//! Cluster backend abstraction
//!
//! This module provides the ClusterBackend trait for abstracting engine-specific
//! differences between ElastiCache Valkey, MemoryDB, and Valkey OSS.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use std::time::Duration;

use crate::client::{ControlPlane, ControlPlaneExt};
use crate::metrics::{EngineType, FtInfoResult};
use crate::utils::RespValue;

/// Connection configuration per engine type
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Maximum idle time before connection is closed
    pub max_idle_time: Duration,
    /// Connection establishment timeout
    pub connection_timeout: Duration,
    /// Read timeout
    pub read_timeout: Duration,
    /// Write timeout
    pub write_timeout: Duration,
    /// Whether this engine supports cluster mode disabled
    pub supports_cluster_mode_disabled: bool,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            max_idle_time: Duration::from_secs(300),
            connection_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            supports_cluster_mode_disabled: true,
        }
    }
}

/// Cluster backend trait for abstracting engine-specific behavior
///
/// This trait provides a unified interface for different Valkey/Redis engines:
/// - ElastiCache Valkey (cluster mode enabled/disabled)
/// - MemoryDB
/// - Valkey OSS
pub trait ClusterBackend: Send + Sync {
    /// Get the engine type
    fn engine_type(&self) -> EngineType;

    /// Get the display name for this backend
    fn name(&self) -> &'static str;

    /// Parse FT.INFO response according to engine format
    fn parse_ft_info(&self, response: &RespValue) -> FtInfoResult {
        FtInfoResult::from_response(response, self.engine_type())
    }

    /// Get connection configuration for this engine
    fn connection_config(&self) -> ConnectionConfig {
        ConnectionConfig::default()
    }

    /// Get the INFO section name for search metrics (if any)
    fn info_search_section(&self) -> Option<&'static str> {
        match self.engine_type() {
            EngineType::MemoryDb => Some("SEARCH"),
            EngineType::ElasticacheValkey | EngineType::ElasticacheServerless => None,
            EngineType::OssValkey | EngineType::Unknown => None,
        }
    }

    /// Check if this backend supports the FT.SEARCH LOCALONLY flag
    /// Note: Skipped per requirements - always returns false
    fn supports_localonly(&self) -> bool {
        false
    }
}

// =============================================================================
// Backend Implementations
// =============================================================================

/// ElastiCache Valkey backend
#[derive(Debug, Clone)]
pub struct ElastiCacheBackend {
    serverless: bool,
}

impl ElastiCacheBackend {
    pub fn new(serverless: bool) -> Self {
        Self { serverless }
    }

    pub fn provisioned() -> Self {
        Self { serverless: false }
    }

    pub fn serverless() -> Self {
        Self { serverless: true }
    }
}

impl ClusterBackend for ElastiCacheBackend {
    fn engine_type(&self) -> EngineType {
        if self.serverless {
            EngineType::ElasticacheServerless
        } else {
            EngineType::ElasticacheValkey
        }
    }

    fn name(&self) -> &'static str {
        if self.serverless {
            "ElastiCache Serverless"
        } else {
            "ElastiCache Valkey"
        }
    }

    fn connection_config(&self) -> ConnectionConfig {
        ConnectionConfig {
            supports_cluster_mode_disabled: !self.serverless,
            ..ConnectionConfig::default()
        }
    }
}

/// MemoryDB backend
#[derive(Debug, Clone, Default)]
pub struct MemoryDBBackend;

impl MemoryDBBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ClusterBackend for MemoryDBBackend {
    fn engine_type(&self) -> EngineType {
        EngineType::MemoryDb
    }

    fn name(&self) -> &'static str {
        "MemoryDB"
    }

    fn info_search_section(&self) -> Option<&'static str> {
        Some("SEARCH")
    }
}

/// Valkey OSS backend
#[derive(Debug, Clone, Default)]
pub struct ValkeyOSSBackend;

impl ValkeyOSSBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ClusterBackend for ValkeyOSSBackend {
    fn engine_type(&self) -> EngineType {
        EngineType::OssValkey
    }

    fn name(&self) -> &'static str {
        "Valkey OSS"
    }
}

/// Unknown backend (fallback)
#[derive(Debug, Clone, Default)]
pub struct UnknownBackend;

impl ClusterBackend for UnknownBackend {
    fn engine_type(&self) -> EngineType {
        EngineType::Unknown
    }

    fn name(&self) -> &'static str {
        "Unknown"
    }
}

// =============================================================================
// Auto-Detection Backend
// =============================================================================

/// Auto-detecting backend that wraps engine detection
///
/// This backend lazily detects the engine type on first access and
/// delegates to the appropriate backend implementation.
pub struct AutoDetectBackend {
    detected: AtomicBool,
    inner: RwLock<Box<dyn ClusterBackend>>,
}

impl Default for AutoDetectBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoDetectBackend {
    pub fn new() -> Self {
        Self {
            detected: AtomicBool::new(false),
            inner: RwLock::new(Box::new(UnknownBackend)),
        }
    }

    /// Detect the engine type from a connection
    ///
    /// This should be called once during initialization with a valid connection.
    pub fn detect<C: ControlPlane>(&self, conn: &mut C) -> std::io::Result<EngineType> {
        // Get INFO server response
        let info_str = conn.info("server")?;

        // Detect engine type
        let engine_type = EngineType::detect(&info_str);

        // Create appropriate backend
        let backend: Box<dyn ClusterBackend> = match engine_type {
            EngineType::ElasticacheValkey => Box::new(ElastiCacheBackend::provisioned()),
            EngineType::ElasticacheServerless => Box::new(ElastiCacheBackend::serverless()),
            EngineType::MemoryDb => Box::new(MemoryDBBackend::new()),
            EngineType::OssValkey => Box::new(ValkeyOSSBackend::new()),
            EngineType::Unknown => Box::new(UnknownBackend),
        };

        // Store the detected backend
        {
            let mut inner = self.inner.write().unwrap();
            *inner = backend;
        }
        self.detected.store(true, Ordering::SeqCst);

        Ok(engine_type)
    }

    /// Check if detection has been performed
    pub fn is_detected(&self) -> bool {
        self.detected.load(Ordering::SeqCst)
    }

    /// Get the detected engine type
    pub fn detected_engine_type(&self) -> EngineType {
        self.inner.read().unwrap().engine_type()
    }
}

impl ClusterBackend for AutoDetectBackend {
    fn engine_type(&self) -> EngineType {
        self.inner.read().unwrap().engine_type()
    }

    fn name(&self) -> &'static str {
        if !self.is_detected() {
            return "Auto-detect (pending)";
        }
        // We need to return a static str, so we match on engine type
        match self.engine_type() {
            EngineType::ElasticacheValkey => "ElastiCache Valkey",
            EngineType::ElasticacheServerless => "ElastiCache Serverless",
            EngineType::MemoryDb => "MemoryDB",
            EngineType::OssValkey => "Valkey OSS",
            EngineType::Unknown => "Unknown",
        }
    }

    fn parse_ft_info(&self, response: &RespValue) -> FtInfoResult {
        self.inner.read().unwrap().parse_ft_info(response)
    }

    fn connection_config(&self) -> ConnectionConfig {
        self.inner.read().unwrap().connection_config()
    }

    fn info_search_section(&self) -> Option<&'static str> {
        match self.engine_type() {
            EngineType::MemoryDb => Some("SEARCH"),
            _ => None,
        }
    }
}

// =============================================================================
// Factory Function
// =============================================================================

/// Create a backend from a known engine type
pub fn create_backend(engine_type: EngineType) -> Box<dyn ClusterBackend> {
    match engine_type {
        EngineType::ElasticacheValkey => Box::new(ElastiCacheBackend::provisioned()),
        EngineType::ElasticacheServerless => Box::new(ElastiCacheBackend::serverless()),
        EngineType::MemoryDb => Box::new(MemoryDBBackend::new()),
        EngineType::OssValkey => Box::new(ValkeyOSSBackend::new()),
        EngineType::Unknown => Box::new(UnknownBackend),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elasticache_backend() {
        let backend = ElastiCacheBackend::provisioned();
        assert_eq!(backend.engine_type(), EngineType::ElasticacheValkey);
        assert_eq!(backend.name(), "ElastiCache Valkey");
        assert!(backend.connection_config().supports_cluster_mode_disabled);

        let serverless = ElastiCacheBackend::serverless();
        assert_eq!(serverless.engine_type(), EngineType::ElasticacheServerless);
        assert!(!serverless.connection_config().supports_cluster_mode_disabled);
    }

    #[test]
    fn test_memorydb_backend() {
        let backend = MemoryDBBackend::new();
        assert_eq!(backend.engine_type(), EngineType::MemoryDb);
        assert_eq!(backend.name(), "MemoryDB");
        assert_eq!(backend.info_search_section(), Some("SEARCH"));
    }

    #[test]
    fn test_valkey_oss_backend() {
        let backend = ValkeyOSSBackend::new();
        assert_eq!(backend.engine_type(), EngineType::OssValkey);
        assert_eq!(backend.name(), "Valkey OSS");
    }

    #[test]
    fn test_create_backend() {
        let backend = create_backend(EngineType::MemoryDb);
        assert_eq!(backend.engine_type(), EngineType::MemoryDb);

        let backend = create_backend(EngineType::ElasticacheValkey);
        assert_eq!(backend.engine_type(), EngineType::ElasticacheValkey);
    }

    #[test]
    fn test_auto_detect_initial_state() {
        let backend = AutoDetectBackend::new();
        assert!(!backend.is_detected());
        assert_eq!(backend.engine_type(), EngineType::Unknown);
    }
}
