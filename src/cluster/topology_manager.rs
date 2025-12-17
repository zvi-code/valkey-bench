//! Dynamic cluster topology manager
//!
//! Handles MOVED/ASK redirections and topology refresh during benchmarks.
//! Similar to fetchClusterSlotsConfiguration in the C implementation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use tracing::{info, warn};

use super::topology::ClusterTopology;
use crate::client::{ConnectionFactory, ControlPlaneExt, RawConnection};

/// Redirect information parsed from MOVED/ASK error
#[derive(Debug, Clone)]
pub struct RedirectInfo {
    /// Target slot
    pub slot: u16,
    /// Target host
    pub host: String,
    /// Target port
    pub port: u16,
    /// Whether this is an ASK redirect (requires ASKING prefix)
    pub is_ask: bool,
}

impl RedirectInfo {
    /// Parse from error message like "MOVED 3999 127.0.0.1:7001" or "ASK 3999 127.0.0.1:7001"
    pub fn parse(error_msg: &str) -> Option<Self> {
        let parts: Vec<&str> = error_msg.split_whitespace().collect();
        if parts.len() < 3 {
            return None;
        }

        let is_ask = parts[0] == "ASK";
        let is_moved = parts[0] == "MOVED";

        if !is_ask && !is_moved {
            return None;
        }

        let slot: u16 = parts[1].parse().ok()?;
        let addr_parts: Vec<&str> = parts[2].split(':').collect();
        if addr_parts.len() != 2 {
            return None;
        }

        let host = addr_parts[0].to_string();
        let port: u16 = addr_parts[1].parse().ok()?;

        Some(Self {
            slot,
            host,
            port,
            is_ask,
        })
    }
}

/// Shared topology manager for cluster mode
///
/// Provides thread-safe access to cluster topology and handles
/// dynamic topology updates when MOVED errors are received.
pub struct TopologyManager {
    /// Current cluster topology
    topology: RwLock<ClusterTopology>,

    /// Connection factory for creating new connections
    connection_factory: ConnectionFactory,

    /// Whether a refresh is currently in progress (prevents concurrent refreshes)
    is_refreshing: AtomicBool,

    /// Last update timestamp (epoch counter)
    last_update: AtomicU64,

    /// Cache of redirect connections (host:port -> connection)
    /// Used for handling ASK/MOVED redirects without full topology refresh
    redirect_connections: RwLock<HashMap<String, Arc<RwLock<RawConnection>>>>,

    /// Seed addresses for topology discovery
    seed_addresses: Vec<(String, u16)>,
}

impl TopologyManager {
    /// Create new topology manager
    pub fn new(
        initial_topology: ClusterTopology,
        connection_factory: ConnectionFactory,
        seed_addresses: Vec<(String, u16)>,
    ) -> Self {
        Self {
            topology: RwLock::new(initial_topology),
            connection_factory,
            is_refreshing: AtomicBool::new(false),
            last_update: AtomicU64::new(1),
            redirect_connections: RwLock::new(HashMap::new()),
            seed_addresses,
        }
    }

    /// Get current topology version
    pub fn version(&self) -> u64 {
        self.last_update.load(Ordering::Relaxed)
    }

    /// Get node for slot
    pub fn get_node_for_slot(&self, slot: u16) -> Option<(String, u16)> {
        let topology = self.topology.read().ok()?;
        topology
            .get_node_for_slot(slot)
            .map(|n| (n.host.clone(), n.port))
    }

    /// Get all primary node addresses
    pub fn get_primary_addresses(&self) -> Vec<(String, u16)> {
        if let Ok(topology) = self.topology.read() {
            topology
                .primaries()
                .filter(|n| n.is_available())
                .map(|n| (n.host.clone(), n.port))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Refresh cluster topology
    ///
    /// Called when MOVED errors are encountered. Uses compare-and-swap
    /// to ensure only one thread performs the refresh.
    ///
    /// Returns true if refresh was performed, false if another thread
    /// is already refreshing or if refresh failed.
    pub fn refresh_topology(&self) -> bool {
        // Try to acquire refresh lock
        if self
            .is_refreshing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            // Another thread is already refreshing
            return false;
        }

        warn!("Cluster topology changed, refreshing slot configuration...");

        let result = self.do_refresh();

        // Release refresh lock
        self.is_refreshing.store(false, Ordering::SeqCst);

        result
    }

    /// Perform the actual topology refresh
    fn do_refresh(&self) -> bool {
        // Try each seed address until we get a successful CLUSTER NODES
        for (host, port) in &self.seed_addresses {
            match self.fetch_topology_from_node(host, *port) {
                Ok(new_topology) => {
                    // Update topology
                    if let Ok(mut topology) = self.topology.write() {
                        info!(
                            "Cluster topology refreshed: {} primaries, {} total nodes",
                            new_topology.num_primaries(),
                            new_topology.num_nodes()
                        );
                        *topology = new_topology;
                        self.last_update.fetch_add(1, Ordering::SeqCst);

                        // Clear redirect connection cache
                        if let Ok(mut cache) = self.redirect_connections.write() {
                            cache.clear();
                        }

                        return true;
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to fetch topology from {}:{}: {}",
                        host, port, e
                    );
                    continue;
                }
            }
        }

        // Also try current topology nodes (collect addresses first to avoid holding lock)
        let topology_node_addresses: Vec<(String, u16)> = {
            if let Ok(topology) = self.topology.read() {
                topology
                    .primaries()
                    .filter(|node| {
                        !self
                            .seed_addresses
                            .iter()
                            .any(|(h, p)| h == &node.host && *p == node.port)
                    })
                    .map(|node| (node.host.clone(), node.port))
                    .collect()
            } else {
                Vec::new()
            }
        };

        for (host, port) in topology_node_addresses {
            match self.fetch_topology_from_node(&host, port) {
                Ok(new_topology) => {
                    if let Ok(mut topo) = self.topology.write() {
                        info!(
                            "Cluster topology refreshed: {} primaries, {} total nodes",
                            new_topology.num_primaries(),
                            new_topology.num_nodes()
                        );
                        *topo = new_topology;
                        self.last_update.fetch_add(1, Ordering::SeqCst);

                        if let Ok(mut cache) = self.redirect_connections.write() {
                            cache.clear();
                        }

                        return true;
                    }
                }
                Err(_) => continue,
            }
        }

        warn!("Failed to refresh cluster topology from any node");
        false
    }

    /// Fetch topology from a specific node
    fn fetch_topology_from_node(&self, host: &str, port: u16) -> Result<ClusterTopology, String> {
        let mut conn = self
            .connection_factory
            .create(host, port)
            .map_err(|e| format!("Connection failed: {}", e))?;

        let nodes_response = conn
            .cluster_nodes()
            .map_err(|e| format!("CLUSTER NODES failed: {}", e))?;

        ClusterTopology::from_cluster_nodes(&nodes_response)
    }

    /// Get or create a connection for redirect target
    ///
    /// Used for handling MOVED/ASK redirects without waiting for full
    /// topology refresh.
    pub fn get_redirect_connection(
        &self,
        host: &str,
        port: u16,
    ) -> Option<Arc<RwLock<RawConnection>>> {
        let key = format!("{}:{}", host, port);

        // Check cache first
        {
            let cache = self.redirect_connections.read().ok()?;
            if let Some(conn) = cache.get(&key) {
                return Some(Arc::clone(conn));
            }
        }

        // Create new connection
        match self.connection_factory.create(host, port) {
            Ok(conn) => {
                let conn = Arc::new(RwLock::new(conn));
                if let Ok(mut cache) = self.redirect_connections.write() {
                    cache.insert(key, Arc::clone(&conn));
                }
                Some(conn)
            }
            Err(e) => {
                warn!("Failed to create redirect connection to {}:{}: {}", host, port, e);
                None
            }
        }
    }

    /// Check if CLUSTERDOWN error
    pub fn is_cluster_down(error_msg: &str) -> bool {
        error_msg.starts_with("CLUSTERDOWN")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redirect_info_parse_moved() {
        let info = RedirectInfo::parse("MOVED 3999 127.0.0.1:7001").unwrap();
        assert_eq!(info.slot, 3999);
        assert_eq!(info.host, "127.0.0.1");
        assert_eq!(info.port, 7001);
        assert!(!info.is_ask);
    }

    #[test]
    fn test_redirect_info_parse_ask() {
        let info = RedirectInfo::parse("ASK 1234 10.0.0.5:6380").unwrap();
        assert_eq!(info.slot, 1234);
        assert_eq!(info.host, "10.0.0.5");
        assert_eq!(info.port, 6380);
        assert!(info.is_ask);
    }

    #[test]
    fn test_redirect_info_parse_invalid() {
        assert!(RedirectInfo::parse("ERR unknown command").is_none());
        assert!(RedirectInfo::parse("MOVED").is_none());
        assert!(RedirectInfo::parse("MOVED 123").is_none());
        assert!(RedirectInfo::parse("MOVED 123 invalid").is_none());
    }

    #[test]
    fn test_is_cluster_down() {
        assert!(TopologyManager::is_cluster_down("CLUSTERDOWN The cluster is down"));
        assert!(!TopologyManager::is_cluster_down("MOVED 123 host:port"));
    }
}
