//! Glide-core based control plane client
//!
//! Uses glide-core for control plane operations while maintaining
//! compatibility with the existing benchmark infrastructure.

use glide_core::client::{Client, ConnectionRequest};
use glide_redis::{Cmd, Value};
use tokio::runtime::Runtime;

use crate::cluster::{ClusterNode, ClusterTopology};
use crate::config::SearchConfig;

/// Glide-based control plane client
pub struct GlideControlPlane {
    /// Tokio runtime for async operations
    runtime: Runtime,
    /// Glide client
    client: Option<Client>,
    /// Connection configuration
    addresses: Vec<(String, u16)>,
    /// TLS enabled
    tls: bool,
    /// Auth password
    password: Option<String>,
    /// Auth username
    username: Option<String>,
}

impl GlideControlPlane {
    /// Create a new control plane client
    pub fn new(
        addresses: Vec<(String, u16)>,
        tls: bool,
        password: Option<String>,
        username: Option<String>,
    ) -> Result<Self, String> {
        // Create tokio runtime for async operations
        let runtime = Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        Ok(Self {
            runtime,
            client: None,
            addresses,
            tls,
            password,
            username,
        })
    }

    /// Create glide client asynchronously
    async fn create_client(&self) -> Result<Client, String> {
        use glide_core::client::NodeAddress;

        let addresses: Vec<NodeAddress> = self.addresses
            .iter()
            .map(|(host, port)| NodeAddress {
                host: host.clone().into(),
                port: *port,
            })
            .collect();

        let request = ConnectionRequest {
            addresses,
            tls_mode: if self.tls {
                Some(glide_core::client::TlsMode::SecureTls)
            } else {
                Some(glide_core::client::TlsMode::NoTls)
            },
            cluster_mode_enabled: true, // Auto-detect
            request_timeout: Some(5000), // 5 seconds
            ..Default::default()
        };

        // Add auth if provided
        let request = if let Some(ref password) = self.password {
            ConnectionRequest {
                authentication_info: Some(glide_core::client::AuthenticationInfo {
                    password: password.clone().into(),
                    username: self.username.clone().map(|s| s.into()),
                    iam_config: None,
                }),
                ..request
            }
        } else {
            request
        };

        Client::new(request, None)
            .await
            .map_err(|e| format!("Failed to create glide client: {}", e))
    }

    /// Execute a command and get the result
    fn execute_cmd(&mut self, cmd: &Cmd) -> Result<Value, String> {
        // First ensure we have a client
        if self.client.is_none() {
            let client = self.runtime.block_on(async {
                self.create_client().await
            })?;
            self.client = Some(client);
        }
        
        // Now execute the command
        let client = self.client.as_mut().unwrap();
        self.runtime.block_on(async {
            client.send_command(cmd, None)
                .await
                .map_err(|e| format!("Command failed: {}", e))
        })
    }

    /// Discover cluster topology
    pub fn discover_cluster(&mut self) -> Result<Option<ClusterTopology>, String> {
        let mut cmd = Cmd::new();
        cmd.arg("CLUSTER").arg("NODES");

        match self.execute_cmd(&cmd) {
            Ok(Value::BulkString(data)) => {
                let response = String::from_utf8_lossy(&data);
                ClusterTopology::from_cluster_nodes(&response).map(Some)
            }
            Ok(Value::SimpleString(s)) => {
                ClusterTopology::from_cluster_nodes(&s).map(Some)
            }
            Ok(Value::Nil) => Ok(None),
            Ok(other) => Err(format!("Unexpected CLUSTER NODES response: {:?}", other)),
            Err(e) if e.contains("cluster") || e.contains("CLUSTER") => {
                // Not a cluster - standalone mode
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Get INFO for a specific section
    pub fn info(&mut self, section: &str) -> Result<String, String> {
        let mut cmd = Cmd::new();
        cmd.arg("INFO").arg(section);

        match self.execute_cmd(&cmd) {
            Ok(Value::BulkString(data)) => {
                Ok(String::from_utf8_lossy(&data).to_string())
            }
            Ok(Value::SimpleString(s)) => Ok(s),
            Ok(other) => Err(format!("Unexpected INFO response: {:?}", other)),
            Err(e) => Err(e),
        }
    }

    /// Create a vector search index using FT.CREATE
    pub fn create_index(&mut self, config: &SearchConfig, overwrite: bool) -> Result<(), String> {
        // Drop existing index if requested
        if overwrite {
            let _ = self.drop_index(&config.index_name);
        }

        // Build FT.CREATE command
        let mut cmd = Cmd::new();
        cmd.arg("FT.CREATE")
            .arg(&config.index_name)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg("1")
            .arg(&config.prefix)
            .arg("SCHEMA")
            .arg(&config.vector_field)
            .arg("VECTOR");

        // Algorithm-specific parameters
        cmd.arg(config.algorithm.as_str());

        // Vector parameters
        let dim_str = config.dim.to_string();

        // Use distance_metric field and its as_str() method
        let metric_str = config.distance_metric.as_str();

        // Build vector parameters based on algorithm
        match config.algorithm {
            crate::config::VectorAlgorithm::Hnsw => {
                let m_str = config.hnsw_m.unwrap_or(16).to_string();
                let ef_construction_str = config.ef_construction.unwrap_or(200).to_string();
                let ef_runtime_str = config.ef_search.unwrap_or(10).to_string();
                
                cmd.arg("12") // Number of params for HNSW
                    .arg("TYPE").arg("FLOAT32")
                    .arg("DIM").arg(&dim_str)
                    .arg("DISTANCE_METRIC").arg(metric_str)
                    .arg("M").arg(&m_str)
                    .arg("EF_CONSTRUCTION").arg(&ef_construction_str)
                    .arg("EF_RUNTIME").arg(&ef_runtime_str);
            }
            crate::config::VectorAlgorithm::Flat => {
                cmd.arg("6") // Number of params for FLAT
                    .arg("TYPE").arg("FLOAT32")
                    .arg("DIM").arg(&dim_str)
                    .arg("DISTANCE_METRIC").arg(metric_str);
            }
        }

        match self.execute_cmd(&cmd) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("FT.CREATE failed: {}", e)),
        }
    }

    /// Drop an index
    pub fn drop_index(&mut self, index_name: &str) -> Result<(), String> {
        let mut cmd = Cmd::new();
        cmd.arg("FT.DROPINDEX").arg(index_name);

        match self.execute_cmd(&cmd) {
            Ok(_) => Ok(()),
            Err(e) if e.contains("Unknown index") || e.contains("Unknown Index") => {
                // Index doesn't exist - that's fine
                Ok(())
            }
            Err(e) => Err(format!("FT.DROPINDEX failed: {}", e)),
        }
    }

    /// Get FT.INFO for an index
    pub fn ft_info(&mut self, index_name: &str) -> Result<Value, String> {
        let mut cmd = Cmd::new();
        cmd.arg("FT.INFO").arg(index_name);

        self.execute_cmd(&cmd)
    }

    /// Execute FT.SEARCH
    pub fn ft_search(
        &mut self,
        index_name: &str,
        query: &str,
        k: usize,
    ) -> Result<Value, String> {
        let mut cmd = Cmd::new();
        cmd.arg("FT.SEARCH")
            .arg(index_name)
            .arg(query)
            .arg("LIMIT")
            .arg("0")
            .arg(k.to_string());

        self.execute_cmd(&cmd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_plane_creation() {
        let addresses = vec![("localhost".to_string(), 6379)];
        let cp = GlideControlPlane::new(addresses, false, None, None);
        assert!(cp.is_ok());
    }
}
