//! TLS configuration

use std::path::PathBuf;

/// TLS configuration
#[derive(Debug, Clone)]
pub struct TlsConfig {
    pub skip_verify: bool,
    pub ca_cert: Option<PathBuf>,
    pub client_cert: Option<PathBuf>,
    pub client_key: Option<PathBuf>,
    pub sni: Option<String>,
}

impl TlsConfig {
    /// Check if client certificate authentication is configured
    pub fn has_client_cert(&self) -> bool {
        self.client_cert.is_some() && self.client_key.is_some()
    }
}
