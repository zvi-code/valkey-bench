//! Raw TCP connection for benchmark traffic
//!
//! This module provides direct TCP (and TLS) connections with
//! pre-allocated buffers for high-performance benchmark traffic.

use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::config::TlsConfig;
use crate::utils::{ConnectionError, RespDecoder, RespEncoder, RespValue};

/// Raw connection wrapper (TCP or TLS)
///
/// For TCP, we split into separate reader/writer for better performance.
/// For TLS, we use a single stream since native-tls doesn't support cloning.
pub enum RawConnection {
    Tcp {
        writer: BufWriter<TcpStream>,
        reader: BufReader<TcpStream>,
    },
    #[cfg(feature = "native-tls-backend")]
    NativeTls {
        stream: native_tls::TlsStream<TcpStream>,
        read_buf: Vec<u8>,
    },
}

impl RawConnection {
    /// Create new TCP connection
    pub fn connect_tcp(
        host: &str,
        port: u16,
        connect_timeout: Duration,
    ) -> Result<Self, ConnectionError> {
        use std::net::ToSocketAddrs;

        let addr_str = format!("{}:{}", host, port);

        // Resolve hostname to socket address
        let addr = addr_str
            .to_socket_addrs()
            .map_err(|e| ConnectionError::ConnectFailed {
                host: host.to_string(),
                port,
                source: e,
            })?
            .next()
            .ok_or_else(|| ConnectionError::ConnectFailed {
                host: host.to_string(),
                port,
                source: io::Error::new(io::ErrorKind::NotFound, "No addresses found"),
            })?;

        let stream = TcpStream::connect_timeout(&addr, connect_timeout).map_err(|e| {
            ConnectionError::ConnectFailed {
                host: host.to_string(),
                port,
                source: e,
            }
        })?;

        // Configure socket
        stream.set_nodelay(true).ok(); // Disable Nagle's algorithm
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

        let writer = BufWriter::with_capacity(
            65536,
            stream
                .try_clone()
                .map_err(|e| ConnectionError::ConnectFailed {
                    host: host.to_string(),
                    port,
                    source: e,
                })?,
        );
        let reader = BufReader::with_capacity(65536, stream);

        Ok(RawConnection::Tcp { writer, reader })
    }

    /// Create new TLS connection
    #[cfg(feature = "native-tls-backend")]
    pub fn connect_tls(
        host: &str,
        port: u16,
        connect_timeout: Duration,
        tls_config: &TlsConfig,
    ) -> Result<Self, ConnectionError> {
        use native_tls::{Certificate, Identity, TlsConnector};

        // Build TLS connector
        let mut builder = TlsConnector::builder();

        if tls_config.skip_verify {
            builder.danger_accept_invalid_certs(true);
            builder.danger_accept_invalid_hostnames(true);
        }

        // Load CA certificate
        if let Some(ref ca_path) = tls_config.ca_cert {
            let ca_data = std::fs::read(ca_path).map_err(|e| {
                ConnectionError::TlsFailed(format!("Failed to read CA cert: {}", e))
            })?;
            let cert = Certificate::from_pem(&ca_data)
                .map_err(|e| ConnectionError::TlsFailed(format!("Invalid CA cert: {}", e)))?;
            builder.add_root_certificate(cert);
        }

        // Load client certificate and key
        if let (Some(ref cert_path), Some(ref key_path)) =
            (&tls_config.client_cert, &tls_config.client_key)
        {
            let cert_data = std::fs::read(cert_path).map_err(|e| {
                ConnectionError::TlsFailed(format!("Failed to read client cert: {}", e))
            })?;
            let key_data = std::fs::read(key_path).map_err(|e| {
                ConnectionError::TlsFailed(format!("Failed to read client key: {}", e))
            })?;

            // Combine cert and key into PKCS8 (native-tls requirement)
            let identity = Identity::from_pkcs8(&cert_data, &key_data).map_err(|e| {
                ConnectionError::TlsFailed(format!("Invalid client identity: {}", e))
            })?;
            builder.identity(identity);
        }

        let connector = builder.build().map_err(|e| {
            ConnectionError::TlsFailed(format!("Failed to build TLS connector: {}", e))
        })?;

        // Connect TCP first
        use std::net::ToSocketAddrs;

        let addr_str = format!("{}:{}", host, port);
        let addr = addr_str
            .to_socket_addrs()
            .map_err(|e| ConnectionError::ConnectFailed {
                host: host.to_string(),
                port,
                source: e,
            })?
            .next()
            .ok_or_else(|| ConnectionError::ConnectFailed {
                host: host.to_string(),
                port,
                source: io::Error::new(io::ErrorKind::NotFound, "No addresses found"),
            })?;

        let tcp_stream = TcpStream::connect_timeout(&addr, connect_timeout).map_err(|e| {
            ConnectionError::ConnectFailed {
                host: host.to_string(),
                port,
                source: e,
            }
        })?;

        tcp_stream.set_nodelay(true).ok();
        tcp_stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .ok();
        tcp_stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .ok();

        // TLS handshake
        let sni_host = tls_config.sni.as_deref().unwrap_or(host);
        let tls_stream = connector
            .connect(sni_host, tcp_stream)
            .map_err(|e| ConnectionError::TlsFailed(format!("TLS handshake failed: {}", e)))?;

        Ok(RawConnection::NativeTls {
            stream: tls_stream,
            read_buf: vec![0u8; 65536],
        })
    }

    /// Write bytes to connection
    pub fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match self {
            RawConnection::Tcp { writer, .. } => writer.write_all(buf),
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { stream, .. } => stream.write_all(buf),
        }
    }

    /// Flush write buffer
    pub fn flush(&mut self) -> io::Result<()> {
        match self {
            RawConnection::Tcp { writer, .. } => writer.flush(),
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { stream, .. } => stream.flush(),
        }
    }

    /// Read a single RESP response
    fn read_response(&mut self) -> io::Result<RespValue> {
        match self {
            RawConnection::Tcp { reader, .. } => {
                let mut decoder = RespDecoder::new(reader);
                decoder.decode()
            }
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { stream, .. } => {
                // For TLS, wrap in BufReader temporarily
                let mut buf_reader = BufReader::new(stream);
                let mut decoder = RespDecoder::new(&mut buf_reader);
                decoder.decode()
            }
        }
    }

    /// Send command and receive response
    pub fn execute(&mut self, encoder: &RespEncoder) -> io::Result<RespValue> {
        self.write_all(encoder.as_bytes())?;
        self.flush()?;
        self.read_response()
    }

    /// Send AUTH command
    pub fn authenticate(
        &mut self,
        password: &str,
        username: Option<&str>,
    ) -> Result<(), ConnectionError> {
        let mut encoder = RespEncoder::with_capacity(256);

        match username {
            Some(user) => encoder.encode_command_str(&["AUTH", user, password]),
            None => encoder.encode_command_str(&["AUTH", password]),
        }

        let response = self
            .execute(&encoder)
            .map_err(|e| ConnectionError::AuthFailed(format!("IO error: {}", e)))?;

        match response {
            RespValue::SimpleString(s) if s == "OK" => Ok(()),
            RespValue::Error(e) => Err(ConnectionError::AuthFailed(e)),
            other => Err(ConnectionError::AuthFailed(format!(
                "Unexpected response: {:?}",
                other
            ))),
        }
    }

    /// Send SELECT command (for standalone mode)
    pub fn select_db(&mut self, db: u32) -> io::Result<RespValue> {
        let mut encoder = RespEncoder::with_capacity(32);
        let db_str = db.to_string();
        encoder.encode_command_str(&["SELECT", &db_str]);
        self.execute(&encoder)
    }

    /// Send PING command
    pub fn ping(&mut self) -> io::Result<bool> {
        let mut encoder = RespEncoder::with_capacity(32);
        encoder.encode_command_str(&["PING"]);

        let response = self.execute(&encoder)?;
        match response {
            RespValue::SimpleString(s) => Ok(s == "PONG"),
            _ => Ok(false),
        }
    }

    /// Send CLUSTER NODES command and return raw response
    pub fn cluster_nodes(&mut self) -> io::Result<String> {
        let mut encoder = RespEncoder::with_capacity(32);
        encoder.encode_command_str(&["CLUSTER", "NODES"]);

        let response = self.execute(&encoder)?;
        match response {
            RespValue::BulkString(data) => {
                String::from_utf8(data).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid UTF-8: {}", e))
                })
            }
            RespValue::Error(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unexpected CLUSTER NODES response: {:?}", other),
            )),
        }
    }

    /// Check if this is a cluster node (returns true if CLUSTER NODES works)
    pub fn is_cluster(&mut self) -> bool {
        self.cluster_nodes().is_ok()
    }

    /// Set read timeout
    pub fn set_read_timeout(&mut self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            RawConnection::Tcp { reader, .. } => reader.get_ref().set_read_timeout(timeout),
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { stream, .. } => stream.get_ref().set_read_timeout(timeout),
        }
    }

    /// Set write timeout
    pub fn set_write_timeout(&mut self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            RawConnection::Tcp { writer, .. } => writer.get_ref().set_write_timeout(timeout),
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { stream, .. } => stream.get_ref().set_write_timeout(timeout),
        }
    }

    /// Read multiple responses (for pipeline)
    pub fn read_responses(&mut self, count: usize) -> io::Result<Vec<RespValue>> {
        let mut responses = Vec::with_capacity(count);
        for _ in 0..count {
            responses.push(self.read_response()?);
        }
        Ok(responses)
    }
}

/// Connection factory for creating connections with common config
#[derive(Clone)]
pub struct ConnectionFactory {
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub tls_config: Option<TlsConfig>,
    pub auth_password: Option<String>,
    pub auth_username: Option<String>,
    pub dbnum: Option<u32>,
}

impl ConnectionFactory {
    /// Create a new connection to the specified host:port
    pub fn create(&self, host: &str, port: u16) -> Result<RawConnection, ConnectionError> {
        // Create connection (TCP or TLS)
        let mut conn = match &self.tls_config {
            #[cfg(feature = "native-tls-backend")]
            Some(tls) => RawConnection::connect_tls(host, port, self.connect_timeout, tls)?,
            #[cfg(not(feature = "native-tls-backend"))]
            Some(_) => {
                return Err(ConnectionError::TlsFailed(
                    "TLS support not compiled in".to_string(),
                ));
            }
            None => RawConnection::connect_tcp(host, port, self.connect_timeout)?,
        };

        // Set timeouts
        conn.set_read_timeout(Some(self.read_timeout)).ok();
        conn.set_write_timeout(Some(self.write_timeout)).ok();

        // Authenticate if configured
        if let Some(ref password) = self.auth_password {
            conn.authenticate(password, self.auth_username.as_deref())?;
        }

        // Select database if configured
        if let Some(db) = self.dbnum {
            conn.select_db(db)
                .map_err(|e| ConnectionError::ConnectFailed {
                    host: host.to_string(),
                    port,
                    source: e,
                })?;
        }

        Ok(conn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a running Valkey server
    // They are marked as ignored by default

    #[test]
    #[ignore]
    fn test_tcp_connection() {
        let mut conn = RawConnection::connect_tcp("127.0.0.1", 6379, Duration::from_secs(5))
            .expect("Failed to connect");

        assert!(conn.ping().expect("Ping failed"));
    }

    #[test]
    #[ignore]
    fn test_connection_factory() {
        let factory = ConnectionFactory {
            connect_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            tls_config: None,
            auth_password: None,
            auth_username: None,
            dbnum: None,
        };

        let mut conn = factory
            .create("127.0.0.1", 6379)
            .expect("Failed to connect");
        assert!(conn.ping().expect("Ping failed"));
    }
}
