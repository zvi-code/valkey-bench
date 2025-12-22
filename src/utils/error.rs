//! Error types for valkey-bench-rs

use std::io;
use thiserror::Error;

/// Top-level application error
#[derive(Error, Debug)]
pub enum BenchmarkError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Connection error: {0}")]
    Connection(#[from] ConnectionError),

    #[error("Protocol error: {0}")]
    Protocol(#[from] ProtocolError),

    #[error("Dataset error: {0}")]
    Dataset(#[from] DatasetError),

    #[error("Cluster error: {0}")]
    Cluster(#[from] ClusterError),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Worker error: {0}")]
    Worker(String),
}

/// Connection-related errors
#[derive(Error, Debug)]
pub enum ConnectionError {
    #[error("Failed to connect to {host}:{port}: {source}")]
    ConnectFailed {
        host: String,
        port: u16,
        source: io::Error,
    },

    #[error("Authentication failed: {0}")]
    AuthFailed(String),

    #[error("TLS handshake failed: {0}")]
    TlsFailed(String),

    #[error("Connection closed unexpectedly")]
    Closed,

    #[error("Connection timeout after {0}ms")]
    Timeout(u64),
}

/// RESP protocol errors
#[derive(Error, Debug)]
pub enum ProtocolError {
    #[error("Invalid RESP type byte: {0}")]
    InvalidType(u8),

    #[error("Invalid bulk string length: {0}")]
    InvalidLength(i64),

    #[error("Unexpected response: expected {expected}, got {actual}")]
    UnexpectedResponse { expected: String, actual: String },

    #[error("Server error: {0}")]
    ServerError(String),

    #[error("MOVED {slot} {host}:{port}")]
    Moved { slot: u16, host: String, port: u16 },

    #[error("ASK {slot} {host}:{port}")]
    Ask { slot: u16, host: String, port: u16 },

    #[error("Parse error: {0}")]
    Parse(String),
}

/// Dataset-related errors
#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("Invalid dataset magic: expected 0x{expected:08X}, got 0x{actual:08X}")]
    InvalidMagic { expected: u32, actual: u32 },

    #[error("Unsupported dataset version: {0}")]
    UnsupportedVersion(u32),

    #[error("Dataset file too small: {size} bytes, minimum {minimum} bytes")]
    FileTooSmall { size: u64, minimum: u64 },

    #[error("Vector index {index} out of bounds (max {max})")]
    IndexOutOfBounds { index: u64, max: u64 },

    #[error("Failed to open dataset: {0}")]
    OpenFailed(io::Error),
}

/// Cluster-related errors
#[derive(Error, Debug)]
pub enum ClusterError {
    #[error("Failed to parse CLUSTER NODES response: {0}")]
    ParseFailed(String),

    #[error("No primary nodes found in cluster")]
    NoPrimaries,

    #[error("Slot {0} has no assigned node")]
    UnassignedSlot(u16),

    #[error("Node {0} not found in topology")]
    NodeNotFound(String),

    #[error("Cluster topology refresh failed: {0}")]
    RefreshFailed(String),
}

pub type Result<T> = std::result::Result<T, BenchmarkError>;
