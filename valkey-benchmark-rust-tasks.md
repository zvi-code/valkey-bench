# Valkey-Search-Benchmark Rust Implementation: Detailed Task List

## Document Overview

This document provides a comprehensive, phase-by-phase task breakdown for implementing the `valkey-search-benchmark` tool in Rust. Each phase concludes with a git commit and validation checkpoint. Tasks are written in an agent-friendly format with explicit instructions, expected outcomes, and validation criteria.

---

## Phase 1: Project Skeleton and CLI Foundation

**Objective:** Establish project structure, dependencies, and complete CLI argument parsing matching all C version options.

**Git Branch:** `phase-1-skeleton`

### Task 1.1: Initialize Cargo Project

**Description:** Create the Rust project with proper directory structure and initial Cargo.toml.

**Instructions:**
1. Create project directory: `mkdir -p valkey-search-benchmark && cd valkey-search-benchmark`
2. Initialize Cargo project: `cargo init --name valkey-search-benchmark`
3. Create directory structure:
```
src/
├── main.rs
├── lib.rs
├── config/
│   ├── mod.rs
│   ├── cli.rs
│   ├── benchmark_config.rs
│   ├── search_config.rs
│   └── tls_config.rs
├── client/
│   └── mod.rs
├── cluster/
│   └── mod.rs
├── workload/
│   └── mod.rs
├── dataset/
│   └── mod.rs
├── benchmark/
│   └── mod.rs
├── metrics/
│   └── mod.rs
├── optimizer/
│   └── mod.rs
└── utils/
    ├── mod.rs
    └── error.rs
```

**Validation:**
- `cargo build` succeeds with no errors
- Directory structure matches specification

---

### Task 1.2: Configure Cargo.toml with Dependencies

**Description:** Add all required dependencies to Cargo.toml with proper feature flags.

**Instructions:**
Create `Cargo.toml` with the following content:

```toml
[package]
name = "valkey-search-benchmark"
version = "0.1.0"
edition = "2021"
authors = ["Valkey Contributors"]
description = "High-performance benchmark tool for Valkey with vector search support"
license = "BSD-3-Clause"

[dependencies]
# CLI parsing
clap = { version = "4", features = ["derive", "env", "string"] }

# Metrics
hdrhistogram = "7"

# Dataset memory mapping
memmap2 = "0.9"

# Concurrency primitives
parking_lot = "0.12"
crossbeam-channel = "0.5"

# Random number generation (fast, non-cryptographic)
fastrand = "2"

# Integer formatting (fast itoa)
itoa = "1"

# Progress and logging
indicatif = "0.17"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Utilities
bytes = "1"
thiserror = "1"
anyhow = "1"

# TLS support
native-tls = { version = "0.2", optional = true }
rustls = { version = "0.23", optional = true }
rustls-pemfile = { version = "2", optional = true }
webpki-roots = { version = "0.26", optional = true }

# Control plane (cluster discovery) - redis-rs standalone
redis = { version = "0.27", features = ["cluster", "tokio-comp"], optional = true }
tokio = { version = "1", features = ["rt", "net", "time", "sync"], optional = true }

[features]
default = ["native-tls-backend"]
native-tls-backend = ["native-tls"]
rustls-backend = ["rustls", "rustls-pemfile", "webpki-roots"]
control-plane = ["redis", "tokio"]

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1

[profile.bench]
opt-level = 3
debug = true
```

**Validation:**
- `cargo check` succeeds
- `cargo tree` shows expected dependency graph

---

### Task 1.3: Create Error Types

**Description:** Define custom error types for the application using thiserror.

**Instructions:**
Create `src/utils/error.rs`:

```rust
//! Error types for valkey-search-benchmark

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
```

Update `src/utils/mod.rs`:
```rust
pub mod error;
pub use error::{BenchmarkError, ConnectionError, ProtocolError, DatasetError, ClusterError, Result};
```

**Validation:**
- `cargo check` succeeds
- All error types compile without warnings

---

### Task 1.4: Implement CLI Argument Parsing

**Description:** Create comprehensive CLI argument parsing that matches ALL options from the C implementation.

**Instructions:**
Create `src/config/cli.rs`:

```rust
//! Command-line argument parsing
//!
//! This module defines all CLI arguments matching the C valkey-benchmark implementation.
//! Arguments are grouped by category for clarity.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// High-performance benchmark tool for Valkey with vector search support
#[derive(Parser, Debug, Clone)]
#[command(name = "valkey-search-benchmark")]
#[command(version, about, long_about = None)]
#[command(arg_required_else_help = false)]
pub struct CliArgs {
    // ===== Connection Options =====
    
    /// Server hostname (can be specified multiple times for cluster)
    #[arg(short = 'h', long = "host", default_value = "127.0.0.1", action = clap::ArgAction::Append)]
    pub hosts: Vec<String>,

    /// Server port
    #[arg(short = 'p', long = "port", default_value_t = 6379)]
    pub port: u16,

    /// Unix socket path (overrides host/port)
    #[arg(short = 's', long = "socket")]
    pub socket: Option<PathBuf>,

    /// Password for AUTH command
    #[arg(short = 'a', long = "auth")]
    pub password: Option<String>,

    /// Username for ACL AUTH (requires --auth)
    #[arg(long = "user")]
    pub username: Option<String>,

    // ===== TLS Options =====

    /// Enable TLS connection
    #[arg(long = "tls")]
    pub tls: bool,

    /// Skip TLS certificate verification (insecure)
    #[arg(long = "tls-skip-verify")]
    pub tls_skip_verify: bool,

    /// CA certificate file for TLS
    #[arg(long = "tls-ca-cert")]
    pub tls_ca_cert: Option<PathBuf>,

    /// Client certificate file for TLS
    #[arg(long = "tls-cert")]
    pub tls_cert: Option<PathBuf>,

    /// Client private key file for TLS
    #[arg(long = "tls-key")]
    pub tls_key: Option<PathBuf>,

    /// Server Name Indication for TLS
    #[arg(long = "tls-sni")]
    pub tls_sni: Option<String>,

    // ===== Benchmark Parameters =====

    /// Number of parallel connections (total across all threads)
    #[arg(short = 'c', long = "clients", default_value_t = 50)]
    pub clients: u32,

    /// Number of worker threads
    #[arg(long = "threads", default_value_t = 0)]
    pub threads: u32,

    /// Total number of requests to issue
    #[arg(short = 'n', long = "requests", default_value_t = 100000)]
    pub requests: u64,

    /// Database number to SELECT
    #[arg(short = 'd', long = "dbnum")]
    pub dbnum: Option<u32>,

    /// Pipeline depth (commands per batch)
    #[arg(short = 'P', long = "pipeline", default_value_t = 1)]
    pub pipeline: u32,

    // ===== Key Generation =====

    /// Size of key/value data in bytes
    #[arg(short = 'D', long = "data-size", default_value_t = 3)]
    pub data_size: usize,

    /// Use random keys within keyspace range
    #[arg(short = 'r', long = "random-keys", default_value_t = 0)]
    pub keyspace_len: u64,

    /// Use sequential keys instead of random
    #[arg(long = "sequential")]
    pub sequential: bool,

    /// Key prefix for generated keys
    #[arg(long = "key-prefix", default_value = "key:")]
    pub key_prefix: String,

    // ===== Cluster Options =====

    /// Enable cluster mode
    #[arg(long = "cluster")]
    pub cluster_mode: bool,

    /// Read from replicas strategy
    #[arg(long = "rfr", value_enum, default_value_t = ReadFromReplica::Primary)]
    pub read_from_replica: ReadFromReplica,

    /// Enable node request balancing
    #[arg(long = "balance-nodes")]
    pub balance_nodes: bool,

    /// Node balance tolerance percentage
    #[arg(long = "balance-tolerance", default_value_t = 10)]
    pub balance_tolerance: u32,

    // ===== Workload Selection =====

    /// Benchmark type(s) to run
    #[arg(short = 't', long = "tests", value_delimiter = ',')]
    pub tests: Option<Vec<String>>,

    /// Custom command to benchmark (RESP format)
    #[arg(long = "command")]
    pub custom_command: Option<String>,

    // ===== Vector Search Options =====

    /// Index name for vector search
    #[arg(long = "search-index", default_value = "idx")]
    pub search_index: String,

    /// Vector field name in hash
    #[arg(long = "search-vector-field", default_value = "embedding")]
    pub search_vector_field: String,

    /// Key prefix for vector data
    #[arg(long = "search-prefix", default_value = "vec:")]
    pub search_prefix: String,

    /// Vector algorithm (HNSW or FLAT)
    #[arg(long = "search-algorithm", value_enum, default_value_t = VectorAlgorithm::Hnsw)]
    pub search_algorithm: VectorAlgorithm,

    /// Distance metric (L2, IP, COSINE)
    #[arg(long = "search-distance", value_enum, default_value_t = DistanceMetric::L2)]
    pub search_distance: DistanceMetric,

    /// Vector dimension
    #[arg(long = "vector-dim", default_value_t = 128)]
    pub vector_dim: u32,

    /// Number of nearest neighbors to return (k)
    #[arg(short = 'k', long = "search-k", default_value_t = 10)]
    pub search_k: u32,

    /// HNSW EF_CONSTRUCTION parameter
    #[arg(long = "ef-construction")]
    pub ef_construction: Option<u32>,

    /// HNSW M parameter
    #[arg(long = "hnsw-m")]
    pub hnsw_m: Option<u32>,

    /// EF_RUNTIME for search queries
    #[arg(long = "ef-search")]
    pub ef_search: Option<u32>,

    /// Use NOCONTENT in FT.SEARCH (return IDs only)
    #[arg(long = "nocontent")]
    pub nocontent: bool,

    // ===== Dataset Options =====

    /// Path to binary dataset file
    #[arg(long = "dataset")]
    pub dataset: Option<PathBuf>,

    /// Enable filtered search (use dataset metadata)
    #[arg(long = "filtered-search")]
    pub filtered_search: bool,

    /// Number of vectors to load (0 = all)
    #[arg(long = "num-vectors", default_value_t = 0)]
    pub num_vectors: u64,

    /// Starting vector index
    #[arg(long = "vector-offset", default_value_t = 0)]
    pub vector_offset: u64,

    // ===== Rate Limiting =====

    /// Target requests per second (0 = unlimited)
    #[arg(long = "rps", default_value_t = 0)]
    pub requests_per_second: u64,

    // ===== Optimizer Options =====

    /// Enable automatic parameter optimization
    #[arg(long = "optimize")]
    pub optimize: bool,

    /// Target recall for optimizer (0.0 - 1.0)
    #[arg(long = "target-recall", default_value_t = 0.95)]
    pub target_recall: f64,

    /// Target QPS for optimizer
    #[arg(long = "target-qps")]
    pub target_qps: Option<u64>,

    /// Target p99 latency in milliseconds
    #[arg(long = "target-p99")]
    pub target_p99_ms: Option<u64>,

    /// Maximum ef_search to try
    #[arg(long = "max-ef-search", default_value_t = 4096)]
    pub max_ef_search: u32,

    // ===== Output Options =====

    /// Output file path
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Output format
    #[arg(long = "output-format", value_enum, default_value_t = OutputFormat::Text)]
    pub output_format: OutputFormat,

    /// Output CSV file for per-second stats
    #[arg(long = "csv")]
    pub csv_output: Option<PathBuf>,

    /// Quiet mode (minimal output)
    #[arg(short = 'q', long = "quiet")]
    pub quiet: bool,

    /// Verbose output
    #[arg(short = 'v', long = "verbose")]
    pub verbose: bool,

    // ===== Timing Options =====

    /// Run duration in seconds (overrides -n)
    #[arg(long = "duration")]
    pub duration_secs: Option<u64>,

    /// Connection timeout in milliseconds
    #[arg(long = "connect-timeout", default_value_t = 5000)]
    pub connect_timeout_ms: u64,

    /// Request timeout in milliseconds
    #[arg(long = "request-timeout", default_value_t = 30000)]
    pub request_timeout_ms: u64,

    // ===== Advanced Options =====

    /// Skip index creation (assume exists)
    #[arg(long = "skip-index-create")]
    pub skip_index_create: bool,

    /// Skip data loading (assume loaded)
    #[arg(long = "skip-load")]
    pub skip_load: bool,

    /// Delete index after benchmark
    #[arg(long = "cleanup")]
    pub cleanup: bool,

    /// Warmup requests before measurement
    #[arg(long = "warmup", default_value_t = 0)]
    pub warmup_requests: u64,

    /// Seed for random number generation (0 = random seed)
    #[arg(long = "seed", default_value_t = 0)]
    pub seed: u64,
}

/// Read-from-replica strategy
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadFromReplica {
    /// Always read from primary
    #[default]
    Primary,
    /// Prefer replicas, fallback to primary
    PreferReplica,
    /// Round-robin across all nodes
    RoundRobin,
    /// Prefer same availability zone
    AzAffinity,
}

/// Vector search algorithm
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorAlgorithm {
    #[default]
    Hnsw,
    Flat,
}

impl VectorAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            VectorAlgorithm::Hnsw => "HNSW",
            VectorAlgorithm::Flat => "FLAT",
        }
    }
}

/// Distance metric for vector search
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    #[default]
    L2,
    #[value(name = "IP")]
    InnerProduct,
    Cosine,
}

impl DistanceMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            DistanceMetric::L2 => "L2",
            DistanceMetric::InnerProduct => "IP",
            DistanceMetric::Cosine => "COSINE",
        }
    }
}

/// Output format for results
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Csv,
}

impl CliArgs {
    /// Parse CLI arguments from command line
    pub fn parse_args() -> Self {
        Self::parse()
    }

    /// Validate argument combinations
    pub fn validate(&self) -> Result<(), String> {
        // Username requires password
        if self.username.is_some() && self.password.is_none() {
            return Err("--user requires --auth to be set".to_string());
        }

        // TLS cert requires TLS key
        if self.tls_cert.is_some() != self.tls_key.is_some() {
            return Err("--tls-cert and --tls-key must both be specified".to_string());
        }

        // Client certificate requires CA cert (usually)
        if self.tls_cert.is_some() && self.tls_ca_cert.is_none() && !self.tls_skip_verify {
            return Err("--tls-cert typically requires --tls-ca-cert (or use --tls-skip-verify)".to_string());
        }

        // Pipeline must be positive
        if self.pipeline == 0 {
            return Err("--pipeline must be at least 1".to_string());
        }

        // Dataset required for vector search tests
        if let Some(ref tests) = self.tests {
            let needs_dataset = tests.iter().any(|t| {
                matches!(t.to_lowercase().as_str(), "vecload" | "vecquery" | "vec-load" | "vec-query")
            });
            if needs_dataset && self.dataset.is_none() {
                return Err("Vector search tests require --dataset".to_string());
            }
        }

        // Target recall must be in valid range
        if !(0.0..=1.0).contains(&self.target_recall) {
            return Err("--target-recall must be between 0.0 and 1.0".to_string());
        }

        // Keyspace length for random keys
        if self.keyspace_len == 0 && !self.sequential {
            // Will use default keyspace
        }

        Ok(())
    }

    /// Get effective number of threads (0 = auto-detect)
    pub fn effective_threads(&self) -> u32 {
        if self.threads == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get() as u32)
                .unwrap_or(4)
        } else {
            self.threads
        }
    }

    /// Get effective keyspace length
    pub fn effective_keyspace(&self) -> u64 {
        if self.keyspace_len == 0 {
            self.requests
        } else {
            self.keyspace_len
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_args() {
        let args = CliArgs::parse_from(["test"]);
        assert_eq!(args.port, 6379);
        assert_eq!(args.clients, 50);
        assert_eq!(args.requests, 100000);
        assert_eq!(args.pipeline, 1);
    }

    #[test]
    fn test_multiple_hosts() {
        let args = CliArgs::parse_from([
            "test", "-h", "host1", "-h", "host2", "-h", "host3"
        ]);
        assert_eq!(args.hosts, vec!["host1", "host2", "host3"]);
    }

    #[test]
    fn test_vector_search_args() {
        let args = CliArgs::parse_from([
            "test",
            "--search-index", "myidx",
            "--vector-dim", "256",
            "-k", "20",
            "--ef-search", "100",
        ]);
        assert_eq!(args.search_index, "myidx");
        assert_eq!(args.vector_dim, 256);
        assert_eq!(args.search_k, 20);
        assert_eq!(args.ef_search, Some(100));
    }

    #[test]
    fn test_validation_user_without_auth() {
        let args = CliArgs::parse_from(["test", "--user", "admin"]);
        assert!(args.validate().is_err());
    }

    #[test]
    fn test_validation_tls_cert_without_key() {
        let args = CliArgs::parse_from(["test", "--tls-cert", "cert.pem"]);
        assert!(args.validate().is_err());
    }
}
```

**Validation:**
- `cargo test --lib` passes all CLI tests
- `cargo run -- --help` displays all options grouped logically
- `cargo run -- -h host1 -h host2 -p 6380 -n 1000` parses correctly

---

### Task 1.5: Create Configuration Structures

**Description:** Transform CLI arguments into validated configuration structures.

**Instructions:**
Create `src/config/benchmark_config.rs`:

```rust
//! Benchmark configuration derived from CLI arguments

use super::cli::{CliArgs, ReadFromReplica, OutputFormat};
use super::search_config::SearchConfig;
use super::tls_config::TlsConfig;
use std::path::PathBuf;
use std::net::SocketAddr;

/// Resolved server address
#[derive(Debug, Clone)]
pub struct ServerAddress {
    pub host: String,
    pub port: u16,
}

impl ServerAddress {
    pub fn to_string(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub password: String,
    pub username: Option<String>,
}

/// Complete benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    // Connection
    pub addresses: Vec<ServerAddress>,
    pub socket_path: Option<PathBuf>,
    pub auth: Option<AuthConfig>,
    pub tls: Option<TlsConfig>,
    pub dbnum: Option<u32>,
    pub connect_timeout_ms: u64,
    pub request_timeout_ms: u64,

    // Parallelism
    pub clients: u32,
    pub threads: u32,
    pub pipeline: u32,

    // Request generation
    pub requests: u64,
    pub duration_secs: Option<u64>,
    pub warmup_requests: u64,
    pub keyspace_len: u64,
    pub sequential: bool,
    pub key_prefix: String,
    pub data_size: usize,
    pub seed: u64,

    // Cluster
    pub cluster_mode: bool,
    pub read_from_replica: ReadFromReplica,
    pub balance_nodes: bool,
    pub balance_tolerance: u32,

    // Rate limiting
    pub requests_per_second: u64,

    // Workload
    pub tests: Vec<String>,
    pub custom_command: Option<String>,

    // Vector search
    pub search_config: Option<SearchConfig>,
    pub dataset_path: Option<PathBuf>,
    pub filtered_search: bool,
    pub num_vectors: u64,
    pub vector_offset: u64,

    // Optimizer
    pub optimize: bool,
    pub target_recall: f64,
    pub target_qps: Option<u64>,
    pub target_p99_ms: Option<u64>,
    pub max_ef_search: u32,

    // Output
    pub output_path: Option<PathBuf>,
    pub output_format: OutputFormat,
    pub csv_output: Option<PathBuf>,
    pub quiet: bool,
    pub verbose: bool,

    // Control
    pub skip_index_create: bool,
    pub skip_load: bool,
    pub cleanup: bool,
}

impl BenchmarkConfig {
    /// Create configuration from CLI arguments
    pub fn from_cli(args: &CliArgs) -> Result<Self, String> {
        // Validate first
        args.validate()?;

        // Build server addresses
        let addresses: Vec<ServerAddress> = args.hosts
            .iter()
            .map(|h| ServerAddress {
                host: h.clone(),
                port: args.port,
            })
            .collect();

        // Build auth config
        let auth = args.password.as_ref().map(|p| AuthConfig {
            password: p.clone(),
            username: args.username.clone(),
        });

        // Build TLS config
        let tls = if args.tls {
            Some(TlsConfig {
                skip_verify: args.tls_skip_verify,
                ca_cert: args.tls_ca_cert.clone(),
                client_cert: args.tls_cert.clone(),
                client_key: args.tls_key.clone(),
                sni: args.tls_sni.clone(),
            })
        } else {
            None
        };

        // Build search config if needed
        let search_config = if args.dataset.is_some() || args.tests.as_ref().map(|t| {
            t.iter().any(|s| s.to_lowercase().contains("vec"))
        }).unwrap_or(false) {
            Some(SearchConfig::from_cli(args))
        } else {
            None
        };

        // Determine tests to run
        let tests = args.tests.clone().unwrap_or_else(|| vec!["ping".to_string()]);

        // Effective thread count
        let threads = args.effective_threads();

        // Effective keyspace
        let keyspace_len = args.effective_keyspace();

        Ok(Self {
            addresses,
            socket_path: args.socket.clone(),
            auth,
            tls,
            dbnum: args.dbnum,
            connect_timeout_ms: args.connect_timeout_ms,
            request_timeout_ms: args.request_timeout_ms,

            clients: args.clients,
            threads,
            pipeline: args.pipeline,

            requests: args.requests,
            duration_secs: args.duration_secs,
            warmup_requests: args.warmup_requests,
            keyspace_len,
            sequential: args.sequential,
            key_prefix: args.key_prefix.clone(),
            data_size: args.data_size,
            seed: args.seed,

            cluster_mode: args.cluster_mode,
            read_from_replica: args.read_from_replica,
            balance_nodes: args.balance_nodes,
            balance_tolerance: args.balance_tolerance,

            requests_per_second: args.requests_per_second,

            tests,
            custom_command: args.custom_command.clone(),

            search_config,
            dataset_path: args.dataset.clone(),
            filtered_search: args.filtered_search,
            num_vectors: args.num_vectors,
            vector_offset: args.vector_offset,

            optimize: args.optimize,
            target_recall: args.target_recall,
            target_qps: args.target_qps,
            target_p99_ms: args.target_p99_ms,
            max_ef_search: args.max_ef_search,

            output_path: args.output.clone(),
            output_format: args.output_format,
            csv_output: args.csv_output.clone(),
            quiet: args.quiet,
            verbose: args.verbose,

            skip_index_create: args.skip_index_create,
            skip_load: args.skip_load,
            cleanup: args.cleanup,
        })
    }

    /// Get clients per thread
    pub fn clients_per_thread(&self) -> u32 {
        (self.clients + self.threads - 1) / self.threads
    }
}
```

Create `src/config/search_config.rs`:

```rust
//! Vector search configuration

use super::cli::{CliArgs, VectorAlgorithm, DistanceMetric};

/// Vector search configuration
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub index_name: String,
    pub vector_field: String,
    pub prefix: String,
    pub algorithm: VectorAlgorithm,
    pub distance_metric: DistanceMetric,
    pub dim: u32,
    pub k: u32,
    pub ef_construction: Option<u32>,
    pub hnsw_m: Option<u32>,
    pub ef_search: Option<u32>,
    pub nocontent: bool,
}

impl SearchConfig {
    pub fn from_cli(args: &CliArgs) -> Self {
        Self {
            index_name: args.search_index.clone(),
            vector_field: args.search_vector_field.clone(),
            prefix: args.search_prefix.clone(),
            algorithm: args.search_algorithm,
            distance_metric: args.search_distance,
            dim: args.vector_dim,
            k: args.search_k,
            ef_construction: args.ef_construction,
            hnsw_m: args.hnsw_m,
            ef_search: args.ef_search,
            nocontent: args.nocontent,
        }
    }

    /// Get vector byte length (dim * sizeof(f32))
    pub fn vec_byte_len(&self) -> usize {
        self.dim as usize * std::mem::size_of::<f32>()
    }
}
```

Create `src/config/tls_config.rs`:

```rust
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
```

Create `src/config/mod.rs`:

```rust
//! Configuration module

pub mod cli;
pub mod benchmark_config;
pub mod search_config;
pub mod tls_config;

pub use cli::{CliArgs, ReadFromReplica, VectorAlgorithm, DistanceMetric, OutputFormat};
pub use benchmark_config::{BenchmarkConfig, ServerAddress, AuthConfig};
pub use search_config::SearchConfig;
pub use tls_config::TlsConfig;
```

**Validation:**
- `cargo check` succeeds
- Unit tests pass for config construction

---

### Task 1.6: Create Main Entry Point

**Description:** Set up main.rs with argument parsing, logging, and stub benchmark entry point.

**Instructions:**
Create `src/main.rs`:

```rust
//! valkey-search-benchmark - High-performance benchmark tool for Valkey
//!
//! This tool supports standard Redis/Valkey benchmarks as well as
//! vector search (FT.SEARCH) benchmarks with recall verification.

use anyhow::Result;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

mod config;
mod utils;
mod client;
mod cluster;
mod workload;
mod dataset;
mod benchmark;
mod metrics;
mod optimizer;

use config::{CliArgs, BenchmarkConfig};

fn setup_logging(verbose: bool, quiet: bool) {
    let level = if quiet {
        Level::ERROR
    } else if verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}

fn print_banner(config: &BenchmarkConfig) {
    if config.quiet {
        return;
    }

    println!("valkey-search-benchmark v{}", env!("CARGO_PKG_VERSION"));
    println!("====================================");
    println!("Hosts: {:?}", config.addresses.iter()
        .map(|a| a.to_string())
        .collect::<Vec<_>>());
    println!("Clients: {}, Threads: {}, Pipeline: {}", 
             config.clients, config.threads, config.pipeline);
    println!("Requests: {}", config.requests);
    if config.cluster_mode {
        println!("Cluster mode: enabled, RFR: {:?}", config.read_from_replica);
    }
    if let Some(ref search) = config.search_config {
        println!("Vector search: dim={}, k={}, algo={:?}", 
                 search.dim, search.k, search.algorithm);
    }
    println!("====================================\n");
}

fn run() -> Result<()> {
    // Parse CLI arguments
    let args = CliArgs::parse_args();

    // Setup logging
    setup_logging(args.verbose, args.quiet);

    // Build configuration
    let config = BenchmarkConfig::from_cli(&args)
        .map_err(|e| anyhow::anyhow!("Configuration error: {}", e))?;

    // Print banner
    print_banner(&config);

    // TODO: Run benchmark
    info!("Benchmark would run here with tests: {:?}", config.tests);
    
    println!("\nBenchmark complete (stub)");
    
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        error!("Error: {:#}", e);
        std::process::exit(1);
    }
}
```

Create `src/lib.rs`:

```rust
//! valkey-search-benchmark library

pub mod config;
pub mod utils;
pub mod client;
pub mod cluster;
pub mod workload;
pub mod dataset;
pub mod benchmark;
pub mod metrics;
pub mod optimizer;
```

Create stub `mod.rs` files for each module:

```rust
// src/client/mod.rs
//! Client connection layer

// src/cluster/mod.rs
//! Cluster topology and node management

// src/workload/mod.rs
//! Workload definitions and command templates

// src/dataset/mod.rs
//! Dataset loading and access

// src/benchmark/mod.rs
//! Benchmark orchestration and workers

// src/metrics/mod.rs
//! Metrics collection and reporting

// src/optimizer/mod.rs
//! Parameter optimization
```

**Validation:**
- `cargo build --release` succeeds
- `cargo run -- --help` shows all options
- `cargo run -- -h localhost -p 6379 -n 1000 -t ping` runs without panic (shows stub message)
- `cargo run -- --version` shows version

---

### Task 1.7: Phase 1 Commit and Validation

**Description:** Commit Phase 1 and run full validation.

**Instructions:**
1. Run all tests: `cargo test`
2. Run clippy: `cargo clippy -- -D warnings`
3. Format code: `cargo fmt`
4. Build release: `cargo build --release`
5. Manual validation:
   ```bash
   # Basic invocation
   ./target/release/valkey-search-benchmark --help
   
   # Multiple hosts
   ./target/release/valkey-search-benchmark -h host1 -h host2 -p 6380
   
   # Vector search args
   ./target/release/valkey-search-benchmark --vector-dim 128 -k 10 --ef-search 100
   
   # Full config
   ./target/release/valkey-search-benchmark \
       -h localhost -p 6379 \
       -c 100 --threads 4 -P 10 \
       -n 1000000 \
       --cluster --rfr prefer-replica \
       --search-index idx --vector-dim 128 \
       --optimize --target-recall 0.95
   ```

6. Git commit:
   ```bash
   git add .
   git commit -m "Phase 1: Project skeleton and CLI foundation

   - Initialize project structure with all modules
   - Add Cargo.toml with dependencies
   - Implement comprehensive CLI argument parsing (clap)
   - Create configuration structures (BenchmarkConfig, SearchConfig, TlsConfig)
   - Add custom error types
   - Setup logging with tracing
   - Create main entry point with banner
   
   All C version CLI options are mapped and validated."
   ```

**Validation Checklist:**
- [ ] `cargo build --release` succeeds with no warnings
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` passes with no warnings
- [ ] `--help` output is readable and complete
- [ ] All CLI options from C version are present
- [ ] Configuration validation catches invalid combinations

---

## Phase 2: RESP Protocol and Raw Connection Layer

**Objective:** Implement RESP protocol encoding/decoding and raw TCP connection handling.

**Git Branch:** `phase-2-connections`

### Task 2.1: Implement RESP Encoder

**Description:** Create RESP protocol encoder for command serialization.

**Instructions:**
Create `src/utils/resp.rs`:

```rust
//! RESP (Redis Serialization Protocol) encoder and decoder
//!
//! This module provides zero-copy RESP encoding for commands and
//! streaming RESP decoding for responses.

use std::io::{self, Write, BufRead};

/// RESP value types
#[derive(Debug, Clone, PartialEq)]
pub enum RespValue {
    /// Simple string (+OK\r\n)
    SimpleString(String),
    /// Error (-ERR message\r\n)
    Error(String),
    /// Integer (:1000\r\n)
    Integer(i64),
    /// Bulk string ($6\r\nfoobar\r\n)
    BulkString(Vec<u8>),
    /// Null bulk string ($-1\r\n)
    Null,
    /// Array (*2\r\n...)
    Array(Vec<RespValue>),
}

impl RespValue {
    /// Check if this is an error response
    pub fn is_error(&self) -> bool {
        matches!(self, RespValue::Error(_))
    }

    /// Check if this is a MOVED error
    pub fn is_moved(&self) -> bool {
        match self {
            RespValue::Error(e) => e.starts_with("MOVED"),
            _ => false,
        }
    }

    /// Check if this is an ASK error
    pub fn is_ask(&self) -> bool {
        match self {
            RespValue::Error(e) => e.starts_with("ASK"),
            _ => false,
        }
    }

    /// Parse MOVED/ASK error to extract slot and target
    /// Returns (slot, host, port)
    pub fn parse_redirect(&self) -> Option<(u16, String, u16)> {
        match self {
            RespValue::Error(e) => {
                let parts: Vec<&str> = e.split_whitespace().collect();
                if parts.len() >= 3 && (parts[0] == "MOVED" || parts[0] == "ASK") {
                    let slot: u16 = parts[1].parse().ok()?;
                    let addr_parts: Vec<&str> = parts[2].split(':').collect();
                    if addr_parts.len() == 2 {
                        let host = addr_parts[0].to_string();
                        let port: u16 = addr_parts[1].parse().ok()?;
                        return Some((slot, host, port));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Get as string (for simple string or bulk string)
    pub fn as_str(&self) -> Option<&str> {
        match self {
            RespValue::SimpleString(s) => Some(s),
            RespValue::BulkString(b) => std::str::from_utf8(b).ok(),
            _ => None,
        }
    }

    /// Get as bytes (for bulk string)
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            RespValue::BulkString(b) => Some(b),
            _ => None,
        }
    }

    /// Get as integer
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            RespValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as array
    pub fn as_array(&self) -> Option<&[RespValue]> {
        match self {
            RespValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

/// RESP encoder with pre-allocated buffer
pub struct RespEncoder {
    buf: Vec<u8>,
}

impl RespEncoder {
    /// Create new encoder with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Clear buffer for reuse
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Get encoded bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Take ownership of buffer
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Get mutable reference to internal buffer
    pub fn buffer_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buf
    }

    /// Encode a command as RESP array
    /// Each argument is encoded as a bulk string
    pub fn encode_command(&mut self, args: &[&[u8]]) {
        // Array header: *<count>\r\n
        self.buf.push(b'*');
        self.write_int(args.len() as i64);
        self.buf.extend_from_slice(b"\r\n");

        // Each argument as bulk string: $<len>\r\n<data>\r\n
        for arg in args {
            self.buf.push(b'$');
            self.write_int(arg.len() as i64);
            self.buf.extend_from_slice(b"\r\n");
            self.buf.extend_from_slice(arg);
            self.buf.extend_from_slice(b"\r\n");
        }
    }

    /// Encode a command from string slices
    pub fn encode_command_str(&mut self, args: &[&str]) {
        let byte_args: Vec<&[u8]> = args.iter().map(|s| s.as_bytes()).collect();
        self.encode_command(&byte_args);
    }

    /// Encode multiple commands (pipeline)
    pub fn encode_pipeline(&mut self, commands: &[Vec<&[u8]>]) {
        for cmd in commands {
            self.encode_command(cmd);
        }
    }

    /// Write integer using fast itoa
    #[inline]
    fn write_int(&mut self, value: i64) {
        itoa::write(&mut self.buf, value).unwrap();
    }

    /// Encode inline command (space-separated, for simple commands)
    pub fn encode_inline(&mut self, args: &[&str]) {
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                self.buf.push(b' ');
            }
            self.buf.extend_from_slice(arg.as_bytes());
        }
        self.buf.extend_from_slice(b"\r\n");
    }
}

/// RESP decoder for streaming reads
pub struct RespDecoder<R> {
    reader: R,
    line_buf: String,
}

impl<R: BufRead> RespDecoder<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_buf: String::with_capacity(256),
        }
    }

    /// Decode next RESP value from stream
    pub fn decode(&mut self) -> io::Result<RespValue> {
        // Read type byte
        self.line_buf.clear();
        self.reader.read_line(&mut self.line_buf)?;

        if self.line_buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Connection closed"
            ));
        }

        let line = self.line_buf.trim_end_matches(&['\r', '\n'][..]);
        if line.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty RESP line"
            ));
        }

        let type_byte = line.as_bytes()[0];
        let content = &line[1..];

        match type_byte {
            b'+' => Ok(RespValue::SimpleString(content.to_string())),
            b'-' => Ok(RespValue::Error(content.to_string())),
            b':' => {
                let value: i64 = content.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid integer")
                })?;
                Ok(RespValue::Integer(value))
            }
            b'$' => {
                let len: i64 = content.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid bulk string length")
                })?;

                if len < 0 {
                    return Ok(RespValue::Null);
                }

                let len = len as usize;
                let mut data = vec![0u8; len];
                self.reader.read_exact(&mut data)?;

                // Read trailing \r\n
                let mut crlf = [0u8; 2];
                self.reader.read_exact(&mut crlf)?;

                Ok(RespValue::BulkString(data))
            }
            b'*' => {
                let count: i64 = content.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid array length")
                })?;

                if count < 0 {
                    return Ok(RespValue::Null);
                }

                let mut elements = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    elements.push(self.decode()?);
                }

                Ok(RespValue::Array(elements))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid RESP type byte: {}", type_byte as char)
            )),
        }
    }

    /// Decode multiple responses (for pipeline)
    pub fn decode_pipeline(&mut self, count: usize) -> io::Result<Vec<RespValue>> {
        let mut responses = Vec::with_capacity(count);
        for _ in 0..count {
            responses.push(self.decode()?);
        }
        Ok(responses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_encode_simple_command() {
        let mut encoder = RespEncoder::with_capacity(64);
        encoder.encode_command_str(&["PING"]);
        assert_eq!(encoder.as_bytes(), b"*1\r\n$4\r\nPING\r\n");
    }

    #[test]
    fn test_encode_set_command() {
        let mut encoder = RespEncoder::with_capacity(64);
        encoder.encode_command_str(&["SET", "key", "value"]);
        assert_eq!(
            encoder.as_bytes(),
            b"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n"
        );
    }

    #[test]
    fn test_decode_simple_string() {
        let data = b"+OK\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(value, RespValue::SimpleString("OK".to_string()));
    }

    #[test]
    fn test_decode_error() {
        let data = b"-ERR unknown command\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(value, RespValue::Error("ERR unknown command".to_string()));
    }

    #[test]
    fn test_decode_integer() {
        let data = b":1000\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(value, RespValue::Integer(1000));
    }

    #[test]
    fn test_decode_bulk_string() {
        let data = b"$6\r\nfoobar\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(value, RespValue::BulkString(b"foobar".to_vec()));
    }

    #[test]
    fn test_decode_array() {
        let data = b"*2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(
            value,
            RespValue::Array(vec![
                RespValue::BulkString(b"foo".to_vec()),
                RespValue::BulkString(b"bar".to_vec()),
            ])
        );
    }

    #[test]
    fn test_parse_moved() {
        let value = RespValue::Error("MOVED 3999 127.0.0.1:7001".to_string());
        let (slot, host, port) = value.parse_redirect().unwrap();
        assert_eq!(slot, 3999);
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 7001);
    }
}
```

Update `src/utils/mod.rs`:

```rust
pub mod error;
pub mod resp;

pub use error::{BenchmarkError, ConnectionError, ProtocolError, DatasetError, ClusterError, Result};
pub use resp::{RespValue, RespEncoder, RespDecoder};
```

**Validation:**
- `cargo test utils::resp` passes all tests
- Encoder output matches expected RESP format
- Decoder handles all RESP types correctly

---

### Task 2.2: Implement Raw TCP Connection

**Description:** Create raw TCP connection wrapper with authentication and timeout support.

**Instructions:**
Create `src/client/raw_connection.rs`:

```rust
//! Raw TCP connection for benchmark traffic
//!
//! This module provides direct TCP (and TLS) connections with
//! pre-allocated buffers for high-performance benchmark traffic.

use std::io::{self, Read, Write, BufReader, BufWriter};
use std::net::TcpStream;
use std::time::Duration;

use crate::config::TlsConfig;
use crate::utils::{RespEncoder, RespDecoder, RespValue, ConnectionError};

/// Raw connection wrapper (TCP or TLS)
pub enum RawConnection {
    Tcp {
        writer: BufWriter<TcpStream>,
        reader: BufReader<TcpStream>,
    },
    #[cfg(feature = "native-tls-backend")]
    NativeTls {
        writer: BufWriter<native_tls::TlsStream<TcpStream>>,
        reader: BufReader<native_tls::TlsStream<TcpStream>>,
    },
}

impl RawConnection {
    /// Create new TCP connection
    pub fn connect_tcp(
        host: &str,
        port: u16,
        connect_timeout: Duration,
    ) -> Result<Self, ConnectionError> {
        let addr = format!("{}:{}", host, port);
        let addr = addr.parse().map_err(|e| ConnectionError::ConnectFailed {
            host: host.to_string(),
            port,
            source: io::Error::new(io::ErrorKind::InvalidInput, e),
        })?;

        let stream = TcpStream::connect_timeout(&addr, connect_timeout)
            .map_err(|e| ConnectionError::ConnectFailed {
                host: host.to_string(),
                port,
                source: e,
            })?;

        // Configure socket
        stream.set_nodelay(true).ok(); // Disable Nagle's algorithm
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

        let writer = BufWriter::with_capacity(65536, stream.try_clone().map_err(|e| {
            ConnectionError::ConnectFailed {
                host: host.to_string(),
                port,
                source: e,
            }
        })?);
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
        use native_tls::{TlsConnector, Certificate, Identity};

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
            let cert = Certificate::from_pem(&ca_data).map_err(|e| {
                ConnectionError::TlsFailed(format!("Invalid CA cert: {}", e))
            })?;
            builder.add_root_certificate(cert);
        }

        // Load client certificate and key
        if let (Some(ref cert_path), Some(ref key_path)) = (&tls_config.client_cert, &tls_config.client_key) {
            let cert_data = std::fs::read(cert_path).map_err(|e| {
                ConnectionError::TlsFailed(format!("Failed to read client cert: {}", e))
            })?;
            let key_data = std::fs::read(key_path).map_err(|e| {
                ConnectionError::TlsFailed(format!("Failed to read client key: {}", e))
            })?;

            // Combine cert and key into PKCS12 (native-tls requirement)
            // For simplicity, we expect PEM format and convert
            let identity = Identity::from_pkcs8(&cert_data, &key_data).map_err(|e| {
                ConnectionError::TlsFailed(format!("Invalid client identity: {}", e))
            })?;
            builder.identity(identity);
        }

        let connector = builder.build().map_err(|e| {
            ConnectionError::TlsFailed(format!("Failed to build TLS connector: {}", e))
        })?;

        // Connect TCP first
        let addr = format!("{}:{}", host, port);
        let addr = addr.parse().map_err(|e| ConnectionError::ConnectFailed {
            host: host.to_string(),
            port,
            source: io::Error::new(io::ErrorKind::InvalidInput, e),
        })?;

        let tcp_stream = TcpStream::connect_timeout(&addr, connect_timeout)
            .map_err(|e| ConnectionError::ConnectFailed {
                host: host.to_string(),
                port,
                source: e,
            })?;

        tcp_stream.set_nodelay(true).ok();

        // TLS handshake
        let sni_host = tls_config.sni.as_deref().unwrap_or(host);
        let tls_stream = connector.connect(sni_host, tcp_stream).map_err(|e| {
            ConnectionError::TlsFailed(format!("TLS handshake failed: {}", e))
        })?;

        let writer = BufWriter::with_capacity(65536, tls_stream.try_clone().map_err(|e| {
            ConnectionError::TlsFailed(format!("Failed to clone TLS stream: {}", e))
        })?);
        let reader = BufReader::with_capacity(65536, tls_stream);

        Ok(RawConnection::NativeTls { writer, reader })
    }

    /// Write bytes to connection
    pub fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match self {
            RawConnection::Tcp { writer, .. } => writer.write_all(buf),
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { writer, .. } => writer.write_all(buf),
        }
    }

    /// Flush write buffer
    pub fn flush(&mut self) -> io::Result<()> {
        match self {
            RawConnection::Tcp { writer, .. } => writer.flush(),
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { writer, .. } => writer.flush(),
        }
    }

    /// Get RESP decoder for this connection
    pub fn decoder(&mut self) -> RespDecoder<&mut dyn std::io::BufRead> {
        match self {
            RawConnection::Tcp { reader, .. } => {
                RespDecoder::new(reader as &mut dyn std::io::BufRead)
            }
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { reader, .. } => {
                RespDecoder::new(reader as &mut dyn std::io::BufRead)
            }
        }
    }

    /// Send command and receive response
    pub fn execute(&mut self, encoder: &RespEncoder) -> io::Result<RespValue> {
        self.write_all(encoder.as_bytes())?;
        self.flush()?;
        self.decoder().decode()
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

        let response = self.execute(&encoder).map_err(|e| {
            ConnectionError::AuthFailed(format!("IO error: {}", e))
        })?;

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

    /// Set read timeout
    pub fn set_read_timeout(&mut self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            RawConnection::Tcp { reader, .. } => {
                reader.get_ref().set_read_timeout(timeout)
            }
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { reader, .. } => {
                reader.get_ref().get_ref().set_read_timeout(timeout)
            }
        }
    }

    /// Set write timeout
    pub fn set_write_timeout(&mut self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            RawConnection::Tcp { writer, .. } => {
                writer.get_ref().set_write_timeout(timeout)
            }
            #[cfg(feature = "native-tls-backend")]
            RawConnection::NativeTls { writer, .. } => {
                writer.get_ref().get_ref().set_write_timeout(timeout)
            }
        }
    }
}

/// Connection factory for creating connections with common config
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
                    "TLS support not compiled in".to_string()
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
            conn.select_db(db).map_err(|e| {
                ConnectionError::ConnectFailed {
                    host: host.to_string(),
                    port,
                    source: e,
                }
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
        let mut conn = RawConnection::connect_tcp(
            "127.0.0.1",
            6379,
            Duration::from_secs(5),
        ).expect("Failed to connect");

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

        let mut conn = factory.create("127.0.0.1", 6379).expect("Failed to connect");
        assert!(conn.ping().expect("Ping failed"));
    }
}
```

Update `src/client/mod.rs`:

```rust
//! Client connection layer

pub mod raw_connection;

pub use raw_connection::{RawConnection, ConnectionFactory};
```

**Validation:**
- `cargo check` succeeds
- Integration tests pass (when server available)
- Code handles TLS feature flags correctly

---

### Task 2.3: Implement Benchmark Client with Pre-allocated Buffers

**Description:** Create high-performance benchmark client with zero-allocation hot path.

**Instructions:**
Create `src/client/benchmark_client.rs`:

```rust
//! Benchmark client with pre-allocated buffers
//!
//! This client is optimized for benchmark traffic with:
//! - Pre-allocated write buffer (command template)
//! - Pre-allocated read buffer (responses)
//! - In-place placeholder replacement
//! - Pipeline support

use std::io::{self, BufRead};
use std::collections::VecDeque;
use std::time::Instant;

use crate::utils::{RespValue, RespDecoder};
use super::raw_connection::RawConnection;

/// Placeholder offset information
#[derive(Debug, Clone)]
pub struct PlaceholderOffset {
    /// Byte offset in write buffer
    pub offset: usize,
    /// Length of placeholder region
    pub len: usize,
    /// Type of placeholder
    pub placeholder_type: PlaceholderType,
}

/// Types of placeholders
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceholderType {
    /// Random or sequential key (fixed-width decimal)
    Key,
    /// Vector data (binary blob)
    Vector,
    /// Cluster routing tag {xxx}
    ClusterTag,
    /// Random integer
    RandInt,
}

/// Pre-computed command template with placeholder offsets
#[derive(Debug, Clone)]
pub struct CommandBuffer {
    /// RESP-encoded command bytes (template)
    pub bytes: Vec<u8>,
    /// Placeholder offsets for each command in pipeline
    pub placeholders: Vec<Vec<PlaceholderOffset>>,
    /// Number of commands in pipeline
    pub pipeline_size: usize,
    /// Bytes per single command (for pipeline offset calculation)
    pub command_len: usize,
}

impl CommandBuffer {
    /// Create a new command buffer from template bytes
    pub fn new(template_bytes: Vec<u8>, pipeline_size: usize) -> Self {
        let command_len = template_bytes.len() / pipeline_size.max(1);
        Self {
            bytes: template_bytes,
            placeholders: Vec::new(),
            pipeline_size,
            command_len,
        }
    }

    /// Register placeholder offset for command at index
    pub fn add_placeholder(&mut self, cmd_idx: usize, offset: PlaceholderOffset) {
        while self.placeholders.len() <= cmd_idx {
            self.placeholders.push(Vec::new());
        }
        self.placeholders[cmd_idx].push(offset);
    }

    /// Get absolute offset for placeholder in command at index
    #[inline]
    pub fn absolute_offset(&self, cmd_idx: usize, relative_offset: usize) -> usize {
        cmd_idx * self.command_len + relative_offset
    }
}

/// Batch response from pipeline execution
#[derive(Debug)]
pub struct BatchResponse {
    /// Response values
    pub values: Vec<RespValue>,
    /// Latency for the batch
    pub latency_us: u64,
    /// Dataset indices for retry (if insert failed)
    pub inflight_indices: Vec<u64>,
    /// Query indices for recall verification
    pub query_indices: Vec<u64>,
}

/// High-performance benchmark client
pub struct BenchmarkClient {
    /// Underlying connection
    conn: RawConnection,
    
    /// Pre-allocated command buffer (write)
    write_buf: CommandBuffer,
    
    /// Pre-allocated read buffer
    read_buf: Vec<u8>,
    
    /// Pipeline depth
    pipeline: usize,
    
    /// Pending responses to read
    pending: usize,
    
    /// Dataset indices currently in flight
    inflight_indices: VecDeque<u64>,
    
    /// Query indices for recall verification
    query_indices: VecDeque<u64>,
    
    /// Assigned node (for cluster routing)
    assigned_node: Option<(String, u16)>,
}

impl BenchmarkClient {
    /// Create new benchmark client from connection and command template
    pub fn new(
        conn: RawConnection,
        command_template: CommandBuffer,
        pipeline: usize,
    ) -> Self {
        Self {
            conn,
            write_buf: command_template,
            read_buf: vec![0u8; 65536], // 64KB read buffer
            pipeline,
            pending: 0,
            inflight_indices: VecDeque::with_capacity(pipeline),
            query_indices: VecDeque::with_capacity(pipeline),
            assigned_node: None,
        }
    }

    /// Set assigned node for this client
    pub fn set_assigned_node(&mut self, host: String, port: u16) {
        self.assigned_node = Some((host, port));
    }

    /// Get assigned node
    pub fn assigned_node(&self) -> Option<&(String, u16)> {
        self.assigned_node.as_ref()
    }

    /// Clear tracking state for new batch
    pub fn clear_batch_state(&mut self) {
        self.inflight_indices.clear();
        self.query_indices.clear();
    }

    /// Replace key placeholder at command index with fixed-width value
    /// 
    /// # Arguments
    /// * `cmd_idx` - Index of command in pipeline (0-based)
    /// * `value` - Key value to write
    /// * `width` - Fixed width for decimal representation
    #[inline]
    pub fn replace_key(&mut self, cmd_idx: usize, value: u64, ph_offset: &PlaceholderOffset) {
        let offset = self.write_buf.absolute_offset(cmd_idx, ph_offset.offset);
        write_fixed_width_u64(&mut self.write_buf.bytes, offset, value, ph_offset.len);
    }

    /// Replace vector placeholder with raw bytes (zero-copy from mmap)
    /// 
    /// # Arguments
    /// * `cmd_idx` - Index of command in pipeline
    /// * `vector_bytes` - Raw vector bytes (directly from mmap)
    #[inline]
    pub fn replace_vector(&mut self, cmd_idx: usize, vector_bytes: &[u8], ph_offset: &PlaceholderOffset) {
        let offset = self.write_buf.absolute_offset(cmd_idx, ph_offset.offset);
        self.write_buf.bytes[offset..offset + vector_bytes.len()]
            .copy_from_slice(vector_bytes);
    }

    /// Replace cluster tag placeholder
    #[inline]
    pub fn replace_cluster_tag(&mut self, cmd_idx: usize, tag: &[u8; 5], ph_offset: &PlaceholderOffset) {
        let offset = self.write_buf.absolute_offset(cmd_idx, ph_offset.offset);
        self.write_buf.bytes[offset..offset + 5].copy_from_slice(tag);
    }

    /// Track inflight dataset index for retry
    #[inline]
    pub fn track_inflight(&mut self, idx: u64) {
        self.inflight_indices.push_back(idx);
    }

    /// Track query index for recall verification
    #[inline]
    pub fn track_query(&mut self, idx: u64) {
        self.query_indices.push_back(idx);
    }

    /// Send the pre-built command buffer
    pub fn send(&mut self) -> io::Result<()> {
        self.conn.write_all(&self.write_buf.bytes)?;
        self.conn.flush()?;
        self.pending = self.pipeline;
        Ok(())
    }

    /// Receive responses for pending commands
    pub fn recv(&mut self) -> io::Result<BatchResponse> {
        let start = Instant::now();
        
        let mut decoder = self.conn.decoder();
        let mut values = Vec::with_capacity(self.pending);
        
        for _ in 0..self.pending {
            values.push(decoder.decode()?);
        }
        
        let latency_us = start.elapsed().as_micros() as u64;
        self.pending = 0;
        
        Ok(BatchResponse {
            values,
            latency_us,
            inflight_indices: self.inflight_indices.drain(..).collect(),
            query_indices: self.query_indices.drain(..).collect(),
        })
    }

    /// Execute batch: send and receive
    pub fn execute_batch(&mut self) -> io::Result<BatchResponse> {
        let start = Instant::now();
        
        self.send()?;
        let mut response = self.recv()?;
        
        // Use total round-trip time
        response.latency_us = start.elapsed().as_micros() as u64;
        
        Ok(response)
    }

    /// Get mutable reference to write buffer for manual manipulation
    pub fn write_buffer_mut(&mut self) -> &mut CommandBuffer {
        &mut self.write_buf
    }

    /// Get reference to write buffer
    pub fn write_buffer(&self) -> &CommandBuffer {
        &self.write_buf
    }
}

/// Write u64 as fixed-width decimal string (zero-padded)
/// 
/// # Arguments
/// * `buf` - Target buffer
/// * `offset` - Starting offset in buffer
/// * `value` - Value to write
/// * `width` - Fixed width (will be zero-padded)
#[inline]
pub fn write_fixed_width_u64(buf: &mut [u8], offset: usize, value: u64, width: usize) {
    let mut v = value;
    for i in (0..width).rev() {
        buf[offset + i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
}

/// Write bytes from source to buffer at offset (single memcpy)
#[inline]
pub fn write_bytes(buf: &mut [u8], offset: usize, src: &[u8]) {
    buf[offset..offset + src.len()].copy_from_slice(src);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_fixed_width_u64() {
        let mut buf = vec![b'X'; 20];
        write_fixed_width_u64(&mut buf, 5, 12345, 10);
        assert_eq!(&buf[5..15], b"0000012345");
    }

    #[test]
    fn test_write_fixed_width_zero() {
        let mut buf = vec![b'X'; 10];
        write_fixed_width_u64(&mut buf, 0, 0, 5);
        assert_eq!(&buf[0..5], b"00000");
    }

    #[test]
    fn test_write_fixed_width_max() {
        let mut buf = vec![b'X'; 20];
        write_fixed_width_u64(&mut buf, 0, 999999999999, 12);
        assert_eq!(&buf[0..12], b"999999999999");
    }
}
```

Update `src/client/mod.rs`:

```rust
//! Client connection layer

pub mod raw_connection;
pub mod benchmark_client;

pub use raw_connection::{RawConnection, ConnectionFactory};
pub use benchmark_client::{
    BenchmarkClient, CommandBuffer, BatchResponse,
    PlaceholderOffset, PlaceholderType,
    write_fixed_width_u64, write_bytes,
};
```

**Validation:**
- `cargo test client::benchmark_client` passes
- Fixed-width integer writing is correct
- Buffer manipulation is zero-copy

---

### Task 2.4: Phase 2 Commit and Validation

**Description:** Commit Phase 2 and run full validation.

**Instructions:**
1. Run all tests: `cargo test`
2. Run clippy: `cargo clippy -- -D warnings`
3. Format code: `cargo fmt`
4. Build release: `cargo build --release`
5. Integration test (requires Valkey server):
   ```bash
   # Start a Valkey server on localhost:6379
   
   # Run ignored tests
   cargo test -- --ignored
   ```

6. Git commit:
   ```bash
   git add .
   git commit -m "Phase 2: RESP protocol and raw connection layer

   - Implement RESP encoder with zero-allocation design
   - Implement RESP decoder with streaming support
   - Create RawConnection for TCP and TLS
   - Add ConnectionFactory with auth and timeout support
   - Create BenchmarkClient with pre-allocated buffers
   - Add fixed-width integer writing for key placeholders
   - Support MOVED/ASK error parsing
   
   Zero-allocation hot path established for benchmark traffic."
   ```

**Validation Checklist:**
- [ ] RESP encoding matches expected format
- [ ] RESP decoding handles all types
- [ ] TCP connection works with timeout
- [ ] TLS connection works (if feature enabled)
- [ ] Authentication succeeds
- [ ] Fixed-width integer writing is correct
- [ ] Pre-allocated buffers are reused correctly

---

## Phase 3: Command Templates and Placeholder System

**Objective:** Implement the command template system with zero-allocation placeholder replacement.

**Git Branch:** `phase-3-templates`

### Task 3.1: Define Workload Types

**Description:** Create workload type enumeration matching all C implementation benchmarks.

**Instructions:**
Create `src/workload/workload_type.rs`:

```rust
//! Workload type definitions

/// Supported benchmark workload types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    // === Standard benchmarks ===
    Ping,
    Set,
    Get,
    Incr,
    Lpush,
    Rpush,
    Lpop,
    Rpop,
    Sadd,
    Spop,
    Hset,
    Zadd,
    Zpopmin,
    Lrange100,
    Lrange300,
    Lrange500,
    Lrange600,
    Mset,
    
    // === Vector search workloads ===
    /// Load vectors with HSET
    VecLoad,
    /// Query vectors with FT.SEARCH
    VecQuery,
    /// Delete vector keys
    VecDelete,
    /// Update existing vectors
    VecUpdate,
    
    // === Custom command ===
    Custom,
}

impl WorkloadType {
    /// Parse workload type from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ping" => Some(Self::Ping),
            "set" => Some(Self::Set),
            "get" => Some(Self::Get),
            "incr" => Some(Self::Incr),
            "lpush" => Some(Self::Lpush),
            "rpush" => Some(Self::Rpush),
            "lpop" => Some(Self::Lpop),
            "rpop" => Some(Self::Rpop),
            "sadd" => Some(Self::Sadd),
            "spop" => Some(Self::Spop),
            "hset" => Some(Self::Hset),
            "zadd" => Some(Self::Zadd),
            "zpopmin" => Some(Self::Zpopmin),
            "lrange" | "lrange_100" | "lrange100" => Some(Self::Lrange100),
            "lrange_300" | "lrange300" => Some(Self::Lrange300),
            "lrange_500" | "lrange500" => Some(Self::Lrange500),
            "lrange_600" | "lrange600" => Some(Self::Lrange600),
            "mset" => Some(Self::Mset),
            "vecload" | "vec-load" | "vec_load" => Some(Self::VecLoad),
            "vecquery" | "vec-query" | "vec_query" => Some(Self::VecQuery),
            "vecdelete" | "vec-delete" | "vec_delete" => Some(Self::VecDelete),
            "vecupdate" | "vec-update" | "vec_update" => Some(Self::VecUpdate),
            _ => None,
        }
    }

    /// Get display name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ping => "PING",
            Self::Set => "SET",
            Self::Get => "GET",
            Self::Incr => "INCR",
            Self::Lpush => "LPUSH",
            Self::Rpush => "RPUSH",
            Self::Lpop => "LPOP",
            Self::Rpop => "RPOP",
            Self::Sadd => "SADD",
            Self::Spop => "SPOP",
            Self::Hset => "HSET",
            Self::Zadd => "ZADD",
            Self::Zpopmin => "ZPOPMIN",
            Self::Lrange100 => "LRANGE_100",
            Self::Lrange300 => "LRANGE_300",
            Self::Lrange500 => "LRANGE_500",
            Self::Lrange600 => "LRANGE_600",
            Self::Mset => "MSET",
            Self::VecLoad => "VECLOAD",
            Self::VecQuery => "VECQUERY",
            Self::VecDelete => "VECDELETE",
            Self::VecUpdate => "VECUPDATE",
            Self::Custom => "CUSTOM",
        }
    }

    /// Check if workload requires dataset
    pub fn requires_dataset(&self) -> bool {
        matches!(self, Self::VecLoad | Self::VecQuery | Self::VecUpdate)
    }

    /// Check if workload is a vector search operation
    pub fn is_vector_search(&self) -> bool {
        matches!(self, Self::VecLoad | Self::VecQuery | Self::VecDelete | Self::VecUpdate)
    }

    /// Check if workload modifies data (for read-from-replica routing)
    pub fn is_write(&self) -> bool {
        !matches!(self, Self::Ping | Self::Get | Self::VecQuery | Self::Lrange100 | 
                  Self::Lrange300 | Self::Lrange500 | Self::Lrange600)
    }
}

impl std::fmt::Display for WorkloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_workload_types() {
        assert_eq!(WorkloadType::from_str("ping"), Some(WorkloadType::Ping));
        assert_eq!(WorkloadType::from_str("PING"), Some(WorkloadType::Ping));
        assert_eq!(WorkloadType::from_str("vecload"), Some(WorkloadType::VecLoad));
        assert_eq!(WorkloadType::from_str("vec-load"), Some(WorkloadType::VecLoad));
        assert_eq!(WorkloadType::from_str("unknown"), None);
    }

    #[test]
    fn test_requires_dataset() {
        assert!(WorkloadType::VecLoad.requires_dataset());
        assert!(WorkloadType::VecQuery.requires_dataset());
        assert!(!WorkloadType::Ping.requires_dataset());
        assert!(!WorkloadType::Set.requires_dataset());
    }
}
```

**Validation:**
- All workload types from C version are defined
- Parsing is case-insensitive
- Dataset requirements are correctly flagged

---

### Task 3.2: Implement Command Template Builder

**Description:** Create command template builder that generates RESP-encoded commands with placeholder markers.

**Instructions:**
Create `src/workload/command_template.rs`:

```rust
//! Command template builder with placeholder support
//!
//! This module creates RESP-encoded command templates where placeholder
//! regions are marked for in-place replacement during the benchmark.

use crate::client::{CommandBuffer, PlaceholderOffset, PlaceholderType};
use crate::config::SearchConfig;
use crate::utils::RespEncoder;

/// Template argument (literal or placeholder)
#[derive(Debug, Clone)]
pub enum TemplateArg {
    /// Literal bytes (copied as-is)
    Literal(Vec<u8>),
    /// Placeholder with type and reserved length
    Placeholder {
        ph_type: PlaceholderType,
        len: usize,
    },
}

/// Command template definition
#[derive(Debug, Clone)]
pub struct CommandTemplate {
    /// Template arguments
    args: Vec<TemplateArg>,
    /// Command name for display
    name: String,
}

impl CommandTemplate {
    /// Create new command template
    pub fn new(name: &str) -> Self {
        Self {
            args: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Add literal argument
    pub fn arg_literal(mut self, value: &[u8]) -> Self {
        self.args.push(TemplateArg::Literal(value.to_vec()));
        self
    }

    /// Add string literal argument
    pub fn arg_str(mut self, value: &str) -> Self {
        self.args.push(TemplateArg::Literal(value.as_bytes().to_vec()));
        self
    }

    /// Add key placeholder (fixed-width decimal)
    pub fn arg_key(mut self, width: usize) -> Self {
        self.args.push(TemplateArg::Placeholder {
            ph_type: PlaceholderType::Key,
            len: width,
        });
        self
    }

    /// Add vector placeholder (binary blob)
    pub fn arg_vector(mut self, byte_len: usize) -> Self {
        self.args.push(TemplateArg::Placeholder {
            ph_type: PlaceholderType::Vector,
            len: byte_len,
        });
        self
    }

    /// Add cluster tag placeholder
    pub fn arg_cluster_tag(mut self) -> Self {
        self.args.push(TemplateArg::Placeholder {
            ph_type: PlaceholderType::ClusterTag,
            len: 5, // {xxx}
        });
        self
    }

    /// Add random integer placeholder
    pub fn arg_rand_int(mut self, width: usize) -> Self {
        self.args.push(TemplateArg::Placeholder {
            ph_type: PlaceholderType::RandInt,
            len: width,
        });
        self
    }

    /// Build RESP-encoded command buffer for given pipeline size
    pub fn build(&self, pipeline: usize) -> CommandBuffer {
        let single_cmd = self.encode_single();
        let command_len = single_cmd.len();
        
        // Repeat for pipeline
        let mut bytes = Vec::with_capacity(command_len * pipeline);
        for _ in 0..pipeline {
            bytes.extend_from_slice(&single_cmd);
        }
        
        // Calculate placeholder offsets for each command in pipeline
        let offsets = self.compute_offsets();
        let mut buffer = CommandBuffer::new(bytes, pipeline);
        
        for cmd_idx in 0..pipeline {
            for offset in &offsets {
                buffer.add_placeholder(cmd_idx, PlaceholderOffset {
                    offset: offset.offset,
                    len: offset.len,
                    placeholder_type: offset.placeholder_type,
                });
            }
        }
        
        buffer
    }

    /// Encode single command to RESP
    fn encode_single(&self) -> Vec<u8> {
        let mut encoder = RespEncoder::with_capacity(1024);
        let buf = encoder.buffer_mut();
        
        // Array header: *<count>\r\n
        buf.push(b'*');
        itoa::write(&mut *buf, self.args.len()).unwrap();
        buf.extend_from_slice(b"\r\n");
        
        // Each argument
        for arg in &self.args {
            match arg {
                TemplateArg::Literal(data) => {
                    buf.push(b'$');
                    itoa::write(&mut *buf, data.len()).unwrap();
                    buf.extend_from_slice(b"\r\n");
                    buf.extend_from_slice(data);
                    buf.extend_from_slice(b"\r\n");
                }
                TemplateArg::Placeholder { len, .. } => {
                    buf.push(b'$');
                    itoa::write(&mut *buf, *len).unwrap();
                    buf.extend_from_slice(b"\r\n");
                    // Fill with zeros (will be overwritten)
                    buf.resize(buf.len() + *len, b'0');
                    buf.extend_from_slice(b"\r\n");
                }
            }
        }
        
        encoder.into_bytes()
    }

    /// Compute placeholder offsets in encoded command
    fn compute_offsets(&self) -> Vec<PlaceholderOffset> {
        let mut offsets = Vec::new();
        let mut pos = 0;
        
        // Skip array header
        pos += 1; // *
        pos += self.args.len().to_string().len(); // count digits
        pos += 2; // \r\n
        
        for arg in &self.args {
            match arg {
                TemplateArg::Literal(data) => {
                    pos += 1; // $
                    pos += data.len().to_string().len(); // length digits
                    pos += 2; // \r\n
                    pos += data.len(); // data
                    pos += 2; // \r\n
                }
                TemplateArg::Placeholder { ph_type, len } => {
                    pos += 1; // $
                    pos += len.to_string().len(); // length digits
                    pos += 2; // \r\n
                    
                    // Record offset at start of placeholder data
                    offsets.push(PlaceholderOffset {
                        offset: pos,
                        len: *len,
                        placeholder_type: *ph_type,
                    });
                    
                    pos += *len; // placeholder data
                    pos += 2; // \r\n
                }
            }
        }
        
        offsets
    }
    
    /// Get command name
    pub fn name(&self) -> &str {
        &self.name
    }
}

// === Standard Command Templates ===

/// PING command
pub fn ping_template() -> CommandTemplate {
    CommandTemplate::new("PING")
        .arg_str("PING")
}

/// SET key value command
pub fn set_template(key_prefix: &str, key_width: usize, value_size: usize) -> CommandTemplate {
    let mut template = CommandTemplate::new("SET")
        .arg_str("SET");
    
    // Key: prefix + key placeholder
    if !key_prefix.is_empty() {
        let mut key_arg = key_prefix.as_bytes().to_vec();
        key_arg.resize(key_arg.len() + key_width, b'0');
        template.args.push(TemplateArg::Literal(key_arg));
        // We need to track this as having a placeholder region
        // This is a simplification - real impl would use composite key
    } else {
        template = template.arg_key(key_width);
    }
    
    // Value: fixed-size data
    template.arg_literal(&vec![b'x'; value_size])
}

/// GET key command
pub fn get_template(key_prefix: &str, key_width: usize) -> CommandTemplate {
    CommandTemplate::new("GET")
        .arg_str("GET")
        .arg_key(key_width)
}

/// INCR key command
pub fn incr_template(key_width: usize) -> CommandTemplate {
    CommandTemplate::new("INCR")
        .arg_str("INCR")
        .arg_key(key_width)
}

/// LPUSH key value command
pub fn lpush_template(key_width: usize, value_size: usize) -> CommandTemplate {
    CommandTemplate::new("LPUSH")
        .arg_str("LPUSH")
        .arg_key(key_width)
        .arg_literal(&vec![b'x'; value_size])
}

/// RPUSH key value command
pub fn rpush_template(key_width: usize, value_size: usize) -> CommandTemplate {
    CommandTemplate::new("RPUSH")
        .arg_str("RPUSH")
        .arg_key(key_width)
        .arg_literal(&vec![b'x'; value_size])
}

/// LPOP key command
pub fn lpop_template(key_width: usize) -> CommandTemplate {
    CommandTemplate::new("LPOP")
        .arg_str("LPOP")
        .arg_key(key_width)
}

/// RPOP key command
pub fn rpop_template(key_width: usize) -> CommandTemplate {
    CommandTemplate::new("RPOP")
        .arg_str("RPOP")
        .arg_key(key_width)
}

/// SADD key member command
pub fn sadd_template(key_width: usize, value_size: usize) -> CommandTemplate {
    CommandTemplate::new("SADD")
        .arg_str("SADD")
        .arg_key(key_width)
        .arg_literal(&vec![b'x'; value_size])
}

/// SPOP key command
pub fn spop_template(key_width: usize) -> CommandTemplate {
    CommandTemplate::new("SPOP")
        .arg_str("SPOP")
        .arg_key(key_width)
}

/// HSET key field value command
pub fn hset_template(key_width: usize, value_size: usize) -> CommandTemplate {
    CommandTemplate::new("HSET")
        .arg_str("HSET")
        .arg_key(key_width)
        .arg_str("field")
        .arg_literal(&vec![b'x'; value_size])
}

/// ZADD key score member command
pub fn zadd_template(key_width: usize, value_size: usize) -> CommandTemplate {
    CommandTemplate::new("ZADD")
        .arg_str("ZADD")
        .arg_key(key_width)
        .arg_str("0") // score
        .arg_literal(&vec![b'x'; value_size])
}

/// ZPOPMIN key command
pub fn zpopmin_template(key_width: usize) -> CommandTemplate {
    CommandTemplate::new("ZPOPMIN")
        .arg_str("ZPOPMIN")
        .arg_key(key_width)
}

/// LRANGE key 0 N command
pub fn lrange_template(key_width: usize, count: usize) -> CommandTemplate {
    CommandTemplate::new(&format!("LRANGE_{}", count))
        .arg_str("LRANGE")
        .arg_key(key_width)
        .arg_str("0")
        .arg_str(&(count - 1).to_string())
}

/// MSET key1 val1 key2 val2 ... (10 pairs)
pub fn mset_template(key_width: usize, value_size: usize) -> CommandTemplate {
    let mut template = CommandTemplate::new("MSET")
        .arg_str("MSET");
    
    for _ in 0..10 {
        template = template
            .arg_key(key_width)
            .arg_literal(&vec![b'x'; value_size]);
    }
    
    template
}

// === Vector Search Templates ===

/// HSET for vector load: HSET {tag}prefix:key field vector_bytes
pub fn vec_load_template(prefix: &str, field: &str, key_width: usize, vec_byte_len: usize, with_cluster_tag: bool) -> CommandTemplate {
    let mut template = CommandTemplate::new("VECLOAD");
    template = template.arg_str("HSET");
    
    if with_cluster_tag {
        // Key with cluster tag: {xxx}prefix:0000000000
        // We'll handle this as a composite - simplified here
        template = template
            .arg_cluster_tag()
            .arg_key(key_width);
    } else {
        template = template.arg_key(key_width);
    }
    
    template
        .arg_str(field)
        .arg_vector(vec_byte_len)
}

/// FT.SEARCH for vector query
pub fn vec_query_template(config: &SearchConfig) -> CommandTemplate {
    let vec_byte_len = config.vec_byte_len();
    
    // Build query string: *=>[KNN k @field $BLOB EF_RUNTIME ef]
    let query = if let Some(ef) = config.ef_search {
        format!("*=>[KNN {} @{} $BLOB EF_RUNTIME {}]", config.k, config.vector_field, ef)
    } else {
        format!("*=>[KNN {} @{} $BLOB]", config.k, config.vector_field)
    };
    
    let mut template = CommandTemplate::new("VECQUERY")
        .arg_str("FT.SEARCH")
        .arg_str(&config.index_name)
        .arg_str(&query)
        .arg_str("PARAMS")
        .arg_str("2")
        .arg_str("BLOB")
        .arg_vector(vec_byte_len);
    
    if config.nocontent {
        template = template.arg_str("NOCONTENT");
    }
    
    template
        .arg_str("LIMIT")
        .arg_str("0")
        .arg_str(&config.k.to_string())
        .arg_str("DIALECT")
        .arg_str("2")
}

/// DEL key for vector delete
pub fn vec_delete_template(key_width: usize) -> CommandTemplate {
    CommandTemplate::new("VECDELETE")
        .arg_str("DEL")
        .arg_key(key_width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_template() {
        let template = ping_template();
        let buffer = template.build(1);
        assert_eq!(buffer.bytes, b"*1\r\n$4\r\nPING\r\n");
    }

    #[test]
    fn test_set_template_encoding() {
        let template = CommandTemplate::new("TEST")
            .arg_str("SET")
            .arg_key(10)
            .arg_literal(b"value");
        
        let buffer = template.build(1);
        
        // Check structure
        assert!(buffer.bytes.starts_with(b"*3\r\n"));
        assert!(buffer.bytes.contains(&b"$3\r\nSET\r\n"[..]));
    }

    #[test]
    fn test_placeholder_offsets() {
        let template = CommandTemplate::new("TEST")
            .arg_str("SET")
            .arg_key(5)
            .arg_str("val");
        
        let buffer = template.build(1);
        
        // Should have one key placeholder
        assert_eq!(buffer.placeholders.len(), 1);
        assert_eq!(buffer.placeholders[0].len(), 1);
        assert_eq!(buffer.placeholders[0][0].placeholder_type, PlaceholderType::Key);
        assert_eq!(buffer.placeholders[0][0].len, 5);
    }

    #[test]
    fn test_pipeline_buffer() {
        let template = ping_template();
        let buffer = template.build(3);
        
        // Should be 3x the single command
        let single = b"*1\r\n$4\r\nPING\r\n";
        assert_eq!(buffer.bytes.len(), single.len() * 3);
        assert_eq!(buffer.pipeline_size, 3);
    }
}
```

**Validation:**
- Templates generate valid RESP encoding
- Placeholder offsets are computed correctly
- Pipeline multiplication works

---

### Task 3.3: Implement Template Factory

**Description:** Create factory function to build templates based on workload type.

**Instructions:**
Create `src/workload/template_factory.rs`:

```rust
//! Template factory for creating command templates from workload types

use crate::config::{BenchmarkConfig, SearchConfig};
use super::workload_type::WorkloadType;
use super::command_template::*;

/// Key width for different keyspace sizes
pub fn key_width_for_keyspace(keyspace: u64) -> usize {
    if keyspace == 0 {
        12 // Default: up to 999,999,999,999
    } else if keyspace < 1_000_000 {
        6
    } else if keyspace < 1_000_000_000 {
        9
    } else {
        12
    }
}

/// Create command template for workload type
pub fn create_template(
    workload: WorkloadType,
    config: &BenchmarkConfig,
) -> Option<CommandTemplate> {
    let key_width = key_width_for_keyspace(config.keyspace_len);
    let value_size = config.data_size;
    
    match workload {
        WorkloadType::Ping => Some(ping_template()),
        
        WorkloadType::Set => Some(
            CommandTemplate::new("SET")
                .arg_str("SET")
                .arg_key(key_width)
                .arg_literal(&vec![b'x'; value_size])
        ),
        
        WorkloadType::Get => Some(
            CommandTemplate::new("GET")
                .arg_str("GET")
                .arg_key(key_width)
        ),
        
        WorkloadType::Incr => Some(incr_template(key_width)),
        
        WorkloadType::Lpush => Some(lpush_template(key_width, value_size)),
        WorkloadType::Rpush => Some(rpush_template(key_width, value_size)),
        WorkloadType::Lpop => Some(lpop_template(key_width)),
        WorkloadType::Rpop => Some(rpop_template(key_width)),
        
        WorkloadType::Sadd => Some(sadd_template(key_width, value_size)),
        WorkloadType::Spop => Some(spop_template(key_width)),
        
        WorkloadType::Hset => Some(hset_template(key_width, value_size)),
        
        WorkloadType::Zadd => Some(zadd_template(key_width, value_size)),
        WorkloadType::Zpopmin => Some(zpopmin_template(key_width)),
        
        WorkloadType::Lrange100 => Some(lrange_template(key_width, 100)),
        WorkloadType::Lrange300 => Some(lrange_template(key_width, 300)),
        WorkloadType::Lrange500 => Some(lrange_template(key_width, 500)),
        WorkloadType::Lrange600 => Some(lrange_template(key_width, 600)),
        
        WorkloadType::Mset => Some(mset_template(key_width, value_size)),
        
        WorkloadType::VecLoad => {
            let search = config.search_config.as_ref()?;
            Some(vec_load_template(
                &search.prefix,
                &search.vector_field,
                key_width,
                search.vec_byte_len(),
                config.cluster_mode,
            ))
        }
        
        WorkloadType::VecQuery => {
            let search = config.search_config.as_ref()?;
            Some(vec_query_template(search))
        }
        
        WorkloadType::VecDelete => Some(vec_delete_template(key_width)),
        
        WorkloadType::VecUpdate => {
            let search = config.search_config.as_ref()?;
            Some(vec_load_template(
                &search.prefix,
                &search.vector_field,
                key_width,
                search.vec_byte_len(),
                config.cluster_mode,
            ))
        }
        
        WorkloadType::Custom => None, // Custom templates handled separately
    }
}

/// Parse custom command string into template
/// Format: "CMD arg1 arg2 __rand_key__ arg3 ..."
pub fn parse_custom_command(cmd: &str, keyspace: u64) -> Option<CommandTemplate> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }
    
    let key_width = key_width_for_keyspace(keyspace);
    let mut template = CommandTemplate::new(parts[0]);
    
    for part in parts {
        if part == "__rand_key__" || part == "__key__" {
            template = template.arg_key(key_width);
        } else if part == "__rand_int__" {
            template = template.arg_rand_int(12);
        } else {
            template = template.arg_str(part);
        }
    }
    
    Some(template)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::cli::CliArgs;
    use clap::Parser;

    fn default_config() -> BenchmarkConfig {
        let args = CliArgs::parse_from(["test"]);
        BenchmarkConfig::from_cli(&args).unwrap()
    }

    #[test]
    fn test_create_standard_templates() {
        let config = default_config();
        
        assert!(create_template(WorkloadType::Ping, &config).is_some());
        assert!(create_template(WorkloadType::Set, &config).is_some());
        assert!(create_template(WorkloadType::Get, &config).is_some());
    }

    #[test]
    fn test_key_width() {
        assert_eq!(key_width_for_keyspace(100), 6);
        assert_eq!(key_width_for_keyspace(1_000_000), 9);
        assert_eq!(key_width_for_keyspace(10_000_000_000), 12);
    }

    #[test]
    fn test_parse_custom_command() {
        let template = parse_custom_command("GET __rand_key__", 1000).unwrap();
        let buffer = template.build(1);
        
        assert!(buffer.bytes.starts_with(b"*2\r\n"));
    }
}
```

Update `src/workload/mod.rs`:

```rust
//! Workload definitions and command templates

pub mod workload_type;
pub mod command_template;
pub mod template_factory;

pub use workload_type::WorkloadType;
pub use command_template::{CommandTemplate, TemplateArg};
pub use template_factory::{create_template, parse_custom_command, key_width_for_keyspace};
```

**Validation:**
- All workload types have templates
- Custom command parsing works
- Key width calculation is correct

---

### Task 3.4: Phase 3 Commit and Validation

**Description:** Commit Phase 3 and run full validation.

**Instructions:**
1. Run all tests: `cargo test`
2. Run clippy: `cargo clippy -- -D warnings`
3. Format code: `cargo fmt`
4. Build release: `cargo build --release`
5. Manual verification:
   ```rust
   // Add temporary test in main.rs to verify templates
   let template = ping_template();
   let buffer = template.build(1);
   println!("PING: {:?}", String::from_utf8_lossy(&buffer.bytes));
   ```

6. Git commit:
   ```bash
   git add .
   git commit -m "Phase 3: Command templates and placeholder system

   - Define all workload types (standard + vector search)
   - Implement CommandTemplate builder with RESP encoding
   - Create PlaceholderOffset tracking for in-place replacement
   - Add template factory for all workload types
   - Support custom command parsing with placeholders
   - Add key width calculation based on keyspace size
   
   Zero-allocation placeholder system ready for benchmark traffic."
   ```

**Validation Checklist:**
- [ ] All standard workloads have templates
- [ ] Vector search templates generate correct FT.SEARCH syntax
- [ ] Placeholder offsets are computed correctly
- [ ] Pipeline buffers are correctly sized
- [ ] Custom command parsing handles __rand_key__

---

## Phase 4: Threading Model and Worker Implementation

**Objective:** Implement multi-threaded benchmark execution with thread-independent workers.

**Git Branch:** `phase-4-workers`

### Task 4.1: Implement Global Counters

**Description:** Create atomic counters for cross-thread synchronization.

**Instructions:**
Create `src/benchmark/counters.rs`:

```rust
//! Global atomic counters for thread synchronization
//!
//! These are the ONLY synchronization points between worker threads.
//! All other state is thread-local.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Global counters shared between all worker threads
/// 
/// Design principle: Minimize contention by using relaxed ordering
/// where possible and keeping counter operations simple (fetch_add).
pub struct GlobalCounters {
    /// Total requests issued (claimed by workers)
    pub requests_issued: AtomicU64,
    
    /// Total requests completed (responses received)
    pub requests_finished: AtomicU64,
    
    /// Sequential key counter (for --sequential mode)
    pub seq_key_counter: AtomicU64,
    
    /// Dataset vector counter (for unique vector insertion)
    pub dataset_counter: AtomicU64,
    
    /// Query vector counter (for sequential query iteration)
    pub query_counter: AtomicU64,
    
    /// Total errors encountered
    pub error_count: AtomicU64,
    
    /// Shutdown signal
    pub shutdown: AtomicBool,
}

impl GlobalCounters {
    /// Create new counters initialized to zero
    pub fn new() -> Self {
        Self {
            requests_issued: AtomicU64::new(0),
            requests_finished: AtomicU64::new(0),
            seq_key_counter: AtomicU64::new(0),
            dataset_counter: AtomicU64::new(0),
            query_counter: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Claim a batch of requests
    /// Returns the starting request number, or None if quota exhausted
    #[inline]
    pub fn claim_requests(&self, batch_size: u64, total_requests: u64) -> Option<u64> {
        let issued = self.requests_issued.fetch_add(batch_size, Ordering::Relaxed);
        if issued >= total_requests {
            // Undo the claim
            self.requests_issued.fetch_sub(batch_size, Ordering::Relaxed);
            None
        } else {
            Some(issued)
        }
    }

    /// Record completed requests
    #[inline]
    pub fn record_finished(&self, count: u64) {
        self.requests_finished.fetch_add(count, Ordering::Relaxed);
    }

    /// Get next sequential key value
    #[inline]
    pub fn next_seq_key(&self, keyspace: u64) -> u64 {
        self.seq_key_counter.fetch_add(1, Ordering::Relaxed) % keyspace
    }

    /// Claim next dataset index for vector insertion
    #[inline]
    pub fn next_dataset_idx(&self) -> u64 {
        self.dataset_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Get next query index
    #[inline]
    pub fn next_query_idx(&self, num_queries: u64) -> u64 {
        self.query_counter.fetch_add(1, Ordering::Relaxed) % num_queries
    }

    /// Record an error
    #[inline]
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Signal shutdown to all workers
    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Check if shutdown has been signaled
    #[inline]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Get current progress
    pub fn progress(&self) -> (u64, u64) {
        (
            self.requests_finished.load(Ordering::Relaxed),
            self.requests_issued.load(Ordering::Relaxed),
        )
    }

    /// Get error count
    pub fn errors(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Reset all counters (for warmup -> measurement transition)
    pub fn reset(&self) {
        self.requests_issued.store(0, Ordering::SeqCst);
        self.requests_finished.store(0, Ordering::SeqCst);
        self.error_count.store(0, Ordering::SeqCst);
        // Note: Don't reset seq_key_counter, dataset_counter, query_counter
        // as those track cumulative position
    }
}

impl Default for GlobalCounters {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_claim_requests() {
        let counters = GlobalCounters::new();
        
        assert_eq!(counters.claim_requests(10, 100), Some(0));
        assert_eq!(counters.claim_requests(10, 100), Some(10));
        
        // Claim past limit
        assert_eq!(counters.claim_requests(100, 100), None);
    }

    #[test]
    fn test_concurrent_claims() {
        let counters = Arc::new(GlobalCounters::new());
        let total = 1000u64;
        let batch = 10u64;
        
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = Arc::clone(&counters);
                thread::spawn(move || {
                    let mut claimed = 0u64;
                    while c.claim_requests(batch, total).is_some() {
                        claimed += batch;
                    }
                    claimed
                })
            })
            .collect();
        
        let total_claimed: u64 = handles.into_iter()
            .map(|h| h.join().unwrap())
            .sum();
        
        // Should claim exactly the total (some threads may overclaim slightly)
        assert!(total_claimed >= total);
        assert!(total_claimed <= total + batch * 4);
    }

    #[test]
    fn test_shutdown_signal() {
        let counters = GlobalCounters::new();
        
        assert!(!counters.is_shutdown());
        counters.signal_shutdown();
        assert!(counters.is_shutdown());
    }
}
```

**Validation:**
- Atomic operations work correctly
- Concurrent claims are safe
- Shutdown signal propagates

---

### Task 4.2: Implement Worker Thread

**Description:** Create benchmark worker that owns its clients and runs independently.

**Instructions:**
Create `src/benchmark/worker.rs`:

```rust
//! Benchmark worker thread implementation
//!
//! Each worker owns its clients exclusively. The only synchronization
//! points are atomic counters for request claiming and progress tracking.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;

use crate::client::{BenchmarkClient, CommandBuffer, PlaceholderType};
use crate::config::BenchmarkConfig;
use crate::dataset::DatasetContext;
use super::counters::GlobalCounters;

/// Result from a worker thread
pub struct WorkerResult {
    /// Worker ID
    pub worker_id: usize,
    /// Local histogram of latencies (microseconds)
    pub histogram: Histogram<u64>,
    /// Recall statistics (for vector queries)
    pub recall_stats: RecallStats,
    /// Number of retries needed
    pub retry_count: u64,
    /// Number of errors
    pub error_count: u64,
    /// Total requests processed
    pub requests_processed: u64,
}

/// Recall statistics for vector search
#[derive(Debug, Default)]
pub struct RecallStats {
    pub total_queries: u64,
    pub sum_recall: f64,
    pub min_recall: f64,
    pub max_recall: f64,
    pub perfect_count: u64,  // recall == 1.0
    pub zero_count: u64,     // recall == 0.0
}

impl RecallStats {
    pub fn new() -> Self {
        Self {
            min_recall: f64::MAX,
            max_recall: f64::MIN,
            ..Default::default()
        }
    }

    pub fn record(&mut self, recall: f64) {
        self.total_queries += 1;
        self.sum_recall += recall;
        self.min_recall = self.min_recall.min(recall);
        self.max_recall = self.max_recall.max(recall);
        
        if (recall - 1.0).abs() < f64::EPSILON {
            self.perfect_count += 1;
        }
        if recall < f64::EPSILON {
            self.zero_count += 1;
        }
    }

    pub fn average(&self) -> f64 {
        if self.total_queries > 0 {
            self.sum_recall / self.total_queries as f64
        } else {
            0.0
        }
    }

    pub fn merge(&mut self, other: &RecallStats) {
        self.total_queries += other.total_queries;
        self.sum_recall += other.sum_recall;
        self.min_recall = self.min_recall.min(other.min_recall);
        self.max_recall = self.max_recall.max(other.max_recall);
        self.perfect_count += other.perfect_count;
        self.zero_count += other.zero_count;
    }
}

/// Benchmark worker (runs in dedicated OS thread)
pub struct BenchmarkWorker {
    /// Worker ID
    id: usize,
    
    /// Owned clients (NOT shared with other threads)
    clients: Vec<BenchmarkClient>,
    
    /// Thread-local RNG (fast, no sync)
    rng: fastrand::Rng,
    
    /// Thread-local histogram
    histogram: Histogram<u64>,
    
    /// Thread-local recall stats
    recall_stats: RecallStats,
    
    /// Local retry queue
    retry_queue: VecDeque<u64>,
    
    /// Configuration
    pipeline: u32,
    keyspace_len: u64,
    sequential: bool,
    total_requests: u64,
    
    /// Rate limiter tokens (if enabled)
    rate_limit_tokens: Option<TokenBucket>,
}

/// Simple token bucket for rate limiting
struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    tokens_per_ms: f64,
    last_update: Instant,
}

impl TokenBucket {
    fn new(rps: u64) -> Self {
        let tokens_per_ms = rps as f64 / 1000.0;
        Self {
            tokens: 0.0,
            max_tokens: rps as f64, // 1 second burst
            tokens_per_ms,
            last_update: Instant::now(),
        }
    }

    fn acquire(&mut self, count: u32) -> Option<Duration> {
        // Refill tokens based on elapsed time
        let now = Instant::now();
        let elapsed_ms = now.duration_since(self.last_update).as_secs_f64() * 1000.0;
        self.tokens = (self.tokens + elapsed_ms * self.tokens_per_ms).min(self.max_tokens);
        self.last_update = now;

        let needed = count as f64;
        if self.tokens >= needed {
            self.tokens -= needed;
            None // Can proceed immediately
        } else {
            // Calculate wait time
            let deficit = needed - self.tokens;
            let wait_ms = deficit / self.tokens_per_ms;
            Some(Duration::from_secs_f64(wait_ms / 1000.0))
        }
    }
}

impl BenchmarkWorker {
    /// Create new worker
    pub fn new(
        id: usize,
        clients: Vec<BenchmarkClient>,
        config: &BenchmarkConfig,
    ) -> Self {
        // Initialize RNG with worker-specific seed
        let seed = if config.seed == 0 {
            // Random seed
            fastrand::u64(..)
        } else {
            // Deterministic seed based on config + worker id
            config.seed.wrapping_add(id as u64)
        };
        let rng = fastrand::Rng::with_seed(seed);

        // Initialize histogram (1us to 1 hour, 3 significant digits)
        let histogram = Histogram::new_with_bounds(1, 3_600_000_000, 3)
            .expect("Failed to create histogram");

        // Rate limiter
        let rate_limit_tokens = if config.requests_per_second > 0 {
            // Divide RPS among threads
            let per_thread_rps = config.requests_per_second / config.threads as u64;
            Some(TokenBucket::new(per_thread_rps.max(1)))
        } else {
            None
        };

        Self {
            id,
            clients,
            rng,
            histogram,
            recall_stats: RecallStats::new(),
            retry_queue: VecDeque::new(),
            pipeline: config.pipeline,
            keyspace_len: config.keyspace_len,
            sequential: config.sequential,
            total_requests: config.requests,
            rate_limit_tokens,
        }
    }

    /// Main worker loop
    pub fn run(
        mut self,
        counters: Arc<GlobalCounters>,
        dataset: Option<Arc<DatasetContext>>,
    ) -> WorkerResult {
        let batch_size = self.pipeline as u64;
        let mut client_idx = 0;
        let mut requests_processed = 0u64;
        let mut error_count = 0u64;

        loop {
            // Check shutdown
            if counters.is_shutdown() {
                break;
            }

            // Claim request batch
            if counters.claim_requests(batch_size, self.total_requests).is_none() {
                break;
            }

            // Rate limiting
            if let Some(ref mut limiter) = self.rate_limit_tokens {
                if let Some(wait) = limiter.acquire(self.pipeline) {
                    std::thread::sleep(wait);
                }
            }

            // Get next client (round-robin within this worker's clients)
            let client = &mut self.clients[client_idx];
            client_idx = (client_idx + 1) % self.clients.len();

            // Clear batch state
            client.clear_batch_state();

            // Fill placeholders
            self.fill_placeholders(client, &counters, dataset.as_deref());

            // Execute batch
            match client.execute_batch() {
                Ok(response) => {
                    // Record latency
                    self.histogram.record(response.latency_us).ok();
                    
                    // Process responses
                    for (i, value) in response.values.iter().enumerate() {
                        if value.is_error() {
                            error_count += 1;
                            counters.record_error();
                            
                            // Handle MOVED/ASK for retry
                            if value.is_moved() || value.is_ask() {
                                if let Some(idx) = response.inflight_indices.get(i) {
                                    self.retry_queue.push_back(*idx);
                                }
                            }
                        }
                    }
                    
                    requests_processed += response.values.len() as u64;
                    counters.record_finished(response.values.len() as u64);
                }
                Err(e) => {
                    error_count += batch_size;
                    counters.record_error();
                    // Log error (in production, use tracing)
                    eprintln!("Worker {}: Batch error: {}", self.id, e);
                }
            }
        }

        WorkerResult {
            worker_id: self.id,
            histogram: self.histogram,
            recall_stats: self.recall_stats,
            retry_count: self.retry_queue.len() as u64,
            error_count,
            requests_processed,
        }
    }

    /// Fill placeholders for all commands in the batch
    fn fill_placeholders(
        &mut self,
        client: &mut BenchmarkClient,
        counters: &GlobalCounters,
        dataset: Option<&DatasetContext>,
    ) {
        let buffer = client.write_buffer();
        
        for cmd_idx in 0..self.pipeline as usize {
            if cmd_idx >= buffer.placeholders.len() {
                continue;
            }
            
            for ph in &buffer.placeholders[cmd_idx].clone() {
                match ph.placeholder_type {
                    PlaceholderType::Key => {
                        let key = if self.sequential {
                            counters.next_seq_key(self.keyspace_len)
                        } else {
                            self.rng.u64(0..self.keyspace_len)
                        };
                        client.replace_key(cmd_idx, key, ph);
                    }
                    PlaceholderType::Vector => {
                        if let Some(ds) = dataset {
                            // Get next dataset index
                            let idx = counters.next_dataset_idx() % ds.num_vectors();
                            let vec_bytes = ds.get_vector_bytes(idx);
                            client.replace_vector(cmd_idx, vec_bytes, ph);
                            client.track_inflight(idx);
                        }
                    }
                    PlaceholderType::ClusterTag => {
                        // Simplified: use static tag for now
                        // Full implementation would use ClusterTagMap
                        let tag = b"{000}";
                        client.replace_cluster_tag(cmd_idx, tag, ph);
                    }
                    PlaceholderType::RandInt => {
                        let value = self.rng.u64(..);
                        client.replace_key(cmd_idx, value, ph);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_stats() {
        let mut stats = RecallStats::new();
        
        stats.record(1.0);
        stats.record(0.5);
        stats.record(0.0);
        
        assert_eq!(stats.total_queries, 3);
        assert!((stats.average() - 0.5).abs() < 0.001);
        assert_eq!(stats.perfect_count, 1);
        assert_eq!(stats.zero_count, 1);
    }

    #[test]
    fn test_token_bucket() {
        let mut bucket = TokenBucket::new(1000); // 1000 RPS
        
        // Should be able to acquire immediately after some time
        std::thread::sleep(Duration::from_millis(10));
        assert!(bucket.acquire(5).is_none());
    }
}
```

**Validation:**
- Worker runs independently
- Rate limiting works
- Recall stats accumulate correctly

---

### Task 4.3: Implement Benchmark Orchestrator

**Description:** Create orchestrator to spawn workers, coordinate execution, and collect results.

**Instructions:**
Create `src/benchmark/orchestrator.rs`:

```rust
//! Benchmark orchestrator
//!
//! Coordinates worker threads, collects results, and manages the benchmark lifecycle.

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;
use indicatif::{ProgressBar, ProgressStyle};

use crate::client::{ConnectionFactory, BenchmarkClient, CommandBuffer};
use crate::config::BenchmarkConfig;
use crate::cluster::ClusterTopology;
use crate::dataset::DatasetContext;
use crate::workload::{WorkloadType, create_template};
use crate::utils::Result;

use super::counters::GlobalCounters;
use super::worker::{BenchmarkWorker, WorkerResult, RecallStats};

/// Benchmark result summary
pub struct BenchmarkResult {
    /// Test name
    pub test_name: String,
    /// Total requests completed
    pub total_requests: u64,
    /// Total duration
    pub duration: Duration,
    /// Throughput (requests per second)
    pub throughput: f64,
    /// Merged latency histogram
    pub histogram: Histogram<u64>,
    /// Merged recall statistics
    pub recall_stats: RecallStats,
    /// Total errors
    pub error_count: u64,
}

impl BenchmarkResult {
    /// Get percentile latency in microseconds
    pub fn percentile_us(&self, p: f64) -> u64 {
        self.histogram.value_at_percentile(p)
    }

    /// Get percentile latency in milliseconds
    pub fn percentile_ms(&self, p: f64) -> f64 {
        self.percentile_us(p) as f64 / 1000.0
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== {} ===", self.test_name);
        println!("Throughput: {:.2} req/sec", self.throughput);
        println!("Total requests: {}", self.total_requests);
        println!("Duration: {:.2}s", self.duration.as_secs_f64());
        println!("Errors: {}", self.error_count);
        println!("\nLatency (ms):");
        println!("  avg: {:.3}", self.histogram.mean() / 1000.0);
        println!("  p50: {:.3}", self.percentile_ms(50.0));
        println!("  p95: {:.3}", self.percentile_ms(95.0));
        println!("  p99: {:.3}", self.percentile_ms(99.0));
        println!("  p99.9: {:.3}", self.percentile_ms(99.9));
        println!("  max: {:.3}", self.histogram.max() as f64 / 1000.0);
        
        if self.recall_stats.total_queries > 0 {
            println!("\nRecall:");
            println!("  avg: {:.4}", self.recall_stats.average());
            println!("  min: {:.4}", self.recall_stats.min_recall);
            println!("  max: {:.4}", self.recall_stats.max_recall);
            println!("  perfect (1.0): {}", self.recall_stats.perfect_count);
            println!("  zero (0.0): {}", self.recall_stats.zero_count);
        }
    }
}

/// Benchmark orchestrator
pub struct Orchestrator {
    config: Arc<BenchmarkConfig>,
    connection_factory: ConnectionFactory,
    topology: Option<ClusterTopology>,
    dataset: Option<Arc<DatasetContext>>,
}

impl Orchestrator {
    /// Create new orchestrator
    pub fn new(config: BenchmarkConfig) -> Result<Self> {
        let connection_factory = ConnectionFactory {
            connect_timeout: Duration::from_millis(config.connect_timeout_ms),
            read_timeout: Duration::from_millis(config.request_timeout_ms),
            write_timeout: Duration::from_millis(config.request_timeout_ms),
            tls_config: config.tls.clone(),
            auth_password: config.auth.as_ref().map(|a| a.password.clone()),
            auth_username: config.auth.as_ref().and_then(|a| a.username.clone()),
            dbnum: config.dbnum,
        };

        Ok(Self {
            config: Arc::new(config),
            connection_factory,
            topology: None,
            dataset: None,
        })
    }

    /// Set cluster topology (for cluster mode)
    pub fn set_topology(&mut self, topology: ClusterTopology) {
        self.topology = Some(topology);
    }

    /// Set dataset (for vector search)
    pub fn set_dataset(&mut self, dataset: DatasetContext) {
        self.dataset = Some(Arc::new(dataset));
    }

    /// Run a single benchmark test
    pub fn run_test(&self, workload: WorkloadType) -> Result<BenchmarkResult> {
        // Create command template
        let template = create_template(workload, &self.config)
            .ok_or_else(|| crate::utils::BenchmarkError::Config(
                format!("No template for workload: {:?}", workload)
            ))?;

        // Build command buffer for pipeline
        let command_buffer = template.build(self.config.pipeline as usize);

        // Create global counters
        let counters = Arc::new(GlobalCounters::new());

        // Calculate clients per thread
        let clients_per_thread = self.config.clients_per_thread() as usize;

        // Spawn worker threads
        let mut handles: Vec<JoinHandle<WorkerResult>> = Vec::with_capacity(
            self.config.threads as usize
        );

        let start_time = Instant::now();

        for worker_id in 0..self.config.threads as usize {
            let config = Arc::clone(&self.config);
            let counters = Arc::clone(&counters);
            let dataset = self.dataset.clone();
            let buffer = command_buffer.clone();
            let factory = self.connection_factory.clone();
            let addr = &self.config.addresses[0]; // Simplified: use first address

            // Create clients for this worker
            let mut clients = Vec::with_capacity(clients_per_thread);
            for _ in 0..clients_per_thread {
                let conn = factory.create(&addr.host, addr.port)?;
                let client = BenchmarkClient::new(conn, buffer.clone(), config.pipeline as usize);
                clients.push(client);
            }

            let handle = thread::Builder::new()
                .name(format!("worker-{}", worker_id))
                .spawn(move || {
                    let worker = BenchmarkWorker::new(worker_id, clients, &config);
                    worker.run(counters, dataset)
                })
                .expect("Failed to spawn worker thread");

            handles.push(handle);
        }

        // Progress reporting (if not quiet)
        if !self.config.quiet {
            self.report_progress(&counters, self.config.requests);
        }

        // Wait for workers to complete
        let results: Vec<WorkerResult> = handles
            .into_iter()
            .map(|h| h.join().expect("Worker thread panicked"))
            .collect();

        let duration = start_time.elapsed();

        // Merge results
        self.merge_results(workload.as_str(), results, duration)
    }

    /// Report progress during benchmark
    fn report_progress(&self, counters: &GlobalCounters, total: u64) {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})")
                .unwrap()
                .progress_chars("#>-")
        );

        while !counters.is_shutdown() {
            let (finished, _) = counters.progress();
            pb.set_position(finished);
            
            if finished >= total {
                break;
            }
            
            thread::sleep(Duration::from_millis(100));
        }

        pb.finish_with_message("Complete");
    }

    /// Merge results from all workers
    fn merge_results(
        &self,
        test_name: &str,
        results: Vec<WorkerResult>,
        duration: Duration,
    ) -> Result<BenchmarkResult> {
        // Initialize merged histogram
        let mut merged_histogram = Histogram::new_with_bounds(1, 3_600_000_000, 3)
            .expect("Failed to create histogram");
        
        let mut merged_recall = RecallStats::new();
        let mut total_requests = 0u64;
        let mut error_count = 0u64;

        for result in results {
            merged_histogram.add(&result.histogram).ok();
            merged_recall.merge(&result.recall_stats);
            total_requests += result.requests_processed;
            error_count += result.error_count;
        }

        let throughput = total_requests as f64 / duration.as_secs_f64();

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            total_requests,
            duration,
            throughput,
            histogram: merged_histogram,
            recall_stats: merged_recall,
            error_count,
        })
    }

    /// Run all configured tests
    pub fn run_all(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for test_name in &self.config.tests {
            let workload = WorkloadType::from_str(test_name)
                .ok_or_else(|| crate::utils::BenchmarkError::Config(
                    format!("Unknown test: {}", test_name)
                ))?;

            println!("\nRunning test: {}", workload);
            let result = self.run_test(workload)?;
            result.print_summary();
            results.push(result);
        }

        Ok(results)
    }
}

// Implement Clone for ConnectionFactory (needed for thread spawning)
impl Clone for ConnectionFactory {
    fn clone(&self) -> Self {
        Self {
            connect_timeout: self.connect_timeout,
            read_timeout: self.read_timeout,
            write_timeout: self.write_timeout,
            tls_config: self.tls_config.clone(),
            auth_password: self.auth_password.clone(),
            auth_username: self.auth_username.clone(),
            dbnum: self.dbnum,
        }
    }
}
```

Update `src/benchmark/mod.rs`:

```rust
//! Benchmark orchestration and workers

pub mod counters;
pub mod worker;
pub mod orchestrator;

pub use counters::GlobalCounters;
pub use worker::{BenchmarkWorker, WorkerResult, RecallStats};
pub use orchestrator::{Orchestrator, BenchmarkResult};
```

**Validation:**
- Workers spawn and complete
- Results merge correctly
- Progress reporting works

---

### Task 4.4: Integrate with Main

**Description:** Update main.rs to run benchmarks using the orchestrator.

**Instructions:**
Update `src/main.rs`:

```rust
//! valkey-search-benchmark - High-performance benchmark tool for Valkey

use anyhow::Result;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

mod config;
mod utils;
mod client;
mod cluster;
mod workload;
mod dataset;
mod benchmark;
mod metrics;
mod optimizer;

use config::{CliArgs, BenchmarkConfig};
use benchmark::Orchestrator;

fn setup_logging(verbose: bool, quiet: bool) {
    let level = if quiet {
        Level::ERROR
    } else if verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}

fn print_banner(config: &BenchmarkConfig) {
    if config.quiet {
        return;
    }

    println!("valkey-search-benchmark v{}", env!("CARGO_PKG_VERSION"));
    println!("====================================");
    println!("Hosts: {:?}", config.addresses.iter()
        .map(|a| a.to_string())
        .collect::<Vec<_>>());
    println!("Clients: {}, Threads: {}, Pipeline: {}", 
             config.clients, config.threads, config.pipeline);
    println!("Requests: {}", config.requests);
    println!("Tests: {:?}", config.tests);
    if config.cluster_mode {
        println!("Cluster mode: enabled, RFR: {:?}", config.read_from_replica);
    }
    if let Some(ref search) = config.search_config {
        println!("Vector search: dim={}, k={}, algo={:?}", 
                 search.dim, search.k, search.algorithm);
    }
    println!("====================================\n");
}

fn run() -> Result<()> {
    // Parse CLI arguments
    let args = CliArgs::parse_args();

    // Setup logging
    setup_logging(args.verbose, args.quiet);

    // Build configuration
    let config = BenchmarkConfig::from_cli(&args)
        .map_err(|e| anyhow::anyhow!("Configuration error: {}", e))?;

    // Print banner
    print_banner(&config);

    // Create orchestrator
    let orchestrator = Orchestrator::new(config)?;

    // Run all tests
    let results = orchestrator.run_all()?;

    // Print summary
    println!("\n====================================");
    println!("BENCHMARK COMPLETE");
    println!("====================================");
    println!("Tests run: {}", results.len());
    
    let total_requests: u64 = results.iter().map(|r| r.total_requests).sum();
    let total_errors: u64 = results.iter().map(|r| r.error_count).sum();
    println!("Total requests: {}", total_requests);
    println!("Total errors: {}", total_errors);
    
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        error!("Error: {:#}", e);
        std::process::exit(1);
    }
}
```

**Validation:**
- `cargo run -- -h localhost -n 1000 -t ping` runs benchmark
- Progress bar shows during execution
- Results print at end

---

### Task 4.5: Phase 4 Commit and Validation

**Description:** Commit Phase 4 and run full validation.

**Instructions:**
1. Run all tests: `cargo test`
2. Run clippy: `cargo clippy -- -D warnings`
3. Format code: `cargo fmt`
4. Build release: `cargo build --release`
5. Integration test (requires Valkey server):
   ```bash
   # Start Valkey server
   valkey-server &
   
   # Run PING benchmark
   ./target/release/valkey-search-benchmark -h localhost -n 10000 -t ping
   
   # Run SET benchmark with pipeline
   ./target/release/valkey-search-benchmark -h localhost -n 100000 -t set -P 10 -c 50 --threads 4
   
   # Run multiple tests
   ./target/release/valkey-search-benchmark -h localhost -n 10000 -t ping,set,get
   ```

6. Git commit:
   ```bash
   git add .
   git commit -m "Phase 4: Threading model and worker implementation

   - Implement GlobalCounters for atomic cross-thread sync
   - Create BenchmarkWorker with thread-local state
   - Add RecallStats for vector search verification
   - Implement TokenBucket rate limiter
   - Create Orchestrator for coordinating workers
   - Add progress bar reporting
   - Integrate with main entry point
   
   Multi-threaded benchmark execution working with standard workloads."
   ```

**Validation Checklist:**
- [ ] Workers spawn correctly
- [ ] Request claiming is atomic and correct
- [ ] Progress bar updates during run
- [ ] Results merge correctly
- [ ] Rate limiting throttles requests
- [ ] Multiple tests run sequentially
- [ ] Latency histogram is accurate

---

---

## Phase 5: Deployment Targets and Discovery Abstraction

**Objective:** Implement deployment-agnostic discovery supporting standalone mode, cluster mode, AWS ElastiCache, AWS MemoryDB, and ElastiCache Serverless.

**Git Branch:** `phase-5-deployment`

### Task 5.1: Define Deployment Target Abstraction

**Description:** Create abstraction layer for different Valkey deployment types with their specific behaviors.

**Instructions:**
Create `src/cluster/deployment.rs`:

```rust
//! Deployment target abstraction
//!
//! Supports: self-managed Valkey, AWS ElastiCache (cluster/standalone),
//! AWS MemoryDB, ElastiCache Serverless

/// Deployment target type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentTarget {
    /// Self-managed Valkey/Redis (open source)
    SelfManaged,
    /// AWS ElastiCache (Valkey or Redis mode)
    ElastiCache,
    /// AWS ElastiCache Serverless
    ElastiCacheServerless,
    /// AWS MemoryDB
    MemoryDB,
}

/// Server mode (cluster vs standalone)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerMode {
    /// Cluster mode enabled (slot-based routing)
    Cluster,
    /// Standalone mode (may still have replicas via replication)
    Standalone,
}

/// Deployment configuration combining target and mode
#[derive(Debug, Clone)]
pub struct DeploymentConfig {
    pub target: DeploymentTarget,
    pub mode: ServerMode,
    /// Original endpoints as provided (may be configuration endpoints)
    pub endpoints: Vec<String>,
    /// Resolved node addresses after discovery
    pub resolved_nodes: Vec<NodeAddress>,
}

/// Resolved node address
#[derive(Debug, Clone)]
pub struct NodeAddress {
    pub host: String,
    pub port: u16,
    pub is_primary: bool,
    pub is_replica: bool,
    /// For ElastiCache: availability zone
    pub availability_zone: Option<String>,
    /// Node identifier (varies by deployment)
    pub node_id: Option<String>,
}

impl DeploymentTarget {
    /// Auto-detect deployment target from endpoint format
    pub fn detect_from_endpoint(endpoint: &str) -> Self {
        if endpoint.contains(".cache.amazonaws.com") {
            if endpoint.contains("serverless") {
                DeploymentTarget::ElastiCacheServerless
            } else {
                DeploymentTarget::ElastiCache
            }
        } else if endpoint.contains(".memorydb.") && endpoint.contains(".amazonaws.com") {
            DeploymentTarget::MemoryDB
        } else {
            DeploymentTarget::SelfManaged
        }
    }

    /// Check if this target uses AWS-specific endpoint formats
    pub fn is_aws(&self) -> bool {
        matches!(self, 
            DeploymentTarget::ElastiCache | 
            DeploymentTarget::ElastiCacheServerless | 
            DeploymentTarget::MemoryDB
        )
    }

    /// Get VSS (Vector Similarity Search) implementation variant
    pub fn vss_variant(&self) -> VssVariant {
        match self {
            DeploymentTarget::SelfManaged => VssVariant::OpenSource,
            DeploymentTarget::ElastiCache => VssVariant::ElastiCache,
            DeploymentTarget::ElastiCacheServerless => VssVariant::ElastiCacheServerless,
            DeploymentTarget::MemoryDB => VssVariant::MemoryDB,
        }
    }
}

/// VSS implementation variant (affects metrics and command behavior)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VssVariant {
    /// Open-source Valkey VSS
    OpenSource,
    /// AWS ElastiCache VSS
    ElastiCache,
    /// AWS ElastiCache Serverless VSS
    ElastiCacheServerless,
    /// AWS MemoryDB VSS (different from ElastiCache)
    MemoryDB,
}

impl VssVariant {
    /// Get INFO section name for VSS metrics
    pub fn info_section(&self) -> &'static str {
        match self {
            VssVariant::OpenSource => "search",
            VssVariant::ElastiCache => "search",
            VssVariant::ElastiCacheServerless => "search",
            VssVariant::MemoryDB => "search", // May differ
        }
    }

    /// Get available VSS-specific metrics for this variant
    pub fn available_metrics(&self) -> &'static [&'static str] {
        match self {
            VssVariant::OpenSource => &[
                "search_number_of_indexes",
                "search_total_indexing_time",
                "search_total_query_time",
                "search_used_memory_bytes",
            ],
            VssVariant::ElastiCache => &[
                "search_number_of_indexes",
                "search_total_indexing_time",
                "search_total_query_time",
                "search_used_memory_bytes",
                // ElastiCache-specific metrics
            ],
            VssVariant::MemoryDB => &[
                "search_number_of_indexes",
                "search_total_indexing_time",
                // MemoryDB may have different metric names
            ],
            VssVariant::ElastiCacheServerless => &[
                // Serverless has limited metrics visibility
            ],
        }
    }
}
```

**Validation:**
- Endpoint detection correctly identifies AWS services
- VSS variants are distinct

---

### Task 5.2: Implement AWS Endpoint Parsing

**Description:** Parse AWS-specific endpoint formats (ElastiCache cluster configuration endpoint, MemoryDB endpoints).

**Instructions:**
Create `src/cluster/aws_endpoint.rs`:

```rust
//! AWS endpoint parsing
//!
//! ElastiCache endpoints:
//! - Configuration endpoint: clustercfg.<name>.<region>.cache.amazonaws.com:6379
//! - Node endpoints: <name>-<shard>-<node>.<name>.<region>.cache.amazonaws.com:6379
//! - Serverless: <name>.<region>.serverless.cache.amazonaws.com:6379
//!
//! MemoryDB endpoints:
//! - Cluster endpoint: clustercfg.<name>.<random>.memorydb.<region>.amazonaws.com:6379

use std::net::ToSocketAddrs;

/// Parsed AWS endpoint information
#[derive(Debug, Clone)]
pub struct AwsEndpoint {
    pub host: String,
    pub port: u16,
    pub endpoint_type: AwsEndpointType,
    pub cluster_name: Option<String>,
    pub region: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwsEndpointType {
    /// ElastiCache configuration endpoint (clustercfg.*)
    ElastiCacheConfig,
    /// ElastiCache individual node
    ElastiCacheNode,
    /// ElastiCache Serverless endpoint
    ElastiCacheServerless,
    /// MemoryDB cluster endpoint
    MemoryDBCluster,
    /// MemoryDB individual node
    MemoryDBNode,
    /// Non-AWS endpoint
    Standard,
}

impl AwsEndpoint {
    /// Parse endpoint string
    pub fn parse(endpoint: &str) -> Self {
        let (host, port) = Self::split_host_port(endpoint);
        let endpoint_type = Self::detect_type(&host);
        let (cluster_name, region) = Self::extract_metadata(&host, endpoint_type);

        Self {
            host,
            port,
            endpoint_type,
            cluster_name,
            region,
        }
    }

    fn split_host_port(endpoint: &str) -> (String, u16) {
        if let Some((host, port_str)) = endpoint.rsplit_once(':') {
            if let Ok(port) = port_str.parse() {
                return (host.to_string(), port);
            }
        }
        (endpoint.to_string(), 6379)
    }

    fn detect_type(host: &str) -> AwsEndpointType {
        if host.contains(".serverless.cache.amazonaws.com") {
            AwsEndpointType::ElastiCacheServerless
        } else if host.starts_with("clustercfg.") && host.contains(".cache.amazonaws.com") {
            AwsEndpointType::ElastiCacheConfig
        } else if host.contains(".cache.amazonaws.com") {
            AwsEndpointType::ElastiCacheNode
        } else if host.starts_with("clustercfg.") && host.contains(".memorydb.") {
            AwsEndpointType::MemoryDBCluster
        } else if host.contains(".memorydb.") && host.contains(".amazonaws.com") {
            AwsEndpointType::MemoryDBNode
        } else {
            AwsEndpointType::Standard
        }
    }

    fn extract_metadata(host: &str, endpoint_type: AwsEndpointType) -> (Option<String>, Option<String>) {
        // Extract cluster name and region from endpoint
        // Example: clustercfg.mycluster.use1.cache.amazonaws.com
        //          -> cluster_name: mycluster, region: us-east-1
        
        let parts: Vec<&str> = host.split('.').collect();
        
        match endpoint_type {
            AwsEndpointType::ElastiCacheConfig => {
                if parts.len() >= 4 {
                    let cluster_name = Some(parts[1].to_string());
                    let region = Self::expand_region(parts[2]);
                    (cluster_name, region)
                } else {
                    (None, None)
                }
            }
            AwsEndpointType::MemoryDBCluster => {
                if parts.len() >= 4 {
                    let cluster_name = Some(parts[1].to_string());
                    // MemoryDB: clustercfg.<name>.<random>.memorydb.<region>...
                    if let Some(region_idx) = parts.iter().position(|&p| p == "memorydb") {
                        if region_idx + 1 < parts.len() {
                            return (cluster_name, Some(parts[region_idx + 1].to_string()));
                        }
                    }
                    (cluster_name, None)
                } else {
                    (None, None)
                }
            }
            _ => (None, None),
        }
    }

    /// Expand AWS region shorthand (e.g., "use1" -> "us-east-1")
    fn expand_region(short: &str) -> Option<String> {
        match short {
            "use1" => Some("us-east-1".to_string()),
            "use2" => Some("us-east-2".to_string()),
            "usw1" => Some("us-west-1".to_string()),
            "usw2" => Some("us-west-2".to_string()),
            "euw1" => Some("eu-west-1".to_string()),
            "euw2" => Some("eu-west-2".to_string()),
            "euc1" => Some("eu-central-1".to_string()),
            "apne1" => Some("ap-northeast-1".to_string()),
            "apne2" => Some("ap-northeast-2".to_string()),
            "apse1" => Some("ap-southeast-1".to_string()),
            "apse2" => Some("ap-southeast-2".to_string()),
            _ => Some(short.to_string()), // Assume it's already full region name
        }
    }

    /// Check if this is a configuration endpoint that needs resolution
    pub fn needs_node_discovery(&self) -> bool {
        matches!(
            self.endpoint_type,
            AwsEndpointType::ElastiCacheConfig | AwsEndpointType::MemoryDBCluster
        )
    }

    /// Resolve DNS to get all IPs (for multi-node discovery)
    pub fn resolve_dns(&self) -> Vec<std::net::IpAddr> {
        let addr = format!("{}:{}", self.host, self.port);
        addr.to_socket_addrs()
            .map(|addrs| addrs.map(|a| a.ip()).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elasticache_config_endpoint() {
        let ep = AwsEndpoint::parse("clustercfg.mycluster.use1.cache.amazonaws.com:6379");
        assert_eq!(ep.endpoint_type, AwsEndpointType::ElastiCacheConfig);
        assert_eq!(ep.cluster_name, Some("mycluster".to_string()));
        assert_eq!(ep.region, Some("us-east-1".to_string()));
    }

    #[test]
    fn test_elasticache_serverless() {
        let ep = AwsEndpoint::parse("myapp.use1.serverless.cache.amazonaws.com:6379");
        assert_eq!(ep.endpoint_type, AwsEndpointType::ElastiCacheServerless);
    }

    #[test]
    fn test_memorydb_endpoint() {
        let ep = AwsEndpoint::parse("clustercfg.mydb.abc123.memorydb.us-east-1.amazonaws.com:6379");
        assert_eq!(ep.endpoint_type, AwsEndpointType::MemoryDBCluster);
    }

    #[test]
    fn test_standard_endpoint() {
        let ep = AwsEndpoint::parse("localhost:6379");
        assert_eq!(ep.endpoint_type, AwsEndpointType::Standard);
    }
}
```

**Validation:**
- AWS endpoint formats are correctly parsed
- Region expansion works
- DNS resolution returns multiple IPs for cluster endpoints

---

### Task 5.3: Implement Discovery Strategy Abstraction

**Description:** Create discovery strategy that handles both cluster mode and standalone mode with replicas.

**Instructions:**
Create `src/cluster/discovery.rs`:

```rust
//! Node discovery strategies
//!
//! Supports:
//! - Cluster mode: CLUSTER NODES
//! - Standalone with replicas: INFO REPLICATION
//! - AWS ElastiCache: Configuration endpoint + CLUSTER NODES
//! - AWS MemoryDB: Cluster endpoint discovery

use crate::client::{RawConnection, ConnectionFactory};
use crate::utils::{RespEncoder, ClusterError};
use super::deployment::{DeploymentConfig, ServerMode, NodeAddress};
use super::topology::ClusterTopology;
use super::aws_endpoint::AwsEndpoint;

/// Discovery result
pub enum DiscoveryResult {
    /// Cluster topology (cluster mode enabled)
    Cluster(ClusterTopology),
    /// Standalone topology (may include replicas)
    Standalone(StandaloneTopology),
}

/// Standalone topology (non-cluster mode)
#[derive(Debug, Clone)]
pub struct StandaloneTopology {
    pub primary: NodeAddress,
    pub replicas: Vec<NodeAddress>,
}

/// Discover topology based on deployment configuration
pub fn discover(
    factory: &ConnectionFactory,
    config: &mut DeploymentConfig,
) -> Result<DiscoveryResult, ClusterError> {
    // First, connect to initial endpoint and detect mode
    let initial = &config.endpoints[0];
    let aws_ep = AwsEndpoint::parse(initial);
    
    let mut conn = factory.create(&aws_ep.host, aws_ep.port)
        .map_err(|e| ClusterError::RefreshFailed(e.to_string()))?;
    
    // Detect server mode
    let mode = detect_server_mode(&mut conn)?;
    config.mode = mode;
    
    match mode {
        ServerMode::Cluster => {
            let topology = discover_cluster(&mut conn, factory, &aws_ep)?;
            Ok(DiscoveryResult::Cluster(topology))
        }
        ServerMode::Standalone => {
            let topology = discover_standalone(&mut conn, factory)?;
            Ok(DiscoveryResult::Standalone(topology))
        }
    }
}

/// Detect if server is in cluster mode
fn detect_server_mode(conn: &mut RawConnection) -> Result<ServerMode, ClusterError> {
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["INFO", "CLUSTER"]);
    
    let response = conn.execute(&encoder)
        .map_err(|e| ClusterError::RefreshFailed(e.to_string()))?;
    
    let info = response.as_str().unwrap_or("");
    
    if info.contains("cluster_enabled:1") {
        Ok(ServerMode::Cluster)
    } else {
        Ok(ServerMode::Standalone)
    }
}

/// Discover cluster topology using CLUSTER NODES
fn discover_cluster(
    conn: &mut RawConnection,
    factory: &ConnectionFactory,
    aws_ep: &AwsEndpoint,
) -> Result<ClusterTopology, ClusterError> {
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["CLUSTER", "NODES"]);
    
    let response = conn.execute(&encoder)
        .map_err(|e| ClusterError::RefreshFailed(e.to_string()))?;
    
    let nodes_str = response.as_str()
        .ok_or_else(|| ClusterError::ParseFailed("Expected string response".to_string()))?;
    
    // For AWS endpoints, may need to replace internal IPs with DNS names
    let nodes_str = if aws_ep.endpoint_type.is_aws_config() {
        rewrite_aws_node_addresses(nodes_str, aws_ep)
    } else {
        nodes_str.to_string()
    };
    
    ClusterTopology::from_cluster_nodes(&nodes_str)
        .map_err(ClusterError::ParseFailed)
}

/// Discover standalone topology using INFO REPLICATION
fn discover_standalone(
    conn: &mut RawConnection,
    factory: &ConnectionFactory,
) -> Result<StandaloneTopology, ClusterError> {
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["INFO", "REPLICATION"]);
    
    let response = conn.execute(&encoder)
        .map_err(|e| ClusterError::RefreshFailed(e.to_string()))?;
    
    let info = response.as_str()
        .ok_or_else(|| ClusterError::ParseFailed("Expected string response".to_string()))?;
    
    parse_replication_info(info)
}

/// Parse INFO REPLICATION to extract primary and replicas
fn parse_replication_info(info: &str) -> Result<StandaloneTopology, ClusterError> {
    let mut role = "master";
    let mut replicas = Vec::new();
    let mut master_host = None;
    let mut master_port = None;
    
    for line in info.lines() {
        let line = line.trim();
        
        if let Some(r) = line.strip_prefix("role:") {
            role = r.trim();
        } else if let Some(host) = line.strip_prefix("master_host:") {
            master_host = Some(host.trim().to_string());
        } else if let Some(port) = line.strip_prefix("master_port:") {
            master_port = port.trim().parse().ok();
        } else if line.starts_with("slave") && line.contains("ip=") {
            // Parse replica info: slave0:ip=127.0.0.1,port=6380,state=online,...
            if let Some(replica) = parse_replica_line(line) {
                replicas.push(replica);
            }
        }
    }
    
    // Build topology based on role
    if role == "master" {
        // We're connected to primary, get local address
        // Note: Need to get actual address from connection or config
        Ok(StandaloneTopology {
            primary: NodeAddress {
                host: "localhost".to_string(), // Replace with actual
                port: 6379,
                is_primary: true,
                is_replica: false,
                availability_zone: None,
                node_id: None,
            },
            replicas,
        })
    } else {
        // We're connected to replica, master info available
        let primary = NodeAddress {
            host: master_host.unwrap_or_else(|| "localhost".to_string()),
            port: master_port.unwrap_or(6379),
            is_primary: true,
            is_replica: false,
            availability_zone: None,
            node_id: None,
        };
        Ok(StandaloneTopology { primary, replicas })
    }
}

/// Parse replica line from INFO REPLICATION
fn parse_replica_line(line: &str) -> Option<NodeAddress> {
    let mut ip = None;
    let mut port = None;
    
    for part in line.split(',') {
        if let Some(val) = part.strip_prefix("ip=") {
            ip = Some(val.to_string());
        } else if let Some(val) = part.strip_prefix("port=") {
            port = val.parse().ok();
        }
    }
    
    Some(NodeAddress {
        host: ip?,
        port: port?,
        is_primary: false,
        is_replica: true,
        availability_zone: None,
        node_id: None,
    })
}

/// Rewrite AWS internal IPs to DNS names if needed
fn rewrite_aws_node_addresses(nodes_str: &str, aws_ep: &AwsEndpoint) -> String {
    // AWS ElastiCache may return internal IPs in CLUSTER NODES
    // that need to be accessed via the original DNS endpoint
    // This is deployment-specific behavior
    nodes_str.to_string() // Placeholder - implement based on AWS behavior
}

impl AwsEndpointType {
    fn is_aws_config(&self) -> bool {
        matches!(self, 
            AwsEndpointType::ElastiCacheConfig | 
            AwsEndpointType::MemoryDBCluster
        )
    }
}
```

**Validation:**
- Cluster mode detection works
- Standalone topology parsing extracts replicas
- AWS endpoint rewriting handles internal IPs

---

### Task 5.4: Implement Cluster Node Representation

**Description:** Create data structures for cluster topology.

**Instructions:**
Create `src/cluster/node.rs`:

```rust
//! Cluster node representation

use super::deployment::VssVariant;

/// Cluster node information
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Node ID from CLUSTER NODES
    pub id: String,
    /// Hostname or IP
    pub host: String,
    /// Port
    pub port: u16,
    /// Cluster bus port
    pub bus_port: u16,
    /// Is this a primary node?
    pub is_primary: bool,
    /// Is this a replica?
    pub is_replica: bool,
    /// Primary node ID (if replica)
    pub primary_id: Option<String>,
    /// Assigned slots (for primaries)
    pub slots: Vec<u16>,
    /// Node flags (fail, handshake, etc.)
    pub flags: Vec<String>,
    /// Availability zone (for AWS deployments)
    pub availability_zone: Option<String>,
    /// VSS variant for this node
    pub vss_variant: VssVariant,
}

impl ClusterNode {
    /// Get address string
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Check if node is available (not failing)
    pub fn is_available(&self) -> bool {
        !self.flags.iter().any(|f| f == "fail" || f == "fail?")
    }

    /// Check if node owns the given slot
    pub fn owns_slot(&self, slot: u16) -> bool {
        self.slots.contains(&slot)
    }

    /// Get unique identifier for metrics tracking
    pub fn metrics_id(&self) -> String {
        if self.id.len() > 8 {
            format!("{}:{}({})", self.host, self.port, &self.id[..8])
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

/// Parse CLUSTER NODES response line
pub fn parse_cluster_node_line(line: &str, vss_variant: VssVariant) -> Option<ClusterNode> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 8 {
        return None;
    }

    let id = parts[0].to_string();
    
    // Parse address: ip:port@bus_port or ip:port
    let addr_part = parts[1].split('@').next()?;
    let addr_parts: Vec<&str> = addr_part.split(':').collect();
    if addr_parts.len() < 2 {
        return None;
    }
    
    let host = addr_parts[0].to_string();
    let port: u16 = addr_parts[1].parse().ok()?;
    let bus_port = parts[1]
        .split('@')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(port + 10000);

    // Parse flags
    let flags: Vec<String> = parts[2].split(',').map(|s| s.to_string()).collect();
    let is_primary = flags.contains(&"master".to_string());
    let is_replica = flags.contains(&"slave".to_string());

    // Parse primary ID (for replicas)
    let primary_id = if is_replica && parts[3] != "-" {
        Some(parts[3].to_string())
    } else {
        None
    };

    // Parse slot ranges (parts[8..])
    let mut slots = Vec::new();
    for slot_spec in parts.iter().skip(8) {
        if slot_spec.starts_with('[') {
            continue; // Skip migrating/importing slots
        }
        if let Some(range) = parse_slot_range(slot_spec) {
            slots.extend(range);
        }
    }

    Some(ClusterNode {
        id,
        host,
        port,
        bus_port,
        is_primary,
        is_replica,
        primary_id,
        slots,
        flags,
        availability_zone: None,
        vss_variant,
    })
}

/// Parse slot range like "0-5460" or "5461"
fn parse_slot_range(spec: &str) -> Option<Vec<u16>> {
    if let Some((start, end)) = spec.split_once('-') {
        let start: u16 = start.parse().ok()?;
        let end: u16 = end.parse().ok()?;
        Some((start..=end).collect())
    } else {
        let slot: u16 = spec.parse().ok()?;
        Some(vec![slot])
    }
}
```

---

### Task 5.5: Phase 5 Commit and Validation

```bash
git add .
git commit -m "Phase 5: Deployment targets and discovery abstraction

- Define DeploymentTarget enum (SelfManaged, ElastiCache, MemoryDB, Serverless)
- Implement AWS endpoint parsing (configuration endpoints, node endpoints)
- Create discovery strategy abstraction (cluster vs standalone)
- Add standalone topology discovery via INFO REPLICATION
- Support VSS variant detection per deployment
- Parse replica information for non-cluster mode

Supports all Valkey deployment scenarios with appropriate discovery."
```

**Validation Checklist:**
- [ ] AWS ElastiCache configuration endpoint detected
- [ ] MemoryDB endpoint format parsed correctly
- [ ] Cluster mode detection via INFO CLUSTER works
- [ ] Standalone mode discovers replicas via INFO REPLICATION
- [ ] VSS variant correctly assigned based on deployment

---

## Phase 6: Cluster Topology and Node Selection

**Objective:** Implement cluster topology management with slot mapping and read-from-replica strategies.

**Git Branch:** `phase-6-cluster`

### Task 6.1: Implement Cluster Topology

**Description:** Create cluster topology with slot mapping and node selection.

**Instructions:**
Create `src/cluster/topology.rs`:

```rust
//! Cluster topology management

use std::collections::HashMap;
use crate::config::ReadFromReplica;
use super::node::{ClusterNode, parse_cluster_node_line};
use super::deployment::VssVariant;

/// Cluster topology snapshot
#[derive(Debug, Clone)]
pub struct ClusterTopology {
    /// All nodes in the cluster
    pub nodes: Vec<ClusterNode>,
    /// Slot to node index mapping
    slot_map: [Option<usize>; 16384],
    /// Primary node indices
    primary_indices: Vec<usize>,
    /// Replica node indices grouped by primary ID
    replica_map: HashMap<String, Vec<usize>>,
    /// VSS variant for this cluster
    pub vss_variant: VssVariant,
}

impl ClusterTopology {
    /// Parse CLUSTER NODES response
    pub fn from_cluster_nodes(response: &str) -> Result<Self, String> {
        Self::from_cluster_nodes_with_variant(response, VssVariant::OpenSource)
    }

    /// Parse CLUSTER NODES response with explicit VSS variant
    pub fn from_cluster_nodes_with_variant(
        response: &str,
        vss_variant: VssVariant,
    ) -> Result<Self, String> {
        let mut nodes = Vec::new();
        let mut slot_map = [None; 16384];
        let mut primary_indices = Vec::new();
        let mut replica_map: HashMap<String, Vec<usize>> = HashMap::new();

        for line in response.lines() {
            if line.is_empty() {
                continue;
            }

            if let Some(node) = parse_cluster_node_line(line, vss_variant) {
                let idx = nodes.len();

                if node.is_primary {
                    primary_indices.push(idx);
                    for &slot in &node.slots {
                        slot_map[slot as usize] = Some(idx);
                    }
                    replica_map.insert(node.id.clone(), Vec::new());
                }

                nodes.push(node);
            }
        }

        // Map replicas to primaries
        for (idx, node) in nodes.iter().enumerate() {
            if node.is_replica {
                if let Some(ref primary_id) = node.primary_id {
                    if let Some(replicas) = replica_map.get_mut(primary_id) {
                        replicas.push(idx);
                    }
                }
            }
        }

        if primary_indices.is_empty() {
            return Err("No primary nodes found".to_string());
        }

        Ok(Self {
            nodes,
            slot_map,
            primary_indices,
            replica_map,
            vss_variant,
        })
    }

    /// Get node for slot
    pub fn get_node_for_slot(&self, slot: u16) -> Option<&ClusterNode> {
        self.slot_map[slot as usize].map(|idx| &self.nodes[idx])
    }

    /// Get node index for slot
    pub fn get_node_idx_for_slot(&self, slot: u16) -> Option<usize> {
        self.slot_map[slot as usize]
    }

    /// Get all primary nodes
    pub fn primaries(&self) -> impl Iterator<Item = &ClusterNode> {
        self.primary_indices.iter().map(|&idx| &self.nodes[idx])
    }

    /// Get primary node indices
    pub fn primary_indices(&self) -> &[usize] {
        &self.primary_indices
    }

    /// Get replicas for a primary
    pub fn replicas_for(&self, primary_id: &str) -> Vec<&ClusterNode> {
        self.replica_map
            .get(primary_id)
            .map(|indices| indices.iter().map(|&idx| &self.nodes[idx]).collect())
            .unwrap_or_default()
    }

    /// Get nodes based on read-from-replica strategy
    pub fn select_nodes(&self, strategy: ReadFromReplica) -> Vec<(usize, &ClusterNode)> {
        match strategy {
            ReadFromReplica::Primary => {
                self.primary_indices.iter()
                    .map(|&idx| (idx, &self.nodes[idx]))
                    .filter(|(_, n)| n.is_available())
                    .collect()
            }
            ReadFromReplica::PreferReplica => {
                // Collect available replicas
                let mut nodes: Vec<(usize, &ClusterNode)> = self.nodes.iter()
                    .enumerate()
                    .filter(|(_, n)| n.is_replica && n.is_available())
                    .collect();
                
                // Fallback to primaries if no replicas
                if nodes.is_empty() {
                    nodes = self.primary_indices.iter()
                        .map(|&idx| (idx, &self.nodes[idx]))
                        .filter(|(_, n)| n.is_available())
                        .collect();
                }
                nodes
            }
            ReadFromReplica::RoundRobin => {
                // All available nodes
                self.nodes.iter()
                    .enumerate()
                    .filter(|(_, n)| n.is_available())
                    .collect()
            }
            ReadFromReplica::AzAffinity => {
                // Group by AZ, prefer nodes in same AZ
                // Simplified: return all available, let caller filter by AZ
                self.nodes.iter()
                    .enumerate()
                    .filter(|(_, n)| n.is_available())
                    .collect()
            }
        }
    }

    /// Get number of primary nodes
    pub fn num_primaries(&self) -> usize {
        self.primary_indices.len()
    }

    /// Get total number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Calculate slot for key using CRC16
    pub fn slot_for_key(key: &[u8]) -> u16 {
        // Check for hash tag {xxx}
        if let Some(start) = key.iter().position(|&b| b == b'{') {
            if let Some(end) = key[start + 1..].iter().position(|&b| b == b'}') {
                if end > 0 {
                    return crc16(&key[start + 1..start + 1 + end]) % 16384;
                }
            }
        }
        crc16(key) % 16384
    }

    /// Get all node addresses for metrics collection
    pub fn all_node_addresses(&self) -> Vec<(String, u16, bool)> {
        self.nodes.iter()
            .map(|n| (n.host.clone(), n.port, n.is_primary))
            .collect()
    }
}

/// CRC16 implementation for Redis cluster slot calculation (XMODEM)
fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_calculation_hash_tag() {
        // Keys with same hash tag should map to same slot
        let slot1 = ClusterTopology::slot_for_key(b"{foo}bar");
        let slot2 = ClusterTopology::slot_for_key(b"{foo}baz");
        assert_eq!(slot1, slot2);
    }

    #[test]
    fn test_slot_calculation_no_tag() {
        let slot = ClusterTopology::slot_for_key(b"hello");
        assert!(slot < 16384);
    }

    #[test]
    fn test_crc16() {
        // Known test vector: "123456789" -> 0x31C3
        assert_eq!(crc16(b"123456789"), 0x31C3);
    }
}
```

Update `src/cluster/mod.rs`:

```rust
//! Cluster topology and node management

pub mod deployment;
pub mod aws_endpoint;
pub mod discovery;
pub mod node;
pub mod topology;

pub use deployment::{DeploymentTarget, DeploymentConfig, ServerMode, VssVariant, NodeAddress};
pub use aws_endpoint::AwsEndpoint;
pub use discovery::{discover, DiscoveryResult, StandaloneTopology};
pub use node::ClusterNode;
pub use topology::ClusterTopology;
```

---

### Task 6.2: Phase 6 Commit

```bash
git add .
git commit -m "Phase 6: Cluster topology and node selection

- Implement ClusterTopology with slot mapping
- Add CRC16 slot calculation (XMODEM variant)
- Support read-from-replica strategies
- Create node selection by strategy
- Add hash tag extraction for slot routing

Full cluster routing support with replica awareness."
```

**Validation:**
- [ ] Slot calculation matches Redis implementation
- [ ] Hash tag extraction works correctly
- [ ] Node selection respects read-from-replica strategy

**Description:** Create data structures for cluster topology.

**Instructions:**
Create `src/cluster/node.rs`:

```rust
//! Cluster node representation

/// Cluster node information
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Node ID from CLUSTER NODES
    pub id: String,
    /// Hostname or IP
    pub host: String,
    /// Port
    pub port: u16,
    /// Cluster bus port
    pub bus_port: u16,
    /// Is this a primary node?
    pub is_primary: bool,
    /// Is this a replica?
    pub is_replica: bool,
    /// Primary node ID (if replica)
    pub primary_id: Option<String>,
    /// Assigned slots (for primaries)
    pub slots: Vec<u16>,
    /// Node flags (fail, handshake, etc.)
    pub flags: Vec<String>,
}

impl ClusterNode {
    /// Get address string
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Check if node is available (not failing)
    pub fn is_available(&self) -> bool {
        !self.flags.iter().any(|f| f == "fail" || f == "fail?")
    }

    /// Check if node owns the given slot
    pub fn owns_slot(&self, slot: u16) -> bool {
        self.slots.contains(&slot)
    }
}

/// Parse CLUSTER NODES response line
pub fn parse_cluster_node_line(line: &str) -> Option<ClusterNode> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 8 {
        return None;
    }

    let id = parts[0].to_string();
    
    // Parse address: ip:port@bus_port or ip:port
    let addr_part = parts[1].split('@').next()?;
    let addr_parts: Vec<&str> = addr_part.split(':').collect();
    if addr_parts.len() < 2 {
        return None;
    }
    
    let host = addr_parts[0].to_string();
    let port: u16 = addr_parts[1].parse().ok()?;
    let bus_port = parts[1]
        .split('@')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(port + 10000);

    // Parse flags
    let flags: Vec<String> = parts[2].split(',').map(|s| s.to_string()).collect();
    let is_primary = flags.contains(&"master".to_string());
    let is_replica = flags.contains(&"slave".to_string());

    // Parse primary ID (for replicas)
    let primary_id = if is_replica && parts[3] != "-" {
        Some(parts[3].to_string())
    } else {
        None
    };

    // Parse slot ranges (parts[8..])
    let mut slots = Vec::new();
    for slot_spec in parts.iter().skip(8) {
        if slot_spec.starts_with('[') {
            continue; // Skip migrating/importing slots
        }
        if let Some(range) = parse_slot_range(slot_spec) {
            slots.extend(range);
        }
    }

    Some(ClusterNode {
        id,
        host,
        port,
        bus_port,
        is_primary,
        is_replica,
        primary_id,
        slots,
        flags,
    })
}

/// Parse slot range like "0-5460" or "5461"
fn parse_slot_range(spec: &str) -> Option<Vec<u16>> {
    if let Some((start, end)) = spec.split_once('-') {
        let start: u16 = start.parse().ok()?;
        let end: u16 = end.parse().ok()?;
        Some((start..=end).collect())
    } else {
        let slot: u16 = spec.parse().ok()?;
        Some(vec![slot])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_primary_node() {
        let line = "07c37dfeb235213a872192d90877d0cd55635b91 127.0.0.1:7000@17000 myself,master - 0 0 1 connected 0-5460";
        let node = parse_cluster_node_line(line).unwrap();
        
        assert_eq!(node.host, "127.0.0.1");
        assert_eq!(node.port, 7000);
        assert!(node.is_primary);
        assert!(!node.is_replica);
        assert_eq!(node.slots.len(), 5461);
    }

    #[test]
    fn test_parse_replica_node() {
        let line = "e7d1eecce10fd6bb5eb35b9f99a514335d9ba9ca 127.0.0.1:7003@17003 slave 07c37dfeb235213a872192d90877d0cd55635b91 0 0 1 connected";
        let node = parse_cluster_node_line(line).unwrap();
        
        assert!(node.is_replica);
        assert!(!node.is_primary);
        assert!(node.primary_id.is_some());
    }
}
```

---

### Task 5.2: Implement Cluster Topology

**Description:** Create cluster topology with slot mapping.

**Instructions:**
Create `src/cluster/topology.rs`:

```rust
//! Cluster topology management

use std::collections::HashMap;
use crate::config::ReadFromReplica;
use super::node::{ClusterNode, parse_cluster_node_line};

/// Cluster topology snapshot
#[derive(Debug, Clone)]
pub struct ClusterTopology {
    /// All nodes in the cluster
    pub nodes: Vec<ClusterNode>,
    /// Slot to node index mapping
    slot_map: [Option<usize>; 16384],
    /// Primary node indices
    primary_indices: Vec<usize>,
    /// Replica node indices grouped by primary
    replica_map: HashMap<String, Vec<usize>>,
}

impl ClusterTopology {
    /// Parse CLUSTER NODES response
    pub fn from_cluster_nodes(response: &str) -> Result<Self, String> {
        let mut nodes = Vec::new();
        let mut slot_map = [None; 16384];
        let mut primary_indices = Vec::new();
        let mut replica_map: HashMap<String, Vec<usize>> = HashMap::new();

        for line in response.lines() {
            if line.is_empty() {
                continue;
            }

            if let Some(node) = parse_cluster_node_line(line) {
                let idx = nodes.len();

                if node.is_primary {
                    primary_indices.push(idx);
                    for &slot in &node.slots {
                        slot_map[slot as usize] = Some(idx);
                    }
                    replica_map.insert(node.id.clone(), Vec::new());
                }

                nodes.push(node);
            }
        }

        // Map replicas to primaries
        for (idx, node) in nodes.iter().enumerate() {
            if node.is_replica {
                if let Some(ref primary_id) = node.primary_id {
                    if let Some(replicas) = replica_map.get_mut(primary_id) {
                        replicas.push(idx);
                    }
                }
            }
        }

        if primary_indices.is_empty() {
            return Err("No primary nodes found".to_string());
        }

        Ok(Self {
            nodes,
            slot_map,
            primary_indices,
            replica_map,
        })
    }

    /// Get node for slot
    pub fn get_node_for_slot(&self, slot: u16) -> Option<&ClusterNode> {
        self.slot_map[slot as usize].map(|idx| &self.nodes[idx])
    }

    /// Get all primary nodes
    pub fn primaries(&self) -> impl Iterator<Item = &ClusterNode> {
        self.primary_indices.iter().map(|&idx| &self.nodes[idx])
    }

    /// Get replicas for a primary
    pub fn replicas_for(&self, primary_id: &str) -> Vec<&ClusterNode> {
        self.replica_map
            .get(primary_id)
            .map(|indices| indices.iter().map(|&idx| &self.nodes[idx]).collect())
            .unwrap_or_default()
    }

    /// Get nodes based on read-from-replica strategy
    pub fn get_selected_nodes(&self, strategy: &ReadFromReplica) -> Vec<&ClusterNode> {
        match strategy {
            ReadFromReplica::Primary => {
                self.primaries().collect()
            }
            ReadFromReplica::PreferReplica => {
                // Return replicas first, then primaries
                let mut nodes: Vec<&ClusterNode> = self.nodes.iter()
                    .filter(|n| n.is_replica && n.is_available())
                    .collect();
                if nodes.is_empty() {
                    nodes = self.primaries().collect();
                }
                nodes
            }
            ReadFromReplica::RoundRobin => {
                // All available nodes
                self.nodes.iter()
                    .filter(|n| n.is_available())
                    .collect()
            }
            ReadFromReplica::AzAffinity => {
                // Simplified: just return all nodes
                // Full implementation would filter by AZ
                self.nodes.iter()
                    .filter(|n| n.is_available())
                    .collect()
            }
        }
    }

    /// Get number of primary nodes
    pub fn num_primaries(&self) -> usize {
        self.primary_indices.len()
    }

    /// Get total number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Calculate slot for key using CRC16
    pub fn slot_for_key(key: &[u8]) -> u16 {
        // Check for hash tag {xxx}
        if let Some(start) = key.iter().position(|&b| b == b'{') {
            if let Some(end) = key[start..].iter().position(|&b| b == b'}') {
                if end > 1 {
                    return crc16(&key[start + 1..start + end]) % 16384;
                }
            }
        }
        crc16(key) % 16384
    }
}

/// CRC16 implementation for Redis cluster slot calculation
fn crc16(data: &[u8]) -> u16 {
    const CRC16_TABLE: [u16; 256] = [
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
        0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
        // ... (full table implementation)
        0x1ef0, 0x0ed1, 0x3eb2, 0x2e93, 0x5e74, 0x4e55, 0x7e36, 0x6e17,
        0xaef8, 0xbed9, 0x8eba, 0x9e9b, 0xee7c, 0xfe5d, 0xce3e, 0xde1f,
    ];
    
    let mut crc: u16 = 0;
    for &byte in data {
        let idx = ((crc >> 8) ^ (byte as u16)) as usize;
        crc = (crc << 8) ^ CRC16_TABLE[idx & 0xFF];
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_calculation() {
        // Test hash tag extraction
        assert_eq!(ClusterTopology::slot_for_key(b"{foo}bar"), ClusterTopology::slot_for_key(b"{foo}baz"));
    }
}
```

Update `src/cluster/mod.rs`:

```rust
//! Cluster topology and node management

pub mod node;
pub mod topology;

pub use node::ClusterNode;
pub use topology::ClusterTopology;
```

---

### Task 5.3: Implement Cluster Discovery

**Description:** Create cluster discovery that fetches topology from any node.

**Instructions:**
Create `src/cluster/discovery.rs`:

```rust
//! Cluster discovery

use crate::client::{RawConnection, ConnectionFactory};
use crate::utils::{RespEncoder, ClusterError};
use super::topology::ClusterTopology;

/// Discover cluster topology
pub fn discover_topology(
    factory: &ConnectionFactory,
    host: &str,
    port: u16,
) -> Result<ClusterTopology, ClusterError> {
    let mut conn = factory.create(host, port)
        .map_err(|e| ClusterError::RefreshFailed(e.to_string()))?;

    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["CLUSTER", "NODES"]);

    let response = conn.execute(&encoder)
        .map_err(|e| ClusterError::RefreshFailed(e.to_string()))?;

    let nodes_str = response.as_str()
        .ok_or_else(|| ClusterError::ParseFailed("Expected string response".to_string()))?;

    ClusterTopology::from_cluster_nodes(nodes_str)
        .map_err(ClusterError::ParseFailed)
}

/// Check if server is in cluster mode
pub fn is_cluster_mode(
    factory: &ConnectionFactory,
    host: &str,
    port: u16,
) -> Result<bool, ClusterError> {
    let mut conn = factory.create(host, port)
        .map_err(|e| ClusterError::RefreshFailed(e.to_string()))?;

    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["INFO", "CLUSTER"]);

    let response = conn.execute(&encoder)
        .map_err(|e| ClusterError::RefreshFailed(e.to_string()))?;

    let info_str = response.as_str().unwrap_or("");
    Ok(info_str.contains("cluster_enabled:1"))
}
```

---

### Task 5.4: Phase 5 Commit

```bash
git add .
git commit -m "Phase 5: Cluster support

- Implement ClusterNode representation
- Create ClusterTopology with slot mapping
- Add CLUSTER NODES response parsing
- Support read-from-replica strategies
- Add CRC16 slot calculation
- Implement cluster discovery

Cluster mode enabled with automatic node selection."
```

**Validation:**
- [ ] CLUSTER NODES parsing works
- [ ] Slot mapping is correct
- [ ] Node selection by strategy works
- [ ] Discovery from any node works

---

## Phase 6: Dataset Integration

**Objective:** Implement memory-mapped dataset access for 10B+ vectors.

**Git Branch:** `phase-6-dataset`

### Task 6.1: Implement Dataset Header

**Description:** Create C-compatible binary header structure.

**Instructions:**
Create `src/dataset/header.rs`:

```rust
//! Dataset header structure (C-compatible)

pub const DATASET_MAGIC: u32 = 0xDECDB001;
pub const HEADER_SIZE: usize = 4096;

/// Binary dataset header - matches C struct exactly
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DatasetHeader {
    pub magic: u32,
    pub version: u32,
    pub dataset_name: [u8; 256],
    pub distance_metric: u8,
    pub dtype: u8,
    pub has_metadata: u8,
    pub padding: [u8; 1],
    pub dim: u32,
    pub num_vectors: u64,
    pub num_queries: u64,
    pub num_neighbors: u32,
    pub vocab_size: u32,
    pub vectors_offset: u64,
    pub queries_offset: u64,
    pub ground_truth_offset: u64,
    pub vector_metadata_offset: u64,
    pub query_metadata_offset: u64,
    pub vocab_offset: u64,
    pub reserved: [u8; 3744],
}

impl DatasetHeader {
    /// Get dataset name as string
    pub fn name(&self) -> &str {
        let end = self.dataset_name.iter()
            .position(|&b| b == 0)
            .unwrap_or(self.dataset_name.len());
        std::str::from_utf8(&self.dataset_name[..end]).unwrap_or("unknown")
    }

    /// Get vector byte length
    pub fn vec_byte_len(&self) -> usize {
        let elem_size = match self.dtype {
            0 => 4, // FLOAT32
            1 => 2, // FLOAT16
            _ => 4,
        };
        self.dim as usize * elem_size
    }

    /// Get distance metric name
    pub fn distance_metric_str(&self) -> &'static str {
        match self.distance_metric {
            0 => "L2",
            1 => "IP",
            2 => "COSINE",
            _ => "UNKNOWN",
        }
    }
}
```

---

### Task 6.2: Implement Memory-Mapped Dataset

**Description:** Create zero-copy dataset access via mmap.

**Instructions:**
Create `src/dataset/binary_dataset.rs`:

```rust
//! Memory-mapped binary dataset

use std::fs::File;
use std::io;
use memmap2::Mmap;

use crate::utils::DatasetError;
use super::header::{DatasetHeader, DATASET_MAGIC, HEADER_SIZE};

/// Memory-mapped dataset context
/// 
/// Thread-safe: safe to share via Arc (mmap is read-only)
pub struct DatasetContext {
    mmap: Mmap,
    // Cached values (avoid packed struct access in hot path)
    dim: usize,
    num_vectors: u64,
    num_queries: u64,
    num_neighbors: usize,
    vectors_offset: usize,
    queries_offset: usize,
    ground_truth_offset: usize,
    vec_byte_len: usize,
}

impl DatasetContext {
    /// Open dataset file
    pub fn open(path: &str) -> Result<Self, DatasetError> {
        let file = File::open(path)
            .map_err(DatasetError::OpenFailed)?;

        let mmap = unsafe { Mmap::map(&file) }
            .map_err(DatasetError::OpenFailed)?;

        if mmap.len() < HEADER_SIZE {
            return Err(DatasetError::FileTooSmall {
                size: mmap.len() as u64,
                minimum: HEADER_SIZE as u64,
            });
        }

        // Parse header (packed struct, use unaligned read)
        let header: DatasetHeader = unsafe {
            std::ptr::read_unaligned(mmap.as_ptr() as *const DatasetHeader)
        };

        if header.magic != DATASET_MAGIC {
            return Err(DatasetError::InvalidMagic {
                expected: DATASET_MAGIC,
                actual: header.magic,
            });
        }

        let vec_byte_len = header.vec_byte_len();

        Ok(Self {
            mmap,
            dim: header.dim as usize,
            num_vectors: header.num_vectors,
            num_queries: header.num_queries,
            num_neighbors: header.num_neighbors as usize,
            vectors_offset: header.vectors_offset as usize,
            queries_offset: header.queries_offset as usize,
            ground_truth_offset: header.ground_truth_offset as usize,
            vec_byte_len,
        })
    }

    // === Accessors ===

    #[inline(always)]
    pub fn dim(&self) -> usize { self.dim }

    #[inline(always)]
    pub fn num_vectors(&self) -> u64 { self.num_vectors }

    #[inline(always)]
    pub fn num_queries(&self) -> u64 { self.num_queries }

    #[inline(always)]
    pub fn num_neighbors(&self) -> usize { self.num_neighbors }

    #[inline(always)]
    pub fn vec_byte_len(&self) -> usize { self.vec_byte_len }

    // === Zero-Copy Data Access ===

    /// Get raw bytes for vector at index (zero-copy)
    #[inline(always)]
    pub fn get_vector_bytes(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.num_vectors);
        let offset = self.vectors_offset + (idx as usize * self.vec_byte_len);
        &self.mmap[offset..offset + self.vec_byte_len]
    }

    /// Get raw bytes for query vector at index (zero-copy)
    #[inline(always)]
    pub fn get_query_bytes(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.num_queries);
        let offset = self.queries_offset + (idx as usize * self.vec_byte_len);
        &self.mmap[offset..offset + self.vec_byte_len]
    }

    /// Get ground truth neighbor IDs for query (zero-copy)
    #[inline]
    pub fn get_neighbor_ids(&self, query_idx: u64) -> &[u64] {
        debug_assert!(query_idx < self.num_queries);
        let offset = self.ground_truth_offset 
            + (query_idx as usize * self.num_neighbors * std::mem::size_of::<u64>());
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(offset) as *const u64,
                self.num_neighbors
            )
        }
    }

    /// Compute recall against ground truth
    pub fn compute_recall(&self, query_idx: u64, result_ids: &[u64], k: usize) -> f64 {
        let gt_ids = self.get_neighbor_ids(query_idx);
        let k = k.min(gt_ids.len()).min(result_ids.len());

        if k == 0 {
            return 0.0;
        }

        // Linear search for small k
        let mut matches = 0usize;
        for &result_id in &result_ids[..k] {
            if gt_ids[..k].contains(&result_id) {
                matches += 1;
            }
        }

        matches as f64 / k as f64
    }
}

// Safe to share across threads (mmap is read-only)
unsafe impl Send for DatasetContext {}
unsafe impl Sync for DatasetContext {}
```

Update `src/dataset/mod.rs`:

```rust
//! Dataset loading and access

pub mod header;
pub mod binary_dataset;

pub use header::DatasetHeader;
pub use binary_dataset::DatasetContext;
```

---

### Task 6.3: Phase 6 Commit

```bash
git add .
git commit -m "Phase 6: Dataset integration

- Implement C-compatible DatasetHeader
- Create memory-mapped DatasetContext
- Add zero-copy vector/query access
- Implement ground truth lookup
- Add recall computation

Zero-copy dataset access for 10B+ vectors."
```

**Validation:**
- [ ] Header parsing works
- [ ] Vector access is zero-copy
- [ ] Recall computation is correct
- [ ] Large files work (test with multi-GB file)

---

## Phase 7: Vector Search Commands

**Objective:** Implement FT.CREATE and FT.SEARCH with binary vectors.

**Git Branch:** `phase-7-vector-search`

### Task 7.1: Implement FT.CREATE Builder

**Description:** Create index creation command builder.

**Instructions:**
Create `src/workload/search_commands.rs`:

```rust
//! Vector search command builders

use crate::config::SearchConfig;
use crate::utils::RespEncoder;

/// Build FT.CREATE command for vector index
pub fn build_ft_create(config: &SearchConfig) -> Vec<u8> {
    let mut encoder = RespEncoder::with_capacity(512);
    
    let mut args: Vec<&str> = vec![
        "FT.CREATE",
        &config.index_name,
        "ON", "HASH",
        "PREFIX", "1", &config.prefix,
        "SCHEMA",
        &config.vector_field,
        "VECTOR",
        config.algorithm.as_str(),
    ];

    // Algorithm-specific parameters
    let dim_str = config.dim.to_string();
    let m_str;
    let ef_str;

    match config.algorithm {
        crate::config::VectorAlgorithm::Hnsw => {
            args.push("6"); // number of params
            args.push("TYPE");
            args.push("FLOAT32");
            args.push("DIM");
            args.push(&dim_str);
            args.push("DISTANCE_METRIC");
            args.push(config.distance_metric.as_str());
        }
        crate::config::VectorAlgorithm::Flat => {
            args.push("6");
            args.push("TYPE");
            args.push("FLOAT32");
            args.push("DIM");
            args.push(&dim_str);
            args.push("DISTANCE_METRIC");
            args.push(config.distance_metric.as_str());
        }
    }

    // Optional HNSW parameters
    if let Some(m) = config.hnsw_m {
        m_str = m.to_string();
        args.push("M");
        args.push(&m_str);
    }
    if let Some(ef) = config.ef_construction {
        ef_str = ef.to_string();
        args.push("EF_CONSTRUCTION");
        args.push(&ef_str);
    }

    let byte_args: Vec<&[u8]> = args.iter().map(|s| s.as_bytes()).collect();
    encoder.encode_command(&byte_args);
    encoder.into_bytes()
}

/// Build FT.DROPINDEX command
pub fn build_ft_drop(index_name: &str, delete_docs: bool) -> Vec<u8> {
    let mut encoder = RespEncoder::with_capacity(64);
    
    if delete_docs {
        encoder.encode_command_str(&["FT.DROPINDEX", index_name, "DD"]);
    } else {
        encoder.encode_command_str(&["FT.DROPINDEX", index_name]);
    }
    
    encoder.into_bytes()
}

/// Parse FT.SEARCH response to extract document IDs
pub fn parse_search_response(response: &crate::utils::RespValue) -> Vec<u64> {
    let mut ids = Vec::new();

    if let crate::utils::RespValue::Array(arr) = response {
        // Response: [total_count, doc_id1, [fields...], doc_id2, ...]
        // With NOCONTENT: [total_count, doc_id1, doc_id2, ...]
        
        for (i, item) in arr.iter().enumerate() {
            if i == 0 {
                continue; // Skip total count
            }
            
            // Document ID is bulk string
            if let crate::utils::RespValue::BulkString(id_bytes) = item {
                // Parse ID from key like "vec:12345"
                if let Ok(id_str) = std::str::from_utf8(id_bytes) {
                    if let Some(id_part) = id_str.rsplit(':').next() {
                        if let Ok(id) = id_part.parse::<u64>() {
                            ids.push(id);
                        }
                    }
                }
            }
        }
    }

    ids
}
```

---

### Task 7.2: Update Worker for Vector Search

**Description:** Integrate recall verification in worker.

**Instructions:**
Update `src/benchmark/worker.rs` to add recall verification:

```rust
// In BenchmarkWorker impl:

/// Verify recall for FT.SEARCH response
fn verify_recall(
    &mut self,
    response: &RespValue,
    query_idx: u64,
    dataset: &DatasetContext,
    k: usize,
) {
    let result_ids = parse_search_response(response);
    let recall = dataset.compute_recall(query_idx, &result_ids, k);
    self.recall_stats.record(recall);
}

// In process_responses, when handling VecQuery:
if let Some(&query_idx) = response.query_indices.get(i) {
    if let Some(ds) = dataset.as_ref() {
        self.verify_recall(&resp, query_idx, ds, self.config.search_config.as_ref().map(|s| s.k as usize).unwrap_or(10));
    }
}
```

---

### Task 7.3: Phase 7 Commit

```bash
git add .
git commit -m "Phase 7: Vector search commands

- Implement FT.CREATE command builder
- Add FT.DROPINDEX command builder
- Create FT.SEARCH response parser
- Integrate recall verification in worker
- Support NOCONTENT mode

Vector search benchmark with recall verification."
```

---

## Phase 8: Metrics and Reporting

**Objective:** Implement comprehensive metrics collection and multiple output formats.

**Git Branch:** `phase-8-metrics`

### Task 8.1: Implement Metrics Collector

**Description:** Create thread-safe metrics collection.

**Instructions:**
Create `src/metrics/collector.rs`:

```rust
//! Metrics collection

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use parking_lot::Mutex;
use hdrhistogram::Histogram;

/// Per-second statistics
#[derive(Debug, Clone)]
pub struct PerSecondStats {
    pub timestamp: Instant,
    pub requests: u64,
    pub throughput: f64,
    pub latency_avg_us: f64,
    pub latency_p50_us: u64,
    pub latency_p99_us: u64,
    pub errors: u64,
}

/// Metrics collector with per-second snapshots
pub struct MetricsCollector {
    histogram: Mutex<Histogram<u64>>,
    requests: AtomicU64,
    errors: AtomicU64,
    start_time: Instant,
    per_second_stats: Mutex<Vec<PerSecondStats>>,
    last_snapshot: Mutex<Instant>,
    last_requests: AtomicU64,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            histogram: Mutex::new(
                Histogram::new_with_bounds(1, 3_600_000_000, 3).unwrap()
            ),
            requests: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            start_time: Instant::now(),
            per_second_stats: Mutex::new(Vec::new()),
            last_snapshot: Mutex::new(Instant::now()),
            last_requests: AtomicU64::new(0),
        }
    }

    pub fn record_latency(&self, latency_us: u64) {
        self.histogram.lock().record(latency_us).ok();
        self.requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Take per-second snapshot
    pub fn snapshot(&self) {
        let now = Instant::now();
        let mut last = self.last_snapshot.lock();
        let elapsed = now.duration_since(*last).as_secs_f64();
        
        if elapsed < 0.9 {
            return; // Not time for snapshot yet
        }

        let current_requests = self.requests.load(Ordering::Relaxed);
        let last_requests = self.last_requests.swap(current_requests, Ordering::Relaxed);
        let delta = current_requests - last_requests;
        let throughput = delta as f64 / elapsed;

        let hist = self.histogram.lock();
        let stats = PerSecondStats {
            timestamp: now,
            requests: delta,
            throughput,
            latency_avg_us: hist.mean(),
            latency_p50_us: hist.value_at_percentile(50.0),
            latency_p99_us: hist.value_at_percentile(99.0),
            errors: self.errors.load(Ordering::Relaxed),
        };

        self.per_second_stats.lock().push(stats);
        *last = now;
    }

    pub fn get_per_second_stats(&self) -> Vec<PerSecondStats> {
        self.per_second_stats.lock().clone()
    }
}
```

---

### Task 8.2: Implement Output Formatters

**Description:** Create text, CSV, and JSON output formatters.

**Instructions:**
Create `src/metrics/reporter.rs`:

```rust
//! Output formatters

use std::io::Write;
use std::fs::File;
use crate::benchmark::BenchmarkResult;
use crate::config::OutputFormat;

/// Write benchmark results to specified format
pub fn write_results(
    results: &[BenchmarkResult],
    format: OutputFormat,
    output: Option<&str>,
) -> std::io::Result<()> {
    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(File::create(path)?),
        None => Box::new(std::io::stdout()),
    };

    match format {
        OutputFormat::Text => write_text(&mut writer, results),
        OutputFormat::Json => write_json(&mut writer, results),
        OutputFormat::Csv => write_csv(&mut writer, results),
    }
}

fn write_text(w: &mut dyn Write, results: &[BenchmarkResult]) -> std::io::Result<()> {
    for result in results {
        writeln!(w, "\n=== {} ===", result.test_name)?;
        writeln!(w, "Throughput: {:.2} req/sec", result.throughput)?;
        writeln!(w, "Requests: {}", result.total_requests)?;
        writeln!(w, "Duration: {:.2}s", result.duration.as_secs_f64())?;
        writeln!(w, "\nLatency (ms):")?;
        writeln!(w, "  avg: {:.3}", result.histogram.mean() / 1000.0)?;
        writeln!(w, "  p50: {:.3}", result.percentile_ms(50.0))?;
        writeln!(w, "  p95: {:.3}", result.percentile_ms(95.0))?;
        writeln!(w, "  p99: {:.3}", result.percentile_ms(99.0))?;
        writeln!(w, "  p99.9: {:.3}", result.percentile_ms(99.9))?;
        
        if result.recall_stats.total_queries > 0 {
            writeln!(w, "\nRecall: {:.4}", result.recall_stats.average())?;
        }
    }
    Ok(())
}

fn write_json(w: &mut dyn Write, results: &[BenchmarkResult]) -> std::io::Result<()> {
    writeln!(w, "{{")?;
    writeln!(w, "  \"results\": [")?;
    
    for (i, result) in results.iter().enumerate() {
        writeln!(w, "    {{")?;
        writeln!(w, "      \"test\": \"{}\",", result.test_name)?;
        writeln!(w, "      \"throughput\": {:.2},", result.throughput)?;
        writeln!(w, "      \"requests\": {},", result.total_requests)?;
        writeln!(w, "      \"latency_avg_ms\": {:.3},", result.histogram.mean() / 1000.0)?;
        writeln!(w, "      \"latency_p99_ms\": {:.3}", result.percentile_ms(99.0))?;
        write!(w, "    }}")?;
        if i < results.len() - 1 {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }
    
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn write_csv(w: &mut dyn Write, results: &[BenchmarkResult]) -> std::io::Result<()> {
    writeln!(w, "test,throughput,requests,latency_avg_ms,latency_p50_ms,latency_p99_ms,recall")?;
    
    for result in results {
        writeln!(w, "{},{:.2},{},{:.3},{:.3},{:.3},{:.4}",
            result.test_name,
            result.throughput,
            result.total_requests,
            result.histogram.mean() / 1000.0,
            result.percentile_ms(50.0),
            result.percentile_ms(99.0),
            result.recall_stats.average()
        )?;
    }
    Ok(())
}
```

Update `src/metrics/mod.rs`:

```rust
pub mod collector;
pub mod reporter;

pub use collector::{MetricsCollector, PerSecondStats};
pub use reporter::write_results;
```

---

### Task 8.3: Phase 8 Commit

```bash
git add .
git commit -m "Phase 8: Metrics and reporting

- Implement MetricsCollector with per-second stats
- Add text, JSON, CSV output formatters
- Support file and stdout output
- Include recall in all formats

Full metrics with multiple output formats."
```

---

## Phase 9: Per-Node Metrics and Temporal Diff Analysis

**Objective:** Implement comprehensive per-node metrics tracking with temporal diff calculation to detect load imbalance and execution issues between nodes.

**Git Branch:** `phase-9-node-metrics`

### Task 9.1: Define Metrics Field Types

**Description:** Create abstraction for user-defined metrics with aggregation and diff types.

**Instructions:**
Create `src/metrics/info_fields.rs`:

```rust
//! INFO command field definitions and parsing
//!
//! Supports user-defined metrics with per-node tracking,
//! aggregation types, and temporal diff calculation.

use std::collections::HashMap;

/// How to aggregate values across nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationType {
    /// Sum values across all nodes
    Sum,
    /// Average across nodes
    Average,
    /// Maximum value
    Max,
    /// Minimum value
    Min,
    /// Show min and max range
    MinMax,
    /// No aggregation (per-node only)
    None,
}

/// How to display the metric value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayFormat {
    /// Raw integer
    Integer,
    /// Floating point with precision
    Float { decimals: u8 },
    /// Memory in human-readable format (KB, MB, GB)
    Memory,
    /// Percentage (0-100)
    Percentage,
    /// Rate per second
    RatePerSecond,
    /// Duration in milliseconds
    DurationMs,
}

/// How to compute diff between snapshots
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffType {
    /// Rate of change (delta / elapsed_seconds)
    RateCount,
    /// Absolute change
    Delta,
    /// Memory growth rate (delta_bytes / elapsed_seconds)
    MemoryGrowth,
    /// Percentage change ((new - old) / old * 100)
    PercentChange,
    /// No diff calculation
    None,
}

/// Which nodes to include in aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeFilter {
    /// Only primary nodes
    PrimaryOnly,
    /// Only replica nodes
    ReplicaOnly,
    /// All nodes
    All,
}

/// Definition of a metric field to collect from INFO
#[derive(Debug, Clone)]
pub struct InfoFieldDef {
    /// Display name for the metric
    pub name: String,
    /// INFO section to search (e.g., "server", "memory", "search")
    pub section: String,
    /// Field prefix or exact name to match
    pub field_match: FieldMatch,
    /// How to parse the value
    pub parse_type: ParseType,
    /// How to aggregate across nodes
    pub aggregation: AggregationType,
    /// How to display
    pub display: DisplayFormat,
    /// How to compute diff
    pub diff_type: DiffType,
    /// Store per-node values
    pub track_per_node: bool,
    /// Which nodes to include
    pub node_filter: NodeFilter,
    /// Description for help/documentation
    pub description: String,
}

/// How to match field names
#[derive(Debug, Clone)]
pub enum FieldMatch {
    /// Exact field name
    Exact(String),
    /// Field name prefix
    Prefix(String),
    /// Custom regex pattern
    Pattern(String),
}

/// How to parse field value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseType {
    /// Parse as i64
    Integer,
    /// Parse as f64
    Float,
    /// Parse as boolean (0/1, yes/no)
    Boolean,
    /// Keep as string
    String,
    /// Parse memory string (e.g., "1.5G", "512M")
    MemoryString,
}

/// Parsed field value
#[derive(Debug, Clone)]
pub enum FieldValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
}

impl FieldValue {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            FieldValue::Integer(v) => Some(*v),
            FieldValue::Float(v) => Some(*v as i64),
            FieldValue::Boolean(v) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FieldValue::Integer(v) => Some(*v as f64),
            FieldValue::Float(v) => Some(*v),
            FieldValue::Boolean(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
}

/// Default INFO fields for vector search benchmarking
pub fn default_vss_fields() -> Vec<InfoFieldDef> {
    vec![
        // Memory metrics
        InfoFieldDef {
            name: "used_memory".to_string(),
            section: "memory".to_string(),
            field_match: FieldMatch::Exact("used_memory".to_string()),
            parse_type: ParseType::Integer,
            aggregation: AggregationType::Sum,
            display: DisplayFormat::Memory,
            diff_type: DiffType::MemoryGrowth,
            track_per_node: true,
            node_filter: NodeFilter::All,
            description: "Total memory used by Valkey".to_string(),
        },
        InfoFieldDef {
            name: "used_memory_rss".to_string(),
            section: "memory".to_string(),
            field_match: FieldMatch::Exact("used_memory_rss".to_string()),
            parse_type: ParseType::Integer,
            aggregation: AggregationType::Sum,
            display: DisplayFormat::Memory,
            diff_type: DiffType::MemoryGrowth,
            track_per_node: true,
            node_filter: NodeFilter::All,
            description: "Resident set size memory".to_string(),
        },
        // Search/VSS metrics
        InfoFieldDef {
            name: "search_used_memory".to_string(),
            section: "search".to_string(),
            field_match: FieldMatch::Exact("search_used_memory_bytes".to_string()),
            parse_type: ParseType::Integer,
            aggregation: AggregationType::Sum,
            display: DisplayFormat::Memory,
            diff_type: DiffType::MemoryGrowth,
            track_per_node: true,
            node_filter: NodeFilter::PrimaryOnly,
            description: "Memory used by search indexes".to_string(),
        },
        InfoFieldDef {
            name: "search_total_indexing_time".to_string(),
            section: "search".to_string(),
            field_match: FieldMatch::Exact("search_total_indexing_time".to_string()),
            parse_type: ParseType::Integer,
            aggregation: AggregationType::Sum,
            display: DisplayFormat::DurationMs,
            diff_type: DiffType::RateCount,
            track_per_node: true,
            node_filter: NodeFilter::PrimaryOnly,
            description: "Cumulative indexing time".to_string(),
        },
        // Request counters
        InfoFieldDef {
            name: "total_commands_processed".to_string(),
            section: "stats".to_string(),
            field_match: FieldMatch::Exact("total_commands_processed".to_string()),
            parse_type: ParseType::Integer,
            aggregation: AggregationType::Sum,
            display: DisplayFormat::RatePerSecond,
            diff_type: DiffType::RateCount,
            track_per_node: true,
            node_filter: NodeFilter::All,
            description: "Total commands processed".to_string(),
        },
        InfoFieldDef {
            name: "instantaneous_ops_per_sec".to_string(),
            section: "stats".to_string(),
            field_match: FieldMatch::Exact("instantaneous_ops_per_sec".to_string()),
            parse_type: ParseType::Integer,
            aggregation: AggregationType::Sum,
            display: DisplayFormat::Integer,
            diff_type: DiffType::None,
            track_per_node: true,
            node_filter: NodeFilter::All,
            description: "Current ops/sec".to_string(),
        },
    ]
}

/// Parse INFO response into field values
pub fn parse_info_response(
    info: &str,
    fields: &[InfoFieldDef],
) -> HashMap<String, FieldValue> {
    let mut result = HashMap::new();
    let mut current_section = String::new();

    for line in info.lines() {
        let line = line.trim();
        
        // Section header
        if line.starts_with('#') {
            current_section = line.trim_start_matches('#').trim().to_lowercase();
            continue;
        }

        // Field: key:value
        if let Some((key, value)) = line.split_once(':') {
            for field in fields {
                if field.section.to_lowercase() != current_section {
                    continue;
                }

                let matches = match &field.field_match {
                    FieldMatch::Exact(name) => key == name,
                    FieldMatch::Prefix(prefix) => key.starts_with(prefix),
                    FieldMatch::Pattern(_pattern) => false, // TODO: regex
                };

                if matches {
                    if let Some(parsed) = parse_value(value, field.parse_type) {
                        result.insert(field.name.clone(), parsed);
                    }
                }
            }
        }
    }

    result
}

fn parse_value(value: &str, parse_type: ParseType) -> Option<FieldValue> {
    match parse_type {
        ParseType::Integer => value.parse().ok().map(FieldValue::Integer),
        ParseType::Float => value.parse().ok().map(FieldValue::Float),
        ParseType::Boolean => {
            match value.to_lowercase().as_str() {
                "1" | "yes" | "true" => Some(FieldValue::Boolean(true)),
                "0" | "no" | "false" => Some(FieldValue::Boolean(false)),
                _ => None,
            }
        }
        ParseType::String => Some(FieldValue::String(value.to_string())),
        ParseType::MemoryString => parse_memory_string(value).map(FieldValue::Integer),
    }
}

fn parse_memory_string(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Some(num) = s.strip_suffix('K') {
        num.parse::<f64>().ok().map(|n| (n * 1024.0) as i64)
    } else if let Some(num) = s.strip_suffix('M') {
        num.parse::<f64>().ok().map(|n| (n * 1024.0 * 1024.0) as i64)
    } else if let Some(num) = s.strip_suffix('G') {
        num.parse::<f64>().ok().map(|n| (n * 1024.0 * 1024.0 * 1024.0) as i64)
    } else {
        s.parse().ok()
    }
}
```

**Validation:**
- INFO parsing extracts correct field values
- Memory string parsing handles K/M/G suffixes
- Field matching works for exact and prefix

---

### Task 9.2: Implement Cluster Snapshot Collection

**Description:** Create cluster-wide snapshot of metrics with per-node values.

**Instructions:**
Create `src/metrics/snapshot.rs`:

```rust
//! Cluster snapshot collection for temporal analysis

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

use crate::client::{RawConnection, ConnectionFactory};
use crate::cluster::{ClusterTopology, ClusterNode, StandaloneTopology};
use crate::utils::RespEncoder;

use super::info_fields::{InfoFieldDef, FieldValue, parse_info_response, NodeFilter};

/// Per-node field value
#[derive(Debug, Clone)]
pub struct NodeFieldValue {
    pub node_id: String,
    pub host: String,
    pub port: u16,
    pub is_primary: bool,
    pub value: FieldValue,
}

/// Snapshot of a single field across the cluster
#[derive(Debug, Clone)]
pub struct FieldSnapshot {
    pub field_name: String,
    /// Aggregated value (sum, avg, etc.)
    pub aggregated: Option<FieldValue>,
    /// Per-node values (if track_per_node is true)
    pub per_node: Vec<NodeFieldValue>,
}

/// Complete cluster snapshot at a point in time
#[derive(Debug, Clone)]
pub struct ClusterSnapshot {
    /// Timestamp when snapshot was taken
    pub timestamp: Instant,
    /// Wall clock time
    pub wall_time: SystemTime,
    /// Number of nodes sampled
    pub num_nodes: usize,
    /// Node identifiers (for correlation)
    pub node_ids: Vec<String>,
    /// Field snapshots
    pub fields: HashMap<String, FieldSnapshot>,
}

impl ClusterSnapshot {
    /// Collect snapshot from cluster topology
    pub fn collect_cluster(
        factory: &ConnectionFactory,
        topology: &ClusterTopology,
        field_defs: &[InfoFieldDef],
    ) -> Result<Self, String> {
        let timestamp = Instant::now();
        let wall_time = SystemTime::now();
        let mut fields: HashMap<String, FieldSnapshot> = HashMap::new();
        let mut node_ids = Vec::new();

        // Initialize field snapshots
        for def in field_defs {
            fields.insert(def.name.clone(), FieldSnapshot {
                field_name: def.name.clone(),
                aggregated: None,
                per_node: Vec::new(),
            });
        }

        // Collect from each node
        for node in &topology.nodes {
            // Filter nodes based on field definitions
            let node_id = node.metrics_id();
            node_ids.push(node_id.clone());

            // Connect and get INFO
            let info = match collect_node_info(factory, node) {
                Ok(info) => info,
                Err(e) => {
                    eprintln!("Warning: Failed to collect INFO from {}: {}", node.address(), e);
                    continue;
                }
            };

            // Parse fields
            let parsed = parse_info_response(&info, field_defs);

            // Store values
            for def in field_defs {
                // Check node filter
                let include = match def.node_filter {
                    NodeFilter::PrimaryOnly => node.is_primary,
                    NodeFilter::ReplicaOnly => node.is_replica,
                    NodeFilter::All => true,
                };

                if !include {
                    continue;
                }

                if let Some(value) = parsed.get(&def.name) {
                    let snapshot = fields.get_mut(&def.name).unwrap();
                    
                    if def.track_per_node {
                        snapshot.per_node.push(NodeFieldValue {
                            node_id: node_id.clone(),
                            host: node.host.clone(),
                            port: node.port,
                            is_primary: node.is_primary,
                            value: value.clone(),
                        });
                    }
                }
            }
        }

        // Compute aggregates
        for def in field_defs {
            let snapshot = fields.get_mut(&def.name).unwrap();
            snapshot.aggregated = compute_aggregate(&snapshot.per_node, def);
        }

        Ok(Self {
            timestamp,
            wall_time,
            num_nodes: node_ids.len(),
            node_ids,
            fields,
        })
    }

    /// Collect snapshot from standalone topology
    pub fn collect_standalone(
        factory: &ConnectionFactory,
        topology: &StandaloneTopology,
        field_defs: &[InfoFieldDef],
    ) -> Result<Self, String> {
        // Similar to cluster, but with primary + replicas
        let timestamp = Instant::now();
        let wall_time = SystemTime::now();
        let mut fields: HashMap<String, FieldSnapshot> = HashMap::new();
        let mut node_ids = Vec::new();

        // Initialize
        for def in field_defs {
            fields.insert(def.name.clone(), FieldSnapshot {
                field_name: def.name.clone(),
                aggregated: None,
                per_node: Vec::new(),
            });
        }

        // Collect from primary
        // ... (similar to cluster collection)

        Ok(Self {
            timestamp,
            wall_time,
            num_nodes: node_ids.len(),
            node_ids,
            fields,
        })
    }
}

fn collect_node_info(factory: &ConnectionFactory, node: &ClusterNode) -> Result<String, String> {
    let mut conn = factory.create(&node.host, node.port)
        .map_err(|e| e.to_string())?;

    let mut encoder = RespEncoder::with_capacity(32);
    encoder.encode_command_str(&["INFO", "ALL"]);

    let response = conn.execute(&encoder)
        .map_err(|e| e.to_string())?;

    response.as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Invalid INFO response".to_string())
}

fn compute_aggregate(
    values: &[NodeFieldValue],
    def: &InfoFieldDef,
) -> Option<FieldValue> {
    if values.is_empty() {
        return None;
    }

    let nums: Vec<f64> = values.iter()
        .filter_map(|v| v.value.as_f64())
        .collect();

    if nums.is_empty() {
        return None;
    }

    use super::info_fields::AggregationType::*;
    
    let result = match def.aggregation {
        Sum => nums.iter().sum(),
        Average => nums.iter().sum::<f64>() / nums.len() as f64,
        Max => nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        Min => nums.iter().cloned().fold(f64::INFINITY, f64::min),
        MinMax => return None, // Handled separately
        None => return None,
    };

    Some(FieldValue::Float(result))
}
```

**Validation:**
- Snapshot collection gathers from all nodes
- Aggregation computes correctly
- Node filtering respects field definitions

---

### Task 9.3: Implement Temporal Diff Calculator

**Description:** Calculate diff between before/after snapshots with rate computation.

**Instructions:**
Create `src/metrics/diff.rs`:

```rust
//! Temporal diff calculation between snapshots

use std::collections::HashMap;
use super::info_fields::{InfoFieldDef, FieldValue, DiffType, DisplayFormat};
use super::snapshot::{ClusterSnapshot, FieldSnapshot, NodeFieldValue};

/// Diff result for a single field
#[derive(Debug, Clone)]
pub struct FieldDiff {
    pub field_name: String,
    /// Aggregated diff
    pub aggregated_diff: Option<DiffValue>,
    /// Per-node diffs
    pub per_node_diffs: Vec<NodeDiff>,
    /// Detected imbalance (if any)
    pub imbalance: Option<ImbalanceInfo>,
}

/// Calculated diff value
#[derive(Debug, Clone)]
pub struct DiffValue {
    pub old_value: f64,
    pub new_value: f64,
    pub delta: f64,
    /// Computed rate (delta / elapsed_seconds)
    pub rate: Option<f64>,
    /// Formatted string for display
    pub formatted: String,
}

/// Per-node diff
#[derive(Debug, Clone)]
pub struct NodeDiff {
    pub node_id: String,
    pub host: String,
    pub port: u16,
    pub diff: DiffValue,
}

/// Detected imbalance between nodes
#[derive(Debug, Clone)]
pub struct ImbalanceInfo {
    /// Node with highest value
    pub max_node: String,
    /// Node with lowest value
    pub min_node: String,
    /// Ratio of max/min (>1.0 indicates imbalance)
    pub ratio: f64,
    /// Percentage deviation from mean
    pub max_deviation_pct: f64,
    /// Description of the imbalance
    pub description: String,
}

/// Complete diff report between two snapshots
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    /// Elapsed time between snapshots
    pub elapsed_secs: f64,
    /// Field diffs
    pub fields: HashMap<String, FieldDiff>,
    /// Nodes that appeared/disappeared
    pub node_changes: NodeChanges,
}

#[derive(Debug, Clone, Default)]
pub struct NodeChanges {
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

impl SnapshotDiff {
    /// Calculate diff between two snapshots
    pub fn calculate(
        before: &ClusterSnapshot,
        after: &ClusterSnapshot,
        field_defs: &[InfoFieldDef],
    ) -> Self {
        let elapsed_secs = after.timestamp.duration_since(before.timestamp).as_secs_f64();
        let mut fields = HashMap::new();

        // Detect node changes
        let node_changes = detect_node_changes(&before.node_ids, &after.node_ids);

        for def in field_defs {
            if def.diff_type == DiffType::None {
                continue;
            }

            let before_field = before.fields.get(&def.name);
            let after_field = after.fields.get(&def.name);

            if let (Some(bf), Some(af)) = (before_field, after_field) {
                let field_diff = calculate_field_diff(bf, af, def, elapsed_secs);
                fields.insert(def.name.clone(), field_diff);
            }
        }

        Self {
            elapsed_secs,
            fields,
            node_changes,
        }
    }

    /// Print human-readable diff report
    pub fn print_report(&self, field_defs: &[InfoFieldDef]) {
        println!("\n=== Metrics Diff Report (elapsed: {:.2}s) ===\n", self.elapsed_secs);

        // Print node changes
        if !self.node_changes.added.is_empty() {
            println!("Nodes added: {:?}", self.node_changes.added);
        }
        if !self.node_changes.removed.is_empty() {
            println!("Nodes removed: {:?}", self.node_changes.removed);
        }

        // Print aggregated diffs
        println!("\n--- Aggregated Metrics ---");
        for def in field_defs {
            if let Some(field_diff) = self.fields.get(&def.name) {
                if let Some(ref agg) = field_diff.aggregated_diff {
                    println!("{}: {}", def.name, agg.formatted);
                }
            }
        }

        // Print per-node diffs with imbalance detection
        println!("\n--- Per-Node Analysis ---");
        for def in field_defs {
            if let Some(field_diff) = self.fields.get(&def.name) {
                if !field_diff.per_node_diffs.is_empty() {
                    println!("\n{}:", def.name);
                    for nd in &field_diff.per_node_diffs {
                        println!("  {}: {}", nd.node_id, nd.diff.formatted);
                    }

                    // Report imbalance
                    if let Some(ref imbalance) = field_diff.imbalance {
                        println!("  ⚠️  IMBALANCE: {}", imbalance.description);
                    }
                }
            }
        }
    }

    /// Get detected imbalances
    pub fn get_imbalances(&self) -> Vec<(&str, &ImbalanceInfo)> {
        self.fields.iter()
            .filter_map(|(name, diff)| {
                diff.imbalance.as_ref().map(|i| (name.as_str(), i))
            })
            .collect()
    }
}

fn detect_node_changes(before: &[String], after: &[String]) -> NodeChanges {
    let before_set: std::collections::HashSet<_> = before.iter().collect();
    let after_set: std::collections::HashSet<_> = after.iter().collect();

    NodeChanges {
        added: after_set.difference(&before_set).map(|s| (*s).clone()).collect(),
        removed: before_set.difference(&after_set).map(|s| (*s).clone()).collect(),
    }
}

fn calculate_field_diff(
    before: &FieldSnapshot,
    after: &FieldSnapshot,
    def: &InfoFieldDef,
    elapsed_secs: f64,
) -> FieldDiff {
    // Aggregated diff
    let aggregated_diff = match (&before.aggregated, &after.aggregated) {
        (Some(bv), Some(av)) => {
            let b = bv.as_f64().unwrap_or(0.0);
            let a = av.as_f64().unwrap_or(0.0);
            Some(compute_diff_value(b, a, def, elapsed_secs))
        }
        _ => None,
    };

    // Per-node diffs
    let mut per_node_diffs = Vec::new();
    for after_node in &after.per_node {
        if let Some(before_node) = before.per_node.iter()
            .find(|n| n.node_id == after_node.node_id)
        {
            let b = before_node.value.as_f64().unwrap_or(0.0);
            let a = after_node.value.as_f64().unwrap_or(0.0);
            
            per_node_diffs.push(NodeDiff {
                node_id: after_node.node_id.clone(),
                host: after_node.host.clone(),
                port: after_node.port,
                diff: compute_diff_value(b, a, def, elapsed_secs),
            });
        }
    }

    // Detect imbalance
    let imbalance = detect_imbalance(&per_node_diffs, def);

    FieldDiff {
        field_name: def.name.clone(),
        aggregated_diff,
        per_node_diffs,
        imbalance,
    }
}

fn compute_diff_value(
    old: f64,
    new: f64,
    def: &InfoFieldDef,
    elapsed_secs: f64,
) -> DiffValue {
    let delta = new - old;
    
    let (rate, formatted) = match def.diff_type {
        DiffType::RateCount => {
            let r = delta / elapsed_secs;
            (Some(r), format!("{:.2}/sec", r))
        }
        DiffType::Delta => {
            (None, format_value(delta, def.display))
        }
        DiffType::MemoryGrowth => {
            let r = delta / elapsed_secs;
            let formatted = if r.abs() > 1_048_576.0 {
                format!("{:.2} MB/sec", r / 1_048_576.0)
            } else if r.abs() > 1024.0 {
                format!("{:.2} KB/sec", r / 1024.0)
            } else {
                format!("{:.2} B/sec", r)
            };
            (Some(r), formatted)
        }
        DiffType::PercentChange => {
            let pct = if old != 0.0 { (delta / old) * 100.0 } else { 0.0 };
            (None, format!("{:+.2}%", pct))
        }
        DiffType::None => {
            (None, format_value(new, def.display))
        }
    };

    DiffValue {
        old_value: old,
        new_value: new,
        delta,
        rate,
        formatted,
    }
}

fn format_value(value: f64, display: DisplayFormat) -> String {
    match display {
        DisplayFormat::Integer => format!("{:.0}", value),
        DisplayFormat::Float { decimals } => format!("{:.*}", decimals as usize, value),
        DisplayFormat::Memory => format_memory(value as i64),
        DisplayFormat::Percentage => format!("{:.2}%", value),
        DisplayFormat::RatePerSecond => format!("{:.2}/sec", value),
        DisplayFormat::DurationMs => format!("{:.2}ms", value),
    }
}

fn format_memory(bytes: i64) -> String {
    let abs = bytes.abs() as f64;
    let sign = if bytes < 0 { "-" } else { "" };
    
    if abs >= 1_073_741_824.0 {
        format!("{}{:.2} GB", sign, abs / 1_073_741_824.0)
    } else if abs >= 1_048_576.0 {
        format!("{}{:.2} MB", sign, abs / 1_048_576.0)
    } else if abs >= 1024.0 {
        format!("{}{:.2} KB", sign, abs / 1024.0)
    } else {
        format!("{}{} B", sign, bytes)
    }
}

fn detect_imbalance(
    node_diffs: &[NodeDiff],
    def: &InfoFieldDef,
) -> Option<ImbalanceInfo> {
    if node_diffs.len() < 2 {
        return None;
    }

    // Get rates or deltas
    let values: Vec<(String, f64)> = node_diffs.iter()
        .map(|nd| {
            let v = nd.diff.rate.unwrap_or(nd.diff.delta);
            (nd.node_id.clone(), v)
        })
        .collect();

    let (min_node, min_val) = values.iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
    let (max_node, max_val) = values.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;

    // Calculate ratio and deviation
    let ratio = if *min_val != 0.0 { max_val / min_val } else { f64::INFINITY };
    let mean: f64 = values.iter().map(|(_, v)| v).sum::<f64>() / values.len() as f64;
    let max_deviation = if mean != 0.0 {
        ((max_val - mean) / mean * 100.0).abs()
    } else {
        0.0
    };

    // Only report significant imbalance (>20% deviation)
    if max_deviation > 20.0 || ratio > 1.5 {
        Some(ImbalanceInfo {
            max_node: max_node.clone(),
            min_node: min_node.clone(),
            ratio,
            max_deviation_pct: max_deviation,
            description: format!(
                "Max ({}) is {:.1}x higher than Min ({}), {:.1}% deviation from mean",
                max_node, ratio, min_node, max_deviation
            ),
        })
    } else {
        None
    }
}
```

**Validation:**
- Diff calculation produces correct rates
- Imbalance detection triggers on significant deviations
- Memory formatting is human-readable

---

### Task 9.4: Implement User-Defined Metrics

**Description:** Allow users to specify custom metrics to track via CLI.

**Instructions:**
Add to CLI in `src/config/cli.rs`:

```rust
// In CliArgs struct:

/// Custom INFO fields to track (format: section:field:aggtype)
#[arg(long = "track-metric", action = clap::ArgAction::Append)]
pub track_metrics: Vec<String>,

/// Imbalance threshold percentage (default 20%)
#[arg(long = "imbalance-threshold", default_value_t = 20.0)]
pub imbalance_threshold: f64,

/// Save metrics snapshot to file
#[arg(long = "metrics-snapshot")]
pub metrics_snapshot_file: Option<PathBuf>,

/// Compare with previous snapshot file
#[arg(long = "compare-snapshot")]
pub compare_snapshot_file: Option<PathBuf>,
```

Create `src/metrics/user_metrics.rs`:

```rust
//! User-defined metrics parsing

use super::info_fields::{InfoFieldDef, FieldMatch, ParseType, AggregationType, DisplayFormat, DiffType, NodeFilter};

/// Parse user metric specification
/// Format: section:field[:aggtype[:difftype]]
/// Example: stats:total_commands_processed:sum:rate
pub fn parse_user_metric(spec: &str) -> Result<InfoFieldDef, String> {
    let parts: Vec<&str> = spec.split(':').collect();
    
    if parts.len() < 2 {
        return Err(format!("Invalid metric spec '{}': expected section:field[:aggtype[:difftype]]", spec));
    }

    let section = parts[0].to_string();
    let field = parts[1].to_string();
    let agg = parts.get(2).copied().unwrap_or("sum");
    let diff = parts.get(3).copied().unwrap_or("rate");

    let aggregation = match agg.to_lowercase().as_str() {
        "sum" => AggregationType::Sum,
        "avg" | "average" => AggregationType::Average,
        "max" => AggregationType::Max,
        "min" => AggregationType::Min,
        "none" => AggregationType::None,
        _ => return Err(format!("Unknown aggregation type: {}", agg)),
    };

    let diff_type = match diff.to_lowercase().as_str() {
        "rate" => DiffType::RateCount,
        "delta" => DiffType::Delta,
        "memory" => DiffType::MemoryGrowth,
        "percent" => DiffType::PercentChange,
        "none" => DiffType::None,
        _ => return Err(format!("Unknown diff type: {}", diff)),
    };

    let display = match diff_type {
        DiffType::RateCount => DisplayFormat::RatePerSecond,
        DiffType::MemoryGrowth => DisplayFormat::Memory,
        DiffType::PercentChange => DisplayFormat::Percentage,
        _ => DisplayFormat::Integer,
    };

    Ok(InfoFieldDef {
        name: format!("{}:{}", section, field),
        section,
        field_match: FieldMatch::Exact(field),
        parse_type: ParseType::Integer,
        aggregation,
        display,
        diff_type,
        track_per_node: true,
        node_filter: NodeFilter::All,
        description: format!("User-defined metric: {}", spec),
    })
}
```

---

### Task 9.5: Phase 9 Commit and Validation

```bash
git add .
git commit -m "Phase 9: Per-node metrics and temporal diff analysis

- Define InfoFieldDef with aggregation and diff types
- Implement cluster-wide snapshot collection
- Add temporal diff calculator with rate computation
- Create imbalance detection algorithm
- Support user-defined metrics via CLI
- Add snapshot save/compare functionality

Comprehensive per-node metrics tracking for identifying load
imbalance and execution issues between nodes."
```

**Validation Checklist:**
- [ ] INFO parsing extracts all defined fields
- [ ] Snapshot collection gathers from all nodes
- [ ] Diff calculation produces correct rates
- [ ] Imbalance detection triggers on >20% deviation
- [ ] User-defined metrics are parsed correctly
- [ ] Memory formatting is human-readable
- [ ] Node changes (added/removed) are detected

---

## Phase 10: Cluster Tag Map and Key Routing

**Objective:** Implement cluster tag mapping for routing vectors to specific nodes via hash tags.

**Git Branch:** `phase-10-cluster-tags`

### Task 10.1: Implement Cluster Tag Map

**Description:** Create cluster tag map with dense/sparse storage for 10B+ vector scale.

**Instructions:**
Create `src/dataset/cluster_tag_map.rs`:

```rust
//! Cluster tag mapping for vector ID to cluster hash tag
//!
//! Supports 10B+ vectors with hybrid dense/sparse storage:
//! - Dense: Vec<Option<[u8; 5]>> for IDs < threshold
//! - Sparse: HashMap for IDs >= threshold

use std::collections::HashMap;
use parking_lot::RwLock;

/// Hash tag format: {xxx} where xxx is 3 characters
pub type ClusterTag = [u8; 5]; // {000} through {zzz}

/// Cluster tag map for vector ID routing
pub struct ClusterTagMap {
    /// Dense storage for low IDs (direct array access)
    dense: Option<Vec<Option<ClusterTag>>>,
    /// Sparse storage for high IDs (hash map)
    sparse: RwLock<HashMap<u64, ClusterTag>>,
    /// Threshold: IDs below this use dense, above use sparse
    dense_capacity: u64,
    /// Total entries stored
    entry_count: std::sync::atomic::AtomicU64,
}

impl ClusterTagMap {
    /// Create new cluster tag map
    /// 
    /// # Arguments
    /// * `dense_capacity` - Maximum ID for dense storage (set based on available RAM)
    ///   - 0 = use sparse only
    ///   - 1B = ~5GB RAM for dense storage
    pub fn new(dense_capacity: u64) -> Self {
        let dense = if dense_capacity > 0 {
            // Allocate dense storage
            // Each entry is Option<[u8; 5]> = 6 bytes
            Some(vec![None; dense_capacity as usize])
        } else {
            None
        };

        Self {
            dense,
            sparse: RwLock::new(HashMap::new()),
            dense_capacity,
            entry_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Insert mapping from vector ID to cluster tag
    #[inline]
    pub fn insert(&self, vector_id: u64, tag: ClusterTag) {
        if vector_id < self.dense_capacity {
            if let Some(ref dense) = self.dense {
                // Safety: dense is pre-allocated to dense_capacity
                // This requires interior mutability - use UnsafeCell in production
                unsafe {
                    let ptr = dense.as_ptr() as *mut Option<ClusterTag>;
                    *ptr.add(vector_id as usize) = Some(tag);
                }
            }
        } else {
            self.sparse.write().insert(vector_id, tag);
        }
        self.entry_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get cluster tag for vector ID
    #[inline]
    pub fn get(&self, vector_id: u64) -> Option<&ClusterTag> {
        if vector_id < self.dense_capacity {
            if let Some(ref dense) = self.dense {
                dense.get(vector_id as usize).and_then(|opt| opt.as_ref())
            } else {
                None
            }
        } else {
            // Sparse lookup requires holding read lock - return copy instead
            None // Caller should use get_copy for sparse
        }
    }

    /// Get cluster tag (with copy for sparse storage)
    #[inline]
    pub fn get_copy(&self, vector_id: u64) -> Option<ClusterTag> {
        if vector_id < self.dense_capacity {
            if let Some(ref dense) = self.dense {
                dense.get(vector_id as usize).and_then(|opt| *opt)
            } else {
                None
            }
        } else {
            self.sparse.read().get(&vector_id).copied()
        }
    }

    /// Check if vector exists in map
    #[inline]
    pub fn contains(&self, vector_id: u64) -> bool {
        self.get_copy(vector_id).is_some()
    }

    /// Get total entry count
    pub fn len(&self) -> u64 {
        self.entry_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        if let Some(ref mut dense) = self.dense {
            dense.fill(None);
        }
        self.sparse.write().clear();
        self.entry_count.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Generate cluster tag for node index
pub fn tag_for_node(node_idx: usize) -> ClusterTag {
    // Generate tag {000} through {zzz} (base 36)
    let mut tag = *b"{000}";
    let mut n = node_idx;
    
    for i in (1..4).rev() {
        let digit = n % 36;
        tag[i] = if digit < 10 {
            b'0' + digit as u8
        } else {
            b'a' + (digit - 10) as u8
        };
        n /= 36;
    }
    
    tag
}

/// Parse cluster tag from key bytes
pub fn parse_tag_from_key(key: &[u8]) -> Option<ClusterTag> {
    // Find {xxx} pattern
    if let Some(start) = key.iter().position(|&b| b == b'{') {
        if start + 4 < key.len() && key[start + 4] == b'}' {
            let mut tag = [0u8; 5];
            tag.copy_from_slice(&key[start..start + 5]);
            return Some(tag);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_generation() {
        assert_eq!(&tag_for_node(0), b"{000}");
        assert_eq!(&tag_for_node(1), b"{001}");
        assert_eq!(&tag_for_node(10), b"{00a}");
        assert_eq!(&tag_for_node(36), b"{010}");
    }

    #[test]
    fn test_cluster_tag_map() {
        let map = ClusterTagMap::new(1000);
        
        map.insert(0, *b"{000}");
        map.insert(500, *b"{001}");
        
        assert_eq!(map.get_copy(0), Some(*b"{000}"));
        assert_eq!(map.get_copy(500), Some(*b"{001}"));
        assert_eq!(map.get_copy(999), None);
    }
}
```

---

### Task 10.2: Implement SCAN-Based Key Discovery

**Description:** Discover existing keys via SCAN to build cluster tag map.

**Instructions:**
Create `src/dataset/scanner.rs`:

```rust
//! SCAN-based key discovery for cluster tag map population

use crate::client::{RawConnection, ConnectionFactory};
use crate::cluster::{ClusterTopology, ClusterNode};
use crate::utils::RespEncoder;
use super::cluster_tag_map::{ClusterTagMap, parse_tag_from_key};

/// Scan options
pub struct ScanOptions {
    /// Key pattern to match
    pub pattern: String,
    /// Batch size per SCAN call
    pub count: u32,
    /// Maximum keys to scan (0 = unlimited)
    pub max_keys: u64,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            pattern: "*".to_string(),
            count: 1000,
            max_keys: 0,
        }
    }
}

/// Scan result
pub struct ScanResult {
    pub keys_scanned: u64,
    pub tags_found: u64,
    pub errors: Vec<String>,
}

/// Scan all nodes in cluster to populate cluster tag map
pub fn scan_cluster_for_tags(
    factory: &ConnectionFactory,
    topology: &ClusterTopology,
    tag_map: &ClusterTagMap,
    key_prefix: &str,
    options: &ScanOptions,
) -> ScanResult {
    let mut result = ScanResult {
        keys_scanned: 0,
        tags_found: 0,
        errors: Vec::new(),
    };

    let pattern = format!("{}*", key_prefix);

    // Scan each primary node
    for node in topology.primaries() {
        match scan_node(factory, node, tag_map, &pattern, options) {
            Ok((scanned, found)) => {
                result.keys_scanned += scanned;
                result.tags_found += found;
            }
            Err(e) => {
                result.errors.push(format!("{}: {}", node.address(), e));
            }
        }

        // Check max keys limit
        if options.max_keys > 0 && result.keys_scanned >= options.max_keys {
            break;
        }
    }

    result
}

fn scan_node(
    factory: &ConnectionFactory,
    node: &ClusterNode,
    tag_map: &ClusterTagMap,
    pattern: &str,
    options: &ScanOptions,
) -> Result<(u64, u64), String> {
    let mut conn = factory.create(&node.host, node.port)
        .map_err(|e| e.to_string())?;

    let mut cursor = "0".to_string();
    let mut scanned = 0u64;
    let mut found = 0u64;
    let count_str = options.count.to_string();

    loop {
        let mut encoder = RespEncoder::with_capacity(128);
        encoder.encode_command_str(&[
            "SCAN", &cursor, "MATCH", pattern, "COUNT", &count_str
        ]);

        let response = conn.execute(&encoder)
            .map_err(|e| e.to_string())?;

        // Parse SCAN response: [cursor, [keys...]]
        if let crate::utils::RespValue::Array(arr) = response {
            if arr.len() != 2 {
                return Err("Invalid SCAN response".to_string());
            }

            // Get new cursor
            cursor = match &arr[0] {
                crate::utils::RespValue::BulkString(b) => {
                    String::from_utf8_lossy(b).to_string()
                }
                _ => return Err("Invalid cursor in SCAN response".to_string()),
            };

            // Process keys
            if let crate::utils::RespValue::Array(keys) = &arr[1] {
                for key in keys {
                    if let crate::utils::RespValue::BulkString(key_bytes) = key {
                        scanned += 1;
                        
                        // Extract tag and vector ID from key
                        if let Some(tag) = parse_tag_from_key(key_bytes) {
                            if let Some(id) = extract_vector_id(key_bytes) {
                                tag_map.insert(id, tag);
                                found += 1;
                            }
                        }
                    }
                }
            }
        }

        // Check if scan complete
        if cursor == "0" {
            break;
        }
    }

    Ok((scanned, found))
}

/// Extract vector ID from key bytes
/// Expects format: {xxx}prefix:12345
fn extract_vector_id(key: &[u8]) -> Option<u64> {
    // Find last colon and parse number after it
    if let Some(pos) = key.iter().rposition(|&b| b == b':') {
        let id_bytes = &key[pos + 1..];
        std::str::from_utf8(id_bytes).ok()?.parse().ok()
    } else {
        None
    }
}
```

---

### Task 10.3: Phase 10 Commit

```bash
git add .
git commit -m "Phase 10: Cluster tag map and key routing

- Implement ClusterTagMap with dense/sparse hybrid storage
- Support 10B+ vectors with configurable dense threshold
- Add SCAN-based key discovery across cluster nodes
- Create tag generation and parsing utilities
- Integrate with placeholder replacement system

Cluster-aware key routing for multi-node deployments."
```

---

## Phase 11: Rate Limiting and Node Balancing

**Objective:** Implement quota-based node balancing to ensure fair load distribution.

**Git Branch:** `phase-11-balancing`

### Task 11.1: Implement Node Balancer

**Description:** Create quota-based node balancing algorithm matching C implementation.

**Instructions:**
Create `src/benchmark/node_balancer.rs`:

```rust
//! Quota-based node balancing
//!
//! Problem: Nodes with different latencies process requests at different rates.
//!          Fast nodes complete their work while slow nodes are still processing.
//!          This leads to uneven load distribution and skewed benchmarks.
//!
//! Solution: Quota-based throttling
//!   1. Each node starts with quota = balance_quota_step
//!   2. Before sending, check if node has quota remaining
//!   3. If quota exhausted:
//!      a. Find min_completed = MIN(node_request_counters[i])
//!      b. If min_completed == 0: throttle (wait for slowest)
//!      c. Else: add quota proportional to min_completed
//!   4. Result: All nodes stay within tolerance_pct of each other

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

/// Per-node quota and counter state
pub struct NodeState {
    /// Remaining quota for this node
    pub quota_remaining: AtomicI64,
    /// Requests sent to this node in current cycle
    pub request_counter: AtomicU64,
}

/// Node balancer for fair distribution across cluster nodes
pub struct NodeBalancer {
    /// Per-node state
    nodes: Vec<NodeState>,
    /// Initial quota per cycle
    quota_step: i64,
    /// Allowed imbalance percentage (0-100)
    tolerance_pct: i64,
    /// Number of nodes being balanced
    num_nodes: usize,
}

impl NodeBalancer {
    /// Create new node balancer
    pub fn new(num_nodes: usize, quota_step: i64, tolerance_pct: i64) -> Self {
        let nodes = (0..num_nodes)
            .map(|_| NodeState {
                quota_remaining: AtomicI64::new(quota_step),
                request_counter: AtomicU64::new(0),
            })
            .collect();

        Self {
            nodes,
            quota_step,
            tolerance_pct,
            num_nodes,
        }
    }

    /// Check if node has quota, returning delay if throttling needed
    /// 
    /// Returns:
    /// - Ok(()) if request can proceed
    /// - Err(delay) if throttling needed (delay in microseconds)
    #[inline]
    pub fn check_quota(&self, node_idx: usize, tokens: i64) -> Result<(), u64> {
        let node = &self.nodes[node_idx];
        
        // Check current quota
        let remaining = node.quota_remaining.load(Ordering::Relaxed);
        if remaining >= tokens {
            return Ok(());
        }

        // Quota exhausted - check if we can start new cycle
        self.try_new_cycle(node_idx, tokens)
    }

    /// Consume quota for a request
    #[inline]
    pub fn consume(&self, node_idx: usize, tokens: i64) {
        let node = &self.nodes[node_idx];
        node.quota_remaining.fetch_sub(tokens, Ordering::Relaxed);
        node.request_counter.fetch_add(tokens as u64, Ordering::Relaxed);
    }

    /// Try to start a new balancing cycle
    fn try_new_cycle(&self, _requesting_node: usize, _tokens: i64) -> Result<(), u64> {
        // Find minimum completed across all nodes
        let min_completed = self.nodes.iter()
            .map(|n| n.request_counter.load(Ordering::Relaxed))
            .min()
            .unwrap_or(0);

        if min_completed == 0 {
            // Slowest node hasn't made any progress - must wait
            return Err(1000); // 1ms delay
        }

        // Calculate quota to add based on slowest node's progress
        let quota_to_add = (min_completed as i64 * (100 + self.tolerance_pct)) / 100;

        // Add quota to all nodes and reset counters
        for node in &self.nodes {
            node.quota_remaining.fetch_add(quota_to_add, Ordering::Relaxed);
            // Don't reset counter here - it's cumulative for imbalance detection
        }

        Ok(())
    }

    /// Record completed response (for tracking)
    #[inline]
    pub fn record_response(&self, node_idx: usize, _count: u64) {
        // Currently tracking is done via request_counter at send time
        // This hook is for future per-response tracking if needed
    }

    /// Get current imbalance statistics
    pub fn get_imbalance_stats(&self) -> ImbalanceStats {
        let counters: Vec<u64> = self.nodes.iter()
            .map(|n| n.request_counter.load(Ordering::Relaxed))
            .collect();

        let min = *counters.iter().min().unwrap_or(&0);
        let max = *counters.iter().max().unwrap_or(&0);
        let sum: u64 = counters.iter().sum();
        let mean = if self.num_nodes > 0 { sum / self.num_nodes as u64 } else { 0 };

        let ratio = if min > 0 { max as f64 / min as f64 } else { f64::INFINITY };
        let deviation_pct = if mean > 0 {
            ((max - mean) as f64 / mean as f64) * 100.0
        } else {
            0.0
        };

        ImbalanceStats {
            min_requests: min,
            max_requests: max,
            mean_requests: mean,
            ratio,
            deviation_pct,
            per_node: counters,
        }
    }

    /// Reset all counters (for new benchmark run)
    pub fn reset(&self) {
        for node in &self.nodes {
            node.quota_remaining.store(self.quota_step, Ordering::Relaxed);
            node.request_counter.store(0, Ordering::Relaxed);
        }
    }
}

/// Imbalance statistics
#[derive(Debug, Clone)]
pub struct ImbalanceStats {
    pub min_requests: u64,
    pub max_requests: u64,
    pub mean_requests: u64,
    pub ratio: f64,
    pub deviation_pct: f64,
    pub per_node: Vec<u64>,
}

impl ImbalanceStats {
    /// Check if imbalance exceeds threshold
    pub fn exceeds_threshold(&self, threshold_pct: f64) -> bool {
        self.deviation_pct > threshold_pct
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "min={}, max={}, ratio={:.2}x, deviation={:.1}%",
            self.min_requests, self.max_requests, self.ratio, self.deviation_pct
        )
    }
}
```

---

### Task 11.2: Phase 11 Commit

```bash
git add .
git commit -m "Phase 11: Rate limiting and node balancing

- Implement quota-based NodeBalancer
- Add imbalance detection and statistics
- Support configurable tolerance percentage
- Create throttling mechanism for fairness

Fair load distribution across heterogeneous nodes."
```

---

## Phase 12: VSS Platform-Specific Handling

**Objective:** Implement platform-specific handling for VSS differences between MemoryDB, ElastiCache, and open-source Valkey.

**Git Branch:** `phase-12-vss-platforms`

### Task 12.1: Define VSS Platform Differences

**Description:** Create abstraction for VSS command and metrics differences.

**Instructions:**
Create `src/workload/vss_platform.rs`:

```rust
//! VSS platform-specific handling
//!
//! Different platforms have variations in:
//! - FT.CREATE parameters
//! - FT.SEARCH query syntax
//! - INFO metrics for vector search
//! - Index management commands

use crate::cluster::VssVariant;
use crate::config::SearchConfig;

/// Platform-specific VSS configuration
#[derive(Debug, Clone)]
pub struct VssPlatformConfig {
    /// Platform variant
    pub variant: VssVariant,
    /// Supported FT.CREATE parameters
    pub create_params: VssCreateParams,
    /// Supported FT.SEARCH features
    pub search_features: VssSearchFeatures,
    /// Available INFO metrics
    pub info_metrics: Vec<String>,
}

/// FT.CREATE parameter support
#[derive(Debug, Clone)]
pub struct VssCreateParams {
    /// Supports EF_CONSTRUCTION parameter
    pub ef_construction: bool,
    /// Supports M parameter for HNSW
    pub hnsw_m: bool,
    /// Supports INITIAL_CAP
    pub initial_cap: bool,
    /// Maximum supported dimension
    pub max_dim: u32,
    /// Supported distance metrics
    pub distance_metrics: Vec<String>,
    /// Supports ON JSON (vs only ON HASH)
    pub json_support: bool,
}

/// FT.SEARCH feature support
#[derive(Debug, Clone)]
pub struct VssSearchFeatures {
    /// Supports EF_RUNTIME parameter
    pub ef_runtime: bool,
    /// Supports DIALECT 2+
    pub dialect_support: u8,
    /// Supports hybrid search (filters + vector)
    pub hybrid_search: bool,
    /// Supports NOCONTENT
    pub nocontent: bool,
    /// Maximum K value
    pub max_k: u32,
}

impl VssPlatformConfig {
    /// Get configuration for platform variant
    pub fn for_variant(variant: VssVariant) -> Self {
        match variant {
            VssVariant::OpenSource => Self::open_source(),
            VssVariant::ElastiCache => Self::elasticache(),
            VssVariant::ElastiCacheServerless => Self::elasticache_serverless(),
            VssVariant::MemoryDB => Self::memorydb(),
        }
    }

    fn open_source() -> Self {
        Self {
            variant: VssVariant::OpenSource,
            create_params: VssCreateParams {
                ef_construction: true,
                hnsw_m: true,
                initial_cap: true,
                max_dim: 32768,
                distance_metrics: vec![
                    "L2".to_string(),
                    "IP".to_string(),
                    "COSINE".to_string(),
                ],
                json_support: true,
            },
            search_features: VssSearchFeatures {
                ef_runtime: true,
                dialect_support: 4,
                hybrid_search: true,
                nocontent: true,
                max_k: 10000,
            },
            info_metrics: vec![
                "search_number_of_indexes".to_string(),
                "search_total_indexing_time".to_string(),
                "search_total_query_time".to_string(),
                "search_used_memory_bytes".to_string(),
                "search_total_queries".to_string(),
                "search_total_index_writes".to_string(),
            ],
        }
    }

    fn elasticache() -> Self {
        Self {
            variant: VssVariant::ElastiCache,
            create_params: VssCreateParams {
                ef_construction: true,
                hnsw_m: true,
                initial_cap: true,
                max_dim: 16384, // May differ from open source
                distance_metrics: vec![
                    "L2".to_string(),
                    "IP".to_string(),
                    "COSINE".to_string(),
                ],
                json_support: true,
            },
            search_features: VssSearchFeatures {
                ef_runtime: true,
                dialect_support: 2,
                hybrid_search: true,
                nocontent: true,
                max_k: 10000,
            },
            info_metrics: vec![
                // ElastiCache may have additional/different metrics
                "search_number_of_indexes".to_string(),
                "search_total_indexing_time".to_string(),
                "search_used_memory_bytes".to_string(),
            ],
        }
    }

    fn elasticache_serverless() -> Self {
        Self {
            variant: VssVariant::ElastiCacheServerless,
            create_params: VssCreateParams {
                ef_construction: true,
                hnsw_m: true,
                initial_cap: false, // May not be supported
                max_dim: 8192, // More limited
                distance_metrics: vec![
                    "L2".to_string(),
                    "COSINE".to_string(),
                ],
                json_support: false, // Limited support
            },
            search_features: VssSearchFeatures {
                ef_runtime: true,
                dialect_support: 2,
                hybrid_search: false, // May be limited
                nocontent: true,
                max_k: 1000, // More limited
            },
            info_metrics: vec![
                // Serverless has limited metrics visibility
                "search_number_of_indexes".to_string(),
            ],
        }
    }

    fn memorydb() -> Self {
        Self {
            variant: VssVariant::MemoryDB,
            create_params: VssCreateParams {
                ef_construction: true,
                hnsw_m: true,
                initial_cap: true,
                max_dim: 16384,
                distance_metrics: vec![
                    "L2".to_string(),
                    "IP".to_string(),
                    "COSINE".to_string(),
                ],
                json_support: true,
            },
            search_features: VssSearchFeatures {
                ef_runtime: true,
                dialect_support: 2,
                hybrid_search: true,
                nocontent: true,
                max_k: 10000,
            },
            info_metrics: vec![
                // MemoryDB may have different metric names
                "search_number_of_indexes".to_string(),
                "search_total_indexing_time_ms".to_string(), // Note: _ms suffix
                "search_memory_bytes".to_string(), // Different name
            ],
        }
    }

    /// Validate search config against platform capabilities
    pub fn validate_config(&self, config: &SearchConfig) -> Result<(), String> {
        // Check dimension limit
        if config.dim > self.create_params.max_dim {
            return Err(format!(
                "Dimension {} exceeds platform maximum {}",
                config.dim, self.create_params.max_dim
            ));
        }

        // Check distance metric support
        let metric = config.distance_metric.as_str();
        if !self.create_params.distance_metrics.contains(&metric.to_string()) {
            return Err(format!(
                "Distance metric {} not supported on {:?}",
                metric, self.variant
            ));
        }

        // Check K limit
        if config.k > self.search_features.max_k {
            return Err(format!(
                "K={} exceeds platform maximum {}",
                config.k, self.search_features.max_k
            ));
        }

        // Check EF_RUNTIME support
        if config.ef_search.is_some() && !self.search_features.ef_runtime {
            return Err(format!(
                "EF_RUNTIME not supported on {:?}",
                self.variant
            ));
        }

        Ok(())
    }

    /// Build FT.CREATE command with platform-appropriate parameters
    pub fn build_ft_create(&self, config: &SearchConfig) -> Vec<String> {
        let mut args = vec![
            "FT.CREATE".to_string(),
            config.index_name.clone(),
            "ON".to_string(),
            "HASH".to_string(),
            "PREFIX".to_string(),
            "1".to_string(),
            config.prefix.clone(),
            "SCHEMA".to_string(),
            config.vector_field.clone(),
            "VECTOR".to_string(),
            config.algorithm.as_str().to_string(),
        ];

        // HNSW parameters
        let mut num_params = 6; // TYPE, DIM, DISTANCE_METRIC
        
        if config.hnsw_m.is_some() && self.create_params.hnsw_m {
            num_params += 2;
        }
        if config.ef_construction.is_some() && self.create_params.ef_construction {
            num_params += 2;
        }

        args.push(num_params.to_string());
        args.push("TYPE".to_string());
        args.push("FLOAT32".to_string());
        args.push("DIM".to_string());
        args.push(config.dim.to_string());
        args.push("DISTANCE_METRIC".to_string());
        args.push(config.distance_metric.as_str().to_string());

        if let Some(m) = config.hnsw_m {
            if self.create_params.hnsw_m {
                args.push("M".to_string());
                args.push(m.to_string());
            }
        }

        if let Some(ef) = config.ef_construction {
            if self.create_params.ef_construction {
                args.push("EF_CONSTRUCTION".to_string());
                args.push(ef.to_string());
            }
        }

        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{VectorAlgorithm, DistanceMetric};

    #[test]
    fn test_platform_configs() {
        let os = VssPlatformConfig::for_variant(VssVariant::OpenSource);
        assert!(os.search_features.ef_runtime);
        assert_eq!(os.search_features.dialect_support, 4);

        let ec = VssPlatformConfig::for_variant(VssVariant::ElastiCache);
        assert_eq!(ec.search_features.dialect_support, 2);

        let serverless = VssPlatformConfig::for_variant(VssVariant::ElastiCacheServerless);
        assert!(!serverless.create_params.initial_cap);
    }
}
```

---

### Task 12.2: Phase 12 Commit

```bash
git add .
git commit -m "Phase 12: VSS platform-specific handling

- Define VssPlatformConfig for each deployment target
- Implement platform capability validation
- Add platform-aware FT.CREATE builder
- Document metric differences across platforms
- Support MemoryDB, ElastiCache, and open-source variations

Platform-appropriate VSS command generation."
```

---

## Phase 13: Optimizer Integration

**Objective:** Implement parameter optimization with phase transitions.

**Git Branch:** `phase-13-optimizer`

### Task 13.1: Implement Optimizer State Machine

**Description:** Create optimizer with feasibility, recall, and throughput phases.

**Instructions:**
Create `src/optimizer/optimizer.rs`:

```rust
//! Adaptive parameter optimization
//!
//! Phase 1: Feasibility - verify basic functionality
//! Phase 2: Recall - find ef_search that meets recall target
//! Phase 3: Throughput - maximize QPS while maintaining constraints

use crate::benchmark::BenchmarkResult;
use crate::config::SearchConfig;

/// Optimizer phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerPhase {
    /// Initial phase - verify system works
    Feasibility,
    /// Find ef_search that meets recall target
    RecallSearch,
    /// Maximize throughput while maintaining constraints
    ThroughputOptimization,
    /// Optimization complete
    Complete,
}

/// Optimizer constraints
#[derive(Debug, Clone)]
pub struct Constraints {
    /// Minimum recall target (0.0 - 1.0)
    pub min_recall: f64,
    /// Maximum p99 latency in milliseconds
    pub max_p99_ms: Option<f64>,
    /// Target QPS (optional)
    pub target_qps: Option<u64>,
}

/// Optimizer state
pub struct Optimizer {
    phase: OptimizerPhase,
    constraints: Constraints,
    /// Binary search state for ef_search
    ef_search_low: u32,
    ef_search_high: u32,
    ef_search_current: u32,
    /// Best result that meets constraints
    best_result: Option<OptimizationResult>,
    /// History of all measurements
    history: Vec<Measurement>,
    /// Maximum iterations
    max_iterations: u32,
    current_iteration: u32,
}

#[derive(Debug, Clone)]
pub struct Measurement {
    pub ef_search: u32,
    pub recall: f64,
    pub qps: f64,
    pub p99_ms: f64,
    pub constraints_met: bool,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub ef_search: u32,
    pub recall: f64,
    pub qps: f64,
    pub p99_ms: f64,
}

impl Optimizer {
    /// Create new optimizer
    pub fn new(constraints: Constraints, max_ef_search: u32) -> Self {
        Self {
            phase: OptimizerPhase::Feasibility,
            constraints,
            ef_search_low: 10,
            ef_search_high: max_ef_search,
            ef_search_current: 100, // Start with reasonable default
            best_result: None,
            history: Vec::new(),
            max_iterations: 20,
            current_iteration: 0,
        }
    }

    /// Get current phase
    pub fn phase(&self) -> OptimizerPhase {
        self.phase
    }

    /// Get current ef_search value to test
    pub fn current_ef_search(&self) -> u32 {
        self.ef_search_current
    }

    /// Process benchmark result and determine next step
    pub fn step(&mut self, result: &BenchmarkResult) -> OptimizerAction {
        self.current_iteration += 1;

        let measurement = Measurement {
            ef_search: self.ef_search_current,
            recall: result.recall_stats.average(),
            qps: result.throughput,
            p99_ms: result.percentile_ms(99.0),
            constraints_met: self.check_constraints(result),
        };

        self.history.push(measurement.clone());

        match self.phase {
            OptimizerPhase::Feasibility => {
                self.handle_feasibility(measurement)
            }
            OptimizerPhase::RecallSearch => {
                self.handle_recall_search(measurement)
            }
            OptimizerPhase::ThroughputOptimization => {
                self.handle_throughput(measurement)
            }
            OptimizerPhase::Complete => {
                OptimizerAction::Done(self.best_result.clone())
            }
        }
    }

    fn check_constraints(&self, result: &BenchmarkResult) -> bool {
        // Check recall
        if result.recall_stats.average() < self.constraints.min_recall {
            return false;
        }

        // Check p99 latency
        if let Some(max_p99) = self.constraints.max_p99_ms {
            if result.percentile_ms(99.0) > max_p99 {
                return false;
            }
        }

        // Check target QPS
        if let Some(target) = self.constraints.target_qps {
            if result.throughput < target as f64 {
                return false;
            }
        }

        true
    }

    fn handle_feasibility(&mut self, m: Measurement) -> OptimizerAction {
        if m.recall > 0.0 && m.qps > 0.0 {
            // System works, move to recall search
            self.phase = OptimizerPhase::RecallSearch;
            self.ef_search_current = (self.ef_search_low + self.ef_search_high) / 2;
            OptimizerAction::Continue(self.ef_search_current)
        } else {
            OptimizerAction::Error("Feasibility check failed".to_string())
        }
    }

    fn handle_recall_search(&mut self, m: Measurement) -> OptimizerAction {
        if m.recall >= self.constraints.min_recall {
            // Recall achieved, try lower ef_search
            self.ef_search_high = self.ef_search_current;
            
            if m.constraints_met {
                self.best_result = Some(OptimizationResult {
                    ef_search: m.ef_search,
                    recall: m.recall,
                    qps: m.qps,
                    p99_ms: m.p99_ms,
                });
            }
        } else {
            // Need higher ef_search
            self.ef_search_low = self.ef_search_current;
        }

        // Binary search convergence check
        if self.ef_search_high - self.ef_search_low <= 10 
            || self.current_iteration >= self.max_iterations 
        {
            if self.best_result.is_some() {
                self.phase = OptimizerPhase::ThroughputOptimization;
            } else {
                return OptimizerAction::Error("Could not achieve target recall".to_string());
            }
        }

        self.ef_search_current = (self.ef_search_low + self.ef_search_high) / 2;
        OptimizerAction::Continue(self.ef_search_current)
    }

    fn handle_throughput(&mut self, m: Measurement) -> OptimizerAction {
        // Simple throughput optimization: use lowest ef_search that meets constraints
        if m.constraints_met {
            if let Some(ref mut best) = self.best_result {
                if m.qps > best.qps {
                    *best = OptimizationResult {
                        ef_search: m.ef_search,
                        recall: m.recall,
                        qps: m.qps,
                        p99_ms: m.p99_ms,
                    };
                }
            }
        }

        self.phase = OptimizerPhase::Complete;
        OptimizerAction::Done(self.best_result.clone())
    }

    /// Get best result found
    pub fn best_result(&self) -> Option<&OptimizationResult> {
        self.best_result.as_ref()
    }
}

/// Action to take after optimization step
pub enum OptimizerAction {
    /// Continue with new ef_search value
    Continue(u32),
    /// Optimization complete
    Done(Option<OptimizationResult>),
    /// Error occurred
    Error(String),
}
```

---

### Task 13.2: Phase 13 Commit

```bash
git add .
git commit -m "Phase 13: Optimizer integration

- Implement Optimizer state machine
- Add feasibility, recall search, and throughput phases
- Create binary search on ef_search parameter
- Define constraint evaluation
- Support optimization history tracking

Adaptive parameter optimization for vector search."
```

---

## Phase 14: Polish, Testing, and Documentation

**Objective:** Complete implementation with comprehensive testing and documentation.

**Git Branch:** `phase-14-polish`

### Task 14.1: Create Integration Test Suite

**Description:** Comprehensive tests for all deployment scenarios.

**Instructions:**
Create `tests/integration/` with tests for:
- Standalone mode discovery
- Cluster mode discovery
- AWS ElastiCache endpoint handling
- MemoryDB endpoint handling
- Per-node metrics collection
- Temporal diff calculation
- Node balancing

### Task 14.2: Performance Benchmarks

**Description:** Compare Rust implementation against C version.

**Instructions:**
- Create benchmark suite using criterion
- Measure: connection setup, command encoding, response parsing
- Compare throughput with C version
- Document any performance differences

### Task 14.3: Documentation

**Description:** Complete documentation for all features.

**Instructions:**
- README.md with quick start and examples
- Deployment-specific guides (AWS, self-managed)
- API documentation (cargo doc)
- Metrics reference guide
- Troubleshooting guide

### Task 14.4: Final Commit

```bash
git add .
git commit -m "Phase 14: Polish, testing, and documentation

- Add comprehensive integration test suite
- Create performance benchmarks vs C version
- Complete documentation and guides
- Add CI/CD pipeline configuration

Production-ready release."
```

---

## Updated Phase Summary and Dependencies

| Phase | Description | Dependencies | Key Deliverables |
|-------|-------------|--------------|------------------|
| 1 | Project skeleton, CLI | None | Buildable project, all CLI args |
| 2 | RESP protocol, connections | Phase 1 | TCP/TLS connections, RESP encode/decode |
| 3 | Command templates | Phase 2 | Template system, all workload types |
| 4 | Threading, workers | Phase 3 | Multi-threaded execution |
| 5 | Deployment targets, discovery | Phase 4 | AWS/self-managed, cluster/standalone |
| 6 | Cluster topology | Phase 5 | Slot mapping, read-from-replica |
| 7 | Dataset integration | Phase 4 | mmap access, zero-copy vectors |
| 8 | Vector search | Phase 6, 7 | FT.CREATE, FT.SEARCH, recall |
| 9 | Per-node metrics | Phase 6 | INFO collection, temporal diff, imbalance |
| 10 | Cluster tag map | Phase 6, 7 | SCAN, key routing |
| 11 | Node balancing | Phase 6 | Quota-based fair distribution |
| 12 | VSS platforms | Phase 8 | MemoryDB, ElastiCache differences |
| 13 | Optimizer | Phase 8, 9 | Phase transitions, binary search |
| 14 | Polish & testing | All | Tests, docs, CI/CD |

---

## Critical Capabilities Checklist

### Deployment Targets
- [ ] Self-managed Valkey/Redis
- [ ] AWS ElastiCache (cluster mode)
- [ ] AWS ElastiCache (non-cluster mode)
- [ ] AWS ElastiCache Serverless
- [ ] AWS MemoryDB
- [ ] Standalone with replicas (non-cluster)

### Discovery Mechanisms
- [ ] CLUSTER NODES for cluster mode
- [ ] INFO REPLICATION for standalone replicas
- [ ] AWS configuration endpoint parsing
- [ ] DNS resolution for multi-node discovery

### Per-Node Metrics
- [ ] INFO field collection across all nodes
- [ ] Configurable field definitions
- [ ] Temporal diff (before/after snapshots)
- [ ] Rate calculation (delta/elapsed)
- [ ] Memory growth tracking
- [ ] Imbalance detection and reporting
- [ ] User-defined custom metrics
- [ ] Snapshot save/load for cross-run comparison

### VSS Platform Differences
- [ ] MemoryDB VSS metrics
- [ ] ElastiCache VSS metrics
- [ ] Open-source Valkey VSS metrics
- [ ] Platform capability validation
- [ ] Appropriate FT.CREATE parameters per platform

| Phase | Description | Dependencies | Key Deliverables |
|-------|-------------|--------------|------------------|
| 1 | Project skeleton, CLI | None | Buildable project, all CLI args |
| 2 | RESP protocol, connections | Phase 1 | TCP/TLS connections, RESP encode/decode |
| 3 | Command templates | Phase 2 | Template system, all workload types |
| 4 | Threading, workers | Phase 3 | Multi-threaded execution |
| 5 | Cluster discovery | Phase 4 | CLUSTER NODES parsing, node selection |
| 6 | Dataset integration | Phase 4 | mmap access, zero-copy vectors |
| 7 | Vector search | Phase 5, 6 | FT.CREATE, FT.SEARCH, binary blobs |
| 8 | Recall verification | Phase 7 | Ground truth comparison |
| 9 | Metrics & reporting | Phase 4 | HDR histogram, CSV/JSON output |
| 10 | Cluster tag map | Phase 5, 6 | SCAN, key discovery |
| 11 | Rate limiting, node balancing | Phase 5 | Token bucket, quota-based balancing |
| 12 | Optimizer | Phase 9 | Parameter tuning, phase transitions |
| 13 | Polish & testing | All | Tests, CI/CD, docs |
