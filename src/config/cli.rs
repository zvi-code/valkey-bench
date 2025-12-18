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
#[command(disable_help_flag = true)]
#[command(trailing_var_arg = true)]
#[allow(clippy::manual_non_exhaustive)]
pub struct CliArgs {
    /// Print help information
    #[arg(long = "help", action = clap::ArgAction::Help)]
    help: (),

    // ===== CLI Mode =====
    /// Run in interactive CLI mode (like valkey-cli)
    #[arg(long = "cli")]
    pub cli_mode: bool,

    /// Command arguments when using --cli (non-interactive mode)
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
    pub command_args: Vec<String>,

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
    #[arg(long = "dbnum")]
    pub dbnum: Option<u32>,

    /// Pipeline depth (commands per batch)
    #[arg(short = 'P', long = "pipeline", default_value_t = 1)]
    pub pipeline: u32,

    // ===== Key Generation =====
    /// Size of key/value data in bytes
    #[arg(short = 'd', long = "data-size", default_value_t = 3)]
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
            return Err(
                "--tls-cert typically requires --tls-ca-cert (or use --tls-skip-verify)"
                    .to_string(),
            );
        }

        // Pipeline must be positive
        if self.pipeline == 0 {
            return Err("--pipeline must be at least 1".to_string());
        }

        // Dataset required for vector search tests
        if let Some(ref tests) = self.tests {
            let needs_dataset = tests.iter().any(|t| {
                matches!(
                    t.to_lowercase().as_str(),
                    "vecload" | "vecquery" | "vec-load" | "vec-query"
                )
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
        let args = CliArgs::parse_from(["test", "-h", "host1", "-h", "host2", "-h", "host3"]);
        assert_eq!(args.hosts, vec!["host1", "host2", "host3"]);
    }

    #[test]
    fn test_vector_search_args() {
        let args = CliArgs::parse_from([
            "test",
            "--search-index",
            "myidx",
            "--vector-dim",
            "256",
            "-k",
            "20",
            "--ef-search",
            "100",
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
