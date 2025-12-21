//! Configurable INFO field definitions for temporal tracking
//!
//! This module provides a flexible system for defining which fields to track
//! from INFO SEARCH, FT.INFO, and other server responses. Each field can have:
//! - Custom parsing strategy (integer, memory, percentile, cmdstat)
//! - Aggregation type (sum, average, max, min/max)
//! - Display format (integer, memory MB, percentage, latency)
//! - Diff type for temporal comparison (rate, memory growth, percentage change)
//! - Node filtering (primary only, replica only, all)

use std::collections::HashMap;

/// Parse strategy for extracting values from INFO responses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseStrategy {
    /// Parse as integer
    Integer,
    /// Parse memory with K/M/G suffixes
    Memory,
    /// Parse float, store as fixed-point (value * 1000)
    FloatFixed,
    /// Extract percentile from histogram string (p50, p99, p99.9)
    Percentile,
    /// Extract from cmdstat format: calls=...,usec=...,usec_per_call=...,rejected=...,failed=...
    CmdStats,
}

/// Configuration for parsing
#[derive(Debug, Clone)]
pub struct ParseConfig {
    pub strategy: ParseStrategy,
    /// For percentile: "p50", "p99", "p99.9"
    /// For cmdstats: "calls", "usec", "usec_per_call", "rejected", "failed"
    pub key: Option<String>,
}

impl ParseConfig {
    pub fn integer() -> Self {
        Self {
            strategy: ParseStrategy::Integer,
            key: None,
        }
    }

    pub fn memory() -> Self {
        Self {
            strategy: ParseStrategy::Memory,
            key: None,
        }
    }

    pub fn float_fixed() -> Self {
        Self {
            strategy: ParseStrategy::FloatFixed,
            key: None,
        }
    }

    pub fn percentile(key: &str) -> Self {
        Self {
            strategy: ParseStrategy::Percentile,
            key: Some(key.to_string()),
        }
    }

    pub fn cmdstats(key: &str) -> Self {
        Self {
            strategy: ParseStrategy::CmdStats,
            key: Some(key.to_string()),
        }
    }
}

/// How to aggregate values across nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationType {
    /// Sum all node values
    Sum,
    /// Average across nodes
    Average,
    /// Take maximum value
    Max,
    /// Track min and max separately
    MinMax,
}

/// How to display the value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayFormat {
    /// Plain integer
    Integer,
    /// Memory in MB
    MemoryMb,
    /// Human-readable memory (auto-scale K/M/G)
    MemoryHuman,
    /// Percentage (from fixed-point scaled by 1000)
    Percentage,
    /// Float (from fixed-point scaled by 1000)
    Float,
    /// Latency in microseconds
    LatencyUsec,
    /// Min/max range display
    MinMax,
}

/// How to calculate diff between snapshots
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffType {
    /// No diff calculation
    None,
    /// Rate: delta / time_sec
    RateCount,
    /// Rate for microsecond counters: (delta / 1M) / time_sec
    RateMicrosec,
    /// Memory growth: (delta_bytes / MB) / time_sec
    MemoryGrowth,
    /// Percentage change: (delta / old_value) * 100
    PercentageChange,
}

/// Which nodes to aggregate from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeFilter {
    /// Only primary nodes
    PrimaryOnly,
    /// Only replica nodes
    ReplicaOnly,
    /// All nodes
    All,
}

/// Field matching strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchStrategy {
    /// Exact field name match
    Exact,
    /// Prefix match
    Prefix,
}

/// Definition of a field to track
#[derive(Debug, Clone)]
pub struct InfoFieldType {
    /// Field name or prefix to match
    pub name: String,
    /// How to match the field name
    pub match_strategy: MatchStrategy,
    /// How to parse the value
    pub parse_config: ParseConfig,
    /// How to aggregate across nodes
    pub aggregation_type: AggregationType,
    /// How to display the value
    pub display_format: DisplayFormat,
    /// How to calculate diff for temporal comparison
    pub diff_type: DiffType,
    /// Whether to track per-node values
    pub track_per_node: bool,
    /// Which nodes to include
    pub node_filter: NodeFilter,
}

impl InfoFieldType {
    /// Create a new field definition with common defaults
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            match_strategy: MatchStrategy::Exact,
            parse_config: ParseConfig::integer(),
            aggregation_type: AggregationType::Sum,
            display_format: DisplayFormat::Integer,
            diff_type: DiffType::None,
            track_per_node: true,
            node_filter: NodeFilter::All,
        }
    }

    // Builder methods
    pub fn prefix_match(mut self) -> Self {
        self.match_strategy = MatchStrategy::Prefix;
        self
    }

    pub fn parse(mut self, config: ParseConfig) -> Self {
        self.parse_config = config;
        self
    }

    pub fn aggregate(mut self, agg: AggregationType) -> Self {
        self.aggregation_type = agg;
        self
    }

    pub fn display(mut self, fmt: DisplayFormat) -> Self {
        self.display_format = fmt;
        self
    }

    pub fn diff(mut self, diff: DiffType) -> Self {
        self.diff_type = diff;
        self
    }

    pub fn from_primary_only(mut self) -> Self {
        self.node_filter = NodeFilter::PrimaryOnly;
        self
    }

    pub fn from_replica_only(mut self) -> Self {
        self.node_filter = NodeFilter::ReplicaOnly;
        self
    }

    pub fn no_per_node(mut self) -> Self {
        self.track_per_node = false;
        self
    }

    /// Check if this field matches a given field name
    pub fn matches(&self, field_name: &str) -> bool {
        match self.match_strategy {
            MatchStrategy::Exact => field_name == self.name,
            MatchStrategy::Prefix => field_name.starts_with(&self.name),
        }
    }
}

/// Parsed field value
#[derive(Debug, Clone)]
pub struct FieldValue {
    /// The raw value (for integers and fixed-point)
    pub value: i64,
    /// Min value (for MinMax aggregation)
    pub min_value: Option<i64>,
    /// Max value (for MinMax aggregation)
    pub max_value: Option<i64>,
    /// String representation (for display)
    pub value_str: Option<String>,
}

impl FieldValue {
    pub fn new(value: i64) -> Self {
        Self {
            value,
            min_value: None,
            max_value: None,
            value_str: None,
        }
    }

    pub fn with_minmax(min: i64, max: i64) -> Self {
        Self {
            value: max,
            min_value: Some(min),
            max_value: Some(max),
            value_str: None,
        }
    }
}

/// Snapshot of a single field across the cluster
#[derive(Debug, Clone)]
pub struct FieldSnapshot {
    /// Field name
    pub field_name: String,
    /// Aggregated value
    pub value: FieldValue,
    /// Per-node values (if tracked)
    pub per_node_values: Option<Vec<(String, i64)>>,
    /// Number of nodes contributing
    pub node_count: usize,
    /// Whether value is valid
    pub valid: bool,
}

/// Parse a value according to the parse config
pub fn parse_value(line: &str, config: &ParseConfig) -> Option<i64> {
    // Extract the value part after ':'
    let value_str = line.split(':').nth(1)?.trim();

    match config.strategy {
        ParseStrategy::Integer => value_str.parse::<i64>().ok(),

        ParseStrategy::Memory => parse_memory_value(value_str),

        ParseStrategy::FloatFixed => {
            let f: f64 = value_str.parse().ok()?;
            Some((f * 1000.0) as i64)
        }

        ParseStrategy::Percentile => {
            // Format: "p50=100,p99=200,p99.9=500"
            let key = config.key.as_ref()?;
            for part in value_str.split(',') {
                let mut kv = part.split('=');
                if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                    if k.trim() == key.as_str() {
                        return v.trim().parse().ok();
                    }
                }
            }
            None
        }

        ParseStrategy::CmdStats => {
            // Format: "calls=100,usec=5000,usec_per_call=50.00,rejected=0,failed=0"
            let key = config.key.as_ref()?;
            for part in value_str.split(',') {
                let mut kv = part.split('=');
                if let (Some(k), Some(v)) = (kv.next(), kv.next()) {
                    if k.trim() == key.as_str() {
                        // Handle both integer and float
                        if let Ok(i) = v.trim().parse::<i64>() {
                            return Some(i);
                        }
                        if let Ok(f) = v.trim().parse::<f64>() {
                            return Some(f as i64);
                        }
                    }
                }
            }
            None
        }
    }
}

/// Parse memory value with K/M/G/T suffixes
fn parse_memory_value(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let (num_str, multiplier) = if s.ends_with('K') || s.ends_with('k') {
        (&s[..s.len() - 1], 1024i64)
    } else if s.ends_with('M') || s.ends_with('m') {
        (&s[..s.len() - 1], 1024 * 1024)
    } else if s.ends_with('G') || s.ends_with('g') {
        (&s[..s.len() - 1], 1024 * 1024 * 1024)
    } else if s.ends_with('T') || s.ends_with('t') {
        (&s[..s.len() - 1], 1024 * 1024 * 1024 * 1024)
    } else {
        (s, 1)
    };

    // Handle decimal values
    if let Ok(f) = num_str.parse::<f64>() {
        Some((f * multiplier as f64) as i64)
    } else {
        num_str.parse::<i64>().ok().map(|v| v * multiplier)
    }
}

/// Format a value for display
pub fn format_value(value: &FieldValue, format: DisplayFormat) -> String {
    match format {
        DisplayFormat::Integer => format!("{}", value.value),

        DisplayFormat::MemoryMb => {
            let mb = value.value as f64 / (1024.0 * 1024.0);
            format!("{:.2} MB", mb)
        }

        DisplayFormat::MemoryHuman => format_memory_human(value.value),

        DisplayFormat::Percentage => {
            let pct = value.value as f64 / 1000.0;
            format!("{:.2}%", pct)
        }

        DisplayFormat::Float => {
            let f = value.value as f64 / 1000.0;
            format!("{:.3}", f)
        }

        DisplayFormat::LatencyUsec => format!("{} µs", value.value),

        DisplayFormat::MinMax => {
            if let (Some(min), Some(max)) = (value.min_value, value.max_value) {
                format!("{}/{}", min, max)
            } else {
                format!("{}", value.value)
            }
        }
    }
}

/// Format bytes as human-readable string
fn format_memory_human(bytes: i64) -> String {
    let abs = bytes.unsigned_abs();
    let sign = if bytes < 0 { "-" } else { "" };

    if abs >= 1024 * 1024 * 1024 * 1024 {
        format!("{}{:.2}T", sign, abs as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0))
    } else if abs >= 1024 * 1024 * 1024 {
        format!("{}{:.2}G", sign, abs as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if abs >= 1024 * 1024 {
        format!("{}{:.2}M", sign, abs as f64 / (1024.0 * 1024.0))
    } else if abs >= 1024 {
        format!("{}{:.2}K", sign, abs as f64 / 1024.0)
    } else {
        format!("{}{}B", sign, abs)
    }
}

/// Calculate diff between two values
pub fn calculate_diff(
    old_value: i64,
    new_value: i64,
    elapsed_secs: f64,
    diff_type: DiffType,
) -> Option<f64> {
    if elapsed_secs <= 0.0 {
        return None;
    }

    let delta = new_value - old_value;

    match diff_type {
        DiffType::None => None,

        DiffType::RateCount => Some(delta as f64 / elapsed_secs),

        DiffType::RateMicrosec => Some((delta as f64 / 1_000_000.0) / elapsed_secs),

        DiffType::MemoryGrowth => {
            let delta_mb = delta as f64 / (1024.0 * 1024.0);
            Some(delta_mb / elapsed_secs)
        }

        DiffType::PercentageChange => {
            if old_value == 0 {
                None
            } else {
                Some((delta as f64 / old_value as f64) * 100.0)
            }
        }
    }
}

/// Default INFO SEARCH fields (supports EC CME, EC CMD, and MemoryDB)
pub fn default_search_info_fields() -> Vec<InfoFieldType> {
    vec![
        // Request rates - Common to all systems
        InfoFieldType::new("search_successful_requests_count")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        InfoFieldType::new("search_failure_requests_count")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        InfoFieldType::new("search_hybrid_requests_count")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        // Memory growth - Common to all systems
        InfoFieldType::new("search_used_memory_bytes")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryMb)
            .diff(DiffType::MemoryGrowth)
            .from_primary_only(),
        InfoFieldType::new("search_index_reclaimable_memory")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryMb)
            .diff(DiffType::MemoryGrowth)
            .from_primary_only(),
        // EC CMD/CME specific - ingestion
        InfoFieldType::new("search_ingest_field_vector")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount)
            .from_primary_only(),
        // EC CMD/CME specific - indexing rates
        InfoFieldType::new("search_total_indexed_documents")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        InfoFieldType::new("search_total_active_write_threads")
            .aggregate(AggregationType::MinMax)
            .display(DisplayFormat::MinMax)
            .diff(DiffType::None),
        // MemoryDB specific - indexing stats
        InfoFieldType::new("search_total_indexed_keys")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount)
            .from_primary_only(),
        InfoFieldType::new("search_total_indexed_vectors")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount)
            .from_primary_only(),
        InfoFieldType::new("search_total_indexed_hash_keys")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount)
            .from_primary_only(),
        InfoFieldType::new("search_total_index_size")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::MemoryGrowth)
            .from_primary_only(),
        InfoFieldType::new("search_total_vector_index_size")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::MemoryGrowth)
            .from_primary_only(),
        InfoFieldType::new("search_max_index_degradation_percentage")
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_max_index_lag_ms")
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        // EC CMD/CME specific - CPU usage
        InfoFieldType::new("search_read_cpu_time_sec")
            .parse(ParseConfig::float_fixed())
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Float)
            .diff(DiffType::RateCount),
        InfoFieldType::new("search_write_cpu_time_sec")
            .parse(ParseConfig::float_fixed())
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Float)
            .diff(DiffType::RateCount)
            .from_primary_only(),
        InfoFieldType::new("search_used_read_cpu")
            .parse(ParseConfig::float_fixed())
            .aggregate(AggregationType::Average)
            .display(DisplayFormat::Percentage)
            .diff(DiffType::None),
        InfoFieldType::new("search_used_write_cpu")
            .parse(ParseConfig::float_fixed())
            .aggregate(AggregationType::Average)
            .display(DisplayFormat::Percentage)
            .diff(DiffType::None)
            .from_primary_only(),
        // EC CMD/CME specific - Queue sizes
        InfoFieldType::new("search_query_queue_size")
            .aggregate(AggregationType::Average)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None),
        InfoFieldType::new("search_writer_queue_size")
            .aggregate(AggregationType::Average)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        // Latencies - EC CMD/CME specific
        InfoFieldType::new("search_hnsw_vector_index_search_latency_usec")
            .parse(ParseConfig::percentile("p50"))
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::LatencyUsec)
            .diff(DiffType::None),
        InfoFieldType::new("search_hnsw_vector_index_search_latency_usec")
            .parse(ParseConfig::percentile("p99"))
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::LatencyUsec)
            .diff(DiffType::None),
        InfoFieldType::new("search_hnsw_vector_index_search_latency_usec")
            .parse(ParseConfig::percentile("p99.9"))
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::LatencyUsec)
            .diff(DiffType::None),
        InfoFieldType::new("search_flat_vector_index_search_latency_usec")
            .parse(ParseConfig::percentile("p99"))
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::LatencyUsec)
            .diff(DiffType::None),
        // EC CME specific - coordinator latencies
        InfoFieldType::new("search_coordinator_server_search_index_partition_success_latency_usec")
            .parse(ParseConfig::percentile("p99"))
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::LatencyUsec)
            .diff(DiffType::None),
        InfoFieldType::new("search_coordinator_client_search_index_partition_success_latency_usec")
            .parse(ParseConfig::percentile("p99"))
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::LatencyUsec)
            .diff(DiffType::None),
        // Error rates
        InfoFieldType::new("search_hnsw_add_exceptions_count")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        InfoFieldType::new("search_bounds_check_errors")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        // Index counters
        InfoFieldType::new("search_num_hnsw_edges")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_num_flat_nodes")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_num_vector_indexes")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_num_hnsw_indexes")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_num_flat_indexes")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_number_of_indexes")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_num_available_indexes")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_vectors_marked_deleted")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_num_hnsw_nodes")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::PercentageChange)
            .from_primary_only(),
        // Status - MemoryDB specific
        InfoFieldType::new("search_background_indexing_status")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
        InfoFieldType::new("search_num_active_backfills")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_current_backfill_progress_percentage")
            .aggregate(AggregationType::Average)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_num_active_queries")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None),
        // Memory stats
        InfoFieldType::new("search_vectors_memory_marked_deleted")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_vectors_bytes")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("search_interned_strings_memory")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::None)
            .from_primary_only(),
        // Network stats - EC CME specific
        InfoFieldType::new("search_network_bytes_in")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::RateCount),
        InfoFieldType::new("search_network_bytes_out")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::RateCount),
        InfoFieldType::new("search_coordinator_bytes_in")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::RateCount),
        InfoFieldType::new("search_coordinator_bytes_out")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::RateCount),
    ]
}

/// Default FT.INFO fields (supports EC and MemoryDB)
pub fn default_ftinfo_fields() -> Vec<InfoFieldType> {
    vec![
        // EC format fields
        InfoFieldType::new("num_docs")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount)
            .from_primary_only(),
        InfoFieldType::new("hash_indexing_failures")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        InfoFieldType::new("mutation_queue_size")
            .aggregate(AggregationType::Average)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("num_records")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("attributes.dim")
            .aggregate(AggregationType::MinMax)
            .display(DisplayFormat::MinMax)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
        InfoFieldType::new("attributes.M")
            .aggregate(AggregationType::MinMax)
            .display(DisplayFormat::MinMax)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
        InfoFieldType::new("attributes.capacity")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
        InfoFieldType::new("attributes.size")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
        // MemoryDB format fields
        InfoFieldType::new("index_name")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
        InfoFieldType::new("num_indexed_vectors")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount)
            .from_primary_only(),
        InfoFieldType::new("space_usage")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::MemoryGrowth)
            .from_primary_only(),
        InfoFieldType::new("vector_space_usage")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::MemoryGrowth)
            .from_primary_only(),
        InfoFieldType::new("fulltext_space_usage")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryHuman)
            .diff(DiffType::MemoryGrowth)
            .from_primary_only(),
        InfoFieldType::new("current_lag")
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("index_status")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
        InfoFieldType::new("index_degradation_percentage")
            .aggregate(AggregationType::Max)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only(),
        InfoFieldType::new("fields.vector_params.dimension")
            .aggregate(AggregationType::MinMax)
            .display(DisplayFormat::MinMax)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
        InfoFieldType::new("fields.vector_params.maximum_edges")
            .aggregate(AggregationType::MinMax)
            .display(DisplayFormat::MinMax)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
        InfoFieldType::new("fields.vector_params.current_capacity")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None)
            .from_primary_only()
            .no_per_node(),
    ]
}

/// Default INFO fields (general server stats)
pub fn default_info_fields() -> Vec<InfoFieldType> {
    vec![
        // Memory
        InfoFieldType::new("used_memory")
            .parse(ParseConfig::memory())
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::MemoryMb)
            .diff(DiffType::None)
            .from_primary_only(),
        // Keyspace hit/miss counters (from INFO stats)
        InfoFieldType::new("keyspace_hits")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        InfoFieldType::new("keyspace_misses")
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        // FT.SEARCH command stats
        InfoFieldType::new("cmdstat_FT.SEARCH")
            .prefix_match()
            .parse(ParseConfig::cmdstats("calls"))
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        InfoFieldType::new("cmdstat_FT.SEARCH")
            .prefix_match()
            .parse(ParseConfig::cmdstats("usec"))
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateMicrosec),
        InfoFieldType::new("cmdstat_FT.SEARCH")
            .prefix_match()
            .parse(ParseConfig::cmdstats("usec_per_call"))
            .aggregate(AggregationType::Average)
            .display(DisplayFormat::Integer)
            .diff(DiffType::None),
        InfoFieldType::new("cmdstat_FT.SEARCH")
            .prefix_match()
            .parse(ParseConfig::cmdstats("rejected"))
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
        InfoFieldType::new("cmdstat_FT.SEARCH")
            .prefix_match()
            .parse(ParseConfig::cmdstats("failed"))
            .aggregate(AggregationType::Sum)
            .display(DisplayFormat::Integer)
            .diff(DiffType::RateCount),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_integer() {
        let config = ParseConfig::integer();
        assert_eq!(parse_value("field:12345", &config), Some(12345));
        assert_eq!(parse_value("field:-100", &config), Some(-100));
    }

    #[test]
    fn test_parse_memory() {
        let config = ParseConfig::memory();
        assert_eq!(parse_value("field:1024", &config), Some(1024));
        assert_eq!(parse_value("field:1K", &config), Some(1024));
        assert_eq!(parse_value("field:1M", &config), Some(1024 * 1024));
        assert_eq!(parse_value("field:1.5G", &config), Some((1.5 * 1024.0 * 1024.0 * 1024.0) as i64));
    }

    #[test]
    fn test_parse_float_fixed() {
        let config = ParseConfig::float_fixed();
        assert_eq!(parse_value("field:1.5", &config), Some(1500));
        assert_eq!(parse_value("field:0.001", &config), Some(1));
    }

    #[test]
    fn test_parse_percentile() {
        let config = ParseConfig::percentile("p99");
        assert_eq!(
            parse_value("field:p50=100,p99=500,p99.9=1000", &config),
            Some(500)
        );
    }

    #[test]
    fn test_parse_cmdstats() {
        let config = ParseConfig::cmdstats("calls");
        assert_eq!(
            parse_value("cmdstat_FT.SEARCH:calls=1000,usec=50000,usec_per_call=50.00", &config),
            Some(1000)
        );
    }

    #[test]
    fn test_format_memory_human() {
        assert_eq!(format_memory_human(500), "500B");
        assert_eq!(format_memory_human(1024), "1.00K");
        assert_eq!(format_memory_human(1024 * 1024), "1.00M");
        assert_eq!(format_memory_human(1024 * 1024 * 1024), "1.00G");
    }

    #[test]
    fn test_calculate_diff() {
        assert_eq!(
            calculate_diff(100, 200, 10.0, DiffType::RateCount),
            Some(10.0)
        );
        assert_eq!(
            calculate_diff(0, 100, 10.0, DiffType::PercentageChange),
            None
        );
        assert_eq!(
            calculate_diff(100, 200, 10.0, DiffType::PercentageChange),
            Some(100.0)
        );
    }

    #[test]
    fn test_field_matching() {
        let exact = InfoFieldType::new("search_used_memory_bytes");
        assert!(exact.matches("search_used_memory_bytes"));
        assert!(!exact.matches("search_used_memory"));

        let prefix = InfoFieldType::new("cmdstat_FT.").prefix_match();
        assert!(prefix.matches("cmdstat_FT.SEARCH"));
        assert!(prefix.matches("cmdstat_FT.INFO"));
        assert!(!prefix.matches("cmdstat_GET"));
    }
}
