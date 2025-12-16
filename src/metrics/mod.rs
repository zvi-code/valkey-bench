//! Metrics collection and reporting
//!
//! This module provides:
//! - Per-node metrics tracking (ops, latency, errors)
//! - Aggregated metrics across all nodes
//! - Temporal diff calculation for INFO statistics
//! - JSON/CSV export

pub mod collector;
pub mod node_metrics;
pub mod reporter;

pub use collector::MetricsCollector;
pub use node_metrics::NodeMetrics;
pub use reporter::MetricsReporter;
