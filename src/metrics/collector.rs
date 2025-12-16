//! Metrics collector - aggregates metrics across all nodes
//!
//! Thread-safe collector that can be shared across worker threads.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use hdrhistogram::Histogram;
use parking_lot::RwLock;

use super::node_metrics::{NodeMetrics, NodeMetricsSnapshot};

/// Aggregated metrics collector
pub struct MetricsCollector {
    /// Per-node metrics
    nodes: RwLock<HashMap<String, Arc<NodeMetrics>>>,
    /// Global start time
    start_time: Instant,
    /// Test name
    test_name: String,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(test_name: &str) -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            start_time: Instant::now(),
            test_name: test_name.to_string(),
        }
    }

    /// Register a node for tracking
    pub fn register_node(&self, node_id: &str, is_primary: bool) -> Arc<NodeMetrics> {
        let mut nodes = self.nodes.write();
        let metrics = Arc::new(NodeMetrics::new(node_id.to_string(), is_primary));
        nodes.insert(node_id.to_string(), Arc::clone(&metrics));
        metrics
    }

    /// Get or create metrics for a node
    pub fn get_or_create_node(&self, node_id: &str, is_primary: bool) -> Arc<NodeMetrics> {
        // First try read lock
        {
            let nodes = self.nodes.read();
            if let Some(metrics) = nodes.get(node_id) {
                return Arc::clone(metrics);
            }
        }

        // Need write lock to create
        self.register_node(node_id, is_primary)
    }

    /// Get node metrics
    pub fn get_node(&self, node_id: &str) -> Option<Arc<NodeMetrics>> {
        self.nodes.read().get(node_id).cloned()
    }

    /// Get all node snapshots
    pub fn all_node_snapshots(&self) -> Vec<NodeMetricsSnapshot> {
        self.nodes
            .read()
            .values()
            .map(|m| m.snapshot())
            .collect()
    }

    /// Get aggregated metrics across all nodes
    pub fn aggregate(&self) -> AggregatedMetrics {
        let nodes = self.nodes.read();

        let mut total_ops = 0u64;
        let mut total_errors = 0u64;
        let merged_histogram =
            Histogram::new_with_bounds(1, 3_600_000_000, 3).expect("Failed to create histogram");

        for metrics in nodes.values() {
            total_ops += metrics.total_ops();
            total_errors += metrics.total_errors();
            metrics.merge_histogram(&merged_histogram);
        }

        let elapsed = self.start_time.elapsed();
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            total_ops as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        AggregatedMetrics {
            test_name: self.test_name.clone(),
            duration_secs: elapsed.as_secs_f64(),
            total_ops,
            total_errors,
            throughput,
            mean_latency_ms: merged_histogram.mean() / 1000.0,
            p50_latency_ms: merged_histogram.value_at_percentile(50.0) as f64 / 1000.0,
            p95_latency_ms: merged_histogram.value_at_percentile(95.0) as f64 / 1000.0,
            p99_latency_ms: merged_histogram.value_at_percentile(99.0) as f64 / 1000.0,
            p999_latency_ms: merged_histogram.value_at_percentile(99.9) as f64 / 1000.0,
            max_latency_ms: merged_histogram.max() as f64 / 1000.0,
            node_count: nodes.len(),
        }
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Get test name
    pub fn test_name(&self) -> &str {
        &self.test_name
    }
}

/// Aggregated metrics across all nodes
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    pub test_name: String,
    pub duration_secs: f64,
    pub total_ops: u64,
    pub total_errors: u64,
    pub throughput: f64,
    pub mean_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub p999_latency_ms: f64,
    pub max_latency_ms: f64,
    pub node_count: usize,
}

impl AggregatedMetrics {
    /// Format as summary string
    pub fn summary(&self) -> String {
        format!(
            "Test: {} | Duration: {:.2}s | Ops: {} | Errors: {} | Throughput: {:.2} ops/sec | p99: {:.3}ms",
            self.test_name,
            self.duration_secs,
            self.total_ops,
            self.total_errors,
            self.throughput,
            self.p99_latency_ms
        )
    }

    /// Convert to JSON object
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "test_name": self.test_name,
            "duration_secs": self.duration_secs,
            "total_ops": self.total_ops,
            "total_errors": self.total_errors,
            "throughput": self.throughput,
            "latency": {
                "mean_ms": self.mean_latency_ms,
                "p50_ms": self.p50_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms,
                "p999_ms": self.p999_latency_ms,
                "max_ms": self.max_latency_ms
            },
            "node_count": self.node_count
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_basic() {
        let collector = MetricsCollector::new("test");

        let node1 = collector.register_node("127.0.0.1:6379", true);
        let node2 = collector.register_node("127.0.0.1:6380", false);

        node1.record_op(1000);
        node1.record_op(2000);
        node2.record_op(1500);
        node2.record_error();

        let agg = collector.aggregate();
        assert_eq!(agg.total_ops, 3);
        assert_eq!(agg.total_errors, 1);
        assert_eq!(agg.node_count, 2);
    }

    #[test]
    fn test_metrics_collector_get_or_create() {
        let collector = MetricsCollector::new("test");

        let m1 = collector.get_or_create_node("node1", true);
        let m2 = collector.get_or_create_node("node1", true);

        // Should return same metrics instance
        assert!(Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn test_aggregated_metrics_json() {
        let metrics = AggregatedMetrics {
            test_name: "SET".to_string(),
            duration_secs: 10.5,
            total_ops: 100000,
            total_errors: 5,
            throughput: 9523.81,
            mean_latency_ms: 1.5,
            p50_latency_ms: 1.2,
            p95_latency_ms: 2.5,
            p99_latency_ms: 5.0,
            p999_latency_ms: 10.0,
            max_latency_ms: 25.0,
            node_count: 3,
        };

        let json = metrics.to_json();
        assert_eq!(json["test_name"], "SET");
        assert_eq!(json["total_ops"], 100000);
    }
}
