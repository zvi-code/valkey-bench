//! Per-node metrics tracking
//!
//! Tracks operations, latency, and errors for individual cluster nodes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use hdrhistogram::Histogram;
use parking_lot::Mutex;

/// Metrics for a single node
pub struct NodeMetrics {
    /// Node identifier (host:port)
    pub node_id: String,
    /// Is this node a primary?
    pub is_primary: bool,
    /// Total operations completed
    pub ops_completed: AtomicU64,
    /// Total errors
    pub errors: AtomicU64,
    /// Latency histogram (thread-safe)
    histogram: Mutex<Histogram<u64>>,
    /// Start time for calculating rates
    start_time: Instant,
    /// Last snapshot time for diff calculations
    last_snapshot_time: Mutex<Instant>,
    /// Last ops count for rate calculation
    last_ops_count: AtomicU64,
}

impl NodeMetrics {
    /// Create new node metrics
    pub fn new(node_id: String, is_primary: bool) -> Self {
        let now = Instant::now();
        Self {
            node_id,
            is_primary,
            ops_completed: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            histogram: Mutex::new(
                Histogram::new_with_bounds(1, 3_600_000_000, 3)
                    .expect("Failed to create histogram"),
            ),
            start_time: now,
            last_snapshot_time: Mutex::new(now),
            last_ops_count: AtomicU64::new(0),
        }
    }

    /// Record a completed operation
    #[inline]
    pub fn record_op(&self, latency_us: u64) {
        self.ops_completed.fetch_add(1, Ordering::Relaxed);
        self.histogram.lock().record(latency_us).ok();
    }

    /// Record an error
    #[inline]
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record multiple operations (for batch)
    #[inline]
    pub fn record_batch(&self, count: u64, latency_us: u64) {
        self.ops_completed.fetch_add(count, Ordering::Relaxed);
        self.histogram.lock().record(latency_us).ok();
    }

    /// Get total operations
    pub fn total_ops(&self) -> u64 {
        self.ops_completed.load(Ordering::Relaxed)
    }

    /// Get total errors
    pub fn total_errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Get overall throughput (ops/sec)
    pub fn throughput(&self) -> f64 {
        let ops = self.total_ops();
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            ops as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get instantaneous throughput since last snapshot
    pub fn instantaneous_throughput(&self) -> f64 {
        let now = Instant::now();
        let current_ops = self.total_ops();

        let mut last_time = self.last_snapshot_time.lock();
        let last_ops = self.last_ops_count.swap(current_ops, Ordering::Relaxed);

        let elapsed = now.duration_since(*last_time).as_secs_f64();
        *last_time = now;

        if elapsed > 0.0 {
            (current_ops - last_ops) as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get latency percentile in microseconds
    pub fn percentile_us(&self, p: f64) -> u64 {
        self.histogram.lock().value_at_percentile(p)
    }

    /// Get latency percentile in milliseconds
    pub fn percentile_ms(&self, p: f64) -> f64 {
        self.percentile_us(p) as f64 / 1000.0
    }

    /// Get mean latency in microseconds
    pub fn mean_latency_us(&self) -> f64 {
        self.histogram.lock().mean()
    }

    /// Get max latency in microseconds
    pub fn max_latency_us(&self) -> u64 {
        self.histogram.lock().max()
    }

    /// Merge another histogram into this one
    pub fn merge_histogram(&self, other: &Histogram<u64>) {
        self.histogram.lock().add(other).ok();
    }

    /// Get snapshot of current metrics
    pub fn snapshot(&self) -> NodeMetricsSnapshot {
        let hist = self.histogram.lock();
        NodeMetricsSnapshot {
            node_id: self.node_id.clone(),
            is_primary: self.is_primary,
            ops_completed: self.total_ops(),
            errors: self.total_errors(),
            throughput: self.throughput(),
            mean_latency_ms: hist.mean() / 1000.0,
            p50_latency_ms: hist.value_at_percentile(50.0) as f64 / 1000.0,
            p95_latency_ms: hist.value_at_percentile(95.0) as f64 / 1000.0,
            p99_latency_ms: hist.value_at_percentile(99.0) as f64 / 1000.0,
            p999_latency_ms: hist.value_at_percentile(99.9) as f64 / 1000.0,
            max_latency_ms: hist.max() as f64 / 1000.0,
        }
    }
}

/// Snapshot of node metrics at a point in time
#[derive(Debug, Clone)]
pub struct NodeMetricsSnapshot {
    pub node_id: String,
    pub is_primary: bool,
    pub ops_completed: u64,
    pub errors: u64,
    pub throughput: f64,
    pub mean_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub p999_latency_ms: f64,
    pub max_latency_ms: f64,
}

impl NodeMetricsSnapshot {
    /// Format as CSV row
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            self.node_id,
            self.is_primary,
            self.ops_completed,
            self.errors,
            self.throughput,
            self.mean_latency_ms,
            self.p50_latency_ms,
            self.p95_latency_ms,
            self.p99_latency_ms,
            self.p999_latency_ms,
            self.max_latency_ms
        )
    }

    /// Get CSV header
    pub fn csv_header() -> &'static str {
        "node_id,is_primary,ops_completed,errors,throughput,mean_ms,p50_ms,p95_ms,p99_ms,p999_ms,max_ms"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_metrics_basic() {
        let metrics = NodeMetrics::new("127.0.0.1:6379".to_string(), true);

        metrics.record_op(1000); // 1ms
        metrics.record_op(2000); // 2ms
        metrics.record_error();

        assert_eq!(metrics.total_ops(), 2);
        assert_eq!(metrics.total_errors(), 1);
    }

    #[test]
    fn test_node_metrics_batch() {
        let metrics = NodeMetrics::new("127.0.0.1:6379".to_string(), true);

        metrics.record_batch(10, 5000);

        assert_eq!(metrics.total_ops(), 10);
    }

    #[test]
    fn test_node_metrics_snapshot() {
        let metrics = NodeMetrics::new("127.0.0.1:6379".to_string(), false);

        for i in 0..100 {
            metrics.record_op(1000 + i * 10);
        }

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.node_id, "127.0.0.1:6379");
        assert!(!snapshot.is_primary);
        assert_eq!(snapshot.ops_completed, 100);
        assert!(snapshot.mean_latency_ms > 0.0);
    }
}
