//! Metrics reporter - output formatting and export
//!
//! Supports multiple output formats:
//! - Console (human-readable)
//! - JSON
//! - CSV

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use super::collector::AggregatedMetrics;
use super::node_metrics::NodeMetricsSnapshot;

/// Output format for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Console,
    Json,
    Csv,
}

/// Metrics reporter
pub struct MetricsReporter {
    format: OutputFormat,
}

impl MetricsReporter {
    /// Create new reporter with specified format
    pub fn new(format: OutputFormat) -> Self {
        Self { format }
    }

    /// Report aggregated metrics to stdout
    pub fn report_aggregated(&self, metrics: &AggregatedMetrics) {
        match self.format {
            OutputFormat::Console => self.report_console(metrics),
            OutputFormat::Json => self.report_json(metrics),
            OutputFormat::Csv => self.report_csv_aggregated(metrics),
        }
    }

    /// Report to console (human-readable)
    fn report_console(&self, metrics: &AggregatedMetrics) {
        println!("\n=== {} ===", metrics.test_name);
        println!("Throughput: {:.2} req/sec", metrics.throughput);
        println!("Total requests: {}", metrics.total_ops);
        println!("Duration: {:.2}s", metrics.duration_secs);
        println!("Errors: {}", metrics.total_errors);
        println!("\nLatency (ms):");
        println!("  avg: {:.3}", metrics.mean_latency_ms);
        println!("  p50: {:.3}", metrics.p50_latency_ms);
        println!("  p95: {:.3}", metrics.p95_latency_ms);
        println!("  p99: {:.3}", metrics.p99_latency_ms);
        println!("  p99.9: {:.3}", metrics.p999_latency_ms);
        println!("  max: {:.3}", metrics.max_latency_ms);
    }

    /// Report as JSON
    fn report_json(&self, metrics: &AggregatedMetrics) {
        println!("{}", serde_json::to_string_pretty(&metrics.to_json()).unwrap());
    }

    /// Report as CSV (single row for aggregated)
    fn report_csv_aggregated(&self, metrics: &AggregatedMetrics) {
        println!(
            "{},{:.2},{},{},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            metrics.test_name,
            metrics.duration_secs,
            metrics.total_ops,
            metrics.total_errors,
            metrics.throughput,
            metrics.mean_latency_ms,
            metrics.p50_latency_ms,
            metrics.p95_latency_ms,
            metrics.p99_latency_ms,
            metrics.p999_latency_ms,
            metrics.max_latency_ms
        );
    }

    /// Write metrics to JSON file
    pub fn write_json_file(
        &self,
        path: &Path,
        aggregated: &AggregatedMetrics,
        nodes: &[NodeMetricsSnapshot],
    ) -> io::Result<()> {
        let json = serde_json::json!({
            "summary": aggregated.to_json(),
            "nodes": nodes.iter().map(|n| {
                serde_json::json!({
                    "node_id": n.node_id,
                    "is_primary": n.is_primary,
                    "ops_completed": n.ops_completed,
                    "errors": n.errors,
                    "throughput": n.throughput,
                    "latency": {
                        "mean_ms": n.mean_latency_ms,
                        "p50_ms": n.p50_latency_ms,
                        "p95_ms": n.p95_latency_ms,
                        "p99_ms": n.p99_latency_ms,
                        "p999_ms": n.p999_latency_ms,
                        "max_ms": n.max_latency_ms
                    }
                })
            }).collect::<Vec<_>>()
        });

        let mut file = File::create(path)?;
        writeln!(file, "{}", serde_json::to_string_pretty(&json).unwrap())?;
        Ok(())
    }

    /// Write node metrics to CSV file
    pub fn write_csv_file(&self, path: &Path, nodes: &[NodeMetricsSnapshot]) -> io::Result<()> {
        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "{}", NodeMetricsSnapshot::csv_header())?;

        // Write rows
        for node in nodes {
            writeln!(file, "{}", node.to_csv_row())?;
        }

        Ok(())
    }

    /// Print per-node summary to console
    pub fn report_nodes_console(&self, nodes: &[NodeMetricsSnapshot]) {
        if nodes.is_empty() {
            return;
        }

        println!("\nPer-node metrics:");
        println!(
            "{:40} {:>10} {:>10} {:>12} {:>10} {:>10}",
            "Node", "Ops", "Errors", "Throughput", "p99 (ms)", "Max (ms)"
        );
        println!("{}", "-".repeat(100));

        for node in nodes {
            println!(
                "{:40} {:>10} {:>10} {:>12.2} {:>10.3} {:>10.3}",
                node.node_id,
                node.ops_completed,
                node.errors,
                node.throughput,
                node.p99_latency_ms,
                node.max_latency_ms
            );
        }
    }
}

/// Benchmark results collection for export
#[derive(Debug)]
pub struct BenchmarkResults {
    /// All test results
    pub tests: Vec<TestResult>,
    /// Configuration summary
    pub config_summary: String,
}

/// Single test result
#[derive(Debug)]
pub struct TestResult {
    pub aggregated: AggregatedMetrics,
    pub nodes: Vec<NodeMetricsSnapshot>,
}

impl BenchmarkResults {
    /// Create new results collection
    pub fn new(config_summary: &str) -> Self {
        Self {
            tests: Vec::new(),
            config_summary: config_summary.to_string(),
        }
    }

    /// Add a test result
    pub fn add_test(&mut self, aggregated: AggregatedMetrics, nodes: Vec<NodeMetricsSnapshot>) {
        self.tests.push(TestResult { aggregated, nodes });
    }

    /// Export all results to JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "config": self.config_summary,
            "tests": self.tests.iter().map(|t| {
                serde_json::json!({
                    "summary": t.aggregated.to_json(),
                    "nodes": t.nodes.iter().map(|n| {
                        serde_json::json!({
                            "node_id": n.node_id,
                            "is_primary": n.is_primary,
                            "ops_completed": n.ops_completed,
                            "errors": n.errors,
                            "throughput": n.throughput,
                            "latency": {
                                "mean_ms": n.mean_latency_ms,
                                "p50_ms": n.p50_latency_ms,
                                "p95_ms": n.p95_latency_ms,
                                "p99_ms": n.p99_latency_ms,
                                "p999_ms": n.p999_latency_ms,
                                "max_ms": n.max_latency_ms
                            }
                        })
                    }).collect::<Vec<_>>()
                })
            }).collect::<Vec<_>>()
        })
    }

    /// Write all results to JSON file
    pub fn write_json(&self, path: &Path) -> io::Result<()> {
        let mut file = File::create(path)?;
        writeln!(file, "{}", serde_json::to_string_pretty(&self.to_json()).unwrap())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format() {
        let reporter = MetricsReporter::new(OutputFormat::Console);
        assert_eq!(reporter.format, OutputFormat::Console);
    }

    #[test]
    fn test_benchmark_results() {
        let mut results = BenchmarkResults::new("test config");

        let agg = AggregatedMetrics {
            test_name: "SET".to_string(),
            duration_secs: 1.0,
            total_ops: 1000,
            total_errors: 0,
            throughput: 1000.0,
            mean_latency_ms: 1.0,
            p50_latency_ms: 0.9,
            p95_latency_ms: 1.5,
            p99_latency_ms: 2.0,
            p999_latency_ms: 3.0,
            max_latency_ms: 5.0,
            node_count: 1,
        };

        results.add_test(agg, vec![]);
        assert_eq!(results.tests.len(), 1);
    }
}
