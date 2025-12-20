//! Adaptive parameter optimization
//!
//! Phase 1: Feasibility - verify basic functionality
//! Phase 2: Recall - find ef_search that meets recall target
//! Phase 3: Throughput - maximize QPS while maintaining constraints

use crate::benchmark::BenchmarkResult;

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

impl Default for Constraints {
    fn default() -> Self {
        Self {
            min_recall: 0.95,
            max_p99_ms: Some(100.0),
            target_qps: None,
        }
    }
}

/// Single measurement from a benchmark run
#[derive(Debug, Clone)]
pub struct Measurement {
    pub ef_search: u32,
    pub recall: f64,
    pub qps: f64,
    pub p99_ms: f64,
    pub constraints_met: bool,
}

/// Result of optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub ef_search: u32,
    pub recall: f64,
    pub qps: f64,
    pub p99_ms: f64,
}

/// Optimizer state machine
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

    /// Get best result found so far
    pub fn best_result(&self) -> Option<&OptimizationResult> {
        self.best_result.as_ref()
    }

    /// Get all measurements
    pub fn history(&self) -> &[Measurement] {
        &self.history
    }

    /// Get constraints
    pub fn constraints(&self) -> &Constraints {
        &self.constraints
    }

    /// Check if optimization is complete
    pub fn is_complete(&self) -> bool {
        self.phase == OptimizerPhase::Complete
    }

    /// Process benchmark result and advance optimization
    ///
    /// Returns the next ef_search value to test, or None if complete
    pub fn process_result(&mut self, result: &BenchmarkResult) -> Option<u32> {
        let recall = result.recall_stats.average();
        let qps = result.throughput;
        let p99_ms = result.percentile_ms(99.0);

        // Check if constraints are met
        let recall_met = recall >= self.constraints.min_recall;
        let latency_met = self
            .constraints
            .max_p99_ms
            .map(|max| p99_ms <= max)
            .unwrap_or(true);
        let constraints_met = recall_met && latency_met;

        // Record measurement
        let measurement = Measurement {
            ef_search: self.ef_search_current,
            recall,
            qps,
            p99_ms,
            constraints_met,
        };
        self.history.push(measurement);
        self.current_iteration += 1;

        // Update best result if constraints met and better QPS
        if constraints_met {
            let is_better = self
                .best_result
                .as_ref()
                .map(|b| qps > b.qps)
                .unwrap_or(true);
            if is_better {
                self.best_result = Some(OptimizationResult {
                    ef_search: self.ef_search_current,
                    recall,
                    qps,
                    p99_ms,
                });
            }
        }

        // Advance state machine
        match self.phase {
            OptimizerPhase::Feasibility => {
                if constraints_met {
                    // System works, move to recall search
                    self.phase = OptimizerPhase::RecallSearch;
                    // Start binary search from current position
                    self.ef_search_low = 10;
                    // Try lower ef_search to find minimum that still meets recall
                    self.ef_search_current = (self.ef_search_low + self.ef_search_high) / 2;
                } else if !recall_met {
                    // Increase ef_search to improve recall
                    self.ef_search_current =
                        (self.ef_search_current * 2).min(self.ef_search_high);
                    if self.ef_search_current >= self.ef_search_high {
                        // Cannot meet recall target, complete with failure
                        self.phase = OptimizerPhase::Complete;
                        return None;
                    }
                } else {
                    // Latency constraint not met, try higher ef_search might help
                    // (usually doesn't, but we need to verify)
                    self.ef_search_current = self.ef_search_current * 2;
                    if self.ef_search_current >= self.ef_search_high
                        || self.current_iteration >= 5
                    {
                        self.phase = OptimizerPhase::Complete;
                        return None;
                    }
                }
            }
            OptimizerPhase::RecallSearch => {
                // Binary search for minimum ef_search that meets recall
                if self.ef_search_high - self.ef_search_low <= 5
                    || self.current_iteration >= self.max_iterations
                {
                    // Converged, move to throughput optimization
                    self.phase = OptimizerPhase::ThroughputOptimization;
                    // Use best known ef_search as starting point
                    if let Some(ref best) = self.best_result {
                        self.ef_search_current = best.ef_search;
                    }
                    return self.next_throughput_test();
                }

                if recall_met {
                    // Can try lower ef_search
                    self.ef_search_high = self.ef_search_current;
                } else {
                    // Need higher ef_search
                    self.ef_search_low = self.ef_search_current;
                }
                self.ef_search_current = (self.ef_search_low + self.ef_search_high) / 2;
            }
            OptimizerPhase::ThroughputOptimization => {
                return self.next_throughput_test();
            }
            OptimizerPhase::Complete => {
                return None;
            }
        }

        if self.current_iteration >= self.max_iterations {
            self.phase = OptimizerPhase::Complete;
            return None;
        }

        Some(self.ef_search_current)
    }

    /// Get next ef_search value for throughput testing
    fn next_throughput_test(&mut self) -> Option<u32> {
        // In throughput phase, we test a few values around the optimal recall point
        // to find the best balance of recall and throughput

        // Find measurements that met recall constraint
        let valid_measurements: Vec<&Measurement> =
            self.history.iter().filter(|m| m.constraints_met).collect();

        if valid_measurements.is_empty() {
            self.phase = OptimizerPhase::Complete;
            return None;
        }

        // Try testing slightly lower ef_search to see if we can maintain recall
        // while improving throughput
        if let Some(ref best) = self.best_result {
            let test_value = (best.ef_search as f64 * 0.9) as u32;
            if test_value >= self.ef_search_low
                && !self.history.iter().any(|m| m.ef_search == test_value)
            {
                self.ef_search_current = test_value;
                return Some(test_value);
            }
        }

        // Optimization complete
        self.phase = OptimizerPhase::Complete;
        None
    }

    /// Format optimization summary
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str(&format!(
            "Optimization Status: {:?}\n",
            self.phase
        ));
        s.push_str(&format!(
            "Iterations: {}\n",
            self.current_iteration
        ));
        s.push_str(&format!(
            "Constraints: min_recall={:.2}, max_p99={:?}ms\n",
            self.constraints.min_recall, self.constraints.max_p99_ms
        ));

        if let Some(ref best) = self.best_result {
            s.push_str("\nBest Result:\n");
            s.push_str(&format!("  ef_search: {}\n", best.ef_search));
            s.push_str(&format!("  recall: {:.4}\n", best.recall));
            s.push_str(&format!("  qps: {:.2}\n", best.qps));
            s.push_str(&format!("  p99: {:.3}ms\n", best.p99_ms));
        } else {
            s.push_str("\nNo result meeting constraints found.\n");
        }

        s.push_str("\nMeasurement History:\n");
        for m in &self.history {
            s.push_str(&format!(
                "  ef_search={:4} recall={:.4} qps={:8.2} p99={:6.3}ms {}\n",
                m.ef_search,
                m.recall,
                m.qps,
                m.p99_ms,
                if m.constraints_met { "[OK]" } else { "[FAIL]" }
            ));
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::RecallStats;
    use hdrhistogram::Histogram;
    use std::time::Duration;

    fn make_result(recall: f64, qps: f64, p99_ms: f64) -> BenchmarkResult {
        let mut histogram =
            Histogram::new_with_bounds(1, 3_600_000_000, 3).expect("histogram");
        // Record p99 value in microseconds
        histogram.record((p99_ms * 1000.0) as u64).ok();

        let mut recall_stats = RecallStats::new();
        recall_stats.record(recall);

        BenchmarkResult {
            test_name: "test".to_string(),
            total_requests: 1000,
            duration: Duration::from_secs(1),
            throughput: qps,
            histogram,
            recall_stats,
            error_count: 0,
            node_metrics: Vec::new(),
        }
    }

    #[test]
    fn test_optimizer_feasibility_pass() {
        let constraints = Constraints {
            min_recall: 0.95,
            max_p99_ms: Some(100.0),
            target_qps: None,
        };

        let mut opt = Optimizer::new(constraints, 1000);
        assert_eq!(opt.phase(), OptimizerPhase::Feasibility);

        // Good result - should pass feasibility
        let result = make_result(0.98, 10000.0, 50.0);
        let next = opt.process_result(&result);

        assert!(next.is_some());
        assert_eq!(opt.phase(), OptimizerPhase::RecallSearch);
    }

    #[test]
    fn test_optimizer_feasibility_fail_recall() {
        let constraints = Constraints {
            min_recall: 0.95,
            max_p99_ms: Some(100.0),
            target_qps: None,
        };

        // Use higher max_ef_search to allow multiple retries
        let mut opt = Optimizer::new(constraints, 500);
        assert_eq!(opt.phase(), OptimizerPhase::Feasibility);

        // Bad recall - should increase ef_search
        let result = make_result(0.80, 10000.0, 50.0);
        let next = opt.process_result(&result);

        assert!(next.is_some());
        assert!(opt.current_ef_search() > 100); // Should have increased
        assert_eq!(opt.phase(), OptimizerPhase::Feasibility);
    }

    #[test]
    fn test_optimizer_binary_search() {
        let constraints = Constraints {
            min_recall: 0.95,
            max_p99_ms: Some(100.0),
            target_qps: None,
        };

        let mut opt = Optimizer::new(constraints, 500);

        // Pass feasibility
        let result = make_result(0.98, 10000.0, 50.0);
        opt.process_result(&result);
        assert_eq!(opt.phase(), OptimizerPhase::RecallSearch);

        // Simulate binary search
        for _ in 0..10 {
            let ef = opt.current_ef_search();
            let recall = if ef >= 100 { 0.96 } else { 0.85 };
            let result = make_result(recall, 10000.0 + (500.0 - ef as f64) * 10.0, 50.0);
            if opt.process_result(&result).is_none() {
                break;
            }
        }

        // Should have found an optimum
        assert!(opt.best_result().is_some());
    }

    #[test]
    fn test_optimizer_constraints() {
        let constraints = Constraints::default();
        assert_eq!(constraints.min_recall, 0.95);
        assert_eq!(constraints.max_p99_ms, Some(100.0));
    }
}
