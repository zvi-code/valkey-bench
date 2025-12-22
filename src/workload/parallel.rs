//! Parallel workload execution
//!
//! This module provides support for running multiple workload types concurrently
//! with weighted traffic ratios. For example, a mix of 80% GET and 20% SET.

use super::{PrepareResult, Workload, WorkloadType};
use crate::utils::{BenchmarkError, Result};

/// A component of a parallel workload with a weight
#[derive(Debug, Clone)]
pub struct ParallelComponent {
    /// Workload type (e.g., GET, SET)
    pub workload_type: WorkloadType,
    /// Weight for this workload (0.0 to 1.0, must sum to 1.0 across all components)
    pub weight: f64,
}

impl ParallelComponent {
    /// Create a new parallel component
    pub fn new(workload_type: WorkloadType, weight: f64) -> Self {
        Self {
            workload_type,
            weight,
        }
    }
}

/// Parallel workload that runs multiple workload types with weighted traffic
#[derive(Debug, Clone)]
pub struct ParallelWorkload {
    /// Components and their weights
    components: Vec<ParallelComponent>,
    /// Precomputed cumulative weights for O(1) selection
    cumulative_weights: Vec<f64>,
    /// Combined name for display
    name: String,
}

impl ParallelWorkload {
    /// Create a new parallel workload from components
    ///
    /// Weights are normalized to sum to 1.0.
    pub fn new(components: Vec<ParallelComponent>) -> Result<Self> {
        if components.is_empty() {
            return Err(BenchmarkError::Config(
                "ParallelWorkload requires at least one component".to_string(),
            ));
        }

        // Normalize weights
        let total_weight: f64 = components.iter().map(|c| c.weight).sum();
        if total_weight <= 0.0 {
            return Err(BenchmarkError::Config(
                "Total weight must be positive".to_string(),
            ));
        }

        let normalized: Vec<ParallelComponent> = components
            .into_iter()
            .map(|c| ParallelComponent {
                workload_type: c.workload_type,
                weight: c.weight / total_weight,
            })
            .collect();

        // Compute cumulative weights for O(1) selection
        let mut cumulative = Vec::with_capacity(normalized.len());
        let mut sum = 0.0;
        for c in &normalized {
            sum += c.weight;
            cumulative.push(sum);
        }

        // Build display name
        let name = normalized
            .iter()
            .map(|c| format!("{}:{:.0}%", c.workload_type.as_str(), c.weight * 100.0))
            .collect::<Vec<_>>()
            .join("+");

        Ok(Self {
            components: normalized,
            cumulative_weights: cumulative,
            name,
        })
    }

    /// Parse parallel workload specification from CLI string
    ///
    /// Format: "workload1:weight1,workload2:weight2,..."
    /// Example: "get:0.8,set:0.2" for 80% GET, 20% SET
    pub fn parse(spec: &str) -> Result<Self> {
        let mut components = Vec::new();

        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            let mut split = part.splitn(2, ':');
            let workload_str = split
                .next()
                .ok_or_else(|| BenchmarkError::Config("Missing workload type".to_string()))?;
            let weight_str = split
                .next()
                .ok_or_else(|| BenchmarkError::Config("Missing weight".to_string()))?;

            let workload_type = WorkloadType::parse(workload_str).ok_or_else(|| {
                BenchmarkError::Config(format!("Unknown workload type: {}", workload_str))
            })?;

            let weight: f64 = weight_str.parse().map_err(|_| {
                BenchmarkError::Config(format!("Invalid weight: {}", weight_str))
            })?;

            if weight < 0.0 {
                return Err(BenchmarkError::Config(format!(
                    "Weight must be non-negative: {}",
                    weight
                )));
            }

            components.push(ParallelComponent::new(workload_type, weight));
        }

        Self::new(components)
    }

    /// Get all components
    pub fn components(&self) -> &[ParallelComponent] {
        &self.components
    }

    /// Get cumulative weights (for weighted selection)
    pub fn cumulative_weights(&self) -> &[f64] {
        &self.cumulative_weights
    }

    /// Select a workload type based on a random value in [0, 1)
    ///
    /// Uses binary search for O(log n) selection (constant for typical 2-4 workloads).
    pub fn select(&self, random: f64) -> &ParallelComponent {
        // Binary search for the correct bucket
        match self.cumulative_weights.binary_search_by(|w| {
            w.partial_cmp(&random).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => &self.components[idx.min(self.components.len() - 1)],
            Err(idx) => &self.components[idx.min(self.components.len() - 1)],
        }
    }

    /// Check if any component is a write workload
    pub fn has_writes(&self) -> bool {
        self.components.iter().any(|c| c.workload_type.is_write())
    }

    /// Check if any component requires a dataset
    pub fn requires_dataset(&self) -> bool {
        self.components
            .iter()
            .any(|c| c.workload_type.requires_dataset())
    }
}

impl Workload for ParallelWorkload {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_write(&self) -> bool {
        self.has_writes()
    }

    fn requires_dataset(&self) -> bool {
        ParallelWorkload::requires_dataset(self)
    }

    fn prepare(&self) -> Result<PrepareResult> {
        // Each component might need preparation
        // For now, return empty - preparation happens in orchestrator
        Ok(PrepareResult::empty())
    }
}

/// Builder for parallel workloads
#[derive(Default)]
pub struct ParallelWorkloadBuilder {
    components: Vec<ParallelComponent>,
}

impl ParallelWorkloadBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a workload with the given weight
    pub fn add(mut self, workload_type: WorkloadType, weight: f64) -> Self {
        self.components.push(ParallelComponent::new(workload_type, weight));
        self
    }

    /// Build the parallel workload
    pub fn build(self) -> Result<ParallelWorkload> {
        ParallelWorkload::new(self.components)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let pw = ParallelWorkload::parse("get:0.8,set:0.2").unwrap();
        assert_eq!(pw.components.len(), 2);
        assert_eq!(pw.components[0].workload_type, WorkloadType::Get);
        assert!((pw.components[0].weight - 0.8).abs() < 0.001);
        assert_eq!(pw.components[1].workload_type, WorkloadType::Set);
        assert!((pw.components[1].weight - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_parse_normalized() {
        // Weights don't need to sum to 1.0 - they get normalized
        let pw = ParallelWorkload::parse("get:80,set:20").unwrap();
        assert!((pw.components[0].weight - 0.8).abs() < 0.001);
        assert!((pw.components[1].weight - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_select() {
        let pw = ParallelWorkload::parse("get:0.8,set:0.2").unwrap();

        // 0.0 - 0.8 should select GET
        assert_eq!(pw.select(0.0).workload_type, WorkloadType::Get);
        assert_eq!(pw.select(0.5).workload_type, WorkloadType::Get);
        assert_eq!(pw.select(0.79).workload_type, WorkloadType::Get);

        // 0.8 - 1.0 should select SET
        assert_eq!(pw.select(0.81).workload_type, WorkloadType::Set);
        assert_eq!(pw.select(0.99).workload_type, WorkloadType::Set);
    }

    #[test]
    fn test_builder() {
        let pw = ParallelWorkloadBuilder::new()
            .add(WorkloadType::Get, 0.7)
            .add(WorkloadType::Set, 0.3)
            .build()
            .unwrap();

        assert_eq!(pw.components.len(), 2);
        assert_eq!(pw.name(), "GET:70%+SET:30%");
    }

    #[test]
    fn test_name_display() {
        let pw = ParallelWorkload::parse("get:50,set:50").unwrap();
        assert_eq!(pw.name(), "GET:50%+SET:50%");
    }

    #[test]
    fn test_has_writes() {
        let pw_read = ParallelWorkload::parse("get:1").unwrap();
        assert!(!pw_read.has_writes());

        let pw_mixed = ParallelWorkload::parse("get:0.8,set:0.2").unwrap();
        assert!(pw_mixed.has_writes());
    }

    #[test]
    fn test_requires_dataset() {
        let pw_basic = ParallelWorkload::parse("get:0.8,set:0.2").unwrap();
        assert!(!pw_basic.requires_dataset());

        let pw_vec = ParallelWorkload::parse("vec-query:1").unwrap();
        assert!(pw_vec.requires_dataset());
    }

    #[test]
    fn test_empty_fails() {
        let result = ParallelWorkload::parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_workload() {
        let result = ParallelWorkload::parse("unknown:0.5");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_weight() {
        let result = ParallelWorkload::parse("get:abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_workload_trait() {
        let pw = ParallelWorkload::parse("get:0.8,set:0.2").unwrap();
        assert_eq!(pw.name(), "GET:80%+SET:20%");
        assert!(pw.is_write()); // has SET which is a write
    }
}
