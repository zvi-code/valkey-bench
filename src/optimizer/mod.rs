//! Parameter optimization
//!
//! This module provides general-purpose parameter optimization for benchmarks.
//! It can tune multiple parameters (clients, threads, pipeline, ef_search) to
//! optimize for a target objective while satisfying constraints.
//!
//! # Multi-Objective Optimization
//!
//! Supports ordered multiple goals with tolerance. For example:
//! - Primary goal: maximize QPS
//! - Secondary goal: minimize p99 (only compared when QPS values are within tolerance)
//!
//! # Bounded Objectives
//!
//! Objectives can have bounds, e.g., "maximize QPS where QPS < 1M" (find the highest
//! achievable QPS that's still under 1M) or "minimize p99 where p99 > 0.1ms".
//!
//! # Examples
//!
//! ```ignore
//! // Vector search: maximize QPS with recall > 0.95
//! --optimize --objective "maximize:qps" --constraint "recall:gt:0.95" --tune "ef_search:10:500:10"
//!
//! // GET benchmark: maximize QPS with p99 < 0.1ms
//! --optimize --objective "maximize:qps" --constraint "p99_ms:lt:0.1" --tune "clients:10:200:10"
//!
//! // Multi-objective: maximize QPS, tiebreak on lowest p99 (4% tolerance)
//! --optimize --objective "maximize:qps,minimize:p99_ms" --tolerance 0.04 --tune "clients:10:200:10"
//!
//! // Bounded: find highest QPS under 1M req/s
//! --optimize --objective "maximize:qps:lt:1000000" --tune "clients:10:200:10"
//! ```

mod optimizer;

pub use optimizer::{
    CompareOp, Constraint, Constraints, Measurement, Metric, Objective, ObjectiveGoal,
    Objectives, OptimizationResult, OptimizeDirection, Optimizer, OptimizerBuilder,
    OptimizerPhase, ParameterType, TestConfig, TunableParameter,
};
