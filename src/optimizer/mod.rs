//! Parameter optimization
//!
//! This module implements adaptive parameter optimization for vector search,
//! with phases for feasibility testing, recall search, and throughput optimization.

mod optimizer;

pub use optimizer::{Constraints, Measurement, OptimizationResult, Optimizer, OptimizerPhase};
