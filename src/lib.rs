//! valkey-search-benchmark library
//!
//! High-performance benchmark tool for Valkey with vector search support.

// Allow dead code during development - fields/types will be used in later phases
#![allow(dead_code)]
#![allow(unused_imports)]

pub mod benchmark;
pub mod client;
pub mod cluster;
pub mod config;
pub mod dataset;
pub mod metrics;
pub mod optimizer;
pub mod utils;
pub mod workload;
