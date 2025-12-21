//! Benchmark orchestration and workers
//!
//! This module provides the multi-threaded benchmark execution system:
//! - GlobalCounters: Atomic counters for cross-thread synchronization
//! - EventWorker: Event-driven worker with non-blocking I/O (like C's ae)
//! - Orchestrator: Coordinates workers and collects results

pub mod counters;
pub mod event_worker;
pub mod orchestrator;

pub use counters::GlobalCounters;
pub use event_worker::{EventWorker, EventWorkerResult, RecallStats};
pub use orchestrator::{format_count, format_throughput, BaseLatency, BenchmarkResult, KeyspaceStats, Orchestrator};
