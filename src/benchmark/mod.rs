//! Benchmark orchestration and workers
//!
//! This module provides the multi-threaded benchmark execution system:
//! - GlobalCounters: Atomic counters for cross-thread synchronization
//! - BenchmarkWorker: Independent worker that owns its clients (blocking I/O)
//! - EventWorker: Event-driven worker with non-blocking I/O (like C's ae)
//! - Orchestrator: Coordinates workers and collects results

pub mod counters;
pub mod event_worker;
pub mod orchestrator;
pub mod worker;

pub use counters::GlobalCounters;
pub use event_worker::{EventWorker, EventWorkerResult};
pub use orchestrator::{BenchmarkResult, Orchestrator};
pub use worker::{BenchmarkWorker, RecallStats, WorkerResult};
