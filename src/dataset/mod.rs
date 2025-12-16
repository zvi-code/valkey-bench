//! Dataset loading and access
//!
//! This module provides memory-mapped access to binary vector datasets.
//! The dataset format is compatible with the C implementation, allowing
//! zero-copy access to vectors for high-performance benchmarking.

pub mod binary_dataset;
pub mod header;

pub use binary_dataset::DatasetContext;
pub use header::{DataType, DatasetHeader, DistanceMetricId, DATASET_MAGIC, HEADER_SIZE};
