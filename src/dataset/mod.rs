//! Dataset loading and access
//!
//! This module provides memory-mapped access to binary vector datasets.
//! The dataset format is compatible with the C implementation, allowing
//! zero-copy access to vectors for high-performance benchmarking.
//!
//! ## Traits
//!
//! - [`DataSource`] - Abstract data source for any binary data
//! - [`VectorDataSource`] - Extensions for vector search (queries, recall)
//!
//! These traits enable adding new data types without modifying core infrastructure.

pub mod binary_dataset;
pub mod header;
pub mod source;

pub use binary_dataset::DatasetContext;
pub use header::{DataType, DatasetHeader, DistanceMetricId, DATASET_MAGIC, HEADER_SIZE};
pub use source::{DataSource, VectorDataSource};
