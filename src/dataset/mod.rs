//! Schema-driven dataset loading and access
//!
//! This module provides memory-mapped access to binary datasets with structure
//! defined by external YAML schema files. All offsets are computed from the
//! schema at load time, enabling O(1) zero-copy access to any field.
//!
//! ## Architecture
//!
//! - **Schema** (`schema.rs`): YAML parsing for dataset structure definition
//! - **Layout** (`layout.rs`): Offset computation from schema
//! - **Context** (`context.rs`): Memory-mapped dataset access
//! - **Source** (`source.rs`): Abstract traits for data sources
//!
//! ## Traits
//!
//! - [`DataSource`] - Abstract data source for any binary data
//! - [`VectorDataSource`] - Extensions for vector search (queries, recall)
//!
//! ## Example
//!
//! ```ignore
//! use valkey_bench_rs::dataset::DatasetContext;
//!
//! // Load dataset from schema and data files
//! let dataset = DatasetContext::open("mnist.yaml", "mnist.bin")?;
//!
//! // Access vector data
//! let vector = dataset.get_vector_bytes(0);
//! let dim = dataset.dim();
//!
//! // Access any field
//! let category = dataset.get_field_bytes(0, "category");
//! ```

pub mod context;
pub mod layout;
pub mod schema;
pub mod source;

pub use context::DatasetContext;
pub use layout::{FieldLayout, RecordLayout, SectionLayout};
pub use schema::{
    DatasetSchema, DType, Encoding, FieldDef, FieldType, LengthSpec,
};
pub use source::{DataSource, VectorDataSource};
