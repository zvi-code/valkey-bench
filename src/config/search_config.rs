//! Vector search configuration

use super::cli::{CliArgs, DistanceMetric, VectorAlgorithm};

/// Vector search configuration
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub index_name: String,
    pub vector_field: String,
    pub prefix: String,
    pub algorithm: VectorAlgorithm,
    pub distance_metric: DistanceMetric,
    pub dim: u32,
    pub k: u32,
    pub ef_construction: Option<u32>,
    pub hnsw_m: Option<u32>,
    pub ef_search: Option<u32>,
    pub nocontent: bool,
}

impl SearchConfig {
    pub fn from_cli(args: &CliArgs) -> Self {
        Self {
            index_name: args.search_index.clone(),
            vector_field: args.search_vector_field.clone(),
            prefix: args.search_prefix.clone(),
            algorithm: args.search_algorithm,
            distance_metric: args.search_distance,
            dim: args.vector_dim,
            k: args.search_k,
            ef_construction: args.ef_construction,
            hnsw_m: args.hnsw_m,
            ef_search: args.ef_search,
            nocontent: args.nocontent,
        }
    }

    /// Update dimension from dataset
    pub fn set_dim(&mut self, dim: u32) {
        self.dim = dim;
    }

    /// Get vector byte length (dim * sizeof(f32))
    pub fn vec_byte_len(&self) -> usize {
        self.dim as usize * std::mem::size_of::<f32>()
    }
}
