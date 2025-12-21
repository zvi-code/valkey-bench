//! Vector search configuration

use super::cli::{CliArgs, DistanceMetric, VectorAlgorithm};
use crate::workload::{NumericFieldConfig, NumericFieldSet, NumericValueType, TagDistributionSet};

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
    /// Tag field name (optional, for filtered search)
    pub tag_field: Option<String>,
    /// Tag distribution set for generating tags during vec-load
    pub tag_distributions: Option<TagDistributionSet>,
    /// Tag filter pattern for vec-query (e.g., "tag1|tag2")
    pub tag_filter: Option<String>,
    /// Maximum tag payload length
    pub tag_max_len: usize,
    /// Numeric field name (optional, for filtered search) - backward compatibility
    pub numeric_field: Option<String>,
    /// Extended numeric field configurations with types and distributions
    pub numeric_fields: NumericFieldSet,
}

impl SearchConfig {
    pub fn from_cli(args: &CliArgs) -> Self {
        // Parse tag distributions if provided
        let tag_distributions = args.search_tags.as_ref().and_then(|tags_str| {
            match TagDistributionSet::parse(tags_str) {
                Ok(set) => Some(set.with_max_len(args.tag_max_len)),
                Err(e) => {
                    eprintln!("Warning: Failed to parse --search-tags: {}", e);
                    None
                }
            }
        });

        // Parse extended numeric field configurations
        let mut numeric_fields = NumericFieldSet::new();
        for config_str in &args.numeric_field_configs {
            match NumericFieldConfig::parse(config_str) {
                Ok(config) => numeric_fields.add(config),
                Err(e) => {
                    eprintln!("Warning: Failed to parse --numeric-field-config '{}': {}", config_str, e);
                }
            }
        }

        // If no extended configs but simple --numeric-field is set, create a key-based config
        if numeric_fields.is_empty() {
            if let Some(ref field_name) = args.numeric_field {
                // Create a simple key-based numeric field for backward compatibility
                let config = NumericFieldConfig::new_key_based(field_name, 0.0, f64::MAX);
                numeric_fields.add(config);
            }
        }

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
            tag_field: args.tag_field.clone(),
            tag_distributions,
            tag_filter: args.tag_filter.clone(),
            tag_max_len: args.tag_max_len,
            numeric_field: args.numeric_field.clone(),
            numeric_fields,
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
