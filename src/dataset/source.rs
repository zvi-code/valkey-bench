//! Abstract data source traits for binary datasets
//!
//! This module provides trait abstractions for data sources, enabling:
//! - Easy addition of new data types (not just vectors)
//! - Unified interface for different binary formats
//! - Separation of vector-specific operations (recall, ground truth)

use std::collections::HashSet;

/// Abstract data source for workloads that load data from binary files
///
/// This trait provides the minimal interface needed for loading items from
/// a binary dataset. It can be implemented for various data types:
/// - Vectors (f32, f16, binary)
/// - Text embeddings
/// - Any fixed-size binary data
pub trait DataSource: Send + Sync {
    /// Number of items in the dataset
    fn num_items(&self) -> u64;

    /// Get raw bytes for item at index (zero-copy)
    ///
    /// # Arguments
    /// * `idx` - Item index (0-based)
    ///
    /// # Panics
    /// May panic if idx >= num_items() in debug mode
    fn get_item_bytes(&self, idx: u64) -> &[u8];

    /// Item byte length (fixed size per item)
    fn item_byte_len(&self) -> usize;
}

/// Vector-specific extensions for search operations
///
/// This trait extends DataSource with operations specific to vector search:
/// - Query vectors (separate from database vectors)
/// - Ground truth neighbors for recall computation
/// - Recall calculation
pub trait VectorDataSource: DataSource {
    /// Number of query vectors in the dataset
    fn num_queries(&self) -> u64;

    /// Get raw bytes for query vector at index (zero-copy)
    fn get_query_bytes(&self, idx: u64) -> &[u8];

    /// Compute recall@k against ground truth
    ///
    /// # Arguments
    /// * `query_idx` - Query index (0-based)
    /// * `result_ids` - Vector IDs returned by search
    /// * `k` - Number of results to consider
    ///
    /// # Returns
    /// Recall value between 0.0 and 1.0
    fn compute_recall(&self, query_idx: u64, result_ids: &[u64], k: usize) -> f64;

    /// Get all unique vector IDs that appear in ground truth
    ///
    /// These vectors are "protected" during deletion benchmarks to ensure
    /// valid recall computation.
    fn get_ground_truth_vector_ids(&self) -> HashSet<u64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock data source for testing
    struct MockDataSource {
        items: Vec<Vec<u8>>,
        item_len: usize,
    }

    impl MockDataSource {
        fn new(num_items: usize, item_len: usize) -> Self {
            let items: Vec<Vec<u8>> = (0..num_items)
                .map(|i| vec![i as u8; item_len])
                .collect();
            Self { items, item_len }
        }
    }

    impl DataSource for MockDataSource {
        fn num_items(&self) -> u64 {
            self.items.len() as u64
        }

        fn get_item_bytes(&self, idx: u64) -> &[u8] {
            &self.items[idx as usize]
        }

        fn item_byte_len(&self) -> usize {
            self.item_len
        }
    }

    #[test]
    fn test_mock_data_source() {
        let ds = MockDataSource::new(100, 32);
        assert_eq!(ds.num_items(), 100);
        assert_eq!(ds.item_byte_len(), 32);

        let bytes = ds.get_item_bytes(5);
        assert_eq!(bytes.len(), 32);
        assert!(bytes.iter().all(|&b| b == 5));
    }
}
