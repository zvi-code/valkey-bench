//! Memory-mapped binary dataset
//!
//! This module provides zero-copy access to vector datasets via memory mapping.
//! The dataset file is mapped into memory read-only, and vectors are accessed
//! directly from the mapped memory without any copying.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use super::header::{DatasetHeader, DistanceMetricId, DATASET_MAGIC, HEADER_SIZE};
use crate::utils::DatasetError;

/// Memory-mapped dataset context
///
/// Provides zero-copy access to vectors, queries, and ground truth data.
/// Thread-safe: safe to share via Arc (mmap is read-only).
pub struct DatasetContext {
    /// Memory-mapped file
    mmap: Mmap,
    /// Cached header values (avoid packed struct access in hot path)
    dim: usize,
    num_vectors: u64,
    num_queries: u64,
    num_neighbors: usize,
    vectors_offset: usize,
    queries_offset: usize,
    ground_truth_offset: usize,
    vec_byte_len: usize,
    distance_metric: DistanceMetricId,
}

impl DatasetContext {
    /// Open dataset file and memory map it
    ///
    /// # Arguments
    /// * `path` - Path to the binary dataset file
    ///
    /// # Returns
    /// * `Ok(DatasetContext)` on success
    /// * `Err(DatasetError)` if file cannot be opened, mapped, or has invalid header
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DatasetError> {
        let file = File::open(path.as_ref()).map_err(DatasetError::OpenFailed)?;

        // SAFETY: The file is opened read-only and we don't modify the memory
        let mmap = unsafe { Mmap::map(&file) }.map_err(DatasetError::OpenFailed)?;

        if mmap.len() < HEADER_SIZE {
            return Err(DatasetError::FileTooSmall {
                size: mmap.len() as u64,
                minimum: HEADER_SIZE as u64,
            });
        }

        // Parse header (packed struct, use unaligned read)
        // SAFETY: We've verified the file is at least HEADER_SIZE bytes
        let header: DatasetHeader =
            unsafe { std::ptr::read_unaligned(mmap.as_ptr() as *const DatasetHeader) };

        if header.magic != DATASET_MAGIC {
            return Err(DatasetError::InvalidMagic {
                expected: DATASET_MAGIC,
                actual: header.magic,
            });
        }

        // Validate version
        if header.version > 2 {
            return Err(DatasetError::UnsupportedVersion(header.version));
        }

        let vec_byte_len = header.vec_byte_len();

        // Validate file size
        let expected_min_size =
            header.vectors_offset as usize + (header.num_vectors as usize * vec_byte_len);
        if mmap.len() < expected_min_size {
            return Err(DatasetError::FileTooSmall {
                size: mmap.len() as u64,
                minimum: expected_min_size as u64,
            });
        }

        Ok(Self {
            mmap,
            dim: header.dim as usize,
            num_vectors: header.num_vectors,
            num_queries: header.num_queries,
            num_neighbors: header.num_neighbors as usize,
            vectors_offset: header.vectors_offset as usize,
            queries_offset: header.queries_offset as usize,
            ground_truth_offset: header.ground_truth_offset as usize,
            vec_byte_len,
            distance_metric: header.distance_metric_type(),
        })
    }

    // === Accessors ===

    /// Get vector dimension
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of database vectors
    #[inline(always)]
    pub fn num_vectors(&self) -> u64 {
        self.num_vectors
    }

    /// Get number of query vectors
    #[inline(always)]
    pub fn num_queries(&self) -> u64 {
        self.num_queries
    }

    /// Get number of ground truth neighbors per query
    #[inline(always)]
    pub fn num_neighbors(&self) -> usize {
        self.num_neighbors
    }

    /// Get vector byte length (dim * element_size)
    #[inline(always)]
    pub fn vec_byte_len(&self) -> usize {
        self.vec_byte_len
    }

    /// Get distance metric
    #[inline(always)]
    pub fn distance_metric(&self) -> DistanceMetricId {
        self.distance_metric
    }

    // === Zero-Copy Data Access ===

    /// Get raw bytes for vector at index (zero-copy)
    ///
    /// # Arguments
    /// * `idx` - Vector index (0-based)
    ///
    /// # Panics
    /// Panics in debug mode if idx >= num_vectors
    #[inline(always)]
    pub fn get_vector_bytes(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.num_vectors, "vector index out of bounds");
        let offset = self.vectors_offset + (idx as usize * self.vec_byte_len);
        &self.mmap[offset..offset + self.vec_byte_len]
    }

    /// Get raw bytes for query vector at index (zero-copy)
    ///
    /// # Arguments
    /// * `idx` - Query index (0-based)
    ///
    /// # Panics
    /// Panics in debug mode if idx >= num_queries
    #[inline(always)]
    pub fn get_query_bytes(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.num_queries, "query index out of bounds");
        let offset = self.queries_offset + (idx as usize * self.vec_byte_len);
        &self.mmap[offset..offset + self.vec_byte_len]
    }

    /// Get ground truth neighbor IDs for query (zero-copy)
    ///
    /// # Arguments
    /// * `query_idx` - Query index (0-based)
    ///
    /// # Returns
    /// Slice of ground truth neighbor IDs (sorted by distance)
    ///
    /// # Panics
    /// Panics in debug mode if query_idx >= num_queries
    #[inline]
    pub fn get_neighbor_ids(&self, query_idx: u64) -> &[u64] {
        debug_assert!(query_idx < self.num_queries, "query index out of bounds");
        let offset = self.ground_truth_offset
            + (query_idx as usize * self.num_neighbors * std::mem::size_of::<u64>());
        // SAFETY: Ground truth data is u64 aligned in the file format
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(offset) as *const u64,
                self.num_neighbors,
            )
        }
    }

    /// Get vector at index with bounds checking
    ///
    /// # Arguments
    /// * `idx` - Vector index (0-based)
    ///
    /// # Returns
    /// * `Some(&[u8])` if index is valid
    /// * `None` if index is out of bounds
    #[inline]
    pub fn try_get_vector_bytes(&self, idx: u64) -> Option<&[u8]> {
        if idx >= self.num_vectors {
            return None;
        }
        let offset = self.vectors_offset + (idx as usize * self.vec_byte_len);
        Some(&self.mmap[offset..offset + self.vec_byte_len])
    }

    /// Get query vector at index with bounds checking
    ///
    /// # Arguments
    /// * `idx` - Query index (0-based)
    ///
    /// # Returns
    /// * `Some(&[u8])` if index is valid
    /// * `None` if index is out of bounds
    #[inline]
    pub fn try_get_query_bytes(&self, idx: u64) -> Option<&[u8]> {
        if idx >= self.num_queries {
            return None;
        }
        let offset = self.queries_offset + (idx as usize * self.vec_byte_len);
        Some(&self.mmap[offset..offset + self.vec_byte_len])
    }

    // === Recall Computation ===

    /// Compute recall@k against ground truth
    ///
    /// # Arguments
    /// * `query_idx` - Query index (0-based)
    /// * `result_ids` - Vector IDs returned by search
    /// * `k` - Number of results to consider
    ///
    /// # Returns
    /// Recall value between 0.0 and 1.0
    pub fn compute_recall(&self, query_idx: u64, result_ids: &[u64], k: usize) -> f64 {
        let gt_ids = self.get_neighbor_ids(query_idx);
        let k = k.min(gt_ids.len()).min(result_ids.len());

        if k == 0 {
            return 0.0;
        }

        // Count matches using linear search (efficient for small k)
        let mut matches = 0usize;
        for &result_id in &result_ids[..k] {
            if gt_ids[..k].contains(&result_id) {
                matches += 1;
            }
        }

        matches as f64 / k as f64
    }

    /// Compute recall@k with custom k values
    ///
    /// # Arguments
    /// * `query_idx` - Query index (0-based)
    /// * `result_ids` - Vector IDs returned by search
    /// * `ks` - List of k values to compute recall for
    ///
    /// # Returns
    /// Vector of (k, recall) pairs
    pub fn compute_recalls_at_k(
        &self,
        query_idx: u64,
        result_ids: &[u64],
        ks: &[usize],
    ) -> Vec<(usize, f64)> {
        ks.iter()
            .map(|&k| (k, self.compute_recall(query_idx, result_ids, k)))
            .collect()
    }

    // === Statistics ===

    /// Get total memory mapped size in bytes
    pub fn mmap_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get dataset summary string
    pub fn summary(&self) -> String {
        format!(
            "Dataset: {} vectors, {} queries, dim={}, metric={}, vec_size={}B",
            self.num_vectors,
            self.num_queries,
            self.dim,
            self.distance_metric.as_str(),
            self.vec_byte_len
        )
    }
}

// Safe to share across threads (mmap is read-only)
// SAFETY: The mmap is created read-only and we never modify the data.
// All methods only read from the mapped memory.
unsafe impl Send for DatasetContext {}
unsafe impl Sync for DatasetContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_computation() {
        // Test recall computation logic without needing an actual file
        // This is a unit test for the recall algorithm

        // Ground truth: [0, 1, 2, 3, 4]
        // Results: [0, 2, 5, 1, 6]
        // At k=5: matches are 0, 2, 1 = 3 matches out of 5 = 0.6 recall

        let gt = vec![0u64, 1, 2, 3, 4];
        let results = vec![0u64, 2, 5, 1, 6];
        let k = 5;

        let mut matches = 0usize;
        for &result_id in &results[..k] {
            if gt[..k].contains(&result_id) {
                matches += 1;
            }
        }
        let recall = matches as f64 / k as f64;

        assert!((recall - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_recall_at_k1() {
        // At k=1, if first result is in ground truth, recall = 1.0
        let gt = vec![5u64, 1, 2, 3, 4];
        let results = vec![5u64, 9, 9, 9, 9];

        let mut matches = 0usize;
        let k = 1;
        for &result_id in &results[..k] {
            if gt[..k].contains(&result_id) {
                matches += 1;
            }
        }
        let recall = matches as f64 / k as f64;

        assert!((recall - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dataset_context_missing_file() {
        let result = DatasetContext::open("/nonexistent/file.bin");
        assert!(result.is_err());
    }
}
