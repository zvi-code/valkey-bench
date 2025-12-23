//! Schema-driven dataset context
//!
//! This module provides zero-copy access to datasets via memory mapping,
//! with structure defined by external YAML schema files. All field offsets
//! are computed from the schema at load time, enabling O(1) access.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use super::layout::{RecordLayout, SectionLayout};
use super::schema::DatasetSchema;
use super::source::{DataSource, VectorDataSource};
use crate::utils::DatasetError;

/// Schema-driven memory-mapped dataset context
///
/// Provides zero-copy access to records, queries, keys, and ground truth.
/// Thread-safe: safe to share via Arc (mmap is read-only).
pub struct DatasetContext {
    /// Memory-mapped data file
    mmap: Mmap,

    /// Schema loaded from YAML
    schema: DatasetSchema,

    /// Computed record layout (field offsets within record)
    record_layout: RecordLayout,

    /// Computed section layout (section offsets within file)
    section_layout: SectionLayout,

    /// Cached vector field info for fast access
    vector_field_offset: Option<usize>,
    vector_byte_len: Option<usize>,
    vector_dimensions: Option<u32>,
}

impl DatasetContext {
    /// Open dataset from schema YAML and data binary file
    ///
    /// # Arguments
    /// * `schema_path` - Path to the YAML schema file
    /// * `data_path` - Path to the binary data file
    ///
    /// # Returns
    /// * `Ok(DatasetContext)` on success
    /// * `Err(DatasetError)` if files cannot be opened or are invalid
    pub fn open<P1: AsRef<Path>, P2: AsRef<Path>>(
        schema_path: P1,
        data_path: P2,
    ) -> Result<Self, DatasetError> {
        // Load schema
        let schema = DatasetSchema::load(schema_path)
            .map_err(|e| DatasetError::Schema(e.to_string()))?;

        // Compute layouts
        let fields = schema
            .record
            .get_fields()
            .ok_or_else(|| DatasetError::InvalidSchema("No fields defined in schema".into()))?;
        let record_layout = RecordLayout::from_fields(fields);
        let section_layout = SectionLayout::compute(&schema, &record_layout);

        // Memory map data file
        let file = File::open(data_path.as_ref()).map_err(DatasetError::OpenFailed)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(DatasetError::OpenFailed)?;

        // Validate file size
        if mmap.len() < section_layout.total_size {
            return Err(DatasetError::FileTooSmall {
                size: mmap.len() as u64,
                minimum: section_layout.total_size as u64,
            });
        }

        // Cache vector field info for hot path
        let (vector_field_offset, vector_byte_len, vector_dimensions) =
            if let Some(vf) = record_layout.first_vector_field() {
                (Some(vf.offset), Some(vf.size), vf.dimensions)
            } else {
                (None, None, None)
            };

        Ok(Self {
            mmap,
            schema,
            record_layout,
            section_layout,
            vector_field_offset,
            vector_byte_len,
            vector_dimensions,
        })
    }

    // === Schema & Layout Access ===

    /// Get the schema
    #[inline]
    pub fn schema(&self) -> &DatasetSchema {
        &self.schema
    }

    /// Get the record layout
    #[inline]
    pub fn record_layout(&self) -> &RecordLayout {
        &self.record_layout
    }

    /// Get the section layout
    #[inline]
    pub fn section_layout(&self) -> &SectionLayout {
        &self.section_layout
    }

    // === Vector-Specific Accessors (for compatibility) ===

    /// Get vector dimension (for vector datasets)
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.vector_dimensions.unwrap_or(0) as usize
    }

    /// Get number of database vectors/records
    #[inline(always)]
    pub fn num_vectors(&self) -> u64 {
        self.section_layout.record_count
    }

    /// Alias for num_vectors
    #[inline(always)]
    pub fn num_records(&self) -> u64 {
        self.section_layout.record_count
    }

    /// Get number of query vectors
    #[inline(always)]
    pub fn num_queries(&self) -> u64 {
        self.section_layout.query_count
    }

    /// Get number of ground truth neighbors per query
    #[inline(always)]
    pub fn num_neighbors(&self) -> usize {
        self.section_layout.neighbors_per_query
    }

    /// Get vector byte length
    #[inline(always)]
    pub fn vec_byte_len(&self) -> usize {
        self.vector_byte_len.unwrap_or(0)
    }

    /// Get record byte length
    #[inline(always)]
    pub fn record_byte_len(&self) -> usize {
        self.section_layout.record_size
    }

    // === Zero-Copy Record Access ===

    /// Get raw bytes for entire record at index
    #[inline(always)]
    pub fn get_record_bytes(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.section_layout.record_count, "record index out of bounds");
        let offset = self.section_layout.records_offset + (idx as usize * self.section_layout.record_size);
        &self.mmap[offset..offset + self.section_layout.record_size]
    }

    /// Get raw bytes for a specific field in a record
    #[inline]
    pub fn get_field_bytes(&self, record_idx: u64, field_name: &str) -> Option<&[u8]> {
        let field = self.record_layout.field(field_name)?;
        let record = self.get_record_bytes(record_idx);

        if field.is_variable {
            // Variable-length: read u32 length prefix, then data
            let len_bytes = &record[field.offset..field.offset + 4];
            let len = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
            let data_start = field.offset + field.data_offset;
            Some(&record[data_start..data_start + len])
        } else {
            // Fixed-length: direct access
            Some(&record[field.offset..field.offset + field.size])
        }
    }

    /// Get raw bytes for vector field at record index (zero-copy)
    ///
    /// This is optimized for the common case of accessing the first vector field.
    #[inline(always)]
    pub fn get_vector_bytes(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.section_layout.record_count, "vector index out of bounds");

        let vec_offset = self.vector_field_offset.expect("no vector field in schema");
        let vec_len = self.vector_byte_len.expect("no vector field in schema");

        let record_offset = self.section_layout.records_offset + (idx as usize * self.section_layout.record_size);
        let offset = record_offset + vec_offset;
        &self.mmap[offset..offset + vec_len]
    }

    /// Get raw bytes for query vector at index (zero-copy)
    #[inline(always)]
    pub fn get_query_bytes(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.section_layout.query_count, "query index out of bounds");

        let queries_offset = self.section_layout.queries_offset.expect("no queries section");
        let query_size = self.section_layout.query_record_size;
        let offset = queries_offset + (idx as usize * query_size);

        // If query has same fields as record, extract vector field
        // Otherwise return full query record
        if let Some(_vec_offset) = self.vector_field_offset {
            let vec_len = self.vector_byte_len.unwrap();
            // Find vector field offset within query record
            // For now, assume query contains vector at start if query_fields subset
            if self.section_layout.query_field_indices.contains(&self.record_layout.first_vector_idx.unwrap_or(usize::MAX)) {
                // Calculate offset within query record
                let query_vec_offset = self.query_vector_offset();
                &self.mmap[offset + query_vec_offset..offset + query_vec_offset + vec_len]
            } else {
                &self.mmap[offset..offset + query_size]
            }
        } else {
            &self.mmap[offset..offset + query_size]
        }
    }

    /// Calculate vector field offset within query record
    fn query_vector_offset(&self) -> usize {
        let vec_idx = self.record_layout.first_vector_idx.unwrap_or(0);
        let mut offset = 0;
        for &idx in &self.section_layout.query_field_indices {
            if idx == vec_idx {
                return offset;
            }
            offset += self.record_layout.fields[idx].size;
        }
        0
    }

    // === Key Access ===

    /// Get key for record at index
    ///
    /// Returns either a generated key from pattern or key from file.
    pub fn get_key(&self, idx: u64) -> String {
        if let Some(ref pattern) = self.schema.sections.keys.pattern {
            // Generate key from pattern
            pattern.replace("{id}", &idx.to_string())
        } else if let Some(keys_offset) = self.section_layout.keys_offset {
            // Read key from file
            let entry_size = self.section_layout.keys_entry_size.unwrap();
            let offset = keys_offset + (idx as usize * entry_size);

            // Check if variable length (has u32 prefix)
            if self.schema.sections.keys.length == Some(super::schema::LengthSpec::Variable) {
                let len_bytes = &self.mmap[offset..offset + 4];
                let len = u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
                let data = &self.mmap[offset + 4..offset + 4 + len];
                String::from_utf8_lossy(data).into_owned()
            } else {
                // Fixed length, trim trailing nulls
                let data = &self.mmap[offset..offset + entry_size];
                let end = data.iter().position(|&b| b == 0).unwrap_or(entry_size);
                String::from_utf8_lossy(&data[..end]).into_owned()
            }
        } else {
            // Default pattern
            format!("key:{}", idx)
        }
    }

    // === Ground Truth Access ===

    /// Get ground truth neighbor IDs for query (zero-copy)
    #[inline]
    pub fn get_neighbor_ids(&self, query_idx: u64) -> &[u64] {
        debug_assert!(query_idx < self.section_layout.query_count, "query index out of bounds");

        let gt_offset = self.section_layout.ground_truth_offset
            .expect("no ground truth in dataset");
        let neighbors = self.section_layout.neighbors_per_query;
        let id_size = self.section_layout.gt_id_size;

        if id_size == 8 {
            // u64 IDs - direct slice
            let offset = gt_offset + (query_idx as usize * neighbors * 8);
            unsafe {
                std::slice::from_raw_parts(
                    self.mmap.as_ptr().add(offset) as *const u64,
                    neighbors,
                )
            }
        } else {
            // This branch is less common; for u32 IDs we'd need conversion
            // For now, panic if u32 ground truth is used
            panic!("u32 ground truth IDs require conversion - use get_neighbor_ids_u32");
        }
    }

    /// Get ground truth neighbor IDs as u32 (for datasets with u32 IDs)
    #[inline]
    pub fn get_neighbor_ids_u32(&self, query_idx: u64) -> &[u32] {
        debug_assert!(query_idx < self.section_layout.query_count, "query index out of bounds");

        let gt_offset = self.section_layout.ground_truth_offset
            .expect("no ground truth in dataset");
        let neighbors = self.section_layout.neighbors_per_query;

        let offset = gt_offset + (query_idx as usize * neighbors * 4);
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(offset) as *const u32,
                neighbors,
            )
        }
    }

    // === Recall Computation ===

    /// Compute recall@k against ground truth
    pub fn compute_recall(&self, query_idx: u64, result_ids: &[u64], k: usize) -> f64 {
        if self.section_layout.ground_truth_offset.is_none() {
            return 0.0;
        }

        let gt_ids = if self.section_layout.gt_id_size == 8 {
            self.get_neighbor_ids(query_idx)
        } else {
            // Convert u32 to u64 for comparison
            // This is slower but handles the u32 case
            let u32_ids = self.get_neighbor_ids_u32(query_idx);
            // Create temporary - not ideal but maintains API
            let converted: Vec<u64> = u32_ids.iter().map(|&id| id as u64).collect();
            return self.compute_recall_with_gt(result_ids, &converted, k);
        };

        self.compute_recall_with_gt(result_ids, gt_ids, k)
    }

    fn compute_recall_with_gt(&self, result_ids: &[u64], gt_ids: &[u64], k: usize) -> f64 {
        let k = k.min(gt_ids.len()).min(result_ids.len());
        if k == 0 {
            return 0.0;
        }

        let mut matches = 0usize;
        for &result_id in &result_ids[..k] {
            if gt_ids[..k].contains(&result_id) {
                matches += 1;
            }
        }

        matches as f64 / k as f64
    }

    /// Get all unique vector IDs that appear in ground truth
    pub fn get_ground_truth_vector_ids(&self) -> HashSet<u64> {
        let mut ids = HashSet::new();

        if self.section_layout.ground_truth_offset.is_none() {
            return ids;
        }

        for query_idx in 0..self.section_layout.query_count {
            if self.section_layout.gt_id_size == 8 {
                let neighbors = self.get_neighbor_ids(query_idx);
                for &id in neighbors {
                    ids.insert(id);
                }
            } else {
                let neighbors = self.get_neighbor_ids_u32(query_idx);
                for &id in neighbors {
                    ids.insert(id as u64);
                }
            }
        }

        ids
    }

    /// Get ground truth vector IDs as sorted vector
    pub fn get_ground_truth_vector_ids_sorted(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.get_ground_truth_vector_ids().into_iter().collect();
        ids.sort_unstable();
        ids
    }

    // === Statistics ===

    /// Get total memory mapped size in bytes
    pub fn mmap_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get dataset summary string
    pub fn summary(&self) -> String {
        let name = self.schema.metadata.name.as_deref().unwrap_or("unnamed");

        if let Some(dim) = self.vector_dimensions {
            format!(
                "Dataset '{}': {} records, {} queries, dim={}, record_size={}B",
                name,
                self.section_layout.record_count,
                self.section_layout.query_count,
                dim,
                self.section_layout.record_size
            )
        } else {
            format!(
                "Dataset '{}': {} records, {} queries, record_size={}B",
                name,
                self.section_layout.record_count,
                self.section_layout.query_count,
                self.section_layout.record_size
            )
        }
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl DataSource for DatasetContext {
    fn num_items(&self) -> u64 {
        self.section_layout.record_count
    }

    fn get_item_bytes(&self, idx: u64) -> &[u8] {
        self.get_record_bytes(idx)
    }

    fn item_byte_len(&self) -> usize {
        self.section_layout.record_size
    }
}

impl VectorDataSource for DatasetContext {
    fn num_queries(&self) -> u64 {
        self.section_layout.query_count
    }

    fn get_query_bytes(&self, idx: u64) -> &[u8] {
        DatasetContext::get_query_bytes(self, idx)
    }

    fn compute_recall(&self, query_idx: u64, result_ids: &[u64], k: usize) -> f64 {
        DatasetContext::compute_recall(self, query_idx, result_ids, k)
    }

    fn get_ground_truth_vector_ids(&self) -> HashSet<u64> {
        DatasetContext::get_ground_truth_vector_ids(self)
    }
}

// Safe to share across threads (mmap is read-only)
unsafe impl Send for DatasetContext {}
unsafe impl Sync for DatasetContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_computation() {
        // Test recall computation logic
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
        let gt = vec![5u64, 1, 2, 3, 4];
        let results = vec![5u64, 9, 9, 9, 9];
        let k = 1;

        let mut matches = 0usize;
        for &result_id in &results[..k] {
            if gt[..k].contains(&result_id) {
                matches += 1;
            }
        }
        let recall = matches as f64 / k as f64;

        assert!((recall - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_key_pattern_generation() {
        // Test key pattern replacement
        let pattern = "vec:{id}:data";
        let key = pattern.replace("{id}", "12345");
        assert_eq!(key, "vec:12345:data");
    }
}
