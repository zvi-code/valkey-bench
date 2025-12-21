//! Workload context trait and implementations
//!
//! This module provides the WorkloadContext trait that abstracts workload-specific
//! behavior from the generic EventWorker infrastructure. This decoupling enables:
//! - Easy addition of new workload types
//! - Workload-specific metrics (e.g., recall for vector search)
//! - Workload-specific ID claiming logic (e.g., skipping existing vectors)
//! - Workload-specific placeholder filling (e.g., tag generation)

use std::sync::Arc;

use crate::benchmark::{GlobalCounters, RecallStats};
use crate::client::PlaceholderType;
use crate::cluster::{ClusterTagMap, ProtectedVectorIds};
use crate::dataset::DatasetContext;
use crate::utils::RespValue;
use crate::workload::{extract_numeric_ids, parse_search_response, NumericFieldSet, TagDistributionSet, WorkloadType};

/// Metrics collected by workload context
#[derive(Debug, Default)]
pub enum WorkloadMetrics {
    #[default]
    None,
    Recall(RecallStats),
}

/// Trait for workload-specific context and behavior
///
/// Implementations encapsulate workload-specific logic like:
/// - ID claiming (with tag_map for partial prefill, protected_ids for deletion)
/// - Response processing (recall computation)
/// - Placeholder filling for workload-specific data (e.g., tag generation)
pub trait WorkloadContext: Send {
    /// Claim the next key/item ID for this workload
    ///
    /// Returns None when no more IDs are available (e.g., all vectors loaded,
    /// all deleteable vectors claimed).
    fn claim_next_id(&self, counters: &GlobalCounters) -> Option<u64>;

    /// Get the dataset index for vector operations
    /// Returns None for non-vector workloads
    fn next_dataset_idx(&self, counters: &GlobalCounters) -> Option<u64>;

    /// Get query vector bytes for FT.SEARCH operations
    /// Returns None for non-query workloads
    fn get_query_bytes(&self, idx: u64) -> Option<&[u8]>;

    /// Get vector bytes for HSET operations
    /// Returns None for non-vector workloads
    fn get_vector_bytes(&self, idx: u64) -> Option<&[u8]>;

    /// Fill a Tag placeholder with generated tag values
    ///
    /// Uses key_num as seed for reproducible tag generation.
    /// Buffer is filled with tag values or padding (commas) if no tags configured.
    fn fill_tag_placeholder(&self, _key_num: u64, buf: &mut [u8]) {
        // Default: fill with commas (no-op for workloads without tag support)
        buf.fill(b',');
    }

    /// Fill a NumericField placeholder with generated values
    ///
    /// Uses key_num as seed for reproducible value generation.
    /// The field_idx identifies which NumericFieldConfig to use.
    fn fill_numeric_field(&self, field_idx: usize, key_num: u64, seq_counter: u64, buf: &mut [u8]) {
        // Default: fill with zeros (no-op for workloads without numeric field support)
        let _ = (field_idx, key_num, seq_counter);
        buf.fill(b'0');
    }

    /// Compute and record recall for a query response
    /// Called after receiving a response with the corresponding query index
    fn compute_and_record_recall(&mut self, query_idx: u64, response: &RespValue);

    /// Take accumulated metrics (consumes the internal state)
    fn take_metrics(&mut self) -> WorkloadMetrics;

    /// Get number of items in dataset (for modulo operations)
    fn num_items(&self) -> u64;

    /// Get number of queries in dataset
    fn num_queries(&self) -> u64;

    /// Check if this workload uses dataset-based keys
    /// When true, the Key placeholder is filled by the Vector handler
    fn uses_dataset_keys(&self) -> bool;

    /// Check if this workload's key is the claimed ID (for delete operations)
    fn key_is_claimed_id(&self) -> bool;
}

// =============================================================================
// SimpleContext - For key-value workloads (SET, GET, INCR, etc.)
// =============================================================================

/// Context for simple key-value workloads without dataset or recall
pub struct SimpleContext {
    keyspace_len: u64,
    sequential: bool,
    seed: u64,
}

impl SimpleContext {
    pub fn new(keyspace_len: u64, sequential: bool, seed: u64) -> Self {
        Self {
            keyspace_len,
            sequential,
            seed,
        }
    }
}

impl WorkloadContext for SimpleContext {
    fn claim_next_id(&self, counters: &GlobalCounters) -> Option<u64> {
        if self.sequential {
            Some(counters.next_seq_key(self.keyspace_len))
        } else {
            Some(counters.next_random_key(self.seed, self.keyspace_len))
        }
    }

    fn next_dataset_idx(&self, _counters: &GlobalCounters) -> Option<u64> {
        None
    }

    fn get_query_bytes(&self, _idx: u64) -> Option<&[u8]> {
        None
    }

    fn get_vector_bytes(&self, _idx: u64) -> Option<&[u8]> {
        None
    }

    fn compute_and_record_recall(&mut self, _query_idx: u64, _response: &RespValue) {
        // No-op for simple workloads
    }

    fn take_metrics(&mut self) -> WorkloadMetrics {
        WorkloadMetrics::None
    }

    fn num_items(&self) -> u64 {
        self.keyspace_len
    }

    fn num_queries(&self) -> u64 {
        0
    }

    fn uses_dataset_keys(&self) -> bool {
        false
    }

    fn key_is_claimed_id(&self) -> bool {
        false
    }
}

// =============================================================================
// VectorLoadContext - For VecLoad with partial prefill support
// =============================================================================

/// Context for vector loading workloads (HSET with vectors)
pub struct VectorLoadContext {
    dataset: Arc<DatasetContext>,
    tag_map: Option<Arc<ClusterTagMap>>,
    tag_distributions: Option<TagDistributionSet>,
    numeric_fields: NumericFieldSet,
}

impl VectorLoadContext {
    pub fn new(
        dataset: Arc<DatasetContext>,
        tag_map: Option<Arc<ClusterTagMap>>,
        tag_distributions: Option<TagDistributionSet>,
        numeric_fields: NumericFieldSet,
    ) -> Self {
        Self {
            dataset,
            tag_map,
            tag_distributions,
            numeric_fields,
        }
    }
}

impl WorkloadContext for VectorLoadContext {
    fn claim_next_id(&self, counters: &GlobalCounters) -> Option<u64> {
        // For VecLoad, use tag_map to skip existing vectors if available
        if let Some(ref tm) = self.tag_map {
            tm.claim_unmapped_id(self.dataset.num_vectors())
        } else {
            Some(counters.next_dataset_idx() % self.dataset.num_vectors())
        }
    }

    fn next_dataset_idx(&self, counters: &GlobalCounters) -> Option<u64> {
        self.claim_next_id(counters)
    }

    fn get_query_bytes(&self, _idx: u64) -> Option<&[u8]> {
        None
    }

    fn get_vector_bytes(&self, idx: u64) -> Option<&[u8]> {
        Some(self.dataset.get_vector_bytes(idx))
    }

    fn fill_tag_placeholder(&self, key_num: u64, buf: &mut [u8]) {
        if let Some(ref tag_dist) = self.tag_distributions {
            if let Some(tags) = tag_dist.select_tags_seeded(key_num) {
                let tag_bytes = tags.as_bytes();
                let copy_len = tag_bytes.len().min(buf.len());
                buf[..copy_len].copy_from_slice(&tag_bytes[..copy_len]);
                buf[copy_len..].fill(b',');
            } else {
                buf.fill(b',');
            }
        } else {
            buf.fill(b',');
        }
    }

    fn fill_numeric_field(&self, field_idx: usize, key_num: u64, seq_counter: u64, buf: &mut [u8]) {
        if let Some(field_config) = self.numeric_fields.get(field_idx) {
            field_config.fill_buffer(key_num, seq_counter, buf);
        } else {
            buf.fill(b'0');
        }
    }

    fn compute_and_record_recall(&mut self, _query_idx: u64, _response: &RespValue) {
        // No-op for load workloads
    }

    fn take_metrics(&mut self) -> WorkloadMetrics {
        WorkloadMetrics::None
    }

    fn num_items(&self) -> u64 {
        self.dataset.num_vectors()
    }

    fn num_queries(&self) -> u64 {
        0
    }

    fn uses_dataset_keys(&self) -> bool {
        true // Key is the vector ID from dataset
    }

    fn key_is_claimed_id(&self) -> bool {
        false // Key is set by Vector handler, not directly from claim_next_id
    }
}

// =============================================================================
// VectorQueryContext - For VecQuery with recall computation
// =============================================================================

/// Context for vector query workloads (FT.SEARCH) with recall tracking
pub struct VectorQueryContext {
    dataset: Arc<DatasetContext>,
    recall_stats: RecallStats,
    k: usize,
    key_prefix: String,
}

impl VectorQueryContext {
    pub fn new(dataset: Arc<DatasetContext>, k: usize, key_prefix: String) -> Self {
        Self {
            dataset,
            recall_stats: RecallStats::new(),
            k,
            key_prefix,
        }
    }
}

impl WorkloadContext for VectorQueryContext {
    fn claim_next_id(&self, counters: &GlobalCounters) -> Option<u64> {
        Some(counters.next_query_idx(self.dataset.num_queries()))
    }

    fn next_dataset_idx(&self, _counters: &GlobalCounters) -> Option<u64> {
        None // Queries don't use dataset vectors
    }

    fn get_query_bytes(&self, idx: u64) -> Option<&[u8]> {
        Some(self.dataset.get_query_bytes(idx))
    }

    fn get_vector_bytes(&self, _idx: u64) -> Option<&[u8]> {
        None
    }

    fn compute_and_record_recall(&mut self, query_idx: u64, response: &RespValue) {
        let doc_ids = parse_search_response(response);
        let result_ids = extract_numeric_ids(&doc_ids, &self.key_prefix);
        let recall = self.dataset.compute_recall(query_idx, &result_ids, self.k);
        self.recall_stats.record(recall);
    }

    fn take_metrics(&mut self) -> WorkloadMetrics {
        let stats = std::mem::take(&mut self.recall_stats);
        // Re-initialize for next batch
        self.recall_stats = RecallStats::new();
        WorkloadMetrics::Recall(stats)
    }

    fn num_items(&self) -> u64 {
        self.dataset.num_vectors()
    }

    fn num_queries(&self) -> u64 {
        self.dataset.num_queries()
    }

    fn uses_dataset_keys(&self) -> bool {
        true // Query uses dataset queries
    }

    fn key_is_claimed_id(&self) -> bool {
        false
    }
}

// =============================================================================
// VectorDeleteContext - For VecDelete with ground truth protection
// =============================================================================

/// Context for vector deletion workloads (DEL) with ground truth protection
pub struct VectorDeleteContext {
    dataset: Arc<DatasetContext>,
    protected_ids: Option<Arc<ProtectedVectorIds>>,
}

impl VectorDeleteContext {
    pub fn new(dataset: Arc<DatasetContext>, protected_ids: Option<Arc<ProtectedVectorIds>>) -> Self {
        Self {
            dataset,
            protected_ids,
        }
    }
}

impl WorkloadContext for VectorDeleteContext {
    fn claim_next_id(&self, counters: &GlobalCounters) -> Option<u64> {
        // Use protected_ids to skip ground truth vectors
        if let Some(ref pids) = self.protected_ids {
            pids.claim_deleteable_id()
        } else {
            Some(counters.next_dataset_idx() % self.dataset.num_vectors())
        }
    }

    fn next_dataset_idx(&self, counters: &GlobalCounters) -> Option<u64> {
        self.claim_next_id(counters)
    }

    fn get_query_bytes(&self, _idx: u64) -> Option<&[u8]> {
        None
    }

    fn get_vector_bytes(&self, _idx: u64) -> Option<&[u8]> {
        None // Delete doesn't need vector bytes
    }

    fn compute_and_record_recall(&mut self, _query_idx: u64, _response: &RespValue) {
        // No-op for delete workloads
    }

    fn take_metrics(&mut self) -> WorkloadMetrics {
        WorkloadMetrics::None
    }

    fn num_items(&self) -> u64 {
        self.dataset.num_vectors()
    }

    fn num_queries(&self) -> u64 {
        0
    }

    fn uses_dataset_keys(&self) -> bool {
        true // Key is the vector ID to delete
    }

    fn key_is_claimed_id(&self) -> bool {
        true // Key is directly the claimed ID (vector ID to delete)
    }
}

// =============================================================================
// VectorUpdateContext - For VecUpdate (similar to VecLoad but updates existing)
// =============================================================================

/// Context for vector update workloads (HSET updating existing vectors)
pub struct VectorUpdateContext {
    dataset: Arc<DatasetContext>,
    tag_distributions: Option<TagDistributionSet>,
    numeric_fields: NumericFieldSet,
}

impl VectorUpdateContext {
    pub fn new(
        dataset: Arc<DatasetContext>,
        tag_distributions: Option<TagDistributionSet>,
        numeric_fields: NumericFieldSet,
    ) -> Self {
        Self {
            dataset,
            tag_distributions,
            numeric_fields,
        }
    }
}

impl WorkloadContext for VectorUpdateContext {
    fn claim_next_id(&self, counters: &GlobalCounters) -> Option<u64> {
        Some(counters.next_dataset_idx() % self.dataset.num_vectors())
    }

    fn next_dataset_idx(&self, counters: &GlobalCounters) -> Option<u64> {
        self.claim_next_id(counters)
    }

    fn get_query_bytes(&self, _idx: u64) -> Option<&[u8]> {
        None
    }

    fn get_vector_bytes(&self, idx: u64) -> Option<&[u8]> {
        Some(self.dataset.get_vector_bytes(idx))
    }

    fn fill_tag_placeholder(&self, key_num: u64, buf: &mut [u8]) {
        if let Some(ref tag_dist) = self.tag_distributions {
            if let Some(tags) = tag_dist.select_tags_seeded(key_num) {
                let tag_bytes = tags.as_bytes();
                let copy_len = tag_bytes.len().min(buf.len());
                buf[..copy_len].copy_from_slice(&tag_bytes[..copy_len]);
                buf[copy_len..].fill(b',');
            } else {
                buf.fill(b',');
            }
        } else {
            buf.fill(b',');
        }
    }

    fn fill_numeric_field(&self, field_idx: usize, key_num: u64, seq_counter: u64, buf: &mut [u8]) {
        if let Some(field_config) = self.numeric_fields.get(field_idx) {
            field_config.fill_buffer(key_num, seq_counter, buf);
        } else {
            buf.fill(b'0');
        }
    }

    fn compute_and_record_recall(&mut self, _query_idx: u64, _response: &RespValue) {
        // No-op for update workloads
    }

    fn take_metrics(&mut self) -> WorkloadMetrics {
        WorkloadMetrics::None
    }

    fn num_items(&self) -> u64 {
        self.dataset.num_vectors()
    }

    fn num_queries(&self) -> u64 {
        0
    }

    fn uses_dataset_keys(&self) -> bool {
        true // Key is the vector ID from dataset
    }

    fn key_is_claimed_id(&self) -> bool {
        false // Key is set by Vector handler
    }
}

// =============================================================================
// Factory function
// =============================================================================

/// Create appropriate WorkloadContext for the given workload type
pub fn create_workload_context(
    workload_type: WorkloadType,
    dataset: Option<Arc<DatasetContext>>,
    tag_map: Option<Arc<ClusterTagMap>>,
    protected_ids: Option<Arc<ProtectedVectorIds>>,
    tag_distributions: Option<TagDistributionSet>,
    numeric_fields: NumericFieldSet,
    keyspace_len: u64,
    sequential: bool,
    seed: u64,
    k: usize,
    key_prefix: &str,
) -> Box<dyn WorkloadContext> {
    match workload_type {
        WorkloadType::VecLoad => {
            let ds = dataset.expect("VecLoad requires dataset");
            Box::new(VectorLoadContext::new(ds, tag_map, tag_distributions, numeric_fields))
        }
        WorkloadType::VecQuery => {
            let ds = dataset.expect("VecQuery requires dataset");
            Box::new(VectorQueryContext::new(ds, k, key_prefix.to_string()))
        }
        WorkloadType::VecDelete => {
            let ds = dataset.expect("VecDelete requires dataset");
            Box::new(VectorDeleteContext::new(ds, protected_ids))
        }
        WorkloadType::VecUpdate => {
            let ds = dataset.expect("VecUpdate requires dataset");
            Box::new(VectorUpdateContext::new(ds, tag_distributions, numeric_fields))
        }
        _ => {
            // All other workloads use simple key-value context
            Box::new(SimpleContext::new(keyspace_len, sequential, seed))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_context_sequential() {
        let ctx = SimpleContext::new(100, true, 0);
        let counters = GlobalCounters::new();

        // Sequential should return 0, 1, 2, ...
        assert_eq!(ctx.claim_next_id(&counters), Some(0));
        assert_eq!(ctx.claim_next_id(&counters), Some(1));
        assert_eq!(ctx.claim_next_id(&counters), Some(2));
    }

    #[test]
    fn test_workload_metrics_default() {
        let metrics = WorkloadMetrics::default();
        assert!(matches!(metrics, WorkloadMetrics::None));
    }

    #[test]
    fn test_simple_context_fill_tag_default() {
        // SimpleContext uses default fill_tag_placeholder which fills with commas
        let ctx = SimpleContext::new(100, true, 0);
        let mut buf = [0u8; 10];
        ctx.fill_tag_placeholder(42, &mut buf);
        assert!(buf.iter().all(|&b| b == b','));
    }

    #[test]
    fn test_tag_distribution_fill_logic() {
        // Test the tag distribution fill logic directly
        let tag_dist = TagDistributionSet::parse("test:100").unwrap();
        let mut buf = [0u8; 10];

        // Simulate what fill_tag_placeholder does
        if let Some(tags) = tag_dist.select_tags_seeded(42) {
            let tag_bytes = tags.as_bytes();
            let copy_len = tag_bytes.len().min(buf.len());
            buf[..copy_len].copy_from_slice(&tag_bytes[..copy_len]);
            buf[copy_len..].fill(b',');
        } else {
            buf.fill(b',');
        }

        // Should have "test" followed by commas
        assert_eq!(&buf[0..4], b"test");
        assert!(buf[4..].iter().all(|&b| b == b','));
    }

    #[test]
    fn test_tag_distribution_deterministic() {
        // Same key_num should produce same tags
        let tag_dist = TagDistributionSet::parse("a:50,b:50").unwrap();

        let tags1 = tag_dist.select_tags_seeded(12345);
        let tags2 = tag_dist.select_tags_seeded(12345);

        assert_eq!(tags1, tags2);
    }

    #[test]
    fn test_tag_distribution_different_keys() {
        // Different key_num can produce different tags
        let tag_dist = TagDistributionSet::parse("a:50,b:50").unwrap();

        // Run many times - at least some should be different
        let mut same_count = 0;
        for i in 0..100 {
            let tags1 = tag_dist.select_tags_seeded(i);
            let tags2 = tag_dist.select_tags_seeded(i + 1000);
            if tags1 == tags2 {
                same_count += 1;
            }
        }
        // With 50% probability for each of 2 tags, we expect significant variation
        assert!(same_count < 90, "Tags should vary across different keys");
    }
}
