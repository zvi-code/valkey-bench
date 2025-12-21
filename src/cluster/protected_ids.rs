//! Protected Vector IDs
//!
//! Manages a set of vector IDs that should be skipped during deletion benchmarks.
//! These are typically ground truth vector IDs that must remain in the cluster
//! for valid recall computation during vec-query operations.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe iterator for claiming deleteable (non-protected) vector IDs
///
/// This structure allows multiple worker threads to atomically claim vector IDs
/// for deletion while skipping those marked as protected (ground truth vectors).
pub struct ProtectedVectorIds {
    /// Set of protected vector IDs (ground truth neighbors)
    protected: HashSet<u64>,
    /// Atomic counter for claiming the next deleteable ID
    delete_counter: AtomicU64,
    /// Maximum vector ID in the dataset
    max_id: u64,
    /// Number of protected IDs
    protected_count: u64,
}

impl ProtectedVectorIds {
    /// Create a new ProtectedVectorIds from a set of protected IDs
    ///
    /// # Arguments
    /// * `protected` - HashSet of vector IDs that should not be deleted
    /// * `max_id` - Maximum vector ID in the dataset (exclusive)
    pub fn new(protected: HashSet<u64>, max_id: u64) -> Self {
        let protected_count = protected.len() as u64;
        Self {
            protected,
            delete_counter: AtomicU64::new(0),
            max_id,
            protected_count,
        }
    }

    /// Create from a sorted vector of protected IDs
    pub fn from_sorted_vec(sorted_ids: Vec<u64>, max_id: u64) -> Self {
        let protected: HashSet<u64> = sorted_ids.into_iter().collect();
        Self::new(protected, max_id)
    }

    /// Check if a vector ID is protected
    #[inline]
    pub fn is_protected(&self, id: u64) -> bool {
        self.protected.contains(&id)
    }

    /// Claim the next deleteable (non-protected) vector ID
    ///
    /// Atomically finds and claims the next vector ID that is not protected.
    /// Returns None when all deleteable vectors have been claimed.
    pub fn claim_deleteable_id(&self) -> Option<u64> {
        loop {
            let candidate = self.delete_counter.fetch_add(1, Ordering::Relaxed);
            if candidate >= self.max_id {
                return None; // All vectors processed
            }
            if !self.is_protected(candidate) {
                return Some(candidate);
            }
            // Vector is protected, try next one
        }
    }

    /// Get number of protected vector IDs
    pub fn protected_count(&self) -> u64 {
        self.protected_count
    }

    /// Get number of deleteable vector IDs
    pub fn deleteable_count(&self) -> u64 {
        self.max_id.saturating_sub(self.protected_count)
    }

    /// Get maximum vector ID
    pub fn max_id(&self) -> u64 {
        self.max_id
    }

    /// Get current counter value (for progress tracking)
    pub fn counter_value(&self) -> u64 {
        self.delete_counter.load(Ordering::Relaxed)
    }

    /// Reset the counter (call before starting a new deletion benchmark)
    pub fn reset_counter(&self) {
        self.delete_counter.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protected_ids_basic() {
        let protected: HashSet<u64> = vec![1, 3, 5].into_iter().collect();
        let pids = ProtectedVectorIds::new(protected, 10);

        assert!(pids.is_protected(1));
        assert!(pids.is_protected(3));
        assert!(pids.is_protected(5));
        assert!(!pids.is_protected(0));
        assert!(!pids.is_protected(2));
        assert!(!pids.is_protected(4));
    }

    #[test]
    fn test_claim_deleteable_skips_protected() {
        let protected: HashSet<u64> = vec![1, 3, 5].into_iter().collect();
        let pids = ProtectedVectorIds::new(protected, 10);

        // Should get 0, 2, 4, 6, 7, 8, 9 (skipping 1, 3, 5)
        assert_eq!(pids.claim_deleteable_id(), Some(0));
        assert_eq!(pids.claim_deleteable_id(), Some(2));
        assert_eq!(pids.claim_deleteable_id(), Some(4));
        assert_eq!(pids.claim_deleteable_id(), Some(6));
        assert_eq!(pids.claim_deleteable_id(), Some(7));
        assert_eq!(pids.claim_deleteable_id(), Some(8));
        assert_eq!(pids.claim_deleteable_id(), Some(9));
        assert_eq!(pids.claim_deleteable_id(), None);
    }

    #[test]
    fn test_counts() {
        let protected: HashSet<u64> = vec![1, 3, 5].into_iter().collect();
        let pids = ProtectedVectorIds::new(protected, 10);

        assert_eq!(pids.protected_count(), 3);
        assert_eq!(pids.deleteable_count(), 7);
        assert_eq!(pids.max_id(), 10);
    }

    #[test]
    fn test_reset_counter() {
        let protected: HashSet<u64> = vec![5].into_iter().collect();
        let pids = ProtectedVectorIds::new(protected, 10);

        // Claim some IDs
        assert_eq!(pids.claim_deleteable_id(), Some(0));
        assert_eq!(pids.claim_deleteable_id(), Some(1));
        assert_eq!(pids.counter_value(), 2);

        // Reset and start over
        pids.reset_counter();
        assert_eq!(pids.counter_value(), 0);
        assert_eq!(pids.claim_deleteable_id(), Some(0));
    }
}
