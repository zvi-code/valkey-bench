//! Global atomic counters for thread synchronization
//!
//! These are the ONLY synchronization points between worker threads.
//! All other state is thread-local.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Global counters shared between all worker threads
///
/// Design principle: Minimize contention by using relaxed ordering
/// where possible and keeping counter operations simple (fetch_add).
pub struct GlobalCounters {
    /// Total requests issued (claimed by workers)
    pub requests_issued: AtomicU64,

    /// Total requests completed (responses received)
    pub requests_finished: AtomicU64,

    /// Sequential key counter (for --sequential mode)
    pub seq_key_counter: AtomicU64,

    /// Random key counter (for deterministic random key generation)
    /// Each call returns a unique index that maps deterministically to a key
    pub random_key_counter: AtomicU64,

    /// Dataset vector counter (for unique vector insertion)
    pub dataset_counter: AtomicU64,

    /// Query vector counter (for sequential query iteration)
    pub query_counter: AtomicU64,

    /// Total errors encountered
    pub error_count: AtomicU64,

    /// Shutdown signal
    pub shutdown: AtomicBool,

    /// Benchmark start time (for duration-based benchmarks)
    start_time: Option<Instant>,

    /// Duration limit (if set, ignores request count)
    duration_limit: Option<Duration>,

    /// Total request limit (for request-count based benchmarks)
    total_requests: u64,
}

impl GlobalCounters {
    /// Base initialization with all counters at zero (internal helper)
    fn base() -> Self {
        Self {
            requests_issued: AtomicU64::new(0),
            requests_finished: AtomicU64::new(0),
            seq_key_counter: AtomicU64::new(0),
            random_key_counter: AtomicU64::new(0),
            dataset_counter: AtomicU64::new(0),
            query_counter: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            start_time: None,
            duration_limit: None,
            total_requests: u64::MAX,
        }
    }

    /// Create new counters initialized to zero (unlimited requests)
    pub fn new() -> Self {
        Self::base()
    }

    /// Create counters with a request limit
    pub fn with_requests(total: u64) -> Self {
        Self {
            total_requests: total,
            ..Self::base()
        }
    }

    /// Create counters with a duration limit (time-based benchmarking)
    pub fn with_duration(duration_secs: u64) -> Self {
        Self {
            start_time: Some(Instant::now()),
            duration_limit: Some(Duration::from_secs(duration_secs)),
            ..Self::base()
        }
    }

    /// Check if duration has been exceeded (for time-based benchmarks)
    #[inline]
    pub fn is_duration_exceeded(&self) -> bool {
        if let (Some(start), Some(limit)) = (self.start_time, self.duration_limit) {
            start.elapsed() >= limit
        } else {
            false
        }
    }

    /// Check if running in duration mode
    #[inline]
    pub fn is_duration_mode(&self) -> bool {
        self.duration_limit.is_some()
    }

    /// Claim a batch of requests (uses internal total_requests limit)
    /// Returns the starting request number, or None if quota exhausted or duration exceeded
    #[inline]
    pub fn claim_batch(&self, batch_size: u64) -> Option<u64> {
        // Check duration first (time-based mode)
        if self.is_duration_exceeded() {
            return None;
        }

        // In duration mode, we don't check request count
        if self.is_duration_mode() {
            return Some(self.requests_issued.fetch_add(batch_size, Ordering::Relaxed));
        }

        // Request count mode - use internal limit
        let issued = self
            .requests_issued
            .fetch_add(batch_size, Ordering::Relaxed);
        if issued >= self.total_requests {
            // Undo the claim
            self.requests_issued
                .fetch_sub(batch_size, Ordering::Relaxed);
            None
        } else {
            Some(issued)
        }
    }

    /// Claim a batch of requests (with explicit total_requests override)
    /// Returns the starting request number, or None if quota exhausted or duration exceeded
    #[inline]
    pub fn claim_requests(&self, batch_size: u64, total_requests: u64) -> Option<u64> {
        // Check duration first (time-based mode)
        if self.is_duration_exceeded() {
            return None;
        }

        // In duration mode, we don't check request count
        if self.is_duration_mode() {
            return Some(self.requests_issued.fetch_add(batch_size, Ordering::Relaxed));
        }

        // Request count mode
        let issued = self
            .requests_issued
            .fetch_add(batch_size, Ordering::Relaxed);
        if issued >= total_requests {
            // Undo the claim
            self.requests_issued
                .fetch_sub(batch_size, Ordering::Relaxed);
            None
        } else {
            Some(issued)
        }
    }

    /// Record completed requests
    #[inline]
    pub fn record_finished(&self, count: u64) {
        self.requests_finished.fetch_add(count, Ordering::Relaxed);
    }

    /// Check if all requests have been issued (for non-duration mode)
    #[inline]
    pub fn is_complete(&self) -> bool {
        if self.is_duration_mode() {
            return false; // Duration mode doesn't complete based on count
        }
        self.requests_issued.load(Ordering::Relaxed) >= self.total_requests
    }

    /// Get next sequential key value
    #[inline]
    pub fn next_seq_key(&self, keyspace: u64) -> u64 {
        self.seq_key_counter.fetch_add(1, Ordering::Relaxed) % keyspace
    }

    /// Get next random key deterministically
    ///
    /// Uses an atomic counter combined with the seed to produce a deterministic
    /// sequence of "random" keys. This ensures that SET and GET with the same
    /// seed and keyspace access the exact same keys, regardless of thread count
    /// or scheduling.
    ///
    /// The mixing function uses SplitMix64-style mixing to produce good
    /// distribution across the keyspace.
    #[inline]
    pub fn next_random_key(&self, seed: u64, keyspace: u64) -> u64 {
        let index = self.random_key_counter.fetch_add(1, Ordering::Relaxed);
        Self::mix_key(seed, index, keyspace)
    }

    /// Deterministically map (seed, index) to a key in [0, keyspace)
    ///
    /// Uses SplitMix64 mixing for good distribution.
    #[inline]
    fn mix_key(seed: u64, index: u64, keyspace: u64) -> u64 {
        // Combine seed and index
        let mut x = seed.wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));

        // SplitMix64 mixing steps
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x = x ^ (x >> 31);

        x % keyspace
    }

    /// Claim next dataset index for vector insertion
    #[inline]
    pub fn next_dataset_idx(&self) -> u64 {
        self.dataset_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Get next query index
    #[inline]
    pub fn next_query_idx(&self, num_queries: u64) -> u64 {
        self.query_counter.fetch_add(1, Ordering::Relaxed) % num_queries
    }

    /// Record an error
    #[inline]
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Signal shutdown to all workers
    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Check if shutdown has been signaled
    #[inline]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Get current progress
    pub fn progress(&self) -> (u64, u64) {
        (
            self.requests_finished.load(Ordering::Relaxed),
            self.requests_issued.load(Ordering::Relaxed),
        )
    }

    /// Get error count
    pub fn errors(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Reset all counters (for warmup -> measurement transition)
    pub fn reset(&self) {
        self.requests_issued.store(0, Ordering::SeqCst);
        self.requests_finished.store(0, Ordering::SeqCst);
        self.error_count.store(0, Ordering::SeqCst);
        // Note: Don't reset seq_key_counter, dataset_counter, query_counter
        // as those track cumulative position
    }
}

impl Default for GlobalCounters {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_claim_requests() {
        let counters = GlobalCounters::new();

        assert_eq!(counters.claim_requests(10, 100), Some(0));
        assert_eq!(counters.claim_requests(10, 100), Some(10));
        assert_eq!(counters.claim_requests(10, 100), Some(20));

        // Claim until exhausted
        for _ in 0..7 {
            counters.claim_requests(10, 100);
        }

        // Now we've claimed 100 total, should return None
        assert_eq!(counters.claim_requests(10, 100), None);
    }

    #[test]
    fn test_concurrent_claims() {
        let counters = Arc::new(GlobalCounters::new());
        let total = 1000u64;
        let batch = 10u64;

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = Arc::clone(&counters);
                thread::spawn(move || {
                    let mut claimed = 0u64;
                    while c.claim_requests(batch, total).is_some() {
                        claimed += batch;
                    }
                    claimed
                })
            })
            .collect();

        let total_claimed: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();

        // Should claim exactly the total (some threads may overclaim slightly)
        assert!(total_claimed >= total);
        assert!(total_claimed <= total + batch * 4);
    }

    #[test]
    fn test_shutdown_signal() {
        let counters = GlobalCounters::new();

        assert!(!counters.is_shutdown());
        counters.signal_shutdown();
        assert!(counters.is_shutdown());
    }

    #[test]
    fn test_seq_key() {
        let counters = GlobalCounters::new();

        assert_eq!(counters.next_seq_key(10), 0);
        assert_eq!(counters.next_seq_key(10), 1);
        assert_eq!(counters.next_seq_key(10), 2);
    }

    #[test]
    fn test_progress() {
        let counters = GlobalCounters::new();

        counters.claim_requests(50, 100);
        counters.record_finished(25);

        let (finished, issued) = counters.progress();
        assert_eq!(finished, 25);
        assert_eq!(issued, 50);
    }

    #[test]
    fn test_reset() {
        let counters = GlobalCounters::new();

        counters.claim_requests(50, 100);
        counters.record_finished(25);
        counters.record_error();

        counters.reset();

        let (finished, issued) = counters.progress();
        assert_eq!(finished, 0);
        assert_eq!(issued, 0);
        assert_eq!(counters.errors(), 0);
    }
}
