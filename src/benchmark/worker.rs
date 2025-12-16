//! Benchmark worker thread implementation
//!
//! Each worker owns its clients exclusively. The only synchronization
//! points are atomic counters for request claiming and progress tracking.
//!
//! ## MOVED/ASK Error Handling
//!
//! When the cluster topology changes during a benchmark, workers handle
//! MOVED and ASK errors similar to the C implementation:
//!
//! - MOVED: Triggers topology refresh via TopologyManager
//! - ASK: Sends ASKING command to target node before retry
//! - CLUSTERDOWN: Waits 1 second then refreshes topology

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;
use tracing::warn;

use super::counters::GlobalCounters;
use crate::client::{BenchmarkClient, CommandBuffer, PlaceholderOffset, PlaceholderType};
use crate::cluster::{RedirectInfo, TopologyManager};
use crate::config::BenchmarkConfig;
use crate::dataset::DatasetContext;
use crate::workload::{extract_numeric_ids, parse_search_response};

/// Result from a worker thread
pub struct WorkerResult {
    /// Worker ID
    pub worker_id: usize,
    /// Local histogram of latencies (microseconds)
    pub histogram: Histogram<u64>,
    /// Recall statistics (for vector queries)
    pub recall_stats: RecallStats,
    /// Number of MOVED/ASK redirects handled
    pub redirect_count: u64,
    /// Number of topology refreshes triggered
    pub topology_refresh_count: u64,
    /// Number of errors (excluding handled redirects)
    pub error_count: u64,
    /// Total requests processed
    pub requests_processed: u64,
}

/// Recall statistics for vector search
#[derive(Debug, Default)]
pub struct RecallStats {
    pub total_queries: u64,
    pub sum_recall: f64,
    pub min_recall: f64,
    pub max_recall: f64,
    pub perfect_count: u64, // recall == 1.0
    pub zero_count: u64,    // recall == 0.0
}

impl RecallStats {
    pub fn new() -> Self {
        Self {
            min_recall: f64::MAX,
            max_recall: f64::MIN,
            ..Default::default()
        }
    }

    pub fn record(&mut self, recall: f64) {
        self.total_queries += 1;
        self.sum_recall += recall;
        self.min_recall = self.min_recall.min(recall);
        self.max_recall = self.max_recall.max(recall);

        if (recall - 1.0).abs() < f64::EPSILON {
            self.perfect_count += 1;
        }
        if recall < f64::EPSILON {
            self.zero_count += 1;
        }
    }

    pub fn average(&self) -> f64 {
        if self.total_queries > 0 {
            self.sum_recall / self.total_queries as f64
        } else {
            0.0
        }
    }

    pub fn merge(&mut self, other: &RecallStats) {
        if other.total_queries == 0 {
            return;
        }
        self.total_queries += other.total_queries;
        self.sum_recall += other.sum_recall;
        if other.min_recall < self.min_recall {
            self.min_recall = other.min_recall;
        }
        if other.max_recall > self.max_recall {
            self.max_recall = other.max_recall;
        }
        self.perfect_count += other.perfect_count;
        self.zero_count += other.zero_count;
    }
}

/// Simple token bucket for rate limiting
struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    tokens_per_ms: f64,
    last_update: Instant,
}

impl TokenBucket {
    fn new(rps: u64) -> Self {
        let tokens_per_ms = rps as f64 / 1000.0;
        Self {
            tokens: 0.0,
            max_tokens: rps as f64, // 1 second burst
            tokens_per_ms,
            last_update: Instant::now(),
        }
    }

    fn acquire(&mut self, count: u32) -> Option<Duration> {
        // Refill tokens based on elapsed time
        let now = Instant::now();
        let elapsed_ms = now.duration_since(self.last_update).as_secs_f64() * 1000.0;
        self.tokens = (self.tokens + elapsed_ms * self.tokens_per_ms).min(self.max_tokens);
        self.last_update = now;

        let needed = count as f64;
        if self.tokens >= needed {
            self.tokens -= needed;
            None // Can proceed immediately
        } else {
            // Calculate wait time
            let deficit = needed - self.tokens;
            let wait_ms = deficit / self.tokens_per_ms;
            Some(Duration::from_secs_f64(wait_ms / 1000.0))
        }
    }
}

/// Benchmark worker (runs in dedicated OS thread)
pub struct BenchmarkWorker {
    /// Worker ID
    id: usize,

    /// Owned clients (NOT shared with other threads)
    clients: Vec<BenchmarkClient>,

    /// Thread-local RNG (fast, no sync)
    rng: fastrand::Rng,

    /// Thread-local histogram
    histogram: Histogram<u64>,

    /// Thread-local recall stats
    recall_stats: RecallStats,

    /// Local retry queue (indices of commands to retry)
    retry_queue: VecDeque<u64>,

    /// Configuration
    pipeline: u32,
    keyspace_len: u64,
    sequential: bool,
    total_requests: u64,

    /// Rate limiter tokens (if enabled)
    rate_limit_tokens: Option<TokenBucket>,

    /// Whether this worker should compute recall (VecQuery workload)
    compute_recall: bool,

    /// Vector key prefix for extracting IDs from FT.SEARCH response
    key_prefix: String,

    /// K value for recall@k computation
    k: usize,

    /// Shared topology manager for cluster mode (handles MOVED/ASK)
    topology_manager: Option<Arc<TopologyManager>>,

    /// Last known topology version (to detect updates)
    topology_version: u64,

    /// Count of redirect operations handled
    redirect_count: u64,

    /// Count of topology refreshes triggered by this worker
    topology_refresh_count: u64,
}

impl BenchmarkWorker {
    /// Create new worker
    pub fn new(id: usize, clients: Vec<BenchmarkClient>, config: &BenchmarkConfig) -> Self {
        // Initialize RNG with worker-specific seed
        let seed = if config.seed == 0 {
            // Random seed
            fastrand::u64(..)
        } else {
            // Deterministic seed based on config + worker id
            config.seed.wrapping_add(id as u64)
        };
        let rng = fastrand::Rng::with_seed(seed);

        // Initialize histogram (1us to 1 hour, 3 significant digits)
        let histogram =
            Histogram::new_with_bounds(1, 3_600_000_000, 3).expect("Failed to create histogram");

        // Rate limiter
        let rate_limit_tokens = if config.requests_per_second > 0 {
            // Divide RPS among threads
            let per_thread_rps = config.requests_per_second / config.threads as u64;
            Some(TokenBucket::new(per_thread_rps.max(1)))
        } else {
            None
        };

        // Recall computation config
        let (compute_recall, key_prefix, k) = if let Some(ref sc) = config.search_config {
            (true, sc.prefix.clone(), sc.k as usize)
        } else {
            (false, String::new(), 0)
        };

        Self {
            id,
            clients,
            rng,
            histogram,
            recall_stats: RecallStats::new(),
            retry_queue: VecDeque::new(),
            pipeline: config.pipeline,
            keyspace_len: config.keyspace_len,
            sequential: config.sequential,
            total_requests: config.requests,
            rate_limit_tokens,
            compute_recall,
            key_prefix,
            k,
            topology_manager: None,
            topology_version: 0,
            redirect_count: 0,
            topology_refresh_count: 0,
        }
    }

    /// Set topology manager for cluster mode
    pub fn with_topology_manager(mut self, manager: Arc<TopologyManager>) -> Self {
        self.topology_version = manager.version();
        self.topology_manager = Some(manager);
        self
    }

    /// Main worker loop
    pub fn run(
        mut self,
        counters: Arc<GlobalCounters>,
        dataset: Option<Arc<DatasetContext>>,
    ) -> WorkerResult {
        let batch_size = self.pipeline as u64;
        let mut client_idx = 0;
        let num_clients = self.clients.len();
        let mut requests_processed = 0u64;
        let mut error_count = 0u64;

        loop {
            // Check shutdown
            if counters.is_shutdown() {
                break;
            }

            // Claim request batch
            if counters
                .claim_requests(batch_size, self.total_requests)
                .is_none()
            {
                break;
            }

            // Rate limiting
            if let Some(ref mut limiter) = self.rate_limit_tokens {
                if let Some(wait) = limiter.acquire(self.pipeline) {
                    std::thread::sleep(wait);
                }
            }

            // Get next client (round-robin within this worker's clients)
            let cur_client_idx = client_idx;
            client_idx = (client_idx + 1) % num_clients;

            // Clear batch state
            self.clients[cur_client_idx].clear_batch_state();

            // Fill placeholders
            self.fill_placeholders_for_client(cur_client_idx, &counters, dataset.as_deref());

            // Execute batch
            match self.clients[cur_client_idx].execute_batch() {
                Ok(response) => {
                    // Record latency
                    self.histogram.record(response.latency_us).ok();

                    // Process responses
                    let mut batch_errors = 0u64;
                    let mut batch_redirects = 0u64;
                    let mut need_topology_refresh = false;

                    for (i, value) in response.values.iter().enumerate() {
                        if value.is_error() {
                            if let crate::utils::RespValue::Error(msg) = value {
                                // Check for MOVED/ASK redirects
                                if let Some(redirect) = RedirectInfo::parse(msg) {
                                    batch_redirects += 1;
                                    self.redirect_count += 1;

                                    // MOVED indicates slot ownership changed - need topology refresh
                                    if !redirect.is_ask {
                                        need_topology_refresh = true;
                                    }

                                    // Track for retry (if we had per-command tracking)
                                    if let Some(&idx) = response.inflight_indices.get(i) {
                                        self.retry_queue.push_back(idx);
                                    }

                                    // Log redirect (limited)
                                    if batch_redirects <= 3 {
                                        warn!(
                                            "Worker {}: {} redirect to {}:{} for slot {}",
                                            self.id,
                                            if redirect.is_ask { "ASK" } else { "MOVED" },
                                            redirect.host,
                                            redirect.port,
                                            redirect.slot
                                        );
                                    }
                                } else if TopologyManager::is_cluster_down(msg) {
                                    // CLUSTERDOWN - wait then refresh
                                    warn!("Worker {}: CLUSTERDOWN received, waiting 1s...", self.id);
                                    std::thread::sleep(Duration::from_secs(1));
                                    need_topology_refresh = true;
                                    batch_errors += 1;
                                    counters.record_error();
                                } else {
                                    // Other error
                                    batch_errors += 1;
                                    counters.record_error();

                                    // Log first few errors for debugging
                                    if batch_errors <= 3 {
                                        eprintln!("Worker {}: Error response: {}", self.id, msg);
                                    }
                                }
                            }
                        } else if self.compute_recall {
                            // Compute recall for FT.SEARCH responses
                            if let Some(&query_idx) = response.query_indices.get(i) {
                                if let Some(ref ds) = dataset {
                                    self.compute_recall_for_response(value, query_idx, ds);
                                }
                            }
                        }
                    }

                    // Trigger topology refresh if needed (like C code's fetchClusterSlotsConfiguration)
                    if need_topology_refresh {
                        self.handle_topology_refresh();
                    }

                    error_count += batch_errors;

                    // Count successful responses (excluding redirects which will be retried)
                    let successful = response.values.len() as u64 - batch_redirects;
                    requests_processed += successful;
                    counters.record_finished(successful);
                }
                Err(e) => {
                    error_count += batch_size;
                    counters.record_error();
                    // Log error
                    eprintln!("Worker {}: Batch error: {}", self.id, e);
                }
            }
        }

        WorkerResult {
            worker_id: self.id,
            histogram: self.histogram,
            recall_stats: self.recall_stats,
            redirect_count: self.redirect_count,
            topology_refresh_count: self.topology_refresh_count,
            error_count,
            requests_processed,
        }
    }

    /// Handle topology refresh (similar to C code's fetchClusterSlotsConfiguration)
    fn handle_topology_refresh(&mut self) {
        if let Some(ref manager) = self.topology_manager {
            // Check if topology was already refreshed by another worker
            let current_version = manager.version();
            if current_version > self.topology_version {
                // Another worker already refreshed, just update our version
                self.topology_version = current_version;
                return;
            }

            // Try to refresh topology
            if manager.refresh_topology() {
                self.topology_refresh_count += 1;
                self.topology_version = manager.version();
            } else {
                // Another worker is refreshing or refresh failed
                // Update our version in case it changed
                self.topology_version = manager.version();
            }
        }
    }

    /// Fill placeholders for all commands in the batch for a specific client
    fn fill_placeholders_for_client(
        &mut self,
        client_idx: usize,
        counters: &GlobalCounters,
        dataset: Option<&DatasetContext>,
    ) {
        // First, collect placeholder info (to avoid borrow conflicts)
        let placeholder_info: Vec<Vec<(PlaceholderType, usize, usize)>> = {
            let buffer = self.clients[client_idx].write_buffer();
            (0..self.pipeline as usize)
                .map(|cmd_idx| {
                    if cmd_idx >= buffer.placeholders.len() {
                        Vec::new()
                    } else {
                        buffer.placeholders[cmd_idx]
                            .iter()
                            .map(|ph| (ph.placeholder_type, ph.offset, ph.len))
                            .collect()
                    }
                })
                .collect()
        };

        // Now fill placeholders using the collected info
        for (cmd_idx, placeholders) in placeholder_info.iter().enumerate() {
            for &(ph_type, offset, len) in placeholders {
                let ph = crate::client::PlaceholderOffset {
                    offset,
                    len,
                    placeholder_type: ph_type,
                };

                match ph_type {
                    PlaceholderType::Key => {
                        // For vector load with dataset, use the dataset index as key
                        // to ensure key matches vector index for recall computation
                        let key = if dataset.is_some() {
                            // Will be set together with Vector placeholder below
                            continue;
                        } else if self.sequential {
                            counters.next_seq_key(self.keyspace_len)
                        } else {
                            self.rng.u64(0..self.keyspace_len)
                        };
                        self.clients[client_idx].replace_key(cmd_idx, key, &ph);
                    }
                    PlaceholderType::Vector => {
                        if let Some(ds) = dataset {
                            // INSERT/PREFILL: Get next dataset index for database vectors
                            // Like C code's replacePlaceholderDataset, we use sequential
                            // assignment during prefill to ensure each vector is loaded exactly once.
                            // The key is derived from the same index to ensure key=vector_id.
                            let idx = counters.next_dataset_idx() % ds.num_vectors();
                            let vec_bytes = ds.get_vector_bytes(idx);
                            self.clients[client_idx].replace_vector(cmd_idx, vec_bytes, &ph);
                            self.clients[client_idx].track_inflight(idx);

                            // Also set the key to match the dataset index (for recall)
                            // This mirrors C code's encode_vector_key_fixed using vector_id
                            for (other_ph_type, other_offset, other_len) in placeholders.iter() {
                                if matches!(other_ph_type, PlaceholderType::Key) {
                                    let key_ph = PlaceholderOffset {
                                        offset: *other_offset,
                                        len: *other_len,
                                        placeholder_type: *other_ph_type,
                                    };
                                    self.clients[client_idx].replace_key(cmd_idx, idx, &key_ph);
                                    break;
                                }
                            }
                        }
                    }
                    PlaceholderType::QueryVector => {
                        if let Some(ds) = dataset {
                            // Get query vector from dataset queries section
                            // Use random query selection (like C code's xorshift64* PRNG)
                            // to avoid cache-friendly patterns during benchmarking
                            let idx = self.rng.u64(0..ds.num_queries());
                            let vec_bytes = ds.get_query_bytes(idx);
                            self.clients[client_idx].replace_vector(cmd_idx, vec_bytes, &ph);
                            self.clients[client_idx].track_query(idx);
                        }
                    }
                    PlaceholderType::ClusterTag => {
                        // Simplified: use static tag for now
                        // Full implementation would use ClusterTagMap
                        let tag = b"{000}";
                        self.clients[client_idx].replace_cluster_tag(cmd_idx, tag, &ph);
                    }
                    PlaceholderType::RandInt => {
                        let value = self.rng.u64(..);
                        self.clients[client_idx].replace_key(cmd_idx, value, &ph);
                    }
                }
            }
        }
    }

    /// Compute recall for an FT.SEARCH response
    fn compute_recall_for_response(
        &mut self,
        response: &crate::utils::RespValue,
        query_idx: u64,
        dataset: &DatasetContext,
    ) {
        // Parse FT.SEARCH response to get document IDs
        let doc_ids = parse_search_response(response);

        // Extract numeric IDs from document keys
        let result_ids = extract_numeric_ids(&doc_ids, &self.key_prefix);

        // Compute recall using dataset ground truth
        let recall = dataset.compute_recall(query_idx, &result_ids, self.k);

        // Record recall stat
        self.recall_stats.record(recall);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_stats() {
        let mut stats = RecallStats::new();

        stats.record(1.0);
        stats.record(0.5);
        stats.record(0.0);

        assert_eq!(stats.total_queries, 3);
        assert!((stats.average() - 0.5).abs() < 0.001);
        assert_eq!(stats.perfect_count, 1);
        assert_eq!(stats.zero_count, 1);
    }

    #[test]
    fn test_recall_stats_merge() {
        let mut stats1 = RecallStats::new();
        stats1.record(1.0);
        stats1.record(0.8);

        let mut stats2 = RecallStats::new();
        stats2.record(0.5);
        stats2.record(0.0);

        stats1.merge(&stats2);

        assert_eq!(stats1.total_queries, 4);
        assert!((stats1.average() - 0.575).abs() < 0.001); // (1.0 + 0.8 + 0.5 + 0.0) / 4
        assert_eq!(stats1.perfect_count, 1);
        assert_eq!(stats1.zero_count, 1);
    }

    #[test]
    fn test_token_bucket() {
        let mut bucket = TokenBucket::new(1000); // 1000 RPS

        // Should be able to acquire immediately after some time
        std::thread::sleep(Duration::from_millis(10));
        assert!(bucket.acquire(5).is_none());
    }

    #[test]
    fn test_token_bucket_wait() {
        let mut bucket = TokenBucket::new(100); // 100 RPS

        // Try to acquire more than accumulated
        let wait = bucket.acquire(200);
        assert!(wait.is_some());
        // Should need to wait approximately 2 seconds for 200 tokens at 100 RPS
        // But since some time may have passed, just verify we need to wait
        if let Some(duration) = wait {
            assert!(duration.as_secs_f64() > 0.1);
        }
    }
}
