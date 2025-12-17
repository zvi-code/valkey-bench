//! Glide-based benchmark worker
//!
//! Uses valkey-glide for all operations including data plane.
//! This provides automatic cluster routing, connection management,
//! and pipelining at the cost of some abstraction overhead.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use hdrhistogram::Histogram;

use super::counters::GlobalCounters;
use super::worker::RecallStats;
use crate::config::BenchmarkConfig;
use crate::dataset::DatasetContext;

#[cfg(feature = "glide-control-plane")]
use glide_core::client::{Client, ConnectionRequest, NodeAddress};
#[cfg(feature = "glide-control-plane")]
use glide_redis::{Cmd, Pipeline, Value};

/// Result from a glide worker
pub struct GlideWorkerResult {
    pub worker_id: usize,
    pub histogram: Histogram<u64>,
    pub recall_stats: RecallStats,
    pub error_count: u64,
    pub requests_processed: u64,
}

/// Glide-based benchmark worker
#[cfg(feature = "glide-control-plane")]
pub struct GlideWorker {
    worker_id: usize,
    config: Arc<BenchmarkConfig>,
    counters: Arc<GlobalCounters>,
    dataset: Option<Arc<DatasetContext>>,
    histogram: Histogram<u64>,
    recall_stats: RecallStats,
    error_count: u64,
    requests_processed: u64,
}

#[cfg(feature = "glide-control-plane")]
impl GlideWorker {
    pub fn new(
        worker_id: usize,
        config: Arc<BenchmarkConfig>,
        counters: Arc<GlobalCounters>,
        dataset: Option<Arc<DatasetContext>>,
    ) -> Self {
        Self {
            worker_id,
            config,
            counters,
            dataset,
            histogram: Histogram::new(3).unwrap(),
            recall_stats: RecallStats::new(),
            error_count: 0,
            requests_processed: 0,
        }
    }

    /// Run the benchmark using glide
    pub fn run(mut self) -> GlideWorkerResult {
        // Create tokio runtime for this worker
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        rt.block_on(async {
            self.run_async().await
        });

        GlideWorkerResult {
            worker_id: self.worker_id,
            histogram: self.histogram,
            recall_stats: self.recall_stats,
            error_count: self.error_count,
            requests_processed: self.requests_processed,
        }
    }

    async fn run_async(&mut self) {
        // Create glide client
        let addresses: Vec<NodeAddress> = self.config.addresses.iter()
            .map(|addr| NodeAddress {
                host: addr.host.clone().into(),
                port: addr.port,
            })
            .collect();

        let tls_mode = match &self.config.tls {
            Some(_) => Some(glide_core::client::TlsMode::SecureTls),
            None => Some(glide_core::client::TlsMode::NoTls),
        };

        let request = ConnectionRequest {
            addresses,
            tls_mode,
            cluster_mode_enabled: self.config.cluster_mode,
            request_timeout: Some(self.config.connect_timeout_ms as u32),
            ..Default::default()
        };

        let mut client = match Client::new(request, None).await {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Worker {} failed to create glide client: {}", self.worker_id, e);
                return;
            }
        };

        // Determine workload from tests
        let test = self.config.tests.first().map(|s| s.as_str()).unwrap_or("set");
        
        match test {
            "set" => self.run_set(&mut client).await,
            "get" => self.run_get(&mut client).await,
            "vec-load" => self.run_vec_load(&mut client).await,
            "vec-search" => self.run_vec_search(&mut client).await,
            _ => {
                tracing::warn!("Workload '{}' not yet implemented for glide worker", test);
            }
        }
    }

    async fn run_set(&mut self, client: &mut Client) {
        let pipeline_size = self.config.pipeline as usize;
        let key_prefix = &self.config.key_prefix;
        let data_size = self.config.data_size;
        let data = vec![b'x'; data_size];
        let total_requests = self.config.requests;

        loop {
            // Claim work
            let start_idx = self.counters.requests_issued.fetch_add(
                pipeline_size as u64,
                Ordering::Relaxed,
            );

            if start_idx >= total_requests {
                break;
            }

            let batch_size = std::cmp::min(
                pipeline_size,
                (total_requests - start_idx) as usize,
            );

            // Build pipeline
            let mut pipeline = Pipeline::new();
            for i in 0..batch_size {
                let key = format!("{}{}", key_prefix, start_idx + i as u64);
                pipeline.cmd("SET").arg(&key).arg(&data);
            }

            // Execute and measure
            let start = Instant::now();
            let result = client.send_pipeline(&pipeline, None, false, None, Default::default()).await;
            let elapsed = start.elapsed();

            match result {
                Ok(_) => {
                    let latency_us = elapsed.as_micros() as u64 / batch_size as u64;
                    for _ in 0..batch_size {
                        let _ = self.histogram.record(latency_us);
                    }
                    self.requests_processed += batch_size as u64;
                    self.counters.requests_finished.fetch_add(
                        batch_size as u64,
                        Ordering::Relaxed,
                    );
                }
                Err(e) => {
                    tracing::debug!("Pipeline error: {}", e);
                    self.error_count += batch_size as u64;
                    self.counters.error_count.fetch_add(batch_size as u64, Ordering::Relaxed);
                }
            }
        }
    }

    async fn run_get(&mut self, client: &mut Client) {
        let pipeline_size = self.config.pipeline as usize;
        let key_prefix = &self.config.key_prefix;
        let keyspace_len = self.config.keyspace_len;
        let total_requests = self.config.requests;

        loop {
            let start_idx = self.counters.requests_issued.fetch_add(
                pipeline_size as u64,
                Ordering::Relaxed,
            );

            if start_idx >= total_requests {
                break;
            }

            let batch_size = std::cmp::min(
                pipeline_size,
                (total_requests - start_idx) as usize,
            );

            // Build pipeline - use random keys within keyspace
            let mut pipeline = Pipeline::new();
            for _ in 0..batch_size {
                let key_idx = fastrand::u64(0..keyspace_len);
                let key = format!("{}{}", key_prefix, key_idx);
                pipeline.cmd("GET").arg(&key);
            }

            let start = Instant::now();
            let result = client.send_pipeline(&pipeline, None, false, None, Default::default()).await;
            let elapsed = start.elapsed();

            match result {
                Ok(_) => {
                    let latency_us = elapsed.as_micros() as u64 / batch_size as u64;
                    for _ in 0..batch_size {
                        let _ = self.histogram.record(latency_us);
                    }
                    self.requests_processed += batch_size as u64;
                    self.counters.requests_finished.fetch_add(
                        batch_size as u64,
                        Ordering::Relaxed,
                    );
                }
                Err(e) => {
                    tracing::debug!("Pipeline error: {}", e);
                    self.error_count += batch_size as u64;
                    self.counters.error_count.fetch_add(batch_size as u64, Ordering::Relaxed);
                }
            }
        }
    }

    async fn run_vec_load(&mut self, client: &mut Client) {
        let dataset = match &self.dataset {
            Some(d) => d.clone(),
            None => {
                tracing::error!("Dataset required for vec-load");
                return;
            }
        };

        let pipeline_size = self.config.pipeline as usize;
        let key_prefix = &self.config.key_prefix;
        let vector_field = self.config.search_config
            .as_ref()
            .map(|s| s.vector_field.as_str())
            .unwrap_or("vec");
        let total_vectors = dataset.base_count() as u64;

        loop {
            let start_idx = self.counters.dataset_counter.fetch_add(
                pipeline_size as u64,
                Ordering::Relaxed,
            );

            if start_idx >= total_vectors {
                break;
            }

            let batch_size = std::cmp::min(
                pipeline_size,
                (total_vectors - start_idx) as usize,
            );

            // Build pipeline with HSET commands
            let mut pipeline = Pipeline::new();
            for i in 0..batch_size {
                let idx = (start_idx + i as u64) as usize;
                let key = format!("{}{}", key_prefix, idx);
                
                if let Some(vector) = dataset.get_base_vector(idx) {
                    // Convert f32 slice to bytes
                    let vector_bytes: Vec<u8> = vector.iter()
                        .flat_map(|f: &f32| f.to_le_bytes())
                        .collect();
                    
                    pipeline.cmd("HSET")
                        .arg(&key)
                        .arg(vector_field)
                        .arg(&vector_bytes);
                }
            }

            let start = Instant::now();
            let result = client.send_pipeline(&pipeline, None, false, None, Default::default()).await;
            let elapsed = start.elapsed();

            match result {
                Ok(_) => {
                    let latency_us = elapsed.as_micros() as u64 / batch_size as u64;
                    for _ in 0..batch_size {
                        let _ = self.histogram.record(latency_us);
                    }
                    self.requests_processed += batch_size as u64;
                    self.counters.requests_finished.fetch_add(
                        batch_size as u64,
                        Ordering::Relaxed,
                    );
                }
                Err(e) => {
                    tracing::debug!("Pipeline error: {}", e);
                    self.error_count += batch_size as u64;
                    self.counters.error_count.fetch_add(batch_size as u64, Ordering::Relaxed);
                }
            }
        }
    }

    async fn run_vec_search(&mut self, client: &mut Client) {
        let dataset = match &self.dataset {
            Some(d) => d.clone(),
            None => {
                tracing::error!("Dataset required for vec-search");
                return;
            }
        };

        let search_config = match &self.config.search_config {
            Some(c) => c,
            None => {
                tracing::error!("Search config required for vec-search");
                return;
            }
        };

        let k = search_config.k as usize;
        let index_name = &search_config.index_name;
        let vector_field = &search_config.vector_field;
        let nocontent = search_config.nocontent;
        let total_requests = self.config.requests;

        loop {
            let query_idx = self.counters.query_counter.fetch_add(1, Ordering::Relaxed);

            if query_idx >= total_requests {
                break;
            }

            // Get query vector (cycle through available queries)
            let actual_idx = (query_idx as usize) % dataset.query_count();
            let query_vector = match dataset.get_query_vector(actual_idx) {
                Some(v) => v,
                None => continue,
            };

            // Build FT.SEARCH command
            let vector_bytes: Vec<u8> = query_vector.iter()
                .flat_map(|f: &f32| f.to_le_bytes())
                .collect();

            let query = format!("*=>[KNN {} @{} $BLOB]", k, vector_field);
            
            let mut cmd = Cmd::new();
            cmd.arg("FT.SEARCH")
                .arg(index_name)
                .arg(&query)
                .arg("PARAMS")
                .arg("2")
                .arg("BLOB")
                .arg(&vector_bytes)
                .arg("DIALECT")
                .arg("2");

            if nocontent {
                cmd.arg("NOCONTENT");
            }

            let start = Instant::now();
            let result = client.send_command(&cmd, None).await;
            let elapsed = start.elapsed();

            match result {
                Ok(value) => {
                    let latency_us = elapsed.as_micros() as u64;
                    let _ = self.histogram.record(latency_us);
                    self.requests_processed += 1;
                    self.counters.requests_finished.fetch_add(1, Ordering::Relaxed);

                    // Calculate recall if ground truth available
                    let neighbors = dataset.get_neighbor_ids(actual_idx as u64);
                    if !neighbors.is_empty() {
                        if let Some(result_ids) = extract_ids_from_value(&value) {
                            let recall = calculate_recall(&result_ids, neighbors, k);
                            self.recall_stats.record(recall);
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("Search error: {}", e);
                    self.error_count += 1;
                    self.counters.error_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Extract document IDs from FT.SEARCH response
#[cfg(feature = "glide-control-plane")]
fn extract_ids_from_value(value: &Value) -> Option<Vec<u64>> {
    // FT.SEARCH returns: [total_count, doc1_key, doc1_fields..., doc2_key, ...]
    // With NOCONTENT: [total_count, doc1_key, doc2_key, ...]
    match value {
        Value::Array(arr) if arr.len() > 1 => {
            let mut ids = Vec::new();
            for item in arr.iter().skip(1) {
                if let Value::BulkString(key_bytes) = item {
                    // Parse key like "vec:12345" to extract numeric ID
                    if let Ok(key_str) = std::str::from_utf8(key_bytes) {
                        if let Some(id_str) = key_str.rsplit(':').next() {
                            if let Ok(id) = id_str.parse::<u64>() {
                                ids.push(id);
                            }
                        }
                    }
                }
            }
            if ids.is_empty() { None } else { Some(ids) }
        }
        _ => None,
    }
}

/// Calculate recall: what fraction of true neighbors were found
#[cfg(feature = "glide-control-plane")]
fn calculate_recall(found: &[u64], ground_truth: &[u64], k: usize) -> f64 {
    let k = k.min(ground_truth.len()).min(found.len());
    if k == 0 {
        return 0.0;
    }

    let gt_set: std::collections::HashSet<u64> = ground_truth.iter()
        .take(k)
        .copied()
        .collect();

    let found_in_gt = found.iter()
        .take(k)
        .filter(|id| gt_set.contains(id))
        .count();

    found_in_gt as f64 / k as f64
}

// Stub for non-glide builds
#[cfg(not(feature = "glide-control-plane"))]
pub struct GlideWorker;

#[cfg(not(feature = "glide-control-plane"))]
impl GlideWorker {
    pub fn new(
        _worker_id: usize,
        _config: Arc<BenchmarkConfig>,
        _counters: Arc<GlobalCounters>,
        _dataset: Option<Arc<DatasetContext>>,
    ) -> Self {
        panic!("GlideWorker requires 'glide-control-plane' feature")
    }

    pub fn run(self) -> GlideWorkerResult {
        panic!("GlideWorker requires 'glide-control-plane' feature")
    }
}
