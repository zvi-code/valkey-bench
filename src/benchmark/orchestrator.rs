//! Benchmark orchestrator
//!
//! Coordinates worker threads, collects results, and manages the benchmark lifecycle.

use std::path::Path;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::info;

use super::counters::GlobalCounters;
use super::event_worker::{EventWorker, EventWorkerResult};
use super::worker::{BenchmarkWorker, RecallStats, WorkerResult};
use crate::client::{BenchmarkClient, ConnectionFactory, ControlPlane, ControlPlaneExt};
use crate::cluster::{
    build_vector_id_mappings, ClusterScanConfig, ClusterTagMap, ClusterTopology, TopologyManager,
};
use crate::config::BenchmarkConfig;
use crate::dataset::DatasetContext;
use crate::metrics::reporter::{BenchmarkResults, OutputFormat};
use crate::metrics::{BackfillWaitConfig, EngineType, MetricsCollector, MetricsReporter, NodeMetrics};
use crate::utils::Result;
use crate::workload::{
    create_index, create_template, drop_index, get_index_info, WorkloadType,
};

/// Benchmark result summary
pub struct BenchmarkResult {
    /// Test name
    pub test_name: String,
    /// Total requests completed
    pub total_requests: u64,
    /// Total duration
    pub duration: Duration,
    /// Throughput (requests per second)
    pub throughput: f64,
    /// Merged latency histogram
    pub histogram: Histogram<u64>,
    /// Merged recall statistics
    pub recall_stats: RecallStats,
    /// Total errors
    pub error_count: u64,
    /// Per-node metrics snapshots
    pub node_metrics: Vec<crate::metrics::node_metrics::NodeMetricsSnapshot>,
}

impl BenchmarkResult {
    /// Get percentile latency in microseconds
    pub fn percentile_us(&self, p: f64) -> u64 {
        self.histogram.value_at_percentile(p)
    }

    /// Get percentile latency in milliseconds
    pub fn percentile_ms(&self, p: f64) -> f64 {
        self.percentile_us(p) as f64 / 1000.0
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== {} ===", self.test_name);
        println!("Throughput: {:.2} req/sec", self.throughput);
        println!("Total requests: {}", self.total_requests);
        println!("Duration: {:.2}s", self.duration.as_secs_f64());
        println!("Errors: {}", self.error_count);
        println!("\nLatency (ms):");
        println!("  avg: {:.3}", self.histogram.mean() / 1000.0);
        println!("  p50: {:.3}", self.percentile_ms(50.0));
        println!("  p95: {:.3}", self.percentile_ms(95.0));
        println!("  p99: {:.3}", self.percentile_ms(99.0));
        println!("  p99.9: {:.3}", self.percentile_ms(99.9));
        println!("  max: {:.3}", self.histogram.max() as f64 / 1000.0);

        if self.recall_stats.total_queries > 0 {
            println!("\nRecall:");
            println!("  avg: {:.4}", self.recall_stats.average());
            println!("  min: {:.4}", self.recall_stats.min_recall);
            println!("  max: {:.4}", self.recall_stats.max_recall);
            println!("  perfect (1.0): {}", self.recall_stats.perfect_count);
            println!("  zero (0.0): {}", self.recall_stats.zero_count);
        } else if self.total_requests > 0 {
            // Debug: show if recall wasn't computed
            println!("\n(Recall not computed - requires dataset with ground truth)");
        }
    }
}

/// Benchmark orchestrator
pub struct Orchestrator {
    config: Arc<BenchmarkConfig>,
    connection_factory: ConnectionFactory,
    dataset: Option<Arc<DatasetContext>>,
    /// Discovered cluster topology (if cluster mode enabled)
    cluster_topology: Option<ClusterTopology>,
    /// Shared topology manager for dynamic cluster updates
    topology_manager: Option<Arc<TopologyManager>>,
    /// Cluster tag map for vector ID to node routing (for vec-query with existing data)
    cluster_tag_map: Option<Arc<ClusterTagMap>>,
}

impl Orchestrator {
    /// Create new orchestrator
    pub fn new(config: BenchmarkConfig) -> Result<Self> {
        let connection_factory = ConnectionFactory {
            connect_timeout: Duration::from_millis(config.connect_timeout_ms),
            read_timeout: Duration::from_millis(config.request_timeout_ms),
            write_timeout: Duration::from_millis(config.request_timeout_ms),
            tls_config: config.tls.clone(),
            auth_password: config.auth.as_ref().map(|a| a.password.clone()),
            auth_username: config.auth.as_ref().and_then(|a| a.username.clone()),
            dbnum: config.dbnum,
        };

        // Discover cluster topology if cluster mode is enabled or auto-detect
        let cluster_topology = Self::discover_cluster(&config, &connection_factory)?;

        // Create topology manager for dynamic updates (cluster mode only)
        let topology_manager = cluster_topology.as_ref().map(|topo| {
            let seed_addresses: Vec<(String, u16)> = config
                .addresses
                .iter()
                .map(|a| (a.host.clone(), a.port))
                .collect();

            Arc::new(TopologyManager::new(
                topo.clone(),
                connection_factory.clone(),
                seed_addresses,
            ))
        });

        Ok(Self {
            config: Arc::new(config),
            connection_factory,
            dataset: None,
            cluster_topology,
            topology_manager,
            cluster_tag_map: None,
        })
    }

    /// Discover cluster topology by connecting to seed node
    fn discover_cluster(
        config: &BenchmarkConfig,
        factory: &ConnectionFactory,
    ) -> Result<Option<ClusterTopology>> {
        if config.addresses.is_empty() {
            return Ok(None);
        }

        let addr = &config.addresses[0];

        // Create a temporary connection to discover topology
        let mut conn = factory.create(&addr.host, addr.port)?;

        // Try CLUSTER NODES to see if this is a cluster
        match conn.cluster_nodes() {
            Ok(nodes_response) => {
                let topology = ClusterTopology::from_cluster_nodes(&nodes_response)
                    .map_err(|e| crate::utils::BenchmarkError::Connection(
                        crate::utils::ConnectionError::ConnectFailed {
                            host: addr.host.clone(),
                            port: addr.port,
                            source: std::io::Error::new(std::io::ErrorKind::Other, e),
                        }
                    ))?;

                info!(
                    "Discovered cluster with {} primaries, {} total nodes",
                    topology.num_primaries(),
                    topology.num_nodes()
                );

                // Log primary addresses
                for primary in topology.primaries() {
                    info!("  Primary: {}:{} (slots: {})", primary.host, primary.port, primary.slots.len());
                }

                Ok(Some(topology))
            }
            Err(e) => {
                // Not a cluster or cluster commands disabled
                if config.cluster_mode {
                    // User explicitly requested cluster mode but it's not available
                    return Err(crate::utils::BenchmarkError::Connection(
                        crate::utils::ConnectionError::ConnectFailed {
                            host: addr.host.clone(),
                            port: addr.port,
                            source: std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("Cluster mode requested but CLUSTER NODES failed: {}", e),
                            ),
                        }
                    ));
                }
                info!("Running in standalone mode (CLUSTER NODES not available)");
                Ok(None)
            }
        }
    }

    /// Get addresses for benchmark clients (primaries for cluster, seed for standalone)
    fn get_benchmark_addresses(&self) -> Vec<(String, u16)> {
        if let Some(ref topology) = self.cluster_topology {
            // Use primary nodes for cluster mode
            topology
                .primaries()
                .filter(|n| n.is_available())
                .map(|n| (n.host.clone(), n.port))
                .collect()
        } else {
            // Use configured addresses for standalone
            self.config
                .addresses
                .iter()
                .map(|a| (a.host.clone(), a.port))
                .collect()
        }
    }

    /// Set dataset (for vector search)
    pub fn set_dataset(&mut self, dataset: DatasetContext) {
        self.dataset = Some(Arc::new(dataset));
    }

    /// Build cluster tag map by scanning cluster for existing vector keys
    ///
    /// This is used for vec-query with existing data to:
    /// 1. Know which vectors exist in the cluster
    /// 2. Map vector IDs to their cluster tags for proper routing
    /// 3. Validate recall only against vectors that actually exist
    pub fn build_cluster_tag_map(&mut self) -> Result<()> {
        let search_config = match &self.config.search_config {
            Some(cfg) => cfg,
            None => {
                info!("No search config - skipping cluster tag map");
                return Ok(());
            }
        };

        // Get capacity from dataset if available
        let capacity = self
            .dataset
            .as_ref()
            .map(|ds| ds.num_vectors())
            .unwrap_or(1_000_000) as u64;

        let is_cluster = self.cluster_topology.is_some();

        info!(
            "Building cluster tag map for prefix '{}' (capacity: {}, cluster: {})",
            search_config.prefix, capacity, is_cluster
        );

        let tag_map = Arc::new(ClusterTagMap::new(
            &search_config.prefix,
            capacity,
            is_cluster,
        ));

        // Only scan if in cluster mode with topology
        if let Some(ref topology) = self.cluster_topology {
            let nodes: Vec<_> = topology.nodes.clone();

            let scan_config = ClusterScanConfig {
                pattern: format!("{}*", search_config.prefix),
                batch_size: 1000,
                timeout: Duration::from_secs(5),
                show_progress: !self.config.quiet,
            };

            match build_vector_id_mappings(&tag_map, &nodes, &scan_config) {
                Ok(results) => {
                    info!(
                        "Cluster tag map built: {} vectors mapped from {} keys",
                        tag_map.count(),
                        results.total_keys
                    );
                }
                Err(e) => {
                    return Err(crate::utils::BenchmarkError::Config(format!(
                        "Failed to build cluster tag map: {}",
                        e
                    )));
                }
            }
        } else {
            info!("Standalone mode - cluster tag map will not route by node");
        }

        self.cluster_tag_map = Some(tag_map);
        Ok(())
    }

    /// Get cluster tag map
    pub fn cluster_tag_map(&self) -> Option<Arc<ClusterTagMap>> {
        self.cluster_tag_map.clone()
    }

    /// Run a single benchmark test
    pub fn run_test(&self, workload: WorkloadType) -> Result<BenchmarkResult> {
        // Create command template (use cluster mode if we have cluster topology)
        let cluster_mode = self.cluster_topology.is_some();
        let template = create_template(
            workload,
            &self.config.key_prefix,
            self.config.data_size,
            self.config.search_config.as_ref(),
            cluster_mode,
        );

        // Build command buffer for pipeline
        let command_buffer = template.build(self.config.pipeline as usize);

        // Create global counters (duration-based or request-count based)
        let counters = Arc::new(if let Some(duration) = self.config.duration_secs {
            GlobalCounters::with_duration(duration)
        } else {
            GlobalCounters::with_requests(self.config.requests)
        });

        // Calculate clients per thread
        let clients_per_thread = self.config.clients_per_thread() as usize;

        // Spawn worker threads
        let mut handles: Vec<JoinHandle<WorkerResult>> =
            Vec::with_capacity(self.config.threads as usize);

        let start_time = Instant::now();

        // Get addresses for workers (cluster primaries or standalone addresses)
        let addresses = self.get_benchmark_addresses();
        if addresses.is_empty() {
            return Err(crate::utils::BenchmarkError::Config(
                "No available server addresses".to_string(),
            ));
        }

        for worker_id in 0..self.config.threads as usize {
            let config = Arc::clone(&self.config);
            let counters = Arc::clone(&counters);
            let dataset = self.dataset.clone();
            let buffer = command_buffer.clone();
            let factory = self.connection_factory.clone();
            let topology_manager = self.topology_manager.clone();
            // Distribute workers across available addresses (round-robin)
            let (host, port) = addresses[worker_id % addresses.len()].clone();

            let handle = thread::Builder::new()
                .name(format!("worker-{}", worker_id))
                .spawn(move || {
                    // Create clients for this worker
                    let mut clients = Vec::with_capacity(clients_per_thread);
                    for _ in 0..clients_per_thread {
                        match factory.create(&host, port) {
                            Ok(conn) => {
                                let client = BenchmarkClient::new(
                                    conn,
                                    buffer.clone(),
                                    config.pipeline as usize,
                                );
                                clients.push(client);
                            }
                            Err(e) => {
                                eprintln!(
                                    "Worker {}: Failed to create connection: {}",
                                    worker_id, e
                                );
                                // Return empty result with error count
                                return WorkerResult {
                                    worker_id,
                                    histogram: Histogram::new_with_bounds(1, 3_600_000_000, 3)
                                        .expect("Failed to create histogram"),
                                    recall_stats: RecallStats::new(),
                                    redirect_count: 0,
                                    topology_refresh_count: 0,
                                    error_count: 1,
                                    requests_processed: 0,
                                };
                            }
                        }
                    }

                    // Create worker with optional topology manager for cluster mode
                    let mut worker = BenchmarkWorker::new(worker_id, clients, &config);
                    if let Some(tm) = topology_manager {
                        worker = worker.with_topology_manager(tm);
                    }
                    worker.run(counters, dataset)
                })
                .expect("Failed to spawn worker thread");

            handles.push(handle);
        }

        // Progress reporting (if not quiet)
        if !self.config.quiet {
            let counters_clone = Arc::clone(&counters);
            let total = self.config.requests;
            let duration_secs = self.config.duration_secs;
            thread::spawn(move || {
                Self::report_progress(&counters_clone, total, duration_secs);
            });
        }

        // Wait for workers to complete
        let results: Vec<WorkerResult> = handles
            .into_iter()
            .map(|h| h.join().expect("Worker thread panicked"))
            .collect();

        let duration = start_time.elapsed();

        // Signal shutdown to progress reporter
        counters.signal_shutdown();

        // Merge results (with empty node metrics for now - can be enhanced later)
        self.merge_results(workload.as_str(), results, duration, Vec::new())
    }

    /// Report progress during benchmark
    fn report_progress(counters: &GlobalCounters, total: u64, duration_secs: Option<u64>) {
        if let Some(duration) = duration_secs {
            // Duration-based progress (show elapsed time and throughput)
            let pb = ProgressBar::new(duration);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}s/{len}s | {msg}",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
            );

            let start = std::time::Instant::now();
            let mut last_finished = 0u64;
            let mut last_time = start;

            while !counters.is_shutdown() && !counters.is_duration_exceeded() {
                let elapsed = start.elapsed().as_secs();
                let (finished, _) = counters.progress();
                pb.set_position(elapsed);

                // Calculate current throughput
                let now = std::time::Instant::now();
                let interval = now.duration_since(last_time).as_secs_f64();
                if interval >= 1.0 {
                    let throughput = (finished - last_finished) as f64 / interval;
                    pb.set_message(format!("{:.0} ops/s, total: {}", throughput, finished));
                    last_finished = finished;
                    last_time = now;
                }

                thread::sleep(Duration::from_millis(100));
            }

            let (finished, _) = counters.progress();
            pb.finish_with_message(format!("Complete - {} total ops", finished));
        } else {
            // Request count based progress
            let pb = ProgressBar::new(total);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
            );

            while !counters.is_shutdown() {
                let (finished, _) = counters.progress();
                pb.set_position(finished);

                if finished >= total {
                    break;
                }

                thread::sleep(Duration::from_millis(100));
            }

            pb.finish_with_message("Complete");
        }
    }

    /// Merge results from all workers
    fn merge_results(
        &self,
        test_name: &str,
        results: Vec<WorkerResult>,
        duration: Duration,
        node_metrics: Vec<crate::metrics::node_metrics::NodeMetricsSnapshot>,
    ) -> Result<BenchmarkResult> {
        // Initialize merged histogram
        let mut merged_histogram =
            Histogram::new_with_bounds(1, 3_600_000_000, 3).expect("Failed to create histogram");

        let mut merged_recall = RecallStats::new();
        let mut total_requests = 0u64;
        let mut error_count = 0u64;

        for result in results {
            merged_histogram.add(&result.histogram).ok();
            merged_recall.merge(&result.recall_stats);
            total_requests += result.requests_processed;
            error_count += result.error_count;
        }

        let throughput = total_requests as f64 / duration.as_secs_f64();

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            total_requests,
            duration,
            throughput,
            histogram: merged_histogram,
            recall_stats: merged_recall,
            error_count,
            node_metrics,
        })
    }

    /// Merge results from event-driven workers
    fn merge_event_results(
        &self,
        test_name: &str,
        results: Vec<EventWorkerResult>,
        duration: Duration,
    ) -> Result<BenchmarkResult> {
        let mut merged_histogram =
            Histogram::new_with_bounds(1, 3_600_000_000, 3).expect("Failed to create histogram");

        let mut merged_recall = RecallStats::new();
        let mut total_requests = 0u64;
        let mut error_count = 0u64;

        for result in results {
            merged_histogram.add(&result.histogram).ok();
            merged_recall.merge(&result.recall_stats);
            total_requests += result.requests_processed;
            error_count += result.error_count;
        }

        let throughput = total_requests as f64 / duration.as_secs_f64();

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            total_requests,
            duration,
            throughput,
            histogram: merged_histogram,
            recall_stats: merged_recall,
            error_count,
            node_metrics: Vec::new(),
        })
    }

    /// Run a single benchmark test using event-driven I/O (like C's ae)
    pub fn run_test_event(&self, workload: WorkloadType) -> Result<BenchmarkResult> {
        // For vec-load with partial prefill, calculate actual vectors to load
        let effective_requests = if matches!(workload, WorkloadType::VecLoad) {
            if let Some(ref tag_map) = self.cluster_tag_map {
                tag_map.reset_unmapped_counter();
                
                // Calculate vectors to actually load (skip existing)
                let already_mapped = tag_map.count();
                let dataset_size = self.dataset.as_ref().map(|ds| ds.num_vectors()).unwrap_or(0);
                let requested = self.config.requests.min(dataset_size);
                let to_load = requested.saturating_sub(already_mapped);
                
                if already_mapped > 0 {
                    info!(
                        "Partial prefill: {} vectors already exist, {} to load (of {} requested)",
                        already_mapped, to_load, self.config.requests
                    );
                }
                
                if to_load == 0 {
                    info!("All {} vectors already loaded, nothing to do", already_mapped);
                    // Return early with empty result
                    return Ok(BenchmarkResult {
                        test_name: workload.to_string(),
                        total_requests: 0,
                        duration: Duration::from_secs(0),
                        throughput: 0.0,
                        histogram: Histogram::new_with_bounds(1, 3_600_000_000, 3)
                            .expect("Failed to create histogram"),
                        recall_stats: super::worker::RecallStats::new(),
                        error_count: 0,
                        node_metrics: Vec::new(),
                    });
                }
                
                to_load
            } else {
                self.config.requests
            }
        } else {
            self.config.requests
        };

        // Create command template (use cluster mode if we have cluster topology)
        let cluster_mode = self.cluster_topology.is_some();
        let template = create_template(
            workload,
            &self.config.key_prefix,
            self.config.data_size,
            self.config.search_config.as_ref(),
            cluster_mode,
        );

        // Build command buffer for pipeline
        let command_buffer = template.build(self.config.pipeline as usize);

        // Create global counters (duration-based or request-count based)
        let counters = Arc::new(if let Some(duration) = self.config.duration_secs {
            GlobalCounters::with_duration(duration)
        } else {
            GlobalCounters::with_requests(effective_requests)
        });

        // Get addresses for workers
        let addresses = self.get_benchmark_addresses();
        if addresses.is_empty() {
            return Err(crate::utils::BenchmarkError::Config(
                "No available server addresses".to_string(),
            ));
        }

        // Calculate clients per thread per node
        // For slot-aware routing, we need at least 1 client per node per thread
        let clients_per_thread = self.config.clients_per_thread() as usize;
        if clients_per_thread == 0 {
            return Err(crate::utils::BenchmarkError::Config(
                format!(
                    "Not enough clients ({}) for threads ({}). Need at least 1 client per thread.",
                    self.config.clients, self.config.threads
                ),
            ));
        }

        // In cluster mode, total clients = clients_per_thread * num_nodes * threads
        let total_clients = clients_per_thread * addresses.len() * self.config.threads as usize;
        if self.cluster_topology.is_some() && total_clients != self.config.clients as usize {
            eprintln!(
                "Note: Using {} total clients ({} per thread × {} nodes × {} threads)",
                total_clients, clients_per_thread, addresses.len(), self.config.threads
            );
        }

        // Spawn event-driven worker threads
        let mut handles: Vec<thread::JoinHandle<EventWorkerResult>> =
            Vec::with_capacity(self.config.threads as usize);

        let start_time = Instant::now();

        // Clone topology for thread sharing
        let topology = self.cluster_topology.clone();

        for worker_id in 0..self.config.threads as usize {
            let config = Arc::clone(&self.config);
            let counters = Arc::clone(&counters);
            let dataset = self.dataset.clone();
            let buffer = command_buffer.clone();
            let addrs = addresses.clone();
            let topo = topology.clone();
            let tag_map = self.cluster_tag_map.clone();
            let wl_type = workload;

            let handle = thread::Builder::new()
                .name(format!("event-worker-{}", worker_id))
                .spawn(move || {
                    // Create event-driven worker with topology for cluster-aware routing
                    match EventWorker::new(
                        worker_id,
                        addrs,
                        clients_per_thread,
                        &config,
                        buffer,
                        topo.as_ref(),
                        tag_map,
                        wl_type,
                    ) {
                        Ok(worker) => worker.run(counters, dataset),
                        Err(e) => {
                            eprintln!("Worker {}: Failed to create: {}", worker_id, e);
                            EventWorkerResult {
                                worker_id,
                                histogram: Histogram::new_with_bounds(1, 3_600_000_000, 3)
                                    .expect("Failed to create histogram"),
                                recall_stats: RecallStats::new(),
                                redirect_count: 0,
                                topology_refresh_count: 0,
                                error_count: 1,
                                requests_processed: 0,
                            }
                        }
                    }
                })
                .expect("Failed to spawn worker thread");

            handles.push(handle);
        }

        // Progress reporting (if not quiet)
        if !self.config.quiet {
            let counters_clone = Arc::clone(&counters);
            let total = self.config.requests;
            let duration_secs = self.config.duration_secs;
            thread::spawn(move || {
                Self::report_progress(&counters_clone, total, duration_secs);
            });
        }

        // Wait for workers to complete
        let results: Vec<EventWorkerResult> = handles
            .into_iter()
            .map(|h| h.join().expect("Worker thread panicked"))
            .collect();

        let duration = start_time.elapsed();

        // Signal shutdown to progress reporter
        counters.signal_shutdown();

        // Merge results
        self.merge_event_results(workload.as_str(), results, duration)
    }

    /// Run all configured tests (uses event-driven I/O by default)
    pub fn run_all(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for test_name in &self.config.tests {
            let workload = WorkloadType::parse(test_name).ok_or_else(|| {
                crate::utils::BenchmarkError::Config(format!("Unknown test: {}", test_name))
            })?;

            println!("\nRunning test: {}", workload);
            // Use event-driven I/O for maximum performance (like C's ae)
            let result = self.run_test_event(workload)?;
            result.print_summary();
            results.push(result);
        }

        Ok(results)
    }

    /// Run all configured tests using blocking I/O (legacy)
    pub fn run_all_blocking(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for test_name in &self.config.tests {
            let workload = WorkloadType::parse(test_name).ok_or_else(|| {
                crate::utils::BenchmarkError::Config(format!("Unknown test: {}", test_name))
            })?;

            println!("\nRunning test: {}", workload);
            let result = self.run_test(workload)?;
            result.print_summary();
            results.push(result);
        }

        Ok(results)
    }

    /// Create vector search index using FT.CREATE
    ///
    /// # Arguments
    /// * `overwrite` - If true, drop existing index first
    pub fn create_search_index(&self, overwrite: bool) -> Result<()> {
        let search_config = self.config.search_config.as_ref().ok_or_else(|| {
            crate::utils::BenchmarkError::Config(
                "Search config required for index creation".to_string(),
            )
        })?;

        let addresses = self.get_benchmark_addresses();
        if addresses.is_empty() {
            return Err(crate::utils::BenchmarkError::Config(
                "No available server addresses".to_string(),
            ));
        }

        // Connect to first available node
        let (host, port) = &addresses[0];
        let mut conn = self.connection_factory.create(host, *port)?;

        info!(
            "Creating index '{}' with algorithm {:?}, dim={}, metric={:?}",
            search_config.index_name,
            search_config.algorithm,
            search_config.dim,
            search_config.distance_metric
        );

        create_index(&mut conn, search_config, overwrite).map_err(|e| {
            crate::utils::BenchmarkError::Config(format!("Failed to create index: {}", e))
        })?;

        info!("Index '{}' created successfully", search_config.index_name);
        Ok(())
    }

    /// Drop vector search index
    pub fn drop_search_index(&self) -> Result<()> {
        let search_config = self.config.search_config.as_ref().ok_or_else(|| {
            crate::utils::BenchmarkError::Config(
                "Search config required for index operations".to_string(),
            )
        })?;

        let addresses = self.get_benchmark_addresses();
        if addresses.is_empty() {
            return Err(crate::utils::BenchmarkError::Config(
                "No available server addresses".to_string(),
            ));
        }

        let (host, port) = &addresses[0];
        let mut conn = self.connection_factory.create(host, *port)?;

        info!("Dropping index '{}'", search_config.index_name);

        drop_index(&mut conn, &search_config.index_name).map_err(|e| {
            crate::utils::BenchmarkError::Config(format!("Failed to drop index: {}", e))
        })?;

        info!("Index '{}' dropped", search_config.index_name);
        Ok(())
    }

    /// Wait for index background indexing to complete
    pub fn wait_for_search_indexing(&self, timeout_secs: u64) -> Result<()> {
        use crate::metrics::backfill::wait_for_backfill;

        let search_config = self.config.search_config.as_ref().ok_or_else(|| {
            crate::utils::BenchmarkError::Config(
                "Search config required for index operations".to_string(),
            )
        })?;

        // Detect engine type from first connection
        let addresses = self.get_benchmark_addresses();
        if addresses.is_empty() {
            return Err(crate::utils::BenchmarkError::Config(
                "No available server addresses".to_string(),
            ));
        }

        let (host, port) = &addresses[0];
        let mut conn = self.connection_factory.create(host, *port)?;
        let engine_type = detect_engine_type(&mut conn);

        // Use cluster-aware backfill wait if we have cluster topology
        if let Some(ref topology) = self.cluster_topology {
            let config = BackfillWaitConfig {
                max_wait: Some(Duration::from_secs(timeout_secs)),
                ..Default::default()
            };

            let index_names = [search_config.index_name.as_str()];
            let conn_factory = &self.connection_factory;

            wait_for_backfill(
                engine_type,
                &index_names,
                &topology.nodes,
                |node| {
                    conn_factory.create(&node.host, node.port).ok()
                },
                &config,
            )
            .map_err(|e| {
                crate::utils::BenchmarkError::Config(format!("Backfill wait failed: {}", e))
            })?;
        } else {
            // Single-node mode: use simple wait
            info!(
                "Waiting for index '{}' to complete indexing (timeout: {}s)",
                search_config.index_name, timeout_secs
            );

            crate::workload::wait_for_indexing(&mut conn, &search_config.index_name, timeout_secs)
                .map_err(|e| {
                    crate::utils::BenchmarkError::Config(format!("Indexing wait failed: {}", e))
                })?;

            info!("Index '{}' is fully indexed", search_config.index_name);
        }

        Ok(())
    }

    /// Get index information
    pub fn get_search_index_info(&self) -> Result<crate::workload::IndexInfo> {
        let search_config = self.config.search_config.as_ref().ok_or_else(|| {
            crate::utils::BenchmarkError::Config(
                "Search config required for index operations".to_string(),
            )
        })?;

        let addresses = self.get_benchmark_addresses();
        if addresses.is_empty() {
            return Err(crate::utils::BenchmarkError::Config(
                "No available server addresses".to_string(),
            ));
        }

        let (host, port) = &addresses[0];
        let mut conn = self.connection_factory.create(host, *port)?;

        get_index_info(&mut conn, &search_config.index_name).map_err(|e| {
            crate::utils::BenchmarkError::Config(format!("Failed to get index info: {}", e))
        })
    }

    /// Export results to JSON file
    pub fn export_json(&self, results: &[BenchmarkResult], path: &Path) -> Result<()> {
        let config_summary = format!(
            "hosts={:?}, clients={}, threads={}, pipeline={}, requests={}",
            self.config.addresses.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
            self.config.clients,
            self.config.threads,
            self.config.pipeline,
            self.config.requests
        );

        let mut benchmark_results = BenchmarkResults::new(&config_summary);

        for result in results {
            let aggregated = crate::metrics::collector::AggregatedMetrics {
                test_name: result.test_name.clone(),
                duration_secs: result.duration.as_secs_f64(),
                total_ops: result.total_requests,
                total_errors: result.error_count,
                throughput: result.throughput,
                mean_latency_ms: result.histogram.mean() / 1000.0,
                p50_latency_ms: result.percentile_ms(50.0),
                p95_latency_ms: result.percentile_ms(95.0),
                p99_latency_ms: result.percentile_ms(99.0),
                p999_latency_ms: result.percentile_ms(99.9),
                max_latency_ms: result.histogram.max() as f64 / 1000.0,
                node_count: result.node_metrics.len().max(1),
            };
            benchmark_results.add_test(aggregated, result.node_metrics.clone());
        }

        benchmark_results.write_json(path).map_err(|e| {
            crate::utils::BenchmarkError::Config(format!("Failed to write JSON: {}", e))
        })
    }

    /// Export results to CSV file
    pub fn export_csv(&self, results: &[BenchmarkResult], path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path).map_err(|e| {
            crate::utils::BenchmarkError::Config(format!("Failed to create CSV file: {}", e))
        })?;

        // Write header
        writeln!(
            file,
            "test_name,duration_secs,total_ops,errors,throughput,mean_ms,p50_ms,p95_ms,p99_ms,p999_ms,max_ms"
        ).map_err(|e| {
            crate::utils::BenchmarkError::Config(format!("Failed to write CSV header: {}", e))
        })?;

        // Write rows
        for result in results {
            writeln!(
                file,
                "{},{:.2},{},{},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
                result.test_name,
                result.duration.as_secs_f64(),
                result.total_requests,
                result.error_count,
                result.throughput,
                result.histogram.mean() / 1000.0,
                result.percentile_ms(50.0),
                result.percentile_ms(95.0),
                result.percentile_ms(99.0),
                result.percentile_ms(99.9),
                result.histogram.max() as f64 / 1000.0
            ).map_err(|e| {
                crate::utils::BenchmarkError::Config(format!("Failed to write CSV row: {}", e))
            })?;
        }

        Ok(())
    }
}

/// Detect engine type from INFO SEARCH response
fn detect_engine_type(conn: &mut crate::client::RawConnection) -> EngineType {
    use crate::utils::{RespEncoder, RespValue};

    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["INFO", "SEARCH"]);

    match conn.execute_encoded(&encoder) {
        Ok(RespValue::BulkString(data)) => {
            let info_str = String::from_utf8_lossy(&data);
            EngineType::detect(&info_str)
        }
        _ => EngineType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_percentiles() {
        let mut histogram =
            Histogram::new_with_bounds(1, 3_600_000_000, 3).expect("Failed to create histogram");

        // Record some values (in microseconds)
        for _ in 0..100 {
            histogram.record(1000).unwrap(); // 1ms
        }
        for _ in 0..10 {
            histogram.record(10000).unwrap(); // 10ms
        }

        let result = BenchmarkResult {
            test_name: "test".to_string(),
            total_requests: 110,
            duration: Duration::from_secs(1),
            throughput: 110.0,
            histogram,
            recall_stats: RecallStats::new(),
            error_count: 0,
            node_metrics: Vec::new(),
        };

        // p50 should be around 1ms
        assert!(result.percentile_ms(50.0) < 2.0);
    }
}
