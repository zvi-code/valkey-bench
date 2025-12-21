//! valkey-search-benchmark - High-performance benchmark tool for Valkey
//!
//! This tool supports standard Redis/Valkey benchmarks as well as
//! vector search (FT.SEARCH) benchmarks with recall verification.
//!
//! When run with `--cli`, operates as an interactive CLI (like valkey-cli).

// Allow dead code during development - fields/types will be used in later phases
#![allow(dead_code)]
#![allow(unused_imports)]

use anyhow::Result;
use std::sync::Arc;
use tracing::{error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

mod benchmark;
mod cli_mode;
mod client;
mod cluster;
mod config;
mod dataset;
mod metrics;
mod optimizer;
mod utils;
mod workload;

use benchmark::Orchestrator;
use config::{BenchmarkConfig, CliArgs};
use dataset::DatasetContext;
use optimizer::{Constraint, Objectives, Optimizer, TunableParameter};

fn setup_logging(verbose: bool, quiet: bool) {
    let level = if quiet {
        Level::ERROR
    } else if verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");
}

fn print_banner(config: &BenchmarkConfig) {
    if config.quiet {
        return;
    }

    println!("valkey-search-benchmark v{}", env!("CARGO_PKG_VERSION"));
    println!("====================================");
    println!(
        "Hosts: {:?}",
        config
            .addresses
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
    );
    println!(
        "Clients: {}, Threads: {}, Pipeline: {}",
        config.clients, config.threads, config.pipeline
    );
    println!("Requests: {}", config.requests);
    println!("Tests: {:?}", config.tests);
    if config.cluster_mode {
        println!("Cluster mode: enabled, RFR: {:?}", config.read_from_replica);
    }
    if let Some(ref search) = config.search_config {
        println!(
            "Vector search: dim={}, k={}, algo={:?}",
            search.dim, search.k, search.algorithm
        );
    }
    println!("====================================\n");
}

/// Run optimization mode: iteratively test configurations to find optimal parameters
fn run_optimization(
    base_config: &BenchmarkConfig,
    dataset: Option<Arc<DatasetContext>>,
) -> Result<()> {
    // Parse objectives (supports multi-goal with tolerance)
    let objectives = Objectives::parse(&base_config.optimize_objective)
        .map_err(|e| anyhow::anyhow!("Invalid objective: {}", e))?
        .with_tolerance(base_config.optimize_tolerance);

    // Parse constraints
    let mut constraints = Vec::new();
    for constraint_str in &base_config.optimize_constraints {
        let constraint = Constraint::parse(constraint_str)
            .map_err(|e| anyhow::anyhow!("Invalid constraint '{}': {}", constraint_str, e))?;
        constraints.push(constraint);
    }

    // Parse parameters to tune
    let mut parameters = Vec::new();
    for param_str in &base_config.optimize_parameters {
        let param = TunableParameter::parse(param_str)
            .map_err(|e| anyhow::anyhow!("Invalid parameter '{}': {}", param_str, e))?;
        parameters.push(param);
    }

    // If no parameters specified, provide a helpful error
    if parameters.is_empty() {
        return Err(anyhow::anyhow!(
            "No parameters to tune. Use --tune to specify parameters.\n\
             Examples:\n\
             --tune \"clients:10:200:10\"     # Tune clients from 10 to 200 with step 10\n\
             --tune \"threads:1:16:1\"        # Tune threads from 1 to 16\n\
             --tune \"ef_search:10:500:10\"   # Tune ef_search for vector search\n\
             --tune \"pipeline:1:20:1\"       # Tune pipeline depth"
        ));
    }

    // Compute base requests for adaptive duration
    // Use config's request count as base, with a minimum for meaningful measurements
    let base_requests = if base_config.requests >= 100_000 {
        base_config.requests
    } else {
        100_000
    };
    let exploitation_multiplier = 5u64;

    println!("\n=== OPTIMIZATION MODE ===\n");
    println!("Objectives: {}", objectives);
    if objectives.goals.len() > 1 {
        println!(
            "  Tolerance: {:.1}% (configs within this range compared by secondary goals)",
            objectives.tolerance * 100.0
        );
    }
    if !constraints.is_empty() {
        println!("Constraints:");
        for c in &constraints {
            println!("  - {}", c);
        }
    }
    println!("Parameters to tune:");
    for p in &parameters {
        println!("  - {}", p);
    }
    println!("Max iterations: {}", base_config.max_optimize_iterations);
    println!(
        "Adaptive duration: {}K requests (exploration) -> {}K (exploitation)\n",
        base_requests / 1000,
        base_requests * exploitation_multiplier / 1000
    );

    let mut optimizer_builder = Optimizer::builder()
        .objectives(objectives)
        .max_iterations(base_config.max_optimize_iterations)
        .base_requests(base_requests)
        .exploitation_multiplier(exploitation_multiplier as u32);

    for constraint in constraints {
        optimizer_builder = optimizer_builder.constraint(constraint);
    }
    for param in parameters {
        optimizer_builder = optimizer_builder.parameter(param);
    }

    let mut optimizer = optimizer_builder.build();

    // Create base orchestrator for index creation etc.
    let mut orchestrator = Orchestrator::new(base_config.clone())?;
    if let Some(ref ds) = dataset {
        orchestrator.set_dataset_arc(ds.clone());
    }

    // Create index if needed (once before optimization loop)
    let has_vec_workload = base_config.tests.iter().any(|t| t.to_lowercase().starts_with("vec"));
    if has_vec_workload && !base_config.skip_index_create {
        if base_config.search_config.is_some() {
            info!("Creating vector search index...");
            orchestrator.create_search_index(true)?;
            info!("Waiting for index to be ready...");
            orchestrator.wait_for_search_indexing(300)?;
        }
    }
    if has_vec_workload && base_config.search_config.is_some() {
        info!("Building cluster tag map...");
        orchestrator.build_cluster_tag_map()?;
    }

    // Extract shared resources from base orchestrator for iteration reuse
    // This avoids rediscovering cluster topology on each iteration (prevents port exhaustion)
    let shared_topology = orchestrator.cluster_topology().cloned();
    let shared_tag_map = orchestrator.cluster_tag_map();

    // Optimization loop
    let mut iteration = 0;
    while let Some(test_config) = optimizer.next_config() {
        iteration += 1;

        // Get recommended request count for current phase
        // Exploration uses base requests, exploitation uses longer runs for accuracy
        let recommended_requests = optimizer.recommended_requests();

        // Apply test configuration to base config
        let mut run_config = base_config.clone();
        run_config.requests = recommended_requests;
        run_config.quiet = true; // Suppress verbose output during optimization iterations

        if let Some(clients) = test_config.clients {
            run_config.clients = clients;
        }
        if let Some(threads) = test_config.threads {
            run_config.threads = threads;
        }
        if let Some(pipeline) = test_config.pipeline {
            run_config.pipeline = pipeline;
        }
        if let Some(ef_search) = test_config.ef_search {
            if let Some(ref mut search_config) = run_config.search_config {
                search_config.ef_search = Some(ef_search);
            }
        }

        // Create orchestrator with shared topology (avoids rediscovery connections)
        let mut iter_orchestrator = Orchestrator::with_topology(run_config.clone(), shared_topology.clone())?;
        if let Some(ref ds) = dataset {
            iter_orchestrator.set_dataset_arc(ds.clone());
        }
        if let Some(ref tag_map) = shared_tag_map {
            iter_orchestrator.set_cluster_tag_map(tag_map.clone());
        }

        // Run benchmark for the first test only
        let results = iter_orchestrator.run_all()?;

        if let Some(result) = results.first() {
            // Record result with optimizer
            optimizer.record_result(test_config.clone(), result);

            // Print single-line result summary
            // Format: [iteration] phase | config | qps | p99 | recall (if applicable) | status
            let qps_str = if result.throughput >= 1_000_000.0 {
                format!("{:.2}M", result.throughput / 1_000_000.0)
            } else if result.throughput >= 1_000.0 {
                format!("{:.0}K", result.throughput / 1_000.0)
            } else {
                format!("{:.0}", result.throughput)
            };

            let recall = result.recall_stats.average();
            let recall_str = if recall > 0.0 {
                format!(" recall={:.3}", recall)
            } else {
                String::new()
            };

            let best_marker = if optimizer.best_result().map(|b| &b.config) == Some(&test_config) {
                " *BEST*"
            } else {
                ""
            };

            println!(
                "[{:2}] {:?} | {} | {} req/s p99={:.2}ms{}{}",
                iteration,
                optimizer.phase(),
                test_config,
                qps_str,
                result.percentile_ms(99.0),
                recall_str,
                best_marker
            );
        } else {
            warn!("No results from benchmark iteration");
        }
    }

    // Print final summary
    println!("\n{}", optimizer.summary());

    // Print prominent warning if didn't converge
    if optimizer.hit_iteration_limit() {
        eprintln!("\n!!! OPTIMIZATION DID NOT CONVERGE !!!");
        eprintln!("The iteration limit ({}) was reached before completing all phases.", base_config.max_optimize_iterations);
        eprintln!("The best result found may not be optimal.\n");
    }

    // Print best configuration as full command line with expected performance
    if let Some(best) = optimizer.best_result() {
        println!("\n=== Recommended Command Line ===\n");

        // Build the full command line including connection options
        let mut cmd_parts = vec!["./valkey-search-benchmark".to_string()];

        // Add host(s)
        for addr in &base_config.addresses {
            cmd_parts.push(format!("-h {}", addr.host));
        }
        if base_config.addresses.first().map(|a| a.port).unwrap_or(6379) != 6379 {
            cmd_parts.push(format!(
                "-p {}",
                base_config.addresses.first().unwrap().port
            ));
        }

        // Add cluster mode if enabled
        if base_config.cluster_mode {
            cmd_parts.push("--cluster".to_string());
        }

        // Add TLS if enabled
        if base_config.tls.is_some() {
            cmd_parts.push("--tls".to_string());
            if base_config
                .tls
                .as_ref()
                .map(|t| t.skip_verify)
                .unwrap_or(false)
            {
                cmd_parts.push("--tls-skip-verify".to_string());
            }
        }

        // Add test type
        if !base_config.tests.is_empty() {
            cmd_parts.push(format!("-t {}", base_config.tests.join(",")));
        }

        // Add optimized parameters
        if let Some(clients) = best.config.clients {
            cmd_parts.push(format!("-c {}", clients));
        }
        if let Some(threads) = best.config.threads {
            cmd_parts.push(format!("--threads {}", threads));
        }
        if let Some(pipeline) = best.config.pipeline {
            cmd_parts.push(format!("-P {}", pipeline));
        }
        if let Some(ef_search) = best.config.ef_search {
            cmd_parts.push(format!("--ef-search {}", ef_search));
        }

        // Add dataset if used
        if let Some(ref dataset_path) = base_config.dataset_path {
            cmd_parts.push(format!("--dataset {}", dataset_path.display()));
        }

        // Add index name if not default
        if let Some(ref search_config) = base_config.search_config {
            if search_config.index_name != "idx" {
                cmd_parts.push(format!("--search-index {}", search_config.index_name));
            }
        }

        // Add request count suggestion (use a reasonable production run size)
        cmd_parts.push(format!("-n {}", std::cmp::max(base_config.requests, 1_000_000)));

        println!("{}", cmd_parts.join(" "));

        // Print expected performance
        let qps = best.metrics.get(&optimizer::Metric::Qps).unwrap_or(&0.0);
        let p99 = best
            .metrics
            .get(&optimizer::Metric::P99Ms)
            .unwrap_or(&0.0);
        let recall = best.metrics.get(&optimizer::Metric::Recall).unwrap_or(&0.0);

        // Format QPS nicely
        let qps_str = if *qps >= 1_000_000.0 {
            format!("{:.2}M", qps / 1_000_000.0)
        } else if *qps >= 1_000.0 {
            format!("{:.0}K", qps / 1_000.0)
        } else {
            format!("{:.0}", qps)
        };

        println!("\nExpected performance: {} req/sec, p99={:.2}ms", qps_str, p99);
        if *recall > 0.0 {
            println!("                      recall={:.4}", recall);
        }
    }

    Ok(())
}

fn run() -> Result<()> {
    // Parse CLI arguments
    let args = CliArgs::parse_args();

    // Check for CLI mode
    if args.cli_mode {
        // Setup minimal logging for CLI mode
        setup_logging(false, true); // quiet mode

        // If there are trailing args, execute them as a command and exit
        if !args.command_args.is_empty() {
            return cli_mode::run_cli_command(&args, &args.command_args);
        }

        // Otherwise run interactive CLI
        return cli_mode::run_cli_mode(&args);
    }

    // Setup logging
    setup_logging(args.verbose, args.quiet);

    // Build configuration
    let mut config = BenchmarkConfig::from_cli(&args)
        .map_err(|e| anyhow::anyhow!("Configuration error: {}", e))?;

    // Load dataset if specified and update config with dataset dimensions
    let dataset = if let Some(ref dataset_path) = config.dataset_path {
        info!("Loading dataset from: {:?}", dataset_path);
        let dataset = DatasetContext::open(dataset_path)
            .map_err(|e| anyhow::anyhow!("Failed to load dataset: {}", e))?;
        info!("{}", dataset.summary());

        // Update search config with dataset dimension
        if let Some(ref mut search_config) = config.search_config {
            search_config.set_dim(dataset.dim() as u32);
        }

        Some(dataset)
    } else {
        None
    };

    // Print banner
    print_banner(&config);

    // If optimization mode is enabled, run the optimizer
    if config.optimize {
        let dataset_arc = dataset.map(Arc::new);
        return run_optimization(&config, dataset_arc);
    }

    // Create orchestrator
    let mut orchestrator = Orchestrator::new(config.clone())?;

    // Set dataset on orchestrator if loaded
    if let Some(dataset) = dataset {
        orchestrator.set_dataset(dataset);
    }

    // Create index if needed for vector search workloads
    // This matches C behavior: create index whenever --search is enabled with vec-* workloads
    let has_vec_workload = config.tests.iter().any(|t| {
        let lower = t.to_lowercase();
        lower.starts_with("vec")  // Match vecload, vecquery, vecdelete, vec-load, vec-query, etc.
    });

    if has_vec_workload && !config.skip_index_create {
        if let Some(ref _search_config) = config.search_config {
            info!("Creating vector search index...");
            orchestrator.create_search_index(true)?; // overwrite existing

            info!("Waiting for index to be ready...");
            orchestrator.wait_for_search_indexing(300)?; // 5 minute timeout
        }
    }

    // Build cluster tag map for all vector workloads
    // This scans the cluster to discover which vectors exist and their cluster tags
    // Used by:
    // - vec-load: skip keys that already exist (only insert missing)
    // - vec-query: know which vectors exist for recall computation
    // - vec-delete/vec-update: operate on existing vectors
    // If no keys exist, scan completes very fast
    if has_vec_workload {
        if config.search_config.is_some() {
            info!("Building cluster tag map for existing vectors...");
            orchestrator.build_cluster_tag_map()?;
        }
    }

    // Run all tests
    let results = orchestrator.run_all()?;

    // Export to JSON if requested
    if let Some(ref output_path) = config.output_path {
        info!("Writing results to: {:?}", output_path);
        orchestrator.export_json(&results, output_path)?;
    }

    // Export to CSV if requested
    if let Some(ref csv_path) = config.csv_output {
        info!("Writing CSV to: {:?}", csv_path);
        orchestrator.export_csv(&results, csv_path)?;
    }

    // Print summary
    println!("\n====================================");
    println!("BENCHMARK COMPLETE");
    println!("====================================");
    println!("Tests run: {}", results.len());

    let total_requests: u64 = results.iter().map(|r| r.total_requests).sum();
    let total_errors: u64 = results.iter().map(|r| r.error_count).sum();
    println!("Total requests: {}", total_requests);
    println!("Total errors: {}", total_errors);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        error!("Error: {:#}", e);
        std::process::exit(1);
    }
}
