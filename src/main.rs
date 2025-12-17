//! valkey-search-benchmark - High-performance benchmark tool for Valkey
//!
//! This tool supports standard Redis/Valkey benchmarks as well as
//! vector search (FT.SEARCH) benchmarks with recall verification.

// Allow dead code during development - fields/types will be used in later phases
#![allow(dead_code)]
#![allow(unused_imports)]

use anyhow::Result;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

mod benchmark;
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

fn run() -> Result<()> {
    // Parse CLI arguments
    let args = CliArgs::parse_args();

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
        lower.contains("vec-load")
            || lower.contains("vec-query")
            || lower.contains("vec-del")
            || lower.contains("vec-insert")
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
