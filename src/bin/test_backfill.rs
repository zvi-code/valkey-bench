//! Test binary for backfill waiting functionality
//!
//! Usage: test-backfill <host> <port> <index_name>

use std::env;
use std::time::Duration;

use valkey_search_benchmark::client::RawConnection;
use valkey_search_benchmark::metrics::{
    wait_for_index_backfill_complete, BackfillWaitConfig, EngineType,
};
use valkey_search_benchmark::utils::{RespEncoder, RespValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <host> <port> <index_name>", args[0]);
        std::process::exit(1);
    }

    let host = &args[1];
    let port: u16 = args[2].parse()?;
    let index_name = &args[3];

    println!("=== Backfill Wait Test ===\n");
    println!("Connecting to {}:{}...", host, port);

    // Connect to server to detect engine type
    let mut conn = RawConnection::connect_tcp(host, port, Duration::from_secs(5))?;

    // Get server info to detect engine type
    println!("Detecting engine type...");
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["INFO", "SERVER"]);

    let info_reply = conn.execute(&encoder)?;
    let info_response = match &info_reply {
        RespValue::BulkString(data) => String::from_utf8_lossy(data).to_string(),
        _ => String::new(),
    };

    let engine_type = EngineType::detect(&info_response);
    println!("Detected engine: {:?}", engine_type);

    // Get cluster nodes
    println!("\nGetting cluster topology...");
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["CLUSTER", "NODES"]);

    let nodes_reply = conn.execute(&encoder)?;
    let nodes_response = match &nodes_reply {
        RespValue::BulkString(data) => String::from_utf8_lossy(data).to_string(),
        _ => String::new(),
    };

    // Parse nodes and get primary hosts
    let primary_hosts: Vec<(String, u16)> = nodes_response
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 && parts[2].contains("master") {
                // Parse host:port@cport format
                let host_port = parts[1].split('@').next()?;
                let mut hp = host_port.split(':');
                let h = hp.next()?;
                let p: u16 = hp.next()?.parse().ok()?;
                Some((h.to_string(), p))
            } else {
                None
            }
        })
        .collect();

    println!("Found {} primary nodes:", primary_hosts.len());
    for (h, p) in &primary_hosts {
        println!("  {}:{}", h, p);
    }

    // Configure backfill wait
    let config = BackfillWaitConfig {
        poll_interval: Duration::from_secs(1),
        initial_delay: Duration::from_millis(500),
        max_wait: Some(Duration::from_secs(120)),
        show_progress: true,
    };

    // Create connection factory
    let node_count = primary_hosts.len();
    let mut connections: Vec<Option<RawConnection>> = primary_hosts
        .iter()
        .filter_map(|(h, p)| RawConnection::connect_tcp(h, *p, Duration::from_secs(5)).ok())
        .map(Some)
        .collect();

    println!("\nWaiting for backfill to complete...\n");

    let index_names = [index_name.as_str()];
    let result = wait_for_index_backfill_complete(
        engine_type,
        &index_names,
        |idx| {
            if idx < connections.len() {
                connections[idx].take()
            } else {
                None
            }
        },
        node_count,
        &config,
    );

    match result {
        Ok(progress) => {
            println!("\n=== Backfill Complete ===");
            println!("  Total docs:          {}", progress.total_docs);
            println!("  Nodes in progress:   {}", progress.nodes_in_progress);
            println!("  Average progress:    {}%", progress.progress_percent);
            println!("  Per-node progress:");
            for np in &progress.node_progress {
                println!(
                    "    {}: {} docs, {}%, {:?}",
                    np.node_id, np.num_docs, np.progress_percent, np.status
                );
            }
        }
        Err(e) => {
            println!("\nBackfill wait failed: {}", e);
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
