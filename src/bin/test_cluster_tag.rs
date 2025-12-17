//! Test binary for cluster tag mapping functionality
//!
//! Usage: test-cluster-tag <host> <port> [prefix]
//!
//! Scans cluster for vector keys and builds tag mapping.

use std::env;
use std::time::Duration;

use valkey_search_benchmark::client::{ControlPlane, RawConnection};
use valkey_search_benchmark::cluster::{
    build_vector_id_mappings, parse_vector_key, ClusterScanConfig, ClusterTagMap, ClusterTopology,
};
use valkey_search_benchmark::utils::{RespEncoder, RespValue};

fn get_cluster_nodes(conn: &mut RawConnection) -> Result<String, Box<dyn std::error::Error>> {
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["CLUSTER", "NODES"]);

    let reply = conn.execute_encoded(&encoder)?;
    match reply {
        RespValue::BulkString(data) => Ok(String::from_utf8_lossy(&data).to_string()),
        RespValue::Error(e) => Err(format!("Cluster error: {}", e).into()),
        _ => Ok(String::new()),
    }
}

fn count_keys_on_node(
    host: &str,
    port: u16,
    pattern: &str,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut conn = RawConnection::connect_tcp(host, port, Duration::from_secs(5))?;

    // Use SCAN to count keys
    let mut count = 0u64;
    let mut cursor: u64 = 0;

    loop {
        let mut encoder = RespEncoder::with_capacity(128);
        encoder.encode_command_str(&[
            "SCAN",
            &cursor.to_string(),
            "MATCH",
            pattern,
            "COUNT",
            "1000",
        ]);

        let reply = conn.execute_encoded(&encoder)?;

        let (new_cursor, keys) = match reply {
            RespValue::Array(arr) if arr.len() == 2 => {
                let cur = match &arr[0] {
                    RespValue::BulkString(s) => {
                        String::from_utf8_lossy(s).parse::<u64>().unwrap_or(0)
                    }
                    RespValue::Integer(i) => *i as u64,
                    _ => 0,
                };

                let key_count = match &arr[1] {
                    RespValue::Array(keys) => keys.len() as u64,
                    _ => 0,
                };

                (cur, key_count)
            }
            _ => (0, 0),
        };

        cursor = new_cursor;
        count += keys;

        if cursor == 0 {
            break;
        }
    }

    Ok(count)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <host> <port> [prefix]", args[0]);
        std::process::exit(1);
    }

    let host = &args[1];
    let port: u16 = args[2].parse()?;
    let prefix = args.get(3).map(|s| s.as_str()).unwrap_or("vec:");

    println!("=== Cluster Tag Mapping Test ===\n");
    println!("Connecting to {}:{}...", host, port);

    // Connect to server
    let mut conn = RawConnection::connect_tcp(host, port, Duration::from_secs(5))?;

    // Discover cluster nodes
    let nodes_str = get_cluster_nodes(&mut conn)?;

    if nodes_str.is_empty() {
        println!("Not a cluster - running standalone test");
        return Ok(());
    }

    let topology = ClusterTopology::from_cluster_nodes(&nodes_str)?;
    println!(
        "Discovered cluster with {} primaries, {} total nodes",
        topology.num_primaries(),
        topology.num_nodes()
    );

    // Print primary addresses
    println!("\nPrimary nodes:");
    for primary in topology.primaries() {
        println!("  {}:{}", primary.host, primary.port);
    }

    // Count keys per node first
    let pattern = format!("{}*", prefix);
    println!("\n--- Key Distribution ---");
    println!("Scanning for pattern: {}", pattern);

    let mut total_keys = 0u64;
    for primary in topology.primaries() {
        let count = count_keys_on_node(&primary.host, primary.port, &pattern)?;
        println!("  {}:{}: {} keys", primary.host, primary.port, count);
        total_keys += count;
    }
    println!("Total keys across all primaries: {}", total_keys);

    if total_keys == 0 {
        println!("\nNo keys found with prefix '{}'. Skipping tag map test.", prefix);
        return Ok(());
    }

    // Build cluster tag map
    println!("\n--- Building Cluster Tag Map ---");
    let capacity = total_keys.max(100_000);
    let tag_map = ClusterTagMap::new(prefix, capacity, true);

    let scan_config = ClusterScanConfig {
        pattern: pattern.clone(),
        batch_size: 1000,
        timeout: Duration::from_secs(5),
        show_progress: true,
    };

    let results = build_vector_id_mappings(&tag_map, &topology.nodes, &scan_config)?;

    println!("\n--- Results ---");
    println!("Total keys scanned: {}", results.total_keys);
    println!("Vectors mapped: {}", tag_map.count());
    println!("Scan time: {}ms", results.total_time_ms);
    println!("Throughput: {:.1} keys/sec", results.keys_per_second);

    // Show sample mappings
    println!("\n--- Sample Mappings (first 10) ---");
    let mut shown = 0;
    for i in 0..capacity as usize {
        if tag_map.vector_exists(i as u64) {
            if let Some(tag) = tag_map.get_tag(i as u64) {
                println!("  Vector {} -> tag {}", i, tag);
                shown += 1;
                if shown >= 10 {
                    break;
                }
            }
        }
    }

    // Test parse_vector_key function
    println!("\n--- Key Parsing Tests ---");
    let test_keys = [
        format!("{}{{ABC}}:000123", prefix),
        format!("{}{{XYZ}}000456", prefix),
        format!("{}{{AAA}}:000001", prefix),
    ];

    for key in &test_keys {
        match parse_vector_key(key, prefix) {
            Some((id, tag)) => println!("  '{}' -> id={}, tag={}", key, id, tag),
            None => println!("  '{}' -> parse failed", key),
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
