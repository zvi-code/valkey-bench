//! Test binary for snapshot/metrics comparison functionality
//!
//! Usage: test-snapshot <host> <port>
//!
//! Takes before/after snapshots of INFO SEARCH metrics and shows diff.

use std::env;
use std::thread;
use std::time::Duration;

use valkey_search_benchmark::client::RawConnection;
use valkey_search_benchmark::metrics::{
    compare_snapshots, default_search_info_fields, print_per_node_diff_all, print_snapshot_diff,
    SnapshotBuilder,
};
use valkey_search_benchmark::utils::{RespEncoder, RespValue};

fn get_info_search(conn: &mut RawConnection) -> Result<String, Box<dyn std::error::Error>> {
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["INFO", "SEARCH"]);

    let reply = conn.execute(&encoder)?;
    match reply {
        RespValue::BulkString(data) => Ok(String::from_utf8_lossy(&data).to_string()),
        RespValue::Error(e) => Err(format!("Server error: {}", e).into()),
        _ => Ok(String::new()),
    }
}

fn run_vector_queries(
    conn: &mut RawConnection,
    index_name: &str,
    dim: usize,
    count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Generate random query vector
    let mut query_vec = Vec::with_capacity(dim * 4);
    for _ in 0..dim {
        let val: f32 = fastrand::f32() * 2.0 - 1.0; // Random -1 to 1
        query_vec.extend_from_slice(&val.to_le_bytes());
    }

    // Build KNN query string
    let knn_query = "*=>[KNN 10 @embedding $BLOB AS score]";

    for i in 0..count {
        // Build FT.SEARCH command with byte slices for the binary blob
        let mut encoder = RespEncoder::with_capacity(256 + query_vec.len());
        encoder.encode_command(&[
            b"FT.SEARCH",
            index_name.as_bytes(),
            knn_query.as_bytes(),
            b"PARAMS",
            b"2",
            b"BLOB",
            &query_vec,
            b"DIALECT",
            b"2",
        ]);

        let reply = conn.execute(&encoder)?;

        // Print first result to show it's working
        if i == 0 {
            if let RespValue::Array(ref arr) = reply {
                if let Some(RespValue::Integer(result_count)) = arr.first() {
                    println!("First query returned {} results", result_count);
                }
            }
        }
    }

    Ok(())
}

/// Get cluster primary nodes
fn get_cluster_primaries(
    conn: &mut RawConnection,
) -> Result<Vec<(String, u16)>, Box<dyn std::error::Error>> {
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["CLUSTER", "NODES"]);

    let reply = conn.execute(&encoder)?;
    let nodes_str = match reply {
        RespValue::BulkString(data) => String::from_utf8_lossy(&data).to_string(),
        RespValue::Error(e) => return Err(format!("Cluster error: {}", e).into()),
        _ => return Ok(vec![]),
    };

    let primaries: Vec<(String, u16)> = nodes_str
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 && parts[2].contains("master") {
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

    Ok(primaries)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <host> <port> [index_name]", args[0]);
        std::process::exit(1);
    }

    let host = &args[1];
    let port: u16 = args[2].parse()?;
    let index_name = args.get(3).map(|s| s.as_str()).unwrap_or("mnist_idx");

    println!("=== Snapshot/Metrics Comparison Test ===\n");
    println!("Connecting to {}:{}...", host, port);

    // Connect to server
    let mut conn = RawConnection::connect_tcp(host, port, Duration::from_secs(5))?;

    // Discover cluster nodes
    let primaries = get_cluster_primaries(&mut conn)?;
    let is_cluster = !primaries.is_empty();

    if is_cluster {
        println!("Discovered {} cluster primary nodes:", primaries.len());
        for (h, p) in &primaries {
            println!("  {}:{}", h, p);
        }
    } else {
        println!("Running in standalone mode");
    }

    // Define fields to track
    let fields = default_search_info_fields();

    println!("\nTracking {} INFO SEARCH fields", fields.len());

    // Take "before" snapshot from all nodes
    println!("\n--- Taking 'before' snapshot ---");
    let mut builder_before = SnapshotBuilder::new("before", fields.clone());

    if is_cluster {
        for (h, p) in &primaries {
            if let Ok(mut node_conn) = RawConnection::connect_tcp(h, *p, Duration::from_secs(5)) {
                if let Ok(info) = get_info_search(&mut node_conn) {
                    let node_id = format!("{}:{}", h, p);
                    builder_before.add_node(&node_id, true, &info);
                }
            }
        }
    } else {
        let info_before = get_info_search(&mut conn)?;
        builder_before.add_node(&format!("{}:{}", host, port), true, &info_before);
    }
    let snapshot_before = builder_before.build();

    println!(
        "Snapshot 'before' captured with {} fields from {} nodes",
        snapshot_before.fields.len(),
        snapshot_before.node_count
    );

    // Print some key values from before
    let key_fields_to_show = [
        "search_total_requests",
        "search_index_scans_count",
        "search_knn_scans_count",
    ];
    for field in &key_fields_to_show {
        if let Some(val) = snapshot_before.get_value(field) {
            println!("  {}: {}", field, val);
        }
    }

    // Run some vector queries to generate metrics changes
    println!("\n--- Running 100 vector queries ---");
    let query_count = 100;
    run_vector_queries(&mut conn, index_name, 784, query_count)?;
    println!("Completed {} queries", query_count);

    // Small delay to let metrics update
    thread::sleep(Duration::from_millis(100));

    // Take "after" snapshot from all nodes
    println!("\n--- Taking 'after' snapshot ---");
    let mut builder_after = SnapshotBuilder::new("after", fields.clone());

    if is_cluster {
        for (h, p) in &primaries {
            if let Ok(mut node_conn) = RawConnection::connect_tcp(h, *p, Duration::from_secs(5)) {
                if let Ok(info) = get_info_search(&mut node_conn) {
                    let node_id = format!("{}:{}", h, p);
                    builder_after.add_node(&node_id, true, &info);
                }
            }
        }
    } else {
        let info_after = get_info_search(&mut conn)?;
        builder_after.add_node(&format!("{}:{}", host, port), true, &info_after);
    }
    let snapshot_after = builder_after.build();

    println!("Snapshot 'after' captured with {} fields", snapshot_after.fields.len());

    // Print some key values from after
    for field in &key_fields_to_show {
        if let Some(val) = snapshot_after.get_value(field) {
            println!("  {}: {}", field, val);
        }
    }

    // Compare snapshots
    println!("\n--- Comparing Snapshots ---");
    let diff = compare_snapshots(&snapshot_before, &snapshot_after, &fields);

    // Print the diff
    print_snapshot_diff(&diff, &fields);

    // Print per-node distribution (shows load spread across cluster nodes)
    print_per_node_diff_all(&diff);

    // Show summary of changes
    println!("=== Summary ===");
    println!("Time elapsed: {:.3}s", diff.elapsed_secs);
    println!("Fields with changes:");
    for field_diff in &diff.fields {
        if field_diff.delta != 0 {
            println!(
                "  {}: {} -> {} (delta: {}, rate: {:?})",
                field_diff.field_name,
                field_diff.old_value,
                field_diff.new_value,
                field_diff.delta,
                field_diff.rate.map(|r| format!("{:.2}/s", r))
            );
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
