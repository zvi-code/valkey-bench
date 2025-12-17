//! Flush all nodes in a cluster

use std::env;
use std::time::Duration;

use valkey_search_benchmark::client::RawConnection;
use valkey_search_benchmark::cluster::ClusterTopology;
use valkey_search_benchmark::utils::{RespEncoder, RespValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <host> <port>", args[0]);
        std::process::exit(1);
    }

    let host = &args[1];
    let port: u16 = args[2].parse()?;

    println!("Connecting to {}:{}...", host, port);
    let mut conn = RawConnection::connect_tcp(host, port, Duration::from_secs(5))?;

    // Get cluster nodes
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["CLUSTER", "NODES"]);
    let reply = conn.execute(&encoder)?;

    let nodes_str = match reply {
        RespValue::BulkString(data) => String::from_utf8_lossy(&data).to_string(),
        _ => {
            println!("Not a cluster, flushing single node...");
            let mut encoder = RespEncoder::with_capacity(64);
            encoder.encode_command_str(&["FLUSHALL"]);
            let reply = conn.execute(&encoder)?;
            println!("Result: {:?}", reply);
            return Ok(());
        }
    };

    let topology = ClusterTopology::from_cluster_nodes(&nodes_str)?;
    println!("Found {} primaries", topology.num_primaries());

    for primary in topology.primaries() {
        println!("Flushing {}:{}...", primary.host, primary.port);
        match RawConnection::connect_tcp(&primary.host, primary.port, Duration::from_secs(5)) {
            Ok(mut node_conn) => {
                let mut encoder = RespEncoder::with_capacity(64);
                encoder.encode_command_str(&["FLUSHALL"]);
                match node_conn.execute(&encoder) {
                    Ok(reply) => println!("  {:?}", reply),
                    Err(e) => println!("  Error: {}", e),
                }
            }
            Err(e) => println!("  Connection failed: {}", e),
        }
    }

    println!("Done!");
    Ok(())
}
