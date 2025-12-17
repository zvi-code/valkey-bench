//! Test binary for FT.INFO parsing
//!
//! Usage: test-ftinfo <host> <port> <index_name>

use std::env;
use std::time::Duration;

use valkey_search_benchmark::client::RawConnection;
use valkey_search_benchmark::metrics::{
    convert_ftinfo_to_lines, convert_memdb_ftinfo_to_lines, EngineType, FtInfoResult,
    get_node_progress, parse_ftinfo_lines,
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

    println!("=== FT.INFO Parsing Test ===\n");
    println!("Connecting to {}:{}...", host, port);

    // Connect to server
    let mut conn = RawConnection::connect_tcp(host, port, Duration::from_secs(5))?;

    // Get server info to detect engine type
    println!("\n--- Detecting Engine Type ---");
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["INFO", "SERVER"]);

    let info_reply = conn.execute(&encoder)?;
    let info_response = match &info_reply {
        RespValue::BulkString(data) => String::from_utf8_lossy(data).to_string(),
        _ => String::new(),
    };

    let engine_type = EngineType::detect(&info_response);
    println!("Detected engine: {:?}", engine_type);

    // Send FT.INFO command
    println!("\n--- Fetching FT.INFO {} ---", index_name);
    let mut encoder = RespEncoder::with_capacity(128);
    encoder.encode_command_str(&["FT.INFO", index_name]);

    let resp_value = conn.execute(&encoder)?;

    // Check for errors
    if let RespValue::Error(ref e) = resp_value {
        println!("Server error: {}", e);
        return Ok(());
    }

    // Convert to lines based on engine type
    println!("\n--- Raw FT.INFO Lines ---");
    let lines = match engine_type {
        EngineType::MemoryDb => convert_memdb_ftinfo_to_lines(&resp_value, None),
        _ => convert_ftinfo_to_lines(&resp_value, None),
    };

    // Print first 50 lines
    for (i, line) in lines.lines().take(50).enumerate() {
        println!("{:3}: {}", i + 1, line);
    }
    if lines.lines().count() > 50 {
        println!("... ({} more lines)", lines.lines().count() - 50);
    }

    // Parse into HashMap
    println!("\n--- Parsed Key Fields ---");
    let parsed = parse_ftinfo_lines(&lines);

    let key_fields = [
        "index_name",
        "num_docs",
        "num_records",
        "backfill_in_progress",
        "backfill_complete_percent",
        "state",
        "index_status",
        "index_degradation_percentage",
        "attributes.dim",
        "attributes.distance_metric",
        "attributes.algorithm",
    ];

    for field in key_fields {
        if let Some(value) = parsed.get(field) {
            println!("  {:35} = {}", field, value);
        }
    }

    // Create FtInfoResult
    println!("\n--- FtInfoResult Struct ---");
    let result = FtInfoResult::from_response(&resp_value, engine_type);
    println!("  index_name:             {:?}", result.index_name);
    println!("  num_docs:               {}", result.num_docs);
    println!("  num_indexed_vectors:    {}", result.num_indexed_vectors);
    println!("  status:                 {:?}", result.status);
    println!("  backfill_in_progress:   {}", result.backfill_in_progress);
    println!("  backfill_complete_pct:  {:.2}%", result.backfill_complete_percent * 100.0);
    println!("  space_usage:            {} bytes", result.space_usage);
    println!("  vector_space_usage:     {} bytes", result.vector_space_usage);
    println!("  is_ready:               {}", result.is_ready());
    println!("  progress_percent:       {}%", result.progress_percent());

    // Test node progress function with fresh connection
    println!("\n--- Testing get_node_progress() ---");
    let mut conn2 = RawConnection::connect_tcp(host, port, Duration::from_secs(5))?;
    match get_node_progress(&mut conn2, index_name, engine_type) {
        Ok(progress) => {
            println!("  num_docs:         {}", progress.num_docs);
            println!("  progress_percent: {}%", progress.progress_percent);
            println!("  in_progress:      {}", progress.in_progress);
            println!("  status:           {:?}", progress.status);
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
