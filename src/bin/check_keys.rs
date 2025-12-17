//! Simple utility to check key format

use std::env;
use std::time::Duration;

use valkey_search_benchmark::client::RawConnection;
use valkey_search_benchmark::utils::{RespEncoder, RespValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <host> <port> [pattern]", args[0]);
        std::process::exit(1);
    }

    let host = &args[1];
    let port: u16 = args[2].parse()?;
    let pattern = args.get(3).map(|s| s.as_str()).unwrap_or("*");

    let mut conn = RawConnection::connect_tcp(host, port, Duration::from_secs(5))?;

    let mut encoder = RespEncoder::with_capacity(128);
    encoder.encode_command_str(&["SCAN", "0", "MATCH", pattern, "COUNT", "10"]);

    let reply = conn.execute(&encoder)?;

    match reply {
        RespValue::Array(arr) if arr.len() == 2 => {
            let cursor = match &arr[0] {
                RespValue::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                RespValue::Integer(i) => i.to_string(),
                _ => "?".to_string(),
            };

            println!("Cursor: {}", cursor);

            if let RespValue::Array(keys) = &arr[1] {
                println!("Found {} keys:", keys.len());
                for key in keys {
                    if let RespValue::BulkString(k) = key {
                        println!("  '{}'", String::from_utf8_lossy(k));
                    }
                }
            }
        }
        _ => {
            println!("Unexpected reply: {:?}", reply);
        }
    }

    Ok(())
}
