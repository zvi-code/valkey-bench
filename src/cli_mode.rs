//! Interactive CLI mode - valkey-cli alternative
//!
//! When `--cli` is specified, the benchmark tool operates as an interactive
//! command-line interface to Valkey/Redis, similar to valkey-cli.

use std::io::{self, BufRead, Write};
use std::time::Duration;

use crate::client::control_plane::{ControlPlane, ControlPlaneExt};
use crate::client::raw_connection::{ConnectionFactory, RawConnection};
use crate::config::{BenchmarkConfig, CliArgs, ServerAddress};
use crate::utils::RespValue;

/// Run interactive CLI mode
pub fn run_cli_mode(args: &CliArgs) -> anyhow::Result<()> {
    // Build connection factory from CLI args
    let factory = build_connection_factory(args);

    // Connect to the first host
    let host = args.hosts.first().map(|s| s.as_str()).unwrap_or("127.0.0.1");
    let port = args.port;

    eprintln!("Connecting to {}:{}...", host, port);

    let mut conn = factory
        .create(host, port)
        .map_err(|e| anyhow::anyhow!("Connection failed: {}", e))?;

    // Check connection with PING
    if !ControlPlaneExt::ping(&mut conn)? {
        return Err(anyhow::anyhow!("Server did not respond to PING"));
    }

    // Print connection info
    let server_info = get_server_info(&mut conn);
    eprintln!(
        "Connected to {} {}:{} ({})",
        server_info.server_type,
        host,
        port,
        server_info.version
    );
    eprintln!("Type 'help' for available commands, 'quit' or Ctrl-D to exit.\n");

    // Run the REPL
    run_repl(&mut conn, host, port)
}

/// Run CLI mode with a command from arguments (non-interactive)
pub fn run_cli_command(args: &CliArgs, command_args: &[String]) -> anyhow::Result<()> {
    let factory = build_connection_factory(args);
    let host = args.hosts.first().map(|s| s.as_str()).unwrap_or("127.0.0.1");
    let port = args.port;

    let mut conn = factory
        .create(host, port)
        .map_err(|e| anyhow::anyhow!("Connection failed: {}", e))?;

    // Execute the command
    let args_refs: Vec<&str> = command_args.iter().map(|s| s.as_str()).collect();
    let response = conn.execute(&args_refs)?;

    // Print the response
    print_response(&response, 0);

    Ok(())
}

/// Build connection factory from CLI args
fn build_connection_factory(args: &CliArgs) -> ConnectionFactory {
    use crate::config::TlsConfig;

    let tls_config = if args.tls {
        Some(TlsConfig {
            skip_verify: args.tls_skip_verify,
            ca_cert: args.tls_ca_cert.clone(),
            client_cert: args.tls_cert.clone(),
            client_key: args.tls_key.clone(),
            sni: args.tls_sni.clone(),
        })
    } else {
        None
    };

    ConnectionFactory {
        connect_timeout: Duration::from_millis(args.connect_timeout_ms),
        read_timeout: Duration::from_millis(args.request_timeout_ms),
        write_timeout: Duration::from_millis(args.request_timeout_ms),
        tls_config,
        auth_password: args.password.clone(),
        auth_username: args.username.clone(),
        dbnum: args.dbnum,
    }
}

/// Server information
struct ServerInfo {
    server_type: String,
    version: String,
}

fn get_server_info(conn: &mut RawConnection) -> ServerInfo {
    // Try to get server info
    let info = ControlPlaneExt::info(conn, "server").unwrap_or_default();

    let mut server_type = "Valkey".to_string();
    let mut version = "unknown".to_string();

    for line in info.lines() {
        if let Some(v) = line.strip_prefix("redis_version:") {
            version = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("valkey_version:") {
            version = v.trim().to_string();
            server_type = "Valkey".to_string();
        }
    }

    // Check if it's Redis or Valkey based on version string
    if version.contains("valkey") || info.contains("valkey_version") {
        server_type = "Valkey".to_string();
    } else if version.len() > 0 && version.chars().next().unwrap().is_ascii_digit() {
        // Numeric version likely means Redis
        if !info.contains("valkey") {
            server_type = "Redis".to_string();
        }
    }

    ServerInfo {
        server_type,
        version,
    }
}

/// Run the Read-Eval-Print Loop
fn run_repl(conn: &mut RawConnection, host: &str, port: u16) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Print prompt
        print!("{}:{}> ", host, port);
        stdout.flush()?;

        // Read line
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF (Ctrl-D)
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                continue;
            }
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Handle special commands
        match line.to_lowercase().as_str() {
            "quit" | "exit" => break,
            "help" => {
                print_help();
                continue;
            }
            "clear" => {
                // Clear screen (ANSI escape)
                print!("\x1b[2J\x1b[H");
                stdout.flush()?;
                continue;
            }
            _ => {}
        }

        // Parse command and arguments
        let args = parse_command_line(line);
        if args.is_empty() {
            continue;
        }

        // Execute command
        let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        match conn.execute(&args_refs) {
            Ok(response) => {
                print_response(&response, 0);
            }
            Err(e) => {
                eprintln!("(error) {}", e);
            }
        }
    }

    Ok(())
}

/// Parse a command line into arguments, handling quotes
fn parse_command_line(line: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escape_next = false;

    for ch in line.chars() {
        if escape_next {
            current.push(ch);
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if !in_single_quote => {
                escape_next = true;
            }
            '\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
            }
            '"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
            }
            ' ' | '\t' if !in_single_quote && !in_double_quote => {
                if !current.is_empty() {
                    args.push(std::mem::take(&mut current));
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    if !current.is_empty() {
        args.push(current);
    }

    args
}

/// Print a RESP response with proper formatting
fn print_response(resp: &RespValue, indent: usize) {
    let prefix = "  ".repeat(indent);

    match resp {
        RespValue::SimpleString(s) => {
            println!("{}{}", prefix, s);
        }
        RespValue::Error(e) => {
            println!("{}(error) {}", prefix, e);
        }
        RespValue::Integer(n) => {
            println!("{}(integer) {}", prefix, n);
        }
        RespValue::BulkString(data) => {
            match String::from_utf8(data.clone()) {
                Ok(s) => {
                    // Check if it's multi-line (like INFO output)
                    if s.contains('\n') {
                        for line in s.lines() {
                            println!("{}{}", prefix, line);
                        }
                    } else {
                        println!("{}\"{}\"", prefix, s);
                    }
                }
                Err(_) => {
                    // Binary data - show as hex
                    println!("{}(binary) {} bytes", prefix, data.len());
                }
            }
        }
        RespValue::Array(items) => {
            if items.is_empty() {
                println!("{}(empty array)", prefix);
            } else {
                for (i, item) in items.iter().enumerate() {
                    print!("{}{}) ", prefix, i + 1);
                    print_array_item(item, indent);
                }
            }
        }
        RespValue::Null => {
            println!("{}(nil)", prefix);
        }
    }
}

/// Print an array item (inline for simple types)
fn print_array_item(resp: &RespValue, indent: usize) {
    match resp {
        RespValue::SimpleString(s) => println!("{}", s),
        RespValue::Error(e) => println!("(error) {}", e),
        RespValue::Integer(n) => println!("(integer) {}", n),
        RespValue::BulkString(data) => {
            match String::from_utf8(data.clone()) {
                Ok(s) => println!("\"{}\"", s),
                Err(_) => println!("(binary) {} bytes", data.len()),
            }
        }
        RespValue::Null => println!("(nil)"),
        RespValue::Array(_) => {
            println!();
            print_response(resp, indent + 1);
        }
    }
}

/// Print help message
fn print_help() {
    println!(
        r#"
valkey-bench-rs CLI mode
================================

This is an interactive command-line interface to Valkey/Redis.
You can type any Valkey/Redis command directly.

Built-in commands:
  help     Show this help message
  quit     Exit the CLI (or use Ctrl-D)
  exit     Exit the CLI
  clear    Clear the screen

Example commands:
  PING                       Check connection
  SET key value              Set a key
  GET key                    Get a key
  INFO server                Get server info
  KEYS *                     List all keys (use with caution!)
  SCAN 0                     Iterate keys
  FT._LIST                   List search indexes
  FT.INFO idx                Get index info
  FT.SEARCH idx "*"          Search query

Tip: Use quotes for values with spaces: SET key "hello world"
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_command() {
        assert_eq!(parse_command_line("SET key value"), vec!["SET", "key", "value"]);
    }

    #[test]
    fn test_parse_quoted_value() {
        assert_eq!(
            parse_command_line("SET key \"hello world\""),
            vec!["SET", "key", "hello world"]
        );
    }

    #[test]
    fn test_parse_single_quoted() {
        assert_eq!(
            parse_command_line("SET key 'hello world'"),
            vec!["SET", "key", "hello world"]
        );
    }

    #[test]
    fn test_parse_escaped() {
        assert_eq!(
            parse_command_line(r#"SET key hello\ world"#),
            vec!["SET", "key", "hello world"]
        );
    }

    #[test]
    fn test_parse_empty() {
        assert!(parse_command_line("").is_empty());
        assert!(parse_command_line("   ").is_empty());
    }
}
