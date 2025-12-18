# Valkey Search Benchmark

A high-performance benchmarking tool for Valkey/Redis, with specialized support for vector search operations. Written in Rust for maximum throughput and minimal latency overhead.

## Features

- **High Performance**: Lock-free architecture with thread-local histograms and atomic counters
- **Pipeline Support**: Configurable command pipelining for maximum throughput
- **Cluster Support**: Automatic topology discovery, slot routing, and MOVED/ASK handling
- **Read-From-Replica**: Distribute read traffic across replicas for horizontal scaling
- **Vector Search**: FT.CREATE, FT.SEARCH with recall@k computation against ground truth
- **Multiple Workloads**: PING, GET, SET, HSET, LPUSH, RPUSH, SADD, ZADD, and vector operations
- **Rate Limiting**: Token bucket rate limiter for controlled load testing
- **TLS Support**: Full TLS/SSL support with certificate authentication
- **CLI Mode**: Interactive command-line interface (like valkey-cli)
- **JSON Output**: Machine-readable results for CI/CD integration

## Supported Platforms

Tested and verified on:
- **Amazon MemoryDB** (cluster mode)
- **Amazon ElastiCache for Valkey** (standalone and cluster mode)
- **Open-source Valkey/Redis** (standalone and cluster mode)

## Building

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- C compiler (for some dependencies)

### Build Commands

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Build and run
cargo run --release -- [OPTIONS]
```

The release binary will be at `target/release/valkey-search-benchmark`.

## Usage

### Basic Usage

```bash
# Benchmark PING command
./valkey-search-benchmark -h localhost -p 6379 -t ping

# Benchmark SET/GET with 100 byte values
./valkey-search-benchmark -h localhost -p 6379 -t set,get -d 100

# Cluster mode with auto-discovery
./valkey-search-benchmark -h node1 --cluster -t ping

# Read from replicas for higher throughput
./valkey-search-benchmark -h node1 --cluster --rfr prefer-replica -t get
```

### CLI Mode (Interactive)

Use the `--cli` flag to run as an interactive command-line interface:

```bash
# Interactive mode
./valkey-search-benchmark --cli -h localhost -p 6379

# Non-interactive: execute a single command
./valkey-search-benchmark --cli -h localhost PING
./valkey-search-benchmark --cli -h localhost INFO server
./valkey-search-benchmark --cli -h localhost SCAN 0 COUNT 10

# With TLS
./valkey-search-benchmark --cli -h secure-host --tls --tls-skip-verify PING
```

### Command Line Options

#### Connection Options

| Option | Description | Default |
|--------|-------------|---------|
| `-h, --host <HOST>` | Server hostname (can be repeated) | `127.0.0.1` |
| `-p, --port <PORT>` | Server port | `6379` |
| `-a, --auth <PASSWORD>` | Authentication password | None |
| `--user <USERNAME>` | Authentication username (ACL) | None |
| `--tls` | Enable TLS | `false` |
| `--tls-skip-verify` | Skip TLS certificate verification | `false` |
| `--tls-cert <FILE>` | TLS client certificate | None |
| `--tls-key <FILE>` | TLS client private key | None |
| `--tls-ca-cert <FILE>` | TLS CA certificate | None |
| `--tls-sni <HOST>` | TLS Server Name Indication | None |
| `--dbnum <NUM>` | Database number | `0` |

#### Cluster Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cluster` | Enable cluster mode | `false` |
| `--rfr <STRATEGY>` | Read-from-replica strategy | `primary` |

Read-from-replica strategies:
- `primary` - Always read from primary (default)
- `prefer-replica` - Prefer replicas, fallback to primary
- `round-robin` - Round-robin across all nodes
- `az-affinity` - Prefer same availability zone

#### Benchmark Options

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --tests <TESTS>` | Workload types (comma-separated) | `ping` |
| `-c, --clients <NUM>` | Number of parallel connections | `50` |
| `--threads <NUM>` | Number of worker threads | `4` |
| `-P, --pipeline <NUM>` | Pipeline depth | `1` |
| `-n, --requests <NUM>` | Total number of requests | `100000` |
| `-r, --keyspace <NUM>` | Key space size (random keys 0 to N-1) | `1000000` |
| `-d, --data-size <BYTES>` | Data size for SET/HSET values | `3` |
| `--rps <NUM>` | Rate limit (requests per second) | Unlimited |
| `--sequential` | Use sequential keys instead of random | `false` |
| `--seed <NUM>` | Random seed (0 = random) | `0` |

#### CLI Mode Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cli` | Run in interactive CLI mode | `false` |

When `--cli` is specified, trailing arguments are executed as a command.
If no command is given, starts an interactive REPL.

#### Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `-q, --quiet` | Quiet mode (no progress bar) | `false` |
| `--json <FILE>` | Export results to JSON file | None |

### Workload Types

| Workload | Description |
|----------|-------------|
| `ping` | PING command |
| `set` | SET key value |
| `get` | GET key |
| `hset` | HSET key field value |
| `lpush` | LPUSH key value |
| `rpush` | RPUSH key value |
| `sadd` | SADD key member |
| `zadd` | ZADD key score member |
| `vec-load` | HSET with vector data (requires dataset) |
| `vec-query` | FT.SEARCH vector query (requires dataset and index) |

### Examples

#### Basic Benchmarks

```bash
# Simple PING test
./valkey-search-benchmark -h localhost -p 6379 -t ping -n 100000

# SET/GET with 100 clients, pipeline of 10
./valkey-search-benchmark -h localhost -p 6379 -t set,get -c 100 -P 10 -n 1000000

# Rate-limited test at 10,000 requests/sec
./valkey-search-benchmark -h localhost -p 6379 -t set --rps 10000 -n 100000

# Sequential keys for cache warming
./valkey-search-benchmark -h localhost -p 6379 -t set --sequential -r 1000000 -n 1000000
```

#### Cluster Mode

```bash
# Enable cluster mode
./valkey-search-benchmark -h cluster-node --cluster -t ping

# Read from replicas for higher read throughput
./valkey-search-benchmark -h cluster-node --cluster --rfr prefer-replica -t get -n 1000000

# Round-robin across all nodes (primary + replicas)
./valkey-search-benchmark -h cluster-node --cluster --rfr round-robin -t get
```

#### TLS Connection

```bash
# TLS with certificate verification disabled (for testing)
./valkey-search-benchmark -h secure-host --tls --tls-skip-verify -t ping

# TLS with CA certificate
./valkey-search-benchmark -h secure-host --tls --tls-ca-cert ca.crt -t ping

# TLS with client certificate authentication
./valkey-search-benchmark -h secure-host --tls \
  --tls-cert client.crt --tls-key client.key --tls-ca-cert ca.crt -t ping
```

#### Authentication

```bash
# Password authentication
./valkey-search-benchmark -h localhost -p 6379 -a mypassword -t ping

# ACL authentication (username + password)
./valkey-search-benchmark -h localhost -p 6379 --user myuser -a mypassword -t ping
```

#### JSON Output

```bash
# Export results to JSON
./valkey-search-benchmark -h localhost -p 6379 -t ping,set,get -o results.json
```

Output format:
```json
{
  "config": "hosts=[\"localhost:6379\"], clients=50, threads=4, pipeline=1, requests=100000",
  "tests": [
    {
      "summary": {
        "test_name": "PING",
        "throughput": 245832.5,
        "total_ops": 100000,
        "total_errors": 0,
        "duration_secs": 0.407,
        "latency": {
          "mean_ms": 0.195,
          "p50_ms": 0.183,
          "p95_ms": 0.312,
          "p99_ms": 0.456,
          "p999_ms": 1.234,
          "max_ms": 2.567
        },
        "node_count": 1
      },
      "nodes": []
    }
  ]
}
```

## CLI Mode

The `--cli` flag enables an interactive command-line interface similar to `valkey-cli`:

### Interactive Mode

```bash
./valkey-search-benchmark --cli -h localhost

# Output:
# Connecting to localhost:6379...
# Connected to Valkey localhost:6379 (8.2.0)
# Type 'help' for available commands, 'quit' or Ctrl-D to exit.
#
# localhost:6379> PING
# PONG
# localhost:6379> SET mykey "hello world"
# OK
# localhost:6379> GET mykey
# "hello world"
# localhost:6379> quit
```

### Non-Interactive Mode

Execute commands directly by appending them after the connection options:

```bash
# Simple commands
./valkey-search-benchmark --cli -h localhost PING
./valkey-search-benchmark --cli -h localhost INFO server
./valkey-search-benchmark --cli -h localhost DBSIZE

# Commands with arguments
./valkey-search-benchmark --cli -h localhost SET foo bar
./valkey-search-benchmark --cli -h localhost GET foo
./valkey-search-benchmark --cli -h localhost SCAN 0 COUNT 10

# Cluster commands
./valkey-search-benchmark --cli -h cluster-node CLUSTER INFO
./valkey-search-benchmark --cli -h cluster-node CLUSTER NODES

# Vector search commands
./valkey-search-benchmark --cli -h localhost FT._LIST
./valkey-search-benchmark --cli -h localhost FT.INFO idx
```

### CLI with TLS and Auth

```bash
# TLS connection
./valkey-search-benchmark --cli -h secure-host --tls --tls-skip-verify PING

# With authentication
./valkey-search-benchmark --cli -h localhost -a mypassword INFO server
```

## Vector Search Benchmarking

### Dataset Format

Vector datasets use a binary format with the following structure:

1. **Header** (128 bytes):
   - Magic number: `VDSET001`
   - Dimensions, vector count, query count, neighbor count
   - Data type (f32, f16, i8, u8, binary)
   - Distance metric (L2, Cosine, InnerProduct)

2. **Sections**:
   - Database vectors
   - Query vectors
   - Ground truth neighbor IDs (for recall computation)

### Vector Search Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset <FILE>` | Binary dataset file | None |
| `--index-name <NAME>` | Vector index name | `idx` |
| `--prefix <PREFIX>` | Key prefix for vectors | `vec:` |
| `--algorithm <ALG>` | HNSW or FLAT | `HNSW` |
| `--distance-metric <M>` | L2, COSINE, or IP | `L2` |
| `--ef-construction <N>` | HNSW build parameter | `200` |
| `--ef-runtime <N>` | HNSW search parameter | `10` |
| `--m <N>` | HNSW max connections | `16` |
| `-k <N>` | Number of neighbors to return | `10` |

### Vector Search Examples

```bash
# Load vectors into database
./valkey-search-benchmark -h localhost -p 6379 \
  --dataset vectors.bin \
  --index-name myindex \
  --prefix "doc:" \
  -t vec-load \
  -n 100000

# Run vector search queries with recall computation
./valkey-search-benchmark -h localhost -p 6379 \
  --dataset vectors.bin \
  --index-name myindex \
  --prefix "doc:" \
  --algorithm HNSW \
  --ef-runtime 100 \
  -k 10 \
  -t vec-query \
  -n 10000
```

## Architecture

### Thread Model

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                         │
│  - Discovers cluster topology                           │
│  - Creates workers and distributes clients              │
│  - Collects and merges results                          │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Worker 0    │  │   Worker 1    │  │   Worker N    │
│ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │
│ │ Client 0  │ │  │ │ Client 0  │ │  │ │ Client 0  │ │
│ │ Client 1  │ │  │ │ Client 1  │ │  │ │ Client 1  │ │
│ │    ...    │ │  │ │    ...    │ │  │ │    ...    │ │
│ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │
│ Thread-local  │  │ Thread-local  │  │ Thread-local  │
│  histogram    │  │  histogram    │  │  histogram    │
└───────────────┘  └───────────────┘  └───────────────┘
```

### Cluster Support

- **Auto-discovery**: Connects to seed node and runs `CLUSTER NODES`
- **Slot routing**: CRC16-based slot calculation with hash tag support
- **Read-from-replica**: Distribute connections across replicas for read scaling
- **MOVED handling**: Automatic topology refresh on MOVED errors
- **ASK handling**: Redirect support for slot migration
- **CLUSTERDOWN**: Wait and retry on cluster failures

### Performance Optimizations

- **Lock-free counters**: Atomic operations for request claiming and progress
- **Thread-local histograms**: No contention during latency recording
- **Pre-computed command templates**: RESP encoding done once, placeholders filled at runtime
- **Pipeline batching**: Multiple commands per network round-trip
- **Memory-mapped datasets**: Zero-copy vector access for large datasets
- **Connection distribution**: Even distribution across nodes for replica reads

## Comparison with redis-benchmark

| Feature | valkey-search-benchmark | redis-benchmark |
|---------|------------------------|-----------------|
| Vector search | Yes | No |
| Recall computation | Yes | No |
| Read-from-replica | Yes | No |
| CLI mode | Yes | No |
| Cluster MOVED handling | Yes (with refresh) | Limited |
| JSON output | Yes | Yes |
| TLS support | Yes | Yes |
| Custom workloads | Extensible | Limited |
| Language | Rust | C |

## Troubleshooting

### Connection Issues

```bash
# Test basic connectivity with CLI mode
./valkey-search-benchmark --cli -h localhost PING

# Test benchmark mode
./valkey-search-benchmark -h localhost -p 6379 -t ping -n 1

# Enable debug logging
RUST_LOG=debug ./valkey-search-benchmark -h localhost -p 6379 -t ping
```

### Cluster Issues

If you see MOVED errors in CLI mode:
- This is expected for key commands - CLI mode doesn't auto-redirect
- Use benchmark mode with `--cluster` for automatic slot handling

In benchmark mode:
- The benchmark automatically refreshes cluster topology
- Check that all cluster nodes are accessible
- Verify cluster is not resharding

### Performance Issues

- Increase pipeline depth (`-P 10` or higher)
- Increase client count (`-c 100`)
- Increase thread count (`--threads 8`)
- Enable read-from-replica for read workloads (`--rfr prefer-replica`)
- Use release build (`cargo build --release`)

## License

This project is part of the Valkey ecosystem.
