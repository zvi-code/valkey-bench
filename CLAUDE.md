# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code Behavior

- **No Claude attribution**: Do not add Claude attribution, signatures, or co-author lines to git commits or any documents/files.
- **No emoji**: Do not use emoji in commits, code, or documentation unless explicitly requested.

## Repository Overview

This repository contains **valkey-bench-rs** - a high-performance benchmarking tool for Valkey/Redis with specialized support for vector search operations. Written in Rust for maximum throughput and minimal latency overhead.

```
valkey-bench-rs/
├── src/                    # Rust source code
├── prep_datasets/          # Python scripts for dataset management
├── bench/                  # Benchmark utilities, wrappers, and perf tools
├── datasets/               # Binary dataset files (generated)
├── docs/                   # Design documents
├── Cargo.toml
├── README.md               # Main user documentation
├── INSTALLATION.md         # Setup and build instructions
├── DATASETS.md             # Dataset download and conversion guide
├── BENCHMARKING.md         # Benchmark usage guide
├── ADVANCED.md             # Optimizer, metadata filtering, binary format
├── EXAMPLES.md             # Comprehensive feature examples
├── THIRD_PARTY_LICENSES.md # Third-party licenses and dataset attributions
└── CLAUDE.md               # This file
```

## Architecture

The benchmark uses a custom mio-based event-driven architecture with zero external Redis client dependencies:

```
src/
├── main.rs                   # Entry point, CLI dispatch
├── lib.rs                    # Library exports
├── cli_mode.rs               # Interactive CLI mode (valkey-cli like)
├── client/                   # Connection layer
│   ├── control_plane.rs      # ControlPlane trait
│   ├── raw_connection.rs     # Direct TCP/TLS connections
│   └── benchmark_client.rs   # High-performance benchmark client
├── cluster/                  # Cluster topology management
│   ├── topology.rs           # Node discovery (CLUSTER NODES)
│   ├── topology_manager.rs   # Dynamic refresh on MOVED/ASK
│   ├── node.rs               # ClusterNode representation
│   ├── backend.rs            # Backend detection (EC, MemoryDB, OSS)
│   ├── cluster_tag_map.rs    # Vector ID to cluster tag mapping
│   └── protected_ids.rs      # Protected IDs for deletion benchmarks
├── benchmark/                # Benchmark execution
│   ├── orchestrator.rs       # Thread spawning, result collection
│   ├── event_worker.rs       # mio-based worker implementation
│   └── counters.rs           # Global atomic counters
├── metrics/                  # Metrics collection and reporting
│   ├── collector.rs          # Metrics aggregation
│   ├── reporter.rs           # Output formatting (text/JSON/CSV)
│   ├── node_metrics.rs       # Per-node metrics tracking
│   ├── snapshot.rs           # Cluster snapshot comparison
│   ├── ft_info.rs            # FT.INFO parsing (EC/MemoryDB)
│   ├── info_fields.rs        # INFO field parsing strategies
│   └── backfill.rs           # Index backfill progress monitoring
├── workload/                 # Command templates and workload types
│   ├── workload_type.rs      # WorkloadType enum
│   ├── lifecycle.rs          # Workload trait and preparation
│   ├── context.rs            # Workload context types
│   ├── command_template.rs   # Command template with placeholders
│   ├── template_factory.rs   # Template creation factory
│   ├── key_format.rs         # Key formatting and cluster tags
│   ├── addressable.rs        # Addressable spaces (key/hash/json)
│   ├── iteration.rs          # Iteration strategies (seq/random/zipfian)
│   ├── parallel.rs           # Parallel workload execution
│   ├── composite.rs          # Composite sequential workload
│   ├── search_ops.rs         # FT.CREATE, FT.SEARCH operations
│   ├── numeric_field.rs      # Numeric field configuration
│   └── tag_distribution.rs   # Tag generation with probabilities
├── dataset/                  # Memory-mapped binary dataset access
│   ├── binary_dataset.rs     # Dataset context and loading
│   ├── header.rs             # Dataset header format
│   └── source.rs             # Data source traits
├── optimizer/                # Adaptive parameter tuning
│   └── optimizer.rs          # Multi-objective optimizer
├── config/                   # Configuration
│   ├── cli.rs                # CLI argument parsing (clap)
│   ├── benchmark_config.rs   # Main configuration struct
│   ├── search_config.rs      # Vector search configuration
│   ├── tls_config.rs         # TLS configuration
│   └── workload_config.rs    # Workload configuration
└── utils/                    # Utilities
    ├── resp.rs               # RESP protocol encoder/decoder
    └── error.rs              # Error types
```

### Key Design Features

- **Zero External Dependencies**: Custom RESP protocol codec, no redis-rs or external clients
- **Event-Driven I/O**: mio polling with non-blocking sockets
- **Lock-Free Hot Path**: Atomic counters, thread-local HDR histograms
- **Memory Efficiency**: Pre-allocated buffers, in-place placeholder replacement
- **Cluster Native**: Full topology discovery, slot routing, MOVED/ASK handling

### Build Commands

```bash
# Standard release build
cargo build --release

# Run tests
cargo test

# Build with rustls TLS backend (alternative to native-tls)
cargo build --release --features rustls-backend
```

### CLI Mode

The benchmark tool includes an interactive CLI mode for direct cluster interaction (similar to valkey-cli):

```bash
# Non-interactive: execute a single command
./target/release/valkey-bench-rs --cli -h CLUSTER_HOST PING
./target/release/valkey-bench-rs --cli -h CLUSTER_HOST INFO server
./target/release/valkey-bench-rs --cli -h CLUSTER_HOST CLUSTER INFO
./target/release/valkey-bench-rs --cli -h CLUSTER_HOST "FT._LIST"
./target/release/valkey-bench-rs --cli -h CLUSTER_HOST "FT.INFO index_name"

# Interactive mode
./target/release/valkey-bench-rs --cli -h CLUSTER_HOST

# With TLS
./target/release/valkey-bench-rs --cli -h CLUSTER_HOST --tls --tls-skip-verify PING

# With authentication
./target/release/valkey-bench-rs --cli -h CLUSTER_HOST -a PASSWORD PING
```

### Benchmark Commands

```bash
# Basic GET/SET benchmark
./target/release/valkey-bench-rs -h CLUSTER_HOST --cluster -t get,set -n 1000000 -c 100

# With pipeline
./target/release/valkey-bench-rs -h CLUSTER_HOST --cluster -t get -n 1000000 -c 100 -P 10

# Vector load (HSET with vector data)
./target/release/valkey-bench-rs -h CLUSTER_HOST --cluster -t vec-load --dataset mnist.bin --prefix "vec:" -n 60000 -c 100

# Vector query (FT.SEARCH)
./target/release/valkey-bench-rs -h CLUSTER_HOST --cluster -t vec-query --dataset mnist.bin --index-name idx -k 10 -n 10000 -c 50
```

### Command Line Options

#### Core Options

| Option | Description | Default |
|--------|-------------|---------|
| `-h, --host` | Server hostname (repeatable for multiple hosts) | 127.0.0.1 |
| `-p, --port` | Server port | 6379 |
| `-a, --auth` | Authentication password | None |
| `--user` | Username for ACL auth | None |
| `--cluster` | Enable cluster mode | false |
| `--rfr` | Read-from-replica: `primary`, `prefer-replica`, `round-robin` | primary |

#### Benchmark Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --tests` | Workload types (comma-separated) | ping |
| `-c, --clients` | Number of parallel connections | 50 |
| `--threads` | Number of worker threads | auto |
| `-P, --pipeline` | Pipeline depth | 1 |
| `-n, --requests` | Total number of requests | 100000 |
| `-r, --keyspace` | Key space size | 1000000 |
| `-d, --data-size` | Data size for SET/HSET values | 3 |
| `--rps` | Rate limit (requests per second) | Unlimited |
| `--sequential` | Use sequential keys | false |

#### Workload Options

| Option | Description | Default |
|--------|-------------|---------|
| `--parallel` | Mixed workload: `"get:0.8,set:0.2"` (weights) | None |
| `--composite` | Sequential phases: `"vec-load:10000,vec-query:1000"` | None |
| `--iteration` | Strategy: `sequential`, `random:seed`, `subset:start:end`, `zipfian:skew:seed` | sequential |
| `--address-type` | Address type: `key:prefix`, `hash:prefix:field1,field2`, `json:prefix:$.path` | key |

#### Vector Search Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Binary dataset file for vector ops | None |
| `--search-index` | FT.SEARCH index name | idx |
| `--search-prefix` | Key prefix for vectors | vec: |
| `--search-vector-field` | Vector field name | embedding |
| `--search-algorithm` | HNSW or FLAT | HNSW |
| `--search-distance` | L2, COSINE, or IP | L2 |
| `-k, --search-k` | KNN neighbors to return | 10 |
| `--ef-construction` | HNSW build parameter | 200 |
| `--ef-search` | HNSW search parameter | 10 |
| `--hnsw-m` | HNSW max connections | 16 |
| `--nocontent` | Return only keys in FT.SEARCH | false |

#### Tag and Numeric Field Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tag-field` | Tag field name for filtered search | None |
| `--search-tags` | Tag distribution: `"tag1:prob1,tag2:prob2"` (prob 0-100) | None |
| `--tag-filter` | Tag filter for vec-query: `"tag1\|tag2"` | None |
| `--numeric-field-config` | Numeric field: `"name:type:dist:params"` (repeatable) | None |
| `--numeric-filter` | Numeric filter: `"field:[min,max]"` (repeatable) | None |

**Numeric field config format:** `name:type:distribution:params`
- Types: `int`, `float:N` (N decimals), `unix_timestamp`, `iso_datetime`, `date_only`
- Distributions: `uniform:min:max`, `zipfian:skew:min:max`, `normal:mean:stddev`, `sequential:start:step`, `constant:value`, `key_based:min:max`

#### Optimizer Options

| Option | Description | Default |
|--------|-------------|---------|
| `--optimize` | Enable optimization mode | false |
| `--objective` | Goal(s): `"maximize:qps"`, `"maximize:qps,minimize:p99_ms"` | maximize:qps |
| `--tolerance` | Multi-objective equivalence tolerance | 0.04 |
| `--constraint` | Constraint (repeatable): `"recall:gt:0.95"`, `"p99_ms:lt:1.0"` | None |
| `--tune` | Parameter to tune (repeatable): `"clients:10:300:10"` | None |
| `--max-optimize-iterations` | Max optimizer iterations | 50 |

## Testing Guidelines

### Unit Tests

```bash
cd valkey-bench-rs
cargo test
```

All 228 tests should pass. Run this before committing any changes.

### Performance Testing with Real Clusters

Performance testing must be done against a real cluster to validate throughput and latency. Use the test cluster:

```
Cluster: zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com
```

#### Test Procedure for GET/SET Performance

1. **Prepare the cluster** (flush existing data):
   ```bash
   # Drop all vector indexes
   ./target/release/valkey-bench-rs --cli -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com "FT._LIST"
   # For each index found:
   ./target/release/valkey-bench-rs --cli -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com "FT.DROPINDEX index_name"

   # Flush all data
   ./target/release/valkey-bench-rs --cli -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com FLUSHALL
   ```

2. **Prefill the database** with 3M keys, 500-byte values:
   ```bash
   ./target/release/valkey-bench-rs -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com --cluster -t set -n 3000000 -r 3000000 -d 500 -c 100 --threads 16
   ```

3. **Run GET benchmark** (after data is loaded):
   ```bash
   ./target/release/valkey-bench-rs -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com --cluster -t get -n 3000000 -r 3000000 -c 500 --threads 52
   ```

4. **Run SET benchmark**:
   ```bash
   ./target/release/valkey-bench-rs -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com --cluster -t set -n 3000000 -r 3000000 -d 500 -c 200 --threads 52
   ```

#### Expected Performance (single CME cluster, node type cache.r7g.16xlarge, 3M keys with 500B values)

| Workload | Throughput |
|----------|------------|
| GET (warm) | ~980K req/sec |
| SET | ~380K req/sec |

Note: First GET run may show lower numbers due to cold cache. Run twice for accurate warm cache numbers.

#### Vector Search Testing

1. **Prepare cluster** (same flush steps as above)

2. **Create index** (via CLI):
   ```bash
   ./target/release/valkey-bench-rs --cli -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com "FT.CREATE mnist_idx ON HASH PREFIX 1 vec: SCHEMA embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 784 DISTANCE_METRIC L2"
   ```

3. **Load vectors**:
   ```bash
   ./target/release/valkey-bench-rs -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com --cluster -t vec-load --dataset datasets/mnist.bin --prefix "vec:" -n 60000 -c 100 --threads 16
   ```

4. **Query vectors**:
   ```bash
   ./target/release/valkey-bench-rs -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com --cluster -t vec-query --dataset datasets/mnist.bin --index-name mnist_idx -k 10 -n 10000 -c 50 --threads 16
   ```

### Parameter Optimizer Testing

The optimizer automatically finds optimal configuration for benchmark parameters:

```bash
# Maximize QPS for GET workload, tune clients and threads
./target/release/valkey-bench-rs -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com --cluster -t get -n 100000 --optimize --objective "maximize:qps" --tune "clients:10:300:10" --tune "threads:1:32:1"

# With p99 latency constraint
./target/release/valkey-bench-rs -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com --cluster -t get -n 100000 --optimize --objective "maximize:qps" --constraint "p99_ms:lt:1.0" --tune "clients:10:300:10" --tune "threads:1:32:1"

# Vector search: maximize QPS with recall constraint
./target/release/valkey-bench-rs -h zvi-vss-16xl.ajfdds.clustercfg.euw1devo.cache.amazonaws.com --cluster -t vec-query --dataset datasets/mnist.bin --index-name mnist_idx -n 10000 --optimize --objective "maximize:qps" --constraint "recall:gt:0.95" --tune "ef_search:10:500:10"
```

#### Optimizer CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--optimize` | Enable optimization mode | false |
| `--objective` | Goal(s): "maximize:qps,minimize:p99_ms" | maximize:qps |
| `--tolerance` | Equivalence tolerance for multi-goal (0.04 = 4%) | 0.04 |
| `--constraint` | Constraints (repeatable): "recall:gt:0.95" | None |
| `--tune` | Parameters (repeatable): "clients:10:300:10" | None |
| `--max-optimize-iterations` | Max iterations before stopping | 50 |

#### Objective Formats

- **Single**: `maximize:qps` or `minimize:p99_ms`
- **Multi-goal**: `maximize:qps,minimize:p99_ms` (ordered, tolerance-based tiebreaker)
- **Bounded**: `maximize:qps:lt:1000000` (max QPS but must be < 1M)

#### Optimizer Phases

1. **Feasibility**: Tests maximum values to find upper bound
2. **Exploration**: Grid sampling (min, 25%, 50%, 75%, max) to map parameter space
3. **Exploitation**: Hill climbing with 1x, 2x, 3x step sizes in all directions

**Adaptive Duration**: Exploration uses base requests (100K), exploitation uses 5x (500K) for accuracy.

**Output Format**: Single line per iteration: `[iter] phase | config | qps p99 [recall] [*BEST*]`

#### Expected Optimizer Results (16-node cluster, 3M keys, 500B values)

| Workload | Optimal Config | Throughput |
|----------|----------------|------------|
| GET | clients=275, threads=24 | ~1.04M req/sec |
| SET | clients=200, threads=16 | ~380K req/sec |

#### Convergence

The optimizer prints a warning if it hits the iteration limit without converging. In that case, increase `--max-optimize-iterations` or narrow the parameter ranges.

### Verifying Changes

Before committing changes:

1. Run unit tests: `cargo test`
2. Build release: `cargo build --release`
3. Run GET/SET benchmark on test cluster (should match expected throughput)
4. If touching vector code, run vector search benchmark
5. If touching optimizer code, run optimization and verify convergence

## Documentation

- [README.md](README.md) - Main user documentation
- [INSTALLATION.md](INSTALLATION.md) - Setup and build instructions
- [DATASETS.md](DATASETS.md) - Dataset download and conversion guide
- [BENCHMARKING.md](BENCHMARKING.md) - Benchmark usage guide
- [ADVANCED.md](ADVANCED.md) - Optimizer, metadata filtering, binary format
- [EXAMPLES.md](EXAMPLES.md) - Comprehensive feature examples
- [docs/](docs/) - Design documents (HLD, LLD)

## Dataset Management

Binary datasets are used for vector search benchmarking:

```bash
# List available datasets
./prep_datasets/dataset.sh list

# Download and convert datasets
./prep_datasets/dataset.sh get mnist           # Small (60K vectors, 784-dim)
./prep_datasets/dataset.sh get sift-128        # Medium (1M vectors, 128-dim)
./prep_datasets/dataset.sh get cohere-medium-1m  # Large (1M vectors, 768-dim)

# Verify dataset
./prep_datasets/dataset.sh verify datasets/mnist.bin
```

Dataset binary format:
```
+------------------+
| Header (128 B)   |  magic, dimensions, counts, offsets
+------------------+
| Database Vectors |  N x dim x sizeof(f32)
+------------------+
| Query Vectors    |  Q x dim x sizeof(f32)
+------------------+
| Ground Truth     |  Q x K neighbor IDs
+------------------+
```

## Build Time Expectations

| Component | First Build | Subsequent |
|-----------|-------------|------------|
| Rust benchmark tool | 1-2 min | 10-20 sec |

First builds are slower due to dependency compilation. Subsequent builds use cached artifacts.
