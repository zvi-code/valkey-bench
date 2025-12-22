# Benchmarking Guide

Complete guide for running vector search benchmarks with valkey-bench-rs.

## Quick Start

### Basic Benchmark Workflow

1. **Load vectors into Valkey** (vec-load phase)
2. **Run query benchmarks** with recall validation (vec-query phase)
3. **Analyze results** (QPS, latency, recall)

### Example: COHERE 1M Dataset

```bash
# Step 1: Load all vectors into Valkey
./target/release/valkey-bench-rs \
  -h localhost --cluster \
  --dataset datasets/cohere-medium-1m.bin \
  -t vec-load \
  --search-index cohere_1m --search-prefix zvec_: \
  -n 1000000 -c 10

# Step 2: Run query benchmark
./target/release/valkey-bench-rs \
  -h localhost --cluster \
  --dataset datasets/cohere-medium-1m.bin \
  -t vec-query \
  --search-index cohere_1m --search-prefix zvec_: \
  -n 10000 -c 10 --threads 10
```

## Understanding Command Options

### Connection Options

```bash
-h <host>           # Valkey/Redis host (default: localhost)
-p <port>           # Port (default: 6379)
-a <password>       # Authentication password
--user <username>   # ACL username
--cluster           # Enable cluster mode
--tls               # Enable TLS connection
```

### Dataset Options

```bash
--dataset <file>    # Path to binary dataset file
```

### Index Options

```bash
--search-index <name>        # Index name (e.g., cohere_1m)
--search-prefix <prefix>     # Key prefix (e.g., zvec_:)
--search-vector-field <name> # Vector field name (default: embedding)
--search-algorithm <alg>     # HNSW or FLAT (default: HNSW)
--search-distance <metric>   # L2, COSINE, or IP (default: L2)
--ef-construction <n>        # HNSW build parameter (default: 200)
--ef-search <n>              # HNSW search parameter (default: 10)
--hnsw-m <n>                 # HNSW max connections (default: 16)
-k, --search-k <n>           # Number of neighbors to return (default: 10)
```

### Performance Options

```bash
-n <count>          # Number of operations
-c <clients>        # Number of parallel clients
--threads <n>       # Number of worker threads
-P <n>              # Pipeline requests (default: 1)
--rps <n>           # Rate limit (requests per second)
```

### Utility Options

```bash
--cleanup           # Delete index after benchmark
--skip-index-create # Skip index creation (assume exists)
--skip-load         # Skip data loading (assume loaded)
-v, --verbose       # Show detailed output
-q, --quiet         # Minimal output
-o <file>           # Output file path
--output-format     # text, json, or csv
```

## Benchmark Phases

### Phase 1: Vector Loading (vec-load)

Loads all training vectors into Valkey and creates the index.

```bash
./target/release/valkey-bench-rs \
  --dataset sift-128.bin \
  -t vec-load \
  --search-index sift_index \
  --search-prefix vec_: \
  --ef-construction 200 \
  --hnsw-m 16 \
  -n 1000000 -c 10
```

**What happens:**
1. Creates index with specified parameters
2. Inserts all vectors with ID mapping
3. Reports insertion throughput

**Typical output:**
```
Running test: vec-load... (12,543/s)

=== vec-load ===
Throughput: 12,543 req/s | Requests: 1,000,000 | Duration: 79.7s
Latency (ms): avg=0.80 p50=0.75 p95=1.20 p99=1.50 max=5.2
```

### Phase 2: Query Benchmark (vec-query)

Runs test queries and validates recall against ground truth.

```bash
./target/release/valkey-bench-rs \
  --dataset sift-128.bin \
  -t vec-query \
  --search-index sift_index \
  --search-prefix vec_: \
  --ef-search 100 \
  -k 10 \
  -n 10000 -c 10 --threads 10
```

**What happens:**
1. Runs k-NN queries from test set
2. Compares results to pre-computed ground truth
3. Reports QPS, latency, and recall

**Typical output:**
```
Running test: vec-query... (1,892/s)

=== vec-query ===
Throughput: 1,892 req/s | Requests: 10,000 | Duration: 5.29s
Latency (ms): avg=5.29 p50=5.12 p95=7.21 p99=8.93 max=15.2
Recall@10: 95.3%
```

## Parameter Tuning: ef_search

The `ef_search` parameter controls the HNSW search quality vs speed trade-off.

### Testing Different ef_search Values

```bash
# Low ef_search - faster, lower recall
./target/release/valkey-bench-rs \
  --dataset sift-128.bin \
  -t vec-query \
  --search-index sift_index \
  --ef-search 50 \
  -k 10 -n 10000

# High ef_search - slower, higher recall
./target/release/valkey-bench-rs \
  --dataset sift-128.bin \
  -t vec-query \
  --search-index sift_index \
  --ef-search 500 \
  -k 10 -n 10000
```

**Recommended values:**

| ef_search | Recall | Latency | Use Case |
|-----------|--------|---------|----------|
| 50-75 | 70-72% | ~1.5ms | High throughput, low latency |
| 100-200 | 73-74% | ~2ms | **Balanced (recommended)** |
| 250-400 | 74-75% | ~3ms | High accuracy |
| 500+ | 74-75% | ~4ms+ | Maximum accuracy (diminishing returns) |

## Automatic Optimization

The optimizer automatically finds the best configuration for your objective.

### Basic Optimization

```bash
./target/release/valkey-bench-rs \
  --optimize \
  --objective "maximize:qps" \
  --constraint "recall:gt:0.95" \
  --tune "clients:10:200:10" \
  --tune "threads:1:16:1" \
  --dataset datasets/cohere-medium-1m.bin \
  -t vec-query \
  --search-index cohere_1m \
  -h localhost --cluster
```

**What the optimizer does:**
1. **Feasibility phase**: Tests maximum values to find upper bound
2. **Exploration phase**: Grid sampling (min, 25%, 50%, 75%, max)
3. **Exploitation phase**: Hill climbing with multiple step sizes

### Optimization Options

```bash
--optimize                              # Enable optimizer
--objective <target>                    # "maximize:qps" or "minimize:p99_ms"
--constraint <condition>                # e.g., "recall:gt:0.95"
--tune <param>                          # e.g., "clients:10:200:10"
--max-optimize-iterations <n>           # Max iterations (default: 50)
--tolerance <n>                         # Multi-goal equivalence tolerance
```

**Multiple constraints:**
```bash
--constraint "recall:gt:0.95" \
--constraint "p99_ms:lt:10.0"
```

### Optimization Examples

```bash
# Maximize QPS for GET workload
./target/release/valkey-bench-rs -h cluster-node --cluster -t get -n 100000 \
  --optimize --objective "maximize:qps" \
  --tune "clients:10:300:10" --tune "threads:1:32:1"

# Maximize QPS with p99 latency constraint
./target/release/valkey-bench-rs -h cluster-node --cluster -t get -n 100000 \
  --optimize --objective "maximize:qps" --constraint "p99_ms:lt:1.0" \
  --tune "clients:10:200:10" --tune "threads:1:16:1"

# Multi-objective: maximize QPS, tiebreak on lowest p99
./target/release/valkey-bench-rs -h cluster-node --cluster -t get -n 100000 \
  --optimize --objective "maximize:qps,minimize:p99_ms" --tolerance 0.04 \
  --tune "clients:10:300:10" --tune "threads:1:32:1"

# Vector search: maximize QPS with recall above 95%
./target/release/valkey-bench-rs -h cluster-node --cluster -t vec-query \
  --dataset vectors.bin --search-index idx -n 100000 \
  --optimize --objective "maximize:qps" --constraint "recall:gt:0.95" \
  --tune "ef_search:10:500:10" --tune "clients:10:100:10"
```

## Interpreting Results

### Throughput Metrics

- **QPS (Queries Per Second)**: Higher is better
  - Good: >1,000 QPS
  - Excellent: >5,000 QPS

### Latency Metrics

- **avg**: Average latency
- **p50**: Median (50th percentile)
- **p95**: 95th percentile (5% of queries are slower)
- **p99**: 99th percentile (1% of queries are slower)
- **p99.9**: 99.9th percentile
- **max**: Worst case

**What to watch:**
- High p99/p50 ratio: Inconsistent performance
- p99 > 10x avg: Possible outliers or GC pauses

### Recall Metrics

- **Recall@K**: Percentage of ground truth neighbors found

**Target recall:**
- Production: >=95%
- High quality: >=98%
- Research: >=99%

## Common Benchmark Scenarios

### Scenario 1: Quick Performance Check

```bash
# Test with 1,000 queries
./target/release/valkey-bench-rs \
  --dataset datasets/mnist.bin \
  -t vec-query \
  --search-index mnist \
  -n 1000 -c 10
```

### Scenario 2: Production Simulation

```bash
# High concurrency, long duration
./target/release/valkey-bench-rs \
  --dataset datasets/cohere-medium-1m.bin \
  -t vec-query \
  --search-index prod_test \
  -n 100000 -c 100 --threads 10
```

### Scenario 3: Latency-Optimized

```bash
# Low concurrency, measure tail latency
./target/release/valkey-bench-rs \
  --dataset datasets/sift-128.bin \
  -t vec-query \
  --search-index low_latency \
  -n 10000 -c 1 --threads 1
```

### Scenario 4: Throughput-Optimized

```bash
# High concurrency, maximize QPS
./target/release/valkey-bench-rs \
  --dataset datasets/gist-960.bin \
  -t vec-query \
  --search-index high_qps \
  -n 50000 -c 200 --threads 20 -P 10
```

## Filtered Search Benchmarks

For datasets with metadata (like YFCC-10M):

```bash
# Load vectors with tags
./target/release/valkey-bench-rs \
  --dataset datasets/yfcc-10m.bin \
  -t vec-load \
  --search-index yfcc \
  --search-prefix vec_: \
  --tag-field category \
  --search-tags "electronics:30,clothing:25,home:20" \
  -n 100000 -c 10

# Query with tag filter
./target/release/valkey-bench-rs \
  --dataset datasets/yfcc-10m.bin \
  -t vec-query \
  --search-index yfcc \
  --tag-field category \
  --tag-filter "electronics" \
  -n 10000 -c 10
```

## Troubleshooting

### Low Recall

**Problem:** Recall < 90%

**Solutions:**
1. Increase ef_search: `--ef-search 400`
2. Check vector normalization (for COSINE metric)
3. Verify dataset integrity: `./prep_datasets/dataset.sh verify dataset.bin`

### Low QPS

**Problem:** QPS < expected

**Solutions:**
1. Increase concurrency: `-c 50 --threads 10`
2. Use pipelining: `-P 10`
3. Check cluster distribution: Ensure vectors spread across shards
4. Monitor server resources: CPU, memory, network

### High Latency

**Problem:** p99 latency very high

**Solutions:**
1. Reduce ef_search for faster queries
2. Decrease concurrency to reduce contention
3. Check server load and GC pauses
4. Verify network latency

### Connection Errors

**Problem:** Connection refused or timeouts

**Solutions:**
```bash
# Verify server is running
./target/release/valkey-bench-rs --cli -h localhost PING

# Check cluster status
./target/release/valkey-bench-rs --cli -h localhost CLUSTER INFO

# Test basic connectivity
./target/release/valkey-bench-rs -h localhost -t ping -n 1000
```

### Memory Issues

**Problem:** Server OOM during loading

**Solutions:**
1. Load in smaller batches: Use `--num-vectors` to limit
2. Increase server memory
3. Reduce INITIAL_CAP in index params

## Performance Tips

### Optimal Concurrency

Rule of thumb: `clients = 10-20 x num_cores`

```bash
# For 8-core server
-c 100 --threads 10
```

### Cluster Considerations

- Ensure even shard distribution
- Use `--cluster` for automatic topology discovery
- Test with `--rfr prefer-replica` for read workloads

### JSON Output Analysis

Save results for analysis:

```bash
./target/release/valkey-bench-rs \
  --dataset datasets/cohere-medium-1m.bin \
  -t vec-query \
  -o results.json --output-format json
```

Load in Python for analysis:

```python
import json
with open('results.json') as f:
    data = json.load(f)
print(data)
```

## Next Steps

- **Advanced Features**: See [ADVANCED.md](ADVANCED.md) for optimizer internals and metadata filtering
- **Dataset Management**: See [DATASETS.md](DATASETS.md) for more datasets
- **Installation**: See [INSTALLATION.md](INSTALLATION.md) for environment setup
- **Examples**: See [EXAMPLES.md](EXAMPLES.md) for comprehensive examples
