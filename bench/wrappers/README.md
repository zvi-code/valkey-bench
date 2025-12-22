# Valkey Search Benchmark - Python Wrapper Framework

Simple, elegant Python wrapper framework for building sophisticated benchmark automation scripts.

## Philosophy: Keep It Simple (KIS)

This framework follows the KIS principle by leveraging existing functionality:
- **Cluster Detection**: The Rust binary supports `--cluster` flag
- **Auto-optimization**: The Rust binary supports `--optimize` for automated parameter tuning

The wrapper orchestrates, parses results, and implements search algorithms.

## Quick Start

### Basic Usage

```python
#!/usr/bin/env python3
from wrappers import ValKeyBenchmarkWrapper, BenchmarkConfig

# Initialize wrapper
wrapper = ValKeyBenchmarkWrapper(verbose=True)

# Create configuration
config = BenchmarkConfig(
    host="localhost",
    dataset="datasets/sift-128.bin",
    num_clients=20,
    ef_search=100
)

# Run benchmark
result = wrapper.run(config, operation="vec-query")

print(f"QPS: {result.qps:.1f}")
print(f"Latency P99: {result.latency_p99:.3f}ms")
print(f"Recall: {result.recall_avg:.2%}")
```

### With Stage Signaling

```python
# Stages emit signals for external monitoring tools
with wrapper.stage("vec-query", tag="ef_100"):
    result = wrapper.run(config)
```

### Binary Search for Max QPS

```python
# Find max QPS at 95% recall
best = wrapper.binary_search_max_qps(
    base_config=config,
    target_recall=0.95,
    param_name="num_clients",
    max_val=200
)

print(f"Optimal clients: {best.config.num_clients}")
print(f"Max QPS: {best.qps:.1f}")
```

### Grid Search

```python
# Test all combinations
results = wrapper.grid_search(
    base_config=config,
    param_grid={
        "num_clients": [10, 20, 50, 100],
        "ef_search": [50, 100, 200]
    },
    filter_fn=lambda r: r.recall_avg >= 0.95  # Only high recall
)

# Find best
best = max(results, key=lambda r: r.qps)
```

### With Constraints

```python
# Max QPS with recall >= 95% AND p99 <= 10ms
best = wrapper.find_max_qps_with_constraints(
    base_config=config,
    min_recall=0.95,
    max_latency_p99=10.0,
    param_name="num_clients"
)
```

## Example Scripts

### run_queries.py

**Auto-optimized query benchmark** - simplest interface, only requires dataset name and host:

```bash
# Simplest usage - automatic optimization
./bench/scripts/run_queries.py \
    --host localhost \
    --dataset openai-large-5m

# The script automatically:
#   - Finds dataset in datasets/ directory
#   - Uses dataset name as index name
#   - Detects dataset dimensions and size
#   - Estimates optimal ef_search range
#   - Finds optimal num_clients via binary search
#   - Reports exact command used
```

### max_qps_recall.py

Find maximum QPS at target recall threshold (more control than run_queries.py):

```bash
# Basic usage
./bench/scripts/max_qps_recall.py \
    --host localhost \
    --dataset datasets/sift-128.bin \
    --target-recall 0.95 \
    --ef-search 100

# With latency constraint
./bench/scripts/max_qps_recall.py \
    --host localhost \
    --dataset datasets/sift-128.bin \
    --target-recall 0.95 \
    --max-p99-latency 10.0 \
    --output results.csv

# Search over ef_search instead of num_clients
./bench/scripts/max_qps_recall.py \
    --host localhost \
    --dataset datasets/sift-128.bin \
    --target-recall 0.95 \
    --search-param ef_search \
    --min-value 50 \
    --max-value 500
```

### stage-monitor.sh

Monitor stage signals and collect perf data:

```bash
# In Terminal 1: Run benchmark
./bench/scripts/max_qps_recall.py ... 2>&1 | tee bench.log

# In Terminal 2: Monitor and collect perf
./bench/scripts/stage-monitor.sh \
    --watch vec-query \
    --collect-perf \
    --server-pid $(pgrep valkey-server) \
    < bench.log

# Or pipe directly
./bench/scripts/max_qps_recall.py ... 2>&1 | \
    ./bench/scripts/stage-monitor.sh --watch vec-query --collect-perf
```

## API Reference

### BenchmarkConfig

Configuration dataclass for benchmark runs.

**Attributes:**
- `host`: Redis/Valkey server hostname
- `dataset`: Path to binary dataset file
- `num_clients`: Number of parallel clients (default: 10)
- `num_threads`: Number of threads (default: 4)
- `num_requests`: Number of requests (default: 10000)
- `ef_search`: HNSW ef_search parameter (optional)
- `operation`: Operation type for `-t` flag (default: "vec-query")
- `extra_args`: Additional CLI arguments as list

### BenchmarkResult

Results dataclass from benchmark execution.

**Attributes:**
- `qps`: Queries per second (throughput)
- `latency_avg/p50/p95/p99/max`: Latency metrics in milliseconds
- `recall_avg/min/max`: Recall metrics (0.0-1.0), None if not applicable
- `baseline_latency_avg`: Baseline network latency, None if not measured
- `config`: The configuration used for this run
- `timestamp`: Unix timestamp

### ValKeyBenchmarkWrapper

Main wrapper class.

**Methods:**

#### `__init__(binary=None, verbose=False)`
Initialize wrapper. Auto-detects binary if not specified.

#### `run(config, operation=None) -> BenchmarkResult`
Run benchmark and return parsed results.

#### `stage(name, tag=None)` [context manager]
Signal test stage for external monitoring tools.

#### `binary_search_max_qps(base_config, target_recall, param_name="num_clients", min_val=1, max_val=200) -> BenchmarkResult`
Binary search to find maximum QPS at target recall.

#### `grid_search(base_config, param_grid, filter_fn=None) -> List[BenchmarkResult]`
Grid search over parameter combinations.

#### `find_max_qps_with_constraints(base_config, min_recall, max_latency_p99=None, param_name="num_clients", ...) -> BenchmarkResult`
Find max QPS with recall and latency constraints.

#### `save_results_csv(results, output_file)`
Save results to CSV file.

### Exceptions

- `BenchmarkError`: Base exception for benchmark errors
- `BinaryNotFoundError`: Binary not found
- `ParseError`: Output parsing failed

## Stage Signaling Protocol

The wrapper emits stage signals to stderr for external tool coordination:

```
[STAGE:START] stage_name[:tag]
[STAGE:END] stage_name[:tag] duration=X.XXXs
```

**Stage Names:**
- Typically match the `-t` operation: `vec-query`, `vec-load`, `get`, etc.
- Can include tags for variants: `vec-query:ef_100`, `vec-query:clients_20`

**External Tool Integration:**
External scripts (like `stage-monitor.sh`) can parse these signals to:
- Start/stop profiling during specific stages
- Tag collected data with stage names
- Skip profiling during warmup/prefill stages

## Environment Variables

- `VALKEY_BENCHMARK_BIN`: Path to valkey-bench-rs binary (uses `--cli` mode for CLI commands)

## Architecture

```
bench/
├── wrappers/
│   ├── __init__.py           # Package exports
│   └── base_wrapper.py       # All core logic (~580 lines)
│
└── scripts/
    ├── max_qps_recall.py     # Example: Find max QPS
    └── stage-monitor.sh      # Example: External monitor
```

**Design Decisions:**
1. **Single-file core** - All logic in `base_wrapper.py` for simplicity
2. **No cluster manager** - Simple detection helper, Rust binary handles distribution
3. **No stage manager module** - Just a 10-line context manager
4. **No perf collector module** - External scripts parse stage signals

## Creating New Wrappers

Use `max_qps_recall.py` as a template:

1. Import the framework:
```python
from wrappers import ValKeyBenchmarkWrapper, BenchmarkConfig
```

2. Create wrapper and config:
```python
wrapper = ValKeyBenchmarkWrapper(verbose=True)
config = BenchmarkConfig(...)
```

3. Use search algorithms or implement custom logic:
```python
result = wrapper.binary_search_max_qps(...)
# or
results = wrapper.grid_search(...)
# or
result = wrapper.run(config)
```

4. Save results:
```python
wrapper.save_results_csv(results, "output.csv")
```

## Future Wrapper Scripts (Planned)

See `TODO.md` for planned wrappers:
- Wrapper #2: Max QPS at recall + latency thresholds ✅ (implemented as `find_max_qps_with_constraints`)
- Wrapper #3: Optimal config discovery (grid search) ✅ (implemented as `grid_search`)
- Wrapper #4: Profiling integration ✅ (implemented as `stage-monitor.sh`)
- Wrapper #5: Memory saturation testing
- Wrapper #6: Payload impact testing

## Contributing

When creating new wrappers:
1. Follow the KIS principle - keep it simple!
2. Use existing search algorithms when possible
3. Emit stage signals for monitoring
4. Add inline documentation
5. Save results to CSV for analysis

## License

BSD 3-Clause License - See LICENSE file in root directory.
