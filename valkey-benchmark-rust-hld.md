# High-Level Design: valkey-search-benchmark

**Version:** 2.0
**Purpose:** Architectural overview of the Rust benchmark tool for Valkey/Redis

---

## 1. Executive Summary

`valkey-search-benchmark` is a high-performance benchmarking tool written in Rust, designed for Valkey/Redis with first-class support for vector search operations. The tool features zero external Redis client dependencies, using a custom mio-based event-driven architecture for maximum throughput and minimal latency.

### 1.1 Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| Zero Dependencies | Custom RESP codec, no redis-rs or external clients |
| Event-Driven I/O | mio polling with non-blocking sockets |
| Lock-Free Hot Path | Atomic counters, thread-local histograms |
| Memory Efficiency | Pre-allocated buffers, in-place placeholder replacement |
| Cluster Native | Full topology discovery, slot routing, MOVED/ASK handling |

### 1.2 Supported Features

- **Multiple Workloads**: PING, GET, SET, HSET, LPUSH, RPUSH, SADD, ZADD, vector operations
- **Cluster Mode**: Automatic topology discovery, read-from-replica strategies
- **Vector Search**: FT.CREATE, FT.SEARCH with recall@k computation
- **Rate Limiting**: Token bucket for controlled load testing
- **TLS/SSL**: Full certificate authentication support
- **CLI Mode**: Interactive command interface (valkey-cli compatible)

---

## 2. Architecture Overview

### 2.1 System Layers

```
+-------------------------------------------------------------------------+
|                           valkey-search-benchmark                        |
+-------------------------------------------------------------------------+
|                                                                         |
|  +-------------------+  +-------------------+  +---------------------+  |
|  |   Configuration   |  |    Orchestrator   |  |  Metrics Reporter   |  |
|  |  - CLI parsing    |  |  - Thread spawn   |  |  - HDR histograms   |  |
|  |  - TLS/Auth       |  |  - Progress bar   |  |  - Recall stats     |  |
|  |  - Workload setup |  |  - Result merge   |  |  - JSON/text output |  |
|  +--------+----------+  +---------+---------+  +----------+----------+  |
|           |                       |                       |             |
|           v                       v                       v             |
|  +----------------------------------------------------------------------+
|  |                        Event Workers (N threads)                      |
|  |  +----------------+  +----------------+       +----------------+      |
|  |  |   Worker 0     |  |   Worker 1     |  ...  |   Worker N     |      |
|  |  | +------------+ |  | +------------+ |       | +------------+ |      |
|  |  | | mio Poll   | |  | | mio Poll   | |       | | mio Poll   | |      |
|  |  | +------------+ |  | +------------+ |       | +------------+ |      |
|  |  | | Clients    | |  | | Clients    | |       | | Clients    | |      |
|  |  | | [0..M/N]   | |  | | [0..M/N]   | |       | | [0..M/N]   | |      |
|  |  | +------------+ |  | +------------+ |       | +------------+ |      |
|  |  | Thread-local: |  | Thread-local: |       | Thread-local: |       |
|  |  | - histogram   |  | - histogram   |       | - histogram   |       |
|  |  | - RNG         |  | - RNG         |       | - RNG         |       |
|  |  +----------------+  +----------------+       +----------------+      |
|  +----------------------------------------------------------------------+
|                                    |                                     |
|                                    v                                     |
|  +----------------------------------------------------------------------+
|  |                          Connection Layer                             |
|  |  +--------------------+  +--------------------+  +-----------------+  |
|  |  |   RawConnection    |  |  BenchmarkClient   |  | ClusterTopology |  |
|  |  | - TCP/TLS streams  |  | - Write buffers    |  | - Node discovery|  |
|  |  | - RESP codec       |  | - Placeholder mgmt |  | - Slot mapping  |  |
|  |  | - Auth handling    |  | - Response parsing |  | - RFR selection |  |
|  |  +--------------------+  +--------------------+  +-----------------+  |
|  +----------------------------------------------------------------------+
|                                                                         |
+-------------------------------------------------------------------------+
```

### 2.2 Request Flow

```
+--------------------------------------------------------------------------+
|                           REQUEST LIFECYCLE                               |
+--------------------------------------------------------------------------+
|                                                                          |
|  1. CLAIM REQUEST QUOTA                                                  |
|     +------------------+                                                 |
|     | atomic_fetch_add |  <-- requests_issued counter                    |
|     | (requests_issued)|                                                 |
|     +--------+---------+                                                 |
|              |                                                           |
|              v                                                           |
|  2. RATE LIMITING (optional)                                             |
|     +------------------+                                                 |
|     | Token Bucket     |  <-- if --rps specified                         |
|     | acquire_tokens() |                                                 |
|     +--------+---------+                                                 |
|              |                                                           |
|              v                                                           |
|  3. PLACEHOLDER REPLACEMENT (in-place, zero-allocation)                  |
|     +------------------+                                                 |
|     | Replace:         |                                                 |
|     | - Random keys    |  random() % keyspace_len                        |
|     | - Dataset vectors|  memcpy from mmap'd region                      |
|     | - Cluster tags   |  lookup from tag map                            |
|     +--------+---------+                                                 |
|              |                                                           |
|              v                                                           |
|  4. SOCKET WRITE (non-blocking)                                          |
|     +------------------+                                                 |
|     | write_buf -> TCP |  <-- mio WRITABLE event                         |
|     | (pipelined cmds) |                                                 |
|     +--------+---------+                                                 |
|              |                                                           |
|              v                                                           |
|  5. SOCKET READ (non-blocking)                                           |
|     +------------------+                                                 |
|     | TCP -> read_buf  |  <-- mio READABLE event                         |
|     | RESP parsing     |                                                 |
|     +--------+---------+                                                 |
|              |                                                           |
|              v                                                           |
|  6. RESPONSE PROCESSING                                                  |
|     +------------------+                                                 |
|     | - Record latency |  thread-local histogram                         |
|     | - Verify recall  |  if FT.SEARCH with ground truth                 |
|     | - Handle errors  |  MOVED/ASK -> queue retry                       |
|     +--------+---------+                                                 |
|              |                                                           |
|              v                                                           |
|  7. UPDATE COUNTERS                                                      |
|     +------------------+                                                 |
|     | atomic_fetch_add |  <-- requests_finished counter                  |
|     +------------------+                                                 |
|                                                                          |
+--------------------------------------------------------------------------+
```

---

## 3. Threading Model

### 3.1 Thread Architecture

```
+------------------------------------------------------------------------+
|                         THREAD HIERARCHY                                |
+------------------------------------------------------------------------+
|                                                                        |
|  MAIN THREAD                                                           |
|  +------------------------------------------------------------------+  |
|  | - Parse CLI arguments                                             |  |
|  | - Discover cluster topology (control plane)                       |  |
|  | - Load dataset (mmap)                                             |  |
|  | - Build command templates                                         |  |
|  | - Spawn worker threads                                            |  |
|  | - Display progress bar                                            |  |
|  | - Collect and merge results                                       |  |
|  +------------------------------------------------------------------+  |
|           |                                                            |
|           v spawns N worker threads                                    |
|  +------------------------------------------------------------------+  |
|  |                     WORKER THREADS (N)                            |  |
|  |                                                                    |  |
|  |  Each worker:                                                      |  |
|  |  - Owns M/N clients exclusively (no sharing)                       |  |
|  |  - Runs its own mio event loop                                     |  |
|  |  - Maintains thread-local histogram                                |  |
|  |  - Uses thread-local RNG (fastrand)                                |  |
|  |                                                                    |  |
|  |  Synchronization (atomic counters only):                           |  |
|  |  - requests_issued: claim quota before sending                     |  |
|  |  - requests_finished: increment after response                     |  |
|  |  - dataset_counter: claim unique vector indices                    |  |
|  |                                                                    |  |
|  +------------------------------------------------------------------+  |
|                                                                        |
+------------------------------------------------------------------------+
```

### 3.2 Synchronization Points

| Counter | Type | Access Pattern | Purpose |
|---------|------|----------------|---------|
| `requests_issued` | AtomicU64 | fetch_add before send | Global request quota |
| `requests_finished` | AtomicU64 | fetch_add after reply | Progress tracking |
| `dataset_counter` | AtomicU64 | fetch_add for unique claim | Vector insertion |
| `seq_key_counter` | AtomicU64 | fetch_add if sequential | Sequential keys |

**Design Decision:** Thread-local histograms are merged only at benchmark completion, avoiding runtime synchronization.

---

## 4. Connection Architecture

### 4.1 Connection Types

```
+------------------------------------------------------------------------+
|                       CONNECTION LAYER                                  |
+------------------------------------------------------------------------+
|                                                                        |
|  RawConnection (Direct TCP/TLS)                                        |
|  +------------------------------------------------------------------+  |
|  | - std::net::TcpStream with non-blocking mode                      |  |
|  | - Optional native-tls wrapper for TLS                             |  |
|  | - Custom RESP protocol encoder/decoder                            |  |
|  | - No external Redis client library                                |  |
|  +------------------------------------------------------------------+  |
|                                                                        |
|  BenchmarkClient (Per-connection state)                                |
|  +------------------------------------------------------------------+  |
|  | - Pre-allocated write buffer (RESP-encoded template)              |  |
|  | - Pre-allocated read buffer                                        |  |
|  | - Placeholder offsets (computed once at init)                      |  |
|  | - Assigned cluster node reference                                  |  |
|  | - Pipeline state (pending responses count)                         |  |
|  | - Query indices queue (for recall verification)                    |  |
|  +------------------------------------------------------------------+  |
|                                                                        |
+------------------------------------------------------------------------+
```

### 4.2 RESP Protocol

The tool implements a custom RESP (Redis Serialization Protocol) codec:

```
Encoding:
  *<num_args>\r\n           -- Array header
  $<len>\r\n<data>\r\n      -- Bulk string for each argument

Decoding:
  +<message>\r\n            -- Simple string
  -<error>\r\n              -- Error
  :<number>\r\n             -- Integer
  $<len>\r\n<data>\r\n      -- Bulk string
  *<count>\r\n...           -- Array
```

---

## 5. Cluster Support

### 5.1 Topology Discovery

```
+------------------------------------------------------------------------+
|                      CLUSTER TOPOLOGY                                   |
+------------------------------------------------------------------------+
|                                                                        |
|  Discovery Flow:                                                       |
|  1. Connect to seed node                                               |
|  2. Execute CLUSTER NODES                                              |
|  3. Parse node list (id, host:port, flags, slots)                      |
|  4. Build slot-to-node mapping (16384 slots)                           |
|  5. Select nodes based on read-from-replica strategy                   |
|                                                                        |
|  Slot Mapping:                                                         |
|  +---------------------------+                                         |
|  | slot_map: [Option<Node>; 16384]                                    |
|  | get_node(key) -> compute_crc16(key) % 16384 -> lookup              |
|  +---------------------------+                                         |
|                                                                        |
|  Read-From-Replica Strategies:                                         |
|  +---------------------------+                                         |
|  | primary        | Only primary nodes (default)                      |
|  | prefer-replica | Prefer replicas, fallback to primary              |
|  | round-robin    | Distribute across all nodes                       |
|  | az-affinity    | Prefer same availability zone                     |
|  +---------------------------+                                         |
|                                                                        |
+------------------------------------------------------------------------+
```

### 5.2 Error Handling

```
MOVED <slot> <host>:<port>
  -> Refresh cluster topology
  -> Retry request to new node

ASK <slot> <host>:<port>
  -> Send ASKING command to target
  -> Retry request to target node

CLUSTERDOWN
  -> Wait and retry
```

---

## 6. Workload System

### 6.1 Command Templates

Templates are RESP-encoded command buffers with placeholder markers:

```
Template Structure:
  +------------------+------------------+------------------+
  | Static RESP      | Placeholder      | Static RESP      |
  | (command name)   | (key/vector)     | (options)        |
  +------------------+------------------+------------------+

Example - HSET for vector insert:
  *4                     -- 4 arguments
  $4 HSET               -- command
  $17 {clt}vec:000000   -- key with cluster tag + dataset key placeholder
  $9 embedding          -- field name
  $3072 <vector bytes>  -- vector data placeholder (768 dim * 4 bytes)
```

### 6.2 Placeholder Types

| Placeholder | Purpose | Replacement |
|-------------|---------|-------------|
| `{clt}` | Cluster routing tag | 5-char slot-based tag |
| Random key | Key identifier | Fixed-width decimal (12 chars) |
| Dataset vector | Vector blob | Raw f32 bytes from mmap |
| Dataset key | Vector ID | Claimed dataset index |

### 6.3 Workload Types

| Type | Command | Description |
|------|---------|-------------|
| ping | PING | Basic connectivity |
| get | GET key | Read operation |
| set | SET key value | Write operation |
| hset | HSET key field value | Hash write |
| vec-load | HSET with vector | Load vectors into index |
| vec-query | FT.SEARCH | Vector similarity search |

---

## 7. Vector Search

### 7.1 Dataset Format

Binary dataset with memory-mapped access:

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

### 7.2 Recall Computation

```
For each FT.SEARCH query:
  1. Extract returned document IDs from response
  2. Load ground truth neighbors for query index
  3. Compute intersection size
  4. recall = |intersection| / k
  5. Aggregate statistics (mean, min, max)
```

---

## 8. Metrics Collection

### 8.1 Latency Tracking

- **HDR Histogram**: High Dynamic Range histograms for accurate percentiles
- **Thread-Local**: Each worker maintains its own histogram
- **Merge at End**: Histograms combined after benchmark completion

### 8.2 Reported Metrics

| Metric | Description |
|--------|-------------|
| Throughput | Requests per second |
| p50/p95/p99/p999 | Latency percentiles |
| Mean/Max latency | Average and worst case |
| Recall@k | Vector search accuracy |
| Error count | Failed requests |

---

## 9. Rate Limiting

Token bucket algorithm for controlled load:

```
Configuration:
  --rps <N>        Target requests per second

Implementation:
  - time_per_token = 1_000_000_000 / rps (nanoseconds)
  - Atomic timestamp for last token issue
  - Workers wait if tokens unavailable
```

---

## 10. CLI Mode

Interactive command interface compatible with valkey-cli:

```
Features:
  - Connect to standalone or cluster
  - Execute any Redis/Valkey command
  - Format output (bulk strings, arrays, etc.)
  - TLS and authentication support

Usage:
  # Interactive mode
  ./valkey-search-benchmark --cli -h host

  # Single command
  ./valkey-search-benchmark --cli -h host PING
  ./valkey-search-benchmark --cli -h host FT.INFO index
```

---

## 11. Build and Dependencies

### 11.1 Core Dependencies

| Dependency | Purpose |
|------------|---------|
| clap | CLI argument parsing |
| mio | Event-driven I/O |
| hdrhistogram | Latency percentiles |
| memmap2 | Memory-mapped datasets |
| native-tls | TLS support |
| fastrand | Fast random number generation |
| indicatif | Progress bars |

### 11.2 Build Commands

```bash
# Standard build
cargo build --release

# Run tests
cargo test

# With specific TLS backend
cargo build --release --features rustls-backend
```

---

## 12. Performance Characteristics

### 12.1 Design Optimizations

| Optimization | Benefit |
|--------------|---------|
| mio event loop | Non-blocking I/O, no thread-per-connection |
| Pre-allocated buffers | Zero allocation in hot path |
| Thread-local histograms | No runtime lock contention |
| Memory-mapped datasets | Zero-copy vector access |
| Pipeline batching | Amortize network round-trips |

### 12.2 Typical Performance

On a 16-node cluster with 500-byte values:
- GET: ~980K requests/second
- SET: ~380K requests/second
- FT.SEARCH: ~13K queries/second (depending on index size)

---

## 13. Module Structure

```
src/
  main.rs              Entry point, CLI dispatch
  lib.rs               Library exports

  config/
    cli.rs             Clap argument definitions
    benchmark_config.rs Runtime configuration

  client/
    raw_connection.rs  TCP/TLS connection wrapper
    benchmark_client.rs Per-client state and buffers
    control_plane.rs   Cluster discovery trait

  cluster/
    topology.rs        Node discovery and slot mapping
    node.rs            ClusterNode representation

  benchmark/
    orchestrator.rs    Thread spawning, result collection
    event_worker.rs    mio-based worker implementation

  workload/
    types.rs           Workload type enum
    template.rs        Command template builder

  dataset/
    binary_dataset.rs  Memory-mapped dataset access

  metrics/
    histogram.rs       HDR histogram wrapper
    recall.rs          Recall statistics
    reporter.rs        Output formatting

  utils/
    resp.rs            RESP protocol codec
    token_bucket.rs    Rate limiter
```
