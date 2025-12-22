# Low Level Design: valkey-search-benchmark (C Implementation Reference)

> **Note: This document describes the LEGACY C IMPLEMENTATION**
>
> This LLD documents the original **C implementation** design patterns that informed the Rust rewrite.
> The current Rust implementation in this directory follows similar architectural principles but uses:
> - **mio** instead of ae event loop
> - **Custom RESP codec** instead of libvalkey
> - **std::thread** instead of pthreads
> - **Rust ownership** instead of manual memory management
>
> For the current Rust architecture, see:
> - [valkey-benchmark-rust-hld.md](valkey-benchmark-rust-hld.md) - Current Rust HLD
> - [README.md](README.md) - User documentation
> - [EXAMPLES.md](EXAMPLES.md) - Comprehensive usage examples

**Version:** 2.0
**Purpose:** Reference document showing C implementation patterns that informed the Rust rewrite

---

## 1. Executive Summary

`valkey-search-benchmark` is a high-performance, multi-threaded benchmarking tool for Valkey/Redis with first-class support for vector search operations. The tool is designed for zero-allocation request generation, cluster topology awareness, recall verification against ground truth datasets, and adaptive parameter optimization.

### Core Design Goals
1. **Zero-allocation hot path**: Pre-computed templates with in-place placeholder replacement
2. **Cluster-native**: Full support for slot-based routing and per-node statistics
3. **Vector search native**: FT.SEARCH/FT.CREATE integration with recall verification
4. **Dataset-driven**: HDF5/binary dataset loading with ground truth comparison
5. **Optimization-aware**: Adaptive tuning of benchmark parameters to meet constraints

---

## 2. Architectural Overview

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         valkey-search-benchmark                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────────────┐  │
│  │   Config    │   │   Optimizer  │   │      Metrics Collector      │  │
│  │  Manager    │   │   (Phased)   │   │  (HDR Histogram, Per-Node)  │  │
│  └──────┬──────┘   └──────┬───────┘   └──────────────┬──────────────┘  │
│         │                 │                          │                  │
│         v                 v                          v                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     Benchmark Orchestrator                        │  │
│  │  - Workload sequencing (sequential/parallel scheduling)           │  │
│  │  - Thread pool management                                         │  │
│  │  - Global counters (atomic requests_issued, requests_finished)    │  │
│  └───────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                    │
│         v                    v                    v                    │
│  ┌─────────────┐    ┌─────────────┐     ┌─────────────┐               │
│  │  Thread 0   │    │  Thread 1   │ ... │  Thread N   │               │
│  │ (ae loop)   │    │ (ae loop)   │     │ (ae loop)   │               │
│  └──────┬──────┘    └──────┬──────┘     └──────┬──────┘               │
│         │                  │                   │                       │
│         v                  v                   v                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                      Client Pool                                 │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │  │
│  │  │Client 0│  │Client 1│  │Client 2│  │   ...  │  │Client M│    │  │
│  │  │(node 0)│  │(node 1)│  │(node 2)│  │        │  │(node K)│    │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                    │
│         v                    v                    v                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Cluster Topology Manager                       │  │
│  │  - Slot-to-node mapping (16384 slots)                            │  │
│  │  - Node selection (primary/replica/all)                          │  │
│  │  - MOVED/ASK handling and slot refresh                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              v                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Dataset Manager                              │  │
│  │  - Memory-mapped binary dataset (vectors, queries, ground truth) │  │
│  │  - Vector ID to cluster tag mapping (clusterTagMap)              │  │
│  │  - Recall verification against ground truth                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Request Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           REQUEST LIFECYCLE                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STARTUP PHASE (once per benchmark run):                                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │ Parse CLI args  │────>│ Create command  │────>│ Scan template   │   │
│  │ Load config     │     │ template string │     │ for placeholders│   │
│  └─────────────────┘     └─────────────────┘     └────────┬────────┘   │
│                                                           │             │
│                                               ┌───────────v───────────┐ │
│                                               │ Build placeholders    │ │
│                                               │ indices struct        │ │
│                                               │ (count, positions)    │ │
│                                               └───────────┬───────────┘ │
│                                                           │             │
│  RUNTIME PHASE (per request batch):                       v             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │ writeHandler()  │────>│replacePlaceholders│────>│ Write to socket │   │
│  │ triggered by ae │     │  (in-place)      │     │ (non-blocking)  │   │
│  └─────────────────┘     └────────┬────────┘     └────────┬────────┘   │
│                                   │                       │             │
│         ┌─────────────────────────┴───────────────────────┘             │
│         │                                                               │
│         │  PLACEHOLDER REPLACEMENT MODES:                               │
│         │  ┌─────────────────────────────────────────────────────────┐ │
│         │  │ KEY placeholders (__rand_int__, __d_key_ph__):          │ │
│         │  │   • Sequential: atomic_fetch_add on thread-safe counter │ │
│         │  │   • Random: random() % keyspacelen                      │ │
│         │  │   • Dataset: iterate through dataset, skip already      │ │
│         │  │             inserted (via clusterTagMap lookup)         │ │
│         │  ├─────────────────────────────────────────────────────────┤ │
│         │  │ VECTOR placeholders (__d_vec_ph____):                   │ │
│         │  │   • Insert: datasetGetVector() → copy to buffer         │ │
│         │  │   • Query: datasetSetQueryVec() → copy query vector     │ │
│         │  │   • Enqueue query_idx for recall tracking               │ │
│         │  ├─────────────────────────────────────────────────────────┤ │
│         │  │ CLUSTER TAG placeholders ({clt}):                       │ │
│         │  │   • Lookup/compute slot from key hash                   │ │
│         │  │   • Overwrite 5-byte tag to route to correct node       │ │
│         │  └─────────────────────────────────────────────────────────┘ │
│         │                                                               │
│  RESPONSE PHASE:                                                        │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │ readHandler()   │────>│ Parse RESP      │────>│ Command-specific│   │
│  │ triggered by ae │     │ reply           │     │ processing      │   │
│  └─────────────────┘     └─────────────────┘     └────────┬────────┘   │
│                                                           │             │
│                          ┌────────────────────────────────┘             │
│                          v                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ COMMAND-SPECIFIC RESPONSE HANDLERS:                               │  │
│  │                                                                    │  │
│  │ vec-query (FT.SEARCH):                                            │  │
│  │   1. Dequeue query_idx from client's tracking queue               │  │
│  │   2. Parse reply → extract returned document IDs                  │  │
│  │   3. Load ground truth: datasetGetNeighbors(query_idx)            │  │
│  │   4. Compute recall = |intersection| / k                          │  │
│  │   5. Update global recall statistics (thread-safe)                │  │
│  │                                                                    │  │
│  │ vec-load (HSET):                                                   │  │
│  │   1. Dequeue inflight dataset_idx from client                     │  │
│  │   2. On error: add to thread's retry queue                        │  │
│  │   3. On success: update clusterTagMap (vector_id → cluster_tag)   │  │
│  │                                                                    │  │
│  │ MOVED/ASK errors:                                                  │  │
│  │   1. Trigger fetchClusterSlotsConfiguration()                     │  │
│  │   2. Re-route pending requests to new node assignment             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Data Structures

### 3.1 Global Configuration (`config` struct)

The `config` struct is a static global containing all benchmark state:

```c
struct config {
    /* Event Loop */
    aeEventLoop *el;                      // Main thread event loop (single-threaded mode)
    
    /* Connection */
    enum valkeyConnectionType ct;         // TCP, UNIX, TLS, RDMA
    cliConnInfo conn_info;                // Host, port, auth, user
    valkeyContext *conn_ctx;              // Shared context for setup commands
    int tls;                              // TLS enabled flag
    struct cliSSLconfig sslconfig;        // TLS certificates, keys, SNI
    
    /* Concurrency */
    int64_t numclients;                   // Total number of simulated clients
    _Atomic int64_t liveclients;          // Currently active clients
    int64_t num_threads;                  // Worker thread count (0 = single-threaded)
    struct benchmarkThread **threads;     // Array of worker thread pointers
    
    /* Request Control */
    int64_t requests;                     // Total requests to issue
    _Atomic int64_t requests_issued;      // Requests sent (atomic counter)
    _Atomic int64_t requests_finished;    // Replies received (atomic counter)
    int64_t pipeline;                     // Commands per pipeline batch
    
    /* Key Generation */
    int64_t keyspacelen;                  // Range of key IDs [0, keyspacelen)
    int64_t sequential_replacement;       // 0=random, 1=sequential key generation
    int64_t replace_placeholders;         // Enable placeholder replacement
    
    /* Cluster Mode */
    int64_t cluster_mode;                 // Cluster mode enabled
    readFromReplica read_from_replica;    // FROM_PRIMARY_ONLY, FROM_REPLICA_ONLY, FROM_ALL
    int64_t cluster_node_count;           // Total discovered nodes
    struct clusterNode **cluster_nodes;   // All cluster nodes
    int64_t selected_node_count;          // Nodes selected for benchmarking
    struct clusterNode **selected_nodes;  // Selected nodes for round-robin assignment
    
    /* Node Balancing */
    int64_t balance_nodes;                // Enable quota-based balancing
    int64_t balance_quota_step;           // Requests per node per cycle
    int64_t balance_tolerance_pct;        // Allowed imbalance percentage
    int64_t *node_request_counters;       // Per-node request counts (current cycle)
    int64_t *node_quota_remaining;        // Per-node remaining quota
    
    /* Vector Search Configuration */
    int64_t use_search;                   // Vector search mode enabled
    searchIndex search;                   // Index name, algorithm, dim, metric, etc.
    
    /* Dataset Integration */
    int64_t use_dataset;                  // Dataset mode enabled
    int64_t use_filtered_search;          // Metadata-filtered search
    sds dataset_name;                     // Dataset file path
    void *dataset_ctx;                    // Opaque dataset context (mmap'd)
    uint64_t dataset_num_vectors;         // Total vectors in dataset
    uint64_t dataset_num_queries;         // Total query vectors
    uint32_t dataset_num_neighbors;       // Ground truth k value
    _Atomic uint64_t dataset_prefill_counter;  // Insert progress
    _Atomic uint64_t dataset_query_counter;    // Query progress
    
    /* Load Optimizer */
    int64_t optimize_enabled;             // Adaptive optimization enabled
    optimizer_t *optimizer;               // Optimizer instance
    sds optimize_objective;               // "maximize:qps" or "minimize:p99_latency"
    
    /* Metrics */
    struct hdr_histogram *latency_histogram;           // Aggregate latency distribution
    struct hdr_histogram *current_sec_latency_histogram; // Per-second histogram
    
    /* Rate Limiting */
    int64_t rps;                          // Requests per second limit
    atomic_uint_fast64_t last_time_ns;    // Token bucket timestamp
    uint64_t time_per_token;              // Nanoseconds per request
    uint64_t time_per_burst;              // Burst window size
    
    /* ... additional fields ... */
};
```

### 3.2 Client Structure

Each simulated connection is represented by a `client` struct:

```c
typedef struct _client {
    /* Connection */
    valkeyContext *context;              // libvalkey connection context
    sds obuf;                            // Output buffer (command templates)
    size_t written;                      // Bytes already written
    
    /* Cluster Routing */
    char **stagptr;                      // Pointers to {clt} tags in obuf
    size_t staglen;                      // Number of slot tag pointers
    size_t stagfree;                     // Unused tag pointer slots
    struct clusterNode *cluster_node;   // Assigned cluster node
    int64_t slots_last_update;           // Slot config version
    
    /* Timing */
    long long start;                     // Request batch start time
    long long latency;                   // Calculated latency (-1 until first read)
    
    /* Pipeline State */
    int64_t seqlen;                      // Commands per sequence
    int64_t pending;                     // Outstanding replies to receive
    int64_t prefix_pending;              // Auth/SELECT prefix commands
    int64_t prefixlen;                   // Prefix command bytes
    
    /* Thread Assignment */
    int64_t thread_id;                   // Owning thread (-1 for main thread)
    
    /* Dataset Query Tracking (circular queue for recall) */
    uint64_t *dataset_query_indices;     // Queue: which query was sent
    int64_t dataset_query_head;          // Consumer position
    int64_t dataset_query_tail;          // Producer position
    int64_t dataset_query_capacity;      // Queue size
    
    /* Inflight Tracking (for retry on connection error) */
    uint64_t *inflight_indices;          // Queue: dataset indices in flight
    int64_t inflight_head;
    int64_t inflight_tail;
    int64_t inflight_capacity;
    
    /* State Flags */
    uint64_t paused : 1;                 // Rate-limited
    uint64_t reuse : 1;                  // Reusing after pause
    uint64_t request_prepared : 1;       // Placeholders replaced
    int64_t running_queries;             // Active query count
} *client;
```

### 3.3 Thread Structure

```c
typedef struct benchmarkThread {
    int64_t index;                       // Thread index
    pthread_t thread;                    // POSIX thread handle
    aeEventLoop *el;                     // Per-thread event loop
    list *clients;                       // Clients owned by this thread
    list *paused_clients;                // Rate-limited clients
    
    /* Per-thread Node Balancing */
    int64_t *node_request_counters;      // Requests per node (this thread)
    int64_t *node_quota_remaining;       // Remaining quota per node
    
    /* Per-thread Retry Queue */
    uint64_t *retry_queue;               // Failed dataset indices to retry
    int64_t retry_queue_head;
    int64_t retry_queue_tail;
    int64_t retry_queue_capacity;
    int64_t retry_count;
} benchmarkThread;
```

### 3.4 Placeholder System

The placeholder system enables zero-allocation request generation:

```c
/* Placeholder indices */
#define CLUSTER_PLACEHOLDER_INDEX 10    // {clt} position
#define VECTOR_PLACEHOLDER_INDEX 11     // __v_rd__ position
#define DATASET_KEY_PLACEHOLDER_INDEX 12    // __d_key_ph__ position
#define DATASET_VECTOR_PLACEHOLDER_INDEX 13 // __d_vec_ph____ position
#define DATASET_TAG_PLACEHOLDER_INDEX 14    // ___tag_field____ position
// ... additional placeholders

/* Placeholder registry */
static const struct {
    const char *name;
    int64_t len;
} PLACEHOLDERS[] = {
    [0] = {"__rand_int__", 12},    // Generic random integer
    [1] = {"__rand_1st__", 12},    // First random value
    [2] = {"__rand_2nd__", 12},    // Second random value
    // ... up to __rand_9th__
    [CLUSTER_PLACEHOLDER_INDEX] = {"{clt}", 5},
    [VECTOR_PLACEHOLDER_INDEX] = {"__v_rd__", 8},
    [DATASET_KEY_PLACEHOLDER_INDEX] = {"__d_key_ph__", 12},
    [DATASET_VECTOR_PLACEHOLDER_INDEX] = {"__d_vec_ph____", 14},
    [DATASET_TAG_PLACEHOLDER_INDEX] = {"___tag_field____", 16},
    // ...
};

/* Runtime placeholder state (global) */
static struct placeholders {
    size_t cmd_len;                      // Total command length
    size_t count[PLACEHOLDER_NUM_OF];    // Count of each placeholder type
    size_t len[PLACEHOLDER_NUM_OF];      // Length of each placeholder string
    size_t *indices[PLACEHOLDER_NUM_OF]; // Byte offsets in command buffer
    size_t *index_data;                  // Backing allocation for all indices
} placeholders;
```

### 3.5 Cluster Node Structure

```c
typedef struct clusterNode {
    valkeyContext *ctx;                  // Connection for topology queries
    int selected;                        // Selected for benchmarking
    int is_replica;                      // Is this a replica node
    char *ip;                            // Node IP address
    int port;                            // Node port
    sds name;                            // Node ID from CLUSTER NODES
    int flags;                           // Node flags
    sds replicate;                       // Primary ID if replica
    int *slots;                          // Owned slots array
    int slots_count;                     // Number of owned slots
    int *updated_slots;                  // Slots after MOVED refresh
    int updated_slots_count;
    int replicas_count;                  // Number of replicas
    struct serverConfig *server_config;  // Server-side config snapshot
} clusterNode;
```

### 3.6 Dataset Structures

```c
/* Binary dataset header (4KB aligned) */
typedef struct __attribute__((packed)) {
    uint32_t magic;                      // 0xDECDB001
    uint32_t version;                    // Format version
    char dataset_name[256];              // Human-readable name
    uint8_t distance_metric;             // L2, COSINE, IP
    uint8_t dtype;                       // FLOAT32, FLOAT16
    uint8_t has_metadata;                // Metadata presence flag
    uint8_t padding[1];
    uint32_t dim;                        // Vector dimension
    uint64_t num_vectors;                // Total vectors
    uint64_t num_queries;                // Query set size
    uint32_t num_neighbors;              // Ground truth k
    uint32_t vocab_size;                 // Tag vocabulary size
    uint64_t vectors_offset;             // Offset to vector data
    uint64_t queries_offset;             // Offset to query vectors
    uint64_t ground_truth_offset;        // Offset to neighbor lists
    uint64_t vector_metadata_offset;     // Offset to vector tags
    uint64_t query_metadata_offset;      // Offset to query predicates
    uint64_t vocab_offset;               // Offset to vocabulary
    uint8_t reserved[3744];              // Pad to 4096 bytes
} dataset_header_t;

/* Vector ID to cluster tag mapping */
typedef struct {
    uint64_t vector_id;
    char cluster_tag[6];                 // e.g., "{ABC}" + null
} vectorClusterMapping;

typedef struct {
    char *prefix;                        // Key prefix for scanning
    int64_t prefix_len;
    vectorClusterMapping *mappings;      // Dense array indexed by vector_id
    uint64_t capacity;                   // Array size
    uint64_t count;                      // Populated entries
    uint64_t keys_scanned;               // Progress tracking
    int64_t is_cluster_mode_enabled;
    pthread_mutex_t mutex;               // Thread safety
    struct progressBar *progress_bar;    // Visual progress
} clusterTagMap;
```

---

## 4. Threading Model

### 4.1 Overview

The benchmark supports three execution modes:

| Mode | Threads | Event Loops | Use Case |
|------|---------|-------------|----------|
| Single-threaded | 0 | 1 (main) | Simple testing, debugging |
| Multi-threaded | 1-500 | N (per thread) | Production benchmarking |
| Main + Workers | N | N+1 | Maximum throughput |

### 4.2 Thread Initialization Flow

```
main()
  │
  ├─► initBenchmarkThreads(num_threads)
  │     │
  │     ├─► For each thread i in [0, num_threads):
  │     │     ├─► benchmarkThread *t = zcalloc(sizeof(benchmarkThread))
  │     │     ├─► t->el = aeCreateEventLoop(eventloop_size)
  │     │     ├─► t->clients = listCreate()
  │     │     ├─► t->paused_clients = listCreate()
  │     │     ├─► Allocate per-thread retry queue
  │     │     ├─► Allocate per-thread node_request_counters[]
  │     │     ├─► Allocate per-thread node_quota_remaining[]
  │     │     └─► pthread_create(&t->thread, execBenchmarkThread, t)
  │     │
  │     └─► config.threads[i] = t
  │
  ├─► createMissingClients()
  │     │
  │     ├─► Distribute numclients across threads:
  │     │     total_per_thread = numclients / num_threads
  │     │     remainder distributed to first (numclients % num_threads) threads
  │     │
  │     └─► For each client to create:
  │           ├─► thread_id = client_index % num_threads
  │           ├─► client c = createClient(cmd, len, seqlen, NULL, thread_id)
  │           └─► listAddNodeTail(thread->clients, c)
  │
  └─► benchmark()
        │
        └─► For single-threaded: aeMain(config.el)
            For multi-threaded: threads already running aeMain() in execBenchmarkThread()
```

### 4.3 Client-to-Node Assignment

```c
/* Round-robin assignment across selected nodes */
static client createClient(..., int64_t thread_id) {
    int64_t node_idx = 0;
    if (thread_id >= 0) {
        int num_clients = listLength(config.threads[thread_id]->clients);
        node_idx = num_clients % config.selected_node_count;
    } else {
        node_idx = config.liveclients % config.selected_node_count;
    }
    
    clusterNode *node = config.selected_nodes[node_idx];
    c->cluster_node = node;
    c->context = valkeyConnect(node->ip, node->port);
    // ... TLS negotiation, AUTH, etc.
}
```

### 4.4 Synchronization Points

| Resource | Protection | Access Pattern |
|----------|------------|----------------|
| `config.requests_issued` | `_Atomic` | Increment before sending |
| `config.requests_finished` | `_Atomic` | Increment after reply |
| `config.liveclients` | `pthread_mutex_t` | Create/destroy clients |
| `dataset_prefill_counter` | `_Atomic` | Claim next dataset index |
| `recall_stats` | `pthread_mutex_t` | Update recall statistics |
| `config.is_updating_slots` | `_Atomic` | Cluster slot refresh |
| `node_quota_remaining` | Per-thread local | No cross-thread access |

---

## 5. Template and Placeholder System

### 5.1 Template Creation Phase

```
createSearchCmdTemplate()
  │
  ├─► Build RESP-encoded command string with placeholders:
  │   
  │   Example for vec-load (HSET):
  │   "*5\r\n$4\r\nHSET\r\n$<keylen>\r\n{clt}<prefix>__d_key_ph__\r\n
  │    $<fieldlen>\r\n<vector_field>\r\n$<veclen>\r\n__d_vec_ph____<padding>\r\n"
  │   
  │   Example for vec-query (FT.SEARCH):
  │   "*10\r\n$9\r\nFT.SEARCH\r\n$<namelen>\r\n<index>\r\n$<querylen>\r\n
  │    *=>[KNN <k> @<field> $QUERY_VECTOR]\r\n...\r\n$<veclen>\r\n__d_vec_ph____<padding>\r\n"
  │
  └─► Return cmd buffer pointer and length
```

### 5.2 Placeholder Index Building

```c
/* Called once after template creation */
static void scanPlaceholders(const char *cmd, size_t len) {
    memset(&placeholders, 0, sizeof(placeholders));
    placeholders.cmd_len = len;
    
    /* First pass: count occurrences of each placeholder */
    for (int ph = 0; ph < PLACEHOLDER_NUM_OF; ph++) {
        const char *pattern = PLACEHOLDERS[ph].name;
        size_t pattern_len = PLACEHOLDERS[ph].len;
        const char *pos = cmd;
        
        while ((pos = memmem(pos, len - (pos - cmd), pattern, pattern_len))) {
            placeholders.count[ph]++;
            pos += pattern_len;
        }
        placeholders.len[ph] = pattern_len;
    }
    
    /* Allocate backing storage for all indices */
    size_t total_indices = 0;
    for (int ph = 0; ph < PLACEHOLDER_NUM_OF; ph++) {
        total_indices += placeholders.count[ph];
    }
    placeholders.index_data = zmalloc(total_indices * sizeof(size_t));
    
    /* Second pass: record byte offsets */
    size_t *current = placeholders.index_data;
    for (int ph = 0; ph < PLACEHOLDER_NUM_OF; ph++) {
        placeholders.indices[ph] = current;
        const char *pattern = PLACEHOLDERS[ph].name;
        size_t pattern_len = PLACEHOLDERS[ph].len;
        const char *pos = cmd;
        
        while ((pos = memmem(pos, len - (pos - cmd), pattern, pattern_len))) {
            *current++ = (pos - cmd);  // Store byte offset
            pos += pattern_len;
        }
    }
}
```

### 5.3 Runtime Placeholder Replacement

```c
/* Called per request batch in writeHandler() */
static void replacePlaceholders(client c, char *cmd, int pipeline) {
    /* Get placeholder metadata */
    size_t *key_indices = placeholders.indices[DATASET_KEY_PLACEHOLDER_INDEX];
    size_t key_count = placeholders.count[DATASET_KEY_PLACEHOLDER_INDEX];
    size_t *vec_indices = placeholders.indices[DATASET_VECTOR_PLACEHOLDER_INDEX];
    size_t vec_count = placeholders.count[DATASET_VECTOR_PLACEHOLDER_INDEX];
    size_t *cluster_tag_indices = placeholders.indices[CLUSTER_PLACEHOLDER_INDEX];
    
    /* Determine operation mode based on placeholder presence */
    int is_insert = (vec_count > 0 && key_count > 0);
    int is_query = (vec_count > 0 && key_count == 0);
    int is_delete = (key_count > 0 && vec_count == 0);
    
    if (is_insert) {
        /* INSERT: Both key and vector placeholders present */
        for (size_t i = 0; i < key_count; i++) {
            uint64_t dataset_idx, vector_id;
            const char *cluster_tag = NULL;
            
            /* Claim unique dataset index (atomic) */
            do {
                dataset_idx = atomic_fetch_add(&config.dataset_prefill_counter, 1);
                dataset_idx %= config.dataset_num_vectors;
            } while (checkVectorExistsInCluster(&cluster_tag_map, dataset_idx));
            
            /* Get vector data */
            float *vec_write_pos = (float *)(cmd + vec_indices[i]);
            datasetGetVector(config.dataset_ctx, dataset_idx, &vector_id, vec_write_pos);
            
            /* Encode key with cluster tag */
            char *key_write_pos = cmd + key_indices[i];
            encode_vector_key_fixed(key_write_pos, key_len, NULL, cluster_tag, vector_id);
            
            /* Track in-flight for retry on error */
            enqueue_inflight(c, dataset_idx);
            
            /* Update cluster tag mapping */
            addClusterTagMapping(&cluster_tag_map, vector_id, cluster_tag);
        }
    }
    else if (is_query) {
        /* QUERY: Only vector placeholder */
        for (size_t i = 0; i < vec_count; i++) {
            float *vec_write_pos = (float *)(cmd + vec_indices[i]);
            
            /* Select random query vector */
            uint64_t query_idx = fast_random() % config.dataset_num_queries;
            datasetSetQueryVec(config.dataset_ctx, query_idx, vec_write_pos);
            
            /* Track for recall verification */
            enqueue_query_index(c, query_idx);
        }
        c->running_queries++;
    }
    else if (is_delete) {
        /* DELETE: Only key placeholder */
        for (size_t i = 0; i < key_count; i++) {
            uint64_t vector_id;
            const char *cluster_tag;
            
            /* Select vector to delete */
            vector_id = (config.sequential_replacement) 
                ? atomic_fetch_add(&vector_counter, 1) 
                : random();
            vector_id %= config.keyspacelen;
            
            /* Get cluster tag for routing */
            cluster_tag = getClusterTagForVector(&cluster_tag_map, vector_id);
            
            /* Encode key */
            char *key_write_pos = cmd + key_indices[i];
            encode_vector_key_fixed(key_write_pos, key_len, NULL, cluster_tag, vector_id);
        }
    }
    
    /* Replace random integer placeholders */
    for (int ph = 0; ph < PLACEHOLDER_NORMAL_NUM_OF; ph++) {
        for (size_t j = 0; j < placeholders.count[ph]; j++) {
            char *pos = cmd + placeholders.indices[ph][j];
            uint64_t value = (config.sequential_replacement)
                ? atomic_fetch_add(&seq_counter, 1)
                : random();
            value %= config.keyspacelen;
            snprintf(pos, 12, "%012lu", value);  // Fixed width, padded
        }
    }
}
```

---

## 6. Event Loop and I/O Handling

### 6.1 Event Loop Integration

The benchmark uses the `ae` (async event) library from Redis:

```c
/* Event types */
#define AE_READABLE 1
#define AE_WRITABLE 2

/* Main event loop structure */
typedef struct aeEventLoop {
    int maxfd;                           // Highest file descriptor
    void *apidata;                       // epoll/kqueue backend data
    aeFileEvent *events;                 // Registered file events
    aeFiredEvent *fired;                 // Fired events per iteration
    aeTimeEvent *timeEventHead;          // Time event linked list
    // ...
} aeEventLoop;
```

### 6.2 writeHandler Flow

```c
static void writeHandler(aeEventLoop *el, int fd, void *privdata, int mask) {
    client c = privdata;
    
    /* STEP 1: Rate limiting check */
    if (config.balance_nodes && c->cluster_node) {
        int64_t node_idx = findNodeIndex(c->cluster_node);
        long long delay = checkNodeBalanceThrottle(c->thread_id, node_idx, config.pipeline);
        if (delay > 0) {
            pauseClient(c, delay);
            return;
        }
        // Decrement quota, increment counter
        config.threads[c->thread_id]->node_quota_remaining[node_idx] -= config.pipeline;
        config.threads[c->thread_id]->node_request_counters[node_idx] += config.pipeline;
    }
    
    if (config.rps > 0 && !c->reuse) {
        long long delay = acquireTokenOrWait(config.pipeline);
        if (delay > 0) {
            pauseClient(c, delay);
            return;
        }
    }
    
    /* STEP 2: Request preparation (once per batch) */
    if (c->written == 0 && !c->request_prepared) {
        /* Atomic claim of request quota */
        int64_t current = atomic_load(&config.requests_issued);
        int64_t increment = config.pipeline * c->seqlen;
        do {
            if (current >= config.requests) {
                aeDeleteFileEvent(el, fd, AE_WRITABLE);
                return;  // All requests issued
            }
        } while (!atomic_compare_exchange_weak(&config.requests_issued, &current, current + increment));
        
        /* Replace placeholders */
        replacePlaceholders(c, c->obuf + c->prefixlen, config.pipeline);
        c->start = ustime();
        c->latency = -1;
        c->request_prepared = 1;
    }
    
    /* STEP 3: Non-blocking write */
    const ssize_t writeLen = sdslen(c->obuf) - c->written;
    if (writeLen > 0) {
        ssize_t nwritten = cliWriteConn(c->context, c->obuf + c->written, writeLen);
        if (nwritten == -1 && errno != EAGAIN) {
            freeClient(c);
            return;
        }
        if (nwritten > 0) {
            c->written += nwritten;
        }
        if (c->written < sdslen(c->obuf)) {
            // Partial write, wait for next writable event
            return;
        }
    }
    
    /* STEP 4: Write complete, switch to read mode */
    aeDeleteFileEvent(el, fd, AE_WRITABLE);
    aeCreateFileEvent(el, fd, AE_READABLE, readHandler, c);
}
```

### 6.3 readHandler Flow

```c
static void readHandler(aeEventLoop *el, int fd, void *privdata, int mask) {
    client c = privdata;
    void *reply = NULL;
    
    /* Calculate latency on first read */
    if (c->latency < 0) {
        c->latency = ustime() - c->start;
    }
    
    /* Read from socket */
    if (valkeyBufferRead(c->context) != VALKEY_OK) {
        atomic_fetch_add(&config.connection_errors, 1);
        atomic_fetch_add(&config.lost_responses, c->pending);
        freeClient(c);
        return;
    }
    
    /* Process all available replies */
    while (c->pending > 0) {
        if (valkeyGetReply(c->context, &reply) != VALKEY_OK) {
            freeClient(c);
            return;
        }
        if (reply == NULL) break;  // No more complete replies
        
        valkeyReply *r = reply;
        
        /* Handle cluster redirects */
        if (r->type == VALKEY_REPLY_ERROR) {
            if (strncmp(r->str, "MOVED", 5) == 0 || 
                strncmp(r->str, "ASK", 3) == 0) {
                fetchClusterSlotsConfiguration();
            }
            // Handle error appropriately
        }
        
        /* Command-specific response processing */
        if (c->running_queries > 0 && r->type == VALKEY_REPLY_ARRAY) {
            /* FT.SEARCH response: verify recall */
            uint64_t query_idx = dequeue_query_index(c);
            processQueryResults(r, query_idx);  // Computes recall
            c->running_queries--;
        }
        
        /* Skip prefix commands (AUTH, SELECT) */
        if (c->prefix_pending > 0) {
            c->prefix_pending--;
            freeReplyObject(reply);
            continue;
        }
        
        /* Record metrics */
        if (c->latency > 0 && !config.csv) {
            hdr_record_value(config.latency_histogram, c->latency);
        }
        
        c->pending--;
        freeReplyObject(reply);
        
        int64_t finished = atomic_fetch_add(&config.requests_finished, 1) + 1;
        if (finished >= config.requests) {
            clientDone(c);
            return;
        }
    }
    
    /* All replies received, prepare for next batch */
    if (c->pending == 0) {
        resetClient(c);
    }
}
```

---

## 7. Workload Scheduling

### 7.1 Current: Sequential Workload Execution

```c
/* Current implementation: workloads run one after another */
void main() {
    // ...
    
    do {
        if (test_is_selected("vec-load")) {
            char *cmd;
            len = createInsertCmdTemplate(&cmd);
            benchmark("VEC-LOAD", cmd, len);  // Runs to completion
            zfree(cmd);
        }
        
        if (test_is_selected("vec-query")) {
            char *cmd;
            len = createSearchCmdTemplate(&cmd);
            benchmark("VEC-QUERY", cmd, len);  // Runs to completion
            zfree(cmd);
        }
        
        // ... other workloads ...
        
    } while (config.loop && !interrupted);
}
```

### 7.2 Future: Mixed/Parallel Workload Scheduling

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      WORKLOAD SCHEDULER (Future)                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Configuration:                                                          │
│    workloads: [                                                          │
│      { type: "vec-query", ratio: 0.80 },                                │
│      { type: "vec-insert", ratio: 0.15 },                               │
│      { type: "vec-delete", ratio: 0.05 }                                │
│    ]                                                                     │
│                                                                          │
│  Scheduling Algorithm:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ For each client requesting work:                                    ││
│  │   1. Generate random r in [0.0, 1.0)                                ││
│  │   2. Select workload w where cumulative_ratio[w-1] <= r < cumulative││
│  │   3. Assign template[w] to client                                   ││
│  │   4. Replace placeholders according to workload type                ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  Implementation:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ typedef struct {                                                     ││
│  │     const char *type;           // "vec-query", "vec-insert", etc.  ││
│  │     double ratio;               // Probability [0.0, 1.0]           ││
│  │     double cumulative;          // Precomputed cumulative sum       ││
│  │     char *cmd_template;         // Pre-built command template       ││
│  │     size_t cmd_len;                                                 ││
│  │     struct placeholders ph;     // Per-workload placeholder indices ││
│  │ } workload_t;                                                       ││
│  │                                                                     ││
│  │ typedef struct {                                                     ││
│  │     workload_t *workloads;                                          ││
│  │     size_t num_workloads;                                           ││
│  │     _Atomic uint64_t *workload_counters;  // Per-type stats         ││
│  │ } workload_scheduler_t;                                             ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  Client Changes:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ typedef struct _client {                                             ││
│  │     // ... existing fields ...                                      ││
│  │     workload_t *active_workload;  // Currently assigned workload    ││
│  │     char *workload_obuf;          // Per-client buffer for workload ││
│  │ } *client;                                                          ││
│  │                                                                     ││
│  │ // In writeHandler():                                               ││
│  │ workload_t *w = scheduler_select_workload(&scheduler);              ││
│  │ memcpy(c->workload_obuf, w->cmd_template, w->cmd_len);              ││
│  │ replacePlaceholders(c, c->workload_obuf, config.pipeline);          ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Response Processing and Recall Verification

### 8.1 Query Result Processing

```c
static void processQueryResults(valkeyReply *reply, uint64_t query_idx) {
    /* Parse FT.SEARCH response format:
     * [total_count, doc1_id, [fields], doc2_id, [fields], ...]
     * or with NOCONTENT:
     * [total_count, doc1_id, doc2_id, ...]
     */
    
    if (!reply || reply->type != VALKEY_REPLY_ARRAY || reply->elements < 1) {
        fprintf(stderr, "Invalid search result format\n");
        return;
    }
    
    int64_t num_results = reply->element[0]->integer;
    
    /* Get ground truth */
    dataset_neighbors_t *gt = config.use_filtered_search
        ? dataset_get_filtered_neighbors(config.dataset_ctx, query_idx)
        : datasetGetNeighbors(config.dataset_ctx, query_idx);
    
    if (!gt || gt->count == 0) {
        fprintf(stderr, "No ground truth for query %lu\n", query_idx);
        return;
    }
    
    /* Extract returned IDs */
    size_t expected_k = (size_t)config.search.k;
    uint64_t result_ids[expected_k];
    size_t actual_results = 0;
    
    size_t stride = config.search.nocontent ? 1 : 2;  // Skip field arrays
    for (size_t i = 1; i < reply->elements && actual_results < expected_k; i += stride) {
        valkeyReply *id_reply = reply->element[i];
        if (id_reply->type != VALKEY_REPLY_STRING) continue;
        
        /* Parse vector ID from key: "{clt}prefix123" → 123 */
        uint64_t vec_id = parse_vector_id_from_key(id_reply->str, id_reply->len);
        result_ids[actual_results++] = vec_id;
    }
    
    /* Build ground truth ID set for fast lookup */
    uint64_t gt_ids[expected_k];
    size_t gt_count = MIN(gt->count, expected_k);
    for (size_t i = 0; i < gt_count; i++) {
        gt_ids[i] = gt->ids[i];
    }
    
    /* Sort both for set intersection */
    qsort(result_ids, actual_results, sizeof(uint64_t), uint64_cmp);
    qsort(gt_ids, gt_count, sizeof(uint64_t), uint64_cmp);
    
    /* Count matches */
    int64_t matches = 0;
    size_t ri = 0, gi = 0;
    while (ri < actual_results && gi < gt_count) {
        if (result_ids[ri] == gt_ids[gi]) {
            matches++;
            ri++; gi++;
        } else if (result_ids[ri] < gt_ids[gi]) {
            ri++;
        } else {
            gi++;
        }
    }
    
    /* Compute recall */
    float recall = (gt_count > 0) ? (float)matches / gt_count : 0.0f;
    
    /* Update global statistics (thread-safe) */
    updateRecallStats(recall);
    
    /* Cleanup */
    if (config.use_filtered_search) {
        dataset_free_neighbors(gt);
    }
}
```

### 8.2 Recall Statistics Structure

```c
typedef struct {
    uint64_t total_queries;        // Total queries processed
    uint64_t total_matches;        // Sum of matches across queries
    double sum_recall;             // Sum of recall values (for average)
    double min_recall;             // Minimum recall observed
    double max_recall;             // Maximum recall observed
    uint64_t perfect_recalls;      // Count where recall == 1.0
    uint64_t zero_recalls;         // Count where recall == 0.0
} recallStats;

static recallStats dataset_recall_stats;
pthread_mutex_t recall_stats_mutex = PTHREAD_MUTEX_INITIALIZER;

static void updateRecallStats(float recall) {
    pthread_mutex_lock(&recall_stats_mutex);
    
    dataset_recall_stats.total_queries++;
    dataset_recall_stats.sum_recall += recall;
    
    if (recall < dataset_recall_stats.min_recall)
        dataset_recall_stats.min_recall = recall;
    if (recall > dataset_recall_stats.max_recall)
        dataset_recall_stats.max_recall = recall;
    
    if (recall >= 0.9999)
        dataset_recall_stats.perfect_recalls++;
    if (recall < 0.0001)
        dataset_recall_stats.zero_recalls++;
    
    dataset_recall_stats.total_matches += (uint64_t)(recall * config.search.k);
    
    pthread_mutex_unlock(&recall_stats_mutex);
}
```

---

## 9. Cluster Management

### 9.1 Topology Discovery

```c
/* Cluster topology discovery flow */
static int fetchClusterConfiguration() {
    valkeyReply *reply = valkeyCommand(config.conn_ctx, "CLUSTER NODES");
    
    /* Parse CLUSTER NODES response:
     * <id> <ip:port@cport> <flags> <master> <ping-sent> <pong-recv> <config-epoch> <link-state> <slot> ...
     */
    
    for (each line in reply) {
        clusterNode *node = parseNodeLine(line);
        
        /* Parse flags */
        node->is_replica = (strstr(flags, "slave") != NULL);
        int is_primary = (strstr(flags, "master") != NULL);
        
        /* Parse slots for primaries */
        if (is_primary) {
            parseSlotRanges(line, &node->slots, &node->slots_count);
        }
        
        /* Filter based on read_from_replica setting */
        if (isSelected(is_primary)) {
            node->selected = 1;
        }
        
        addNodeToList(node);
    }
    
    /* Build slot-to-node mapping */
    buildSlotMapping();
}

/* Slot mapping: O(1) lookup */
static clusterNode *slot_to_node[CLUSTER_SLOTS];  // 16384 slots

static void buildSlotMapping() {
    for (int i = 0; i < config.cluster_node_count; i++) {
        clusterNode *node = config.cluster_nodes[i];
        for (int j = 0; j < node->slots_count; j++) {
            slot_to_node[node->slots[j]] = node;
        }
    }
}
```

### 9.2 Node Balancing Algorithm

```c
/*
 * Quota-based fair node balancing
 *
 * Problem: Nodes with different latencies process requests at different rates.
 *          Fast nodes complete their work while slow nodes are still processing.
 *          This leads to uneven load distribution and skewed benchmarks.
 *
 * Solution: Quota-based throttling
 *   1. Each node starts with quota = balance_quota_step (e.g., 1000)
 *   2. Before sending, check if node has quota remaining
 *   3. If quota exhausted:
 *      a. Find min_completed = MIN(node_request_counters[i])
 *      b. If min_completed == 0: throttle (wait for slowest)
 *      c. Else: add quota_to_add = min_completed * (100 + tolerance_pct) / 100
 *              to all nodes, reset counters
 *   4. Result: All nodes stay within tolerance_pct of each other
 */

static long long checkNodeBalanceThrottle(int64_t thread_id, int64_t node_idx, int64_t tokens) {
    int64_t *node_quota_remaining;
    int64_t *node_request_counters;
    
    if (thread_id == -1) {
        /* Single-threaded: global counters */
        node_quota_remaining = config.node_quota_remaining;
        node_request_counters = config.node_request_counters;
    } else {
        /* Multi-threaded: per-thread counters */
        node_quota_remaining = config.threads[thread_id]->node_quota_remaining;
        node_request_counters = config.threads[thread_id]->node_request_counters;
    }
    
    /* Check if node has quota */
    if (node_quota_remaining[node_idx] >= tokens) {
        return 0;  // No throttle needed
    }
    
    /* Find slowest node's progress */
    int64_t min_completed = INT64_MAX;
    for (int64_t i = 0; i < config.selected_node_count; i++) {
        if (node_request_counters[i] < min_completed) {
            min_completed = node_request_counters[i];
        }
    }
    
    if (min_completed == 0) {
        return 1;  // Throttle: slowest node hasn't made progress
    }
    
    /* Start new cycle: add quota based on slowest node's progress */
    int64_t quota_to_add = (min_completed * (100 + config.balance_tolerance_pct)) / 100;
    for (int64_t i = 0; i < config.selected_node_count; i++) {
        node_quota_remaining[i] += quota_to_add;
        node_request_counters[i] = 0;
    }
    
    return 0;  // Quota replenished
}
```

---

## 10. Load Optimization

### 10.1 Optimizer Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        LOAD OPTIMIZER                                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PARAMETERS (Tunable):                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ Name          Group        Range        Step   Effect                ││
│  │ ─────────────────────────────────────────────────────────────────── ││
│  │ ef_search     RECALL       [10, 1000]   10     Recall, Latency       ││
│  │ clients       THROUGHPUT   [1, 1000]    10     QPS, Latency          ││
│  │ threads       THROUGHPUT   [1, 32]      1      QPS, CPU              ││
│  │ pipeline      THROUGHPUT   [1, 100]     5      QPS, Latency          ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  OPTIMIZATION PHASES:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                                                                     ││
│  │  PHASE_INIT ──────► PHASE_FEASIBILITY ──────► PHASE_RECALL         ││
│  │                           │                        │                ││
│  │     Collect first         │ Exponential           │ Binary search  ││
│  │     measurement           │ resource growth       │ on ef_search   ││
│  │                           │ until feasible        │ to minimize    ││
│  │                           │                        │                ││
│  │                           ▼                        ▼                ││
│  │              PHASE_THROUGHPUT ◄────────────────────┘                ││
│  │                     │                                               ││
│  │                     │ Grid search on clients, threads, pipeline    ││
│  │                     │                                               ││
│  │                     ▼                                               ││
│  │              PHASE_HILL_CLIMB                                       ││
│  │                     │                                               ││
│  │                     │ Joint gradient descent on all parameters     ││
│  │                     │                                               ││
│  │                     ▼                                               ││
│  │              PHASE_REFINEMENT                                       ││
│  │                     │                                               ││
│  │                     │ Coordinate descent, one param at a time      ││
│  │                     │                                               ││
│  │                     ▼                                               ││
│  │              PHASE_CONVERGED                                        ││
│  │                                                                     ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  CONVERGENCE CRITERIA:                                                   │
│    - Objective improvement < 2% for 5 consecutive iterations            │
│    - All parameters at bounds with no improvement                       │
│    - Maximum iterations reached (default: 100)                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Optimizer Step Function

```c
status_t optimizer_step(optimizer_t *opt, const double metrics[METRIC_COUNT]) {
    /* Record measurement */
    measurement_t *m = &opt->history[opt->history_size++];
    memcpy(m->metrics, metrics, sizeof(m->metrics));
    snapshot_current_params(opt, m);
    m->iteration = opt->total_iterations++;
    
    /* Evaluate constraints */
    m->constraints_satisfied = evaluate_constraints(opt, metrics);
    
    /* Compute objective score */
    m->objective_score = compute_objective(opt, metrics);
    
    /* Update best solutions */
    if (m->constraints_satisfied) {
        if (!opt->has_feasible_solution || m->objective_score > opt->best_feasible.objective_score) {
            opt->best_feasible = *m;
        }
        opt->has_feasible_solution = true;
    }
    
    /* Early abort check for catastrophically poor configs */
    if (should_abort_early(opt, m)) {
        reset_to_best_feasible(opt);
        return STATUS_ERROR;
    }
    
    /* Phase-specific logic */
    switch (opt->phase) {
        case PHASE_INIT:
            opt->baseline_objective = m->objective_score;
            opt->phase = PHASE_FEASIBILITY;
            break;
            
        case PHASE_FEASIBILITY:
            if (m->constraints_satisfied) {
                opt->phase = PHASE_RECALL;
                init_binary_search_for_recall(opt);
            } else {
                double_resources(opt);  // Exponential growth
            }
            break;
            
        case PHASE_RECALL:
            if (update_binary_search(opt, m->constraints_satisfied)) {
                opt->phase = PHASE_THROUGHPUT;
                init_grid_search_for_throughput(opt);
            }
            break;
            
        case PHASE_THROUGHPUT:
            if (update_grid_search(opt, m->objective_score, m->constraints_satisfied)) {
                opt->phase = PHASE_HILL_CLIMB;
            }
            break;
            
        case PHASE_HILL_CLIMB:
            estimate_gradients(opt);
            apply_gradient_update(opt, PARAM_GROUP_MIXED);
            if (check_convergence(opt)) {
                opt->phase = PHASE_REFINEMENT;
            }
            break;
            
        case PHASE_REFINEMENT:
            coordinate_descent_step(opt);
            if (check_convergence(opt)) {
                opt->phase = PHASE_CONVERGED;
                return STATUS_CONVERGED;
            }
            break;
            
        case PHASE_CONVERGED:
            return STATUS_CONVERGED;
    }
    
    return opt->iterations_since_change < opt->stabilization_window
        ? STATUS_WAIT_STABILIZATION
        : STATUS_OK;
}
```

---

## 11. Dataset Integration

### 11.1 Memory-Mapped Dataset Access

```c
/* Dataset initialization with mmap */
dataset_ctx_t* dataset_init(const char *path, dataset_info_t *info) {
    dataset_ctx_t *ctx = calloc(1, sizeof(dataset_ctx_t));
    
    /* Open file */
    ctx->fd = open(path, O_RDONLY);
    struct stat st;
    fstat(ctx->fd, &st);
    ctx->file_size = st.st_size;
    
    /* Memory map entire file */
    ctx->mmap_base = mmap(NULL, ctx->file_size, PROT_READ, MAP_PRIVATE, ctx->fd, 0);
    
    /* Parse header */
    dataset_header_t *header = (dataset_header_t *)ctx->mmap_base;
    if (header->magic != DATASET_MAGIC) {
        return NULL;
    }
    
    /* Set up data pointers (all pointing into mmap'd region) */
    ctx->vectors = (float *)(ctx->mmap_base + header->vectors_offset);
    ctx->queries = (float *)(ctx->mmap_base + header->queries_offset);
    ctx->ground_truth = (uint64_t *)(ctx->mmap_base + header->ground_truth_offset);
    
    /* Populate info struct */
    info->dim = header->dim;
    info->num_vectors = header->num_vectors;
    info->num_queries = header->num_queries;
    info->num_neighbors = header->num_neighbors;
    
    return ctx;
}

/* Zero-copy vector access */
int datasetGetVector(dataset_ctx_t *ctx, uint64_t index, uint64_t *id_out, float *vec_out) {
    if (index >= ctx->header->num_vectors) return -1;
    
    /* Direct pointer into mmap'd region */
    float *src = ctx->vectors + (index * ctx->header->dim);
    
    /* Copy to output buffer (in command template) */
    memcpy(vec_out, src, ctx->header->dim * sizeof(float));
    
    *id_out = index;  // Vector ID == array index
    return 0;
}

/* Ground truth access */
dataset_neighbors_t* datasetGetNeighbors(dataset_ctx_t *ctx, uint64_t query_idx) {
    if (query_idx >= ctx->header->num_queries) return NULL;
    
    dataset_neighbors_t *result = malloc(sizeof(dataset_neighbors_t));
    result->count = ctx->header->num_neighbors;
    result->ids = malloc(result->count * sizeof(uint64_t));
    result->dists = malloc(result->count * sizeof(float));
    
    /* Ground truth layout: [query0_neighbors..., query1_neighbors..., ...] */
    size_t offset = query_idx * ctx->header->num_neighbors;
    
    memcpy(result->ids, ctx->ground_truth + offset, result->count * sizeof(uint64_t));
    /* Distances stored after all IDs */
    memcpy(result->dists, ctx->ground_truth_dists + offset, result->count * sizeof(float));
    
    return result;
}
```

### 11.2 Cluster Tag Map Building

```c
/* Parallel cluster scan to build vector ID → cluster tag mapping */
int buildVectorIdMappings(int64_t is_cluster_mode, const char *prefix,
                         clusterNode **nodes, int64_t node_count,
                         clusterTagMap *tag_map,
                         keyProcessorCallback key_processor,
                         connectionFactoryCallback connection_factory) {
    
    /* Create scan pattern */
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "%s*", prefix);
    
    /* Initialize tag map */
    tag_map->prefix = strdup(prefix);
    tag_map->prefix_len = strlen(prefix);
    tag_map->is_cluster_mode_enabled = is_cluster_mode;
    
    /* Configure cluster scan */
    clusterScanConfig scan_config;
    initClusterScanConfig(&scan_config, pattern, nodes, node_count, key_processor, tag_map);
    scan_config.silent_mode = 1;
    
    /* Set TLS/auth connection factory */
    setClusterScanConnectionFactory(&scan_config, connection_factory);
    
    /* Set performance parameters */
    setClusterScanPerformance(&scan_config, 
        1000,        // batch_size: keys per SCAN
        node_count,  // max_workers: one per node
        50000);      // progress_interval
    
    /* Execute parallel scan */
    clusterScanResults results;
    int ret = executeClusterScan(&scan_config, &results);
    
    printf("[VECTOR-MAPPING] Built %lu mappings from %lu keys in %.2f sec (%.1f keys/sec)\n",
           tag_map->count, results.total_keys_processed,
           results.total_scan_time_ms / 1000.0, results.keys_per_second);
    
    return ret;
}

/* Per-key callback for scan (called from worker threads) */
int64_t vectorKeyProcessor(const char *key, void *user_data, int64_t thread_id) {
    clusterTagMap *tag_map = (clusterTagMap *)user_data;
    
    /* Parse key: "{ABC}prefix123" → cluster_tag="{ABC}", vector_id=123 */
    const char *tag_start = strchr(key, '{');
    const char *tag_end = strchr(key, '}');
    const char *id_start = key + tag_map->prefix_len;
    
    if (tag_map->is_cluster_mode_enabled) {
        id_start = tag_end + 1 + tag_map->prefix_len;
    }
    
    char cluster_tag[6] = {0};
    if (tag_start && tag_end) {
        size_t tag_len = tag_end - tag_start + 1;
        memcpy(cluster_tag, tag_start, MIN(tag_len, 5));
    }
    
    uint64_t vector_id = strtoull(id_start, NULL, 10);
    
    /* Thread-safe insertion (mutex inside) */
    addClusterTagMapping(tag_map, vector_id, cluster_tag);
    
    return 0;  // Success
}
```

---

## 12. Metrics and Statistics

### 12.1 HDR Histogram Integration

```c
/* Latency histogram configuration */
#define CONFIG_LATENCY_HISTOGRAM_MIN_VALUE 10L          // >= 10 usec
#define CONFIG_LATENCY_HISTOGRAM_MAX_VALUE 3000000L     // <= 3 sec

/* Global histogram */
struct hdr_histogram *latency_histogram;

void initHistograms() {
    hdr_init(CONFIG_LATENCY_HISTOGRAM_MIN_VALUE,
             CONFIG_LATENCY_HISTOGRAM_MAX_VALUE,
             config.precision,  // 3 = 0.1% precision
             &config.latency_histogram);
}

/* Record latency in readHandler */
if (c->latency > 0) {
    hdr_record_value(config.latency_histogram, c->latency);
}

/* Report percentiles */
void printLatencyReport() {
    double avg = hdr_mean(config.latency_histogram) / 1000.0;
    double p50 = hdr_value_at_percentile(config.latency_histogram, 50.0) / 1000.0;
    double p99 = hdr_value_at_percentile(config.latency_histogram, 99.0) / 1000.0;
    double max = hdr_max(config.latency_histogram) / 1000.0;
    
    printf("Latency: avg=%.3fms, p50=%.3fms, p99=%.3fms, max=%.3fms\n",
           avg, p50, p99, max);
}
```

### 12.2 Per-Node Statistics Collection

```c
/* INFO command field collection for per-node analysis */
typedef struct infoFieldType {
    char *prefix_match;                  // Field prefix to match
    matcherCallBack match;               // Custom matcher function
    ParseConfig parse_config;            // How to parse the value
    AggregationType aggregation_type;    // SUM, AVG, MAX, MINMAX
    DisplayFormat display_format;        // INTEGER, MEMORY_MB, PERCENTAGE
    DiffType diff_type;                  // RATE_COUNT, MEMORY_GROWTH, etc.
    int track_per_node;                  // Store per-node values
    readFromReplica nodes_to_aggregate;  // Which nodes to include
    int is_last;                         // Stop processing after this field
} infoFieldType;

/* Cluster snapshot for temporal diff */
typedef struct clusterSnapshot {
    long long timestamp_ms;
    int num_fields;
    fieldSnapshot *fields;
    int num_nodes;
    sds *node_identifiers;
} clusterSnapshot;

/* Collect INFO across cluster */
clusterSnapshot* getInfoCluster(int node_count, clusterNode **nodes,
                                enum valkeyConnectionType ct) {
    clusterSnapshot *snapshot = zcalloc(sizeof(clusterSnapshot));
    snapshot->timestamp_ms = mstime();
    snapshot->num_nodes = node_count;
    
    for (int i = 0; i < node_count; i++) {
        valkeyContext *ctx = getValkeyContext(ct, nodes[i]->ip, nodes[i]->port);
        valkeyReply *reply = valkeyCommand(ctx, "INFO ALL");
        
        parseInfoFields(reply->str, snapshot, i);
        
        freeReplyObject(reply);
        valkeyFree(ctx);
    }
    
    return snapshot;
}

/* Compare snapshots for temporal analysis */
void compareInfoSnapshots(clusterSnapshot *before, clusterSnapshot *after) {
    double elapsed_sec = (after->timestamp_ms - before->timestamp_ms) / 1000.0;
    
    for (int f = 0; f < before->num_fields; f++) {
        fieldSnapshot *old = &before->fields[f];
        fieldSnapshot *new = &after->fields[f];
        
        int64_t delta = new->value - old->value;
        
        switch (field_types[f].diff_type) {
            case DIFF_RATE_COUNT:
                printf("%s: %.2f/sec\n", old->field_name, delta / elapsed_sec);
                break;
            case DIFF_MEMORY_GROWTH:
                printf("%s: %.2f MB/sec\n", old->field_name, 
                       (delta / 1048576.0) / elapsed_sec);
                break;
            // ...
        }
    }
}
```

---

## 13. Future Roadmap Integration

Based on `TODO.md`, the following features require architectural support:

### 13.1 Mixed Workloads (TODO #23)

**Required Changes:**
- `workload_scheduler_t` struct with ratio-based selection
- Per-client template pointer instead of shared `config.obuf`
- Workload-aware placeholder replacement
- Separate statistics per workload type

### 13.2 Multiple Indexes (TODO #20)

**Required Changes:**
- Array of `searchIndex` structs
- Index selection in command template
- Per-index ground truth loading
- Index-specific recall tracking

### 13.3 JSON Support (TODO #18)

**Required Changes:**
- `JSON.SET` command template builder
- JSON path-based field addressing
- Integration with existing placeholder system
- Response parsing for JSON data types

### 13.4 Dynamic Scaling (TODO #10)

**Required Changes:**
- Hot-add/remove of threads without restart
- Client migration between threads
- Connection pool resizing
- Histogram merging on scale-down

### 13.5 Advanced Ranking Metrics (TODO #33)

**Required Changes:**
```c
typedef struct rankingMetrics {
    double map;         // Mean Average Precision
    double ndcg;        // Normalized DCG
    double mrr;         // Mean Reciprocal Rank
} rankingMetrics;

/* MAP calculation */
double computeMAP(uint64_t *results, uint64_t *ground_truth, size_t k) {
    double ap = 0.0;
    int relevant = 0;
    
    for (size_t i = 0; i < k; i++) {
        if (is_in_ground_truth(results[i], ground_truth, k)) {
            relevant++;
            ap += (double)relevant / (i + 1);  // Precision at rank i+1
        }
    }
    
    return relevant > 0 ? ap / k : 0.0;
}
```

---

## 14. Module Dependencies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       MODULE DEPENDENCY GRAPH                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  External Libraries:                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ libvalkey│  │hdr_histog│  │ pthread  │  │ openssl  │               │
│  │ (RESP)   │  │ (latency)│  │ (threads)│  │ (TLS)    │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │             │             │             │                       │
│       └──────┬──────┴──────┬──────┴──────┬──────┘                      │
│              │             │             │                              │
│              v             v             v                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    valkey-benchmark.c (main)                      │  │
│  │  - CLI parsing, workload orchestration, event loop                │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│              │                │                │                        │
│              v                v                v                        │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐         │
│  │ search_utils  │  │  dataset_api    │  │ load_optimizer   │         │
│  │ - cluster ops │  │  - mmap dataset │  │ - param tuning   │         │
│  │ - INFO parse  │  │  - GT access    │  │ - gradient desc  │         │
│  └───────┬───────┘  └────────┬────────┘  └────────┬─────────┘         │
│          │                   │                    │                    │
│          v                   v                    v                    │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐         │
│  │ mapping_scan  │  │dataset_id_mapping│  │ config_persist   │         │
│  │ - SCAN engine │  │ - vec_id→tag    │  │ - save/load cfg  │         │
│  └───────────────┘  └─────────────────┘  └──────────────────┘         │
│                                                                        │
│  Redis/Valkey Utilities (vendored):                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │   ae     │  │   sds    │  │  dict    │  │ zmalloc  │               │
│  │(ev loop)│  │ (strings)│  │ (hash)   │  │ (alloc)  │               │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘               │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Build and Test Commands

```bash
# Build (release)
./build.sh --release

# Build (debug with sanitizers)
./build.sh --debug --sanitizers

# Run unit tests
./build.sh --run-tests

# Run end-to-end tests
cd tests
python -m pytest -v

# Example benchmark command
./build/bin/valkey-benchmark \
    -h <host> --cluster \
    --rfr all \
    --dataset /path/to/dataset.bin \
    -t vec-query \
    --search \
    --vector-dim 1536 \
    --search-name my-index \
    --search-prefix vec: \
    -n 100000 \
    -c 200 \
    --threads 10 \
    --ef-search 256 \
    --nocontent \
    --k 100
```

---

## 16. Summary

This document provides a comprehensive technical reference for `valkey-search-benchmark`, covering:

1. **Architecture**: Multi-threaded, event-driven design with per-thread event loops
2. **Zero-allocation**: Template-based request generation with in-place placeholder replacement
3. **Cluster support**: Full topology awareness, slot routing, MOVED/ASK handling
4. **Dataset integration**: Memory-mapped binary datasets with ground truth verification
5. **Optimization**: Phased parameter tuning with gradient descent and constraints
6. **Metrics**: HDR histograms, recall statistics, per-node analysis
7. **Future extensibility**: Support for mixed workloads, multiple indexes, JSON

This design enables high-performance benchmarking while maintaining accuracy for vector search quality assessment.
