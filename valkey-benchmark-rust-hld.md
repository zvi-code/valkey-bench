# High-Level Design: valkey-search-benchmark Rust Rewrite

**Version:** 1.0  
**Target:** Rust implementation using valkey-glide client  
**Purpose:** Comprehensive architectural blueprint for language migration

---

## 1. Executive Summary

This document describes the Rust rewrite of `valkey-search-benchmark`, replacing the C implementation's custom connection handling with the `valkey-glide` Rust client library. The design preserves all existing functionality while leveraging Rust's safety guarantees, async runtime, and the battle-tested valkey-glide infrastructure.

### 1.1 Key Architectural Decisions

| Aspect | C Implementation | Rust Implementation |
|--------|------------------|---------------------|
| Runtime | ae event loop (single-threaded per worker) | Tokio async runtime (multi-threaded) |
| Connections | Manual socket + RESP parsing | valkey-glide Client abstraction |
| Cluster | Manual slot routing, MOVED/ASK handling | Automatic via glide-core |
| Memory | Manual allocation, sds strings | Rust ownership, `bytes` crate |
| Concurrency | pthreads + atomic counters | Tokio tasks + `std::sync::atomic` |

### 1.2 Critical Integration Notes

**⚠️ GAP: valkey-glide Rust crate is NOT published to crates.io.**  
Must use git dependency:
```toml
[dependencies]
glide-core = { git = "https://github.com/valkey-io/valkey-glide.git", branch = "main" }
```
Requires compile-time environment variables: `GLIDE_NAME`, `GLIDE_VERSION`.

---

## 2. Project Structure

```
valkey-search-benchmark/
├── Cargo.toml
├── build.rs                    # Set GLIDE_NAME/GLIDE_VERSION env vars
├── src/
│   ├── main.rs                 # Entry point, CLI parsing
│   ├── lib.rs                  # Library root, re-exports
│   │
│   ├── config/
│   │   ├── mod.rs
│   │   ├── cli.rs              # CLI argument definitions (clap)
│   │   ├── benchmark_config.rs # Global config struct
│   │   ├── search_config.rs    # Vector search params
│   │   └── tls_config.rs       # TLS certificate configuration
│   │
│   ├── client/
│   │   ├── mod.rs
│   │   ├── glide_wrapper.rs    # Thin wrapper over glide-core
│   │   ├── client_pool.rs      # Client pool management
│   │   ├── connection_factory.rs # Client creation with config
│   │   └── routing.rs          # Node selection / balancing
│   │
│   ├── cluster/
│   │   ├── mod.rs
│   │   ├── topology.rs         # Cluster node discovery
│   │   ├── node.rs             # ClusterNode struct
│   │   ├── slot_map.rs         # Slot-to-node mapping (if exposed)
│   │   └── node_selector.rs    # RFR (read-from-replica) logic
│   │
│   ├── workload/
│   │   ├── mod.rs
│   │   ├── workload_type.rs    # Enum: VecLoad, VecQuery, etc.
│   │   ├── command_template.rs # Template builder
│   │   ├── placeholder.rs      # Placeholder system
│   │   └── scheduler.rs        # Workload selection (ratio-based)
│   │
│   ├── dataset/
│   │   ├── mod.rs
│   │   ├── binary_dataset.rs   # Memory-mapped dataset access
│   │   ├── header.rs           # Dataset header parsing
│   │   ├── ground_truth.rs     # Neighbor list access
│   │   ├── cluster_tag_map.rs  # Vector ID → cluster tag
│   │   └── scanner.rs          # SCAN-based key mapping
│   │
│   ├── benchmark/
│   │   ├── mod.rs
│   │   ├── orchestrator.rs     # Main benchmark loop
│   │   ├── worker.rs           # Per-worker task
│   │   ├── rate_limiter.rs     # Token bucket rate limiting
│   │   └── node_balancer.rs    # Quota-based node balancing
│   │
│   ├── metrics/
│   │   ├── mod.rs
│   │   ├── latency.rs          # HDR histogram wrapper
│   │   ├── recall.rs           # Recall statistics
│   │   ├── counters.rs         # Atomic request counters
│   │   ├── reporter.rs         # Output formatting (text/CSV/JSON)
│   │   └── info_collector.rs   # INFO command aggregation
│   │
│   ├── optimizer/
│   │   ├── mod.rs
│   │   ├── optimizer.rs        # Main optimizer state machine
│   │   ├── phases.rs           # Phase definitions
│   │   ├── parameters.rs       # Tunable parameter definitions
│   │   └── constraints.rs      # Constraint evaluation
│   │
│   └── utils/
│       ├── mod.rs
│       ├── resp.rs             # RESP protocol helpers
│       ├── encoding.rs         # Vector/key encoding
│       ├── progress.rs         # Progress bar
│       └── logging.rs          # Tracing setup
│
└── tests/
    ├── integration/
    └── unit/
```

---

## 3. Core Traits and Type Hierarchy

### 3.1 Client Abstraction Layer

```rust
// src/client/glide_wrapper.rs

use async_trait::async_trait;
use glide_core::{client::Client, ConnectionRequest, Value};

/// Unified client trait abstracting standalone vs cluster modes
#[async_trait]
pub trait ValkeyClient: Send + Sync {
    /// Execute a command and return the response
    async fn execute(&self, cmd: Vec<String>) -> Result<Value, ClientError>;
    
    /// Execute with explicit routing (cluster mode)
    async fn execute_routed(
        &self, 
        cmd: Vec<String>, 
        route: Option<Route>
    ) -> Result<Value, ClientError>;
    
    /// Execute a batch of commands (pipeline)
    async fn execute_batch(&self, cmds: Vec<Vec<String>>) -> Result<Vec<Value>, ClientError>;
    
    /// Get cluster topology info (if cluster mode)
    async fn get_cluster_info(&self) -> Result<Option<ClusterInfo>, ClientError>;
    
    /// Check if running in cluster mode
    fn is_cluster(&self) -> bool;
}

/// Routing hint for cluster mode
pub enum Route {
    /// Route to a specific node by address
    Node(String, u16),
    /// Route to primary for slot
    SlotPrimary(u16),
    /// Route to any replica for slot
    SlotReplica(u16),
    /// Route to random node
    Random,
    /// Route to all primaries
    AllPrimaries,
    /// Route to all nodes
    AllNodes,
}

/// Wrapper around glide-core GlideClient / GlideClusterClient
pub struct GlideClientWrapper {
    inner: GlideClientInner,
    config: ClientConfig,
}

enum GlideClientInner {
    Standalone(glide_core::GlideClient),
    Cluster(glide_core::GlideClusterClient),
}

impl GlideClientWrapper {
    pub async fn connect(config: &ClientConfig) -> Result<Self, ClientError> {
        if config.cluster_mode {
            let conn_req = build_cluster_connection_request(config);
            let client = glide_core::GlideClusterClient::new(conn_req).await?;
            Ok(Self {
                inner: GlideClientInner::Cluster(client),
                config: config.clone(),
            })
        } else {
            let conn_req = build_standalone_connection_request(config);
            let client = glide_core::GlideClient::new(conn_req).await?;
            Ok(Self {
                inner: GlideClientInner::Standalone(client),
                config: config.clone(),
            })
        }
    }
}
```

### 3.2 Command Template to RESP Encoding

```rust
// src/workload/command_template.rs

impl CommandTemplate {
    /// Convert template to RESP-encoded bytes for a pipeline of N commands
    /// This is called ONCE at initialization - the buffer is then reused
    pub fn to_resp_bytes(&self, pipeline: usize) -> Vec<u8> {
        // Calculate total size
        let single_cmd_size = self.estimate_resp_size();
        let total_size = single_cmd_size * pipeline;
        
        let mut buf = Vec::with_capacity(total_size);
        
        for _ in 0..pipeline {
            self.encode_to_resp(&mut buf);
        }
        
        buf
    }
    
    /// Encode single command to RESP format
    fn encode_to_resp(&self, buf: &mut Vec<u8>) {
        // Write array header: *<num_args>\r\n
        buf.extend_from_slice(b"*");
        itoa::write(&mut *buf, self.args.len()).unwrap();
        buf.extend_from_slice(b"\r\n");
        
        for arg in &self.args {
            match arg {
                TemplateArg::Literal(bytes) => {
                    // $<len>\r\n<data>\r\n
                    buf.extend_from_slice(b"$");
                    itoa::write(&mut *buf, bytes.len()).unwrap();
                    buf.extend_from_slice(b"\r\n");
                    buf.extend_from_slice(bytes);
                    buf.extend_from_slice(b"\r\n");
                }
                TemplateArg::Placeholder(ph_type, reserved_len) => {
                    // Placeholder with fixed-width reservation
                    // $<reserved_len>\r\n<placeholder_marker>\r\n
                    buf.extend_from_slice(b"$");
                    itoa::write(&mut *buf, *reserved_len).unwrap();
                    buf.extend_from_slice(b"\r\n");
                    
                    // Fill with placeholder marker (will be overwritten)
                    let marker_start = buf.len();
                    buf.resize(buf.len() + *reserved_len, b'0');
                    buf.extend_from_slice(b"\r\n");
                    
                    // Record offset for later replacement
                    // (stored in PlaceholderOffsets, computed separately)
                }
            }
        }
    }
    
    /// Compute placeholder offsets after RESP encoding
    pub fn compute_offsets(&self, encoded_buf: &[u8], pipeline: usize) -> PlaceholderOffsets {
        let mut offsets = PlaceholderOffsets {
            commands: Vec::with_capacity(pipeline),
        };
        
        // Parse the encoded buffer to find placeholder positions
        let single_len = encoded_buf.len() / pipeline;
        
        for cmd_idx in 0..pipeline {
            let cmd_start = cmd_idx * single_len;
            let cmd_offsets = self.find_offsets_in_cmd(&encoded_buf[cmd_start..cmd_start + single_len], cmd_start);
            offsets.commands.push(cmd_offsets);
        }
        
        offsets
    }
}

/// Example: Building HSET command for vector insert
/// 
/// Command: HSET {clt}vec:__d_key_ph__ embedding __d_vec_ph____
/// 
/// RESP encoding (with placeholder markers):
/// ```
/// *5\r\n                           // 5 arguments
/// $4\r\n HSET\r\n                  // "HSET"
/// $22\r\n {clt}vec:000000000000\r\n  // Key with placeholders (5+4+12=21 chars + padding)
/// $9\r\n embedding\r\n             // Field name
/// $512\r\n <512 bytes of zeros>\r\n // Vector placeholder (dim=128 * 4 bytes)
/// ```
pub fn build_hset_vector_template(prefix: &str, field: &str, dim: usize) -> CommandTemplate {
    let vec_byte_len = dim * std::mem::size_of::<f32>();
    
    // Key structure: {clt}<prefix><12-digit-id>
    // {clt} is 5 bytes, id is 12 bytes (up to 999,999,999,999)
    let key_len = 5 + prefix.len() + 12;
    
    CommandTemplate {
        args: vec![
            TemplateArg::Literal(b"HSET".to_vec()),
            TemplateArg::Placeholder(PlaceholderType::CompositeKey, key_len),
            TemplateArg::Literal(field.as_bytes().to_vec()),
            TemplateArg::Placeholder(PlaceholderType::DatasetVector, vec_byte_len),
        ],
        ..Default::default()
    }
}

/// Example: Building FT.SEARCH command for vector query
pub fn build_ft_search_template(config: &SearchConfig) -> CommandTemplate {
    let vec_byte_len = config.dim * std::mem::size_of::<f32>();
    
    // Query: *=>[KNN k @field $BLOB EF_RUNTIME ef]
    let query = format!(
        "*=>[KNN {} @{} $BLOB{}]",
        config.k,
        config.vector_field,
        config.ef_search.map(|ef| format!(" EF_RUNTIME {}", ef)).unwrap_or_default()
    );
    
    let mut args = vec![
        TemplateArg::Literal(b"FT.SEARCH".to_vec()),
        TemplateArg::Literal(config.index_name.as_bytes().to_vec()),
        TemplateArg::Literal(query.into_bytes()),
        TemplateArg::Literal(b"PARAMS".to_vec()),
        TemplateArg::Literal(b"2".to_vec()),
        TemplateArg::Literal(b"BLOB".to_vec()),
        TemplateArg::Placeholder(PlaceholderType::DatasetVector, vec_byte_len),
    ];
    
    if config.nocontent {
        args.push(TemplateArg::Literal(b"NOCONTENT".to_vec()));
    }
    
    args.extend([
        TemplateArg::Literal(b"LIMIT".to_vec()),
        TemplateArg::Literal(b"0".to_vec()),
        TemplateArg::Literal(config.k.to_string().into_bytes()),
        TemplateArg::Literal(b"DIALECT".to_vec()),
        TemplateArg::Literal(b"2".to_vec()),
    ]);
    
    CommandTemplate { args, ..Default::default() }
}
```


```rust
// src/workload/command_template.rs

/// Pre-built command template with placeholder positions
#[derive(Debug, Clone)]
pub struct CommandTemplate {
    /// Raw RESP-encoded bytes (or argument list)
    pub args: Vec<TemplateArg>,
    /// Total byte length when serialized
    pub byte_len: usize,
    /// Placeholder positions indexed by type
    pub placeholders: PlaceholderRegistry,
}

#[derive(Debug, Clone)]
pub enum TemplateArg {
    /// Static literal value
    Literal(Vec<u8>),
    /// Placeholder to be replaced at runtime
    Placeholder(PlaceholderType, usize), // (type, reserved_len)
}

impl CommandTemplate {
    /// Build FT.SEARCH template for vector query
    pub fn ft_search(config: &SearchConfig) -> Self {
        let mut args = vec![
            TemplateArg::Literal(b"FT.SEARCH".to_vec()),
            TemplateArg::Literal(config.index_name.as_bytes().to_vec()),
        ];
        
        // Build query string: "*=>[KNN k @field $BLOB]"
        let query = format!(
            "*=>[KNN {} @{} $BLOB{}]",
            config.k,
            config.vector_field,
            config.ef_search.map(|ef| format!(" EF_RUNTIME {}", ef)).unwrap_or_default()
        );
        args.push(TemplateArg::Literal(query.into_bytes()));
        
        // PARAMS 2 BLOB <vector_placeholder>
        args.push(TemplateArg::Literal(b"PARAMS".to_vec()));
        args.push(TemplateArg::Literal(b"2".to_vec()));
        args.push(TemplateArg::Literal(b"BLOB".to_vec()));
        args.push(TemplateArg::Placeholder(
            PlaceholderType::DatasetVector,
            config.dim * 4, // f32 = 4 bytes
        ));
        
        // Optional: NOCONTENT, LIMIT, DIALECT
        if config.nocontent {
            args.push(TemplateArg::Literal(b"NOCONTENT".to_vec()));
        }
        args.push(TemplateArg::Literal(b"LIMIT".to_vec()));
        args.push(TemplateArg::Literal(b"0".to_vec()));
        args.push(TemplateArg::Literal(config.k.to_string().into_bytes()));
        args.push(TemplateArg::Literal(b"DIALECT".to_vec()));
        args.push(TemplateArg::Literal(b"2".to_vec()));
        
        Self::from_args(args)
    }
    
    /// Build HSET template for vector insert
    pub fn hset_vector(config: &SearchConfig, prefix: &str) -> Self {
        let mut args = vec![
            TemplateArg::Literal(b"HSET".to_vec()),
        ];
        
        // Key with cluster tag and dataset key placeholder
        // Format: {clt}<prefix><dataset_key>
        args.push(TemplateArg::Placeholder(PlaceholderType::ClusterTag, 5));
        args.push(TemplateArg::Literal(prefix.as_bytes().to_vec()));
        args.push(TemplateArg::Placeholder(PlaceholderType::DatasetKey, 12));
        
        // Vector field and value
        args.push(TemplateArg::Literal(config.vector_field.as_bytes().to_vec()));
        args.push(TemplateArg::Placeholder(
            PlaceholderType::DatasetVector,
            config.dim * 4,
        ));
        
        Self::from_args(args)
    }
}
```

### 3.3 Placeholder System

```rust
// src/workload/placeholder.rs

use std::sync::atomic::{AtomicU64, Ordering};

/// Types of placeholders that can appear in command templates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaceholderType {
    /// __rand_int__ - Random integer in keyspace
    RandInt,
    /// __rand_1st__ through __rand_9th__ - Named random values
    RandNth(u8),
    /// {clt} - Cluster routing tag
    ClusterTag,
    /// __d_key_ph__ - Dataset vector key/ID
    DatasetKey,
    /// __d_vec_ph____ - Dataset vector blob
    DatasetVector,
    /// ___tag_field____ - Metadata tag field
    TagField,
}

/// Registry of placeholder positions in a template
#[derive(Debug, Clone, Default)]
pub struct PlaceholderRegistry {
    /// Positions indexed by placeholder type
    positions: HashMap<PlaceholderType, Vec<PlaceholderPosition>>,
}

#[derive(Debug, Clone)]
pub struct PlaceholderPosition {
    /// Argument index in command args
    pub arg_index: usize,
    /// Byte offset within the argument (for compound args)
    pub byte_offset: usize,
    /// Reserved length for replacement
    pub reserved_len: usize,
}

/// Runtime placeholder replacement context
pub struct PlaceholderContext<'a> {
    /// Sequential counter for key generation
    pub seq_counter: &'a AtomicU64,
    /// Dataset prefill counter
    pub dataset_counter: &'a AtomicU64,
    /// Dataset context for vector access
    pub dataset: Option<&'a DatasetContext>,
    /// Cluster tag map for routing
    pub cluster_tag_map: Option<&'a ClusterTagMap>,
    /// Keyspace length for random generation
    pub keyspace_len: u64,
    /// Use sequential vs random key generation
    pub sequential: bool,
    /// RNG state (thread-local)
    pub rng: &'a mut fastrand::Rng,
}

impl PlaceholderRegistry {
    /// Replace all placeholders in command arguments
    pub fn replace(
        &self,
        args: &mut Vec<Vec<u8>>,
        ctx: &mut PlaceholderContext<'_>,
    ) -> Result<PlaceholderMetadata, PlaceholderError> {
        let mut metadata = PlaceholderMetadata::default();
        
        // Replace random int placeholders
        for pos in self.positions.get(&PlaceholderType::RandInt).unwrap_or(&vec![]) {
            let value = if ctx.sequential {
                ctx.seq_counter.fetch_add(1, Ordering::Relaxed) % ctx.keyspace_len
            } else {
                ctx.rng.u64(0..ctx.keyspace_len)
            };
            write_fixed_int(&mut args[pos.arg_index], pos.byte_offset, value, pos.reserved_len);
        }
        
        // Replace dataset key placeholders
        for pos in self.positions.get(&PlaceholderType::DatasetKey).unwrap_or(&vec![]) {
            let dataset = ctx.dataset.ok_or(PlaceholderError::NoDataset)?;
            let idx = ctx.dataset_counter.fetch_add(1, Ordering::Relaxed);
            let idx = idx % dataset.num_vectors();
            
            let vector_id = dataset.get_vector_id(idx);
            write_fixed_int(&mut args[pos.arg_index], pos.byte_offset, vector_id, pos.reserved_len);
            
            metadata.inflight_indices.push(idx);
        }
        
        // Replace dataset vector placeholders
        for pos in self.positions.get(&PlaceholderType::DatasetVector).unwrap_or(&vec![]) {
            let dataset = ctx.dataset.ok_or(PlaceholderError::NoDataset)?;
            
            // For queries: use query vectors
            // For inserts: use dataset vectors (coordinated with key placeholder)
            if let Some(&idx) = metadata.inflight_indices.last() {
                // Insert mode: use same index as key
                let vec_data = dataset.get_vector(idx);
                args[pos.arg_index][pos.byte_offset..pos.byte_offset + vec_data.len()]
                    .copy_from_slice(vec_data);
            } else {
                // Query mode: random query vector
                let query_idx = ctx.rng.u64(0..dataset.num_queries());
                let vec_data = dataset.get_query_vector(query_idx);
                args[pos.arg_index][pos.byte_offset..pos.byte_offset + vec_data.len()]
                    .copy_from_slice(vec_data);
                metadata.query_indices.push(query_idx);
            }
        }
        
        // Replace cluster tag placeholders
        if let Some(tag_map) = ctx.cluster_tag_map {
            for pos in self.positions.get(&PlaceholderType::ClusterTag).unwrap_or(&vec![]) {
                if let Some(&vec_id) = metadata.inflight_indices.last() {
                    if let Some(tag) = tag_map.get_tag(vec_id) {
                        args[pos.arg_index][pos.byte_offset..pos.byte_offset + tag.len()]
                            .copy_from_slice(tag.as_bytes());
                    }
                }
            }
        }
        
        Ok(metadata)
    }
}

/// Metadata returned after placeholder replacement (for response tracking)
#[derive(Debug, Default)]
pub struct PlaceholderMetadata {
    /// Dataset indices of in-flight inserts (for retry on error)
    pub inflight_indices: Vec<u64>,
    /// Query indices (for recall verification)
    pub query_indices: Vec<u64>,
}
```

---

## 4. Threading and Concurrency Model

### 4.1 Architecture Overview

**Key Principle: Thread Independence**
- Each thread owns its clients exclusively - NO sharing
- Sync points are minimal and well-defined
- No multiplexing: each client = one dedicated TCP connection

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Benchmark System                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  CONTROL PLANE (glide-core for discovery/setup only):                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  - Cluster topology discovery (CLUSTER NODES)                    │  │
│  │  - Index creation (FT.CREATE)                                    │  │
│  │  - Pre-flight checks (INFO, CONFIG GET)                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  SYNC POINTS (Atomic Counters):                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  requests_issued: AtomicU64     │  dataset_counter: AtomicU64    │  │
│  │  requests_finished: AtomicU64   │  query_counter: AtomicU64      │  │
│  │  seq_key_counter: AtomicU64     │  shutdown_flag: AtomicBool     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  DATA PLANE (raw connections for benchmark traffic):                   │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐  │
│  │     Thread 0       │ │     Thread 1       │ │     Thread N       │  │
│  │  (std::thread)     │ │  (std::thread)     │ │  (std::thread)     │  │
│  │                    │ │                    │ │                    │  │
│  │ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ┌────────────────┐ │  │
│  │ │   Client 0     │ │ │ │   Client 0     │ │ │ │   Client 0     │ │  │
│  │ │ ┌────────────┐ │ │ │ │ ┌────────────┐ │ │ │ │ ┌────────────┐ │ │  │
│  │ │ │ TcpStream  │ │ │ │ │ │ TcpStream  │ │ │ │ │ │ TcpStream  │ │ │  │
│  │ │ │ write_buf  │ │ │ │ │ │ write_buf  │ │ │ │ │ │ write_buf  │ │ │  │
│  │ │ │ read_buf   │ │ │ │ │ │ read_buf   │ │ │ │ │ │ read_buf   │ │ │  │
│  │ │ └────────────┘ │ │ │ │ └────────────┘ │ │ │ │ └────────────┘ │ │  │
│  │ ├────────────────┤ │ │ ├────────────────┤ │ │ ├────────────────┤ │  │
│  │ │   Client 1     │ │ │ │   Client 1     │ │ │ │   Client 1     │ │  │
│  │ │ ┌────────────┐ │ │ │ │ ┌────────────┐ │ │ │ │ ┌────────────┐ │ │  │
│  │ │ │ TcpStream  │ │ │ │ │ │ TcpStream  │ │ │ │ │ │ TcpStream  │ │ │  │
│  │ │ │ write_buf  │ │ │ │ │ │ write_buf  │ │ │ │ │ │ write_buf  │ │ │  │
│  │ │ │ read_buf   │ │ │ │ │ │ read_buf   │ │ │ │ │ │ read_buf   │ │ │  │
│  │ │ └────────────┘ │ │ │ │ └────────────┘ │ │ │ │ └────────────┘ │ │  │
│  │ ├────────────────┤ │ │ ├────────────────┤ │ │ ├────────────────┤ │  │
│  │ │      ...       │ │ │ │      ...       │ │ │ │      ...       │ │  │
│  │ ├────────────────┤ │ │ ├────────────────┤ │ │ ├────────────────┤ │  │
│  │ │   Client M/N   │ │ │ │   Client M/N   │ │ │ │   Client M/N   │ │  │
│  │ └────────────────┘ │ │ └────────────────┘ │ │ └────────────────┘ │  │
│  │                    │ │                    │ │                    │  │
│  │ Local State:       │ │ Local State:       │ │ Local State:       │  │
│  │ - retry_queue      │ │ - retry_queue      │ │ - retry_queue      │  │
│  │ - rng              │ │ - rng              │ │ - rng              │  │
│  │ - node_counters    │ │ - node_counters    │ │ - node_counters    │  │
│  │ - local_histogram  │ │ - local_histogram  │ │ - local_histogram  │  │
│  └────────────────────┘ └────────────────────┘ └────────────────────┘  │
│           │                      │                      │              │
│           └──────────────────────┼──────────────────────┘              │
│                                  v                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Shared Read-Only State                        │  │
│  │  - Dataset (mmap'd, read-only)                                   │  │
│  │  - Command templates (immutable after init)                      │  │
│  │  - Cluster topology snapshot (Arc<ClusterTopology>)              │  │
│  │  - Benchmark config (Arc<BenchmarkConfig>)                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Synchronization Points (Explicit)

| Sync Point | Type | Access Pattern | Purpose |
|------------|------|----------------|---------|
| `requests_issued` | `AtomicU64` | `fetch_add` before send | Global request quota |
| `requests_finished` | `AtomicU64` | `fetch_add` after reply | Progress tracking |
| `dataset_counter` | `AtomicU64` | `fetch_add` for insert | Unique dataset index claim |
| `seq_key_counter` | `AtomicU64` | `fetch_add` if sequential | Sequential key generation |
| `shutdown_flag` | `AtomicBool` | `load` in loop | Coordinated shutdown |
| `recall_accumulator` | `Mutex<RecallAccum>` | `lock` on reply | Recall aggregation |
| `global_histogram` | Per-thread local, merge at end | No runtime sync | Latency stats |

### 4.3 Raw Connection Client (Zero-Allocation Hot Path)

```rust
// src/client/raw_client.rs

use std::io::{Read, Write, BufWriter, BufReader};
use std::net::TcpStream;

/// Low-level benchmark client with pre-allocated buffers
/// One dedicated TCP connection per client - NO multiplexing
pub struct RawBenchmarkClient {
    /// Dedicated TCP connection (or TLS wrapper)
    stream: BufWriter<TcpStream>,
    reader: BufReader<TcpStream>,
    
    /// Pre-allocated write buffer containing RESP-encoded command template
    /// Placeholders are overwritten in-place each iteration
    write_buf: Vec<u8>,
    
    /// Pre-allocated read buffer for responses
    read_buf: Vec<u8>,
    
    /// Byte offsets of placeholders in write_buf (computed once at init)
    placeholder_offsets: PlaceholderOffsets,
    
    /// Assigned cluster node (for node balancing)
    assigned_node: Option<ClusterNodeRef>,
    
    /// Pipeline depth
    pipeline: usize,
    
    /// Pending responses to read
    pending: usize,
    
    /// Inflight dataset indices (for retry on error)
    inflight_indices: VecDeque<u64>,
    
    /// Query indices for recall verification
    query_indices: VecDeque<u64>,
}

impl RawBenchmarkClient {
    pub fn new(
        addr: &str,
        template: &CommandTemplate,
        pipeline: usize,
        tls_config: Option<&TlsConfig>,
    ) -> io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;  // Disable Nagle for latency
        
        // Clone write buffer from template - this is the only allocation
        // After this, we only overwrite in-place
        let write_buf = template.to_resp_bytes(pipeline);
        let placeholder_offsets = template.compute_offsets(&write_buf, pipeline);
        
        // Pre-allocate read buffer (64KB typical)
        let read_buf = vec![0u8; 65536];
        
        Ok(Self {
            stream: BufWriter::new(stream.try_clone()?),
            reader: BufReader::new(stream),
            write_buf,
            read_buf,
            placeholder_offsets,
            assigned_node: None,
            pipeline,
            pending: 0,
            inflight_indices: VecDeque::with_capacity(pipeline),
            query_indices: VecDeque::with_capacity(pipeline),
        })
    }
    
    /// Send AUTH command (called once at connection init)
    pub fn authenticate(&mut self, password: &str, username: Option<&str>) -> io::Result<()> {
        let cmd = match username {
            Some(user) => format!("*3\r\n$4\r\nAUTH\r\n${}\r\n{}\r\n${}\r\n{}\r\n",
                                  user.len(), user, password.len(), password),
            None => format!("*2\r\n$4\r\nAUTH\r\n${}\r\n{}\r\n", 
                           password.len(), password),
        };
        self.stream.write_all(cmd.as_bytes())?;
        self.stream.flush()?;
        self.read_simple_response()?;
        Ok(())
    }
    
    /// Replace placeholders in-place and send request batch
    /// ZERO ALLOCATIONS in this hot path
    #[inline]
    pub fn send_batch(&mut self, ctx: &mut PlaceholderContext<'_>) -> io::Result<()> {
        // Clear tracking queues
        self.inflight_indices.clear();
        self.query_indices.clear();
        
        // In-place placeholder replacement (same pattern as C)
        for i in 0..self.pipeline {
            self.replace_placeholders_at(i, ctx);
        }
        
        // Write pre-filled buffer to socket
        self.stream.write_all(&self.write_buf)?;
        self.pending = self.pipeline;
        
        Ok(())
    }
    
    /// In-place placeholder replacement for command i in pipeline
    #[inline]
    fn replace_placeholders_at(&mut self, cmd_idx: usize, ctx: &mut PlaceholderContext<'_>) {
        let offsets = &self.placeholder_offsets;
        let buf = &mut self.write_buf;
        
        // Replace key placeholder (if present)
        if let Some(key_offset) = offsets.get_key_offset(cmd_idx) {
            let (key_value, dataset_idx) = ctx.next_key();
            write_fixed_width_u64(buf, key_offset, key_value, offsets.key_len);
            
            if let Some(idx) = dataset_idx {
                self.inflight_indices.push_back(idx);
            }
        }
        
        // Replace vector placeholder (if present) - ZERO COPY from mmap
        if let Some(vec_offset) = offsets.get_vector_offset(cmd_idx) {
            if let Some(dataset) = ctx.dataset {
                // Determine if this is insert (has key) or query (no key)
                if let Some(&idx) = self.inflight_indices.back() {
                    // Insert mode: use same dataset index as key
                    let vec_bytes = dataset.get_vector_bytes(idx);
                    buf[vec_offset..vec_offset + vec_bytes.len()].copy_from_slice(vec_bytes);
                } else {
                    // Query mode: random query vector
                    let query_idx = ctx.next_query_idx();
                    let vec_bytes = dataset.get_query_bytes(query_idx);
                    buf[vec_offset..vec_offset + vec_bytes.len()].copy_from_slice(vec_bytes);
                    self.query_indices.push_back(query_idx);
                }
            }
        }
        
        // Replace cluster tag placeholder (if present)
        if let Some(tag_offset) = offsets.get_cluster_tag_offset(cmd_idx) {
            if let Some(tag_map) = ctx.cluster_tag_map {
                if let Some(&idx) = self.inflight_indices.back() {
                    if let Some(tag) = tag_map.get_tag(idx) {
                        buf[tag_offset..tag_offset + 5].copy_from_slice(tag);
                    }
                }
            }
        }
        
        // Replace random int placeholders
        for rand_offset in offsets.get_rand_offsets(cmd_idx) {
            let value = ctx.next_rand_key();
            write_fixed_width_u64(buf, *rand_offset, value, 12);
        }
    }
    
    /// Read and process response batch
    pub fn recv_batch(&mut self) -> io::Result<BatchResponse> {
        let mut responses = Vec::with_capacity(self.pending);
        
        while self.pending > 0 {
            let resp = self.read_resp_value()?;
            responses.push(resp);
            self.pending -= 1;
        }
        
        Ok(BatchResponse {
            values: responses,
            inflight_indices: std::mem::take(&mut self.inflight_indices),
            query_indices: std::mem::take(&mut self.query_indices),
        })
    }
    
    /// Low-level RESP parser (minimal allocation)
    fn read_resp_value(&mut self) -> io::Result<RespValue> {
        // Read type byte
        let mut type_buf = [0u8; 1];
        self.reader.read_exact(&mut type_buf)?;
        
        match type_buf[0] {
            b'+' => self.read_simple_string(),
            b'-' => self.read_error(),
            b':' => self.read_integer(),
            b'$' => self.read_bulk_string(),
            b'*' => self.read_array(),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid RESP type")),
        }
    }
    
    // ... RESP parsing methods ...
}

/// Pre-computed byte offsets for placeholders in command buffer
pub struct PlaceholderOffsets {
    /// Per-command offsets (for pipeline of N commands)
    commands: Vec<CommandOffsets>,
}

struct CommandOffsets {
    key_offset: Option<usize>,
    vector_offset: Option<usize>,
    cluster_tag_offset: Option<usize>,
    rand_offsets: Vec<usize>,
}

/// Write u64 as fixed-width decimal string (zero-padded)
#[inline]
fn write_fixed_width_u64(buf: &mut [u8], offset: usize, value: u64, width: usize) {
    // Fast path: write digits right-to-left with zero padding
    let mut v = value;
    for i in (0..width).rev() {
        buf[offset + i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
}
```

### 4.4 Thread Worker Implementation

```rust
// src/benchmark/worker.rs

use std::thread;
use std::time::{Duration, Instant};

/// Per-thread benchmark worker - owns all its clients exclusively
pub struct BenchmarkWorker {
    /// Thread index
    id: usize,
    
    /// Owned clients (NOT shared with other threads)
    clients: Vec<RawBenchmarkClient>,
    
    /// Thread-local RNG (fast, no sync)
    rng: fastrand::Rng,
    
    /// Thread-local histogram (merged at end)
    local_histogram: hdrhistogram::Histogram<u64>,
    
    /// Thread-local recall accumulator
    local_recall: RecallAccumulator,
    
    /// Local retry queue for failed inserts
    retry_queue: VecDeque<u64>,
    
    /// Per-node request counters (for node balancing)
    node_counters: Vec<u64>,
    node_quota: Vec<i64>,
}

impl BenchmarkWorker {
    /// Main worker loop - runs in dedicated OS thread
    pub fn run(
        mut self,
        config: Arc<BenchmarkConfig>,
        dataset: Option<Arc<DatasetContext>>,
        cluster_tag_map: Option<Arc<ClusterTagMap>>,
        counters: Arc<GlobalCounters>,
    ) -> WorkerResult {
        let pipeline = config.pipeline as usize;
        let total_requests = config.requests;
        
        // Round-robin across owned clients
        let mut client_idx = 0;
        
        loop {
            // === SYNC POINT 1: Check shutdown ===
            if counters.shutdown.load(Ordering::Relaxed) {
                break;
            }
            
            // === SYNC POINT 2: Atomic claim of request quota ===
            let issued = counters.requests_issued.fetch_add(
                pipeline as u64, 
                Ordering::Relaxed
            );
            if issued >= total_requests {
                // Undo the claim, we're done
                counters.requests_issued.fetch_sub(pipeline as u64, Ordering::Relaxed);
                break;
            }
            
            // Rate limiting (if enabled)
            if let Some(ref limiter) = config.rate_limiter {
                limiter.acquire(pipeline);
            }
            
            // Get next client (round-robin within this thread's clients)
            let client = &mut self.clients[client_idx];
            client_idx = (client_idx + 1) % self.clients.len();
            
            // Build placeholder context
            // === SYNC POINT 3: Key/dataset index generation ===
            let mut ctx = PlaceholderContext {
                seq_counter: &counters.seq_key_counter,
                dataset_counter: &counters.dataset_counter,
                query_counter: &counters.query_counter,
                dataset: dataset.as_deref(),
                cluster_tag_map: cluster_tag_map.as_deref(),
                keyspace_len: config.keyspace_len,
                sequential: config.sequential,
                rng: &mut self.rng,
            };
            
            // Send request batch (zero-allocation hot path)
            let start = Instant::now();
            if let Err(e) = client.send_batch(&mut ctx) {
                self.handle_send_error(e, client);
                continue;
            }
            
            // Receive response batch
            match client.recv_batch() {
                Ok(responses) => {
                    // Record latency
                    let latency_us = start.elapsed().as_micros() as u64;
                    self.local_histogram.record(latency_us).ok();
                    
                    // === SYNC POINT 4: Update finished counter ===
                    counters.requests_finished.fetch_add(
                        responses.values.len() as u64,
                        Ordering::Relaxed
                    );
                    
                    // Process responses (recall verification, error handling)
                    self.process_responses(&responses, &config);
                }
                Err(e) => {
                    self.handle_recv_error(e, client);
                }
            }
        }
        
        WorkerResult {
            worker_id: self.id,
            local_histogram: self.local_histogram,
            local_recall: self.local_recall,
            retry_count: self.retry_queue.len(),
        }
    }
    
    fn process_responses(&mut self, batch: &BatchResponse, config: &BenchmarkConfig) {
        for (i, resp) in batch.values.iter().enumerate() {
            match resp {
                RespValue::Error(msg) => {
                    if msg.starts_with("MOVED") || msg.starts_with("ASK") {
                        // Queue for retry with correct node
                        // (In phase 2: trigger topology refresh)
                    } else {
                        // Other error - log and continue
                    }
                }
                RespValue::Array(arr) if !batch.query_indices.is_empty() => {
                    // FT.SEARCH response - verify recall
                    if let Some(query_idx) = batch.query_indices.get(i) {
                        self.verify_recall(arr, *query_idx, config);
                    }
                }
                _ => {
                    // Success response - nothing special to do
                }
            }
        }
    }
}

/// Global atomic counters (the ONLY cross-thread sync)
pub struct GlobalCounters {
    pub requests_issued: AtomicU64,
    pub requests_finished: AtomicU64,
    pub seq_key_counter: AtomicU64,
    pub dataset_counter: AtomicU64,
    pub query_counter: AtomicU64,
    pub shutdown: AtomicBool,
}
```

### 4.5 Placeholder Context (Bridge to Sync Points)

```rust
// src/workload/placeholder.rs

/// Context for placeholder replacement
/// References atomic counters for cross-thread synchronization
pub struct PlaceholderContext<'a> {
    // === Atomic counters (cross-thread sync points) ===
    pub seq_counter: &'a AtomicU64,
    pub dataset_counter: &'a AtomicU64,
    pub query_counter: &'a AtomicU64,
    
    // === Read-only shared state ===
    pub dataset: Option<&'a DatasetContext>,
    pub cluster_tag_map: Option<&'a ClusterTagMap>,
    pub keyspace_len: u64,
    pub sequential: bool,
    
    // === Thread-local state (no sync) ===
    pub rng: &'a mut fastrand::Rng,
}

impl<'a> PlaceholderContext<'a> {
    /// Get next key value - SYNC POINT for sequential mode
    #[inline]
    pub fn next_key(&mut self) -> (u64, Option<u64>) {
        if self.sequential {
            let v = self.seq_counter.fetch_add(1, Ordering::Relaxed);
            (v % self.keyspace_len, None)
        } else {
            (self.rng.u64(0..self.keyspace_len), None)
        }
    }
    
    /// Claim next dataset index - SYNC POINT for unique vector insertion
    #[inline]
    pub fn next_dataset_idx(&mut self) -> u64 {
        self.dataset_counter.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Get next query index - SYNC POINT for ordered query iteration
    #[inline]
    pub fn next_query_idx(&mut self) -> u64 {
        if self.sequential {
            self.query_counter.fetch_add(1, Ordering::Relaxed) 
                % self.dataset.map(|d| d.num_queries()).unwrap_or(1)
        } else {
            self.rng.u64(0..self.dataset.map(|d| d.num_queries()).unwrap_or(1))
        }
    }
    
    /// Get random key (thread-local RNG, no sync)
    #[inline]
    pub fn next_rand_key(&mut self) -> u64 {
        self.rng.u64(0..self.keyspace_len)
    }
}
```

### 4.6 Runtime Choice: std::thread vs Tokio

**Decision: Use `std::thread` (not Tokio) for worker threads**

| Factor | std::thread | Tokio |
|--------|-------------|-------|
| Scheduling | OS-controlled, predictable | Work-stealing, can migrate tasks |
| Overhead | Minimal | Runtime overhead, async machinery |
| Control | Direct thread affinity possible | Abstracted |
| I/O Model | Blocking (fine for benchmark) | Non-blocking (overkill here) |
| Complexity | Simple | Requires async/await throughout |
| C Parity | Matches pthread model | Different model |

**Rationale:**
1. Benchmark workers are CPU-bound (placeholder replacement, RESP parsing)
2. Each client has dedicated connection - no benefit from async I/O multiplexing
3. Predictable scheduling is important for latency measurement
4. Simpler code without async/await propagation
5. Direct match to C implementation's pthread model

**Where we DO use async (via glide/redis-rs):**
- Control plane operations (cluster discovery, index creation)
- These are infrequent and don't affect benchmark accuracy

```rust
// Main orchestrator - uses std::thread for workers
pub fn run_benchmark(config: &BenchmarkConfig) -> BenchmarkResult {
    // Control plane setup (can use async internally)
    let topology = discover_cluster(&config)?;
    
    // Spawn OS threads for workers
    let handles: Vec<_> = (0..config.threads)
        .map(|worker_id| {
            let config = Arc::clone(&config);
            let counters = Arc::clone(&counters);
            let dataset = dataset.clone();
            
            std::thread::Builder::new()
                .name(format!("worker-{}", worker_id))
                .spawn(move || {
                    let worker = BenchmarkWorker::new(worker_id, &config, &dataset);
                    worker.run(counters)
                })
                .expect("Failed to spawn worker thread")
        })
        .collect();
    
    // Wait for completion
    let results: Vec<WorkerResult> = handles
        .into_iter()
        .map(|h| h.join().expect("Worker panicked"))
        .collect();
    
    // Merge results
    merge_results(results)
}
```

---

## 5. valkey-glide Integration (Control Plane Only)

### 5.1 Hybrid Architecture Decision

**Key Insight:** valkey-glide is designed for application workloads (multiplexed, high-level API), not benchmark-grade raw throughput.

**Our Approach:**
- **Control Plane (glide):** Cluster discovery, index creation, INFO collection, setup commands
- **Data Plane (raw):** Direct TCP/TLS connections with pre-allocated buffers for benchmark traffic

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        CONNECTION STRATEGY                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  CONTROL PLANE (valkey-glide / redis-rs high-level)                        │
│  ├── Cluster topology discovery (CLUSTER NODES, CLUSTER SLOTS)             │
│  ├── Index creation (FT.CREATE)                                            │
│  ├── Pre-flight validation (INFO, CONFIG GET)                              │
│  ├── Cluster tag map building (SCAN across nodes)                          │
│  └── Topology refresh on MOVED errors                                      │
│                                                                            │
│  DATA PLANE (raw TCP + manual RESP)                                        │
│  ├── Benchmark request traffic (GET, SET, HSET, FT.SEARCH, etc.)           │
│  ├── Pre-allocated write buffers with in-place placeholder replacement     │
│  ├── One dedicated TCP connection per client (no multiplexing)             │
│  └── Direct socket I/O for minimum latency                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Dependency Configuration

```toml
# Cargo.toml

[package]
name = "valkey-search-benchmark"
version = "0.1.0"
edition = "2021"

[dependencies]
# === Control Plane (glide for cluster discovery) ===
# Note: glide-core is NOT published to crates.io
glide-core = { git = "https://github.com/valkey-io/valkey-glide.git", branch = "main" }
# redis-rs is used by glide internally, we also use it directly for control plane
redis = { git = "https://github.com/valkey-io/valkey-glide.git", branch = "main" }

# === Data Plane (raw connections) ===
# No external dependency - we implement RESP protocol directly
# Using std::net::TcpStream + native-tls/rustls for TLS

# === TLS ===
native-tls = { version = "0.2", optional = true }
rustls = { version = "0.21", optional = true }
webpki-roots = { version = "0.25", optional = true }

# === CLI ===
clap = { version = "4", features = ["derive", "env"] }

# === Metrics ===
hdrhistogram = "7"

# === Dataset ===
memmap2 = "0.9"

# === Concurrency ===
parking_lot = "0.12"
crossbeam-channel = "0.5"

# === Random ===
fastrand = "2"

# === Progress/Logging ===
indicatif = "0.17"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# === Utilities ===
bytes = "1"
thiserror = "1"
anyhow = "1"

[features]
default = ["native-tls-backend"]
native-tls-backend = ["native-tls"]
rustls-backend = ["rustls", "webpki-roots"]

[build-dependencies]
# For setting GLIDE_NAME and GLIDE_VERSION env vars
```

```rust
// build.rs

fn main() {
    // Required for glide-core compilation
    println!("cargo:rustc-env=GLIDE_NAME=valkey-search-benchmark");
    println!("cargo:rustc-env=GLIDE_VERSION={}", env!("CARGO_PKG_VERSION"));
}
```

### 5.3 Control Plane: Cluster Discovery

```rust
// src/cluster/discovery.rs

use redis::{Client, Commands, Connection};

/// Discover cluster topology using redis-rs (or glide control connection)
pub struct ClusterDiscovery {
    /// Control connection (not used for benchmark traffic)
    control_conn: Connection,
}

impl ClusterDiscovery {
    pub fn new(addr: &str, auth: Option<&AuthConfig>, tls: Option<&TlsConfig>) -> Result<Self> {
        let client = Client::open(format!("redis://{}", addr))?;
        let mut conn = client.get_connection()?;
        
        // AUTH if needed
        if let Some(auth) = auth {
            if let Some(ref user) = auth.username {
                redis::cmd("AUTH").arg(user).arg(&auth.password).query(&mut conn)?;
            } else {
                redis::cmd("AUTH").arg(&auth.password).query(&mut conn)?;
            }
        }
        
        Ok(Self { control_conn: conn })
    }
    
    /// Parse CLUSTER NODES response
    pub fn discover_topology(&mut self) -> Result<ClusterTopology> {
        let nodes_str: String = redis::cmd("CLUSTER").arg("NODES")
            .query(&mut self.control_conn)?;
        
        let mut nodes = Vec::new();
        let mut slot_map = [None; 16384];
        
        for line in nodes_str.lines() {
            if let Some(node) = parse_cluster_node_line(line)? {
                // Update slot map
                for &slot in &node.slots {
                    slot_map[slot as usize] = Some(nodes.len());
                }
                nodes.push(node);
            }
        }
        
        Ok(ClusterTopology { nodes, slot_map })
    }
    
    /// Execute FT.CREATE for index setup
    pub fn create_search_index(&mut self, config: &SearchConfig) -> Result<()> {
        let mut cmd = redis::cmd("FT.CREATE");
        cmd.arg(&config.index_name)
            .arg("ON").arg("HASH")
            .arg("PREFIX").arg("1").arg(&config.prefix)
            .arg("SCHEMA")
            .arg(&config.vector_field)
            .arg("VECTOR")
            .arg(config.algorithm.to_str())
            .arg("6")  // num attributes
            .arg("TYPE").arg("FLOAT32")
            .arg("DIM").arg(config.dim)
            .arg("DISTANCE_METRIC").arg(config.distance_metric.to_str());
        
        if let Some(m) = config.hnsw_m {
            cmd.arg("M").arg(m);
        }
        if let Some(ef) = config.ef_construction {
            cmd.arg("EF_CONSTRUCTION").arg(ef);
        }
        
        cmd.query(&mut self.control_conn)?;
        Ok(())
    }
    
    /// Collect INFO from a specific node
    pub fn get_node_info(&mut self, node: &ClusterNode) -> Result<NodeInfo> {
        // Create temporary connection to specific node
        let client = Client::open(format!("redis://{}:{}", node.host, node.port))?;
        let mut conn = client.get_connection()?;
        
        let info: String = redis::cmd("INFO").arg("ALL").query(&mut conn)?;
        parse_info_response(&info)
    }
}

fn parse_cluster_node_line(line: &str) -> Result<Option<ClusterNode>> {
    // Format: <id> <ip:port@cport> <flags> <master> <ping> <pong> <epoch> <state> [slot ...]
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 8 {
        return Ok(None);
    }
    
    let addr_part = parts[1].split('@').next().unwrap_or("");
    let (host, port) = parse_host_port(addr_part)?;
    
    let flags = parts[2];
    let is_primary = flags.contains("master");
    let is_replica = flags.contains("slave");
    
    // Parse slot ranges (parts[8..])
    let mut slots = Vec::new();
    for slot_spec in &parts[8..] {
        if let Some((start, end)) = parse_slot_range(slot_spec) {
            for slot in start..=end {
                slots.push(slot);
            }
        }
    }
    
    Ok(Some(ClusterNode {
        id: parts[0].to_string(),
        host,
        port,
        is_primary,
        is_replica,
        slots,
        primary_id: if is_replica { Some(parts[3].to_string()) } else { None },
    }))
}
```

### 5.4 Data Plane: Raw Connection Factory

```rust
// src/client/connection_factory.rs

use std::net::TcpStream;
use std::io::{self, Write, BufWriter, BufReader};
use std::time::Duration;

pub struct ConnectionFactory {
    tls_config: Option<TlsConnector>,
    auth: Option<AuthConfig>,
    connect_timeout: Duration,
    read_timeout: Option<Duration>,
    write_timeout: Option<Duration>,
}

impl ConnectionFactory {
    /// Create a raw TCP connection to a Valkey node
    /// Used for benchmark traffic (NOT glide)
    pub fn create_connection(&self, host: &str, port: u16) -> io::Result<RawConnection> {
        let addr = format!("{}:{}", host, port);
        
        // Connect with timeout
        let stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?,
            self.connect_timeout
        )?;
        
        // Configure socket
        stream.set_nodelay(true)?;  // Disable Nagle
        if let Some(timeout) = self.read_timeout {
            stream.set_read_timeout(Some(timeout))?;
        }
        if let Some(timeout) = self.write_timeout {
            stream.set_write_timeout(Some(timeout))?;
        }
        
        let mut conn = if let Some(ref tls) = self.tls_config {
            // TLS handshake
            let tls_stream = tls.connect(host, stream)?;
            RawConnection::Tls(BufWriter::new(tls_stream.try_clone()?), 
                               BufReader::new(tls_stream))
        } else {
            RawConnection::Tcp(BufWriter::new(stream.try_clone()?),
                               BufReader::new(stream))
        };
        
        // Authenticate
        if let Some(ref auth) = self.auth {
            conn.authenticate(&auth.password, auth.username.as_deref())?;
        }
        
        Ok(conn)
    }
    
    /// Create N connections distributed across nodes for a worker thread
    pub fn create_worker_clients(
        &self,
        worker_id: usize,
        clients_per_worker: usize,
        topology: &ClusterTopology,
        template: &CommandTemplate,
        pipeline: usize,
        config: &BenchmarkConfig,
    ) -> io::Result<Vec<RawBenchmarkClient>> {
        let mut clients = Vec::with_capacity(clients_per_worker);
        
        let selected_nodes = topology.get_selected_nodes(&config.read_from_replica);
        
        for i in 0..clients_per_worker {
            // Round-robin node assignment
            let node_idx = (worker_id * clients_per_worker + i) % selected_nodes.len();
            let node = &selected_nodes[node_idx];
            
            let conn = self.create_connection(&node.host, node.port)?;
            let client = RawBenchmarkClient::from_connection(
                conn,
                template,
                pipeline,
                Some(node.clone()),
            )?;
            
            clients.push(client);
        }
        
        Ok(clients)
    }
}

pub enum RawConnection {
    Tcp(BufWriter<TcpStream>, BufReader<TcpStream>),
    #[cfg(feature = "native-tls-backend")]
    Tls(BufWriter<native_tls::TlsStream<TcpStream>>, 
        BufReader<native_tls::TlsStream<TcpStream>>),
}

impl RawConnection {
    pub fn authenticate(&mut self, password: &str, username: Option<&str>) -> io::Result<()> {
        // Build AUTH command
        let cmd = match username {
            Some(user) => format!(
                "*3\r\n$4\r\nAUTH\r\n${}\r\n{}\r\n${}\r\n{}\r\n",
                user.len(), user, password.len(), password
            ),
            None => format!(
                "*2\r\n$4\r\nAUTH\r\n${}\r\n{}\r\n",
                password.len(), password
            ),
        };
        
        self.write_all(cmd.as_bytes())?;
        self.flush()?;
        
        // Read response
        let resp = self.read_response()?;
        if let RespValue::Error(e) = resp {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, e));
        }
        
        Ok(())
    }
}
```
### 5.3 Command Execution Mapping

| C Operation | valkey-glide Rust Equivalent |
|-------------|------------------------------|
| `valkeyCommand(ctx, "SET %s %s", key, val)` | `client.execute(vec!["SET", key, val])` |
| `valkeyAppendCommand` + `valkeyGetReply` (pipeline) | `client.execute_batch(commands)` |
| Manual RESP parsing | `Value` enum pattern matching |
| `CLUSTER NODES` parsing | `client.get_cluster_info()` (**see gap below**) |
| `FT.SEARCH` with binary blob | `client.execute(vec!["FT.SEARCH", ..., blob])` |
| `MOVED` handling | **Automatic in glide-core** |

---

## 6. Command Mapping: C to Rust/Glide

### 6.1 Standard Commands

```rust
// Examples of command execution via valkey-glide

// PING
let response = client.execute(vec!["PING".into()]).await?;

// SET key value
let response = client.execute(vec![
    "SET".into(),
    key.into(),
    value.into(),
]).await?;

// HSET key field value
let response = client.execute(vec![
    "HSET".into(),
    key.into(),
    field.into(),
    value.into(),
]).await?;

// Pipeline execution
let commands = vec![
    vec!["SET".into(), "key1".into(), "val1".into()],
    vec!["SET".into(), "key2".into(), "val2".into()],
    vec!["GET".into(), "key1".into()],
];
let responses = client.execute_batch(commands).await?;
```

### 6.2 Vector Search Commands

```rust
// FT.CREATE - Index creation
async fn create_search_index(
    client: &dyn ValkeyClient,
    config: &SearchConfig,
) -> Result<(), ClientError> {
    let mut args = vec![
        "FT.CREATE".to_string(),
        config.index_name.clone(),
        "ON".to_string(),
        "HASH".to_string(),
        "PREFIX".to_string(),
        "1".to_string(),
        config.prefix.clone(),
        "SCHEMA".to_string(),
        config.vector_field.clone(),
        "VECTOR".to_string(),
        config.algorithm.to_string(), // HNSW or FLAT
        "6".to_string(), // number of attributes
        "TYPE".to_string(),
        "FLOAT32".to_string(),
        "DIM".to_string(),
        config.dim.to_string(),
        "DISTANCE_METRIC".to_string(),
        config.distance_metric.to_string(), // L2, IP, COSINE
    ];
    
    // Add HNSW-specific parameters
    if config.algorithm == VectorAlgorithm::HNSW {
        args.extend([
            "M".to_string(),
            config.m.unwrap_or(16).to_string(),
            "EF_CONSTRUCTION".to_string(),
            config.ef_construction.unwrap_or(200).to_string(),
        ]);
    }
    
    client.execute(args).await?;
    Ok(())
}

// FT.SEARCH - Vector query
async fn vector_search(
    client: &dyn ValkeyClient,
    config: &SearchConfig,
    query_vector: &[f32],
) -> Result<SearchResult, ClientError> {
    let query = format!(
        "*=>[KNN {} @{} $BLOB{}]",
        config.k,
        config.vector_field,
        config.ef_search.map(|ef| format!(" EF_RUNTIME {}", ef)).unwrap_or_default()
    );
    
    // Serialize vector to bytes
    let blob: Vec<u8> = query_vector
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    
    let mut args = vec![
        "FT.SEARCH".to_string(),
        config.index_name.clone(),
        query,
        "PARAMS".to_string(),
        "2".to_string(),
        "BLOB".to_string(),
    ];
    
    // Add binary blob as raw bytes
    // Note: glide-core handles binary data via GlideString or similar
    let response = client.execute_with_binary(args, vec![blob]).await?;
    
    parse_search_response(response, config.nocontent)
}
```

### 6.3 Cluster Operations

```rust
// Cluster topology discovery
async fn discover_cluster_topology(
    client: &dyn ValkeyClient,
) -> Result<ClusterTopology, ClientError> {
    // Using INFO and CLUSTER commands
    let cluster_info = client.execute(vec!["CLUSTER".into(), "INFO".into()]).await?;
    let cluster_nodes = client.execute(vec!["CLUSTER".into(), "NODES".into()]).await?;
    
    // Parse CLUSTER NODES output
    // Format: <id> <ip:port@cport> <flags> <master> <ping> <pong> <epoch> <state> <slots>
    parse_cluster_nodes(cluster_nodes)
}

// Per-node INFO collection
async fn collect_node_info(
    client: &dyn ValkeyClient,
    node: &ClusterNode,
) -> Result<NodeInfo, ClientError> {
    // Route to specific node
    let response = client.execute_routed(
        vec!["INFO".into(), "ALL".into()],
        Some(Route::Node(node.host.clone(), node.port)),
    ).await?;
    
    parse_info_response(response)
}
```

---

## 7. Configuration and CLI Argument Mapping

### 7.1 Complete CLI Structure

```rust
// src/config/cli.rs

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(name = "valkey-benchmark")]
#[command(about = "High-performance Valkey/Redis benchmark tool with vector search support")]
pub struct Cli {
    // === Connection Options ===
    
    /// Server hostname
    #[arg(short = 'h', long, default_value = "127.0.0.1")]
    pub host: String,
    
    /// Server port
    #[arg(short = 'p', long, default_value_t = 6379)]
    pub port: u16,
    
    /// Unix socket path (overrides host/port)
    #[arg(short = 's', long)]
    pub socket: Option<String>,
    
    /// Password for AUTH
    #[arg(short = 'a', long, env = "VALKEY_AUTH")]
    pub auth: Option<String>,
    
    /// Username for ACL authentication
    #[arg(long)]
    pub user: Option<String>,
    
    // === TLS Options ===
    
    /// Enable TLS connection
    #[arg(long)]
    pub tls: bool,
    
    /// Skip TLS certificate verification
    #[arg(long)]
    pub tls_skip_verify: bool,
    
    /// CA certificate file
    #[arg(long)]
    pub cacert: Option<String>,
    
    /// Client certificate file
    #[arg(long)]
    pub cert: Option<String>,
    
    /// Client private key file
    #[arg(long)]
    pub key: Option<String>,
    
    /// TLS SNI hostname
    #[arg(long)]
    pub sni: Option<String>,
    
    // === Concurrency Options ===
    
    /// Number of parallel connections (simulated clients)
    #[arg(short = 'c', long, default_value_t = 50)]
    pub clients: u64,
    
    /// Number of worker threads (0 = auto-detect)
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    
    /// Pipeline depth (commands per batch)
    #[arg(short = 'P', long, default_value_t = 1)]
    pub pipeline: u64,
    
    // === Request Options ===
    
    /// Total number of requests
    #[arg(short = 'n', long, default_value_t = 100000)]
    pub requests: u64,
    
    /// Data size in bytes (for SET/GET)
    #[arg(short = 'd', long, default_value_t = 3)]
    pub datasize: usize,
    
    /// Keyspace length (range of key IDs)
    #[arg(short = 'r', long, default_value_t = 1000000)]
    pub keyspacelen: u64,
    
    /// Use sequential keys instead of random
    #[arg(long)]
    pub sequential: bool,
    
    /// Enable placeholder replacement in commands
    #[arg(long)]
    pub enable_placeholders: bool,
    
    // === Rate Limiting ===
    
    /// Requests per second limit (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    pub rps: u64,
    
    // === Cluster Options ===
    
    /// Enable cluster mode
    #[arg(long)]
    pub cluster: bool,
    
    /// Read from replica strategy
    #[arg(long, value_enum, default_value_t = ReadFromReplica::Primary)]
    pub rfr: ReadFromReplica,
    
    /// Enable node balancing (quota-based)
    #[arg(long)]
    pub balance_nodes: bool,
    
    /// Balance quota step size
    #[arg(long, default_value_t = 1000)]
    pub balance_quota_step: u64,
    
    /// Balance tolerance percentage
    #[arg(long, default_value_t = 10)]
    pub balance_tolerance: u64,
    
    // === Database Options ===
    
    /// Database number to SELECT
    #[arg(long)]
    pub dbnum: Option<u32>,
    
    // === Workload Selection ===
    
    /// Test type(s) to run (comma-separated)
    #[arg(short = 't', long, value_delimiter = ',')]
    pub tests: Vec<WorkloadArg>,
    
    /// Custom command to benchmark
    #[arg(long)]
    pub command: Option<String>,
    
    /// Loop continuously
    #[arg(short = 'l', long)]
    pub loop_mode: bool,
    
    // === Vector Search Options ===
    
    /// Enable vector search mode
    #[arg(long)]
    pub search: bool,
    
    /// Search index name
    #[arg(long, default_value = "idx")]
    pub search_name: String,
    
    /// Key prefix for vectors
    #[arg(long, default_value = "vec:")]
    pub search_prefix: String,
    
    /// Vector dimension
    #[arg(long, default_value_t = 128)]
    pub vector_dim: usize,
    
    /// Vector field name in hash
    #[arg(long, default_value = "vec")]
    pub vector_field: String,
    
    /// Distance metric (L2, IP, COSINE)
    #[arg(long, value_enum, default_value_t = DistanceMetric::L2)]
    pub distance_metric: DistanceMetric,
    
    /// Vector algorithm (HNSW, FLAT)
    #[arg(long, value_enum, default_value_t = VectorAlgorithm::HNSW)]
    pub algorithm: VectorAlgorithm,
    
    /// Number of neighbors to return (k)
    #[arg(short = 'k', long, default_value_t = 10)]
    pub k: usize,
    
    /// HNSW M parameter
    #[arg(long)]
    pub hnsw_m: Option<usize>,
    
    /// HNSW EF_CONSTRUCTION parameter
    #[arg(long)]
    pub ef_construction: Option<usize>,
    
    /// Runtime EF_SEARCH parameter
    #[arg(long)]
    pub ef_search: Option<usize>,
    
    /// Use NOCONTENT in FT.SEARCH
    #[arg(long)]
    pub nocontent: bool,
    
    // === Dataset Options ===
    
    /// Path to binary dataset file
    #[arg(long)]
    pub dataset: Option<String>,
    
    /// Enable filtered search mode
    #[arg(long)]
    pub filtered_search: bool,
    
    // === Optimizer Options ===
    
    /// Enable load optimization
    #[arg(long)]
    pub optimize: bool,
    
    /// Optimization objective (e.g., "maximize:qps", "minimize:p99")
    #[arg(long)]
    pub optimize_objective: Option<String>,
    
    /// Target recall constraint (e.g., 0.95)
    #[arg(long)]
    pub target_recall: Option<f64>,
    
    /// Target QPS constraint
    #[arg(long)]
    pub target_qps: Option<f64>,
    
    /// Target p99 latency constraint (ms)
    #[arg(long)]
    pub target_p99: Option<f64>,
    
    // === Output Options ===
    
    /// Output format
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub output_format: OutputFormat,
    
    /// Output file path
    #[arg(short = 'o', long)]
    pub output: Option<String>,
    
    /// CSV output mode
    #[arg(long)]
    pub csv: bool,
    
    /// Enable quiet mode (minimal output)
    #[arg(short = 'q', long)]
    pub quiet: bool,
    
    /// Show per-second stats during benchmark
    #[arg(long)]
    pub show_per_second: bool,
    
    /// Include percentile latency in output
    #[arg(long)]
    pub show_percentiles: bool,
    
    /// Collect INFO stats before/after benchmark
    #[arg(long)]
    pub collect_info: bool,
    
    // === Misc Options ===
    
    /// Histogram precision (1-5)
    #[arg(long, default_value_t = 3)]
    pub precision: u8,
    
    /// Idle timeout in seconds
    #[arg(short = 'I', long)]
    pub idle_timeout: Option<u64>,
    
    /// Seed for random number generator
    #[arg(long)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ReadFromReplica {
    /// Only read from primary nodes
    Primary,
    /// Prefer replicas, fall back to primary
    PreferReplica,
    /// Round-robin across all nodes
    RoundRobin,
    /// AZ-affinity based selection
    AZAffinity,
    /// Alias for PreferReplica
    All,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DistanceMetric {
    L2,
    IP,
    Cosine,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum VectorAlgorithm {
    HNSW,
    Flat,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Text,
    Csv,
    Json,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum WorkloadArg {
    Ping,
    Set,
    Get,
    Incr,
    Lpush,
    Rpush,
    Lpop,
    Rpop,
    Sadd,
    Hset,
    Spop,
    Zadd,
    Zpopmin,
    Lrange,
    Mset,
    VecLoad,
    VecQuery,
    VecDelete,
}
```

### 7.2 Option Group to Component Mapping

| Option Group | Component(s) | Responsibility |
|--------------|--------------|----------------|
| `-h`, `-p`, `-s`, `-a`, `--user`, `--tls*` | `ConnectionFactory` | Client connection setup |
| `-c`, `--threads`, `-P` | `BenchmarkOrchestrator`, `BenchmarkWorker` | Concurrency model |
| `-n`, `-r`, `--sequential` | `PlaceholderContext`, `AtomicCounters` | Request/key generation |
| `--cluster`, `--rfr`, `--balance*` | `ClusterTopology`, `NodeBalancer`, `NodeSelector` | Cluster routing |
| `-t`, `--command` | `WorkloadScheduler`, `CommandTemplate` | Workload definition |
| `--search*`, `--vector*`, `-k`, `--ef*` | `SearchConfig`, `CommandTemplate::ft_search` | Vector search |
| `--dataset`, `--filtered-search` | `DatasetContext`, `ClusterTagMap` | Dataset integration |
| `--optimize*`, `--target*` | `Optimizer`, `Constraints` | Load optimization |
| `--rps` | `RateLimiter` | Rate control |
| `-o`, `--csv`, `--output-format` | `Reporter` | Output formatting |

---

## 8. Dataset Integration (Zero-Copy, 10B+ Scale)

### 8.1 Design Principles

**Critical Requirements:**
1. **NO full dataset load** - File can be larger than RAM
2. **Zero-copy access** - Vector bytes go directly from mmap to socket buffer
3. **10B+ key support** - O(1) access regardless of dataset size
4. **Thread-safe reads** - Multiple threads read simultaneously without locks

**Data Flow (single memcpy):**
```
mmap'd file → &[u8] slice → copy_from_slice() → write_buf → socket
     │                            │
     │                            └── Single memcpy (unavoidable)
     └── Zero-copy pointer arithmetic
```

### 8.2 Memory-Mapped Dataset Implementation

```rust
// src/dataset/binary_dataset.rs

use memmap2::Mmap;
use std::fs::File;
use std::io;

const DATASET_MAGIC: u32 = 0xDECDB001;

/// Binary dataset header (4KB, matches C struct exactly for compatibility)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DatasetHeader {
    pub magic: u32,
    pub version: u32,
    pub dataset_name: [u8; 256],
    pub distance_metric: u8,      // 0=L2, 1=IP, 2=COSINE
    pub dtype: u8,                // 0=FLOAT32, 1=FLOAT16
    pub has_metadata: u8,
    pub padding: [u8; 1],
    pub dim: u32,
    pub num_vectors: u64,         // Can be 10B+
    pub num_queries: u64,
    pub num_neighbors: u32,       // Ground truth k
    pub vocab_size: u32,
    pub vectors_offset: u64,
    pub queries_offset: u64,
    pub ground_truth_offset: u64,
    pub vector_metadata_offset: u64,
    pub query_metadata_offset: u64,
    pub vocab_offset: u64,
    pub reserved: [u8; 3744],     // Pad to 4096 bytes
}

/// Memory-mapped dataset context
/// 
/// # Thread Safety
/// Safe to share across threads via Arc - mmap is read-only
/// 
/// # Memory Usage
/// Only accessed pages are loaded (OS page cache handles this)
/// A 10B vector dataset with dim=128 is ~5TB on disk
/// Only ~1GB of pages will be resident at any time (depending on access pattern)
pub struct DatasetContext {
    /// Memory-mapped file (read-only)
    mmap: Mmap,
    
    /// Cached header values (avoid packed struct access in hot path)
    dim: usize,
    num_vectors: u64,
    num_queries: u64,
    num_neighbors: usize,
    
    /// Pre-computed offsets and sizes
    vectors_offset: usize,
    queries_offset: usize,
    ground_truth_offset: usize,
    vec_byte_len: usize,    // dim * sizeof(f32)
}

impl DatasetContext {
    pub fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        
        // Memory-map the file read-only
        // SAFETY: We never write to the mmap
        let mmap = unsafe { Mmap::map(&file)? };
        
        // Parse header (first 4KB)
        if mmap.len() < std::mem::size_of::<DatasetHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small for header"
            ));
        }
        
        let header: DatasetHeader = unsafe {
            std::ptr::read_unaligned(mmap.as_ptr() as *const DatasetHeader)
        };
        
        if header.magic != DATASET_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid magic: expected 0x{:08X}, got 0x{:08X}", 
                        DATASET_MAGIC, header.magic)
            ));
        }
        
        let dim = header.dim as usize;
        let vec_byte_len = dim * std::mem::size_of::<f32>();
        
        Ok(Self {
            mmap,
            dim,
            num_vectors: header.num_vectors,
            num_queries: header.num_queries,
            num_neighbors: header.num_neighbors as usize,
            vectors_offset: header.vectors_offset as usize,
            queries_offset: header.queries_offset as usize,
            ground_truth_offset: header.ground_truth_offset as usize,
            vec_byte_len,
        })
    }
    
    // === Accessors (inlined for performance) ===
    
    #[inline(always)]
    pub fn dim(&self) -> usize { self.dim }
    
    #[inline(always)]
    pub fn num_vectors(&self) -> u64 { self.num_vectors }
    
    #[inline(always)]
    pub fn num_queries(&self) -> u64 { self.num_queries }
    
    #[inline(always)]
    pub fn num_neighbors(&self) -> usize { self.num_neighbors }
    
    #[inline(always)]
    pub fn vec_byte_len(&self) -> usize { self.vec_byte_len }
    
    // === Zero-Copy Data Access ===
    
    /// Get raw bytes for vector at index
    /// 
    /// # Performance
    /// - O(1) pointer arithmetic
    /// - Returns slice directly into mmap (zero-copy)
    /// - Page fault on first access loads data from disk
    /// 
    /// # Panics
    /// Panics if idx >= num_vectors (debug builds)
    #[inline(always)]
    pub fn get_vector_bytes(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.num_vectors, "Vector index out of bounds");
        
        let offset = self.vectors_offset + (idx as usize * self.vec_byte_len);
        // SAFETY: bounds checked by dataset format + debug_assert
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(offset),
                self.vec_byte_len
            )
        }
    }
    
    /// Get raw bytes for query vector at index
    #[inline(always)]
    pub fn get_query_bytes(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.num_queries, "Query index out of bounds");
        
        let offset = self.queries_offset + (idx as usize * self.vec_byte_len);
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(offset),
                self.vec_byte_len
            )
        }
    }
    
    /// Get vector ID for dataset index (typically idx == id, but may differ)
    #[inline(always)]
    pub fn get_vector_id(&self, idx: u64) -> u64 {
        // For standard datasets, vector_id == array_index
        // Override this for datasets with explicit ID arrays
        idx
    }
    
    /// Get ground truth neighbor IDs for a query
    /// 
    /// # Returns
    /// Slice of neighbor IDs (NOT a copy)
    #[inline]
    pub fn get_neighbor_ids(&self, query_idx: u64) -> &[u64] {
        debug_assert!(query_idx < self.num_queries, "Query index out of bounds");
        
        let offset = self.ground_truth_offset 
            + (query_idx as usize * self.num_neighbors * std::mem::size_of::<u64>());
        
        unsafe {
            std::slice::from_raw_parts(
                self.mmap.as_ptr().add(offset) as *const u64,
                self.num_neighbors
            )
        }
    }
}

// SAFETY: Mmap is read-only, safe to share across threads
unsafe impl Send for DatasetContext {}
unsafe impl Sync for DatasetContext {}
```

### 8.3 Using Dataset in Hot Path (Zero-Copy)

```rust
// In RawBenchmarkClient::replace_placeholders_at()

// Replace vector placeholder - ZERO COPY from mmap
if let Some(vec_offset) = offsets.get_vector_offset(cmd_idx) {
    if let Some(dataset) = ctx.dataset {
        // Insert mode: use same dataset index as key
        if let Some(&idx) = self.inflight_indices.back() {
            // This is THE critical path for 10B+ datasets:
            // 1. get_vector_bytes() returns &[u8] into mmap (pointer arithmetic only)
            // 2. copy_from_slice() is a single memcpy into pre-allocated buffer
            // 3. NO allocation, NO intermediate Vec<f32>
            let vec_bytes = dataset.get_vector_bytes(idx);
            self.write_buf[vec_offset..vec_offset + vec_bytes.len()]
                .copy_from_slice(vec_bytes);
        } else {
            // Query mode: random query vector
            let query_idx = ctx.next_query_idx();
            let vec_bytes = dataset.get_query_bytes(query_idx);
            self.write_buf[vec_offset..vec_offset + vec_bytes.len()]
                .copy_from_slice(vec_bytes);
            self.query_indices.push_back(query_idx);
        }
    }
}
```

### 8.4 Ground Truth Access for Recall Verification

```rust
// src/dataset/ground_truth.rs

impl DatasetContext {
    /// Verify recall of search results against ground truth
    /// 
    /// # Performance
    /// - Ground truth access is also zero-copy (mmap slice)
    /// - HashSet allocation happens once per verification
    /// - For very high QPS, consider pre-allocated scratch buffers
    pub fn compute_recall(&self, query_idx: u64, result_ids: &[u64], k: usize) -> f64 {
        let gt_ids = self.get_neighbor_ids(query_idx);
        let k = k.min(gt_ids.len()).min(result_ids.len());
        
        if k == 0 {
            return 0.0;
        }
        
        // For small k (typical: 10-100), linear search is faster than HashSet
        if k <= 64 {
            let mut matches = 0usize;
            for &result_id in &result_ids[..k] {
                for &gt_id in &gt_ids[..k] {
                    if result_id == gt_id {
                        matches += 1;
                        break;
                    }
                }
            }
            return matches as f64 / k as f64;
        }
        
        // For larger k, use HashSet
        let gt_set: std::collections::HashSet<u64> = gt_ids[..k].iter().copied().collect();
        let matches = result_ids[..k].iter().filter(|id| gt_set.contains(id)).count();
        
        matches as f64 / k as f64
    }
}
```

### 8.5 Cluster Tag Map (for 10B+ vectors)

```rust
// src/dataset/cluster_tag_map.rs

use parking_lot::RwLock;

/// Efficient mapping from vector ID to cluster routing tag
/// 
/// Design for 10B+ vectors:
/// - For dense sequential IDs (0..N): Use pre-allocated Vec
/// - For sparse or very large IDs: Use HashMap with sharding
/// 
/// Memory usage for 10B vectors with 5-byte tags:
/// - Dense Vec: ~50GB (may exceed RAM)
/// - Sparse HashMap: ~16 bytes per entry = 160GB (definitely exceeds RAM)
/// 
/// Solution: Lazy population during SCAN + overflow to disk or skip
pub struct ClusterTagMap {
    /// Dense storage for IDs in [0, dense_capacity)
    /// Using Option<[u8;5]> costs 6 bytes per entry
    dense: Option<Vec<Option<[u8; 5]>>>,
    
    /// Sparse storage for IDs >= dense_capacity
    sparse: RwLock<std::collections::HashMap<u64, [u8; 5]>>,
    
    /// Threshold for dense vs sparse storage
    dense_capacity: u64,
    
    /// Count of populated entries
    count: std::sync::atomic::AtomicU64,
    
    /// Key prefix
    prefix: String,
}

impl ClusterTagMap {
    /// Create a new tag map
    /// 
    /// # Arguments
    /// * `dense_capacity` - Use dense Vec for IDs < this value
    ///   Set to 0 for fully sparse (large datasets)
    ///   Set to dataset size for fully dense (smaller datasets)
    pub fn new(dense_capacity: u64, prefix: &str) -> Self {
        let dense = if dense_capacity > 0 && dense_capacity <= 1_000_000_000 {
            // Only allocate dense array if reasonable size (< 6GB)
            Some(vec![None; dense_capacity as usize])
        } else {
            None
        };
        
        Self {
            dense,
            sparse: RwLock::new(std::collections::HashMap::new()),
            dense_capacity,
            count: std::sync::atomic::AtomicU64::new(0),
            prefix: prefix.to_string(),
        }
    }
    
    /// Get cluster tag for vector ID (lock-free for dense region)
    #[inline]
    pub fn get_tag(&self, vector_id: u64) -> Option<&[u8; 5]> {
        if vector_id < self.dense_capacity {
            if let Some(ref dense) = self.dense {
                return dense.get(vector_id as usize)
                    .and_then(|opt| opt.as_ref());
            }
        }
        
        // Sparse lookup (requires read lock)
        self.sparse.read().get(&vector_id).copied().as_ref()
        // Note: This copies the 5-byte tag on return due to lock scope
        // For ultimate performance, consider an alternative design
    }
    
    /// Set cluster tag for vector ID
    pub fn set_tag(&self, vector_id: u64, tag: [u8; 5]) {
        if vector_id < self.dense_capacity {
            if let Some(ref mut dense) = unsafe { 
                // SAFETY: Concurrent writes to different indices are safe
                // Same index writes are idempotent (same tag value expected)
                &mut *(self as *const Self as *mut Self) 
            }.dense {
                dense[vector_id as usize] = Some(tag);
                self.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        }
        
        self.sparse.write().insert(vector_id, tag);
        self.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Check if vector ID has a mapping
    #[inline]
    pub fn contains(&self, vector_id: u64) -> bool {
        if vector_id < self.dense_capacity {
            if let Some(ref dense) = self.dense {
                return dense.get(vector_id as usize)
                    .map(|opt| opt.is_some())
                    .unwrap_or(false);
            }
        }
        
        self.sparse.read().contains_key(&vector_id)
    }
    
    pub fn count(&self) -> u64 {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// Safe to share across threads (RwLock protects sparse map)
unsafe impl Send for ClusterTagMap {}
unsafe impl Sync for ClusterTagMap {}
```

### 8.3 SCAN-Based Key Mapping Builder

```rust
// src/dataset/scanner.rs

/// Build cluster tag map by scanning existing keys
pub async fn build_tag_map_from_scan(
    client: &dyn ValkeyClient,
    prefix: &str,
    cluster_nodes: &[ClusterNode],
    progress: Option<&indicatif::ProgressBar>,
) -> Result<ClusterTagMap, ScanError> {
    let mut tag_map = ClusterTagMap::new(1_000_000, prefix);
    
    // In cluster mode, scan each node in parallel
    if client.is_cluster() {
        let handles: Vec<_> = cluster_nodes.iter().map(|node| {
            let client = client.clone();
            let prefix = prefix.to_string();
            let node_addr = (node.host.clone(), node.port);
            
            tokio::spawn(async move {
                scan_node(&client, &prefix, node_addr).await
            })
        }).collect();
        
        for handle in handles {
            let mappings = handle.await??;
            for (vec_id, tag) in mappings {
                tag_map.set_tag(vec_id, tag);
            }
            if let Some(pb) = progress {
                pb.inc(1);
            }
        }
    } else {
        // Standalone mode: single scan
        let mappings = scan_standalone(client, prefix).await?;
        for (vec_id, tag) in mappings {
            tag_map.set_tag(vec_id, tag);
        }
    }
    
    Ok(tag_map)
}

async fn scan_node(
    client: &dyn ValkeyClient,
    prefix: &str,
    node: (String, u16),
) -> Result<Vec<(u64, [u8; 5])>, ScanError> {
    let mut mappings = Vec::new();
    let mut cursor = "0".to_string();
    let pattern = format!("{}*", prefix);
    
    loop {
        let response = client.execute_routed(
            vec![
                "SCAN".into(),
                cursor.clone(),
                "MATCH".into(),
                pattern.clone(),
                "COUNT".into(),
                "1000".into(),
            ],
            Some(Route::Node(node.0.clone(), node.1)),
        ).await?;
        
        // Parse SCAN response: [cursor, [keys...]]
        let (new_cursor, keys) = parse_scan_response(response)?;
        
        for key in keys {
            if let Some((vec_id, tag)) = parse_vector_key(&key, prefix) {
                mappings.push((vec_id, tag));
            }
        }
        
        cursor = new_cursor;
        if cursor == "0" {
            break;
        }
    }
    
    Ok(mappings)
}
```

---

## 9. Metrics and Statistics

### 9.1 Latency Histogram

```rust
// src/metrics/latency.rs

use hdrhistogram::Histogram;
use parking_lot::Mutex;
use std::time::Duration;

/// Thread-safe HDR histogram for latency tracking
pub struct LatencyHistogram {
    histogram: Mutex<Histogram<u64>>,
    current_second: Mutex<Histogram<u64>>,
}

impl LatencyHistogram {
    pub fn new(precision: u8) -> Self {
        Self {
            histogram: Mutex::new(
                Histogram::new_with_bounds(10, 3_000_000, precision).unwrap()
            ),
            current_second: Mutex::new(
                Histogram::new_with_bounds(10, 3_000_000, precision).unwrap()
            ),
        }
    }
    
    /// Record a latency sample (in microseconds)
    pub fn record(&self, latency: Duration) {
        let micros = latency.as_micros() as u64;
        self.histogram.lock().record(micros).ok();
        self.current_second.lock().record(micros).ok();
    }
    
    /// Get percentile value
    pub fn percentile(&self, p: f64) -> Duration {
        let micros = self.histogram.lock().value_at_percentile(p);
        Duration::from_micros(micros)
    }
    
    /// Reset per-second histogram and return stats
    pub fn rotate_second(&self) -> SecondStats {
        let mut current = self.current_second.lock();
        let stats = SecondStats {
            count: current.len(),
            mean: Duration::from_micros(current.mean() as u64),
            p50: Duration::from_micros(current.value_at_percentile(50.0)),
            p99: Duration::from_micros(current.value_at_percentile(99.0)),
            max: Duration::from_micros(current.max()),
        };
        current.reset();
        stats
    }
}

pub struct SecondStats {
    pub count: u64,
    pub mean: Duration,
    pub p50: Duration,
    pub p99: Duration,
    pub max: Duration,
}
```

### 9.2 Recall Statistics

```rust
// src/metrics/recall.rs

use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::Mutex;

/// Thread-safe recall statistics accumulator
pub struct RecallStats {
    total_queries: AtomicU64,
    sum_recall: Mutex<f64>,
    min_recall: Mutex<f64>,
    max_recall: Mutex<f64>,
    perfect_count: AtomicU64,  // recall >= 0.9999
    zero_count: AtomicU64,     // recall < 0.0001
}

impl RecallStats {
    pub fn new() -> Self {
        Self {
            total_queries: AtomicU64::new(0),
            sum_recall: Mutex::new(0.0),
            min_recall: Mutex::new(1.0),
            max_recall: Mutex::new(0.0),
            perfect_count: AtomicU64::new(0),
            zero_count: AtomicU64::new(0),
        }
    }
    
    /// Record a recall measurement
    pub fn record(&self, recall: f64) {
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        
        {
            let mut sum = self.sum_recall.lock();
            *sum += recall;
        }
        
        {
            let mut min = self.min_recall.lock();
            if recall < *min {
                *min = recall;
            }
        }
        
        {
            let mut max = self.max_recall.lock();
            if recall > *max {
                *max = recall;
            }
        }
        
        if recall >= 0.9999 {
            self.perfect_count.fetch_add(1, Ordering::Relaxed);
        }
        if recall < 0.0001 {
            self.zero_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Get average recall
    pub fn average(&self) -> f64 {
        let total = self.total_queries.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        *self.sum_recall.lock() / total as f64
    }
}
```

### 9.3 Recall Verification

```rust
// src/benchmark/worker.rs (continued)

impl BenchmarkWorker {
    /// Verify recall against ground truth
    async fn verify_recall(&self, response: &Value, query_idx: u64) {
        let dataset = match &self.config.dataset {
            Some(d) => d,
            None => return,
        };
        
        // Parse FT.SEARCH response
        let result_ids = match parse_ft_search_response(response, self.config.search.nocontent) {
            Ok(ids) => ids,
            Err(e) => {
                tracing::warn!("Failed to parse search response: {}", e);
                return;
            }
        };
        
        // Get ground truth
        let ground_truth = dataset.get_neighbors(query_idx);
        let k = self.config.search.k.min(ground_truth.count);
        
        // Compute intersection
        let mut result_set: HashSet<u64> = result_ids.iter().take(k).copied().collect();
        let gt_set: HashSet<u64> = ground_truth.ids.iter().take(k).copied().collect();
        
        let matches = result_set.intersection(&gt_set).count();
        let recall = matches as f64 / k as f64;
        
        self.metrics.recall.record(recall);
    }
}

fn parse_ft_search_response(
    response: &Value,
    nocontent: bool,
) -> Result<Vec<u64>, ParseError> {
    // FT.SEARCH returns: [total, doc1_id, [fields], doc2_id, [fields], ...]
    // With NOCONTENT: [total, doc1_id, doc2_id, ...]
    
    match response {
        Value::Array(arr) if arr.len() >= 1 => {
            let stride = if nocontent { 1 } else { 2 };
            let mut ids = Vec::new();
            
            for i in (1..arr.len()).step_by(stride) {
                if let Value::BulkString(key) = &arr[i] {
                    // Parse vector ID from key: "{tag}prefix123" -> 123
                    if let Some(id) = extract_vector_id(key) {
                        ids.push(id);
                    }
                }
            }
            
            Ok(ids)
        }
        _ => Err(ParseError::InvalidFormat),
    }
}
```

---

## 10. Optimizer Integration

### 10.1 Optimizer State Machine

```rust
// src/optimizer/optimizer.rs

/// Optimization phases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerPhase {
    Init,
    Feasibility,
    Recall,
    Throughput,
    HillClimb,
    Refinement,
    Converged,
}

/// Tunable parameter definition
#[derive(Debug, Clone)]
pub struct TunableParam {
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub group: ParamGroup,
}

#[derive(Debug, Clone, Copy)]
pub enum ParamGroup {
    Recall,      // ef_search
    Throughput,  // clients, threads, pipeline
}

/// Optimizer state
pub struct Optimizer {
    phase: OptimizerPhase,
    params: Vec<TunableParam>,
    objective: OptimizeObjective,
    constraints: Vec<Constraint>,
    history: Vec<Measurement>,
    best_feasible: Option<Measurement>,
    baseline_objective: f64,
    iterations_since_change: usize,
    stabilization_window: usize,
}

impl Optimizer {
    pub fn new(config: &OptimizerConfig) -> Self {
        let params = vec![
            TunableParam {
                name: "ef_search".into(),
                value: config.initial_ef_search as f64,
                min: 10.0,
                max: 1000.0,
                step: 10.0,
                group: ParamGroup::Recall,
            },
            TunableParam {
                name: "clients".into(),
                value: config.initial_clients as f64,
                min: 1.0,
                max: 1000.0,
                step: 10.0,
                group: ParamGroup::Throughput,
            },
            TunableParam {
                name: "threads".into(),
                value: config.initial_threads as f64,
                min: 1.0,
                max: 32.0,
                step: 1.0,
                group: ParamGroup::Throughput,
            },
            TunableParam {
                name: "pipeline".into(),
                value: config.initial_pipeline as f64,
                min: 1.0,
                max: 100.0,
                step: 5.0,
                group: ParamGroup::Throughput,
            },
        ];
        
        Self {
            phase: OptimizerPhase::Init,
            params,
            objective: config.objective.clone(),
            constraints: config.constraints.clone(),
            history: Vec::new(),
            best_feasible: None,
            baseline_objective: 0.0,
            iterations_since_change: 0,
            stabilization_window: 5,
        }
    }
    
    /// Process a measurement and return next parameter values
    pub fn step(&mut self, metrics: &BenchmarkMetrics) -> OptimizerResult {
        let measurement = Measurement {
            params: self.params.iter().map(|p| p.value).collect(),
            metrics: metrics.clone(),
            constraints_satisfied: self.evaluate_constraints(metrics),
            objective_score: self.compute_objective(metrics),
        };
        
        self.history.push(measurement.clone());
        
        // Update best feasible
        if measurement.constraints_satisfied {
            if self.best_feasible.is_none() 
                || measurement.objective_score > self.best_feasible.as_ref().unwrap().objective_score 
            {
                self.best_feasible = Some(measurement.clone());
            }
        }
        
        // Phase-specific logic
        match self.phase {
            OptimizerPhase::Init => {
                self.baseline_objective = measurement.objective_score;
                self.phase = OptimizerPhase::Feasibility;
            }
            OptimizerPhase::Feasibility => {
                if measurement.constraints_satisfied {
                    self.phase = OptimizerPhase::Recall;
                } else {
                    self.double_resources();
                }
            }
            OptimizerPhase::Recall => {
                // Binary search on ef_search
                if self.binary_search_step(&measurement) {
                    self.phase = OptimizerPhase::Throughput;
                }
            }
            OptimizerPhase::Throughput => {
                // Grid search on throughput params
                if self.grid_search_step(&measurement) {
                    self.phase = OptimizerPhase::HillClimb;
                }
            }
            OptimizerPhase::HillClimb => {
                self.gradient_step();
                if self.check_convergence() {
                    self.phase = OptimizerPhase::Refinement;
                }
            }
            OptimizerPhase::Refinement => {
                self.coordinate_descent_step();
                if self.check_convergence() {
                    self.phase = OptimizerPhase::Converged;
                }
            }
            OptimizerPhase::Converged => {}
        }
        
        OptimizerResult {
            phase: self.phase,
            params: self.params.clone(),
            should_continue: self.phase != OptimizerPhase::Converged,
        }
    }
}
```

---

## 11. Implementation Phases and Node Balancing Strategy

### 11.1 Phased Approach for Node Balancing

**Phase 1: Simple (Use Glide Routing)**
- Accept automatic slot-based routing from cluster discovery
- Round-robin client-to-node assignment at connection time
- No runtime balancing
- Suitable for: Initial implementation, homogeneous clusters

**Phase 2: Manual Control (Full Parity with C)**
- Implement our own slot-to-node mapping
- Quota-based throttling per node
- Per-thread node counters
- Rebalancing logic when nodes fall behind
- Suitable for: Production benchmarks, heterogeneous clusters

```rust
// Phase 2: Per-thread node balancing (same as C implementation)

pub struct NodeBalancer {
    /// Per-node quota remaining (starts at quota_step)
    quota_remaining: Vec<AtomicI64>,
    /// Per-node request counters (reset each cycle)
    request_counters: Vec<AtomicU64>,
    /// Quota replenishment step
    quota_step: i64,
    /// Tolerance percentage (e.g., 10%)
    tolerance_pct: i64,
}

impl NodeBalancer {
    /// Check if request can proceed to this node, or should pause
    /// 
    /// Returns: delay_ns (0 = proceed, >0 = wait this long)
    pub fn check_quota(&self, node_idx: usize, tokens: i64) -> u64 {
        let remaining = self.quota_remaining[node_idx].load(Ordering::Relaxed);
        
        if remaining >= tokens {
            // Quota available, proceed
            self.quota_remaining[node_idx].fetch_sub(tokens, Ordering::Relaxed);
            self.request_counters[node_idx].fetch_add(tokens as u64, Ordering::Relaxed);
            return 0;
        }
        
        // Quota exhausted - find slowest node
        let min_completed = self.request_counters.iter()
            .map(|c| c.load(Ordering::Relaxed))
            .min()
            .unwrap_or(0);
        
        if min_completed == 0 {
            // Slowest node hasn't made progress, must wait
            return 1_000_000; // 1ms delay
        }
        
        // Replenish quota for all nodes based on slowest
        let quota_to_add = (min_completed as i64 * (100 + self.tolerance_pct)) / 100;
        for i in 0..self.quota_remaining.len() {
            self.quota_remaining[i].fetch_add(quota_to_add, Ordering::Relaxed);
            self.request_counters[i].store(0, Ordering::Relaxed);
        }
        
        0 // Can proceed now
    }
}
```

### 11.2 MOVED/ASK Handling Strategy

**Phase 1:**
- Log MOVED/ASK errors
- Retry with error (benchmark continues, may have lower accuracy)
- Flag for post-run topology check

**Phase 2:**
- Parse MOVED target: `MOVED <slot> <host>:<port>`
- Trigger control plane topology refresh
- Re-route affected clients to new node
- Track slot migration events

---

## 12. Identified Gaps and Decisions

### 12.1 Resolved by This Design

| Previous Gap | Resolution |
|--------------|------------|
| Multiplexed connections | ✅ Raw TCP, one connection per client |
| Thread-shared clients | ✅ Each thread owns its clients exclusively |
| Buffer allocation per request | ✅ Pre-allocated buffers, in-place replacement |
| Dataset memory loading | ✅ mmap with zero-copy slicing |
| Node balancing abstraction | ✅ Phased approach (simple → full control) |

### 12.2 Design Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Runtime | `std::thread` (not Tokio) | Matches C model, predictable scheduling |
| Client connections | Raw TCP + manual RESP | Maximum control, zero-allocation hot path |
| Glide usage | Control plane only | Discovery, index creation, INFO |
| Buffer strategy | Per-client `Vec<u8>` with in-place mutation | Same efficiency as C |
| Dataset access | `memmap2` with `&[u8]` slices | Zero-copy, 10B+ scale |
| Thread sync | Atomic counters only | Minimal contention |
| Node balancing | Phased (simple first) | Incremental complexity |

### 12.3 Remaining Technical Considerations

| Item | Status | Notes |
|------|--------|-------|
| TLS performance | Verify | native-tls vs rustls benchmark |
| RESP3 support | Future | Currently RESP2 only |
| Pub/Sub for abort | Consider | Alternative: periodic atomic check |
| Mixed workloads | Phase 2 | Per-client workload template selection |
| JSON path commands | Future | JSON.SET with placeholder |

---

## 13. Implementation Phases (Detailed)

### Phase 1: Core Infrastructure (Week 1)
**Goal:** Buildable skeleton with CLI parsing

- [ ] Project setup with Cargo.toml, build.rs (GLIDE env vars)
- [ ] CLI argument parsing with clap (all args from C version)
- [ ] `BenchmarkConfig` struct with validation
- [ ] `SearchConfig` for vector search parameters
- [ ] Logging setup with tracing
- [ ] Error types with thiserror

**Deliverable:** `cargo build` succeeds, `--help` works

### Phase 2: Raw Connection Layer (Week 2)
**Goal:** Direct TCP connections with RESP protocol

- [ ] `RawConnection` (TCP and TLS variants)
- [ ] RESP encoder (command → bytes)
- [ ] RESP parser (bytes → RespValue)
- [ ] AUTH command support
- [ ] SELECT command support
- [ ] Connection factory (standalone mode)

**Deliverable:** Can PING a Valkey server

### Phase 3: Command Templates (Week 3)
**Goal:** Zero-allocation command preparation

- [ ] `CommandTemplate` struct
- [ ] `TemplateArg` enum (Literal vs Placeholder)
- [ ] RESP encoding with placeholder markers
- [ ] `PlaceholderOffsets` computation
- [ ] In-place replacement functions
- [ ] Standard command templates (SET, GET, PING, HSET)

**Deliverable:** Can send SET/GET with random keys

### Phase 4: Threading Model (Week 4)
**Goal:** Multi-threaded benchmark execution

- [ ] `GlobalCounters` (atomic counters)
- [ ] `BenchmarkWorker` struct
- [ ] `std::thread` spawning
- [ ] Round-robin client-to-node assignment
- [ ] Worker main loop (send → recv → record)
- [ ] Shutdown coordination

**Deliverable:** Multi-threaded SET/GET benchmark runs

### Phase 5: Cluster Support - Phase 1 (Week 5)
**Goal:** Basic cluster mode (simple routing)

- [ ] CLUSTER NODES parsing
- [ ] `ClusterTopology` struct
- [ ] Node selection based on `--rfr`
- [ ] Round-robin node assignment per client
- [ ] MOVED/ASK error logging (no retry yet)

**Deliverable:** Benchmark runs against cluster

### Phase 6: Dataset Integration (Week 6)
**Goal:** Memory-mapped dataset with zero-copy

- [ ] `DatasetHeader` parsing
- [ ] `DatasetContext` with mmap
- [ ] Zero-copy vector access
- [ ] Ground truth access
- [ ] Dataset counter sync point
- [ ] Vector placeholder replacement

**Deliverable:** Can load vectors from binary dataset

### Phase 7: Vector Search Commands (Week 7)
**Goal:** FT.SEARCH and FT.CREATE support

- [ ] `SearchConfig` integration
- [ ] FT.CREATE via control connection
- [ ] FT.SEARCH template builder
- [ ] Binary vector blob in RESP
- [ ] Search response parsing
- [ ] VecLoad (HSET) and VecQuery templates

**Deliverable:** Vector load and query benchmarks

### Phase 8: Recall Verification (Week 8)
**Goal:** Ground truth comparison

- [ ] Search response ID extraction
- [ ] Ground truth lookup
- [ ] Recall computation
- [ ] `RecallAccumulator` (per-thread + merge)
- [ ] Query index tracking
- [ ] Recall statistics in output

**Deliverable:** Accurate recall measurement

### Phase 9: Metrics and Reporting (Week 9)
**Goal:** Full statistics output

- [ ] HDR histogram integration
- [ ] Per-second stats collection
- [ ] Thread-local histograms + merge
- [ ] Text/CSV/JSON reporters
- [ ] INFO collection (control plane)
- [ ] Per-node statistics

**Deliverable:** Production-quality output

### Phase 10: Cluster Tag Map (Week 10)
**Goal:** Existing key discovery

- [ ] SCAN across nodes (control plane)
- [ ] Key parsing (extract tag, vector ID)
- [ ] `ClusterTagMap` population
- [ ] Sparse vs dense storage decision
- [ ] Progress reporting
- [ ] Integration with placeholder replacement

**Deliverable:** Resume benchmark on existing data

### Phase 11: Rate Limiting & Node Balancing (Week 11)
**Goal:** Traffic shaping

- [ ] Token bucket rate limiter
- [ ] Per-thread rate limiter integration
- [ ] Node balancer (quota-based)
- [ ] Per-node counters
- [ ] Throttle delay mechanism

**Deliverable:** `--rps` and `--balance-nodes` work

### Phase 12: Optimizer Integration (Week 12)
**Goal:** Adaptive parameter tuning

- [ ] `Optimizer` state machine
- [ ] Phase transitions
- [ ] Parameter adjustment
- [ ] Constraint evaluation
- [ ] Config persistence
- [ ] Integration with benchmark loop

**Deliverable:** `--optimize` finds optimal params

### Phase 13: Polish and Testing (Week 13+)
- [ ] Unit tests for all components
- [ ] Integration tests (requires Valkey cluster)
- [ ] Performance validation vs C version
- [ ] Documentation
- [ ] CI/CD setup
- [ ] Edge case handling

---

## 14. Summary

This HLD provides a complete blueprint for the Rust rewrite with:

1. **Thread Independence:** Each worker owns its clients, sync via atomic counters only
2. **Zero-Allocation Hot Path:** Pre-allocated buffers with in-place placeholder replacement
3. **Zero-Copy Dataset:** mmap with direct `&[u8]` slices for 10B+ scale
4. **Hybrid Architecture:** glide for control plane, raw TCP for data plane
5. **Phased Node Balancing:** Simple first, full control later
6. **Full CLI Parity:** All C version options mapped to clap derive

The design enables incremental implementation while maintaining the performance characteristics of the C version.
