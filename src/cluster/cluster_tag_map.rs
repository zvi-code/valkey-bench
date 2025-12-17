//! Cluster Tag Mapping
//!
//! Maps vector IDs to their cluster tags (hash tags) for proper routing
//! and recall validation in cluster mode.
//!
//! Key format: `prefix + cluster_tag + ':' + vector_id`
//! Example: `vec:{ABC}:000001` where `vec:` is prefix, `{ABC}` is cluster tag
//!
//! The cluster tag determines which hash slot (and thus which node) a key belongs to.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::client::RawConnection;
use crate::cluster::ClusterNode;
use crate::utils::{RespEncoder, RespValue};

/// Vector ID to cluster tag mapping entry
#[derive(Debug, Clone, Default)]
pub struct VectorClusterMapping {
    /// Cluster tag (e.g., "{ABC}") - 5 chars + null
    pub cluster_tag: [u8; 6],
}

impl VectorClusterMapping {
    /// Check if this mapping has a valid tag
    pub fn has_tag(&self) -> bool {
        self.cluster_tag[0] == b'{'
    }

    /// Get cluster tag as string slice
    pub fn tag_str(&self) -> Option<&str> {
        if self.has_tag() {
            // Find the actual length (up to null terminator or end)
            let len = self.cluster_tag.iter().position(|&b| b == 0).unwrap_or(5);
            std::str::from_utf8(&self.cluster_tag[..len]).ok()
        } else {
            None
        }
    }

    /// Set cluster tag from string
    pub fn set_tag(&mut self, tag: &str) {
        let bytes = tag.as_bytes();
        let len = bytes.len().min(5);
        self.cluster_tag[..len].copy_from_slice(&bytes[..len]);
        self.cluster_tag[len] = 0;
    }
}

/// Thread-safe cluster tag mapping table
pub struct ClusterTagMap {
    /// Key prefix (e.g., "vec:")
    pub prefix: String,
    /// Mappings from vector_id to cluster_tag
    mappings: Vec<VectorClusterMapping>,
    /// Number of valid mappings
    count: AtomicU64,
    /// Total keys scanned (for progress tracking)
    keys_scanned: AtomicU64,
    /// Whether cluster mode is enabled
    pub is_cluster_mode: bool,
    /// Mutex for concurrent updates
    update_mutex: Mutex<()>,
    /// Atomic counter for claiming unmapped vector IDs (for partial prefill)
    unmapped_counter: AtomicU64,
}

impl ClusterTagMap {
    /// Create a new cluster tag map with given capacity
    pub fn new(prefix: &str, capacity: u64, is_cluster_mode: bool) -> Self {
        Self {
            prefix: prefix.to_string(),
            mappings: vec![VectorClusterMapping::default(); capacity as usize],
            count: AtomicU64::new(0),
            keys_scanned: AtomicU64::new(0),
            is_cluster_mode,
            update_mutex: Mutex::new(()),
            unmapped_counter: AtomicU64::new(0),
        }
    }

    /// Add a vector ID to cluster tag mapping
    pub fn add_mapping(&self, vector_id: u64, cluster_tag: &str) {
        if vector_id >= self.mappings.len() as u64 {
            return;
        }

        self.keys_scanned.fetch_add(1, Ordering::Relaxed);

        // Safety: We're using interior mutability with proper synchronization
        let _lock = self.update_mutex.lock().unwrap();

        // SAFETY: We hold the mutex, so we can safely mutate
        let mapping = unsafe {
            let ptr = self.mappings.as_ptr() as *mut VectorClusterMapping;
            &mut *ptr.add(vector_id as usize)
        };

        if !mapping.has_tag() {
            // New entry
            if self.is_cluster_mode {
                mapping.set_tag(cluster_tag);
            } else {
                // Dummy tag for non-cluster mode
                mapping.set_tag("{CMD}");
            }
            self.count.fetch_add(1, Ordering::Relaxed);
        } else if self.is_cluster_mode {
            // Update existing
            mapping.set_tag(cluster_tag);
        }
    }

    /// Get cluster tag for a vector ID
    pub fn get_tag(&self, vector_id: u64) -> Option<&str> {
        if vector_id >= self.mappings.len() as u64 || !self.is_cluster_mode {
            return None;
        }
        self.mappings[vector_id as usize].tag_str()
    }

    /// Check if a vector exists in the cluster
    pub fn vector_exists(&self, vector_id: u64) -> bool {
        if vector_id >= self.mappings.len() as u64 {
            return false;
        }
        self.mappings[vector_id as usize].has_tag()
    }

    /// Get number of mapped vectors
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get number of keys scanned
    pub fn keys_scanned(&self) -> u64 {
        self.keys_scanned.load(Ordering::Relaxed)
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.mappings.len()
    }

    /// Claim the next unmapped vector ID (for partial prefill support)
    ///
    /// Atomically finds and claims the next vector ID that doesn't exist in the map.
    /// This allows multiple workers to efficiently skip existing vectors without
    /// duplicate work.
    ///
    /// Returns None when all vectors up to max_id have been processed or mapped.
    pub fn claim_unmapped_id(&self, max_id: u64) -> Option<u64> {
        loop {
            let candidate = self.unmapped_counter.fetch_add(1, Ordering::Relaxed);
            if candidate >= max_id {
                return None; // All vectors processed
            }
            if !self.vector_exists(candidate) {
                return Some(candidate);
            }
            // Vector already exists, try next one
        }
    }

    /// Reset the unmapped counter (call before starting vec-load)
    pub fn reset_unmapped_counter(&self) {
        self.unmapped_counter.store(0, Ordering::Relaxed);
    }

    /// Get current unmapped counter value (for progress tracking)
    pub fn unmapped_counter_value(&self) -> u64 {
        self.unmapped_counter.load(Ordering::Relaxed)
    }
}

/// Configuration for cluster scan
pub struct ClusterScanConfig {
    /// Key pattern to scan for (e.g., "vec:*")
    pub pattern: String,
    /// SCAN batch size
    pub batch_size: usize,
    /// Connection timeout
    pub timeout: Duration,
    /// Whether to show progress
    pub show_progress: bool,
}

impl Default for ClusterScanConfig {
    fn default() -> Self {
        Self {
            pattern: "*".to_string(),
            batch_size: 1000,
            timeout: Duration::from_secs(5),
            show_progress: true,
        }
    }
}

/// Results from cluster scan
#[derive(Debug, Clone)]
pub struct ClusterScanResults {
    /// Total keys processed
    pub total_keys: u64,
    /// Total scan time in milliseconds
    pub total_time_ms: u64,
    /// Number of nodes scanned
    pub nodes_scanned: usize,
    /// Errors encountered
    pub errors: usize,
    /// Keys per second
    pub keys_per_second: f64,
}

/// Extract vector ID and cluster tag from a key
///
/// Uses the unified key format from workload::key_format module.
/// Key format: `prefix{tag}:vector_id`
/// Example: `vec:{ABC}:000123`
///
/// Returns (vector_id, cluster_tag) if successfully parsed
pub fn parse_vector_key(key: &str, prefix: &str) -> Option<(u64, String)> {
    use crate::workload::key_format::{KeyFormat, DEFAULT_KEY_WIDTH};

    let format = KeyFormat::with_cluster_tags(prefix, DEFAULT_KEY_WIDTH);
    let (vector_id, tag_opt) = format.parse_key(key)?;

    // Cluster tag is required for this function
    let cluster_tag = tag_opt?;
    Some((vector_id, cluster_tag))
}

/// Build vector ID mappings by scanning cluster nodes
///
/// This function scans all primary nodes in parallel to discover existing keys
/// and build a mapping from vector_id to cluster_tag.
pub fn build_vector_id_mappings(
    tag_map: &ClusterTagMap,
    nodes: &[ClusterNode],
    config: &ClusterScanConfig,
) -> Result<ClusterScanResults, String> {
    use std::sync::Arc;
    use std::thread;

    let start_time = Instant::now();
    let total_keys = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));

    // Filter to primary nodes only
    let primaries: Vec<_> = nodes.iter().filter(|n| n.is_primary).collect();

    if primaries.is_empty() {
        return Err("No primary nodes found".to_string());
    }

    if config.show_progress {
        println!(
            "[CLUSTER-SCAN] Scanning {} primary nodes for pattern '{}'",
            primaries.len(),
            config.pattern
        );
    }

    // Scan each node in a separate thread
    let handles: Vec<_> = primaries
        .iter()
        .enumerate()
        .map(|(idx, node)| {
            let host = node.host.clone();
            let port = node.port;
            let pattern = config.pattern.clone();
            let batch_size = config.batch_size;
            let timeout = config.timeout;
            let prefix = tag_map.prefix.clone();
            let total_keys = Arc::clone(&total_keys);
            let errors = Arc::clone(&errors);

            // We need to pass the tag_map reference carefully
            // Since ClusterTagMap uses interior mutability, we can share it across threads
            let tag_map_ptr = tag_map as *const ClusterTagMap as usize;

            thread::spawn(move || {
                let result = scan_node(tag_map_ptr, &host, port, &pattern, batch_size, timeout, &prefix, idx);
                match result {
                    Ok(keys) => {
                        total_keys.fetch_add(keys, Ordering::Relaxed);
                    }
                    Err(e) => {
                        eprintln!("[CLUSTER-SCAN] Worker {}: Error: {}", idx, e);
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        let _ = handle.join();
    }

    let elapsed = start_time.elapsed();
    let total_time_ms = elapsed.as_millis() as u64;
    let total_keys_count = total_keys.load(Ordering::Relaxed);
    let keys_per_second = if total_time_ms > 0 {
        (total_keys_count as f64 * 1000.0) / total_time_ms as f64
    } else {
        0.0
    };

    let results = ClusterScanResults {
        total_keys: total_keys_count,
        total_time_ms,
        nodes_scanned: primaries.len(),
        errors: errors.load(Ordering::Relaxed) as usize,
        keys_per_second,
    };

    if config.show_progress {
        println!(
            "[CLUSTER-SCAN] Complete: {} keys in {}ms ({:.1} keys/sec)",
            results.total_keys, results.total_time_ms, results.keys_per_second
        );
        println!(
            "[CLUSTER-SCAN] Mapped {} vectors from {} scanned keys",
            tag_map.count(),
            tag_map.keys_scanned()
        );
    }

    Ok(results)
}

/// Scan a single node for keys matching pattern
fn scan_node(
    tag_map_ptr: usize,
    host: &str,
    port: u16,
    pattern: &str,
    batch_size: usize,
    timeout: Duration,
    prefix: &str,
    _worker_id: usize,
) -> Result<u64, String> {
    // Reconstruct tag_map reference
    let tag_map = unsafe { &*(tag_map_ptr as *const ClusterTagMap) };

    // Connect to node
    let mut conn = RawConnection::connect_tcp(host, port, timeout)
        .map_err(|e| format!("Connection failed: {}", e))?;

    let mut cursor: u64 = 0;
    let mut keys_processed: u64 = 0;

    loop {
        // Build SCAN command
        let mut encoder = RespEncoder::with_capacity(128);
        encoder.encode_command_str(&[
            "SCAN",
            &cursor.to_string(),
            "MATCH",
            pattern,
            "COUNT",
            &batch_size.to_string(),
        ]);

        let reply = conn.execute(&encoder).map_err(|e| format!("SCAN failed: {}", e))?;

        // Parse response: [cursor, [keys...]]
        let (new_cursor, keys) = match reply {
            RespValue::Array(arr) if arr.len() == 2 => {
                let cur = match &arr[0] {
                    RespValue::BulkString(s) => {
                        String::from_utf8_lossy(s).parse::<u64>().unwrap_or(0)
                    }
                    RespValue::Integer(i) => *i as u64,
                    _ => 0,
                };

                let key_list = match &arr[1] {
                    RespValue::Array(keys) => keys
                        .iter()
                        .filter_map(|k| match k {
                            RespValue::BulkString(s) => {
                                String::from_utf8(s.clone()).ok()
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                    _ => vec![],
                };

                (cur, key_list)
            }
            _ => return Err("Invalid SCAN response".to_string()),
        };

        cursor = new_cursor;

        // Process keys
        for key in keys {
            if let Some((vector_id, cluster_tag)) = parse_vector_key(&key, prefix) {
                tag_map.add_mapping(vector_id, &cluster_tag);
            }
            keys_processed += 1;
        }

        // Done when cursor returns to 0
        if cursor == 0 {
            break;
        }
    }

    Ok(keys_processed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vector_key() {
        // Standard format: prefix{tag}:id
        let result = parse_vector_key("vec:{ABC}:000123", "vec:");
        assert_eq!(result, Some((123, "{ABC}".to_string())));

        // Without colon separator
        let result = parse_vector_key("vec:{XYZ}000456", "vec:");
        assert_eq!(result, Some((456, "{XYZ}".to_string())));

        // Wrong prefix
        let result = parse_vector_key("other:{ABC}:000123", "vec:");
        assert!(result.is_none());

        // No tag
        let result = parse_vector_key("vec:000123", "vec:");
        assert!(result.is_none());
    }

    #[test]
    fn test_cluster_tag_map() {
        let map = ClusterTagMap::new("vec:", 1000, true);

        map.add_mapping(0, "{ABC}");
        map.add_mapping(1, "{XYZ}");

        assert!(map.vector_exists(0));
        assert!(map.vector_exists(1));
        assert!(!map.vector_exists(2));

        assert_eq!(map.get_tag(0), Some("{ABC}"));
        assert_eq!(map.get_tag(1), Some("{XYZ}"));
        assert_eq!(map.count(), 2);
    }

    #[test]
    fn test_non_cluster_mode() {
        let map = ClusterTagMap::new("vec:", 1000, false);

        map.add_mapping(0, "{ABC}");

        // In non-cluster mode, get_tag returns None
        assert!(map.get_tag(0).is_none());
        // But vector_exists still works
        assert!(map.vector_exists(0));
    }

    #[test]
    fn test_vector_cluster_mapping() {
        let mut mapping = VectorClusterMapping::default();
        assert!(!mapping.has_tag());

        mapping.set_tag("{ABC}");
        assert!(mapping.has_tag());
        assert_eq!(mapping.tag_str(), Some("{ABC}"));
    }
}
