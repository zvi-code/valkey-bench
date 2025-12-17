//! Index backfill progress tracking
//!
//! Provides functions to monitor index backfill progress across cluster nodes,
//! supporting both EC (ElastiCache Valkey) and MemoryDB engines.

use std::thread;
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};

use super::ft_info::{
    convert_ftinfo_to_lines, convert_memdb_ftinfo_to_lines, parse_ftinfo_lines, EngineType,
    FtInfoResult, IndexStatus,
};
use crate::client::{ControlPlane, RawConnection};
use crate::cluster::ClusterNode;
use crate::utils::{RespEncoder, RespValue};

/// Progress information for a single node
#[derive(Debug, Clone)]
pub struct NodeProgress {
    /// Node identifier
    pub node_id: String,
    /// Number of documents indexed
    pub num_docs: i64,
    /// Progress percentage (0-100)
    pub progress_percent: i32,
    /// Whether backfill is in progress
    pub in_progress: bool,
    /// Index status
    pub status: IndexStatus,
}

/// Get node progress for EC (ElastiCache Valkey)
pub fn get_node_progress_ec(
    conn: &mut RawConnection,
    index_name: &str,
) -> Result<NodeProgress, String> {
    // Send FT.INFO command
    let mut encoder = RespEncoder::with_capacity(128);
    encoder.encode_command_str(&["FT.INFO", index_name]);

    let reply = conn
        .execute_encoded(&encoder)
        .map_err(|e| format!("FT.INFO failed: {}", e))?;

    let lines = convert_ftinfo_to_lines(&reply, None);
    let info = parse_ftinfo_lines(&lines);

    // Extract backfill status
    let backfill_in_progress: i32 = info
        .get("backfill_in_progress")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let backfill_complete_percent: f64 = info
        .get("backfill_complete_percent")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    let num_docs: i64 = info
        .get("num_docs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let progress_percent = (backfill_complete_percent * 100.0) as i32;

    Ok(NodeProgress {
        node_id: String::new(), // Set by caller
        num_docs,
        progress_percent,
        in_progress: backfill_in_progress != 0,
        status: if backfill_in_progress != 0 {
            IndexStatus::Backfilling
        } else {
            IndexStatus::Available
        },
    })
}

/// Get node progress for MemoryDB
pub fn get_node_progress_memorydb(
    conn: &mut RawConnection,
    index_name: &str,
) -> Result<NodeProgress, String> {
    // Send FT.INFO command
    let mut encoder = RespEncoder::with_capacity(128);
    encoder.encode_command_str(&["FT.INFO", index_name]);

    let ft_info_reply = conn
        .execute_encoded(&encoder)
        .map_err(|e| format!("FT.INFO failed: {}", e))?;

    let ft_info_lines = convert_memdb_ftinfo_to_lines(&ft_info_reply, None);
    let ft_info = parse_ftinfo_lines(&ft_info_lines);

    // Send INFO SEARCH command
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["INFO", "SEARCH"]);

    let search_info_reply = conn
        .execute_encoded(&encoder)
        .map_err(|e| format!("INFO SEARCH failed: {}", e))?;

    let search_info_lines = match &search_info_reply {
        RespValue::BulkString(data) => String::from_utf8_lossy(data).to_string(),
        _ => convert_memdb_ftinfo_to_lines(&search_info_reply, None),
    };
    let search_info = parse_ftinfo_lines(&search_info_lines);

    // Parse FT.INFO fields
    let status_str = ft_info.get("index_status").map(|s| s.as_str()).unwrap_or("AVAILABLE");
    let status = IndexStatus::from_str(status_str);

    let degradation: i32 = ft_info
        .get("index_degradation_percentage")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let num_docs: i64 = ft_info
        .get("num_indexed_vectors")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Parse INFO SEARCH fields
    let active_backfills: i32 = search_info
        .get("search_num_active_backfills")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let backfill_progress: i32 = search_info
        .get("search_current_backfill_progress_percentage")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    // Determine progress
    let (in_progress, progress_percent) = if status == IndexStatus::Backfilling {
        (true, backfill_progress)
    } else if degradation > 0 {
        (true, 100 - degradation)
    } else if status.is_in_progress() || active_backfills > 0 {
        (true, backfill_progress)
    } else {
        (false, 100)
    };

    Ok(NodeProgress {
        node_id: String::new(), // Set by caller
        num_docs,
        progress_percent,
        in_progress,
        status,
    })
}

/// Get node progress based on engine type
pub fn get_node_progress(
    conn: &mut RawConnection,
    index_name: &str,
    engine_type: EngineType,
) -> Result<NodeProgress, String> {
    match engine_type {
        EngineType::MemoryDb => get_node_progress_memorydb(conn, index_name),
        _ => get_node_progress_ec(conn, index_name),
    }
}

/// Cluster-wide backfill progress
#[derive(Debug, Clone)]
pub struct ClusterBackfillProgress {
    /// Total documents across all nodes
    pub total_docs: i64,
    /// Number of nodes still backfilling
    pub nodes_in_progress: usize,
    /// Average progress percentage
    pub progress_percent: i32,
    /// Per-node progress
    pub node_progress: Vec<NodeProgress>,
}

/// Configuration for waiting on backfill completion
#[derive(Debug, Clone)]
pub struct BackfillWaitConfig {
    /// Poll interval
    pub poll_interval: Duration,
    /// Initial delay before first check
    pub initial_delay: Duration,
    /// Maximum wait time (None for unlimited)
    pub max_wait: Option<Duration>,
    /// Whether to show progress bar
    pub show_progress: bool,
}

impl Default for BackfillWaitConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(1),
            initial_delay: Duration::from_secs(2),
            max_wait: None,
            show_progress: true,
        }
    }
}

/// Wait for index backfill to complete on all nodes
///
/// This function polls all cluster nodes until backfill is complete or timeout.
pub fn wait_for_index_backfill_complete<F>(
    engine_type: EngineType,
    index_names: &[&str],
    mut get_connection: F,
    node_count: usize,
    config: &BackfillWaitConfig,
) -> Result<ClusterBackfillProgress, String>
where
    F: FnMut(usize) -> Option<RawConnection>,
{
    if index_names.is_empty() {
        return Ok(ClusterBackfillProgress {
            total_docs: 0,
            nodes_in_progress: 0,
            progress_percent: 100,
            node_progress: vec![],
        });
    }

    // Print waiting message
    let engine_name = match engine_type {
        EngineType::MemoryDb => "MemoryDB",
        _ => "ValkeySearch",
    };

    let index_list = index_names.join(", ");
    println!(
        "{}: Waiting for index{}: '{}' backfill to complete on all nodes...",
        engine_name,
        if index_names.len() > 1 { "es" } else { "" },
        index_list
    );

    // Initial delay
    thread::sleep(config.initial_delay);

    // Set up progress bar
    let progress_bar = if config.show_progress {
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}% | {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        None
    };

    let start_time = Instant::now();
    let mut last_progress = ClusterBackfillProgress {
        total_docs: 0,
        nodes_in_progress: node_count,
        progress_percent: 0,
        node_progress: vec![],
    };

    loop {
        // Check timeout
        if let Some(max_wait) = config.max_wait {
            if start_time.elapsed() > max_wait {
                if let Some(pb) = &progress_bar {
                    pb.finish_with_message("Timeout");
                }
                return Err("Backfill wait timeout".to_string());
            }
        }

        // Collect progress from all nodes
        let mut total_docs = 0i64;
        let mut nodes_in_progress = 0usize;
        let mut total_progress = 0i32;
        let mut node_progress = Vec::with_capacity(node_count * index_names.len());

        for node_idx in 0..node_count {
            let Some(mut conn) = get_connection(node_idx) else {
                continue;
            };

            for index_name in index_names {
                match get_node_progress(&mut conn, index_name, engine_type) {
                    Ok(mut progress) => {
                        progress.node_id = format!("node-{}", node_idx);

                        if progress.in_progress {
                            nodes_in_progress += 1;
                        }

                        total_docs += progress.num_docs;
                        total_progress += progress.progress_percent;
                        node_progress.push(progress);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to get progress from node {}: {}", node_idx, e);
                        nodes_in_progress += 1; // Assume still in progress on error
                    }
                }
            }
        }

        let entry_count = node_count * index_names.len();
        let avg_progress = if entry_count > 0 {
            total_progress / entry_count as i32
        } else {
            100
        };

        last_progress = ClusterBackfillProgress {
            total_docs,
            nodes_in_progress,
            progress_percent: avg_progress,
            node_progress,
        };

        // Update progress bar
        if let Some(pb) = &progress_bar {
            pb.set_position(avg_progress as u64);
            pb.set_message(format!(
                "{} docs, {} nodes in progress",
                total_docs, nodes_in_progress
            ));
        }

        // Check if complete
        if nodes_in_progress == 0 {
            break;
        }

        thread::sleep(config.poll_interval);
    }

    // Finish progress bar
    if let Some(pb) = &progress_bar {
        pb.set_position(100);
        pb.finish_with_message(format!("Complete - {} docs", last_progress.total_docs));
    }

    println!(
        "{} Index{} backfill process has completed on all nodes. Total docs indexed: {}",
        index_names.len(),
        if index_names.len() > 1 { "es" } else { "" },
        last_progress.total_docs
    );

    Ok(last_progress)
}

/// Simple callback-based backfill wait for use with ClusterTopology
pub fn wait_for_backfill<'a>(
    engine_type: EngineType,
    index_names: &[&str],
    nodes: &'a [ClusterNode],
    create_connection: impl Fn(&'a ClusterNode) -> Option<RawConnection>,
    config: &BackfillWaitConfig,
) -> Result<ClusterBackfillProgress, String> {
    let node_connections: Vec<_> = nodes
        .iter()
        .filter_map(|node| {
            if node.is_primary {
                create_connection(node).map(|conn| (node.id.clone(), conn))
            } else {
                None
            }
        })
        .collect();

    let node_count = node_connections.len();
    let mut connections: Vec<Option<RawConnection>> =
        node_connections.into_iter().map(|(_, c)| Some(c)).collect();

    wait_for_index_backfill_complete(
        engine_type,
        index_names,
        |idx| {
            if idx < connections.len() {
                connections[idx].take()
            } else {
                None
            }
        },
        node_count,
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backfill_wait_config_default() {
        let config = BackfillWaitConfig::default();
        assert_eq!(config.poll_interval, Duration::from_secs(1));
        assert_eq!(config.initial_delay, Duration::from_secs(2));
        assert!(config.max_wait.is_none());
        assert!(config.show_progress);
    }

    #[test]
    fn test_cluster_progress_empty() {
        let progress = ClusterBackfillProgress {
            total_docs: 0,
            nodes_in_progress: 0,
            progress_percent: 100,
            node_progress: vec![],
        };
        assert_eq!(progress.progress_percent, 100);
    }
}
