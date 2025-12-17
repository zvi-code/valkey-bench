//! Cluster topology and node management
//!
//! This module provides cluster support including:
//! - Topology discovery via CLUSTER NODES
//! - Slot mapping and CRC16 calculation
//! - Read-from-replica strategies
//! - Node selection
//! - Dynamic topology refresh on MOVED/ASK errors
//! - Cluster tag mapping for vector ID to node routing

pub mod cluster_tag_map;
pub mod node;
pub mod topology;
pub mod topology_manager;

pub use cluster_tag_map::{
    build_vector_id_mappings, parse_vector_key, ClusterScanConfig, ClusterScanResults,
    ClusterTagMap,
};
pub use node::ClusterNode;
pub use topology::ClusterTopology;
pub use topology_manager::{RedirectInfo, TopologyManager};
