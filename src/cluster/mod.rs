//! Cluster topology and node management
//!
//! This module provides cluster support including:
//! - Topology discovery via CLUSTER NODES
//! - Slot mapping and CRC16 calculation
//! - Read-from-replica strategies
//! - Node selection
//! - Dynamic topology refresh on MOVED/ASK errors

pub mod node;
pub mod topology;
pub mod topology_manager;

pub use node::ClusterNode;
pub use topology::ClusterTopology;
pub use topology_manager::{RedirectInfo, TopologyManager};
