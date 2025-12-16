//! Cluster topology management

use std::collections::HashMap;

use super::node::{parse_cluster_node_line, ClusterNode};
use crate::config::ReadFromReplica;

/// Cluster topology snapshot
#[derive(Debug, Clone)]
pub struct ClusterTopology {
    /// All nodes in the cluster
    pub nodes: Vec<ClusterNode>,
    /// Slot to node index mapping
    slot_map: [Option<usize>; 16384],
    /// Primary node indices
    primary_indices: Vec<usize>,
    /// Replica node indices grouped by primary ID
    replica_map: HashMap<String, Vec<usize>>,
}

impl ClusterTopology {
    /// Parse CLUSTER NODES response
    pub fn from_cluster_nodes(response: &str) -> Result<Self, String> {
        let mut nodes = Vec::new();
        let mut slot_map = [None; 16384];
        let mut primary_indices = Vec::new();
        let mut replica_map: HashMap<String, Vec<usize>> = HashMap::new();

        for line in response.lines() {
            if line.is_empty() {
                continue;
            }

            if let Some(node) = parse_cluster_node_line(line) {
                let idx = nodes.len();

                if node.is_primary {
                    primary_indices.push(idx);
                    for &slot in &node.slots {
                        slot_map[slot as usize] = Some(idx);
                    }
                    replica_map.insert(node.id.clone(), Vec::new());
                }

                nodes.push(node);
            }
        }

        // Map replicas to primaries
        for (idx, node) in nodes.iter().enumerate() {
            if node.is_replica {
                if let Some(ref primary_id) = node.primary_id {
                    if let Some(replicas) = replica_map.get_mut(primary_id) {
                        replicas.push(idx);
                    }
                }
            }
        }

        if primary_indices.is_empty() {
            return Err("No primary nodes found".to_string());
        }

        Ok(Self {
            nodes,
            slot_map,
            primary_indices,
            replica_map,
        })
    }

    /// Get node for slot
    pub fn get_node_for_slot(&self, slot: u16) -> Option<&ClusterNode> {
        self.slot_map[slot as usize].map(|idx| &self.nodes[idx])
    }

    /// Get node index for slot
    pub fn get_node_idx_for_slot(&self, slot: u16) -> Option<usize> {
        self.slot_map[slot as usize]
    }

    /// Get all primary nodes
    pub fn primaries(&self) -> impl Iterator<Item = &ClusterNode> {
        self.primary_indices.iter().map(|&idx| &self.nodes[idx])
    }

    /// Get primary node indices
    pub fn primary_indices(&self) -> &[usize] {
        &self.primary_indices
    }

    /// Get replicas for a primary
    pub fn replicas_for(&self, primary_id: &str) -> Vec<&ClusterNode> {
        self.replica_map
            .get(primary_id)
            .map(|indices| indices.iter().map(|&idx| &self.nodes[idx]).collect())
            .unwrap_or_default()
    }

    /// Get nodes based on read-from-replica strategy
    pub fn select_nodes(&self, strategy: ReadFromReplica) -> Vec<(usize, &ClusterNode)> {
        match strategy {
            ReadFromReplica::Primary => self
                .primary_indices
                .iter()
                .map(|&idx| (idx, &self.nodes[idx]))
                .filter(|(_, n)| n.is_available())
                .collect(),
            ReadFromReplica::PreferReplica => {
                // Collect available replicas
                let mut nodes: Vec<(usize, &ClusterNode)> = self
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.is_replica && n.is_available())
                    .collect();

                // Fallback to primaries if no replicas
                if nodes.is_empty() {
                    nodes = self
                        .primary_indices
                        .iter()
                        .map(|&idx| (idx, &self.nodes[idx]))
                        .filter(|(_, n)| n.is_available())
                        .collect();
                }
                nodes
            }
            ReadFromReplica::RoundRobin => {
                // All available nodes
                self.nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.is_available())
                    .collect()
            }
            ReadFromReplica::AzAffinity => {
                // Group by AZ, prefer nodes in same AZ
                // Simplified: return all available, let caller filter by AZ
                self.nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| n.is_available())
                    .collect()
            }
        }
    }

    /// Get number of primary nodes
    pub fn num_primaries(&self) -> usize {
        self.primary_indices.len()
    }

    /// Get total number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Calculate slot for key using CRC16
    pub fn slot_for_key(key: &[u8]) -> u16 {
        // Check for hash tag {xxx}
        if let Some(start) = key.iter().position(|&b| b == b'{') {
            if let Some(end) = key[start + 1..].iter().position(|&b| b == b'}') {
                if end > 0 {
                    return crc16(&key[start + 1..start + 1 + end]) % 16384;
                }
            }
        }
        crc16(key) % 16384
    }

    /// Get all node addresses for metrics collection
    pub fn all_node_addresses(&self) -> Vec<(String, u16, bool)> {
        self.nodes
            .iter()
            .map(|n| (n.host.clone(), n.port, n.is_primary))
            .collect()
    }
}

/// CRC16 implementation for Redis cluster slot calculation (XMODEM)
fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_calculation_hash_tag() {
        // Keys with same hash tag should map to same slot
        let slot1 = ClusterTopology::slot_for_key(b"{foo}bar");
        let slot2 = ClusterTopology::slot_for_key(b"{foo}baz");
        assert_eq!(slot1, slot2);
    }

    #[test]
    fn test_slot_calculation_no_tag() {
        let slot = ClusterTopology::slot_for_key(b"hello");
        assert!(slot < 16384);
    }

    #[test]
    fn test_crc16() {
        // Known test vector: "123456789" -> 0x31C3
        assert_eq!(crc16(b"123456789"), 0x31C3);
    }

    #[test]
    fn test_parse_cluster_nodes() {
        let response = r#"
07c37dfeb235213a872192d90877d0cd55635b91 127.0.0.1:30001@31001 master - 0 1426238316232 1 connected 0-5460
e7d1eecce10fd6bb5eb35b9f99a514335d9ba9ca 127.0.0.1:30002@31002 master - 0 1426238316232 2 connected 5461-10922
67ed2db8d677e59ec4a4cefb06858cf2a1a89fa1 127.0.0.1:30003@31003 master - 0 1426238316232 3 connected 10923-16383
292f8b365bb7edb5e285caf0b7e6ddc7265d2f4f 127.0.0.1:30004@31004 slave 07c37dfeb235213a872192d90877d0cd55635b91 0 1426238316232 1 connected
"#;

        let topology = ClusterTopology::from_cluster_nodes(response).unwrap();

        assert_eq!(topology.num_primaries(), 3);
        assert_eq!(topology.num_nodes(), 4);

        // Check slot mapping
        assert!(topology.get_node_for_slot(0).is_some());
        assert!(topology.get_node_for_slot(5460).is_some());
        assert!(topology.get_node_for_slot(16383).is_some());
    }

    #[test]
    fn test_select_nodes_primary() {
        let response = r#"
07c37dfeb235213a872192d90877d0cd55635b91 127.0.0.1:30001@31001 master - 0 1426238316232 1 connected 0-5460
292f8b365bb7edb5e285caf0b7e6ddc7265d2f4f 127.0.0.1:30004@31004 slave 07c37dfeb235213a872192d90877d0cd55635b91 0 1426238316232 1 connected
"#;

        let topology = ClusterTopology::from_cluster_nodes(response).unwrap();
        let nodes = topology.select_nodes(ReadFromReplica::Primary);

        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].1.is_primary);
    }
}
