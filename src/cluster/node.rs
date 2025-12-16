//! Cluster node representation

/// Cluster node information
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Node ID from CLUSTER NODES
    pub id: String,
    /// Hostname or IP
    pub host: String,
    /// Port
    pub port: u16,
    /// Cluster bus port
    pub bus_port: u16,
    /// Is this a primary node?
    pub is_primary: bool,
    /// Is this a replica?
    pub is_replica: bool,
    /// Primary node ID (if replica)
    pub primary_id: Option<String>,
    /// Assigned slots (for primaries)
    pub slots: Vec<u16>,
    /// Node flags (fail, handshake, etc.)
    pub flags: Vec<String>,
    /// Availability zone (for AWS)
    pub az: Option<String>,
    /// Is node currently available?
    pub available: bool,
}

impl ClusterNode {
    /// Check if node is available
    pub fn is_available(&self) -> bool {
        self.available && !self.flags.iter().any(|f| f == "fail" || f == "handshake")
    }

    /// Get node address as string
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Parse a line from CLUSTER NODES response
///
/// Format: `<id> <ip:port@cport> <flags> <master> <ping-sent> <pong-recv> <config-epoch> <link-state> <slot> <slot> ... <slot>`
///
/// Example:
/// ```text
/// 07c37dfeb235213a872192d90877d0cd55635b91 127.0.0.1:30004@31004 slave e7d1eecce10fd6bb5eb35b9f99a514335d9ba9ca 0 1426238317239 4 connected
/// ```
pub fn parse_cluster_node_line(line: &str) -> Option<ClusterNode> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 8 {
        return None;
    }

    let id = parts[0].to_string();

    // Parse address: ip:port@cport or ip:port
    let addr_str = parts[1];
    let (host, port, bus_port) = parse_node_address(addr_str)?;

    // Parse flags
    let flags: Vec<String> = parts[2].split(',').map(String::from).collect();

    let is_primary = flags.contains(&"master".to_string());
    let is_replica = flags.contains(&"slave".to_string()) || flags.contains(&"replica".to_string());

    // Parse primary ID (for replicas)
    let primary_id = if is_replica && parts[3] != "-" {
        Some(parts[3].to_string())
    } else {
        None
    };

    // Parse link state
    let link_state = parts[7];
    let available = link_state == "connected";

    // Parse slots (for primaries, starting at index 8)
    let mut slots = Vec::new();
    if is_primary {
        for &slot_str in &parts[8..] {
            if let Some((start, end)) = parse_slot_range(slot_str) {
                for slot in start..=end {
                    slots.push(slot);
                }
            }
        }
    }

    Some(ClusterNode {
        id,
        host,
        port,
        bus_port,
        is_primary,
        is_replica,
        primary_id,
        slots,
        flags,
        az: None, // Not available in standard CLUSTER NODES
        available,
    })
}

/// Parse node address from CLUSTER NODES
/// Formats: "host:port@cport", "host:port", "host:port@cport,hostname"
fn parse_node_address(addr: &str) -> Option<(String, u16, u16)> {
    // Handle ElastiCache format: ip:port@cport,hostname
    let addr = addr.split(',').next().unwrap_or(addr);

    // Split on @ for cluster bus port
    let parts: Vec<&str> = addr.split('@').collect();
    let host_port = parts[0];
    let bus_port = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    // Split host:port
    let hp: Vec<&str> = host_port.rsplitn(2, ':').collect();
    if hp.len() != 2 {
        return None;
    }

    let port: u16 = hp[0].parse().ok()?;
    let host = hp[1].to_string();

    Some((host, port, bus_port))
}

/// Parse slot range: "0-5460" or "0"
fn parse_slot_range(s: &str) -> Option<(u16, u16)> {
    // Skip importing slots like "[123->-node_id]"
    if s.contains('[') {
        return None;
    }

    if let Some(idx) = s.find('-') {
        let start: u16 = s[..idx].parse().ok()?;
        let end: u16 = s[idx + 1..].parse().ok()?;
        Some((start, end))
    } else {
        let slot: u16 = s.parse().ok()?;
        Some((slot, slot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_primary_node() {
        let line = "07c37dfeb235213a872192d90877d0cd55635b91 127.0.0.1:30001@31001 master - 0 1426238316232 1 connected 0-5460";
        let node = parse_cluster_node_line(line).unwrap();

        assert_eq!(node.id, "07c37dfeb235213a872192d90877d0cd55635b91");
        assert_eq!(node.host, "127.0.0.1");
        assert_eq!(node.port, 30001);
        assert_eq!(node.bus_port, 31001);
        assert!(node.is_primary);
        assert!(!node.is_replica);
        assert!(node.available);
        assert_eq!(node.slots.len(), 5461); // 0-5460 inclusive
    }

    #[test]
    fn test_parse_replica_node() {
        let line = "07c37dfeb235213a872192d90877d0cd55635b91 127.0.0.1:30004@31004 slave e7d1eecce10fd6bb5eb35b9f99a514335d9ba9ca 0 1426238317239 4 connected";
        let node = parse_cluster_node_line(line).unwrap();

        assert!(node.is_replica);
        assert!(!node.is_primary);
        assert_eq!(
            node.primary_id,
            Some("e7d1eecce10fd6bb5eb35b9f99a514335d9ba9ca".to_string())
        );
        assert!(node.slots.is_empty());
    }

    #[test]
    fn test_parse_node_address() {
        let (host, port, bus) = parse_node_address("127.0.0.1:6379@16379").unwrap();
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 6379);
        assert_eq!(bus, 16379);
    }

    #[test]
    fn test_parse_node_address_no_bus_port() {
        let (host, port, _) = parse_node_address("127.0.0.1:6379").unwrap();
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 6379);
    }

    #[test]
    fn test_parse_elasticache_address() {
        let (host, port, bus) =
            parse_node_address("10.0.0.1:6379@16379,hostname.example.com").unwrap();
        assert_eq!(host, "10.0.0.1");
        assert_eq!(port, 6379);
        assert_eq!(bus, 16379);
    }

    #[test]
    fn test_parse_slot_range() {
        assert_eq!(parse_slot_range("0-5460"), Some((0, 5460)));
        assert_eq!(parse_slot_range("5461"), Some((5461, 5461)));
        assert_eq!(parse_slot_range("[123->-abc]"), None);
    }
}
