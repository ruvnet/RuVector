//! Virtual networking for QEMU swarm.
//!
//! This module handles network configuration and topology management
//! for inter-node communication.

use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::error::{SwarmError, SwarmResult};
use crate::node::NodeId;

/// MAC address for network interfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MacAddress([u8; 6]);

impl MacAddress {
    /// Create a MAC address from bytes.
    pub fn new(bytes: [u8; 6]) -> Self {
        Self(bytes)
    }

    /// Create a MAC address from a node index.
    /// Uses the QEMU locally-administered unicast prefix (52:54:00).
    pub fn from_index(index: usize) -> Self {
        Self([
            0x52, 0x54, 0x00,
            0x12, 0x34,
            (index & 0xFF) as u8,
        ])
    }

    /// Parse a MAC address from string.
    pub fn parse(s: &str) -> SwarmResult<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 6 {
            return Err(SwarmError::network(format!("Invalid MAC address: {}", s)));
        }

        let mut bytes = [0u8; 6];
        for (i, part) in parts.iter().enumerate() {
            bytes[i] = u8::from_str_radix(part, 16)
                .map_err(|_| SwarmError::network(format!("Invalid MAC byte: {}", part)))?;
        }

        Ok(Self(bytes))
    }

    /// Get the bytes.
    pub fn bytes(&self) -> &[u8; 6] {
        &self.0
    }
}

impl fmt::Display for MacAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5]
        )
    }
}

impl Default for MacAddress {
    fn default() -> Self {
        Self::from_index(0)
    }
}

/// Network topology types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Topology {
    /// Fully connected mesh - all nodes can communicate directly.
    #[default]
    Mesh,

    /// Ring topology - each node connects to two neighbors.
    Ring,

    /// Star topology - one central node, others connect through it.
    Star,

    /// Tree topology - hierarchical structure.
    Tree,

    /// Random connections with specified connectivity.
    Random { connectivity: u8 },

    /// Custom topology from adjacency matrix.
    Custom,
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mesh => write!(f, "mesh"),
            Self::Ring => write!(f, "ring"),
            Self::Star => write!(f, "star"),
            Self::Tree => write!(f, "tree"),
            Self::Random { connectivity } => write!(f, "random({}%)", connectivity),
            Self::Custom => write!(f, "custom"),
        }
    }
}

/// Network link between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLink {
    /// Source node ID.
    pub from: NodeId,

    /// Destination node ID.
    pub to: NodeId,

    /// Link latency in microseconds.
    pub latency_us: u32,

    /// Bandwidth in Mbps.
    pub bandwidth_mbps: u32,

    /// Packet loss percentage.
    pub packet_loss_pct: u8,

    /// Whether the link is currently active.
    pub active: bool,
}

impl NetworkLink {
    /// Create a new network link with default parameters.
    pub fn new(from: NodeId, to: NodeId) -> Self {
        Self {
            from,
            to,
            latency_us: 0,
            bandwidth_mbps: 1000,
            packet_loss_pct: 0,
            active: true,
        }
    }

    /// Builder: set latency.
    pub fn with_latency(mut self, us: u32) -> Self {
        self.latency_us = us;
        self
    }

    /// Builder: set bandwidth.
    pub fn with_bandwidth(mut self, mbps: u32) -> Self {
        self.bandwidth_mbps = mbps;
        self
    }

    /// Builder: set packet loss.
    pub fn with_packet_loss(mut self, pct: u8) -> Self {
        self.packet_loss_pct = pct.min(100);
        self
    }
}

/// Network topology configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    /// Topology type.
    pub topology_type: Topology,

    /// Number of nodes.
    pub node_count: usize,

    /// Adjacency matrix (node_count x node_count).
    /// True if nodes are connected.
    adjacency: Vec<Vec<bool>>,

    /// Links with their properties.
    links: Vec<NetworkLink>,
}

impl NetworkTopology {
    /// Create a new network topology.
    pub fn new(topology_type: Topology, node_count: usize) -> Self {
        let mut topology = Self {
            topology_type,
            node_count,
            adjacency: vec![vec![false; node_count]; node_count],
            links: Vec::new(),
        };

        topology.generate_topology();
        topology
    }

    /// Generate the topology based on type.
    fn generate_topology(&mut self) {
        match self.topology_type {
            Topology::Mesh => self.generate_mesh(),
            Topology::Ring => self.generate_ring(),
            Topology::Star => self.generate_star(),
            Topology::Tree => self.generate_tree(),
            Topology::Random { connectivity } => self.generate_random(connectivity),
            Topology::Custom => {} // User must set manually
        }
    }

    /// Generate mesh topology (fully connected).
    fn generate_mesh(&mut self) {
        for i in 0..self.node_count {
            for j in 0..self.node_count {
                if i != j {
                    self.adjacency[i][j] = true;
                    if i < j {
                        self.links.push(NetworkLink::new(
                            NodeId::from_index(i),
                            NodeId::from_index(j),
                        ));
                    }
                }
            }
        }
    }

    /// Generate ring topology.
    fn generate_ring(&mut self) {
        for i in 0..self.node_count {
            let next = (i + 1) % self.node_count;
            self.adjacency[i][next] = true;
            self.adjacency[next][i] = true;
            self.links.push(NetworkLink::new(
                NodeId::from_index(i),
                NodeId::from_index(next),
            ));
        }
    }

    /// Generate star topology (node 0 is the center).
    fn generate_star(&mut self) {
        for i in 1..self.node_count {
            self.adjacency[0][i] = true;
            self.adjacency[i][0] = true;
            self.links.push(NetworkLink::new(
                NodeId::from_index(0),
                NodeId::from_index(i),
            ));
        }
    }

    /// Generate tree topology.
    fn generate_tree(&mut self) {
        // Binary tree structure
        for i in 1..self.node_count {
            let parent = (i - 1) / 2;
            self.adjacency[parent][i] = true;
            self.adjacency[i][parent] = true;
            self.links.push(NetworkLink::new(
                NodeId::from_index(parent),
                NodeId::from_index(i),
            ));
        }
    }

    /// Generate random topology with specified connectivity percentage.
    fn generate_random(&mut self, connectivity: u8) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let threshold = connectivity as f64 / 100.0;

        for i in 0..self.node_count {
            for j in (i + 1)..self.node_count {
                if rng.gen::<f64>() < threshold {
                    self.adjacency[i][j] = true;
                    self.adjacency[j][i] = true;
                    self.links.push(NetworkLink::new(
                        NodeId::from_index(i),
                        NodeId::from_index(j),
                    ));
                }
            }
        }

        // Ensure connectivity by connecting isolated nodes
        self.ensure_connected();
    }

    /// Ensure the network is connected.
    fn ensure_connected(&mut self) {
        let mut visited = vec![false; self.node_count];
        let mut stack = vec![0];

        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;

            for (neighbor, &connected) in self.adjacency[node].iter().enumerate() {
                if connected && !visited[neighbor] {
                    stack.push(neighbor);
                }
            }
        }

        // Connect any unvisited nodes to node 0
        for i in 0..self.node_count {
            if !visited[i] {
                self.adjacency[0][i] = true;
                self.adjacency[i][0] = true;
                self.links.push(NetworkLink::new(
                    NodeId::from_index(0),
                    NodeId::from_index(i),
                ));
            }
        }
    }

    /// Check if two nodes are directly connected.
    pub fn are_connected(&self, a: usize, b: usize) -> bool {
        if a >= self.node_count || b >= self.node_count {
            return false;
        }
        self.adjacency[a][b]
    }

    /// Get neighbors of a node.
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        if node >= self.node_count {
            return Vec::new();
        }
        self.adjacency[node]
            .iter()
            .enumerate()
            .filter_map(|(i, &connected)| if connected { Some(i) } else { None })
            .collect()
    }

    /// Get all links.
    pub fn links(&self) -> &[NetworkLink] {
        &self.links
    }

    /// Get link between two nodes.
    pub fn get_link(&self, from: usize, to: usize) -> Option<&NetworkLink> {
        self.links.iter().find(|l| {
            let from_idx = from;
            let to_idx = to;
            (l.from == NodeId::from_index(from_idx) && l.to == NodeId::from_index(to_idx))
                || (l.from == NodeId::from_index(to_idx) && l.to == NodeId::from_index(from_idx))
        })
    }

    /// Get the diameter of the network (longest shortest path).
    pub fn diameter(&self) -> usize {
        let mut max_dist = 0;
        for i in 0..self.node_count {
            let distances = self.shortest_paths(i);
            for d in distances {
                if d != usize::MAX && d > max_dist {
                    max_dist = d;
                }
            }
        }
        max_dist
    }

    /// Compute shortest paths from a node using BFS.
    pub fn shortest_paths(&self, from: usize) -> Vec<usize> {
        let mut distances = vec![usize::MAX; self.node_count];
        let mut queue = std::collections::VecDeque::new();

        distances[from] = 0;
        queue.push_back(from);

        while let Some(node) = queue.pop_front() {
            for neighbor in self.neighbors(node) {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[node] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }

    /// Print ASCII visualization of the topology.
    pub fn ascii_diagram(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Topology: {} ({} nodes)\n", self.topology_type, self.node_count));
        output.push_str("Adjacency:\n");

        // Header
        output.push_str("     ");
        for i in 0..self.node_count {
            output.push_str(&format!(" N{:<2}", i));
        }
        output.push('\n');

        // Rows
        for i in 0..self.node_count {
            output.push_str(&format!(" N{:<2} |", i));
            for j in 0..self.node_count {
                if i == j {
                    output.push_str("  - ");
                } else if self.adjacency[i][j] {
                    output.push_str("  X ");
                } else {
                    output.push_str("  . ");
                }
            }
            output.push('\n');
        }

        output.push_str(&format!("\nDiameter: {}\n", self.diameter()));
        output.push_str(&format!("Links: {}\n", self.links.len()));

        output
    }
}

/// Virtual network manager for the swarm.
pub struct VirtualNetwork {
    /// Network topology.
    topology: NetworkTopology,

    /// Multicast groups for each isolation group.
    multicast_groups: HashMap<String, SocketAddr>,

    /// Base multicast address.
    base_multicast: SocketAddr,

    /// Node to group mapping.
    node_groups: HashMap<NodeId, String>,
}

impl VirtualNetwork {
    /// Create a new virtual network.
    pub fn new(topology: NetworkTopology, base_multicast: SocketAddr) -> Self {
        let mut multicast_groups = HashMap::new();
        multicast_groups.insert("default".to_string(), base_multicast);

        Self {
            topology,
            multicast_groups,
            base_multicast,
            node_groups: HashMap::new(),
        }
    }

    /// Get the network topology.
    pub fn topology(&self) -> &NetworkTopology {
        &self.topology
    }

    /// Get multicast group for a node.
    pub fn multicast_for_node(&self, node_id: NodeId) -> SocketAddr {
        if let Some(group_name) = self.node_groups.get(&node_id) {
            if let Some(addr) = self.multicast_groups.get(group_name) {
                return *addr;
            }
        }
        self.base_multicast
    }

    /// Create an isolation group with its own multicast address.
    pub fn create_isolation_group(&mut self, name: &str, nodes: &[NodeId]) -> SwarmResult<()> {
        // Generate a unique multicast address for this group
        let group_index = self.multicast_groups.len();
        let base_ip = self.base_multicast.ip();
        let new_ip = match base_ip {
            std::net::IpAddr::V4(ip) => {
                let octets = ip.octets();
                std::net::IpAddr::V4(std::net::Ipv4Addr::new(
                    octets[0],
                    octets[1],
                    octets[2],
                    (octets[3] as usize + group_index) as u8,
                ))
            }
            _ => return Err(SwarmError::network("IPv6 not supported for multicast")),
        };

        let addr = SocketAddr::new(new_ip, self.base_multicast.port());
        self.multicast_groups.insert(name.to_string(), addr);

        for node_id in nodes {
            self.node_groups.insert(*node_id, name.to_string());
        }

        info!(
            group = name,
            addr = %addr,
            nodes = nodes.len(),
            "Created isolation group"
        );

        Ok(())
    }

    /// Check if two nodes can communicate.
    pub fn can_communicate(&self, a: NodeId, b: NodeId) -> bool {
        // Check if they're in the same isolation group
        let group_a = self.node_groups.get(&a);
        let group_b = self.node_groups.get(&b);

        match (group_a, group_b) {
            (Some(ga), Some(gb)) => ga == gb,
            (None, None) => true, // Both in default group
            _ => false,           // One isolated, one not
        }
    }

    /// Simulate network partition by isolating nodes.
    pub fn partition(&mut self, group_a: &[NodeId], group_b: &[NodeId]) -> SwarmResult<()> {
        self.create_isolation_group("partition_a", group_a)?;
        self.create_isolation_group("partition_b", group_b)?;
        info!(
            "Network partitioned: {} vs {} nodes",
            group_a.len(),
            group_b.len()
        );
        Ok(())
    }

    /// Heal a network partition (return all nodes to default group).
    pub fn heal_partition(&mut self) {
        self.node_groups.clear();
        info!("Network partition healed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_address_from_index() {
        let mac = MacAddress::from_index(0);
        assert_eq!(mac.to_string(), "52:54:00:12:34:00");

        let mac = MacAddress::from_index(255);
        assert_eq!(mac.to_string(), "52:54:00:12:34:ff");
    }

    #[test]
    fn test_mac_address_parse() {
        let mac = MacAddress::parse("52:54:00:12:34:56").unwrap();
        assert_eq!(mac.bytes(), &[0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
    }

    #[test]
    fn test_mesh_topology() {
        let topology = NetworkTopology::new(Topology::Mesh, 4);

        // All nodes should be connected to all others
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(topology.are_connected(i, j));
                }
            }
        }

        // Diameter should be 1 for fully connected mesh
        assert_eq!(topology.diameter(), 1);
    }

    #[test]
    fn test_ring_topology() {
        let topology = NetworkTopology::new(Topology::Ring, 4);

        // Each node connected to neighbors only
        assert!(topology.are_connected(0, 1));
        assert!(topology.are_connected(1, 2));
        assert!(topology.are_connected(2, 3));
        assert!(topology.are_connected(3, 0));

        // Non-adjacent nodes not directly connected
        assert!(!topology.are_connected(0, 2));
        assert!(!topology.are_connected(1, 3));

        // Diameter should be 2 for 4-node ring
        assert_eq!(topology.diameter(), 2);
    }

    #[test]
    fn test_star_topology() {
        let topology = NetworkTopology::new(Topology::Star, 5);

        // Center (node 0) connected to all
        for i in 1..5 {
            assert!(topology.are_connected(0, i));
        }

        // Periphery nodes not connected to each other
        assert!(!topology.are_connected(1, 2));
        assert!(!topology.are_connected(2, 3));

        // Diameter should be 2
        assert_eq!(topology.diameter(), 2);
    }

    #[test]
    fn test_topology_ascii() {
        let topology = NetworkTopology::new(Topology::Mesh, 3);
        let diagram = topology.ascii_diagram();
        assert!(diagram.contains("Topology: mesh"));
        assert!(diagram.contains("3 nodes"));
    }
}
