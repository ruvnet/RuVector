//! Topology-aware routing for MCP tool calls across agent networks.
//!
//! Provides [`TopologyRouter`] that directs tool calls based on the
//! deployment topology (standalone, hierarchical, mesh, adaptive).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Topology enums
// ---------------------------------------------------------------------------

/// Topology type for agent deployment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TopologyType {
    /// Single agent, no coordination.
    Standalone,
    /// Tree structure with queen/leader at root.
    Hierarchical,
    /// Fully connected peer-to-peer.
    Mesh,
    /// Dynamic switching based on load/topology.
    Adaptive,
}

impl Default for TopologyType {
    fn default() -> Self {
        Self::Standalone
    }
}

/// Role of a node in the topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeRole {
    Queen,
    Worker,
    Scout,
    Specialist,
    Router,
}

/// Status of a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeStatus {
    Active,
    Idle,
    Busy,
    Failed,
    Draining,
}

/// Consensus algorithm type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsensusType {
    Raft,
    Byzantine,
    Gossip,
    None,
}

// ---------------------------------------------------------------------------
// TopologyNode
// ---------------------------------------------------------------------------

/// A node in the topology graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyNode {
    pub id: String,
    pub role: NodeRole,
    pub status: NodeStatus,
    pub tools: Vec<String>,
    pub connections: Vec<String>,
}

// ---------------------------------------------------------------------------
// TopologyConfig
// ---------------------------------------------------------------------------

/// Topology configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    pub topology_type: TopologyType,
    pub max_agents: usize,
    pub consensus: ConsensusType,
    pub health_check_interval_ms: u64,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            topology_type: TopologyType::Standalone,
            max_agents: 8,
            consensus: ConsensusType::Raft,
            health_check_interval_ms: 5000,
        }
    }
}

// ---------------------------------------------------------------------------
// TopologyRouter
// ---------------------------------------------------------------------------

/// Router that directs tool calls through the topology.
pub struct TopologyRouter {
    config: TopologyConfig,
    nodes: HashMap<String, TopologyNode>,
}

impl TopologyRouter {
    /// Create a new router with the given configuration.
    pub fn new(config: TopologyConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
        }
    }

    /// Create a standalone router (single agent, no coordination).
    pub fn standalone() -> Self {
        Self::new(TopologyConfig::default())
    }

    /// Create a hierarchical router.
    pub fn hierarchical(max_agents: usize) -> Self {
        Self::new(TopologyConfig {
            topology_type: TopologyType::Hierarchical,
            max_agents,
            consensus: ConsensusType::Raft,
            ..Default::default()
        })
    }

    /// Create a mesh router.
    pub fn mesh(max_agents: usize) -> Self {
        Self::new(TopologyConfig {
            topology_type: TopologyType::Mesh,
            max_agents,
            consensus: ConsensusType::Byzantine,
            ..Default::default()
        })
    }

    /// Create an adaptive router.
    pub fn adaptive(max_agents: usize) -> Self {
        Self::new(TopologyConfig {
            topology_type: TopologyType::Adaptive,
            max_agents,
            consensus: ConsensusType::Gossip,
            ..Default::default()
        })
    }

    /// Add a node to the topology.
    pub fn add_node(&mut self, node: TopologyNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Remove a node from the topology.
    pub fn remove_node(&mut self, id: &str) -> Option<TopologyNode> {
        self.nodes.remove(id)
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&TopologyNode> {
        self.nodes.get(id)
    }

    /// Get all active nodes.
    pub fn active_nodes(&self) -> Vec<&TopologyNode> {
        self.nodes
            .values()
            .filter(|n| n.status == NodeStatus::Active)
            .collect()
    }

    /// Get the topology type.
    pub fn topology_type(&self) -> &TopologyType {
        &self.config.topology_type
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the topology configuration.
    pub fn config(&self) -> &TopologyConfig {
        &self.config
    }

    /// Route a tool call to the best available node.
    ///
    /// Returns `None` for standalone (handle locally) or when no node
    /// is available. Returns `Some(node_id)` for the target node.
    pub fn route_tool_call(&self, tool_name: &str) -> Option<String> {
        match &self.config.topology_type {
            TopologyType::Standalone => None,
            TopologyType::Hierarchical => self.route_hierarchical(tool_name),
            TopologyType::Mesh => self.route_mesh(tool_name),
            TopologyType::Adaptive => self.route_adaptive(tool_name),
        }
    }

    fn route_hierarchical(&self, tool_name: &str) -> Option<String> {
        // Find a specialist with the tool, or fall back to queen
        self.nodes
            .values()
            .find(|n| {
                n.status == NodeStatus::Active
                    && n.tools.contains(&tool_name.to_string())
            })
            .or_else(|| {
                self.nodes.values().find(|n| {
                    n.role == NodeRole::Queen && n.status == NodeStatus::Active
                })
            })
            .map(|n| n.id.clone())
    }

    fn route_mesh(&self, tool_name: &str) -> Option<String> {
        // Find first active node with the tool
        self.nodes
            .values()
            .find(|n| {
                n.status == NodeStatus::Active
                    && n.tools.contains(&tool_name.to_string())
            })
            .map(|n| n.id.clone())
    }

    fn route_adaptive(&self, tool_name: &str) -> Option<String> {
        // Prefer idle nodes, then active, then busy
        self.nodes
            .values()
            .filter(|n| n.tools.contains(&tool_name.to_string()))
            .min_by_key(|n| match n.status {
                NodeStatus::Idle => 0,
                NodeStatus::Active => 1,
                NodeStatus::Busy => 2,
                _ => 3,
            })
            .map(|n| n.id.clone())
    }

    /// Get topology status as JSON.
    pub fn status(&self) -> serde_json::Value {
        serde_json::json!({
            "topology": self.config.topology_type,
            "max_agents": self.config.max_agents,
            "node_count": self.nodes.len(),
            "active_nodes": self.active_nodes().len(),
            "consensus": self.config.consensus,
            "nodes": self.nodes.values().map(|n| serde_json::json!({
                "id": n.id,
                "role": n.role,
                "status": n.status,
                "tools": n.tools,
                "connections": n.connections,
            })).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, role: NodeRole, status: NodeStatus, tools: Vec<&str>) -> TopologyNode {
        TopologyNode {
            id: id.into(),
            role,
            status,
            tools: tools.into_iter().map(|s| s.to_string()).collect(),
            connections: vec![],
        }
    }

    #[test]
    fn test_standalone_routing_returns_none() {
        let router = TopologyRouter::standalone();
        assert_eq!(router.route_tool_call("read_file"), None);
    }

    #[test]
    fn test_hierarchical_routing_finds_specialist() {
        let mut router = TopologyRouter::hierarchical(8);
        router.add_node(make_node(
            "queen-1",
            NodeRole::Queen,
            NodeStatus::Active,
            vec!["execute"],
        ));
        router.add_node(make_node(
            "spec-1",
            NodeRole::Specialist,
            NodeStatus::Active,
            vec!["read_file", "write_file"],
        ));
        let target = router.route_tool_call("read_file");
        assert_eq!(target, Some("spec-1".into()));
    }

    #[test]
    fn test_hierarchical_falls_back_to_queen() {
        let mut router = TopologyRouter::hierarchical(8);
        router.add_node(make_node(
            "queen-1",
            NodeRole::Queen,
            NodeStatus::Active,
            vec!["execute"],
        ));
        // No specialist has "unknown_tool"
        let target = router.route_tool_call("unknown_tool");
        assert_eq!(target, Some("queen-1".into()));
    }

    #[test]
    fn test_mesh_routing_finds_active_node() {
        let mut router = TopologyRouter::mesh(4);
        router.add_node(make_node(
            "node-1",
            NodeRole::Worker,
            NodeStatus::Active,
            vec!["grep"],
        ));
        let target = router.route_tool_call("grep");
        assert_eq!(target, Some("node-1".into()));
    }

    #[test]
    fn test_mesh_routing_no_match() {
        let mut router = TopologyRouter::mesh(4);
        router.add_node(make_node(
            "node-1",
            NodeRole::Worker,
            NodeStatus::Active,
            vec!["grep"],
        ));
        let target = router.route_tool_call("read_file");
        assert_eq!(target, None);
    }

    #[test]
    fn test_adaptive_prefers_idle_nodes() {
        let mut router = TopologyRouter::adaptive(8);
        router.add_node(make_node(
            "busy-1",
            NodeRole::Worker,
            NodeStatus::Busy,
            vec!["ls"],
        ));
        router.add_node(make_node(
            "idle-1",
            NodeRole::Worker,
            NodeStatus::Idle,
            vec!["ls"],
        ));
        router.add_node(make_node(
            "active-1",
            NodeRole::Worker,
            NodeStatus::Active,
            vec!["ls"],
        ));
        let target = router.route_tool_call("ls");
        assert_eq!(target, Some("idle-1".into()));
    }

    #[test]
    fn test_add_remove_node() {
        let mut router = TopologyRouter::standalone();
        router.add_node(make_node(
            "n1",
            NodeRole::Worker,
            NodeStatus::Active,
            vec![],
        ));
        assert_eq!(router.node_count(), 1);
        let removed = router.remove_node("n1");
        assert!(removed.is_some());
        assert_eq!(router.node_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_node() {
        let mut router = TopologyRouter::standalone();
        assert!(router.remove_node("nope").is_none());
    }

    #[test]
    fn test_get_node() {
        let mut router = TopologyRouter::standalone();
        router.add_node(make_node(
            "n1",
            NodeRole::Scout,
            NodeStatus::Idle,
            vec![],
        ));
        let node = router.get_node("n1").unwrap();
        assert_eq!(node.role, NodeRole::Scout);
    }

    #[test]
    fn test_get_node_not_found() {
        let router = TopologyRouter::standalone();
        assert!(router.get_node("missing").is_none());
    }

    #[test]
    fn test_active_nodes_filtering() {
        let mut router = TopologyRouter::standalone();
        router.add_node(make_node(
            "a",
            NodeRole::Worker,
            NodeStatus::Active,
            vec![],
        ));
        router.add_node(make_node(
            "b",
            NodeRole::Worker,
            NodeStatus::Failed,
            vec![],
        ));
        router.add_node(make_node(
            "c",
            NodeRole::Worker,
            NodeStatus::Active,
            vec![],
        ));
        assert_eq!(router.active_nodes().len(), 2);
    }

    #[test]
    fn test_topology_type_accessor() {
        let router = TopologyRouter::mesh(4);
        assert_eq!(router.topology_type(), &TopologyType::Mesh);
    }

    #[test]
    fn test_node_count() {
        let mut router = TopologyRouter::standalone();
        assert_eq!(router.node_count(), 0);
        router.add_node(make_node(
            "x",
            NodeRole::Worker,
            NodeStatus::Idle,
            vec![],
        ));
        assert_eq!(router.node_count(), 1);
    }

    #[test]
    fn test_status_json_shape() {
        let mut router = TopologyRouter::hierarchical(8);
        router.add_node(make_node(
            "q",
            NodeRole::Queen,
            NodeStatus::Active,
            vec!["ls"],
        ));
        let status = router.status();
        assert_eq!(status["topology"], "hierarchical");
        assert_eq!(status["max_agents"], 8);
        assert_eq!(status["node_count"], 1);
        assert_eq!(status["active_nodes"], 1);
        assert!(status["nodes"].is_array());
    }

    #[test]
    fn test_topology_config_defaults() {
        let config = TopologyConfig::default();
        assert_eq!(config.topology_type, TopologyType::Standalone);
        assert_eq!(config.max_agents, 8);
        assert_eq!(config.consensus, ConsensusType::Raft);
        assert_eq!(config.health_check_interval_ms, 5000);
    }

    #[test]
    fn test_topology_type_serde() {
        let tt = TopologyType::Hierarchical;
        let json = serde_json::to_string(&tt).unwrap();
        assert_eq!(json, "\"hierarchical\"");
        let back: TopologyType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, TopologyType::Hierarchical);
    }

    #[test]
    fn test_node_role_serde() {
        for role in &[
            NodeRole::Queen,
            NodeRole::Worker,
            NodeRole::Scout,
            NodeRole::Specialist,
            NodeRole::Router,
        ] {
            let json = serde_json::to_string(role).unwrap();
            let back: NodeRole = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, role);
        }
    }

    #[test]
    fn test_node_status_serde() {
        for status in &[
            NodeStatus::Active,
            NodeStatus::Idle,
            NodeStatus::Busy,
            NodeStatus::Failed,
            NodeStatus::Draining,
        ] {
            let json = serde_json::to_string(status).unwrap();
            let back: NodeStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, status);
        }
    }

    #[test]
    fn test_consensus_type_serde() {
        for ct in &[
            ConsensusType::Raft,
            ConsensusType::Byzantine,
            ConsensusType::Gossip,
            ConsensusType::None,
        ] {
            let json = serde_json::to_string(ct).unwrap();
            let back: ConsensusType = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, ct);
        }
    }

    #[test]
    fn test_topology_node_serde_roundtrip() {
        let node = TopologyNode {
            id: "n1".into(),
            role: NodeRole::Worker,
            status: NodeStatus::Active,
            tools: vec!["ls".into(), "grep".into()],
            connections: vec!["n2".into()],
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: TopologyNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "n1");
        assert_eq!(back.tools.len(), 2);
    }

    #[test]
    fn test_topology_config_serde_roundtrip() {
        let config = TopologyConfig {
            topology_type: TopologyType::Mesh,
            max_agents: 16,
            consensus: ConsensusType::Byzantine,
            health_check_interval_ms: 3000,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: TopologyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.topology_type, TopologyType::Mesh);
        assert_eq!(back.max_agents, 16);
    }

    #[test]
    fn test_adaptive_routing_skips_failed() {
        let mut router = TopologyRouter::adaptive(4);
        router.add_node(make_node(
            "failed-1",
            NodeRole::Worker,
            NodeStatus::Failed,
            vec!["ls"],
        ));
        router.add_node(make_node(
            "active-1",
            NodeRole::Worker,
            NodeStatus::Active,
            vec!["ls"],
        ));
        let target = router.route_tool_call("ls");
        assert_eq!(target, Some("active-1".into()));
    }

    #[test]
    fn test_config_accessor() {
        let router = TopologyRouter::standalone();
        let config = router.config();
        assert_eq!(config.max_agents, 8);
    }
}
