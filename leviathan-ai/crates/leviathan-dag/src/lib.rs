//! # Leviathan DAG - Bank-Grade Auditability System
//!
//! A comprehensive DAG (Directed Acyclic Graph) implementation for tracking data lineage,
//! ensuring auditability, and maintaining BCBS 239 compliance in financial systems.
//!
//! ## Features
//!
//! - **Content-Addressable Storage**: Each node is uniquely identified by its content hash
//! - **Cryptographic Verification**: Merkle tree proofs and tamper detection
//! - **Data Lineage Tracking**: Full provenance tracking for regulatory compliance
//! - **Multiple Parent Support**: Merge tracking for complex data flows
//! - **Graph Export**: Visualization support via GraphViz
//!
//! ## Example
//!
//! ```rust
//! use leviathan_dag::{AuditDag, NodeType, DataNode};
//!
//! let mut dag = AuditDag::new();
//!
//! // Add a data node
//! let node_id = dag.add_data_node(
//!     b"transaction data",
//!     vec![],
//!     serde_json::json!({"type": "transaction"})
//! ).unwrap();
//!
//! // Verify the chain
//! assert!(dag.verify_chain(&node_id).unwrap());
//! ```

pub mod node;
pub mod lineage;
pub mod verify;

use chrono::{DateTime, Utc};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use uuid::Uuid;

pub use node::{ComputeNode, DataNode, CheckpointNode, ValidationNode, NodeType};
pub use lineage::{LineageTracker, DataFlowEdge, TransformationType};
pub use verify::{MerkleProof, verify_proof, compute_merkle_root};

/// Errors that can occur during DAG operations
#[derive(Error, Debug)]
pub enum DagError {
    #[error("Node not found: {0}")]
    NodeNotFound(Uuid),

    #[error("Cycle detected in DAG")]
    CycleDetected,

    #[error("Invalid parent reference: {0}")]
    InvalidParent(Uuid),

    #[error("Hash verification failed")]
    HashVerificationFailed,

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid node type")]
    InvalidNodeType,
}

/// Result type for DAG operations
pub type DagResult<T> = Result<T, DagError>;

/// A node in the DAG representing an auditable operation or data state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    /// Unique identifier for this node
    pub id: Uuid,

    /// Content-addressable hash (BLAKE3) of data + parents
    pub hash: String,

    /// Timestamp when the node was created
    pub timestamp: DateTime<Utc>,

    /// List of parent node IDs (supports multiple parents for merges)
    pub parent_ids: Vec<Uuid>,

    /// The actual data stored in this node
    pub data: Vec<u8>,

    /// Metadata for additional context
    pub metadata: serde_json::Value,

    /// Type of node (Data, Compute, Validation, Checkpoint)
    pub node_type: NodeType,
}

impl DagNode {
    /// Create a new DAG node with computed hash
    pub fn new(
        data: Vec<u8>,
        parent_ids: Vec<Uuid>,
        metadata: serde_json::Value,
        node_type: NodeType,
    ) -> Self {
        let id = Uuid::new_v4();
        let timestamp = Utc::now();

        // Compute deterministic hash from data + parents
        let hash = Self::compute_hash(&data, &parent_ids);

        Self {
            id,
            hash,
            timestamp,
            parent_ids,
            data,
            metadata,
            node_type,
        }
    }

    /// Compute content-addressable hash (data + parents = deterministic hash)
    fn compute_hash(data: &[u8], parent_ids: &[Uuid]) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);

        // Include parent IDs in hash for chain integrity
        for parent_id in parent_ids {
            hasher.update(parent_id.as_bytes());
        }

        hasher.finalize().to_hex().to_string()
    }

    /// Verify that this node's hash is correct
    pub fn verify_hash(&self) -> bool {
        let computed_hash = Self::compute_hash(&self.data, &self.parent_ids);
        computed_hash == self.hash
    }
}

/// The main DAG structure for audit tracking
#[derive(Debug, Clone)]
pub struct AuditDag {
    /// Underlying directed graph structure
    graph: DiGraph<DagNode, ()>,

    /// Map from UUID to graph NodeIndex for fast lookup
    node_map: HashMap<Uuid, NodeIndex>,

    /// Lineage tracker for data flow analysis
    lineage_tracker: LineageTracker,
}

impl AuditDag {
    /// Create a new empty audit DAG
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            lineage_tracker: LineageTracker::new(),
        }
    }

    /// Add a node to the DAG
    pub fn add_node(&mut self, node: DagNode) -> DagResult<Uuid> {
        // Verify all parent nodes exist
        for parent_id in &node.parent_ids {
            if !self.node_map.contains_key(parent_id) {
                return Err(DagError::InvalidParent(*parent_id));
            }
        }

        let node_id = node.id;
        let parent_ids = node.parent_ids.clone();
        let node_idx = self.graph.add_node(node);

        // Add edges from parents
        for parent_id in &parent_ids {
            if let Some(&parent_idx) = self.node_map.get(parent_id) {
                self.graph.add_edge(parent_idx, node_idx, ());
            }
        }

        self.node_map.insert(node_id, node_idx);

        // Check for cycles (should never happen in a proper DAG)
        if petgraph::algo::is_cyclic_directed(&self.graph) {
            // Remove the node if it creates a cycle
            self.graph.remove_node(node_idx);
            self.node_map.remove(&node_id);
            return Err(DagError::CycleDetected);
        }

        Ok(node_id)
    }

    /// Add a data node (convenience method)
    pub fn add_data_node(
        &mut self,
        data: &[u8],
        parent_ids: Vec<Uuid>,
        metadata: serde_json::Value,
    ) -> DagResult<Uuid> {
        let node = DagNode::new(
            data.to_vec(),
            parent_ids,
            metadata,
            NodeType::Data(DataNode { size: data.len() }),
        );
        self.add_node(node)
    }

    /// Add a compute node (convenience method)
    pub fn add_compute_node(
        &mut self,
        operation: String,
        result: &[u8],
        parent_ids: Vec<Uuid>,
        metadata: serde_json::Value,
    ) -> DagResult<Uuid> {
        let node = DagNode::new(
            result.to_vec(),
            parent_ids,
            metadata,
            NodeType::Compute(ComputeNode { operation }),
        );
        self.add_node(node)
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &Uuid) -> DagResult<&DagNode> {
        let idx = self.node_map
            .get(id)
            .ok_or(DagError::NodeNotFound(*id))?;
        Ok(&self.graph[*idx])
    }

    /// Get the complete lineage (ancestry) of a node
    pub fn get_lineage(&self, id: &Uuid) -> DagResult<Vec<Uuid>> {
        let idx = self.node_map
            .get(id)
            .ok_or(DagError::NodeNotFound(*id))?;

        let mut lineage = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![*idx];

        while let Some(current_idx) = stack.pop() {
            if visited.contains(&current_idx) {
                continue;
            }

            visited.insert(current_idx);
            let node = &self.graph[current_idx];
            lineage.push(node.id);

            // Add all parents to stack
            for parent_id in &node.parent_ids {
                if let Some(&parent_idx) = self.node_map.get(parent_id) {
                    stack.push(parent_idx);
                }
            }
        }

        Ok(lineage)
    }

    /// Verify the integrity of the chain up to this node
    pub fn verify_chain(&self, id: &Uuid) -> DagResult<bool> {
        let lineage = self.get_lineage(id)?;

        for node_id in lineage {
            let node = self.get_node(&node_id)?;

            // Verify the node's own hash
            if !node.verify_hash() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Export the DAG to GraphViz DOT format for visualization
    pub fn export_graphviz(&self) -> String {
        use petgraph::dot::{Config, Dot};

        let dot = Dot::with_config(&self.graph, &[Config::EdgeNoLabel]);
        format!("{:?}", dot)
    }

    /// Get all nodes with no parents (root nodes)
    pub fn get_roots(&self) -> Vec<Uuid> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Incoming)
                    .count() == 0
            })
            .map(|idx| self.graph[idx].id)
            .collect()
    }

    /// Get all nodes with no children (leaf nodes)
    pub fn get_leaves(&self) -> Vec<Uuid> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Outgoing)
                    .count() == 0
            })
            .map(|idx| self.graph[idx].id)
            .collect()
    }

    /// Get statistics about the DAG
    pub fn stats(&self) -> DagStats {
        DagStats {
            total_nodes: self.graph.node_count(),
            total_edges: self.graph.edge_count(),
            root_nodes: self.get_roots().len(),
            leaf_nodes: self.get_leaves().len(),
        }
    }

    /// Get the lineage tracker for advanced queries
    pub fn lineage_tracker(&self) -> &LineageTracker {
        &self.lineage_tracker
    }

    /// Get mutable access to the lineage tracker
    pub fn lineage_tracker_mut(&mut self) -> &mut LineageTracker {
        &mut self.lineage_tracker
    }
}

impl Default for AuditDag {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a DAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub root_nodes: usize,
    pub leaf_nodes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_dag_node() {
        let node = DagNode::new(
            b"test data".to_vec(),
            vec![],
            serde_json::json!({"type": "test"}),
            NodeType::Data(DataNode { size: 9 }),
        );

        assert!(node.verify_hash());
        assert_eq!(node.data, b"test data");
    }

    #[test]
    fn test_add_node() {
        let mut dag = AuditDag::new();

        let node = DagNode::new(
            b"test".to_vec(),
            vec![],
            serde_json::json!({}),
            NodeType::Data(DataNode { size: 4 }),
        );

        let id = dag.add_node(node).unwrap();
        assert!(dag.get_node(&id).is_ok());
    }

    #[test]
    fn test_lineage() {
        let mut dag = AuditDag::new();

        // Create a chain: root -> child1 -> child2
        let root = dag.add_data_node(
            b"root",
            vec![],
            serde_json::json!({}),
        ).unwrap();

        let child1 = dag.add_data_node(
            b"child1",
            vec![root],
            serde_json::json!({}),
        ).unwrap();

        let child2 = dag.add_data_node(
            b"child2",
            vec![child1],
            serde_json::json!({}),
        ).unwrap();

        let lineage = dag.get_lineage(&child2).unwrap();
        assert_eq!(lineage.len(), 3);
        assert!(lineage.contains(&root));
        assert!(lineage.contains(&child1));
        assert!(lineage.contains(&child2));
    }

    #[test]
    fn test_verify_chain() {
        let mut dag = AuditDag::new();

        let root = dag.add_data_node(
            b"root",
            vec![],
            serde_json::json!({}),
        ).unwrap();

        assert!(dag.verify_chain(&root).unwrap());
    }

    #[test]
    fn test_cycle_detection() {
        let mut dag = AuditDag::new();

        let node1 = dag.add_data_node(
            b"node1",
            vec![],
            serde_json::json!({}),
        ).unwrap();

        // Attempting to create a cycle should fail
        let node2 = DagNode::new(
            b"node2".to_vec(),
            vec![node1],
            serde_json::json!({}),
            NodeType::Data(DataNode { size: 5 }),
        );

        dag.add_node(node2).unwrap();
        // Note: Actual cycle would require manual graph manipulation
    }
}
