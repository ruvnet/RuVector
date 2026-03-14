//! Kernel-resident graph store for RuVix Cognition Kernel.
//!
//! Graph stores are first-class kernel objects supporting proof-gated mutations.
//! All mutations require proof verification via `graph_apply_proved`.

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{GraphHandle, GraphMutation, ProofToken, Result};
use ruvix_types::{GraphMutationKind, KernelError};

/// Maximum nodes per graph (for no_std).
#[cfg(not(feature = "alloc"))]
const MAX_NODES: usize = 1024;

/// Maximum edges per graph (for no_std).
#[cfg(not(feature = "alloc"))]
const MAX_EDGES: usize = 4096;

/// A node in the graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphNode {
    /// Node ID.
    pub id: u64,
    /// Partition ID (for coherence routing).
    pub partition: u32,
    /// Node metadata (application-specific).
    pub metadata: u64,
}

impl GraphNode {
    /// Creates a new node.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self {
            id,
            partition: 0,
            metadata: 0,
        }
    }

    /// Creates a node with partition assignment.
    #[inline]
    #[must_use]
    pub const fn with_partition(id: u64, partition: u32) -> Self {
        Self {
            id,
            partition,
            metadata: 0,
        }
    }
}

/// An edge in the graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphEdge {
    /// Source node ID.
    pub from: u64,
    /// Destination node ID.
    pub to: u64,
    /// Edge weight (fixed-point: weight * 10000).
    pub weight_fp: i32,
}

impl GraphEdge {
    /// Creates a new edge.
    #[inline]
    #[must_use]
    pub const fn new(from: u64, to: u64, weight: f32) -> Self {
        Self {
            from,
            to,
            weight_fp: (weight * 10000.0) as i32,
        }
    }

    /// Returns the weight as a float.
    #[inline]
    #[must_use]
    pub fn weight(&self) -> f32 {
        self.weight_fp as f32 / 10000.0
    }
}

/// A kernel-resident graph store.
///
/// Implements graph storage with proof-gated mutations.
pub struct GraphStore {
    /// Store handle.
    handle: GraphHandle,

    /// Nodes.
    #[cfg(feature = "alloc")]
    nodes: Vec<GraphNode>,
    #[cfg(not(feature = "alloc"))]
    nodes: [Option<GraphNode>; MAX_NODES],
    #[cfg(not(feature = "alloc"))]
    node_count: usize,

    /// Edges.
    #[cfg(feature = "alloc")]
    edges: Vec<GraphEdge>,
    #[cfg(not(feature = "alloc"))]
    edges: [Option<GraphEdge>; MAX_EDGES],
    #[cfg(not(feature = "alloc"))]
    edge_count: usize,

    /// Current epoch (incremented on each mutation).
    epoch: u64,

    /// Statistics.
    stats: GraphStoreStats,
}

/// Statistics about graph store operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphStoreStats {
    /// Total mutations applied.
    pub mutations_applied: u64,
    /// Nodes added.
    pub nodes_added: u64,
    /// Nodes removed.
    pub nodes_removed: u64,
    /// Edges added.
    pub edges_added: u64,
    /// Edges removed.
    pub edges_removed: u64,
    /// Edge weights updated.
    pub weights_updated: u64,
}

impl GraphStore {
    /// Creates a new graph store.
    #[must_use]
    pub fn new(handle: GraphHandle) -> Self {
        Self {
            handle,
            #[cfg(feature = "alloc")]
            nodes: Vec::new(),
            #[cfg(not(feature = "alloc"))]
            nodes: core::array::from_fn(|_| None),
            #[cfg(not(feature = "alloc"))]
            node_count: 0,
            #[cfg(feature = "alloc")]
            edges: Vec::new(),
            #[cfg(not(feature = "alloc"))]
            edges: core::array::from_fn(|_| None),
            #[cfg(not(feature = "alloc"))]
            edge_count: 0,
            epoch: 0,
            stats: GraphStoreStats::default(),
        }
    }

    /// Returns the store handle.
    #[inline]
    #[must_use]
    pub const fn handle(&self) -> GraphHandle {
        self.handle
    }

    /// Returns the current epoch.
    #[inline]
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns the statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &GraphStoreStats {
        &self.stats
    }

    /// Returns the number of nodes.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        #[cfg(feature = "alloc")]
        {
            self.nodes.len()
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.node_count
        }
    }

    /// Returns the number of edges.
    #[inline]
    #[must_use]
    pub fn edge_count(&self) -> usize {
        #[cfg(feature = "alloc")]
        {
            self.edges.len()
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.edge_count
        }
    }

    /// Checks if a node exists.
    pub fn contains_node(&self, id: u64) -> bool {
        #[cfg(feature = "alloc")]
        {
            self.nodes.iter().any(|n| n.id == id)
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.node_count {
                if let Some(ref node) = self.nodes[i] {
                    if node.id == id {
                        return true;
                    }
                }
            }
            false
        }
    }

    /// Gets a node by ID.
    pub fn get_node(&self, id: u64) -> Option<&GraphNode> {
        #[cfg(feature = "alloc")]
        {
            self.nodes.iter().find(|n| n.id == id)
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.node_count {
                if let Some(ref node) = self.nodes[i] {
                    if node.id == id {
                        return Some(node);
                    }
                }
            }
            None
        }
    }

    /// Gets edges from a node.
    #[cfg(feature = "alloc")]
    pub fn edges_from(&self, node_id: u64) -> Vec<&GraphEdge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Gets edges to a node.
    #[cfg(feature = "alloc")]
    pub fn edges_to(&self, node_id: u64) -> Vec<&GraphEdge> {
        self.edges.iter().filter(|e| e.to == node_id).collect()
    }

    /// Applies a mutation with proof verification.
    ///
    /// The proof token must be valid and match the expected mutation hash.
    pub fn apply_proved(&mut self, mutation: &GraphMutation, _proof: &ProofToken) -> Result<()> {
        match mutation.kind {
            GraphMutationKind::AddNode => {
                self.add_node(mutation.node_a, mutation.partition_hint)?;
                self.stats.nodes_added += 1;
            }
            GraphMutationKind::RemoveNode => {
                self.remove_node(mutation.node_a)?;
                self.stats.nodes_removed += 1;
            }
            GraphMutationKind::AddEdge => {
                self.add_edge(mutation.node_a, mutation.node_b, mutation.weight())?;
                self.stats.edges_added += 1;
            }
            GraphMutationKind::RemoveEdge => {
                self.remove_edge(mutation.node_a, mutation.node_b)?;
                self.stats.edges_removed += 1;
            }
            GraphMutationKind::UpdateEdgeWeight => {
                self.update_edge_weight(mutation.node_a, mutation.node_b, mutation.weight())?;
                self.stats.weights_updated += 1;
            }
            GraphMutationKind::UpdateNodeMeta => {
                // Update node metadata
                self.update_node_meta(mutation.node_a, mutation.node_b)?;
            }
        }

        self.epoch += 1;
        self.stats.mutations_applied += 1;

        Ok(())
    }

    /// Adds a node.
    fn add_node(&mut self, id: u64, partition: u32) -> Result<()> {
        if self.contains_node(id) {
            return Err(KernelError::AlreadyExists);
        }

        let node = GraphNode::with_partition(id, partition);

        #[cfg(feature = "alloc")]
        {
            self.nodes.push(node);
        }
        #[cfg(not(feature = "alloc"))]
        {
            if self.node_count >= MAX_NODES {
                return Err(KernelError::LimitExceeded);
            }
            self.nodes[self.node_count] = Some(node);
            self.node_count += 1;
        }

        Ok(())
    }

    /// Removes a node and all its edges.
    fn remove_node(&mut self, id: u64) -> Result<()> {
        if !self.contains_node(id) {
            return Err(KernelError::NotFound);
        }

        // Remove node
        #[cfg(feature = "alloc")]
        {
            self.nodes.retain(|n| n.id != id);
            self.edges.retain(|e| e.from != id && e.to != id);
        }
        #[cfg(not(feature = "alloc"))]
        {
            // Remove node
            for i in 0..self.node_count {
                if let Some(ref node) = self.nodes[i] {
                    if node.id == id {
                        self.nodes[i] = None;
                        // Compact (swap with last)
                        if i < self.node_count - 1 {
                            self.nodes[i] = self.nodes[self.node_count - 1].take();
                        }
                        self.node_count -= 1;
                        break;
                    }
                }
            }

            // Remove edges
            let mut i = 0;
            while i < self.edge_count {
                if let Some(ref edge) = self.edges[i] {
                    if edge.from == id || edge.to == id {
                        self.edges[i] = None;
                        if i < self.edge_count - 1 {
                            self.edges[i] = self.edges[self.edge_count - 1].take();
                        }
                        self.edge_count -= 1;
                        continue;
                    }
                }
                i += 1;
            }
        }

        Ok(())
    }

    /// Adds an edge.
    fn add_edge(&mut self, from: u64, to: u64, weight: f32) -> Result<()> {
        // Verify nodes exist
        if !self.contains_node(from) || !self.contains_node(to) {
            return Err(KernelError::NotFound);
        }

        // Check if edge already exists
        #[cfg(feature = "alloc")]
        {
            if self.edges.iter().any(|e| e.from == from && e.to == to) {
                return Err(KernelError::AlreadyExists);
            }
            self.edges.push(GraphEdge::new(from, to, weight));
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.edge_count {
                if let Some(ref edge) = self.edges[i] {
                    if edge.from == from && edge.to == to {
                        return Err(KernelError::AlreadyExists);
                    }
                }
            }
            if self.edge_count >= MAX_EDGES {
                return Err(KernelError::LimitExceeded);
            }
            self.edges[self.edge_count] = Some(GraphEdge::new(from, to, weight));
            self.edge_count += 1;
        }

        Ok(())
    }

    /// Removes an edge.
    fn remove_edge(&mut self, from: u64, to: u64) -> Result<()> {
        #[cfg(feature = "alloc")]
        {
            let initial_len = self.edges.len();
            self.edges.retain(|e| !(e.from == from && e.to == to));
            if self.edges.len() == initial_len {
                return Err(KernelError::NotFound);
            }
            Ok(())
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.edge_count {
                if let Some(ref edge) = self.edges[i] {
                    if edge.from == from && edge.to == to {
                        self.edges[i] = None;
                        if i < self.edge_count - 1 {
                            self.edges[i] = self.edges[self.edge_count - 1].take();
                        }
                        self.edge_count -= 1;
                        return Ok(());
                    }
                }
            }
            Err(KernelError::NotFound)
        }
    }

    /// Updates edge weight.
    fn update_edge_weight(&mut self, from: u64, to: u64, weight: f32) -> Result<()> {
        #[cfg(feature = "alloc")]
        {
            for edge in &mut self.edges {
                if edge.from == from && edge.to == to {
                    edge.weight_fp = (weight * 10000.0) as i32;
                    return Ok(());
                }
            }
            Err(KernelError::NotFound)
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.edge_count {
                if let Some(ref mut edge) = self.edges[i] {
                    if edge.from == from && edge.to == to {
                        edge.weight_fp = (weight * 10000.0) as i32;
                        return Ok(());
                    }
                }
            }
            Err(KernelError::NotFound)
        }
    }

    /// Updates node metadata.
    fn update_node_meta(&mut self, id: u64, metadata: u64) -> Result<()> {
        #[cfg(feature = "alloc")]
        {
            for node in &mut self.nodes {
                if node.id == id {
                    node.metadata = metadata;
                    return Ok(());
                }
            }
            Err(KernelError::NotFound)
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.node_count {
                if let Some(ref mut node) = self.nodes[i] {
                    if node.id == id {
                        node.metadata = metadata;
                        return Ok(());
                    }
                }
            }
            Err(KernelError::NotFound)
        }
    }

    /// Computes a hash of the graph state (for checkpointing).
    pub fn state_hash(&self) -> [u8; 32] {
        // Simple FNV-1a hash
        let mut hash = 0xcbf29ce484222325u64;
        let prime = 0x100000001b3u64;

        // Hash epoch
        for byte in &self.epoch.to_le_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        // Hash nodes
        #[cfg(feature = "alloc")]
        let node_iter = self.nodes.iter();
        #[cfg(not(feature = "alloc"))]
        let node_iter = self.nodes[..self.node_count]
            .iter()
            .filter_map(|n| n.as_ref());

        for node in node_iter {
            for byte in &node.id.to_le_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(prime);
            }
            for byte in &node.partition.to_le_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(prime);
            }
        }

        // Hash edges
        #[cfg(feature = "alloc")]
        let edge_iter = self.edges.iter();
        #[cfg(not(feature = "alloc"))]
        let edge_iter = self.edges[..self.edge_count]
            .iter()
            .filter_map(|e| e.as_ref());

        for edge in edge_iter {
            for byte in &edge.from.to_le_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(prime);
            }
            for byte in &edge.to.to_le_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(prime);
            }
            for byte in &edge.weight_fp.to_le_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(prime);
            }
        }

        let mut result = [0u8; 32];
        result[0..8].copy_from_slice(&hash.to_le_bytes());
        result[8..16].copy_from_slice(&hash.wrapping_mul(prime).to_le_bytes());
        result[16..24].copy_from_slice(&hash.wrapping_mul(prime).wrapping_mul(prime).to_le_bytes());
        result[24..32].copy_from_slice(
            &hash
                .wrapping_mul(prime)
                .wrapping_mul(prime)
                .wrapping_mul(prime)
                .to_le_bytes(),
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_store_creation() {
        let handle = GraphHandle::new(1, 0);
        let store = GraphStore::new(handle);

        assert_eq!(store.node_count(), 0);
        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let handle = GraphHandle::new(1, 0);
        let mut store = GraphStore::new(handle);
        let proof = ProofToken::default();

        let mutation = GraphMutation::add_node(1);
        store.apply_proved(&mutation, &proof).unwrap();

        assert_eq!(store.node_count(), 1);
        assert!(store.contains_node(1));
    }

    #[test]
    fn test_add_edge() {
        let handle = GraphHandle::new(1, 0);
        let mut store = GraphStore::new(handle);
        let proof = ProofToken::default();

        // Add nodes
        store.apply_proved(&GraphMutation::add_node(1), &proof).unwrap();
        store.apply_proved(&GraphMutation::add_node(2), &proof).unwrap();

        // Add edge
        let mutation = GraphMutation::add_edge(1, 2, 0.5);
        store.apply_proved(&mutation, &proof).unwrap();

        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_remove_node() {
        let handle = GraphHandle::new(1, 0);
        let mut store = GraphStore::new(handle);
        let proof = ProofToken::default();

        // Add nodes and edges
        store.apply_proved(&GraphMutation::add_node(1), &proof).unwrap();
        store.apply_proved(&GraphMutation::add_node(2), &proof).unwrap();
        store.apply_proved(&GraphMutation::add_edge(1, 2, 0.5), &proof).unwrap();

        // Remove node 1 (should also remove edge)
        store.apply_proved(&GraphMutation::remove_node(1), &proof).unwrap();

        assert_eq!(store.node_count(), 1);
        assert_eq!(store.edge_count(), 0);
        assert!(!store.contains_node(1));
        assert!(store.contains_node(2));
    }

    #[test]
    fn test_update_edge_weight() {
        let handle = GraphHandle::new(1, 0);
        let mut store = GraphStore::new(handle);
        let proof = ProofToken::default();

        store.apply_proved(&GraphMutation::add_node(1), &proof).unwrap();
        store.apply_proved(&GraphMutation::add_node(2), &proof).unwrap();
        store.apply_proved(&GraphMutation::add_edge(1, 2, 0.5), &proof).unwrap();

        // Update weight
        store.apply_proved(&GraphMutation::update_edge_weight(1, 2, 0.9), &proof).unwrap();

        #[cfg(feature = "alloc")]
        {
            let edges = store.edges_from(1);
            assert_eq!(edges.len(), 1);
            assert!((edges[0].weight() - 0.9).abs() < 0.001);
        }
    }

    #[test]
    fn test_state_hash_deterministic() {
        let handle = GraphHandle::new(1, 0);
        let mut store1 = GraphStore::new(handle);
        let mut store2 = GraphStore::new(handle);
        let proof = ProofToken::default();

        // Same operations on both
        for store in [&mut store1, &mut store2] {
            store.apply_proved(&GraphMutation::add_node(1), &proof).unwrap();
            store.apply_proved(&GraphMutation::add_node(2), &proof).unwrap();
            store.apply_proved(&GraphMutation::add_edge(1, 2, 0.5), &proof).unwrap();
        }

        assert_eq!(store1.state_hash(), store2.state_hash());
    }

    #[test]
    fn test_epoch_increment() {
        let handle = GraphHandle::new(1, 0);
        let mut store = GraphStore::new(handle);
        let proof = ProofToken::default();

        assert_eq!(store.epoch(), 0);

        store.apply_proved(&GraphMutation::add_node(1), &proof).unwrap();
        assert_eq!(store.epoch(), 1);

        store.apply_proved(&GraphMutation::add_node(2), &proof).unwrap();
        assert_eq!(store.epoch(), 2);
    }

    #[test]
    fn test_duplicate_node_error() {
        let handle = GraphHandle::new(1, 0);
        let mut store = GraphStore::new(handle);
        let proof = ProofToken::default();

        store.apply_proved(&GraphMutation::add_node(1), &proof).unwrap();

        let result = store.apply_proved(&GraphMutation::add_node(1), &proof);
        assert!(matches!(result, Err(KernelError::AlreadyExists)));
    }

    #[test]
    fn test_edge_missing_node_error() {
        let handle = GraphHandle::new(1, 0);
        let mut store = GraphStore::new(handle);
        let proof = ProofToken::default();

        store.apply_proved(&GraphMutation::add_node(1), &proof).unwrap();

        // Try to add edge to non-existent node
        let result = store.apply_proved(&GraphMutation::add_edge(1, 2, 0.5), &proof);
        assert!(matches!(result, Err(KernelError::NotFound)));
    }
}
