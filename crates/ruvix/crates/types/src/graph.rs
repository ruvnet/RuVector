//! Graph store types for kernel-resident graph operations.
//!
//! The kernel maintains graph stores as first-class objects. All mutations
//! are proof-gated via the `graph_apply_proved` syscall.

use crate::handle::Handle;

/// Handle to a kernel-resident graph store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GraphHandle(pub Handle);

impl GraphHandle {
    /// Creates a new graph handle.
    #[inline]
    #[must_use]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self(Handle::new(id, generation))
    }

    /// Creates a null (invalid) graph handle.
    #[inline]
    #[must_use]
    pub const fn null() -> Self {
        Self(Handle::null())
    }

    /// Checks if this handle is null.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the raw handle.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> Handle {
        self.0
    }
}

impl Default for GraphHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Kind of graph mutation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GraphMutationKind {
    /// Add a new node to the graph.
    AddNode = 0,

    /// Remove a node and all its edges.
    RemoveNode = 1,

    /// Add a new edge between two nodes.
    AddEdge = 2,

    /// Remove an edge between two nodes.
    RemoveEdge = 3,

    /// Update the weight of an existing edge.
    UpdateEdgeWeight = 4,

    /// Update node metadata.
    UpdateNodeMeta = 5,
}

impl GraphMutationKind {
    /// Returns the mutation kind as a string.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::AddNode => "AddNode",
            Self::RemoveNode => "RemoveNode",
            Self::AddEdge => "AddEdge",
            Self::RemoveEdge => "RemoveEdge",
            Self::UpdateEdgeWeight => "UpdateEdgeWeight",
            Self::UpdateNodeMeta => "UpdateNodeMeta",
        }
    }

    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::AddNode),
            1 => Some(Self::RemoveNode),
            2 => Some(Self::AddEdge),
            3 => Some(Self::RemoveEdge),
            4 => Some(Self::UpdateEdgeWeight),
            5 => Some(Self::UpdateNodeMeta),
            _ => None,
        }
    }
}

/// A graph mutation request.
///
/// Graph mutations are passed to `graph_apply_proved` and require
/// a valid proof token to execute.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GraphMutation {
    /// The kind of mutation.
    pub kind: GraphMutationKind,

    /// Source node ID (for edge operations) or target node ID.
    pub node_a: u64,

    /// Destination node ID (for edge operations).
    pub node_b: u64,

    /// Edge weight (for AddEdge and UpdateEdgeWeight).
    /// Represented as fixed-point: weight * 10000.
    pub weight_fp: i32,

    /// Partition ID hint (for coherence routing).
    pub partition_hint: u32,
}

impl GraphMutation {
    /// Creates an AddNode mutation.
    #[inline]
    #[must_use]
    pub const fn add_node(node_id: u64) -> Self {
        Self {
            kind: GraphMutationKind::AddNode,
            node_a: node_id,
            node_b: 0,
            weight_fp: 0,
            partition_hint: 0,
        }
    }

    /// Creates a RemoveNode mutation.
    #[inline]
    #[must_use]
    pub const fn remove_node(node_id: u64) -> Self {
        Self {
            kind: GraphMutationKind::RemoveNode,
            node_a: node_id,
            node_b: 0,
            weight_fp: 0,
            partition_hint: 0,
        }
    }

    /// Creates an AddEdge mutation.
    #[inline]
    #[must_use]
    pub const fn add_edge(from: u64, to: u64, weight: f32) -> Self {
        Self {
            kind: GraphMutationKind::AddEdge,
            node_a: from,
            node_b: to,
            weight_fp: (weight * 10000.0) as i32,
            partition_hint: 0,
        }
    }

    /// Creates a RemoveEdge mutation.
    #[inline]
    #[must_use]
    pub const fn remove_edge(from: u64, to: u64) -> Self {
        Self {
            kind: GraphMutationKind::RemoveEdge,
            node_a: from,
            node_b: to,
            weight_fp: 0,
            partition_hint: 0,
        }
    }

    /// Creates an UpdateEdgeWeight mutation.
    #[inline]
    #[must_use]
    pub const fn update_edge_weight(from: u64, to: u64, weight: f32) -> Self {
        Self {
            kind: GraphMutationKind::UpdateEdgeWeight,
            node_a: from,
            node_b: to,
            weight_fp: (weight * 10000.0) as i32,
            partition_hint: 0,
        }
    }

    /// Returns the weight as a float.
    #[inline]
    #[must_use]
    pub fn weight(&self) -> f32 {
        self.weight_fp as f32 / 10000.0
    }

    /// Sets a partition hint for coherence-aware routing.
    #[inline]
    #[must_use]
    pub const fn with_partition_hint(mut self, partition_id: u32) -> Self {
        self.partition_hint = partition_id;
        self
    }
}

impl Default for GraphMutation {
    fn default() -> Self {
        Self {
            kind: GraphMutationKind::AddNode,
            node_a: 0,
            node_b: 0,
            weight_fp: 0,
            partition_hint: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_handle() {
        let h = GraphHandle::new(42, 7);
        assert!(!h.is_null());
        assert_eq!(h.raw().id, 42);
    }

    #[test]
    fn test_graph_mutation_add_edge() {
        let mutation = GraphMutation::add_edge(1, 2, 0.75);
        assert_eq!(mutation.kind, GraphMutationKind::AddEdge);
        assert_eq!(mutation.node_a, 1);
        assert_eq!(mutation.node_b, 2);
        assert!((mutation.weight() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_graph_mutation_with_partition() {
        let mutation = GraphMutation::add_node(100).with_partition_hint(5);
        assert_eq!(mutation.partition_hint, 5);
    }
}
