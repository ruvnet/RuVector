//! Fixed-size coherence graph for partition communication topology.
//!
//! `CoherenceGraph` stores partition nodes and weighted edges in a
//! fixed-capacity adjacency structure with zero heap allocation,
//! suitable for `no_std` microhypervisor use.

use rvm_types::PartitionId;

/// Index into the node array.
pub type NodeIdx = u16;

/// Index into the edge array.
pub type EdgeIdx = u16;

/// Sentinel value indicating an unused slot.
const INVALID: u16 = u16::MAX;

/// A node in the coherence graph, representing a partition.
#[derive(Debug, Clone, Copy)]
struct Node {
    /// The partition this node represents, or `None` if the slot is free.
    partition: Option<PartitionId>,
    /// Index of the first outgoing edge in the edge array (linked list head).
    first_edge: u16,
}

impl Node {
    const EMPTY: Self = Self {
        partition: None,
        first_edge: INVALID,
    };
}

/// A directed weighted edge in the coherence graph.
#[derive(Debug, Clone, Copy)]
struct Edge {
    /// Source node index.
    from: NodeIdx,
    /// Destination node index.
    to: NodeIdx,
    /// Edge weight (communication volume, decayed per epoch).
    weight: u64,
    /// Next edge in the adjacency list for the `from` node.
    next_from: u16,
    /// Whether this slot is in use.
    active: bool,
}

impl Edge {
    const EMPTY: Self = Self {
        from: 0,
        to: 0,
        weight: 0,
        next_from: INVALID,
        active: false,
    };
}

/// Maximum number of slots probed in the `id_to_node` open-addressing
/// index before falling back to a full linear scan.
const PROBE_LEN: usize = 8;

/// Slot state for the open-addressed `id_to_node` index.
///
/// Distinguishing `Empty` (never used) from `Tombstone` (deleted) lets
/// lookups prove absence early: a probe chain that reaches an `Empty`
/// slot cannot have skipped a live entry, so the full-scan fallback is
/// only needed when the probe window is exhausted inconclusively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexSlot {
    /// Slot has never held an entry.
    Empty,
    /// Slot previously held an entry that was removed; probe chains
    /// continue past it.
    Tombstone,
    /// Slot holds the node index for some partition.
    Occupied(u8),
}

/// Result type for graph operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphError {
    /// No free node slots available.
    NodeCapacityExhausted,
    /// No free edge slots available.
    EdgeCapacityExhausted,
    /// The specified node was not found.
    NodeNotFound,
    /// The specified edge was not found.
    EdgeNotFound,
    /// A node for this partition already exists.
    DuplicateNode,
}

/// Fixed-size coherence graph.
///
/// `MAX_NODES` bounds the number of partition nodes, and `MAX_EDGES`
/// bounds the number of directed communication edges. Both are
/// compile-time constants to enable fully stack-allocated operation.
/// Size of the partition-ID-to-node index. 256 is sufficient since
/// `MAX_NODES` is typically 32 and partition IDs are bounded by VMID width.
const NODE_INDEX_SIZE: usize = 256;

/// Maximum dimension of the adjacency matrix (matches typical `MAX_NODES`).
/// Kept as a separate constant so the matrix size is fixed regardless of
/// the generic `MAX_NODES` parameter (which must be <= this value).
const ADJ_DIM: usize = 32;

/// Compile-time guard: `id_to_node` stores node indices as `u8`, so the
/// adjacency dimension (which bounds `MAX_NODES`) must never exceed 256.
const _ASSERT_ADJ_DIM_FITS_U8_INDEX: () = assert!(
    ADJ_DIM <= 256,
    "ADJ_DIM must be <= 256: id_to_node stores node indices as u8"
);

/// Stack-allocated coherence graph tracking inter-partition communication weights.
///
/// # Compile-time invariants
///
/// `MAX_NODES` must be at most `ADJ_DIM` (32, the fixed adjacency-matrix
/// dimension) and at most 256 (the `u8` width of the node index).
/// Violating instantiations are rejected at compile time:
///
/// ```compile_fail
/// use rvm_coherence::graph::CoherenceGraph;
/// // 64 nodes exceeds the 32x32 adjacency matrix: fails to compile.
/// let g = CoherenceGraph::<64, 256>::new();
/// ```
pub struct CoherenceGraph<const MAX_NODES: usize, const MAX_EDGES: usize> {
    nodes: [Node; MAX_NODES],
    edges: [Edge; MAX_EDGES],
    /// Open-addressed lookup: maps `PartitionId % NODE_INDEX_SIZE` (with
    /// bounded linear probing) to node index. Enables O(1) expected
    /// `find_node` instead of O(MAX_NODES) linear scan.
    id_to_node: [IndexSlot; NODE_INDEX_SIZE],
    /// Adjacency matrix: `adj_matrix[from][to]` holds the sum of edge
    /// weights from node `from` to node `to`.  Provides O(1)
    /// `edge_weight_between` lookups instead of O(E) scans.
    adj_matrix: [[u64; ADJ_DIM]; ADJ_DIM],
    /// Cached per-node total outgoing edge weight.
    cached_outgoing: [u64; ADJ_DIM],
    /// Cached per-node total incoming edge weight.
    cached_incoming: [u64; ADJ_DIM],
    node_count: u16,
    edge_count: u16,
}

impl<const MAX_NODES: usize, const MAX_EDGES: usize> CoherenceGraph<MAX_NODES, MAX_EDGES> {
    /// Compile-time assertion: `MAX_NODES` must fit the fixed
    /// `ADJ_DIM x ADJ_DIM` adjacency matrix. Combined with
    /// `_ASSERT_ADJ_DIM_FITS_U8_INDEX` (`ADJ_DIM <= 256`), this also
    /// guarantees node indices fit the `u8` entries of `id_to_node`.
    /// Referenced from [`Self::new`] so that violating instantiations
    /// fail compilation instead of panicking at the 33rd node (same
    /// pattern as `rvm-witness` `_ASSERT_N_NONZERO` and `rvm-partition`
    /// `_CAPACITY_IS_POWER_OF_TWO`).
    const _ASSERT_MAX_NODES_FITS: () = assert!(
        MAX_NODES <= ADJ_DIM,
        "CoherenceGraph MAX_NODES must be <= ADJ_DIM (32)"
    );

    /// Create a new empty coherence graph.
    ///
    /// # Compile-time invariant
    ///
    /// `MAX_NODES` must be `<= ADJ_DIM` (32) and `<= 256`. Instantiating
    /// a graph that violates this is a compile-time error.
    #[must_use]
    pub const fn new() -> Self {
        // Reference the const so the compile-time check fires for every
        // instantiation of this graph.
        let () = Self::_ASSERT_MAX_NODES_FITS;
        Self {
            nodes: [Node::EMPTY; MAX_NODES],
            edges: [Edge::EMPTY; MAX_EDGES],
            id_to_node: [IndexSlot::Empty; NODE_INDEX_SIZE],
            adj_matrix: [[0u64; ADJ_DIM]; ADJ_DIM],
            cached_outgoing: [0u64; ADJ_DIM],
            cached_incoming: [0u64; ADJ_DIM],
            node_count: 0,
            edge_count: 0,
        }
    }

    /// Number of active nodes.
    #[must_use]
    pub const fn node_count(&self) -> u16 {
        self.node_count
    }

    /// Number of active edges.
    #[must_use]
    pub const fn edge_count(&self) -> u16 {
        self.edge_count
    }

    /// Add a partition node to the graph. Returns the node index.
    pub fn add_node(&mut self, partition_id: PartitionId) -> Result<NodeIdx, GraphError> {
        // Check for duplicate
        if self.find_node(partition_id).is_some() {
            return Err(GraphError::DuplicateNode);
        }

        // Find a free slot
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if node.partition.is_none() {
                node.partition = Some(partition_id);
                node.first_edge = INVALID;
                self.node_count += 1;
                // Populate the lookup index with bounded linear probing:
                // place the entry in the first non-occupied slot of the
                // probe window. If the window is fully occupied the node
                // is simply not indexed; `find_node` falls back to the
                // linear scan for it.
                let hash = (partition_id.as_u32() as usize) % NODE_INDEX_SIZE;
                for k in 0..PROBE_LEN {
                    let slot = (hash + k) % NODE_INDEX_SIZE;
                    if !matches!(self.id_to_node[slot], IndexSlot::Occupied(_)) {
                        self.id_to_node[slot] = IndexSlot::Occupied(i as u8);
                        break;
                    }
                }
                return Ok(i as NodeIdx);
            }
        }
        Err(GraphError::NodeCapacityExhausted)
    }

    /// Remove a partition node and all its incident edges.
    pub fn remove_node(&mut self, partition_id: PartitionId) -> Result<(), GraphError> {
        let idx = self.find_node(partition_id).ok_or(GraphError::NodeNotFound)?;

        // Remove all edges where this node is source or destination.
        // remove_edge_by_index maintains adj_matrix and cached weights.
        for i in 0..MAX_EDGES {
            if self.edges[i].active
                && (self.edges[i].from == idx || self.edges[i].to == idx)
            {
                self.remove_edge_by_index(i as EdgeIdx);
            }
        }

        // Tombstone the lookup index entry. The entry, if indexed, lives
        // in the probe window; an `Empty` slot terminates the search since
        // entries are always placed at the first non-occupied slot and
        // slots never revert to `Empty`.
        let hash = (partition_id.as_u32() as usize) % NODE_INDEX_SIZE;
        for k in 0..PROBE_LEN {
            let slot = (hash + k) % NODE_INDEX_SIZE;
            match self.id_to_node[slot] {
                IndexSlot::Occupied(stored) if stored as usize == idx as usize => {
                    self.id_to_node[slot] = IndexSlot::Tombstone;
                    break;
                }
                IndexSlot::Empty => break,
                _ => {}
            }
        }
        self.nodes[idx as usize].partition = None;
        self.nodes[idx as usize].first_edge = INVALID;

        // Clear adjacency matrix row and column for this node, and
        // reset cached weights (should already be zero after edge removal,
        // but clear explicitly for safety).
        let ni = idx as usize;
        for j in 0..ADJ_DIM {
            self.adj_matrix[ni][j] = 0;
            self.adj_matrix[j][ni] = 0;
        }
        self.cached_outgoing[ni] = 0;
        self.cached_incoming[ni] = 0;

        self.node_count = self.node_count.saturating_sub(1);
        Ok(())
    }

    /// Add a directed weighted edge from one node to another. Returns the edge index.
    pub fn add_edge(
        &mut self,
        from: PartitionId,
        to: PartitionId,
        weight: u64,
    ) -> Result<EdgeIdx, GraphError> {
        let from_idx = self.find_node(from).ok_or(GraphError::NodeNotFound)?;
        let to_idx = self.find_node(to).ok_or(GraphError::NodeNotFound)?;

        // Find a free edge slot
        let edge_idx = self.alloc_edge()?;

        let old_head = self.nodes[from_idx as usize].first_edge;
        self.edges[edge_idx as usize] = Edge {
            from: from_idx,
            to: to_idx,
            weight,
            next_from: old_head,
            active: true,
        };
        self.nodes[from_idx as usize].first_edge = edge_idx;
        self.edge_count += 1;

        // Update adjacency matrix and cached weights.
        let fi = from_idx as usize;
        let ti = to_idx as usize;
        self.adj_matrix[fi][ti] = self.adj_matrix[fi][ti].saturating_add(weight);
        self.cached_outgoing[fi] = self.cached_outgoing[fi].saturating_add(weight);
        self.cached_incoming[ti] = self.cached_incoming[ti].saturating_add(weight);

        Ok(edge_idx)
    }

    /// Update the weight of an edge by adding `delta` (saturating).
    pub fn update_weight(&mut self, edge_id: EdgeIdx, delta: i64) -> Result<(), GraphError> {
        let idx = edge_id as usize;
        if idx >= MAX_EDGES || !self.edges[idx].active {
            return Err(GraphError::EdgeNotFound);
        }
        let old_weight = self.edges[idx].weight;
        let new_weight = if delta >= 0 {
            old_weight.saturating_add(delta as u64)
        } else {
            old_weight.saturating_sub(delta.unsigned_abs())
        };
        self.edges[idx].weight = new_weight;

        // Update adjacency matrix and cached weights.
        let fi = self.edges[idx].from as usize;
        let ti = self.edges[idx].to as usize;
        // Adjust: remove old, add new.
        self.adj_matrix[fi][ti] = self.adj_matrix[fi][ti]
            .saturating_sub(old_weight)
            .saturating_add(new_weight);
        self.cached_outgoing[fi] = self.cached_outgoing[fi]
            .saturating_sub(old_weight)
            .saturating_add(new_weight);
        self.cached_incoming[ti] = self.cached_incoming[ti]
            .saturating_sub(old_weight)
            .saturating_add(new_weight);

        Ok(())
    }

    /// Get the weight of an edge by index.
    #[must_use]
    pub fn edge_weight(&self, edge_id: EdgeIdx) -> Option<u64> {
        let idx = edge_id as usize;
        if idx < MAX_EDGES && self.edges[idx].active {
            Some(self.edges[idx].weight)
        } else {
            None
        }
    }

    /// Get the source and destination partition IDs for an edge.
    #[must_use]
    pub fn edge_endpoints(&self, edge_id: EdgeIdx) -> Option<(PartitionId, PartitionId)> {
        let idx = edge_id as usize;
        if idx >= MAX_EDGES || !self.edges[idx].active {
            return None;
        }
        let from_pid = self.nodes[self.edges[idx].from as usize].partition?;
        let to_pid = self.nodes[self.edges[idx].to as usize].partition?;
        Some((from_pid, to_pid))
    }

    /// Iterate over neighbor node indices of a given partition.
    ///
    /// Returns `(neighbor_node_idx, edge_weight)` pairs for all outgoing
    /// edges from the given partition.
    pub fn neighbors(
        &self,
        partition_id: PartitionId,
    ) -> Option<NeighborIter<'_, MAX_NODES, MAX_EDGES>> {
        let idx = self.find_node(partition_id)?;
        Some(NeighborIter {
            graph: self,
            current_edge: self.nodes[idx as usize].first_edge,
        })
    }

    /// Sum of all incident edge weights for a partition (outgoing + incoming).
    ///
    /// Uses cached per-node weight sums for O(1) lookup instead of scanning
    /// all edges.
    #[must_use]
    pub fn total_weight(&self, partition_id: PartitionId) -> u64 {
        match self.find_node(partition_id) {
            Some(idx) => {
                let i = idx as usize;
                self.cached_outgoing[i].saturating_add(self.cached_incoming[i])
            }
            None => 0,
        }
    }

    /// Sum of internal edge weights (edges where both endpoints are the
    /// given partition or edges between two specific partitions).
    ///
    /// For coherence scoring, "internal" edges are those where both
    /// endpoints belong to the same logical group. In the single-partition
    /// case, this is self-loops. For multi-partition queries, callers
    /// should use `edge_weight_between`.
    ///
    /// O(1): the self-loop weight sum is maintained at
    /// `adj_matrix[idx][idx]` by `add_edge`, `update_weight`,
    /// `decay_weights`, and `remove_edge_by_index`.
    #[must_use]
    pub fn internal_weight(&self, partition_id: PartitionId) -> u64 {
        match self.find_node(partition_id) {
            Some(idx) => self.adj_matrix[idx as usize][idx as usize],
            None => 0,
        }
    }

    /// Sum of edge weights between two specific partitions (in either direction).
    ///
    /// Uses the adjacency matrix for O(1) lookup instead of scanning all edges.
    #[must_use]
    pub fn edge_weight_between(&self, a: PartitionId, b: PartitionId) -> u64 {
        let a_idx = match self.find_node(a) {
            Some(i) => i as usize,
            None => return 0,
        };
        let b_idx = match self.find_node(b) {
            Some(i) => i as usize,
            None => return 0,
        };
        self.adj_matrix[a_idx][b_idx].saturating_add(self.adj_matrix[b_idx][a_idx])
    }

    /// Find the edge index of a directed edge from `from` to `to`.
    ///
    /// Uses the adjacency matrix for a fast O(1) existence check, then walks
    /// the source node's adjacency list (O(out-degree)) to find the edge index.
    /// Returns `None` if no such edge exists.
    #[must_use]
    pub fn find_directed_edge(&self, from: PartitionId, to: PartitionId) -> Option<EdgeIdx> {
        let from_idx = self.find_node(from)?;
        let to_idx = self.find_node(to)?;

        // Fast path: adjacency matrix says no weight => no edge.
        if self.adj_matrix[from_idx as usize][to_idx as usize] == 0 {
            return None;
        }

        // Walk the source node's outgoing edge list.
        let mut cur = self.nodes[from_idx as usize].first_edge;
        while cur != INVALID {
            let ci = cur as usize;
            if ci >= MAX_EDGES {
                break;
            }
            if self.edges[ci].active && self.edges[ci].to == to_idx {
                return Some(cur);
            }
            cur = self.edges[ci].next_from;
        }
        None
    }

    /// Get the node index for a partition, or `None` if not present.
    ///
    /// O(1) expected via the open-addressed index with bounded linear
    /// probing. A probe chain that reaches a never-used `Empty` slot
    /// proves absence without scanning; the full linear scan only runs
    /// when the probe window is exhausted inconclusively (all slots
    /// occupied by colliding entries or tombstones).
    #[inline]
    #[must_use]
    pub fn find_node(&self, partition_id: PartitionId) -> Option<NodeIdx> {
        let hash = (partition_id.as_u32() as usize) % NODE_INDEX_SIZE;
        for k in 0..PROBE_LEN {
            let slot = (hash + k) % NODE_INDEX_SIZE;
            match self.id_to_node[slot] {
                IndexSlot::Occupied(idx) => {
                    let i = idx as usize;
                    if i < MAX_NODES && self.nodes[i].partition == Some(partition_id) {
                        return Some(i as NodeIdx);
                    }
                    // Collision: keep probing.
                }
                // Never-used slot: the entry would have been placed here
                // (or earlier) on insert, so the partition is absent.
                IndexSlot::Empty => return None,
                // Deleted slot: probe chains continue past it.
                IndexSlot::Tombstone => {}
            }
        }
        // Inconclusive probe chain: fall back to a linear scan (covers
        // nodes that could not be indexed because their window was full).
        for (i, node) in self.nodes.iter().enumerate() {
            if node.partition == Some(partition_id) {
                return Some(i as NodeIdx);
            }
        }
        None
    }

    /// Get the partition ID for a node index.
    #[must_use]
    pub fn partition_at(&self, idx: NodeIdx) -> Option<PartitionId> {
        if (idx as usize) < MAX_NODES {
            self.nodes[idx as usize].partition
        } else {
            None
        }
    }

    /// Iterate over all active node indices and their partition IDs.
    pub fn active_nodes(&self) -> impl Iterator<Item = (NodeIdx, PartitionId)> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.partition.map(|pid| (i as NodeIdx, pid)))
    }

    /// Iterate over all active edges as `(edge_idx, from_node, to_node, weight)`.
    pub fn active_edges(&self) -> impl Iterator<Item = (EdgeIdx, NodeIdx, NodeIdx, u64)> + '_ {
        self.edges
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if e.active {
                    Some((i as EdgeIdx, e.from, e.to, e.weight))
                } else {
                    None
                }
            })
    }

    /// Iterate over node indices that have a nonzero-weight edge into `to`.
    ///
    /// Reads the adjacency-matrix column in O(MAX_NODES) instead of
    /// scanning all edges (O(MAX_EDGES)). Nodes connected to `to` only by
    /// zero-weight edges are not yielded.
    pub fn in_neighbors_of(&self, to: NodeIdx) -> impl Iterator<Item = NodeIdx> + '_ {
        let ti = to as usize;
        (0..MAX_NODES).filter_map(move |j| {
            if ti < ADJ_DIM
                && self.adj_matrix[j][ti] > 0
                && self.nodes[j].partition.is_some()
            {
                Some(j as NodeIdx)
            } else {
                None
            }
        })
    }

    /// Decay all edge weights by the given percentage (in basis points).
    ///
    /// `decay_bp` = 1000 means decay by 10% per call. Edges whose weight
    /// falls to zero are automatically pruned. Returns the number of
    /// edges pruned.
    ///
    /// Call once per epoch to prevent stale communication patterns from
    /// dominating the coherence graph.
    pub fn decay_weights(&mut self, decay_bp: u16) -> u16 {
        let mut pruned = 0u16;
        let factor = 10_000u64.saturating_sub(decay_bp as u64);
        for i in 0..MAX_EDGES {
            if !self.edges[i].active {
                continue;
            }
            let old_w = self.edges[i].weight;
            let new_w = old_w.saturating_mul(factor) / 10_000;
            if new_w == 0 {
                // remove_edge_by_index handles adj_matrix and cached weights.
                self.remove_edge_by_index(i as EdgeIdx);
                pruned += 1;
            } else {
                // Update the weight directly and adjust caches.
                let fi = self.edges[i].from as usize;
                let ti = self.edges[i].to as usize;
                let diff = old_w - new_w;
                self.edges[i].weight = new_w;
                self.adj_matrix[fi][ti] = self.adj_matrix[fi][ti].saturating_sub(diff);
                self.cached_outgoing[fi] = self.cached_outgoing[fi].saturating_sub(diff);
                self.cached_incoming[ti] = self.cached_incoming[ti].saturating_sub(diff);
            }
        }
        pruned
    }

    /// Remove **all** directed edges from `from` to `to`, returning the
    /// total weight removed.
    ///
    /// Used by the split executor to re-home a neighbor's edges across
    /// a mincut boundary (remove from the source partition, re-add to
    /// the child). O(MAX_EDGES) — epoch/split path only, not hot path.
    pub fn remove_directed_edges(
        &mut self,
        from: PartitionId,
        to: PartitionId,
    ) -> Result<u64, GraphError> {
        let from_idx = self.find_node(from).ok_or(GraphError::NodeNotFound)?;
        let to_idx = self.find_node(to).ok_or(GraphError::NodeNotFound)?;

        let mut total = 0u64;
        for i in 0..MAX_EDGES {
            if self.edges[i].active
                && self.edges[i].from == from_idx
                && self.edges[i].to == to_idx
            {
                total = total.saturating_add(self.edges[i].weight);
                self.remove_edge_by_index(i as EdgeIdx);
            }
        }
        Ok(total)
    }

    /// Allocate a free edge slot.
    fn alloc_edge(&self) -> Result<EdgeIdx, GraphError> {
        for (i, e) in self.edges.iter().enumerate() {
            if !e.active {
                return Ok(i as EdgeIdx);
            }
        }
        Err(GraphError::EdgeCapacityExhausted)
    }

    /// Remove an edge by its index, repairing the adjacency linked list.
    fn remove_edge_by_index(&mut self, edge_idx: EdgeIdx) {
        let idx = edge_idx as usize;
        if idx >= MAX_EDGES || !self.edges[idx].active {
            return;
        }

        let from_node = self.edges[idx].from as usize;
        let to_node = self.edges[idx].to as usize;
        let weight = self.edges[idx].weight;

        // Update adjacency matrix and cached weights before removing.
        self.adj_matrix[from_node][to_node] =
            self.adj_matrix[from_node][to_node].saturating_sub(weight);
        self.cached_outgoing[from_node] =
            self.cached_outgoing[from_node].saturating_sub(weight);
        self.cached_incoming[to_node] =
            self.cached_incoming[to_node].saturating_sub(weight);

        // Repair the linked list: remove this edge from the from-node's list
        if self.nodes[from_node].first_edge == edge_idx {
            self.nodes[from_node].first_edge = self.edges[idx].next_from;
        } else {
            // Walk the list to find the predecessor
            let mut cur = self.nodes[from_node].first_edge;
            while cur != INVALID {
                let cur_idx = cur as usize;
                if self.edges[cur_idx].next_from == edge_idx {
                    self.edges[cur_idx].next_from = self.edges[idx].next_from;
                    break;
                }
                cur = self.edges[cur_idx].next_from;
            }
        }

        self.edges[idx].active = false;
        self.edge_count = self.edge_count.saturating_sub(1);
    }
}

/// Iterator over the neighbors of a node.
pub struct NeighborIter<'a, const MAX_NODES: usize, const MAX_EDGES: usize> {
    graph: &'a CoherenceGraph<MAX_NODES, MAX_EDGES>,
    current_edge: u16,
}

impl<const MAX_NODES: usize, const MAX_EDGES: usize> Iterator
    for NeighborIter<'_, MAX_NODES, MAX_EDGES>
{
    /// `(neighbor_node_idx, edge_weight)`
    type Item = (NodeIdx, u64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_edge != INVALID {
            let idx = self.current_edge as usize;
            if idx >= MAX_EDGES {
                break;
            }
            let edge = &self.graph.edges[idx];
            self.current_edge = edge.next_from;
            if edge.active {
                return Some((edge.to, edge.weight));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(n: u32) -> PartitionId {
        PartitionId::new(n)
    }

    #[test]
    fn add_and_find_nodes() {
        let mut g = CoherenceGraph::<8, 16>::new();
        let n0 = g.add_node(pid(1)).unwrap();
        let n1 = g.add_node(pid(2)).unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.find_node(pid(1)), Some(n0));
        assert_eq!(g.find_node(pid(2)), Some(n1));
        assert_eq!(g.find_node(pid(99)), None);
    }

    #[test]
    fn duplicate_node_rejected() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        assert_eq!(g.add_node(pid(1)), Err(GraphError::DuplicateNode));
    }

    #[test]
    fn node_capacity_exhausted() {
        let mut g = CoherenceGraph::<2, 4>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        assert_eq!(g.add_node(pid(3)), Err(GraphError::NodeCapacityExhausted));
    }

    #[test]
    fn add_edge_and_query_weight() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        let e = g.add_edge(pid(1), pid(2), 100).unwrap();
        assert_eq!(g.edge_weight(e), Some(100));
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn update_weight_positive_and_negative() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        let e = g.add_edge(pid(1), pid(2), 100).unwrap();

        g.update_weight(e, 50).unwrap();
        assert_eq!(g.edge_weight(e), Some(150));

        g.update_weight(e, -200).unwrap();
        assert_eq!(g.edge_weight(e), Some(0)); // saturating
    }

    #[test]
    fn neighbors_iteration() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_node(pid(3)).unwrap();
        g.add_edge(pid(1), pid(2), 10).unwrap();
        g.add_edge(pid(1), pid(3), 20).unwrap();

        let mut count = 0u32;
        let mut total = 0u64;
        for (_idx, w) in g.neighbors(pid(1)).unwrap() {
            count += 1;
            total += w;
        }
        assert_eq!(count, 2);
        assert_eq!(total, 30);
    }

    #[test]
    fn total_weight_includes_incoming_and_outgoing() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_edge(pid(1), pid(2), 100).unwrap();
        g.add_edge(pid(2), pid(1), 50).unwrap();

        // pid(1): outgoing 100 + incoming 50
        assert_eq!(g.total_weight(pid(1)), 150);
    }

    #[test]
    fn remove_node_clears_edges() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_edge(pid(1), pid(2), 100).unwrap();
        g.add_edge(pid(2), pid(1), 50).unwrap();
        assert_eq!(g.edge_count(), 2);

        g.remove_node(pid(1)).unwrap();
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.find_node(pid(1)), None);
    }

    #[test]
    fn edge_weight_between_bidirectional() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_edge(pid(1), pid(2), 100).unwrap();
        g.add_edge(pid(2), pid(1), 50).unwrap();
        assert_eq!(g.edge_weight_between(pid(1), pid(2)), 150);
        assert_eq!(g.edge_weight_between(pid(2), pid(1)), 150);
    }

    #[test]
    fn edge_endpoints_retrieval() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        let e = g.add_edge(pid(1), pid(2), 42).unwrap();
        assert_eq!(g.edge_endpoints(e), Some((pid(1), pid(2))));
    }

    #[test]
    fn decay_weights_reduces_values() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_edge(pid(1), pid(2), 1000).unwrap();

        // 10% decay
        let pruned = g.decay_weights(1000);
        assert_eq!(pruned, 0);
        // 1000 * 0.9 = 900
        assert_eq!(g.edge_weight_between(pid(1), pid(2)), 900);
    }

    #[test]
    fn decay_prunes_zero_weight_edges() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_edge(pid(1), pid(2), 1).unwrap();
        assert_eq!(g.edge_count(), 1);

        // 50% decay on weight=1 → 0, should prune.
        let pruned = g.decay_weights(5000);
        assert_eq!(pruned, 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn decay_100_percent_prunes_all() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_edge(pid(1), pid(2), 500).unwrap();
        g.add_edge(pid(2), pid(1), 300).unwrap();

        let pruned = g.decay_weights(10_000); // 100% decay
        assert_eq!(pruned, 2);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn internal_weight_tracks_add_update_decay_remove() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();

        // Add: self-loop and an external edge.
        let self_loop = g.add_edge(pid(1), pid(1), 1000).unwrap();
        g.add_edge(pid(1), pid(2), 500).unwrap();
        assert_eq!(g.internal_weight(pid(1)), 1000);
        assert_eq!(g.internal_weight(pid(2)), 0);

        // A second self-loop accumulates.
        let self_loop2 = g.add_edge(pid(1), pid(1), 200).unwrap();
        assert_eq!(g.internal_weight(pid(1)), 1200);

        // Update: positive and negative deltas.
        g.update_weight(self_loop, 300).unwrap();
        assert_eq!(g.internal_weight(pid(1)), 1500);
        g.update_weight(self_loop, -800).unwrap();
        assert_eq!(g.internal_weight(pid(1)), 700);

        // Decay 10%: each self-loop decays individually.
        // self_loop: 500 -> 450, self_loop2: 200 -> 180.
        g.decay_weights(1000);
        assert_eq!(g.internal_weight(pid(1)), 630);

        // Decay that prunes: drive self_loop2 to zero via updates first.
        g.update_weight(self_loop2, -180).unwrap();
        assert_eq!(g.internal_weight(pid(1)), 450);
        // 100% decay prunes everything.
        g.decay_weights(10_000);
        assert_eq!(g.internal_weight(pid(1)), 0);

        // Re-add and remove the node entirely.
        g.add_edge(pid(1), pid(1), 77).unwrap();
        assert_eq!(g.internal_weight(pid(1)), 77);
        g.remove_node(pid(1)).unwrap();
        assert_eq!(g.internal_weight(pid(1)), 0);
    }

    #[test]
    fn find_node_with_index_collisions() {
        // pid(1) and pid(257) hash to the same id_to_node slot
        // (257 % 256 == 1). Bounded probing must resolve both.
        let mut g = CoherenceGraph::<8, 16>::new();
        let n0 = g.add_node(pid(1)).unwrap();
        let n1 = g.add_node(pid(257)).unwrap();
        let n2 = g.add_node(pid(513)).unwrap(); // also collides

        assert_eq!(g.find_node(pid(1)), Some(n0));
        assert_eq!(g.find_node(pid(257)), Some(n1));
        assert_eq!(g.find_node(pid(513)), Some(n2));
        // Colliding miss: hashes into the same chain but absent.
        assert_eq!(g.find_node(pid(769)), None);
    }

    #[test]
    fn remove_with_index_collisions_keeps_other_entries() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        let n1 = g.add_node(pid(257)).unwrap();

        // Remove the first-inserted collider; the evicted-chain entry
        // must still be found (tombstone is probed past).
        g.remove_node(pid(1)).unwrap();
        assert_eq!(g.find_node(pid(1)), None);
        assert_eq!(g.find_node(pid(257)), Some(n1));

        // Reinsert into the tombstoned chain and find both again.
        let n0 = g.add_node(pid(1)).unwrap();
        assert_eq!(g.find_node(pid(1)), Some(n0));
        assert_eq!(g.find_node(pid(257)), Some(n1));
    }

    #[test]
    fn in_neighbors_of_matches_incoming_edges() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_node(pid(3)).unwrap();
        g.add_edge(pid(2), pid(1), 10).unwrap();
        g.add_edge(pid(3), pid(1), 20).unwrap();
        g.add_edge(pid(1), pid(2), 30).unwrap(); // outgoing, not incoming

        let root = g.find_node(pid(1)).unwrap();
        let mut found = [false; 8];
        for j in g.in_neighbors_of(root) {
            found[j as usize] = true;
        }
        assert!(found[g.find_node(pid(2)).unwrap() as usize]);
        assert!(found[g.find_node(pid(3)).unwrap() as usize]);
        assert!(!found[root as usize]);
    }

    #[test]
    fn remove_directed_edges_returns_total_weight() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        // Two parallel directed edges 1->2, plus a reverse edge 2->1.
        g.add_edge(pid(1), pid(2), 100).unwrap();
        g.add_edge(pid(1), pid(2), 50).unwrap();
        g.add_edge(pid(2), pid(1), 30).unwrap();

        let removed = g.remove_directed_edges(pid(1), pid(2)).unwrap();
        assert_eq!(removed, 150);
        // Only the reverse edge remains.
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.edge_weight_between(pid(1), pid(2)), 30);
        assert_eq!(g.total_weight(pid(1)), 30);
        assert_eq!(g.total_weight(pid(2)), 30);
    }

    #[test]
    fn remove_directed_edges_missing_node_fails() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        assert_eq!(
            g.remove_directed_edges(pid(1), pid(99)),
            Err(GraphError::NodeNotFound)
        );
    }

    #[test]
    fn remove_directed_edges_none_present_is_zero() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        assert_eq!(g.remove_directed_edges(pid(1), pid(2)), Ok(0));
    }

    #[test]
    fn decay_zero_is_noop() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_edge(pid(1), pid(2), 1000).unwrap();

        let pruned = g.decay_weights(0);
        assert_eq!(pruned, 0);
        assert_eq!(g.edge_weight_between(pid(1), pid(2)), 1000);
    }
}
