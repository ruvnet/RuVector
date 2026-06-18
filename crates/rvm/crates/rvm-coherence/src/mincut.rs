//! MinCut bridge: budgeted approximate minimum cut computation.
//!
//! This module provides a self-contained `no_std` approximate mincut
//! implementation for small graphs, with a hard time budget per
//! ADR-132 DC-2 (50 microseconds per epoch). If the budget is exceeded,
//! the last known cut is returned.
//!
//! The v1 algorithm uses a greedy heuristic inspired by Stoer-Wagner:
//! iteratively merge the most-connected pair of nodes until two
//! super-nodes remain. This is exact for small graphs and a good
//! approximation for larger ones, all without heap allocation.

use rvm_types::PartitionId;

use crate::graph::{CoherenceGraph, NodeIdx};

/// Maximum number of nodes supported by the mincut computation.
/// Kept small for the budgeted `no_std` implementation.
const MINCUT_MAX_NODES: usize = 32;

/// Result of a minimum cut computation.
#[derive(Debug, Clone)]
pub struct MinCutResult {
    /// Partition IDs in the "left" side of the cut.
    pub left: [Option<PartitionId>; MINCUT_MAX_NODES],
    /// Number of active entries in `left`.
    pub left_count: u16,
    /// Partition IDs in the "right" side of the cut.
    pub right: [Option<PartitionId>; MINCUT_MAX_NODES],
    /// Number of active entries in `right`.
    pub right_count: u16,
    /// Total weight of edges crossing the cut.
    pub cut_weight: u64,
    /// Whether the computation was completed within budget.
    pub within_budget: bool,
}

impl MinCutResult {
    /// Create an empty result.
    const fn empty() -> Self {
        Self {
            left: [None; MINCUT_MAX_NODES],
            left_count: 0,
            right: [None; MINCUT_MAX_NODES],
            right_count: 0,
            cut_weight: 0,
            within_budget: true,
        }
    }
}

/// Budgeted minimum cut bridge.
///
/// Wraps the approximate mincut algorithm with epoch tracking and
/// budget enforcement. When the budget is exceeded, the last known
/// cut result is reused.
pub struct MinCutBridge<const N: usize> {
    /// Last known cut result (reused when budget is exceeded).
    last_known_cut: MinCutResult,
    /// Current epoch counter.
    epoch: u32,
    /// Number of times the budget was exceeded.
    pub budget_exceeded_count: u32,
    /// Number of successful computations.
    pub compute_count: u32,
    /// Maximum iterations allowed per computation (budget proxy in `no_std`).
    max_iterations: u32,
    /// Reusable adjacency working buffer for Stoer-Wagner. Built directly
    /// from the graph each call (no separate copy) and consumed in place,
    /// avoiding ~16 KB of per-call stack traffic.
    scratch_adj: [[u64; MINCUT_MAX_NODES]; MINCUT_MAX_NODES],
    /// Reusable super-node group-membership scratch buffer.
    scratch_groups: [[bool; MINCUT_MAX_NODES]; MINCUT_MAX_NODES],
}

impl<const N: usize> MinCutBridge<N> {
    /// Create a new `MinCutBridge` with the given iteration budget.
    ///
    /// `max_iterations` controls the maximum number of merge steps
    /// per mincut computation. For a graph with `k` nodes, the
    /// Stoer-Wagner-like algorithm needs `k-1` merge steps, so set
    /// this to at least the expected node count.
    #[must_use]
    pub const fn new(max_iterations: u32) -> Self {
        Self {
            last_known_cut: MinCutResult::empty(),
            epoch: 0,
            budget_exceeded_count: 0,
            compute_count: 0,
            max_iterations,
            scratch_adj: [[0u64; MINCUT_MAX_NODES]; MINCUT_MAX_NODES],
            scratch_groups: [[false; MINCUT_MAX_NODES]; MINCUT_MAX_NODES],
        }
    }

    /// Current epoch.
    #[must_use]
    pub const fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Advance the epoch counter.
    pub fn advance_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Compute the approximate minimum cut for the subgraph rooted at
    /// `partition_id` and its neighbors.
    ///
    /// If the computation exceeds the iteration budget, the last known
    /// cut is returned and `within_budget` is set to `false`.
    pub fn find_min_cut<const MN: usize, const ME: usize>(
        &mut self,
        graph: &CoherenceGraph<MN, ME>,
        partition_id: PartitionId,
    ) -> &MinCutResult {
        self.advance_epoch();

        // Collect the local subgraph: partition_id + its neighbors
        let mut sub_nodes: [Option<(NodeIdx, PartitionId)>; MINCUT_MAX_NODES] =
            [None; MINCUT_MAX_NODES];
        let mut sub_count = 0usize;
        // Membership bitmask over graph node indices (the graph enforces
        // MAX_NODES <= 32 at compile time, so indices fit in a u64).
        // Replaces the previous O(sub_count) duplicate scans.
        let mut present: u64 = 0;

        // Add the target partition
        if let Some(root_idx) = graph.find_node(partition_id) {
            sub_nodes[0] = Some((root_idx, partition_id));
            present |= 1u64 << root_idx;
            sub_count = 1;

            // Add neighbors
            if let Some(iter) = graph.neighbors(partition_id) {
                for (neighbor_idx, _weight) in iter {
                    if sub_count >= MINCUT_MAX_NODES {
                        break;
                    }
                    if present & (1u64 << neighbor_idx) != 0 {
                        continue;
                    }
                    if let Some(npid) = graph.partition_at(neighbor_idx) {
                        sub_nodes[sub_count] = Some((neighbor_idx, npid));
                        present |= 1u64 << neighbor_idx;
                        sub_count += 1;
                    }
                }
            }

            // Also add nodes that have incoming edges to partition_id,
            // derived from the adjacency-matrix column (O(MAX_NODES))
            // instead of scanning all active edges (O(MAX_EDGES)).
            for from in graph.in_neighbors_of(root_idx) {
                if sub_count >= MINCUT_MAX_NODES {
                    break;
                }
                if present & (1u64 << from) != 0 {
                    continue;
                }
                if let Some(fpid) = graph.partition_at(from) {
                    sub_nodes[sub_count] = Some((from, fpid));
                    present |= 1u64 << from;
                    sub_count += 1;
                }
            }
        }

        // Need at least 2 nodes for a meaningful cut
        if sub_count < 2 {
            self.last_known_cut = MinCutResult::empty();
            self.last_known_cut.within_budget = true;
            if sub_count == 1 {
                if let Some((_, pid)) = sub_nodes[0] {
                    self.last_known_cut.left[0] = Some(pid);
                    self.last_known_cut.left_count = 1;
                }
            }
            self.compute_count += 1;
            return &self.last_known_cut;
        }

        // Build the local adjacency weight matrix for the subgraph
        // directly into the reusable working buffer; Stoer-Wagner then
        // consumes it in place (no separate copy). Only the upper-left
        // sub_count x sub_count region is read, so the rest can stay stale.
        for i in 0..sub_count {
            self.scratch_adj[i][i] = 0;
            if let Some((_, pi)) = sub_nodes[i] {
                for j in (i + 1)..sub_count {
                    if let Some((_, pj)) = sub_nodes[j] {
                        // Sum edges in both directions for undirected treatment
                        let w = graph.edge_weight_between(pi, pj);
                        self.scratch_adj[i][j] = w;
                        self.scratch_adj[j][i] = w;
                    }
                }
            }
        }

        // Stoer-Wagner-like minimum cut on the local adjacency matrix,
        // writing the result directly into `last_known_cut` (avoids two
        // ~600 B MinCutResult copies per call).
        let within_budget = stoer_wagner_mincut(
            &mut self.scratch_adj,
            &mut self.scratch_groups,
            sub_count,
            &sub_nodes,
            self.max_iterations,
            &mut self.last_known_cut,
        );

        if within_budget {
            self.compute_count += 1;
        } else {
            self.budget_exceeded_count += 1;
        }

        &self.last_known_cut
    }

    /// Return the last known cut result without recomputing.
    #[must_use]
    pub fn last_known_cut(&self) -> &MinCutResult {
        &self.last_known_cut
    }
}

/// Stoer-Wagner minimum cut on a small adjacency matrix.
///
/// Operates directly on the caller-provided working buffers: `w` is the
/// adjacency matrix (consumed/mutated in place) and `groups` is the
/// super-node membership scratch (re-initialized here). The result is
/// written into `out`; the return value is `true` if the computation
/// completed within the iteration budget, `false` if it was exceeded
/// (in which case `out` holds the best partial result).
fn stoer_wagner_mincut(
    w: &mut [[u64; MINCUT_MAX_NODES]; MINCUT_MAX_NODES],
    groups: &mut [[bool; MINCUT_MAX_NODES]; MINCUT_MAX_NODES],
    n: usize,
    sub_nodes: &[Option<(NodeIdx, PartitionId)>; MINCUT_MAX_NODES],
    max_iterations: u32,
    out: &mut MinCutResult,
) -> bool {
    if n < 2 {
        *out = MinCutResult::empty();
        return true;
    }

    // Re-initialize the group membership scratch for the region in use:
    // each original node starts in its own super-node.
    for i in 0..n {
        for j in 0..n {
            groups[i][j] = i == j;
        }
    }

    // Track which super-nodes are still active
    let mut active = [false; MINCUT_MAX_NODES];
    for i in 0..n {
        active[i] = true;
    }

    let mut best_cut_weight = u64::MAX;
    let mut best_cut_node = 0usize; // The last node added in the best phase
    let mut iteration_count = 0u32;
    let mut active_count = n;

    // Run n-1 phases
    while active_count > 1 {
        if iteration_count >= max_iterations {
            // Budget exceeded: build partial result from best so far
            build_result(sub_nodes, groups, best_cut_node, n, out);
            out.cut_weight = best_cut_weight;
            out.within_budget = false;
            return false;
        }

        // Minimum cut phase: find the most tightly connected pair
        let (s, t, cut_of_phase) =
            minimum_cut_phase(w, &active, active_count, n);

        if cut_of_phase < best_cut_weight {
            best_cut_weight = cut_of_phase;
            best_cut_node = t;
        }

        // Merge t into s
        for i in 0..n {
            if active[i] && i != s && i != t {
                w[s][i] = w[s][i].saturating_add(w[t][i]);
                w[i][s] = w[i][s].saturating_add(w[i][t]);
            }
        }

        // Transfer group membership
        for i in 0..n {
            if groups[t][i] {
                groups[s][i] = true;
            }
        }

        active[t] = false;
        active_count -= 1;
        iteration_count += 1;
    }

    build_result(sub_nodes, groups, best_cut_node, n, out);
    out.cut_weight = best_cut_weight;
    out.within_budget = true;
    true
}

/// Single phase of the Stoer-Wagner algorithm: maximum adjacency ordering.
///
/// Returns `(second_to_last, last, cut_of_phase_weight)`.
fn minimum_cut_phase(
    w: &[[u64; MINCUT_MAX_NODES]; MINCUT_MAX_NODES],
    active: &[bool; MINCUT_MAX_NODES],
    active_count: usize,
    n: usize,
) -> (usize, usize, u64) {
    let mut in_a = [false; MINCUT_MAX_NODES];
    let mut key = [0u64; MINCUT_MAX_NODES]; // tightness of connection to A

    let mut second_to_last = 0usize;

    // Pick any active node as start
    let start = (0..n).find(|&i| active[i]).unwrap_or(0);
    in_a[start] = true;
    let mut last = start;

    // Initialize keys
    for i in 0..n {
        if active[i] && !in_a[i] {
            key[i] = w[start][i];
        }
    }

    for _ in 1..active_count {
        // Find the most tightly connected non-A node
        let mut max_key = 0u64;
        let mut max_node = 0usize;
        let mut found = false;
        for i in 0..n {
            if active[i] && !in_a[i] && (key[i] > max_key || !found) {
                max_key = key[i];
                max_node = i;
                found = true;
            }
        }

        second_to_last = last;
        last = max_node;
        in_a[max_node] = true;

        // Update keys
        for i in 0..n {
            if active[i] && !in_a[i] {
                key[i] = key[i].saturating_add(w[max_node][i]);
            }
        }
    }

    // The cut of this phase is the key value of `last` when it was added
    let cut_weight = key[last];
    (second_to_last, last, cut_weight)
}

/// Build the final `MinCutResult` from group membership, writing into
/// the caller-provided result slot (avoids a ~600 B return-value copy).
fn build_result(
    sub_nodes: &[Option<(NodeIdx, PartitionId)>; MINCUT_MAX_NODES],
    groups: &[[bool; MINCUT_MAX_NODES]; MINCUT_MAX_NODES],
    cut_node: usize,
    n: usize,
    result: &mut MinCutResult,
) {
    *result = MinCutResult::empty();

    // Nodes in the cut_node's group go to "left", rest go to "right"
    for i in 0..n {
        if let Some((_, pid)) = sub_nodes[i] {
            if groups[cut_node][i] {
                if (result.left_count as usize) < MINCUT_MAX_NODES {
                    result.left[result.left_count as usize] = Some(pid);
                    result.left_count += 1;
                }
            } else {
                if (result.right_count as usize) < MINCUT_MAX_NODES {
                    result.right[result.right_count as usize] = Some(pid);
                    result.right_count += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CoherenceGraph;

    fn pid(n: u32) -> PartitionId {
        PartitionId::new(n)
    }

    #[test]
    fn single_node_no_cut() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();

        let mut bridge = MinCutBridge::<8>::new(100);
        let result = bridge.find_min_cut(&g, pid(1));
        assert!(result.within_budget);
        assert_eq!(result.left_count, 1);
        assert_eq!(result.right_count, 0);
    }

    #[test]
    fn two_nodes_simple_cut() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_edge(pid(1), pid(2), 100).unwrap();
        g.add_edge(pid(2), pid(1), 100).unwrap();

        let mut bridge = MinCutBridge::<8>::new(100);
        let result = bridge.find_min_cut(&g, pid(1));
        assert!(result.within_budget);
        // Should split into two sides
        let total = result.left_count + result.right_count;
        assert_eq!(total, 2);
        assert!(result.cut_weight > 0);
    }

    #[test]
    fn three_nodes_finds_weakest_link() {
        let mut g = CoherenceGraph::<8, 32>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_node(pid(3)).unwrap();
        // Strong link between 1 and 2
        g.add_edge(pid(1), pid(2), 1000).unwrap();
        g.add_edge(pid(2), pid(1), 1000).unwrap();
        // Weak link between 2 and 3
        g.add_edge(pid(2), pid(3), 10).unwrap();
        g.add_edge(pid(3), pid(2), 10).unwrap();
        // Weak link between 1 and 3
        g.add_edge(pid(1), pid(3), 5).unwrap();
        g.add_edge(pid(3), pid(1), 5).unwrap();

        let mut bridge = MinCutBridge::<8>::new(100);
        let result = bridge.find_min_cut(&g, pid(1));
        assert!(result.within_budget);
        // The min cut should separate node 3 from {1, 2}.
        // edge_weight_between sums both directions, so the adjacency
        // matrix has w[2,3] = 20 and w[1,3] = 10. Cut weight = 30.
        assert_eq!(result.cut_weight, 30);
    }

    #[test]
    fn budget_exceeded_returns_last_known() {
        let mut g = CoherenceGraph::<8, 32>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_node(pid(3)).unwrap();
        g.add_node(pid(4)).unwrap();
        g.add_edge(pid(1), pid(2), 100).unwrap();
        g.add_edge(pid(2), pid(3), 100).unwrap();
        g.add_edge(pid(3), pid(4), 100).unwrap();
        g.add_edge(pid(4), pid(1), 100).unwrap();

        // Set max_iterations to 1 -- not enough for 4 nodes (needs 3 phases)
        let mut bridge = MinCutBridge::<8>::new(1);
        let result = bridge.find_min_cut(&g, pid(1));
        assert!(!result.within_budget);
        assert_eq!(bridge.budget_exceeded_count, 1);
    }

    #[test]
    fn epoch_tracking() {
        let mut bridge = MinCutBridge::<8>::new(100);
        assert_eq!(bridge.epoch(), 0);

        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();

        bridge.find_min_cut(&g, pid(1));
        assert_eq!(bridge.epoch(), 1);
        bridge.find_min_cut(&g, pid(1));
        assert_eq!(bridge.epoch(), 2);
    }

    #[test]
    fn compute_count_tracking() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_node(pid(2)).unwrap();
        g.add_edge(pid(1), pid(2), 50).unwrap();

        let mut bridge = MinCutBridge::<8>::new(100);
        bridge.find_min_cut(&g, pid(1));
        bridge.find_min_cut(&g, pid(1));
        assert_eq!(bridge.compute_count, 2);
    }
}
