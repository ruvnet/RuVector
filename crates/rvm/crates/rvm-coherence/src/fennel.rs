//! Fennel-style streaming placement: the HOT PATH of the two-tier
//! split-decision architecture.
//!
//! ## Two-tier architecture (roadmap item 5)
//!
//! - **Hot path (this module)**: when a new node arrives, it is placed
//!   greedily onto one of two sides in O(degree) work using the Fennel
//!   objective (Tsourakakis et al., WSDM 2014):
//!
//!   ```text
//!   side = argmax_P  w(v, P) - alpha * |P|^gamma      (gamma = 1.5)
//!   ```
//!
//!   where `w(v, P)` is the total edge weight between the new node `v`
//!   and the nodes already assigned to side `P`, and `alpha * |P|^1.5`
//!   is the balance penalty discouraging one side from absorbing
//!   everything. No full-graph work is performed: only `v`'s incident
//!   edges and O(1) cached side sizes are consulted.
//!
//! - **Epoch task (see [`crate::engine`])**: the exact Stoer-Wagner
//!   mincut runs periodically (or on pressure triggers) off the hot
//!   path and produces a [`crate::engine::SplitPlan`]. The plan seeds
//!   this placer, and subsequent arrivals are routed incrementally
//!   until the next epoch recomputation.
//!
//! ## Fixed-point arithmetic
//!
//! All scoring is integer-only (`no_std`, no FPU). Weights are scaled
//! by 1000 ("milli" units) and `|P|^1.5` is computed as
//! `|P| * isqrt(|P| * 10^6)` so that `gamma = 1.5` is exact to three
//! decimal places. To avoid signed arithmetic, the comparison
//! `w0 - pen0 >= w1 - pen1` is evaluated as `w0 + pen1 >= w1 + pen0`.

use rvm_types::PartitionId;

use crate::graph::CoherenceGraph;

/// Number of sides the placer routes between. Matches the binary
/// split (keep / move-to-child) produced by the epoch mincut.
pub const FENNEL_SIDES: usize = 2;

/// Maximum tracked assignments. Matches the coherence graph's 32-node
/// adjacency bound (`ADJ_DIM`).
const FENNEL_MAX_ENTRIES: usize = 32;

/// Default balance penalty coefficient, in milli (fixed-point / 1000).
///
/// `100` means `alpha = 0.1` weight-units. Rationale: side sizes are
/// bounded by 32, so the maximum possible penalty difference is
/// `0.1 * 32^1.5 ~= 18` weight units. Communication edge weights in the
/// coherence graph are message counts (typically tens to thousands),
/// so neighbor affinity dominates whenever any real affinity exists,
/// and the balance penalty only decides ties and near-ties. This is
/// deliberately the single tuning constant of the hot path.
pub const DEFAULT_ALPHA_MILLI: u64 = 100;

/// Integer square root (Newton's method), exact floor for all `u64`.
const fn isqrt(v: u64) -> u64 {
    if v < 2 {
        return v;
    }
    let mut x = v;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + v / x) / 2;
    }
    x
}

/// `|P|^1.5` in milli units: `s * sqrt(s) * 1000`, computed as
/// `s * isqrt(s * 10^6)`.
const fn pow_1_5_milli(s: u32) -> u64 {
    (s as u64).saturating_mul(isqrt((s as u64).saturating_mul(1_000_000)))
}

/// Incremental two-way Fennel placer.
///
/// Holds the side assignment of up to 32 partitions plus O(1) cached
/// side sizes. [`FennelPlacer::place`] is the hot-path entry point:
/// O(degree) per placement, no full-graph scans.
#[derive(Debug, Clone)]
pub struct FennelPlacer {
    /// Assigned partition IDs (association list; bounded, `no_std`).
    ids: [Option<PartitionId>; FENNEL_MAX_ENTRIES],
    /// Side per entry, parallel to `ids`.
    sides: [u8; FENNEL_MAX_ENTRIES],
    /// Cached number of partitions on each side.
    sizes: [u32; FENNEL_SIDES],
    /// Balance penalty coefficient in milli units.
    alpha_milli: u64,
}

impl FennelPlacer {
    /// Create a placer with an explicit balance coefficient (milli).
    #[must_use]
    pub const fn new(alpha_milli: u64) -> Self {
        Self {
            ids: [None; FENNEL_MAX_ENTRIES],
            sides: [0; FENNEL_MAX_ENTRIES],
            sizes: [0; FENNEL_SIDES],
            alpha_milli,
        }
    }

    /// Create a placer with [`DEFAULT_ALPHA_MILLI`].
    #[must_use]
    pub const fn with_default_alpha() -> Self {
        Self::new(DEFAULT_ALPHA_MILLI)
    }

    /// Remove all assignments (e.g., before reseeding from a new epoch plan).
    pub fn clear(&mut self) {
        self.ids = [None; FENNEL_MAX_ENTRIES];
        self.sides = [0; FENNEL_MAX_ENTRIES];
        self.sizes = [0; FENNEL_SIDES];
    }

    /// Pre-assign a partition to a side (used to seed from the epoch
    /// mincut plan). Returns `false` if the side is out of range or the
    /// table is full. Re-seeding an existing partition moves it.
    pub fn seed(&mut self, pid: PartitionId, side: u8) -> bool {
        if (side as usize) >= FENNEL_SIDES {
            return false;
        }
        // Existing entry: move it.
        for i in 0..FENNEL_MAX_ENTRIES {
            if self.ids[i] == Some(pid) {
                let old = self.sides[i] as usize;
                self.sizes[old] = self.sizes[old].saturating_sub(1);
                self.sides[i] = side;
                self.sizes[side as usize] += 1;
                return true;
            }
        }
        // New entry: first free slot.
        for i in 0..FENNEL_MAX_ENTRIES {
            if self.ids[i].is_none() {
                self.ids[i] = Some(pid);
                self.sides[i] = side;
                self.sizes[side as usize] += 1;
                return true;
            }
        }
        false
    }

    /// Forget a partition's assignment (e.g., when it is destroyed).
    pub fn remove(&mut self, pid: PartitionId) {
        for i in 0..FENNEL_MAX_ENTRIES {
            if self.ids[i] == Some(pid) {
                let side = self.sides[i] as usize;
                self.sizes[side] = self.sizes[side].saturating_sub(1);
                self.ids[i] = None;
                self.sides[i] = 0;
                return;
            }
        }
    }

    /// The side a partition is assigned to, if any.
    #[must_use]
    pub fn side_of(&self, pid: PartitionId) -> Option<u8> {
        for i in 0..FENNEL_MAX_ENTRIES {
            if self.ids[i] == Some(pid) {
                return Some(self.sides[i]);
            }
        }
        None
    }

    /// Number of partitions assigned to `side` (0 for out-of-range sides).
    #[must_use]
    pub fn size(&self, side: u8) -> u32 {
        if (side as usize) < FENNEL_SIDES {
            self.sizes[side as usize]
        } else {
            0
        }
    }

    /// Total number of assigned partitions.
    #[must_use]
    pub fn assigned_count(&self) -> u32 {
        self.sizes[0] + self.sizes[1]
    }

    /// HOT PATH: place `pid` greedily by the Fennel objective.
    ///
    /// Cost is O(degree of `pid`) edge work plus a bounded 32-entry
    /// membership scan — no full-graph (edge-set) traversal and no
    /// mincut recomputation. If `pid` is already assigned, its existing
    /// side is returned unchanged. If `pid` has no edges (or is not in
    /// the graph), placement is purely balance-driven (smaller side).
    pub fn place<const N: usize, const E: usize>(
        &mut self,
        graph: &CoherenceGraph<N, E>,
        pid: PartitionId,
    ) -> u8 {
        if let Some(side) = self.side_of(pid) {
            return side;
        }

        // Accumulate w(v, P) per side over v's distinct neighbors.
        // Outgoing neighbors come from the adjacency list (O(out-deg));
        // incoming neighbors from the adjacency-matrix column
        // (O(32), bounded). Dedupe via a node-index bitmask, then read
        // each pairwise weight in O(1) from the adjacency matrix.
        let mut side_weight = [0u64; FENNEL_SIDES];
        if let Some(idx) = graph.find_node(pid) {
            let mut mask: u64 = 0;
            if let Some(iter) = graph.neighbors(pid) {
                for (n, _w) in iter {
                    mask |= 1u64 << n;
                }
            }
            for n in graph.in_neighbors_of(idx) {
                mask |= 1u64 << n;
            }
            mask &= !(1u64 << idx); // self-loops carry no side affinity

            let mut n = 0u16;
            while mask != 0 {
                if mask & 1 != 0 {
                    if let Some(q) = graph.partition_at(n) {
                        if let Some(s) = self.side_of(q) {
                            side_weight[s as usize] = side_weight[s as usize]
                                .saturating_add(graph.edge_weight_between(pid, q));
                        }
                    }
                }
                mask >>= 1;
                n += 1;
            }
        }

        // score(P) = w(v,P)*1000 - alpha_milli * |P|^1.5(milli) / 1000.
        // Compare side 0 vs side 1 with penalties moved to the opposite
        // side so everything stays unsigned.
        let pen0 = self
            .alpha_milli
            .saturating_mul(pow_1_5_milli(self.sizes[0]))
            / 1000;
        let pen1 = self
            .alpha_milli
            .saturating_mul(pow_1_5_milli(self.sizes[1]))
            / 1000;
        let lhs = side_weight[0].saturating_mul(1000).saturating_add(pen1);
        let rhs = side_weight[1].saturating_mul(1000).saturating_add(pen0);

        let side = match lhs.cmp(&rhs) {
            core::cmp::Ordering::Greater => 0u8,
            core::cmp::Ordering::Less => 1u8,
            // Exact tie (equal affinity and equal penalty): smaller
            // side first, then side 0.
            core::cmp::Ordering::Equal => u8::from(self.sizes[0] > self.sizes[1]),
        };

        let _ = self.seed(pid, side);
        side
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(n: u32) -> PartitionId {
        PartitionId::new(n)
    }

    #[test]
    fn isqrt_exact_values() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(3), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(1_000_000), 1000);
        assert_eq!(isqrt(32_000_000), 5656); // sqrt(32)*1000 = 5656.85
    }

    #[test]
    fn pow_1_5_is_monotonic() {
        let mut prev = 0u64;
        for s in 0..=32u32 {
            let v = pow_1_5_milli(s);
            assert!(v >= prev, "pow_1_5 must be monotonic");
            prev = v;
        }
        // Spot check: 4^1.5 = 8 -> 8000 milli.
        assert_eq!(pow_1_5_milli(4), 8000);
    }

    #[test]
    fn balanced_growth_without_edges() {
        // No edges: placement is purely balance-driven and must
        // alternate sides, never letting one side run away.
        let g = CoherenceGraph::<8, 16>::new();
        let mut placer = FennelPlacer::with_default_alpha();

        for n in 1..=6u32 {
            placer.place(&g, pid(n));
        }
        assert_eq!(placer.size(0), 3);
        assert_eq!(placer.size(1), 3);
    }

    #[test]
    fn neighbor_affinity_wins_at_equal_sizes() {
        let mut g = CoherenceGraph::<8, 32>::new();
        for n in 1..=5u32 {
            g.add_node(pid(n)).unwrap();
        }
        // Cluster A = {1,2}, cluster B = {3,4}; new node 5 talks to B.
        g.add_edge(pid(5), pid(3), 500).unwrap();
        g.add_edge(pid(4), pid(5), 500).unwrap();

        let mut placer = FennelPlacer::with_default_alpha();
        placer.seed(pid(1), 0);
        placer.seed(pid(2), 0);
        placer.seed(pid(3), 1);
        placer.seed(pid(4), 1);

        assert_eq!(placer.place(&g, pid(5)), 1);
        assert_eq!(placer.size(1), 3);
    }

    #[test]
    fn affinity_beats_mild_imbalance() {
        let mut g = CoherenceGraph::<8, 32>::new();
        for n in 1..=6u32 {
            g.add_node(pid(n)).unwrap();
        }
        // Side 1 is already bigger (3 vs 1), but node 6 communicates
        // with side 1: real affinity must override the balance penalty.
        g.add_edge(pid(6), pid(3), 100).unwrap();

        let mut placer = FennelPlacer::with_default_alpha();
        placer.seed(pid(1), 0);
        placer.seed(pid(3), 1);
        placer.seed(pid(4), 1);
        placer.seed(pid(5), 1);

        assert_eq!(placer.place(&g, pid(6)), 1);
    }

    #[test]
    fn balance_penalty_routes_isolated_node_to_smaller_side() {
        let g = CoherenceGraph::<8, 16>::new();
        let mut placer = FennelPlacer::with_default_alpha();
        for n in 1..=5u32 {
            placer.seed(pid(n), 0);
        }
        placer.seed(pid(6), 1);

        // Isolated node: no affinity, must go to the smaller side.
        assert_eq!(placer.place(&g, pid(7)), 1);
    }

    #[test]
    fn already_assigned_is_stable() {
        let g = CoherenceGraph::<8, 16>::new();
        let mut placer = FennelPlacer::with_default_alpha();
        placer.seed(pid(1), 1);
        assert_eq!(placer.place(&g, pid(1)), 1);
        assert_eq!(placer.size(1), 1);
        assert_eq!(placer.assigned_count(), 1);
    }

    #[test]
    fn self_loops_carry_no_affinity() {
        let mut g = CoherenceGraph::<8, 16>::new();
        g.add_node(pid(1)).unwrap();
        g.add_edge(pid(1), pid(1), 10_000).unwrap();

        let mut placer = FennelPlacer::with_default_alpha();
        placer.seed(pid(2), 0); // side 0 has one member
        // Node 1's only edge is a self-loop: balance decides -> side 1.
        assert_eq!(placer.place(&g, pid(1)), 1);
    }

    #[test]
    fn seed_moves_existing_entry() {
        let mut placer = FennelPlacer::with_default_alpha();
        assert!(placer.seed(pid(1), 0));
        assert_eq!(placer.size(0), 1);
        assert!(placer.seed(pid(1), 1));
        assert_eq!(placer.size(0), 0);
        assert_eq!(placer.size(1), 1);
    }

    #[test]
    fn seed_rejects_out_of_range_side() {
        let mut placer = FennelPlacer::with_default_alpha();
        assert!(!placer.seed(pid(1), 2));
    }

    #[test]
    fn remove_updates_sizes() {
        let mut placer = FennelPlacer::with_default_alpha();
        placer.seed(pid(1), 0);
        placer.seed(pid(2), 1);
        placer.remove(pid(1));
        assert_eq!(placer.size(0), 0);
        assert_eq!(placer.side_of(pid(1)), None);
        assert_eq!(placer.side_of(pid(2)), Some(1));
    }

    #[test]
    fn clear_resets_everything() {
        let mut placer = FennelPlacer::with_default_alpha();
        placer.seed(pid(1), 0);
        placer.seed(pid(2), 1);
        placer.clear();
        assert_eq!(placer.assigned_count(), 0);
        assert_eq!(placer.side_of(pid(1)), None);
    }

    #[test]
    fn unassigned_neighbors_are_ignored() {
        let mut g = CoherenceGraph::<8, 32>::new();
        for n in 1..=3u32 {
            g.add_node(pid(n)).unwrap();
        }
        // Node 3 talks to node 2, but node 2 has no side assignment:
        // that edge contributes no affinity, so balance decides.
        g.add_edge(pid(3), pid(2), 1000).unwrap();

        let mut placer = FennelPlacer::with_default_alpha();
        placer.seed(pid(1), 0);
        assert_eq!(placer.place(&g, pid(3)), 1); // smaller side
    }
}
