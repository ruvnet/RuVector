//! Symbolic contraction → chordal upward graph + elimination tree (ADR-197 §3).
//!
//! Everything here is in **rank space**: vertices are relabelled to their
//! contraction rank, so `up[r]` lists higher-ranked neighbours in ascending
//! order (cache-friendly, SIMD-amenable per ADR-197 §4). This phase is fully
//! metric-independent — the shortcut *set* depends only on order + topology.

use crate::graph::{Graph, NodeId};
use std::collections::BTreeSet;

pub const NONE: u32 = u32::MAX;

/// Metric-independent skeleton built once from `(Graph, order)`.
pub struct Topology {
    pub n: usize,
    /// `rank[orig] = contraction rank`.
    pub rank: Vec<u32>,
    /// `orig[rank] = original id` (inverse of `rank`).
    pub orig: Vec<NodeId>,
    /// Upward chordal arcs in rank space: `up[r]` = higher-ranked neighbours of
    /// rank `r`, ascending. Includes original edges + fill-in shortcuts.
    pub up: Vec<Vec<u32>>,
    /// Initial weight of each upward arc, parallel to `up`. `+inf` for shortcuts
    /// (filled in by customization); finite for original edges.
    pub w0: Vec<Vec<f64>>,
    /// Elimination-tree parent (rank space): lowest higher-ranked neighbour.
    pub elim_parent: Vec<u32>,
}

impl Topology {
    /// Number of upward arcs (|G+| restricted to upward orientation) — the
    /// numerator of the shortcut-blowup ratio in ADR-199.
    #[must_use]
    pub fn arc_count(&self) -> usize {
        self.up.iter().map(Vec::len).sum()
    }

    /// Index of arc `r -> hi` within `up[r]` (arcs are sorted, so binary search).
    #[inline]
    pub fn arc_pos(&self, r: u32, hi: u32) -> Option<usize> {
        self.up[r as usize].binary_search(&hi).ok()
    }
}

/// Build the chordal upward graph and elimination tree from a contraction order.
#[must_use]
pub fn contract(g: &Graph, order: &[NodeId]) -> Topology {
    let n = g.n;
    let mut rank = vec![0u32; n];
    for (r, &v) in order.iter().enumerate() {
        rank[v as usize] = r as u32;
    }
    let orig: Vec<NodeId> = order.to_vec();

    // Original adjacency in rank space, plus original weights keyed by arc.
    let mut nbrs: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
    // up-weight lookup during build: orig weight of (min,max) ranks.
    let mut orig_w: Vec<std::collections::HashMap<u32, f64>> = vec![Default::default(); n];
    for (u, v, w) in g.edges() {
        let (ru, rv) = (rank[u as usize], rank[v as usize]);
        nbrs[ru as usize].insert(rv);
        nbrs[rv as usize].insert(ru);
        let (lo, hi) = if ru < rv { (ru, rv) } else { (rv, ru) };
        orig_w[lo as usize].insert(hi, w);
    }

    let mut up: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut elim_parent = vec![NONE; n];

    // Contract in increasing rank; eliminating r makes its higher neighbours a clique.
    for r in 0..n as u32 {
        let hi: Vec<u32> = nbrs[r as usize]
            .iter()
            .copied()
            .filter(|&x| x > r)
            .collect();
        elim_parent[r as usize] = hi.first().copied().unwrap_or(NONE);
        // Fill-in: every pair of higher neighbours becomes adjacent.
        for i in 0..hi.len() {
            for j in (i + 1)..hi.len() {
                let (a, b) = (hi[i], hi[j]);
                nbrs[a as usize].insert(b);
                nbrs[b as usize].insert(a);
            }
        }
        up[r as usize] = hi; // already ascending (BTreeSet order preserved)
    }

    // Initialise arc weights: original edges get their weight, shortcuts +inf.
    let mut w0: Vec<Vec<f64>> = up
        .iter()
        .map(|row| vec![f64::INFINITY; row.len()])
        .collect();
    for r in 0..n {
        for (i, &hi) in up[r].iter().enumerate() {
            if let Some(&w) = orig_w[r].get(&hi) {
                w0[r][i] = w;
            }
        }
    }

    Topology { n, rank, orig, up, w0, elim_parent }
}
