//! Metric-dependent customization (ADR-198, Phase 2).
//!
//! Computes the weight of every upward arc (original + shortcut) by a bottom-up
//! triangle sweep over the elimination order. Re-run whenever the metric changes
//! — topology is untouched. For each lower vertex `r` and each pair of its upward
//! neighbours `(a, b)`, relax `w(a,b) = min(w(a,b), w(r,a) + w(r,b))`.
//!
//! Processing in increasing rank is correct: arc `(r, a)` is finalised by the
//! time we process `r`, because any improvement to it comes from a triangle with
//! a strictly lower-ranked apex, already processed.

use crate::contraction::Topology;

/// Per-arc weights, parallel to `Topology::up`. One `Metric` per relevance lens.
#[derive(Clone, Debug)]
pub struct Metric {
    pub w: Vec<Vec<f64>>,
}

/// Run customization for the metric whose original-edge weights are already in
/// `topo.w0`. (At M2 the initial weights come from a GNN edge head; for M0 they
/// are the graph's own edge weights.)
#[must_use]
pub fn customize(topo: &Topology) -> Metric {
    let mut w = topo.w0.clone();
    for r in 0..topo.n as u32 {
        let hi = &topo.up[r as usize];
        for i in 0..hi.len() {
            let wri = w[r as usize][i];
            if !wri.is_finite() {
                continue;
            }
            for j in (i + 1)..hi.len() {
                let wrj = w[r as usize][j];
                if !wrj.is_finite() {
                    continue;
                }
                let (a, b) = (hi[i], hi[j]); // a < b, both > r
                let cand = wri + wrj;
                // Relax arc (a -> b).
                if let Some(idx) = topo.arc_pos(a, b) {
                    let slot = &mut w[a as usize][idx];
                    if cand < *slot {
                        *slot = cand;
                    }
                }
            }
        }
    }
    Metric { w }
}
