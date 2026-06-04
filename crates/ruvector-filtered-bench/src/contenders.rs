//! M1 — incumbents (B/D) and the post-filter floor (C), each reporting exact
//! distance-evals via the instrumented `ruvector-acorn` search.
//!
//! All three drive the **real** `AcornGraph` + `acorn_search_counted` — not a
//! re-implementation — so the head-to-head measures ACORN as shipped (protocol
//! rule #2). Contender A (region-pruned IVF) arrives in M2 (`prune` module).

use ruvector_acorn::graph::AcornGraph;
use ruvector_acorn::search::acorn_search_counted;

/// ACORN edge budget base (γ·M neighbors/node); matches `AcornIndexGamma::M`.
pub const ACORN_M: usize = 16;

/// Outcome of one filtered query: the returned ids (nearest-first) and the exact
/// number of distance evaluations spent — the pre-registered primary cost metric.
pub struct QueryResult {
    pub ids: Vec<u32>,
    pub evals: u64,
}

/// A real ACORN-γ graph. Drives **B** (predicate-agnostic search) and **C** (the
/// post-filter floor) off one graph, so the only variable between them is the
/// traversal policy — the cleanest demonstration that post-filter, not graph
/// density, is what collapses at low selectivity.
pub struct Acorn {
    pub graph: AcornGraph,
    pub gamma: usize,
    pub ef: usize,
}

impl Acorn {
    /// Build the incumbent graph. `gamma` = 2 is `AcornIndexGamma`'s default
    /// (32 edges/node); `gamma` = 3 is the "tune harder" variant (D's denser graph).
    pub fn build(feats: &[Vec<f32>], gamma: usize, ef: usize) -> Self {
        let graph = AcornGraph::build(feats.to_vec(), ACORN_M * gamma)
            .expect("acorn graph build");
        Acorn { graph, gamma, ef }
    }

    /// **Contender B** — ACORN predicate-agnostic search (expands all neighbors).
    pub fn search(&self, query: &[f32], k: usize, predicate: impl Fn(u32) -> bool) -> QueryResult {
        let (got, evals) = acorn_search_counted(&self.graph, query, k, self.ef, predicate);
        QueryResult { ids: got.into_iter().map(|(id, _)| id).collect(), evals }
    }

    /// **Contender C** — classic post-filter: retrieve the `pool` nearest neighbors
    /// *ignoring* the predicate, then keep the first `k` that pass. At low
    /// selectivity the unfiltered pool is almost all non-matching, so few (or zero)
    /// survive → recall collapses. This is the floor ACORN was designed to beat;
    /// reproducing the collapse proves the benchmark has teeth.
    pub fn postfilter(
        &self,
        query: &[f32],
        k: usize,
        pool: usize,
        predicate: impl Fn(u32) -> bool,
    ) -> QueryResult {
        let pool = pool.max(k);
        // Unfiltered retrieval (predicate = always-true); cost is the search's evals.
        let (cands, evals) = acorn_search_counted(&self.graph, query, pool, self.ef, |_| true);
        let ids = cands
            .into_iter()
            .map(|(id, _)| id)
            .filter(|&id| predicate(id))
            .take(k)
            .collect();
        QueryResult { ids, evals }
    }
}

/// Recall@k against an exact filtered-kNN truth set: fraction of the true top-k
/// that the contender returned. `truth` may be shorter than k when matches < k.
pub fn recall(truth: &[u32], got: &[u32]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let got_set: std::collections::HashSet<u32> = got.iter().copied().collect();
    let hit = truth.iter().filter(|id| got_set.contains(id)).count();
    hit as f64 / truth.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_acorn::graph::exact_filtered_knn;

    fn ramp(n: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| vec![i as f32, (i % 7) as f32]).collect()
    }

    #[test]
    fn agnostic_beats_postfilter_when_selective() {
        // A predicate matching only every 11th node (~9%) should let ACORN's
        // agnostic search keep recall while post-filter (pool=k) starves.
        let feats = ramp(600);
        let acorn = Acorn::build(&feats, 2, 80);
        let k = 5;
        let pred = |id: u32| id.is_multiple_of(11);

        let (mut agn_hits, mut pf_hits, mut n) = (0.0, 0.0, 0.0);
        for qi in (0..600).step_by(97) {
            let truth = exact_filtered_knn(&feats, &feats[qi], k, pred);
            let agn = acorn.search(&feats[qi], k, pred);
            let pf = acorn.postfilter(&feats[qi], k, k, pred); // tight pool → starves
            agn_hits += recall(&truth, &agn.ids);
            pf_hits += recall(&truth, &pf.ids);
            n += 1.0;
        }
        assert!(
            agn_hits / n >= pf_hits / n,
            "agnostic recall {:.2} should be >= post-filter recall {:.2}",
            agn_hits / n,
            pf_hits / n
        );
    }

    #[test]
    fn evals_are_recorded() {
        let feats = ramp(300);
        let acorn = Acorn::build(&feats, 2, 64);
        let r = acorn.search(&feats[10], 5, |_| true);
        assert!(r.evals > 0);
    }
}
