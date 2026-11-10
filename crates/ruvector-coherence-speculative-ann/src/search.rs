//! Graph-traversal draft stage: bounded beam search over `DraftGraph`,
//! ranked by cheap u8 distances, with an optional traversal-coherence
//! expansion gate. Returns a candidate id pool for the verify stage plus a
//! cost accounting (`pops`, `expansions`) that is the algorithmic analogue of
//! the prior crate's O(n) draft scan size.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::coherence::traversal_coherence;
use crate::graph::DraftGraph;
use crate::sq_l2_u8;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OrdU64(u64);

/// Draft-stage output: a bounded candidate pool plus algorithmic cost.
pub struct DraftResult {
    /// Candidate ids ranked by u8 distance (nearest first), length <= ef.
    pub candidate_ids: Vec<u32>,
    /// Candidates popped from the beam frontier.
    pub pops: usize,
    /// Candidates whose neighborhoods were expanded (the coherence gate
    /// reduces this relative to `pops`).
    pub expansions: usize,
}

pub trait DraftSearcher: Send + Sync {
    /// `k_prime` is the draft candidate-pool size handed to the verify stage;
    /// `ef` is the beam width bound during traversal (ef >= k_prime).
    fn draft(
        &self,
        graph: &DraftGraph,
        query_f32: &[f32],
        query_u8: &[u8],
        k_prime: usize,
        ef: usize,
        entry_id: usize,
    ) -> DraftResult;

    fn name(&self) -> &str;
}

/// Plain graph beam draft: every popped candidate's neighbors are expanded
/// unconditionally (the coherence gate always returns true). Isolates the
/// effect of "graph traversal instead of linear scan" without the coherence
/// gate's contribution.
pub struct BaselineGraphDraft;

impl DraftSearcher for BaselineGraphDraft {
    fn draft(
        &self,
        graph: &DraftGraph,
        query_f32: &[f32],
        query_u8: &[u8],
        k_prime: usize,
        ef: usize,
        entry_id: usize,
    ) -> DraftResult {
        beam_draft(
            graph,
            query_f32,
            query_u8,
            k_prime,
            ef,
            entry_id,
            |_e, _c, _q| true,
        )
    }

    fn name(&self) -> &str {
        "BaselineGraphDraft"
    }
}

/// Coherence-gated graph beam draft: a popped candidate's neighborhood is
/// only expanded if its traversal coherence (computed on exact f32 vectors)
/// is at or above `threshold`. The candidate itself is still placed in the
/// result pool regardless of the gate.
pub struct CoherenceGatedGraphDraft {
    pub threshold: f32,
}

impl DraftSearcher for CoherenceGatedGraphDraft {
    fn draft(
        &self,
        graph: &DraftGraph,
        query_f32: &[f32],
        query_u8: &[u8],
        k_prime: usize,
        ef: usize,
        entry_id: usize,
    ) -> DraftResult {
        let t = self.threshold;
        beam_draft(
            graph,
            query_f32,
            query_u8,
            k_prime,
            ef,
            entry_id,
            move |entry, cand, query| traversal_coherence(entry, cand, query) >= t,
        )
    }

    fn name(&self) -> &str {
        "CoherenceGatedGraphDraft"
    }
}

#[allow(clippy::too_many_arguments)]
fn beam_draft<F>(
    graph: &DraftGraph,
    query_f32: &[f32],
    query_u8: &[u8],
    k_prime: usize,
    ef: usize,
    entry_id: usize,
    expand_gate: F,
) -> DraftResult
where
    F: Fn(&[f32], &[f32], &[f32]) -> bool,
{
    if graph.is_empty() {
        return DraftResult {
            candidate_ids: vec![],
            pops: 0,
            expansions: 0,
        };
    }

    let n = graph.n;
    let ef = ef.max(k_prime);
    let entry = entry_id.min(n - 1);
    let entry_f32 = graph.row_f32(entry);

    let mut visited: Vec<bool> = vec![false; n];
    // Min-heap on u8 squared distance: pop closest-by-cheap-metric first.
    let mut candidates: BinaryHeap<Reverse<(OrdU64, u32)>> = BinaryHeap::with_capacity(ef + 1);
    // Bounded max-heap of the draft pool (worst-at-top), sized `ef`.
    let mut pool: BinaryHeap<(OrdU64, u32)> = BinaryHeap::with_capacity(ef + 1);

    let d0 = sq_l2_u8(query_u8, graph.row_u8(entry));
    candidates.push(Reverse((OrdU64(d0), entry as u32)));
    visited[entry] = true;

    let mut pops = 0usize;
    let mut expansions = 0usize;

    while let Some(Reverse((OrdU64(curr_d), curr))) = candidates.pop() {
        pops += 1;

        if pool.len() >= ef {
            if let Some(&(OrdU64(worst), _)) = pool.peek() {
                if curr_d > worst {
                    break;
                }
            }
        }

        pool.push((OrdU64(curr_d), curr));
        if pool.len() > ef {
            pool.pop();
        }

        let curr_f32 = graph.row_f32(curr as usize);
        if !expand_gate(entry_f32, curr_f32, query_f32) {
            continue;
        }
        expansions += 1;

        let worst_pending = candidates
            .iter()
            .map(|Reverse((OrdU64(d), _))| *d)
            .max()
            .unwrap_or(u64::MAX);

        for &nbr in &graph.neighbors[curr as usize] {
            let ni = nbr as usize;
            if visited[ni] {
                continue;
            }
            visited[ni] = true;
            let nd = sq_l2_u8(query_u8, graph.row_u8(ni));

            if candidates.len() < ef || nd < worst_pending {
                candidates.push(Reverse((OrdU64(nd), nbr)));
                if candidates.len() > ef * 2 {
                    let mut items: Vec<_> = candidates.drain().collect();
                    items.sort_unstable_by_key(|a| a.0 .0 .0);
                    items.truncate(ef);
                    candidates.extend(items);
                }
            }
        }
    }

    let mut candidate_ids: Vec<(u64, u32)> =
        pool.into_iter().map(|(OrdU64(d), id)| (d, id)).collect();
    candidate_ids.sort_unstable_by_key(|&(d, _)| d);
    candidate_ids.truncate(k_prime.max(candidate_ids.len().min(k_prime)));

    DraftResult {
        candidate_ids: candidate_ids.into_iter().map(|(_, id)| id).collect(),
        pops,
        expansions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::graph::GraphConfig;
    use crate::ScalarQuantizer;

    fn build_test_graph() -> (DraftGraph, ScalarQuantizer, Dataset) {
        let ds = Dataset::generate(200, 16, 5, 123);
        let q = ScalarQuantizer::train(&ds.corpus_vectors());
        let g = DraftGraph::build(
            ds.corpus.clone(),
            &q,
            GraphConfig {
                m: 8,
                m_longjump: 4,
                dims: 16,
            },
        );
        (g, q, ds)
    }

    #[test]
    fn baseline_draft_returns_bounded_pool() {
        let (g, quantizer, ds) = build_test_graph();
        let query = ds.query(0);
        let query_u8 = quantizer.quantize(query);
        let r = BaselineGraphDraft.draft(&g, query, &query_u8, 10, 20, 0);
        assert!(r.candidate_ids.len() <= 20);
        assert!(!r.candidate_ids.is_empty());
    }

    #[test]
    fn coherence_gate_never_expands_more_than_baseline() {
        let (g, quantizer, ds) = build_test_graph();
        let query = ds.query(1);
        let query_u8 = quantizer.quantize(query);
        let base = BaselineGraphDraft.draft(&g, query, &query_u8, 10, 20, 0);
        let gated =
            CoherenceGatedGraphDraft { threshold: 0.2 }.draft(&g, query, &query_u8, 10, 20, 0);
        assert!(gated.expansions <= base.expansions);
    }
}
