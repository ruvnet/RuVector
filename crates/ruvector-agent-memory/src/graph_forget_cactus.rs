//! Canonical-cactus-cut forgetting: attacking ADR-345's rejection root cause
//! (ADR-346, docs/research/nightly/2026-09-10-canonical-cactus-forgetting).
//!
//! [`crate::graph_forget::MincutGatedForgetting`] (ADR-345) implemented the
//! same idea this module implements — a k-NN graph min-cut boundary as a
//! structural "don't evict the bridge" signal for compaction — using
//! [`ruvector_mincut::RuVectorGraphAnalyzer`], the crate's general dynamic
//! min-cut wrapper. That experiment was **rejected**: `partition()` returned
//! an empty (unusable) result on 50% of repeated calls on byte-identical
//! input, and cost 1,800-2,700x the scalar baseline even on an 84-vertex
//! corpus (see ADR-345's "Failure modes").
//!
//! `ruvector-mincut` separately ships a `canonical` feature
//! (`source_anchored`/`tree_packing`/`dynamic` tiers, plus a standalone
//! [`ruvector_mincut::CactusGraph`]) whose entire purpose is a
//! *deterministic* global min-cut: it runs dense-array Stoer-Wagner to
//! enumerate every minimum-cut partition, encodes them in a cactus, and
//! picks the lexicographically smallest one as `canonical_cut()`. It was not
//! used by ADR-345 and nobody had measured it against that experiment's own
//! rejection criteria. This module does that: same boundary-signal idea, same
//! `ForgetMode::{Soft,Hard}` policies, same acceptance-test shape, swapped
//! backend.

use crate::compaction::{weighted_importance, CoherenceWeights, CompactionPolicy};
use crate::graph_forget::ForgetMode;
use crate::memory::MemoryEntry;
use crate::scoring::cosine_sim;
use ruvector_mincut::CactusGraph;
use ruvector_mincut::DynamicGraph;
use std::collections::HashSet;
use std::sync::Arc;

/// Cactus-canonical-cut forgetting compaction policy (ADR-346 candidates).
///
/// Identical scoring/eviction logic to
/// [`crate::graph_forget::MincutGatedForgetting`]; the only difference is
/// [`Self::boundary_indices`]'s backend.
#[derive(Debug, Clone)]
pub struct CactusGatedForgetting {
    pub weights: CoherenceWeights,
    pub mode: ForgetMode,
    /// Max neighbors per vertex when building the similarity graph.
    pub k_neighbors: usize,
    /// Minimum cosine similarity for an edge to be added.
    pub min_similarity: f32,
    /// [`ForgetMode::Soft`] only: bonus added to a boundary vertex's scalar
    /// importance score.
    pub structural_bonus: f32,
    /// [`ForgetMode::Hard`] only: fraction of `target_size` reserved for
    /// boundary vertices.
    pub protect_fraction: f32,
}

impl CactusGatedForgetting {
    /// [`ForgetMode::Soft`] with the given weights and bonus.
    pub fn soft(weights: CoherenceWeights, structural_bonus: f32) -> Self {
        Self {
            weights,
            mode: ForgetMode::Soft,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus,
            protect_fraction: 0.0,
        }
    }

    /// [`ForgetMode::Hard`] with the given weights and protected fraction.
    pub fn hard(weights: CoherenceWeights, protect_fraction: f32) -> Self {
        Self {
            weights,
            mode: ForgetMode::Hard,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus: 0.0,
            protect_fraction,
        }
    }

    /// Build a k-NN cosine-similarity graph (identical construction to
    /// [`crate::graph_forget::MincutGatedForgetting::boundary_indices`]) and
    /// return the indices of vertices with at least one neighbor edge
    /// crossing [`ruvector_mincut::CactusGraph::canonical_cut`]'s partition.
    ///
    /// Unlike the ADR-345 backend this makes exactly one min-cut call: the
    /// canonical cut is deterministic by construction (dense Stoer-Wagner
    /// plus lexicographic tie-break), so there is no retry-and-union
    /// mitigation to apply. See
    /// `docs/research/nightly/2026-09-10-canonical-cactus-forgetting/README.md`
    /// for the measured determinism and latency comparison against ADR-345.
    fn boundary_indices(&self, entries: &[MemoryEntry]) -> HashSet<usize> {
        let n = entries.len();
        if n < 4 {
            return HashSet::new();
        }

        let k = self.k_neighbors.max(1);
        let graph = Arc::new(DynamicGraph::new());
        let mut any_edge = false;
        for i in 0..n {
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_sim(&entries[i].vector, &entries[j].vector)))
                .filter(|&(_, s)| s >= self.min_similarity)
                .collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims.truncate(k);
            for (j, s) in sims {
                // Mirror `RuVectorGraphAnalyzer::from_knn`: weight = 1/distance,
                // so near-duplicate pairs get heavy, cut-resistant edges.
                //
                // Deliberately unconditional on `i < j`: k-NN truncation is
                // asymmetric (i can be in j's top-k without j being in i's,
                // e.g. a low-degree "gateway" vertex whose k-nearest are all
                // same-cluster and so never lists a farther bridge, even
                // though the bridge's own short candidate list lists the
                // gateway). An `i < j` guard would silently require *both*
                // endpoints to agree, which can drop exactly the
                // low-similarity bridging edges this policy exists to find.
                // `DynamicGraph::insert_edge` is undirected and returns
                // `EdgeExists` on the second, redundant attempt, so calling
                // it from both endpoints is a correct no-op, not a bug.
                let weight = (1.0 / (1.0 - s).max(1e-4)) as f64;
                let _ = graph.insert_edge(i as u64, j as u64, weight);
                any_edge = true;
            }
        }
        if !any_edge {
            return HashSet::new();
        }

        let cactus = CactusGraph::build_from_graph(&graph);
        let cut = cactus.canonical_cut();
        let (side_a, _side_b) = &cut.partition;
        if side_a.is_empty() || side_a.len() == n {
            return HashSet::new();
        }
        let side_a_set: HashSet<usize> = side_a.iter().copied().collect();

        let mut boundary = HashSet::new();
        for edge in graph.edges() {
            let u = edge.source as usize;
            let v = edge.target as usize;
            if side_a_set.contains(&u) != side_a_set.contains(&v) {
                boundary.insert(u);
                boundary.insert(v);
            }
        }
        boundary
    }
}

impl CompactionPolicy for CactusGatedForgetting {
    fn name(&self) -> &str {
        match self.mode {
            ForgetMode::Soft => "CactusGatedForgetting-Soft",
            ForgetMode::Hard => "CactusGatedForgetting-Hard",
        }
    }

    fn select_survivors(
        &self,
        entries: &[MemoryEntry],
        target_size: usize,
        context: &[Vec<f32>],
    ) -> Vec<usize> {
        if entries.is_empty() {
            return Vec::new();
        }

        let boundary = self.boundary_indices(entries);
        let scalar = weighted_importance(entries, &self.weights, context);

        match self.mode {
            ForgetMode::Soft => {
                let mut scored: Vec<(usize, f32)> = scalar
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| {
                        let bonus = if boundary.contains(&i) {
                            self.structural_bonus
                        } else {
                            0.0
                        };
                        (i, s + bonus)
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored
                    .into_iter()
                    .take(target_size)
                    .map(|(i, _)| i)
                    .collect()
            }
            ForgetMode::Hard => {
                let protect_budget =
                    ((target_size as f32) * self.protect_fraction.clamp(0.0, 1.0)).floor() as usize;
                let protect_budget = protect_budget.min(target_size).min(boundary.len());

                let mut boundary_ranked: Vec<(usize, f32)> =
                    boundary.iter().map(|&i| (i, scalar[i])).collect();
                boundary_ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let protected: Vec<usize> = boundary_ranked
                    .into_iter()
                    .take(protect_budget)
                    .map(|(i, _)| i)
                    .collect();
                let protected_set: HashSet<usize> = protected.iter().copied().collect();

                let remaining_budget = target_size - protected.len();
                let mut rest: Vec<(usize, f32)> = (0..entries.len())
                    .filter(|i| !protected_set.contains(i))
                    .map(|i| (i, scalar[i]))
                    .collect();
                rest.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut survivors = protected;
                survivors.extend(rest.into_iter().take(remaining_budget).map(|(i, _)| i));
                survivors
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryEntry;

    fn normalize3(v: [f32; 3]) -> Vec<f32> {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        vec![v[0] / n, v[1] / n, v[2] / n]
    }

    /// Identical dataset to `graph_forget::tests::bridge_dataset` (see that
    /// module for the full topology rationale): two 9-member orthogonal
    /// clusters joined by a single degree-2 bridge vertex.
    fn bridge_dataset() -> (Vec<MemoryEntry>, usize) {
        let mut entries = Vec::new();
        let mut id = 0u64;
        let mut push = |v: Vec<f32>, access_count: u64, entries: &mut Vec<MemoryEntry>| {
            let mut e = MemoryEntry::new(id, v, 0);
            e.access_count = access_count;
            entries.push(e);
            id += 1;
        };

        for axis in 0..2 {
            let plain = if axis == 0 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let gateway = if axis == 0 {
                normalize3([1.0, 0.0, 0.5])
            } else {
                normalize3([0.0, 1.0, 0.5])
            };
            for _ in 0..8 {
                push(plain.to_vec(), 1, &mut entries);
            }
            push(gateway, 0, &mut entries);
        }
        push(vec![0.0, 0.0, 1.0], 0, &mut entries); // bridge
        let bridge_idx = entries.len() - 1;
        (entries, bridge_idx)
    }

    #[test]
    fn soft_mode_protects_the_structural_bridge_deterministically() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy = CactusGatedForgetting::soft(CoherenceWeights::default(), 1.0);
        // Single call, no retry budget (unlike ADR-345's `mincut_trials`):
        // the point of this experiment is that one call should suffice.
        for _ in 0..5 {
            let survivors = policy.select_survivors(&entries, 16, &[]);
            assert!(
                survivors.contains(&bridge_idx),
                "cactus-gated forgetting must retain the sole cross-cluster bridge on every call"
            );
        }
    }

    #[test]
    fn hard_mode_reserves_budget_for_boundary_vertices() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy = CactusGatedForgetting::hard(CoherenceWeights::default(), 0.3);
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(
            survivors.contains(&bridge_idx),
            "hard cactus-gated forgetting must protect the bridge within its reserved budget"
        );
    }

    #[test]
    fn falls_back_gracefully_below_minimum_size() {
        let entries: Vec<MemoryEntry> = (0..3)
            .map(|i| MemoryEntry::new(i, vec![i as f32, 0.0], 0))
            .collect();
        let policy = CactusGatedForgetting::soft(CoherenceWeights::default(), 1.0);
        let survivors = policy.select_survivors(&entries, 2, &[]);
        assert_eq!(survivors.len(), 2);
    }
}
