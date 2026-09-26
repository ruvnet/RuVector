//! Approximate-mincut-gated forgetting: a follow-up to [`crate::graph_forget`]
//! (ADR-345, nightly 2026-09-05) attacking the bottleneck that experiment
//! identified but did not fix — swapping `ruvector_mincut::RuVectorGraphAnalyzer`
//! (measured 1,800-2,700x slower than the scalar baseline at an 84-memory
//! corpus, per the 2026-09-05 nightly's scaling probe) for
//! `ruvector_mincut::ApproxMinCut`, the crate's own "(1+ε)-approximate min-cut
//! for all cut sizes" entry point (docstring cites SODA 2025, arXiv:2412.15069)
//! that the earlier experiment never tried.
//!
//! Same k-NN cosine-similarity graph construction and the same
//! Soft/Hard boundary-signal combination as [`crate::graph_forget`] — this
//! module isolates the *engine* as the only variable, so any difference in
//! bridge-survival or latency between the two policies is attributable to
//! `RuVectorGraphAnalyzer` vs. `ApproxMinCut`, not to a different graph or a
//! different scoring rule.
//!
//! # Measured limitation (nightly 2026-09-19 finding)
//!
//! `ApproxMinCut::min_cut()`'s returned `partition` field is **not derived
//! from the cut it just computed**: `ApproxMinCut::compute_partition` ignores
//! its `cut_value` argument entirely and instead does a plain BFS from an
//! arbitrary starting vertex, stopping once it has visited half the vertices
//! (`crates/ruvector-mincut/src/algorithm/approximate.rs`,
//! `compute_partition`). This makes the exposed partition unrelated to the
//! actual (approximate) minimum cut for any graph with more than one valid
//! near-half bisection — which is the general case, not a corner case. See
//! `examples/approx_mincut_partition_probe.rs` for a minimal, executable
//! demonstration on the classic "two triangles joined by a bridge" graph, and
//! the nightly research doc for the corpus-level bridge-survival numbers this
//! produces. This module still ships the integration (Darwin's job is to
//! evolve *within* a bounded scope, not to silently work around a dependency
//! bug), but reports the resulting correctness gap honestly rather than
//! hiding it behind a favorable latency number.

use crate::compaction::{weighted_importance, CoherenceWeights, CompactionPolicy};
use crate::memory::MemoryEntry;
use crate::scoring::cosine_sim;
use ruvector_mincut::ApproxMinCut;
use std::collections::{HashMap, HashSet};

/// How the approximate-mincut boundary signal is combined with the scalar
/// [`crate::compaction::CoherencePolicy`] importance score. Mirrors
/// [`crate::graph_forget::ForgetMode`] exactly so the two engines are
/// comparable at the policy level, not just the graph level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproxForgetMode {
    /// Additive bonus on top of scalar importance, then rank normally.
    Soft,
    /// Reserve `protect_fraction` of the retained budget for the
    /// highest-scoring boundary vertices before ranking the rest.
    Hard,
}

/// Approximate-mincut-gated forgetting compaction policy (nightly
/// 2026-09-19 follow-up to ADR-345).
#[derive(Debug, Clone)]
pub struct ApproxMincutForgetting {
    pub weights: CoherenceWeights,
    pub mode: ApproxForgetMode,
    /// Max neighbors per vertex when building the similarity graph.
    pub k_neighbors: usize,
    /// Minimum cosine similarity for an edge to be added.
    pub min_similarity: f32,
    /// [`ApproxForgetMode::Soft`] only: bonus added to a boundary vertex's
    /// scalar importance score.
    pub structural_bonus: f32,
    /// [`ApproxForgetMode::Hard`] only: fraction of `target_size` reserved
    /// for boundary vertices.
    pub protect_fraction: f32,
    /// Approximation parameter passed to `ApproxMinCut` (0 < ε ≤ 1).
    pub epsilon: f64,
}

impl ApproxMincutForgetting {
    /// [`ApproxForgetMode::Soft`] with the given weights and bonus.
    pub fn soft(weights: CoherenceWeights, structural_bonus: f32) -> Self {
        Self {
            weights,
            mode: ApproxForgetMode::Soft,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus,
            protect_fraction: 0.0,
            epsilon: 0.1,
        }
    }

    /// [`ApproxForgetMode::Hard`] with the given weights and protected
    /// fraction.
    pub fn hard(weights: CoherenceWeights, protect_fraction: f32) -> Self {
        Self {
            weights,
            mode: ApproxForgetMode::Hard,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus: 0.0,
            protect_fraction,
            epsilon: 0.1,
        }
    }

    /// Build the deduplicated undirected k-NN edge set feeding both mincut
    /// engines: `(vertex_i, vertex_j, weight)` with `weight = 1/distance`,
    /// matching `ruvector_mincut::RuVectorGraphAnalyzer::from_knn`'s
    /// convention exactly (see [`crate::graph_forget`]) so a latency or
    /// survival difference cannot be explained by a difference in the input
    /// graph. Ties are broken "first vertex's neighbor list wins", the same
    /// semantics `from_knn` gets implicitly from `DynamicGraph::insert_edge`
    /// rejecting the second, reciprocal insert of an already-present edge.
    fn dedup_knn_edges(&self, entries: &[MemoryEntry]) -> Vec<(usize, usize, f64)> {
        let n = entries.len();
        let k = self.k_neighbors.max(1);
        let mut edge_map: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..n {
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_sim(&entries[i].vector, &entries[j].vector)))
                .filter(|&(_, s)| s >= self.min_similarity)
                .collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims.truncate(k);
            for (j, s) in sims {
                let dist = (1.0 - s).max(1e-4) as f64;
                let weight = 1.0 / dist;
                let key = (i.min(j), i.max(j));
                edge_map.entry(key).or_insert(weight);
            }
        }
        edge_map.into_iter().map(|((i, j), w)| (i, j, w)).collect()
    }

    /// Vertices with at least one neighbor edge crossing `ApproxMinCut`'s
    /// reported partition. Empty when the corpus is too small (`< 4`
    /// entries), the graph has no edges above `min_similarity`, or
    /// `ApproxMinCut` reports a degenerate (empty-sided) partition.
    fn boundary_indices(&self, entries: &[MemoryEntry]) -> HashSet<usize> {
        if entries.len() < 4 {
            return HashSet::new();
        }
        let edges = self.dedup_knn_edges(entries);
        if edges.is_empty() {
            return HashSet::new();
        }

        let mut approx = ApproxMinCut::with_epsilon(self.epsilon);
        for &(i, j, w) in &edges {
            approx.insert_edge(i as u64, j as u64, w);
        }
        let result = approx.min_cut();

        let side_a = match result.partition {
            Some((side_a, side_b)) if !side_a.is_empty() && !side_b.is_empty() => side_a,
            _ => return HashSet::new(),
        };
        let side_a_set: HashSet<u64> = side_a.into_iter().collect();

        let mut boundary = HashSet::new();
        for &(i, j, _) in &edges {
            let i_in_a = side_a_set.contains(&(i as u64));
            let j_in_a = side_a_set.contains(&(j as u64));
            if i_in_a != j_in_a {
                boundary.insert(i);
                boundary.insert(j);
            }
        }
        boundary
    }
}

impl CompactionPolicy for ApproxMincutForgetting {
    fn name(&self) -> &str {
        match self.mode {
            ApproxForgetMode::Soft => "ApproxMincutForgetting-Soft",
            ApproxForgetMode::Hard => "ApproxMincutForgetting-Hard",
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
            ApproxForgetMode::Soft => {
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
            ApproxForgetMode::Hard => {
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

    /// Identical construction to `graph_forget::tests::bridge_dataset` (see
    /// that function's doc comment for the full rationale): two 9-member
    /// clusters plus a bridge vertex whose only two edges are the unambiguous
    /// cheapest cut of the graph. Duplicated rather than shared so this
    /// module's tests do not depend on `graph_forget`'s private test helpers,
    /// and so the two engines are exercised against byte-identical input.
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

    /// Measured result (nightly 2026-09-19): this test's assertion was
    /// originally written to expect the bridge to *unreliably* survive, on
    /// the theory that `ApproxMinCut`'s partition is essentially arbitrary.
    /// Running it showed the opposite in isolation: the bridge survived on
    /// every one of 5 epsilons tried, in that one process. Investigating why
    /// (rather than loosening the assertion to fit) found the true cause,
    /// and a second, independent problem this experiment did not originally
    /// go looking for: `ApproxMinCut`'s internal `vertices: HashSet<VertexId>`
    /// uses Rust's default *randomized* hasher, and `compute_partition`'s BFS
    /// start vertex is `self.vertices.iter().next()` — so the outcome
    /// depends on this process's hasher seed, not just on the graph. This
    /// dataset (two exactly-9-vertex clusters + a 19th bridge vertex) also
    /// happens to make a "visited half of 19" BFS stop near a true cluster
    /// boundary regardless of start vertex, so many (not all — confirmed by
    /// running this exact assertion across several separate `cargo test`
    /// process invocations during development, which did occasionally fail)
    /// hasher seeds land on the correct-looking answer by construction, the
    /// same coincidence `examples/approx_mincut_partition_probe.rs`'s
    /// "balanced" graph demonstrates deliberately. Because the outcome
    /// depends on this process's random hasher state, an assertion pinned to
    /// one specific outcome is inherently flaky (confirmed: this exact test,
    /// in its original form, failed 2 of 6 separate process invocations
    /// tried during development) and would be a false signal in CI either
    /// way. This test instead asserts the invariant that *does* hold
    /// regardless of hasher state (a well-formed survivor set of the
    /// requested size) and documents the underlying non-determinism finding
    /// here and in the nightly research doc's "Non-determinism" section,
    /// where it is characterized properly via repeated in-process trials in
    /// `examples/approx_mincut_forgetting_bench.rs` instead of a single
    /// pass/fail unit-test assertion.
    #[test]
    fn soft_mode_runs_without_panicking_on_bridge_dataset() {
        let (entries, bridge_idx) = bridge_dataset();
        let scalar = weighted_importance(&entries, &CoherenceWeights::default(), &[]);
        assert_eq!(scalar[bridge_idx], 0.0);

        let epsilons = [0.05, 0.1, 0.2, 0.3, 0.5];
        for &eps in &epsilons {
            let mut policy = ApproxMincutForgetting::soft(CoherenceWeights::default(), 1.0);
            policy.epsilon = eps;
            let survivors = policy.select_survivors(&entries, 16, &[]);
            assert_eq!(
                survivors.len(),
                16,
                "epsilon={eps}: policy must always return exactly target_size survivors"
            );
        }
    }

    #[test]
    fn falls_back_gracefully_below_minimum_size() {
        let entries: Vec<MemoryEntry> = (0..3)
            .map(|i| MemoryEntry::new(i, vec![i as f32, 0.0], 0))
            .collect();
        let policy = ApproxMincutForgetting::soft(CoherenceWeights::default(), 1.0);
        let survivors = policy.select_survivors(&entries, 2, &[]);
        assert_eq!(survivors.len(), 2);
    }

    #[test]
    fn hard_mode_runs_without_panicking_on_bridge_dataset() {
        let (entries, _bridge_idx) = bridge_dataset();
        let policy = ApproxMincutForgetting::hard(CoherenceWeights::default(), 0.3);
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert_eq!(survivors.len(), 16);
    }
}
