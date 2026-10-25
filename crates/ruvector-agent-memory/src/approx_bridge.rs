//! Approximate structural-bridge detection for agent-memory compaction.
//!
//! Follow-up to [`crate::graph_forget`] (ADR-345, nightly 2026-09-05) and its
//! determinism fix (ADR-346, nightly 2026-09-11). Both of those runs used
//! `ruvector-mincut`'s exact global min-cut (`RuVectorGraphAnalyzer`) as the
//! structural "is this a bridge?" signal and found it **1,800-2,700x slower**
//! than the scalar baseline even at an 84-memory corpus (76ms-11.4s per call
//! for 50-400 vertices) — the 2026-09-11 nightly's own "Next Research" item 2
//! names this latency as "the remaining blocker" for the whole approach, and
//! item 3 of the 2026-09-05 doc's rejected-alternatives flags "approximate
//! min-cut (Karger-Stein sampling)" as untried.
//!
//! This module tries a *different* approximate family: **sampled betweenness
//! centrality** (Brandes 2001, "A faster algorithm for betweenness
//! centrality"; approximated via pivot sampling per Bader, Kintali, Madduri &
//! Mihail 2007, "Approximating Betweenness Centrality") over the same k-NN
//! cosine-similarity graph [`crate::graph_forget`] builds. A vertex that sits
//! on many shortest paths between other vertex pairs is, by construction, a
//! cut-vertex-like bridge candidate — the bridge in `graph_forget`'s own test
//! fixture (the sole degree-2 link between two 9-cliques) carries *every*
//! cross-cluster shortest path, so it has the highest possible betweenness in
//! that graph. Unlike global min-cut, betweenness from `p` sampled pivots is
//! `O(p * (V + E))`: with `p` chosen independently of `V` (see
//! [`default_pivot_count`]) this is asymptotically linear in corpus size,
//! not cubic-or-worse.
//!
//! A related, simpler signal — plain degree centrality — is exactly what
//! "Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM
//! Agents" (arXiv:2608.28978) uses for its own graph-memory pruning score,
//! and that paper reports its graph-based policy did *not* beat a flat vector
//! baseline on LongMemEval. That is a real prior against structural signals
//! helping at all, consistent with the 2026-09-05 nightly's own finding of
//! "zero measurable difference" from exact min-cut on Gaussian-cluster data
//! at n=84. This module's benchmark (`examples/approx_bridge_bench.rs`)
//! re-tests that question honestly, including the possibility of another
//! REJECT, rather than assuming betweenness fixes it.
//!
//! ## Determinism
//!
//! `graph_forget`'s own doc comment documents a real, measured
//! non-determinism bug in `RuVectorGraphAnalyzer::partition()` traced to
//! `DashMap`'s randomly-seeded iteration order (fixed upstream in
//! `ruvector-mincut` by ADR-346, but a cautionary precedent for *any* new
//! graph code in this area). This module avoids the whole category by never
//! iterating a hash-keyed structure where order affects output: adjacency is
//! `Vec<BTreeSet<usize>>` (sorted iteration), pivot selection uses a
//! seeded RNG with no retry/race, and the final ranking breaks score ties by
//! ascending vertex index. [`determinism_is_stable`] in the test module
//! below exercises this directly.

use crate::compaction::{weighted_importance, CoherenceWeights, CompactionPolicy};
use crate::memory::MemoryEntry;
use crate::scoring::cosine_sim;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::{BTreeSet, HashSet, VecDeque};

/// How the betweenness-based boundary signal is combined with the scalar
/// [`crate::compaction::CoherencePolicy`] importance score. Mirrors
/// `graph_forget::ForgetMode`'s two strategies exactly, as an independent
/// type so this module never needs the `mincut-forget` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeMode {
    /// Additive bonus on top of scalar importance, then rank normally.
    Soft,
    /// Reserve `protect_fraction` of the retained budget for the
    /// highest-scoring boundary vertices before ranking the rest.
    Hard,
}

/// A reasonable pivot-sample count for approximate betweenness: grows with
/// `sqrt(n)` (so cost stays sub-linear in the dominant `V` term of
/// `O(p * (V + E))`) but never drops below 16 (so small graphs still get a
/// meaningfully large sample) or exceeds `n` itself. This is an engineering
/// default, not a formal accuracy guarantee from the sampling literature
/// cited above — callers needing a specific error bound should pick `p`
/// themselves via [`ApproxBridgeForgetting::n_pivots`].
pub fn default_pivot_count(n: usize) -> usize {
    let scaled = (4.0 * (n as f64).sqrt()).ceil() as usize;
    scaled.max(16).min(n.max(1))
}

/// Approximate-betweenness-gated forgetting: candidate B of the follow-up
/// experiment to ADR-345/ADR-346.
#[derive(Debug, Clone)]
pub struct ApproxBridgeForgetting {
    pub weights: CoherenceWeights,
    pub mode: BridgeMode,
    /// Max neighbors per vertex when building the k-NN similarity graph.
    pub k_neighbors: usize,
    /// Minimum cosine similarity for an edge to be added.
    pub min_similarity: f32,
    /// [`BridgeMode::Soft`] only: bonus added to a boundary vertex's scalar
    /// importance score.
    pub structural_bonus: f32,
    /// [`BridgeMode::Hard`] only: fraction of `target_size` reserved for
    /// boundary vertices.
    pub protect_fraction: f32,
    /// Number of BFS pivots sampled for approximate betweenness. See
    /// [`default_pivot_count`].
    pub n_pivots: usize,
    /// Fraction of vertices (by descending betweenness score, ties broken by
    /// ascending index) treated as "boundary" / structurally load-bearing.
    pub boundary_fraction: f32,
    /// Seed for deterministic pivot sampling.
    pub seed: u64,
}

impl ApproxBridgeForgetting {
    /// [`BridgeMode::Soft`] with the given weights and bonus.
    pub fn soft(weights: CoherenceWeights, structural_bonus: f32) -> Self {
        Self {
            weights,
            mode: BridgeMode::Soft,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus,
            protect_fraction: 0.0,
            n_pivots: 0, // 0 => use default_pivot_count(n) at call time
            boundary_fraction: 0.15,
            seed: 0xA6E5_B41D_6E75,
        }
    }

    /// [`BridgeMode::Hard`] with the given weights and protected fraction.
    pub fn hard(weights: CoherenceWeights, protect_fraction: f32) -> Self {
        Self {
            weights,
            mode: BridgeMode::Hard,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus: 0.0,
            protect_fraction,
            n_pivots: 0,
            boundary_fraction: 0.15,
            seed: 0xA6E5_B41D_6E75,
        }
    }

    /// Build a symmetric k-NN cosine-similarity graph and return the indices
    /// (into `entries`) of vertices in the top `boundary_fraction` by
    /// approximate betweenness centrality.
    ///
    /// Returns an empty set when there are fewer than 4 entries or the graph
    /// has no edges at all (same fallback contract as
    /// `graph_forget::MincutGatedForgetting::boundary_indices`).
    fn boundary_indices(&self, entries: &[MemoryEntry]) -> HashSet<usize> {
        let n = entries.len();
        if n < 4 {
            return HashSet::new();
        }

        let adjacency = build_knn_adjacency(entries, self.k_neighbors.max(1), self.min_similarity);
        if adjacency.iter().all(|nbrs| nbrs.is_empty()) {
            return HashSet::new();
        }

        let pivots = select_pivots(n, self.pivot_count(n), self.seed);
        let scores = approx_betweenness(&adjacency, &pivots);

        let budget = ((n as f32) * self.boundary_fraction.clamp(0.0, 1.0))
            .ceil()
            .max(1.0) as usize;
        let mut ranked: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .filter(|&(_, score)| score > 0.0)
            .take(budget)
            .map(|(i, _)| i)
            .collect()
    }

    fn pivot_count(&self, n: usize) -> usize {
        if self.n_pivots == 0 {
            default_pivot_count(n)
        } else {
            self.n_pivots.min(n)
        }
    }
}

/// Build a symmetric adjacency list from a k-NN cosine-similarity graph over
/// `entries`. Deliberately self-contained (does not share code with
/// `graph_forget::MincutGatedForgetting::boundary_indices`'s equivalent
/// private helper) so this module compiles and is benchmarkable without the
/// `mincut-forget` feature — the same "duplicate rather than couple a
/// standalone nightly artifact to another module's internals" tradeoff
/// `examples/mincut_gated_forgetting_bench.rs` already makes explicitly for
/// its own vector-math helpers.
fn build_knn_adjacency(
    entries: &[MemoryEntry],
    k: usize,
    min_similarity: f32,
) -> Vec<BTreeSet<usize>> {
    let n = entries.len();
    let mut adjacency = vec![BTreeSet::new(); n];
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_sim(&entries[i].vector, &entries[j].vector)))
            .filter(|&(_, s)| s >= min_similarity)
            .collect();
        // Sort by descending similarity, ties broken by ascending index, so
        // truncation to `k` is deterministic regardless of float equality.
        sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        sims.truncate(k);
        for (j, _) in sims {
            adjacency[i].insert(j);
            adjacency[j].insert(i);
        }
    }
    adjacency
}

/// Pick `p` distinct pivots from `0..n` deterministically from `seed`.
/// Returned in ascending order so downstream accumulation order never
/// depends on the RNG's internal draw sequence, only on the fixed `(n, p,
/// seed)` triple.
fn select_pivots(n: usize, p: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut chosen = rand::seq::index::sample(&mut rng, n, p.min(n)).into_vec();
    chosen.sort_unstable();
    chosen
}

/// Approximate betweenness centrality via Brandes' algorithm (2001) run from
/// each vertex in `pivots` only, on an unweighted graph given as a symmetric
/// adjacency list. `O(pivots.len() * (V + E))`.
///
/// Returns one score per vertex (index = vertex id), the sum of each
/// pivot's dependency contribution. Unnormalized: only the *relative order*
/// of scores is used by this module, not their absolute scale.
fn approx_betweenness(adjacency: &[BTreeSet<usize>], pivots: &[usize]) -> Vec<f64> {
    let n = adjacency.len();
    let mut total = vec![0.0f64; n];

    for &s in pivots {
        let mut dist = vec![-1i64; n];
        let mut sigma = vec![0.0f64; n];
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut order: Vec<usize> = Vec::with_capacity(n);

        dist[s] = 0;
        sigma[s] = 1.0;
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            order.push(v);
            for &w in &adjacency[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    preds[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0f64; n];
        for &w in order.iter().rev() {
            for &v in &preds[w] {
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                total[w] += delta[w];
            }
        }
    }

    total
}

impl CompactionPolicy for ApproxBridgeForgetting {
    fn name(&self) -> &str {
        match self.mode {
            BridgeMode::Soft => "ApproxBridgeForgetting-Soft",
            BridgeMode::Hard => "ApproxBridgeForgetting-Hard",
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
            BridgeMode::Soft => {
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
            BridgeMode::Hard => {
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

    /// Identical fixture to `graph_forget::tests::bridge_dataset`: two
    /// 9-member clusters, each with one "gateway" blended toward a third,
    /// orthogonal axis, plus one "bridge" vector sitting exactly on that
    /// axis. The bridge's only two edges are to the two gateways, so every
    /// shortest path between the two clusters passes through it — maximal
    /// betweenness by construction, independent of tie-breaking.
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
    fn soft_mode_protects_the_structural_bridge() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy = ApproxBridgeForgetting::soft(CoherenceWeights::default(), 1.0);
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(
            survivors.contains(&bridge_idx),
            "soft approx-bridge forgetting must retain the sole cross-cluster bridge"
        );
    }

    #[test]
    fn hard_mode_reserves_budget_for_boundary_vertices() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy = ApproxBridgeForgetting::hard(CoherenceWeights::default(), 0.3);
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(
            survivors.contains(&bridge_idx),
            "hard approx-bridge forgetting must protect the bridge within its reserved budget"
        );
    }

    #[test]
    fn falls_back_gracefully_below_minimum_size() {
        let entries: Vec<MemoryEntry> = (0..3)
            .map(|i| MemoryEntry::new(i, vec![i as f32, 0.0], 0))
            .collect();
        let policy = ApproxBridgeForgetting::soft(CoherenceWeights::default(), 1.0);
        let survivors = policy.select_survivors(&entries, 2, &[]);
        assert_eq!(survivors.len(), 2);
    }

    /// Guards against the exact category of bug ADR-346 fixed in
    /// `ruvector-mincut`: repeated calls on a byte-identical, unchanged graph
    /// must return the identical boundary set every time.
    #[test]
    fn determinism_is_stable() {
        let (entries, _) = bridge_dataset();
        let policy = ApproxBridgeForgetting::soft(CoherenceWeights::default(), 1.0);
        let mut first: Vec<usize> = policy.boundary_indices(&entries).into_iter().collect();
        first.sort_unstable();
        for _ in 0..30 {
            let mut again: Vec<usize> = policy.boundary_indices(&entries).into_iter().collect();
            again.sort_unstable();
            assert_eq!(
                first, again,
                "boundary set must be identical across repeated calls"
            );
        }
    }

    #[test]
    fn default_pivot_count_is_bounded() {
        assert_eq!(default_pivot_count(4), 4); // n too small: capped at n
        assert!(default_pivot_count(100) >= 16);
        assert!(default_pivot_count(100) <= 100);
        assert!(default_pivot_count(10_000) < 10_000); // sublinear in n
    }

    #[test]
    fn betweenness_zero_on_a_clique() {
        // A clique has a direct edge between every pair, so no vertex lies
        // strictly between any other two: all betweenness scores are 0.
        let n = 6;
        let adjacency: Vec<BTreeSet<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).collect())
            .collect();
        let pivots: Vec<usize> = (0..n).collect();
        let scores = approx_betweenness(&adjacency, &pivots);
        for s in scores {
            assert!(s.abs() < 1e-9);
        }
    }

    #[test]
    fn betweenness_peaks_on_a_path_midpoint() {
        // Path graph 0-1-2-3-4: vertex 2 lies on every shortest path between
        // {0,1} and {3,4}, so it must have the highest betweenness.
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
        let mut adjacency = vec![BTreeSet::new(); 5];
        for (a, b) in edges {
            adjacency[a].insert(b);
            adjacency[b].insert(a);
        }
        let pivots: Vec<usize> = (0..5).collect();
        let scores = approx_betweenness(&adjacency, &pivots);
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2);
    }
}
