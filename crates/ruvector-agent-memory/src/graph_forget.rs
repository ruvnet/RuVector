//! Mincut-gated forgetting: a structural signal for agent-memory compaction
//! (ADR-345, docs/research/nightly/2026-09-05-mincut-gated-forgetting).
//!
//! [`crate::compaction::CoherencePolicy`] scores every memory *independently*:
//! `I = alpha*recency + beta*frequency + gamma*coherence`. It has no notion of
//! *structure* — a "bridge" memory that is the only semantic link between two
//! otherwise well-separated topic clusters can score low on all three terms
//! (rarely accessed, old, off-topic vs. the current context window) and get
//! evicted, silently disconnecting the surviving store even though
//! cross-cluster retrieval and `fusion::CausalEpisodicGraph` connectivity
//! depend on such bridges surviving.
//!
//! This module builds a k-NN cosine-similarity graph over the compaction
//! candidates and hands it to `ruvector-mincut`'s
//! [`ruvector_mincut::RuVectorGraphAnalyzer`] (the crate's own vector-graph
//! integration layer — reused as-is, not reimplemented) to find the graph's
//! global minimum-cut partition. Every memory with at least one neighbor edge
//! crossing that partition is treated as *structurally load-bearing*. Two
//! policies use the signal differently:
//!
//! - [`ForgetMode::Soft`]: add a fixed bonus to a boundary vertex's scalar
//!   [`crate::compaction::weighted_importance`] before ranking.
//! - [`ForgetMode::Hard`]: reserve up to `protect_fraction` of the retained
//!   budget for boundary vertices (highest scalar score first), then fill the
//!   rest by scalar importance exactly like `CoherencePolicy`.
//!
//! Both fall back to plain `CoherencePolicy` behavior when the corpus is too
//! small (`< 4` entries) or the similarity graph has no crossing edges (e.g.
//! it is already disconnected, or every pair is above/below threshold
//! uniformly) — there is no boundary signal to add in that case.

use crate::compaction::{weighted_importance, CoherenceWeights, CompactionPolicy};
use crate::memory::MemoryEntry;
use crate::scoring::cosine_sim;
use ruvector_mincut::{DynamicGraph, DynamicMinCut, MinCutConfig, RuVectorGraphAnalyzer};
use std::collections::HashSet;

/// How the mincut-boundary structural signal is combined with the scalar
/// [`crate::compaction::CoherencePolicy`] importance score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForgetMode {
    /// Additive bonus on top of scalar importance, then rank normally.
    Soft,
    /// Reserve `protect_fraction` of the retained budget for the
    /// highest-scoring boundary vertices before ranking the rest.
    Hard,
}

/// Which `ruvector-mincut` API computes the boundary partition.
///
/// # Background (nightly 2026-09-28, following up ADR-345 / ADR-346)
///
/// [`MincutBackend::GraphAnalyzer`] (the original, and still the default —
/// changing the default is a separate decision from adding the option) uses
/// [`RuVectorGraphAnalyzer`], which delegates to `MinCutWrapper`'s
/// subpolynomial dynamic algorithm: a geometric ladder of `BoundedInstance`s
/// sized for graphs under *repeated* edge insert/delete churn. Every
/// `partition()` call here is a one-shot query against a graph built once
/// and never mutated again, so that machinery's amortized-update guarantees
/// buy nothing and its per-call setup cost (documented at ADR-345:
/// 76ms-11.4s for 50-400 vertices, ~1,800-2,700x the scalar baseline at this
/// module's own 84-entry benchmark corpus) is pure overhead here.
///
/// [`MincutBackend::DynamicMinCut`] instead uses the crate's *other*,
/// lower-level exact solver: [`DynamicMinCut`]
/// (`ruvector_mincut::algorithm::DynamicMinCut`), a sparse Stoer-Wagner
/// implementation (`algorithm::exact::minimum_cut`, O(n) phases over a
/// max-adjacency heap, O(n + m) storage) that computes its cut once at
/// construction (`DynamicMinCut::from_graph`) and serves `partition()` from
/// that cached result — no geometric instance ladder, and no `DashMap`
/// iteration-order dependency (ADR-346 root-caused that determinism bug in
/// the *`RuVectorGraphAnalyzer`* code path specifically: `DynamicGraph`'s
/// `vertices()`/`edges()`; `DynamicMinCut::recompute_min_cut` separately
/// sorts its vertex and edge lists before indexing them, so it does not
/// inherit that bug even though it walks the same `DynamicGraph` type).
/// See `examples/mincut_scaling_probe.rs` for a head-to-head latency
/// comparison of both backends at matched graph sizes, and
/// `docs/research/nightly/2026-09-28-mincut-backend-latency/README.md` for
/// the full write-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MincutBackend {
    /// `RuVectorGraphAnalyzer` / `MinCutWrapper` (original ADR-345 backend).
    #[default]
    GraphAnalyzer,
    /// `ruvector_mincut::DynamicMinCut` (sparse Stoer-Wagner, one-shot).
    DynamicMinCut,
}

/// Mincut-gated forgetting compaction policy (candidates A/B of the nightly
/// 2026-09-05 experiment).
#[derive(Debug, Clone)]
pub struct MincutGatedForgetting {
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
    /// Number of times to recompute the min-cut partition on an unchanged
    /// graph, unioning the boundary vertices found each time (see the
    /// "Measured limitation" note on [`Self::boundary_indices`]). `1`
    /// disables retrying.
    pub mincut_trials: usize,
    /// Which `ruvector-mincut` API computes the boundary partition. See
    /// [`MincutBackend`].
    pub backend: MincutBackend,
}

impl MincutGatedForgetting {
    /// [`ForgetMode::Soft`] with the given weights and bonus.
    pub fn soft(weights: CoherenceWeights, structural_bonus: f32) -> Self {
        Self {
            weights,
            mode: ForgetMode::Soft,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus,
            protect_fraction: 0.0,
            mincut_trials: 3,
            backend: MincutBackend::default(),
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
            mincut_trials: 3,
            backend: MincutBackend::default(),
        }
    }

    /// Use `backend` for boundary-partition computation instead of the
    /// default. See [`MincutBackend`].
    pub fn with_backend(mut self, backend: MincutBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Build a k-NN cosine-similarity graph and return the indices (into
    /// `entries`) of vertices with at least one neighbor edge crossing the
    /// graph's global min-cut partition.
    ///
    /// Returns an empty set when there is no usable structural signal: fewer
    /// than 4 entries, or no edges survive `min_similarity`.
    ///
    /// # Measured limitation (nightly 2026-09-05 finding)
    ///
    /// `ruvector_mincut::RuVectorGraphAnalyzer::partition()` is NOT
    /// deterministic across repeated calls on an *identical, unchanged*
    /// graph, and is expensive even on tiny graphs: on a synthetic 19-vertex
    /// graph with a unique-by-construction weakest link (a degree-2 "bridge"
    /// vertex whose two edges are the only ones connecting two
    /// otherwise-disjoint 9-vertex cliques), 30 repeated
    /// `from_knn(...).partition()` calls on byte-identical input averaged
    /// 841ms/call and returned an empty side (no usable cut) in 15/30 calls
    /// (50%); of the 15 non-empty calls, the bridge was correctly flagged as
    /// boundary in all 15 (the two valid minimum-cut partitions both cross at
    /// least one of the bridge's two edges in this particular topology, so
    /// this graph cannot distinguish "wrong partition" from "empty result",
    /// only "no signal" from "some signal"). This is consistent with internal
    /// tie-breaking that depends on hash-map iteration order rather than any
    /// property of the graph, not with an intentional randomized algorithm
    /// (no direct `rand` usage was found in `ruvector-mincut`'s
    /// instance/witness/algorithm modules). See
    /// `examples/mincut_determinism_probe.rs` (exact reproduction of the
    /// numbers above) and
    /// `docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md`
    /// ("Failure modes", which also covers latency scaling up to 400
    /// vertices). Filed as a follow-up hardening item against
    /// `ruvector-mincut` rather than worked around there.
    ///
    /// This method mitigates it locally by taking the union of boundary
    /// vertices found across [`Self::mincut_trials`] independent calls: a
    /// vertex flagged as boundary in *any* trial genuinely does sit on some
    /// minimum cut of the graph, so the union only adds true positives (never
    /// false ones) at the cost of also protecting bystander vertices caught
    /// by an alternate, equally-valid partition.
    /// Number of `entries` currently receiving the structural signal (i.e.
    /// `self.boundary_indices(entries).len()`), exposed so a caller can
    /// observe whether the k-NN graph produced *any* boundary at all before
    /// relying on [`ForgetMode::Soft`]/[`ForgetMode::Hard`] to act on it.
    ///
    /// # Measured limitation (nightly 2026-09-28 finding, ADR-350)
    ///
    /// This is not a rare-edge-case concern: at the 84-entry benchmark
    /// corpus in `examples/mincut_gated_forgetting_bench.rs`, the boundary
    /// set is non-empty (6/84) but the bonus still moves zero survivors
    /// relative to baseline. At 10x and 50x that corpus (840 and 4,200
    /// entries, `examples/mincut_gated_forgetting_scale_probe.rs`, same
    /// generator, same seed family, only feasible to test at all once the
    /// [`MincutBackend::DynamicMinCut`] backend made the mincut call itself
    /// cheap), the boundary set was empty at *every* sampled scale — not
    /// because there is no cross-cluster structure to find, but because
    /// [`Self::boundary_indices`] returns an empty set whenever the k-NN
    /// graph as a whole is disconnected (an isolated low-degree bridge
    /// vertex, whose nearest neighbors by cosine similarity all fall below
    /// `min_similarity`, is a likely-enough event per bridge that a larger
    /// bridge count makes it near-certain at least one occurs), silently
    /// disabling the structural signal for the *entire* compaction pass,
    /// not just the isolated vertex. See
    /// `docs/research/nightly/2026-09-28-mincut-backend-latency/README.md`.
    pub fn boundary_size(&self, entries: &[MemoryEntry]) -> usize {
        self.boundary_indices(entries).len()
    }

    fn boundary_indices(&self, entries: &[MemoryEntry]) -> HashSet<usize> {
        let n = entries.len();
        if n < 4 {
            return HashSet::new();
        }

        let k = self.k_neighbors.max(1);
        let neighbors: Vec<(usize, Vec<(usize, f64)>)> = (0..n)
            .map(|i| {
                let mut sims: Vec<(usize, f32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, cosine_sim(&entries[i].vector, &entries[j].vector)))
                    .filter(|&(_, s)| s >= self.min_similarity)
                    .collect();
                sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                sims.truncate(k);
                // `from_knn` treats the second tuple element as a *distance*
                // (weight = 1/distance): invert similarity so near-duplicate
                // pairs get heavy, cut-resistant edges.
                let dists = sims
                    .into_iter()
                    .map(|(j, s)| (j, (1.0 - s).max(1e-4) as f64))
                    .collect();
                (i, dists)
            })
            .collect();

        if neighbors.iter().all(|(_, nbrs)| nbrs.is_empty()) {
            return HashSet::new();
        }

        let mut boundary = HashSet::new();
        for _ in 0..self.mincut_trials.max(1) {
            boundary.extend(match self.backend {
                MincutBackend::GraphAnalyzer => Self::boundary_from_one_partition(&neighbors),
                MincutBackend::DynamicMinCut => Self::boundary_from_dynamic_mincut(&neighbors),
            });
        }
        boundary
    }

    /// One min-cut partition attempt over an already-built k-NN graph using
    /// [`RuVectorGraphAnalyzer`]; see [`Self::boundary_indices`]'s "Measured
    /// limitation" note for why this is called more than once.
    fn boundary_from_one_partition(neighbors: &[(usize, Vec<(usize, f64)>)]) -> HashSet<usize> {
        let mut analyzer = RuVectorGraphAnalyzer::from_knn(neighbors);
        let (side_a, side_b) = match analyzer.partition() {
            Some(p) => p,
            None => return HashSet::new(),
        };
        Self::boundary_from_sides(neighbors, side_a, side_b)
    }

    /// Same boundary computation as [`Self::boundary_from_one_partition`],
    /// but via [`MincutBackend::DynamicMinCut`] — see that variant's doc
    /// comment for why this is expected to be far cheaper for the one-shot,
    /// never-mutated-again graphs this module builds per compaction call.
    /// Builds its own `DynamicGraph` (mirroring exactly what
    /// `RuVectorGraphAnalyzer::from_knn` does: one `insert_edge` per (i, j,
    /// 1/distance) neighbor pair, duplicates silently dropped by
    /// `DynamicGraph::insert_edge`'s own `EdgeExists` handling) rather than
    /// reusing `RuVectorGraphAnalyzer::from_knn`'s construction, since
    /// `DynamicMinCut::from_graph` takes an owned `DynamicGraph`, not the
    /// `Arc<DynamicGraph>` `RuVectorGraphAnalyzer` holds.
    fn boundary_from_dynamic_mincut(neighbors: &[(usize, Vec<(usize, f64)>)]) -> HashSet<usize> {
        let graph = DynamicGraph::new();
        for (i, nbrs) in neighbors {
            for &(j, dist) in nbrs {
                let weight = if dist > 0.0 { 1.0 / dist } else { 1.0 };
                let _ = graph.insert_edge(*i as u64, j as u64, weight);
            }
        }
        let mincut = match DynamicMinCut::from_graph(graph, MinCutConfig::default()) {
            Ok(m) => m,
            Err(_) => return HashSet::new(),
        };
        if !mincut.is_connected() {
            return HashSet::new();
        }
        let (side_a, side_b) = mincut.partition();
        Self::boundary_from_sides(neighbors, side_a, side_b)
    }

    /// Shared "which vertices touch a crossing edge" step for both backends.
    fn boundary_from_sides(
        neighbors: &[(usize, Vec<(usize, f64)>)],
        side_a: Vec<u64>,
        side_b: Vec<u64>,
    ) -> HashSet<usize> {
        if side_a.is_empty() || side_b.is_empty() {
            return HashSet::new();
        }
        let side_a_set: HashSet<u64> = side_a.into_iter().collect();
        let _ = side_b;

        let mut boundary = HashSet::new();
        for (i, nbrs) in neighbors {
            let i_in_a = side_a_set.contains(&(*i as u64));
            for &(j, _) in nbrs {
                let j_in_a = side_a_set.contains(&(j as u64));
                if i_in_a != j_in_a {
                    boundary.insert(*i);
                    boundary.insert(j);
                }
            }
        }
        boundary
    }
}

impl CompactionPolicy for MincutGatedForgetting {
    fn name(&self) -> &str {
        match self.mode {
            ForgetMode::Soft => "MincutGatedForgetting-Soft",
            ForgetMode::Hard => "MincutGatedForgetting-Hard",
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

    /// Two 9-member clusters along orthogonal axes (`[1,0,0]` / `[0,1,0]`),
    /// each with one "gateway" member blended slightly toward a third,
    /// orthogonal "bridge" axis (`[0,0,1]`). The bridge vector sits exactly
    /// on that third axis: cosine to a gateway is ~0.45 (passes
    /// `min_similarity`), cosine to every plain member and to the other
    /// cluster is exactly 0 (filtered out). So the bridge's only two graph
    /// edges are to the two gateways — a strictly lower degree (2) than any
    /// plain member's (~8, one per same-cluster peer) — making "cut the
    /// bridge's 1-2 edges" the unambiguous cheapest separation of the graph,
    /// independent of tie-breaking among same-score plain members.
    ///
    /// Ordinary members get `access_count = 1`; the bridge and both gateways
    /// get `access_count = 0`, so they are the lowest-scoring group under
    /// plain [`weighted_importance`] (recency and coherence are equalized:
    /// identical timestamps, empty context) — exactly the entries a
    /// scalar-only policy would evict first.
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
        // Sanity: under plain scalar importance the bridge ties for last
        // place (score 0, alongside the two gateways) — a plausible eviction
        // target for any purely scalar policy.
        let scalar = weighted_importance(&entries, &CoherenceWeights::default(), &[]);
        assert_eq!(scalar[bridge_idx], 0.0);

        let mut policy = MincutGatedForgetting::soft(CoherenceWeights::default(), 1.0);
        // Raised from the default 3: this dataset has two equal-cost minimum
        // cuts (see the "Measured limitation" doc on `boundary_indices`), so
        // a single low-trial-count run can occasionally miss the boundary
        // signal by chance; 10 keeps this deterministic unit test's flake
        // rate negligible at a runtime cost that is fine for `cargo test`
        // (19 vertices, not the multi-second cost measured at production
        // scale).
        policy.mincut_trials = 10;
        // Evict 3 of 19: exactly the tied-lowest group (bridge + 2 gateways)
        // under the scalar baseline.
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(
            survivors.contains(&bridge_idx),
            "soft mincut-gated forgetting must retain the sole cross-cluster bridge"
        );
    }

    #[test]
    fn hard_mode_reserves_budget_for_boundary_vertices() {
        let (entries, bridge_idx) = bridge_dataset();
        let mut policy = MincutGatedForgetting::hard(CoherenceWeights::default(), 0.3);
        policy.mincut_trials = 10; // see the sibling soft-mode test's comment
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(
            survivors.contains(&bridge_idx),
            "hard mincut-gated forgetting must protect the bridge within its reserved budget"
        );
    }

    /// Same scenario as [`soft_mode_protects_the_structural_bridge`], but
    /// with [`MincutBackend::DynamicMinCut`] instead of the default
    /// [`MincutBackend::GraphAnalyzer`] — confirms the new backend's
    /// boundary detection is not silently empty (a real risk worth a
    /// dedicated positive test, since `boundary_indices` treats "no signal"
    /// and "not evicted anyway" identically at the `select_survivors` level,
    /// which is exactly what the nightly 2026-09-28 acceptance benchmark's
    /// 0.0pp gap on both backends could otherwise be confused with a
    /// boundary-detection bug rather than a genuine "no effect at this
    /// corpus" finding).
    #[test]
    fn soft_mode_dynamic_mincut_backend_protects_the_structural_bridge() {
        let (entries, bridge_idx) = bridge_dataset();
        let mut policy = MincutGatedForgetting::soft(CoherenceWeights::default(), 1.0)
            .with_backend(MincutBackend::DynamicMinCut);
        policy.mincut_trials = 10; // see the GraphAnalyzer-backend test's comment
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(
            survivors.contains(&bridge_idx),
            "DynamicMinCut-backed soft mincut-gated forgetting must retain the sole cross-cluster bridge"
        );
    }

    /// Direct check that [`MincutGatedForgetting::boundary_indices`] returns
    /// a *non-empty* boundary set containing the bridge under
    /// [`MincutBackend::DynamicMinCut`] on a single deterministic trial (no
    /// retry union) — isolates boundary detection itself from the
    /// downstream ranking logic the two tests above also exercise.
    #[test]
    fn dynamic_mincut_backend_finds_nonempty_boundary_in_one_trial() {
        let (entries, bridge_idx) = bridge_dataset();
        let mut policy = MincutGatedForgetting::soft(CoherenceWeights::default(), 1.0)
            .with_backend(MincutBackend::DynamicMinCut);
        policy.mincut_trials = 1;
        let boundary = policy.boundary_indices(&entries);
        assert!(!boundary.is_empty(), "boundary set must not be empty");
        assert!(
            boundary.contains(&bridge_idx),
            "the bridge vertex must be flagged as boundary"
        );
    }

    #[test]
    fn boundary_size_reports_the_same_count_boundary_indices_finds() {
        let (entries, _bridge_idx) = bridge_dataset();
        let mut policy = MincutGatedForgetting::soft(CoherenceWeights::default(), 1.0)
            .with_backend(MincutBackend::DynamicMinCut);
        policy.mincut_trials = 1;
        assert_eq!(
            policy.boundary_size(&entries),
            policy.boundary_indices(&entries).len()
        );
        assert!(policy.boundary_size(&entries) > 0);
    }

    #[test]
    fn falls_back_gracefully_below_minimum_size() {
        let entries: Vec<MemoryEntry> = (0..3)
            .map(|i| MemoryEntry::new(i, vec![i as f32, 0.0], 0))
            .collect();
        let policy = MincutGatedForgetting::soft(CoherenceWeights::default(), 1.0);
        let survivors = policy.select_survivors(&entries, 2, &[]);
        assert_eq!(survivors.len(), 2);
    }
}
