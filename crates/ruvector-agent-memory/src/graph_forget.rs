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
use ruvector_mincut::{RuVectorGraphAnalyzer, SourceAnchoredConfig, SourceAnchoredMinCut};
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

/// Which `ruvector-mincut` engine computes the boundary partition.
///
/// Added by the 2026-09-20 nightly follow-up to ADR-345, which attacked
/// that experiment's two open findings against `MincutBackend::Legacy`
/// (measured non-determinism, and a ~1,800-2,700x compaction slowdown vs.
/// baseline that made the original hypothesis fail its acceptance
/// thresholds): see
/// `docs/research/nightly/2026-09-20_canonical-mincut-forgetting/README.md`
/// and the crate's `examples/mincut_canonical_probe.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MincutBackend {
    /// `ruvector_mincut::RuVectorGraphAnalyzer` (wraps `MinCutWrapper`).
    /// Measured non-deterministic across repeated calls on an unchanged
    /// graph; mitigated here via [`MincutGatedForgetting::mincut_trials`]
    /// union-of-repeats. Kept as the default to preserve ADR-345's original,
    /// already-rejected behavior unchanged.
    #[default]
    Legacy,
    /// `ruvector_mincut::canonical::source_anchored::SourceAnchoredMinCut`
    /// (ADR-117 pseudo-deterministic canonical min-cut). Measured
    /// deterministic (identical partition across repeated calls on
    /// byte-identical input) and substantially faster at every corpus size
    /// tested; see the nightly doc above for raw numbers.
    /// [`MincutGatedForgetting::mincut_trials`] is ignored in this mode
    /// (a single call already returns the canonical, repeatable partition).
    Canonical,
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
    /// disables retrying. Ignored when `backend` is
    /// [`MincutBackend::Canonical`] (see that variant's doc).
    pub mincut_trials: usize,
    /// Which `ruvector-mincut` engine computes the boundary partition.
    /// Defaults to [`MincutBackend::Legacy`] to keep `soft()`/`hard()`
    /// behavior-identical to ADR-345; use
    /// [`Self::soft_canonical`]/[`Self::hard_canonical`] for the faster,
    /// deterministic backend evaluated by the 2026-09-20 nightly follow-up.
    pub backend: MincutBackend,
}

impl MincutGatedForgetting {
    /// [`ForgetMode::Soft`] with the given weights and bonus.
    /// Uses [`MincutBackend::Legacy`] (ADR-345's original, rejected
    /// configuration) — see [`Self::soft_canonical`] for the faster,
    /// deterministic backend.
    pub fn soft(weights: CoherenceWeights, structural_bonus: f32) -> Self {
        Self {
            weights,
            mode: ForgetMode::Soft,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus,
            protect_fraction: 0.0,
            mincut_trials: 3,
            backend: MincutBackend::Legacy,
        }
    }

    /// [`ForgetMode::Hard`] with the given weights and protected fraction.
    /// Uses [`MincutBackend::Legacy`] (ADR-345's original, rejected
    /// configuration) — see [`Self::hard_canonical`] for the faster,
    /// deterministic backend.
    pub fn hard(weights: CoherenceWeights, protect_fraction: f32) -> Self {
        Self {
            weights,
            mode: ForgetMode::Hard,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus: 0.0,
            protect_fraction,
            mincut_trials: 3,
            backend: MincutBackend::Legacy,
        }
    }

    /// [`ForgetMode::Soft`] on [`MincutBackend::Canonical`] (2026-09-20
    /// nightly follow-up to ADR-345).
    pub fn soft_canonical(weights: CoherenceWeights, structural_bonus: f32) -> Self {
        Self {
            backend: MincutBackend::Canonical,
            mincut_trials: 1,
            ..Self::soft(weights, structural_bonus)
        }
    }

    /// [`ForgetMode::Hard`] on [`MincutBackend::Canonical`] (2026-09-20
    /// nightly follow-up to ADR-345).
    pub fn hard_canonical(weights: CoherenceWeights, protect_fraction: f32) -> Self {
        Self {
            backend: MincutBackend::Canonical,
            mincut_trials: 1,
            ..Self::hard(weights, protect_fraction)
        }
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

        match self.backend {
            MincutBackend::Legacy => {
                let mut boundary = HashSet::new();
                for _ in 0..self.mincut_trials.max(1) {
                    boundary.extend(Self::boundary_from_one_partition(&neighbors));
                }
                boundary
            }
            // Deterministic backend: one call already returns the
            // repeatable canonical partition, so retrying adds cost with no
            // change in output (measured: 30/30 identical partitions on
            // byte-identical input, see `examples/mincut_canonical_probe.rs`).
            MincutBackend::Canonical => Self::boundary_from_canonical_partition(&neighbors),
        }
    }

    /// One min-cut partition attempt over an already-built k-NN graph; see
    /// [`Self::boundary_indices`]'s "Measured limitation" note for why this
    /// is called more than once.
    fn boundary_from_one_partition(neighbors: &[(usize, Vec<(usize, f64)>)]) -> HashSet<usize> {
        let mut analyzer = RuVectorGraphAnalyzer::from_knn(neighbors);
        let (side_a, side_b) = match analyzer.partition() {
            Some(p) => p,
            None => return HashSet::new(),
        };
        if side_a.is_empty() || side_b.is_empty() {
            return HashSet::new();
        }
        let side_a_set: HashSet<u64> = side_a.into_iter().collect();

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

    /// Single-call boundary computation via
    /// `ruvector_mincut::canonical::source_anchored::SourceAnchoredMinCut`
    /// (ADR-117 pseudo-deterministic canonical min-cut), given the same k-NN
    /// edge list used by [`Self::boundary_from_one_partition`]. See
    /// [`MincutBackend::Canonical`].
    fn boundary_from_canonical_partition(
        neighbors: &[(usize, Vec<(usize, f64)>)],
    ) -> HashSet<usize> {
        // Same distance-to-weight convention as `RuVectorGraphAnalyzer::from_knn`
        // (weight = 1/distance), deduplicated into an undirected edge list.
        let mut seen: HashSet<(u64, u64)> = HashSet::new();
        let mut edges: Vec<(u64, u64, f64)> = Vec::new();
        for &(i, ref nbrs) in neighbors {
            for &(j, dist) in nbrs {
                let (u, v) = if i < j {
                    (i as u64, j as u64)
                } else {
                    (j as u64, i as u64)
                };
                if seen.insert((u, v)) {
                    let weight = if dist > 0.0 { 1.0 / dist } else { 1.0 };
                    edges.push((u, v, weight));
                }
            }
        }
        if edges.is_empty() {
            return HashSet::new();
        }

        let mut engine =
            match SourceAnchoredMinCut::with_edges(edges, SourceAnchoredConfig::default()) {
                Ok(engine) => engine,
                Err(_) => return HashSet::new(),
            };
        let cut = match engine.canonical_cut() {
            Some(cut) => cut,
            None => return HashSet::new(),
        };
        if cut.side_vertices.is_empty() || cut.cut_edges.is_empty() {
            return HashSet::new();
        }

        let mut boundary = HashSet::new();
        for &(u, v) in &cut.cut_edges {
            boundary.insert(u as usize);
            boundary.insert(v as usize);
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

    #[test]
    fn falls_back_gracefully_below_minimum_size() {
        let entries: Vec<MemoryEntry> = (0..3)
            .map(|i| MemoryEntry::new(i, vec![i as f32, 0.0], 0))
            .collect();
        let policy = MincutGatedForgetting::soft(CoherenceWeights::default(), 1.0);
        let survivors = policy.select_survivors(&entries, 2, &[]);
        assert_eq!(survivors.len(), 2);
    }

    /// Canonical-backend counterpart of `soft_mode_protects_the_structural_bridge`:
    /// unlike the legacy backend, a single call already suffices (no
    /// `mincut_trials` retry needed — see `MincutBackend::Canonical`'s doc
    /// and the 2026-09-20 nightly follow-up).
    #[test]
    fn soft_canonical_protects_the_structural_bridge() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy = MincutGatedForgetting::soft_canonical(CoherenceWeights::default(), 1.0);
        assert_eq!(policy.mincut_trials, 1);
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(
            survivors.contains(&bridge_idx),
            "canonical-backend soft mincut-gated forgetting must retain the sole cross-cluster bridge"
        );
    }

    #[test]
    fn hard_canonical_reserves_budget_for_boundary_vertices() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy = MincutGatedForgetting::hard_canonical(CoherenceWeights::default(), 0.3);
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(
            survivors.contains(&bridge_idx),
            "canonical-backend hard mincut-gated forgetting must protect the bridge within its reserved budget"
        );
    }

    /// The finding this backend was added to attack: repeated calls on
    /// byte-identical input must return the identical boundary set (the
    /// legacy backend measurably does not — see `boundary_indices`'s doc).
    #[test]
    fn canonical_backend_boundary_is_deterministic_across_repeated_calls() {
        let (entries, _bridge_idx) = bridge_dataset();
        let policy = MincutGatedForgetting::soft_canonical(CoherenceWeights::default(), 1.0);
        let first = policy.boundary_indices(&entries);
        for _ in 0..9 {
            assert_eq!(
                policy.boundary_indices(&entries),
                first,
                "canonical backend must return an identical boundary set on every call"
            );
        }
    }
}
