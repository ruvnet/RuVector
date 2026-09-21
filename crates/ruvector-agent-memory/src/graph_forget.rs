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
use ruvector_mincut::RuVectorGraphAnalyzer;
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
    /// graph, unioning the boundary vertices found each time. Retained for
    /// defense-in-depth, but no longer required for correctness: see the
    /// "Resolved" update on [`Self::boundary_indices`]'s doc comment. `1`
    /// (the default) disables retrying.
    pub mincut_trials: usize,
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
            mincut_trials: 1,
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
            mincut_trials: 1,
        }
    }

    /// Build a k-NN cosine-similarity graph and return the indices (into
    /// `entries`) of vertices with at least one neighbor edge crossing the
    /// graph's global min-cut partition.
    ///
    /// Returns an empty set when there is no usable structural signal: fewer
    /// than 4 entries, or no edges survive `min_similarity`.
    ///
    /// # Measured limitation (nightly 2026-09-05 finding) — Resolved (nightly 2026-09-21)
    ///
    /// `ruvector_mincut::RuVectorGraphAnalyzer::partition()` was NOT
    /// deterministic across repeated calls on an *identical, unchanged*
    /// graph: on a synthetic 19-vertex graph with a unique-by-construction
    /// weakest link (a degree-2 "bridge" vertex whose two edges are the only
    /// ones connecting two otherwise-disjoint 9-vertex cliques), 30 repeated
    /// `from_knn(...).partition()` calls on byte-identical input averaged
    /// 841ms/call and returned an empty side (no usable cut) in 15/30 calls
    /// (50%).
    ///
    /// The 2026-09-21 nightly root-caused this to two independent bugs in
    /// `ruvector-mincut`, both now fixed there (not worked around here):
    ///
    /// 1. `BoundedInstance::brute_force_min_cut`/`search_for_cuts`
    ///    (`instance/bounded.rs`) built their vertex enumeration/seed order
    ///    from `self.vertices: HashSet<VertexId>` iteration order, which
    ///    Rust's default hasher reseeds on every `BoundedInstance::new()`
    ///    call. Both functions pick the *first* minimum/in-range cut they
    ///    find, so a randomly-reordered enumeration silently changed which
    ///    of several equal-cost cuts was returned. Fixed by sorting both
    ///    vectors before use.
    /// 2. `WitnessHandle::materialize_partition()` (`instance/witness.rs`)
    ///    computed the cut's second side as `0..=max(membership)` minus
    ///    `membership` — i.e. it assumed the graph's highest vertex ID was
    ///    the highest ID *in the winning side*. Whenever the winning side
    ///    didn't happen to contain the graph's actual highest-ID vertex, the
    ///    "other side" came back empty instead of the rest of the real
    ///    graph, which `graph_forget`'s `side_a.is_empty() ||
    ///    side_b.is_empty()` fallback (correctly, given the bad input) then
    ///    read as "no signal". This alone accounts for most of the 50%
    ///    empty-result rate above: it doesn't require any randomness to
    ///    trigger, only that the (possibly still nondeterministic) winning
    ///    side excludes the max-ID vertex. Fixed at the one production call
    ///    site, `RuVectorGraphAnalyzer::partition()`
    ///    (`integration/mod.rs`), by computing the complement against the
    ///    analyzer's actual graph instead of trusting the witness's
    ///    self-reported range.
    ///
    /// Re-running the unmodified `examples/mincut_determinism_probe.rs` (200
    /// trials, up from 30) after both fixes: 0/200 empty results, 200/200
    /// correct boundary detection, byte-identical output on every call.
    /// Latency is unchanged (~1.0s/call on this probe's 19-vertex graph,
    /// dominated by `brute_force_min_cut`'s O(2^n) exhaustive search, not by
    /// either bug) — still an open, separately-tracked cost, not addressed
    /// here. See `docs/research/nightly/2026-09-05-mincut-gated-forgetting/`
    /// and `docs/research/nightly/2026-09-21-mincut-determinism-fix/`.
    ///
    /// [`Self::mincut_trials`] (now defaulting to `1`) is kept only as
    /// defense-in-depth: with the above fixed, repeating an identical query
    /// against an unchanged graph returns an identical answer, so trials
    /// beyond the first are pure overhead, not added correctness.
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
            boundary.extend(Self::boundary_from_one_partition(&neighbors));
        }
        boundary
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
        policy.mincut_trials = 1;
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
        policy.mincut_trials = 1;
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
}
