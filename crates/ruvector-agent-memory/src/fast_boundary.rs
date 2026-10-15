//! Fast, dependency-light structural-boundary proxies for agent-memory
//! compaction (ADR-353, nightly `docs/research/nightly/2026_10_15_fast-boundary-forgetting`).
//!
//! [`crate::graph_forget::MincutGatedForgetting`] (ADR-345/346) showed that an
//! exact minimum-cut boundary signal protects structural "bridge" memories
//! that a scalar-only [`crate::compaction::CoherencePolicy`] would evict, but
//! the 2026-09-05 nightly rejected it for production use because
//! `ruvector_mincut::RuVectorGraphAnalyzer::partition()` is 1,800-2,700x
//! slower than the scalar baseline even on an 84-memory corpus and scales to
//! multi-second/call latency by n=400 (`examples/mincut_scaling_probe.rs`) —
//! infeasible at the 500-2,000-memory corpus sizes a real agent deployment
//! would compact.
//!
//! This module asks whether a *cheap* local-structure proxy — one that never
//! calls a global min-cut solver at all — recovers enough of the same signal
//! to be useful where exact mincut cannot run. It implements two independent
//! O(V·k) / O(rounds·V·k) heuristics over the same k-NN similarity graph
//! [`crate::graph_forget`] builds:
//!
//! - [`BoundaryHeuristic::ClusteringCoefficient`] (candidate A): a vertex
//!   with few triangles among its neighbors (low local clustering
//!   coefficient) tends to be a bridge between otherwise-separate
//!   neighborhoods — a classical social-network-analysis proxy for
//!   "structural hole" vertices, computed with no global graph algorithm.
//! - [`BoundaryHeuristic::LabelPropagation`] (candidate B): a few
//!   synchronous rounds of label-propagation community detection, then flag
//!   any vertex whose neighbors do not agree on a single label as sitting on
//!   a community boundary.
//!
//! Both heuristics build their k-NN adjacency by sorting `(similarity, index)`
//! pairs — never iterating a hash map — so the result is deterministic by
//! construction, with none of the `DashMap`-iteration-order non-determinism
//! ADR-346 had to fix downstream of `ruvector-mincut`.
//!
//! This module has **no dependency on `ruvector-mincut`** and compiles under
//! the `fast-boundary-forget` feature alone — the `mincut-forget` feature is
//! only needed to *compare against* the exact-mincut baseline (see
//! `examples/boundary_fidelity_and_scale_bench.rs`), not to use these
//! policies in production.

use crate::compaction::{weighted_importance, CoherenceWeights, CompactionPolicy};
use crate::memory::MemoryEntry;
use crate::scoring::cosine_sim;
use std::collections::{HashMap, HashSet};

/// Which cheap local-structure proxy flags "bridge-like" vertices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryHeuristic {
    /// Candidate A: local clustering coefficient.
    ClusteringCoefficient,
    /// Candidate B: label-propagation community boundary.
    LabelPropagation,
}

/// How the fast boundary signal combines with scalar importance — the same
/// semantics as [`crate::graph_forget::ForgetMode`], duplicated locally so
/// this module carries no dependency on the `mincut-forget` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForgetMode {
    /// Additive bonus on top of scalar importance, then rank normally.
    Soft,
    /// Reserve `protect_fraction` of the retained budget for the
    /// highest-scoring boundary vertices before ranking the rest.
    Hard,
}

/// Fast boundary-gated forgetting compaction policy (candidates A/B of the
/// 2026-10-15 nightly).
#[derive(Debug, Clone)]
pub struct FastBoundaryForgetting {
    pub weights: CoherenceWeights,
    pub mode: ForgetMode,
    pub heuristic: BoundaryHeuristic,
    /// Max neighbors per vertex when building the k-NN similarity graph.
    pub k_neighbors: usize,
    /// Minimum cosine similarity for an edge to be added.
    pub min_similarity: f32,
    /// [`ForgetMode::Soft`] only: bonus added to a boundary vertex's scalar
    /// importance score.
    pub structural_bonus: f32,
    /// [`ForgetMode::Hard`] only: fraction of `target_size` reserved for
    /// boundary vertices.
    pub protect_fraction: f32,
    /// [`BoundaryHeuristic::ClusteringCoefficient`] only: a vertex with
    /// local clustering coefficient `<=` this threshold (and degree `>=`
    /// `min_degree`) is flagged as a boundary vertex.
    pub clustering_threshold: f32,
    /// [`BoundaryHeuristic::ClusteringCoefficient`] only: vertices with
    /// fewer neighbors than this are skipped (too little evidence either
    /// way), matching the intuition that an isolated low-degree vertex is
    /// "sparse", not necessarily "bridging".
    pub min_degree: usize,
    /// [`BoundaryHeuristic::LabelPropagation`] only: number of synchronous
    /// propagation rounds.
    pub lp_rounds: usize,
}

impl FastBoundaryForgetting {
    fn base(heuristic: BoundaryHeuristic, mode: ForgetMode, weights: CoherenceWeights) -> Self {
        Self {
            weights,
            mode,
            heuristic,
            k_neighbors: 8,
            min_similarity: 0.05,
            structural_bonus: 0.0,
            protect_fraction: 0.0,
            clustering_threshold: 0.15,
            min_degree: 2,
            lp_rounds: 8,
        }
    }

    /// Candidate A, [`ForgetMode::Soft`].
    pub fn clustering_soft(weights: CoherenceWeights, structural_bonus: f32) -> Self {
        let mut s = Self::base(
            BoundaryHeuristic::ClusteringCoefficient,
            ForgetMode::Soft,
            weights,
        );
        s.structural_bonus = structural_bonus;
        s
    }

    /// Candidate A, [`ForgetMode::Hard`].
    pub fn clustering_hard(weights: CoherenceWeights, protect_fraction: f32) -> Self {
        let mut s = Self::base(
            BoundaryHeuristic::ClusteringCoefficient,
            ForgetMode::Hard,
            weights,
        );
        s.protect_fraction = protect_fraction;
        s
    }

    /// Candidate B, [`ForgetMode::Soft`].
    pub fn label_propagation_soft(weights: CoherenceWeights, structural_bonus: f32) -> Self {
        let mut s = Self::base(
            BoundaryHeuristic::LabelPropagation,
            ForgetMode::Soft,
            weights,
        );
        s.structural_bonus = structural_bonus;
        s
    }

    /// Candidate B, [`ForgetMode::Hard`].
    pub fn label_propagation_hard(weights: CoherenceWeights, protect_fraction: f32) -> Self {
        let mut s = Self::base(
            BoundaryHeuristic::LabelPropagation,
            ForgetMode::Hard,
            weights,
        );
        s.protect_fraction = protect_fraction;
        s
    }

    /// Build a symmetric k-NN adjacency list: vertex `i`'s top-`k_neighbors`
    /// neighbors by cosine similarity (`>= min_similarity`), unioned with the
    /// reverse direction so the result is a genuine undirected graph (a
    /// one-directional top-k relation is not symmetric in general).
    ///
    /// Determinism: ties in similarity are broken by ascending neighbor
    /// index, and the adjacency is built from a sorted `Vec`, never a hash
    /// map's iteration order — this is the same class of bug ADR-346 had to
    /// fix downstream of `ruvector-mincut`'s `DashMap` usage, avoided here by
    /// construction.
    fn build_adjacency(&self, entries: &[MemoryEntry]) -> Vec<Vec<usize>> {
        let n = entries.len();
        let k = self.k_neighbors.max(1);
        let mut adj_sets: Vec<HashSet<usize>> = vec![HashSet::new(); n];

        for i in 0..n {
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_sim(&entries[i].vector, &entries[j].vector)))
                .filter(|&(_, s)| s >= self.min_similarity)
                .collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            sims.truncate(k);
            for (j, _) in sims {
                adj_sets[i].insert(j);
                adj_sets[j].insert(i);
            }
        }

        adj_sets
            .into_iter()
            .map(|s| {
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect()
    }

    /// Candidate A: flag vertices whose local clustering coefficient
    /// (triangles among neighbors / possible neighbor pairs) is at or below
    /// [`Self::clustering_threshold`], among vertices with degree at least
    /// [`Self::min_degree`].
    fn clustering_boundary(&self, adj: &[Vec<usize>]) -> HashSet<usize> {
        let neighbor_sets: Vec<HashSet<usize>> =
            adj.iter().map(|v| v.iter().copied().collect()).collect();

        let mut boundary = HashSet::new();
        for (i, nbrs) in adj.iter().enumerate() {
            let deg = nbrs.len();
            if deg < self.min_degree {
                continue;
            }
            let mut triangles = 0usize;
            for a in 0..nbrs.len() {
                for b in (a + 1)..nbrs.len() {
                    if neighbor_sets[nbrs[a]].contains(&nbrs[b]) {
                        triangles += 1;
                    }
                }
            }
            let possible = deg * (deg.saturating_sub(1)) / 2;
            let coeff = if possible == 0 {
                0.0
            } else {
                triangles as f32 / possible as f32
            };
            if coeff <= self.clustering_threshold {
                boundary.insert(i);
            }
        }
        boundary
    }

    /// Candidate B: run [`Self::lp_rounds`] synchronous label-propagation
    /// rounds (each vertex adopts its neighbors' most-common label, ties
    /// broken by smallest label id for determinism — never by iteration
    /// order), then flag any vertex whose neighbor labels are not unanimous.
    fn label_propagation_boundary(&self, adj: &[Vec<usize>]) -> HashSet<usize> {
        let n = adj.len();
        let mut labels: Vec<usize> = (0..n).collect();

        for _ in 0..self.lp_rounds.max(1) {
            let mut new_labels = labels.clone();
            for (i, nbrs) in adj.iter().enumerate() {
                if nbrs.is_empty() {
                    continue;
                }
                let mut counts: HashMap<usize, usize> = HashMap::new();
                for &j in nbrs {
                    *counts.entry(labels[j]).or_insert(0) += 1;
                }
                let mut best: Option<(usize, usize)> = None; // (label, count)
                for (label, count) in counts {
                    best = match best {
                        None => Some((label, count)),
                        Some((bl, bc)) => {
                            if count > bc || (count == bc && label < bl) {
                                Some((label, count))
                            } else {
                                Some((bl, bc))
                            }
                        }
                    };
                }
                if let Some((label, _)) = best {
                    new_labels[i] = label;
                }
            }
            if new_labels == labels {
                break; // converged
            }
            labels = new_labels;
        }

        let mut boundary = HashSet::new();
        for (i, nbrs) in adj.iter().enumerate() {
            if nbrs.is_empty() {
                continue;
            }
            let distinct: HashSet<usize> = nbrs.iter().map(|&j| labels[j]).collect();
            if distinct.len() > 1 || !distinct.contains(&labels[i]) {
                boundary.insert(i);
            }
        }
        boundary
    }

    /// Public accessor returning the raw boundary-vertex index set (used by
    /// `select_survivors` and by the fidelity benchmark comparing this
    /// against `MincutGatedForgetting`'s exact boundary).
    pub fn boundary_set(&self, entries: &[MemoryEntry]) -> HashSet<usize> {
        if entries.len() < 4 {
            return HashSet::new();
        }
        let adj = self.build_adjacency(entries);
        if adj.iter().all(|n| n.is_empty()) {
            return HashSet::new();
        }
        match self.heuristic {
            BoundaryHeuristic::ClusteringCoefficient => self.clustering_boundary(&adj),
            BoundaryHeuristic::LabelPropagation => self.label_propagation_boundary(&adj),
        }
    }
}

impl CompactionPolicy for FastBoundaryForgetting {
    fn name(&self) -> &str {
        match (self.heuristic, self.mode) {
            (BoundaryHeuristic::ClusteringCoefficient, ForgetMode::Soft) => {
                "FastBoundaryForgetting-Clustering-Soft"
            }
            (BoundaryHeuristic::ClusteringCoefficient, ForgetMode::Hard) => {
                "FastBoundaryForgetting-Clustering-Hard"
            }
            (BoundaryHeuristic::LabelPropagation, ForgetMode::Soft) => {
                "FastBoundaryForgetting-LabelProp-Soft"
            }
            (BoundaryHeuristic::LabelPropagation, ForgetMode::Hard) => {
                "FastBoundaryForgetting-LabelProp-Hard"
            }
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

        let boundary = self.boundary_set(entries);
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

    /// Identical topology to `graph_forget::tests::bridge_dataset` — two
    /// 9-member clusters joined by a single bridge vertex through two
    /// "gateway" members — so results are directly comparable to the exact
    /// min-cut baseline.
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
    fn clustering_flags_the_bridge() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy = FastBoundaryForgetting::clustering_soft(CoherenceWeights::default(), 1.0);
        let boundary = policy.boundary_set(&entries);
        assert!(
            boundary.contains(&bridge_idx),
            "clustering-coefficient heuristic must flag the sole cross-cluster bridge; got {boundary:?}"
        );
    }

    #[test]
    fn label_propagation_flags_the_bridge() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy =
            FastBoundaryForgetting::label_propagation_soft(CoherenceWeights::default(), 1.0);
        let boundary = policy.boundary_set(&entries);
        assert!(
            boundary.contains(&bridge_idx),
            "label-propagation heuristic must flag the sole cross-cluster bridge; got {boundary:?}"
        );
    }

    #[test]
    fn clustering_soft_mode_protects_the_bridge_in_compaction() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy = FastBoundaryForgetting::clustering_soft(CoherenceWeights::default(), 1.0);
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(survivors.contains(&bridge_idx));
    }

    #[test]
    fn label_propagation_hard_mode_protects_the_bridge_in_compaction() {
        let (entries, bridge_idx) = bridge_dataset();
        let policy =
            FastBoundaryForgetting::label_propagation_hard(CoherenceWeights::default(), 0.3);
        let survivors = policy.select_survivors(&entries, 16, &[]);
        assert!(survivors.contains(&bridge_idx));
    }

    #[test]
    fn falls_back_gracefully_below_minimum_size() {
        let entries: Vec<MemoryEntry> = (0..3)
            .map(|i| MemoryEntry::new(i, vec![i as f32, 0.0], 0))
            .collect();
        let policy = FastBoundaryForgetting::clustering_soft(CoherenceWeights::default(), 1.0);
        let survivors = policy.select_survivors(&entries, 2, &[]);
        assert_eq!(survivors.len(), 2);
    }

    #[test]
    fn adjacency_build_is_deterministic_across_repeated_calls() {
        // Guards the exact property ADR-346 had to retrofit onto
        // ruvector-mincut: no hash-map iteration order anywhere in the path.
        let (entries, _) = bridge_dataset();
        let policy = FastBoundaryForgetting::clustering_soft(CoherenceWeights::default(), 1.0);
        let first = policy.build_adjacency(&entries);
        for _ in 0..10 {
            assert_eq!(first, policy.build_adjacency(&entries));
        }
    }

    #[test]
    fn triangle_counting_matches_a_hand_computed_clustering_coefficient() {
        // Two triangles {0,1,2} and {3,4,5} joined by a single bridge edge
        // 2-3. Vertex 2's neighbors are {0,1,3}: (0,1) is a triangle edge
        // (counts), (0,3) and (1,3) are not adjacent -> 1 of 3 possible
        // pairs -> coefficient = 1/3, well below a "normal" triangle
        // vertex's 1.0.
        let adj: Vec<Vec<usize>> = vec![
            vec![1, 2],    // 0
            vec![0, 2],    // 1
            vec![0, 1, 3], // 2: bridge vertex
            vec![2, 4, 5], // 3: bridge vertex
            vec![3, 5],    // 4
            vec![3, 4],    // 5
        ];
        let policy = FastBoundaryForgetting::clustering_soft(CoherenceWeights::default(), 1.0);
        let mut strict = policy.clone();
        strict.clustering_threshold = 0.34; // 1/3 rounds to ~0.333, <= 0.34
        strict.min_degree = 2;
        let boundary = strict.clustering_boundary(&adj);
        assert!(
            boundary.contains(&2) && boundary.contains(&3),
            "both bridge vertices (coefficient 1/3) should be flagged at threshold 0.34; got {boundary:?}"
        );
        assert!(
            !boundary.contains(&0) && !boundary.contains(&4),
            "plain triangle vertices (coefficient 1.0) must not be flagged; got {boundary:?}"
        );
    }
}
