//! Three retrieval variants for head-to-head comparison.
//!
//! All implement the [`Retriever`] trait so benchmarks and tests are uniform.
//!
//! ## Key insight from benchmarking
//!
//! In the **CrossCluster** regime (satellites are semantically distant from hub),
//! naive cosine re-ranking after graph expansion produces IDENTICAL recall to
//! VectorOnly — the satellites rank indistinguishably from noise.  Graph
//! augmentation only helps when the retriever **incorporates graph-edge weight
//! into the final score**.  The `hop_discount` parameter in `OneHopExpander` and
//! `CoherenceGatedHopper` represents this "trust in graph edges" — the degree
//! to which being graph-connected to a relevant entity is evidence of relevance.
//! This mirrors the α parameter in real GraphRAG systems (HippoRAG, PathRAG).

use std::collections::HashSet;

use crate::{
    coherence::should_expand,
    graph::KnowledgeGraph,
    vector::{cosine_distance, l2_norm, FlatIndex},
    Hit, MhgarError,
};

/// Common retrieval interface.
pub trait Retriever {
    fn name(&self) -> &str;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, MhgarError>;
}

// ─── Variant 1: VectorOnly ───────────────────────────────────────────────────

/// Pure approximate-nearest-neighbor baseline. No graph traversal.
pub struct VectorOnlyRetriever<'a> {
    pub index: &'a FlatIndex,
}

impl<'a> Retriever for VectorOnlyRetriever<'a> {
    fn name(&self) -> &str {
        "VectorOnly"
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, MhgarError> {
        self.index.search(query, k)
    }
}

// ─── Variant 2: OneHopExpander ────────────────────────────────────────────────

/// ANN search + targeted graph-expansion + graph-weight-influenced reranking.
///
/// Algorithm:
/// 1. Find `initial_k` seeds via vector ANN search.
/// 2. Expand graph neighbors from the **top `num_seeds_to_expand` seeds only**.
///    Expanding from the top-m (not all) seeds avoids injecting too many
///    irrelevant cross-cluster neighbours into the candidate pool.
/// 3. Score seeds by raw cosine distance; score graph-found entities with a
///    multiplicative discount: `score = cosine_distance × (1 − hop_discount)`.
///    Lower `hop_discount` = less trust in graph edges.
/// 4. Return top-k by effective score.
///
/// ## Parameter guidance
/// - `num_seeds_to_expand = 1`: only the closest ANN result is expanded; maximises
///   precision of graph traversal but may miss entities linked to 2nd-ranked seeds.
/// - `hop_discount ∈ [0.3, 0.6]`: reasonable range for production graphs with
///   strong semantic edges.  Use 0.0 to observe "naive expansion" (no benefit
///   in CrossCluster regime).
pub struct OneHopExpander<'a> {
    pub index: &'a FlatIndex,
    pub graph: &'a KnowledgeGraph,
    /// How many initial ANN seeds to retrieve.
    pub initial_k: usize,
    /// How many of the top seeds to expand via graph (≤ initial_k).
    pub num_seeds_to_expand: usize,
    /// Score discount for graph-found 1-hop entities.  0.0 = no graph trust;
    /// 0.5 = graph-found entities score at 50% of their raw cosine distance.
    pub hop_discount: f32,
}

impl<'a> Retriever for OneHopExpander<'a> {
    fn name(&self) -> &str {
        "OneHopExpander"
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, MhgarError> {
        let seeds = self.index.search(query, self.initial_k)?;
        let seed_ids: HashSet<u32> = seeds.iter().map(|h| h.id).collect();
        let mut candidate_ids: HashSet<u32> = seed_ids.clone();

        // Only expand from the top-m seeds to avoid flooding with cross-cluster noise.
        for seed in seeds.iter().take(self.num_seeds_to_expand) {
            for edge in self.graph.neighbors(seed.id) {
                candidate_ids.insert(edge.target);
            }
        }

        let q_norm = l2_norm(query);
        let mut scored: Vec<Hit> = candidate_ids
            .iter()
            .filter_map(|&id| {
                let v = self.index.get_vector(id)?;
                let dist = cosine_distance(query, v, q_norm);
                let (hops, effective_score) = if seed_ids.contains(&id) {
                    (0u32, dist)
                } else {
                    // Apply graph-edge discount: being reachable via a strong edge
                    // from a relevant seed is evidence of relevance.
                    let hop_score = dist * (1.0 - self.hop_discount).max(0.0);
                    (1u32, hop_score)
                };
                Some(Hit::new(id, effective_score, hops))
            })
            .collect();

        scored.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(scored.into_iter().take(k).collect())
    }
}

// ─── Variant 3: CoherenceGatedHopper ─────────────────────────────────────────

/// Adaptive multi-hop retrieval controlled by coherence scoring + hop discount.
///
/// Extends `OneHopExpander` with an adaptive stopping criterion:
/// - After each hop, measure the mean query-distance of the current candidate set.
/// - If distance is already low (candidates are collectively close to query), stop.
/// - Otherwise, expand one more hop from the current frontier.
///
/// Graph-found entities receive the same `hop_discount` treatment as in
/// `OneHopExpander`.  Setting `num_seeds_to_expand = 1` mirrors the
/// OneHopExpander recommendation for CrossCluster: only the top ANN seed (the
/// hub) drives graph expansion, preventing noise from wrong-cluster neighbors.
pub struct CoherenceGatedHopper<'a> {
    pub index: &'a FlatIndex,
    pub graph: &'a KnowledgeGraph,
    pub initial_k: usize,
    /// How many of the top ANN seeds to use as the initial expansion frontier.
    /// Set to 1 for CrossCluster; higher for NearHub where multiple seeds
    /// are close and their neighborhoods are worth exploring.
    pub num_seeds_to_expand: usize,
    pub max_hops: u32,
    /// Mean cosine distance threshold below which expansion stops.
    pub expansion_threshold: f32,
    /// Score discount for graph-found entities (same semantics as OneHopExpander).
    pub hop_discount: f32,
}

impl<'a> Retriever for CoherenceGatedHopper<'a> {
    fn name(&self) -> &str {
        "CoherenceGatedHopper"
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, MhgarError> {
        let seeds = self.index.search(query, self.initial_k)?;
        let seed_ids: HashSet<u32> = seeds.iter().map(|h| h.id).collect();

        let mut visited: HashSet<u32> = seed_ids.clone();
        // Only expand from the top-m seeds to avoid cross-cluster noise flooding.
        let mut frontier: Vec<u32> = seeds
            .iter()
            .take(self.num_seeds_to_expand)
            .map(|h| h.id)
            .collect();
        let mut all_graph_found: HashSet<u32> = HashSet::new();
        let mut hop = 0u32;

        loop {
            let cand_vecs: Vec<Vec<f32>> = visited
                .iter()
                .filter_map(|&id| self.index.get_vector(id).map(|v| v.to_vec()))
                .collect();

            if !should_expand(query, &cand_vecs, hop, self.max_hops, self.expansion_threshold) {
                break;
            }

            let new_candidates = self.graph.expand_one_hop(frontier.iter().copied(), &visited);
            if new_candidates.is_empty() {
                break;
            }

            frontier.clear();
            for (id, _weight) in new_candidates {
                visited.insert(id);
                all_graph_found.insert(id);
                frontier.push(id);
            }
            hop += 1;
        }

        let q_norm = l2_norm(query);
        let mut scored: Vec<Hit> = visited
            .iter()
            .filter_map(|&id| {
                let v = self.index.get_vector(id)?;
                let dist = cosine_distance(query, v, q_norm);
                let (hops, effective_score) = if seed_ids.contains(&id) {
                    (0u32, dist)
                } else {
                    let hop_score = dist * (1.0 - self.hop_discount).max(0.0);
                    (hop, hop_score)
                };
                Some(Hit::new(id, effective_score, hops))
            })
            .collect();

        scored.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(scored.into_iter().take(k).collect())
    }
}

// ─── Ground-truth oracle ─────────────────────────────────────────────────────

/// Compute the exact ground truth top-k by exhaustive search over ALL entities
/// reachable within max_hops from any direct vector neighbor.
pub fn ground_truth(
    query: &[f32],
    index: &FlatIndex,
    graph: &KnowledgeGraph,
    k: usize,
    max_hops: u32,
) -> Vec<u32> {
    let mut reachable: HashSet<u32> = (0..index.len() as u32).collect();
    let mut frontier: HashSet<u32> = reachable.clone();

    for _ in 0..max_hops {
        let mut next_frontier = HashSet::new();
        for &id in &frontier {
            for edge in graph.neighbors(id) {
                if reachable.insert(edge.target) {
                    next_frontier.insert(edge.target);
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    let q_norm = l2_norm(query);
    let mut scored: Vec<(u32, f32)> = reachable
        .iter()
        .filter_map(|&id| {
            let v = index.get_vector(id)?;
            Some((id, cosine_distance(query, v, q_norm)))
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}
