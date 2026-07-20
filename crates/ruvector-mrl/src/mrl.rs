//! MRL-aware index variants.
//!
//! Two concrete implementations sit on top of the core graph primitive:
//!
//! * [`MrlLinear`] — brute-force prefix scan + full-dim rerank.
//!   Shows the speedup from dimension reduction alone, no graph structure.
//! * [`MrlGraph`] — greedy graph on D_FAST prefix + full-dim rerank.
//!   Full approach: sub-linear graph navigation in low-dim space.

use crate::{dot, graph::GreedyGraph, MrlSearch, SearchResult};

/// Brute-force two-stage index.
///
/// Stage 1: score all N vectors using only `d_fast` prefix dimensions.
/// Stage 2: exact cosine on `d_full` dims for the top `k × k_over` candidates.
///
/// Cost per query: O(N·D_FAST + k·k_over·D_FULL).
pub struct MrlLinear {
    /// Prefix-only vectors stored for fast pass.
    fast_vecs: Vec<Vec<f32>>,
    /// Full vectors stored for exact rerank.
    full_vecs: Vec<Vec<f32>>,
    d_fast: usize,
    d_full: usize,
    /// Oversample factor: retrieve k × k_over candidates in stage 1.
    k_over: usize,
}

impl MrlLinear {
    /// Create a linear MRL index.
    pub fn new(d_fast: usize, d_full: usize, k_over: usize) -> Self {
        assert!(d_fast <= d_full);
        MrlLinear {
            fast_vecs: Vec::new(),
            full_vecs: Vec::new(),
            d_fast,
            d_full,
            k_over,
        }
    }

    fn fast_score(&self, id: usize, query: &[f32]) -> f32 {
        dot(&self.fast_vecs[id], &query[..self.d_fast])
    }

    fn full_score(&self, id: usize, query: &[f32]) -> f32 {
        dot(&self.full_vecs[id], &query[..self.d_full])
    }
}

impl MrlSearch for MrlLinear {
    fn insert(&mut self, id: u32, vector: &[f32]) {
        assert_eq!(vector.len(), self.d_full, "dimension mismatch");
        assert_eq!(id as usize, self.full_vecs.len(), "ids must be sequential");
        self.fast_vecs.push(vector[..self.d_fast].to_vec());
        self.full_vecs.push(vector.to_vec());
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let n = self.full_vecs.len();
        if n == 0 {
            return Vec::new();
        }

        // Stage 1: score on d_fast prefix.
        let shortlist = (k * self.k_over).max(k + 1).min(n);
        let mut scored: Vec<(u32, f32)> = (0..n)
            .map(|i| (i as u32, self.fast_score(i, query)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(shortlist);

        // Stage 2: exact rerank on full dims.
        let mut reranked: Vec<SearchResult> = scored
            .iter()
            .map(|&(id, _)| SearchResult {
                id,
                score: self.full_score(id as usize, query),
            })
            .collect();
        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(k);
        reranked
    }

    fn len(&self) -> usize {
        self.full_vecs.len()
    }
}

/// Graph-based two-stage MRL index.
///
/// Stage 1: beam search on `d_fast` prefix via greedy kNN graph.
/// Stage 2: exact cosine on `d_full` dims for the top `k × oversample` candidates.
///
/// Build cost: O(N² · D_FAST) — reasonable for N ≤ 10 K.
/// Search cost: O(ef · M · D_FAST + k·oversample · D_FULL).
pub struct MrlGraph {
    graph: GreedyGraph,
    /// Beam width during graph navigation.
    ef: usize,
    /// Candidate oversample factor for rerank.
    oversample: usize,
}

impl MrlGraph {
    /// Create a graph-based MRL index.
    ///
    /// * `d_fast` — prefix dimension for graph navigation.
    /// * `d_full` — total vector dimension.
    /// * `m` — max neighbours per graph node.
    /// * `ef` — beam width during search.
    /// * `oversample` — fetch `k × oversample` before full rerank.
    pub fn new(d_fast: usize, d_full: usize, m: usize, ef: usize, oversample: usize) -> Self {
        MrlGraph {
            graph: GreedyGraph::new(d_fast, d_full, m),
            ef,
            oversample,
        }
    }
}

impl MrlSearch for MrlGraph {
    fn insert(&mut self, id: u32, vector: &[f32]) {
        let assigned = self.graph.insert(vector);
        assert_eq!(assigned, id, "ids must be sequential");
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let k_over = (k * self.oversample).max(k + 1);
        let candidates = self.graph.beam_fast(query, k_over, self.ef);
        self.graph.rerank(&candidates, query, k)
    }

    fn len(&self) -> usize {
        self.graph.len()
    }
}

impl MrlGraph {
    /// Finalise edge construction.  Must be called once after all [`insert`]
    /// calls and before any [`search`] call.
    pub fn build_edges(&mut self) {
        self.graph.build_edges();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{normalize, recall_at_k};

    fn unit_vec(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[hot % dim] = 1.0;
        normalize(&mut v);
        v
    }

    #[test]
    fn test_mrl_linear_top1() {
        let (d_fast, d_full) = (8, 32);
        let mut idx = MrlLinear::new(d_fast, d_full, 4);
        for i in 0..50u32 {
            idx.insert(i, &unit_vec(d_full, i as usize));
        }
        let q = unit_vec(d_full, 0);
        let results = idx.search(&q, 1);
        assert_eq!(results[0].id, 0);
    }

    /// Gaussian vector normalized to unit sphere.
    /// MRL works well on these because information is spread across all dims,
    /// so the d_fast prefix is genuinely informative.
    fn gauss_vec(dim: usize, seed_extra: u64) -> Vec<f32> {
        use std::f32::consts::PI;
        // Simple Box-Muller using a linear congruential seed.
        let mut v = Vec::with_capacity(dim);
        let mut s = seed_extra.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        for _ in 0..dim / 2 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u1 = (s >> 33) as f32 / u32::MAX as f32;
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u2 = (s >> 33) as f32 / u32::MAX as f32;
            let r = (-2.0 * (u1 + 1e-9).ln()).sqrt();
            v.push(r * (2.0 * PI * u2).cos());
            v.push(r * (2.0 * PI * u2).sin());
        }
        if dim % 2 == 1 { v.push(0.0); }
        normalize(&mut v);
        v
    }

    #[test]
    fn test_mrl_linear_recall_at_10() {
        // d_fast = 16/32 = 50% dimensionality ratio; this gives good recall
        // with modest oversample. At 25% ratio recall drops to ~0.65 — also
        // a valid finding captured in the benchmark binary.
        let (d_fast, d_full) = (16, 32);
        let mut flat_idx = crate::flat::FlatIndex::new(d_full);
        let mut mrl_idx = MrlLinear::new(d_fast, d_full, 5);
        for i in 0..200u32 {
            let v = gauss_vec(d_full, i as u64 * 1_000_003);
            flat_idx.insert(i, &v);
            mrl_idx.insert(i, &v);
        }
        let mut total_recall = 0.0f32;
        let num_q = 20u64;
        for qi in 0..num_q {
            let q = gauss_vec(d_full, qi.wrapping_mul(999_983) ^ 0xABCD);
            let gt: Vec<u32> = flat_idx.search(&q, 10).iter().map(|r| r.id).collect();
            let found: Vec<u32> = mrl_idx.search(&q, 10).iter().map(|r| r.id).collect();
            total_recall += recall_at_k(&gt, &found);
        }
        let mean_recall = total_recall / num_q as f32;
        assert!(
            mean_recall >= 0.75,
            "MrlLinear recall@10 = {:.3}, expected >= 0.75",
            mean_recall
        );
    }

    #[test]
    fn test_mrl_graph_top1() {
        let (d_fast, d_full, m, ef, os) = (8, 32, 8, 20, 3);
        let mut idx = MrlGraph::new(d_fast, d_full, m, ef, os);
        for i in 0..100u32 {
            idx.insert(i, &unit_vec(d_full, i as usize));
        }
        idx.build_edges();
        let q = unit_vec(d_full, 3);
        let results = idx.search(&q, 1);
        assert_eq!(results[0].id, 3);
    }

    #[test]
    fn test_recall_metric_helpers() {
        assert!((recall_at_k(&[0, 1, 2], &[0, 1, 2]) - 1.0).abs() < 1e-6);
        assert!((recall_at_k(&[0, 1, 2], &[0, 1, 5]) - 2.0 / 3.0).abs() < 1e-6);
        assert!((recall_at_k(&[0, 1, 2], &[9, 8, 7])).abs() < 1e-6);
    }
}
