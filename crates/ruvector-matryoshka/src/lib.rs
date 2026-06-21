//! Matryoshka-aware coarse-to-fine vector search for RuVector.
//!
//! Three ANN search variants that trade distance-computation cost against recall:
//!
//! | Variant       | Stages                          | Distance ops            |
//! |---------------|---------------------------------|-------------------------|
//! | `FullDim`     | Single HNSW at full dim         | All at D dims           |
//! | `TwoStage`    | Coarse HNSW + full-dim rerank   | Traverse at D1, rerank at D |
//! | `ThreeStage`  | Coarse → mid → full-dim funnel  | Traverse D1, filter D2, rerank D |
//!
//! The primary metric is *recall@k*: fraction of the true top-k (found by brute-force
//! at full dimension) that each variant recovers.

pub mod dataset;
pub mod hnsw;

use hnsw::{l2_sq_prefix, HnswConfig, HnswGraph};

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct MatryoshkaConfig {
    pub full_dim: usize,
    pub coarse_dim: usize,
    pub mid_dim: usize,
    /// HNSW graph degree.
    pub m: usize,
    pub ef_construction: usize,
    /// Candidate set size for TwoStage coarse retrieval.
    pub two_stage_candidates: usize,
    /// Candidate set sizes for ThreeStage (coarse → mid).
    pub three_stage_coarse_candidates: usize,
    pub three_stage_mid_candidates: usize,
}

impl MatryoshkaConfig {
    /// Sensible defaults for a 128-dim collection with 32/64 coarse/mid dims.
    pub fn default_128() -> Self {
        Self {
            full_dim: 128,
            coarse_dim: 32,
            mid_dim: 64,
            m: 16,
            ef_construction: 100,
            two_stage_candidates: 100,
            three_stage_coarse_candidates: 150,
            three_stage_mid_candidates: 50,
        }
    }
}

// ─── Distance helpers ─────────────────────────────────────────────────────────

#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    l2_sq_prefix(a, b, a.len().min(b.len()))
}

/// L2-normalise a vector in-place.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        let inv = 1.0 / norm;
        v.iter_mut().for_each(|x| *x *= inv);
    }
}

/// Return the first `dim` elements, L2-normalised.
pub fn prefix_project(v: &[f32], dim: usize) -> Vec<f32> {
    let mut out: Vec<f32> = v[..dim.min(v.len())].to_vec();
    l2_normalize(&mut out);
    out
}

// ─── Searcher trait ───────────────────────────────────────────────────────────

pub trait Searcher {
    /// Build an index over the provided full-dim vectors.
    fn build(config: &MatryoshkaConfig, vectors: &[Vec<f32>]) -> Self
    where
        Self: Sized;

    /// Return approximate top-k full-dim nearest-neighbour node ids.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<usize>;

    fn name(&self) -> &'static str;
}

// ─── Variant 1: FullDimIndex ─────────────────────────────────────────────────

/// Standard HNSW search at the full embedding dimension.
/// Baseline for both recall and latency.
pub struct FullDimIndex {
    graph: HnswGraph,
}

impl Searcher for FullDimIndex {
    fn build(config: &MatryoshkaConfig, vectors: &[Vec<f32>]) -> Self {
        let hcfg = HnswConfig::new(config.full_dim, config.m, config.ef_construction);
        let mut graph = HnswGraph::new(hcfg);
        for v in vectors {
            let projected = prefix_project(v, config.full_dim);
            graph.insert(projected);
        }
        Self { graph }
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<usize> {
        let q_proj = prefix_project(query, self.graph.config.dim);
        self.graph
            .search(&q_proj, k, ef)
            .into_iter()
            .map(|id| id as usize)
            .collect()
    }

    fn name(&self) -> &'static str {
        "FullDimHNSW"
    }
}

// ─── Variant 2: TwoStageIndex ─────────────────────────────────────────────────

/// Coarse HNSW at `coarse_dim`, then full-dim rerank of the candidate set.
///
/// Distance ops breakdown:
///   - Graph traversal: O(ef × M) comparisons at `coarse_dim` dims
///   - Rerank: O(candidates) comparisons at `full_dim` dims
pub struct TwoStageIndex {
    config: MatryoshkaConfig,
    /// HNSW built on coarse-projected vectors.
    coarse_graph: HnswGraph,
    /// Full-dim vectors for reranking.
    full_vecs: Vec<Vec<f32>>,
}

impl Searcher for TwoStageIndex {
    fn build(config: &MatryoshkaConfig, vectors: &[Vec<f32>]) -> Self {
        let hcfg = HnswConfig::new(config.coarse_dim, config.m, config.ef_construction);
        let mut coarse_graph = HnswGraph::new(hcfg);
        let mut full_vecs = Vec::with_capacity(vectors.len());
        for v in vectors {
            let coarse = prefix_project(v, config.coarse_dim);
            coarse_graph.insert(coarse);
            full_vecs.push(prefix_project(v, config.full_dim));
        }
        Self {
            config: config.clone(),
            coarse_graph,
            full_vecs,
        }
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<usize> {
        let candidates = self.config.two_stage_candidates.max(k);
        let q_coarse = prefix_project(query, self.config.coarse_dim);
        let coarse_ids = self
            .coarse_graph
            .search(&q_coarse, candidates, ef.max(candidates));

        // Rerank at full_dim.
        let q_full = prefix_project(query, self.config.full_dim);
        let mut scored: Vec<(f32, usize)> = coarse_ids
            .iter()
            .map(|&id| (l2_sq(&q_full, &self.full_vecs[id as usize]), id as usize))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    fn name(&self) -> &'static str {
        "TwoStage"
    }
}

// ─── Variant 3: ThreeStageIndex ──────────────────────────────────────────────

/// Three-stage funnel: coarse_dim → mid_dim filter → full_dim rerank.
///
/// Distance ops breakdown:
///   - Stage 1 traversal: O(ef × M) at `coarse_dim` dims
///   - Stage 2 filter:    O(coarse_candidates) at `mid_dim` dims
///   - Stage 3 rerank:    O(mid_candidates) at `full_dim` dims
pub struct ThreeStageIndex {
    config: MatryoshkaConfig,
    coarse_graph: HnswGraph,
    mid_vecs: Vec<Vec<f32>>,
    full_vecs: Vec<Vec<f32>>,
}

impl Searcher for ThreeStageIndex {
    fn build(config: &MatryoshkaConfig, vectors: &[Vec<f32>]) -> Self {
        let hcfg = HnswConfig::new(config.coarse_dim, config.m, config.ef_construction);
        let mut coarse_graph = HnswGraph::new(hcfg);
        let mut mid_vecs = Vec::with_capacity(vectors.len());
        let mut full_vecs = Vec::with_capacity(vectors.len());
        for v in vectors {
            let coarse = prefix_project(v, config.coarse_dim);
            coarse_graph.insert(coarse);
            mid_vecs.push(prefix_project(v, config.mid_dim));
            full_vecs.push(prefix_project(v, config.full_dim));
        }
        Self {
            config: config.clone(),
            coarse_graph,
            mid_vecs,
            full_vecs,
        }
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<usize> {
        let coarse_n = self.config.three_stage_coarse_candidates.max(k);
        let mid_n = self.config.three_stage_mid_candidates.max(k);

        // Stage 1: coarse HNSW retrieval.
        let q_coarse = prefix_project(query, self.config.coarse_dim);
        let coarse_ids = self
            .coarse_graph
            .search(&q_coarse, coarse_n, ef.max(coarse_n));

        // Stage 2: mid-dim filtering.
        let q_mid = prefix_project(query, self.config.mid_dim);
        let mut mid_scored: Vec<(f32, u32)> = coarse_ids
            .iter()
            .map(|&id| (l2_sq(&q_mid, &self.mid_vecs[id as usize]), id))
            .collect();
        mid_scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mid_ids: Vec<u32> = mid_scored
            .into_iter()
            .take(mid_n)
            .map(|(_, id)| id)
            .collect();

        // Stage 3: full-dim rerank.
        let q_full = prefix_project(query, self.config.full_dim);
        let mut full_scored: Vec<(f32, usize)> = mid_ids
            .iter()
            .map(|&id| (l2_sq(&q_full, &self.full_vecs[id as usize]), id as usize))
            .collect();
        full_scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        full_scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    fn name(&self) -> &'static str {
        "ThreeStage"
    }
}

// ─── Recall and ground-truth ──────────────────────────────────────────────────

/// Brute-force exact top-k at full dimension (ground truth).
pub fn brute_force_knn(vectors: &[Vec<f32>], query: &[f32], k: usize, dim: usize) -> Vec<usize> {
    let mut dists: Vec<(f32, usize)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (l2_sq_prefix(query, v, dim), i))
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Recall@k: fraction of ground-truth top-k that appear in the result set.
pub fn recall_at_k(results: &[usize], ground_truth: &[usize]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let hits: usize = results
        .iter()
        .filter(|&&r| ground_truth.contains(&r))
        .count();
    hits as f32 / ground_truth.len() as f32
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dataset::generate_matryoshka_dataset;

    const N: usize = 500;
    const K: usize = 10;
    const EF: usize = 50;
    const N_QUERIES: usize = 20;
    const SEED: u64 = 0xDEAD_BEEF;

    fn build_and_recall<S: Searcher>(seed: u64) -> f32 {
        let cfg = MatryoshkaConfig::default_128();
        let (vectors, queries) =
            generate_matryoshka_dataset(N, N_QUERIES, cfg.full_dim, cfg.coarse_dim, seed);
        let idx = S::build(&cfg, &vectors);
        let mut total = 0.0f32;
        for q in &queries {
            let gt = brute_force_knn(&vectors, q, K, cfg.full_dim);
            let res = idx.search(q, K, EF);
            total += recall_at_k(&res, &gt);
        }
        total / N_QUERIES as f32
    }

    #[test]
    fn full_dim_recall_passes_threshold() {
        // N=500, ef=50, M=16: small unit-test params. Benchmark uses N=3000, ef=64.
        let recall = build_and_recall::<FullDimIndex>(SEED);
        assert!(
            recall >= 0.75,
            "FullDimHNSW recall@10 = {:.3} < 0.75",
            recall
        );
    }

    #[test]
    fn two_stage_recall_passes_threshold() {
        let recall = build_and_recall::<TwoStageIndex>(SEED);
        assert!(recall >= 0.65, "TwoStage recall@10 = {:.3} < 0.65", recall);
    }

    #[test]
    fn three_stage_recall_passes_threshold() {
        let recall = build_and_recall::<ThreeStageIndex>(SEED);
        assert!(
            recall >= 0.58,
            "ThreeStage recall@10 = {:.3} < 0.58",
            recall
        );
    }

    #[test]
    fn brute_force_is_perfect() {
        let cfg = MatryoshkaConfig::default_128();
        let (vectors, queries) =
            generate_matryoshka_dataset(200, 10, cfg.full_dim, cfg.coarse_dim, 42);
        for q in &queries {
            let gt = brute_force_knn(&vectors, q, K, cfg.full_dim);
            let recall = recall_at_k(&gt, &gt);
            assert!((recall - 1.0).abs() < 1e-6, "brute force must be perfect");
        }
    }
}
