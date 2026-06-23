//! Deterministic synthetic dataset generator for semantic cache benchmarks.
//!
//! Produces:
//! - A "corpus" of N vectors simulating a vector store (dim D).
//! - A "query history" set: K query vectors, each answered by top-10 corpus neighbors.
//! - A "test" set: M queries, each semantically close to one history query (perturbed)
//!   plus M random queries (expected cache misses).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct Dataset {
    pub corpus: Vec<Vec<f32>>,
    /// Ground truth: for each history query, the true top-k corpus IDs.
    pub history_queries: Vec<Vec<f32>>,
    pub history_ground_truth: Vec<Vec<u32>>,
    /// Near-duplicate queries (semantically close to history, should hit cache).
    pub near_dup_queries: Vec<Vec<f32>>,
    /// Expected result for near_dup_queries[i]: same as history_ground_truth[i].
    pub near_dup_expected: Vec<Vec<u32>>,
    /// Random queries (no overlap with history, should miss cache).
    pub random_queries: Vec<Vec<f32>>,
    pub dim: usize,
    pub k: usize,
}

impl Dataset {
    /// Build a deterministic dataset.
    ///
    /// - `n_corpus`: number of vectors in the corpus.
    /// - `n_history`: number of pre-warmed cached queries.
    /// - `n_test_per_class`: number of near-dup and random test queries each.
    /// - `dim`: embedding dimension.
    /// - `k`: top-k for ground truth.
    /// - `noise_scale`: L2 perturbation applied to near-duplicate queries.
    pub fn generate(
        n_corpus: usize,
        n_history: usize,
        n_test_per_class: usize,
        dim: usize,
        k: usize,
        noise_scale: f32,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_1234_5678);

        // ── Corpus ───────────────────────────────────────────────────────────
        let corpus: Vec<Vec<f32>> = (0..n_corpus).map(|_| unit_vec(&mut rng, dim)).collect();

        // ── History queries and ground truth ──────────────────────────────────
        let history_queries: Vec<Vec<f32>> =
            (0..n_history).map(|_| unit_vec(&mut rng, dim)).collect();

        let history_ground_truth: Vec<Vec<u32>> = history_queries
            .iter()
            .map(|q| brute_force_topk(q, &corpus, k))
            .collect();

        // ── Near-duplicate queries ─────────────────────────────────────────────
        // Cycle through history_queries to generate exactly n_test_per_class
        // near-duplicate queries, each independently perturbed.
        let near_dup_queries: Vec<Vec<f32>> = (0..n_test_per_class)
            .map(|i| {
                let base = &history_queries[i % history_queries.len()];
                let mut perturbed = base.clone();
                for v in perturbed.iter_mut() {
                    *v += rng.gen_range(-noise_scale..noise_scale);
                }
                normalize(&perturbed)
            })
            .collect();

        let near_dup_expected: Vec<Vec<u32>> = near_dup_queries
            .iter()
            .enumerate()
            .map(|(i, _)| history_ground_truth[i % history_ground_truth.len()].clone())
            .collect();

        // ── Random queries ─────────────────────────────────────────────────────
        let random_queries: Vec<Vec<f32>> = (0..n_test_per_class)
            .map(|_| unit_vec(&mut rng, dim))
            .collect();

        Self {
            corpus,
            history_queries,
            history_ground_truth,
            near_dup_queries,
            near_dup_expected,
            random_queries,
            dim,
            k,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        let f32_size = std::mem::size_of::<f32>();
        let corpus_bytes = self.corpus.len() * self.dim * f32_size;
        let history_bytes = self.history_queries.len() * self.dim * f32_size;
        let test_bytes =
            (self.near_dup_queries.len() + self.random_queries.len()) * self.dim * f32_size;
        corpus_bytes + history_bytes + test_bytes
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub fn unit_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    normalize(&raw)
}

pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn brute_force_topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut scores: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_sim(query, v)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.iter().take(k).map(|(i, _)| *i as u32).collect()
}

/// Compute recall@k: fraction of ground truth IDs present in retrieved IDs.
pub fn recall_at_k(retrieved: &[u32], ground_truth: &[u32]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let hits = retrieved
        .iter()
        .filter(|id| ground_truth.contains(id))
        .count();
    hits as f32 / ground_truth.len() as f32
}
