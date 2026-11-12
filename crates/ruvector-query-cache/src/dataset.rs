//! Deterministic dataset generator for semantic-query-cache benchmarks.
//!
//! Produces a corpus of random unit vectors and a query set with a controlled
//! repeat fraction: `repeat_rate` queries are drawn near existing query vectors,
//! simulating the "agent repeats similar questions" scenario.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A deterministic dataset for reproducible benchmarks.
pub struct Dataset {
    /// Indexed corpus vectors.
    pub corpus: Vec<Vec<f32>>,
    /// Queries; some are near-duplicates of earlier queries.
    pub queries: Vec<Vec<f32>>,
    /// Ground-truth top-k ids for each query (relative to corpus).
    pub ground_truth: Vec<Vec<usize>>,
    pub dim: usize,
    pub n_corpus: usize,
    pub n_queries: usize,
    pub k: usize,
    /// Fraction of queries that are near-duplicates of a prior query.
    pub repeat_rate: f32,
}

impl Dataset {
    /// Build a dataset with the given parameters.
    ///
    /// `seed`        – RNG seed for reproducibility.
    /// `n_corpus`    – number of corpus vectors.
    /// `dim`         – embedding dimensionality.
    /// `n_queries`   – total number of queries to issue.
    /// `k`           – nearest-neighbour count.
    /// `repeat_rate` – fraction [0,1] of queries drawn near a prior query.
    /// `jitter_scale`– std-dev of noise added to repeated queries (smaller = more similar).
    pub fn generate(
        seed: u64,
        n_corpus: usize,
        dim: usize,
        n_queries: usize,
        k: usize,
        repeat_rate: f32,
        jitter_scale: f32,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let corpus: Vec<Vec<f32>> = (0..n_corpus)
            .map(|_| random_unit_vec(&mut rng, dim))
            .collect();

        let mut issued_queries: Vec<Vec<f32>> = Vec::with_capacity(n_queries);
        let mut queries: Vec<Vec<f32>> = Vec::with_capacity(n_queries);

        for i in 0..n_queries {
            let q = if i > 0 && rng.gen::<f32>() < repeat_rate {
                // Draw near a prior query.
                let base_idx = rng.gen_range(0..issued_queries.len());
                let base = &issued_queries[base_idx];
                jitter_vec(&mut rng, base, jitter_scale)
            } else {
                // Fresh random query.
                random_unit_vec(&mut rng, dim)
            };
            issued_queries.push(q.clone());
            queries.push(q);
        }

        // Compute ground-truth top-k ids (exact L2) for each query.
        let ground_truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| exact_topk_ids(&corpus, q, k))
            .collect();

        Dataset {
            corpus,
            queries,
            ground_truth,
            dim,
            n_corpus,
            n_queries,
            k,
            repeat_rate,
        }
    }
}

// ─── private helpers ─────────────────────────────────────────────────────────

fn random_unit_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    normalize(v)
}

fn jitter_vec(rng: &mut StdRng, base: &[f32], scale: f32) -> Vec<f32> {
    let noisy: Vec<f32> = base
        .iter()
        .map(|x| x + (rng.gen::<f32>() * 2.0 - 1.0) * scale)
        .collect();
    normalize(noisy)
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn exact_topk_ids(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = corpus
        .iter()
        .enumerate()
        .map(|(id, v)| {
            let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
            (d, id)
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.iter().take(k).map(|(_, id)| *id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_sizes_correct() {
        let ds = Dataset::generate(42, 200, 32, 50, 5, 0.3, 0.05);
        assert_eq!(ds.corpus.len(), 200);
        assert_eq!(ds.queries.len(), 50);
        assert_eq!(ds.ground_truth.len(), 50);
        assert!(ds.ground_truth.iter().all(|gt| gt.len() == 5));
    }

    #[test]
    fn corpus_vectors_are_unit_norm() {
        let ds = Dataset::generate(1, 50, 16, 10, 3, 0.0, 0.0);
        for v in &ds.corpus {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
        }
    }

    #[test]
    fn ground_truth_ids_in_range() {
        let ds = Dataset::generate(7, 100, 8, 20, 3, 0.3, 0.05);
        for gt in &ds.ground_truth {
            for &id in gt {
                assert!(id < 100, "id={id} out of corpus bounds");
            }
        }
    }
}
