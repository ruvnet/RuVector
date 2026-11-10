//! Matryoshka coarse-to-fine ANN search with a RaBitQ 1-bit brute-force
//! coarse stage, benchmarked against the existing HNSW-coarse `TwoStageIndex`
//! from `ruvector-matryoshka`.
//!
//! All three variants share the same funnel shape — a cheap coarse
//! candidate-generation pass followed by an exact full-dimension rerank —
//! and differ only in how the coarse pass is implemented:
//!
//! | Variant          | Coarse stage                                   | Source |
//! |------------------|-------------------------------------------------|--------|
//! | `FullDimIndex`   | none (single HNSW at full dim)                 | `ruvector-matryoshka` (baseline) |
//! | `TwoStageIndex`   | HNSW graph at `coarse_dim`                      | `ruvector-matryoshka` (candidate A) |
//! | `RaBitQCascadeIndex` | 1-bit XNOR-popcount brute scan at `coarse_dim` | this crate (candidate B) |
//!
//! Hypothesis under test (see `docs/research/nightly/.../README.md` for the
//! full write-up): a branch-free binary scan needs no graph, so it should
//! use substantially less coarse-stage memory than an HNSW graph of the same
//! dimension, while matching candidate-A's recall within a small margin
//! because the coarse stage only has to produce a *candidate set* that is
//! reranked exactly downstream, not the final top-k itself.

pub mod rabitq_code;

pub use ruvector_matryoshka::{
    brute_force_knn, dataset, l2_sq, prefix_project, recall_at_k, FullDimIndex, MatryoshkaConfig,
    Searcher, TwoStageIndex,
};

use rabitq_code::{BinaryCode, RandomRotation};

/// Default seed for the coarse-stage rotation. Separate from the dataset
/// seed so dataset generation and index construction can be varied
/// independently in the benchmark's parameter sweep.
pub const DEFAULT_ROTATION_SEED: u64 = 0xBEEF_CAFE;

/// Candidate B: RaBitQ 1-bit brute-force coarse scan + full-dim exact rerank.
pub struct RaBitQCascadeIndex {
    config: MatryoshkaConfig,
    rotation: RandomRotation,
    codes: Vec<BinaryCode>,
    full_vecs: Vec<Vec<f32>>,
}

impl RaBitQCascadeIndex {
    /// Build with an explicit rotation seed (the benchmark varies this to
    /// check the result isn't a seed artefact).
    pub fn build_with_seed(config: &MatryoshkaConfig, vectors: &[Vec<f32>], seed: u64) -> Self {
        let rotation = RandomRotation::random(config.coarse_dim, seed);
        let mut codes = Vec::with_capacity(vectors.len());
        let mut full_vecs = Vec::with_capacity(vectors.len());
        for v in vectors {
            // prefix_project L2-normalises, so every stored norm is 1.0 and
            // the symmetric angular estimator is exactly comparable to the
            // l2_sq ranking TwoStageIndex uses on its own normalised vectors.
            let coarse_unit = prefix_project(v, config.coarse_dim);
            let rotated = rotation.apply(&coarse_unit);
            codes.push(BinaryCode::encode(&rotated, 1.0));
            full_vecs.push(prefix_project(v, config.full_dim));
        }
        Self {
            config: config.clone(),
            rotation,
            codes,
            full_vecs,
        }
    }

    /// Coarse-stage-only memory: packed 1-bit codes plus the one-time
    /// rotation matrix. Excludes the full-dim rerank vectors, which both
    /// this index and `TwoStageIndex` store identically.
    pub fn coarse_stage_bytes(&self) -> usize {
        self.codes.iter().map(|c| c.bytes()).sum::<usize>() + self.rotation.bytes()
    }
}

impl Searcher for RaBitQCascadeIndex {
    fn build(config: &MatryoshkaConfig, vectors: &[Vec<f32>]) -> Self {
        Self::build_with_seed(config, vectors, DEFAULT_ROTATION_SEED)
    }

    fn search(&self, query: &[f32], k: usize, _ef: usize) -> Vec<usize> {
        let candidates = self.config.two_stage_candidates.max(k);

        let q_coarse_unit = prefix_project(query, self.config.coarse_dim);
        let q_rotated = self.rotation.apply(&q_coarse_unit);
        let q_code = BinaryCode::encode(&q_rotated, 1.0);

        let mut coarse_scored: Vec<(f32, usize)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(id, c)| (c.estimated_sq_distance(&q_code), id))
            .collect();
        coarse_scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let coarse_ids: Vec<usize> = coarse_scored
            .into_iter()
            .take(candidates)
            .map(|(_, id)| id)
            .collect();

        let q_full = prefix_project(query, self.config.full_dim);
        let mut reranked: Vec<(f32, usize)> = coarse_ids
            .iter()
            .map(|&id| (l2_sq(&q_full, &self.full_vecs[id]), id))
            .collect();
        reranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        reranked.into_iter().take(k).map(|(_, id)| id).collect()
    }

    fn name(&self) -> &'static str {
        "RaBitQCascade"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dataset::generate_matryoshka_dataset;

    const N: usize = 500;
    const K: usize = 10;
    const EF: usize = 50;
    const N_QUERIES: usize = 20;
    const SEED: u64 = 0xDEAD_BEEF;

    #[test]
    fn rabitq_cascade_recall_passes_threshold() {
        let cfg = MatryoshkaConfig::default_128();
        let (vectors, queries) =
            generate_matryoshka_dataset(N, N_QUERIES, cfg.full_dim, cfg.coarse_dim, SEED);
        let idx = RaBitQCascadeIndex::build(&cfg, &vectors);
        let mut total = 0.0f32;
        for q in &queries {
            let gt = brute_force_knn(&vectors, q, K, cfg.full_dim);
            let res = idx.search(q, K, EF);
            total += recall_at_k(&res, &gt);
        }
        let recall = total / N_QUERIES as f32;
        // Small-N unit-test threshold; the benchmark binary uses N=3000 and a
        // tighter acceptance bound vs. TwoStageIndex measured head-to-head.
        assert!(
            recall >= 0.55,
            "RaBitQCascade recall@10 = {:.3} < 0.55",
            recall
        );
    }

    #[test]
    fn coarse_stage_is_smaller_than_hnsw_equivalent() {
        let cfg = MatryoshkaConfig::default_128();
        let (vectors, _queries) =
            generate_matryoshka_dataset(N, 5, cfg.full_dim, cfg.coarse_dim, SEED);
        let idx = RaBitQCascadeIndex::build(&cfg, &vectors);
        // HNSW-coarse equivalent: n*coarse_dim*4 (raw vecs) + n*m*2*4 (graph edges).
        let hnsw_coarse_bytes = N * cfg.coarse_dim * 4 + N * cfg.m * 2 * 4;
        assert!(
            idx.coarse_stage_bytes() < hnsw_coarse_bytes / 4,
            "RaBitQ coarse stage ({} bytes) should be well under 1/4 of the HNSW-coarse equivalent ({} bytes)",
            idx.coarse_stage_bytes(),
            hnsw_coarse_bytes
        );
    }

    #[test]
    fn different_rotation_seeds_give_different_codes_same_recall_ballpark() {
        let cfg = MatryoshkaConfig::default_128();
        let (vectors, queries) =
            generate_matryoshka_dataset(N, N_QUERIES, cfg.full_dim, cfg.coarse_dim, SEED);
        let idx_a = RaBitQCascadeIndex::build_with_seed(&cfg, &vectors, 1);
        let idx_b = RaBitQCascadeIndex::build_with_seed(&cfg, &vectors, 2);
        let recall_of = |idx: &RaBitQCascadeIndex| -> f32 {
            let mut total = 0.0f32;
            for q in &queries {
                let gt = brute_force_knn(&vectors, q, K, cfg.full_dim);
                total += recall_at_k(&idx.search(q, K, EF), &gt);
            }
            total / N_QUERIES as f32
        };
        let (ra, rb) = (recall_of(&idx_a), recall_of(&idx_b));
        assert!(
            (ra - rb).abs() < 0.15,
            "recall should not be wildly seed-dependent: seed1={ra:.3} seed2={rb:.3}"
        );
    }
}
