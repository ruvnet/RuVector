//! Residual Vector Quantization (RVQ) — main contribution.
//!
//! Unlike PQ (which partitions the vector into independent sub-spaces), RVQ applies
//! K successive full-dimensional k-means quantisations to the residual error.
//!
//!   residual_0 = v
//!   code_k = argmin_{c} ||residual_k - codeword_k[c]||²
//!   residual_{k+1} = residual_k - codeword_k[code_k]
//!   v̂ = Σ_k codeword_k[code_k]
//!
//! This lets each stage correct the approximation error of all prior stages,
//! unlike PQ's independent sub-space approximations.
//!
//! Query-time ADC:
//!   Precompute K × N_cw inner-product lookup tables (query · codeword_k[c]).
//!   Approximate score = Σ_k LUT_k[code_k[i]]   (K additions per database vector)
//!
//! Memory:
//!   Codebooks: K × N_cw × D × 4 bytes  (larger than PQ, fixed overhead)
//!   Codes:     N × K bytes              (same as PQ at equal bit budget)

use crate::{dot, kmeans::kmeans, nearest_centroid, Hit, VectorIndex};

pub struct RvqIndex {
    dim: usize,
    n_stages: usize,
    n_codewords: usize,
    /// codebooks[stage][cw] = full-dim vector
    codebooks: Vec<Vec<Vec<f32>>>,
    /// codes[vector_id][stage] = codeword index
    codes: Vec<Vec<u8>>,
}

impl RvqIndex {
    pub fn with_config(vectors: &[Vec<f32>], n_stages: usize, n_codewords: usize) -> Self {
        Self::with_config_iters(vectors, n_stages, n_codewords, 20)
    }

    pub fn with_config_iters(
        vectors: &[Vec<f32>],
        n_stages: usize,
        n_codewords: usize,
        max_iter: usize,
    ) -> Self {
        assert!(n_codewords <= 256, "n_codewords must fit in u8");
        let n = vectors.len();
        let dim = vectors[0].len();

        let mut residuals: Vec<Vec<f32>> = vectors.to_vec();
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_stages);
        let mut all_codes: Vec<Vec<u8>> = vec![Vec::with_capacity(n_stages); n];

        for stage in 0..n_stages {
            let nc = n_codewords.min(residuals.len());
            let cb = kmeans(&residuals, nc, max_iter, 0xF00D_CAFE ^ stage as u64);

            // Assign and subtract nearest codeword from each residual.
            for (i, r) in residuals.iter_mut().enumerate() {
                let (idx, _) = nearest_centroid(r, &cb);
                all_codes[i].push(idx as u8);
                for (rj, cj) in r.iter_mut().zip(cb[idx].iter()) {
                    *rj -= cj;
                }
            }

            codebooks.push(cb);
        }

        Self {
            dim,
            n_stages,
            n_codewords,
            codebooks,
            codes: all_codes,
        }
    }

    /// Reconstruct the approximate vector for `codes` (useful for verification).
    pub fn reconstruct(&self, codes: &[u8]) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for (stage, &c) in codes.iter().enumerate().take(self.n_stages) {
            let cw = &self.codebooks[stage][c as usize];
            for (vi, &cj) in v.iter_mut().zip(cw.iter()) {
                *vi += cj;
            }
        }
        v
    }

    /// Mean squared reconstruction error across the dataset (lower = better).
    pub fn reconstruction_error(&self, vectors: &[Vec<f32>]) -> f32 {
        use crate::l2_sq;
        let total: f32 = self
            .codes
            .iter()
            .zip(vectors.iter())
            .map(|(codes, v)| {
                let v_hat = self.reconstruct(codes);
                l2_sq(v, &v_hat)
            })
            .sum();
        total / vectors.len() as f32
    }
}

impl RvqIndex {
    pub fn build(vectors: &[Vec<f32>]) -> Self {
        // Default: 8 stages × 256 codewords → 8 bytes/vector (same bit budget as PQ-8sub).
        Self::with_config(vectors, 8, 256)
    }
}

impl VectorIndex for RvqIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        // ADC: query inner-product with every codeword in every stage.
        let lut: Vec<Vec<f32>> = self
            .codebooks
            .iter()
            .map(|cb| cb.iter().map(|cw| dot(query, cw)).collect())
            .collect();

        let mut hits: Vec<Hit> = self
            .codes
            .iter()
            .enumerate()
            .map(|(id, codes)| {
                let score: f32 = codes
                    .iter()
                    .enumerate()
                    .map(|(s, &c)| lut[s][c as usize])
                    .sum();
                Hit { id, score }
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str {
        "RVQ"
    }

    fn memory_bytes(&self) -> usize {
        // Codebooks: K stages × N_cw codewords × D dims × 4 bytes.
        let codebook_bytes = self.n_stages * self.n_codewords * self.dim * 4;
        // Codes: N vectors × K stages × 1 byte.
        let code_bytes = self.codes.len() * self.n_stages;
        codebook_bytes + code_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Dataset, DatasetConfig};
    use crate::{exact::ExactSearch, l2_sq, recall_at_k};

    #[test]
    fn rvq_returns_k_hits() {
        let cfg = DatasetConfig {
            n_vectors: 512,
            dims: 64,
            n_queries: 5,
            ..Default::default()
        };
        let ds = Dataset::generate(&cfg);
        let idx = RvqIndex::with_config(&ds.vectors, 4, 64);
        let hits = idx.search(&ds.queries[0], 10);
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn rvq_reconstruction_error_decreases_with_stages() {
        let cfg = DatasetConfig {
            n_vectors: 1_000,
            dims: 32,
            n_queries: 0,
            ..Default::default()
        };
        let ds = Dataset::generate(&cfg);

        let rvq2 = RvqIndex::with_config(&ds.vectors, 2, 64);
        let rvq4 = RvqIndex::with_config(&ds.vectors, 4, 64);
        let rvq8 = RvqIndex::with_config(&ds.vectors, 8, 64);

        let e2 = rvq2.reconstruction_error(&ds.vectors);
        let e4 = rvq4.reconstruction_error(&ds.vectors);
        let e8 = rvq8.reconstruction_error(&ds.vectors);

        // Each additional stage should reduce error.
        assert!(
            e4 < e2,
            "4-stage error {e4:.4} should be < 2-stage error {e2:.4}"
        );
        assert!(
            e8 < e4,
            "8-stage error {e8:.4} should be < 4-stage error {e4:.4}"
        );
    }

    #[test]
    fn rvq_recall_at_k_above_threshold() {
        let cfg = DatasetConfig {
            n_vectors: 2_000,
            dims: 64,
            n_queries: 50,
            ..Default::default()
        };
        let ds = Dataset::generate(&cfg);
        let exact = ExactSearch::build(&ds.vectors);
        let rvq = RvqIndex::with_config(&ds.vectors, 4, 256);
        let k = 10;

        let avg_recall: f32 = ds
            .queries
            .iter()
            .map(|q| {
                let truth = exact.search(q, k);
                let approx = rvq.search(q, k);
                recall_at_k(&truth, &approx, k)
            })
            .sum::<f32>()
            / ds.queries.len() as f32;

        // RVQ-4stage on 64-dim random unit vectors: recall varies 20-40% due to isotropic distribution.
        assert!(
            avg_recall >= 0.20,
            "RVQ recall@{k} = {avg_recall:.3} below floor (0.20)"
        );
    }

    #[test]
    fn rvq_reconstruction_converges() {
        // A single perfectly-fitting stage should reconstruct to near-zero error.
        let vecs: Vec<Vec<f32>> = (0..50u8)
            .map(|i| vec![i as f32 / 50.0, (50 - i) as f32 / 50.0])
            .collect();
        // 50 codewords for 50 points → each point maps to its own centroid.
        let idx = RvqIndex::with_config(&vecs, 1, 50);
        let err = idx.reconstruction_error(&vecs);
        assert!(
            err < 1e-3,
            "single-stage reconstruction error {err:.6} too high"
        );
    }
}
