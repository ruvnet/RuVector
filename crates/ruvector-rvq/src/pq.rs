//! Product Quantization (PQ) — baseline compressed index.
//!
//! Splits each D-dim vector into M non-overlapping sub-spaces of size D/M.
//! Trains one k-means codebook per sub-space (K codewords each).
//! Encodes each vector as M code bytes; decodes via lookup.
//!
//! Query uses Asymmetric Distance Computation (ADC):
//!   Precompute M×K lookup tables of query-sub·codeword inner products.
//!   Score = Σ_m LUT[m][code[m]]  (integer-indexed dot product)
//!
//! This is O(M × N) per query rather than O(D × N), with M ≤ D.

use crate::{dot, kmeans::kmeans, nearest_centroid, Hit, VectorIndex};

pub struct PqIndex {
    dim: usize,
    m_subspaces: usize,
    n_codewords: usize,
    /// codebooks[m][cw] = sub-vector of length (dim / m_subspaces)
    codebooks: Vec<Vec<Vec<f32>>>,
    /// codes[vector_id][subspace] = codeword index (u8, so n_codewords ≤ 256)
    codes: Vec<Vec<u8>>,
}

impl PqIndex {
    pub fn with_config(vectors: &[Vec<f32>], m_subspaces: usize, n_codewords: usize) -> Self {
        Self::with_config_iters(vectors, m_subspaces, n_codewords, 20)
    }

    pub fn with_config_iters(
        vectors: &[Vec<f32>],
        m_subspaces: usize,
        n_codewords: usize,
        max_iter: usize,
    ) -> Self {
        assert!(n_codewords <= 256, "n_codewords must fit in u8");
        let dim = vectors[0].len();
        assert_eq!(dim % m_subspaces, 0, "dim must be divisible by m_subspaces");
        let sub_dim = dim / m_subspaces;

        // Train one codebook per sub-space.
        let codebooks: Vec<Vec<Vec<f32>>> = (0..m_subspaces)
            .map(|m| {
                let sub_data: Vec<Vec<f32>> = vectors
                    .iter()
                    .map(|v| v[m * sub_dim..(m + 1) * sub_dim].to_vec())
                    .collect();
                let nc = n_codewords.min(sub_data.len());
                kmeans(&sub_data, nc, max_iter, 0xABCD_EF00 ^ m as u64)
            })
            .collect();

        // Encode every vector.
        let codes: Vec<Vec<u8>> = vectors
            .iter()
            .map(|v| {
                (0..m_subspaces)
                    .map(|m| {
                        let sub = &v[m * sub_dim..(m + 1) * sub_dim];
                        nearest_centroid(sub, &codebooks[m]).0 as u8
                    })
                    .collect()
            })
            .collect();

        Self {
            dim,
            m_subspaces,
            n_codewords,
            codebooks,
            codes,
        }
    }
}

impl PqIndex {
    pub fn build(vectors: &[Vec<f32>]) -> Self {
        // Default: 8 sub-spaces × 256 codewords → 8 bytes/vector (64× vs f32).
        Self::with_config(vectors, 8, 256)
    }
}

impl VectorIndex for PqIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let sub_dim = self.dim / self.m_subspaces;

        // ADC: query-to-codeword inner products for each sub-space.
        let lut: Vec<Vec<f32>> = (0..self.m_subspaces)
            .map(|m| {
                let q_sub = &query[m * sub_dim..(m + 1) * sub_dim];
                self.codebooks[m].iter().map(|cw| dot(q_sub, cw)).collect()
            })
            .collect();

        let mut hits: Vec<Hit> = self
            .codes
            .iter()
            .enumerate()
            .map(|(id, code)| {
                let score: f32 = code
                    .iter()
                    .enumerate()
                    .map(|(m, &c)| lut[m][c as usize])
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
        "PQ"
    }

    fn memory_bytes(&self) -> usize {
        let sub_dim = self.dim / self.m_subspaces;
        // Codebooks: M sub-spaces × K codewords × sub_dim f32s.
        let codebook_bytes = self.m_subspaces * self.n_codewords * sub_dim * 4;
        // Codes: N vectors × M bytes.
        let code_bytes = self.codes.len() * self.m_subspaces;
        codebook_bytes + code_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Dataset, DatasetConfig};
    use crate::{exact::ExactSearch, recall_at_k};

    #[test]
    fn pq_returns_k_hits() {
        let cfg = DatasetConfig {
            n_vectors: 512,
            dims: 64,
            n_queries: 5,
            ..Default::default()
        };
        let ds = Dataset::generate(&cfg);
        let idx = PqIndex::with_config(&ds.vectors, 8, 64);
        let hits = idx.search(&ds.queries[0], 10);
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn pq_recall_at_k_above_threshold() {
        let cfg = DatasetConfig {
            n_vectors: 2_000,
            dims: 64,
            n_queries: 50,
            ..Default::default()
        };
        let ds = Dataset::generate(&cfg);
        let exact = ExactSearch::build(&ds.vectors);
        // 4 sub-spaces × 256 codewords on 2K×64 dataset (256 data pts/centroid avg).
        let pq = PqIndex::with_config(&ds.vectors, 4, 256);
        let k = 10;
        let avg_recall: f32 = ds
            .queries
            .iter()
            .map(|q| {
                let truth = exact.search(q, k);
                let approx = pq.search(q, k);
                recall_at_k(&truth, &approx, k)
            })
            .sum::<f32>()
            / ds.queries.len() as f32;

        // PQ-4sub on 64-dim random unit vectors: recall varies 20-40% due to isotropic distribution.
        assert!(
            avg_recall >= 0.20,
            "PQ recall@{k} = {avg_recall:.3} below floor (0.20)"
        );
    }
}
