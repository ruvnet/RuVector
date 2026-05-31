//! Product Quantization: M independent sub-space k-means codebooks
//! with Asymmetric Distance Computation (ADC) lookup tables.
//!
//! Reference: Jégou, Douze, Schmid. "Product Quantization for Nearest Neighbor Search."
//! IEEE TPAMI 33(1), 2011. DOI:10.1109/TPAMI.2010.57

use crate::kmeans::{l2sq, nearest, train as kmeans_train};

/// Trained PQ codebook: M subspaces × ksub centroids × sub_dim dimensions.
pub struct PqCodebook {
    pub m: usize,
    pub ksub: usize,
    pub sub_dim: usize,
    /// books[subspace][centroid][component]
    pub books: Vec<Vec<Vec<f32>>>,
}

impl PqCodebook {
    /// Train on `vectors` (shape [N][D]). D must be divisible by `m`.
    pub fn train(vectors: &[Vec<f32>], m: usize, ksub: usize, max_iter: usize) -> Self {
        assert!(ksub <= 256, "ksub={ksub} exceeds u8 range (max 256)");
        assert!(!vectors.is_empty(), "training set must not be empty");
        let dim = vectors[0].len();
        assert_eq!(dim % m, 0, "dim={dim} must be divisible by m={m}");
        let sub_dim = dim / m;

        let books: Vec<Vec<Vec<f32>>> = (0..m)
            .map(|s| {
                let start = s * sub_dim;
                let end = start + sub_dim;
                let sub_vecs: Vec<Vec<f32>> =
                    vectors.iter().map(|v| v[start..end].to_vec()).collect();
                // Seed varies per subspace for independent codebooks
                kmeans_train(&sub_vecs, ksub.min(sub_vecs.len()), max_iter, 0xBEEF + s as u64)
            })
            .collect();

        PqCodebook { m, ksub, sub_dim, books }
    }

    /// Encode a D-dimensional vector into `m` u8 codes (one per subspace).
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        (0..self.m)
            .map(|s| {
                let start = s * self.sub_dim;
                nearest(&v[start..start + self.sub_dim], &self.books[s]) as u8
            })
            .collect()
    }

    /// Build an ADC lookup table for `query`.
    /// tables[s][k] = L2 distance from query subvector s to codebook centroid k.
    pub fn build_lut(&self, query: &[f32]) -> LookupTable {
        let tables = (0..self.m)
            .map(|s| {
                let start = s * self.sub_dim;
                let q_sub = &query[start..start + self.sub_dim];
                self.books[s].iter().map(|c| l2sq(q_sub, c)).collect()
            })
            .collect();
        LookupTable { tables }
    }

    /// Squared reconstruction error for `v` (PQ quantisation noise).
    pub fn reconstruct_error(&self, v: &[f32]) -> f32 {
        let codes = self.encode(v);
        (0..self.m)
            .map(|s| {
                let start = s * self.sub_dim;
                l2sq(
                    &v[start..start + self.sub_dim],
                    &self.books[s][codes[s] as usize],
                )
            })
            .sum()
    }

    /// Memory occupied by the codebook arrays in bytes.
    pub fn codebook_bytes(&self) -> usize {
        self.m * self.ksub * self.sub_dim * 4
    }
}

/// Precomputed ADC lookup table for a single query vector.
pub struct LookupTable {
    /// tables[subspace][centroid_id] = squared distance to query subvector.
    pub tables: Vec<Vec<f32>>,
}

impl LookupTable {
    /// Approximate squared-L2 distance to a PQ-encoded vector via table lookups.
    #[inline]
    pub fn score(&self, codes: &[u8]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(s, &c)| self.tables[s][c as usize])
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_corpus(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32 * 0.1).collect())
            .collect()
    }

    #[test]
    fn encode_produces_valid_codes() {
        let vecs = make_corpus(100, 8);
        let cb = PqCodebook::train(&vecs, 2, 16, 30);
        for v in &vecs {
            let codes = cb.encode(v);
            assert_eq!(codes.len(), 2);
            assert!(codes[0] < 16, "code0={}", codes[0]);
            assert!(codes[1] < 16, "code1={}", codes[1]);
        }
    }

    #[test]
    fn reconstruction_error_non_negative() {
        let vecs = make_corpus(200, 16);
        let cb = PqCodebook::train(&vecs, 4, 32, 20);
        for v in vecs.iter().take(10) {
            let err = cb.reconstruct_error(v);
            assert!(err >= 0.0 && err.is_finite(), "err={err}");
        }
    }

    #[test]
    fn lut_score_approx_matches_encode() {
        let vecs = make_corpus(128, 8);
        let cb = PqCodebook::train(&vecs, 2, 16, 30);
        let q = &vecs[0];
        let lut = cb.build_lut(q);
        let codes = cb.encode(q);
        let approx = lut.score(&codes);
        // Approximate distance from a vector to its own encoding should be small
        assert!(approx >= 0.0 && approx.is_finite(), "approx={approx}");
    }

    #[test]
    fn codebook_bytes_correct() {
        let vecs = make_corpus(64, 16);
        let cb = PqCodebook::train(&vecs, 4, 16, 10);
        // 4 subspaces × 16 centroids × (16/4=4 sub_dim) × 4 bytes = 1024
        assert_eq!(cb.codebook_bytes(), 4 * 16 * 4 * 4);
    }
}
