//! Minimal Product Quantization (PQ) for Tier 3B.
//!
//! # Scope
//!
//! This is a *pragmatic* PQ implementation — not a research-grade reference.
//! It exists to pair with [`crate::pq_corrector::PqDistanceCorrector`] and
//! prove that PQ codes + a learned distance corrector can hold recall on
//! SIFT1M at 4× memory reduction. No SIMD, no OPQ, no IVF.
//!
//! # Layout
//!
//! A `d`-dim vector is split into `n_subspaces` contiguous chunks, each of
//! size `d / n_subspaces`. Each chunk is quantized to one of `n_centroids`
//! centroids (u8 index), giving one byte per subspace. For SIFT1M at
//! `d = 128`, `n_subspaces = 8`, `n_centroids = 256`, each vector compresses
//! from 128 × 4 B = 512 B to 8 B (64×).
//!
//! # Distance
//!
//! Asymmetric PQ distance: for a query `q`, precompute a `n_subspaces ×
//! n_centroids` table `D[s][c] = ||q_s - centroid[s][c]||²`. Distance to a
//! coded vector is then the sum of `D[s][code[s]]` across subspaces — a pure
//! table lookup. This is ~8 adds per distance for SIFT1M.

use serde::{Deserialize, Serialize};

/// Maximum iterations we will ever run k-means for a single subspace.
/// Hard cap so we never burn the CPU when the caller passes a silly value.
const KMEANS_HARD_CAP: usize = 100;

/// Early-stop threshold on relative MSE change between k-means iterations.
/// Each subspace stops early when `(prev_mse - mse) / prev_mse < EPS`.
const KMEANS_CONVERGE_EPS: f64 = 1e-4;

/// A trained PQ codebook.
///
/// `centroids[s][c]` is the `c`-th centroid of subspace `s`, a vector of
/// length `sub_dim = d / n_subspaces`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PqCodebook {
    pub n_subspaces: usize,
    pub n_centroids: u16, // up to 256 but store as u16 for room
    pub sub_dim: usize,
    pub centroids: Vec<Vec<Vec<f32>>>,
    /// Mean squared reconstruction error per subspace from the final iter.
    /// Exposed so tests can print convergence evidence.
    pub final_mse_per_subspace: Vec<f64>,
    /// Number of iterations actually run per subspace (early-stop aware).
    pub iters_per_subspace: Vec<usize>,
}

impl PqCodebook {
    /// Dimension of the input vectors this codebook was trained for.
    pub fn dim(&self) -> usize {
        self.n_subspaces * self.sub_dim
    }

    /// Bytes used per encoded vector (currently one byte per subspace).
    pub fn code_bytes(&self) -> usize {
        debug_assert!(self.n_centroids <= 256, "u8 codes require ≤256 centroids");
        self.n_subspaces
    }

    /// Mean of `final_mse_per_subspace`.
    pub fn mean_final_mse(&self) -> f64 {
        if self.final_mse_per_subspace.is_empty() {
            0.0
        } else {
            self.final_mse_per_subspace.iter().sum::<f64>()
                / self.final_mse_per_subspace.len() as f64
        }
    }

    /// Encode a single full-dim vector as one byte per subspace.
    ///
    /// Panics if `v.len() != self.dim()`.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.dim(), "encode expects dim == n_subspaces*sub_dim");
        let mut code = Vec::with_capacity(self.n_subspaces);
        for s in 0..self.n_subspaces {
            let start = s * self.sub_dim;
            let sub = &v[start..start + self.sub_dim];
            let mut best_c = 0u16;
            let mut best_d = f64::MAX;
            for (c, centroid) in self.centroids[s].iter().enumerate() {
                let mut d = 0.0f64;
                for i in 0..self.sub_dim {
                    let diff = (sub[i] - centroid[i]) as f64;
                    d += diff * diff;
                    if d >= best_d {
                        break;
                    }
                }
                if d < best_d {
                    best_d = d;
                    best_c = c as u16;
                }
            }
            code.push(best_c as u8);
        }
        code
    }

    /// Encode a batch of vectors.
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<u8>> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Reconstruct a full-dim vector from its PQ code (used for HNSW graph
    /// storage so we can reuse the stock L2 distance).
    pub fn reconstruct(&self, code: &[u8]) -> Vec<f32> {
        assert_eq!(code.len(), self.n_subspaces, "code length mismatch");
        let mut out = Vec::with_capacity(self.dim());
        for s in 0..self.n_subspaces {
            let c = code[s] as usize;
            out.extend_from_slice(&self.centroids[s][c]);
        }
        out
    }

    /// Precompute the asymmetric-distance lookup table for a query.
    ///
    /// Returns a flat `n_subspaces * n_centroids` table of squared-L2
    /// distances between each subspace slice of the query and each centroid.
    /// Row-major: `table[s * n_centroids + c]`.
    pub fn build_query_table(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim(), "query dim mismatch");
        let nc = self.n_centroids as usize;
        let mut table = vec![0.0f32; self.n_subspaces * nc];
        for s in 0..self.n_subspaces {
            let start = s * self.sub_dim;
            let q_sub = &query[start..start + self.sub_dim];
            for c in 0..nc {
                let centroid = &self.centroids[s][c];
                let mut d = 0.0f32;
                for i in 0..self.sub_dim {
                    let diff = q_sub[i] - centroid[i];
                    d += diff * diff;
                }
                table[s * nc + c] = d;
            }
        }
        table
    }

    /// Asymmetric squared-L2 distance from query (pre-tabulated) to a code.
    ///
    /// `table` must come from [`Self::build_query_table`] with the same
    /// codebook and the query that is being scored.
    #[inline]
    pub fn asymmetric_distance_with_table(&self, table: &[f32], code: &[u8]) -> f32 {
        debug_assert_eq!(code.len(), self.n_subspaces);
        let nc = self.n_centroids as usize;
        let mut sum = 0.0f32;
        for s in 0..self.n_subspaces {
            sum += table[s * nc + code[s] as usize];
        }
        sum
    }

    /// Convenience: compute asymmetric distance directly from a query.
    ///
    /// Rebuilds the query table each call; for batched scoring, build the
    /// table once and use [`Self::asymmetric_distance_with_table`].
    pub fn asymmetric_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        let table = self.build_query_table(query);
        self.asymmetric_distance_with_table(&table, code)
    }
}

/// Train a PQ codebook via naive per-subspace k-means.
///
/// - `vectors`: training set. All vectors must have length
///   `n_subspaces * (d / n_subspaces)`; `d` is inferred from `vectors[0]`.
/// - `n_subspaces`: number of subspaces (and bytes per code).
/// - `n_centroids`: number of centroids per subspace (must be ≤ 256 for u8
///   codes).
/// - `iters`: maximum k-means iterations per subspace (capped at
///   [`KMEANS_HARD_CAP`]; relative-MSE early-stop applies).
pub fn train(
    vectors: &[Vec<f32>],
    n_subspaces: usize,
    n_centroids: u16,
    iters: usize,
) -> PqCodebook {
    assert!(!vectors.is_empty(), "cannot train PQ on empty training set");
    let d = vectors[0].len();
    assert!(
        d % n_subspaces == 0,
        "dim {d} must be divisible by n_subspaces {n_subspaces}"
    );
    assert!((2..=256).contains(&(n_centroids as u32)), "n_centroids must be in 2..=256 for u8 codes");
    assert!(
        vectors.len() >= n_centroids as usize,
        "need at least n_centroids training vectors; got {}",
        vectors.len()
    );

    let sub_dim = d / n_subspaces;
    let nc = n_centroids as usize;
    let iters = iters.min(KMEANS_HARD_CAP);

    let mut centroids: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_subspaces);
    let mut final_mse_per_subspace = Vec::with_capacity(n_subspaces);
    let mut iters_per_subspace = Vec::with_capacity(n_subspaces);

    for s in 0..n_subspaces {
        let start = s * sub_dim;
        // Gather the s-th slice of each training vector.
        let sub_vectors: Vec<&[f32]> = vectors
            .iter()
            .map(|v| &v[start..start + sub_dim])
            .collect();

        // Deterministic farthest-first-ish init: pick every (n / nc) sample
        // as an initial centroid. Using a seeded LCG would be more principled
        // but strided sampling is enough for SIFT where training samples are
        // already iid.
        let mut cents: Vec<Vec<f32>> = Vec::with_capacity(nc);
        let stride = (sub_vectors.len() / nc).max(1);
        for c in 0..nc {
            let idx = (c * stride) % sub_vectors.len();
            cents.push(sub_vectors[idx].to_vec());
        }

        let mut prev_mse = f64::MAX;
        let mut actual_iters = 0usize;
        let mut final_mse = 0.0f64;
        for it in 0..iters {
            actual_iters = it + 1;
            // Assignment step.
            let mut sums: Vec<Vec<f64>> = vec![vec![0.0; sub_dim]; nc];
            let mut counts: Vec<usize> = vec![0; nc];
            let mut total_sq_err = 0.0f64;

            for v in &sub_vectors {
                let mut best_c = 0usize;
                let mut best_d = f64::MAX;
                for (c, centroid) in cents.iter().enumerate() {
                    let mut d = 0.0f64;
                    for i in 0..sub_dim {
                        let diff = (v[i] - centroid[i]) as f64;
                        d += diff * diff;
                        if d >= best_d {
                            break;
                        }
                    }
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                for i in 0..sub_dim {
                    sums[best_c][i] += v[i] as f64;
                }
                counts[best_c] += 1;
                total_sq_err += best_d;
            }

            // Update step.
            for c in 0..nc {
                if counts[c] == 0 {
                    // Empty cluster: re-seed from a random-ish point.
                    let idx = (c * 2654435761u64.wrapping_mul(it as u64 + 1) as usize)
                        % sub_vectors.len();
                    cents[c] = sub_vectors[idx].to_vec();
                    continue;
                }
                let n = counts[c] as f64;
                for i in 0..sub_dim {
                    cents[c][i] = (sums[c][i] / n) as f32;
                }
            }

            let mse = total_sq_err / sub_vectors.len() as f64;
            final_mse = mse;
            let rel = if prev_mse > 0.0 && prev_mse.is_finite() {
                (prev_mse - mse).abs() / prev_mse
            } else {
                1.0
            };
            prev_mse = mse;
            if rel < KMEANS_CONVERGE_EPS && it > 0 {
                break;
            }
        }

        centroids.push(cents);
        final_mse_per_subspace.push(final_mse);
        iters_per_subspace.push(actual_iters);

        let _ = s; // silence unused when s index is implicit
    }

    PqCodebook {
        n_subspaces,
        n_centroids,
        sub_dim,
        centroids,
        final_mse_per_subspace,
        iters_per_subspace,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        // Deterministic LCG.
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                s = s.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f32 / u32::MAX as f32) - 0.5;
                v.push(u);
            }
            out.push(v);
        }
        out
    }

    #[test]
    fn train_produces_expected_shape() {
        let data = random_vecs(512, 32, 1);
        let book = train(&data, 4, 16, 10);
        assert_eq!(book.n_subspaces, 4);
        assert_eq!(book.sub_dim, 8);
        assert_eq!(book.centroids.len(), 4);
        for cs in &book.centroids {
            assert_eq!(cs.len(), 16);
            for c in cs {
                assert_eq!(c.len(), 8);
            }
        }
        assert_eq!(book.iters_per_subspace.len(), 4);
    }

    #[test]
    fn encode_round_trip_length() {
        let data = random_vecs(256, 16, 2);
        let book = train(&data, 4, 8, 5);
        let code = book.encode(&data[0]);
        assert_eq!(code.len(), 4);
        for c in &code {
            assert!(*c < 8);
        }
    }

    #[test]
    fn asymmetric_distance_matches_precomputed() {
        let data = random_vecs(256, 16, 3);
        let book = train(&data, 4, 8, 5);
        let code = book.encode(&data[5]);
        let q = &data[7];
        let d_direct = book.asymmetric_distance(q, &code);
        let table = book.build_query_table(q);
        let d_table = book.asymmetric_distance_with_table(&table, &code);
        assert!((d_direct - d_table).abs() < 1e-5);
    }

    #[test]
    fn reconstruct_has_right_dim() {
        let data = random_vecs(256, 16, 4);
        let book = train(&data, 4, 8, 5);
        let code = book.encode(&data[0]);
        let r = book.reconstruct(&code);
        assert_eq!(r.len(), 16);
    }

    #[test]
    fn kmeans_mse_decreases_across_iters() {
        // Cluster-structured synthetic: mse after more iters should not be
        // higher than mse after fewer iters on the same init scheme.
        let data = random_vecs(1024, 32, 5);
        let a = train(&data, 4, 16, 2);
        let b = train(&data, 4, 16, 20);
        assert!(b.mean_final_mse() <= a.mean_final_mse() + 1e-6);
    }
}
