//! Distance kernels for multi-vector scoring.
//!
//! Three aggregation strategies:
//!   - `maxsim_exact`   — ColBERT MaxSim: sum_i max_j dot(q_i, d_j)
//!   - `chamfer`        — Chamfer distance (bidirectional MinSim → lower = closer)
//!   - `centroid_dot`   — pool doc/query tokens to centroid, then plain dot

/// L2-normalise a vector in-place. No-op if norm is ~0.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// ColBERT MaxSim score: sum over query tokens of the max dot product
/// against any document token. Both sides should be L2-normalised.
///
/// Complexity: O(|Q| × |D| × dim)
pub fn maxsim_exact(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    query_tokens
        .iter()
        .map(|qt| {
            doc_tokens
                .iter()
                .map(|dt| dot(qt, dt))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Centroid pooling: average all token vectors, then dot with query centroid.
/// Cheapest but losses token-level signal.
pub fn centroid_dot(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    let dim = query_tokens[0].len();
    let mut qc = vec![0.0f32; dim];
    for qt in query_tokens {
        qc.iter_mut().zip(qt.iter()).for_each(|(a, &b)| *a += b);
    }
    let qscale = 1.0 / query_tokens.len() as f32;
    qc.iter_mut().for_each(|x| *x *= qscale);

    let mut dc = vec![0.0f32; dim];
    for dt in doc_tokens {
        dc.iter_mut().zip(dt.iter()).for_each(|(a, &b)| *a += b);
    }
    let dscale = 1.0 / doc_tokens.len() as f32;
    dc.iter_mut().for_each(|x| *x *= dscale);

    dot(&qc, &dc)
}

/// Chamfer score (turned into a *higher-is-better* similarity):
///   score = -(forward_chamfer + backward_chamfer) / 2
/// where forward_chamfer  = mean_q  max_d dot(q, d)
///       backward_chamfer = mean_d  max_q dot(d, q)
///
/// Symmetric — avoids the asymmetry bias of pure MaxSim.
pub fn chamfer_score(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    let fwd: f32 = query_tokens
        .iter()
        .map(|qt| {
            doc_tokens
                .iter()
                .map(|dt| dot(qt, dt))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum::<f32>()
        / query_tokens.len() as f32;

    let bwd: f32 = doc_tokens
        .iter()
        .map(|dt| {
            query_tokens
                .iter()
                .map(|qt| dot(qt, dt))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum::<f32>()
        / doc_tokens.len() as f32;

    (fwd + bwd) / 2.0
}

/// MUVERA Fixed-Dimensional Encoding (FDE) — approximate MaxSim.
///
/// Algorithm (Karpukhin et al. 2024, simplified):
///   For each of R repetitions:
///     1. Sample a random orthogonal partition of the dim dimensions into M
///        contiguous subspaces of size dim/M each.
///     2. For each doc token, find which centroid it falls in (via argmax dot
///        with M random unit vectors — one per subspace).
///     3. Accumulate the token vector into its centroid bucket.
///   Stack buckets from all R repetitions → FDE vector of length R×M×dim.
///   Query encoded the same way; MaxSim ≈ dot(fde_q, fde_d).
///
/// We use a lightweight version: R=1, M=subspaces, clusters via top-1 random
/// projection (no k-means training). Encoding is O(T×D×M) per doc.
pub struct FdeEncoder {
    pub dim: usize,
    pub m: usize,
    pub r: usize,
    /// Random projection vectors [r][m][dim] used for cluster assignment.
    projections: Vec<Vec<Vec<f32>>>,
}

impl FdeEncoder {
    pub fn new(dim: usize, m: usize, r: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f64, 1.0).unwrap();

        let projections = (0..r)
            .map(|_| {
                (0..m)
                    .map(|_| {
                        let mut v: Vec<f32> = (0..dim)
                            .map(|_| normal.sample(&mut rng) as f32)
                            .collect();
                        l2_normalize(&mut v);
                        v
                    })
                    .collect()
            })
            .collect();
        Self { dim, m, r, projections }
    }

    /// Encode a set of token vectors into a single FDE vector of length r×m×dim.
    pub fn encode(&self, tokens: &[Vec<f32>]) -> Vec<f32> {
        let out_len = self.r * self.m * self.dim;
        let mut fde = vec![0.0f32; out_len];

        for rep in 0..self.r {
            let rep_offset = rep * self.m * self.dim;
            for tok in tokens {
                // Find which of M cluster projections this token is closest to.
                let cluster = (0..self.m)
                    .map(|c| dot(tok, &self.projections[rep][c]))
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let bucket_offset = rep_offset + cluster * self.dim;
                fde[bucket_offset..bucket_offset + self.dim]
                    .iter_mut()
                    .zip(tok.iter())
                    .for_each(|(a, &b)| *a += b);
            }
        }
        fde
    }

    /// FDE output dimension.
    pub fn fde_dim(&self) -> usize {
        self.r * self.m * self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(d: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; d];
        v[i] = 1.0;
        v
    }

    #[test]
    fn maxsim_identical_docs() {
        // Query == doc → MaxSim should equal |Q| (one 1.0 per query token).
        let d = 4;
        let tokens: Vec<Vec<f32>> = (0..4).map(|i| unit(d, i)).collect();
        let score = maxsim_exact(&tokens, &tokens);
        assert!((score - 4.0).abs() < 1e-5, "got {score}");
    }

    #[test]
    fn centroid_dot_identical() {
        let d = 4;
        let tokens: Vec<Vec<f32>> = (0..4).map(|i| unit(d, i)).collect();
        let score = centroid_dot(&tokens, &tokens);
        // centroid of 4 orthogonal unit vectors dotted with itself
        // = (0.25,0.25,0.25,0.25)·(0.25,0.25,0.25,0.25) = 0.25
        assert!((score - 0.25).abs() < 1e-5, "got {score}");
    }

    #[test]
    fn chamfer_symmetric_identical() {
        let d = 4;
        let tokens: Vec<Vec<f32>> = (0..4).map(|i| unit(d, i)).collect();
        let score = chamfer_score(&tokens, &tokens);
        // fwd = bwd = mean(max over identical set) = mean(1.0) = 1.0
        assert!((score - 1.0).abs() < 1e-5, "got {score}");
    }

    #[test]
    fn maxsim_orthogonal_docs_zero() {
        // Query token [1,0,0,0]; doc token [0,1,0,0] → MaxSim = 0.
        let q = vec![vec![1.0f32, 0.0, 0.0, 0.0]];
        let d = vec![vec![0.0f32, 1.0, 0.0, 0.0]];
        let score = maxsim_exact(&q, &d);
        assert!(score.abs() < 1e-5, "got {score}");
    }

    #[test]
    fn fde_encoder_same_doc_high_score() {
        // Two identical documents should have the same FDE, giving high dot score.
        let dim = 8;
        let m = 2;
        let r = 2;
        let enc = FdeEncoder::new(dim, m, r, 42);
        let tokens: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect();
        let fde_a = enc.encode(&tokens);
        let fde_b = enc.encode(&tokens);
        let score = dot(&fde_a, &fde_b);
        let self_score = dot(&fde_a, &fde_a);
        // Same document → fde_a == fde_b → score == self_score
        assert!((score - self_score).abs() < 1e-5, "got {score} vs self {self_score}");
    }
}
