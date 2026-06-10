/// Dot product of two equal-length slices. Both must be L2-normalised for cosine semantics.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 squared distance.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Normalize a vector to unit length in place.
pub fn normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// MaxSim score for one query against one document.
///
/// MaxSim(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} cosine(q, d)
///
/// With normalised vectors: cosine(q, d) = dot(q, d).
pub fn maxsim_score(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
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

/// SQ8 scalar quantization: f32 ∈ [-1, 1] → i8 ∈ [-127, 127].
pub struct Sq8Codec {
    pub dim: usize,
}

impl Sq8Codec {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn encode(&self, v: &[f32]) -> Vec<i8> {
        v.iter()
            .map(|&x| (x.clamp(-1.0, 1.0) * 127.0).round() as i8)
            .collect()
    }

    /// Integer dot product, dequantized to float.
    pub fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
        let sum: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        sum as f32 / (127.0 * 127.0)
    }

    pub fn bytes_per_token(&self) -> usize {
        self.dim
    }
}

/// k-means clustering (Lloyd's algorithm).
///
/// Returns `k` centroids computed from `tokens` with `iters` iterations.
/// Uses a seeded RNG for reproducibility.
pub fn kmeans_centroids(
    tokens: &[Vec<f32>],
    k: usize,
    dim: usize,
    iters: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    use rand::Rng;
    use rand::SeedableRng;

    let n = tokens.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Initialize centroids by random unique sampling.
    let mut chosen = std::collections::HashSet::new();
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    while centroids.len() < k {
        let idx = rng.gen_range(0..n);
        if chosen.insert(idx) {
            centroids.push(tokens[idx].clone());
        }
    }

    for _ in 0..iters {
        // Assignment: find nearest centroid for each token.
        let assignments: Vec<usize> = tokens
            .iter()
            .map(|tok| {
                (0..k)
                    .min_by(|&a, &b| {
                        l2sq(&centroids[a], tok)
                            .partial_cmp(&l2sq(&centroids[b], tok))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            })
            .collect();

        // Update: recompute centroid as mean of assigned tokens.
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0_f32; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for (i, &c) in assignments.iter().enumerate() {
            for (j, &x) in tokens[i].iter().enumerate() {
                sums[c][j] += x;
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cnt = counts[c] as f32;
                for x in &mut sums[c] {
                    *x /= cnt;
                }
                centroids[c] = sums[c].clone();
            } else {
                centroids[c] = tokens[rng.gen_range(0..n)].clone();
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_unit_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        assert!((dot(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn maxsim_identical_query_and_doc() {
        let tok = vec![vec![1.0_f32, 0.0, 0.0], vec![0.0_f32, 1.0, 0.0]];
        let score = maxsim_score(&tok, &tok);
        // Each query token exactly matches one doc token → score = 2.0
        assert!((score - 2.0).abs() < 1e-5, "score={score}");
    }

    #[test]
    fn sq8_roundtrip_accuracy() {
        let codec = Sq8Codec::new(4);
        let v = vec![0.5_f32, -0.5, 1.0, -1.0];
        let enc = codec.encode(&v);
        // Re-decode manually and check approximate reconstruction
        let dec: Vec<f32> = enc.iter().map(|&x| x as f32 / 127.0).collect();
        for (a, b) in v.iter().zip(dec.iter()) {
            assert!(
                (a - b).abs() < 0.02,
                "roundtrip error too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn kmeans_returns_k_centroids() {
        let tokens: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32, (i % 10) as f32]).collect();
        let centroids = kmeans_centroids(&tokens, 8, 2, 5, 42);
        assert_eq!(centroids.len(), 8);
    }
}
