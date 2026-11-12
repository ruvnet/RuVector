//! Deterministic dataset generation using a simple LCG.
//! No external `rand` dependency required.

/// Minimal LCG pseudo-random generator (Knuth parameters).
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 6364136223846793005,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Sample uniformly in [-1.0, 1.0].
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u64();
        let f = (u >> 11) as f32 / (1u64 << 53) as f32; // [0, 1)
        f * 2.0 - 1.0
    }
}

/// Generate `n` random f32 vectors of `dim` dimensions.
pub fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
        .collect()
}

/// Generate `nq` query vectors (different seed to avoid overlap with corpus).
pub fn generate_queries(nq: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    generate_vectors(nq, dim, seed.wrapping_add(999_999_937))
}

/// Pick `k` initial centroid indices (k-means++ style, deterministic).
/// First centroid is index 0; each subsequent centroid maximises min-distance
/// to the already-chosen centroids.
pub fn initial_centroid_indices(vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
    let n = vectors.len();
    assert!(k <= n, "k must be ≤ n");
    let mut chosen = vec![0usize];
    for _ in 1..k {
        // For each point, compute min squared distance to any chosen centroid.
        let mut max_dist = f32::NEG_INFINITY;
        let mut best = 0;
        for i in 0..n {
            if chosen.contains(&i) {
                continue;
            }
            let d = chosen
                .iter()
                .map(|&c| crate::l2_sq(&vectors[i], &vectors[c]))
                .fold(f32::INFINITY, f32::min);
            if d > max_dist {
                max_dist = d;
                best = i;
            }
        }
        chosen.push(best);
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_produces_distinct_values() {
        let mut rng = Lcg::new(42);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b);
    }

    #[test]
    fn generated_vectors_have_correct_shape() {
        let vecs = generate_vectors(100, 32, 0);
        assert_eq!(vecs.len(), 100);
        assert_eq!(vecs[0].len(), 32);
    }

    #[test]
    fn initial_centroids_are_distinct() {
        let vecs = generate_vectors(100, 32, 7);
        let indices = initial_centroid_indices(&vecs, 8);
        let unique: std::collections::HashSet<_> = indices.iter().copied().collect();
        assert_eq!(unique.len(), 8);
    }
}
