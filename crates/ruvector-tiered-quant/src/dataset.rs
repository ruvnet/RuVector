//! Deterministic synthetic dataset generation for benchmarking.
//!
//! Uses a simple LCG PRNG — no external crates required — so benchmark runs
//! are fully reproducible across machines.

/// Lightweight LCG PRNG (Knuth constants; same as speculative-ann crate).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed ^ 0xcafe_babe_dead_beef }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform f32 in [-1.0, 1.0].
    fn next_f32(&mut self) -> f32 {
        // Use the upper 32 bits for good statistical quality
        let bits = (self.next_u64() >> 32) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Generate `n` random f32 vectors of `dims` dimensions and `nq` query vectors.
///
/// Vectors are sampled uniformly in [-1, 1]^dims.
pub fn generate_dataset(
    n: usize,
    dims: usize,
    nq: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut lcg = Lcg::new(seed);

    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dims).map(|_| lcg.next_f32()).collect())
        .collect();

    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|_| (0..dims).map(|_| lcg.next_f32()).collect())
        .collect();

    (vecs, queries)
}

/// Generate a dataset with a "hot" cluster: the first `hot_n` vectors are
/// concentrated near the origin, mimicking a frequently-accessed topic cluster.
pub fn generate_clustered_dataset(
    n: usize,
    dims: usize,
    nq: usize,
    hot_n: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut lcg = Lcg::new(seed);

    let mut vecs = Vec::with_capacity(n);

    // Hot cluster: tight around a random centroid.
    let centroid: Vec<f32> = (0..dims).map(|_| lcg.next_f32() * 0.3).collect();
    for _ in 0..hot_n.min(n) {
        let v: Vec<f32> = centroid.iter().map(|&c| c + lcg.next_f32() * 0.05).collect();
        vecs.push(v);
    }
    // Remaining: random background.
    for _ in hot_n..n {
        let v: Vec<f32> = (0..dims).map(|_| lcg.next_f32()).collect();
        vecs.push(v);
    }

    // Queries: 50% near the hot cluster, 50% random.
    let mut queries = Vec::with_capacity(nq);
    for i in 0..nq {
        let v: Vec<f32> = if i % 2 == 0 {
            centroid.iter().map(|&c| c + lcg.next_f32() * 0.1).collect()
        } else {
            (0..dims).map(|_| lcg.next_f32()).collect()
        };
        queries.push(v);
    }

    (vecs, queries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_is_deterministic() {
        let (a, _) = generate_dataset(10, 8, 2, 42);
        let (b, _) = generate_dataset(10, 8, 2, 42);
        assert_eq!(a, b, "same seed must produce identical dataset");
    }

    #[test]
    fn dataset_sizes_match() {
        let (vecs, queries) = generate_dataset(100, 32, 20, 1);
        assert_eq!(vecs.len(), 100);
        assert_eq!(queries.len(), 20);
        assert!(vecs.iter().all(|v| v.len() == 32));
    }

    #[test]
    fn dataset_has_mixed_signs() {
        let (vecs, _) = generate_dataset(50, 64, 0, 777);
        // Each vector should have roughly half positive, half negative dimensions
        for v in &vecs {
            let pos = v.iter().filter(|&&x| x > 0.0).count();
            // For 64-dim, expect roughly 20-44 positive (Binomial(64, 0.5))
            assert!(pos > 5 && pos < 59, "dataset values appear non-uniform: {pos} positive dims");
        }
    }
}
