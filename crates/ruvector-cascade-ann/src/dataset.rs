//! Deterministic synthetic dataset generator.
//!
//! Uses a Lehmer LCG (no external deps) seeded at a fixed value so every
//! benchmark run produces identical vectors.  Vectors are drawn from a
//! mixture of `n_clusters` Gaussian clusters to create non-trivial recall
//! structure (pure uniform random makes every brute-force answer equally hard).

/// Multiplicative LCG — returns values in [0, 1).
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}

/// Draw a sample from N(0,1) via Box-Muller.
fn standard_normal(state: &mut u64) -> f32 {
    let u1 = lcg_next(state).max(1e-10);
    let u2 = lcg_next(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Generate `n` random f32 vectors of dimension `dim` from a mixture of
/// `n_clusters` isotropic Gaussian clusters.  All arithmetic is deterministic.
pub fn generate_clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed ^ 0xDEAD_BEEF_CAFE_BABE;

    // Build cluster centres.
    let centres: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            (0..dim)
                .map(|_| standard_normal(&mut state) * 10.0)
                .collect()
        })
        .collect();

    // Assign vectors round-robin to clusters, add small noise.
    (0..n)
        .map(|i| {
            let c = &centres[i % n_clusters];
            c.iter()
                .map(|&cx| cx + standard_normal(&mut state) * 1.0)
                .collect()
        })
        .collect()
}

/// Generate `n` pure standard-Gaussian vectors (no clustering).
/// In high dimensions, distance concentration makes quantization rank inversions
/// visible even with small quantization steps.
pub fn generate_gaussian(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed ^ 0xCAFE_BABE_1234_5678;
    (0..n)
        .map(|_| (0..dim).map(|_| standard_normal(&mut state)).collect())
        .collect()
}

/// Normalise a slice of vectors to unit length (for cosine search).
pub fn normalise(vecs: &mut Vec<Vec<f32>>) {
    for v in vecs.iter_mut() {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
