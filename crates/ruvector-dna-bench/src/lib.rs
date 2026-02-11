//! RuVector DNA Analyzer Benchmark Suite
//!
//! Comprehensive benchmarks for core crates covering:
//! - HNSW index operations (genomic k-mer embeddings)
//! - Min-cut on genome-scale graphs
//! - Attention mechanisms for genomic sequences
//! - Delta propagation and checkpointing
//! - Quantization throughput and accuracy

/// Deterministic pseudo-random vector generator for reproducible benchmarks.
///
/// Uses a fast LCG-based PRNG seeded per-vector so results are consistent
/// across runs without pulling in `rand` as a benchmark dependency.
pub fn deterministic_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((state >> 33) ^ state) as u32;
            // Map to [-1.0, 1.0]
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Generate a batch of deterministic vectors.
pub fn deterministic_vectors(count: usize, dim: usize, base_seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| deterministic_vector(dim, base_seed.wrapping_add(i as u64)))
        .collect()
}

/// Normalize a vector to unit length (for cosine-based indices).
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}
