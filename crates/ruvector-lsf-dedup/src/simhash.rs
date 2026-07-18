//! SimHash fingerprinting for float vectors.
//!
//! Projects `dim`-dimensional vectors onto `bits` random hyperplanes and
//! encodes the sign of each projection as one bit in a 64-bit hash.
//! Hamming distance between two hashes estimates the angle between the
//! original vectors, and therefore their cosine similarity.
//!
//! Formula:  cos_sim ≈ cos(π · hamming / bits)

use crate::SemanticFingerprinter;
use std::f32::consts::PI;

/// A 64-bit SimHash fingerprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimHashFp {
    pub hash: u64,
}

/// Hasher that maps float vectors to SimHash fingerprints.
pub struct SimHasher {
    /// Row-major matrix: `bits` rows, each `dim` floats.
    planes: Vec<f32>,
    bits: usize,
    dim: usize,
}

impl SimHasher {
    /// Create a new `SimHasher` with `bits` hyperplanes (≤ 64) for
    /// vectors of length `dim`.  `seed` controls deterministic generation.
    pub fn new(dim: usize, bits: usize, seed: u64) -> Self {
        assert!(bits <= 64, "SimHasher supports at most 64 bits");
        assert!(dim > 0, "dimension must be positive");
        let planes = generate_planes(dim, bits, seed);
        Self { planes, bits, dim }
    }

    /// Hamming distance between two 64-bit hashes, masked to `self.bits`.
    pub fn hamming(a: u64, b: u64, bits: usize) -> u32 {
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        ((a ^ b) & mask).count_ones()
    }

    /// Convert hamming distance to estimated cosine similarity.
    pub fn hamming_to_cos(hamming: u32, bits: usize) -> f32 {
        (PI * hamming as f32 / bits as f32).cos()
    }
}

/// Deterministic, seedable normal-distribution plane generator.
/// Uses a simple LCG to produce uniform samples and Box-Muller transform.
fn generate_planes(dim: usize, num_planes: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0xdeadbeef_cafebabe;
    let mut planes = Vec::with_capacity(dim * num_planes);

    for _ in 0..num_planes {
        let mut row = Vec::with_capacity(dim);
        // Box-Muller pairs
        let mut i = 0;
        while i < dim {
            let u1 = lcg_f32(&mut state);
            let u2 = lcg_f32(&mut state);
            let u1 = (u1 + 1e-10).min(1.0 - 1e-10);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            row.push(r * theta.cos());
            if i + 1 < dim {
                row.push(r * theta.sin());
            }
            i += 2;
        }
        row.truncate(dim);
        // normalise the hyperplane
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
        planes.extend_from_slice(&row);
    }
    planes
}

fn lcg_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // extract upper 23 mantissa bits → uniform [0, 1)
    let mantissa = (*state >> 41) as u32;
    f32::from_bits(0x3f80_0000 | mantissa) - 1.0
}

impl SemanticFingerprinter for SimHasher {
    type Fingerprint = SimHashFp;

    fn fingerprint(&self, vec: &[f32]) -> SimHashFp {
        debug_assert_eq!(vec.len(), self.dim, "vector dimension mismatch");
        let mut hash = 0u64;
        for b in 0..self.bits {
            let plane = &self.planes[b * self.dim..(b + 1) * self.dim];
            let dot: f32 = vec.iter().zip(plane.iter()).map(|(a, p)| a * p).sum();
            if dot >= 0.0 {
                hash |= 1u64 << b;
            }
        }
        SimHashFp { hash }
    }

    fn estimate_similarity(&self, a: &SimHashFp, b: &SimHashFp) -> f32 {
        let h = Self::hamming(a.hash, b.hash, self.bits);
        Self::hamming_to_cos(h, self.bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cosine_similarity, normalize};

    fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut v: Vec<f32> = (0..dim).map(|_| lcg_f32(&mut state) * 2.0 - 1.0).collect();
        normalize(&mut v);
        v
    }

    #[test]
    fn identical_vectors_hash_equal() {
        let dim = 64;
        let hasher = SimHasher::new(dim, 32, 42);
        let v = random_unit_vector(dim, 1);
        let fp_a = hasher.fingerprint(&v);
        let fp_b = hasher.fingerprint(&v);
        assert_eq!(fp_a.hash, fp_b.hash);
        assert!((hasher.estimate_similarity(&fp_a, &fp_b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similar_vectors_have_high_estimated_sim() {
        let dim = 128;
        let hasher = SimHasher::new(dim, 64, 99);
        let mut v = random_unit_vector(dim, 2);
        normalize(&mut v);
        // near-duplicate: small perturbation
        let mut v2 = v.clone();
        for (i, x) in v2.iter_mut().enumerate() {
            *x += 0.02 * ((i as f32).sin());
        }
        normalize(&mut v2);
        let exact = cosine_similarity(&v, &v2);
        let fp_a = hasher.fingerprint(&v);
        let fp_b = hasher.fingerprint(&v2);
        let est = hasher.estimate_similarity(&fp_a, &fp_b);
        // estimated should be within 0.2 of exact for highly similar vectors
        assert!(
            (est - exact).abs() < 0.25,
            "exact={exact:.3} est={est:.3} diff={:.3}",
            (est - exact).abs()
        );
        assert!(
            est > 0.7,
            "similar vectors should have est_sim > 0.7, got {est:.3}"
        );
    }

    #[test]
    fn orthogonal_vectors_have_low_estimated_sim() {
        let dim = 128;
        let hasher = SimHasher::new(dim, 64, 7);
        // construct two nearly orthogonal unit vectors
        let mut a = vec![0.0f32; dim];
        let mut b = vec![0.0f32; dim];
        a[0] = 1.0;
        b[1] = 1.0;
        let fp_a = hasher.fingerprint(&a);
        let fp_b = hasher.fingerprint(&b);
        let est = hasher.estimate_similarity(&fp_a, &fp_b);
        assert!(
            est.abs() < 0.3,
            "orthogonal vectors should have |est_sim| < 0.3, got {est:.3}"
        );
    }

    #[test]
    fn hamming_to_cos_identity() {
        // hamming=0 → similarity=1.0
        assert!((SimHasher::hamming_to_cos(0, 32) - 1.0).abs() < 1e-5);
        // hamming=bits/2 → similarity≈0
        let bits = 32;
        let mid = SimHasher::hamming_to_cos(bits as u32 / 2, bits);
        assert!(
            mid.abs() < 0.1,
            "half hamming should be near 0, got {mid:.3}"
        );
        // hamming=bits → similarity=-1
        let anti = SimHasher::hamming_to_cos(bits as u32, bits);
        assert!(
            (anti + 1.0).abs() < 1e-5,
            "full hamming should be -1, got {anti:.3}"
        );
    }
}
