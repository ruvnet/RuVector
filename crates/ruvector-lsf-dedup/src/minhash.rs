//! MinHash fingerprinting for float vectors.
//!
//! Discretises each vector dimension into `bins` quantisation levels, then
//! applies `num_hashes` universal-hash functions to the resulting feature set.
//! The fraction of hash functions that produce equal minima approximates the
//! Jaccard similarity between the feature sets of two vectors.
//!
//! Near-duplicate float vectors (cosine > 0.95) will share most of their
//! discretised bins, yielding high estimated Jaccard similarity.
//!
//! ## Details
//! - Feature for dimension `i` with value `v` → `(i * bins + bin_index(v))`
//! - Hash function: `(a * feature + b) mod P`, Mersenne prime P = 2^31 - 1
//! - Signature: `[min_hash(h_1), ..., min_hash(h_k)]`

use crate::SemanticFingerprinter;

const MERSENNE: u64 = (1u64 << 31) - 1;

/// MinHash fingerprint: a k-length array of minimum hash values.
#[derive(Clone, Debug)]
pub struct MinHashFp {
    pub signature: Vec<u32>,
}

/// Hasher that maps float vectors to MinHash signatures.
pub struct MinHasher {
    /// (a, b) coefficients for each hash function.
    coeffs: Vec<(u64, u64)>,
    num_hashes: usize,
    bins: usize,
    dim: usize,
}

impl MinHasher {
    /// Create a new MinHasher.
    ///
    /// `dim`        — embedding dimension
    /// `num_hashes` — length of signature (more → better recall estimate)
    /// `bins`       — quantisation levels per dimension (8 is a good default)
    /// `seed`       — deterministic initialisation
    pub fn new(dim: usize, num_hashes: usize, bins: usize, seed: u64) -> Self {
        assert!(dim > 0 && num_hashes > 0 && bins >= 2);
        let coeffs = gen_coefficients(num_hashes, seed);
        Self {
            coeffs,
            num_hashes,
            bins,
            dim,
        }
    }

    /// Convert a float vector to a set of discrete feature integers.
    ///
    /// Vectors are first unit-normalised so that scale-variant duplicates
    /// (the same semantic content at different magnitudes) still share bins.
    fn featurize(&self, vec: &[f32]) -> Vec<u64> {
        // project onto unit sphere
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);

        let mut features = Vec::with_capacity(self.dim);
        for (i, &v) in vec.iter().enumerate() {
            let normalised = v / norm; // in [-1, 1]
                                       // map [-1, 1] → [0, bins)
            let t = ((normalised + 1.0) * 0.5 * self.bins as f32) as u64;
            let bin = t.min(self.bins as u64 - 1);
            features.push(i as u64 * self.bins as u64 + bin);
        }
        features
    }

    fn min_hash_for(&self, features: &[u64], a: u64, b: u64) -> u32 {
        features
            .iter()
            .map(|&f| {
                // (a * f + b) mod MERSENNE, avoiding overflow via u128
                let v = ((a as u128 * f as u128 + b as u128) % MERSENNE as u128) as u64;
                v as u32
            })
            .min()
            .unwrap_or(u32::MAX)
    }
}

fn gen_coefficients(n: usize, seed: u64) -> Vec<(u64, u64)> {
    let mut state = seed ^ 0x12345678_9abcdef0;
    (0..n)
        .map(|_| {
            let a = (lcg64(&mut state) % (MERSENNE - 1)) + 1; // a ∈ [1, P-1]
            let b = lcg64(&mut state) % MERSENNE; // b ∈ [0, P-1]
            (a, b)
        })
        .collect()
}

fn lcg64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

impl SemanticFingerprinter for MinHasher {
    type Fingerprint = MinHashFp;

    fn fingerprint(&self, vec: &[f32]) -> MinHashFp {
        debug_assert_eq!(vec.len(), self.dim, "vector dimension mismatch");
        let features = self.featurize(vec);
        let signature = self
            .coeffs
            .iter()
            .map(|&(a, b)| self.min_hash_for(&features, a, b))
            .collect();
        MinHashFp { signature }
    }

    fn estimate_similarity(&self, a: &MinHashFp, b: &MinHashFp) -> f32 {
        let matches = a
            .signature
            .iter()
            .zip(b.signature.iter())
            .filter(|(x, y)| x == y)
            .count();
        matches as f32 / self.num_hashes as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state as i64 as f32) / i64::MAX as f32
            })
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn identical_vectors_full_signature_match() {
        let dim = 64;
        let hasher = MinHasher::new(dim, 64, 8, 1);
        let v = unit_vec(dim, 10);
        let fp1 = hasher.fingerprint(&v);
        let fp2 = hasher.fingerprint(&v);
        let sim = hasher.estimate_similarity(&fp1, &fp2);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors: sim={sim:.4}");
    }

    #[test]
    fn near_duplicate_high_minhash_sim() {
        let dim = 128;
        let hasher = MinHasher::new(dim, 128, 8, 2);
        let v1 = unit_vec(dim, 20);
        // near-duplicate: add tiny noise
        let norm: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut v2 = v1.clone();
        for (i, x) in v2.iter_mut().enumerate() {
            *x += 0.01 * ((i as f32 + 1.0).sin()) / norm;
        }
        let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v2.iter_mut() {
            *x /= norm2;
        }

        let fp1 = hasher.fingerprint(&v1);
        let fp2 = hasher.fingerprint(&v2);
        let sim = hasher.estimate_similarity(&fp1, &fp2);
        assert!(
            sim > 0.5,
            "near-duplicate should have MinHash sim > 0.5, got {sim:.3}"
        );
    }

    #[test]
    fn distinct_vectors_low_minhash_sim() {
        let dim = 128;
        let hasher = MinHasher::new(dim, 128, 8, 3);
        let v1 = unit_vec(dim, 30);
        let v2 = unit_vec(dim, 31_000);
        let fp1 = hasher.fingerprint(&v1);
        let fp2 = hasher.fingerprint(&v2);
        let sim = hasher.estimate_similarity(&fp1, &fp2);
        assert!(
            sim < 0.5,
            "distinct vectors should have MinHash sim < 0.5, got {sim:.3}"
        );
    }

    #[test]
    fn scale_invariant_near_duplicate() {
        // scale-variant duplicate: same direction, 10x larger magnitude
        let dim = 64;
        let hasher = MinHasher::new(dim, 64, 8, 4);
        let v1 = unit_vec(dim, 40);
        let v2: Vec<f32> = v1.iter().map(|x| x * 10.0).collect();
        let fp1 = hasher.fingerprint(&v1);
        let fp2 = hasher.fingerprint(&v2);
        let sim = hasher.estimate_similarity(&fp1, &fp2);
        // After internal normalisation both should produce nearly identical features
        assert!(
            sim > 0.8,
            "scale-variant duplicate should have high sim, got {sim:.3}"
        );
    }
}
