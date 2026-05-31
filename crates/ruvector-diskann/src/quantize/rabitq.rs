//! RaBitQ-backed [`Quantizer`] implementation.
//!
//! This is the **direct-embed** integration described in the RaBitQ research
//! note (`docs/research/nightly/2026-04-23-rabitq/README.md`) and ADR-154.
//! Pattern 1 from the architectural-patterns memo: DiskANN takes a path
//! dependency on `ruvector-rabitq` and uses `RabitqIndex` directly. The
//! `VectorKernel` trait route is reserved for ruLake (ADR-156) once it wires
//! `register_kernel`.
//!
//! ## Why this is a tighter compression than PQ for DiskANN's use case
//!
//! At D=128 and M=16, PQ stores 16 bytes per code (≈ 32× compression vs f32).
//! RaBitQ stores `⌈D/8⌉ + 4` bytes per code (16 + 4 = 20 bytes at D=128, but
//! only 16 of them are the *code* — the 4-byte norm is per-vector metadata).
//! At D=768 (sentence-transformer / OpenAI embeddings) RaBitQ shrinks to 96
//! bytes vs PQ's 32 bytes for M=32 subspaces, but it gives a *theoretical*
//! O(1/√D) error bound where PQ degrades on high-D distributions.
//!
//! ## Determinism
//!
//! [`RabitqQuantizer::new`] takes an explicit `seed`. The rotation matrix and
//! resulting bit-codes are reproducible across runs given `(seed, dim,
//! vectors)` — this is what ADR-154 mandates and what `ruvector-rabitq`
//! already guarantees in its `RandomRotation::random` constructor.

use crate::error::{DiskAnnError, Result};
use crate::quantize::Quantizer;

use ruvector_rabitq::index::RabitqIndex;
use ruvector_rabitq::quantize::BinaryCode;

/// Per-query precomputed state for RaBitQ.
///
/// We use the **symmetric** Charikar-style estimator (`E[B/D] = 1 − θ/π`):
/// both query and database side are 1-bit codes, distance is computed via
/// XNOR-popcount. Two reasons over the asymmetric variant:
///
///   1. Self-query exactness: agreement = D ⇒ cos(0) = 1 ⇒ est_sq ≈ 0,
///      which makes the index trivially correct on existing vectors. The
///      asymmetric IP estimator is unbiased *in expectation* but not exact
///      on a single query, so a self-query on a single vector returns
///      `‖q‖²·(1 − √(2/π))` which surprises callers.
///   2. Hot-loop cost: O(D/64) popcount instead of O(D) f32 arithmetic.
///
/// Asymmetric is still available via [`RabitqQuantizer::inner`] for callers
/// who want it (e.g. rerank-light pipelines).
pub struct RabitqQuery {
    /// Encoded binary code for the query (rotation-aware, unit-norm).
    pub code: BinaryCode,
}

/// RaBitQ-backed quantizer. Wraps a [`RabitqIndex`] purely for its rotation
/// matrix + encoding kernel — DiskANN owns the byte storage itself.
pub struct RabitqQuantizer {
    inner: RabitqIndex,
    dim: usize,
    /// `ceil(D/64)` — the u64-word-length of the bit-packed code.
    n_words: usize,
    /// Total bytes per encoded vector: `n_words * 8` (the code) + `4` (the
    /// f32 norm). Matches what [`Self::encode`] writes and what
    /// [`Self::decode_code`] expects on the inverse path.
    code_bytes_total: usize,
    /// Whether [`Quantizer::train`] has been called. RaBitQ doesn't actually
    /// *learn* anything (the rotation is data-independent), but we still gate
    /// `encode` behind a train call to match the trait's contract.
    trained: bool,
}

impl RabitqQuantizer {
    /// Construct a fresh RaBitQ quantizer for `dim`-dimensional vectors. The
    /// `seed` controls the random rotation matrix; passing the same `(seed,
    /// dim)` pair across runs yields bit-identical codes.
    pub fn new(dim: usize, seed: u64) -> Self {
        let inner = RabitqIndex::new(dim, seed);
        let n_words = (dim + 63) / 64;
        // u64-aligned bit storage + 4-byte f32 norm. Storing the bit code at
        // u64 alignment keeps the popcount hot-path branch-free at the cost
        // of `0..7` padding bytes per vector — negligible compared to the
        // f32 baseline.
        let code_bytes_total = n_words * 8 + 4;
        Self {
            inner,
            dim,
            n_words,
            code_bytes_total,
            trained: false,
        }
    }

    /// Bytes consumed by the rotation matrix (amortised across all vectors).
    pub fn rotation_bytes(&self) -> usize {
        self.inner.rotation().bytes()
    }

    /// Underlying RabitQ encoder — exposed for tests / advanced callers.
    pub fn inner(&self) -> &RabitqIndex {
        &self.inner
    }

    /// Decode `code` back into a [`BinaryCode`] view (zero-copy on the bytes,
    /// minus the 4-byte norm header).
    fn decode_code<'a>(&self, code: &'a [u8]) -> BinaryCode {
        debug_assert_eq!(code.len(), self.code_bytes_total);
        // Layout: [n_words * 8 bytes of u64 code][4 bytes f32 norm LE].
        // We stored only `ceil(D/8)` byte-payload but fixed-padded to
        // `n_words * 8` for u64 alignment / fast popcount.
        let mut words = vec![0u64; self.n_words];
        for (i, w) in words.iter_mut().enumerate().take(self.n_words) {
            let s = i * 8;
            *w = u64::from_le_bytes(code[s..s + 8].try_into().expect("exact 8 bytes"));
        }
        let norm_off = self.n_words * 8;
        let norm = f32::from_le_bytes(
            code[norm_off..norm_off + 4]
                .try_into()
                .expect("exact 4 bytes"),
        );
        BinaryCode {
            words,
            norm,
            dim: self.dim,
        }
    }
}

impl Quantizer for RabitqQuantizer {
    type Query = RabitqQuery;

    fn dim(&self) -> usize {
        self.dim
    }

    fn code_bytes(&self) -> usize {
        // Total per-vector byte cost — match the storage layout produced by
        // `encode`. Caller can subtract `4` if they only want the code bits.
        self.n_words * 8 + 4
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, vectors: &[Vec<f32>], _iterations: usize) -> Result<()> {
        if vectors.is_empty() {
            return Err(DiskAnnError::Empty);
        }
        if vectors[0].len() != self.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.dim,
                actual: vectors[0].len(),
            });
        }
        // RaBitQ's rotation is Haar-uniform and data-independent — there is
        // nothing to fit. We still check dim consistency so a misconfigured
        // caller fails fast at train() rather than mid-encode.
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(DiskAnnError::DimensionMismatch {
                    expected: self.dim,
                    actual: v.len(),
                });
            }
            if i >= 4 {
                break;
            }
        }
        self.trained = true;
        Ok(())
    }

    fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if !self.trained {
            return Err(DiskAnnError::PqNotTrained);
        }
        if vector.len() != self.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        let bc = self.inner.encode_vector(vector);
        let mut out = Vec::with_capacity(self.code_bytes_total);
        for w in &bc.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        // BinaryCode stores `ceil(D/64) = n_words` u64s, so this is exactly
        // `n_words * 8` bytes.
        debug_assert_eq!(out.len(), self.n_words * 8);
        out.extend_from_slice(&bc.norm.to_le_bytes());
        debug_assert_eq!(out.len(), self.code_bytes_total);
        Ok(out)
    }

    fn prepare_query(&self, query: &[f32]) -> Result<Self::Query> {
        if !self.trained {
            return Err(DiskAnnError::PqNotTrained);
        }
        if query.len() != self.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        // Symmetric path: encode the query as a BinaryCode so `distance` is a
        // pure XNOR-popcount + LUT cosine. See [`RabitqQuery`] for the
        // rationale on choosing symmetric over asymmetric.
        let code = self.inner.encode_vector(query);
        Ok(RabitqQuery { code })
    }

    #[inline]
    fn distance(&self, query: &Self::Query, code: &[u8]) -> f32 {
        let bc = self.decode_code(code);
        bc.estimated_sq_distance(&query.code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                v.into_iter().map(|x| x / n).collect()
            })
            .collect()
    }

    #[test]
    fn encode_then_self_distance_is_near_zero() {
        let dim = 128;
        let mut q = RabitqQuantizer::new(dim, 42);
        let vecs = random_unit_vectors(8, dim, 7);
        q.train(&vecs, 0).unwrap();

        for v in &vecs {
            let code = q.encode(v).unwrap();
            let prep = q.prepare_query(v).unwrap();
            let d = q.distance(&prep, &code);
            // Symmetric/asym RaBitQ on a unit vector against its own code: the
            // angular estimator is exact at agreement = D, so cosθ = 1 and
            // est_sq ≈ 0. Allow a small numerical slack from f32 rounding.
            assert!(d < 1e-3, "self-distance too large: {d}");
        }
    }

    #[test]
    fn deterministic_codes_for_same_seed() {
        let dim = 96;
        let vecs = random_unit_vectors(4, dim, 9);
        let mut a = RabitqQuantizer::new(dim, 1234);
        let mut b = RabitqQuantizer::new(dim, 1234);
        a.train(&vecs, 0).unwrap();
        b.train(&vecs, 0).unwrap();
        for v in &vecs {
            let ea = a.encode(v).unwrap();
            let eb = b.encode(v).unwrap();
            assert_eq!(ea, eb, "RaBitQ codes must be bit-identical for same seed");
        }
    }

    #[test]
    fn different_seeds_produce_different_rotations() {
        let dim = 64;
        let vecs = random_unit_vectors(4, dim, 11);
        let mut a = RabitqQuantizer::new(dim, 1);
        let mut b = RabitqQuantizer::new(dim, 2);
        a.train(&vecs, 0).unwrap();
        b.train(&vecs, 0).unwrap();
        let ea = a.encode(&vecs[0]).unwrap();
        let eb = b.encode(&vecs[0]).unwrap();
        // Almost surely different (collision probability << 1e-6 for D=64).
        assert_ne!(ea, eb);
    }

    #[test]
    fn dim_mismatch_is_an_error() {
        let dim = 32;
        let mut q = RabitqQuantizer::new(dim, 0);
        let vecs = random_unit_vectors(2, dim, 0);
        q.train(&vecs, 0).unwrap();
        let bad = vec![0.0f32; dim + 1];
        assert!(q.encode(&bad).is_err());
        assert!(q.prepare_query(&bad).is_err());
    }
}
