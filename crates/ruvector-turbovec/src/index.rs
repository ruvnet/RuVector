//! `TurboVecIndex` — the multi-bit TurboQuant flat ANN index (ADR-254).
//!
//! Pipeline per database vector `x`:
//!
//! 1. `norm = ‖x‖`, `û = x / norm`         — strip & store length (§T1)
//! 2. `r = P · û`  (randomized Hadamard)    — reuse `ruvector_rabitq` (§T1)
//! 3. `z_i = (r_i − shift_i)/scale_i`        — TQ+ calibration (§T3)
//! 4. `q_i = argmin |z_i − centroid|`        — Lloyd–Max 2/3/4-bit SQ (§T2)
//! 5. `c_x = ⟨r, r̂⟩ / ⟨r̂, r̂⟩`              — per-vector bias correction (§T4)
//!
//! Scoring a query `q` (orthogonal `P` preserves inner products, so
//! `⟨r,s⟩ = ⟨û,q̂⟩ = cos`):
//!
//! ```text
//!   s = P · q̂                                  (unit, rotated once / query)
//!   cos_est = c_x · Σ_i r̂_i · s_i              (empirical bias correction)
//!   ⟨q,x⟩_est = ‖q‖ · ‖x‖ · cos_est
//!   score = ‖q‖² + ‖x‖² − 2⟨q,x⟩_est ≈ ‖q − x‖²
//! ```
//!
//! Smaller `score` = nearer, matching `ruvector_rabitq::AnnIndex` ordering, so
//! recall is directly comparable to exact brute-force L2.

use crate::calibrate::Calibration;
use crate::error::{Result, TurboVecError};
use crate::quantize::{pack, quantize_coord, unpack, BitWidth};
use ruvector_rabitq::rotation::normalize_inplace;
use ruvector_rabitq::{AnnIndex, RandomRotation, SearchResult};

/// Default number of vectors collected before calibration is frozen.
pub const DEFAULT_WARMUP: usize = 256;

/// A flat multi-bit TurboQuant index over `dim`-dimensional vectors.
pub struct TurboVecIndex {
    /// Public input dimensionality.
    dim: usize,
    /// Full power-of-two Hadamard space. Keeping every rotated coordinate is
    /// required for exact norm and inner-product preservation.
    rotated_dim: usize,
    bw: BitWidth,
    rotation: RandomRotation,
    warmup: usize,

    /// Raw vectors awaiting calibration. Empty once finalized (memory freed).
    staging: Vec<(usize, Vec<f32>)>,
    calibration: Option<Calibration>,

    // ── SoA compressed storage (post-finalize) ───────────────────────────
    ids: Vec<usize>,
    codes: Vec<u8>, // flattened: code_bytes(dim) per vector
    norms: Vec<f32>,
    corr: Vec<f32>, // per-vector c_x
}

impl TurboVecIndex {
    /// Build an empty index. `seed` makes the Hadamard rotation reproducible.
    pub fn with_seed(dim: usize, bw: BitWidth, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TurboVecError::ZeroDim);
        }
        let rotated_dim = dim.next_power_of_two();
        Ok(Self {
            dim,
            rotated_dim,
            bw,
            rotation: RandomRotation::hadamard(rotated_dim, seed),
            warmup: DEFAULT_WARMUP,
            staging: Vec::new(),
            calibration: None,
            ids: Vec::new(),
            codes: Vec::new(),
            norms: Vec::new(),
            corr: Vec::new(),
        })
    }

    /// Build with the default seed (42).
    pub fn new(dim: usize, bw: BitWidth) -> Result<Self> {
        Self::with_seed(dim, bw, 42)
    }

    /// Override the warm-up count (vectors gathered before calibration freezes).
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup.max(1);
        self
    }

    /// Whether calibration has been frozen and staged vectors encoded.
    #[inline]
    pub fn is_finalized(&self) -> bool {
        self.calibration.is_some()
    }

    /// Number of indexed vectors (staged + encoded).
    #[inline]
    pub fn count(&self) -> usize {
        self.ids.len() + self.staging.len()
    }

    /// Bytes per encoded vector: packed code + norm(f32) + corr(f32) + id(usize).
    #[inline]
    pub fn bytes_per_vector(&self) -> usize {
        self.bw.code_bytes(self.rotated_dim) + 4 + 4 + std::mem::size_of::<usize>()
    }

    /// Compression ratio of encoded storage vs raw `f32` (codes+scalars only,
    /// rotation/calibration are amortized shared overhead).
    pub fn compression_ratio(&self) -> f32 {
        let raw = (self.dim * 4) as f32;
        let comp = (self.bw.code_bytes(self.rotated_dim) + 8) as f32; // code + norm + corr
        raw / comp
    }

    /// Rotate a normalized input in the full padded Hadamard space.
    ///
    /// Truncating a padded Hadamard output is not orthogonal when the input
    /// dimension is far from a power of two (notably 1536 → 2048). Retaining
    /// all padded coordinates preserves L2 geometry exactly.
    fn rotate_unit(&self, unit: &[f32]) -> Vec<f32> {
        debug_assert_eq!(unit.len(), self.dim);
        let mut padded = vec![0.0; self.rotated_dim];
        padded[..self.dim].copy_from_slice(unit);
        self.rotation.apply(&padded)
    }

    /// Freeze calibration on the staged batch and encode everything. Idempotent.
    pub fn finalize(&mut self) {
        if self.is_finalized() {
            return;
        }
        // Rotate staged unit vectors to fit calibration. Zero vectors are not
        // unit-normalizable (they would contribute all-zero rows and bias
        // shift/scale toward zero), so they are excluded here — `encode_and_store`
        // already special-cases them at encode time.
        let rotated: Vec<Vec<f32>> = self
            .staging
            .iter()
            .filter(|(_, v)| v.iter().map(|x| x * x).sum::<f32>() > 1e-20)
            .map(|(_, v)| {
                let mut u = v.clone();
                normalize_inplace(&mut u);
                self.rotate_unit(&u)
            })
            .collect();
        let cal = Calibration::fit(&rotated, self.rotated_dim);
        // Encode each staged vector using the frozen calibration.
        let staged = std::mem::take(&mut self.staging);
        self.calibration = Some(cal);
        self.ids.reserve(staged.len());
        for (id, v) in staged {
            self.encode_and_store(id, &v);
        }
        // staging is now empty → raw f32 memory released.
    }

    /// Encode one vector with the (already frozen) calibration and append it.
    fn encode_and_store(&mut self, id: usize, v: &[f32]) {
        let cal = self.calibration.as_ref().expect("finalized");
        let centroids = self.bw.centroids();

        let mut u = v.to_vec();
        let norm = {
            let n: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            n
        };
        if norm < 1e-10 {
            // Degenerate zero vector: store an all-zero code, neutral scalars.
            self.ids.push(id);
            self.codes.extend(std::iter::repeat_n(
                0u8,
                self.bw.code_bytes(self.rotated_dim),
            ));
            self.norms.push(0.0);
            self.corr.push(0.0);
            return;
        }
        for x in u.iter_mut() {
            *x /= norm;
        }
        let r = self.rotate_unit(&u);

        // Quantize + reconstruct, accumulating ⟨r,r̂⟩ and ⟨r̂,r̂⟩ for c_x.
        let mut q_codes = vec![0u8; self.rotated_dim];
        let mut dot_r_rhat = 0.0f32;
        let mut dot_rhat = 0.0f32;
        for (i, (&ri, slot)) in r.iter().zip(q_codes.iter_mut()).enumerate() {
            let z = cal.calibrate(i, ri);
            let code = quantize_coord(centroids, z);
            *slot = code;
            let rhat = cal.reconstruct(i, centroids[code as usize]);
            dot_r_rhat += ri * rhat;
            dot_rhat += rhat * rhat;
        }
        let c_x = if dot_rhat > 1e-20 {
            dot_r_rhat / dot_rhat
        } else {
            0.0
        };

        self.ids.push(id);
        self.codes.extend_from_slice(&pack(&q_codes, self.bw));
        self.norms.push(norm);
        self.corr.push(c_x);
    }

    /// The encoded code-slice for stored vector `pos`.
    #[inline]
    fn code_slice(&self, pos: usize) -> &[u8] {
        let cb = self.bw.code_bytes(self.rotated_dim);
        &self.codes[pos * cb..(pos + 1) * cb]
    }

    /// Estimated `‖q − x_pos‖²` given the rotated unit query `s` and `‖q‖`.
    /// `centroids` and `cal` are hoisted by the caller for the hot loop.
    #[inline]
    fn estimate_l2(&self, pos: usize, s: &[f32], nq: f32, cal: &Calibration) -> f32 {
        let centroids = self.bw.centroids();
        let codes = unpack(self.code_slice(pos), self.rotated_dim, self.bw);
        let mut ip_unit = 0.0f32;
        for (i, (&code, &si)) in codes.iter().zip(s.iter()).enumerate() {
            let rhat = cal.reconstruct(i, centroids[code as usize]);
            ip_unit += rhat * si;
        }
        let cos_est = self.corr[pos] * ip_unit;
        let ip_est = nq * self.norms[pos] * cos_est;
        nq * nq + self.norms[pos] * self.norms[pos] - 2.0 * ip_est
    }

    /// Estimated cosine similarity between `query` and stored vector `pos`.
    /// Exposed for bias measurement in tests and the demo.
    pub fn estimate_cosine(&self, query: &[f32], pos: usize) -> Result<f32> {
        if query.len() != self.dim {
            return Err(TurboVecError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if !query.iter().all(|value| value.is_finite()) {
            return Err(TurboVecError::NonFiniteQuery);
        }
        let cal = self
            .calibration
            .as_ref()
            .ok_or(TurboVecError::NotFinalized)?;
        if pos >= self.ids.len() {
            return Err(TurboVecError::PositionOutOfBounds {
                pos,
                len: self.ids.len(),
            });
        }
        let mut qn = query.to_vec();
        normalize_inplace(&mut qn);
        let s = self.rotate_unit(&qn);
        let centroids = self.bw.centroids();
        let codes = unpack(self.code_slice(pos), self.rotated_dim, self.bw);
        let mut ip_unit = 0.0f32;
        for (i, (&code, &si)) in codes.iter().zip(s.iter()).enumerate() {
            ip_unit += cal.reconstruct(i, centroids[code as usize]) * si;
        }
        Ok(self.corr[pos] * ip_unit)
    }

    /// Shared search core (used by `AnnIndex::search`). When the index has not
    /// been finalized (tiny indexes below warm-up that never called
    /// `finalize`), falls back to an exact brute-force L2 over the staged raw
    /// vectors — correct, just uncompressed.
    fn search_core(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(TurboVecError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if !query.iter().all(|value| value.is_finite()) {
            return Err(TurboVecError::NonFiniteQuery);
        }
        if self.count() == 0 {
            return Err(TurboVecError::EmptyIndex);
        }
        let mut scored: Vec<SearchResult> = if let Some(cal) = self.calibration.as_ref() {
            let mut qn = query.to_vec();
            let nq: f32 = qn.iter().map(|x| x * x).sum::<f32>().sqrt();
            normalize_inplace(&mut qn);
            let s = self.rotate_unit(&qn);
            (0..self.ids.len())
                .map(|pos| SearchResult {
                    id: self.ids[pos],
                    score: self.estimate_l2(pos, &s, nq, cal),
                })
                .collect()
        } else {
            // Exact fallback for not-yet-finalized tiny indexes.
            self.staging
                .iter()
                .map(|(id, v)| SearchResult {
                    id: *id,
                    score: v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum(),
                })
                .collect()
        };
        let compare = |a: &SearchResult, b: &SearchResult| {
            a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id))
        };
        if k < scored.len() {
            scored.select_nth_unstable_by(k, compare);
            scored.truncate(k);
        }
        scored.sort_unstable_by(compare);
        Ok(scored)
    }

    /// Exposed total memory accounting (used by `AnnIndex::memory_bytes`).
    fn memory(&self) -> usize {
        let cal = self.calibration.as_ref().map(|c| c.bytes()).unwrap_or(0);
        let staged: usize = self
            .staging
            .iter()
            .map(|(_, v)| v.len() * 4 + std::mem::size_of::<usize>())
            .sum();
        self.rotation.bytes()
            + cal
            + self.codes.len()
            + self.norms.len() * 4
            + self.corr.len() * 4
            + self.ids.len() * std::mem::size_of::<usize>()
            + staged
    }
}

impl AnnIndex for TurboVecIndex {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> ruvector_rabitq::error::Result<()> {
        // Surface the same `DimensionMismatch` other `AnnIndex` impls do, in
        // release builds too — never silently accept a wrong-length vector.
        if vector.len() != self.dim {
            return Err(ruvector_rabitq::error::RabitqError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        if !vector.iter().all(|value| value.is_finite()) {
            return Err(ruvector_rabitq::error::RabitqError::InvalidParameter(
                "vector contains a non-finite value".to_string(),
            ));
        }
        if self.is_finalized() {
            self.encode_and_store(id, &vector);
        } else {
            self.staging.push((id, vector));
            if self.staging.len() >= self.warmup {
                self.finalize();
            }
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> ruvector_rabitq::error::Result<Vec<SearchResult>> {
        // Propagate dimension mismatches rather than masking them as an empty
        // result (which would hide caller bugs).
        self.search_core(query, k).map_err(|error| match error {
            TurboVecError::DimMismatch { expected, got } => {
                ruvector_rabitq::error::RabitqError::DimensionMismatch {
                    expected,
                    actual: got,
                }
            }
            TurboVecError::EmptyIndex => ruvector_rabitq::error::RabitqError::EmptyIndex,
            other => ruvector_rabitq::error::RabitqError::InvalidParameter(other.to_string()),
        })
    }

    fn len(&self) -> usize {
        self.count()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn memory_bytes(&self) -> usize {
        self.memory()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    fn brute_topk(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
        let mut s: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum()))
            .collect();
        s.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        s.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn determinism_same_seed_same_codes() {
        let data = random_vectors(300, 64, 1);
        let build = |seed| {
            let mut ix = TurboVecIndex::with_seed(64, BitWidth::Four, seed).unwrap();
            for (i, v) in data.iter().enumerate() {
                ix.add(i, v.clone()).unwrap();
            }
            ix.finalize();
            ix.codes.clone()
        };
        assert_eq!(build(7), build(7), "same seed must give identical codes");
        assert_ne!(build(7), build(8), "different seed should differ");
    }

    #[test]
    fn recall_beats_threshold_4bit() {
        let dim = 128;
        let data = random_vectors(2000, dim, 10);
        let mut ix = TurboVecIndex::with_seed(dim, BitWidth::Four, 42).unwrap();
        for (i, v) in data.iter().enumerate() {
            ix.add(i, v.clone()).unwrap();
        }
        ix.finalize();
        assert!(ix.is_finalized());

        let queries = random_vectors(100, dim, 99);
        let k = 10;
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth = brute_topk(&data, q, k);
            let got: Vec<usize> = ix.search(q, k).unwrap().into_iter().map(|r| r.id).collect();
            for g in &got {
                if truth.contains(g) {
                    hits += 1;
                }
            }
            total += k;
        }
        let recall = hits as f32 / total as f32;
        // 4-bit scalar quant on 128-d uniform data: expect comfortably > 0.5.
        assert!(recall > 0.5, "recall@10 too low: {recall:.3}");
    }

    #[test]
    fn estimator_is_approximately_unbiased() {
        // Mean signed cosine error over many (query, db) pairs should be ~0,
        // which is the whole point of the c_x length-renormalization (§T4).
        let dim = 256;
        let data = random_vectors(500, dim, 3);
        let mut ix = TurboVecIndex::with_seed(dim, BitWidth::Four, 5).unwrap();
        for (i, v) in data.iter().enumerate() {
            ix.add(i, v.clone()).unwrap();
        }
        ix.finalize();

        let queries = random_vectors(50, dim, 77);
        let mut sum_err = 0.0f64;
        let mut n = 0u64;
        for q in &queries {
            let nq: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            for (pos, x) in data.iter().enumerate().take(200) {
                let nx: f32 = x.iter().map(|t| t * t).sum::<f32>().sqrt();
                let true_cos = q.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() / (nq * nx);
                let est_cos = ix.estimate_cosine(q, pos).unwrap();
                sum_err += (est_cos - true_cos) as f64;
                n += 1;
            }
        }
        let mean_err = (sum_err / n as f64).abs();
        assert!(
            mean_err < 0.02,
            "estimator biased: mean |err| = {mean_err:.4}"
        );
    }

    #[test]
    fn add_and_search_reject_wrong_dim() {
        let mut ix = TurboVecIndex::new(16, BitWidth::Four).unwrap();
        // add with wrong-length vector must error, not silently accept.
        assert!(ix.add(0, vec![0.0; 8]).is_err());
        ix.add(0, vec![0.1; 16]).unwrap();
        ix.finalize();
        // search with wrong-length query must error, not return empty.
        assert!(ix.search(&[0.0; 8], 1).is_err());
        assert!(ix.search(&[0.0; 16], 1).is_ok());
    }

    #[test]
    fn rejects_non_finite_vectors_and_queries() {
        let mut ix = TurboVecIndex::new(16, BitWidth::Four).unwrap();
        let mut bad = vec![0.0; 16];
        bad[3] = f32::NAN;
        assert!(ix.add(0, bad).is_err());
        ix.add(0, vec![0.1; 16]).unwrap();
        ix.finalize();

        let mut bad_query = vec![0.0; 16];
        bad_query[5] = f32::INFINITY;
        assert!(ix.search(&bad_query, 1).is_err());
        assert!(matches!(
            ix.estimate_cosine(&bad_query, 0),
            Err(TurboVecError::NonFiniteQuery)
        ));
    }

    #[test]
    fn padded_rotation_preserves_geometry_at_1536_dimensions() {
        let dim = 1536;
        let ix = TurboVecIndex::with_seed(dim, BitWidth::Four, 17).unwrap();
        assert_eq!(ix.rotated_dim, 2048);

        let mut a: Vec<f32> = (0..dim).map(|i| ((i * 17) as f32).sin()).collect();
        let mut b: Vec<f32> = (0..dim).map(|i| ((i * 29) as f32).cos()).collect();
        normalize_inplace(&mut a);
        normalize_inplace(&mut b);
        let dot_before: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let ra = ix.rotate_unit(&a);
        let rb = ix.rotate_unit(&b);
        let dot_after: f32 = ra.iter().zip(&rb).map(|(x, y)| x * y).sum();
        let norm_a: f32 = ra.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = rb.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!((norm_a - 1.0).abs() < 1e-4);
        assert!((norm_b - 1.0).abs() < 1e-4);
        assert!((dot_before - dot_after).abs() < 1e-4);
    }

    #[test]
    fn empty_index_returns_empty_index_error() {
        let ix = TurboVecIndex::new(8, BitWidth::Two).unwrap();
        assert!(matches!(
            ix.search(&[0.0; 8], 1),
            Err(ruvector_rabitq::error::RabitqError::EmptyIndex)
        ));
    }

    #[test]
    fn finalize_excludes_zero_vectors_from_calibration() {
        // A pile of zero vectors mixed into the warm-up must NOT corrupt the
        // calibration fitted from the real vectors. If zeros leaked in they
        // would collapse `scale` toward zero and wreck reconstruction; we prove
        // calibration stays faithful via self-retrieval of a stored vector.
        let dim = 64;
        let real = random_vectors(50, dim, 2);
        let mut ix = TurboVecIndex::with_seed(dim, BitWidth::Four, 1).unwrap();
        for _ in 0..50 {
            ix.add(0, vec![0.0; dim]).unwrap();
        }
        for (i, v) in real.iter().enumerate() {
            ix.add(i + 1, v.clone()).unwrap();
        }
        ix.finalize();
        // Querying a stored real vector must return that same vector as nearest
        // (distance to self ≈ quantization error ≪ distance to any other point).
        let hits = ix.search(&real[7], 1).unwrap();
        assert_eq!(
            hits[0].id, 8,
            "self-retrieval failed → calibration corrupted"
        );
    }

    #[test]
    fn tiny_index_uses_exact_fallback() {
        let dim = 32;
        let data = random_vectors(5, dim, 2); // below warm-up, never finalized
        let mut ix = TurboVecIndex::new(dim, BitWidth::Two).unwrap();
        for (i, v) in data.iter().enumerate() {
            ix.add(i, v.clone()).unwrap();
        }
        assert!(!ix.is_finalized());
        let truth = brute_topk(&data, &data[0], 1)[0];
        let got = ix.search(&data[0], 1).unwrap()[0].id;
        assert_eq!(got, truth, "exact fallback must be perfect");
    }
}
