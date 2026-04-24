//! Random orthogonal rotation.
//!
//! Two flavours are supported:
//!
//! * `HaarDense` — Haar-uniform `D×D` matrix built via Gram–Schmidt on an
//!   i.i.d. Gaussian block. `apply` is `O(D²)`; storage is `4·D²` bytes. This
//!   is the default and stays bit-identical to previous snapshots.
//!
//! * `HadamardSigned` — randomised Hadamard rotation `D₁·H·D₂·H·D₃` where
//!   each `Dᵢ` is a ±1 diagonal and `H` is the Fast Walsh–Hadamard Transform.
//!   Cost is `O(D log D)` with no matrix stored (just `3·D` signs). TurboQuant
//!   (arXiv:2504.19874 §3.2) shows this hits the "close to Haar-uniform"
//!   regime RaBitQ needs for its Johnson–Lindenstrauss-style error bound.
//!
//! For arbitrary `dim` the Hadamard flavour zero-pads up to the next power of
//! two, runs the butterfly there, then truncates back to `dim` — standard
//! FWHT-on-non-dyadic trick.

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

/// Which random-rotation construction a `RandomRotation` is backed by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RandomRotationKind {
    /// Dense `D×D` Haar-uniform orthogonal matrix.
    HaarDense,
    /// Randomised Hadamard: three random ±1 diagonals interleaved with FWHT.
    HadamardSigned,
}

/// Internal storage mode. Kept private so we can evolve it without breaking
/// callers — users interact via `apply` / `apply_into` / `bytes` / `kind`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
enum Mode {
    /// Flattened row-major `D×D` matrix.
    HaarDense { matrix: Vec<f32> },
    /// Three ±1 sign vectors of length `padded_dim`, applied as `D₁·H·D₂·H·D₃`.
    HadamardSigned {
        signs: [Vec<f32>; 3],
        padded_dim: usize,
    },
}

/// A random (approximately) orthogonal rotation.
///
/// Build once, apply many times. The default constructor `random` yields a
/// Haar-uniform `D×D` matrix for backward compatibility; `hadamard` opts in
/// to the `O(D log D)` HD-HD-HD variant.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomRotation {
    mode: Mode,
    pub dim: usize,
    /// Kept for backward compatibility with snapshots that accessed the raw
    /// matrix. Populated only for `HaarDense`; empty for Hadamard.
    #[serde(default)]
    pub matrix: Vec<f32>,
}

impl RandomRotation {
    /// Sample a Haar-uniform orthogonal matrix of size `dim × dim`.
    ///
    /// Backward-compatible default: existing callers that expect a dense
    /// matrix under `self.matrix` keep working unchanged.
    pub fn random(dim: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        // Fill a dim×dim matrix with N(0,1) entries.
        let mut m: Vec<Vec<f32>> = (0..dim)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                            as f32
                    })
                    .collect()
            })
            .collect();

        // Gram–Schmidt orthonormalisation (in-place).
        for i in 0..dim {
            // Subtract projections of all previous basis vectors.
            for j in 0..i {
                let dot: f32 = (0..dim).map(|k| m[i][k] * m[j][k]).sum();
                for k in 0..dim {
                    let v = m[j][k];
                    m[i][k] -= dot * v;
                }
            }
            // Normalise.
            let norm: f32 = m[i].iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                m[i].iter_mut().for_each(|x| *x /= norm);
            }
        }

        let matrix: Vec<f32> = m.into_iter().flatten().collect();
        Self {
            mode: Mode::HaarDense {
                matrix: matrix.clone(),
            },
            dim,
            matrix,
        }
    }

    /// Construct a randomised Hadamard rotation `D₁·H·D₂·H·D₃`.
    ///
    /// Stores only `3 × padded_dim` ±1 entries — no matrix materialised.
    /// `padded_dim` is the next power of two `≥ dim`; for dyadic `dim` it
    /// equals `dim`.
    pub fn hadamard(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "RandomRotation::hadamard: dim must be > 0");
        let padded_dim = dim.next_power_of_two();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        // Three independent ±1 sign vectors.
        let make_signs = |rng: &mut rand::rngs::StdRng| -> Vec<f32> {
            (0..padded_dim)
                .map(|_| if rng.gen::<bool>() { 1.0_f32 } else { -1.0_f32 })
                .collect()
        };
        let signs = [
            make_signs(&mut rng),
            make_signs(&mut rng),
            make_signs(&mut rng),
        ];

        Self {
            mode: Mode::HadamardSigned { signs, padded_dim },
            dim,
            matrix: Vec::new(),
        }
    }

    /// Which construction backs this rotation.
    #[inline]
    pub fn kind(&self) -> RandomRotationKind {
        match &self.mode {
            Mode::HaarDense { .. } => RandomRotationKind::HaarDense,
            Mode::HadamardSigned { .. } => RandomRotationKind::HadamardSigned,
        }
    }

    /// Apply the rotation: out = P · v  (length must equal dim).
    #[inline]
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        debug_assert_eq!(v.len(), self.dim);
        let mut out = vec![0.0f32; self.dim];
        self.apply_into(v, &mut out);
        out
    }

    /// In-place variant that writes into a caller-provided buffer.
    /// Callers doing many rotations (hot query path) should reuse one
    /// `Vec<f32>` instead of allocating per call — saves one malloc
    /// per query in the ANN index's `encode_query_packed` path.
    #[inline]
    pub fn apply_into(&self, v: &[f32], out: &mut [f32]) {
        debug_assert_eq!(v.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        match &self.mode {
            Mode::HaarDense { matrix } => {
                let d = self.dim;
                for (i, out_i) in out.iter_mut().enumerate() {
                    let row = &matrix[i * d..(i + 1) * d];
                    *out_i = row.iter().zip(v.iter()).map(|(&r, &x)| r * x).sum();
                }
            }
            Mode::HadamardSigned { signs, padded_dim } => {
                // Scratch buffer at padded size — zero-pad the tail.
                let mut buf = vec![0.0_f32; *padded_dim];
                buf[..self.dim].copy_from_slice(v);
                // D₃
                for (b, s) in buf.iter_mut().zip(signs[2].iter()) {
                    *b *= *s;
                }
                fwht_inplace(&mut buf);
                // D₂
                for (b, s) in buf.iter_mut().zip(signs[1].iter()) {
                    *b *= *s;
                }
                fwht_inplace(&mut buf);
                // D₁
                for (b, s) in buf.iter_mut().zip(signs[0].iter()) {
                    *b *= *s;
                }
                // Normalise: two FWHT passes multiply the norm by `padded_dim`
                // (each H is orthogonal only after dividing by √padded_dim),
                // so the combined scale factor is 1 / padded_dim.
                let scale = 1.0_f32 / (*padded_dim as f32);
                for (o, b) in out.iter_mut().zip(buf.iter().take(self.dim)) {
                    *o = b * scale;
                }
            }
        }
    }

    /// Memory usage in bytes of the rotation's internal storage.
    pub fn bytes(&self) -> usize {
        match &self.mode {
            Mode::HaarDense { matrix } => matrix.len() * 4,
            Mode::HadamardSigned { signs, .. } => signs.iter().map(|s| s.len() * 4).sum::<usize>(),
        }
    }
}

/// Fast in-place L2 normalisation.
pub fn normalize_inplace(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// In-place Fast Walsh–Hadamard Transform (unnormalised, natural order).
///
/// Requires `buf.len()` to be a power of two. Runs the iterative butterfly:
/// at stage `h`, pairs of elements `(buf[i+j], buf[i+j+h])` are replaced by
/// their sum and difference. After completion, `buf` holds `H · buf_in`
/// where `H` is the unnormalised Hadamard matrix with `H Hᵀ = N · I`.
#[inline]
fn fwht_inplace(buf: &mut [f32]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-two length");
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..(i + h) {
                let x = buf[j];
                let y = buf[j + h];
                buf[j] = x + y;
                buf[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand_distr::StandardNormal;

    /// Full orthogonality check — every pair of rows must be orthonormal.
    /// Stricter than the shipped version at `f2dbb6efb` which only tested
    /// (row 0, row 1).
    #[test]
    fn orthogonality_all_pairs_d64() {
        check_orthonormal(64, 42, 1e-4);
    }

    #[test]
    fn orthogonality_all_pairs_d128() {
        check_orthonormal(128, 7, 1e-4);
    }

    /// At D=256 classical Gram-Schmidt accumulates enough f32 round-off
    /// that we widen the tolerance to 1e-3 — still tight enough for the
    /// estimator not to drift but surfaces that GS is not numerically
    /// stable at large D. Reminder to move to Householder / modified GS
    /// when we start shipping D ≥ 1024.
    #[test]
    fn orthogonality_all_pairs_d256() {
        check_orthonormal(256, 11, 1e-3);
    }

    fn check_orthonormal(dim: usize, seed: u64, tol: f32) {
        let rot = RandomRotation::random(dim, seed);
        let d = rot.dim;
        for i in 0..d {
            let ri = &rot.matrix[i * d..(i + 1) * d];
            // Unit norm.
            let ni: f32 = ri.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((ni - 1.0).abs() < tol, "row {i} norm = {ni}, D={d}");
            // Orthogonal to all later rows.
            for j in (i + 1)..d {
                let rj = &rot.matrix[j * d..(j + 1) * d];
                let dot: f32 = ri.iter().zip(rj.iter()).map(|(&a, &b)| a * b).sum();
                assert!(dot.abs() < tol, "rows {i},{j} dot={dot}, D={d}");
            }
        }
    }

    #[test]
    fn apply_preserves_norm() {
        let rot = RandomRotation::random(128, 7);
        let v: Vec<f32> = (0..128_u32).map(|i| (i as f32).sin()).collect();
        let rv = rot.apply(&v);
        let norm_in: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_out: f32 = rv.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm_in - norm_out).abs() / norm_in < 1e-3);
    }

    /// Determinism: same seed + same dim → bit-identical rotation matrix.
    #[test]
    fn seed_reproducibility() {
        let a = RandomRotation::random(64, 1234);
        let b = RandomRotation::random(64, 1234);
        assert_eq!(a.matrix, b.matrix);
    }

    // ----- Randomised Hadamard (HD-HD-HD) tests --------------------------------

    /// Sample random unit vectors via StdRng + StandardNormal (seeded → reproducible).
    fn random_unit_vecs(dim: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| {
                        <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                            as f32
                    })
                    .collect();
                normalize_inplace(&mut v);
                v
            })
            .collect()
    }

    fn hadamard_norm_check(dim: usize, seed: u64) {
        let rot = RandomRotation::hadamard(dim, seed);
        assert_eq!(rot.kind(), RandomRotationKind::HadamardSigned);
        let vecs = random_unit_vecs(dim, 100, seed ^ 0xDEAD_BEEF);
        for v in &vecs {
            let rv = rot.apply(v);
            let n: f32 = rv.iter().map(|&x| x * x).sum::<f32>().sqrt();
            // Isotropy is approximate (truncation + padding break exact
            // orthogonality) — loose ±5 % band keeps RaBitQ estimator safe.
            assert!(
                (0.95..=1.05).contains(&n),
                "D={dim}: rotated unit vector has norm {n}",
            );
        }
    }

    /// D=128 and D=256 are powers of two — no padding path.
    #[test]
    fn hadamard_apply_preserves_norm_power_of_two() {
        hadamard_norm_check(128, 7);
        hadamard_norm_check(256, 11);
    }

    /// D=1000 exercises the zero-pad-to-1024 branch plus the truncation
    /// back to `dim`. Looser isotropy is expected and allowed by the ±5 %
    /// tolerance.
    #[test]
    fn hadamard_apply_preserves_norm_non_power_of_two() {
        hadamard_norm_check(1000, 3);
    }

    /// Same seed → bit-identical output for both sign vectors (via apply).
    #[test]
    fn hadamard_is_deterministic() {
        let a = RandomRotation::hadamard(128, 0xC0FFEE);
        let b = RandomRotation::hadamard(128, 0xC0FFEE);
        let v: Vec<f32> = (0..128_u32).map(|i| (i as f32).cos()).collect();
        assert_eq!(a.apply(&v), b.apply(&v));
        // Different seed must change the output.
        let c = RandomRotation::hadamard(128, 0xC0FFEE + 1);
        assert_ne!(a.apply(&v), c.apply(&v));
    }

    /// Correctness smoke: for a dyadic dim, the all-ones input after the
    /// first FWHT collapses to `(N, 0, 0, …)` — a cheap way to verify the
    /// butterfly without timing.
    #[test]
    fn hadamard_is_fast() {
        // FWHT of `[1; 8]` must be `[8, 0, 0, 0, 0, 0, 0, 0]`.
        let mut buf = vec![1.0_f32; 8];
        fwht_inplace(&mut buf);
        assert!((buf[0] - 8.0).abs() < 1e-6);
        for v in &buf[1..] {
            assert!(v.abs() < 1e-6);
        }

        // Storage footprint: Hadamard must be dramatically smaller than Haar
        // at non-trivial dim (3·D floats vs D² floats).
        let had = RandomRotation::hadamard(128, 1);
        let haar = RandomRotation::random(128, 1);
        assert!(had.bytes() < haar.bytes() / 10);
        assert_eq!(had.kind(), RandomRotationKind::HadamardSigned);
        assert_eq!(haar.kind(), RandomRotationKind::HaarDense);
    }
}
