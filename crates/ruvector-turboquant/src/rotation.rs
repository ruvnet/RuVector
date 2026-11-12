//! Deterministic randomized rotation for Turbo4 codes (ADR-296).
//!
//! Construction: `ROUNDS` rounds of
//!
//! ```text
//!   v ← blockFWHT( P · (s ⊙ v) )
//! ```
//!
//! where `s` is a ±1 sign diagonal, `P` a uniform permutation, and
//! `blockFWHT` applies the Fast Walsh-Hadamard Transform independently on the
//! power-of-two blocks of the binary decomposition of `D` (e.g. 1536 = 1024 +
//! 512). Each factor is exactly orthogonal, so norms are preserved to floating
//! point accuracy and the composition Gaussianizes coordinate marginals like a
//! Haar-uniform rotation (TurboQuant arXiv:2504.19874 §3.2) — without
//! zero-padding, so downstream code size stays `ceil(D/2)` bytes.
//!
//! Randomness comes from an in-crate SplitMix64 stream seeded by the caller.
//! No `rand` dependency: encoded bytes are a *persisted storage format*, so
//! the rotation must stay bit-identical across platforms, architectures, and
//! dependency upgrades forever.

/// Number of sign→permute→FWHT rounds. Three rounds is the standard
/// HD₁·HD₂·HD₃ recipe that reaches the near-Haar regime.
const ROUNDS: usize = 3;

/// SplitMix64 — tiny, seedable, platform-stable PRNG (public domain
/// construction, Steele et al. 2014). Used only at build time of a
/// [`Rotation`]; never on the encode hot path.
pub(crate) struct SplitMix64(pub u64);

impl SplitMix64 {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `0..bound` via Lemire-style rejection (bias-free).
    #[inline]
    fn next_below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        loop {
            let v = self.next_u64();
            // Rejection zone keeps the mapping exactly uniform.
            if v < u64::MAX - (u64::MAX % bound) {
                return v % bound;
            }
        }
    }
}

/// One round's parameters: sign bits (LSB-first packed), permutation, and the
/// (shared) block decomposition of `dim`.
struct Round {
    /// ±1 signs packed 64 per word; bit set ⇒ negate.
    sign_words: Vec<u64>,
    /// `perm[i]` = source index for output slot `i` (gather form).
    perm: Vec<u32>,
}

/// Deterministic randomized rotation. Build once per (dim, seed); apply many
/// times. `apply` is `O(D log D)` with no matrix stored.
pub struct Rotation {
    dim: usize,
    rounds: Vec<Round>,
    /// Power-of-two block sizes covering `dim` (descending), from the binary
    /// decomposition of `dim`. Blocks of size 1 are identity for the FWHT but
    /// still mix through the permutations.
    blocks: Vec<usize>,
}

impl Rotation {
    /// Build the rotation for `dim` dimensions from `seed`.
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim >= 2, "Turbo4 rotation requires dim >= 2, got {dim}");
        let mut rng = SplitMix64(seed ^ 0x5175_6472_616E_7434); // "QudranT4" domain sep
        let n_words = dim.div_ceil(64);

        let rounds = (0..ROUNDS)
            .map(|_| {
                let sign_words: Vec<u64> = (0..n_words).map(|_| rng.next_u64()).collect();
                // Fisher–Yates with the bias-free sampler.
                let mut perm: Vec<u32> = (0..dim as u32).collect();
                for i in (1..dim).rev() {
                    let j = rng.next_below(i as u64 + 1) as usize;
                    perm.swap(i, j);
                }
                Round { sign_words, perm }
            })
            .collect();

        // Binary decomposition: dim = Σ 2^k over set bits, descending.
        let mut blocks = Vec::new();
        let mut bit = usize::BITS - 1 - dim.leading_zeros();
        loop {
            if dim & (1 << bit) != 0 {
                blocks.push(1usize << bit);
            }
            if bit == 0 {
                break;
            }
            bit -= 1;
        }

        Self {
            dim,
            rounds,
            blocks,
        }
    }

    /// Dimensionality this rotation was built for.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Rotate `v` in place. `scratch` must be `dim` long (used for the
    /// permutation gather); contents are clobbered.
    pub fn apply_in_place(&self, v: &mut [f32], scratch: &mut [f32]) {
        assert_eq!(v.len(), self.dim);
        assert_eq!(scratch.len(), self.dim);
        for round in &self.rounds {
            // Signs.
            for (i, x) in v.iter_mut().enumerate() {
                if round.sign_words[i / 64] >> (i % 64) & 1 != 0 {
                    *x = -*x;
                }
            }
            // Permutation (gather into scratch, swap back).
            for (i, &src) in round.perm.iter().enumerate() {
                scratch[i] = v[src as usize];
            }
            v.copy_from_slice(scratch);
            // Blockwise FWHT with 1/sqrt(block) normalization (orthogonal).
            let mut off = 0;
            for &b in &self.blocks {
                fwht_normalized(&mut v[off..off + b]);
                off += b;
            }
        }
    }

    /// Rotate `v`, returning a new vector.
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        let mut out = v.to_vec();
        let mut scratch = vec![0.0f32; self.dim];
        self.apply_in_place(&mut out, &mut scratch);
        out
    }

    /// Inverse rotation (for tests/debugging; never needed on the search path).
    pub fn apply_inverse(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.dim);
        let mut out = v.to_vec();
        let mut scratch = vec![0.0f32; self.dim];
        for round in self.rounds.iter().rev() {
            // Inverse blockwise FWHT (self-inverse when normalized).
            let mut off = 0;
            for &b in &self.blocks {
                fwht_normalized(&mut out[off..off + b]);
                off += b;
            }
            // Inverse permutation (scatter).
            for (i, &src) in round.perm.iter().enumerate() {
                scratch[src as usize] = out[i];
            }
            out.copy_from_slice(&scratch);
            // Signs are self-inverse.
            for (i, x) in out.iter_mut().enumerate() {
                if round.sign_words[i / 64] >> (i % 64) & 1 != 0 {
                    *x = -*x;
                }
            }
        }
        out
    }
}

/// In-place Fast Walsh–Hadamard Transform, scaled by `1/sqrt(len)` so the
/// transform is orthogonal (and self-inverse). `len` must be a power of two;
/// `len == 1` is the identity.
fn fwht_normalized(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two());
    if n == 1 {
        return;
    }
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for x in v.iter_mut() {
        *x *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss_vec(dim: usize, seed: u64) -> Vec<f32> {
        // Box–Muller over SplitMix64 — deterministic test vectors, no rand dep.
        let mut rng = SplitMix64(seed);
        let mut out = Vec::with_capacity(dim);
        while out.len() < dim {
            let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let r = (-2.0 * u1.max(1e-12).ln()).sqrt();
            let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
            out.push((r * c) as f32);
            if out.len() < dim {
                out.push((r * s) as f32);
            }
        }
        out
    }

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn preserves_norm_pow2_and_non_pow2() {
        for dim in [64usize, 128, 96, 1536, 1000, 3] {
            let rot = Rotation::new(dim, 42);
            let v = gauss_vec(dim, 7);
            let r = rot.apply(&v);
            let (n0, n1) = (norm(&v), norm(&r));
            assert!(
                (n0 - n1).abs() < 1e-3 * n0.max(1.0),
                "dim {dim}: norm {n0} -> {n1}"
            );
        }
    }

    #[test]
    fn preserves_inner_products() {
        let dim = 96; // 64 + 32: exercises cross-block mixing
        let rot = Rotation::new(dim, 9);
        let a = gauss_vec(dim, 1);
        let b = gauss_vec(dim, 2);
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        let (ra, rb) = (rot.apply(&a), rot.apply(&b));
        assert!((dot(&a, &b) - dot(&ra, &rb)).abs() < 1e-2 * dim as f32);
    }

    #[test]
    fn inverse_roundtrips() {
        let dim = 200; // 128+64+8
        let rot = Rotation::new(dim, 5);
        let v = gauss_vec(dim, 3);
        let back = rot.apply_inverse(&rot.apply(&v));
        for (x, y) in v.iter().zip(&back) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn deterministic_across_builds() {
        let dim = 128;
        let (r1, r2) = (Rotation::new(dim, 42), Rotation::new(dim, 42));
        let v = gauss_vec(dim, 11);
        assert_eq!(r1.apply(&v), r2.apply(&v));
        // Different seed ⇒ different rotation.
        let r3 = Rotation::new(dim, 43);
        assert_ne!(r1.apply(&v), r3.apply(&v));
    }

    #[test]
    fn spreads_spike_across_coordinates() {
        // A one-hot vector must be spread out (max |coord| well below 1).
        let dim = 1536;
        let rot = Rotation::new(dim, 42);
        let mut v = vec![0.0f32; dim];
        v[17] = 1.0;
        let r = rot.apply(&v);
        let max = r.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(max < 0.25, "spike not spread: max coord {max}");
    }
}
