//! Tiny deterministic RNG (SplitMix64) for reproducible noise and baselines.
//!
//! We do not use the `rand` crate in the core so that noise generation is
//! fully platform-independent and bit-reproducible: this is what guarantees
//! the determinism invariant (ADR-260 §21) for noisy detector models.

use core::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    #[inline]
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point.
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits -> mantissa.
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Standard normal sample via Box–Muller.
    #[inline]
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = (self.next_f32()).max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_stream() {
        let mut a = DeterministicRng::new(42);
        let mut b = DeterministicRng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_in_range() {
        let mut r = DeterministicRng::new(7);
        for _ in 0..10_000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn gaussian_mean_near_zero() {
        let mut r = DeterministicRng::new(99);
        let n = 50_000;
        let mean: f32 = (0..n).map(|_| r.next_gaussian()).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean was {mean}");
    }
}
