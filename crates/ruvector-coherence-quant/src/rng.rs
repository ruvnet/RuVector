//! Deterministic PRNG for synthetic dataset generation.
//!
//! SplitMix64 (public-domain construction, Steele et al. 2014) — the same
//! generator `ruvector-turboquant` uses internally for its rotation seeds,
//! reimplemented here (it is `pub(crate)` there) so benchmark datasets stay
//! bit-reproducible across runs without pulling in the `rand` crate.

pub struct SplitMix64(pub u64);

impl SplitMix64 {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller standard normal sample.
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn gaussian_vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_gaussian() as f32).collect()
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_unit(&mut self) -> f32 {
        self.next_f64() as f32
    }
}
