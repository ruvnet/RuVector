//! Deterministic xorshift64* PRNG. Not cryptographically secure — used
//! solely for reproducible synthetic benchmark data generation, matching
//! the convention set by `ruvector-retrieval-receipt`'s benchmark harness.

pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Standard normal sample via Box-Muller.
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Uniform integer in [0, bound).
    pub fn next_below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_given_same_seed() {
        let mut a = Xorshift64::new(42);
        let mut b = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Xorshift64::new(1);
        let mut b = Xorshift64::new(2);
        let seq_a: Vec<u64> = (0..10).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..10).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn gaussian_is_roughly_standard_normal() {
        let mut r = Xorshift64::new(7);
        let n = 20_000;
        let samples: Vec<f32> = (0..n).map(|_| r.next_gaussian()).collect();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let var: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean={mean}");
        assert!((var - 1.0).abs() < 0.1, "var={var}");
    }
}
