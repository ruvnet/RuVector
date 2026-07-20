//! A tiny deterministic PRNG (SplitMix64) so the crate stays dependency-free
//! and every benchmark / anneal run is byte-for-byte reproducible.

/// SplitMix64 generator. Not cryptographic; for reproducible sampling only.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)`.
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform `f32` in `[-1, 1)`.
    #[inline]
    pub fn signed(&mut self) -> f32 {
        self.f32() * 2.0 - 1.0
    }

    /// Uniform index in `[0, n)`. Returns `0` when `n == 0`.
    #[inline]
    pub fn range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        // Two generators with the same seed produce identical streams.
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            let x = a.f32();
            assert_eq!(x, b.f32());
            assert!((0.0..1.0).contains(&x));
            assert_eq!(a.range(7), b.range(7));
        }
        // range stays in bounds for a fresh stream.
        let mut c = Rng::new(99);
        for _ in 0..1000 {
            assert!(c.range(7) < 7);
        }
        assert_eq!(Rng::new(0).range(0), 0);
    }

    #[test]
    fn roughly_uniform() {
        let mut r = Rng::new(1);
        let mut buckets = [0usize; 4];
        for _ in 0..40_000 {
            buckets[r.range(4)] += 1;
        }
        for b in buckets {
            assert!((8_000..12_000).contains(&b), "bucket skew: {b}");
        }
    }
}
