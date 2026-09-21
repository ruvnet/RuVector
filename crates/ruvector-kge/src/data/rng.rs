//! Deterministic PRNG shared by splits, negative sampling and tie-breaking
//! (no `rand`, no `RandomState`): a splitmix64-seeded xorshift64 generator.

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A small, deterministic xorshift64 generator seeded through splitmix64.
#[derive(Debug, Clone)]
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn seeded(seed: u64) -> Self {
        // splitmix64 can return 0; xorshift64 must never be seeded with 0.
        Self {
            state: splitmix64(seed) | 1,
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform integer in `[0, n)` with rejection sampling (no modulo bias).
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        if n <= 1 {
            return 0;
        }
        let threshold = 0u64.wrapping_sub(n) % n; // 2^64 mod n
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return x % n;
            }
        }
    }

    /// Uniform integer in `[0, m]` inclusive.
    pub(crate) fn range_inclusive(&mut self, m: u64) -> u64 {
        self.below(m.saturating_add(1))
    }
}
