//! Caller-supplied randomness for HNSW level selection.
//!
//! No `getrandom`, no clock: ADR §6.1 derives randomness as
//! `splitmix64(seq ^ collection_salt)`, so the store can stay stateless and
//! a replay of the op log rebuilds a bit-identical graph.

/// Source of uniformly distributed `u64`s for level selection.
pub trait LevelRng {
    /// Next uniformly distributed value.
    fn next_u64(&mut self) -> u64;
}

/// SplitMix64 generator (Steele, Lea, Flood 2014).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

impl SplitMix64 {
    /// Generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Stateless finaliser: `mix(seq ^ salt)` is the ADR §6.1 per-op value.
    pub fn mix(x: u64) -> u64 {
        let mut z = x.wrapping_add(GOLDEN);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl LevelRng for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        let out = Self::mix(self.state);
        self.state = self.state.wrapping_add(GOLDEN);
        out
    }
}

/// HNSW level for a uniform `x`: geometric with `P(level >= l) = m^-l`
/// (the `1/ln m` multiplier of the paper), capped at `max_level`. `m < 2`
/// always yields level 0.
///
/// Pure integer arithmetic (count base-`m` trailing zero digits of `x`):
/// no `ln`, so native and wasm32 libm differences can never change a level
/// and an off-platform rebuild replays to the identical graph.
pub fn level_from_u64(x: u64, m: usize, max_level: u8) -> u8 {
    if m < 2 {
        return 0;
    }
    let m = m as u64;
    let (mut x, mut level) = (x, 0u8);
    while level < max_level && x % m == 0 {
        x /= m;
        level += 1;
    }
    level
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_geometric() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
        let mut r = SplitMix64::new(1);
        let n = 200_000;
        let mut hist = [0u32; 8];
        for _ in 0..n {
            let l = level_from_u64(r.next_u64(), 16, 7);
            hist[l as usize] += 1;
        }
        // P(level >= 1) = 1/16.
        let upper = n - hist[0];
        let p = f64::from(upper) / f64::from(n);
        assert!((p - 1.0 / 16.0).abs() < 0.005, "p = {p}");
        let two = n - hist[0] - hist[1];
        let p2 = f64::from(two) / f64::from(n);
        assert!((p2 - 1.0 / 256.0).abs() < 0.001, "p2 = {p2}");
        assert_eq!(level_from_u64(0, 16, 5), 5);
        assert_eq!(level_from_u64(u64::MAX, 16, 5), 0);
        assert_eq!(level_from_u64(0, 1, 5), 0);
    }
}
