/// Deterministic synthetic dataset generation (no external dependencies).
///
/// Uses a simple LCG PRNG seeded with a fixed value so benchmark results are
/// reproducible across runs.
use crate::{CapMask, VecEntry};

/// Minimal LCG: period 2^64, reproducible across platforms.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x6c62272e07bb0142)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Uniform [0, 1)
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    #[inline]
    pub fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Uniform [0, n)
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Configuration for dataset generation.
pub struct DatasetConfig {
    pub n_vectors: usize,
    pub dims: usize,
    /// Number of distinct capability bits (max 64).
    pub n_caps: u8,
    /// Each vector requires exactly this many capability bits.
    pub required_per_vector: u8,
    pub seed: u64,
}

/// Build a synthetic dataset with random vectors and per-vector capability requirements.
///
/// Vectors are drawn from a standard Normal-like distribution (Box-Muller
/// approximated via LCG sums) to roughly match real embedding distributions.
pub fn generate(cfg: &DatasetConfig) -> Vec<VecEntry> {
    assert!(cfg.n_caps <= 64);
    assert!(cfg.required_per_vector as usize <= cfg.n_caps as usize);

    let mut rng = Lcg::new(cfg.seed);
    let mut entries = Vec::with_capacity(cfg.n_vectors);

    for id in 0..cfg.n_vectors {
        let vector: Vec<f32> = (0..cfg.dims)
            .map(|_| {
                // Sum of 8 uniform → approximately Normal(4, 8/12) → centre + scale
                let s: f32 = (0..8).map(|_| rng.next_f32()).sum::<f32>();
                (s - 4.0) / 1.63 // ≈ Normal(0, 1)
            })
            .collect();

        // Assign required_per_vector distinct capability bits
        let required = pick_caps(&mut rng, cfg.n_caps, cfg.required_per_vector);
        entries.push(VecEntry {
            id,
            vector,
            required,
        });
    }
    entries
}

/// Pick `k` distinct bits from [0, n_caps) and return as a CapMask.
fn pick_caps(rng: &mut Lcg, n_caps: u8, k: u8) -> CapMask {
    if k == 0 || n_caps == 0 {
        return CapMask::NONE;
    }
    let mut bits: Vec<u8> = (0..n_caps).collect();
    // Partial Fisher-Yates
    for i in 0..k as usize {
        let j = i + rng.next_usize(n_caps as usize - i);
        bits.swap(i, j);
    }
    let mut mask = 0u64;
    for &b in &bits[..k as usize] {
        mask |= 1u64 << b;
    }
    CapMask(mask)
}

/// Generate `n_queries` random query vectors + a querier capability mask.
///
/// The querier holds `held_caps` bits chosen randomly from the available caps.
pub fn generate_queries(
    n_queries: usize,
    dims: usize,
    n_caps: u8,
    held_caps: u8,
    seed: u64,
) -> (Vec<Vec<f32>>, CapMask) {
    let mut rng = Lcg::new(seed ^ 0xdeadbeef_cafef00d);
    let queries = (0..n_queries)
        .map(|_| {
            (0..dims)
                .map(|_| {
                    let s: f32 = (0..8).map(|_| rng.next_f32()).sum::<f32>();
                    (s - 4.0) / 1.63
                })
                .collect()
        })
        .collect();
    let holder = pick_caps(&mut rng, n_caps, held_caps);
    (queries, holder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_deterministic() {
        let cfg = DatasetConfig {
            n_vectors: 10,
            dims: 4,
            n_caps: 8,
            required_per_vector: 2,
            seed: 42,
        };
        let a = generate(&cfg);
        let b = generate(&cfg);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.required, y.required);
            for (xv, yv) in x.vector.iter().zip(y.vector.iter()) {
                assert!((xv - yv).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn pick_caps_correct_count() {
        let mut rng = Lcg::new(0);
        for _ in 0..50 {
            let mask = pick_caps(&mut rng, 8, 3);
            assert_eq!(mask.count(), 3);
        }
    }

    #[test]
    fn pick_caps_within_range() {
        let mut rng = Lcg::new(1);
        for _ in 0..100 {
            let mask = pick_caps(&mut rng, 6, 2);
            // Only bits 0..6 should be set
            assert_eq!(mask.0 & !((1u64 << 6) - 1), 0);
        }
    }
}
