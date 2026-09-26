//! Deterministic seeded synthetic data generation.
//!
//! No `rand` dependency: a splitmix64 core gives byte-identical output for a
//! given seed on any platform, which matters because the whole experiment is a
//! paired comparison — any hidden nondeterminism would show up as leakage.

/// splitmix64 PRNG. Deterministic, seedable, no external dependency.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal via Box-Muller (one sample per call, second discarded —
    /// determinism is worth more here than the halved PRNG cost).
    #[inline]
    pub fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_f32() as f64).max(1e-9);
        let u2 = self.next_f32() as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    /// Uniform integer in `[0, n)`.
    #[inline]
    pub fn next_below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

/// Configuration for the clustered synthetic corpus.
#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub dim: usize,
    pub clusters: usize,
    /// Standard deviation of the per-coordinate Gaussian jitter around a centre.
    pub sigma: f32,
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        // clusters=64 / sigma=0.60 is the regime the feasibility probe
        // (`erasure-probe`, Probe 1) identified as non-degenerate: recall@10
        // = 0.93 at ef=64, matching the 0.914 the 2026-06-18 nightly measured
        // on its uniform corpus. The initially chosen 16/0.18 produced
        // recall@10 = 0.22 even at ef=256 — ground-truth neighbours were
        // effectively ties, so no deletion could be observable.
        Self {
            dim: 32,
            clusters: 64,
            sigma: 0.60,
            seed: 0x5EED_0001,
        }
    }
}

/// A generator that produces vectors from a fixed set of cluster centres.
///
/// Held-out targets and decoys must come from the *same* distribution as the
/// base corpus, otherwise the distinguisher would be detecting "this query is
/// weird", not "this vector was deleted".
pub struct ClusteredSource {
    pub cfg: DatasetConfig,
    centres: Vec<Vec<f32>>,
    rng: Rng,
}

impl ClusteredSource {
    pub fn new(cfg: DatasetConfig) -> Self {
        let mut crng = Rng::new(cfg.seed ^ 0xC0FF_EE00);
        let centres: Vec<Vec<f32>> = (0..cfg.clusters)
            .map(|_| (0..cfg.dim).map(|_| crng.next_f32()).collect())
            .collect();
        let rng = Rng::new(cfg.seed);
        Self { cfg, centres, rng }
    }

    /// Draw one vector: pick a cluster uniformly, add Gaussian jitter.
    pub fn sample(&mut self) -> Vec<f32> {
        let c = self.rng.next_below(self.cfg.clusters);
        let sigma = self.cfg.sigma;
        let mut out = Vec::with_capacity(self.cfg.dim);
        for d in 0..self.cfg.dim {
            out.push(self.centres[c][d] + sigma * self.rng.next_normal());
        }
        out
    }

    pub fn sample_many(&mut self, n: usize) -> Vec<Vec<f32>> {
        (0..n).map(|_| self.sample()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_for_a_seed() {
        let a: Vec<u64> = (0..8).map(|_| Rng::new(7).next_u64()).collect();
        let mut r = Rng::new(7);
        let b: Vec<u64> = (0..8).map(|_| r.next_u64()).collect();
        // Every fresh Rng::new(7) yields the same first value.
        assert!(a.iter().all(|&x| x == a[0]));
        assert_eq!(a[0], b[0]);
        // And the stream itself advances (not a constant generator).
        assert_ne!(b[0], b[1]);
    }

    #[test]
    fn uniform_sample_is_in_unit_interval() {
        let mut r = Rng::new(99);
        for _ in 0..1000 {
            let f = r.next_f32();
            assert!((0.0..1.0).contains(&f), "f32 out of range: {f}");
        }
    }

    #[test]
    fn clustered_source_is_reproducible() {
        let cfg = DatasetConfig::default();
        let mut a = ClusteredSource::new(cfg.clone());
        let mut b = ClusteredSource::new(cfg);
        assert_eq!(a.sample_many(16), b.sample_many(16));
    }

    #[test]
    fn clusters_are_separated_relative_to_jitter() {
        // Sanity: within-cluster spread must be smaller than the corpus spread,
        // otherwise "clustered" data is really uniform noise.
        let cfg = DatasetConfig {
            dim: 32,
            clusters: 8,
            sigma: 0.10,
            seed: 4242,
        };
        let mut src = ClusteredSource::new(cfg.clone());
        let pts = src.sample_many(500);
        let mut mean = vec![0.0f32; cfg.dim];
        for p in &pts {
            for d in 0..cfg.dim {
                mean[d] += p[d] / pts.len() as f32;
            }
        }
        let global_var: f32 = pts
            .iter()
            .map(|p| (0..cfg.dim).map(|d| (p[d] - mean[d]).powi(2)).sum::<f32>())
            .sum::<f32>()
            / pts.len() as f32;
        let within_var = cfg.sigma * cfg.sigma * cfg.dim as f32;
        assert!(
            within_var < global_var,
            "within-cluster variance {within_var} should be < global {global_var}"
        );
    }
}
