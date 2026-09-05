//! Search-time genome for `ruvector-coherence-hnsw`'s `CoherenceGatedSearch`.
//!
//! Only *query-time* knobs are evolved — the coherence threshold and beam
//! width `ef`. Graph-build-time knobs (`m`, `m_longjump`) are fixed for the
//! whole run: mutating them would require rebuilding the O(N²) k-NN graph
//! every generation, which is a different (index-time) tuning problem.

use rand::Rng;

/// Two-parameter genome: `[threshold, ef]`, stored as `f32` so it doubles as
/// a `ruvector_proof_gate::WritePayload` vector with no re-encoding step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Genome {
    /// Coherence-gate threshold, clamped to `traversal_coherence`'s range.
    pub threshold: f32,
    /// Beam width, clamped to a sane search range.
    pub ef: f32,
}

/// Mutation and clamp bounds, fixed before any search is run.
pub const THRESHOLD_MIN: f32 = -1.0;
pub const THRESHOLD_MAX: f32 = 1.0;
pub const EF_MIN: f32 = 10.0;
pub const EF_MAX: f32 = 300.0;

/// Default (unturned) genome: the value a production team would pick by hand
/// — a middling coherence threshold and the `ruvector-coherence-hnsw`
/// benchmark's default `ef`.
pub const DEFAULT_GENOME: Genome = Genome {
    threshold: 0.50,
    ef: 80.0,
};

impl Genome {
    /// Round `ef` to the nearest valid beam width for `Searcher::search`.
    pub fn ef_usize(&self) -> usize {
        self.ef.round().clamp(EF_MIN, EF_MAX) as usize
    }

    /// Gaussian-perturb both genes and clamp to the legal range. `sigma_*`
    /// are the per-generation step sizes (fixed, not self-adapted — keeps
    /// the (1+1)-ES trivially deterministic given a seeded RNG).
    pub fn mutate(&self, rng: &mut impl Rng, sigma_threshold: f32, sigma_ef: f32) -> Genome {
        let dt = sample_normal(rng) * sigma_threshold;
        let de = sample_normal(rng) * sigma_ef;
        Genome {
            threshold: (self.threshold + dt).clamp(THRESHOLD_MIN, THRESHOLD_MAX),
            ef: (self.ef + de).clamp(EF_MIN, EF_MAX),
        }
    }

    /// Canonical `[threshold, ef]` encoding — this is exactly what gets
    /// written into `WritePayload.vector` and hashed into the witness chain.
    pub fn to_vec(self) -> Vec<f32> {
        vec![self.threshold, self.ef]
    }

    /// Inverse of [`Genome::to_vec`]. Used by replay to reconstruct the
    /// genome from a committed `WritePayload` without trusting any other
    /// field.
    pub fn from_vec(v: &[f32]) -> Option<Genome> {
        if v.len() != 2 {
            return None;
        }
        Some(Genome {
            threshold: v[0],
            ef: v[1],
        })
    }
}

/// Box-Muller standard normal sample — avoids pulling in `rand_distr` for a
/// single distribution.
fn sample_normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.gen_range(1e-9..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn mutate_stays_in_bounds() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut g = DEFAULT_GENOME;
        for _ in 0..1000 {
            g = g.mutate(&mut rng, 5.0, 500.0); // deliberately huge steps
            assert!(g.threshold >= THRESHOLD_MIN && g.threshold <= THRESHOLD_MAX);
            assert!(g.ef >= EF_MIN && g.ef <= EF_MAX);
        }
    }

    #[test]
    fn vec_roundtrip_is_exact() {
        let g = Genome {
            threshold: 0.314,
            ef: 123.0,
        };
        let v = g.to_vec();
        assert_eq!(Genome::from_vec(&v), Some(g));
    }

    #[test]
    fn same_seed_produces_identical_mutation_sequence() {
        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        let mut ga = DEFAULT_GENOME;
        let mut gb = DEFAULT_GENOME;
        for _ in 0..50 {
            ga = ga.mutate(&mut a, 0.05, 10.0);
            gb = gb.mutate(&mut b, 0.05, 10.0);
        }
        assert_eq!(ga, gb);
    }
}
