//! Deterministic synthetic scenario generator for the merge-policy benchmark.
//!
//! Ground-truth design (kept honest — see `docs/research/nightly` write-up):
//! each write carries a hidden `alpha` (structural-shift magnitude towards a
//! drifting "ideal" target) that a policy never observes directly. `alpha`
//! drives three *independent, noisy* channels a policy CAN observe:
//!
//! 1. the structural snapshot (embedding/entropy/graph/coherence/pred_error
//!    deltas consumed by `StructuralProperTime`),
//! 2. the coherence context window (a separately-noised sample of the ideal
//!    point, not the same noise draw as the write embedding), and
//! 3. wall-clock timestamp, corrupted by a persistent per-agent skew.
//!
//! `true_quality = alpha + independent noise` is the evaluation label. No
//! policy ever reads `alpha` or `true_quality` — only the noisy observables.
//! This keeps the "correct resolution rate" metric from leaking its own
//! answer into the algorithm under test.

use crate::vclock::{AgentId, VectorClock};
use crate::{MemoryKey, MemoryWrite};
use emergent_time::structural_clock::StateSnapshot;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    /// Uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in `[-1, 1)`.
    fn next_signed(&mut self) -> f64 {
        self.next_unit() * 2.0 - 1.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScenarioConfig {
    pub num_conflicts: usize,
    pub num_causal_controls: usize,
    pub dim: usize,
    pub num_agents: u32,
    /// Max magnitude (ms) of each agent's fixed wall-clock skew bias.
    pub wall_skew_ms: f64,
    /// Intra-event jitter (ms) applied even at zero skew.
    pub jitter_ms: f64,
    pub seed: u64,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        ScenarioConfig {
            num_conflicts: 2000,
            num_causal_controls: 500,
            dim: 16,
            num_agents: 6,
            wall_skew_ms: 400.0,
            jitter_ms: 30.0,
            seed: 0xA6E17,
        }
    }
}

pub struct ConflictCase {
    pub key: MemoryKey,
    pub a: MemoryWrite,
    pub b: MemoryWrite,
    /// Hidden ground truth — visible only to the evaluator, never a policy.
    pub quality_a: f64,
    pub quality_b: f64,
    pub context: Vec<Vec<f64>>,
    /// `false` for the causal-control group (b causally follows a).
    pub is_concurrent: bool,
}

fn rand_vec(rng: &mut Rng, dim: usize) -> Vec<f64> {
    (0..dim).map(|_| rng.next_signed()).collect()
}

fn lerp_add(base: &[f64], target: &[f64], alpha: f64, noise: &[f64], noise_scale: f64) -> Vec<f64> {
    base.iter()
        .zip(target)
        .zip(noise)
        .map(|((b, t), n)| b + alpha * (t - b) + noise_scale * n)
        .collect()
}

/// Build one write whose hidden shift magnitude towards `ideal` is `alpha`.
#[allow(clippy::too_many_arguments)]
fn make_write(
    rng: &mut Rng,
    agent: AgentId,
    key: MemoryKey,
    vclock: VectorClock,
    base_ms: f64,
    skew_ms: f64,
    jitter_ms: f64,
    ideal: &[f64],
    alpha: f64,
    dim: usize,
) -> (MemoryWrite, f64) {
    let prev_embedding = rand_vec(rng, dim);
    let struct_noise = rand_vec(rng, dim);
    let new_embedding = lerp_add(&prev_embedding, ideal, alpha, &struct_noise, 0.15);

    let prev_snapshot = StateSnapshot::full(prev_embedding, 1.0, 0.5, 0.0, 0.0);
    let snapshot = StateSnapshot::full(
        new_embedding,
        1.0 - 0.6 * alpha + 0.05 * rng.next_signed(), // entropy drops as the write converges
        (0.3 + 0.6 * alpha + 0.05 * rng.next_signed()).clamp(0.0, 1.0),
        0.4 * alpha + 0.03 * rng.next_signed().abs(),
        (1.0 - alpha + 0.05 * rng.next_signed()).clamp(0.0, 1.0),
    );

    let wall_ts_ms = base_ms + jitter_ms * rng.next_signed() + skew_ms;
    let write = MemoryWrite {
        agent_id: agent,
        key,
        wall_ts_ms,
        vclock,
        prev_snapshot,
        snapshot,
    };
    let quality = (alpha + 0.20 * rng.next_signed()).clamp(0.0, 1.5);
    (write, quality)
}

/// Generate the benchmark corpus: `num_conflicts` genuinely concurrent write
/// pairs (the primary test set) followed by `num_causal_controls`
/// causally-ordered pairs (sanity/control set for causal-order preservation).
pub fn generate(cfg: &ScenarioConfig) -> Vec<ConflictCase> {
    let mut rng = Rng::new(cfg.seed);
    let mut skew = Vec::with_capacity(cfg.num_agents as usize);
    for _ in 0..cfg.num_agents {
        skew.push(cfg.wall_skew_ms * rng.next_signed());
    }

    let mut cases = Vec::with_capacity(cfg.num_conflicts + cfg.num_causal_controls);
    let step_ms = 1000.0;

    for i in 0..cfg.num_conflicts {
        let ideal = rand_vec(&mut rng, cfg.dim);
        let ag_a = (rng.next_unit() * cfg.num_agents as f64) as u32 % cfg.num_agents;
        let mut ag_b = (rng.next_unit() * cfg.num_agents as f64) as u32 % cfg.num_agents;
        if ag_b == ag_a {
            ag_b = (ag_b + 1) % cfg.num_agents;
        }
        let alpha_a = rng.next_unit();
        let alpha_b = rng.next_unit();
        let base_ms = i as f64 * step_ms;

        let mut vc_a = VectorClock::new();
        vc_a.tick(ag_a);
        let mut vc_b = VectorClock::new();
        vc_b.tick(ag_b);

        let (wa, qa) = make_write(
            &mut rng,
            ag_a,
            i as MemoryKey,
            vc_a,
            base_ms,
            skew[ag_a as usize],
            cfg.jitter_ms,
            &ideal,
            alpha_a,
            cfg.dim,
        );
        let (wb, qb) = make_write(
            &mut rng,
            ag_b,
            i as MemoryKey,
            vc_b,
            base_ms,
            skew[ag_b as usize],
            cfg.jitter_ms,
            &ideal,
            alpha_b,
            cfg.dim,
        );

        // Context window: independently-noised samples around the same ideal
        // point (different noise draw than either write's embedding).
        let context: Vec<Vec<f64>> = (0..3)
            .map(|_| {
                lerp_add(
                    &vec![0.0; cfg.dim],
                    &ideal,
                    1.0,
                    &rand_vec(&mut rng, cfg.dim),
                    0.35,
                )
            })
            .collect();

        cases.push(ConflictCase {
            key: i as MemoryKey,
            a: wa,
            b: wb,
            quality_a: qa,
            quality_b: qb,
            context,
            is_concurrent: true,
        });
    }

    for j in 0..cfg.num_causal_controls {
        let idx = cfg.num_conflicts + j;
        let ideal = rand_vec(&mut rng, cfg.dim);
        let ag_a = (rng.next_unit() * cfg.num_agents as f64) as u32 % cfg.num_agents;
        let mut ag_b = (rng.next_unit() * cfg.num_agents as f64) as u32 % cfg.num_agents;
        if ag_b == ag_a {
            ag_b = (ag_b + 1) % cfg.num_agents;
        }
        let base_ms = idx as f64 * step_ms;

        let mut vc_a = VectorClock::new();
        vc_a.tick(ag_a);
        let alpha_a = rng.next_unit();
        let (wa, qa) = make_write(
            &mut rng,
            ag_a,
            idx as MemoryKey,
            vc_a.clone(),
            base_ms,
            skew[ag_a as usize],
            cfg.jitter_ms,
            &ideal,
            alpha_a,
            cfg.dim,
        );

        // b causally observes a's write, then ticks locally: b happens-after a.
        let mut vc_b = vc_a.clone();
        vc_b.merge(&wa.vclock);
        vc_b.tick(ag_b);
        let alpha_b = rng.next_unit();
        let (wb, qb) = make_write(
            &mut rng,
            ag_b,
            idx as MemoryKey,
            vc_b,
            base_ms + step_ms * 0.5,
            skew[ag_b as usize],
            cfg.jitter_ms,
            &ideal,
            alpha_b,
            cfg.dim,
        );

        let context: Vec<Vec<f64>> = (0..3)
            .map(|_| {
                lerp_add(
                    &vec![0.0; cfg.dim],
                    &ideal,
                    1.0,
                    &rand_vec(&mut rng, cfg.dim),
                    0.35,
                )
            })
            .collect();

        cases.push(ConflictCase {
            key: idx as MemoryKey,
            a: wa,
            b: wb,
            quality_a: qa,
            quality_b: qb,
            context,
            is_concurrent: false,
        });
    }

    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_pairs_are_actually_concurrent() {
        let cfg = ScenarioConfig {
            num_conflicts: 50,
            num_causal_controls: 10,
            ..Default::default()
        };
        let cases = generate(&cfg);
        for c in cases.iter().filter(|c| c.is_concurrent) {
            assert_eq!(
                c.a.vclock.compare(&c.b.vclock),
                crate::vclock::CausalOrder::Concurrent
            );
        }
    }

    #[test]
    fn control_pairs_are_causally_ordered() {
        let cfg = ScenarioConfig {
            num_conflicts: 10,
            num_causal_controls: 50,
            ..Default::default()
        };
        let cases = generate(&cfg);
        for c in cases.iter().filter(|c| !c.is_concurrent) {
            assert_eq!(
                c.a.vclock.compare(&c.b.vclock),
                crate::vclock::CausalOrder::Before
            );
        }
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let cfg = ScenarioConfig::default();
        let a = generate(&cfg);
        let b = generate(&cfg);
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].quality_a, b[0].quality_a);
        assert_eq!(a[0].a.wall_ts_ms, b[0].a.wall_ts_ms);
    }
}
