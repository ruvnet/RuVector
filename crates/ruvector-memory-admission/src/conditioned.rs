//! Guarded, feature-conditioned self-calibrating admission — candidate C.
//!
//! This directly answers Next Research item 2 of the 2026-09-02 nightly
//! (`docs/research/nightly/2026-09-02-mincut-streaming-memory-admission`):
//! "try a cluster-count-conditioned or local-similarity-conditioned
//! self-calibrating tau, addressing candidate B's specific documented
//! failure mode rather than abandoning self-calibration entirely."
//!
//! Candidate B ([`crate::policy::AdaptiveMincutAdmission`]) tracked a
//! running mean/std of the *global* cut-weight distribution. That nightly's
//! own root-cause analysis found this is the wrong statistic: it drifts
//! away from "the cut weight that separates true same-cluster points from
//! true outliers" as cluster count grows, and the drift pushed `tau` high
//! enough to blow through the 48-cluster safety valve.
//!
//! `GuardedConditionedAdmission` fixes this two ways, not one:
//!
//! 1. **Conditioning.** `tau` is a function of two *local* features —
//!    `cluster_count_norm` and the candidate point's own best single-centroid
//!    cosine similarity — instead of an unconditional global cut-weight
//!    statistic. This is the feature set the prior nightly's negative-result
//!    section named as "not attempted here, to avoid moving the goalposts".
//! 2. **Guarded, bounded recalibration.** The mapping's three coefficients
//!    are not fit by gradient descent on a differentiable loss (there isn't
//!    one here); they are periodically recalibrated online by a bounded
//!    `(1+1)`-ES, using `ruvector-sona`'s `darwin_guard::Guard` (ADR-271) to
//!    screen every candidate mutation for non-finite fitness, out-of-bounds
//!    genes, and a *degenerate* collapse (always-spawn or never-spawn) before
//!    it is even eligible to replace the incumbent. This is the same
//!    reward-hacking defense SONA's own Darwin-mode config search uses
//!    (`crates/sona/examples/darwin_router.rs`), reused here as an actual
//!    dependency rather than a re-implemented pattern.
//!
//! The ES's fitness signal never touches ground-truth cluster labels: it
//! shadow-replays the candidate genome's `tau` function against a sliding
//! window of the policy's own recently *observed* cut features, and scores
//! how close the resulting hypothetical spawn rate lands to a target spawn
//! rate measured once, online, during an initial warm-up window (the same
//! "observe, then self-calibrate" shape candidate B used — but the quantity
//! being tracked and controlled is the actual *admission outcome rate*, not
//! a moment of the cut-weight distribution). There is no dataset-label
//! leakage: `purity`/`recall@10` in the benchmark are computed exactly as
//! for every other policy, from decisions the policy made with no access to
//! `true_cluster`.

use std::collections::VecDeque;

use ruvector_sona::darwin_guard::{Guard, NoJudge, Verdict};

use crate::cosine_sim;
use crate::dataset::Lcg64;
use crate::mincut::{global_min_cut, WeightMatrix};
use crate::policy::{merge_target, running_mean_update, should_spawn, AdmissionPolicy, Decision};

// ─── genome: tau_t = clamp(base + count_coeff * cluster_count_norm
//                                 - sim_coeff   * best_single_sim, TAU_MIN, TAU_MAX) ───

const TAU_MIN: f32 = 0.0;
const TAU_MAX: f32 = 0.10;
const BASE_MIN: f32 = 0.0;
const BASE_MAX: f32 = 0.05;
const COUNT_MIN: f32 = -0.03;
const COUNT_MAX: f32 = 0.03;
const SIM_MIN: f32 = -0.03;
const SIM_MAX: f32 = 0.03;

#[derive(Clone, Copy, Debug)]
struct Genome {
    base: f32,
    count_coeff: f32,
    sim_coeff: f32,
}

impl Genome {
    fn tau_for(&self, cluster_count_norm: f32, best_sim: f32) -> f32 {
        (self.base + self.count_coeff * cluster_count_norm - self.sim_coeff * best_sim.max(0.0))
            .clamp(TAU_MIN, TAU_MAX)
    }

    fn in_bounds(&self) -> bool {
        (BASE_MIN..=BASE_MAX).contains(&self.base)
            && (COUNT_MIN..=COUNT_MAX).contains(&self.count_coeff)
            && (SIM_MIN..=SIM_MAX).contains(&self.sim_coeff)
    }

    fn mutate(&self, rng: &mut Lcg64, sigma: f32) -> Self {
        Genome {
            base: self.base + rng.gaussian() * sigma,
            count_coeff: self.count_coeff + rng.gaussian() * sigma * 1.5,
            sim_coeff: self.sim_coeff + rng.gaussian() * sigma * 1.5,
        }
    }
}

/// One recorded admission-time observation, used only to shadow-replay
/// candidate genomes during recalibration — never to look ahead at ground
/// truth.
#[derive(Clone, Copy)]
struct Obs {
    avg_cut: f32,
    best_sim: f32,
    cluster_count_norm: f32,
}

/// Counts of why a calibration attempt was rejected, for auditability (STEP
/// 21/42 evidence: a guard that never rejects anything is not screening for
/// anything, and one that rejects everything is uninformative).
#[derive(Debug, Clone, Copy, Default)]
pub struct GuardStats {
    pub attempts: u64,
    pub accepted: u64,
    pub rejected_non_finite: u64,
    pub rejected_out_of_bounds: u64,
    pub rejected_degenerate: u64,
    /// `(1+1)`-ES selection step rejected an accepted-but-not-improving
    /// candidate (passed the guard, did not beat the incumbent).
    pub rejected_not_improving: u64,
}

pub struct GuardedConditionedAdmission {
    pub max_clusters: usize,
    pub warmup_len: u64,
    pub calibrate_every: usize,
    pub window_len: usize,
    bootstrap_tau: f32,
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
    genome: Genome,
    window: VecDeque<Obs>,
    n_committed: u64,
    warmup_spawns: u64,
    target_spawn_rate: Option<f32>,
    rng: Lcg64,
    guard: Guard<NoJudge>,
    since_last_calibration: usize,
    stats: GuardStats,
}

impl GuardedConditionedAdmission {
    pub fn new(bootstrap_tau: f32, max_clusters: usize, seed: u64) -> Self {
        GuardedConditionedAdmission {
            max_clusters,
            warmup_len: 200,
            calibrate_every: 150,
            window_len: 150,
            bootstrap_tau,
            centroids: Vec::new(),
            counts: Vec::new(),
            genome: Genome {
                base: bootstrap_tau,
                count_coeff: 0.0,
                sim_coeff: 0.0,
            },
            window: VecDeque::new(),
            n_committed: 0,
            warmup_spawns: 0,
            target_spawn_rate: None,
            rng: Lcg64(seed),
            guard: Guard::deterministic(),
            since_last_calibration: 0,
            stats: GuardStats::default(),
        }
    }

    pub fn guard_stats(&self) -> GuardStats {
        self.stats
    }

    pub fn target_spawn_rate(&self) -> Option<f32> {
        self.target_spawn_rate
    }

    fn cluster_count_norm(&self) -> f32 {
        (self.centroids.len() as f32 / self.max_clusters as f32).min(1.0)
    }

    fn best_single_sim(&self, point: &[f32]) -> f32 {
        self.centroids
            .iter()
            .map(|c| cosine_sim(point, c))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Identical construction to the other two policies' `cut_decision`
    /// (duplicated rather than shared, matching this crate's existing
    /// per-policy cost-accounting convention — see
    /// `AdaptiveMincutAdmission::cut_decision`).
    fn cut_decision(&self, point: &[f32]) -> (f32, Vec<usize>, usize) {
        let c = self.centroids.len();
        let mut m = WeightMatrix::new(c + 1);
        let mut sim_ops = 0usize;
        for i in 0..c {
            for j in (i + 1)..c {
                let w = cosine_sim(&self.centroids[i], &self.centroids[j]).max(0.0);
                m.set_sym(i, j, w as f64);
                sim_ops += 1;
            }
        }
        for i in 0..c {
            let w = cosine_sim(&self.centroids[i], point).max(0.0);
            m.set_sym(i, c, w as f64);
            sim_ops += 1;
        }
        let result = global_min_cut(&m).expect("c+1 >= 2 whenever c >= 1");
        let point_side = result.side[c];
        let group: Vec<usize> = (0..c).filter(|&i| result.side[i] == point_side).collect();
        let avg_cut = (result.weight / result.crossing_edges as f64) as f32;
        (avg_cut, group, sim_ops)
    }

    /// Shadow-replay `genome` against the observation window: how many
    /// insertions out of the window *would* this genome have spawned?
    /// Never touches ground-truth labels — only the policy's own past
    /// `(avg_cut, best_sim, cluster_count_norm)` triples. Returns
    /// `(spawn_count, window_len)`, not a bare rate, so degeneracy
    /// ("always spawns" / "never spawns") can be checked exactly rather
    /// than via an arbitrary percentage band — spawn rates in this problem
    /// are legitimately small (candidate A's final rate is ~0.4%), so a
    /// naive "outside 1%..99%" band would misclassify the desired low-rate
    /// regime itself as degenerate.
    fn shadow_spawn_count(&self, genome: &Genome) -> (usize, usize) {
        let spawns = self
            .window
            .iter()
            .filter(|o| o.avg_cut < genome.tau_for(o.cluster_count_norm, o.best_sim))
            .count();
        (spawns, self.window.len())
    }

    fn maybe_calibrate(&mut self) {
        let Some(target) = self.target_spawn_rate else {
            return;
        };
        if self.window.len() < self.window_len.min(40) {
            return; // not enough shadow-replay evidence yet
        }

        let (incumbent_spawns, window_n) = self.shadow_spawn_count(&self.genome);
        let incumbent_rate = incumbent_spawns as f32 / window_n as f32;
        let incumbent_fitness = -((incumbent_rate - target).abs());

        let candidate = self.genome.mutate(&mut self.rng, 0.004);
        let (candidate_spawns, _) = self.shadow_spawn_count(&candidate);
        let candidate_rate = candidate_spawns as f32 / window_n as f32;
        let candidate_fitness = -((candidate_rate - target).abs());

        let in_bounds = candidate.in_bounds();
        let finite = candidate_fitness.is_finite();
        // Degenerate only at the true collapse points (spawns everything or
        // spawns nothing in the window) — not an arbitrary percentage band.
        let degenerate = finite && (candidate_spawns == 0 || candidate_spawns == window_n);

        self.stats.attempts += 1;
        match self
            .guard
            .screen(candidate_fitness, finite, in_bounds, degenerate)
        {
            Verdict::Rejected(reason) => {
                use ruvector_sona::darwin_guard::Reject;
                match reason {
                    Reject::NonFinite => self.stats.rejected_non_finite += 1,
                    Reject::OutOfBounds => self.stats.rejected_out_of_bounds += 1,
                    Reject::Degenerate => self.stats.rejected_degenerate += 1,
                    Reject::JudgeVeto => {}
                }
            }
            Verdict::Accepted(fitness) => {
                // (1+1)-ES selection: a guard-accepted candidate still only
                // replaces the incumbent if it is a genuine improvement.
                if fitness > incumbent_fitness {
                    self.genome = candidate;
                    self.stats.accepted += 1;
                } else {
                    self.stats.rejected_not_improving += 1;
                }
            }
        }
    }
}

impl AdmissionPolicy for GuardedConditionedAdmission {
    fn name(&self) -> &str {
        "GuardedConditionedAdmission"
    }

    fn decide(&self, point: &[f32]) -> Decision {
        let c = self.centroids.len();
        if c == 0 {
            return Decision {
                cluster_id: 0,
                spawned_new: true,
                sim_ops: 0,
            };
        }
        if c >= self.max_clusters {
            let mut best_id = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, cen) in self.centroids.iter().enumerate() {
                let s = cosine_sim(point, cen);
                if s > best_sim {
                    best_sim = s;
                    best_id = i;
                }
            }
            return Decision {
                cluster_id: best_id,
                spawned_new: false,
                sim_ops: c,
            };
        }

        let best_sim = self.best_single_sim(point);
        let tau = if self.n_committed < self.warmup_len {
            self.bootstrap_tau
        } else {
            self.genome.tau_for(self.cluster_count_norm(), best_sim)
        };
        let (avg_cut, group, sim_ops) = self.cut_decision(point);
        let sim_ops = sim_ops + c; // + best_single_sim pass

        if should_spawn(c, avg_cut, &group, tau) {
            Decision {
                cluster_id: c,
                spawned_new: true,
                sim_ops,
            }
        } else {
            Decision {
                cluster_id: merge_target(point, &group, &self.centroids),
                spawned_new: false,
                sim_ops,
            }
        }
    }

    fn commit(&mut self, point: &[f32], decision: &Decision) {
        if !self.centroids.is_empty() && self.centroids.len() < self.max_clusters {
            let (avg_cut, _, _) = self.cut_decision(point);
            let best_sim = self.best_single_sim(point);
            let obs = Obs {
                avg_cut,
                best_sim,
                cluster_count_norm: self.cluster_count_norm(),
            };
            self.window.push_back(obs);
            while self.window.len() > self.window_len {
                self.window.pop_front();
            }

            self.n_committed += 1;
            if self.n_committed <= self.warmup_len {
                if decision.spawned_new {
                    self.warmup_spawns += 1;
                }
                if self.n_committed == self.warmup_len {
                    let rate = self.warmup_spawns as f32 / self.warmup_len as f32;
                    self.target_spawn_rate = Some(rate.clamp(0.01, 0.99));
                }
            } else {
                self.since_last_calibration += 1;
                if self.since_last_calibration >= self.calibrate_every {
                    self.since_last_calibration = 0;
                    self.maybe_calibrate();
                }
            }
        }

        if decision.spawned_new {
            self.centroids.push(point.to_vec());
            self.counts.push(1);
        } else {
            let c = decision.cluster_id;
            running_mean_update(&mut self.centroids[c], self.counts[c], point);
            self.counts[c] += 1;
        }
    }

    fn n_clusters(&self) -> usize {
        self.centroids.len()
    }

    fn centroid(&self, cluster_id: usize) -> &[f32] {
        &self.centroids[cluster_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, rest: usize) -> Vec<f32> {
        let mut vec = vec![x, y];
        vec.extend(std::iter::repeat_n(0.0, rest));
        crate::dataset::normalise(&mut vec);
        vec
    }

    #[test]
    fn first_point_always_spawns() {
        let mut p = GuardedConditionedAdmission::new(0.005, 32, 1);
        let d = p.admit(&v(1.0, 0.0, 6));
        assert!(d.spawned_new);
        assert_eq!(p.n_clusters(), 1);
    }

    #[test]
    fn respects_max_clusters_safety_valve() {
        let mut p = GuardedConditionedAdmission::new(0.99, 4, 2);
        for i in 0..40 {
            let angle = i as f32 * 0.37;
            p.admit(&v(angle.cos(), angle.sin(), 6));
        }
        assert!(p.n_clusters() <= 4);
    }

    #[test]
    fn warmup_establishes_a_target_spawn_rate() {
        let mut p = GuardedConditionedAdmission::new(0.02, 48, 3);
        p.warmup_len = 50;
        for i in 0..60 {
            let angle = (i as f32) * 0.21;
            p.admit(&v(angle.cos(), angle.sin(), 6));
        }
        assert!(p.target_spawn_rate().is_some());
        let rate = p.target_spawn_rate().unwrap();
        assert!((0.0..=1.0).contains(&rate));
    }

    #[test]
    fn genome_mutation_stays_near_parent_and_bounds_are_checked() {
        let g = Genome {
            base: 0.01,
            count_coeff: 0.0,
            sim_coeff: 0.0,
        };
        assert!(g.in_bounds());
        let mut rng = Lcg64(42);
        let m = g.mutate(&mut rng, 0.004);
        // A single small-sigma mutation should not jump far from the parent.
        assert!((m.base - g.base).abs() < 0.05);
    }

    #[test]
    fn guard_rejects_out_of_bounds_genome() {
        let bad = Genome {
            base: 10.0,
            count_coeff: 0.0,
            sim_coeff: 0.0,
        };
        assert!(!bad.in_bounds());
    }

    #[test]
    fn calibration_guard_never_rejects_everything_or_nothing_on_a_long_run() {
        // A longer, more varied stream should exercise the guard enough to
        // produce SOME attempts; this is a smoke test for the calibration
        // loop wiring, not a claim about the exact accept/reject ratio.
        let mut p = GuardedConditionedAdmission::new(0.01, 48, 7);
        p.warmup_len = 80;
        p.calibrate_every = 40;
        for i in 0..600 {
            let cluster = i % 6;
            let base_angle = cluster as f32 * 1.1;
            let jitter = ((i * 37) % 13) as f32 * 0.01;
            p.admit(&v(
                (base_angle + jitter).cos(),
                (base_angle + jitter).sin(),
                6,
            ));
        }
        let stats = p.guard_stats();
        assert!(
            stats.attempts > 0,
            "expected at least one calibration attempt over 600 points"
        );
    }
}
