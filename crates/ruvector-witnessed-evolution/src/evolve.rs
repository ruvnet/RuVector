//! The (1+1)-evolution strategy shared by the witnessed and unwitnessed
//! variants. Both call the same mutation/acceptance logic with the same
//! seed, so their search trajectories are provably identical — the only
//! difference is whether each generation is committed to a
//! [`crate::witness::WitnessedLineage`].

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::fitness::{Fitness, Workload};
use crate::genome::{Genome, DEFAULT_GENOME};
use crate::witness::WitnessedLineage;

/// Mutation step sizes, fixed before any generation is evaluated.
pub const SIGMA_THRESHOLD: f32 = 0.08;
pub const SIGMA_EF: f32 = 12.0;

/// Best genome/fitness an ES run converged to, plus how many mutation
/// attempts it made.
#[derive(Debug, Clone, Copy)]
pub struct EsOutcome {
    pub best_genome: Genome,
    pub best_fitness: Fitness,
    pub generations: usize,
}

/// Plain (1+1)-ES, no witnessing: `generations` mutation attempts starting
/// from [`DEFAULT_GENOME`], greedily keeping whichever of incumbent/mutant
/// has higher composite fitness.
pub fn run_unwitnessed(workload: &Workload, generations: usize, seed: u64) -> EsOutcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut incumbent = DEFAULT_GENOME;
    let mut incumbent_fitness = workload.evaluate(incumbent);
    for _ in 0..generations {
        let candidate = incumbent.mutate(&mut rng, SIGMA_THRESHOLD, SIGMA_EF);
        let candidate_fitness = workload.evaluate(candidate);
        if candidate_fitness.composite > incumbent_fitness.composite {
            incumbent = candidate;
            incumbent_fitness = candidate_fitness;
        }
    }
    EsOutcome {
        best_genome: incumbent,
        best_fitness: incumbent_fitness,
        generations,
    }
}

/// Identical algorithm and seed to [`run_unwitnessed`], but every
/// generation (including generation 0, the initial incumbent) is committed
/// to a [`WitnessedLineage`] before the loop continues.
pub fn run_witnessed(
    workload: &Workload,
    generations: usize,
    seed: u64,
) -> (EsOutcome, WitnessedLineage) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut lineage = WitnessedLineage::new();
    let mut incumbent = DEFAULT_GENOME;
    let mut incumbent_fitness = workload.evaluate(incumbent);
    lineage.record(0, incumbent, incumbent_fitness, true);

    for gen in 1..=generations {
        let candidate = incumbent.mutate(&mut rng, SIGMA_THRESHOLD, SIGMA_EF);
        let candidate_fitness = workload.evaluate(candidate);
        let accepted = candidate_fitness.composite > incumbent_fitness.composite;
        lineage.record(gen as u64, candidate, candidate_fitness, accepted);
        if accepted {
            incumbent = candidate;
            incumbent_fitness = candidate_fitness;
        }
    }

    (
        EsOutcome {
            best_genome: incumbent,
            best_fitness: incumbent_fitness,
            generations,
        },
        lineage,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::WorkloadConfig;

    fn tiny_workload() -> Workload {
        Workload::build(&WorkloadConfig {
            n_clusters: 4,
            n_per_cluster: 30,
            dims: 12,
            cluster_std: 0.15,
            m: 10,
            m_longjump: 4,
            n_queries: 20,
            k: 8,
            entry_id: 0,
            data_seed: 0x3333,
            query_seed: 0x4444,
        })
    }

    #[test]
    fn witnessed_and_unwitnessed_reach_identical_optimum() {
        let w = tiny_workload();
        let unwitnessed = run_unwitnessed(&w, 15, 7);
        let (witnessed, lineage) = run_witnessed(&w, 15, 7);
        assert_eq!(unwitnessed.best_genome, witnessed.best_genome);
        assert_eq!(unwitnessed.best_fitness, witnessed.best_fitness);
        assert_eq!(lineage.len(), 16); // generation 0 + 15 mutation attempts
    }

    #[test]
    fn es_never_regresses_below_default_genome() {
        let w = tiny_workload();
        let default_fitness = w.evaluate(DEFAULT_GENOME);
        let outcome = run_unwitnessed(&w, 20, 99);
        assert!(outcome.best_fitness.composite >= default_fitness.composite);
    }

    #[test]
    fn witnessed_lineage_replays_clean() {
        let w = tiny_workload();
        let (_, lineage) = run_witnessed(&w, 15, 7);
        let report = lineage.replay_verify(&w);
        assert!(report.verified, "{report:?}");
    }
}
