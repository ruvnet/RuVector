//! # ruvector-darwin
//!
//! A bounded, deterministic evolutionary parameter-search engine.
//!
//! RuVector's nightly research has repeatedly hand-set scalar hyperparameters
//! (e.g. `ruvector-gnn-rerank`'s diffusion `alpha`, `ruvector-coherence-hnsw`'s
//! `adaptation_rate`, `ruvector-memory-admission`'s `tau`) and then explicitly
//! flagged "adaptive"/"evolved" tuning of that constant as unimplemented future
//! work (2026-05-21, 2026-06-16, 2026-09-16 nightlies). No evolutionary or
//! "Darwin" search primitive existed anywhere in the repository to do this with
//! (confirmed by grep across `docs/research/nightly` and `crates/`). This crate
//! is that primitive: a small, generic, elitist (1+λ) evolution strategy over a
//! bounded real-valued genome, with a hard, caller-specified budget
//! (`generations × candidates_per_generation`, `max_promotions`) so it can
//! never "freely rewrite" a search space — it only ever compares a fixed number
//! of mutations of the current best against the caller's own fitness function.
//!
//! ## What this is not
//!
//! This is not a general NAS/AutoML framework, not LLM-guided, and not a
//! replacement for `npx metaharness`'s project-scaffolding "Darwin Mode" (a
//! different, unrelated thing with the same name — see the nightly README for
//! the disambiguation). It is a plain numerical optimizer: Gaussian mutation,
//! elitist selection, deterministic seeded RNG, no gradient information, no
//! surrogate model. That simplicity is deliberate — it is the smallest thing
//! that can honestly be called a bounded evolutionary search and audited end
//! to end.
//!
//! ## Hard constraints (enforced, not advisory)
//!
//! - A candidate whose fitness function returns `None` (caller-signalled
//!   invalid/non-finite result) is recorded as rejected and can never be
//!   promoted.
//! - A mutated gene is always clamped back into its declared `[min, max]`
//!   bound before evaluation — a candidate is never evaluated out of bounds.
//! - At most one genome is ever promoted (`EvolutionConfig::max_promotions`
//!   is accepted for interface symmetry with the nightly harness's
//!   vocabulary, but this engine's selection rule is inherently single-winner
//!   per run: the overall best-of-all-generations candidate, if and only if
//!   it strictly beats the parent).
//! - If no evaluated candidate strictly beats the parent's fitness, the
//!   parent is retained and `EvolutionReport::promoted` is `None`. That is a
//!   correct, successful run, not a failure.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// One bounded, real-valued gene: `value` must stay within `[min, max]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bound {
    pub min: f64,
    pub max: f64,
}

impl Bound {
    pub fn new(min: f64, max: f64) -> Self {
        assert!(min <= max, "Bound::new: min ({min}) must be <= max ({max})");
        Self { min, max }
    }

    fn clamp(&self, v: f64) -> f64 {
        v.clamp(self.min, self.max)
    }

    fn range(&self) -> f64 {
        self.max - self.min
    }
}

/// A point in the search space: one real value per declared [`Bound`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub genes: Vec<f64>,
}

impl Genome {
    pub fn new(genes: Vec<f64>) -> Self {
        Self { genes }
    }
}

/// Budget and mutation parameters for a bounded evolutionary run.
///
/// Defaults mirror the nightly harness's own stated default evolutionary
/// budget: 3 generations, 4 candidates per generation, at most 1 promotion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub generations: usize,
    pub candidates_per_generation: usize,
    pub max_promotions: usize,
    /// Mutation step size, expressed as a fraction of each gene's
    /// `max - min` range (e.g. `0.2` mutates by up to ~20% of the gene's
    /// range per step, on average).
    pub mutation_sigma_frac: f64,
    pub seed: u64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            generations: 3,
            candidates_per_generation: 4,
            max_promotions: 1,
            mutation_sigma_frac: 0.20,
            seed: 0,
        }
    }
}

/// Outcome of evaluating one genome: a finite fitness, or a rejection reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Evaluation {
    Fitness(f64),
    Rejected(String),
}

/// One evaluated candidate, retained in the lineage regardless of outcome —
/// rejected and non-winning candidates are evidence, not discarded noise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub genome: Genome,
    pub evaluation: Evaluation,
}

/// All candidates evaluated within one generation, plus which (if any)
/// became the new running best.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub index: usize,
    pub candidates: Vec<CandidateRecord>,
    /// Index into `candidates` that became the new running best, if any
    /// candidate this generation strictly beat the running best going in.
    pub improved_at: Option<usize>,
    /// Running best fitness after this generation (unchanged from the
    /// previous generation if `improved_at` is `None`).
    pub running_best_fitness: f64,
}

/// Full, auditable record of one bounded evolutionary run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionReport {
    pub config: EvolutionConfig,
    pub bounds: Vec<Bound>,
    pub parent: CandidateRecord,
    pub generations: Vec<GenerationRecord>,
    /// `Some(record)` iff some evaluated candidate strictly beat the
    /// parent's fitness; that candidate, not necessarily the last
    /// generation's best, is always the overall best found. `None` means
    /// the parent is retained — a legitimate, successful outcome.
    pub promoted: Option<CandidateRecord>,
}

impl EvolutionReport {
    /// `true` iff the search found and promoted a genome strictly better
    /// than the parent. Mirrors the nightly harness's `beats_parent` gate.
    pub fn beats_parent(&self) -> bool {
        self.promoted.is_some()
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("EvolutionReport is always serializable")
    }
}

fn fitness_value(e: &Evaluation) -> Option<f64> {
    match e {
        Evaluation::Fitness(f) if f.is_finite() => Some(*f),
        _ => None,
    }
}

/// Mutate `base` into a new, in-bounds [`Genome`] using Gaussian perturbation
/// per gene, scaled by `config.mutation_sigma_frac * bound.range()`.
fn mutate(base: &Genome, bounds: &[Bound], config: &EvolutionConfig, rng: &mut StdRng) -> Genome {
    let genes = base
        .genes
        .iter()
        .zip(bounds)
        .map(|(&v, b)| {
            let sigma = (config.mutation_sigma_frac * b.range()).max(1e-12);
            // Box-Muller via two uniform draws — avoids pulling in `rand_distr`
            // for a single Gaussian sample per gene.
            let u1: f64 = rng.gen_range(1e-12..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            b.clamp(v + z * sigma)
        })
        .collect();
    Genome::new(genes)
}

/// Run a bounded, elitist evolutionary search.
///
/// `parent` is the caller's existing hand-set baseline — the thing a
/// candidate must beat, not a straw man. `bounds.len()` must equal
/// `parent.genes.len()`. `fitness_fn` is evaluated at most
/// `1 + config.generations * config.candidates_per_generation` times, total,
/// over the whole run (the parent once, then every generation's children) —
/// evaluation count is `O(search budget)`, never a function of search
/// outcome, so a fitness function cannot cause this to run away.
///
/// # Panics
/// Panics if `parent.genes.len() != bounds.len()`, or if `config.generations
/// == 0` or `config.candidates_per_generation == 0` (a zero-budget search is
/// a caller bug, not a valid "no-op" run).
pub fn run_evolution<F>(
    parent: Genome,
    bounds: Vec<Bound>,
    config: EvolutionConfig,
    mut fitness_fn: F,
) -> EvolutionReport
where
    F: FnMut(&Genome) -> Evaluation,
{
    assert_eq!(
        parent.genes.len(),
        bounds.len(),
        "ruvector_darwin::run_evolution: parent has {} genes but {} bounds were given",
        parent.genes.len(),
        bounds.len()
    );
    assert!(
        config.generations > 0 && config.candidates_per_generation > 0,
        "ruvector_darwin::run_evolution: generations and candidates_per_generation must be > 0"
    );

    let mut rng = StdRng::seed_from_u64(config.seed);

    let parent_eval = fitness_fn(&parent);
    let parent_record = CandidateRecord {
        genome: parent.clone(),
        evaluation: parent_eval.clone(),
    };
    let mut running_best_genome = parent.clone();
    let mut running_best_fitness = fitness_value(&parent_eval).unwrap_or(f64::NEG_INFINITY);
    // Tracks the single best-of-all-generations candidate for promotion,
    // separately from the per-generation "running best used as the next
    // generation's mutation source" — they coincide under elitist selection,
    // kept as two names for clarity at the call site below.
    let mut best_overall: Option<CandidateRecord> = None;

    let mut generations = Vec::with_capacity(config.generations);

    for gen_idx in 0..config.generations {
        let mut candidates = Vec::with_capacity(config.candidates_per_generation);
        let mut improved_at = None;

        for cand_idx in 0..config.candidates_per_generation {
            let genome = mutate(&running_best_genome, &bounds, &config, &mut rng);
            let eval = fitness_fn(&genome);
            if let Some(f) = fitness_value(&eval) {
                if f > running_best_fitness {
                    running_best_fitness = f;
                    running_best_genome = genome.clone();
                    improved_at = Some(cand_idx);
                    best_overall = Some(CandidateRecord {
                        genome: genome.clone(),
                        evaluation: eval.clone(),
                    });
                }
            }
            candidates.push(CandidateRecord {
                genome,
                evaluation: eval,
            });
        }

        generations.push(GenerationRecord {
            index: gen_idx,
            candidates,
            improved_at,
            running_best_fitness,
        });
    }

    // Respect `max_promotions`: this engine only ever has one winner
    // candidate to offer, so `max_promotions == 0` means "do not promote
    // even if one was found" (a caller opting out of promotion entirely,
    // e.g. to inspect evidence only), and `>= 1` promotes the single winner.
    let promoted = if config.max_promotions == 0 {
        None
    } else {
        best_overall
    };

    EvolutionReport {
        config,
        bounds,
        parent: parent_record,
        generations,
        promoted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_fitness(target: &[f64]) -> impl FnMut(&Genome) -> Evaluation + '_ {
        move |g: &Genome| {
            let neg_sq_err: f64 = g
                .genes
                .iter()
                .zip(target)
                .map(|(&v, &t)| -(v - t).powi(2))
                .sum();
            Evaluation::Fitness(neg_sq_err)
        }
    }

    #[test]
    fn deterministic_given_same_seed() {
        let parent = Genome::new(vec![0.0, 0.0]);
        let bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
        let cfg = EvolutionConfig {
            seed: 7,
            ..Default::default()
        };
        let r1 = run_evolution(
            parent.clone(),
            bounds.clone(),
            cfg.clone(),
            sphere_fitness(&[0.5, -0.5]),
        );
        let r2 = run_evolution(parent, bounds, cfg, sphere_fitness(&[0.5, -0.5]));
        assert_eq!(r1.to_json_pretty(), r2.to_json_pretty());
    }

    #[test]
    fn different_seeds_can_diverge() {
        let parent = Genome::new(vec![0.0, 0.0]);
        let bounds = vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)];
        let cfg_a = EvolutionConfig {
            seed: 1,
            ..Default::default()
        };
        let cfg_b = EvolutionConfig {
            seed: 2,
            ..Default::default()
        };
        let r1 = run_evolution(
            parent.clone(),
            bounds.clone(),
            cfg_a,
            sphere_fitness(&[0.9, 0.9]),
        );
        let r2 = run_evolution(parent, bounds, cfg_b, sphere_fitness(&[0.9, 0.9]));
        // Not asserting inequality (two seeds could coincidentally agree),
        // only that both independently found an improvement over the
        // deliberately-bad parent at the origin.
        assert!(r1.beats_parent());
        assert!(r2.beats_parent());
    }

    #[test]
    fn running_best_fitness_never_decreases_across_generations() {
        let parent = Genome::new(vec![0.0]);
        let bounds = vec![Bound::new(-5.0, 5.0)];
        let cfg = EvolutionConfig {
            seed: 42,
            generations: 5,
            candidates_per_generation: 6,
            ..Default::default()
        };
        let report = run_evolution(parent, bounds, cfg, sphere_fitness(&[3.0]));
        let mut prev = f64::NEG_INFINITY;
        for g in &report.generations {
            assert!(
                g.running_best_fitness >= prev,
                "running best fitness decreased: {prev} -> {}",
                g.running_best_fitness
            );
            prev = g.running_best_fitness;
        }
    }

    #[test]
    fn mutations_always_respect_bounds() {
        let parent = Genome::new(vec![0.0]);
        let bounds = vec![Bound::new(-0.1, 0.1)];
        // Huge mutation sigma to try to force out-of-bounds values if clamping
        // were broken.
        let cfg = EvolutionConfig {
            seed: 3,
            generations: 4,
            candidates_per_generation: 8,
            mutation_sigma_frac: 50.0,
            ..Default::default()
        };
        let report = run_evolution(parent, bounds.clone(), cfg, sphere_fitness(&[0.0]));
        for gen in &report.generations {
            for c in &gen.candidates {
                for (&v, b) in c.genome.genes.iter().zip(&bounds) {
                    assert!(
                        v >= b.min && v <= b.max,
                        "gene {v} escaped bound [{}, {}]",
                        b.min,
                        b.max
                    );
                }
            }
        }
    }

    #[test]
    fn rejected_candidates_never_promoted() {
        let parent = Genome::new(vec![0.0]);
        let bounds = vec![Bound::new(-1.0, 1.0)];
        let cfg = EvolutionConfig {
            seed: 9,
            ..Default::default()
        };
        // Every child is rejected outright; parent itself is finite (0.0 fitness).
        let report = run_evolution(parent, bounds, cfg, |g: &Genome| {
            if g.genes == vec![0.0] {
                Evaluation::Fitness(0.0)
            } else {
                Evaluation::Rejected("synthetic rejection for test".into())
            }
        });
        assert!(!report.beats_parent(), "no non-parent genome should ever equal exactly [0.0] after mutation with sigma>0, so nothing should be promotable");
    }

    #[test]
    fn parent_retained_when_nothing_improves() {
        let parent = Genome::new(vec![1.0]);
        let bounds = vec![Bound::new(-1.0, 1.0)];
        let cfg = EvolutionConfig {
            seed: 5,
            ..Default::default()
        };
        // Fitness is constant everywhere: nothing can strictly beat the parent.
        let report = run_evolution(parent.clone(), bounds, cfg, |_: &Genome| {
            Evaluation::Fitness(0.0)
        });
        assert!(!report.beats_parent());
        assert_eq!(report.parent.genome, parent);
    }

    #[test]
    fn max_promotions_zero_suppresses_promotion() {
        let parent = Genome::new(vec![0.0]);
        let bounds = vec![Bound::new(-5.0, 5.0)];
        let cfg = EvolutionConfig {
            seed: 11,
            max_promotions: 0,
            ..Default::default()
        };
        let report = run_evolution(parent, bounds, cfg, sphere_fitness(&[2.0]));
        assert!(report.promoted.is_none());
    }

    #[test]
    #[should_panic(expected = "generations and candidates_per_generation must be > 0")]
    fn zero_budget_panics() {
        let parent = Genome::new(vec![0.0]);
        let bounds = vec![Bound::new(-1.0, 1.0)];
        let cfg = EvolutionConfig {
            generations: 0,
            ..Default::default()
        };
        run_evolution(parent, bounds, cfg, sphere_fitness(&[0.0]));
    }

    #[test]
    #[should_panic(expected = "parent has 2 genes but 1 bounds")]
    fn mismatched_bounds_panics() {
        let parent = Genome::new(vec![0.0, 0.0]);
        let bounds = vec![Bound::new(-1.0, 1.0)];
        run_evolution(
            parent,
            bounds,
            EvolutionConfig::default(),
            sphere_fitness(&[0.0, 0.0]),
        );
    }
}
