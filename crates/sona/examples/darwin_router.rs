//! Per-task-category genome router (ADR-271, Ornith-1.0 borrow #2).
//!
//! Ornith-1.0's main empirical result is that **per-task-category strategies
//! emerge** — no single scaffold is optimal across workload types. The metaharness
//! analogue: instead of evolving ONE global `EwcConfig`, evolve a **router**
//! `task-class → genome`, so each workload class gets its own specialized config.
//!
//! Two continual-learning workload classes with *conflicting* optima:
//!   * `STABLE`   — few tasks, long training, small shifts → wants to *learn*
//!                  (low lambda; aggressive EWC protection only wastes plasticity).
//!   * `VOLATILE` — many tasks, short training, large shifts → wants to *retain*
//!                  (high lambda; without it, later tasks erase earlier ones).
//!
//! Baseline = the single best global genome (the PR-#615 approach, evolved over
//! both classes pooled). Router = the best genome PER class. We report both on
//! HELD-OUT sequences. Beyond-SOTA result: the router beats the single global
//! genome on unseen sequences — because one config cannot serve conflicting
//! workloads. Selection is screened by `darwin_guard`.
//!
//! Run: `cargo run -p ruvector-sona --release --example darwin_router`

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_sona::darwin_guard::{Guard, Verdict};
use ruvector_sona::{EwcConfig, EwcPlusPlus};

const PARAM_COUNT: usize = 96;
// Enough per-class data that specialization generalizes (Ornith's regime: many
// per-category samples) rather than overfitting a handful of sequences.
const TRAIN_SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const TEST_SEEDS: &[u64] = &[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]; // held out

/// A workload class (task-category).
#[derive(Clone, Copy)]
struct Class {
    name: &'static str,
    n_tasks: usize,
    steps: usize,
    shift: f32, // inter-task target magnitude (forgetting pressure)
}
const STABLE: Class = Class {
    name: "STABLE",
    n_tasks: 3,
    steps: 120,
    shift: 0.25,
};
const VOLATILE: Class = Class {
    name: "VOLATILE",
    n_tasks: 9,
    steps: 35,
    shift: 1.2,
};
const CLASSES: [Class; 2] = [STABLE, VOLATILE];

#[derive(Clone, Debug)]
struct Genome {
    lr: f32,
    initial_lambda: f32,
    max_lambda: f32,
    fisher_ema_decay: f32,
    boundary_threshold: f32,
}
impl Genome {
    fn baseline() -> Self {
        let d = EwcConfig::default();
        Self {
            lr: 0.1,
            initial_lambda: d.initial_lambda,
            max_lambda: d.max_lambda,
            fisher_ema_decay: d.fisher_ema_decay,
            boundary_threshold: d.boundary_threshold,
        }
    }
    fn to_config(&self) -> EwcConfig {
        EwcConfig {
            param_count: PARAM_COUNT,
            max_tasks: 12,
            initial_lambda: self.initial_lambda,
            min_lambda: 50.0,
            max_lambda: self.max_lambda,
            fisher_ema_decay: self.fisher_ema_decay,
            boundary_threshold: self.boundary_threshold,
            gradient_history_size: 60,
        }
    }
    fn clamp(&mut self) {
        self.lr = self.lr.clamp(0.01, 0.5);
        self.initial_lambda = self.initial_lambda.clamp(10.0, 20_000.0);
        self.max_lambda = self.max_lambda.clamp(2_000.0, 50_000.0);
        self.fisher_ema_decay = self.fisher_ema_decay.clamp(0.90, 0.9999);
        self.boundary_threshold = self.boundary_threshold.clamp(0.5, 6.0);
    }
}

fn run_sequence(class: &Class, g: &Genome, seed: u64) -> f32 {
    let p = PARAM_COUNT;
    let mut rng = StdRng::seed_from_u64(seed ^ (class.n_tasks as u64).wrapping_mul(0x9E37));
    let targets: Vec<Vec<f32>> = (0..class.n_tasks)
        .map(|_| {
            (0..p)
                .map(|_| class.shift * rng.gen_range(-1.0f32..1.0))
                .collect()
        })
        .collect();
    let mut ewc = EwcPlusPlus::new(g.to_config());
    let mut w = vec![0.0f32; p];
    for target in &targets {
        for _ in 0..class.steps {
            let grad: Vec<f32> = w.iter().zip(target).map(|(wi, ti)| wi - ti).collect();
            if ewc.detect_task_boundary(&grad) {
                ewc.start_new_task();
            }
            let gc = ewc.apply_constraints(&grad);
            for (wi, gi) in w.iter_mut().zip(gc.iter()) {
                *wi -= g.lr * gi;
            }
            ewc.update_fisher(&grad);
            ewc.set_optimal_weights(&w);
        }
    }
    targets
        .iter()
        .map(|tg| {
            0.5 * w
                .iter()
                .zip(tg)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / p as f32
        })
        .sum::<f32>()
        / class.n_tasks as f32
}

fn eval_class(class: &Class, g: &Genome, seeds: &[u64]) -> f32 {
    seeds
        .iter()
        .map(|&s| run_sequence(class, g, s))
        .sum::<f32>()
        / seeds.len() as f32
}

fn mutate(g: &Genome, rng: &mut StdRng) -> Genome {
    let mut c = g.clone();
    if rng.gen::<f32>() < 0.5 {
        c.lr *= 0.6 + rng.gen::<f32>() * 0.8;
    }
    if rng.gen::<f32>() < 0.6 {
        c.initial_lambda *= 0.35 + rng.gen::<f32>() * 1.5;
    }
    if rng.gen::<f32>() < 0.5 {
        c.max_lambda *= 0.6 + rng.gen::<f32>() * 0.9;
    }
    if rng.gen::<f32>() < 0.4 {
        c.fisher_ema_decay += rng.gen::<f32>() * 0.04 - 0.02;
    }
    if rng.gen::<f32>() < 0.5 {
        c.boundary_threshold += rng.gen::<f32>() * 2.0 - 1.0;
    }
    c.clamp();
    c
}

/// Evolve a genome minimising `fit` (a loss closure), guard-screened.
fn evolve(seed: u64, fit: &dyn Fn(&Genome) -> f32) -> Genome {
    let guard = Guard::deterministic();
    let mut rng = StdRng::seed_from_u64(seed);
    let base = Genome::baseline();
    const POP: usize = 18;
    const GEN: usize = 14;
    let mut pop: Vec<Genome> = std::iter::once(base.clone())
        .chain((0..POP - 1).map(|_| mutate(&base, &mut rng)))
        .collect();
    let mut best = (base.clone(), fit(&base));
    for _ in 0..GEN {
        let mut scored: Vec<(Genome, f32)> = Vec::new();
        for g in &pop {
            let loss = fit(g);
            // Guard: exclude non-finite / degenerate (negative loss is impossible here).
            match guard.screen(-loss, loss.is_finite(), true, loss < 0.0) {
                Verdict::Accepted(_) => scored.push((g.clone(), loss)),
                Verdict::Rejected(_) => {}
            }
        }
        if scored.is_empty() {
            scored.push(best.clone());
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored[0].1 < best.1 {
            best = scored[0].clone();
        }
        let elites: Vec<Genome> = scored.iter().take(5).map(|(g, _)| g.clone()).collect();
        let mut next = elites.clone();
        while next.len() < POP {
            next.push(mutate(&elites[rng.gen_range(0..elites.len())], &mut rng));
        }
        pop = next;
    }
    best.0
}

fn main() {
    println!("== SONA · per-task-category genome ROUTER (ADR-271, Ornith borrow) ==");
    let base = Genome::baseline();
    for c in &CLASSES {
        println!(
            "class {:8} baseline held-out loss {:.4}",
            c.name,
            eval_class(c, &base, TEST_SEEDS)
        );
    }

    // Single GLOBAL genome (PR #615 approach): one config over BOTH classes pooled.
    let global = evolve(0xA10BA1, &|g| {
        CLASSES
            .iter()
            .map(|c| eval_class(c, g, TRAIN_SEEDS))
            .sum::<f32>()
            / CLASSES.len() as f32
    });

    // ROUTER: one specialized genome PER class.
    let router: Vec<(Class, Genome)> = CLASSES
        .iter()
        .map(|c| {
            (
                *c,
                evolve(0x12345 ^ c.n_tasks as u64, &|g| {
                    eval_class(c, g, TRAIN_SEEDS)
                }),
            )
        })
        .collect();

    // Held-out comparison.
    let global_loss = CLASSES
        .iter()
        .map(|c| eval_class(c, &global, TEST_SEEDS))
        .sum::<f32>()
        / CLASSES.len() as f32;
    let router_loss = router
        .iter()
        .map(|(c, g)| eval_class(c, g, TEST_SEEDS))
        .sum::<f32>()
        / router.len() as f32;

    println!("\n-- per-class evolved configs --");
    println!(
        "  GLOBAL (one config): λ0={:.0} lr={:.3} bθ={:.2}",
        global.initial_lambda, global.lr, global.boundary_threshold
    );
    for (c, g) in &router {
        println!(
            "  {:8} (router) : λ0={:.0} lr={:.3} bθ={:.2}  held-out {:.4}",
            c.name,
            g.initial_lambda,
            g.lr,
            g.boundary_threshold,
            eval_class(c, g, TEST_SEEDS)
        );
    }

    println!("\n-- HELD-OUT (mean over classes) --");
    println!("  single global genome (PR #615): {global_loss:.4}");
    println!("  per-category router            : {router_loss:.4}");
    let gain = (global_loss - router_loss) / global_loss * 100.0;
    println!(
        "  → router is {gain:.1}% better on held-out (specialization beats one-config-fits-all)"
    );
    println!(
        "{}",
        if router_loss < global_loss - 1e-4 {
            "BEYOND SOTA — per-task-category routing beats the single best global config on unseen sequences"
        } else {
            "no improvement over the single global genome this run"
        }
    );
    println!(
        "  (caveat: the gain is the value of specialization — modest here, and it REVERSES\n   \
         when per-class data is scarce: a specialized config then overfits while the pooled\n   \
         global generalizes. Per-category routing needs enough per-category samples — Ornith's regime.)"
    );
    assert!(router_loss.is_finite() && global_loss.is_finite());
}
