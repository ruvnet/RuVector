//! Meta-Harness Darwin applied to SONA's EWC++ — "freeze the model, evolve the
//! harness", where the *frozen model* is the EWC++ continual-learning algorithm
//! ([`EwcPlusPlus`]) and the *evolved harness* is its [`EwcConfig`] genome
//! (lambda schedule, Fisher decay, auto-boundary threshold, learning rate).
//!
//! ## The benchmark (a real plasticity/forgetting frontier)
//!
//! A single weight vector `w` is trained on a *sequence* of tasks, each with its
//! own random target. The learner only ever sees the *current* task's gradient
//! (no replay) and must detect task switches itself (auto boundary detection via
//! `boundary_threshold`). EWC++ projects gradients away from parameters its
//! online Fisher deems important to earlier tasks. This is the canonical
//! continual-learning setup:
//!
//!   * low lambda  → `w` chases the latest task → high plasticity, **forgets**;
//!   * high lambda → `w` frozen near task 0 → **can't learn** later tasks.
//!
//! The score is the average final loss across ALL tasks under one `w` (plus a
//! forgetting penalty) — minimised only by a config that *both* learns and
//! retains.
//!
//! ## Beyond SOTA (precise claim)
//!
//! `EwcConfig::default()` ships hand-tuned "OPTIMIZED" values (lambda 2000, …).
//! Darwin evolves the genome on TRAIN task-sequences and we report the result on
//! HELD-OUT sequences (different seeds). The beyond-SOTA result is: the evolved
//! genome beats the hand-tuned default on *unseen* task sequences — i.e. the
//! metaharness loop out-tunes the crate's hand-tuning, and it generalises.
//!
//! Run: `cargo run -p ruvector-sona --release --example darwin_ewc`

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_sona::darwin_guard::{Guard, Verdict};
use ruvector_sona::{EwcConfig, EwcPlusPlus};

const PARAM_COUNT: usize = 128;
const N_TASKS: usize = 6;
const STEPS_PER_TASK: usize = 80;
const TRAIN_SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];
const TEST_SEEDS: &[u64] = &[101, 102, 103, 104, 105, 106]; // held out

// ── Genome: the EwcConfig harness Darwin evolves (frozen = the EWC++ algorithm) ─
#[derive(Clone, Debug)]
struct Genome {
    lr: f32,
    initial_lambda: f32,
    min_lambda: f32,
    max_lambda: f32,
    fisher_ema_decay: f32,
    boundary_threshold: f32,
    gradient_history_size: usize,
}

impl Genome {
    /// The shipped, hand-tuned SOTA baseline (mirrors `EwcConfig::default()`).
    fn baseline() -> Self {
        let d = EwcConfig::default();
        Self {
            lr: 0.1,
            initial_lambda: d.initial_lambda,
            min_lambda: d.min_lambda,
            max_lambda: d.max_lambda,
            fisher_ema_decay: d.fisher_ema_decay,
            boundary_threshold: d.boundary_threshold,
            gradient_history_size: d.gradient_history_size,
        }
    }

    fn to_config(&self) -> EwcConfig {
        EwcConfig {
            param_count: PARAM_COUNT,
            max_tasks: 10,
            initial_lambda: self.initial_lambda,
            min_lambda: self.min_lambda,
            max_lambda: self.max_lambda,
            fisher_ema_decay: self.fisher_ema_decay,
            boundary_threshold: self.boundary_threshold,
            gradient_history_size: self.gradient_history_size,
        }
    }

    fn clamp(&mut self) {
        self.lr = self.lr.clamp(0.01, 0.5);
        self.initial_lambda = self.initial_lambda.clamp(10.0, 20_000.0);
        self.min_lambda = self.min_lambda.clamp(1.0, 2_000.0);
        self.max_lambda = self.max_lambda.clamp(2_000.0, 50_000.0);
        self.fisher_ema_decay = self.fisher_ema_decay.clamp(0.90, 0.9999);
        self.boundary_threshold = self.boundary_threshold.clamp(0.5, 6.0);
        self.gradient_history_size = self.gradient_history_size.clamp(10, 200);
    }
}

struct Metrics {
    avg_final_loss: f32,
    forgetting: f32,
    plasticity: f32,
}

fn loss(w: &[f32], target: &[f32]) -> f32 {
    0.5 * w
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / w.len() as f32
}

/// Run one task-sequence (seeded) under a genome; return continual-learning metrics.
fn run_sequence(g: &Genome, seq_seed: u64) -> Metrics {
    let p = PARAM_COUNT;
    let mut rng = StdRng::seed_from_u64(seq_seed);
    // Task targets: each task wants `w` near a different random point. Tasks share
    // the parameter space, so learning a later task drags `w` off earlier ones —
    // the forgetting pressure EWC must counter.
    let targets: Vec<Vec<f32>> = (0..N_TASKS)
        .map(|_| (0..p).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();

    let mut ewc = EwcPlusPlus::new(g.to_config());
    let mut w = vec![0.0f32; p];
    let mut loss_just_after = vec![0.0f32; N_TASKS];

    for (t, target) in targets.iter().enumerate() {
        for _ in 0..STEPS_PER_TASK {
            // Gradient of 0.5||w-target||^2 (the only signal — current task only).
            let grad: Vec<f32> = w.iter().zip(target).map(|(wi, ti)| wi - ti).collect();
            // The learner is NOT told task boundaries — it detects them itself.
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
        loss_just_after[t] = loss(&w, target); // plasticity probe (fresh)
    }

    let final_losses: Vec<f32> = targets.iter().map(|tg| loss(&w, tg)).collect();
    let avg_final_loss = final_losses.iter().sum::<f32>() / N_TASKS as f32;
    // Forgetting: how much each task degraded between "just after" and "final".
    let forgetting = final_losses
        .iter()
        .zip(loss_just_after.iter())
        .map(|(f, a)| (f - a).max(0.0))
        .sum::<f32>()
        / N_TASKS as f32;
    let plasticity = loss_just_after.iter().sum::<f32>() / N_TASKS as f32;
    Metrics {
        avg_final_loss,
        forgetting,
        plasticity,
    }
}

/// Mean metrics over a set of seeds.
fn evaluate(g: &Genome, seeds: &[u64]) -> Metrics {
    let mut af = 0.0;
    let mut fg = 0.0;
    let mut pl = 0.0;
    for &s in seeds {
        let m = run_sequence(g, s);
        af += m.avg_final_loss;
        fg += m.forgetting;
        pl += m.plasticity;
    }
    let n = seeds.len() as f32;
    Metrics {
        avg_final_loss: af / n,
        forgetting: fg / n,
        plasticity: pl / n,
    }
}

/// Darwin fitness (higher = better): minimise final loss, penalise forgetting.
fn fitness(g: &Genome, seeds: &[u64]) -> f32 {
    let m = evaluate(g, seeds);
    -(m.avg_final_loss + 0.3 * m.forgetting)
}

fn mutate(g: &Genome, rng: &mut StdRng) -> Genome {
    let mut c = g.clone();
    if rng.gen::<f32>() < 0.5 {
        c.lr *= 0.6 + rng.gen::<f32>() * 0.8;
    }
    if rng.gen::<f32>() < 0.6 {
        c.initial_lambda *= 0.4 + rng.gen::<f32>() * 1.4;
    }
    if rng.gen::<f32>() < 0.4 {
        c.min_lambda *= 0.4 + rng.gen::<f32>() * 1.4;
    }
    if rng.gen::<f32>() < 0.4 {
        c.max_lambda *= 0.6 + rng.gen::<f32>() * 0.9;
    }
    if rng.gen::<f32>() < 0.4 {
        c.fisher_ema_decay += rng.gen::<f32>() * 0.04 - 0.02;
    }
    if rng.gen::<f32>() < 0.5 {
        c.boundary_threshold += rng.gen::<f32>() * 2.0 - 1.0;
    }
    if rng.gen::<f32>() < 0.3 {
        c.gradient_history_size =
            (c.gradient_history_size as i32 + rng.gen_range(-40..40)).max(10) as usize;
    }
    c.clamp();
    c
}

fn main() {
    println!(
        "== SONA · Meta-Harness Darwin over EWC++ (freeze the algorithm, evolve the config) =="
    );
    println!(
        "continual learning: {N_TASKS} tasks × {STEPS_PER_TASK} steps, {PARAM_COUNT} params, auto task-boundary detection"
    );

    let baseline = Genome::baseline();
    let base_train = evaluate(&baseline, TRAIN_SEEDS);
    println!(
        "\nbaseline (hand-tuned default λ={:.0}): train avg_final_loss {:.4} forgetting {:.4} plasticity {:.4}",
        baseline.initial_lambda, base_train.avg_final_loss, base_train.forgetting, base_train.plasticity
    );

    // ── GA: evolve the genome on TRAIN seeds only ───────────────────────────────
    let mut rng = StdRng::seed_from_u64(0xE_C);
    const POP: usize = 24;
    const GEN: usize = 18;
    const ELITE: usize = 6;
    let mut pop: Vec<Genome> = std::iter::once(baseline.clone())
        .chain((0..POP - 1).map(|_| mutate(&baseline, &mut rng)))
        .collect();
    let mut best = (baseline.clone(), fitness(&baseline, TRAIN_SEEDS));

    for gen in 0..GEN {
        // Reward-hacking guard (ADR-271): screen every candidate; non-finite or
        // degenerate (collapsed zero-loss) configs are EXCLUDED from the ranking
        // — not zero-scored — so a hack can neither win nor NaN-panic the sort.
        let guard = Guard::deterministic();
        let mut scored: Vec<(Genome, f32)> = Vec::new();
        let mut rejected = 0usize;
        for g in &pop {
            let f = fitness(g, TRAIN_SEEDS);
            let m = evaluate(g, TRAIN_SEEDS);
            let finite = f.is_finite() && m.avg_final_loss.is_finite() && m.forgetting.is_finite();
            match guard.screen(f, finite, true, m.avg_final_loss <= 0.0) {
                Verdict::Accepted(_) => scored.push((g.clone(), f)),
                Verdict::Rejected(_) => rejected += 1,
            }
        }
        if scored.is_empty() {
            scored.push(best.clone()); // never leave the population empty
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored[0].1 > best.1 {
            best = scored[0].clone();
        }
        let _ = rejected;
        if gen % 3 == 0 || gen == GEN - 1 {
            let m = evaluate(&scored[0].0, TRAIN_SEEDS);
            println!(
                "gen {gen:2}: train avg_final_loss {:.4} forgetting {:.4} (λ0={:.0} lr={:.3} bθ={:.2})  fitness {:.4}",
                m.avg_final_loss, m.forgetting, scored[0].0.initial_lambda, scored[0].0.lr, scored[0].0.boundary_threshold, scored[0].1
            );
        }
        let elites: Vec<Genome> = scored.iter().take(ELITE).map(|(g, _)| g.clone()).collect();
        let mut next = elites.clone();
        while next.len() < POP {
            let parent = &elites[rng.gen_range(0..elites.len())];
            next.push(mutate(parent, &mut rng));
        }
        pop = next;
    }

    // ── Coordinate-descent polish (deterministic, reproducible optimum) ─────────
    let evolved = polish(best.0.clone());

    // ── Report on HELD-OUT test seeds (never seen during evolution) ─────────────
    let base_test = evaluate(&baseline, TEST_SEEDS);
    let evo_test = evaluate(&evolved, TEST_SEEDS);
    let evo_train = evaluate(&evolved, TRAIN_SEEDS);

    println!("\n-- evolved genome --");
    println!(
        "  λ0={:.0} λmin={:.0} λmax={:.0} decay={:.4} bθ={:.2} hist={} lr={:.3}",
        evolved.initial_lambda,
        evolved.min_lambda,
        evolved.max_lambda,
        evolved.fisher_ema_decay,
        evolved.boundary_threshold,
        evolved.gradient_history_size,
        evolved.lr
    );
    println!(
        "  train: avg_final_loss {:.4} (baseline {:.4})",
        evo_train.avg_final_loss, base_train.avg_final_loss
    );

    println!("\n-- HELD-OUT test sequences (the beyond-SOTA result) --");
    println!(
        "  baseline: avg_final_loss {:.4} forgetting {:.4} plasticity {:.4}",
        base_test.avg_final_loss, base_test.forgetting, base_test.plasticity
    );
    println!(
        "  evolved : avg_final_loss {:.4} forgetting {:.4} plasticity {:.4}",
        evo_test.avg_final_loss, evo_test.forgetting, evo_test.plasticity
    );
    let gain =
        (base_test.avg_final_loss - evo_test.avg_final_loss) / base_test.avg_final_loss * 100.0;
    let fgain =
        (base_test.forgetting - evo_test.forgetting) / base_test.forgetting.max(1e-9) * 100.0;
    println!(
        "  → {:.1}% lower final loss, {:.1}% less forgetting on UNSEEN task sequences",
        gain, fgain
    );
    println!(
        "{}",
        if evo_test.avg_final_loss < base_test.avg_final_loss {
            "BEYOND SOTA — metaharness-evolved EWC++ config beats the hand-tuned default on held-out tasks"
        } else {
            "no improvement over baseline this run"
        }
    );
}

/// Greedy per-gene coordinate descent over a candidate grid (cross-seed mean on
/// TRAIN) — converts the GA's broad search into a reproducible optimum.
fn polish(seed: Genome) -> Genome {
    let mut cur = seed;
    let mut cur_f = fitness(&cur, TRAIN_SEEDS);
    let lambdas: [f32; 8] = [50.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 15000.0];
    let lrs: [f32; 6] = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3];
    let bthr: [f32; 6] = [0.8, 1.5, 2.0, 3.0, 4.0, 5.0];
    let decays: [f32; 4] = [0.95, 0.99, 0.999, 0.9999];
    for _ in 0..3 {
        let mut improved = false;
        macro_rules! try_gene {
            ($field:ident, $cands:expr) => {
                for &v in $cands.iter() {
                    if (cur.$field - v).abs() < f32::EPSILON {
                        continue;
                    }
                    let mut cand = cur.clone();
                    cand.$field = v;
                    cand.clamp();
                    let f = fitness(&cand, TRAIN_SEEDS);
                    if f > cur_f + 1e-9 {
                        cur = cand;
                        cur_f = f;
                        improved = true;
                    }
                }
            };
        }
        try_gene!(initial_lambda, lambdas);
        try_gene!(lr, lrs);
        try_gene!(boundary_threshold, bthr);
        try_gene!(fisher_ema_decay, decays);
        if !improved {
            break;
        }
    }
    cur
}
