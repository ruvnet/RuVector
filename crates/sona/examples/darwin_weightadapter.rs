//! The `weightAdapter` gene — Darwin selects a fine-tuned LoRA adapter the way it
//! selects any other gene, and *empirically prunes* adapters that regress.
//!
//! Premise (the "autonomous data engine"): a fine-tune produces a candidate
//! adapter delta `Δw` (e.g. a LoRA distilled from verified SWE-bench trajectories).
//! Instead of *assuming* the new weights are better, expose the adapter as a gene
//! `(which_adapter, alpha)` and let evolutionary selection decide whether — and
//! how much — to apply it (`w_eff = w_base + alpha·Δw`), scored by held-out task
//! performance. A fine-tune that overfits gets pruned by selection.
//!
//! ## The subtlety this demonstrates (and why it matters)
//!
//! "Selection prunes overfit adapters" is TRUE — **but only if the fitness is
//! evaluated per-domain.** With a single *aggregate* fitness, an adapter whose
//! in-distribution gain outweighs its out-of-distribution loss is *selected* and
//! silently regresses the out-of-dist domain. Only a **per-domain / no-regression
//! (Pareto)** rule — "must not regress ANY repository" — actually prunes it.
//!
//! Two candidate adapters, distilled from TRAIN sequences:
//!   * `general`  — captures the signal common to both domains → helps both.
//!   * `overfit`  — captures domain-A-specific structure → helps A, hurts B.
//!
//! We then pick the best `(adapter, alpha)` under each selection rule on HELD-OUT
//! sequences and show: aggregate accepts `overfit` (and regresses B); per-domain
//! prunes it and keeps `general`.
//!
//! Run: `cargo run -p ruvector-sona --release --example darwin_weightadapter`

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 64;
const TRAIN_SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];
// Held-out eval is IMBALANCED toward in-dist (A) — the realistic case: the eval
// pool is dominated by the same repos the adapter was fine-tuned on. A naive
// (volume-weighted) aggregate fitness is therefore easy to fool.
const TEST_A: &[u64] = &[101, 102, 103, 104, 105, 106]; // in-dist (majority)
const TEST_B: &[u64] = &[201, 202]; // out-of-dist (minority)
const NA: f32 = 6.0;
const NB: f32 = 2.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Adapter {
    None,
    General,
    Overfit,
}

/// Fixed structure of the two domains: a shared signal + per-domain offsets.
struct World {
    mu_common: Vec<f32>,
    delta_a: Vec<f32>, // domain-A-specific direction
    delta_b: Vec<f32>, // domain-B-specific direction
}

impl World {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let rv = |rng: &mut StdRng| {
            (0..DIM)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>()
        };
        let delta_a = rv(&mut rng);
        // Domain B's structure OPPOSES domain A's — a small shared signal plus
        // conflicting domain-specific directions, so an A-overfit adapter that
        // captures `delta_a` actively hurts B (the realistic "fine-tune helped
        // some repos, regressed others" case).
        let delta_b: Vec<f32> = delta_a.iter().map(|x| -x).collect();
        let mu_common: Vec<f32> = rv(&mut rng).iter().map(|x| 0.3 * x).collect(); // weak shared signal
        Self {
            mu_common,
            delta_a,
            delta_b,
        }
    }

    /// A target for domain `a_side` (true = domain A / in-dist, false = B / out-dist).
    fn target(&self, a_side: bool, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed ^ if a_side { 0xA } else { 0xB });
        let d = if a_side { &self.delta_a } else { &self.delta_b };
        (0..DIM)
            .map(|i| self.mu_common[i] + d[i] + 0.05 * rng.gen_range(-1.0f32..1.0))
            .collect()
    }
}

fn loss(w: &[f32], target: &[f32]) -> f32 {
    w.iter()
        .zip(target)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / w.len() as f32
}

/// Distil an adapter delta from TRAIN targets (mean target = the "fine-tune").
/// `general` averages both domains (shared signal); `overfit` uses domain A only.
fn distil(world: &World, adapter: Adapter) -> Vec<f32> {
    match adapter {
        Adapter::None => vec![0.0; DIM],
        Adapter::Overfit => mean_target(world, &[true], TRAIN_SEEDS),
        Adapter::General => mean_target(world, &[true, false], TRAIN_SEEDS),
    }
}

fn mean_target(world: &World, sides: &[bool], seeds: &[u64]) -> Vec<f32> {
    let mut acc = vec![0.0f32; DIM];
    let mut n = 0.0;
    for &side in sides {
        for &s in seeds {
            let t = world.target(side, s);
            for (a, ti) in acc.iter_mut().zip(t.iter()) {
                *a += ti;
            }
            n += 1.0;
        }
    }
    acc.iter().map(|x| x / n).collect()
}

/// Mean task loss on a domain (seeds) with `w_eff = alpha·Δw` (base w = 0: the
/// adapter supplies all signal — what matters here is *relative* improvement).
fn domain_loss(world: &World, a_side: bool, delta: &[f32], alpha: f32, seeds: &[u64]) -> f32 {
    let w: Vec<f32> = delta.iter().map(|d| alpha * d).collect();
    seeds
        .iter()
        .map(|&s| loss(&w, &world.target(a_side, s)))
        .sum::<f32>()
        / seeds.len() as f32
}

fn main() {
    let world = World::new();
    let alphas: Vec<f32> = (0..=20).map(|i| i as f32 / 20.0).collect();
    let candidates = [Adapter::None, Adapter::General, Adapter::Overfit];

    // Base (no adapter) reference loss per domain, on held-out seeds.
    let base_a = domain_loss(&world, true, &vec![0.0; DIM], 0.0, TEST_A);
    let base_b = domain_loss(&world, false, &vec![0.0; DIM], 0.0, TEST_B);

    println!("== SONA · the `weightAdapter` gene — Darwin selects/prunes a fine-tuned adapter ==");
    println!("two domains (A = in-dist, B = out-dist); base (no adapter) held-out loss: A {base_a:.4}  B {base_b:.4}\n");

    // Evaluate every (adapter, alpha) on held-out seeds → improvement per domain.
    struct Cand {
        adapter: Adapter,
        alpha: f32,
        gain_a: f32,
        gain_b: f32,
    }
    let mut grid = Vec::new();
    for &adapter in &candidates {
        let delta = distil(&world, adapter);
        for &alpha in &alphas {
            let la = domain_loss(&world, true, &delta, alpha, TEST_A);
            let lb = domain_loss(&world, false, &delta, alpha, TEST_B);
            grid.push(Cand {
                adapter,
                alpha,
                gain_a: base_a - la, // >0 = improves domain A
                gain_b: base_b - lb, // >0 = improves domain B
            });
        }
    }

    // Volume-weighted (pooled) aggregate fitness — in-dist dominates the eval.
    let pooled = |c: &Cand| (NA * c.gain_a + NB * c.gain_b) / (NA + NB);

    // Per-adapter best (illustrate the overfit asymmetry).
    println!("-- best alpha per adapter (held-out) --");
    for &adapter in &candidates {
        if adapter == Adapter::None {
            continue;
        }
        let best = grid
            .iter()
            .filter(|c| c.adapter == adapter)
            .max_by(|x, y| pooled(x).partial_cmp(&pooled(y)).unwrap())
            .unwrap();
        println!(
            "  {:8?}: α={:.2}  ΔA {:+.4}  ΔB {:+.4}",
            adapter, best.alpha, best.gain_a, best.gain_b
        );
    }

    // ── Selection rule 1: AGGREGATE fitness (mean gain) ─────────────────────────
    let agg = grid
        .iter()
        .max_by(|x, y| pooled(x).partial_cmp(&pooled(y)).unwrap())
        .unwrap();
    // ── Selection rule 2: PER-DOMAIN no-regression (Pareto) ─────────────────────
    // Accept only candidates that do NOT regress either domain, then maximise gain.
    let perdomain = grid
        .iter()
        .filter(|c| c.gain_a >= -1e-4 && c.gain_b >= -1e-4)
        .max_by(|x, y| pooled(x).partial_cmp(&pooled(y)).unwrap())
        .unwrap();

    println!("\n-- selection outcome --");
    println!(
        "  AGGREGATE fitness  → picks {:8?} α={:.2}: ΔA {:+.4}  ΔB {:+.4}  {}",
        agg.adapter,
        agg.alpha,
        agg.gain_a,
        agg.gain_b,
        if agg.gain_b < -1e-4 {
            "← REGRESSES domain B (silently accepted)"
        } else {
            ""
        }
    );
    println!(
        "  PER-DOMAIN (Pareto)→ picks {:8?} α={:.2}: ΔA {:+.4}  ΔB {:+.4}  {}",
        perdomain.adapter,
        perdomain.alpha,
        perdomain.gain_a,
        perdomain.gain_b,
        if perdomain.adapter != agg.adapter {
            "← pruned the overfit adapter"
        } else {
            ""
        }
    );

    println!("\n-- conclusion --");
    println!("The weightAdapter gene lets Darwin keep a generalizing fine-tune AND reject an");
    println!("overfit one — but ONLY under per-domain (no-regression) selection. A single");
    println!("aggregate fitness is fooled by an adapter whose in-dist gain hides an out-dist");
    println!("regression. Evolve the adapter as a gene, but score it per-repository.");

    assert!(
        perdomain.gain_a >= -1e-4 && perdomain.gain_b >= -1e-4,
        "Pareto pick must not regress"
    );
}
