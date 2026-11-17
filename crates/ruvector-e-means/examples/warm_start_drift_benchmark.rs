//! Nightly research experiment: does warm-starting `fit_warm_start` from an
//! existing codebook make periodic PQ/IVF codebook refresh cheaper than a
//! cold `fit` restart under streaming drift, without losing quality?
//!
//! No extra dependencies or CPU flags; deterministic xorshift64* data
//! generation, following the style of `examples/benchmark.rs`.
//!
//! ```text
//! cargo run -p ruvector-e-means --release --example warm_start_drift_benchmark
//! ```
//!
//! Given a codebook `M0` trained on an initial window, and a second window
//! where a `drift_frac` share of rows are drawn from shifted cluster
//! centers (simulating streaming semantic drift in agent/vector memory),
//!
//! when the refresh is trained with `fit_warm_start(window1, cfg, M0.centroids)`
//! instead of a cold `fit(window1, cfg)`,
//!
//! then held-out MSE on the post-drift distribution should not regress by
//! more than `QUALITY_TOLERANCE`, and — at a reduced epoch budget — wall
//! time should drop by at least `SPEED_TARGET`,
//!
//! subject to both holding across light/moderate/heavy drift, not just one
//! cherry-picked level. The acceptance thresholds below are fixed before any
//! result is read and are not adjusted afterward.

use ruvector_e_means::{fit, fit_warm_start, Config};
use std::hint::black_box;
use std::time::Instant;

const QUALITY_TOLERANCE: f64 = 0.05; // candidate holdout MSE vs. baseline.
const PARITY_TOLERANCE: f64 = 0.02; // same-budget candidate A vs. baseline.
const SPEED_TARGET: f64 = 0.30; // required wall-time reduction for candidate B.
const DIMENSIONS: usize = 32;
const CLUSTERS: usize = 24;
const SAMPLES: usize = 6_000;
const HOLDOUT_SAMPLES: usize = 1_500;
const FULL_EPOCHS: usize = 4;
const REDUCED_EPOCHS: usize = 2; // ceil(FULL_EPOCHS / 2), candidate B's budget.
const REPS: usize = 6;
const DRIFT_LEVELS: [(&str, f64); 3] = [("light", 0.15), ("moderate", 0.35), ("heavy", 0.60)];

/// xorshift64* in [0, 1), matching the inline generator style already used
/// by `examples/benchmark.rs`; avoids adding a dependency for this example.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        // Box-Muller; good enough for a synthetic benchmark, not a stats library.
        let (u1, u2) = (self.unit().max(1e-12), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn cluster_centers(seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..CLUSTERS * DIMENSIONS)
        .map(|i| {
            let cluster = i / DIMENSIONS;
            let spread = (cluster as f64) * 7.0; // keep clusters well separated.
            (spread + rng.normal() * 2.0) as f32
        })
        .collect()
}

fn shifted_centers(base: &[f32], seed: u64, magnitude: f64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    base.iter()
        .map(|&v| (f64::from(v) + rng.normal() * magnitude) as f32)
        .collect()
}

/// Draws `n` rows: `drift_frac` of them around `drifted`, the rest around
/// `stable`, with noise `sigma`. Cluster choice and noise draws are
/// independent xorshift64* streams seeded from `seed`.
fn drifted_window(
    stable: &[f32],
    drifted: &[f32],
    n: usize,
    drift_frac: f64,
    seed: u64,
) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(n * DIMENSIONS);
    for _ in 0..n {
        let centers = if rng.unit() < drift_frac {
            drifted
        } else {
            stable
        };
        let cluster = (rng.unit() * CLUSTERS as f64) as usize % CLUSTERS;
        let start = cluster * DIMENSIONS;
        for d in 0..DIMENSIONS {
            data.push(centers[start + d] + (rng.normal() * 1.2) as f32);
        }
    }
    data
}

fn holdout_mse(data: &[f32], dimensions: usize, centroids: &[f32]) -> f64 {
    let assignments =
        ruvector_e_means::predict(data, dimensions, centroids).expect("valid holdout predict");
    let mut loss = 0.0;
    for (row, &cluster) in data.chunks_exact(dimensions).zip(&assignments) {
        let centroid = &centroids[cluster * dimensions..(cluster + 1) * dimensions];
        loss += row
            .iter()
            .zip(centroid)
            .map(|(&x, &c)| (f64::from(x) - f64::from(c)).powi(2))
            .sum::<f64>();
    }
    loss / assignments.len() as f64
}

struct Trial {
    refresh_ms: f64,
    holdout_mse: f64,
    train_mse: f64,
    updates: usize,
    revived: usize,
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}
fn stderr(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0).max(1.0);
    var.sqrt() / (xs.len() as f64).sqrt()
}

/// Runs one (variant, drift level, rep) trial: builds M0 on the stable
/// window, then measures only the *refresh* step against the drifted
/// window — M0's own construction cost is excluded, since in a real
/// streaming deployment M0 already exists before the refresh is triggered.
fn run_trial(drift_frac: f64, epochs: usize, warm: bool, rep_seed: u64) -> Trial {
    let stable = cluster_centers(rep_seed ^ 0x51);
    let drifted = shifted_centers(&stable, rep_seed ^ 0x52, 9.0);
    let old_window = drifted_window(&stable, &drifted, SAMPLES, 0.0, rep_seed ^ 0x53);
    let new_window = drifted_window(&stable, &drifted, SAMPLES, drift_frac, rep_seed ^ 0x54);
    let holdout = drifted_window(
        &stable,
        &drifted,
        HOLDOUT_SAMPLES,
        drift_frac,
        rep_seed ^ 0x55,
    );

    let base_config = Config {
        clusters: CLUSTERS,
        epochs: FULL_EPOCHS,
        seed: rep_seed ^ 0x56,
        ..Config::default()
    };
    let m0 = fit(black_box(&old_window), DIMENSIONS, &base_config).expect("M0 trains");

    let refresh_config = Config {
        epochs,
        seed: rep_seed ^ 0x57,
        ..base_config.clone()
    };
    let start = Instant::now();
    let model = if warm {
        fit_warm_start(
            black_box(&new_window),
            DIMENSIONS,
            &refresh_config,
            black_box(&m0.centroids),
        )
        .expect("warm refresh trains")
    } else {
        fit(black_box(&new_window), DIMENSIONS, &refresh_config).expect("cold refresh trains")
    };
    let refresh_ms = start.elapsed().as_secs_f64() * 1000.0;
    Trial {
        refresh_ms,
        holdout_mse: holdout_mse(&holdout, DIMENSIONS, &model.centroids),
        train_mse: model.mse,
        updates: model.updates,
        revived: model.revived_clusters,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Verdict {
    Accept,
    Reject,
    Inconclusive,
}
impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Verdict::Accept => "ACCEPT",
            Verdict::Reject => "REJECT",
            Verdict::Inconclusive => "INCONCLUSIVE",
        }
    }
    fn combine(self, other: Verdict) -> Verdict {
        match (self, other) {
            (Verdict::Reject, _) | (_, Verdict::Reject) => Verdict::Reject,
            (Verdict::Inconclusive, _) | (_, Verdict::Inconclusive) => Verdict::Inconclusive,
            _ => Verdict::Accept,
        }
    }
}

/// Compares a candidate's mean metric against baseline's with a simple
/// standard-error check: if the two means are within one combined standard
/// error of each other, the comparison is too noisy to call either way.
fn compare(
    baseline: &[f64],
    candidate: &[f64],
    tolerance: f64,
    lower_is_better: bool,
) -> (f64, Verdict) {
    let (bm, cm) = (mean(baseline), mean(candidate));
    let relative = if lower_is_better {
        (cm - bm) / bm
    } else {
        (bm - cm) / bm
    };
    let combined_sem = (stderr(baseline).powi(2) + stderr(candidate).powi(2)).sqrt();
    let verdict = if relative.abs() * bm < combined_sem && combined_sem > 0.0 {
        Verdict::Inconclusive
    } else if relative <= tolerance {
        Verdict::Accept
    } else {
        Verdict::Reject
    };
    (relative, verdict)
}

fn main() {
    println!("# Given/When/Then: fit_warm_start vs. cold fit under simulated streaming drift");
    println!(
        "# dims={DIMENSIONS} clusters={CLUSTERS} samples={SAMPLES} holdout={HOLDOUT_SAMPLES} \
         full_epochs={FULL_EPOCHS} reduced_epochs={REDUCED_EPOCHS} reps={REPS} \
         quality_tolerance={QUALITY_TOLERANCE} parity_tolerance={PARITY_TOLERANCE} speed_target={SPEED_TARGET}"
    );

    let mut overall = Verdict::Accept;
    for &(name, drift_frac) in &DRIFT_LEVELS {
        let mut baseline = Vec::new();
        let mut candidate_a = Vec::new();
        let mut candidate_b = Vec::new();
        for rep in 0..REPS {
            let seed = 1_000 * (drift_frac * 1000.0) as u64 + rep as u64;
            baseline.push(run_trial(drift_frac, FULL_EPOCHS, false, seed));
            candidate_a.push(run_trial(drift_frac, FULL_EPOCHS, true, seed));
            candidate_b.push(run_trial(drift_frac, REDUCED_EPOCHS, true, seed));
        }
        let report = |label: &str, trials: &[Trial]| {
            let holdout: Vec<f64> = trials.iter().map(|t| t.holdout_mse).collect();
            let ms: Vec<f64> = trials.iter().map(|t| t.refresh_ms).collect();
            println!(
                "{{\"drift\":\"{name}\",\"variant\":\"{label}\",\"meanHoldoutMse\":{:.6},\"meanRefreshMs\":{:.4},\
                 \"meanTrainMse\":{:.6},\"meanUpdates\":{:.1},\"meanRevived\":{:.2}}}",
                mean(&holdout),
                mean(&ms),
                mean(trials.iter().map(|t| t.train_mse).collect::<Vec<_>>().as_slice()),
                trials.iter().map(|t| t.updates as f64).sum::<f64>() / trials.len() as f64,
                trials.iter().map(|t| t.revived as f64).sum::<f64>() / trials.len() as f64,
            );
            (holdout, ms)
        };
        let (base_holdout, base_ms) = report("baseline_cold_fit", &baseline);
        let (a_holdout, _a_ms) = report("candidate_A_warm_same_epochs", &candidate_a);
        let (b_holdout, b_ms) = report("candidate_B_warm_reduced_epochs", &candidate_b);

        let (a_rel, a_verdict) = compare(&base_holdout, &a_holdout, PARITY_TOLERANCE, true);
        let (b_quality_rel, b_quality_verdict) =
            compare(&base_holdout, &b_holdout, QUALITY_TOLERANCE, true);
        let (b_speed_rel, b_speed_verdict) = compare(&base_ms, &b_ms, -SPEED_TARGET, true);
        // Speed is a "faster is better" requirement, phrased as -SPEED_TARGET
        // in the same "lower is better, relative <= tolerance" shape as quality.
        let b_verdict = b_quality_verdict.combine(b_speed_verdict);
        let level_verdict = a_verdict.combine(b_verdict);
        println!(
            "{{\"drift\":\"{name}\",\"candidateA_vs_baseline_relHoldoutMse\":{:.4},\"candidateA_verdict\":\"{}\",\
             \"candidateB_vs_baseline_relHoldoutMse\":{:.4},\"candidateB_relRefreshMsChange\":{:.4},\
             \"candidateB_verdict\":\"{}\",\"levelVerdict\":\"{}\"}}",
            a_rel,
            a_verdict.label(),
            b_quality_rel,
            b_speed_rel,
            b_verdict.label(),
            level_verdict.label()
        );
        overall = overall.combine(level_verdict);
    }
    println!("{{\"acceptance\":\"{}\"}}", overall.label());

    darwin_epoch_fraction_search();
}

/// Bounded, in-process search over the refresh epoch fraction, scoped to
/// this experiment only. This is explicitly *not* an invocation of the
/// ecosystem's `@metaharness/darwin` package (see the nightly report's
/// "MetaHarness and Darwin scoping" note for why) — it is a small internal
/// evolutionary search, structured like Darwin's generations/candidates and
/// fitness/hard-constraint contract, over a single scalar genome.
fn darwin_epoch_fraction_search() {
    const DRIFT_FRAC: f64 = 0.35; // the "moderate" level, as a representative point.
    const DARWIN_REPS: usize = 3;
    const SEED_OFFSET: u64 = 0x4441_5731; // disjoint from the main experiment's seeds.

    let fitness = |baseline_holdout: f64, baseline_ms: f64, holdout: f64, ms: f64| -> Option<f64> {
        if !holdout.is_finite()
            || !ms.is_finite()
            || holdout > baseline_holdout * (1.0 + QUALITY_TOLERANCE)
        {
            return None; // hard constraint: Step 20 of the nightly harness.
        }
        let quality = (1.0
            - ((holdout - baseline_holdout) / baseline_holdout).max(0.0) / QUALITY_TOLERANCE)
            .clamp(0.0, 1.0);
        let speed = ((baseline_ms - ms) / baseline_ms).clamp(0.0, 1.0);
        Some(0.6 * quality + 0.4 * speed)
    };

    let measure = |fraction: f64, gen: usize| -> (f64, f64, f64) {
        let epochs = ((FULL_EPOCHS as f64 * fraction).round() as usize).max(1);
        let mut holdouts = Vec::new();
        let mut mss = Vec::new();
        for rep in 0..DARWIN_REPS {
            let seed = SEED_OFFSET + (gen as u64) * 100 + rep as u64;
            let trial = run_trial(DRIFT_FRAC, epochs, true, seed);
            holdouts.push(trial.holdout_mse);
            mss.push(trial.refresh_ms);
        }
        (mean(&holdouts), mean(&mss), epochs as f64)
    };
    let baseline_measure = |gen: usize| -> (f64, f64) {
        let mut holdouts = Vec::new();
        let mut mss = Vec::new();
        for rep in 0..DARWIN_REPS {
            let seed = SEED_OFFSET + (gen as u64) * 100 + rep as u64;
            let trial = run_trial(DRIFT_FRAC, FULL_EPOCHS, false, seed);
            holdouts.push(trial.holdout_mse);
            mss.push(trial.refresh_ms);
        }
        (mean(&holdouts), mean(&mss))
    };

    println!("# Darwin-lite: bounded epoch-fraction search (3 generations x 4 candidates, max 1 promotion)");
    let (baseline_holdout, baseline_ms) = baseline_measure(0);
    let parent_fraction = 1.0; // candidate A's fraction; the fallback if nothing wins.
    let mut best_fraction = parent_fraction;
    let mut best_fitness =
        fitness(baseline_holdout, baseline_ms, baseline_holdout, baseline_ms).unwrap_or(0.0);

    let generations: [Vec<f64>; 3] = [
        vec![1.0, 0.75, 0.5, 0.25],
        Vec::new(), // filled in below, refined around the running best.
        Vec::new(),
    ];
    let mut candidates_g1 = generations[0].clone();
    let _ = generations;
    for generation in 0..3 {
        let fractions: Vec<f64> = if generation == 0 {
            std::mem::take(&mut candidates_g1)
        } else {
            let step = 0.1 / 2.0_f64.powi(generation as i32 - 1);
            vec![
                (best_fraction + step).min(1.0),
                (best_fraction - step).max(0.01),
                (best_fraction + step / 2.0).min(1.0),
                (best_fraction - step / 2.0).max(0.01),
            ]
        };
        for fraction in fractions {
            let (holdout, ms, epochs) = measure(fraction, generation + 1);
            let score = fitness(baseline_holdout, baseline_ms, holdout, ms);
            let eligible = score.is_some();
            let score_value = score.unwrap_or(f64::NEG_INFINITY);
            if eligible && score_value > best_fitness {
                best_fitness = score_value;
                best_fraction = fraction;
            }
            println!(
                "{{\"generation\":{},\"fraction\":{:.4},\"epochs\":{:.0},\"holdoutMse\":{:.6},\"refreshMs\":{:.4},\
                 \"eligible\":{},\"fitness\":{}}}",
                generation + 1,
                fraction,
                epochs,
                holdout,
                ms,
                eligible,
                score.map(|s| format!("{s:.4}")).unwrap_or_else(|| "null".into())
            );
        }
    }
    let promoted = best_fraction < parent_fraction;
    println!(
        "{{\"darwin\":{{\"parentFraction\":{parent_fraction},\"winnerFraction\":{:.4},\"winnerFitness\":{:.4},\
         \"promoted\":{promoted},\"maxPromotions\":1}}}}",
        best_fraction, best_fitness
    );
}
