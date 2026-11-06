//! Nightly research benchmark (2026-10-25): sampled-betweenness
//! (`ApproxBridgeForgetting`) vs exact global min-cut
//! (`MincutGatedForgetting`) structural bridge protection for agent-memory
//! compaction. Direct follow-up to ADR-345 (2026-09-05 nightly, REJECT —
//! exact min-cut was both too slow and showed no measured benefit at the
//! n=84 corpus size performance forced) and ADR-346 (2026-09-11 nightly,
//! fixed `RuVectorGraphAnalyzer::partition()` determinism but explicitly did
//! not re-attempt this benchmark or touch its latency).
//!
//! ## Hypothesis (fixed before this exact run; not changed after seeing
//! results)
//!
//! Given synthetic agent-memory corpora of Gaussian topic clusters plus
//! interpolated "bridge" memories (same generator family as the 2026-09-05
//! nightly: 50/50 midpoint between two random cluster centroids, Gaussian
//! noise=0.15 on bridges / 0.35 on core members), with a k-NN (k=8,
//! cosine >= 0.05) similarity graph over compaction candidates, release
//! build, seed=341 (documented, fixed):
//!
//! **Part 1 — matched n=84 (identical corpus size to ADR-345/ADR-346, for
//! direct comparability).** When the 84-entry store is compacted to 50% (42
//! entries) using baseline (`CoherencePolicy`, no structural signal),
//! candidate_a (`MincutGatedForgetting::hard`, 20% protect budget,
//! `mincut_trials=1` — identical settings to the ADR-345 production run),
//! and candidate_b (`ApproxBridgeForgetting::hard`, 20% protect budget,
//! default pivot count):
//!
//! - **T1 (speed, PRE-REGISTERED):** candidate_b's compaction wall-clock is
//!   <= candidate_a's / 20 (>= 20x faster at matched n).
//! - **T2 (effectiveness, PRE-REGISTERED):** candidate_b's bridge-survival
//!   gap over baseline is >= 15 percentage points — the identical absolute
//!   bar ADR-345 pre-registered for exact min-cut.
//! - **T3 (quality, PRE-REGISTERED):** candidate_b's |recall@10 delta| vs
//!   baseline is <= 2 percentage points — the identical bar ADR-345 used.
//!
//! **Part 2 — scale probe (EXPLORATORY, no prior bar exists; reported
//! descriptively).** At n=1,000 and n=4,000 — sizes ADR-345/ADR-346 never
//! reached (their own scaling probe measured 76ms-11.4s per
//! `partition()` call at only 50-400 *vertices*, and neither nightly ran a
//! real compaction at 1,000+; this run does not run `MincutGatedForgetting`
//! there either, for the same reason) — only baseline and candidate_b are
//! run, reporting bridge-survival gap, recall delta, and wall-clock.
//!
//! - **T4 (practicality, PRE-REGISTERED):** candidate_b's compaction
//!   wall-clock at n=4,000 is <= 5 seconds.
//!
//! Acceptance: **ACCEPT** iff T1-T4 all pass. A failure of T2 in particular
//! (no measured bridge-survival benefit) would echo ADR-345's own "zero
//! measured difference" finding and would be reported as REJECT, not
//! reframed.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example approx_bridge_bench --features mincut-forget

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, ApproxBridgeForgetting, CoherencePolicy, CoherenceWeights,
    CompactionPolicy, MemoryStore, MincutGatedForgetting,
};
use std::collections::HashSet;
use std::time::{Duration, Instant};

// ── Fixed hypothesis parameters ─────────────────────────────────────────────
const SEED: u64 = 341; // identical to the ADR-345/ADR-346 benchmarks
const K: usize = 5;
const N_QUERIES: usize = 20;
const N_HOT_CLUSTERS: usize = 2;
const CONTEXT_WINDOW_SIZE: usize = 10;
const HOT_ERA_HOT_FRAC: f64 = 0.90;

const PROTECT_FRACTION: f32 = 0.2;
const MINCUT_TRIALS: usize = 1; // identical to the ADR-345 production run

const T2_BRIDGE_SURVIVAL_GAP_PP: f32 = 15.0;
const T3_RECALL_TOLERANCE_PP: f32 = 2.0;
const T1_MIN_SPEEDUP_X: f64 = 20.0;
const T4_MAX_LATENCY_AT_4000_SECS: f64 = 5.0;

// ── Corpus shape (one row per size tested) ──────────────────────────────────
struct CorpusShape {
    label: &'static str,
    n_clusters: usize,
    per_cluster: usize,
    n_bridges: usize,
    dims: usize,
    /// Access-simulation accesses in the "cold" era (uniform over the whole
    /// corpus) and "hot" era (`HOT_ERA_HOT_FRAC` concentrated on
    /// `N_HOT_CLUSTERS`). For n=84 these are the exact literal counts the
    /// 2026-09-05 nightly (ADR-345) used, for a true apples-to-apples
    /// reproduction of its access pattern, not just its hyperparameters.
    cold_accesses: usize,
    hot_accesses: usize,
}

impl CorpusShape {
    fn n_core(&self) -> usize {
        self.n_clusters * self.per_cluster
    }
    fn n_total(&self) -> usize {
        self.n_core() + self.n_bridges
    }
}

// Identical to ADR-345/ADR-346's corpus: 6 clusters x 12 + 12 bridges = 84.
const SHAPE_84: CorpusShape = CorpusShape {
    label: "n=84 (matched to ADR-345/ADR-346)",
    n_clusters: 6,
    per_cluster: 12,
    n_bridges: 12,
    dims: 32,
    cold_accesses: 40, // literal ADR-345 constant
    hot_accesses: 80,  // literal ADR-345 constant
};
const SHAPE_1000: CorpusShape = CorpusShape {
    label: "n=1,000 (exploratory)",
    n_clusters: 10,
    per_cluster: 88,
    n_bridges: 120,
    dims: 64,
    cold_accesses: 476, // same ~0.48x corpus ratio as ADR-345's 40/84
    hot_accesses: 952,  // same ~0.95x corpus ratio as ADR-345's 80/84
};
const SHAPE_4000: CorpusShape = CorpusShape {
    label: "n=4,000 (exploratory)",
    n_clusters: 16,
    per_cluster: 220,
    n_bridges: 480,
    dims: 64,
    cold_accesses: 1905,
    hot_accesses: 3810,
};

// ── Vector utilities (mirrors mincut_gated_forgetting_bench.rs / src/main.rs;
// duplicated deliberately so this is a standalone, reproducible artifact —
// same tradeoff that file's own header comment documents) ──────────────────

fn unit_gaussian(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn scale_vec(v: &[f32], s: f32) -> Vec<f32> {
    v.iter().map(|x| x * s).collect()
}

fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}

fn perturb(centroid: &[f32], noise: f32, rng: &mut StdRng) -> Vec<f32> {
    let n = unit_gaussian(rng, centroid.len());
    normalize_vec(&add_vecs(centroid, &scale_vec(&n, noise)))
}

fn midpoint(a: &[f32], b: &[f32]) -> Vec<f32> {
    normalize_vec(&add_vecs(a, b))
}

// ── Dataset ──────────────────────────────────────────────────────────────────

struct Dataset {
    centroids: Vec<Vec<f32>>,
    cluster_of: Vec<usize>,
    bridge_indices: HashSet<usize>,
    queries: Vec<(Vec<f32>, Vec<u64>)>,
}

fn generate_dataset(shape: &CorpusShape, store: &mut MemoryStore, rng: &mut StdRng) -> Dataset {
    let centroids: Vec<Vec<f32>> = (0..shape.n_clusters)
        .map(|_| unit_gaussian(rng, shape.dims))
        .collect();
    let mut cluster_of = Vec::with_capacity(shape.n_total());

    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..shape.per_cluster {
            let v = perturb(centroid, 0.35, rng);
            store.insert(v);
            cluster_of.push(c);
        }
    }

    let mut bridge_indices = HashSet::new();
    for _ in 0..shape.n_bridges {
        let a = rng.gen_range(0..shape.n_clusters);
        let mut b = rng.gen_range(0..shape.n_clusters);
        while b == a && shape.n_clusters > 1 {
            b = rng.gen_range(0..shape.n_clusters);
        }
        let mid = midpoint(&centroids[a], &centroids[b]);
        let v = perturb(&mid, 0.15, rng);
        let idx = store.len();
        store.insert(v);
        bridge_indices.insert(idx);
        cluster_of.push(usize::MAX);
    }

    let mut queries = Vec::with_capacity(N_QUERIES);
    for i in 0..N_QUERIES {
        let hot_cluster = i % N_HOT_CLUSTERS.min(shape.n_clusters);
        let q = perturb(&centroids[hot_cluster], 0.30, rng);
        let truth: Vec<u64> = store.search(&q, K).into_iter().map(|r| r.id).collect();
        queries.push((q, truth));
    }

    Dataset {
        centroids,
        cluster_of,
        bridge_indices,
        queries,
    }
}

fn simulate_accesses(
    shape: &CorpusShape,
    store: &mut MemoryStore,
    dataset: &Dataset,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let n_total = shape.n_total();
    let n_hot_clusters = N_HOT_CLUSTERS.min(shape.n_clusters);

    for _ in 0..shape.cold_accesses {
        let idx = rng.gen_range(0..n_total);
        store.access_by_index(idx);
    }

    let mut context_accesses: Vec<Vec<f32>> = Vec::new();
    for _ in 0..shape.hot_accesses {
        let idx = if rng.gen_bool(HOT_ERA_HOT_FRAC) {
            let hot_c = rng.gen_range(0..n_hot_clusters);
            hot_c * shape.per_cluster + rng.gen_range(0..shape.per_cluster)
        } else {
            let cold_c = rng.gen_range(n_hot_clusters..shape.n_clusters.max(n_hot_clusters + 1));
            let cold_c = cold_c.min(shape.n_clusters - 1);
            cold_c * shape.per_cluster + rng.gen_range(0..shape.per_cluster)
        };
        store.access_by_index(idx);
        let cluster = dataset.cluster_of[idx];
        if cluster != usize::MAX {
            context_accesses.push(dataset.centroids[cluster].clone());
        }
    }

    let start = context_accesses.len().saturating_sub(CONTEXT_WINDOW_SIZE);
    context_accesses[start..].to_vec()
}

fn measure_recall(queries: &[(Vec<f32>, Vec<u64>)], store: &MemoryStore) -> f32 {
    let mut total = 0.0f32;
    for (q, truth) in queries {
        let candidates: Vec<u64> = store.search(q, K).into_iter().map(|r| r.id).collect();
        total += recall_at_k(truth, &candidates);
    }
    total / queries.len() as f32
}

/// Rebuilds a fresh, identically-seeded store+dataset, runs one compaction
/// policy, and reports (bridge survival rate, recall, compaction wall-clock).
fn run_policy(
    shape: &CorpusShape,
    policy: &dyn CompactionPolicy,
    seed: u64,
) -> (f32, f32, Duration) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut store = MemoryStore::new(shape.dims);
    let dataset = generate_dataset(shape, &mut store, &mut rng);
    let mut rng2 = StdRng::seed_from_u64(seed + 1);
    let context_window = simulate_accesses(shape, &mut store, &dataset, &mut rng2);
    assert_eq!(store.len(), shape.n_total());

    let bridge_ids: HashSet<u64> = dataset
        .bridge_indices
        .iter()
        .map(|&i| store.entries()[i].id)
        .collect();

    let target_size = shape.n_total() / 2;
    let t0 = Instant::now();
    compact(&mut store, policy, target_size, &context_window);
    let elapsed = t0.elapsed();

    assert_eq!(store.len(), target_size);
    let surviving_bridges = store
        .entries()
        .iter()
        .filter(|e| bridge_ids.contains(&e.id))
        .count();
    let survival_rate = surviving_bridges as f32 / bridge_ids.len().max(1) as f32;

    let recall = measure_recall(&dataset.queries, &store);
    (survival_rate, recall, elapsed)
}

struct Row {
    name: String,
    survival: f32,
    recall: f32,
    micros: u128,
}

fn print_table(rows: &[Row]) {
    println!(
        "{:<32} {:>14} {:>12} {:>18}",
        "Policy", "Bridge Surv.", "Recall@5", "Compaction (us)"
    );
    println!("{}", "-".repeat(80));
    for r in rows {
        println!(
            "{:<32} {:>13.1}% {:>11.1}% {:>18}",
            r.name,
            r.survival * 100.0,
            r.recall * 100.0,
            r.micros
        );
    }
    println!();
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-agent-memory — Approximate Bridge Forgetting            ║");
    println!("║  (sampled betweenness vs exact min-cut; follow-up to ADR-345/346) ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    println!("Platform  : {}", std::env::consts::OS);
    println!("Arch      : {}", std::env::consts::ARCH);
    println!("Seed      : {SEED}\n");

    // ── Part 1: matched n=84 three-way comparison ───────────────────────────
    println!("## Part 1 — {} ##\n", SHAPE_84.label);
    println!(
        "  Clusters: {}, per-cluster: {}, bridges: {}, dims: {}, target: 50% compaction\n",
        SHAPE_84.n_clusters, SHAPE_84.per_cluster, SHAPE_84.n_bridges, SHAPE_84.dims
    );

    let baseline = CoherencePolicy::default();
    let mut exact = MincutGatedForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION);
    exact.mincut_trials = MINCUT_TRIALS;
    let approx = ApproxBridgeForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION);

    let mut rows84 = Vec::new();
    for policy in [&baseline as &dyn CompactionPolicy, &exact, &approx] {
        let (survival, recall, dur) = run_policy(&SHAPE_84, policy, SEED);
        rows84.push(Row {
            name: policy.name().to_string(),
            survival,
            recall,
            micros: dur.as_micros(),
        });
    }
    print_table(&rows84);

    let base84 = &rows84[0];
    let exact84 = &rows84[1];
    let approx84 = &rows84[2];

    let approx_gap_pp = (approx84.survival - base84.survival) * 100.0;
    let exact_gap_pp = (exact84.survival - base84.survival) * 100.0;
    let recall_delta_pp = (approx84.recall - base84.recall).abs() * 100.0;
    let speedup_x = exact84.micros as f64 / approx84.micros.max(1) as f64;

    println!("Part 1 acceptance checks");
    let t1_pass = speedup_x >= T1_MIN_SPEEDUP_X;
    println!(
        "  T1 speedup vs exact mincut ({speedup_x:.1}x) >= {T1_MIN_SPEEDUP_X:.0}x        : {}",
        if t1_pass { "PASS" } else { "FAIL" }
    );
    let t2_pass = approx_gap_pp >= T2_BRIDGE_SURVIVAL_GAP_PP;
    println!(
        "  T2 bridge-survival gap vs baseline ({approx_gap_pp:+.1}pp) >= {T2_BRIDGE_SURVIVAL_GAP_PP:.0}pp : {}",
        if t2_pass { "PASS" } else { "FAIL" }
    );
    let t3_pass = recall_delta_pp <= T3_RECALL_TOLERANCE_PP;
    println!(
        "  T3 |recall delta| ({recall_delta_pp:.2}pp) <= {T3_RECALL_TOLERANCE_PP:.0}pp              : {}",
        if t3_pass { "PASS" } else { "FAIL" }
    );
    println!("  (reference) exact mincut's own survival gap vs baseline: {exact_gap_pp:+.1}pp\n");

    // ── Part 2: exploratory scale probe (no exact min-cut — established
    // impractical at this size by ADR-345/ADR-346) ──────────────────────────
    println!("## Part 2 — exploratory scale probe (baseline vs approx only) ##\n");

    let mut scale_rows: Vec<(String, Row, Row, f32, f32)> = Vec::new();
    for shape in [&SHAPE_1000, &SHAPE_4000] {
        println!(
            "  {} — clusters: {}, per-cluster: {}, bridges: {}, dims: {}",
            shape.label, shape.n_clusters, shape.per_cluster, shape.n_bridges, shape.dims
        );
        let (b_survival, b_recall, b_dur) = run_policy(shape, &baseline, SEED);
        let (a_survival, a_recall, a_dur) = run_policy(shape, &approx, SEED);
        let base_row = Row {
            name: "CoherenceWeighted".into(),
            survival: b_survival,
            recall: b_recall,
            micros: b_dur.as_micros(),
        };
        let approx_row = Row {
            name: "ApproxBridgeForgetting-Hard".into(),
            survival: a_survival,
            recall: a_recall,
            micros: a_dur.as_micros(),
        };
        print_table(&[
            Row {
                name: base_row.name.clone(),
                survival: base_row.survival,
                recall: base_row.recall,
                micros: base_row.micros,
            },
            Row {
                name: approx_row.name.clone(),
                survival: approx_row.survival,
                recall: approx_row.recall,
                micros: approx_row.micros,
            },
        ]);
        let gap_pp = (a_survival - b_survival) * 100.0;
        let recall_delta = (a_recall - b_recall).abs() * 100.0;
        scale_rows.push((
            shape.label.to_string(),
            base_row,
            approx_row,
            gap_pp,
            recall_delta,
        ));
    }

    for (label, _base_row, approx_row, gap_pp, recall_delta) in &scale_rows {
        println!(
            "  {label}: approx bridge-survival gap vs baseline = {gap_pp:+.1}pp, |recall delta| = {recall_delta:.2}pp, compaction = {:.3}ms",
            approx_row.micros as f64 / 1000.0
        );
    }
    println!();

    let n4000_secs = scale_rows
        .iter()
        .find(|(label, ..)| label.starts_with("n=4,000"))
        .map(|(_, _, approx_row, ..)| approx_row.micros as f64 / 1_000_000.0)
        .unwrap_or(f64::INFINITY);
    let t4_pass = n4000_secs <= T4_MAX_LATENCY_AT_4000_SECS;
    println!("Part 2 acceptance check (practicality bound only — effectiveness at scale is exploratory/descriptive, no pre-existing bar)");
    println!(
        "  T4 compaction wall-clock at n=4,000 ({n4000_secs:.3}s) <= {T4_MAX_LATENCY_AT_4000_SECS:.0}s : {}",
        if t4_pass { "PASS" } else { "FAIL" }
    );
    println!();

    let all_pass = t1_pass && t2_pass && t3_pass && t4_pass;
    if all_pass {
        println!("=> ACCEPT: approximate (sampled-betweenness) bridge protection matches the pre-registered speed, bridge-survival, and recall thresholds at matched n=84, and remains practical at n=4,000.");
    } else {
        println!(
            "=> REJECT: one or more mandatory (T1-T4) acceptance thresholds failed (see above)."
        );
        std::process::exit(1);
    }
}
