//! Nightly research benchmark (2026-09-19): does `ruvector_mincut::ApproxMinCut`
//! fix the performance bottleneck the 2026-09-05 nightly (ADR-345) found in
//! `RuVectorGraphAnalyzer::partition()` (1,800-2,700x slower than the scalar
//! `CoherencePolicy` baseline at an 84-memory corpus) while keeping the same
//! bridge-survival benefit?
//!
//! Hypothesis (fixed before running this benchmark, based on reading
//! `ruvector-mincut`'s `algorithm::approximate` source — not on any prior run
//! of *this* benchmark):
//!
//! Given the same synthetic corpus, k-NN graph construction, and acceptance
//! bar as the 2026-09-05 experiment (72 core + 12 bridge = 84 memories, 32-dim,
//! k=5, cosine >= 0.05, compacted to 42 = 50%),
//!
//! when boundary detection uses `ApproxMinCut` (candidate B:
//! `ApproxMincutForgetting`-Soft/Hard) instead of `RuVectorGraphAnalyzer`
//! (candidate A: `MincutGatedForgetting`-Soft/Hard, kept here only as the
//! within-run speed reference, not re-litigated on its own merits) versus
//! plain `CoherencePolicy` (baseline),
//!
//! then candidate B should be at least 10x faster per compaction call than
//! candidate A (operationalizing "fix the bottleneck"),
//!
//! subject to: candidate B's bridge-survival gap over baseline must still be
//! at least 15 percentage points (the original 2026-09-05 bar) for ACCEPT — a
//! candidate that is fast but does not protect bridges is not a fix, it is a
//! different, also-unusable policy. Based on reading
//! `ApproxMinCut::compute_partition` (a BFS half-split unrelated to the cut
//! value it reports; see `examples/approx_mincut_partition_probe.rs`), the
//! pre-registered expectation is that the *speed* criterion will likely pass
//! and the *survival* criterion will likely fail — this run measures whether
//! that is actually true rather than assuming it.
//!
//! Tamper-detection (eviction witness chain) is not re-tested here: it is
//! generic over any `CompactionPolicy` and was already validated 20/20 by the
//! 2026-09-05 benchmark against `MincutGatedForgetting`; nothing about the
//! witness chain changes when the boundary-detection engine changes.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example approx_mincut_forgetting_bench --features mincut-forget

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, ApproxMincutForgetting, CoherencePolicy, CoherenceWeights,
    CompactionPolicy, MemoryStore, MincutGatedForgetting,
};
use std::collections::HashSet;
use std::time::{Duration, Instant};

// ── Dataset parameters (identical to the 2026-09-05 benchmark for direct
// comparability; see that experiment's README for why this size was chosen)
const N_CLUSTERS: usize = 6;
const PER_CLUSTER: usize = 12;
const N_CORE: usize = N_CLUSTERS * PER_CLUSTER; // 72
const N_BRIDGES: usize = 12;
const N_MEMORIES: usize = N_CORE + N_BRIDGES; // 84
const N_HOT_CLUSTERS: usize = 2;
const DIMS: usize = 32;
const N_QUERIES: usize = 20;
const K: usize = 5;
const TARGET_SIZE: usize = N_MEMORIES / 2; // 42, 50% compaction
const CONTEXT_WINDOW_SIZE: usize = 10;

const N_COLD_ERA_ACCESSES: usize = 40;
const N_HOT_ERA_ACCESSES: usize = 80;
const HOT_ERA_HOT_FRAC: f64 = 0.90;

const STRUCTURAL_BONUS: f32 = 0.5;
const PROTECT_FRACTION: f32 = 0.2;
const BRIDGE_SURVIVAL_GAP_THRESHOLD_PP: f32 = 15.0;
const RECALL_TOLERANCE: f32 = 0.02;
const MINCUT_TRIALS: usize = 1; // matches the 2026-09-05 benchmark's corpus-size compromise
const MIN_SPEEDUP_VS_EXACT: f64 = 10.0;
// ApproxMinCut's internal HashSet iteration order (Rust's default randomized
// hasher) makes its reported partition vary call to call against the same
// input graph (see the in-`main` comment above the approx-policy loop); this
// many in-process trials characterize that instead of reporting one
// arbitrary sample.
const N_APPROX_TRIALS: usize = 10;

// ── Vector utilities (mirrors mincut_gated_forgetting_bench.rs / src/main.rs) ──

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

fn generate_dataset(store: &mut MemoryStore, rng: &mut StdRng) -> Dataset {
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS).map(|_| unit_gaussian(rng, DIMS)).collect();
    let mut cluster_of = Vec::with_capacity(N_MEMORIES);

    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..PER_CLUSTER {
            let v = perturb(centroid, 0.35, rng);
            store.insert(v);
            cluster_of.push(c);
        }
    }

    let mut bridge_indices = HashSet::new();
    for _ in 0..N_BRIDGES {
        let a = rng.gen_range(0..N_CLUSTERS);
        let mut b = rng.gen_range(0..N_CLUSTERS);
        while b == a {
            b = rng.gen_range(0..N_CLUSTERS);
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
        let hot_cluster = i % N_HOT_CLUSTERS;
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
    store: &mut MemoryStore,
    dataset: &Dataset,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    for _ in 0..N_COLD_ERA_ACCESSES {
        let idx = rng.gen_range(0..N_MEMORIES);
        store.access_by_index(idx);
    }

    let mut context_accesses: Vec<Vec<f32>> = Vec::new();
    for _ in 0..N_HOT_ERA_ACCESSES {
        let idx = if rng.gen_bool(HOT_ERA_HOT_FRAC) {
            let hot_c = rng.gen_range(0..N_HOT_CLUSTERS);
            hot_c * PER_CLUSTER + rng.gen_range(0..PER_CLUSTER)
        } else {
            let cold_c = rng.gen_range(N_HOT_CLUSTERS..N_CLUSTERS);
            cold_c * PER_CLUSTER + rng.gen_range(0..PER_CLUSTER)
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

fn run_policy(policy: &dyn CompactionPolicy, seed: u64) -> (f32, f32, Duration) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut store = MemoryStore::new(DIMS);
    let dataset = generate_dataset(&mut store, &mut rng);
    let mut rng2 = StdRng::seed_from_u64(seed + 1);
    let context_window = simulate_accesses(&mut store, &dataset, &mut rng2);
    assert_eq!(store.len(), N_MEMORIES);

    let bridge_ids: HashSet<u64> = dataset
        .bridge_indices
        .iter()
        .map(|&i| store.entries()[i].id)
        .collect();

    let t0 = Instant::now();
    compact(&mut store, policy, TARGET_SIZE, &context_window);
    let elapsed = t0.elapsed();

    assert_eq!(store.len(), TARGET_SIZE);
    let surviving_bridges = store
        .entries()
        .iter()
        .filter(|e| bridge_ids.contains(&e.id))
        .count();
    let survival_rate = surviving_bridges as f32 / bridge_ids.len() as f32;

    let recall = measure_recall(&dataset.queries, &store);
    (survival_rate, recall, elapsed)
}

fn main() {
    let seed: u64 = 341; // same seed as the 2026-09-05 benchmark: identical dataset
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-agent-memory — Approx-Mincut-Gated Forgetting            ║");
    println!("║  (nightly 2026-09-19 follow-up to ADR-345)                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Platform  : {}", std::env::consts::OS);
    println!("Arch      : {}", std::env::consts::ARCH);
    println!();

    println!("Dataset (identical to the 2026-09-05 benchmark, seed={seed})");
    println!("  Clusters        : {N_CLUSTERS} ({PER_CLUSTER} core memories each = {N_CORE})");
    println!("  Bridge memories : {N_BRIDGES}");
    println!("  Total memories  : {N_MEMORIES}");
    println!("  Target size     : {TARGET_SIZE} (50% compaction)");
    println!();

    let cow = CoherencePolicy::default();
    let mut exact_soft = MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
    exact_soft.mincut_trials = MINCUT_TRIALS;
    let mut exact_hard = MincutGatedForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION);
    exact_hard.mincut_trials = MINCUT_TRIALS;
    let approx_soft = ApproxMincutForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
    let approx_hard = ApproxMincutForgetting::hard(CoherenceWeights::default(), PROTECT_FRACTION);

    struct Row {
        name: String,
        survival: f32,
        recall: f32,
        micros: u128,
    }
    let mut rows = Vec::new();
    // Baseline and the exact (ADR-345) engine: one run each. Both are stable
    // across repeated calls in this corpus (no dependency on this process's
    // HashSet iteration order was found for either — see the note below on
    // why the *approximate* engine's rows are handled differently).
    for policy in [&cow as &dyn CompactionPolicy, &exact_soft, &exact_hard] {
        let (survival, recall, dur) = run_policy(policy, seed);
        rows.push(Row {
            name: policy.name().to_string(),
            survival,
            recall,
            micros: dur.as_micros(),
        });
    }

    // The approximate engine's rows are NOT single runs. Discovered while
    // developing this benchmark: `ApproxMinCut`'s internal `vertices:
    // HashSet<VertexId>` uses Rust's default (randomized) hasher, and its
    // `compute_partition`'s BFS start vertex is `self.vertices.iter().next()`
    // — so which vertex it starts from, and therefore which vertices end up
    // flagged as "boundary," varies from call to call even against the
    // byte-identical input graph this benchmark rebuilds every time (dataset
    // generation is reseeded identically; only ApproxMinCut's internal
    // hasher state differs). This is a second, independent non-determinism
    // finding beyond ADR-345's own (which was in `RuVectorGraphAnalyzer`,
    // for a different reason). `N_APPROX_TRIALS` repeated calls in the same
    // process characterize this instead of reporting one arbitrary sample;
    // see the printed range below and the research doc's "Non-determinism"
    // section for the full account, including separate-process confirmation.
    struct TrialStats {
        name: String,
        mean_survival: f32,
        min_survival: f32,
        max_survival: f32,
        mean_recall: f32,
        mean_micros: u128,
    }
    let mut approx_stats = Vec::new();
    for policy in [&approx_soft as &dyn CompactionPolicy, &approx_hard] {
        let mut survivals = Vec::with_capacity(N_APPROX_TRIALS);
        let mut recalls = Vec::with_capacity(N_APPROX_TRIALS);
        let mut micros = Vec::with_capacity(N_APPROX_TRIALS);
        for _ in 0..N_APPROX_TRIALS {
            let (survival, recall, dur) = run_policy(policy, seed);
            survivals.push(survival);
            recalls.push(recall);
            micros.push(dur.as_micros());
        }
        let mean_survival = survivals.iter().sum::<f32>() / survivals.len() as f32;
        let min_survival = survivals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_survival = survivals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
        let mean_micros = (micros.iter().sum::<u128>()) / micros.len() as u128;
        rows.push(Row {
            name: policy.name().to_string(),
            survival: mean_survival,
            recall: mean_recall,
            micros: mean_micros,
        });
        approx_stats.push(TrialStats {
            name: policy.name().to_string(),
            mean_survival,
            min_survival,
            max_survival,
            mean_recall,
            mean_micros,
        });
    }

    println!(
        "{:<32} {:>13} {:>11} {:>16}",
        "Policy", "Bridge Surv.", "Recall@10", "Compaction (us)"
    );
    println!("{}", "-".repeat(78));
    for r in &rows[..3] {
        println!(
            "{:<32} {:>12.1}% {:>10.1}% {:>16}",
            r.name,
            r.survival * 100.0,
            r.recall * 100.0,
            r.micros
        );
    }
    for s in &approx_stats {
        println!(
            "{:<32} {:>11.1}%* {:>10.1}% {:>16}",
            s.name,
            s.mean_survival * 100.0,
            s.mean_recall * 100.0,
            s.mean_micros
        );
    }
    println!(
        "  * mean of {N_APPROX_TRIALS} in-process trials (non-deterministic per-trial — see above); ranges:"
    );
    for s in &approx_stats {
        println!(
            "    {:<30} min={:>5.1}%  max={:>5.1}%  mean={:>5.1}%",
            s.name,
            s.min_survival * 100.0,
            s.max_survival * 100.0,
            s.mean_survival * 100.0
        );
    }
    println!();

    let baseline = &rows[0];
    let exact_soft_row = &rows[1];
    let exact_hard_row = &rows[2];
    let approx_soft_row = &rows[3];
    let approx_hard_row = &rows[4];

    let speedup_soft = exact_soft_row.micros as f64 / approx_soft_row.micros.max(1) as f64;
    let speedup_hard = exact_hard_row.micros as f64 / approx_hard_row.micros.max(1) as f64;

    // Acceptance uses the *mean* across N_APPROX_TRIALS, not a single
    // cherry-pickable run, precisely because the per-run variance is itself
    // part of this experiment's finding.
    let survival_gap_soft = (approx_soft_row.survival - baseline.survival) * 100.0;
    let survival_gap_hard = (approx_hard_row.survival - baseline.survival) * 100.0;

    let recall_delta_soft = (approx_soft_row.recall - baseline.recall).abs();
    let recall_delta_hard = (approx_hard_row.recall - baseline.recall).abs();

    println!("Acceptance test (candidate B = ApproxMincutForgetting)");
    let speed_soft_pass = speedup_soft >= MIN_SPEEDUP_VS_EXACT;
    let speed_hard_pass = speedup_hard >= MIN_SPEEDUP_VS_EXACT;
    println!(
        "  Soft speedup vs. exact (RuVectorGraphAnalyzer) ({speedup_soft:.1}x) >= {MIN_SPEEDUP_VS_EXACT:.0}x : {}",
        if speed_soft_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Hard speedup vs. exact (RuVectorGraphAnalyzer) ({speedup_hard:.1}x) >= {MIN_SPEEDUP_VS_EXACT:.0}x : {}",
        if speed_hard_pass { "PASS" } else { "FAIL" }
    );

    let survival_soft_pass = survival_gap_soft >= BRIDGE_SURVIVAL_GAP_THRESHOLD_PP;
    let survival_hard_pass = survival_gap_hard >= BRIDGE_SURVIVAL_GAP_THRESHOLD_PP;
    println!(
        "  Soft bridge-survival gap ({survival_gap_soft:+.1}pp) >= {BRIDGE_SURVIVAL_GAP_THRESHOLD_PP:.0}pp        : {}",
        if survival_soft_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Hard bridge-survival gap ({survival_gap_hard:+.1}pp) >= {BRIDGE_SURVIVAL_GAP_THRESHOLD_PP:.0}pp        : {}",
        if survival_hard_pass { "PASS" } else { "FAIL" }
    );

    let recall_soft_pass = recall_delta_soft <= RECALL_TOLERANCE;
    let recall_hard_pass = recall_delta_hard <= RECALL_TOLERANCE;
    println!(
        "  Soft |recall delta| ({:.2}pp) <= {:.0}pp                              : {}",
        recall_delta_soft * 100.0,
        RECALL_TOLERANCE * 100.0,
        if recall_soft_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Hard |recall delta| ({:.2}pp) <= {:.0}pp                              : {}",
        recall_delta_hard * 100.0,
        RECALL_TOLERANCE * 100.0,
        if recall_hard_pass { "PASS" } else { "FAIL" }
    );
    println!();

    let soft_accept = speed_soft_pass && survival_soft_pass && recall_soft_pass;
    let hard_accept = speed_hard_pass && survival_hard_pass && recall_hard_pass;

    if soft_accept && hard_accept {
        println!(
            "=> ACCEPT: ApproxMincutForgetting fixes the ADR-345 latency bottleneck without \
             losing the bridge-survival benefit."
        );
    } else if !speed_soft_pass && !speed_hard_pass {
        println!(
            "=> REJECT: ApproxMincutForgetting is not meaningfully faster than the exact engine \
             at this corpus size, so it does not address the bottleneck it was meant to fix \
             (see the research doc for why: for graphs this size, ApproxMinCut's internal \
             sparsifier target size exceeds the edge count, so it degenerates into re-running \
             full Stoer-Wagner num_samples=3 times instead of once)."
        );
    } else if !survival_soft_pass && !survival_hard_pass {
        println!(
            "=> REJECT: ApproxMincutForgetting is faster but does not reliably protect \
             structural bridges, because ApproxMinCut::compute_partition returns a BFS \
             half-split unrelated to the cut value it reports (see \
             examples/approx_mincut_partition_probe.rs), and which vertices it flags varies \
             from run to run against the identical input graph (see the min/max range printed \
             above). Speed without correctness is not a fix for ADR-345's rejected candidate — \
             it is a different, also-unusable, *and* non-deterministic policy."
        );
    } else {
        println!("=> REJECT: one or more mandatory acceptance thresholds failed (see above).");
    }
    if !(soft_accept && hard_accept) {
        std::process::exit(1);
    }
}
