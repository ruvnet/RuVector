//! Nightly research, 2026-09-18: Structural-Time-Gated Compaction Scheduling.
//!
//! docs/research/nightly/2026-09-18-structural-time-gated-memory-compaction
//!
//! Compares three *compaction triggers* (when to run [`compact`], as
//! opposed to [`CoherencePolicy`], which decides what survives) on a single
//! deterministic write stream that alternates quiet epochs (near-duplicate
//! writes into one existing topic cluster) with burst epochs (writes
//! introducing a brand-new, never-before-seen topic cluster):
//!
//! - `FixedInterval(50)`  — compacts every 50 writes, regardless of content.
//! - `Capacity(2x target)` — compacts once the store exceeds 2x target_size.
//! - `StructuralGate`     — compacts once accumulated `StructuralProperTime`
//!   (emergent-time's arc-length-through-state-manifold clock) since the
//!   last compaction crosses a threshold calibrated from an initial quiet
//!   baseline.
//!
//! Hypothesis (fixed before running; see the nightly research doc §Hypothesis
//! for the full Given/When/Then and acceptance thresholds):
//!
//!   Given this bursty write stream, compacted to `target_size` by the same
//!   `CoherencePolicy` at each trigger fire,
//!
//!   when compaction is scheduled by `StructuralGate` instead of
//!   `FixedInterval(50)`,
//!
//!   then StructuralGate's time-integrated excess-store-size (memory
//!   overhead while waiting to compact) and total compaction-call count
//!   should both be lower than FixedInterval's by at least 20%,
//!
//!   subject to: final Recall@10 on hot-cluster queries staying within 2
//!   percentage points of FixedInterval's, and wall-clock trigger+compaction
//!   overhead staying within 2x of FixedInterval's (StructuralGate does
//!   `O(window*dims)` work per write vs `O(1)`).
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --features structural-gate \
//!     --example structural_gated_compaction_bench

use emergent_time::structural_clock::StructuralMetric;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, CapacityTrigger, CoherencePolicy, CompactionTrigger,
    FixedIntervalTrigger, MemoryStore, StructuralGateTrigger,
};
use std::time::Instant;

// ── Dataset parameters (fixed a priori) ────────────────────────────────────
const DIMS: usize = 32;
const TARGET_SIZE: usize = 200;
const QUIET_EPOCH_WRITES: usize = 300;
const BURST_EPOCH_WRITES: usize = 50;
const N_EPOCH_PAIRS: usize = 4; // 4x(quiet, burst) = 1400 total writes
const NOISE: f32 = 0.02;
const SEED: u64 = 0x5EED_C0DE;
const N_HOT_QUERIES: usize = 30;
const K: usize = 10;

// Trigger configuration (fixed a priori, not tuned on results).
const FIXED_INTERVAL: u64 = 50;
const CAPACITY_MULTIPLIER: usize = 2;
const STRUCTURAL_WINDOW: usize = 24;
const STRUCTURAL_CALIBRATION_WRITES: usize = 50;
const STRUCTURAL_MULTIPLIER: f64 = 20.0;

fn unit_vec(rng: &mut StdRng, dims: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    normalize(&v)
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}

fn noisy_sample(rng: &mut StdRng, centroid: &[f32], noise: f32) -> Vec<f32> {
    let v: Vec<f32> = centroid
        .iter()
        .map(|&c| c + noise * (rng.gen::<f32>() * 2.0 - 1.0))
        .collect();
    normalize(&v)
}

/// A write-stream event: which cluster the write is drawn from.
#[derive(Clone, Copy)]
enum Event {
    Quiet(usize), // draw from existing cluster index
    Burst(usize), // draw from a freshly introduced cluster index
}

/// Build the deterministic quiet/burst write schedule and the full set of
/// cluster centroids used to generate it (existing clusters are reused
/// across quiet epochs; each burst epoch introduces one brand-new cluster).
fn build_schedule() -> (Vec<Event>, usize) {
    let mut schedule = Vec::new();
    let mut n_clusters = 1; // cluster 0 exists from the start
    for _ in 0..N_EPOCH_PAIRS {
        for _ in 0..QUIET_EPOCH_WRITES {
            schedule.push(Event::Quiet(n_clusters - 1)); // most recent cluster
        }
        let new_cluster = n_clusters;
        n_clusters += 1;
        for _ in 0..BURST_EPOCH_WRITES {
            schedule.push(Event::Burst(new_cluster));
        }
    }
    (schedule, n_clusters)
}

struct RunResult {
    name: String,
    n_compactions: usize,
    excess_size_integral: u64,
    final_recall_at_10: f32,
    wall_time: std::time::Duration,
    final_store_size: usize,
    /// Diagnostic only (not part of the pre-registered acceptance test):
    /// how many fires landed on a write whose `Event` was `Quiet` vs
    /// `Burst`, to characterize *where* a trigger fires.
    fires_during_quiet: usize,
    fires_during_burst: usize,
}

fn run_trigger(
    name: &str,
    mut trigger: Box<dyn CompactionTrigger>,
    schedule: &[Event],
    centroids: &[Vec<f32>],
    hot_queries: &[Vec<f32>],
    truth_ids_by_query: &[Vec<u64>],
) -> RunResult {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x1234_5678);
    let mut store = MemoryStore::new(DIMS);
    let policy = CoherencePolicy::default();
    let mut n_compactions = 0usize;
    let mut excess_size_integral: u64 = 0;
    let mut fires_during_quiet = 0usize;
    let mut fires_during_burst = 0usize;
    let context_window: Vec<Vec<f32>> = hot_queries.to_vec();

    let start = Instant::now();
    for ev in schedule {
        let v = match *ev {
            Event::Quiet(c) => noisy_sample(&mut rng, &centroids[c], NOISE),
            Event::Burst(c) => noisy_sample(&mut rng, &centroids[c], NOISE),
        };
        store.insert(v);

        let excess = store.len().saturating_sub(TARGET_SIZE) as u64;
        excess_size_integral += excess;

        if trigger.on_write(store.entries()) && store.len() > TARGET_SIZE {
            compact(&mut store, &policy, TARGET_SIZE, &context_window);
            trigger.on_compacted(store.entries());
            n_compactions += 1;
            match *ev {
                Event::Quiet(_) => fires_during_quiet += 1,
                Event::Burst(_) => fires_during_burst += 1,
            }
        }
    }
    // Final compaction pass so every trigger ends at a comparable state for
    // the quality metric (does not count toward n_compactions: it measures
    // end-state quality, not scheduling behavior).
    if store.len() > TARGET_SIZE {
        compact(&mut store, &policy, TARGET_SIZE, &context_window);
    }
    let wall_time = start.elapsed();

    let mut recalls = Vec::with_capacity(hot_queries.len());
    for (q, truth) in hot_queries.iter().zip(truth_ids_by_query) {
        let results = store.search(q, K);
        let candidate_ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        recalls.push(recall_at_k(truth, &candidate_ids));
    }
    let final_recall_at_10 = recalls.iter().sum::<f32>() / recalls.len().max(1) as f32;

    RunResult {
        name: name.to_string(),
        n_compactions,
        excess_size_integral,
        final_recall_at_10,
        wall_time,
        final_store_size: store.len(),
        fires_during_quiet,
        fires_during_burst,
    }
}

fn main() {
    let (schedule, n_clusters) = build_schedule();
    let mut cgen = StdRng::seed_from_u64(SEED);
    let centroids: Vec<Vec<f32>> = (0..n_clusters).map(|_| unit_vec(&mut cgen, DIMS)).collect();

    // Hot queries: drawn from the final (most recent, "still relevant")
    // cluster, so recall@10 measures whether compaction retained enough of
    // the freshest, most-queried topic.
    let mut qgen = StdRng::seed_from_u64(SEED ^ 0xABCD);
    let hot_cluster = n_clusters - 1;
    let hot_queries: Vec<Vec<f32>> = (0..N_HOT_QUERIES)
        .map(|_| noisy_sample(&mut qgen, &centroids[hot_cluster], NOISE))
        .collect();

    // Ground truth for recall@10: on a *fresh, fully populated* (pre-any-
    // compaction) store built from the same schedule, the true top-10
    // nearest neighbors per hot query. This is independent of which trigger
    // is under test.
    let truth_ids_by_query: Vec<Vec<u64>> = {
        let mut rng = StdRng::seed_from_u64(SEED ^ 0x1234_5678);
        let mut full_store = MemoryStore::new(DIMS);
        for ev in &schedule {
            let v = match *ev {
                Event::Quiet(c) => noisy_sample(&mut rng, &centroids[c], NOISE),
                Event::Burst(c) => noisy_sample(&mut rng, &centroids[c], NOISE),
            };
            full_store.insert(v);
        }
        hot_queries
            .iter()
            .map(|q| full_store.search(q, K).into_iter().map(|r| r.id).collect())
            .collect()
    };

    // Calibrate the structural threshold from an initial quiet baseline
    // slice of the actual schedule (the first QUIET_EPOCH writes of cluster
    // 0), before running any full comparison.
    let calibration_writes: Vec<Vec<f32>> = {
        let mut rng = StdRng::seed_from_u64(SEED ^ 0x1234_5678);
        (0..STRUCTURAL_CALIBRATION_WRITES)
            .map(|_| noisy_sample(&mut rng, &centroids[0], NOISE))
            .collect()
    };
    let metric = StructuralMetric::default();
    let structural_threshold = StructuralGateTrigger::calibrate_threshold(
        metric,
        &calibration_writes,
        STRUCTURAL_WINDOW,
        STRUCTURAL_MULTIPLIER,
    );

    println!("=== Structural-Time-Gated Compaction Scheduling ===");
    println!(
        "writes={} clusters={} target_size={} dims={}",
        schedule.len(),
        n_clusters,
        TARGET_SIZE,
        DIMS
    );
    println!("calibrated structural_threshold = {structural_threshold:.6}\n");

    let triggers: Vec<(&str, Box<dyn CompactionTrigger>)> = vec![
        (
            "FixedInterval",
            Box::new(FixedIntervalTrigger::new(FIXED_INTERVAL)),
        ),
        (
            "Capacity",
            Box::new(CapacityTrigger::new(TARGET_SIZE * CAPACITY_MULTIPLIER)),
        ),
        (
            "StructuralGate",
            Box::new(StructuralGateTrigger::new(
                metric,
                structural_threshold,
                STRUCTURAL_WINDOW,
            )),
        ),
    ];

    let mut results = Vec::new();
    for (name, trigger) in triggers {
        let r = run_trigger(
            name,
            trigger,
            &schedule,
            &centroids,
            &hot_queries,
            &truth_ids_by_query,
        );
        results.push(r);
    }

    println!(
        "{:<16} {:>12} {:>20} {:>14} {:>12} {:>10}",
        "trigger", "compactions", "excess_size_integral", "recall@10", "wall_ms", "final_size"
    );
    for r in &results {
        println!(
            "{:<16} {:>12} {:>20} {:>14.4} {:>12.3} {:>10}",
            r.name,
            r.n_compactions,
            r.excess_size_integral,
            r.final_recall_at_10,
            r.wall_time.as_secs_f64() * 1000.0,
            r.final_store_size
        );
    }

    // Diagnostic only (not part of acceptance): where do fires land? 1200 of
    // the 1400 writes are Quiet events, 200 are Burst events. A trigger that
    // tracks genuine regime structure should fire overwhelmingly during
    // Burst events despite them being the minority of writes.
    println!("\n=== Diagnostic: fire location (1200 Quiet writes, 200 Burst writes) ===");
    println!(
        "{:<16} {:>14} {:>14}",
        "trigger", "fires@quiet", "fires@burst"
    );
    for r in &results {
        println!(
            "{:<16} {:>14} {:>14}",
            r.name, r.fires_during_quiet, r.fires_during_burst
        );
    }

    // Acceptance evaluation against the pre-registered thresholds above.
    let fixed = results.iter().find(|r| r.name == "FixedInterval").unwrap();
    let structural = results.iter().find(|r| r.name == "StructuralGate").unwrap();

    let calls_reduction =
        1.0 - (structural.n_compactions as f64 / fixed.n_compactions.max(1) as f64);
    let excess_reduction =
        1.0 - (structural.excess_size_integral as f64 / fixed.excess_size_integral.max(1) as f64);
    let recall_gap = (fixed.final_recall_at_10 - structural.final_recall_at_10).abs();
    let wall_ratio = structural.wall_time.as_secs_f64() / fixed.wall_time.as_secs_f64().max(1e-9);

    println!("\n=== Acceptance (vs. FixedInterval, thresholds fixed pre-run) ===");
    println!(
        "compaction-call reduction:   {:>7.1}%  (need >= 20%)",
        calls_reduction * 100.0
    );
    println!(
        "excess-size-integral reduction: {:>7.1}%  (need >= 20%)",
        excess_reduction * 100.0
    );
    println!(
        "recall@10 gap:                {:>7.4}  (need <= 0.02)",
        recall_gap
    );
    println!(
        "wall-clock ratio (struct/fixed): {:>7.2}x (need <= 2.0x)",
        wall_ratio
    );

    let accept = calls_reduction >= 0.20
        && excess_reduction >= 0.20
        && recall_gap <= 0.02
        && wall_ratio <= 2.0;

    println!(
        "\nACCEPTANCE RESULT: {}",
        if accept { "ACCEPT" } else { "REJECT" }
    );
}
