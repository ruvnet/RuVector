//! Nightly research benchmark (2026-08-29, ADR-340): does a structural-time
//! recency signal beat wall-clock recency for memory compaction under a
//! **bursty-idle** workload?
//!
//! Hypothesis (pre-registered, not adjusted after seeing results):
//!
//!   Given a memory stream with a long idle gap between a dense "phase 1"
//!   writing burst and a small "phase 2" burst right after the agent returns,
//!
//!   when compaction recency is computed from `emergent_time`'s Structural
//!   Proper Time (embedding arc length) instead of `last_accessed_at`,
//!
//!   then Recall@K for queries about the *end of phase 1* (what the agent was
//!   working on right before the gap) should exceed the wall-clock
//!   `CoherencePolicy` baseline by at least 3.0 percentage points,
//!
//!   subject to: on a steady (no-idle-gap) control workload of the same
//!   size, the structural-time policy must not regress recall by more than
//!   1.0pp relative to the baseline (no free lunch — it must not be winning
//!   only because it changed behavior on ordinary workloads too).
//!
//! Mechanism: `CoherenceWeights`-style recency is min-max normalized over the
//! whole store. A single idle gap that is orders of magnitude larger than the
//! wall-clock ticks spent on real writes dominates that range, so every
//! pre-gap memory's recency score collapses toward 0 — indistinguishable from
//! genuinely stale memories — regardless of how close to the gap (and how
//! relevant) it actually is. Structural Proper Time accumulates only with
//! embedding movement, so an idle gap with zero new memories contributes zero
//! structural time: pre-gap memories keep a recency score reflecting their
//! true position in the write sequence.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example temporal_compaction_bench

use emergent_time::witness::fnv1a64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, CoherencePolicy, CompactionPolicy, GatedStructuralTimePolicy, LruPolicy,
    MemoryStore, StructuralTimePolicy,
};
use std::time::Instant;

// ── Shared dataset parameters ──────────────────────────────────────────────
const DIMS: usize = 64;
// Phase 1 ("before the break"): dense writing burst across many topics.
const N_PHASE1_CLUSTERS: usize = 15;
// Phase 2 ("after the break"): the agent's *current* activity — a smaller,
// differently-themed burst. The compaction context window is drawn from
// here, never from the recall clusters, so coherence cannot leak the answer.
const N_PHASE2_CLUSTERS: usize = 5;
const N_CLUSTERS: usize = N_PHASE1_CLUSTERS + N_PHASE2_CLUSTERS; // 20
const ENTRIES_PER_CLUSTER: usize = 100;
const N_MEMORIES: usize = N_CLUSTERS * ENTRIES_PER_CLUSTER; // 2 000
                                                            // Recall clusters: the last N_RECALL_CLUSTERS topics written in phase 1 —
                                                            // "what the agent was working on right before the break" — evaluated by a
                                                            // held-out query set the compaction policy never sees.
const N_RECALL_CLUSTERS: usize = 3;
const N_QUERIES_PER_CLUSTER: usize = 20;
const K: usize = 10;
// Aggressive compaction: phase 2 (500 entries) plus all of phase 1 (1 500)
// cannot both fit in 700 slots, so recall-cluster survival is a real contest,
// not a formality (a 50% ratio left every policy at a 100% recall ceiling in
// an earlier pilot run — not discriminative, so it was tightened here before
// any acceptance numbers were computed).
const TARGET_SIZE: usize = 700;
const IDLE_GAP_TICKS: u64 = 500_000; // >> the ~2 000 ticks spent on real writes
const RECALL_MARGIN_PP: f32 = 3.0; // pre-registered acceptance margin
const REGRESSION_TOLERANCE_PP: f32 = 1.0; // pre-registered non-regression tolerance

// ── Vector helpers ──────────────────────────────────────────────────────────

fn unit_gaussian(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    normalize_vec(&v)
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

// ── Dataset: two workloads sharing one generator, differing only in
//    whether an idle gap separates phase 1 from phase 2 ───────────────────

struct Workload {
    name: &'static str,
    store: MemoryStore,
    context_window: Vec<Vec<f32>>,
    queries: Vec<(Vec<f32>, Vec<u64>)>,
}

/// `gap_ticks = 0` builds the steady control workload; `gap_ticks > 0` builds
/// the bursty-idle workload. Both write the same 20 clusters x 100 entries in
/// the same order with the same noise; only the wall clock between the last
/// phase-1 cluster and the first phase-2 cluster differs.
fn build_workload(name: &'static str, gap_ticks: u64, seed: u64) -> Workload {
    let mut rng = StdRng::seed_from_u64(seed);
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| unit_gaussian(&mut rng, DIMS))
        .collect();

    let mut store = MemoryStore::new(DIMS);
    for (c, centroid) in centroids.iter().enumerate() {
        if c == N_PHASE1_CLUSTERS {
            // The gap happens once, between phase 1 and phase 2.
            store.advance_clock(gap_ticks);
        }
        for _ in 0..ENTRIES_PER_CLUSTER {
            let v = perturb(centroid, 0.35, &mut rng);
            store.insert(v);
        }
    }

    // Recall clusters: the last N_RECALL_CLUSTERS topics written in phase 1 —
    // held out and evaluated after compaction, never shown to the policy.
    let recall_cluster_ids: Vec<usize> =
        ((N_PHASE1_CLUSTERS - N_RECALL_CLUSTERS)..N_PHASE1_CLUSTERS).collect();
    let mut queries = Vec::with_capacity(N_RECALL_CLUSTERS * N_QUERIES_PER_CLUSTER);
    for &c in &recall_cluster_ids {
        for _ in 0..N_QUERIES_PER_CLUSTER {
            let q = perturb(&centroids[c], 0.30, &mut rng);
            let truth: Vec<u64> = store.search(&q, K).into_iter().map(|r| r.id).collect();
            queries.push((q, truth));
        }
    }

    // Context window: what the agent is *currently* discussing, drawn only
    // from phase-2 topics. Never overlaps the recall clusters, so
    // `CoherencePolicy`'s coherence term cannot leak the evaluation answer —
    // it actively favors phase 2 over the (semantically unrelated) recall
    // clusters, isolating the recency term as the only lever that can save
    // pre-gap memories.
    let mut context_window = Vec::new();
    for centroid in &centroids[N_PHASE1_CLUSTERS..N_CLUSTERS] {
        for _ in 0..4 {
            context_window.push(perturb(centroid, 0.30, &mut rng));
        }
    }

    Workload {
        name,
        store,
        context_window,
        queries,
    }
}

// ── Evaluation ────────────────────────────────────────────────────────────

fn measure_recall(queries: &[(Vec<f32>, Vec<u64>)], store: &MemoryStore) -> f32 {
    let mut total = 0.0f32;
    for (q, truth) in queries {
        let candidates: Vec<u64> = store.search(q, K).into_iter().map(|r| r.id).collect();
        total += recall_at_k(truth, &candidates);
    }
    total / queries.len() as f32
}

struct PolicyResult {
    name: String,
    recall: f32,
    compaction_us: u64,
}

fn run_all_policies(workload: &Workload) -> Vec<PolicyResult> {
    let cow = CoherencePolicy::default();
    let structural = StructuralTimePolicy::default();
    let gated = GatedStructuralTimePolicy::default();
    let policies: Vec<&dyn CompactionPolicy> = vec![&LruPolicy, &cow, &structural, &gated];

    let mut results = Vec::with_capacity(policies.len());
    for policy in policies {
        let mut store = MemoryStore::new(DIMS);
        // Clone entries verbatim (including `created_at`/`last_accessed_at`,
        // which carry the idle-gap timestamps) since `compact()` mutates in
        // place and every policy must see the identical dataset. Re-inserting
        // through `MemoryStore::insert` would retimestamp from a fresh
        // sequential clock and silently discard the gap.
        store.replace_entries(workload.store.entries().to_vec());

        let t0 = Instant::now();
        compact(&mut store, policy, TARGET_SIZE, &workload.context_window);
        let dt = t0.elapsed();
        assert_eq!(store.len(), TARGET_SIZE);

        let recall = measure_recall(&workload.queries, &store);
        results.push(PolicyResult {
            name: policy.name().to_string(),
            recall,
            compaction_us: dt.as_micros() as u64,
        });
    }
    results
}

fn print_results(workload_name: &str, results: &[PolicyResult]) {
    println!("\n[{workload_name}]");
    println!(
        "{:<22} {:>12} {:>18}",
        "Policy", "Recall@10", "Compaction (µs)"
    );
    println!("{}", "-".repeat(56));
    for r in results {
        println!(
            "{:<22} {:>11.1}% {:>17}",
            r.name,
            r.recall * 100.0,
            r.compaction_us
        );
    }
}

fn find<'a>(results: &'a [PolicyResult], name: &str) -> &'a PolicyResult {
    results
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("policy {name} missing from results"))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Structural-Time Memory Compaction — Nightly Benchmark        ║");
    println!("║  ADR-340 / 2026-08-29                                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Platform : {} / {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!();
    println!("Dataset");
    println!(
        "  Memories            : {N_MEMORIES} ({N_CLUSTERS} clusters x {ENTRIES_PER_CLUSTER})"
    );
    println!("  Dimensions          : {DIMS}");
    println!("  Recall clusters     : last {N_RECALL_CLUSTERS} clusters before the gap/tail");
    println!(
        "  Queries             : {} (K={K})",
        N_RECALL_CLUSTERS * N_QUERIES_PER_CLUSTER
    );
    println!(
        "  Target size         : {TARGET_SIZE} ({:.0}% compaction)",
        100.0 * TARGET_SIZE as f32 / N_MEMORIES as f32
    );
    println!("  Idle gap            : {IDLE_GAP_TICKS} ticks (steady workload: 0)");
    println!();

    let seed = 340u64;
    let idle = build_workload("bursty-idle", IDLE_GAP_TICKS, seed);
    let steady = build_workload("steady (control)", 0, seed);

    let idle_results = run_all_policies(&idle);
    let steady_results = run_all_policies(&steady);

    print_results(idle.name, &idle_results);
    print_results(steady.name, &steady_results);

    // ── Acceptance ──────────────────────────────────────────────────────
    let idle_cow = find(&idle_results, "CoherenceWeighted").recall;
    let idle_struct = find(&idle_results, "StructuralTime").recall;
    let idle_gated = find(&idle_results, "GatedStructuralTime").recall;
    let steady_cow = find(&steady_results, "CoherenceWeighted").recall;
    let steady_struct = find(&steady_results, "StructuralTime").recall;

    let idle_delta_a = (idle_struct - idle_cow) * 100.0;
    let idle_delta_b = (idle_gated - idle_cow) * 100.0;
    let steady_delta = (steady_struct - steady_cow) * 100.0;

    let pass_a = idle_delta_a >= RECALL_MARGIN_PP;
    let pass_b = idle_delta_b >= RECALL_MARGIN_PP;
    let pass_no_regression = steady_delta >= -REGRESSION_TOLERANCE_PP;

    println!("\nAcceptance test (pre-registered thresholds)");
    println!(
        "  [A] StructuralTime beats CoherenceWeighted by >= {RECALL_MARGIN_PP:.1}pp on bursty-idle: {:+.1}pp -> {}",
        idle_delta_a,
        if pass_a { "PASS" } else { "FAIL" }
    );
    println!(
        "  [B] GatedStructuralTime beats CoherenceWeighted by >= {RECALL_MARGIN_PP:.1}pp on bursty-idle: {:+.1}pp -> {}",
        idle_delta_b,
        if pass_b { "PASS" } else { "FAIL" }
    );
    println!(
        "  [C] StructuralTime does not regress > {REGRESSION_TOLERANCE_PP:.1}pp vs CoherenceWeighted on steady control: {:+.1}pp -> {}",
        steady_delta,
        if pass_no_regression { "PASS" } else { "FAIL" }
    );

    let verdict = if pass_a && pass_no_regression {
        "ACCEPT"
    } else if !pass_no_regression {
        "REJECT (regresses steady-state recall)"
    } else {
        "REJECT (does not beat baseline on bursty-idle workload)"
    };
    println!("\n  Verdict: {verdict}");

    // ── Witness seal: FNV-1a hash over (seed, params, rounded results) so a
    // re-run with the same seed/params can be checked byte-for-byte against
    // this evidence (reuses emergent_time::witness::fnv1a64 — no new hash). ──
    let mut payload = Vec::new();
    payload.extend_from_slice(&seed.to_le_bytes());
    payload.extend_from_slice(&(N_MEMORIES as u64).to_le_bytes());
    payload.extend_from_slice(&IDLE_GAP_TICKS.to_le_bytes());
    for r in idle_results.iter().chain(steady_results.iter()) {
        payload.extend_from_slice(&((r.recall * 1e6).round() as i64).to_le_bytes());
    }
    let witness_hash = fnv1a64(&payload);
    println!("\nWitness (FNV-1a over seed+params+rounded recall values): {witness_hash:016x}");
    println!("  Reproduce with: cargo run --release -p ruvector-agent-memory --example temporal_compaction_bench");

    if !(pass_a && pass_no_regression) {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the acceptance test above so `cargo test` also exercises it
    /// (in debug mode, so this checks correctness/direction, not the release
    /// benchmark's exact numbers).
    #[test]
    fn structural_time_beats_wall_clock_recency_under_idle_gap() {
        let seed = 340u64;
        let idle = build_workload("bursty-idle", IDLE_GAP_TICKS, seed);
        let results = run_all_policies(&idle);
        let cow = find(&results, "CoherenceWeighted").recall;
        let structural = find(&results, "StructuralTime").recall;
        assert!(
            structural > cow,
            "structural-time recall ({:.3}) should exceed wall-clock recall ({:.3}) under an idle gap",
            structural,
            cow
        );
    }

    #[test]
    fn steady_workload_does_not_regress() {
        let seed = 340u64;
        let steady = build_workload("steady", 0, seed);
        let results = run_all_policies(&steady);
        let cow = find(&results, "CoherenceWeighted").recall;
        let structural = find(&results, "StructuralTime").recall;
        assert!(
            structural >= cow - REGRESSION_TOLERANCE_PP / 100.0,
            "structural-time recall ({:.3}) should not regress vs wall-clock ({:.3}) with no idle gap",
            structural,
            cow
        );
    }

    /// Robustness check: the direction of the effect (structural-time beats
    /// wall-clock recency under the idle gap) must hold across independently
    /// seeded dataset draws, not just the one seed shipped in `main`.
    #[test]
    fn structural_time_wins_across_multiple_seeds() {
        for seed in [1u64, 2, 3, 4, 5, 340, 7777, 99991] {
            let idle = build_workload("bursty-idle", IDLE_GAP_TICKS, seed);
            let results = run_all_policies(&idle);
            let cow = find(&results, "CoherenceWeighted").recall;
            let structural = find(&results, "StructuralTime").recall;
            assert!(
                structural > cow,
                "seed {seed}: structural-time recall ({:.3}) should exceed wall-clock recall ({:.3})",
                structural,
                cow
            );
        }
    }
}
