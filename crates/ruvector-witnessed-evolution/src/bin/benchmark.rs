//! Witnessed Evolution — benchmark binary.
//!
//! Given: a fixed, seeded `ruvector-coherence-hnsw` workload (clustered
//! dataset, flat k-NN graph, queries, brute-force ground truth).
//!
//! When: a (1+1)-ES searches the coherence-threshold/beam-width genome for
//! `N_GENERATIONS` mutation attempts, once unwitnessed (raw) and once
//! witnessed (every generation committed to a `ruvector-proof-gate` hash
//! chain via `WitnessedLineage`), both from the identical seed.
//!
//! Then: the witnessed run should reach the byte-identical optimum as the
//! unwitnessed run, at bounded wall-clock overhead, while producing a
//! lineage that an independent replayer can verify — and a single tampered
//! byte in that lineage must be caught.
//!
//! Usage: `cargo run --release -p ruvector-witnessed-evolution --bin benchmark`

use std::time::Instant;

use ruvector_coherence_hnsw::metrics::LatencyStats;
use ruvector_coherence_hnsw::search::{CoherenceGatedSearch, Searcher};
use ruvector_witnessed_evolution::{
    evolve::{run_unwitnessed, run_witnessed},
    genome::DEFAULT_GENOME,
    Workload, WorkloadConfig,
};

// ─── Dataset parameters (fixed before any run) ────────────────────────────
const N_CLUSTERS: usize = 8;
const N_PER_CLUSTER: usize = 250; // 2000 vectors total, matches the coherence-hnsw benchmark
const DIMS: usize = 32;
const CLUSTER_STD: f32 = 0.15;
const M: usize = 16;
const M_LONGJUMP: usize = 6;
const N_QUERIES: usize = 150;
const K: usize = 10;
const ENTRY: usize = 0;
const DATA_SEED: u64 = 0xDEAD_BEEF;
const QUERY_SEED: u64 = 0xCAFE_BABE;

// ─── Search parameters (fixed before any run) ──────────────────────────────
const N_GENERATIONS: usize = 40;
const ES_SEED: u64 = 0x5EED_1234;
const LATENCY_REPS: usize = 3; // repeated timing passes per genome, report the min pass

// ─── Acceptance thresholds (fixed before any run) ──────────────────────────
const MAX_WITNESS_OVERHEAD_PCT: f64 = 15.0;

fn main() {
    println!("=== Witnessed Evolution: Merkle-Chained Provenance for ANN Parameter Search ===\n");

    eprintln!(
        "[bench] Building workload: {N_CLUSTERS} clusters x {N_PER_CLUSTER} = {} vectors, D={DIMS}, {N_QUERIES} queries, k={K}...",
        N_CLUSTERS * N_PER_CLUSTER
    );
    let workload = Workload::build(&WorkloadConfig {
        n_clusters: N_CLUSTERS,
        n_per_cluster: N_PER_CLUSTER,
        dims: DIMS,
        cluster_std: CLUSTER_STD,
        m: M,
        m_longjump: M_LONGJUMP,
        n_queries: N_QUERIES,
        k: K,
        entry_id: ENTRY,
        data_seed: DATA_SEED,
        query_seed: QUERY_SEED,
    });

    // ─── Baseline: hand-picked fixed genome, no search at all ──────────────
    let baseline_fitness = workload.evaluate(DEFAULT_GENOME);
    let baseline_latency = measure_latency(
        &workload,
        DEFAULT_GENOME.threshold,
        DEFAULT_GENOME.ef_usize(),
    );
    println!(
        "[baseline]      threshold={:.3} ef={:>3}  recall={:.4}  avg_expansions={:.1}  composite={:.4}  p50={:.1}us",
        DEFAULT_GENOME.threshold,
        DEFAULT_GENOME.ef_usize(),
        baseline_fitness.recall_mean,
        baseline_fitness.avg_expansions,
        baseline_fitness.composite,
        baseline_latency.p50_us(),
    );

    // ─── Candidate A: unwitnessed (1+1)-ES ──────────────────────────────────
    let t0 = Instant::now();
    let unwitnessed = run_unwitnessed(&workload, N_GENERATIONS, ES_SEED);
    let unwitnessed_wall = t0.elapsed();
    let a_latency = measure_latency(
        &workload,
        unwitnessed.best_genome.threshold,
        unwitnessed.best_genome.ef_usize(),
    );
    println!(
        "[candidate_A]    threshold={:.3} ef={:>3}  recall={:.4}  avg_expansions={:.1}  composite={:.4}  p50={:.1}us  wall={:.2}ms  ({} generations, unwitnessed)",
        unwitnessed.best_genome.threshold,
        unwitnessed.best_genome.ef_usize(),
        unwitnessed.best_fitness.recall_mean,
        unwitnessed.best_fitness.avg_expansions,
        unwitnessed.best_fitness.composite,
        a_latency.p50_us(),
        unwitnessed_wall.as_secs_f64() * 1000.0,
        N_GENERATIONS,
    );

    // ─── Candidate B: witnessed (1+1)-ES, identical seed ────────────────────
    let t1 = Instant::now();
    let (witnessed, lineage) = run_witnessed(&workload, N_GENERATIONS, ES_SEED);
    let witnessed_wall = t1.elapsed();
    let b_latency = measure_latency(
        &workload,
        witnessed.best_genome.threshold,
        witnessed.best_genome.ef_usize(),
    );
    println!(
        "[candidate_B]    threshold={:.3} ef={:>3}  recall={:.4}  avg_expansions={:.1}  composite={:.4}  p50={:.1}us  wall={:.2}ms  ({} generations, witnessed, chain_len={})",
        witnessed.best_genome.threshold,
        witnessed.best_genome.ef_usize(),
        witnessed.best_fitness.recall_mean,
        witnessed.best_fitness.avg_expansions,
        witnessed.best_fitness.composite,
        b_latency.p50_us(),
        witnessed_wall.as_secs_f64() * 1000.0,
        N_GENERATIONS,
        lineage.len(),
    );

    let overhead_pct =
        (witnessed_wall.as_secs_f64() / unwitnessed_wall.as_secs_f64() - 1.0) * 100.0;
    println!("\nwitnessing overhead: {overhead_pct:.2}% wall-clock ({unwitnessed_wall:?} unwitnessed vs {witnessed_wall:?} witnessed)");
    println!("chain root: {}", hex(&lineage.chain_root()));

    // ─── Replay verification: honest lineage ────────────────────────────────
    let honest_report = lineage.replay_verify(&workload);
    println!(
        "\nreplay_verify(honest lineage)   -> verified={} chain_integrity={} first_divergence={:?} ({} generations checked)",
        honest_report.verified,
        honest_report.chain_integrity_ok,
        honest_report.first_divergence,
        honest_report.generations_checked,
    );

    // ─── Replay verification: tampered lineage (adversarial test) ──────────
    let mut tampered_lineage = run_witnessed(&workload, N_GENERATIONS, ES_SEED).1;
    let tamper_idx = N_GENERATIONS / 2;
    let forged = tampered_lineage.records()[tamper_idx].fitness.composite + 0.5;
    tampered_lineage.tamper_composite(tamper_idx, forged);
    let tampered_report = tampered_lineage.replay_verify(&workload);
    println!(
        "replay_verify(tampered gen {tamper_idx}) -> verified={} first_divergence={:?}  (forged composite {forged:.4} into an otherwise-honest chain)",
        tampered_report.verified, tampered_report.first_divergence,
    );

    // ─── Acceptance ──────────────────────────────────────────────────────────
    let identical = unwitnessed.best_genome == witnessed.best_genome
        && unwitnessed.best_fitness == witnessed.best_fitness;
    let beats_baseline = witnessed.best_fitness.composite > baseline_fitness.composite;
    let overhead_ok = overhead_pct <= MAX_WITNESS_OVERHEAD_PCT;
    let honest_verifies = honest_report.verified;
    let tamper_detected =
        !tampered_report.verified && tampered_report.first_divergence == Some(tamper_idx);

    println!("\n=== Acceptance ===");
    println!("  witnessed run bit-identical to unwitnessed run : {identical}");
    println!(
        "  witnessed ES beats fixed baseline               : {beats_baseline}  ({:.4} vs {:.4})",
        witnessed.best_fitness.composite, baseline_fitness.composite
    );
    println!("  witnessing overhead <= {MAX_WITNESS_OVERHEAD_PCT:.1}%                : {overhead_ok}  (measured {overhead_pct:.2}%)");
    println!("  honest lineage replay-verifies                  : {honest_verifies}");
    println!("  tampered lineage is caught at the tampered gen  : {tamper_detected}");

    let all_mandatory = identical && beats_baseline && honest_verifies && tamper_detected;
    let verdict = if !all_mandatory {
        "REJECT"
    } else if !overhead_ok {
        // Correctness/security all hold; only the overhead budget missed.
        "REJECT"
    } else {
        "ACCEPT"
    };
    println!("\nACCEPTANCE RESULT: {verdict}");
}

fn measure_latency(workload: &Workload, threshold: f32, ef: usize) -> LatencyStats {
    let searcher = CoherenceGatedSearch { threshold };
    let mut best: Option<LatencyStats> = None;
    for _ in 0..LATENCY_REPS {
        let mut samples = Vec::with_capacity(workload.queries.len());
        for q in &workload.queries {
            let t = Instant::now();
            let _ = searcher.search(&workload.graph, q, workload.k, ef, workload.entry_id);
            samples.push(t.elapsed().as_nanos() as u64);
        }
        let stats = LatencyStats::compute(samples);
        best = Some(match best {
            Some(b) if b.p50_ns <= stats.p50_ns => b,
            _ => stats,
        });
    }
    best.expect("LATENCY_REPS > 0")
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
