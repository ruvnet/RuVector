//! Nightly research 2026-10-08: does ACORN's low-selectivity fix generalise
//! from generic metadata predicates (`ruvector-acorn`, 2026-04-26 nightly) to
//! capability-token access control (`ruvector-capgated`)?
//!
//! `CapGraphIndex` already implements ACORN's core traversal insight
//! (expand neighbours regardless of authorisation, so the beam doesn't
//! starve). What it has NOT adopted is ACORN's other lever — γ-augmented
//! degree — and it still seeds search from fixed, evenly-spaced-by-index
//! entry points rather than a navigable hierarchy (the crate's own
//! documented "replace with HNSW for production" TODO).
//!
//! This benchmark freezes two hypotheses before running:
//!
//! ```text
//! Given a 64-bit-capability corpus (n=4000, d=64) where each vector
//! requires exactly one capability bit and a querier holds exactly one bit
//! (low-access scenario, authorised fraction ~= 1/64 = 1.5625%),
//!
//! H1 (gamma-augmented degree):
//!   when CapGraphIndex is built with degree=48 (gamma=4x the baseline's
//!   degree=12) instead of degree=12, holding traversal and ef budget fixed,
//!   then recall@10 should increase by >= 10 percentage points vs baseline,
//!
//! H2 (hierarchical seeding):
//!   when a sparse top layer (1-in-16 nodes) is used to find the base-layer
//!   entry point by greedy descent, instead of fixed evenly-spaced indices,
//!   holding degree and ef budget fixed,
//!   then recall@10 should increase by >= 5 percentage points vs baseline,
//!
//! subject to: build time for either candidate staying within 3x of
//! baseline, QPS not dropping below 30% (H1) / 70% (H2) of baseline, and no
//! recall regression > 2 points at the existing high-access scenario
//! (N_CAPS=8, held=3, auth ~= 37.5%) relative to baseline.
//! ```
//!
//! Run: `cargo run --release -p ruvector-capgated --example selectivity_acorn_bench`
use ruvector_capgated::{
    cap_graph::CapGraphIndex,
    dataset::{generate, generate_queries, DatasetConfig},
    hierarchical::HierarchicalCapGraphIndex,
    oracle::Oracle,
    recall_at_k, CapGatedIndex, CapMask,
};
use std::time::Instant;

const N_VECTORS: usize = 4_000;
const DIMS: usize = 64;
const N_QUERIES: usize = 150;
const K: usize = 10;
const BASE_DEGREE: usize = 12;
const GAMMA: usize = 4; // candidate_A degree = BASE_DEGREE * GAMMA
const ENTRY_POINTS: usize = 8;
const PROMOTE_RATIO: usize = 16; // candidate_B top layer = n / PROMOTE_RATIO nodes
const SEED: u64 = 0x5e1e_c7f1_17f0_a1b2;

// Frozen before running; not adjusted after seeing results.
const H1_MIN_DELTA_PP: f32 = 0.10;
const H2_MIN_DELTA_PP: f32 = 0.05;
const MAX_BUILD_RATIO: f64 = 3.0;
const H1_MIN_QPS_RATIO: f64 = 0.30;
const H2_MIN_QPS_RATIO: f64 = 0.70;
const MAX_HIGH_ACCESS_REGRESSION_PP: f32 = 0.02;

struct Scenario {
    name: &'static str,
    n_caps: u8,
    held_caps: u8,
}

const SCENARIOS: [Scenario; 2] = [
    Scenario {
        name: "high-access (3/8 caps, ~37.5%)",
        n_caps: 8,
        held_caps: 3,
    },
    Scenario {
        name: "low-access (1/64 caps, ~1.56%)",
        n_caps: 64,
        held_caps: 1,
    },
];

struct Measured {
    variant: &'static str,
    build_ms: f64,
    mean_us: f64,
    qps: f64,
    recall: f32,
    mem_mb: f64,
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * p / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn mem_mb(n: usize, dims: usize, degree: usize, extra_edges: usize) -> f64 {
    let vec_bytes = n * dims * 4;
    let graph_bytes = (n * degree + extra_edges) * 8;
    let cap_bytes = n * 8;
    (vec_bytes + graph_bytes + cap_bytes) as f64 / 1_048_576.0
}

fn bench(
    idx: &mut dyn CapGatedIndex,
    queries: &[Vec<f32>],
    oracle_results: &[Vec<ruvector_capgated::SearchResult>],
    holder: CapMask,
) -> (f32, f64, f64) {
    let mut latencies_us: Vec<u128> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;
    for (q, oracle) in queries.iter().zip(oracle_results.iter()) {
        let t0 = Instant::now();
        let res = idx.search(q, K, holder);
        latencies_us.push(t0.elapsed().as_micros());
        total_recall += recall_at_k(oracle, &res, K);
    }
    latencies_us.sort_unstable();
    let mean_us = latencies_us.iter().sum::<u128>() as f64 / latencies_us.len() as f64;
    let qps = 1_000_000.0 / mean_us;
    let recall = total_recall / queries.len() as f32;
    (recall, mean_us, qps)
}

fn run_scenario(scenario: &Scenario) -> Vec<Measured> {
    let cfg = DatasetConfig {
        n_vectors: N_VECTORS,
        dims: DIMS,
        n_caps: scenario.n_caps,
        required_per_vector: 1,
        seed: SEED,
    };
    let entries = generate(&cfg);
    let (queries, holder) = generate_queries(
        N_QUERIES,
        DIMS,
        scenario.n_caps,
        scenario.held_caps,
        SEED ^ 0xabcd,
    );
    let auth_count = entries
        .iter()
        .filter(|e| holder.satisfies(e.required))
        .count();
    let auth_frac = auth_count as f32 / N_VECTORS as f32;
    println!(
        "\n  Holder: 0b{:064b} (popcount {}) | Authorised: {}/{} ({:.3}%)",
        holder.0,
        holder.count(),
        auth_count,
        N_VECTORS,
        auth_frac * 100.0
    );

    let mut oracle = Oracle::new(DIMS);
    for e in &entries {
        oracle.insert(e.id, e.vector.clone(), e.required);
    }
    let oracle_results: Vec<Vec<_>> = queries
        .iter()
        .map(|q| oracle.search(q, K, holder))
        .collect();

    let mut out = Vec::new();

    // ─── baseline: CapGraph, degree=12 ─────────────────────────────────
    let t0 = Instant::now();
    let mut baseline = CapGraphIndex::new(DIMS, BASE_DEGREE, ENTRY_POINTS);
    baseline.batch_build(entries.iter().map(|e| (e.id, e.vector.clone(), e.required)));
    let baseline_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (rec, mean_us, qps) = bench(&mut baseline, &queries, &oracle_results, holder);
    out.push(Measured {
        variant: "baseline: CapGraph(deg=12)",
        build_ms: baseline_build_ms,
        mean_us,
        qps,
        recall: rec,
        mem_mb: mem_mb(N_VECTORS, DIMS, BASE_DEGREE, 0),
    });

    // ─── candidate_A: gamma-augmented degree ───────────────────────────
    let gamma_degree = BASE_DEGREE * GAMMA;
    let t0 = Instant::now();
    let mut cand_a = CapGraphIndex::new(DIMS, gamma_degree, ENTRY_POINTS);
    cand_a.batch_build(entries.iter().map(|e| (e.id, e.vector.clone(), e.required)));
    let cand_a_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (rec, mean_us, qps) = bench(&mut cand_a, &queries, &oracle_results, holder);
    out.push(Measured {
        variant: "candidate_A: CapGraph(deg=48, gamma=4)",
        build_ms: cand_a_build_ms,
        mean_us,
        qps,
        recall: rec,
        mem_mb: mem_mb(N_VECTORS, DIMS, gamma_degree, 0),
    });

    // ─── candidate_B: hierarchical seeding, same degree as baseline ────
    let t0 = Instant::now();
    let mut cand_b = HierarchicalCapGraphIndex::new(DIMS, BASE_DEGREE, PROMOTE_RATIO);
    cand_b.batch_build(entries.iter().map(|e| (e.id, e.vector.clone(), e.required)));
    let cand_b_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (rec, mean_us, qps) = bench(&mut cand_b, &queries, &oracle_results, holder);
    let top_edges = (N_VECTORS / PROMOTE_RATIO).max(1) * BASE_DEGREE;
    out.push(Measured {
        variant: "candidate_B: HierarchicalCapGraph(deg=12,1/16 top)",
        build_ms: cand_b_build_ms,
        mean_us,
        qps,
        recall: rec,
        mem_mb: mem_mb(N_VECTORS, DIMS, BASE_DEGREE, top_edges),
    });

    out
}

fn main() {
    println!("════════════════════════════════════════════════════════════════════");
    println!("  ruvector-capgated: ACORN-style selectivity audit (nightly 2026-10-08)");
    println!("════════════════════════════════════════════════════════════════════");
    println!(
        "  OS: {} | Arch: {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("  n={N_VECTORS} d={DIMS} queries={N_QUERIES} k={K}");
    println!(
        "  baseline degree={BASE_DEGREE} | candidate_A degree={} (gamma={GAMMA})",
        BASE_DEGREE * GAMMA
    );
    println!("  candidate_B top-layer: 1-in-{PROMOTE_RATIO} nodes");
    println!("════════════════════════════════════════════════════════════════════");

    let mut by_scenario: Vec<(String, Vec<Measured>)> = Vec::new();
    for scenario in &SCENARIOS {
        println!("\n▶ Scenario: {}", scenario.name);
        let results = run_scenario(scenario);
        println!(
            "  {:<42} {:>10} {:>10} {:>10} {:>8} {:>8}",
            "Variant", "Build(ms)", "Mean(us)", "QPS", "Recall", "Mem(MB)"
        );
        for r in &results {
            println!(
                "  {:<42} {:>10.1} {:>10.1} {:>10.0} {:>8.3} {:>8.2}",
                r.variant, r.build_ms, r.mean_us, r.qps, r.recall, r.mem_mb
            );
        }
        by_scenario.push((scenario.name.to_string(), results));
    }

    // ─── frozen acceptance evaluation ──────────────────────────────────
    let high = &by_scenario[0].1;
    let low = &by_scenario[1].1;
    let (base_high, a_high, b_high) = (&high[0], &high[1], &high[2]);
    let (base_low, a_low, b_low) = (&low[0], &low[1], &low[2]);

    let h1_delta = a_low.recall - base_low.recall;
    let h1_holds = h1_delta >= H1_MIN_DELTA_PP;
    let h2_delta = b_low.recall - base_low.recall;
    let h2_holds = h2_delta >= H2_MIN_DELTA_PP;

    let a_build_ratio = a_low.build_ms / base_low.build_ms.max(0.001);
    let b_build_ratio = b_low.build_ms / base_low.build_ms.max(0.001);
    let a_qps_ratio = a_low.qps / base_low.qps.max(0.001);
    let b_qps_ratio = b_low.qps / base_low.qps.max(0.001);

    let a_regression_high = base_high.recall - a_high.recall;
    let b_regression_high = base_high.recall - b_high.recall;

    let subject_to_ok = a_build_ratio <= MAX_BUILD_RATIO
        && b_build_ratio <= MAX_BUILD_RATIO
        && a_qps_ratio >= H1_MIN_QPS_RATIO
        && b_qps_ratio >= H2_MIN_QPS_RATIO
        && a_regression_high <= MAX_HIGH_ACCESS_REGRESSION_PP
        && b_regression_high <= MAX_HIGH_ACCESS_REGRESSION_PP;

    println!("\n════════════════════════════════════════════════════════════════════");
    println!("  HYPOTHESIS EVALUATION (low-access scenario, auth ~= 1.56%)");
    println!("════════════════════════════════════════════════════════════════════");
    println!(
        "  H1 gamma=4 degree:    delta_recall = {:+.4} (need >= {:+.4})  -> {}",
        h1_delta,
        H1_MIN_DELTA_PP,
        if h1_holds { "HOLDS" } else { "REJECT" }
    );
    println!(
        "  H2 hierarchical seed: delta_recall = {:+.4} (need >= {:+.4})  -> {}",
        h2_delta,
        H2_MIN_DELTA_PP,
        if h2_holds { "HOLDS" } else { "REJECT" }
    );
    println!(
        "  candidate_A build ratio={:.2}x (<= {MAX_BUILD_RATIO}x) qps ratio={:.2}x (>= {H1_MIN_QPS_RATIO}x)",
        a_build_ratio, a_qps_ratio
    );
    println!(
        "  candidate_B build ratio={:.2}x (<= {MAX_BUILD_RATIO}x) qps ratio={:.2}x (>= {H2_MIN_QPS_RATIO}x)",
        b_build_ratio, b_qps_ratio
    );
    println!(
        "  high-access regression: candidate_A={:+.4}pp candidate_B={:+.4}pp (both must be <= {MAX_HIGH_ACCESS_REGRESSION_PP:+.4})",
        a_regression_high, b_regression_high
    );
    println!(
        "  subject-to constraints: {}",
        if subject_to_ok { "PASS" } else { "FAIL" }
    );

    let verdict = if !subject_to_ok {
        "REJECT"
    } else if h1_holds || h2_holds {
        "ACCEPT"
    } else {
        "REJECT"
    };
    println!("\n  OVERALL ACCEPTANCE: {verdict}");
    println!("════════════════════════════════════════════════════════════════════");

    // ─── supplementary root-cause probe ─────────────────────────────────
    // Not part of the frozen H1/H2 decision above (already computed and
    // printed). Sweeps the ef (visited-node) budget to check whether
    // candidate_A's low-access deficit is a fixed-ef artifact (narrows or
    // closes as ef grows) or a structural ceiling (persists regardless of
    // visited-node budget).
    run_ef_sweep_diagnostic();

    if verdict == "REJECT" {
        std::process::exit(1);
    }
}

fn run_ef_sweep_diagnostic() {
    println!("\n════════════════════════════════════════════════════════════════════");
    println!("  SUPPLEMENTARY DIAGNOSTIC: ef-budget sweep (low-access, not part of H1/H2)");
    println!("════════════════════════════════════════════════════════════════════");

    let scenario = &SCENARIOS[1]; // low-access
    let cfg = DatasetConfig {
        n_vectors: N_VECTORS,
        dims: DIMS,
        n_caps: scenario.n_caps,
        required_per_vector: 1,
        seed: SEED,
    };
    let entries = generate(&cfg);
    let (queries, holder) = generate_queries(
        N_QUERIES,
        DIMS,
        scenario.n_caps,
        scenario.held_caps,
        SEED ^ 0xabcd,
    );
    let mut oracle = Oracle::new(DIMS);
    for e in &entries {
        oracle.insert(e.id, e.vector.clone(), e.required);
    }
    let oracle_results: Vec<Vec<_>> = queries
        .iter()
        .map(|q| oracle.search(q, K, holder))
        .collect();

    let mut base = CapGraphIndex::new(DIMS, BASE_DEGREE, ENTRY_POINTS);
    base.batch_build(entries.iter().map(|e| (e.id, e.vector.clone(), e.required)));
    let mut cand_a = CapGraphIndex::new(DIMS, BASE_DEGREE * GAMMA, ENTRY_POINTS);
    cand_a.batch_build(entries.iter().map(|e| (e.id, e.vector.clone(), e.required)));

    println!(
        "  {:>6} {:>12} {:>10} {:>12} {:>10}",
        "ef_mul", "base_recall", "base_qps", "candA_recall", "candA_qps"
    );
    for &m in &[10usize, 30, 100, 300, 1000] {
        base = base.with_ef_multiplier(m);
        cand_a = cand_a.with_ef_multiplier(m);
        let (br, _, bq) = bench(&mut base, &queries, &oracle_results, holder);
        let (ar, _, aq) = bench(&mut cand_a, &queries, &oracle_results, holder);
        println!(
            "  {:>6} {:>12.3} {:>10.0} {:>12.3} {:>10.0}",
            m, br, bq, ar, aq
        );
    }
    println!(
        "  (ef = k * ef_mul visited-node cap; n={N_VECTORS} so ef_mul>={} visits the whole graph)",
        N_VECTORS / K
    );
}
