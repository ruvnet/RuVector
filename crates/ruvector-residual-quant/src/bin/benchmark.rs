//! Nightly research benchmark for ADR-362.
//!
//! Races three bit-allocation policies — `random` (baseline),
//! `residual_gated` (candidate A, the sheaf-residual hypothesis under
//! test) and `density_gated` (candidate B, the local-density control
//! ADR-353 flagged as untested) — at an identical matched byte budget, and
//! measures whether the novel signal actually buys anything over either
//! alternative.
//!
//! Run with: `cargo run --release -p ruvector-residual-quant --bin benchmark`
//! Override via env vars: N_VECTORS, DIM, N_CLUSTERS, CLUSTER_STD,
//! CENTER_SCALE, BOUNDARY_FRAC, K_NEIGHBORS, EDGE_DIM, HI_BITS, LO_BITS,
//! HI_FRACTION, N_QUERIES, SEEDS (comma separated).

use ruvector_residual_quant::*;
use std::time::Instant;

struct SeedResult {
    policy_results: Vec<PolicyRunResult>,
    residual_auc: f32,
    density_auc: f32,
    residual_density_corr: f32,
    graph_build_ms: f64,
}

#[derive(Clone)]
struct PolicyRunResult {
    policy: &'static str,
    overall_recall: f32,
    boundary_recall: f32,
    total_bytes: usize,
    quantize_build_us: f64,
    query_us: f64,
    witness_ok: bool,
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len().max(1) as f32
}

fn stddev(xs: &[f32]) -> f32 {
    let m = mean(xs);
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / xs.len().max(1) as f32).sqrt()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_u8(key: &str, default: u8) -> u8 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct RunParams {
    k_neighbors: usize,
    edge_dim: usize,
    hi_bits: u8,
    lo_bits: u8,
    hi_fraction: f32,
    n_queries: usize,
    query_seed: u64,
}

fn run_seed(cfg: &DatasetConfig, p: &RunParams) -> SeedResult {
    let RunParams {
        k_neighbors,
        edge_dim,
        hi_bits,
        lo_bits,
        hi_fraction,
        n_queries,
        query_seed,
    } = *p;
    let data = gen_dataset(cfg);
    let vectors: Vec<Vector> = data.iter().map(|e| e.vec.clone()).collect();
    let labels: Vec<bool> = data.iter().map(|e| e.is_boundary).collect();
    let n = vectors.len();
    let raw_bytes_per_vec = cfg.dim * 4;

    let t0 = Instant::now();
    let edges = build_knn_graph(&vectors, k_neighbors);
    let graph_build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let residual_scores =
        node_residual_energies(n, &edges, &vectors, cfg.seed ^ 0xA11CE, edge_dim, cfg.dim);
    let density_scores = node_density_outlier_scores(n, &edges, &vectors);
    let residual_auc = auc(&residual_scores, &labels);
    let density_auc = auc(&density_scores, &labels);
    let residual_density_corr = pearson(&residual_scores, &density_scores);

    // Ground truth top-10 (exact, fp32) is shared across all policies —
    // computed once per seed, never influenced by any candidate.
    let mut qrng = Lcg(query_seed);
    let query_idx: Vec<usize> = qrng
        .shuffled_indices(n)
        .into_iter()
        .take(n_queries.min(n))
        .collect();
    let exact_topk: Vec<Vec<u32>> = query_idx
        .iter()
        .map(|&q| topk_ids(q, &vectors, 10))
        .collect();

    let mut policy_results = Vec::new();
    for policy in [Policy::Random, Policy::ResidualGated, Policy::DensityGated] {
        let scores = match policy {
            Policy::Random => &residual_scores, // ignored by Random allocation, kept for witness scoring only
            Policy::ResidualGated => &residual_scores,
            Policy::DensityGated => &density_scores,
        };
        let t0 = Instant::now();
        let bits = allocate_bits(
            scores,
            hi_bits,
            lo_bits,
            hi_fraction,
            policy,
            cfg.seed ^ 0xB17E55,
        );
        let quantized: Vec<QuantizedVector> = vectors
            .iter()
            .zip(bits.iter())
            .map(|(v, &b)| quantize_vector(v, b))
            .collect();
        let dequantized: Vec<Vector> = quantized.iter().map(dequantize).collect();
        let total_bytes: usize = quantized.iter().map(packed_bytes).sum();

        let mut witness = WitnessLog::new();
        for (i, &b) in bits.iter().enumerate() {
            witness.append(data[i].id, policy.name(), scores[i], b);
        }
        let witness_ok = witness.verify();
        let quantize_build_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        let t1 = Instant::now();
        let mut recalls = Vec::with_capacity(query_idx.len());
        let mut boundary_recalls = Vec::new();
        for (qi, (&q, exact)) in query_idx.iter().zip(exact_topk.iter()).enumerate() {
            let approx = topk_ids(q, &dequantized, 10);
            let r = recall_at_k(&approx, exact);
            recalls.push(r);
            if labels[query_idx[qi]] {
                boundary_recalls.push(r);
            }
        }
        let query_us = t1.elapsed().as_secs_f64() * 1_000_000.0;

        policy_results.push(PolicyRunResult {
            policy: policy.name(),
            overall_recall: mean(&recalls),
            boundary_recall: if boundary_recalls.is_empty() {
                f32::NAN
            } else {
                mean(&boundary_recalls)
            },
            total_bytes,
            quantize_build_us,
            query_us,
            witness_ok,
        });
    }

    println!(
        "  raw fp32 reference: {} bytes/vector, {} vectors total = {:.2} MB",
        raw_bytes_per_vec,
        n,
        (raw_bytes_per_vec * n) as f64 / (1024.0 * 1024.0)
    );

    SeedResult {
        policy_results,
        residual_auc,
        density_auc,
        residual_density_corr,
        graph_build_ms,
    }
}

fn main() {
    let n: usize = env_usize("N_VECTORS", 1500);
    let dim: usize = env_usize("DIM", 48);
    let n_clusters: usize = env_usize("N_CLUSTERS", 6);
    let cluster_std: f32 = env_f32("CLUSTER_STD", 0.12);
    let center_scale: f32 = env_f32("CENTER_SCALE", 3.0);
    let boundary_frac: f32 = env_f32("BOUNDARY_FRAC", 0.08);
    let k_neighbors: usize = env_usize("K_NEIGHBORS", 10);
    let edge_dim: usize = env_usize("EDGE_DIM", 12);
    let hi_bits: u8 = env_u8("HI_BITS", 8);
    let lo_bits: u8 = env_u8("LO_BITS", 2);
    let hi_fraction: f32 = env_f32("HI_FRACTION", 0.25);
    let n_queries: usize = env_usize("N_QUERIES", 250);
    let seeds: Vec<u64> = std::env::var("SEEDS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<u64>| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2, 3, 4, 5]);

    println!("================================================================");
    println!(" ADR-362: Sheaf-residual-gated quantization bit-width benchmark");
    println!("================================================================");
    println!(
        "Platform : {} / {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("Vectors  : {n}   Dim: {dim}   Clusters: {n_clusters}");
    println!("Boundary fraction (planted contradictions): {boundary_frac:.2}");
    println!("kNN k: {k_neighbors}   Edge projection dim: {edge_dim}");
    println!("Bit budget: hi={hi_bits} lo={lo_bits} hi_fraction={hi_fraction:.2}  -> matched avg bits/dim = {:.2}", hi_fraction * hi_bits as f32 + (1.0 - hi_fraction) * lo_bits as f32);
    println!("Queries/seed: {n_queries}   Seeds: {seeds:?}");
    println!();

    let mut per_policy_overall: std::collections::HashMap<&'static str, Vec<f32>> =
        Default::default();
    let mut per_policy_boundary: std::collections::HashMap<&'static str, Vec<f32>> =
        Default::default();
    let mut bytes_by_policy: std::collections::HashMap<&'static str, usize> = Default::default();
    let mut residual_aucs = Vec::new();
    let mut density_aucs = Vec::new();
    let mut corrs = Vec::new();
    let mut all_witness_ok = true;
    let mut all_bytes_equal = true;

    for &seed in &seeds {
        println!("--- seed {seed} ---");
        let cfg = DatasetConfig {
            n,
            dim,
            n_clusters,
            cluster_std,
            center_scale,
            boundary_frac,
            seed,
        };
        let params = RunParams {
            k_neighbors,
            edge_dim,
            hi_bits,
            lo_bits,
            hi_fraction,
            n_queries,
            query_seed: seed ^ 0xF00D,
        };
        let r = run_seed(&cfg, &params);
        println!(
            "  graph build: {:.1}ms   residual_auc={:.3}  density_auc={:.3}  residual_vs_density_corr={:.3}",
            r.graph_build_ms, r.residual_auc, r.density_auc, r.residual_density_corr
        );
        residual_aucs.push(r.residual_auc);
        density_aucs.push(r.density_auc);
        corrs.push(r.residual_density_corr);

        let mut byte_values = Vec::new();
        for pr in &r.policy_results {
            println!(
                "  [{:<14}] overall_recall={:.3}  boundary_recall={:.3}  bytes={}  quant_build={:.0}us  query={:.0}us  witness_ok={}",
                pr.policy, pr.overall_recall, pr.boundary_recall, pr.total_bytes, pr.quantize_build_us, pr.query_us, pr.witness_ok
            );
            per_policy_overall
                .entry(pr.policy)
                .or_default()
                .push(pr.overall_recall);
            per_policy_boundary
                .entry(pr.policy)
                .or_default()
                .push(pr.boundary_recall);
            bytes_by_policy.insert(pr.policy, pr.total_bytes);
            byte_values.push(pr.total_bytes);
            all_witness_ok &= pr.witness_ok;
        }
        if byte_values.iter().any(|&b| b != byte_values[0]) {
            all_bytes_equal = false;
        }
        println!();
    }

    println!("================================================================");
    println!(" Aggregate across {} seeds", seeds.len());
    println!("================================================================");
    println!(
        "Residual-energy AUC (separating planted boundary vectors): mean={:.3} std={:.3}",
        mean(&residual_aucs),
        stddev(&residual_aucs)
    );
    println!(
        "Local-density AUC (same task):                              mean={:.3} std={:.3}",
        mean(&density_aucs),
        stddev(&density_aucs)
    );
    println!(
        "Pearson corr(residual_score, density_score):                mean={:.3} std={:.3}",
        mean(&corrs),
        stddev(&corrs)
    );
    println!();
    println!("┌────────────────┬─────────────────────┬─────────────────────┬──────────┐");
    println!("│ Policy         │ Overall recall@10    │ Boundary recall@10  │ Bytes    │");
    println!("├────────────────┼─────────────────────┼─────────────────────┼──────────┤");
    for policy in ["random", "residual_gated", "density_gated"] {
        let ov = &per_policy_overall[policy];
        let bd = &per_policy_boundary[policy];
        println!(
            "│ {:<14} │ {:>8.3} ± {:>7.3} │ {:>8.3} ± {:>7.3} │ {:>8} │",
            policy,
            mean(ov),
            stddev(ov),
            mean(bd),
            stddev(bd),
            bytes_by_policy[policy]
        );
    }
    println!("└────────────────┴─────────────────────┴─────────────────────┴──────────┘");
    println!();

    // ── Pre-registered acceptance gates (ADR-362 §Hypothesis) ──────────────
    let random_overall = mean(&per_policy_overall["random"]);
    let random_boundary = mean(&per_policy_boundary["random"]);
    let residual_overall = mean(&per_policy_overall["residual_gated"]);
    let residual_boundary = mean(&per_policy_boundary["residual_gated"]);
    let density_boundary = mean(&per_policy_boundary["density_gated"]);

    const BOUNDARY_IMPROVEMENT_MARGIN: f32 = 0.05;
    const OVERALL_REGRESSION_TOLERANCE: f32 = 0.02;

    let boundary_gate = residual_boundary - random_boundary >= BOUNDARY_IMPROVEMENT_MARGIN;
    let overall_gate = residual_overall >= random_overall - OVERALL_REGRESSION_TOLERANCE;
    let budget_gate = all_bytes_equal;
    let witness_gate = all_witness_ok;

    println!("Gate: residual_boundary_recall - random_boundary_recall >= {BOUNDARY_IMPROVEMENT_MARGIN:.2}  -> {:.3} - {:.3} = {:.3}  [{}]",
        residual_boundary, random_boundary, residual_boundary - random_boundary, if boundary_gate {"PASS"} else {"FAIL"});
    println!("Gate: residual_overall_recall >= random_overall_recall - {OVERALL_REGRESSION_TOLERANCE:.2}         -> {:.3} >= {:.3}  [{}]",
        residual_overall, random_overall - OVERALL_REGRESSION_TOLERANCE, if overall_gate {"PASS"} else {"FAIL"});
    println!(
        "Gate: matched byte budget across all policies (every seed)              [{}]",
        if budget_gate { "PASS" } else { "FAIL" }
    );
    println!(
        "Gate: witness hash chain verifies for every policy run (every seed)      [{}]",
        if witness_gate { "PASS" } else { "FAIL" }
    );
    println!();
    println!(
        "Context (not a gate): residual vs. density boundary_recall = {:.3} vs. {:.3}",
        residual_boundary, density_boundary
    );

    let acceptance = if !budget_gate || !witness_gate {
        "INCONCLUSIVE"
    } else if boundary_gate && overall_gate {
        "ACCEPT"
    } else {
        "REJECT"
    };
    println!();
    println!("ACCEPTANCE RESULT: {acceptance}");

    if acceptance != "ACCEPT" {
        std::process::exit(1);
    }
}
