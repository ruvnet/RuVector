//! Nightly 2026-10-26 benchmark: Fisher-weighted ANN vs. brute force vs.
//! plain-L2 ANN for EWC++ "nearest prior task" lookup.
//!
//! Run with: `cargo run --release -p ruvector-fisher-ann --bin fisher_ann_bench`

use ruvector_fisher_ann::dataset::{generate_queries, generate_tasks};
use ruvector_fisher_ann::{
    brute_force_topk, single_task_regularization_loss, FisherWeighted, HnswIndex, PlainL2,
};
use std::collections::HashSet;
use std::time::Instant;

const DIM: usize = 128;
const N_FAMILIES: usize = 40;
const N_QUERIES: usize = 300;
const HNSW_M: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 100;
const HNSW_EF_SEARCH: usize = 100;
const LAMBDA: f32 = 1234.0;

// --- Acceptance thresholds, fixed before any run (Step 10/40 of the
// nightly protocol: never changed after seeing results). ---
const MIN_SPEEDUP_AT_LARGEST_N: f64 = 5.0;
const MIN_RECALL_AT_10: f32 = 0.90;
const MAX_LOSS_GAP_FRACTION: f32 = 0.05;

struct RunStats {
    variant: &'static str,
    n: usize,
    build_ms: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    recall_at_1: f32,
    recall_at_10: f32,
    mean_loss_gap: f32,
    mean_gt_loss: f32,
    memory_bytes: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn summarize(mut durations: Vec<f64>) -> (f64, f64, f64, f64) {
    durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = durations.iter().sum::<f64>() / durations.len().max(1) as f64;
    (
        mean,
        percentile(&durations, 50.0),
        percentile(&durations, 95.0),
        percentile(&durations, 99.0),
    )
}

fn run_for_n(n: usize) -> Vec<RunStats> {
    let tasks = generate_tasks(n, DIM, N_FAMILIES, 1_000 + n as u64);
    let queries = generate_queries(&tasks, N_QUERIES, 2_000 + n as u64);

    // ---- Baseline: Fisher-weighted brute-force linear scan. ----
    // This also establishes ground truth for recall and for the "best
    // achievable" EWC regularization loss at each query.
    let mut durations = Vec::with_capacity(queries.len());
    let mut gt_top1 = Vec::with_capacity(queries.len());
    let mut gt_top10: Vec<HashSet<usize>> = Vec::with_capacity(queries.len());
    let mut gt_loss_sum = 0.0f32;
    for (q, _) in &queries {
        let t0 = Instant::now();
        let top = brute_force_topk(FisherWeighted, &tasks, q, 10);
        durations.push(t0.elapsed().as_secs_f64() * 1e6);
        gt_top1.push(top[0].1);
        gt_top10.push(top.iter().map(|&(_, i)| i).collect());
        gt_loss_sum += single_task_regularization_loss(q, &tasks[top[0].1], LAMBDA);
    }
    let (mean, p50, p95, p99) = summarize(durations);
    let mem = tasks
        .iter()
        .map(|t| (t.weights.len() + t.fisher.len()) * 4)
        .sum::<usize>();
    let mean_gt_loss = gt_loss_sum / queries.len() as f32;
    let baseline = RunStats {
        variant: "baseline_fisher_bruteforce",
        n,
        build_ms: 0.0,
        mean_us: mean,
        p50_us: p50,
        p95_us: p95,
        p99_us: p99,
        recall_at_1: 1.0,
        recall_at_10: 1.0,
        mean_loss_gap: 0.0,
        mean_gt_loss,
        memory_bytes: mem,
    };

    // ---- Candidate A: Fisher-weighted HNSW ----
    let t0 = Instant::now();
    let mut idx_a = HnswIndex::new(FisherWeighted, HNSW_M, HNSW_EF_CONSTRUCTION, 42);
    for t in tasks.clone() {
        idx_a.insert(t);
    }
    let build_ms_a = t0.elapsed().as_secs_f64() * 1000.0;
    let candidate_a = evaluate_hnsw(
        "candidate_A_fisher_hnsw",
        n,
        &idx_a,
        &tasks,
        &queries,
        &gt_top1,
        &gt_top10,
        build_ms_a,
    );

    // ---- Candidate B: plain-L2 HNSW (ablation: index speedup without Fisher) ----
    let t0 = Instant::now();
    let mut idx_b = HnswIndex::new(PlainL2, HNSW_M, HNSW_EF_CONSTRUCTION, 42);
    for t in tasks.clone() {
        idx_b.insert(t);
    }
    let build_ms_b = t0.elapsed().as_secs_f64() * 1000.0;
    let candidate_b = evaluate_hnsw(
        "candidate_B_plain_l2_hnsw",
        n,
        &idx_b,
        &tasks,
        &queries,
        &gt_top1,
        &gt_top10,
        build_ms_b,
    );

    vec![baseline, candidate_a, candidate_b]
}

#[allow(clippy::too_many_arguments)]
fn evaluate_hnsw<M: ruvector_fisher_ann::TaskMetric>(
    name: &'static str,
    n: usize,
    idx: &HnswIndex<M>,
    tasks: &[ruvector_fisher_ann::TaskRecord],
    queries: &[(Vec<f32>, usize)],
    gt_top1: &[usize],
    gt_top10: &[HashSet<usize>],
    build_ms: f64,
) -> RunStats {
    let mut durations = Vec::with_capacity(queries.len());
    let mut hits1 = 0usize;
    let mut hits10 = 0usize;
    let mut loss_gap_sum = 0.0f32;
    for (i, (q, _)) in queries.iter().enumerate() {
        let t0 = Instant::now();
        let res = idx.search(q, 10, HNSW_EF_SEARCH);
        durations.push(t0.elapsed().as_secs_f64() * 1e6);
        if !res.is_empty() {
            if res[0].1 == gt_top1[i] {
                hits1 += 1;
            }
            hits10 += res
                .iter()
                .filter(|&&(_, id)| gt_top10[i].contains(&id))
                .count();
            let chosen_loss = single_task_regularization_loss(q, &tasks[res[0].1], LAMBDA);
            let gt_loss = single_task_regularization_loss(q, &tasks[gt_top1[i]], LAMBDA);
            loss_gap_sum += (chosen_loss - gt_loss).max(0.0);
        }
    }
    let (mean, p50, p95, p99) = summarize(durations);
    RunStats {
        variant: name,
        n,
        build_ms,
        mean_us: mean,
        p50_us: p50,
        p95_us: p95,
        p99_us: p99,
        recall_at_1: hits1 as f32 / queries.len() as f32,
        recall_at_10: hits10 as f32 / (queries.len() * 10) as f32,
        mean_loss_gap: loss_gap_sum / queries.len() as f32,
        mean_gt_loss: 0.0,
        memory_bytes: idx.memory_bytes(),
    }
}

fn main() {
    println!("ruvector-fisher-ann nightly benchmark (2026-10-26)");
    println!(
        "dim={DIM} n_families={N_FAMILIES} n_queries={N_QUERIES} hnsw_m={HNSW_M} ef_construction={HNSW_EF_CONSTRUCTION} ef_search={HNSW_EF_SEARCH}"
    );
    println!(
        "{:<28} {:>8} {:>10} {:>10} {:>9} {:>9} {:>9} {:>10} {:>10} {:>12} {:>10}",
        "variant",
        "n",
        "build_ms",
        "mean_us",
        "p50_us",
        "p95_us",
        "p99_us",
        "recall@1",
        "recall@10",
        "loss_gap",
        "mem_KB"
    );

    let mut all_n_results: Vec<(usize, Vec<RunStats>)> = Vec::new();
    for &n in &[2_000usize, 10_000usize] {
        let results = run_for_n(n);
        for r in &results {
            println!(
                "{:<28} {:>8} {:>10.2} {:>10.2} {:>9.2} {:>9.2} {:>9.2} {:>10.3} {:>10.3} {:>12.4} {:>10.1}",
                r.variant,
                r.n,
                r.build_ms,
                r.mean_us,
                r.p50_us,
                r.p95_us,
                r.p99_us,
                r.recall_at_1,
                r.recall_at_10,
                r.mean_loss_gap,
                r.memory_bytes as f64 / 1024.0,
            );
        }
        all_n_results.push((n, results));
    }

    // ---- Acceptance gate, computed from the largest N run. ----
    let (largest_n, largest_results) = all_n_results.last().unwrap();
    let baseline = &largest_results[0];
    let cand_a = &largest_results[1];
    let cand_b = &largest_results[2];

    let speedup = baseline.mean_us / cand_a.mean_us;
    let recall_ok = cand_a.recall_at_10 >= MIN_RECALL_AT_10;
    let loss_ok = cand_a.mean_loss_gap <= MAX_LOSS_GAP_FRACTION * baseline.mean_gt_loss.max(1e-6);
    let speedup_ok = speedup >= MIN_SPEEDUP_AT_LARGEST_N;
    let ablation_shows_fisher_matters =
        cand_a.recall_at_10 > cand_b.recall_at_10 || cand_a.mean_loss_gap < cand_b.mean_loss_gap;

    println!("\n--- Acceptance check (N={largest_n}) ---");
    println!("speedup candidate_A vs baseline: {speedup:.1}x (threshold >= {MIN_SPEEDUP_AT_LARGEST_N}x) -> {}", if speedup_ok { "PASS" } else { "FAIL" });
    println!(
        "candidate_A recall@10: {:.3} (threshold >= {MIN_RECALL_AT_10}) -> {}",
        cand_a.recall_at_10,
        if recall_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "candidate_A mean loss gap: {:.4} vs threshold {:.4} ({:.0}% of mean ground-truth loss {:.4}) -> {}",
        cand_a.mean_loss_gap,
        MAX_LOSS_GAP_FRACTION * baseline.mean_gt_loss,
        MAX_LOSS_GAP_FRACTION * 100.0,
        baseline.mean_gt_loss,
        if loss_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "ablation (Fisher weighting matters, not just having an index): candidate_A recall@10={:.3} loss_gap={:.4} vs candidate_B recall@10={:.3} loss_gap={:.4} -> {}",
        cand_a.recall_at_10, cand_a.mean_loss_gap, cand_b.recall_at_10, cand_b.mean_loss_gap,
        if ablation_shows_fisher_matters { "PASS" } else { "FAIL" }
    );

    // A clean REJECT is any outcome that actively falsifies the hypothesis
    // (the ablation shows plain L2 is as good as or better than the
    // Fisher-weighted metric it was supposed to beat) rather than merely
    // an ambiguous measurement. INCONCLUSIVE is reserved for the case
    // where the ablation still favors Fisher but one secondary threshold
    // (speedup or absolute recall/loss) narrowly missed.
    let falsified = !ablation_shows_fisher_matters;
    let verdict = if speedup_ok && recall_ok && loss_ok && ablation_shows_fisher_matters {
        "ACCEPT"
    } else if falsified {
        "REJECT"
    } else {
        "INCONCLUSIVE"
    };
    println!("\nACCEPTANCE RESULT: {verdict}");

    // ---- Diagnostic (not part of the acceptance gate): is candidate A's
    // low recall a search-budget problem (fixable by raising ef_search) or
    // a graph-navigability problem (the construction-time metric produced
    // a poorly connected graph, so no amount of search budget helps)? ----
    println!("\n--- Diagnostic: candidate_A recall@10 vs ef_search (N=2000, same built graph) ---");
    let tasks = generate_tasks(2_000, DIM, N_FAMILIES, 1_000 + 2_000u64);
    let queries = generate_queries(&tasks, N_QUERIES, 2_000 + 2_000u64);
    let mut idx_a = HnswIndex::new(FisherWeighted, HNSW_M, HNSW_EF_CONSTRUCTION, 42);
    for t in tasks.clone() {
        idx_a.insert(t);
    }
    let mut gt_top10: Vec<HashSet<usize>> = Vec::with_capacity(queries.len());
    for (q, _) in &queries {
        let top = brute_force_topk(FisherWeighted, &tasks, q, 10);
        gt_top10.push(top.iter().map(|&(_, i)| i).collect());
    }
    for &ef in &[100usize, 500, 2_000] {
        let mut hits10 = 0usize;
        for (i, (q, _)) in queries.iter().enumerate() {
            let res = idx_a.search(q, 10, ef);
            hits10 += res
                .iter()
                .filter(|&&(_, id)| gt_top10[i].contains(&id))
                .count();
        }
        let recall = hits10 as f32 / (queries.len() * 10) as f32;
        println!(
            "ef_search={ef:>5} (index size {}): recall@10={recall:.3}",
            idx_a.len()
        );
    }
}
