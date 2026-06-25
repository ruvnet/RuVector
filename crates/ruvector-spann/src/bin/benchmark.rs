//! SPANN Partition Spilling Benchmark
//!
//! Measures recall@10, throughput (QPS), and memory for three partition-spilling
//! variants across multiple nprobe values.
//!
//! The key SPANN insight: for a given recall target, spilled indexes need
//! fewer partitions probed than hard-assignment IVF. This benchmark shows that
//! directly by sweeping nprobe for each variant.
//!
//! Usage:
//!   cargo run --release -p ruvector-spann --bin benchmark

use ruvector_spann::index::{
    CoherenceSpill, CoherenceSpillConfig, PartitionIndex, SinglePartition, SinglePartitionConfig,
    SpillPartition, SpillPartitionConfig,
};
use std::collections::HashSet;
use std::time::Instant;

// ── Dataset generation ────────────────────────────────────────────────────────

struct Xorshift64(u64);

impl Xorshift64 {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f32) / (u64::MAX as f32)
    }

    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

fn generate_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xorshift64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.next_normal()).collect())
        .collect()
}

// ── Brute-force ground truth ──────────────────────────────────────────────────

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn brute_knn(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_sq(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

fn recall_at_k(results: &[ruvector_spann::SearchResult], gt: &[usize]) -> f32 {
    let gt_set: HashSet<usize> = gt.iter().copied().collect();
    results.iter().filter(|r| gt_set.contains(&r.id)).count() as f32 / gt.len() as f32
}

// ── Percentile helper ─────────────────────────────────────────────────────────

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Benchmark one (variant, nprobe) pair ─────────────────────────────────────

struct ProbeSweepResult {
    nprobe: usize,
    recall: f32,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
}

fn probe_sweep(
    idx: &dyn PartitionIndex,
    queries: &[Vec<f32>],
    gt: &[Vec<usize>],
    nprobe_values: &[usize],
    k: usize,
) -> Vec<ProbeSweepResult> {
    nprobe_values
        .iter()
        .map(|&nprobe| {
            let mut recalls = Vec::with_capacity(queries.len());
            let mut latencies: Vec<f64> = Vec::with_capacity(queries.len());
            for (qi, q) in queries.iter().enumerate() {
                let t = Instant::now();
                let results = idx.search(q, k, nprobe);
                latencies.push(t.elapsed().as_nanos() as f64 / 1000.0);
                recalls.push(recall_at_k(&results, &gt[qi]));
            }
            let mean_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
            let mean_us = latencies.iter().sum::<f64>() / latencies.len() as f64;
            let qps = 1_000_000.0 / mean_us;
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ProbeSweepResult {
                nprobe,
                recall: mean_recall,
                mean_us,
                p50_us: percentile(&sorted, 50.0),
                p95_us: percentile(&sorted, 95.0),
                qps,
            }
        })
        .collect()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(" ruvector-spann: SPANN Partition Spilling Benchmark");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(" OS   : {}", std::env::consts::OS);
    println!(" Arch : {}", std::env::consts::ARCH);
    println!("───────────────────────────────────────────────────────────────────────");

    // Two dataset sizes.
    let configs: &[(usize, usize, usize, usize)] = &[
        // (N, dim, n_centroids, kmeans_iters)
        (5_000, 128, 32, 20),
        (10_000, 128, 40, 20),
    ];

    let k = 10;
    let n_queries = 300;
    // Sweep nprobe from low to high to show the recall curve.
    let nprobe_values = vec![2, 4, 6, 8, 12, 16];

    for &(n, dim, n_centroids, kmeans_iters) in configs {
        println!("\n─── Dataset: N={n}, D={dim}, K={k}, queries={n_queries} ───────────────");
        let corpus = generate_corpus(n, dim, 42);
        let queries = generate_corpus(n_queries, dim, 99);

        eprint!("  Computing brute-force ground truth ({n} vectors × {n_queries} queries)... ");
        let gt: Vec<Vec<usize>> = queries.iter().map(|q| brute_knn(q, &corpus, k)).collect();
        eprintln!("done");

        // Build all three variants.
        eprint!("  Building SinglePartition... ");
        let mut single = SinglePartition::new(SinglePartitionConfig {
            n_centroids,
            kmeans_iters,
            dim,
        });
        let t = Instant::now();
        single.build(&corpus);
        eprintln!(
            "{}ms  ({} assignments, {:.2} MB)",
            t.elapsed().as_millis(),
            single.total_assignments(),
            single.memory_bytes() as f64 / (1024.0 * 1024.0)
        );

        eprint!("  Building SpillPartition (ratio=1.20)... ");
        let mut spill = SpillPartition::new(SpillPartitionConfig {
            n_centroids,
            kmeans_iters,
            dim,
            spill_ratio: 1.20,
        });
        let t = Instant::now();
        spill.build(&corpus);
        eprintln!(
            "{}ms  ({} assignments = {:.2}× overhead, {:.2} MB)",
            t.elapsed().as_millis(),
            spill.total_assignments(),
            spill.total_assignments() as f32 / single.total_assignments() as f32,
            spill.memory_bytes() as f64 / (1024.0 * 1024.0)
        );

        eprint!("  Building CoherenceSpill (pct=0.30)... ");
        let mut coh = CoherenceSpill::new(CoherenceSpillConfig {
            n_centroids,
            kmeans_iters,
            dim,
            coherence_percentile: 0.30,
        });
        let t = Instant::now();
        coh.build(&corpus);
        eprintln!(
            "{}ms  ({} assignments = {:.2}× overhead, {:.2} MB, threshold={:.4})",
            t.elapsed().as_millis(),
            coh.total_assignments(),
            coh.total_assignments() as f32 / single.total_assignments() as f32,
            coh.memory_bytes() as f64 / (1024.0 * 1024.0),
            coh.derived_spill_threshold
        );

        // Sweep nprobe.
        let single_sweep = probe_sweep(&single, &queries, &gt, &nprobe_values, k);
        let spill_sweep = probe_sweep(&spill, &queries, &gt, &nprobe_values, k);
        let coh_sweep = probe_sweep(&coh, &queries, &gt, &nprobe_values, k);

        // Print nprobe sweep table.
        println!("\n  nprobe sweep (recall@10 / QPS):");
        println!(
            "  {:>7}  {:>14}  {:>14}  {:>14}",
            "nprobe", "Single(recall/QPS)", "Spill(recall/QPS)", "CoherenceSpill(r/QPS)"
        );
        println!("  {}", "─".repeat(70));
        for i in 0..nprobe_values.len() {
            println!(
                "  {:>7}  {:>6.3}/{:>6.0}  {:>6.3}/{:>6.0}  {:>6.3}/{:>6.0}",
                single_sweep[i].nprobe,
                single_sweep[i].recall,
                single_sweep[i].qps,
                spill_sweep[i].recall,
                spill_sweep[i].qps,
                coh_sweep[i].recall,
                coh_sweep[i].qps,
            );
        }

        // Print full results table for nprobe=8 (representative).
        let nprobe_idx = nprobe_values.iter().position(|&x| x == 8).unwrap_or(3);
        let s = &single_sweep[nprobe_idx];
        let sp = &spill_sweep[nprobe_idx];
        let c = &coh_sweep[nprobe_idx];
        println!("\n  Detailed latency at nprobe=8:");
        println!(
            "  {:<42} {:>8} {:>8} {:>8} {:>7} {:>6}",
            "Variant", "Recall", "Mean µs", "p50 µs", "p95 µs", "QPS"
        );
        println!("  {}", "─".repeat(82));
        for (name, r) in [
            ("SinglePartition (IVF baseline)", s),
            ("SpillPartition (SPANN fixed-threshold)", sp),
            ("CoherenceSpill (adaptive percentile)", c),
        ] {
            println!(
                "  {:<42} {:>8.3} {:>8.1} {:>8.1} {:>8.1} {:>6.0}",
                &name[..name.len().min(42)],
                r.recall,
                r.mean_us,
                r.p50_us,
                r.p95_us,
                r.qps,
            );
        }

        // Acceptance: SpillPartition must improve recall ≥ 1.40× over Single at same nprobe
        // for at least one nprobe value in the sweep.
        println!("\n  Recall improvement from spilling over SinglePartition:");
        let mut best_spill_gain = 0.0f32;
        let mut best_coh_gain = 0.0f32;
        for i in 0..nprobe_values.len() {
            let gain_spill = if single_sweep[i].recall > 0.0 {
                spill_sweep[i].recall / single_sweep[i].recall
            } else {
                1.0
            };
            let gain_coh = if single_sweep[i].recall > 0.0 {
                coh_sweep[i].recall / single_sweep[i].recall
            } else {
                1.0
            };
            best_spill_gain = best_spill_gain.max(gain_spill);
            best_coh_gain = best_coh_gain.max(gain_coh);
            println!(
                "  nprobe={}: SpillPartition={:.3} ({:+.3}, {:.2}×)  CoherenceSpill={:.3} ({:+.3}, {:.2}×)",
                nprobe_values[i],
                spill_sweep[i].recall,
                spill_sweep[i].recall - single_sweep[i].recall,
                gain_spill,
                coh_sweep[i].recall,
                coh_sweep[i].recall - single_sweep[i].recall,
                gain_coh,
            );
        }

        // Acceptance gate: SpillPartition peak gain ≥ 1.40× over Single.
        // CoherenceSpill threshold is 1.15× — it spills only 30% of the corpus
        // (vs ~100% for SpillPartition), so its recall gain is smaller but
        // its memory overhead is also smaller (1.30× vs 2.00×).
        let spill_passes = best_spill_gain >= 1.40;
        let coh_passes = best_coh_gain >= 1.15;
        println!("\n  ┌─ Acceptance Gate ─────────────────────────────────────────────┐");
        println!(
            "  │ SpillPartition peak recall gain: {:.2}× (threshold: 1.40×) {}  │",
            best_spill_gain,
            if spill_passes { "PASS" } else { "FAIL" }
        );
        println!(
            "  │ CoherenceSpill peak recall gain: {:.2}× (threshold: 1.15×) {}  │",
            best_coh_gain,
            if coh_passes { "PASS" } else { "FAIL" }
        );
        println!("  └───────────────────────────────────────────────────────────────┘");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}
