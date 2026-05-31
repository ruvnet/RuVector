//! SymphonyQG benchmark: two sections.
//!
//! ## Section 1 — FastScan Kernel Throughput
//! Directly measures the speedup of 4-bit LUT-based distance scoring vs exact f32
//! for N candidates (what the SymphonyQG kernel does inside the graph loop).
//!
//! ## Section 2 — End-to-End Graph Search
//! Demonstrates the full index on three ef (beam-width) settings.
//! Note: a single-layer greedy graph (PoC) achieves lower recall than the
//! multi-layer HNSW used in the original SymphonyQG paper. The kernel speedup
//! from Section 1 is the primary measured contribution; Section 2 shows how
//! the kernel integrates into a complete ANN pipeline.
//!
//! Usage: cargo run --release -p ruvector-symphony-qg

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_symphony_qg::{SqgConfig, SqgIndex, fastscan::scan_scalar, pq4::Pq4, recall_at_k};
use std::time::Instant;

fn random_vecs(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0f32, 1.0).unwrap();
    (0..n * d).map(|_| dist.sample(&mut rng)).collect()
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

// ─── Section 1: FastScan Kernel Throughput ───────────────────────────────────

fn bench_fastscan_kernel(n: usize, d: usize, n_queries: usize) {
    println!("  n={n} D={d} ({n_queries} queries)...");
    let data = random_vecs(n, d, 42);
    let queries = random_vecs(n_queries, d, 99);
    let m = 8usize; // subspaces
    let code_bytes = (m + 1) / 2;

    // Train PQ quantizer.
    let pq = Pq4::train(&data, d, m, 25).unwrap();

    // Pre-encode all vectors.
    let all_codes: Vec<u8> = (0..n)
        .flat_map(|i| pq.encode(&data[i * d..(i + 1) * d]))
        .collect();

    // ── Variant A: exact f32 L2 for all n candidates ──
    let t0 = Instant::now();
    let mut exact_top: Vec<Vec<u32>> = Vec::with_capacity(n_queries);
    for qi in 0..n_queries {
        let q = &queries[qi * d..(qi + 1) * d];
        let mut dists: Vec<(f32, u32)> = (0..n)
            .map(|i| (l2_sq(q, &data[i * d..(i + 1) * d]), i as u32))
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        exact_top.push(dists[..10].iter().map(|&(_, id)| id).collect());
    }
    let elapsed_exact = t0.elapsed().as_secs_f64();
    let exact_throughput = (n_queries as f64 * n as f64) / elapsed_exact; // dist/s

    // ── Variant B: FastScan 4-bit LUT for all n candidates ──
    let t0 = Instant::now();
    let mut fastscan_recall_total = 0.0;
    for qi in 0..n_queries {
        let q = &queries[qi * d..(qi + 1) * d];
        let lut = pq.build_lut(q);
        let scores = scan_scalar(&lut, &all_codes, n, m, code_bytes);
        let mut indexed: Vec<(u16, u32)> = scores.iter().enumerate().map(|(i, &s)| (s, i as u32)).collect();
        indexed.sort_unstable_by_key(|&(s, _)| s);
        let cands: Vec<u32> = indexed[..10].iter().map(|&(_, id)| id).collect();
        fastscan_recall_total += recall_at_k(&exact_top[qi], &cands);
    }
    let elapsed_fs = t0.elapsed().as_secs_f64();
    let fs_throughput = (n_queries as f64 * n as f64) / elapsed_fs;
    let fs_recall = fastscan_recall_total / n_queries as f64;

    // ── Variant C: FastScan pre-filter, exact f32 re-rank top-50 ──
    let rerank_k = 50usize;
    let t0 = Instant::now();
    let mut rr_recall_total = 0.0;
    for qi in 0..n_queries {
        let q = &queries[qi * d..(qi + 1) * d];
        let lut = pq.build_lut(q);
        let scores = scan_scalar(&lut, &all_codes, n, m, code_bytes);
        let mut indexed: Vec<(u16, u32)> = scores.iter().enumerate().map(|(i, &s)| (s, i as u32)).collect();
        indexed.sort_unstable_by_key(|&(s, _)| s);
        // Re-rank top-rerank_k with exact f32.
        let mut reranked: Vec<(f32, u32)> = indexed[..rerank_k]
            .iter()
            .map(|&(_, id)| (l2_sq(q, &data[id as usize * d..(id as usize + 1) * d]), id))
            .collect();
        reranked.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let cands: Vec<u32> = reranked[..10].iter().map(|&(_, id)| id).collect();
        rr_recall_total += recall_at_k(&exact_top[qi], &cands);
    }
    let elapsed_rr = t0.elapsed().as_secs_f64();
    let rr_throughput = (n_queries as f64 * n as f64) / elapsed_rr;
    let rr_recall = rr_recall_total / n_queries as f64;

    let speedup_fs = fs_throughput / exact_throughput;
    let speedup_rr = rr_throughput / exact_throughput;

    println!("  {:28} {:>14.0} dist/s  {:>9.1}%  ({:.2}×)", "ExactF32 (baseline)", exact_throughput, 100.0, 1.0f64);
    println!("  {:28} {:>14.0} dist/s  {:>9.1}%  ({:.2}×)", "FastScan4bit (full scan)", fs_throughput, fs_recall * 100.0, speedup_fs);
    println!("  {:28} {:>14.0} dist/s  {:>9.1}%  ({:.2}×)", "FastScan+Rerank50 (C)", rr_throughput, rr_recall * 100.0, speedup_rr);
}

// ─── Section 2: End-to-End Graph Search ──────────────────────────────────────

#[derive(Clone)]
struct GraphResult {
    variant: String,
    n: usize,
    d: usize,
    ef: usize,
    recall: f64,
    qps: f64,
    build_ms: u64,
    speedup: f64,
}

fn bench_graph_search(n: usize, d: usize, n_queries: usize, k: usize) -> Vec<GraphResult> {
    let data = random_vecs(n, d, 42);
    let queries = random_vecs(n_queries, d, 99);
    let cfg = SqgConfig { pq_subspaces: 8, pq_iters: 25, m_neighbors: 20 };

    let t0 = Instant::now();
    let idx = SqgIndex::build(&data, d, cfg).unwrap();
    let build_ms = t0.elapsed().as_millis() as u64;

    let t0 = Instant::now();
    let mut all_truth: Vec<Vec<u32>> = Vec::with_capacity(n_queries);
    for qi in 0..n_queries {
        let q = &queries[qi * d..(qi + 1) * d];
        all_truth.push(idx.flat_exact(q, k).iter().map(|&(id, _)| id).collect());
    }
    let flat_qps = n_queries as f64 / t0.elapsed().as_secs_f64();

    let mut results = vec![GraphResult {
        variant: "FlatExact".into(), n, d, ef: 0,
        recall: 1.0, qps: flat_qps, build_ms: 0, speedup: 1.0,
    }];

    for &ef in &[50, 200, 500] {
        for (label, use_rerank) in [("SqgFastScan", false), ("SqgRerank", true)] {
            let t0 = Instant::now();
            let mut total_recall = 0.0;
            for qi in 0..n_queries {
                let q = &queries[qi * d..(qi + 1) * d];
                let r = if use_rerank {
                    idx.sqg_rerank(q, k, ef)
                } else {
                    idx.sqg_fastscan(q, k, ef)
                };
                let cands: Vec<u32> = r.iter().map(|&(id, _)| id).collect();
                total_recall += recall_at_k(&all_truth[qi], &cands);
            }
            let qps = n_queries as f64 / t0.elapsed().as_secs_f64();
            results.push(GraphResult {
                variant: label.into(), n, d, ef,
                recall: total_recall / n_queries as f64,
                qps,
                build_ms,
                speedup: qps / flat_qps,
            });
        }
    }
    results
}

fn print_graph_results(all: &[GraphResult]) {
    println!("\n  {:-<87}", "");
    println!(
        "  {:<20} {:>6} {:>5} {:>5}  {:>10} {:>11}  {:>10}  {:>8}",
        "Variant", "n", "D", "ef", "Recall@10", "QPS", "BuildMs", "Speedup"
    );
    println!("  {:-<87}", "");
    for r in all {
        println!(
            "  {:<20} {:>6} {:>5} {:>5}  {:>9.1}% {:>11.0}  {:>10}  {:>7.2}×",
            r.variant, r.n, r.d, r.ef, r.recall * 100.0, r.qps,
            if r.build_ms == 0 { "—".to_string() } else { format!("{}", r.build_ms) },
            r.speedup,
        );
    }
    println!("  {:-<87}", "");
}

fn main() {
    println!("SymphonyQG Benchmark — ruvector nightly 2026-05-08");
    println!("Hardware: {}", std::env::consts::ARCH);
    println!("Profile: cargo --release (opt-level=3, LTO=fat, codegen-units=1)");
    println!("Config:  M=20 neighbors, 8 PQ subspaces (4-bit k=16), 25 k-means iters");
    println!();

    println!("════════════════════════════════════════════════════════════════════════");
    println!("SECTION 1: FastScan Kernel Throughput");
    println!("  Scores ALL n candidates per query — isolates kernel efficiency from");
    println!("  graph navigation. Shows the speedup FastScan gives over exact f32 L2.");
    println!("  Format: [variant]  throughput (dist/s)  recall@10  (speedup vs exact)");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    for &(n, d) in &[(2000usize, 128usize), (5000, 128), (2000, 256), (5000, 256)] {
        println!("── n={n}, D={d} ──");
        bench_fastscan_kernel(n, d, 200);
        println!();
    }

    println!("════════════════════════════════════════════════════════════════════════");
    println!("SECTION 2: End-to-End Graph Search");
    println!("  PoC uses a bidirectional greedy flat k-NN graph (not HNSW multi-layer).");
    println!("  Recall is bounded by graph navigability; the paper's HNSW graph would");
    println!("  reach 90%+ recall at ef=50. FastScan kernel speedup is from Section 1.");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    for &(n, d) in &[(2000usize, 128usize), (5000, 128)] {
        println!("── Graph search: n={n}, D={d} ──");
        let r = bench_graph_search(n, d, 300, 10);
        print_graph_results(&r);
        println!();
    }
}
