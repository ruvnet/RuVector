//! CoDEQ benchmark demo.
//!
//! Measures three variants on a Gaussian dataset and simulates 10% streaming drift.
//!
//! Usage: cargo run --release -p ruvector-codeq

use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_codeq::{CoDEQIndex, recall_at_k};
use ruvector_codeq::pq_baseline::{FlatL2IndexCoDEQ, StaticPqIndex};

const N: usize = 5_000;
const DIM: usize = 128;
const N_QUERIES: usize = 500;
const K: usize = 10;
const BITS: usize = 8;         // 8-bit codes → 256 leaf cells
const PQ_M: usize = 8;         // 8 PQ subspaces, dim/m = 16 dims each
const PQ_K: usize = 64;        // 64 centroids per subspace (compact)
const PQ_ITER: usize = 10;

fn make_vecs(n: usize, dim: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n as u64)
        .map(|id| (id, (0..dim).map(|_| dist.sample(&mut rng)).collect()))
        .collect()
}

fn bench_flat_qps(idx: &FlatL2IndexCoDEQ, queries: &[(u64, Vec<f32>)], k: usize) -> f64 {
    let start = Instant::now();
    for (_qid, qv) in queries { let _ = idx.search(qv, k); }
    queries.len() as f64 / start.elapsed().as_secs_f64()
}

fn bench_pq_qps(idx: &StaticPqIndex, queries: &[(u64, Vec<f32>)], k: usize) -> f64 {
    let start = Instant::now();
    for (_qid, qv) in queries { let _ = idx.search_adc(qv, k); }
    queries.len() as f64 / start.elapsed().as_secs_f64()
}

fn bench_codeq_qps(idx: &CoDEQIndex, queries: &[(u64, Vec<f32>)], k: usize) -> f64 {
    let start = Instant::now();
    for (_qid, qv) in queries { let _ = idx.search_adc(qv, k); }
    queries.len() as f64 / start.elapsed().as_secs_f64()
}

fn recall_flat(idx: &FlatL2IndexCoDEQ, flat_ref: &FlatL2IndexCoDEQ, queries: &[(u64, Vec<f32>)], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (_qid, qv) in queries {
        let Ok(approx) = idx.search(qv, k) else { continue };
        let Ok(exact)  = flat_ref.search(qv, k) else { continue };
        let approx_ids: std::collections::HashSet<u64> = approx.iter().map(|(id,_)| *id).collect();
        let exact_ids:  std::collections::HashSet<u64> = exact.iter().map(|(id,_)| *id).collect();
        hits  += approx_ids.intersection(&exact_ids).count();
        total += exact_ids.len();
    }
    if total == 0 { 0.0 } else { hits as f64 / total as f64 }
}

fn recall_pq(idx: &StaticPqIndex, flat_ref: &FlatL2IndexCoDEQ, queries: &[(u64, Vec<f32>)], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (_qid, qv) in queries {
        let Ok(approx) = idx.search_adc(qv, k) else { continue };
        let Ok(exact)  = flat_ref.search(qv, k) else { continue };
        let approx_ids: std::collections::HashSet<u64> = approx.iter().map(|(id,_)| *id).collect();
        let exact_ids:  std::collections::HashSet<u64> = exact.iter().map(|(id,_)| *id).collect();
        hits  += approx_ids.intersection(&exact_ids).count();
        total += exact_ids.len();
    }
    if total == 0 { 0.0 } else { hits as f64 / total as f64 }
}

fn main() {
    println!("CoDEQ Streaming Quantizer Benchmark");
    println!("Dataset: n={N}, D={DIM}, queries={N_QUERIES}, k={K}");
    println!("CoDEQ: bits={BITS} (2^{BITS}={} leaves)", 1 << BITS);
    println!("PQ:    m={PQ_M}, k_sub={PQ_K}, iters={PQ_ITER}");
    println!("Hardware: {}", std::env::consts::ARCH);
    println!();

    let data    = make_vecs(N, DIM, 42);
    let queries = make_vecs(N_QUERIES, DIM, 99);

    // ── Build all three indices ───────────────────────────────────────────────

    let t = Instant::now();
    let flat = FlatL2IndexCoDEQ::from_vecs(data.clone()).unwrap();
    let flat_build_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let pq = StaticPqIndex::build(&data, PQ_M, PQ_K, PQ_ITER).unwrap();
    let pq_build_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let codeq = CoDEQIndex::from_vecs(&data, BITS, 1234).unwrap();
    let codeq_build_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!("Build times:");
    println!("  FlatL2:   {:>8.1} ms", flat_build_ms);
    println!("  StaticPQ: {:>8.1} ms  (k-means: slow for large n)", pq_build_ms);
    println!("  CoDEQ:    {:>8.1} ms  (median split: O(n·D), no k-means)", codeq_build_ms);
    println!();

    // ── Static benchmark (no drift) ───────────────────────────────────────────

    let flat_recall = recall_flat(&flat, &flat, &queries, K);
    let flat_qps    = bench_flat_qps(&flat, &queries, K);
    let flat_mem_mb = flat.memory_bytes() as f64 / 1_048_576.0;

    let pq_recall   = recall_pq(&pq, &flat, &queries, K);
    let pq_qps      = bench_pq_qps(&pq, &queries, K);
    let pq_mem_mb   = pq.memory_bytes() as f64 / 1_048_576.0;

    let codeq_recall = recall_at_k(&codeq, &queries, K);
    let codeq_qps    = bench_codeq_qps(&codeq, &queries, K);
    let codeq_mem_mb = codeq.memory_bytes() as f64 / 1_048_576.0;

    println!("Static benchmark (no drift):");
    println!("{:<30} {:>9} {:>10} {:>12} {:>12}",
             "Variant", "Rec@10", "QPS", "Mem(MB)", "Build(ms)");
    println!("{}", "-".repeat(77));
    println!("{:<30} {:>8.1}% {:>10.0} {:>11.2} {:>12.1}", flat.name(), flat_recall*100.0, flat_qps, flat_mem_mb, flat_build_ms);
    println!("{:<30} {:>8.1}% {:>10.0} {:>11.2} {:>12.1}", pq.name(),   pq_recall*100.0,   pq_qps,   pq_mem_mb,   pq_build_ms);
    println!("{:<30} {:>8.1}% {:>10.0} {:>11.2} {:>12.1}", "CoDEQ (kd-tree, 8-bit)", codeq_recall*100.0, codeq_qps, codeq_mem_mb, codeq_build_ms);
    println!();

    // ── Memory compression ratios ─────────────────────────────────────────────

    println!("Memory compression vs FlatL2:");
    println!("  StaticPQ: {:.2}×", flat_mem_mb / pq_mem_mb);
    println!("  CoDEQ:    {:.2}×", flat_mem_mb / codeq_mem_mb);
    println!("  Theoretical CoDEQ code-only ratio: {:.1}× (n×8 bits vs n×D×32 bits)",
             (DIM as f64 * 32.0) / BITS as f64);
    println!();

    // ── Streaming drift benchmark ─────────────────────────────────────────────
    // Simulate 10% drift: delete 500 old vectors, insert 500 new ones.

    println!("Streaming drift: 10% replace ({} deletes + {} inserts):", N/10, N/10);
    let new_vecs = make_vecs(N / 10, DIM, 777);

    // CoDEQ streaming update.
    let mut codeq_live = CoDEQIndex::from_vecs(&data, BITS, 1234).unwrap();
    let t = Instant::now();
    for i in 0..N/10 {
        codeq_live.delete(i as u64).unwrap();
        codeq_live.insert(100_000 + i as u64, new_vecs[i].1.clone()).unwrap();
    }
    let drift_update_ms = t.elapsed().as_secs_f64() * 1000.0;
    let codeq_live_recall = recall_at_k(&codeq_live, &queries, K);
    let codeq_live_qps    = bench_codeq_qps(&codeq_live, &queries, K);

    // FlatL2 streaming update (trivial baseline).
    let mut flat_live = FlatL2IndexCoDEQ::from_vecs(data.clone()).unwrap();
    for i in 0..N/10 {
        flat_live.delete(i as u64).unwrap();
        flat_live.insert(100_000 + i as u64, new_vecs[i].1.clone()).unwrap();
    }
    let flat_live_recall = recall_flat(&flat_live, &flat_live, &queries, K);
    let flat_live_qps    = bench_flat_qps(&flat_live, &queries, K);

    // StaticPQ cannot stream — measure recall degradation without rebuild.
    let pq_after_drift_recall = {
        // Queries still go to the original PQ (no updates possible).
        // Ground truth is the updated flat (has drifted data).
        let mut hits = 0usize;
        let mut total = 0usize;
        for (_qid, qv) in &queries {
            let Ok(approx) = pq.search_adc(qv, K) else { continue };
            let Ok(exact)  = flat_live.search(qv, K) else { continue };
            let approx_ids: std::collections::HashSet<u64> = approx.iter().map(|(id,_)| *id).collect();
            let exact_ids:  std::collections::HashSet<u64> = exact.iter().map(|(id,_)| *id).collect();
            hits  += approx_ids.intersection(&exact_ids).count();
            total += exact_ids.len();
        }
        if total == 0 { 0.0 } else { hits as f64 / total as f64 }
    };

    println!("{:<30} {:>9} {:>10} {:>20}", "Variant", "Rec@10", "QPS", "Update cost");
    println!("{}", "-".repeat(73));
    println!("{:<30} {:>8.1}% {:>10.0} {:>20}", "FlatL2 (after drift)", flat_live_recall*100.0, flat_live_qps, "trivial (O(1))");
    println!("{:<30} {:>8.1}% {:>10.0} {:>20}", "StaticPQ (stale, no update)", pq_after_drift_recall*100.0, pq_qps, "N/A (full rebuild req.)");
    println!("{:<30} {:>8.1}% {:>10.0} {:>19.1}ms", "CoDEQ (live)", codeq_live_recall*100.0, codeq_live_qps, drift_update_ms);
    println!();
    println!("CoDEQ update throughput: {:.0} ops/sec ({} ops in {:.3} s)",
             (N as f64 / 5.0) * 2.0 / (drift_update_ms / 1000.0),
             N / 5,
             drift_update_ms / 1000.0);
    println!();

    // ── Recall vs code bits ───────────────────────────────────────────────────

    println!("CoDEQ recall@10 vs code bits (n={N}, D={DIM}):");
    println!("{:>6}  {:>10}  {:>10}  {:>12}", "Bits", "Rec@10", "Mem(MB)", "Build(ms)");
    println!("{}", "-".repeat(44));
    for bits in [4, 5, 6, 7, 8] {
        let t = Instant::now();
        let idx = CoDEQIndex::from_vecs(&data, bits, 42).unwrap();
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;
        let r = recall_at_k(&idx, &queries, K);
        let mb = idx.memory_bytes() as f64 / 1_048_576.0;
        println!("{:>6}  {:>9.1}%  {:>10.2}  {:>11.1}", bits, r * 100.0, mb, build_ms);
    }

    println!();
    println!("Done.");
}
