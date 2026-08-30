//! Real-HNSW retrieval-receipt overhead benchmark.
//!
//! Measures receipt-build latency, verification cost, and proof size against
//! a real multi-layer approximate HNSW index (`ruvector-hnsw-repair`),
//! instead of the brute-force index `ruvector-retrieval-receipt` uses to
//! isolate provenance cost from ANN recall. This directly answers "Next
//! Research" item 1 from the `2026-08-13-retrieval-receipts` nightly report.
//!
//! Usage:
//!   cargo run --release -p ruvector-hnsw-receipt --bin benchmark
//!   cargo run --release -p ruvector-hnsw-receipt --bin benchmark -- 5000 64 10 64 300

use std::time::{Duration, Instant};

use ruvector_hnsw_receipt::{
    query_hash, synthetic_queries, HnswReceiptIndex, ReceiptVariant, RetrievalReceipt,
};

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn mean_ns(durations: &[Duration]) -> f64 {
    if durations.is_empty() {
        return 0.0;
    }
    durations.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / durations.len() as f64
}

struct Stage {
    name: &'static str,
    mean_ns: f64,
    p50_ns: u64,
    p95_ns: u64,
}

fn measure_stage(name: &'static str, mut samples: Vec<Duration>) -> Stage {
    samples.sort();
    Stage {
        name,
        mean_ns: mean_ns(&samples),
        p50_ns: percentile(&samples, 50.0).as_nanos() as u64,
        p95_ns: percentile(&samples, 95.0).as_nanos() as u64,
    }
}

fn print_stage(s: &Stage) {
    println!(
        "  {:<26} mean={:>10.0}ns  p50={:>10}ns  p95={:>10}ns",
        s.name, s.mean_ns, s.p50_ns, s.p95_ns
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let n: usize = argv.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let dims: usize = argv.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = argv.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let ef: usize = argv.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let num_queries: usize = argv.get(5).and_then(|s| s.parse().ok()).unwrap_or(300);
    let warmup = 20usize.min(num_queries / 10 + 1);

    println!("=== ruvector-hnsw-receipt benchmark ===");
    println!("n={n} dims={dims} k={k} ef={ef} queries={num_queries} (warmup={warmup} discarded)");
    println!(
        "Hardware: {} logical CPUs (via std::thread::available_parallelism)",
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(0)
    );

    let build_start = Instant::now();
    let index = HnswReceiptIndex::ingest(n, dims, 0xC0FF_EE01_1234);
    let build_time = build_start.elapsed();
    assert!(
        index.verify_write_history(),
        "write history must verify post-ingestion"
    );
    println!(
        "\nIndex construction: {n} vectors, {dims}D, HNSW insert+gate-admit: {:.3}s ({:.1} inserts/sec)",
        build_time.as_secs_f64(),
        n as f64 / build_time.as_secs_f64()
    );

    let all_queries = synthetic_queries(num_queries + warmup, dims, 0xA5A5_5A5A_1122);
    let (warmup_queries, queries) = all_queries.split_at(warmup);

    // Warmup: run every stage once per warmup query so allocator/cache state
    // is representative before timed samples begin.
    for q in warmup_queries {
        let _ = index.search_raw(q, k, ef);
        let items = index.search_items(q, k, ef);
        let qh = query_hash(q);
        let root = index.index_state_root();
        let _ = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &items);
        let _ = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &items);
    }

    let mut search_raw_d = Vec::with_capacity(queries.len());
    let mut search_items_d = Vec::with_capacity(queries.len());
    let mut per_result_build_d = Vec::with_capacity(queries.len());
    let mut merkle_build_d = Vec::with_capacity(queries.len());
    let mut per_result_verify_d = Vec::with_capacity(queries.len());
    let mut merkle_verify_d = Vec::with_capacity(queries.len());

    let mut per_result_bytes_worst = Vec::with_capacity(queries.len());
    let mut merkle_bytes_worst = Vec::with_capacity(queries.len());
    let mut per_result_verified = 0usize;
    let mut merkle_verified = 0usize;

    let mut recall_hits = 0usize;
    let root = index.index_state_root();

    for q in queries {
        let t0 = Instant::now();
        let raw_ids = index.search_raw(q, k, ef);
        search_raw_d.push(t0.elapsed());

        let t1 = Instant::now();
        let items = index.search_items(q, k, ef);
        search_items_d.push(t1.elapsed());

        let qh = query_hash(q);

        let t2 = Instant::now();
        let per_result = RetrievalReceipt::build(ReceiptVariant::PerResult, qh, root, &items);
        per_result_build_d.push(t2.elapsed());

        let t3 = Instant::now();
        let merkle = RetrievalReceipt::build(ReceiptVariant::Merkle, qh, root, &items);
        merkle_build_d.push(t3.elapsed());

        let worst = items.len().saturating_sub(1);
        per_result_bytes_worst.push(per_result.proof_bytes_for(worst));
        merkle_bytes_worst.push(merkle.proof_bytes_for(worst));

        let t4 = Instant::now();
        let pr_ok = per_result.verify_full(qh, root, &items);
        per_result_verify_d.push(t4.elapsed());
        if pr_ok {
            per_result_verified += 1;
        }

        let t5 = Instant::now();
        let mk_ok = merkle.verify_full(qh, root, &items);
        merkle_verify_d.push(t5.elapsed());
        if mk_ok {
            merkle_verified += 1;
        }

        // Correctness invariant: receipt construction must not perturb the
        // search path — items are exactly search_raw's ids in order.
        for (id, item) in raw_ids.iter().zip(items.iter()) {
            assert_eq!(
                *id, item.vector_id as u32,
                "receipt path must not alter search results"
            );
        }

        let gt = index.brute_force_topk(q, k);
        let gt_set: std::collections::HashSet<u32> = gt.into_iter().collect();
        recall_hits += raw_ids.iter().filter(|id| gt_set.contains(id)).count();
    }

    let search_raw_stage = measure_stage("search_raw (baseline)", search_raw_d);
    let search_items_stage = measure_stage("search_items (+rescoring)", search_items_d);
    let pr_build_stage = measure_stage("PerResult receipt build", per_result_build_d);
    let mk_build_stage = measure_stage("Merkle receipt build", merkle_build_d);
    let pr_verify_stage = measure_stage("PerResult verify_full", per_result_verify_d);
    let mk_verify_stage = measure_stage("Merkle verify_full", merkle_verify_d);

    println!("\n--- Latency (search on real multi-layer HNSW, N={n}, ef={ef}, k={k}) ---");
    print_stage(&search_raw_stage);
    print_stage(&search_items_stage);
    print_stage(&pr_build_stage);
    print_stage(&mk_build_stage);
    print_stage(&pr_verify_stage);
    print_stage(&mk_verify_stage);

    let pr_overhead_ratio = pr_build_stage.p50_ns as f64 / search_raw_stage.p50_ns as f64;
    let mk_overhead_ratio = mk_build_stage.p50_ns as f64 / search_raw_stage.p50_ns as f64;

    println!("\n--- Overhead ratio: receipt_build.p50 / search_raw.p50 ---");
    println!("  PerResult: {pr_overhead_ratio:.4}x");
    println!("  Merkle:    {mk_overhead_ratio:.4}x");

    let pr_bytes_mean =
        per_result_bytes_worst.iter().sum::<usize>() as f64 / per_result_bytes_worst.len() as f64;
    let mk_bytes_mean =
        merkle_bytes_worst.iter().sum::<usize>() as f64 / merkle_bytes_worst.len() as f64;
    println!("\n--- Proof size (worst-case index, k={k}) ---");
    println!("  PerResult proof_bytes_for(k-1): mean={pr_bytes_mean:.1} bytes");
    println!("  Merkle    proof_bytes_for(k-1): mean={mk_bytes_mean:.1} bytes");
    println!(
        "  Merkle/PerResult ratio: {:.4} (expect < 1.0 — O(log k) vs O(k))",
        mk_bytes_mean / pr_bytes_mean
    );

    let recall = recall_hits as f64 / (queries.len() * k) as f64;
    println!("\n--- ANN quality context (not the acceptance metric) ---");
    println!("  recall@{k} vs brute-force cosine ground truth: {recall:.4}");

    println!("\n--- Verification integrity (subject-to condition) ---");
    println!(
        "  PerResult verify_full success: {per_result_verified}/{} ({:.2}%)",
        queries.len(),
        100.0 * per_result_verified as f64 / queries.len() as f64
    );
    println!(
        "  Merkle    verify_full success: {merkle_verified}/{} ({:.2}%)",
        queries.len(),
        100.0 * merkle_verified as f64 / queries.len() as f64
    );

    // ── Acceptance ────────────────────────────────────────────────────────
    // Hypothesis fixed before this run (see docs/research/nightly report):
    //   1. 100% verify_full success for PerResult and Merkle.
    //   2. Merkle worst-case proof bytes < PerResult worst-case proof bytes.
    //   3. Merkle receipt-build p50 latency overhead < 0.50x raw HNSW
    //      search p50 latency (production "cheap enough to always-on" bar).
    let cond_verify = per_result_verified == queries.len() && merkle_verified == queries.len();
    let cond_proof_size = mk_bytes_mean < pr_bytes_mean;
    let cond_overhead = mk_overhead_ratio < 0.50;

    println!("\n=== ACCEPTANCE ===");
    println!(
        "  [{}] 100% verify_full success (PerResult + Merkle)",
        if cond_verify { "PASS" } else { "FAIL" }
    );
    println!(
        "  [{}] Merkle proof bytes < PerResult proof bytes",
        if cond_proof_size { "PASS" } else { "FAIL" }
    );
    println!(
        "  [{}] Merkle receipt-build p50 overhead < 0.50x raw search p50 (measured {mk_overhead_ratio:.4}x)",
        if cond_overhead { "PASS" } else { "FAIL" }
    );

    let verdict = if cond_verify && cond_proof_size && cond_overhead {
        "ACCEPT"
    } else {
        "REJECT"
    };
    println!("\n  RESULT: {verdict}");
}
