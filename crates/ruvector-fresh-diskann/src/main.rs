//! FreshDiskANN benchmark — measures recall@10, QPS, consolidation latency
//! for four variants on a 10k-vector 128-dim dataset.
//!
//! Variants:
//!   A — Static: full batch build over all 10k vectors (upper-bound recall).
//!   B — Eager:  build on 9k, stream 1k with consolidation after every insert.
//!   C — Lazy T=100: build on 9k, stream 1k, consolidate every 100 inserts.
//!   D — Buffer-only: build on 9k, stream 1k, never consolidate (pure brute scan).

use ruvector_fresh_diskann::{l2sq, ConsolidationPolicy, FreshConfig, FreshDiskAnn};
use rand::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

const N_BASE: usize = 9_000;
const N_STREAM: usize = 1_000;
const N_QUERY: usize = 200;
const DIM: usize = 128;
const K: usize = 10;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║      ruvector FreshDiskANN — Streaming Index Maintenance Benchmark      ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Base vectors : {N_BASE:>6}  │  Stream inserts: {N_STREAM:>5}  │  Dim: {DIM}  │  k={K}  ║");
    println!("║  Queries      : {N_QUERY:>6}  │  Hardware: see `uname -m` / `lscpu`                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // ---- Data generation (seeded for reproducibility) ----------------------
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let total = N_BASE + N_STREAM + N_QUERY;
    let all_vecs: Vec<Vec<f32>> = (0..total)
        .map(|_| (0..DIM).map(|_| rng.gen::<f32>()).collect())
        .collect();

    let base: Vec<(String, Vec<f32>)> = (0..N_BASE)
        .map(|i| (format!("b{i}"), all_vecs[i].clone()))
        .collect();
    let stream: Vec<(String, Vec<f32>)> = (0..N_STREAM)
        .map(|i| (format!("s{i}"), all_vecs[N_BASE + i].clone()))
        .collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERY)
        .map(|i| all_vecs[N_BASE + N_STREAM + i].clone())
        .collect();

    // Ground truth: brute-force k-NN over the full corpus (base + stream).
    let corpus: Vec<(String, Vec<f32>)> =
        base.iter().chain(stream.iter()).cloned().collect();
    print!("Computing brute-force ground truth ({} vectors × {} queries)... ", corpus.len(), N_QUERY);
    let t_gt = Instant::now();
    let ground_truth: Vec<HashSet<String>> = queries.iter().map(|q| {
        let mut ds: Vec<(usize, f32)> = corpus.iter().enumerate()
            .map(|(i, (_, v))| (i, l2sq(v, q))).collect();
        ds.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        ds[..K].iter().map(|(i, _)| corpus[*i].0.clone()).collect()
    }).collect();
    println!("done ({} ms)\n", t_gt.elapsed().as_millis());

    // ---- Run variants ------------------------------------------------------
    let results = [
        run_variant("A — Static  (full batch, no stream)",
            &corpus, &[], ConsolidationPolicy::Manual, false),
        run_variant("B — Eager   (consolidate per insert)",
            &base, &stream, ConsolidationPolicy::Eager, false),
        run_variant("C — Lazy T=100",
            &base, &stream, ConsolidationPolicy::Lazy(100), false),
        run_variant("D — Buffer  (no consolidation)",
            &base, &stream, ConsolidationPolicy::Manual, false),
    ];

    // Evaluate recall for each variant.
    println!("\n{}", "─".repeat(88));
    println!("{:<42} {:>10} {:>10} {:>12} {:>10}",
        "Variant", "Recall@10", "QPS", "Build (ms)", "Consol (ms)");
    println!("{}", "─".repeat(88));

    for (name, idx, build_ms) in &results {
        let mut total_recall = 0.0f64;
        for (q, gt) in queries.iter().zip(ground_truth.iter()) {
            let found: HashSet<String> = idx.search(q, K).unwrap()
                .into_iter().map(|r| r.id).collect();
            total_recall += gt.intersection(&found).count() as f64 / K as f64;
        }
        let recall = total_recall / queries.len() as f64;

        // QPS: warm up then time.
        for q in queries.iter().take(20) { let _ = idx.search(q, K); }
        let iters = 500usize;
        let t0 = Instant::now();
        for i in 0..iters { let _ = idx.search(&queries[i % queries.len()], K); }
        let qps = iters as f64 / t0.elapsed().as_secs_f64();

        let consol_ms = idx.stats.consolidation_ms;
        println!("{:<42} {:>10.3} {:>10.0} {:>12} {:>10}",
            name, recall, qps, build_ms, consol_ms);
    }

    println!("{}", "─".repeat(88));
    println!("\nNotes:");
    println!("  • Recall@10: fraction of true top-10 neighbours returned by the index.");
    println!("  • QPS measured over {iters} queries after 20-query warm-up.", iters = 500);
    println!("  • Build (ms): time to build the Vamana graph on the base corpus.");
    println!("  • Consol (ms): total time spent wiring buffer vectors into the graph.");
    println!("  • Variant D recall < A/B/C shows the cost of never consolidating.");
    println!("  • Variant B/C recall ≈ A shows streaming inserts preserve search quality.");
}

/// Build the index, apply streaming inserts, return (name, index, build_ms).
fn run_variant<'a>(
    name: &'a str,
    base: &[(String, Vec<f32>)],
    stream: &[(String, Vec<f32>)],
    policy: ConsolidationPolicy,
    _debug: bool,
) -> (&'a str, FreshDiskAnn, u64) {
    print!("  Building {name} ... ");
    let config = FreshConfig {
        dim: DIM,
        max_degree: 32,
        build_beam: 64,
        search_beam: 64,
        alpha: 1.2,
        policy,
    };
    let mut idx = FreshDiskAnn::new(config);
    for (id, v) in base {
        idx.preload(id.clone(), v.clone()).unwrap();
    }
    let t0 = Instant::now();
    idx.build().unwrap();
    let build_ms = t0.elapsed().as_millis() as u64;

    for (id, v) in stream {
        idx.insert(id.clone(), v.clone()).unwrap();
    }
    // Flush remaining buffer (for Manual policy or partial batch in Lazy).
    idx.consolidate();

    println!("done (build={build_ms}ms consol={}ms)", idx.stats.consolidation_ms);
    (name, idx, build_ms)
}
