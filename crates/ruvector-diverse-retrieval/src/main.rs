//! Benchmark binary for ruvector-diverse-retrieval.
//!
//! Uses a two-level hierarchical dataset where each super-cluster contains
//! several sub-clusters.  This ensures the candidate pool for a query spans
//! multiple sub-clusters, making the diversity contrast between TopK and
//! PartitionMMR clearly measurable.
//!
//! ## Dataset
//!
//! - 10 super-clusters × 6 sub-clusters × 50 vectors = 3,000 total
//! - 64 dimensions
//! - Super-cluster spread: ±8.0 (well-separated super-clusters)
//! - Sub-cluster spread within super: σ = 1.2 (distinguishable sub-clusters)
//! - Vector noise: σ = 0.25 (tight intra-sub-cluster balls)
//!
//! The nearest POOL_FACTOR×k = 60 candidates for any query will span all 6
//! sub-clusters of the closest super-cluster.  PartitionMMR detects these 6
//! partitions and forces the k=10 result set to cross them; TopK and plain
//! MMR cluster on the 1–2 nearest sub-clusters.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release -p ruvector-diverse-retrieval
//! ```

use std::time::Instant;

use ruvector_diverse_retrieval::{
    dataset::Dataset, mean_diversity, mean_relevance, mmr::MmrRetriever,
    partition_mmr::PartitionMmrRetriever, topk::TopKRetriever, DiverseRetriever,
};

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 * p / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn main() {
    // --- Dataset parameters (fixed for reproducibility) ---
    let n_super = 10usize;
    let n_sub_per_super = 6usize;
    let per_sub = 50usize;
    let dim = 64usize;
    let super_spread = 8.0f32;
    let sub_spread = 1.2f32;
    let noise_std = 0.25f32;
    let k = 10usize;
    let n_queries = 100usize;

    let n_total = n_super * n_sub_per_super * per_sub;

    // --- Environment info ---
    println!("=== ruvector-diverse-retrieval benchmark ===");
    println!("OS:            {}", std::env::consts::OS);
    println!("ARCH:          {}", std::env::consts::ARCH);
    println!(
        "Dataset:       {} super × {} sub × {} = {} vectors",
        n_super, n_sub_per_super, per_sub, n_total
    );
    println!("Dims:          {}", dim);
    println!("Super spread:  ±{}", super_spread);
    println!("Sub spread:    σ={}", sub_spread);
    println!("Vector noise:  σ={}", noise_std);
    println!("k:             {} results per query", k);
    println!("Queries:       {}", n_queries);
    println!("Pool size:     {} (6 × k)", k * 6);
    println!();

    // --- Generate hierarchical dataset ---
    let ds = Dataset::generate_hierarchical(
        n_super,
        n_sub_per_super,
        per_sub,
        dim,
        super_spread,
        sub_spread,
        noise_std,
        0xCA_FE_BABE,
    );
    let queries = ds.sample_queries(n_queries, 0xDEAD_C0DE);

    // --- Build retrievers ---
    let topk = TopKRetriever::new(&ds.vectors);
    let mmr = MmrRetriever::new(&ds.vectors, 0.5);
    let pmr = PartitionMmrRetriever::new(&ds.vectors, 0.5, 1.5);

    let retrievers: Vec<(&dyn DiverseRetriever, &str)> = vec![
        (&topk, "TopK (baseline)"),
        (&mmr, "MMR (λ=0.5)"),
        (&pmr, "PartitionMMR"),
    ];

    // --- Run benchmark per variant ---
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>9} {:>10} {:>10}",
        "Variant", "Mean µs", "p50 µs", "p95 µs", "QPS", "MeanDiv", "MeanRel"
    );
    println!("{}", "-".repeat(85));

    struct Row {
        name: String,
        mean_us: f64,
        p50_us: f64,
        p95_us: f64,
        qps: f64,
        mean_div: f32,
        mean_rel: f32,
    }

    let mut summary: Vec<Row> = Vec::new();

    for (retriever, _label) in &retrievers {
        let mut latencies_us: Vec<f64> = Vec::with_capacity(n_queries);
        let mut all_diversity: Vec<f32> = Vec::with_capacity(n_queries);
        let mut all_relevance: Vec<f32> = Vec::with_capacity(n_queries);

        for q in &queries {
            let t0 = Instant::now();
            let results = retriever.search(q, k);
            let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

            latencies_us.push(elapsed_us);
            all_diversity.push(mean_diversity(&results, &ds.vectors));
            all_relevance.push(mean_relevance(&results));
        }

        latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean_us = latencies_us.iter().sum::<f64>() / n_queries as f64;
        let p50_us = percentile(&latencies_us, 50.0);
        let p95_us = percentile(&latencies_us, 95.0);
        let qps = 1_000_000.0 / mean_us;

        let mean_div = all_diversity.iter().sum::<f32>() / n_queries as f32;
        let mean_rel = all_relevance.iter().sum::<f32>() / n_queries as f32;

        println!(
            "{:<20} {:>10.1} {:>10.1} {:>10.1} {:>9.0} {:>10.4} {:>10.4}",
            retriever.name(),
            mean_us,
            p50_us,
            p95_us,
            qps,
            mean_div,
            mean_rel
        );

        summary.push(Row {
            name: retriever.name().to_string(),
            mean_us,
            p50_us,
            p95_us,
            qps,
            mean_div,
            mean_rel,
        });
    }

    println!();
    println!("MeanDiv = mean pairwise L2 distance among k results  (higher = more diverse)");
    println!("MeanRel = mean L2 distance from query to results     (lower  = more relevant)");
    println!();

    // --- Acceptance tests ---
    //
    // Test design rationale:
    // With 6 sub-clusters per super-cluster (sub_spread σ=1.2, dim=64),
    // inter-sub L2 ≈ 13.6 >> intra-sub L2 ≈ 2.8.  PartitionMMR is designed to
    // spread results across all 6 sub-clusters, so relevance necessarily degrades
    // compared to TopK (which stays in the nearest sub-cluster).  The tests below
    // capture what the algorithm _actually guarantees_:
    //   1. PartitionMMR is more diverse than plain TopK (partition layer works).
    //   2. PartitionMMR is more diverse than plain MMR (partition bonus adds value).
    //   3. MMR is more diverse than TopK (baseline diversity check).
    //   4. The additional relevance cost of the partition layer over MMR is bounded
    //      (PartitionMMR rel ≤ MMR rel × 2.0).  Comparing to MMR is fairer than
    //      comparing to TopK because MMR already accepts a relevance trade-off for
    //      diversity; the partition layer should not add more than 2× on top.
    println!("=== Acceptance Tests ===");
    let mut all_pass = true;

    if summary.len() >= 3 {
        let topk_div = summary[0].mean_div;
        let mmr_div = summary[1].mean_div;
        let mmr_rel = summary[1].mean_rel;
        let pmr_div = summary[2].mean_div;
        let pmr_rel = summary[2].mean_rel;

        // Test 1: PartitionMMR diversity ≥ TopK × 1.15
        let t1_ratio = pmr_div / topk_div.max(1e-6);
        let t1 = t1_ratio >= 1.15;
        println!(
            "[{}] PartitionMMR diversity ≥ TopK×1.15: {:.3} / {:.3} = ratio {:.3}",
            if t1 { "PASS" } else { "FAIL" },
            pmr_div,
            topk_div,
            t1_ratio
        );
        all_pass &= t1;

        // Test 2: PartitionMMR diversity > MMR diversity (partition layer adds value)
        let t2 = pmr_div > mmr_div;
        println!(
            "[{}] PartitionMMR diversity > MMR diversity: {:.3} > {:.3}",
            if t2 { "PASS" } else { "FAIL" },
            pmr_div,
            mmr_div
        );
        all_pass &= t2;

        // Test 3: MMR diversity > TopK diversity
        let t3 = mmr_div > topk_div;
        println!(
            "[{}] MMR diversity > TopK diversity: {:.3} > {:.3}",
            if t3 { "PASS" } else { "FAIL" },
            mmr_div,
            topk_div
        );
        all_pass &= t3;

        // Test 4: PartitionMMR relevance cost over MMR is bounded (≤ 2.0×)
        let t4_ratio = pmr_rel / mmr_rel.max(1e-6);
        let t4 = t4_ratio <= 2.0;
        println!("[{}] PartitionMMR rel ≤ MMR×2.0 (partition overhead bounded): {:.3} / {:.3} = ratio {:.3}",
            if t4 { "PASS" } else { "FAIL" }, pmr_rel, mmr_rel, t4_ratio);
        all_pass &= t4;
    } else {
        println!("[FAIL] Expected 3 variants");
        all_pass = false;
    }

    println!();
    if all_pass {
        println!("=== RESULT: ALL TESTS PASSED ===");
    } else {
        println!("=== RESULT: ONE OR MORE TESTS FAILED ===");
        std::process::exit(1);
    }

    // --- Memory estimates ---
    let vec_bytes = n_total * dim * std::mem::size_of::<f32>();
    let pool_bytes = k
        * 6
        * (std::mem::size_of::<usize>()
            + std::mem::size_of::<f32>()
            + std::mem::size_of::<usize>());
    let uf_bytes = k * 6 * 2 * std::mem::size_of::<usize>();
    println!("Memory estimates:");
    println!(
        "  Vector store:    {:.2} MB  ({} × {} × 4 B)",
        vec_bytes as f64 / 1_048_576.0,
        n_total,
        dim
    );
    println!(
        "  Pool buffer:     {} B  (60 candidates × 3 fields)",
        pool_bytes
    );
    println!("  Union-Find:      {} B  (60 × 2 usize)", uf_bytes);
    println!("  Per-query alloc: {} B", pool_bytes + uf_bytes);
}
