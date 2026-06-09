//! Benchmark binary for ruvector-memory-compact.
//!
//! Generates a synthetic clustered dataset (realistic for agent episodic memory:
//! groups of related memories around topic centroids), then measures compaction
//! ratio, recall@10, and wall-clock time for three variants.
//!
//! Variants:
//!   1. naive-kmeans     — Lloyd's K-means centroid replacement
//!   2. graph-merge      — threshold-based k-NN graph merge
//!   3. coherence-gated  — adaptive coherence-weighted graph merge
//!
//! Usage:
//!   cargo run --release -p ruvector-memory-compact
//!   N_TOPICS=20 VECS_PER_TOPIC=100 cargo run --release -p ruvector-memory-compact

use std::time::Instant;

use rand::prelude::*;
use rand::rngs::StdRng;
use ruvector_memory_compact::{
    run_compaction, CoherenceGatedCompactor, Compactor, GraphMergeCompactor, MemoryStore,
    NaiveCompactor,
};

/// Generate a clustered dataset: `n_topics` topic centroids, each with
/// `vecs_per_topic` memories drawn from a von-Mises-like Gaussian around
/// the centroid. All vectors are L2-normalised to the unit sphere.
fn generate_clustered(
    n_topics: usize,
    vecs_per_topic: usize,
    dim: usize,
    noise_scale: f32,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Draw `n_topics` random centroids.
    let centroids: Vec<Vec<f32>> = (0..n_topics)
        .map(|_| l2_normalise((0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()))
        .collect();

    // For each centroid, draw `vecs_per_topic` noisy variants.
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n_topics * vecs_per_topic);
    for c in &centroids {
        for _ in 0..vecs_per_topic {
            let noisy: Vec<f32> = c
                .iter()
                .map(|&x| x + rng.gen::<f32>() * noise_scale)
                .collect();
            embeddings.push(l2_normalise(noisy));
        }
    }
    embeddings.shuffle(&mut rng);

    // Queries are drawn from the same distribution (topic centroids + small noise).
    let queries: Vec<Vec<f32>> = centroids
        .iter()
        .map(|c| {
            let noisy: Vec<f32> = c
                .iter()
                .map(|&x| x + rng.gen::<f32>() * noise_scale * 0.5)
                .collect();
            l2_normalise(noisy)
        })
        .collect();

    (embeddings, queries)
}

fn l2_normalise(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn build_store(embeddings: &[Vec<f32>]) -> MemoryStore {
    let mut store = MemoryStore::new();
    for (i, emb) in embeddings.iter().enumerate() {
        store.insert(emb.clone(), format!("mem-{i}"));
    }
    store
}

fn latency_sweep(
    compactor: &dyn Compactor,
    embeddings: &[Vec<f32>],
    queries: &[Vec<f32>],
    target_ratio: f64,
    k: usize,
    runs: usize,
) -> (f64, u128, u128) {
    let mut times: Vec<u128> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut store = build_store(embeddings);
        let t0 = Instant::now();
        let _ = run_compaction(compactor, &mut store, target_ratio, queries, k);
        times.push(t0.elapsed().as_millis());
    }
    times.sort();
    let mean = times.iter().sum::<u128>() as f64 / times.len() as f64;
    let p50 = times[times.len() / 2];
    let p95 = times[((times.len() as f64) * 0.95) as usize];
    (mean, p50, p95)
}

fn main() {
    let n_topics: usize = std::env::var("N_TOPICS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let vecs_per_topic: usize = std::env::var("VECS_PER_TOPIC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let dim: usize = std::env::var("DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let noise_scale: f32 = 0.15; // Controls cluster tightness (lower = tighter)
    let target_ratio: f64 = 0.40; // Keep 40% of vectors (60% compaction)
    let k: usize = 10;
    let sweep_runs: usize = 5;

    let n = n_topics * vecs_per_topic;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          ruvector-memory-compact  benchmark                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("OS          : {}", std::env::consts::OS);
    println!("Arch        : {}", std::env::consts::ARCH);
    if let Ok(v) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        if let Ok(s) = std::str::from_utf8(&v.stdout) {
            println!("Rust        : {}", s.trim());
        }
    }
    println!();
    println!("Dataset     : {n_topics} topics × {vecs_per_topic} vecs = N={n}  dim={dim}");
    println!("Noise scale : {noise_scale}  (intra-topic perturbation)");
    println!(
        "Target keep : {:.0}%  ({:.0}% compaction)",
        target_ratio * 100.0,
        (1.0 - target_ratio) * 100.0
    );
    println!("Queries     : {n_topics} (one per topic centroid)  k={k}");
    println!();

    let (embeddings, queries) = generate_clustered(n_topics, vecs_per_topic, dim, noise_scale, 42);

    // ── Compactors under test ─────────────────────────────────────────────
    let compactors: Vec<(&str, Box<dyn Compactor>)> = vec![
        ("naive-kmeans", Box::new(NaiveCompactor::default())),
        (
            "graph-merge",
            Box::new(GraphMergeCompactor {
                graph_k: 15,
                merge_threshold: None,
            }),
        ),
        (
            "coherence-gated",
            Box::new(CoherenceGatedCompactor {
                graph_k: 15,
                coherence_floor: 0.30,
                max_cluster: 30,
            }),
        ),
    ];

    let recall_threshold = 0.55_f64;
    let mut all_pass = true;

    // ── Primary results table ─────────────────────────────────────────────
    println!(
        "{:<20} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Variant", "N→M", "Compact%", "Recall@10", "Time(ms)", "Mem(MB)", "Pass"
    );
    println!("{}", "─".repeat(78));

    let mut summary: Vec<(String, usize, usize, f64, f64, u64)> = Vec::new();

    for (name, compactor) in &compactors {
        let mut store = build_store(&embeddings);
        let result = run_compaction(compactor.as_ref(), &mut store, target_ratio, &queries, k);
        let pass = result.recall_at_k >= recall_threshold;
        if !pass {
            all_pass = false;
        }
        let mem_after_mb = (result.compacted_count * dim * 4) as f64 / 1_048_576.0;

        println!(
            "{:<20} {:>4}→{:<4} {:>9.1}% {:>10.3} {:>10} {:>9.3} {:>8}",
            name,
            result.original_count,
            result.compacted_count,
            result.compaction_ratio * 100.0,
            result.recall_at_k,
            result.duration_ms,
            mem_after_mb,
            if pass { "PASS" } else { "FAIL" },
        );
        summary.push((
            name.to_string(),
            result.original_count,
            result.compacted_count,
            result.compaction_ratio,
            result.recall_at_k,
            result.duration_ms,
        ));
    }
    println!("{}", "─".repeat(78));
    println!();

    // ── Latency sweep ─────────────────────────────────────────────────────
    println!("Latency sweep ({sweep_runs} runs each):");
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>14}",
        "Variant", "Mean(ms)", "p50(ms)", "p95(ms)", "Throughput/s"
    );
    println!("{}", "─".repeat(66));
    for (name, compactor) in &compactors {
        let (mean, p50, p95) = latency_sweep(
            compactor.as_ref(),
            &embeddings,
            &queries,
            target_ratio,
            k,
            sweep_runs,
        );
        let tput = n as f64 / (mean / 1000.0).max(1e-9);
        println!(
            "{:<20} {:>10.1} {:>10} {:>10} {:>14.0}",
            name, mean, p50, p95, tput
        );
    }
    println!("{}", "─".repeat(66));
    println!();

    // ── Memory math ───────────────────────────────────────────────────────
    let raw_mb = (n * dim * 4) as f64 / 1_048_576.0;
    let target_count = ((n as f64) * target_ratio) as usize;
    let compact_mb = (target_count * dim * 4) as f64 / 1_048_576.0;
    println!("Memory math:");
    println!("  Raw store          : {n} × {dim} × 4 B = {raw_mb:.3} MB");
    println!(
        "  Target keep ({target_ratio:.0}%) : {target_count} × {dim} × 4 B = {compact_mb:.3} MB"
    );
    println!(
        "  Theoretical reduction : {:.2}x",
        raw_mb / compact_mb.max(1e-9)
    );
    println!();

    // ── Witness chain sample ──────────────────────────────────────────────
    {
        let mut store = build_store(&embeddings);
        let cg = CoherenceGatedCompactor {
            graph_k: 15,
            coherence_floor: 0.30,
            max_cluster: 30,
        };
        let result = run_compaction(&cg, &mut store, target_ratio, &queries, k);
        let n_witnesses = result.witness_records.len();
        let total_merged: usize = result
            .witness_records
            .iter()
            .map(|w| w.merged_ids.len())
            .sum();
        let avg_size = if n_witnesses > 0 {
            total_merged as f64 / n_witnesses as f64
        } else {
            1.0
        };
        let avg_intra: f32 = if n_witnesses > 0 {
            result
                .witness_records
                .iter()
                .map(|w| w.intra_sim)
                .sum::<f32>()
                / n_witnesses as f32
        } else {
            1.0
        };
        println!("Coherence-gated witness chain sample:");
        println!("  Clusters (witness records) : {n_witnesses}");
        println!("  Total original IDs recorded: {total_merged}");
        println!("  Avg cluster size            : {avg_size:.2}");
        println!("  Avg intra-cluster cosine sim: {avg_intra:.4}");
        if let Some(w) = result.witness_records.first() {
            println!(
                "  Example: centroid #{} ← merged {:?} (intra={:.4})",
                w.centroid_id,
                &w.merged_ids[..w.merged_ids.len().min(4)],
                w.intra_sim
            );
        }
    }
    println!();

    // ── Acceptance ────────────────────────────────────────────────────────
    println!(
        "Acceptance threshold : recall@{k} ≥ {recall_threshold:.2}  →  {}",
        if all_pass {
            "ALL PASS ✓"
        } else {
            "SOME FAIL ✗ — see details above"
        }
    );

    if !all_pass {
        std::process::exit(1);
    }
}
