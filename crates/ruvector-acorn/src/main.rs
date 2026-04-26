//! ACORN filtered-HNSW demo and benchmark harness.
//!
//! Runs three index variants at three predicate selectivities and prints
//! a table of recall@10, QPS, memory (MB), and build time (ms).
//!
//! Usage: cargo run --release -p ruvector-acorn

use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_acorn::{recall_at_k, AcornIndex1, AcornIndexGamma, FilteredIndex, FlatFilteredIndex};

const N: usize = 5_000;
const DIM: usize = 128;
const N_QUERIES: usize = 500;
const K: usize = 10;
fn gaussian_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

/// Measure QPS by running `n_queries` searches and timing the total.
fn bench_qps(
    index: &dyn FilteredIndex,
    queries: &[Vec<f32>],
    k: usize,
    predicate: &dyn Fn(u32) -> bool,
) -> f64 {
    let start = Instant::now();
    for q in queries {
        let _ = index.search(q, k, predicate).unwrap_or_default();
    }
    let elapsed = start.elapsed().as_secs_f64();
    queries.len() as f64 / elapsed
}

/// Selectivity: fraction of n nodes that pass the predicate.
fn selectivity_predicate(n: usize, fraction: f64) -> impl Fn(u32) -> bool + Copy {
    let threshold = (n as f64 * fraction) as u32;
    move |id: u32| id < threshold
}

fn print_header() {
    println!(
        "\n{:<26} {:>6}  {:>8} {:>10} {:>12} {:>10}",
        "Variant", "Sel%", "Rec@10", "QPS", "Mem(MB)", "Build(ms)"
    );
    println!("{}", "-".repeat(78));
}

fn run_variant(
    label: &str,
    index: &dyn FilteredIndex,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    build_ms: f64,
    sel_pct: f64,
    predicate: &(dyn Fn(u32) -> bool + Sync),
) {
    let recall = recall_at_k(data, queries, K, |id| predicate(id), index);
    let qps = bench_qps(index, queries, K, predicate);
    let mem_mb = index.memory_bytes() as f64 / 1_048_576.0;
    println!(
        "{:<26} {:>5.0}%  {:>7.1}% {:>10.0} {:>11.2} {:>10.1}",
        label,
        sel_pct * 100.0,
        recall * 100.0,
        qps,
        mem_mb,
        build_ms,
    );
}

fn main() {
    println!("ACORN Filtered-HNSW Benchmark");
    println!("Dataset: n={N}, D={DIM}, queries={N_QUERIES}, k={K}");
    println!("Hardware: {}", std::env::consts::ARCH);

    let data = gaussian_vectors(N, DIM, 42);
    let queries = gaussian_vectors(N_QUERIES, DIM, 99);

    // --- Build all three indices and record build times ---
    let t0 = Instant::now();
    let flat = FlatFilteredIndex::build(data.clone()).unwrap();
    let flat_build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let acorn1 = AcornIndex1::build(data.clone()).unwrap();
    let acorn1_build_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let t2 = Instant::now();
    let acorng = AcornIndexGamma::build(data.clone()).unwrap();
    let acorng_build_ms = t2.elapsed().as_secs_f64() * 1000.0;

    println!("\nBuild times:");
    println!("  FlatFiltered:  {flat_build_ms:.1} ms");
    println!("  ACORN-1:       {acorn1_build_ms:.1} ms");
    println!("  ACORN-γ (γ=2): {acorng_build_ms:.1} ms");

    // --- Benchmark at three selectivity levels ---
    let selectivities: &[(f64, &str)] = &[(0.50, "50%"), (0.10, "10%"), (0.01, "1%")];

    print_header();

    for &(sel, sel_label) in selectivities {
        let pred = selectivity_predicate(N, sel);

        // Count valid nodes.
        let n_valid = (0..N as u32).filter(|&id| pred(id)).count();
        if n_valid == 0 {
            println!("  [skip {sel_label}: no valid nodes]");
            continue;
        }

        run_variant(
            flat.name(),
            &flat,
            &data,
            &queries,
            flat_build_ms,
            sel,
            &pred,
        );
        run_variant(
            acorn1.name(),
            &acorn1,
            &data,
            &queries,
            acorn1_build_ms,
            sel,
            &pred,
        );
        run_variant(
            acorng.name(),
            &acorng,
            &data,
            &queries,
            acorng_build_ms,
            sel,
            &pred,
        );
        println!();
    }

    // --- Recall vs selectivity sweep for ACORN-γ ---
    println!("\nRecall@10 sweep across selectivities (ACORN-γ vs FlatFiltered):");
    println!(
        "{:>8}  {:>16}  {:>16}",
        "Sel%", "FlatFiltered R@10", "ACORN-γ R@10"
    );
    println!("{}", "-".repeat(44));
    for sel_frac in [0.50, 0.20, 0.10, 0.05, 0.02, 0.01] {
        let pred = selectivity_predicate(N, sel_frac);
        let r_flat = recall_at_k(&data, &queries, K, |id| pred(id), &flat);
        let r_acorn = recall_at_k(&data, &queries, K, |id| pred(id), &acorng);
        println!(
            "{:>7.0}%  {:>16.1}%  {:>16.1}%",
            sel_frac * 100.0,
            r_flat * 100.0,
            r_acorn * 100.0
        );
    }

    // --- Edge count statistics ---
    println!("\nGraph edge statistics:");
    let acorn1_edges: usize = {
        // Access via memory estimate: edges × 4 bytes of the edge list portion.
        // We re-derive from memory_bytes which includes both vectors and edges.
        // Approximation: edges ≈ (memory_bytes - raw_vecs) / 4
        let raw_vecs = N * DIM * 4;
        (acorn1.memory_bytes().saturating_sub(raw_vecs)) / 4
    };
    let acorng_edges: usize = {
        let raw_vecs = N * DIM * 4;
        (acorng.memory_bytes().saturating_sub(raw_vecs)) / 4
    };
    println!("  ACORN-1 total edges: ~{acorn1_edges}");
    println!("  ACORN-γ total edges: ~{acorng_edges}");
    println!(
        "  Edge ratio γ/1: {:.2}×",
        acorng_edges as f64 / acorn1_edges.max(1) as f64
    );

    println!("\nDone.");
}
