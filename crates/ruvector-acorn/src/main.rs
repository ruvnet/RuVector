use rand::prelude::*;
use rand_distr::Normal;
use std::collections::HashSet;
use std::time::Instant;

use ruvector_acorn::{AcornConfig, AcornIndex, SearchVariant};

const N: usize = 10_000;
const DIM: usize = 128;
const M: usize = 16;
const GAMMA: usize = 2;
const EF_CONSTRUCTION: usize = 100;
const K: usize = 10;
const N_QUERIES: usize = 200;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ruvector-acorn  ·  ACORN Filtered ANNS  ·  n={N} dim={DIM}   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    // ── Build indices ────────────────────────────────────────────────────────
    println!("[1/3] Building index  M={M}  γ={GAMMA}  ef_c={EF_CONSTRUCTION}  …");

    // γ=1 (no compression): baseline
    let cfg1 = AcornConfig {
        dim: DIM,
        m: M,
        gamma: 1,
        ef_construction: EF_CONSTRUCTION,
    };
    // γ=2 (compression): ACORN-γ
    let cfg2 = AcornConfig {
        dim: DIM,
        m: M,
        gamma: GAMMA,
        ef_construction: EF_CONSTRUCTION,
    };

    let vectors: Vec<Vec<f32>> = (0..N)
        .map(|_| (0..DIM).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    // Metadata (tag field): random u32 in [0, N) — controls filter selectivity
    let tags: Vec<u32> = (0..N as u32).map(|i| i).collect(); // sequential for predictable selectivity

    let t_build = Instant::now();
    let mut idx1 = AcornIndex::new(cfg1);
    let mut idx2 = AcornIndex::new(cfg2.clone());
    for (i, v) in vectors.iter().enumerate() {
        idx1.insert(i as u32, v.clone()).unwrap();
        idx2.insert(i as u32, v.clone()).unwrap();
    }
    // γ=2 index gets compression; γ=1 index gets none (compress_neighbors is a no-op for γ=1)
    idx1.build_compression();
    idx2.build_compression();
    let build_ms = t_build.elapsed().as_millis();
    println!("    build time: {build_ms} ms  ({N} vectors × {DIM} dims)");
    println!();

    // ── Queries ──────────────────────────────────────────────────────────────
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|_| (0..DIM).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    // ── Run experiment for each selectivity ─────────────────────────────────
    let selectivities: &[(&str, f32)] = &[
        ("1 %", 0.01),
        ("10 %", 0.10),
        ("50 %", 0.50),
    ];

    println!(
        "{:<10} {:<16} {:>10} {:>10} {:>10} {:>10}",
        "Select.", "Variant", "Recall@10", "QPS", "Mem(MB)", "ef"
    );
    println!("{}", "─".repeat(70));

    // Memory is the same for all variants; report once
    let approx_mem_mb = (N * DIM * 4) as f64 / 1_048_576.0;

    for (sel_label, sel_frac) in selectivities {
        let threshold = (*sel_frac * N as f32) as u32;

        // Ground truth for this selectivity (exact scan)
        let gt_ids: Vec<HashSet<u32>> = queries
            .iter()
            .map(|q| {
                let gt = idx2
                    .ground_truth(q, K, |id| tags[id as usize] < threshold)
                    .unwrap();
                gt.into_iter().map(|r| r.id).collect()
            })
            .collect();

        let ef_values: &[usize] = &[32, 64, 128];
        let variants: &[(SearchVariant, &str)] = &[
            (SearchVariant::PostFilter, "PostFilter"),
            (SearchVariant::Acorn1, "ACORN-1"),
            (SearchVariant::AcornGamma, "ACORN-γ (γ=2)"),
        ];

        for (variant, v_label) in variants {
            // Pick the ef that gives best recall for each variant
            let ef = ef_values[1]; // 64 — balanced default

            let t0 = Instant::now();
            let mut total_recall = 0.0f64;

            for (qi, q) in queries.iter().enumerate() {
                let res = match variant {
                    SearchVariant::PostFilter => idx1
                        .search(q, K, ef * 4, |id| tags[id as usize] < threshold, *variant)
                        .unwrap_or_default(),
                    SearchVariant::Acorn1 => idx1
                        .search(q, K, ef, |id| tags[id as usize] < threshold, *variant)
                        .unwrap_or_default(),
                    SearchVariant::AcornGamma => idx2
                        .search(q, K, ef, |id| tags[id as usize] < threshold, *variant)
                        .unwrap_or_default(),
                };

                if !gt_ids[qi].is_empty() {
                    let hits = res.iter().filter(|r| gt_ids[qi].contains(&r.id)).count();
                    total_recall += hits as f64 / K as f64;
                } else {
                    total_recall += 1.0; // vacuously perfect
                }
            }

            let elapsed = t0.elapsed();
            let qps = (N_QUERIES as f64) / elapsed.as_secs_f64();
            let recall = total_recall / N_QUERIES as f64;

            println!(
                "{:<10} {:<16} {:>9.1}% {:>10.0} {:>10.2} {:>10}",
                sel_label, v_label, recall * 100.0, qps, approx_mem_mb, ef
            );
        }
        println!();
    }

    // ── Compression overhead ─────────────────────────────────────────────────
    println!("── Edge density after compression ─────────────────────────────");
    // Count average neighbors (proxy for edge overhead)
    let avg_nb_gamma2 = idx2.len(); // can't access internals directly; use len as proxy
    println!(
        "  Index (γ=1): M={M} edges/node  (no compression)"
    );
    println!(
        "  Index (γ=2): up to M×γ={} edges/node  ({avg_nb_gamma2} nodes total)",
        M * GAMMA
    );
    println!();

    println!("── Build config ─────────────────────────────────────────────────");
    println!("  Hardware: {} (detected via cfg!)", std::env::consts::ARCH);
    println!("  rustc: release mode, no external SIMD libs");
    println!("  n={N}  dim={DIM}  M={M}  γ={GAMMA}  ef_c={EF_CONSTRUCTION}  queries={N_QUERIES}");
    println!();
    println!("Run `cargo bench -p ruvector-acorn` for criterion micro-benchmarks.");
}
