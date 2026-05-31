//! Streaming HNSW benchmark demo.
//!
//! Usage: cargo run --release -p ruvector-streaming-hnsw

use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use ruvector_streaming_hnsw::{AnnIndex, FlatL2Index, StaticHnsw, StreamingHnsw, recall_at_k};

const N: usize = 5_000;
const DIM: usize = 128;
const N_QUERIES: usize = 500;
const K: usize = 10;
const M: usize = 16;   // neighbors per node
const EF: usize = 40;  // beam width during search

fn gaussian_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0_f32, 1.0).unwrap();
    (0..n).map(|_| (0..dim).map(|_| dist.sample(&mut rng)).collect()).collect()
}

fn bench_qps(index: &dyn AnnIndex, queries: &[Vec<f32>], k: usize) -> f64 {
    let start = Instant::now();
    for q in queries {
        let _ = index.search(q, k).unwrap_or_default();
    }
    queries.len() as f64 / start.elapsed().as_secs_f64()
}

fn main() {
    println!("Streaming HNSW Benchmark");
    println!("Dataset: n={N}, D={DIM}, queries={N_QUERIES}, k={K}, M={M}, ef={EF}");
    println!("Hardware: {}", std::env::consts::ARCH);
    println!();

    let data = gaussian_vecs(N, DIM, 42);
    let queries = gaussian_vecs(N_QUERIES, DIM, 99);

    // ── Flat baseline ────────────────────────────────────────────────────────
    let t = Instant::now();
    let mut flat = FlatL2Index::new(DIM);
    for v in &data { flat.insert(v.clone()).unwrap(); }
    let flat_build_ms = t.elapsed().as_secs_f64() * 1000.0;

    // ── Static HNSW ──────────────────────────────────────────────────────────
    let t = Instant::now();
    let static_idx = StaticHnsw::build(data.clone(), M, EF).unwrap();
    let static_build_ms = t.elapsed().as_secs_f64() * 1000.0;

    // ── Streaming HNSW ────────────────────────────────────────────────────────
    let t = Instant::now();
    let stream_idx = StreamingHnsw::from_vecs(data.clone(), M, EF).unwrap();
    let stream_build_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!("Build times:");
    println!("  FlatL2:        {:>8.1} ms", flat_build_ms);
    println!("  StaticHnsw:    {:>8.1} ms", static_build_ms);
    println!("  StreamingHnsw: {:>8.1} ms", stream_build_ms);
    println!();

    println!("{:<28} {:>9} {:>10} {:>12} {:>11}",
             "Variant", "Rec@10", "QPS", "Mem(MB)", "Build(ms)");
    println!("{}", "-".repeat(74));

    for (name, idx, build_ms) in &[
        (flat.name(),         &flat as &dyn AnnIndex,         flat_build_ms),
        (static_idx.name(),   &static_idx as &dyn AnnIndex,   static_build_ms),
        (stream_idx.name(),   &stream_idx as &dyn AnnIndex,   stream_build_ms),
    ] {
        let recall = recall_at_k(&data, &queries, K, *idx);
        let qps    = bench_qps(*idx, &queries, K);
        let mem_mb = idx.memory_bytes() as f64 / 1_048_576.0;
        println!("{:<28} {:>8.1}% {:>10.0} {:>11.2} {:>11.1}",
                 name, recall * 100.0, qps, mem_mb, build_ms);
    }

    // ── Streaming insert benchmark ────────────────────────────────────────────
    println!();
    println!("Concurrent-insert throughput (4 threads × 500 inserts):");
    use std::sync::Arc;
    let concurrent_idx = Arc::new(StreamingHnsw::new(DIM, M, EF));
    // Seed index with 1K vectors first
    for v in gaussian_vecs(1_000, DIM, 77) { concurrent_idx.insert(v).unwrap(); }

    let idx_ref = Arc::clone(&concurrent_idx);
    let t = Instant::now();
    let mut handles = Vec::new();
    for thread in 0u64..4 {
        let idx = Arc::clone(&idx_ref);
        handles.push(std::thread::spawn(move || {
            let vecs = gaussian_vecs(500, DIM, thread * 1000 + 1);
            for v in vecs { idx.insert(v).unwrap(); }
        }));
    }
    for h in handles { h.join().unwrap(); }
    let elapsed = t.elapsed().as_secs_f64();
    let total_inserts = 2_000u64;
    println!("  Inserted {} vecs in {:.3} s → {:.0} inserts/sec",
             total_inserts, elapsed, total_inserts as f64 / elapsed);
    println!("  Final index size: {} nodes", concurrent_idx.len());

    println!();
    println!("Done.");
}
