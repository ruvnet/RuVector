//! SQ-HNSW benchmark: four variants, real latency and recall measurements.
//!
//! Usage:
//!   cargo run --release -p ruvector-sq-hnsw --example benchmark
//!   cargo run --release -p ruvector-sq-hnsw --example benchmark -- 20000 128 200

use ruvector_sq_hnsw::{
    recall_at_k, FlatExact, FlatSq8, GraphSq4, GraphSq8, NnSearch, ScalarQuantizer, SqHnsw2,
};
use std::time::Instant;

// ─── Deterministic LCG ───────────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn seed(s: u64) -> Self {
        Self(s)
    }
    fn u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.u64() >> 33) as f32 / u32::MAX as f32
    }
    fn gaussian(&mut self) -> f32 {
        let u1 = self.f32().max(1e-9);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn vec(&mut self, dims: usize) -> Vec<f32> {
        (0..dims).map(|_| self.gaussian()).collect()
    }
}

// ─── Stat helpers ────────────────────────────────────────────────────────────

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() as f64 - 1.0) * p / 100.0).round() as usize;
    v[idx.min(v.len() - 1)]
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let n_queries: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let k: usize = 10;

    // NSW parameters
    let m: usize = 16;
    let ef_build: usize = 200;
    let ef_search8: usize = 200;
    let ef_search4: usize = 250;
    let ef_flat: usize = k * 10;
    // seed_step: one seed per ~100 insertions → ~100 seeds at n=10K
    let seed_step: usize = n / 100 + 1;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        RuVector SQ-HNSW Nightly Benchmark 2026-07-11           ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ OS:     {:<57}║", std::env::consts::OS);
    println!("║ Arch:   {:<57}║", std::env::consts::ARCH);
    println!("║ N:      {:<57}║", n);
    println!("║ Dims:   {:<57}║", dims);
    println!("║ K:      {:<57}║", k);
    println!("║ Queries:{:<57}║", n_queries);
    println!("║ M:      {:<57}║", m);
    println!("║ ef_build:{:<56}║", ef_build);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Generate corpus ─────────────────────────────────────────────────────
    print!("Generating corpus ({n} × {dims}) … ");
    let mut rng = Rng::seed(0xDEAD_BEEF);
    let corpus: Vec<Vec<f32>> = (0..n).map(|_| rng.vec(dims)).collect();
    println!("done");

    // Train quantizers
    let sq8 = ScalarQuantizer::train(&corpus, 8);
    let sq4 = ScalarQuantizer::train(&corpus, 4);

    // ── Build indices ────────────────────────────────────────────────────────
    print!("Building FlatExact … ");
    let t0 = Instant::now();
    let mut flat_exact = FlatExact::new();
    for v in &corpus {
        flat_exact.insert(v.clone());
    }
    println!("{:.1?}", t0.elapsed());

    print!("Building FlatSq8 … ");
    let t0 = Instant::now();
    let mut flat_sq8 = FlatSq8::new(ef_flat);
    flat_sq8.init_quantizer(ScalarQuantizer::train(&corpus, 8));
    for v in &corpus {
        flat_sq8.insert(v.clone());
    }
    println!("{:.1?}", t0.elapsed());

    print!("Building GraphSq8 (M={m}, ef_build={ef_build}) … ");
    let t0 = Instant::now();
    let mut graph_sq8 = GraphSq8::new(sq8, m, ef_build, seed_step, ef_search8);
    for v in &corpus {
        graph_sq8.insert(v.clone());
    }
    let build_sq8 = t0.elapsed();
    println!("{:.1?}", build_sq8);

    print!("Building GraphSq4 (M={m}, ef_build={ef_build}) … ");
    let t0 = Instant::now();
    let mut graph_sq4 = GraphSq4::new(sq4, m, ef_build, seed_step, ef_search4);
    for v in &corpus {
        graph_sq4.insert(v.clone());
    }
    let build_sq4 = t0.elapsed();
    println!("{:.1?}", build_sq4);

    // 2-layer HNSW: L1 period = M → ~n/M sparse upper nodes
    let sq8_hnsw = ScalarQuantizer::train(&corpus, 8);
    let l1_period = m; // every M-th node joins L1 → n/M ≈ 625 L1 nodes
    let ef_hnsw = ef_search8;
    print!(
        "Building SqHnsw2 (M0={}, M1={m}, L1 period={l1_period}) … ",
        m * 2
    );
    let t0 = Instant::now();
    let mut sq_hnsw2 = SqHnsw2::new(sq8_hnsw, m * 2, m, ef_build, m * 4, l1_period, ef_hnsw);
    for v in &corpus {
        sq_hnsw2.insert(v.clone());
    }
    let build_hnsw = t0.elapsed();
    println!("{:.1?}", build_hnsw);

    println!();

    // ── Queries ──────────────────────────────────────────────────────────────
    let mut q_rng = Rng::seed(0xCAFE_BABE);
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| q_rng.vec(dims)).collect();

    struct Stats {
        name: &'static str,
        latencies_us: Vec<f64>,
        recalls: Vec<f32>,
        mem_mb: f64,
    }

    let run = |idx: &dyn NnSearch, name: &'static str| -> Stats {
        let mut latencies_us = Vec::with_capacity(n_queries);
        let mut recalls = Vec::with_capacity(n_queries);
        for q in &queries {
            let gt = flat_exact.search(q, k);
            let t = Instant::now();
            let res = idx.search(q, k);
            latencies_us.push(t.elapsed().as_secs_f64() * 1e6);
            recalls.push(recall_at_k(&gt, &res, k));
        }
        let mem_mb = idx.memory_bytes() as f64 / 1_048_576.0;
        Stats {
            name,
            latencies_us,
            recalls,
            mem_mb,
        }
    };

    let variants: Vec<Stats> = vec![
        run(&flat_exact, "FlatExact"),
        run(&flat_sq8, "FlatSq8  "),
        run(&graph_sq8, "NSW-SQ8  "),
        run(&graph_sq4, "NSW-SQ4  "),
        run(&sq_hnsw2, "HNSW2-SQ8"),
    ];

    // ── Results table ────────────────────────────────────────────────────────
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>12} {:>10} {:>9}",
        "Variant", "Mean(μs)", "p50(μs)", "p95(μs)", "QPS", "Mem(MB)", "Recall@10"
    );
    println!("{}", "─".repeat(82));

    let mut accept = true;
    for s in &variants {
        let mean = s.latencies_us.iter().sum::<f64>() / s.latencies_us.len() as f64;
        let p50 = percentile(s.latencies_us.clone(), 50.0);
        let p95 = percentile(s.latencies_us.clone(), 95.0);
        let qps = 1e6 / mean;
        let recall = s.recalls.iter().sum::<f32>() / s.recalls.len() as f32;

        println!(
            "{:<12} {:>10.1} {:>10.1} {:>10.1} {:>12.0} {:>10.2} {:>9.3}",
            s.name, mean, p50, p95, qps, s.mem_mb, recall
        );

        // Acceptance thresholds (NSW recall < HNSW due to flat graph; see research doc)
        let threshold = match s.name.trim() {
            "FlatSq8" => 0.90,
            "NSW-SQ8" => 0.70,   // NSW flat graph limit at 128 dims
            "NSW-SQ4" => 0.60,   // 4-bit compounds NSW limitation
            "HNSW2-SQ8" => 0.88, // 2-layer HNSW target
            _ => 0.00,
        };
        if threshold > 0.0 && recall < threshold {
            eprintln!(
                "FAIL: {} recall@{k} = {:.3} < threshold {:.2}",
                s.name.trim(),
                recall,
                threshold
            );
            accept = false;
        }
    }

    println!();
    println!("Memory comparison:");
    println!(
        "  Full-precision f32:  {:.2} MB",
        (n * dims * 4) as f64 / 1_048_576.0
    );
    println!(
        "  SQ8 codes only:      {:.2} MB (4× compression)",
        (n * dims) as f64 / 1_048_576.0
    );
    println!(
        "  SQ4 codes only:      {:.2} MB (8× compression)",
        (n * (dims + 1) / 2) as f64 / 1_048_576.0
    );
    println!(
        "  NSW-SQ8 total:       {:.2} MB (codes + originals + edges)",
        variants[2].mem_mb
    );
    println!(
        "  NSW-SQ4 total:       {:.2} MB (codes + originals + edges)",
        variants[3].mem_mb
    );
    println!(
        "  HNSW2-SQ8 total:     {:.2} MB (2-layer codes + originals + edges)",
        variants[4].mem_mb
    );
    println!();
    println!("Build times:");
    println!("  NSW-SQ8:   {:.1?}", build_sq8);
    println!("  NSW-SQ4:   {:.1?}", build_sq4);
    println!("  HNSW2-SQ8: {:.1?}", build_hnsw);
    println!();
    println!("Note: NSW recall is bounded by flat-graph connectivity at high dimensions.");
    println!("      HNSW2 adds a sparse upper layer for diverse entry points.");
    println!();

    if accept {
        println!("ACCEPTANCE: PASS — all recall thresholds met.");
        std::process::exit(0);
    } else {
        println!("ACCEPTANCE: FAIL — see errors above.");
        std::process::exit(1);
    }
}
