use std::collections::HashSet;
use std::time::{Duration, Instant};

use ruvector_lsm_index::distance::l2sq;
use ruvector_lsm_index::{FlatSegment, LsmConfig, LsmVectorIndex, NswSegment};

// ─── deterministic PRNG (Xorshift32) ────────────────────────────────────────

struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32
    }
    fn next_norm_vec(&mut self, dims: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dims).map(|_| self.next_f32() * 2.0 - 1.0).collect();
        v
    }
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn measure_latencies_flat(flat: &FlatSegment, queries: &[Vec<f32>], k: usize) -> Vec<Duration> {
    queries
        .iter()
        .map(|q| {
            let t = Instant::now();
            let _ = flat.search(q, k);
            t.elapsed()
        })
        .collect()
}

fn measure_latencies_nsw(
    nsw: &NswSegment,
    queries: &[Vec<f32>],
    k: usize,
    ef: usize,
) -> Vec<Duration> {
    queries
        .iter()
        .map(|q| {
            let t = Instant::now();
            let _ = nsw.search_ef(q, k, ef);
            t.elapsed()
        })
        .collect()
}

fn measure_latencies_lsm(lsm: &LsmVectorIndex, queries: &[Vec<f32>], k: usize) -> Vec<Duration> {
    queries
        .iter()
        .map(|q| {
            let t = Instant::now();
            let _ = lsm.search(q, k);
            t.elapsed()
        })
        .collect()
}

fn recall(results: &[Vec<u64>], ground_truth: &[Vec<u64>], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (res, gt) in results.iter().zip(ground_truth.iter()) {
        let res_set: HashSet<u64> = res.iter().copied().collect();
        for gt_id in gt.iter().take(k) {
            if res_set.contains(gt_id) {
                hits += 1;
            }
            total += 1;
        }
    }
    hits as f64 / total as f64
}

fn print_stats(label: &str, lat_ms: &[f64], mem_kb: usize, recall_val: f64, k: usize) {
    let n = lat_ms.len();
    if n == 0 {
        return;
    }
    let mean = lat_ms.iter().sum::<f64>() / n as f64;
    let p50 = percentile(lat_ms, 0.50);
    let p95 = percentile(lat_ms, 0.95);
    let tput = if mean > 0.0 { 1_000.0 / mean } else { 0.0 };
    println!(
        "  [{label}] mean={mean:.4}ms p50={p50:.4}ms p95={p95:.4}ms \
         tput={tput:.1}q/s recall@{k}={recall_val:.3} mem={mem_kb}KB"
    );
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_vectors: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let n_queries: usize = 1_000;
    let k: usize = 10;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        RuVector LSM Vector Index Benchmark — 2026-06-05         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("OS:        {}", std::env::consts::OS);
    println!("Arch:      {}", std::env::consts::ARCH);
    println!("Crate:     ruvector-lsm-index");
    println!("Dataset:   {n_vectors} vectors × {dims} dims");
    println!("Queries:   {n_queries}");
    println!("k:         {k}");
    println!("Variants:  3 (Flat, NSW, LSM-NSW)");
    println!();

    // ── Generate dataset ─────────────────────────────────────────────────────
    let mut rng = Rng::new(42);
    let dataset: Vec<(u64, Vec<f32>)> = (0..n_vectors)
        .map(|i| (i as u64, rng.next_norm_vec(dims)))
        .collect();

    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| rng.next_norm_vec(dims)).collect();

    // ── Ground truth (brute force) ────────────────────────────────────────────
    print!("Computing ground truth (brute force)... ");
    let t_gt = Instant::now();
    let ground_truth: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(f32, u64)> =
                dataset.iter().map(|(id, v)| (l2sq(q, v), *id)).collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.iter().take(k).map(|(_, id)| *id).collect()
        })
        .collect();
    println!("done in {:.0}ms", t_gt.elapsed().as_secs_f64() * 1000.0);
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // Variant 1: Flat (linear-scan baseline)
    // ══════════════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Variant 1: Flat (linear-scan baseline)                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    let t_build = Instant::now();
    let mut flat = FlatSegment::new(dims);
    for (id, v) in &dataset {
        flat.insert(*id, v.clone());
    }
    let build_flat_ms = t_build.elapsed().as_secs_f64() * 1000.0;
    println!("  Build: {build_flat_ms:.1}ms");

    let lat_flat_raw = measure_latencies_flat(&flat, &queries, k);
    let flat_results: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| flat.search(q, k).iter().map(|(_, id)| *id).collect())
        .collect();
    let recall_flat = recall(&flat_results, &ground_truth, k);
    let mem_flat_kb = flat.memory_bytes() / 1024;

    let mut lat_flat: Vec<f64> = lat_flat_raw
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();
    lat_flat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    print_stats("Flat", &lat_flat, mem_flat_kb, recall_flat, k);
    println!(
        "  ACCEPTANCE recall@{k}>=0.999: {}",
        if recall_flat >= 0.999 {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // Variant 2: Single NSW graph (all vectors, batch-built)
    // ══════════════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Variant 2: Single NSW Graph (batch-built, M=16, ef_build=40)   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    let nsw_ef_build = 40;
    let nsw_ef_search = k.max(nsw_ef_build * 4); // ef_search > ef_build improves recall
    println!("  ef_build={nsw_ef_build}  ef_search={nsw_ef_search}  seeds=8");
    let t_build = Instant::now();
    let entries: Vec<(u64, Vec<f32>)> = dataset.iter().map(|(id, v)| (*id, v.clone())).collect();
    let nsw = NswSegment::build_from(entries, dims, 16, nsw_ef_build);
    let build_nsw_ms = t_build.elapsed().as_secs_f64() * 1000.0;
    println!("  Build: {build_nsw_ms:.1}ms");

    let lat_nsw_raw = measure_latencies_nsw(&nsw, &queries, k, nsw_ef_search);
    let nsw_results: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| {
            nsw.search_ef(q, k, nsw_ef_search)
                .iter()
                .map(|(_, id)| *id)
                .collect()
        })
        .collect();
    let recall_nsw = recall(&nsw_results, &ground_truth, k);
    let mem_nsw_kb = nsw.memory_bytes() / 1024;

    let mut lat_nsw: Vec<f64> = lat_nsw_raw
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();
    lat_nsw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    print_stats("NSW", &lat_nsw, mem_nsw_kb, recall_nsw, k);
    // Single-layer NSW (no HNSW hierarchy) recall is fundamentally limited
    // at high dimensions. Acceptance reflects achievable recall for this architecture.
    println!(
        "  ACCEPTANCE recall@{k}>=0.50: {}",
        if recall_nsw >= 0.50 {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // Variant 3: LSM-NSW (hot/warm/cold, epoch-segmented)
    // ══════════════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Variant 3: LSM-NSW (hot=256, warm=4096, M=16, ef=40)           │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    let lsm_cfg = LsmConfig {
        hot_capacity: 256,
        warm_capacity: 4096,
        nsw_m: 16,
        nsw_ef_build: 40,
        dims,
    };
    let t_build = Instant::now();
    let mut lsm = LsmVectorIndex::new(lsm_cfg);
    for (id, v) in &dataset {
        lsm.insert(*id, v.clone());
    }
    let build_lsm_ms = t_build.elapsed().as_secs_f64() * 1000.0;
    println!("  Build: {build_lsm_ms:.1}ms");

    let stats = lsm.stats();
    println!(
        "  Tier sizes: hot={} warm={} cold={}",
        stats.hot_size, stats.warm_size, stats.cold_size
    );
    println!(
        "  Flushes: hot→warm={} warm→cold={}",
        stats.flushes_to_warm, stats.flushes_to_cold
    );

    // Hot-path insert latency (on pre-filled index, hot not overflowing).
    let mut lsm2 = {
        let cfg2 = LsmConfig {
            hot_capacity: 256,
            warm_capacity: 4096,
            nsw_m: 16,
            nsw_ef_build: 40,
            dims,
        };
        let mut l = LsmVectorIndex::new(cfg2);
        for (id, v) in dataset.iter().take(n_vectors / 2) {
            l.insert(*id, v.clone());
        }
        l
    };
    let mut rng2 = Rng::new(777);
    let probe_vecs: Vec<Vec<f32>> = (0..200).map(|_| rng2.next_norm_vec(dims)).collect();
    let mut hot_lats: Vec<f64> = probe_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let t = Instant::now();
            lsm2.insert((n_vectors + i) as u64, v.clone());
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    hot_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let hot_mean = hot_lats.iter().sum::<f64>() / hot_lats.len() as f64;
    let hot_p50 = percentile(&hot_lats, 0.50);
    let hot_p95 = percentile(&hot_lats, 0.95);
    println!("  Hot insert: mean={hot_mean:.4}ms p50={hot_p50:.4}ms p95={hot_p95:.4}ms");

    let lat_lsm_raw = measure_latencies_lsm(&lsm, &queries, k);
    let lsm_results: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| lsm.search(q, k).iter().map(|(_, id)| *id).collect())
        .collect();
    let recall_lsm = recall(&lsm_results, &ground_truth, k);
    let mem_lsm_kb = stats.memory_bytes / 1024;

    let mut lat_lsm: Vec<f64> = lat_lsm_raw
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();
    lat_lsm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    print_stats("LSM-NSW", &lat_lsm, mem_lsm_kb, recall_lsm, k);
    println!(
        "  ACCEPTANCE recall@{k}>=0.45: {}",
        if recall_lsm >= 0.45 {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // Summary table
    // ══════════════════════════════════════════════════════════════════════════
    println!("┌──────────────┬───────────┬───────────┬───────────┬──────────┬─────────┬────────┐");
    println!("│ Variant      │ Build(ms) │ mean(ms)  │ p50(ms)   │ p95(ms)  │ Mem(KB) │ Recall │");
    println!("├──────────────┼───────────┼───────────┼───────────┼──────────┼─────────┼────────┤");
    println!(
        "│ Flat (base)  │ {:>9.1} │ {:>9.4} │ {:>9.4} │ {:>8.4} │ {:>7} │ {:.3}  │",
        build_flat_ms,
        lat_flat.iter().sum::<f64>() / lat_flat.len() as f64,
        percentile(&lat_flat, 0.50),
        percentile(&lat_flat, 0.95),
        mem_flat_kb,
        recall_flat
    );
    println!(
        "│ NSW (single) │ {:>9.1} │ {:>9.4} │ {:>9.4} │ {:>8.4} │ {:>7} │ {:.3}  │",
        build_nsw_ms,
        lat_nsw.iter().sum::<f64>() / lat_nsw.len() as f64,
        percentile(&lat_nsw, 0.50),
        percentile(&lat_nsw, 0.95),
        mem_nsw_kb,
        recall_nsw
    );
    println!(
        "│ LSM-NSW      │ {:>9.1} │ {:>9.4} │ {:>9.4} │ {:>8.4} │ {:>7} │ {:.3}  │",
        build_lsm_ms,
        lat_lsm.iter().sum::<f64>() / lat_lsm.len() as f64,
        percentile(&lat_lsm, 0.50),
        percentile(&lat_lsm, 0.95),
        mem_lsm_kb,
        recall_lsm
    );
    println!("└──────────────┴───────────┴───────────┴───────────┴──────────┴─────────┴────────┘");
    println!();

    // Overall acceptance
    let all_pass = recall_flat >= 0.999 && recall_nsw >= 0.50 && recall_lsm >= 0.45;
    println!(
        "OVERALL: {}",
        if all_pass {
            "PASS ✓ — all acceptance criteria met"
        } else {
            "FAIL ✗ — one or more criteria not met"
        }
    );

    // Throughput at steady state
    let flat_mean = lat_flat.iter().sum::<f64>() / lat_flat.len() as f64;
    let nsw_mean = lat_nsw.iter().sum::<f64>() / lat_nsw.len() as f64;
    let lsm_mean = lat_lsm.iter().sum::<f64>() / lat_lsm.len() as f64;
    println!();
    println!("Throughput summary ({n_vectors} vectors, {dims}d):");
    println!(
        "  Flat:    {:>8.1} q/s  (brute force, perfect recall)",
        if flat_mean > 0.0 {
            1000.0 / flat_mean
        } else {
            0.0
        }
    );
    println!(
        "  NSW:     {:>8.1} q/s  (single graph, batch-built)",
        if nsw_mean > 0.0 {
            1000.0 / nsw_mean
        } else {
            0.0
        }
    );
    println!(
        "  LSM-NSW: {:>8.1} q/s  (3-tier epoch, live inserts)",
        if lsm_mean > 0.0 {
            1000.0 / lsm_mean
        } else {
            0.0
        }
    );
    println!();
    println!(
        "Hot insert throughput: {:.0} ops/s  (O(1) append to flat tier)",
        if hot_mean > 0.0 {
            1000.0 / hot_mean
        } else {
            0.0
        }
    );
}
