//! Semantic Search Cache Benchmark
//!
//! Measures cache hit rate, mean latency, p50/p95 latency, throughput, and
//! recall quality for three variants:
//!   1. NoCache      — brute-force corpus scan every query (baseline)
//!   2. CacheCoarse  — flat cosine cache, threshold=0.90
//!   3. CacheFine    — flat cosine cache, threshold=0.97
//!
//! Dataset: deterministic, no external files required.

use ruvector_semantic_cache::{
    cache::{coarse, fine, overlap_recall, FlatSemanticCache, NoCache},
    corpus::{generate_prototypes, generate_workload_mixed, FlatCorpus},
    CacheStats, SemanticCacheLayer,
};

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ─── Generic benchmark driver ─────────────────────────────────────────────────

struct BenchResult {
    name: String,
    stats: CacheStats,
    all_latencies_ns: Vec<u64>,
    elapsed_ns: u64,
    mem_bytes: usize,
}

fn run_variant<C: SemanticCacheLayer>(
    mut cache: C,
    corpus: &FlatCorpus,
    queries: &[Vec<f32>],
    k: usize,
    // Fraction of queries used to warm the cache before the measured window.
    warmup_frac: f64,
    // Second corpus scan for hit-recall validation (only used during miss path).
    validate_recall: bool,
) -> BenchResult {
    let name = cache.name().to_string();
    let warmup_n = (queries.len() as f64 * warmup_frac) as usize;
    let mut stats = CacheStats::default();
    let mut all_latencies: Vec<u64> = Vec::with_capacity(queries.len() - warmup_n);

    // ── Warmup: populate cache but don't record stats ──
    for q in &queries[..warmup_n] {
        if cache.lookup(q).is_none() {
            let res = corpus.search(q, k);
            cache.insert(q.clone(), res);
        }
    }
    cache.invalidate_all(); // Reset stats; keep no stale entries from warmup.
                            // Re-warm from scratch so the benchmark window starts with a cold cache.
    for q in &queries[..warmup_n] {
        if cache.lookup(q).is_none() {
            let res = corpus.search(q, k);
            cache.insert(q.clone(), res);
        }
    }

    let bench_start = now_ns();

    for q in &queries[warmup_n..] {
        let t0 = now_ns();

        let found = cache.lookup(q);
        if let Some(cached_results) = found {
            let t1 = now_ns();
            let lat = t1 - t0;
            stats.hits += 1;
            stats.hit_latency_ns_sum += lat;
            all_latencies.push(lat);

            if validate_recall {
                // Run a fresh corpus scan to measure result quality.
                let fresh = corpus.search(q, k);
                let rec = overlap_recall(&cached_results, &fresh);
                stats.hit_recall_sum += rec;
                stats.hit_recall_count += 1;
            }
        } else {
            let fresh = corpus.search(q, k);
            let t1 = now_ns();
            let lat = t1 - t0;
            stats.misses += 1;
            stats.miss_latency_ns_sum += lat;
            all_latencies.push(lat);
            cache.insert(q.clone(), fresh);
        }

        stats.queries += 1;
    }

    let elapsed_ns = now_ns() - bench_start;
    all_latencies.sort_unstable();

    // Rough memory: cache internal storage (for NoCache this is 0).
    let mem_bytes = match name.as_str() {
        "NoCache" => 0,
        _ => corpus.memory_bytes(), // corpus is always in memory; cache adds query vectors
    };

    BenchResult {
        name,
        stats,
        all_latencies_ns: all_latencies,
        elapsed_ns,
        mem_bytes,
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // ── Parameters ──
    let corpus_n = 50_000usize;
    let dim = 128usize;
    let n_prototypes = 200usize; // unique "topics" in the workload
    let n_queries = 3_000usize; // total queries (warmup + bench window)
    let warmup_frac = 0.20; // 20 % used for cache warm-up
    let k = 10usize; // top-k results
    let cache_capacity = 500usize; // max entries in the semantic cache

    // Mixed workload tiers (see corpus::generate_workload_mixed for derivation).
    // With dim=128:
    //   near_sigma=0.02 → cosine(q1,q2) ≈ 0.95 (hits coarse t=0.90, misses fine t=0.97)
    //   far_sigma =0.10 → cosine(q1,q2) ≈ 0.44 (misses both thresholds)
    let exact_frac = 0.35_f32; // fraction of exact prototype repeats (cosine = 1.0)
    let near_frac = 0.40_f32; // near-duplicates (near_sigma)
    let near_sigma = 0.02_f32;
    let far_sigma = 0.10_f32;

    // ── Environment info ──
    eprintln!("=== Semantic Search Cache Benchmark ===");
    eprintln!("OS:           {}", std::env::consts::OS);
    eprintln!("ARCH:         {}", std::env::consts::ARCH);
    eprintln!("Corpus:       {corpus_n} × {dim}-dim f32 vectors");
    eprintln!(
        "Queries:      {n_queries} (warmup={} + bench={})",
        (n_queries as f64 * warmup_frac) as usize,
        n_queries - (n_queries as f64 * warmup_frac) as usize
    );
    eprintln!("Prototypes:   {n_prototypes}");
    eprintln!("Workload:     {:.0}% exact | {:.0}% near (σ={near_sigma}) | {:.0}% diverse (σ={far_sigma})",
        exact_frac * 100.0, near_frac * 100.0, (1.0 - exact_frac - near_frac) * 100.0);
    eprintln!("k:            {k}");
    eprintln!("Cache cap:    {cache_capacity}");
    eprintln!();

    // ── Build dataset ──
    eprintln!("Building corpus ({corpus_n}×{dim})...");
    let corpus = FlatCorpus::generate(corpus_n, dim, 12345);
    eprintln!(
        "  corpus memory: {:.1} MB",
        corpus.memory_bytes() as f64 / 1e6
    );

    eprintln!("Generating {n_prototypes} query prototypes...");
    let prototypes = generate_prototypes(n_prototypes, dim, 99);

    eprintln!("Generating {n_queries} mixed workload queries...");
    let queries = generate_workload_mixed(
        &prototypes,
        n_queries,
        exact_frac,
        near_frac,
        near_sigma,
        far_sigma,
        777,
    );

    eprintln!();

    // ── Run variants ──
    eprintln!("Running NoCache (baseline)...");
    let r_nocache = run_variant(NoCache, &corpus, &queries, k, warmup_frac, false);

    eprintln!("Running SemanticCacheCoarse (threshold=0.90)...");
    let r_coarse = run_variant_flat(
        coarse(cache_capacity),
        &corpus,
        &queries,
        k,
        warmup_frac,
        true,
    );

    eprintln!("Running SemanticCacheFine (threshold=0.97)...");
    let r_fine = run_variant_flat(
        fine(cache_capacity),
        &corpus,
        &queries,
        k,
        warmup_frac,
        true,
    );

    // ── Print results ──
    print_results(&r_nocache);
    print_results(&r_coarse);
    print_results(&r_fine);

    // ── Summary table ──
    println!();
    println!("=== Summary ===");
    println!(
        "{:<32} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Variant", "HitRate%", "MeanLatµs", "p50µs", "p95µs", "QPS", "MemMB", "HitRecall"
    );
    println!("{}", "-".repeat(100));

    for r in [&r_nocache, &r_coarse, &r_fine] {
        let hit_rate = r.stats.hit_rate() * 100.0;
        let mean_lat = r.stats.mean_overall_latency_us();
        let p50 = percentile(&r.all_latencies_ns, 50.0) as f64 / 1_000.0;
        let p95 = percentile(&r.all_latencies_ns, 95.0) as f64 / 1_000.0;
        let qps = r.stats.throughput_qps(r.elapsed_ns);
        let mem_mb =
            (r.mem_bytes as f64 + r.stats.hits as f64 * (dim as f64 * 4.0 + k as f64 * 8.0)) / 1e6;
        let recall = r.stats.mean_hit_recall();
        println!(
            "{:<32} {:>8.1} {:>10.1} {:>8.1} {:>8.1} {:>8.0} {:>8.1} {:>10.3}",
            r.name, hit_rate, mean_lat, p50, p95, qps, mem_mb, recall
        );
    }

    // ── Acceptance test ──
    println!();
    println!("=== Acceptance Tests ===");
    let corpus_mean_us = r_nocache.stats.mean_overall_latency_us();

    // 1. Coarse cache must have > 40 % hit rate given the query workload.
    let coarse_hit_rate = r_coarse.stats.hit_rate();
    let t1_pass = coarse_hit_rate > 0.40;
    println!(
        "[{}] Coarse hit rate > 40 %: {:.1} %",
        if t1_pass { "PASS" } else { "FAIL" },
        coarse_hit_rate * 100.0
    );

    // 2. Fine cache must have > 20 % hit rate.
    let fine_hit_rate = r_fine.stats.hit_rate();
    let t2_pass = fine_hit_rate > 0.20;
    println!(
        "[{}] Fine hit rate > 20 %: {:.1} %",
        if t2_pass { "PASS" } else { "FAIL" },
        fine_hit_rate * 100.0
    );

    // 3. Coarse cache mean latency must be < 60 % of NoCache mean latency.
    let coarse_mean = r_coarse.stats.mean_overall_latency_us();
    let t3_pass = coarse_mean < corpus_mean_us * 0.60;
    println!(
        "[{}] Coarse mean latency < 60 % of NoCache: {:.1} µs vs {:.1} µs",
        if t3_pass { "PASS" } else { "FAIL" },
        coarse_mean,
        corpus_mean_us
    );

    // 4. Fine cache hit recall must be ≥ 0.85 (cached results match fresh results).
    let fine_recall = r_fine.stats.mean_hit_recall();
    let t4_pass = fine_recall >= 0.85;
    println!(
        "[{}] Fine cache hit recall ≥ 0.85: {:.3}",
        if t4_pass { "PASS" } else { "FAIL" },
        fine_recall
    );

    // 5. Coarse cache hit recall must be ≥ 0.75.
    let coarse_recall = r_coarse.stats.mean_hit_recall();
    let t5_pass = coarse_recall >= 0.75;
    println!(
        "[{}] Coarse cache hit recall ≥ 0.75: {:.3}",
        if t5_pass { "PASS" } else { "FAIL" },
        coarse_recall
    );

    println!();
    let all_pass = t1_pass && t2_pass && t3_pass && t4_pass && t5_pass;
    if all_pass {
        println!("ACCEPTANCE: PASS — all 5 tests passed.");
    } else {
        println!("ACCEPTANCE: FAIL — one or more tests failed.");
        std::process::exit(1);
    }
}

// ─── FlatSemanticCache-specific runner (needs access to hit/miss counters) ────

fn run_variant_flat(
    mut cache: FlatSemanticCache,
    corpus: &FlatCorpus,
    queries: &[Vec<f32>],
    k: usize,
    warmup_frac: f64,
    validate_recall: bool,
) -> BenchResult {
    let name = cache.name().to_string();
    let warmup_n = (queries.len() as f64 * warmup_frac) as usize;
    let mut stats = CacheStats::default();
    let mut all_latencies: Vec<u64> = Vec::with_capacity(queries.len() - warmup_n);

    // Warm up.
    for q in &queries[..warmup_n] {
        if cache.lookup(q).is_none() {
            let res = corpus.search(q, k);
            cache.insert(q.clone(), res);
        }
    }
    // Reset counters; keep warm entries.
    cache.hits = 0;
    cache.misses = 0;

    let bench_start = now_ns();

    for q in &queries[warmup_n..] {
        let t0 = now_ns();
        let found = cache.lookup(q);
        if let Some(cached_results) = found {
            let t1 = now_ns();
            stats.hit_latency_ns_sum += t1 - t0;
            all_latencies.push(t1 - t0);
            stats.hits += 1;

            if validate_recall {
                let fresh = corpus.search(q, k);
                stats.hit_recall_sum += overlap_recall(&cached_results, &fresh);
                stats.hit_recall_count += 1;
            }
        } else {
            let fresh = corpus.search(q, k);
            let t1 = now_ns();
            stats.miss_latency_ns_sum += t1 - t0;
            all_latencies.push(t1 - t0);
            stats.misses += 1;
            cache.insert(q.clone(), fresh);
        }
        stats.queries += 1;
    }

    let elapsed_ns = now_ns() - bench_start;
    all_latencies.sort_unstable();

    let cache_mem = cache.memory_bytes();
    let corpus_mem = corpus.memory_bytes();

    BenchResult {
        name,
        stats,
        all_latencies_ns: all_latencies,
        elapsed_ns,
        mem_bytes: corpus_mem + cache_mem,
    }
}

fn print_results(r: &BenchResult) {
    println!("--- {} ---", r.name);
    println!("  Queries (bench window): {}", r.stats.queries);
    println!(
        "  Cache hits:             {} ({:.1}%)",
        r.stats.hits,
        r.stats.hit_rate() * 100.0
    );
    println!("  Cache misses:           {}", r.stats.misses);
    println!(
        "  Mean latency (overall): {:.1} µs",
        r.stats.mean_overall_latency_us()
    );
    if r.stats.hits > 0 {
        println!(
            "  Mean latency (hit):     {:.1} µs",
            r.stats.mean_hit_latency_us()
        );
    }
    if r.stats.misses > 0 {
        println!(
            "  Mean latency (miss):    {:.1} µs",
            r.stats.mean_miss_latency_us()
        );
    }
    let p50 = percentile(&r.all_latencies_ns, 50.0) as f64 / 1_000.0;
    let p95 = percentile(&r.all_latencies_ns, 95.0) as f64 / 1_000.0;
    println!("  p50 latency:            {:.1} µs", p50);
    println!("  p95 latency:            {:.1} µs", p95);
    println!(
        "  Throughput:             {:.0} QPS",
        r.stats.throughput_qps(r.elapsed_ns)
    );
    if r.stats.hit_recall_count > 0 {
        println!("  Hit recall@k:           {:.3}", r.stats.mean_hit_recall());
    }
    println!();
}

#[inline]
fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
