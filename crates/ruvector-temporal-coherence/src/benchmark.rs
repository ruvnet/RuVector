//! Benchmark binary: temporal coherence decay — three variants.
//!
//! Reports mean/p50/p95 latency, throughput, memory estimate, and variant-
//! specific quality metrics:
//!   FlatSearch      → cosine recall@K vs cosine ground truth
//!   TemporalSearch  → mean recency score of retrieved memories (want high)
//!   CoherenceSearch → mean coherence gate of retrieved memories (want high)
//!
//! Lower cosine recall for temporal/coherence variants is *expected and correct*:
//! they intentionally trade some cosine similarity for recency or coherence.
//!
//! Usage:
//!   cargo run --release -p ruvector-temporal-coherence --bin tcd-benchmark
//!   cargo run --release -p ruvector-temporal-coherence --bin tcd-benchmark -- --n 5000 --dims 128

use rand::SeedableRng;
use ruvector_temporal_coherence::{
    estimate_memory_bytes, generate_memory_corpus, ground_truth_topk, recall_at_k, CoherenceGraph,
    CoherenceSearch, DecayConfig, FlatSearch, MemoryStore, TemporalSearch, VectorSearch,
};
use std::time::{Duration, Instant};

const DEFAULT_N: usize = 5_000;
const DEFAULT_DIMS: usize = 128;
const DEFAULT_QUERIES: usize = 200;
const DEFAULT_K: usize = 10;
const COHERENCE_THRESHOLD: f32 = 0.55;
const COHERENCE_WEIGHT: f32 = 0.30;
const HALF_LIFE_FRAC: f64 = 0.30; // 30 % of time_span
const TIME_SPAN: u64 = 1_000_000;
const NUM_CLUSTERS: usize = 20;
// Acceptance thresholds
const MIN_FLAT_RECALL: f32 = 0.95;
// Temporal/coherence are scored by their OWN fitness metric (recency/coherence),
// not by cosine recall.  Thresholds are in [0,1].
const MIN_TEMPORAL_RECENCY: f32 = 0.55; // retrieved memories must be in top 55% by time
const MIN_COHERENCE_GATE: f32 = 0.50; // retrieved memories must have coherence gate >= 0.50 mean
const MAX_MEAN_LATENCY_US: u128 = 500_000; // 500 ms per query (conservative for n=5k O(n²) build)

fn percentile(mut data: Vec<Duration>, p: f64) -> Duration {
    data.sort();
    let idx = ((p / 100.0) * data.len() as f64).floor() as usize;
    data[idx.min(data.len().saturating_sub(1))]
}

/// Mean normalised timestamp [0,1] of retrieved memories — measures recency.
fn mean_recency(ids: &[u64], store: &MemoryStore) -> f32 {
    if ids.is_empty() {
        return 0.0;
    }
    let sum: f64 = ids
        .iter()
        .filter_map(|&id| store.get(id))
        .map(|r| r.metadata.timestamp as f64 / TIME_SPAN as f64)
        .sum();
    (sum / ids.len() as f64) as f32
}

/// Mean coherence gate of retrieved memories — measures community relevance.
fn mean_coherence_gate(ids: &[u64], graph: &CoherenceGraph) -> f32 {
    if ids.is_empty() {
        return 0.0;
    }
    let sum: f32 = ids.iter().map(|&id| graph.gate(id)).sum();
    sum / ids.len() as f32
}

fn print_hw_info() {
    println!("--- Hardware / Runtime ---");
    println!("  OS      : {}", std::env::consts::OS);
    println!("  Arch    : {}", std::env::consts::ARCH);
    println!(
        "  rustc   : {}",
        option_env!("CARGO_BUILD_RUSTC_VERSION").unwrap_or("(see rustc --version)")
    );
    println!();
}

fn parse_args() -> (usize, usize, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut n = DEFAULT_N;
    let mut dims = DEFAULT_DIMS;
    let mut queries = DEFAULT_QUERIES;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => {
                n = args[i + 1].parse().unwrap_or(n);
                i += 2;
            }
            "--dims" => {
                dims = args[i + 1].parse().unwrap_or(dims);
                i += 2;
            }
            "--queries" => {
                queries = args[i + 1].parse().unwrap_or(queries);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    (n, dims, queries)
}

struct VariantStats {
    name: &'static str,
    latencies: Vec<Duration>,
    /// cosine recall vs flat gt
    cosine_recalls: Vec<f32>,
    /// variant-specific quality (recency or coherence gate)
    quality: Vec<f32>,
    quality_label: &'static str,
    memory_bytes: usize,
}

impl VariantStats {
    fn new(name: &'static str, quality_label: &'static str, memory_bytes: usize) -> Self {
        Self {
            name,
            latencies: Vec::new(),
            cosine_recalls: Vec::new(),
            quality: Vec::new(),
            quality_label,
            memory_bytes,
        }
    }

    fn add(&mut self, lat: Duration, recall: f32, quality: f32) {
        self.latencies.push(lat);
        self.cosine_recalls.push(recall);
        self.quality.push(quality);
    }

    fn print(&self) {
        let mean_lat = self.latencies.iter().sum::<Duration>() / self.latencies.len().max(1) as u32;
        let p50 = percentile(self.latencies.clone(), 50.0);
        let p95 = percentile(self.latencies.clone(), 95.0);
        let total_secs = self.latencies.iter().sum::<Duration>().as_secs_f64();
        let throughput = self.latencies.len() as f64 / total_secs.max(1e-9);
        let mean_recall: f32 =
            self.cosine_recalls.iter().sum::<f32>() / self.cosine_recalls.len().max(1) as f32;
        let mean_quality: f32 = self.quality.iter().sum::<f32>() / self.quality.len().max(1) as f32;
        let mem_kb = self.memory_bytes / 1024;

        println!(
            "  {:<20} mean={:>7}µs  p50={:>7}µs  p95={:>7}µs  tput={:>7.1}q/s  mem={:>5}KB  recall@K={:.3}  {}={:.3}",
            self.name,
            mean_lat.as_micros(),
            p50.as_micros(),
            p95.as_micros(),
            throughput,
            mem_kb,
            mean_recall,
            self.quality_label,
            mean_quality,
        );
    }

    fn mean_latency_us(&self) -> u128 {
        (self.latencies.iter().sum::<Duration>() / self.latencies.len().max(1) as u32).as_micros()
    }

    fn mean_cosine_recall(&self) -> f32 {
        self.cosine_recalls.iter().sum::<f32>() / self.cosine_recalls.len().max(1) as f32
    }

    fn mean_quality(&self) -> f32 {
        self.quality.iter().sum::<f32>() / self.quality.len().max(1) as f32
    }
}

fn main() {
    print_hw_info();

    let (n, dims, num_queries) = parse_args();
    let half_life = (TIME_SPAN as f64 * HALF_LIFE_FRAC) as u64;

    println!("--- Dataset ---");
    println!("  N={n}  dims={dims}  queries={num_queries}  K={DEFAULT_K}");
    println!("  clusters={NUM_CLUSTERS}  time_span={TIME_SPAN}  half_life={half_life}");
    println!("  coherence_threshold={COHERENCE_THRESHOLD}  coherence_weight={COHERENCE_WEIGHT}");
    println!();

    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xDEAD_BEEF);

    println!("Building corpus ({n} × {dims}D)…");
    let t0 = Instant::now();
    let store = generate_memory_corpus(n, dims, TIME_SPAN, NUM_CLUSTERS, &mut rng);
    println!(
        "  corpus built in {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    println!("Building coherence graph (threshold={COHERENCE_THRESHOLD})…");
    let tg = Instant::now();
    let graph = CoherenceGraph::build(&store, COHERENCE_THRESHOLD);
    println!(
        "  graph built in {:.1}ms  nodes={}  edges={}  mean_gate={:.3}",
        tg.elapsed().as_secs_f64() * 1000.0,
        graph.node_count(),
        graph.edge_count(),
        graph.mean_gate(),
    );
    println!();

    let now = TIME_SPAN;
    let decay = DecayConfig::exponential(now, half_life);
    let flat = FlatSearch;
    let temporal = TemporalSearch {
        decay: decay.clone(),
    };
    let coherence_search = CoherenceSearch::new(
        decay.clone(),
        CoherenceGraph::build(&store, COHERENCE_THRESHOLD),
        COHERENCE_WEIGHT,
    );

    let mem_vec = estimate_memory_bytes(n, dims);

    let mut stat_flat = VariantStats::new("FlatSearch", "cosine_recall", mem_vec);
    let mut stat_temp = VariantStats::new("TemporalSearch", "recency", mem_vec);
    let mut stat_coh = VariantStats::new("CoherenceSearch", "coh_gate", mem_vec + n * 4);

    use rand::distributions::{Distribution, Uniform};
    let uni = Uniform::new(-1.0f32, 1.0);

    println!("Running {num_queries} queries…");
    for _ in 0..num_queries {
        let query: Vec<f32> = (0..dims).map(|_| uni.sample(&mut rng)).collect();
        let gt = ground_truth_topk(&query, &store, DEFAULT_K);

        // FlatSearch — quality = cosine recall (should be ~1.0)
        let t = Instant::now();
        let r_flat = flat.search(&query, DEFAULT_K, &store);
        let lat = t.elapsed();
        let ids_flat: Vec<u64> = r_flat.iter().map(|x| x.id).collect();
        let rc = recall_at_k(&ids_flat, &gt);
        stat_flat.add(lat, rc, rc);

        // TemporalSearch — quality = mean recency of retrieved memories
        let t = Instant::now();
        let r_temp = temporal.search(&query, DEFAULT_K, &store);
        let lat = t.elapsed();
        let ids_temp: Vec<u64> = r_temp.iter().map(|x| x.id).collect();
        let rc_t = recall_at_k(&ids_temp, &gt);
        let recency = mean_recency(&ids_temp, &store);
        stat_temp.add(lat, rc_t, recency);

        // CoherenceSearch — quality = mean coherence gate of retrieved memories
        let t = Instant::now();
        let r_coh = coherence_search.search(&query, DEFAULT_K, &store);
        let lat = t.elapsed();
        let ids_coh: Vec<u64> = r_coh.iter().map(|x| x.id).collect();
        let rc_c = recall_at_k(&ids_coh, &gt);
        let coh_gate = mean_coherence_gate(&ids_coh, &graph);
        stat_coh.add(lat, rc_c, coh_gate);
    }

    println!();
    println!("--- Results ---");
    println!(
        "  {:<20} {:>10}  {:>10}  {:>10}  {:>12}  {:>8}  {:>12}  quality",
        "Variant", "mean_lat", "p50_lat", "p95_lat", "throughput", "mem", "recall@K"
    );
    stat_flat.print();
    stat_temp.print();
    stat_coh.print();

    println!();
    println!("--- Quality metric explanation ---");
    println!("  FlatSearch.cosine_recall  = overlap with cosine-only ground truth (expect ~1.0)");
    println!("  TemporalSearch.recency    = mean normalised timestamp of retrieved results [0,1]");
    println!("                             (1.0 = always retrieves newest memories)");
    println!("  CoherenceSearch.coh_gate  = mean graph-coherence gate of retrieved results [0,1]");
    println!("                             (1.0 = always retrieves most graph-connected memories)");
    println!();
    println!("  Temporal/coherence cosine_recall vs flat is expected to be < 1.0 —");
    println!("  the variants deliberately trade cosine similarity for recency/coherence.");
    println!();

    // Acceptance tests — each variant is tested on its PRIMARY fitness metric
    println!("--- Acceptance ---");
    let flat_ok = stat_flat.mean_cosine_recall() >= MIN_FLAT_RECALL;
    let temp_ok = stat_temp.mean_quality() >= MIN_TEMPORAL_RECENCY;
    let coh_ok = stat_coh.mean_quality() >= MIN_COHERENCE_GATE;
    let lat_ok = stat_flat.mean_latency_us() <= MAX_MEAN_LATENCY_US;

    println!(
        "  FlatSearch cosine_recall >= {MIN_FLAT_RECALL}       : {} ({:.3})",
        if flat_ok { "PASS" } else { "FAIL" },
        stat_flat.mean_cosine_recall()
    );
    println!(
        "  TemporalSearch recency >= {MIN_TEMPORAL_RECENCY}         : {} ({:.3})",
        if temp_ok { "PASS" } else { "FAIL" },
        stat_temp.mean_quality()
    );
    println!(
        "  CoherenceSearch coh_gate >= {MIN_COHERENCE_GATE}       : {} ({:.3})",
        if coh_ok { "PASS" } else { "FAIL" },
        stat_coh.mean_quality()
    );
    println!(
        "  FlatSearch mean_lat <= {MAX_MEAN_LATENCY_US}µs        : {} ({}µs)",
        if lat_ok { "PASS" } else { "FAIL" },
        stat_flat.mean_latency_us()
    );

    let all_ok = flat_ok && temp_ok && coh_ok && lat_ok;
    println!();
    if all_ok {
        println!("✓ All acceptance tests PASSED.");
        std::process::exit(0);
    } else {
        println!("✗ One or more acceptance tests FAILED.");
        std::process::exit(1);
    }
}
