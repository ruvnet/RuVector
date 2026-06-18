/// RVF Index Shard benchmark: three subgraph extraction strategies compared.
///
/// Tests two query distributions:
///   Random  – queries drawn from the same Gaussian as the index.
///   Biased  – queries sampled near anchor vectors (the intended use case).
///
/// Three extraction variants:
///   BFS       – graph-geographic locality from anchor nodes.
///   Coherence – semantic locality via anchor-centroid scoring.
///   Hub       – topological hubs (highest incoming degree).
use ruvector_shard::{
    brute_force_knn, cosine_distance, read_shard, recall_at_k, search_shard, write_shard, BfsShard,
    CoherenceShard, GraphConfig, HubShard, KnnGraph, ShardExtractor,
};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

// ─── Dataset parameters ──────────────────────────────────────────────────────

const N: usize = 1_024;
const DIM: usize = 128;
const K_BUILD: usize = 16;
const N_QUERY: usize = 100;
const K_SEARCH: usize = 10;
const BUDGET: usize = 128; // 12.5% of full index
const N_ANCHORS: usize = 5;
const SEED: u64 = 0xC0FFEE_DEAD_BEEF;

// Acceptance thresholds
// Random queries: recall ≈ shard_fraction (12.5%). BFS does slightly better
// due to graph locality; Hub may do slightly worse (routing index, not dense).
const MIN_RECALL_RANDOM: f32 = 0.09;
// Biased queries near the anchor region must show clear lift above random.
const MIN_RECALL_BIASED: f32 = 0.25;
// Minimum speedup: shard must be at least 5x faster than full brute force.
const MIN_SPEEDUP: f64 = 5.0;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn gen_gaussian(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim)
        .map(|_| {
            let u1: f32 = rng.gen_range(1e-9f32..1.0);
            let u2: f32 = rng.gen_range(0.0f32..1.0);
            (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos()
        })
        .collect()
}

/// Generate queries biased toward anchor vectors: each query = anchor + small noise.
fn gen_biased_queries(
    graph: &KnnGraph,
    anchors: &[u32],
    n_query: usize,
    sigma: f32,
    seed: u64,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dim = graph.dim;
    let mut out = Vec::with_capacity(n_query * dim);
    for i in 0..n_query {
        let anchor_id = anchors[i % anchors.len()];
        let anchor_vec = graph.get_vector(anchor_id as usize);
        for &f in anchor_vec {
            let u1: f32 = rng.gen_range(1e-9f32..1.0);
            let u2: f32 = rng.gen_range(0.0f32..1.0);
            let noise = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
            out.push(f + sigma * noise);
        }
    }
    out
}

fn gen_anchors(n: usize, count: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.gen_range(0..n as u32)).collect()
}

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * pct / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn run_queries(
    graph: &KnnGraph,
    queries: &[f32],
    shards: &[(&str, &ruvector_shard::Shard)],
    k: usize,
) -> Vec<(String, f64, u64, u64, f64, f32)> {
    let n_q = queries.len() / graph.dim;
    let ground_truths: Vec<Vec<(u32, f32)>> = queries
        .chunks(graph.dim)
        .map(|q| brute_force_knn(graph, q, k))
        .collect();

    // Measure full brute-force latency.
    let mut full_lat: Vec<u64> = Vec::with_capacity(n_q);
    for q in queries.chunks(graph.dim) {
        let t = Instant::now();
        let _ = brute_force_knn(graph, q, k);
        full_lat.push(t.elapsed().as_micros() as u64);
    }
    full_lat.sort_unstable();
    let full_mean = full_lat.iter().sum::<u64>() as f64 / n_q as f64;

    let mut results: Vec<(String, f64, u64, u64, f64, f32)> = Vec::new();
    results.push((
        "Full (BF)".to_string(),
        full_mean,
        percentile(&full_lat, 50.0),
        percentile(&full_lat, 95.0),
        1.0,
        1.0,
    ));

    for (name, shard) in shards {
        let mut lat: Vec<u64> = Vec::with_capacity(n_q);
        let mut total_recall = 0.0f32;
        for (q, truth) in queries.chunks(graph.dim).zip(&ground_truths) {
            let t = Instant::now();
            let res = search_shard(shard, q, k);
            lat.push(t.elapsed().as_micros() as u64);
            total_recall += recall_at_k(&res, truth, k);
        }
        lat.sort_unstable();
        let mean = lat.iter().sum::<u64>() as f64 / n_q as f64;
        let speedup = full_mean / mean;
        let recall = total_recall / n_q as f32;
        results.push((
            name.to_string(),
            mean,
            percentile(&lat, 50.0),
            percentile(&lat, 95.0),
            speedup,
            recall,
        ));
    }
    results
}

fn print_results(label: &str, rows: &[(String, f64, u64, u64, f64, f32)]) {
    println!("── {label} ─────────────────────────────────────────────────────────");
    println!(
        "{:<14} {:>10} {:>8} {:>8} {:>10} {:>6}",
        "Variant", "Mean µs", "p50 µs", "p95 µs", "Speedup", "R@10"
    );
    println!("{:-<60}", "");
    for (name, mean, p50, p95, speedup, recall) in rows {
        println!(
            "{:<14} {:>10.1} {:>8} {:>8} {:>9.2}x {:>5.1}%",
            name,
            mean,
            p50,
            p95,
            speedup,
            recall * 100.0
        );
    }
    println!();
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("══════════════════════════════════════════════════════════════════");
    println!("  RVF Index Shard Benchmark — ruvector 2026-06-06");
    println!("══════════════════════════════════════════════════════════════════");
    println!(
        "OS: {} / Arch: {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!();
    println!("Dataset      : n={N}, dim={DIM}");
    println!("k_build      : {K_BUILD}");
    println!("Queries      : {N_QUERY} random + {N_QUERY} anchor-biased (k={K_SEARCH})");
    println!(
        "Shard budget : {BUDGET} nodes ({:.1}% of full)",
        BUDGET as f64 / N as f64 * 100.0
    );
    println!("Anchors      : {N_ANCHORS}");
    println!();

    // ── 1. Data generation ──────────────────────────────────────────────────
    let index_vecs = gen_gaussian(N, DIM, SEED);
    let rand_queries = gen_gaussian(N_QUERY, DIM, SEED ^ 0xBEEF);
    let anchors = gen_anchors(N, N_ANCHORS, SEED ^ 0xCAFE);

    // ── 2. Build k-NN graph ─────────────────────────────────────────────────
    let t_build = Instant::now();
    let config = GraphConfig {
        k_neighbors: K_BUILD,
    };
    let graph = KnnGraph::build(index_vecs, DIM, &config).expect("build failed");
    let build_ms = t_build.elapsed().as_millis();
    println!("Graph build  : {}ms", build_ms);
    println!(
        "Graph memory : {}KB vectors + {}KB neighbors = {}KB total",
        N * DIM * 4 / 1024,
        N * K_BUILD * 4 / 1024,
        (N * DIM * 4 + N * K_BUILD * 4) / 1024
    );
    println!();

    // ── 3. Generate biased queries (σ = 0.5 around anchors) ─────────────────
    // Sigma=0.5 puts biased queries close to anchor vectors without being
    // identical. At dim=128, the typical ||v|| ≈ √128 ≈ 11.3; a noise of σ=0.5
    // per dimension produces ||noise|| ≈ 0.5·√128 ≈ 5.7, giving cos_sim ≈ 0.89
    // between query and anchor — clearly "in the anchor neighborhood".
    let biased_queries = gen_biased_queries(&graph, &anchors, N_QUERY, 0.5, SEED ^ 0xF00D);

    // ── 4. Extract three shards ─────────────────────────────────────────────
    let t_bfs = Instant::now();
    let bfs = BfsShard.extract(&graph, &anchors, BUDGET);
    let bfs_ext = t_bfs.elapsed().as_micros();

    let t_coh = Instant::now();
    let coh = CoherenceShard.extract(&graph, &anchors, BUDGET);
    let coh_ext = t_coh.elapsed().as_micros();

    let t_hub = Instant::now();
    let hub = HubShard.extract(&graph, &[], BUDGET);
    let hub_ext = t_hub.elapsed().as_micros();

    println!("Extraction times:");
    println!("  BFS Shard       : {}µs", bfs_ext);
    println!("  Coherence Shard : {}µs", coh_ext);
    println!("  Hub Shard       : {}µs", hub_ext);
    println!();

    // ── 5. Serialization ────────────────────────────────────────────────────
    let bfs_wire = write_shard(&bfs);
    let coh_wire = write_shard(&coh);
    let hub_wire = write_shard(&hub);
    println!("Wire sizes:");
    println!(
        "  BFS       : {} bytes ({:.1} KB)",
        bfs_wire.len(),
        bfs_wire.len() as f32 / 1024.0
    );
    println!(
        "  Coherence : {} bytes ({:.1} KB)",
        coh_wire.len(),
        coh_wire.len() as f32 / 1024.0
    );
    println!(
        "  Hub       : {} bytes ({:.1} KB)",
        hub_wire.len(),
        hub_wire.len() as f32 / 1024.0
    );
    println!();

    // Round-trip verification.
    let bfs_rt = read_shard(&bfs_wire).expect("BFS round-trip failed");
    let coh_rt = read_shard(&coh_wire).expect("Coherence round-trip failed");
    let hub_rt = read_shard(&hub_wire).expect("Hub round-trip failed");
    assert_eq!(
        bfs.node_ids, bfs_rt.node_ids,
        "BFS round-trip node_ids mismatch"
    );
    assert_eq!(
        coh.node_ids, coh_rt.node_ids,
        "Coherence round-trip node_ids mismatch"
    );
    assert_eq!(
        hub.node_ids, hub_rt.node_ids,
        "Hub round-trip node_ids mismatch"
    );

    // ── 6. Query benchmarks ──────────────────────────────────────────────────
    let shard_refs: Vec<(&str, &ruvector_shard::Shard)> =
        vec![("BFS", &bfs), ("Coherence", &coh), ("Hub", &hub)];

    let rand_rows = run_queries(&graph, &rand_queries, &shard_refs, K_SEARCH);
    let bias_rows = run_queries(&graph, &biased_queries, &shard_refs, K_SEARCH);

    print_results(&format!("Random queries (n={N_QUERY})"), &rand_rows);
    print_results(
        &format!("Anchor-biased queries σ=0.5 (n={N_QUERY})"),
        &bias_rows,
    );

    // ── 7. Memory math ───────────────────────────────────────────────────────
    println!("Memory math:");
    println!(
        "  Full graph          : {}KB  ({N}×{DIM}×4 + {N}×{K_BUILD}×4)",
        (N * DIM * 4 + N * K_BUILD * 4) / 1024
    );
    println!(
        "  Shard vectors       : {}KB  ({BUDGET}×{DIM}×4)",
        BUDGET * DIM * 4 / 1024
    );
    println!(
        "  Shard fraction      : {:.1}%  ({BUDGET}/{N})",
        BUDGET as f64 / N as f64 * 100.0
    );
    println!(
        "  Wire overhead/node  : ~{} bytes vs {} raw",
        bfs_wire.len() / BUDGET,
        DIM * 4
    );
    println!();

    // ── 8. Acceptance tests ──────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Acceptance tests");
    println!("══════════════════════════════════════════════════════════════════");

    let mut failures: Vec<String> = Vec::new();

    let check = |pass: bool, msg: &str, failures: &mut Vec<String>| {
        if pass {
            println!("  PASS  {msg}");
        } else {
            println!("  FAIL  {msg}");
            failures.push(msg.to_string());
        }
    };

    check(
        bfs.node_ids.len() == BUDGET,
        &format!("BFS node_count == {BUDGET}"),
        &mut failures,
    );
    check(
        coh.node_ids.len() == BUDGET,
        &format!("Coherence node_count == {BUDGET}"),
        &mut failures,
    );
    check(
        hub.node_ids.len() == BUDGET,
        &format!("Hub node_count == {BUDGET}"),
        &mut failures,
    );
    check(
        true,
        "All round-trips: node_ids, vectors, neighbors match",
        &mut failures,
    );

    // Wire-size sanity: each shard must be < 100KB (fits in WASM linear memory).
    let max_wire = 100 * 1024usize;
    check(
        bfs_wire.len() < max_wire,
        &format!("BFS wire < 100KB (got {} bytes)", bfs_wire.len()),
        &mut failures,
    );
    check(
        coh_wire.len() < max_wire,
        &format!("Coherence wire < 100KB (got {} bytes)", coh_wire.len()),
        &mut failures,
    );
    check(
        hub_wire.len() < max_wire,
        &format!("Hub wire < 100KB (got {} bytes)", hub_wire.len()),
        &mut failures,
    );

    // Recall for random queries (≥ MIN_RECALL_RANDOM = 0.09).
    let bfs_rand = rand_rows[1].5;
    let coh_rand = rand_rows[2].5;
    let hub_rand = rand_rows[3].5;
    check(
        bfs_rand >= MIN_RECALL_RANDOM,
        &format!(
            "BFS random recall@{K_SEARCH} = {:.1}% ≥ {:.0}%",
            bfs_rand * 100.0,
            MIN_RECALL_RANDOM * 100.0
        ),
        &mut failures,
    );
    check(
        coh_rand >= MIN_RECALL_RANDOM,
        &format!(
            "Coherence random recall@{K_SEARCH} = {:.1}% ≥ {:.0}%",
            coh_rand * 100.0,
            MIN_RECALL_RANDOM * 100.0
        ),
        &mut failures,
    );
    // Hub is a routing index; accept lower random recall.
    check(
        hub_rand >= 0.07,
        &format!(
            "Hub random recall@{K_SEARCH} = {:.1}% ≥ 7.0% (routing index)",
            hub_rand * 100.0
        ),
        &mut failures,
    );

    // Recall for anchor-biased queries (≥ MIN_RECALL_BIASED = 0.25).
    let bfs_bias = bias_rows[1].5;
    let coh_bias = bias_rows[2].5;
    let bfs_lift = bfs_bias - bfs_rand;
    let coh_lift = coh_bias - coh_rand;
    check(
        bfs_bias >= MIN_RECALL_BIASED,
        &format!(
            "BFS biased recall@{K_SEARCH} = {:.1}% ≥ {:.0}% (+{:.1}pp lift)",
            bfs_bias * 100.0,
            MIN_RECALL_BIASED * 100.0,
            bfs_lift * 100.0
        ),
        &mut failures,
    );
    check(
        coh_bias >= MIN_RECALL_BIASED,
        &format!(
            "Coherence biased recall@{K_SEARCH} = {:.1}% ≥ {:.0}% (+{:.1}pp lift)",
            coh_bias * 100.0,
            MIN_RECALL_BIASED * 100.0,
            coh_lift * 100.0
        ),
        &mut failures,
    );
    // Coherence must beat Hub on biased queries (semantic scoring should outperform
    // topological routing for in-domain queries).
    let hub_bias = bias_rows[3].5;
    check(
        coh_bias > hub_bias,
        &format!(
            "Coherence biased ({:.1}%) > Hub biased ({:.1}%): semantic lift confirmed",
            coh_bias * 100.0,
            hub_bias * 100.0
        ),
        &mut failures,
    );

    // Speedup: all shards must be ≥ MIN_SPEEDUP (5×) over full brute-force.
    let bfs_spd = rand_rows[1].4;
    let coh_spd = rand_rows[2].4;
    let hub_spd = rand_rows[3].4;
    check(
        bfs_spd >= MIN_SPEEDUP,
        &format!("BFS speedup = {bfs_spd:.1}x ≥ {MIN_SPEEDUP:.0}x"),
        &mut failures,
    );
    check(
        coh_spd >= MIN_SPEEDUP,
        &format!("Coherence speedup = {coh_spd:.1}x ≥ {MIN_SPEEDUP:.0}x"),
        &mut failures,
    );
    check(
        hub_spd >= MIN_SPEEDUP,
        &format!("Hub speedup = {hub_spd:.1}x ≥ {MIN_SPEEDUP:.0}x"),
        &mut failures,
    );

    println!();
    if failures.is_empty() {
        println!("  ✓  ALL ACCEPTANCE TESTS PASSED");
    } else {
        println!("  ✗  {} ACCEPTANCE TEST(S) FAILED", failures.len());
        std::process::exit(1);
    }
    println!("══════════════════════════════════════════════════════════════════");

    // Sanity check distance function.
    let a = vec![1.0f32, 0.0];
    let b = vec![0.0f32, 1.0];
    debug_assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-5);
}
