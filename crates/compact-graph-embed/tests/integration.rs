use compact_graph_embed::{
    AnchorTokenizer, CsrGraph, EmbeddingTableF32, EmbeddingTableI8, MeanEmbedder, NodeEmbedder,
    NodeTokenizer, TokenStorage,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Helper: cosine similarity between two f32 slices
// ---------------------------------------------------------------------------

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Helper: generate a synthetic random graph
// ---------------------------------------------------------------------------

fn gen_graph(num_nodes: usize, num_edges: usize, seed: u64) -> CsrGraph {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(num_edges);
    for _ in 0..num_edges {
        let u = rng.gen_range(0..num_nodes);
        let v = rng.gen_range(0..num_nodes);
        edges.push((u, v));
    }
    CsrGraph::new(num_nodes, &edges)
}

// ---------------------------------------------------------------------------
// Test 1: CsrGraph neighbors and degree
// ---------------------------------------------------------------------------

#[test]
fn test_csr_graph_basics() {
    // Triangle: 0-1, 1-2, 0-2
    let edges = vec![(0, 1), (1, 2), (0, 2)];
    let g = CsrGraph::new(3, &edges);

    // Each node has degree 2 (connected to the other two)
    assert_eq!(g.degree(0), 2, "node 0 degree");
    assert_eq!(g.degree(1), 2, "node 1 degree");
    assert_eq!(g.degree(2), 2, "node 2 degree");

    // Neighbors of node 0 should contain 1 and 2
    let n0 = g.neighbors(0);
    assert!(n0.contains(&1), "node 0 should neighbor 1");
    assert!(n0.contains(&2), "node 0 should neighbor 2");

    // Star graph: 0 is hub connected to 1,2,3,4
    let star_edges: Vec<(usize, usize)> = (1..5).map(|i| (0, i)).collect();
    let star = CsrGraph::new(5, &star_edges);
    assert_eq!(star.degree(0), 4, "hub degree");
    for i in 1..5 {
        assert_eq!(star.degree(i), 1, "leaf {} degree", i);
    }

    // Self-loops should be excluded
    let loop_edges = vec![(0, 0), (0, 1), (1, 1)];
    let lg = CsrGraph::new(2, &loop_edges);
    assert_eq!(lg.degree(0), 1, "self-loop excluded for node 0");
    assert_eq!(lg.degree(1), 1, "self-loop excluded for node 1");

    // Duplicate edges should be deduplicated
    let dup_edges = vec![(0, 1), (0, 1), (1, 0)];
    let dg = CsrGraph::new(2, &dup_edges);
    assert_eq!(dg.degree(0), 1, "duplicate edges deduplicated");
}

// ---------------------------------------------------------------------------
// Test 2: AnchorTokenizer produces tokens for all nodes
// ---------------------------------------------------------------------------

#[test]
fn test_anchor_tokenizer_all_nodes() {
    // 10-node ring graph
    let edges: Vec<(usize, usize)> = (0..10).map(|i| (i, (i + 1) % 10)).collect();
    let graph = CsrGraph::new(10, &edges);

    let k = 3;
    let max_dist = 3u8;
    let tokenizer = AnchorTokenizer::new(&graph, k, max_dist, 42);

    assert_eq!(tokenizer.num_nodes(), 10);
    assert_eq!(tokenizer.num_anchors(), k);
    assert_eq!(tokenizer.max_dist(), max_dist);

    // Every node must have a non-empty token list
    for node in 0..10 {
        let tokens = tokenizer.tokenize(node).expect("should produce tokens");
        assert!(!tokens.is_empty(), "node {} has no tokens", node);
        // Token anchor indices must be in range
        for t in &tokens {
            assert!(
                (t.anchor_idx as usize) < k,
                "anchor_idx {} out of range",
                t.anchor_idx
            );
            assert!(t.dist >= 1 || t.dist == 0, "dist should be >= 0");
            assert!(t.dist <= max_dist, "dist {} > max_dist {}", t.dist, max_dist);
        }
        // Sorted by anchor_idx
        let sorted = tokens.windows(2).all(|w| w[0].anchor_idx <= w[1].anchor_idx);
        assert!(sorted, "tokens for node {} not sorted by anchor_idx", node);
    }

    // Out-of-range node should return error
    let err = tokenizer.tokenize(999);
    assert!(err.is_err(), "out-of-range node should error");
}

// ---------------------------------------------------------------------------
// Test 3: EmbeddingTableF32 and EmbeddingTableI8 produce same-dim output
// ---------------------------------------------------------------------------

#[test]
fn test_storage_dim_consistency() {
    let num_anchors = 4;
    let max_dist = 3u8;
    let dim = 64;

    let f32_table = EmbeddingTableF32::new_random(num_anchors, max_dist, dim, 7);
    let i8_table = EmbeddingTableI8::from_f32(&f32_table);

    assert_eq!(f32_table.dim(), dim);
    assert_eq!(i8_table.dim(), dim);
    assert_eq!(f32_table.num_entries(), num_anchors * max_dist as usize);
    assert_eq!(i8_table.num_entries(), num_anchors * max_dist as usize);

    // Check that embed_token returns correct dimension
    for anchor in 0..num_anchors as u32 {
        for d in 1..=max_dist {
            let f32_emb = f32_table.embed_token(anchor, d).unwrap();
            let i8_emb = i8_table.embed_token(anchor, d).unwrap();
            assert_eq!(f32_emb.len(), dim, "f32 dim mismatch anchor={} dist={}", anchor, d);
            assert_eq!(i8_emb.len(), dim, "i8 dim mismatch anchor={} dist={}", anchor, d);
        }
    }

    // Out-of-range accesses should error
    assert!(f32_table.embed_token(num_anchors as u32, 1).is_err());
    assert!(i8_table.embed_token(num_anchors as u32, 1).is_err());
    assert!(f32_table.embed_token(0, max_dist + 1).is_err());
    assert!(i8_table.embed_token(0, max_dist + 1).is_err());
}

// ---------------------------------------------------------------------------
// Test 4: Cosine similarity between f32 and i8 embeddings > 0.97
// ---------------------------------------------------------------------------

#[test]
fn test_f32_i8_cosine_similarity() {
    let num_nodes = 100;
    let num_edges = 500;
    let k = 8;
    let max_dist = 3u8;
    let dim = 64;

    let graph = gen_graph(num_nodes, num_edges, 11);
    let tokenizer_f32 = AnchorTokenizer::new(&graph, k, max_dist, 17);
    let tokenizer_i8 = AnchorTokenizer::new(&graph, k, max_dist, 17); // same seed = same anchors
    let f32_table = EmbeddingTableF32::new_random(k, max_dist, dim, 31);
    let i8_table = EmbeddingTableI8::from_f32(&f32_table);

    let embedder_f32 = MeanEmbedder::new(tokenizer_f32, f32_table);
    let embedder_i8 = MeanEmbedder::new(tokenizer_i8, i8_table);

    let mut total_sim = 0.0_f64;
    let mut count = 0usize;

    for node in 0..num_nodes {
        let emb_f32 = embedder_f32.embed(node).unwrap();
        let emb_i8 = embedder_i8.embed(node).unwrap();
        let sim = cosine_sim(&emb_f32, &emb_i8) as f64;
        total_sim += sim;
        count += 1;
    }

    let mean_sim = total_sim / count as f64;
    println!("Mean cosine similarity f32 vs i8: {:.4}", mean_sim);
    assert!(
        mean_sim > 0.97,
        "Mean cosine similarity {:.4} <= 0.97 — quantization too lossy",
        mean_sim
    );
}

// ---------------------------------------------------------------------------
// Test 5a: Acceptance test — ≥10x RAM compression on large graph
// ---------------------------------------------------------------------------

#[test]
fn test_acceptance_ram_compression() {
    const N: usize = 50_000;
    const E: usize = 500_000;
    const K: usize = 16;
    const MAX_DIST: u8 = 4;
    const DIM: usize = 128;

    let graph = gen_graph(N, E, 42);
    let tokenizer = AnchorTokenizer::new(&graph, K, MAX_DIST, 42);
    let token_bytes = tokenizer.byte_size();
    let f32_table = EmbeddingTableF32::new_random(K, MAX_DIST, DIM, 42);
    let i8_table = EmbeddingTableI8::from_f32(&f32_table);
    let embedder = MeanEmbedder::new(tokenizer, i8_table);

    let dense_ram = N * DIM * 4; // bytes for N x DIM f32
    let compact_ram = embedder.ram_bytes() + token_bytes;

    println!(
        "Dense RAM:   {} MB ({} bytes)",
        dense_ram / (1024 * 1024),
        dense_ram
    );
    println!(
        "Compact RAM: {} MB ({} bytes)",
        compact_ram / (1024 * 1024),
        compact_ram
    );
    println!(
        "Compression: {:.1}x",
        dense_ram as f64 / compact_ram as f64
    );

    assert!(
        compact_ram * 10 <= dense_ram,
        "Insufficient compression: compact={} bytes, dense={} bytes, need {}x got {:.1}x",
        compact_ram,
        dense_ram,
        10,
        dense_ram as f64 / compact_ram as f64
    );
}

// ---------------------------------------------------------------------------
// Test 5b: Acceptance test — mean cosine similarity > 0.97 (recall proxy)
// ---------------------------------------------------------------------------

#[test]
fn test_acceptance_embedding_quality() {
    const N: usize = 50_000;
    const E: usize = 500_000;
    const K: usize = 16;
    const MAX_DIST: u8 = 4;
    const DIM: usize = 128;

    let graph = gen_graph(N, E, 42);

    // Build both embedders with identical anchors/storage seeds
    let tok_f32 = AnchorTokenizer::new(&graph, K, MAX_DIST, 42);
    let tok_i8 = AnchorTokenizer::new(&graph, K, MAX_DIST, 42);
    let f32_table = EmbeddingTableF32::new_random(K, MAX_DIST, DIM, 42);
    let i8_table = EmbeddingTableI8::from_f32(&f32_table);

    let emb_f32 = MeanEmbedder::new(tok_f32, f32_table);
    let emb_i8 = MeanEmbedder::new(tok_i8, i8_table);

    // Sample 1000 random nodes
    let mut rng = SmallRng::seed_from_u64(999);
    let sample: Vec<usize> = (0..1000).map(|_| rng.gen_range(0..N)).collect();

    let mut total_sim = 0.0_f64;
    for &node in &sample {
        let f = emb_f32.embed(node).unwrap();
        let i = emb_i8.embed(node).unwrap();
        total_sim += cosine_sim(&f, &i) as f64;
    }
    let mean_sim = total_sim / sample.len() as f64;
    println!(
        "Acceptance: mean cosine similarity f32 vs i8 over {} nodes: {:.4}",
        sample.len(),
        mean_sim
    );
    assert!(
        mean_sim > 0.97,
        "Mean cosine sim {:.4} <= 0.97 (>3% recall drop)",
        mean_sim
    );
}

// ---------------------------------------------------------------------------
// Test 6: Latency — p50 of 10,000 embed() calls < 10 microseconds
// ---------------------------------------------------------------------------

#[test]
fn test_latency_p50() {
    const N: usize = 50_000;
    const E: usize = 500_000;
    const K: usize = 16;
    const MAX_DIST: u8 = 4;
    const DIM: usize = 128;

    let graph = gen_graph(N, E, 42);
    let tokenizer = AnchorTokenizer::new(&graph, K, MAX_DIST, 42);
    let f32_table = EmbeddingTableF32::new_random(K, MAX_DIST, DIM, 42);
    let i8_table = EmbeddingTableI8::from_f32(&f32_table);
    let embedder = MeanEmbedder::new(tokenizer, i8_table);

    let mut rng = SmallRng::seed_from_u64(777);
    let queries: Vec<usize> = (0..10_000).map(|_| rng.gen_range(0..N)).collect();

    // Warm up
    for &q in queries.iter().take(100) {
        let _ = embedder.embed(q).unwrap();
    }

    // Measure
    let mut latencies_ns: Vec<u64> = Vec::with_capacity(10_000);
    for &q in &queries {
        let start = std::time::Instant::now();
        let _ = embedder.embed(q).unwrap();
        latencies_ns.push(start.elapsed().as_nanos() as u64);
    }

    latencies_ns.sort_unstable();
    let p50_ns = latencies_ns[latencies_ns.len() / 2];
    let p50_us = p50_ns as f64 / 1_000.0;
    let p99_us = latencies_ns[(latencies_ns.len() * 99) / 100] as f64 / 1_000.0;

    println!("Latency p50: {:.2} µs", p50_us);
    println!("Latency p99: {:.2} µs", p99_us);

    assert!(
        p50_ns < 10_000,
        "p50 latency {:.2} µs exceeds 10 µs limit",
        p50_us
    );
}
