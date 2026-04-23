use compact_graph_embed::{
    AnchorTokenizer, CsrGraph, EmbeddingTableF32, EmbeddingTableI8, MeanEmbedder, NodeEmbedder,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Generate a synthetic random graph with N nodes and E edges.
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

fn bench_dense_lookup(c: &mut Criterion) {
    const N: usize = 10_000;
    const DIM: usize = 128;

    // Dense per-node embedding: flat Vec<f32> of size N * DIM
    let mut rng = SmallRng::seed_from_u64(1);
    let dense: Vec<f32> = (0..N * DIM).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();

    let mut query_rng = SmallRng::seed_from_u64(99);

    c.bench_function("dense_lookup", |b| {
        b.iter(|| {
            let node = query_rng.gen_range(0..N);
            let start = node * DIM;
            let slice = &dense[start..start + DIM];
            black_box(slice);
        });
    });
}

fn bench_compact_f32_embed(c: &mut Criterion) {
    const N: usize = 10_000;
    const E: usize = 100_000;
    const K: usize = 16;
    const MAX_DIST: u8 = 3;
    const DIM: usize = 128;

    let graph = gen_graph(N, E, 42);
    let tokenizer = AnchorTokenizer::new(&graph, K, MAX_DIST, 42);
    let storage = EmbeddingTableF32::new_random(K, MAX_DIST, DIM, 42);
    let embedder = MeanEmbedder::new(tokenizer, storage);

    let mut query_rng = SmallRng::seed_from_u64(99);

    c.bench_function("compact_f32_embed", |b| {
        b.iter(|| {
            let node = query_rng.gen_range(0..N);
            black_box(embedder.embed(node).unwrap());
        });
    });
}

fn bench_compact_i8_embed(c: &mut Criterion) {
    const N: usize = 10_000;
    const E: usize = 100_000;
    const K: usize = 16;
    const MAX_DIST: u8 = 3;
    const DIM: usize = 128;

    let graph = gen_graph(N, E, 42);
    let tokenizer = AnchorTokenizer::new(&graph, K, MAX_DIST, 42);
    let f32_table = EmbeddingTableF32::new_random(K, MAX_DIST, DIM, 42);
    let i8_table = EmbeddingTableI8::from_f32(&f32_table);
    let embedder = MeanEmbedder::new(tokenizer, i8_table);

    let mut query_rng = SmallRng::seed_from_u64(99);

    c.bench_function("compact_i8_embed", |b| {
        b.iter(|| {
            let node = query_rng.gen_range(0..N);
            black_box(embedder.embed(node).unwrap());
        });
    });
}

criterion_group!(
    benches,
    bench_dense_lookup,
    bench_compact_f32_embed,
    bench_compact_i8_embed
);
criterion_main!(benches);
