//! Benchmarks for the schema-first typed graph (ADR-252 P1/P2/P4).
//!
//! Measures the fused `search_then_traverse` operator at scale, schema
//! validation overhead, and RRF fusion. Run with:
//! `cargo bench -p ruvector-graph --bench typed_graph_bench`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvector_graph::schema::{
    reciprocal_rank_fusion, DistanceMetric, EdgeSchema, GraphSchema, NodeSchema, PropertySchema,
    PropertyType, VectorSchema,
};
use ruvector_graph::types::PropertyValue;
use ruvector_graph::{
    Edge, Embedder, GraphDB, HashEmbedder, NodeBuilder, TraverseSpec, TypedGraph,
};
use std::sync::Arc;

fn make_schema(dims: usize) -> GraphSchema {
    let mut s = GraphSchema::new();
    s.add_node(
        NodeSchema::new("Doc")
            .property(PropertySchema::new("title", PropertyType::String).required())
            .property(PropertySchema::new("embedding", PropertyType::Vector)),
    );
    s.add_node(
        NodeSchema::new("Topic").property(PropertySchema::new("name", PropertyType::String)),
    );
    s.add_edge(EdgeSchema::new("ABOUT", "Doc", "Topic"));
    s.add_vector(VectorSchema::new(
        "DocEmb",
        "Doc",
        "embedding",
        dims,
        DistanceMetric::Cosine,
    ));
    s
}

/// Deterministic pseudo-random embedding so benches are reproducible.
fn embedding(seed: u64, dims: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..dims)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f32 / u64::MAX as f32) - 0.5
        })
        .collect()
}

fn build_graph(n: usize, dims: usize, topics: usize) -> TypedGraph {
    let tg = TypedGraph::new(GraphDB::new(), make_schema(dims)).unwrap();
    for t in 0..topics {
        tg.create_node(
            NodeBuilder::new()
                .id(format!("t{t}"))
                .label("Topic")
                .property("name", format!("topic{t}"))
                .build(),
        )
        .unwrap();
    }
    for i in 0..n {
        tg.create_node(
            NodeBuilder::new()
                .id(format!("d{i}"))
                .label("Doc")
                .property("title", format!("doc{i}"))
                .property(
                    "embedding",
                    PropertyValue::FloatArray(embedding(i as u64, dims)),
                )
                .build(),
        )
        .unwrap();
        // Two ABOUT edges per doc so traversal does real work.
        tg.create_edge(Edge::create(
            format!("d{i}"),
            format!("t{}", i % topics),
            "ABOUT",
        ))
        .unwrap();
        tg.create_edge(Edge::create(
            format!("d{i}"),
            format!("t{}", (i + 1) % topics),
            "ABOUT",
        ))
        .unwrap();
    }
    tg
}

fn bench_search_then_traverse(c: &mut Criterion) {
    let dims = 128;
    let mut group = c.benchmark_group("search_then_traverse");
    for &n in &[1_000usize, 10_000, 50_000] {
        let tg = build_graph(n, dims, 64);
        let query = embedding(424242, dims);
        let spec = TraverseSpec::out("ABOUT").target_label("Topic");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let res = tg
                    .search_then_traverse(
                        black_box("DocEmb"),
                        black_box(&query),
                        black_box(10),
                        &spec,
                    )
                    .unwrap();
                black_box(res);
            });
        });
    }
    group.finish();
}

fn bench_indexed_vs_scan(c: &mut Criterion) {
    let dims = 128;
    let n = 50_000usize;
    let query = embedding(424242, dims);
    let spec = TraverseSpec::out("ABOUT").target_label("Topic");

    // Brute-force scan (no index).
    let scan = build_graph(n, dims, 64);
    // Same data, with an HNSW push-down index built.
    let mut indexed = build_graph(n, dims, 64);
    indexed.build_vector_index("DocEmb").unwrap();

    let mut group = c.benchmark_group("search_50k_k10");
    group.bench_function("brute_force_scan", |b| {
        b.iter(|| {
            black_box(
                scan.search_then_traverse(black_box("DocEmb"), black_box(&query), 10, &spec)
                    .unwrap(),
            );
        });
    });
    group.bench_function("hnsw_pushdown", |b| {
        b.iter(|| {
            black_box(
                indexed
                    .search_then_traverse(black_box("DocEmb"), black_box(&query), 10, &spec)
                    .unwrap(),
            );
        });
    });
    group.finish();
}

fn bench_embed_and_hybrid(c: &mut Criterion) {
    use ruvector_graph::{
        DistanceMetric, EdgeSchema, GraphSchema, NodeSchema, PropertySchema, PropertyType,
        VectorSchema,
    };

    let dims = 256;
    let embedder = HashEmbedder::new(dims);
    c.bench_function("hash_embed_256", |b| {
        b.iter(|| {
            black_box(
                embedder
                    .embed(black_box("vector database semantic similarity search"))
                    .unwrap(),
            )
        });
    });

    // Build a 10k-doc tri-modal graph: text + inline embeddings + BM25 + ANN.
    let n = 10_000usize;
    let words = [
        "vector",
        "database",
        "graph",
        "search",
        "embedding",
        "model",
        "index",
        "neural",
        "semantic",
        "traversal",
        "cluster",
        "ranking",
        "query",
        "fusion",
    ];
    let text_for = |i: usize| -> String {
        (0..6)
            .map(|j| words[(i * 7 + j * 13) % words.len()])
            .collect::<Vec<_>>()
            .join(" ")
    };

    let mut schema = GraphSchema::new();
    schema.add_node(
        NodeSchema::new("Doc")
            .property(PropertySchema::new("body", PropertyType::String).required())
            .property(PropertySchema::new("embedding", PropertyType::Vector)),
    );
    schema.add_node(NodeSchema::new("Topic"));
    schema.add_edge(EdgeSchema::new("ABOUT", "Doc", "Topic"));
    schema.add_vector(VectorSchema::new(
        "DocEmb",
        "Doc",
        "embedding",
        dims,
        DistanceMetric::Cosine,
    ));

    let mut tg = TypedGraph::new(GraphDB::new(), schema)
        .unwrap()
        .with_embedder(Arc::new(HashEmbedder::new(dims)));
    for i in 0..n {
        let body = text_for(i);
        let node = NodeBuilder::new()
            .id(format!("d{i}"))
            .label("Doc")
            .property("body", body.clone())
            .build();
        tg.create_node_from_text(node, "DocEmb", &body).unwrap();
    }
    tg.build_vector_index("DocEmb").unwrap();
    tg.build_text_index("Doc", "body").unwrap();

    let spec = TraverseSpec::out("ABOUT");
    c.bench_function("tri_modal_hybrid_10k_k10", |b| {
        b.iter(|| {
            black_box(
                tg.hybrid_search_text(
                    black_box("DocEmb"),
                    "body",
                    black_box("vector semantic search ranking"),
                    10,
                    60.0,
                    &spec,
                )
                .unwrap(),
            );
        });
    });
}

fn bench_validation(c: &mut Criterion) {
    let schema = make_schema(128);
    let node = NodeBuilder::new()
        .id("d1")
        .label("Doc")
        .property("title", "hello")
        .property("embedding", PropertyValue::FloatArray(embedding(1, 128)))
        .build();
    c.bench_function("validate_node", |b| {
        b.iter(|| {
            schema.validate_node(black_box(&node)).unwrap();
        });
    });
}

fn bench_rrf(c: &mut Criterion) {
    let a: Vec<String> = (0..1000).map(|i| format!("id{i}")).collect();
    let b_list: Vec<String> = (0..1000).map(|i| format!("id{}", (i * 7) % 1000)).collect();
    c.bench_function("rrf_2x1000", |b| {
        b.iter(|| {
            black_box(reciprocal_rank_fusion(
                black_box(&[a.clone(), b_list.clone()]),
                60.0,
            ))
        });
    });
}

criterion_group!(
    benches,
    bench_search_then_traverse,
    bench_indexed_vs_scan,
    bench_embed_and_hybrid,
    bench_validation,
    bench_rrf
);
criterion_main!(benches);
