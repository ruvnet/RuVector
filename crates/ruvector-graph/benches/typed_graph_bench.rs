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
use ruvector_graph::{Edge, GraphDB, NodeBuilder, TraverseSpec, TypedGraph};

fn make_schema(dims: usize) -> GraphSchema {
    let mut s = GraphSchema::new();
    s.add_node(
        NodeSchema::new("Doc")
            .property(PropertySchema::new("title", PropertyType::String).required())
            .property(PropertySchema::new("embedding", PropertyType::Vector)),
    );
    s.add_node(NodeSchema::new("Topic").property(PropertySchema::new("name", PropertyType::String)));
    s.add_edge(EdgeSchema::new("ABOUT", "Doc", "Topic"));
    s.add_vector(VectorSchema::new("DocEmb", "Doc", "embedding", dims, DistanceMetric::Cosine));
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
            NodeBuilder::new().id(format!("t{t}")).label("Topic").property("name", format!("topic{t}")).build(),
        )
        .unwrap();
    }
    for i in 0..n {
        tg.create_node(
            NodeBuilder::new()
                .id(format!("d{i}"))
                .label("Doc")
                .property("title", format!("doc{i}"))
                .property("embedding", PropertyValue::FloatArray(embedding(i as u64, dims)))
                .build(),
        )
        .unwrap();
        // Two ABOUT edges per doc so traversal does real work.
        tg.create_edge(Edge::create(format!("d{i}"), format!("t{}", i % topics), "ABOUT")).unwrap();
        tg.create_edge(Edge::create(format!("d{i}"), format!("t{}", (i + 1) % topics), "ABOUT")).unwrap();
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
                    .search_then_traverse(black_box("DocEmb"), black_box(&query), black_box(10), &spec)
                    .unwrap();
                black_box(res);
            });
        });
    }
    group.finish();
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
        b.iter(|| black_box(schema.validate_node(black_box(&node)).unwrap()));
    });
}

fn bench_rrf(c: &mut Criterion) {
    let a: Vec<String> = (0..1000).map(|i| format!("id{i}")).collect();
    let b_list: Vec<String> = (0..1000).map(|i| format!("id{}", (i * 7) % 1000)).collect();
    c.bench_function("rrf_2x1000", |b| {
        b.iter(|| black_box(reciprocal_rank_fusion(black_box(&[a.clone(), b_list.clone()]), 60.0)));
    });
}

criterion_group!(benches, bench_search_then_traverse, bench_validation, bench_rrf);
criterion_main!(benches);
