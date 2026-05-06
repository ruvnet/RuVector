//! Acceptance-shaped bench for `VectorPropertyIndex`. The roadmap target
//! is recall@10 ≥ 0.95 at 100k×768 against brute force, with index
//! memory ≤ 1/16 of the f32 baseline. Default `n` here is small enough
//! to run on CI; override with `VECTOR_PROPERTY_INDEX_N=100000` to hit
//! the full acceptance scale.

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use ruvector_graph::{
    GraphDB, NodeBuilder, PropertyValue, VectorPropertyIndex, VectorPropertyIndexConfig,
};

const PROP: &str = "embedding";

fn clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 4.0 - 2.0).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter()
                .map(|&x| x + (rng.gen::<f32>() - 0.5) * 0.3)
                .collect()
        })
        .collect()
}

fn build_graph(vectors: &[Vec<f32>]) -> GraphDB {
    let g = GraphDB::new();
    for (i, v) in vectors.iter().enumerate() {
        let node = NodeBuilder::new()
            .id(format!("n-{i:08}"))
            .label("Doc")
            .property(PROP, PropertyValue::FloatArray(v.clone()))
            .build();
        g.create_node(node).unwrap();
    }
    g
}

fn run_bench(c: &mut Criterion) {
    let n: usize = std::env::var("VECTOR_PROPERTY_INDEX_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let dim: usize = std::env::var("VECTOR_PROPERTY_INDEX_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    let vectors = clustered(n, dim, 32, 0xACCE57);
    let graph = build_graph(&vectors);

    c.bench_function(
        &format!("vector_property_index/build/n={n}/dim={dim}"),
        |b| {
            b.iter(|| {
                let _idx =
                    VectorPropertyIndex::build(&graph, PROP, VectorPropertyIndexConfig::default())
                        .unwrap();
            });
        },
    );

    let idx =
        VectorPropertyIndex::build(&graph, PROP, VectorPropertyIndexConfig::default()).unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(0xBA5E1);
    let queries: Vec<Vec<f32>> = (0..50)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    let mut q_idx = 0usize;
    c.bench_function(
        &format!("vector_property_index/knn/k=10/n={n}/dim={dim}"),
        |b| {
            b.iter(|| {
                let q = &queries[q_idx % queries.len()];
                q_idx = q_idx.wrapping_add(1);
                let _ = idx.knn(q, 10).unwrap();
            });
        },
    );

    eprintln!(
        "[vector_property_index bench] n={n} dim={dim} codes={} B originals={} B",
        idx.codes_bytes(),
        idx.original_bytes()
    );
}

criterion_group!(benches, run_bench);
criterion_main!(benches);
