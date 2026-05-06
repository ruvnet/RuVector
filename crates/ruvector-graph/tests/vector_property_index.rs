//! Integration tests for `VectorPropertyIndex` (Phase 1 / item #2 of the
//! RaBitQ-integration roadmap). Smaller-scale assertions than the 100k×768
//! acceptance test — that lives in `benches/vector_property_index.rs` and
//! is gated behind the `rabitq` feature so CI can skip it by default.

#![cfg(feature = "rabitq")]

use rand::{Rng, SeedableRng};
use ruvector_graph::{
    GraphDB, NodeBuilder, PropertyValue, VectorPropertyIndex, VectorPropertyIndexConfig,
};
use std::collections::HashSet;

const PROP: &str = "embedding";

/// Make `n` clustered `dim`-D vectors. Clustered data is what every recall
/// number in the RaBitQ paper is reported on, and uniform random gives
/// pathologically low recall at small n that wouldn't tell us anything
/// about the index implementation.
fn clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            (0..dim)
                .map(|_| rng.gen::<f32>() * 4.0 - 2.0)
                .collect::<Vec<_>>()
        })
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

fn populate_graph(graph: &GraphDB, vectors: &[Vec<f32>]) -> Vec<String> {
    let mut ids = Vec::with_capacity(vectors.len());
    for (i, v) in vectors.iter().enumerate() {
        let id = format!("node-{i:06}");
        let node = NodeBuilder::new()
            .id(id.clone())
            .label("Doc")
            .property(PROP, PropertyValue::FloatArray(v.clone()))
            .build();
        graph.create_node(node).expect("create_node");
        ids.push(id);
    }
    ids
}

/// Brute-force squared-L2 NN over the same property table — ground truth
/// for the recall assertion.
fn brute_force_topk(vectors: &[Vec<f32>], ids: &[String], q: &[f32], k: usize) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> = vectors
        .iter()
        .zip(ids.iter())
        .map(|(v, id)| {
            let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
            (d, id.as_str())
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored
        .into_iter()
        .take(k)
        .map(|(_, id)| id.to_string())
        .collect()
}

/// Smallest viable smoke test: self-query distance is ~0 and the closest
/// match is the node we queried with.
#[test]
fn self_query_returns_self_at_distance_zero() {
    let dim = 64;
    let n = 256;
    let vectors = clustered(n, dim, 8, 7);
    let graph = GraphDB::new();
    let node_ids = populate_graph(&graph, &vectors);

    let idx = VectorPropertyIndex::build(&graph, PROP, VectorPropertyIndexConfig::default())
        .expect("build");
    assert_eq!(idx.len(), n);
    assert_eq!(idx.dim(), dim);
    assert_eq!(idx.property(), PROP);

    // Pick a deterministic node, query with its own vector.
    let target_pos = 42usize;
    let q = &vectors[target_pos];
    let target_id = &node_ids[target_pos];

    let results = idx.knn(q, 5).expect("knn");
    assert_eq!(results.len(), 5, "should return 5 results");
    assert_eq!(results[0].0, *target_id, "self-match should be top-1");
    assert!(
        results[0].1 < 1e-3,
        "self-distance {} should be ~0",
        results[0].1
    );
    // Distances must be non-decreasing.
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1 + 1e-6, "results not sorted ascending");
    }
}

/// Recall@10 ≥ 0.85 vs brute force on 1k×128 with the default
/// `rerank_factor = 20`. The 100k×768 acceptance number is 0.95; we
/// shave down to 0.85 here so the assertion is solid even on noisy
/// random clusters at small n.
#[test]
fn recall_at_10_above_85_percent_at_1k_x_128() {
    let dim = 128;
    let n = 1000;
    let n_queries = 50;
    let total = clustered(n + n_queries, dim, 16, 2026);
    let (db, queries) = total.split_at(n);

    let graph = GraphDB::new();
    let node_ids = populate_graph(&graph, db);
    let idx =
        VectorPropertyIndex::build(&graph, PROP, VectorPropertyIndexConfig::default()).unwrap();

    let k = 10;
    let mut hits = 0usize;
    for q in queries {
        let truth: HashSet<String> = brute_force_topk(db, &node_ids, q, k).into_iter().collect();
        let got = idx.knn(q, k).unwrap();
        for (id, _) in got {
            if truth.contains(&id) {
                hits += 1;
            }
        }
    }
    let recall = hits as f64 / (n_queries * k) as f64;
    eprintln!("recall@10 at 1k×128 = {:.3}", recall);
    assert!(recall >= 0.85, "recall@10={:.3} below 0.85 floor", recall);
}

/// Memory ratio: codes bytes (rotation matrix + packed 1-bit codes + cos
/// LUT) must come in at ≤ originals/16 + a fixed rotation overhead.
/// The roadmap acceptance is ≤ originals/16 — at small n the rotation
/// matrix dominates, so we fold in a `dim*dim*4` overhead allowance.
#[test]
fn codes_memory_below_one_sixteenth_plus_rotation() {
    let dim = 128;
    let n = 1000;
    let vectors = clustered(n, dim, 16, 31);
    let graph = GraphDB::new();
    populate_graph(&graph, &vectors);
    let idx =
        VectorPropertyIndex::build(&graph, PROP, VectorPropertyIndexConfig::default()).unwrap();

    let originals = idx.original_bytes();
    let codes = idx.codes_bytes();
    // 1-bit per dim → dim/8 bytes per row + small SoA overhead. At
    // dim=128 the rotation matrix is 128*128*4=64 KiB, which dominates
    // at n=1k. Allow that overhead in the budget.
    let rotation_overhead = dim * dim * 4;
    let budget = originals / 16 + rotation_overhead + 4096;
    eprintln!(
        "codes={codes}B, originals={originals}B, ratio={:.3}, budget={budget}",
        codes as f64 / originals as f64
    );
    assert!(
        codes <= budget,
        "codes={codes}B > budget={budget}B (originals={originals}B)"
    );
}

/// ADR-154 determinism: same `(seed, graph)` → byte-identical packed
/// codes across builds. We can't reach the inner SoA bytes without
/// exposing accessors we don't want to expose, so we use the next-best
/// proxy: identical query → identical (NodeId, score-bits) sequence.
#[test]
fn determinism_same_seed_byte_identical_results() {
    let dim = 96;
    let n = 500;
    let vectors = clustered(n, dim, 12, 99);
    let graph = GraphDB::new();
    populate_graph(&graph, &vectors);
    let cfg = VectorPropertyIndexConfig {
        seed: 0xC0FFEE,
        rerank_factor: 8,
    };

    let a = VectorPropertyIndex::build(&graph, PROP, cfg.clone()).unwrap();
    let b = VectorPropertyIndex::build(&graph, PROP, cfg.clone()).unwrap();
    assert_eq!(a.len(), b.len());
    assert_eq!(a.dim(), b.dim());

    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    for _ in 0..10 {
        let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let ra = a.knn(&q, 8).unwrap();
        let rb = b.knn(&q, 8).unwrap();
        assert_eq!(ra.len(), rb.len(), "result count differs across builds");
        for ((id_a, sc_a), (id_b, sc_b)) in ra.iter().zip(rb.iter()) {
            assert_eq!(id_a, id_b, "NodeId differs");
            assert_eq!(
                sc_a.to_bits(),
                sc_b.to_bits(),
                "score bits differ for {id_a}",
            );
        }
    }
}

/// Nodes that lack the property (or carry it as a non-FloatArray) are
/// silently skipped, not errored. The index simply contains fewer rows.
#[test]
fn nodes_without_property_are_skipped() {
    let dim = 32;
    let graph = GraphDB::new();
    // 10 nodes with the property, 5 without.
    let with_vec = clustered(10, dim, 4, 1);
    populate_graph(&graph, &with_vec);
    for i in 0..5 {
        let n = NodeBuilder::new()
            .id(format!("plain-{i}"))
            .label("Doc")
            .property("name", "alice")
            .build();
        graph.create_node(n).unwrap();
    }
    let idx =
        VectorPropertyIndex::build(&graph, PROP, VectorPropertyIndexConfig::default()).unwrap();
    assert_eq!(idx.len(), 10);
}

/// Building over a graph with zero matching nodes returns a clear
/// `InvalidInput`, not a panic from the underlying rabitq crate.
#[test]
fn build_fails_cleanly_on_empty_property_set() {
    let graph = GraphDB::new();
    let n = NodeBuilder::new()
        .id("only")
        .label("Doc")
        .property("name", "no embedding here")
        .build();
    graph.create_node(n).unwrap();
    let res = VectorPropertyIndex::build(&graph, PROP, VectorPropertyIndexConfig::default());
    let err = match res {
        Ok(_) => panic!("should fail with no FloatArray properties"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("FloatArray") || msg.contains(PROP),
        "unexpected error: {msg}"
    );
}

/// Dim mismatch on the query is an error, not a panic.
#[test]
fn knn_rejects_dim_mismatch() {
    let dim = 32;
    let vectors = clustered(64, dim, 4, 5);
    let graph = GraphDB::new();
    populate_graph(&graph, &vectors);
    let idx =
        VectorPropertyIndex::build(&graph, PROP, VectorPropertyIndexConfig::default()).unwrap();

    let bad_q = vec![0.0_f32; dim + 1];
    let err = match idx.knn(&bad_q, 5) {
        Ok(_) => panic!("expected dim mismatch"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("dim"));
}
