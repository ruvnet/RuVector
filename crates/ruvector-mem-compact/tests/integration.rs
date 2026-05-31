//! Integration tests for all three compaction strategies.
//!
//! Test categories:
//!
//! 1. **Ratio tests** — each compactor must hit the target ratio.
//!
//! 2. **ID-exact recall (moderate data)** — GraphCutCompact recall@K must be
//!    within -5 pp of AgeTtl.  On a uniform Gaussian dataset, both strategies
//!    approach the 50% theoretical ceiling; graph-cut's goal here is *parity*,
//!    not improvement.
//!
//! 3. **Cluster-coverage recall (redundant data)** — the primary use case.
//!    Agent memory accumulates near-duplicate embeddings of the same concept.
//!    GraphCutCompact must achieve ≥ 90% cluster-coverage recall at 50%
//!    compaction, while AgeTtl drops entire old-concept clusters.
//!
//! 4. **Correctness** — all returned IDs must belong to the original store.

use ruvector_mem_compact::dataset::{
    generate_concept_queries, generate_queries, generate_redundant_store, generate_store,
    DatasetParams, RedundantParams,
};
use ruvector_mem_compact::metrics::{cluster_coverage_recall, recall_at_k};
use ruvector_mem_compact::{
    AgeTtlCompactor, GraphCutCompactor, MemoryCompactor, ThresholdCompactor,
};

fn moderate_params() -> DatasetParams {
    DatasetParams {
        n: 500,
        dims: 64,
        n_clusters: 10,
        cluster_std: 0.2,
        seed: 42,
    }
}

fn redundant_params() -> RedundantParams {
    RedundantParams {
        n_concepts: 30,
        copies_per_concept: 20,
        dims: 64,
        dup_noise: 0.04,
        seed: 1337,
    }
}

/// Build a cluster-label array: entry i belongs to cluster (i % n_clusters)
/// because `generate_store` assigns round-robin.
fn make_cluster_labels(n: usize, n_clusters: usize) -> Vec<usize> {
    (0..n).map(|i| i % n_clusters).collect()
}

/// Build cluster labels for the redundant store: entry i belongs to concept
/// (i / copies_per_concept).
fn make_redundant_labels(n_concepts: usize, copies: usize) -> Vec<usize> {
    (0..n_concepts * copies).map(|i| i / copies).collect()
}

// ── Ratio tests ───────────────────────────────────────────────────────────────

#[test]
fn age_ttl_compacts_to_target_ratio() {
    let store = generate_store(&moderate_params());
    let result = AgeTtlCompactor.compact(&store, 0.5);
    let ratio = result.kept_ids.len() as f32 / store.len() as f32;
    assert!((ratio - 0.5).abs() < 0.02, "AgeTtl ratio {ratio:.3}");
}

#[test]
fn threshold_compacts_within_range() {
    let store = generate_store(&moderate_params());
    let result = ThresholdCompactor::new(0.90).compact(&store, 0.5);
    let ratio = result.kept_ids.len() as f32 / store.len() as f32;
    assert!(
        ratio >= 0.40 && ratio <= 0.65,
        "ThresholdMerge ratio {ratio:.3}"
    );
}

#[test]
fn graph_cut_compacts_within_range() {
    let store = generate_store(&moderate_params());
    let result = GraphCutCompactor::new(8, 0.70).compact(&store, 0.5);
    let ratio = result.kept_ids.len() as f32 / store.len() as f32;
    assert!(
        ratio >= 0.40 && ratio <= 0.65,
        "GraphCutCompact ratio {ratio:.3}"
    );
}

// ── ID-exact recall: moderate data (parity, not improvement) ─────────────────

#[test]
fn graph_cut_recall_within_5pp_of_age_ttl() {
    let params = moderate_params();
    let store = generate_store(&params);
    let queries = generate_queries(&params, 40);
    let k = 10;

    let gc_r = recall_at_k(
        &queries,
        &store,
        &store.apply(
            &GraphCutCompactor::new(8, 0.70)
                .compact(&store, 0.5)
                .kept_ids,
        ),
        k,
    );
    let ttl_r = recall_at_k(
        &queries,
        &store,
        &store.apply(&AgeTtlCompactor.compact(&store, 0.5).kept_ids),
        k,
    );

    // Graph-cut's ID-exact recall may be slightly lower on uniform Gaussian data;
    // accept up to 5 pp deficit.  The real advantage is cluster-coverage recall.
    assert!(
        gc_r >= ttl_r - 0.05,
        "GraphCutCompact ID-recall {gc_r:.3} more than 5 pp below AgeTtl {ttl_r:.3}"
    );
}

// ── Cluster-coverage recall: redundant data (primary use case) ────────────────

/// With near-duplicate episodic memories (σ=0.04), GraphCutCompact correctly
/// identifies each concept cluster and keeps a representative.  The concept
/// queries are the concept centres — any same-concept vector in the top-K is a
/// valid "hit" for cluster-coverage recall.
///
/// AgeTtl drops entire old-concept clusters (the first 15 of 30 concepts are
/// old and get fully evicted), so its cluster-coverage recall ≈ 50%.
///
/// GraphCutCompact cluster-coverage threshold: ≥ 90%.
/// Basis: the 30 component representatives are always retained, covering all
/// concepts, so coverage is limited only by the padding quality.
#[test]
fn graph_cut_cluster_coverage_on_redundant_data() {
    let params = redundant_params();
    let store = generate_redundant_store(&params);
    let queries = generate_concept_queries(&params);
    let cluster_labels = make_redundant_labels(params.n_concepts, params.copies_per_concept);
    let k = 5;

    let result = GraphCutCompactor::new(8, 0.70).compact(&store, 0.5);
    let compacted = store.apply(&result.kept_ids);
    let ccr = cluster_coverage_recall(&queries, &store, &compacted, &cluster_labels, k);

    assert!(
        ccr >= 0.90,
        "GraphCutCompact cluster-coverage recall = {ccr:.3} < required 0.90"
    );
}

/// AgeTtl drops old concepts entirely, achieving much lower cluster-coverage.
/// This documents the baseline degradation and confirms graph-cut improves on it.
#[test]
fn graph_cut_cluster_coverage_beats_age_ttl() {
    let params = redundant_params();
    let store = generate_redundant_store(&params);
    let queries = generate_concept_queries(&params);
    let cluster_labels = make_redundant_labels(params.n_concepts, params.copies_per_concept);
    let k = 5;

    let gc_result = GraphCutCompactor::new(8, 0.70).compact(&store, 0.5);
    let ttl_result = AgeTtlCompactor.compact(&store, 0.5);

    let gc_ccr = cluster_coverage_recall(
        &queries,
        &store,
        &store.apply(&gc_result.kept_ids),
        &cluster_labels,
        k,
    );
    let ttl_ccr = cluster_coverage_recall(
        &queries,
        &store,
        &store.apply(&ttl_result.kept_ids),
        &cluster_labels,
        k,
    );

    assert!(
        gc_ccr > ttl_ccr,
        "GraphCutCompact CCR {gc_ccr:.3} should beat AgeTtl CCR {ttl_ccr:.3}"
    );
}

// ── Correctness ───────────────────────────────────────────────────────────────

#[test]
fn all_kept_ids_are_valid() {
    let store = generate_store(&moderate_params());
    let valid: std::collections::HashSet<u64> = store.entries.iter().map(|e| e.id).collect();
    for compactor in [
        &AgeTtlCompactor as &dyn MemoryCompactor,
        &ThresholdCompactor::new(0.90),
        &GraphCutCompactor::new(8, 0.70),
    ] {
        let result = compactor.compact(&store, 0.5);
        for id in &result.kept_ids {
            assert!(
                valid.contains(id),
                "{} returned unknown id {id}",
                compactor.name()
            );
        }
    }
}
