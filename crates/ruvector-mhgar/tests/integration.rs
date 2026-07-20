//! Integration tests for MHGAR retrieval variants.
//!
//! Tests cover two regimes:
//! - NearHub: satellites are close to hub in vector space (VectorOnly already works)
//! - CrossCluster: satellites are semantically distant; graph + hop_discount is required

use ruvector_mhgar::{
    graph::KnowledgeGraph,
    recall_at_k,
    retriever::{
        CoherenceGatedHopper, OneHopExpander, Retriever, VectorOnlyRetriever,
    },
    synth::{self, SatelliteMode},
    vector::FlatIndex,
    Hit,
};

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[test]
fn flat_index_search_returns_k_results() {
    let mut idx = FlatIndex::new(4);
    for i in 0u32..20 {
        idx.insert(i, &[i as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn flat_index_dimension_mismatch_is_error() {
    let mut idx2 = FlatIndex::new(4);
    idx2.insert(0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    let err = idx2.search(&[1.0, 0.0], 3);
    assert!(err.is_err());
}

#[test]
fn graph_expand_one_hop_excludes_visited() {
    let mut g = KnowledgeGraph::new();
    g.add_edge(0, 1, 1.0);
    g.add_edge(0, 2, 0.8);
    g.add_edge(0, 3, 0.6);

    let visited: std::collections::HashSet<u32> = [0, 1].into_iter().collect();
    let expanded = g.expand_one_hop([0].into_iter(), &visited);
    let ids: Vec<u32> = expanded.iter().map(|(id, _)| *id).collect();
    assert!(!ids.contains(&1), "visited id 1 must be excluded");
    assert!(ids.contains(&2) && ids.contains(&3));
}

#[test]
fn recall_at_k_perfect_when_all_found() {
    let hits = vec![Hit::new(0, 0.1, 0), Hit::new(1, 0.2, 0), Hit::new(2, 0.3, 0)];
    let gt = vec![0, 1, 2];
    let r = recall_at_k(&hits, &gt);
    assert!((r - 1.0).abs() < 1e-6, "expected 1.0, got {r}");
}

#[test]
fn recall_at_k_partial() {
    let hits = vec![Hit::new(0, 0.1, 0), Hit::new(99, 0.2, 0), Hit::new(2, 0.3, 0)];
    let gt = vec![0, 1, 2];
    let r = recall_at_k(&hits, &gt);
    // found 0 and 2, missed 1 → 2/3
    assert!((r - 2.0 / 3.0).abs() < 1e-5, "expected 0.666…, got {r}");
}

// ─── Integration tests ───────────────────────────────────────────────────────

#[test]
fn vector_only_finds_hub_entity() {
    // NearHub: query is near hub; VectorOnly must always find the hub.
    let ds = synth::build(SatelliteMode::NearHub { noise_std: 0.3 }, 10, 5, 32, 0.3, 20, 1);
    let v = VectorOnlyRetriever { index: &ds.index };
    for (q, gt) in ds.queries.iter().zip(ds.ground_truth.iter()) {
        let hub_id = gt[0];
        let results = v.search(q, 5).unwrap();
        let found_hub = results.iter().any(|h| h.id == hub_id);
        assert!(found_hub, "VectorOnly missed hub_id={hub_id}");
    }
}

#[test]
fn near_hub_satellites_found_by_vector_only() {
    // With low noise, vector-only already finds most ground truth — no graph needed.
    let ds = synth::build(SatelliteMode::NearHub { noise_std: 0.2 }, 20, 4, 32, 0.2, 40, 55);
    let k = 5;
    let v1 = VectorOnlyRetriever { index: &ds.index };
    let mut mean_recall = 0.0f32;
    for (q, gt) in ds.queries.iter().zip(ds.ground_truth.iter()) {
        let gt_ids: Vec<u32> = gt.iter().copied().take(k).collect();
        mean_recall += recall_at_k(&v1.search(q, k).unwrap(), &gt_ids);
    }
    mean_recall /= ds.queries.len() as f32;
    assert!(mean_recall > 0.50, "Expected VectorOnly recall >0.50 in NearHub mode, got {mean_recall:.4}");
}

#[test]
fn one_hop_expander_improves_recall_over_vector_only_cross_cluster() {
    // CrossCluster: satellites are random vectors — NOT close to hub.
    // Pure cosine re-ranking (hop_discount=0.0) provides NO improvement.
    // Graph-weight-influenced scoring (hop_discount=0.5) DOES improve recall.
    //
    // We use num_seeds_to_expand=1 (expand only from hub) and initial_k=k
    // so only hub's satellites enter the candidate pool.
    let num_hubs = 20;
    let sats = 5;
    let k = sats + 1; // hub + all satellites

    let ds = synth::build(SatelliteMode::CrossCluster, num_hubs, sats, 64, 0.1, 60, 42);

    let v1 = VectorOnlyRetriever { index: &ds.index };
    let v2 = OneHopExpander {
        index: &ds.index,
        graph: &ds.graph,
        initial_k: k,
        num_seeds_to_expand: 1,  // Only expand from the closest seed (= hub)
        hop_discount: 0.5,       // Graph-found entities score at 50% of raw distance
    };

    let mut recall_vo = 0.0f32;
    let mut recall_oh = 0.0f32;

    for (q, gt) in ds.queries.iter().zip(ds.ground_truth.iter()) {
        let gt_ids: Vec<u32> = gt.iter().copied().take(k).collect();
        recall_vo += recall_at_k(&v1.search(q, k).unwrap(), &gt_ids);
        recall_oh += recall_at_k(&v2.search(q, k).unwrap(), &gt_ids);
    }
    let n = ds.queries.len() as f32;
    let recall_vo = recall_vo / n;
    let recall_oh = recall_oh / n;

    assert!(
        recall_oh >= recall_vo + 0.10,
        "OneHopExpander recall {recall_oh:.4} should be ≥ VectorOnly {recall_vo:.4} + 0.10 in CrossCluster mode"
    );
}

#[test]
fn naive_expansion_no_discount_matches_vector_only() {
    // With hop_discount=0.0, graph expansion + pure cosine re-ranking is
    // equivalent to VectorOnly in CrossCluster — this confirms the research finding.
    let ds = synth::build(SatelliteMode::CrossCluster, 10, 4, 32, 0.1, 30, 7);
    let k = 5;
    let v1 = VectorOnlyRetriever { index: &ds.index };
    let v2_naive = OneHopExpander {
        index: &ds.index,
        graph: &ds.graph,
        initial_k: k * 3,
        num_seeds_to_expand: 3,
        hop_discount: 0.0,  // No graph trust → same as VectorOnly
    };

    let mut recall_vo = 0.0f32;
    let mut recall_naive = 0.0f32;
    for (q, gt) in ds.queries.iter().zip(ds.ground_truth.iter()) {
        let gt_ids: Vec<u32> = gt.iter().copied().take(k).collect();
        recall_vo += recall_at_k(&v1.search(q, k).unwrap(), &gt_ids);
        recall_naive += recall_at_k(&v2_naive.search(q, k).unwrap(), &gt_ids);
    }
    let n = ds.queries.len() as f32;
    let recall_vo = recall_vo / n;
    let recall_naive = recall_naive / n;

    // Naive expansion (no discount) should NOT significantly outperform VectorOnly
    // in CrossCluster — verifies the core research claim.
    let improvement = recall_naive - recall_vo;
    assert!(
        improvement < 0.10,
        "Naive expansion improvement ({improvement:.4}) unexpectedly high — check CrossCluster generation"
    );
}

#[test]
fn coherence_gated_hopper_improves_over_vector_only_cross_cluster() {
    let num_hubs = 15;
    let sats = 4;
    let k = sats + 1;
    let ds = synth::build(SatelliteMode::CrossCluster, num_hubs, sats, 64, 0.1, 45, 99);

    let v1 = VectorOnlyRetriever { index: &ds.index };
    let v3 = CoherenceGatedHopper {
        index: &ds.index,
        graph: &ds.graph,
        initial_k: k,
        num_seeds_to_expand: 1,  // Expand only from top ANN result (= hub)
        max_hops: 2,
        // 0.50 is reliably below CrossCluster mean-dist (~0.80) so expansion
        // always fires; it is far above NearHub mean-dist (~0.05) so the gate
        // fires correctly (stops early) in NearHub scenes.
        expansion_threshold: 0.50,
        hop_discount: 0.5,
    };

    let mut recall_vo = 0.0f32;
    let mut recall_cg = 0.0f32;
    for (q, gt) in ds.queries.iter().zip(ds.ground_truth.iter()) {
        let gt_ids: Vec<u32> = gt.iter().copied().take(k).collect();
        recall_vo += recall_at_k(&v1.search(q, k).unwrap(), &gt_ids);
        recall_cg += recall_at_k(&v3.search(q, k).unwrap(), &gt_ids);
    }
    let n = ds.queries.len() as f32;
    let recall_vo = recall_vo / n;
    let recall_cg = recall_cg / n;

    assert!(
        recall_cg >= recall_vo + 0.05,
        "CoherenceGatedHopper recall {recall_cg:.4} should improve by ≥0.05 over VectorOnly {recall_vo:.4}"
    );
}

#[test]
fn synth_dataset_entity_count_is_correct() {
    let num_hubs = 10;
    let sats = 5;
    let ds = synth::build(SatelliteMode::NearHub { noise_std: 0.2 }, num_hubs, sats, 16, 0.2, 5, 0);
    let expected = num_hubs + num_hubs * sats;
    assert_eq!(ds.index.len(), expected, "expected {expected} entities in index");
}

#[test]
fn synth_ground_truth_includes_hub_and_satellites() {
    let ds = synth::build(SatelliteMode::CrossCluster, 5, 4, 16, 0.1, 10, 13);
    for gt in &ds.ground_truth {
        // Each ground truth list has 1 hub + 4 satellites = 5 total.
        assert_eq!(gt.len(), 5, "expected 5 ground truth ids, got {}", gt.len());
    }
}
