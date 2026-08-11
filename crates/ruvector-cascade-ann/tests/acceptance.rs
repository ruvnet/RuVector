//! Acceptance tests for graph-neighbour cascade ANN.
//!
//! All thresholds are based on actual measured performance (N=2000, D=64, k=10,
//! ef_mult=1, K_graph=16).  Tests are deterministic: same seed → same dataset.

use ruvector_cascade_ann::{
    dataset::generate_gaussian,
    recall_at_k,
    variants::{GraphNeighbourCascade, LinearFull, QuantizedCascade},
    AnnVariant,
};

const N: usize = 2_000;
const DIM: usize = 64;
const N_QUERIES: usize = 100;
const K: usize = 10;
const K_GRAPH: usize = 16;
const DELTA: f32 = 0.30;
const SEED_C: u64 = 0xABCD_EF01_2345_6789;
const SEED_Q: u64 = 0x9876_5432_10FE_DCBA;

fn build_all() -> (LinearFull, QuantizedCascade, GraphNeighbourCascade) {
    let corpus = generate_gaussian(N, DIM, SEED_C);
    let linear = LinearFull::build(&corpus);
    let qc = QuantizedCascade::build(&corpus, 1);
    let gnc = GraphNeighbourCascade::build(&corpus, K_GRAPH, 1, DELTA);
    (linear, qc, gnc)
}

#[test]
fn linear_full_achieves_perfect_recall() {
    let (linear, _, _) = build_all();
    let queries = generate_gaussian(N_QUERIES, DIM, SEED_Q);
    for q in &queries {
        let gt = linear.search(q, K);
        let result = linear.search(q, K);
        let recall = recall_at_k(&result, &gt, K);
        assert!(
            (recall - 1.0).abs() < 1e-6,
            "LinearFull recall should be 1.0, got {recall}"
        );
    }
}

#[test]
fn quantized_cascade_recall_above_floor() {
    let (linear, qc, _) = build_all();
    let queries = generate_gaussian(N_QUERIES, DIM, SEED_Q);
    let avg_recall: f32 = queries
        .iter()
        .map(|q| {
            let gt = linear.search(q, K);
            let result = qc.search(q, K);
            recall_at_k(&result, &gt, K)
        })
        .sum::<f32>()
        / N_QUERIES as f32;
    assert!(
        avg_recall >= 0.85,
        "QuantizedCascade recall should be ≥ 0.85, got {avg_recall}"
    );
}

#[test]
fn graph_cascade_beats_quantized_cascade() {
    let (linear, qc, gnc) = build_all();
    let queries = generate_gaussian(N_QUERIES, DIM, SEED_Q);

    let recall_qc: f32 = queries
        .iter()
        .map(|q| {
            let gt = linear.search(q, K);
            let result = qc.search(q, K);
            recall_at_k(&result, &gt, K)
        })
        .sum::<f32>()
        / N_QUERIES as f32;

    let recall_gnc: f32 = queries
        .iter()
        .map(|q| {
            let gt = linear.search(q, K);
            let result = gnc.search(q, K);
            recall_at_k(&result, &gt, K)
        })
        .sum::<f32>()
        / N_QUERIES as f32;

    assert!(
        recall_gnc >= recall_qc,
        "GraphNeighbourCascade ({recall_gnc:.4}) must be ≥ QuantizedCascade ({recall_qc:.4})"
    );
}

#[test]
fn graph_cascade_recall_above_target() {
    let (linear, _, gnc) = build_all();
    let queries = generate_gaussian(N_QUERIES, DIM, SEED_Q);
    let avg: f32 = queries
        .iter()
        .map(|q| {
            let gt = linear.search(q, K);
            let result = gnc.search(q, K);
            recall_at_k(&result, &gt, K)
        })
        .sum::<f32>()
        / N_QUERIES as f32;
    assert!(avg >= 0.90, "GraphNeighbourCascade recall should be ≥ 0.90, got {avg}");
}

#[test]
fn memory_overhead_is_bounded() {
    let corpus = generate_gaussian(N, DIM, SEED_C);
    let qc = QuantizedCascade::build(&corpus, 1);
    let gnc = GraphNeighbourCascade::build(&corpus, K_GRAPH, 1, DELTA);
    // Graph adds at most K_GRAPH * 4 bytes per vector on top of QC.
    let expected_graph_overhead = N * K_GRAPH * 4;
    let actual_overhead = gnc.memory_bytes().saturating_sub(qc.memory_bytes());
    assert!(
        actual_overhead <= expected_graph_overhead + 1024,
        "Graph overhead {actual_overhead}B exceeds budget {expected_graph_overhead}B"
    );
}

#[test]
fn search_returns_k_results() {
    let corpus = generate_gaussian(100, DIM, SEED_C);
    let gnc = GraphNeighbourCascade::build(&corpus, 8, 1, 0.30);
    let query = generate_gaussian(1, DIM, SEED_Q).remove(0);
    let hits = gnc.search(&query, K);
    assert_eq!(hits.len(), K.min(100));
}
