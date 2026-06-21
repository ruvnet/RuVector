//! Quick demo: temporal coherence decay for agent memory.
//!
//! Generates 1 000 memories, runs 20 queries, prints recall@10 for each variant.

use rand::SeedableRng;
use ruvector_temporal_coherence::{
    generate_memory_corpus, ground_truth_topk, recall_at_k, CoherenceGraph, CoherenceSearch,
    DecayConfig, FlatSearch, TemporalSearch, VectorSearch,
};

const N: usize = 1_000;
const DIMS: usize = 64;
const TIME_SPAN: u64 = 1_000_000;
const NUM_CLUSTERS: usize = 10;
const K: usize = 10;
const NUM_QUERIES: usize = 20;
const COHERENCE_THRESHOLD: f32 = 0.60;
const COHERENCE_WEIGHT: f32 = 0.30;
const HALF_LIFE: u64 = 300_000; // 30% of time_span

fn main() {
    println!("=== Temporal Coherence Decay — Agent Memory Demo ===");
    println!("Corpus : {N} memories, {DIMS}D, {NUM_CLUSTERS} clusters");
    println!(
        "Queries: {NUM_QUERIES}  K={K}  half_life={HALF_LIFE}  coherence_w={COHERENCE_WEIGHT}"
    );
    println!();

    let mut rng = rand::rngs::SmallRng::seed_from_u64(1337);
    let store = generate_memory_corpus(N, DIMS, TIME_SPAN, NUM_CLUSTERS, &mut rng);

    let now = TIME_SPAN; // query at end of time window
    let decay = DecayConfig::exponential(now, HALF_LIFE);
    let graph = CoherenceGraph::build(&store, COHERENCE_THRESHOLD);

    println!(
        "Coherence graph: {} nodes, {} edges, mean_gate={:.3}",
        graph.node_count(),
        graph.edge_count(),
        graph.mean_gate()
    );
    println!();

    use rand::distributions::{Distribution, Uniform};
    let uni = Uniform::new(-1.0f32, 1.0);

    let flat = FlatSearch;
    let temporal = TemporalSearch {
        decay: decay.clone(),
    };
    let coherence = CoherenceSearch::new(
        decay.clone(),
        CoherenceGraph::build(&store, COHERENCE_THRESHOLD),
        COHERENCE_WEIGHT,
    );

    let (mut total_flat, mut total_temp, mut total_coh) = (0.0f32, 0.0f32, 0.0f32);

    for q in 0..NUM_QUERIES {
        let query: Vec<f32> = (0..DIMS).map(|_| uni.sample(&mut rng)).collect();
        let gt = ground_truth_topk(&query, &store, K);

        let r_flat = flat.search(&query, K, &store);
        let r_temp = temporal.search(&query, K, &store);
        let r_coh = coherence.search(&query, K, &store);

        let rc_flat = recall_at_k(&r_flat.iter().map(|r| r.id).collect::<Vec<_>>(), &gt);
        let rc_temp = recall_at_k(&r_temp.iter().map(|r| r.id).collect::<Vec<_>>(), &gt);
        let rc_coh = recall_at_k(&r_coh.iter().map(|r| r.id).collect::<Vec<_>>(), &gt);

        println!(
            "Query {:02}: flat={:.3}  temporal={:.3}  coherence={:.3}",
            q, rc_flat, rc_temp, rc_coh
        );

        total_flat += rc_flat;
        total_temp += rc_temp;
        total_coh += rc_coh;
    }

    let n = NUM_QUERIES as f32;
    println!();
    println!("=== Mean recall@{K} ===");
    println!("  FlatSearch     : {:.3}", total_flat / n);
    println!("  TemporalSearch : {:.3}", total_temp / n);
    println!("  CoherenceSearch: {:.3}", total_coh / n);
    println!();
    println!("Note: temporal/coherence variants trade raw recall for recency/relevance.");
    println!("Ground truth is cosine-only; lower recall with temporal/coherence is expected");
    println!("when old similar memories exist — the point is retrieval *fitness*, not raw recall.");
}
