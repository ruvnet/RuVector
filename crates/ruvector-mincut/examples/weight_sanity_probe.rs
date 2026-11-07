//! Throwaway diagnostic (not part of the shipped research artifact): does
//! `RuVectorGraphAnalyzer::min_cut()` / `MinCutWrapper` respond to edge
//! *weight*, or only to edge *count*? Two weight-1 triangles joined by a
//! single weight-100 bridge: an edge-count-only (unweighted) engine still
//! picks the 1-edge bridge cut (cheapest by count); a weight-aware engine
//! must instead pick the 2-edge, weight-2 cut that isolates a single
//! triangle vertex, since severing the weight-100 bridge costs far more.
use ruvector_mincut::{DynamicGraph, RuVectorGraphAnalyzer, StaticCutAnalyzer};
use std::sync::Arc;

fn main() {
    let graph = Arc::new(DynamicGraph::new());
    for (u, v) in [(0u64, 1u64), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
        let _ = graph.insert_edge(u, v, 1.0);
    }
    let _ = graph.insert_edge(2, 3, 100.0);

    let mut baseline = RuVectorGraphAnalyzer::new(Arc::clone(&graph));
    println!(
        "baseline min_cut = {} (edge-count-only engine would say 1; weight-aware engine would say 2)",
        baseline.min_cut()
    );
    println!("baseline partition = {:?}", baseline.partition());

    let candidate = StaticCutAnalyzer::new(Arc::clone(&graph));
    println!("candidate min_cut = {}", candidate.min_cut());
    println!("candidate partition = {:?}", candidate.partition());
}
