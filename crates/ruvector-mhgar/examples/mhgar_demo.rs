//! Small interactive demo of MHGAR retrieval.
//!
//! Builds a 20-hub × 5-satellite graph (total 120 entities, D=16),
//! runs a single query against all three variants, and prints results.
//!
//! cargo run --release -p ruvector-mhgar --example mhgar_demo

use ruvector_mhgar::{
    recall_at_k,
    retriever::{CoherenceGatedHopper, OneHopExpander, Retriever, VectorOnlyRetriever},
    synth::{self, SatelliteMode},
};

fn main() {
    let ds = synth::build(SatelliteMode::CrossCluster, 20, 5, 16, 0.1, 10, 7);
    let query = &ds.queries[0];
    let gt = &ds.ground_truth[0];
    let k = 10;

    println!("Query ground truth ids: {:?}", &gt[..gt.len().min(k)]);
    println!();

    let v1 = VectorOnlyRetriever { index: &ds.index };
    let v2 = OneHopExpander {
        index: &ds.index, graph: &ds.graph,
        initial_k: k, num_seeds_to_expand: 1, hop_discount: 0.5,
    };
    let v3 = CoherenceGatedHopper {
        index: &ds.index,
        graph: &ds.graph,
        initial_k: k * 2,
        num_seeds_to_expand: 1,
        max_hops: 3,
        expansion_threshold: 0.50,
        hop_discount: 0.5,
    };

    let gt_ids: Vec<u32> = gt.iter().copied().take(k).collect();

    for ret in &[
        (&v1 as &dyn Retriever, "VectorOnly"),
        (&v2 as &dyn Retriever, "OneHopExpander"),
        (&v3 as &dyn Retriever, "CoherenceGatedHopper"),
    ] {
        let results = ret.0.search(query, k).expect("search failed");
        let recall = recall_at_k(&results, &gt_ids);
        let ids: Vec<u32> = results.iter().map(|h| h.id).collect();
        println!(
            "{:<25} recall={:.2}  ids={:?}",
            ret.1, recall, &ids[..ids.len().min(10)]
        );
    }
}
