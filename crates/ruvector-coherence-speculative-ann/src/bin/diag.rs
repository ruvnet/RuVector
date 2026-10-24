//! Reproducibility tool for the "no usable middle ground" finding in the
//! 2026-10-21 nightly research write-up: traces per-query pops/expansions
//! for the baseline (ungated) vs coherence-gated graph draft directly,
//! bypassing the `AnnVariant` wrapper, to show exactly where the gate starts
//! to fire and why low thresholds (<=0.45) have zero measurable effect.
//!
//! Usage: `cargo run --release -p ruvector-coherence-speculative-ann --bin diag`
use ruvector_coherence_speculative_ann::dataset::ClusteredDataset;
use ruvector_coherence_speculative_ann::graph::{DraftGraph, GraphConfig};
use ruvector_coherence_speculative_ann::search::{
    BaselineGraphDraft, CoherenceGatedGraphDraft, DraftSearcher,
};
use ruvector_coherence_speculative_ann::ScalarQuantizer;

fn main() {
    let ds = ClusteredDataset::generate(20, 200, 64, 0.15, 20, 0xC0DE_CAFE_BABE_7777u64);
    let quantizer = ScalarQuantizer::train(&ds.corpus_vectors());
    let graph = DraftGraph::build(
        ds.corpus.clone(),
        &quantizer,
        GraphConfig {
            m: 16,
            m_longjump: 6,
            dims: 64,
        },
    );

    let baseline = BaselineGraphDraft;
    let gated_00 = CoherenceGatedGraphDraft { threshold: 0.0 };
    let gated_45 = CoherenceGatedGraphDraft { threshold: 0.45 };
    let gated_80 = CoherenceGatedGraphDraft { threshold: 0.80 };

    for qi in 0..10 {
        let query = &ds.queries[qi];
        let qu8 = quantizer.quantize(query);
        let b = baseline.draft(&graph, query, &qu8, 50, 50, 0);
        let g0 = gated_00.draft(&graph, query, &qu8, 50, 50, 0);
        let g45 = gated_45.draft(&graph, query, &qu8, 50, 50, 0);
        let g80 = gated_80.draft(&graph, query, &qu8, 50, 50, 0);
        println!(
            "q{qi}: baseline(pops={},exp={}) t0.00(pops={},exp={}) t0.45(pops={},exp={}) t0.80(pops={},exp={})",
            b.pops, b.expansions,
            g0.pops, g0.expansions,
            g45.pops, g45.expansions,
            g80.pops, g80.expansions,
        );
    }
}
