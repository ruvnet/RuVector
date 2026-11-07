//! Demonstrates `ruvector_sona::darwin_pareto` on a synthetic Darwin
//! evolutionary-search generation (ADR-335 WP31 / ADR-353 / 2026-10-16
//! nightly research: `docs/research/nightly/2026-10-16-darwin-pareto-frontier/`).
//!
//! The candidate shape mirrors `ruvector-sota-bench`'s HNSW tuning levers
//! (`ef_search`, `m`, `ef_construction` — see
//! `schemas/optimization-manifest-v1.json`'s `lever_name` enum): each
//! synthetic candidate reports `(recall_at_10, p95_latency_ms)`, the
//! two-objective shape `promotion.pareto.objectives` declares for this kind
//! of gate. This example does not run a real HNSW build — it demonstrates
//! the frontier-selection module's contract against representative,
//! hand-authored tradeoff data; the actual integration point
//! (`darwin.ts`'s `pareto` selection mode) is a documented next step, not
//! built this run (see the nightly research README's "Limitations").
//!
//! Run: `cargo run -p ruvector-sona --release --example darwin_pareto_frontier`

use ruvector_sona::darwin_pareto::{
    crowding_distance, is_dominated_by_frontier, select_frontier, Direction,
};

struct Candidate {
    label: &'static str,
    ef_search: u32,
    recall_at_10: f64,
    p95_latency_ms: f64,
}

fn main() {
    // One Darwin generation: 4 candidates per generation (STEP 18's default
    // budget), each an HNSW `ef_search` perturbation of a parent config.
    let generation = [
        Candidate {
            label: "parent (ef_search=64)",
            ef_search: 64,
            recall_at_10: 0.956,
            p95_latency_ms: 4.10,
        },
        Candidate {
            label: "candidate (ef_search=48)",
            ef_search: 48,
            recall_at_10: 0.941,
            p95_latency_ms: 3.20,
        },
        Candidate {
            label: "candidate (ef_search=96)",
            ef_search: 96,
            recall_at_10: 0.971,
            p95_latency_ms: 5.65,
        },
        Candidate {
            label: "candidate (ef_search=32)",
            ef_search: 32,
            recall_at_10: 0.902,
            p95_latency_ms: 2.55,
        },
        // Strictly worse than the parent on both axes -> must not appear on
        // the frontier (a reward-hacking-style "regression that still
        // claims improvement" would show up here as a bug, not a feature).
        Candidate {
            label: "candidate (ef_search=64, warm cache bug)",
            ef_search: 64,
            recall_at_10: 0.930,
            p95_latency_ms: 4.80,
        },
    ];

    let objectives = [Direction::Maximize, Direction::Minimize]; // recall up, latency down
    let points: Vec<[f64; 2]> = generation
        .iter()
        .map(|c| [c.recall_at_10, c.p95_latency_ms])
        .collect();
    let point_refs: Vec<&[f64]> = points.iter().map(|p| p.as_slice()).collect();

    let frontier = select_frontier(&objectives, &point_refs);

    println!("Generation ({} candidates), objectives = [recall_at_10: maximize, p95_latency_ms: minimize]\n", generation.len());
    for (i, c) in generation.iter().enumerate() {
        let on_frontier = frontier.contains(&i);
        println!(
            "  [{i}] {:<42} ef_search={:<3} recall@10={:.3} p95={:.2}ms  {}",
            c.label,
            c.ef_search,
            c.recall_at_10,
            c.p95_latency_ms,
            if on_frontier {
                "<- ON FRONTIER"
            } else {
                "   dominated"
            }
        );
    }

    println!("\nFrontier indices (non-dominated, eligible for promotion): {frontier:?}");

    let frontier_rows: Vec<&[f64]> = frontier.iter().map(|&i| point_refs[i]).collect();
    let distances = crowding_distance(&objectives, &point_refs, &frontier);
    println!("\nCrowding distance within the frontier (diversity-preserving tie-break when");
    println!("more candidates clear the frontier than `maximum_promotions` allows):");
    for (slot, &idx) in frontier.iter().enumerate() {
        println!(
            "  [{idx}] {:<42} crowding_distance={:.4}",
            generation[idx].label, distances[slot]
        );
    }

    // The literal optimization-manifest-v1.json gate: would a NEW candidate
    // (e.g. the next generation's mutation) be promotable against the
    // frontier just computed?
    let next_gen_candidate = [0.965_f64, 3.90]; // better recall than parent, worse than ef_search=96, cheaper latency than ef_search=96
    let dominated = is_dominated_by_frontier(&objectives, &next_gen_candidate, &frontier_rows);
    println!(
        "\nNext-generation probe (recall=0.965, p95=3.90ms): dominated_by_frontier={dominated} -> {}",
        if dominated { "REJECT (require_non_dominated)" } else { "ELIGIBLE for promotion" }
    );
}
