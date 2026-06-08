//! WorldGraph condensation demo — RuView `WorldGraph -> condense -> OccWorld`.
//!
//! Run: `cargo run -p ruvector-graph-condense --example worldgraph`
//!
//! RuView (github.com/ruvnet/RuView) records `WorldGraph` snapshots — a stream
//! of spatial-occupancy observations from WiFi CSI sensing — and feeds them to
//! an OccWorld world-model retrainer. A day of sensing is millions of
//! observations; training on all of them on an edge device is impractical.
//!
//! This example simulates a small "day" of WorldGraph observations as a feature
//! graph (observations = nodes with occupancy embeddings + an activity label;
//! edges = spatial-temporal adjacency, heavy inside an activity, light across
//! transitions) and condenses it into a handful of **event summaries** — exactly
//! the `EventSummary { embedding, confidence, ... }` shape from the design brief,
//! realised as [`CondensedNode`]s with provenance.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_graph_condense::{
    condense, evaluate_full, CondenseConfig, CondenseMethod, DiffCutConfig, GraphCondenser,
    NodeFeatures,
};
use ruvector_mincut::DynamicGraph;

/// A simulated "day": `num_events` activities, each spanning `obs_per_event`
/// consecutive observations, joined by light transition edges.
struct DaySim {
    num_events: usize,
    obs_per_event: usize,
    num_activities: usize,
    dim: usize,
    seed: u64,
}

impl DaySim {
    #[allow(clippy::needless_range_loop)] // `e` is the event index, used widely
    fn generate(&self) -> (DynamicGraph, NodeFeatures, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let g = DynamicGraph::new();
        let mut feats = NodeFeatures::new(self.dim, self.num_activities);
        let mut true_event = Vec::new(); // ground-truth event id per observation

        // Each event gets a distinct occupancy centroid and an activity label.
        let centroids: Vec<Vec<f32>> = (0..self.num_events)
            .map(|e| {
                let mut c = vec![0f32; self.dim];
                c[e % self.dim] = 5.0 + (e / self.dim) as f32 * 5.0;
                c
            })
            .collect();
        let activity_of = |e: usize| e % self.num_activities;

        let mut id = 0u64;
        let mut first_of_event = Vec::new();
        for e in 0..self.num_events {
            first_of_event.push(id);
            let prev_first = id;
            for i in 0..self.obs_per_event {
                let mut emb = centroids[e].clone();
                for x in &mut emb {
                    *x += rng.gen_range(-0.4..0.4);
                }
                feats.set(id, emb, activity_of(e)).unwrap();
                true_event.push(e);
                // Temporal chain inside the event (heavy edges).
                if i > 0 {
                    let _ = g.insert_edge(id - 1, id, 1.0);
                }
                // Dense intra-event co-occurrence: link to a few random earlier
                // observations of the same event, so each event is a coherent
                // community (not a thin chain).
                let links = i.min(3);
                for _ in 0..links {
                    let other = prev_first + rng.gen_range(0..i as u64);
                    let _ = g.insert_edge(other, id, 1.0);
                }
                id += 1;
            }
        }
        // Light transition edges between consecutive events (person moves zones).
        for e in 1..self.num_events {
            let a = first_of_event[e] - 1; // last obs of previous event
            let b = first_of_event[e]; // first obs of this event
            let _ = g.insert_edge(a, b, 0.1);
        }
        (g, feats, true_event)
    }
}

fn report(title: &str, g: &DynamicGraph, condensed: &ruvector_graph_condense::CondensedGraph) {
    let m = evaluate_full(g, condensed);
    println!("\n=== {title} ===");
    println!(
        "  observations (nodes): {}  ->  condensed events: {}   ({:.1}x reduction)",
        m.source_nodes, m.condensed_nodes, m.node_reduction_ratio
    );
    println!(
        "  edges: {} -> {} ({:.1}x)   intra-weight kept: {:.1}%   mean coherence: {:.2}",
        m.source_edges,
        m.condensed_edges,
        m.edge_reduction_ratio,
        m.intra_weight_ratio * 100.0,
        m.mean_coherence
    );
    println!(
        "  activity purity: {:.1}%   cut inflation: {}",
        m.label_purity * 100.0,
        m.cut_inflation
            .map(|c| format!("{c:.3} (1.0 = global cut preserved)"))
            .unwrap_or_else(|| "n/a".into())
    );
    println!("  event summaries (CondensedNode == EventSummary):");
    for n in condensed.nodes.iter().take(6) {
        println!(
            "    event {:>2}: {:>3} obs | representative=obs#{:<3} | activity={:?} | confidence(coherence)={:.2}",
            n.id,
            n.weight,
            n.representative,
            n.dominant_class(),
            n.coherence
        );
    }
    if condensed.nodes.len() > 6 {
        println!("    ... ({} more)", condensed.nodes.len() - 6);
    }
}

fn main() {
    let day = DaySim {
        num_events: 12,
        obs_per_event: 50,
        num_activities: 4,
        dim: 8,
        seed: 2026,
    };
    let (graph, features, _truth) = day.generate();
    println!(
        "Simulated WorldGraph: {} observations across {} events, {} edges.",
        graph.num_vertices(),
        day.num_events,
        graph.num_edges()
    );

    // 1) Default structure-preserving condensation (weak-boundary) — the
    //    recommended pipeline for a full-day, many-event WorldGraph.
    let weak = condense(&graph, &features).expect("condense");
    report("WeakBoundary (default)", &graph, &weak);
    println!(
        "  -> a day of {} observations becomes {} deployable event summaries \
         (the artifact OccWorld would retrain on).",
        graph.num_vertices(),
        weak.node_count()
    );

    // 2) Trained differentiable min-cut on the SAME large-K WorldGraph. With
    //    Adam + warm-start init (default) it now recovers all 12 events — the
    //    optimisation work that made the trained method viable at scale.
    let diff = GraphCondenser::new(CondenseConfig {
        method: CondenseMethod::DiffMinCut(DiffCutConfig {
            num_clusters: day.num_events,
            ..Default::default()
        }),
        normalize_centroids: false,
    })
    .condense(&graph, &features)
    .expect("diff condense");
    report("DiffMinCut (trained, Adam + warm-start)", &graph, &diff);
    println!(
        "\nBoth methods recover the day's events; DiffMinCut now scales to large K \
         via Adam + warm-start (it refines the WeakBoundary prior with the \
         differentiable normalized-cut objective)."
    );
}
