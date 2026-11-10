//! Coherence-gated graph draft for speculative ANN — benchmark binary.
//!
//! Tests the open question left by `ruvector-speculative-ann`'s nightly
//! research (2026-07-27): does replacing the linear O(n) u8 draft scan with
//! a coherence-gated graph beam traversal reduce draft cost while keeping
//! recall within an acceptable band of the exact baseline?
//!
//! Usage: `cargo run --release -p ruvector-coherence-speculative-ann --bin benchmark`

use std::time::Instant;

use ruvector_coherence_speculative_ann::dataset::ClusteredDataset;
use ruvector_coherence_speculative_ann::graph::{DraftGraph, GraphConfig};
use ruvector_coherence_speculative_ann::linear_full::LinearFull;
use ruvector_coherence_speculative_ann::metrics::LatencyStats;
use ruvector_coherence_speculative_ann::search::{BaselineGraphDraft, CoherenceGatedGraphDraft};
use ruvector_coherence_speculative_ann::variants::GraphDraftSpeculative;
use ruvector_coherence_speculative_ann::{recall_at_k, AnnVariant, ScalarQuantizer};

// ─── Dataset parameters ────────────────────────────────────────────────────
const N_CLUSTERS: usize = 20;
const N_PER_CLUSTER: usize = 200; // 20 x 200 = 4000 total
const DIMS: usize = 64;
const STD_DEV: f32 = 0.15;
const N_QUERIES: usize = 300;
const K: usize = 10;
const SEED: u64 = 0xC0DE_CAFE_BABE_7777;

// Graph / draft parameters.
const M: usize = 16;
const M_LONGJUMP: usize = 6;
const EF: usize = 50; // draft candidate pool width (k' in speculative terms)

// ─── Acceptance thresholds (declared before running, per CLAUDE.md Step 10) ─
const MIN_COHERENCE_RECALL: f32 = 0.85;
const MAX_RECALL_GAP_VS_BASELINE_DRAFT: f32 = 0.02; // coherence recall >= baseline_draft - 2pp
const REQUIRE_COHERENCE_FASTER_THAN_LINEAR: bool = true;
// A "< baseline" draft-cost check alone is gameable by per-query pop-order
// noise (the gate can occasionally take a different, same-length path through
// the frontier). Require a reduction large enough to not be noise.
const MIN_MEANINGFUL_DRAFT_COST_REDUCTION: f64 = 0.10; // >= 10% fewer expansions/query

fn bench_variant(
    variant: &dyn AnnVariant,
    queries: &[Vec<f32>],
    gt: &[Vec<(usize, f32)>],
) -> (f32, LatencyStats) {
    let mut latencies_us = Vec::with_capacity(queries.len());
    let mut recalls = Vec::with_capacity(queries.len());

    for (qi, q) in queries.iter().enumerate() {
        let start = Instant::now();
        let hits = variant.search(q, K);
        latencies_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);

        let gt_hits: Vec<ruvector_coherence_speculative_ann::Hit> = gt[qi]
            .iter()
            .map(|&(id, dist)| ruvector_coherence_speculative_ann::Hit { id, dist })
            .collect();
        recalls.push(recall_at_k(&hits, &gt_hits, K));
    }

    let mean_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
    (mean_recall, LatencyStats::from_samples(&latencies_us))
}

fn main() {
    println!("=== Coherence-Gated Graph Draft for Speculative ANN ===");
    println!(
        "Dataset: {N_CLUSTERS} clusters x {N_PER_CLUSTER} = {} vectors, dims={DIMS}, std={STD_DEV}",
        N_CLUSTERS * N_PER_CLUSTER
    );
    println!("Queries: {N_QUERIES}, k={K}, ef(draft pool)={EF}, seed=0x{SEED:X}");
    println!();

    eprintln!("[bench] generating clustered dataset + queries...");
    let ds = ClusteredDataset::generate(N_CLUSTERS, N_PER_CLUSTER, DIMS, STD_DEV, N_QUERIES, SEED);

    eprintln!("[bench] computing brute-force ground truth...");
    let gt: Vec<Vec<(usize, f32)>> = ds
        .queries
        .iter()
        .map(|q| {
            ds.ground_truth(q, K)
                .into_iter()
                .map(|h| (h.id, h.dist))
                .collect()
        })
        .collect();

    eprintln!("[bench] training scalar quantizer...");
    let quantizer = ScalarQuantizer::train(&ds.corpus_vectors());

    eprintln!("[bench] building draft graph (M={M}, long-jump={M_LONGJUMP})...");
    let build_start = Instant::now();
    let _graph_for_timing = DraftGraph::build(
        ds.corpus.clone(),
        &quantizer,
        GraphConfig {
            m: M,
            m_longjump: M_LONGJUMP,
            dims: DIMS,
        },
    );
    let build_ms = build_start.elapsed().as_millis();
    eprintln!("[bench] graph built in {build_ms} ms (brute-force O(N^2 . D) index time).");
    println!(
        "Index construction time: {build_ms} ms (one-time cost, excluded from per-query latency)"
    );
    println!();

    // ── LinearFull: exact ground-truth baseline ──────────────────────────────
    let linear = LinearFull::new(ds.corpus.clone(), DIMS);
    let (linear_recall, linear_lat) = bench_variant(&linear, &ds.queries, &gt);

    // ── GraphDraftSpeculative<BaselineGraphDraft>: graph instead of linear scan ─
    let quantizer_b = ScalarQuantizer::train(&ds.corpus_vectors());
    let graph_b = DraftGraph::build(
        ds.corpus.clone(),
        &quantizer_b,
        GraphConfig {
            m: M,
            m_longjump: M_LONGJUMP,
            dims: DIMS,
        },
    );
    let baseline_draft =
        GraphDraftSpeculative::new(graph_b, quantizer_b, BaselineGraphDraft, EF, EF);
    let (base_draft_recall, base_draft_lat) = bench_variant(&baseline_draft, &ds.queries, &gt);
    let base_draft_cost = baseline_draft.last_draft_cost().unwrap_or(0.0);

    // ── GraphDraftSpeculative<CoherenceGatedGraphDraft>: threshold sweep ──────
    let thresholds = [
        0.0f32, 0.30, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
    ];
    let mut coherence_results = Vec::new();
    for &t in &thresholds {
        let quantizer_c = ScalarQuantizer::train(&ds.corpus_vectors());
        let graph_c = DraftGraph::build(
            ds.corpus.clone(),
            &quantizer_c,
            GraphConfig {
                m: M,
                m_longjump: M_LONGJUMP,
                dims: DIMS,
            },
        );
        let v = GraphDraftSpeculative::new(
            graph_c,
            quantizer_c,
            CoherenceGatedGraphDraft { threshold: t },
            EF,
            EF,
        );
        let (recall, lat) = bench_variant(&v, &ds.queries, &gt);
        let draft_cost = v.last_draft_cost().unwrap_or(0.0);
        coherence_results.push((t, recall, lat, draft_cost, v.memory_bytes()));
    }

    // ── Report ────────────────────────────────────────────────────────────
    println!(
        "{:<32} {:>9} {:>10} {:>9} {:>9} {:>12} {:>12}",
        "Variant", "Recall@10", "Mean(us)", "p50(us)", "p95(us)", "DraftCost", "Memory"
    );
    println!("{}", "-".repeat(100));
    println!(
        "{:<32} {:>9.3} {:>10.1} {:>9.1} {:>9.1} {:>12} {:>12}",
        "LinearFull",
        linear_recall,
        linear_lat.mean_us,
        linear_lat.p50_us,
        linear_lat.p95_us,
        "n/a (O(n))",
        fmt_bytes(linear.memory_bytes())
    );
    println!(
        "{:<32} {:>9.3} {:>10.1} {:>9.1} {:>9.1} {:>12.1} {:>12}",
        "GraphDraft(Baseline,ef=50)",
        base_draft_recall,
        base_draft_lat.mean_us,
        base_draft_lat.p50_us,
        base_draft_lat.p95_us,
        base_draft_cost,
        fmt_bytes(baseline_draft.memory_bytes())
    );
    for (t, recall, lat, cost, mem) in &coherence_results {
        println!(
            "{:<32} {:>9.3} {:>10.1} {:>9.1} {:>9.1} {:>12.1} {:>12}",
            format!("GraphDraft(Coherence,t={t:.2})"),
            recall,
            lat.mean_us,
            lat.p50_us,
            lat.p95_us,
            cost,
            fmt_bytes(*mem)
        );
    }
    println!();

    // ── Acceptance ────────────────────────────────────────────────────────
    // Pick the coherence-gated configuration with the lowest draft cost that
    // still clears the recall floor — this is the "promoted" configuration,
    // decided by the rule above, not by cherry-picking after the fact.
    let best = coherence_results
        .iter()
        .filter(|(_, recall, ..)| *recall >= MIN_COHERENCE_RECALL)
        .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

    let mut checks: Vec<(String, bool)> = Vec::new();
    match best {
        Some((t, recall, lat, cost, _)) => {
            checks.push((
                format!("coherence-gated recall@10 >= {MIN_COHERENCE_RECALL} (best: t={t:.2}, got {recall:.3})"),
                *recall >= MIN_COHERENCE_RECALL,
            ));
            checks.push((
                format!(
                    "coherence-gated recall within {MAX_RECALL_GAP_VS_BASELINE_DRAFT} of baseline graph draft (baseline {base_draft_recall:.3}, got {recall:.3})"
                ),
                *recall >= base_draft_recall - MAX_RECALL_GAP_VS_BASELINE_DRAFT,
            ));
            if REQUIRE_COHERENCE_FASTER_THAN_LINEAR {
                checks.push((
                    format!(
                        "coherence-gated mean latency < LinearFull mean latency ({:.1}us vs {:.1}us)",
                        lat.mean_us, linear_lat.mean_us
                    ),
                    lat.mean_us < linear_lat.mean_us,
                ));
            }
            let reduction = 1.0 - (*cost / base_draft_cost);
            checks.push((
                format!(
                    "coherence-gated draft cost reduced by >= {:.0}% vs baseline graph-draft ({cost:.1} vs {base_draft_cost:.1} expansions/query, actual reduction {:.1}%)",
                    MIN_MEANINGFUL_DRAFT_COST_REDUCTION * 100.0,
                    reduction * 100.0
                ),
                reduction >= MIN_MEANINGFUL_DRAFT_COST_REDUCTION,
            ));
        }
        None => {
            checks.push((
                format!(
                    "at least one coherence threshold clears recall@10 >= {MIN_COHERENCE_RECALL}"
                ),
                false,
            ));
        }
    }

    let mut all_pass = true;
    for (desc, pass) in &checks {
        let mark = if *pass { "PASS" } else { "FAIL" };
        println!("[{mark}] {desc}");
        all_pass &= pass;
    }

    println!();
    if all_pass {
        println!("ACCEPTANCE RESULT: ACCEPT");
    } else {
        println!("ACCEPTANCE RESULT: REJECT");
    }
}

fn fmt_bytes(b: usize) -> String {
    format!("{:.2}MB", b as f64 / (1024.0 * 1024.0))
}
