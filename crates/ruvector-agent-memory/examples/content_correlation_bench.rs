//! Nightly research benchmark (2026-11-01, ADR-350): content-correlation
//! augmented arbitration.
//!
//! Hypothesis (fixed before this run; see
//! docs/research/nightly/2026-11-01-content-correlation-arbitration/README.md):
//!
//! Given a retrieved memory set containing an astroturf cluster (N
//! parentless memories whose content is near-identical, with no declared
//! causal parent — the exact blind spot `tests/arbitration.rs`'s
//! `content_copying_yields_no_correlation_downgrade_under_default_reliability`
//! locks in),
//!
//! when `arbitrate_with_content_correlation` (candidate A: `HardMerge`,
//! candidate B: `SoftDiscount{0.5}`) is applied instead of baseline
//! `arbitrate` (declared ancestry only),
//!
//! then the astroturf cluster's lineage count must drop by at least 70%
//! relative to baseline, while a disjoint set of genuinely independent
//! memories (orthogonal embeddings) must see 0% false merges,
//!
//! subject to: the added pairwise-similarity pass costing under 50ms at
//! N=500 total memories on this hardware (release build), and the
//! downgrade-only invariant holding throughout.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example content_correlation_bench

use std::collections::BTreeMap;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    arbitrate, arbitrate_with_content_correlation, ArbitrationConfig, AtomicObservation,
    CausalEpisodicGraph, ContentCorrelationConfig, CorrelationMode, NodeRef, ObservationSource,
    ReliabilityModel, SourceKind, Tenant,
};
use rvf_types::Ed25519Keypair;

const SIMILARITY_THRESHOLD: f32 = 0.95;
const EMBED_DIM: usize = 16;

fn source(id: &str) -> ObservationSource {
    ObservationSource {
        kind: SourceKind::AgentObservation,
        id: id.to_string(),
        public_key: [0u8; 32],
    }
}

fn signed(
    keypair: &Ed25519Keypair,
    time_ns: u64,
    confidence: f32,
    payload: &[u8],
) -> AtomicObservation {
    AtomicObservation::new_signed(
        source("agent"),
        time_ns,
        confidence,
        Tenant::new("acme"),
        vec![],
        payload.to_vec(),
        keypair,
    )
    .unwrap()
}

fn config() -> ArbitrationConfig {
    ArbitrationConfig {
        now_ns: 1,
        half_life_ns: u64::MAX,
        sufficiency_threshold: 1,
        reliability: ReliabilityModel::default(),
    }
}

/// `n` parentless memories whose embeddings cluster tightly around one
/// random center (cosine similarity > 0.999 for every pair) — the astroturf
/// shape: repeated content, no declared derivation.
fn astroturf_fixture(
    keypair: &Ed25519Keypair,
    graph: &mut CausalEpisodicGraph,
    n: usize,
    rng: &mut StdRng,
) -> (Vec<NodeRef>, BTreeMap<NodeRef, Vec<f32>>) {
    astroturf_fixture_tagged(keypair, graph, n, rng, 0)
}

/// `tag` disambiguates content addressing when two fixtures are ingested
/// into the same graph (scenario 3): without it, identical
/// (source, time, confidence, tenant, parents, payload) tuples across calls
/// collide on `ObservationId` and `ingest` rejects the duplicate.
fn astroturf_fixture_tagged(
    keypair: &Ed25519Keypair,
    graph: &mut CausalEpisodicGraph,
    n: usize,
    rng: &mut StdRng,
    tag: u8,
) -> (Vec<NodeRef>, BTreeMap<NodeRef, Vec<f32>>) {
    let center: Vec<f32> = (0..EMBED_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut nodes = Vec::with_capacity(n);
    let mut embeddings = BTreeMap::new();
    for i in 0..n {
        let mut payload = (i as u32).to_le_bytes().to_vec();
        payload.push(tag);
        let id = graph
            .ingest(signed(keypair, i as u64, 0.9, &payload))
            .unwrap();
        let node = NodeRef::Observation(id);
        let v: Vec<f32> = center
            .iter()
            .map(|x| x + rng.gen_range(-0.001..0.001))
            .collect();
        embeddings.insert(node, v);
        nodes.push(node);
    }
    (nodes, embeddings)
}

/// `n` parentless memories with genuinely independent embeddings (random,
/// high-dimensional — expected pairwise cosine similarity near 0).
fn independent_fixture(
    keypair: &Ed25519Keypair,
    graph: &mut CausalEpisodicGraph,
    n: usize,
    rng: &mut StdRng,
) -> (Vec<NodeRef>, BTreeMap<NodeRef, Vec<f32>>) {
    independent_fixture_tagged(keypair, graph, n, rng, 0)
}

/// See [`astroturf_fixture_tagged`] for why `tag` exists.
fn independent_fixture_tagged(
    keypair: &Ed25519Keypair,
    graph: &mut CausalEpisodicGraph,
    n: usize,
    rng: &mut StdRng,
    tag: u8,
) -> (Vec<NodeRef>, BTreeMap<NodeRef, Vec<f32>>) {
    let mut nodes = Vec::with_capacity(n);
    let mut embeddings = BTreeMap::new();
    for i in 0..n {
        let mut payload = (i as u32).to_le_bytes().to_vec();
        payload.push(tag);
        let id = graph
            .ingest(signed(keypair, i as u64, 0.9, &payload))
            .unwrap();
        let node = NodeRef::Observation(id);
        let v: Vec<f32> = (0..EMBED_DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
        embeddings.insert(node, v);
        nodes.push(node);
    }
    (nodes, embeddings)
}

fn hard_merge() -> ContentCorrelationConfig {
    ContentCorrelationConfig {
        similarity_threshold: SIMILARITY_THRESHOLD,
        mode: CorrelationMode::HardMerge,
    }
}

fn soft_discount() -> ContentCorrelationConfig {
    ContentCorrelationConfig {
        similarity_threshold: SIMILARITY_THRESHOLD,
        mode: CorrelationMode::SoftDiscount { strength: 0.5 },
    }
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((xs.len() as f64 - 1.0) * p).round() as usize;
    xs[idx]
}

fn main() {
    println!("=== Content-correlation-augmented arbitration — nightly bench (2026-11-01) ===\n");

    let keypair = Ed25519Keypair::generate(&mut rand::rngs::OsRng);
    let mut accept = true;

    // ── Scenario 1: astroturf detection rate ────────────────────────────────
    println!("--- Scenario 1: astroturf cluster (N=20, no declared parent) ---");
    let n_astroturf = 20usize;
    let mut rng = StdRng::seed_from_u64(0xA57A0);
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let (nodes, embeddings) = astroturf_fixture(&keypair, &mut graph, n_astroturf, &mut rng);

    let baseline = arbitrate(&graph, &nodes, &config()).unwrap();
    let baseline_lineages = baseline.verdict().independent_lineages();

    let hard =
        arbitrate_with_content_correlation(&graph, &nodes, &embeddings, &config(), &hard_merge())
            .unwrap();
    let hard_lineages = hard.verdict().independent_lineages();

    let soft = arbitrate_with_content_correlation(
        &graph,
        &nodes,
        &embeddings,
        &config(),
        &soft_discount(),
    )
    .unwrap();

    let reduction = 1.0 - (hard_lineages as f64 / baseline_lineages as f64);
    println!(
        "baseline (declared-only):  lineages={baseline_lineages:<4} naive={:.4} arbitrated={:.4}",
        baseline.verdict().naive_confidence,
        baseline.verdict().arbitrated_confidence
    );
    println!(
        "candidate A (HardMerge):   lineages={hard_lineages:<4} naive={:.4} arbitrated={:.4}  (lineage reduction: {:.1}%)",
        hard.verdict().naive_confidence,
        hard.verdict().arbitrated_confidence,
        reduction * 100.0
    );
    println!(
        "candidate B (SoftDiscount):lineages={:<4} naive={:.4} arbitrated={:.4}",
        soft.verdict().independent_lineages(),
        soft.verdict().naive_confidence,
        soft.verdict().arbitrated_confidence
    );
    let astroturf_pass = reduction >= 0.70;
    println!(
        "ACCEPTANCE (astroturf lineage reduction >= 70%): {}\n",
        if astroturf_pass { "PASS" } else { "FAIL" }
    );
    accept &= astroturf_pass;

    // ── Scenario 2: independent evidence, false-merge rate ─────────────────
    println!("--- Scenario 2: independent evidence (N=20, orthogonal content) ---");
    let n_indep = 20usize;
    let mut rng2 = StdRng::seed_from_u64(0x1D0E57);
    let mut graph2 = CausalEpisodicGraph::new(Tenant::new("acme"));
    let (nodes2, embeddings2) = independent_fixture(&keypair, &mut graph2, n_indep, &mut rng2);

    let baseline2 = arbitrate(&graph2, &nodes2, &config()).unwrap();
    let hard2 = arbitrate_with_content_correlation(
        &graph2,
        &nodes2,
        &embeddings2,
        &config(),
        &hard_merge(),
    )
    .unwrap();
    let false_merges =
        baseline2.verdict().independent_lineages() - hard2.verdict().independent_lineages();
    let false_merge_rate = false_merges as f64 / baseline2.verdict().independent_lineages() as f64;
    println!(
        "baseline lineages={} -> candidate A lineages={} (false merges: {false_merges}, rate: {:.1}%)",
        baseline2.verdict().independent_lineages(),
        hard2.verdict().independent_lineages(),
        false_merge_rate * 100.0
    );
    let false_merge_pass = false_merges == 0;
    println!(
        "ACCEPTANCE (false-merge rate == 0%): {}\n",
        if false_merge_pass { "PASS" } else { "FAIL" }
    );
    accept &= false_merge_pass;

    // ── Scenario 3: mixed — astroturf + independent in the same call ───────
    println!("--- Scenario 3: mixed (astroturf cluster + independent witnesses, same call) ---");
    let mut rng3 = StdRng::seed_from_u64(0x33D33D);
    let mut graph3 = CausalEpisodicGraph::new(Tenant::new("acme"));
    let (mut nodes3, mut embeddings3) =
        astroturf_fixture_tagged(&keypair, &mut graph3, 10, &mut rng3, 0);
    let (more_nodes, more_embeddings) =
        independent_fixture_tagged(&keypair, &mut graph3, 10, &mut rng3, 1);
    nodes3.extend(more_nodes);
    embeddings3.extend(more_embeddings);

    let baseline3 = arbitrate(&graph3, &nodes3, &config()).unwrap();
    let hard3 = arbitrate_with_content_correlation(
        &graph3,
        &nodes3,
        &embeddings3,
        &config(),
        &hard_merge(),
    )
    .unwrap();
    println!(
        "baseline lineages={} -> candidate A lineages={} (expect ~11: 1 merged astroturf lineage + 10 independent)",
        baseline3.verdict().independent_lineages(),
        hard3.verdict().independent_lineages()
    );
    let mixed_pass = hard3.verdict().independent_lineages() == 11;
    println!(
        "ACCEPTANCE (mixed scenario collapses to exactly 11 lineages): {}\n",
        if mixed_pass { "PASS" } else { "FAIL" }
    );
    accept &= mixed_pass;

    // ── Scenario 4: latency scaling (HardMerge, astroturf worst case) ──────
    println!("--- Scenario 4: latency scaling (HardMerge, worst case L≈N, 20 reps/size) ---");
    println!(
        "{:<8}{:>12}{:>12}{:>12}",
        "N", "mean(ms)", "p50(ms)", "p95(ms)"
    );
    let sizes = [20usize, 50, 100, 200, 500];
    let mut overhead_at_500_ms = 0.0;
    for &n in &sizes {
        let mut durations = Vec::with_capacity(20);
        for rep in 0..20u64 {
            let mut rng_s = StdRng::seed_from_u64(0x0005_CA1E_0000 + n as u64 * 100 + rep);
            let mut g = CausalEpisodicGraph::new(Tenant::new("acme"));
            let (ns, es) = astroturf_fixture(&keypair, &mut g, n, &mut rng_s);
            let t0 = Instant::now();
            let _ =
                arbitrate_with_content_correlation(&g, &ns, &es, &config(), &hard_merge()).unwrap();
            durations.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean: f64 = durations.iter().sum::<f64>() / durations.len() as f64;
        let p50 = percentile(durations.clone(), 0.50);
        let p95 = percentile(durations.clone(), 0.95);
        println!("{n:<8}{mean:>12.3}{p50:>12.3}{p95:>12.3}");
        if n == 500 {
            overhead_at_500_ms = p95;
        }
    }
    let latency_pass = overhead_at_500_ms < 50.0;
    println!(
        "ACCEPTANCE (p95 latency at N=500 < 50ms): {} (measured: {:.3}ms)\n",
        if latency_pass { "PASS" } else { "FAIL" },
        overhead_at_500_ms
    );
    accept &= latency_pass;

    println!(
        "=== Overall acceptance: {} ===",
        if accept { "ACCEPT" } else { "REJECT" }
    );
}
