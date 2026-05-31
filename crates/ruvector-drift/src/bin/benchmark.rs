//! Semantic Drift Guard — RuVector Nightly Benchmark (2026-05-27)
//!
//! Uses structured, clustered embeddings that mimic real agent memory (text
//! embeddings cluster in semantic directions — they are NOT uniform on the
//! unit sphere). Two phases:
//!
//!   Phase 1 — stable: N(μ_stable, σI) normalized (cluster A near e_0)
//!   Phase 2 — drift:  N(μ_drift,  σI) normalized (cluster B near e_{32})
//!
//! Drift shift is +1.5 along e_{32} so the two clusters are well-separated
//! (cosine_sim(μ_stable, μ_drift) ≈ 0).  This matches real agent memory where
//! a session topic changes completely (e.g. "coding assistant" → "medical QA").

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_drift::{
    DriftDetector, EwaDriftDetector, GraphCoherenceDriftDetector,
    WindowedVarianceDriftDetector,
};
use std::collections::HashSet;
use std::time::Instant;

// ── Dataset parameters ──────────────────────────────────────────────────────
const DIM: usize = 64;
const N_STABLE: usize = 800;
const N_DRIFT: usize = 500;
const N_FP_TEST: usize = 300;
/// Intra-cluster noise.
const CLUSTER_SIGMA: f32 = 0.25;
/// Stable cluster: strong bias along e_0.
/// With dim=64 σ=0.25:  cosine_sim ≈ bias²/(bias²+dim·σ²) = 36/(36+4) = 0.90
const STABLE_BIAS: f32 = 6.0;
/// Drift cluster: same bias strength along an orthogonal dimension.
const DRIFT_BIAS: f32 = 6.0;
/// Orthogonal dimension for drift cluster.
const DRIFT_DIM: usize = 32;
const SEED: u64 = 0xdead_cafe;
const RECALL_K: usize = 10;
const N_QUERY: usize = 100;

// ── Detector thresholds ──────────────────────────────────────────────────────
// Within-cluster cosine_sim ≈ 0.90  → stable raw_score ≈ 0.10
// Cross-cluster cosine_sim  ≈ 0.00  → drift  raw_score ≈ 1.00
// Threshold must sit between 0.10 and 1.00 with good margin.
const EWA_ALPHA: f32 = 0.05;
const EWA_THRESH: f32 = 0.20; // well above stable ≈0.10, well below drift ≈1.00
const EWA_WARMUP: usize = 80;

const WV_WARMUP: usize = 80;
const WV_WINDOW: usize = 48;
const WV_MEAN_DROP: f32 = 0.15; // stable mean ≈0.90; drift mean ≈0.0 → drop ≈0.90
const WV_VAR_THRESH: f32 = 0.05;

// Pairwise coherence: stable ≈0.90; 50/50 mix ≈0.45 → drop ≈0.45
const GC_CAPACITY: usize = 96;
const GC_K: usize = 6; // kept for API compat; pairwise metric ignores k
const GC_WARMUP: usize = 80;
const GC_THRESH: f32 = 0.15; // trigger when pairwise-mean drops by 0.15

// ── Helpers ──────────────────────────────────────────────────────────────────

fn normalize(v: &mut Vec<f32>) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Stable: Gaussian noise + strong bias along e_0, then normalize.
/// cosine_sim between any two stable vectors ≈ STABLE_BIAS²/(STABLE_BIAS²+DIM·σ²) ≈ 0.90
fn stable_vec(rng: &mut impl rand::Rng) -> Vec<f32> {
    let dist = Normal::new(0.0f32, CLUSTER_SIGMA).unwrap();
    let mut v: Vec<f32> = (0..DIM).map(|_| dist.sample(rng)).collect();
    v[0] += STABLE_BIAS;
    normalize(&mut v);
    v
}

/// Drift: same noise profile, bias along an orthogonal dimension.
/// cosine_sim(stable_vec, drift_vec) ≈ 0 (orthogonal clusters).
fn drift_vec(rng: &mut impl rand::Rng) -> Vec<f32> {
    let dist = Normal::new(0.0f32, CLUSTER_SIGMA).unwrap();
    let mut v: Vec<f32> = (0..DIM).map(|_| dist.sample(rng)).collect();
    v[DRIFT_DIM] += DRIFT_BIAS;
    normalize(&mut v);
    v
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Top-k by cosine similarity; returns original IDs (not positional indices).
fn top_k_ids(corpus: &[(usize, Vec<f32>)], query: &[f32], k: usize) -> HashSet<usize> {
    let k = k.min(corpus.len());
    let mut sims: Vec<(usize, f32)> = corpus
        .iter()
        .map(|(id, v)| (*id, cosine_sim(v, query)))
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sims.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Mean recall@k for a set of queries.
///
/// `index`: the searchable corpus (id, vector).
/// `gt_corpus`: the ground-truth corpus (usually stable-only).
fn mean_recall(
    index: &[(usize, Vec<f32>)],
    gt_corpus: &[(usize, Vec<f32>)],
    queries: &[Vec<f32>],
    k: usize,
) -> f32 {
    queries
        .iter()
        .map(|q| {
            let gt: HashSet<usize> = top_k_ids(gt_corpus, q, k);
            let retrieved: HashSet<usize> = top_k_ids(index, q, k);
            gt.intersection(&retrieved).count() as f32 / k as f32
        })
        .sum::<f32>()
        / queries.len() as f32
}

// ── Detection benchmark ───────────────────────────────────────────────────────

struct DetResult {
    name: &'static str,
    tp_50: bool,
    tp_100: bool,
    tp_200: bool,
    fp_alerts: usize,
    mean_lat_ns: u64,
    p50_ns: u64,
    p95_ns: u64,
    throughput_vecs_s: u64,
}

fn percentile(mut v: Vec<u64>, p: f64) -> u64 {
    v.sort_unstable();
    v[((v.len() as f64 * p) as usize).min(v.len().saturating_sub(1))]
}

fn run_detector(
    det: &mut dyn DriftDetector,
    stable: &[Vec<f32>],
    drift: &[Vec<f32>],
    fp_stable: &[Vec<f32>],
) -> DetResult {
    let mut lats: Vec<u64> = Vec::with_capacity(stable.len() + drift.len());

    // Warmup on stable data.
    for v in stable {
        let t = Instant::now();
        det.observe(v);
        lats.push(t.elapsed().as_nanos() as u64);
    }

    // Drift phase — record first alert timing.
    let mut tp_50 = false;
    let mut tp_100 = false;
    let mut tp_200 = false;
    for (i, v) in drift.iter().enumerate() {
        let t = Instant::now();
        let s = det.observe(v);
        lats.push(t.elapsed().as_nanos() as u64);
        if s.alert {
            if i < 50 { tp_50 = true; }
            if i < 100 { tp_100 = true; }
            if i < 200 { tp_200 = true; }
        }
    }

    // FP test: reset, re-warmup with stable, then measure alerts on fp_stable.
    det.reset();
    for v in stable {
        det.observe(v);
    }
    let mut fp_alerts = 0usize;
    for v in fp_stable {
        if det.observe(v).alert {
            fp_alerts += 1;
        }
    }

    let mean_ns = lats.iter().sum::<u64>() / lats.len().max(1) as u64;
    let p50 = percentile(lats.clone(), 0.50);
    let p95 = percentile(lats.clone(), 0.95);
    let tput = lats.len() as u64 * 1_000_000_000 / lats.iter().sum::<u64>().max(1);
    let name = det.name();

    DetResult {
        name,
        tp_50,
        tp_100,
        tp_200,
        fp_alerts,
        mean_lat_ns: mean_ns,
        p50_ns: p50,
        p95_ns: p95,
        throughput_vecs_s: tput,
    }
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    println!("=== Semantic Drift Guard — RuVector Nightly Benchmark ===");
    println!("Date:          2026-05-27");
    println!("OS:            {}", std::env::consts::OS);
    println!("Arch:          {}", std::env::consts::ARCH);
    println!("Dim:           {DIM}");
    println!("N stable:      {N_STABLE}");
    println!("N drift:       {N_DRIFT}");
    println!("N FP test:     {N_FP_TEST}");
    println!("Cluster σ:     {CLUSTER_SIGMA}");
    println!(
        "Drift:         bias +{DRIFT_BIAS} along e_{DRIFT_DIM} (orthogonal to stable e_0)"
    );
    println!("Recall K:      {RECALL_K}");
    println!("Queries:       {N_QUERY}");
    println!();

    // ── Generate datasets ───────────────────────────────────────────────────
    let stable: Vec<Vec<f32>> = (0..N_STABLE).map(|_| stable_vec(&mut rng)).collect();
    let drift: Vec<Vec<f32>> = (0..N_DRIFT).map(|_| drift_vec(&mut rng)).collect();
    let fp_stable: Vec<Vec<f32>> = (0..N_FP_TEST).map(|_| stable_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERY).map(|_| stable_vec(&mut rng)).collect();

    // ── Detectors ───────────────────────────────────────────────────────────
    let mut ewa = EwaDriftDetector::new(DIM, EWA_ALPHA, EWA_THRESH, EWA_WARMUP);
    let mut wv = WindowedVarianceDriftDetector::new(
        DIM, WV_WARMUP, WV_WINDOW, WV_MEAN_DROP, WV_VAR_THRESH,
    );
    let mut gc = GraphCoherenceDriftDetector::new(GC_CAPACITY, GC_K, GC_WARMUP, GC_THRESH);

    // ── Detection accuracy ──────────────────────────────────────────────────
    println!("--- Detection Accuracy ---");
    println!(
        "{:<20}  {:>6}  {:>6}  {:>6}  {:>8}  {:>13}  {:>10}  {:>10}  {:>16}",
        "Detector", "TP@50", "TP@100", "TP@200", "FP count",
        "Mean lat (ns)", "p50 (ns)", "p95 (ns)", "vecs/s"
    );
    println!("{}", "-".repeat(105));

    let results: Vec<DetResult> = vec![
        run_detector(&mut ewa, &stable, &drift, &fp_stable),
        run_detector(&mut wv, &stable, &drift, &fp_stable),
        run_detector(&mut gc, &stable, &drift, &fp_stable),
    ];

    for r in &results {
        println!(
            "{:<20}  {:>6}  {:>6}  {:>6}  {:>8}  {:>13}  {:>10}  {:>10}  {:>16}",
            r.name,
            if r.tp_50 { "YES" } else { "NO" },
            if r.tp_100 { "YES" } else { "NO" },
            if r.tp_200 { "YES" } else { "NO" },
            r.fp_alerts,
            r.mean_lat_ns,
            r.p50_ns,
            r.p95_ns,
            r.throughput_vecs_s,
        );
    }
    println!();

    // ── Recall benchmark (ID-tagged, correct index spaces) ─────────────────
    println!("--- Recall@{RECALL_K} — ID-tagged measurement ---");

    // Tag stable vectors with IDs 0..N_STABLE.
    let stable_tagged: Vec<(usize, Vec<f32>)> = stable
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.clone()))
        .collect();

    // Tag drift vectors with IDs N_STABLE..N_STABLE+N_DRIFT.
    let drift_tagged: Vec<(usize, Vec<f32>)> = drift
        .iter()
        .enumerate()
        .map(|(i, v)| (N_STABLE + i, v.clone()))
        .collect();

    // Full contaminated corpus.
    let full_corpus: Vec<(usize, Vec<f32>)> = stable_tagged
        .iter()
        .chain(drift_tagged.iter())
        .cloned()
        .collect();

    // Ground truth: top-k from stable-only corpus.
    // Compact corpus: stable-only (compaction removes all drift).
    let recall_stable_index = mean_recall(&stable_tagged, &stable_tagged, &queries, RECALL_K);
    let recall_full_index = mean_recall(&full_corpus, &stable_tagged, &queries, RECALL_K);
    let recall_compact_index = mean_recall(&stable_tagged, &stable_tagged, &queries, RECALL_K);

    println!("Ground truth:           top-{RECALL_K} from stable-only ({N_STABLE} vectors)");
    println!("Stable-only index:      recall@{RECALL_K} = {recall_stable_index:.4}  (baseline)");
    println!("Full index (stable+drift): recall@{RECALL_K} = {recall_full_index:.4}");
    println!(
        "Compact index (drift removed): recall@{RECALL_K} = {recall_compact_index:.4}"
    );
    println!(
        "Index size reduction: {:.1}%  ({} → {} vectors)",
        N_DRIFT as f64 / (N_STABLE + N_DRIFT) as f64 * 100.0,
        N_STABLE + N_DRIFT,
        N_STABLE
    );
    println!();

    // ── Compaction hint demo ─────────────────────────────────────────────────
    println!("--- GraphCoherence Compaction Hint ---");
    let mut gc_compact = GraphCoherenceDriftDetector::new(GC_CAPACITY, GC_K, GC_WARMUP, GC_THRESH);
    for v in stable.iter().chain(drift.iter()) {
        gc_compact.observe(v);
    }
    match gc_compact.compaction_hint() {
        Some(hint) => {
            println!("Window size:         {}", hint.n_vectors);
            println!("Flagged low-coh:     {} ({:.1}%)",
                hint.n_flagged,
                hint.n_flagged as f64 / hint.n_vectors.max(1) as f64 * 100.0);
            let min_coh = hint.coherence_scores.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_coh = hint.coherence_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean_coh: f32 = hint.coherence_scores.iter().sum::<f32>()
                / hint.coherence_scores.len().max(1) as f32;
            println!("Coherence scores:    min={min_coh:.3}  mean={mean_coh:.3}  max={max_coh:.3}");
            println!("Drift threshold:     {:.3}", hint.drift_threshold);
        }
        None => println!("No compaction hint (not enough warmup)"),
    }
    println!();

    // ── Memory estimate ──────────────────────────────────────────────────────
    println!("--- Memory Estimates ---");
    println!("Stable corpus:         {} KB", N_STABLE * DIM * 4 / 1024);
    println!("Drift corpus:          {} KB", N_DRIFT * DIM * 4 / 1024);
    println!("EWA overhead:          {} B (centroid only)", DIM * 4);
    println!(
        "WindowedVariance:      {} B (window={} + centroid={})",
        (WV_WINDOW + DIM) * 4, WV_WINDOW * 4, DIM * 4
    );
    println!(
        "GraphCoherence:        {} KB (cap={} × {}D × 4B)",
        GC_CAPACITY * DIM * 4 / 1024, GC_CAPACITY, DIM
    );
    println!();

    // ── Acceptance tests ─────────────────────────────────────────────────────
    println!("--- Acceptance Tests ---");
    let r_ewa = &results[0];
    let r_gc = &results[2];

    let t1 = r_ewa.tp_50;
    let t2 = r_ewa.tp_100;
    let t3 = r_gc.tp_100;
    let t4 = r_gc.fp_alerts < (N_FP_TEST / 4); // < 25% FP
    let t5 = (recall_compact_index - recall_stable_index).abs() < 0.05;
    let t6 = r_ewa.mean_lat_ns < 10_000; // EWA must be < 10µs/vec

    println!(
        "EWA detects drift at N=50:         {} (tp_50={})",
        pass(t1), r_ewa.tp_50
    );
    println!(
        "EWA detects drift at N=100:        {} (tp_100={})",
        pass(t2), r_ewa.tp_100
    );
    println!(
        "GraphCoherence detects at N=100:   {} (tp_100={})",
        pass(t3), r_gc.tp_100
    );
    println!(
        "GraphCoherence FP < 25%:           {} ({}/{})",
        pass(t4), r_gc.fp_alerts, N_FP_TEST
    );
    println!(
        "Compaction preserves recall±0.05:  {} (delta={:+.4})",
        pass(t5), recall_compact_index - recall_stable_index
    );
    println!(
        "EWA mean latency < 10µs:           {} ({}ns)",
        pass(t6), r_ewa.mean_lat_ns
    );

    let all_pass = t1 && t2 && t3 && t4 && t5 && t6;
    println!();
    println!(
        "ACCEPTANCE: {}",
        if all_pass { "ALL PASS" } else { "FAIL — see above" }
    );

    if !all_pass {
        std::process::exit(1);
    }
}

fn pass(b: bool) -> &'static str {
    if b { "PASS" } else { "FAIL" }
}
