//! Nightly research 2026-10-28: filtered-retrieval reranker benchmark.
//!
//! Compares `NoisyScoreReranker` (baseline), `GnnMincutReranker` (existing
//! global-threshold coherence gate), and `AdaptiveCoherenceReranker` (new
//! per-node top-fraction coherence gate) when the first-stage candidate pool
//! has first been thinned by a metadata-style filter at varying selectivity.
//!
//! ## Methodology
//!
//! For each trial `t`, a single seeded RNG stream (seed = `BASE_SEED + t`)
//! deterministically produces: a query near a random cluster centroid, a
//! noisy first-stage candidate pool of size `RETRIEVAL_K` (true top-K by L2
//! distance, scored with independent Gaussian noise — same noise model as
//! `src/main.rs`), and one filter-admission draw `u_i ~ Uniform(0,1)` per
//! candidate. A selectivity level `s` keeps exactly the candidates with
//! `u_i < s`. Because the admission draws are generated once per trial and
//! reused across every `s`, the filtered pool at a lower selectivity is a
//! strict *subset* of the pool at a higher selectivity for the same trial —
//! this isolates the effect of selectivity from trial-to-trial query/noise
//! variance, which is a tighter control than redrawing independent filters
//! per selectivity level. All reranker variants see the identical filtered
//! pool within a trial (no variant-specific advantage from filter luck).
//!
//! Ground truth for recall@10 is `ExactL2Reranker`'s top-10 **on that same
//! filtered pool** (not the corpus-global top-10), since a restrictive
//! filter can legitimately remove the global top-k entirely — the fair
//! question is how close each cheap reranker gets to the best achievable
//! ordering of what the filter actually left behind.
//!
//! A trial is skipped for a given `s` (and the attempt counter keeps
//! advancing) if the filtered pool has fewer than `K` candidates; the
//! benchmark keeps drawing trials per `s` until `MIN_TRIALS` succeed,
//! reporting the exact count actually used.
//!
//! Run: `cargo run --release -p ruvector-gnn-rerank --bin filtered_rerank_bench`

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use ruvector_gnn_rerank::{
    AdaptiveCoherenceReranker, Candidate, CandidateReranker, ExactL2Reranker, GnnMincutReranker,
    NoisyScoreReranker,
};

// ── configuration ─────────────────────────────────────────────────────────────

const N: usize = 5_000;
const DIM: usize = 128;
const N_CLUSTERS: usize = 20;
const CLUSTER_SIGMA: f32 = 0.5;
const RETRIEVAL_K: usize = 300;
const NOISE_SIGMA: f32 = 0.40;
const K_GRAPH: usize = 8;
const K: usize = 10;
const MIN_TRIALS: usize = 200;
const MAX_ATTEMPTS: usize = 5_000;
const BASE_SEED: u64 = 777;
const CORPUS_SEED: u64 = 42;
const SELECTIVITIES: [f32; 5] = [1.00, 0.50, 0.25, 0.10, 0.05];

// ── data generation (mirrors src/main.rs) ────────────────────────────────────

fn gen_corpus(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-4.0_f32..4.0)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter()
                .map(|&x| x + rng.gen_range(-CLUSTER_SIGMA..CLUSTER_SIGMA))
                .collect()
        })
        .collect()
}

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// One trial's raw material: a query and its noisy first-stage pool, each
/// candidate carrying its own filter-admission draw `u_i`.
struct Trial {
    query: Vec<f32>,
    pool: Vec<Candidate>,
    admission: Vec<f32>,
}

fn gen_trial(corpus: &[Vec<f32>], seed: u64) -> Trial {
    let mut rng = StdRng::seed_from_u64(seed);

    // Query near (not on) a random cluster centroid, following src/main.rs's
    // gen_queries convention of jittering a real corpus point.
    let base = &corpus[rng.gen_range(0..corpus.len())];
    let query: Vec<f32> = base
        .iter()
        .map(|&x| x + rng.gen_range(-0.1_f32..0.1))
        .collect();

    let noise = Normal::new(0.0_f32, NOISE_SIGMA).unwrap();
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let true_l2 = l2sq(&query, v).sqrt();
            (i, -true_l2 + noise.sample(&mut rng))
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(RETRIEVAL_K);

    let pool: Vec<Candidate> = scored
        .iter()
        .map(|&(id, noisy_score)| Candidate {
            id: id as u32,
            vector: corpus[id].clone(),
            noisy_score,
        })
        .collect();

    // One admission draw per pool member, generated after the pool is fixed
    // so the pool's composition does not depend on how many draws are taken.
    let admission: Vec<f32> = (0..pool.len())
        .map(|_| rng.gen_range(0.0_f32..1.0))
        .collect();

    Trial {
        query,
        pool,
        admission,
    }
}

fn filter_pool(trial: &Trial, s: f32) -> Vec<Candidate> {
    trial
        .pool
        .iter()
        .zip(trial.admission.iter())
        .filter(|(_, &u)| u < s)
        .map(|(c, _)| Candidate {
            id: c.id,
            vector: c.vector.clone(),
            noisy_score: c.noisy_score,
        })
        .collect()
}

// ── stats helpers ─────────────────────────────────────────────────────────────

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn stderr(v: &[f64]) -> f64 {
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0).max(1.0);
    (var / v.len() as f64).sqrt()
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((values.len() as f64 * p / 100.0) as usize).min(values.len() - 1);
    values[idx]
}

struct VariantStats {
    name: &'static str,
    recalls: Vec<f64>,
    latencies_us: Vec<f64>,
}

impl VariantStats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            recalls: Vec::new(),
            latencies_us: Vec::new(),
        }
    }

    fn record<R: CandidateReranker>(
        &mut self,
        r: &R,
        query: &[f32],
        pool: &[Candidate],
        gt: &[u32],
    ) {
        let t0 = Instant::now();
        let ranked = r.rerank(query, pool, K).expect("rerank failed");
        let us = t0.elapsed().as_nanos() as f64 / 1_000.0;
        let hits = ranked.iter().filter(|r| gt.contains(&r.id)).count() as f64;
        self.recalls.push(hits / K as f64);
        self.latencies_us.push(us);
    }

    fn report(&self) -> (f64, f64, f64, f64) {
        let mean_recall = mean(&self.recalls);
        let se = stderr(&self.recalls);
        let mean_us = mean(&self.latencies_us);
        let p95_us = percentile(&mut self.latencies_us.clone(), 95.0);
        (mean_recall, se, mean_us, p95_us)
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("ruvector-gnn-rerank :: filtered_rerank_bench (nightly 2026-10-28)");
    println!(
        "corpus N={N} dim={DIM} clusters={N_CLUSTERS} retrieval_k={RETRIEVAL_K} \
         noise_sigma={NOISE_SIGMA} k_graph={K_GRAPH} k={K} min_trials={MIN_TRIALS} \
         base_seed={BASE_SEED} corpus_seed={CORPUS_SEED}"
    );

    let corpus = gen_corpus(N, DIM, N_CLUSTERS, CORPUS_SEED);

    let noisy_r = NoisyScoreReranker;
    let mincut_r = GnnMincutReranker::default();
    let adaptive_r = AdaptiveCoherenceReranker::default();
    let exact_r = ExactL2Reranker;

    println!(
        "\nGnnMincutReranker defaults: alpha={} coherence_threshold={} k_graph={}",
        mincut_r.alpha, mincut_r.coherence_threshold, mincut_r.k_graph
    );
    println!(
        "AdaptiveCoherenceReranker defaults: alpha={} keep_fraction={} min_keep={} k_graph={}",
        adaptive_r.alpha, adaptive_r.keep_fraction, adaptive_r.min_keep, adaptive_r.k_graph
    );

    println!(
        "\n{:<6} {:<14} {:>12} {:>10} {:>14} {:>14} {:>10}",
        "sel", "variant", "mean_recall%", "±stderr%", "mean_lat_us", "p95_lat_us", "n_trials"
    );
    println!("{}", "-".repeat(90));

    struct SelResult {
        s: f32,
        noisy: (f64, f64, f64, f64),
        mincut: (f64, f64, f64, f64),
        adaptive: (f64, f64, f64, f64),
        n: usize,
    }
    let mut all_results: Vec<SelResult> = Vec::new();

    for &s in &SELECTIVITIES {
        let mut st_noisy = VariantStats::new("NoisyScore");
        let mut st_mincut = VariantStats::new("GnnMincut");
        let mut st_adaptive = VariantStats::new("AdaptiveCoherence");
        let mut succeeded = 0usize;

        for t in 0..MAX_ATTEMPTS {
            if succeeded >= MIN_TRIALS {
                break;
            }
            let trial = gen_trial(&corpus, BASE_SEED + t as u64);
            let filtered = filter_pool(&trial, s);
            if filtered.len() < K {
                continue;
            }

            let gt_ranked = exact_r
                .rerank(&trial.query, &filtered, K)
                .expect("oracle rerank failed");
            let gt: Vec<u32> = gt_ranked.iter().map(|r| r.id).collect();

            st_noisy.record(&noisy_r, &trial.query, &filtered, &gt);
            st_mincut.record(&mincut_r, &trial.query, &filtered, &gt);
            st_adaptive.record(&adaptive_r, &trial.query, &filtered, &gt);
            succeeded += 1;
        }

        if succeeded < MIN_TRIALS {
            eprintln!(
                "WARNING: selectivity {s:.2} only reached {succeeded}/{MIN_TRIALS} trials \
                 within {MAX_ATTEMPTS} attempts"
            );
        }

        let rn = st_noisy.report();
        let rm = st_mincut.report();
        let ra = st_adaptive.report();

        println!(
            "{:<6.2} {:<14} {:>11.2}% {:>9.2}% {:>14.2} {:>14.2} {:>10}",
            s,
            "NoisyScore",
            rn.0 * 100.0,
            rn.1 * 100.0,
            rn.2,
            rn.3,
            succeeded
        );
        println!(
            "{:<6.2} {:<14} {:>11.2}% {:>9.2}% {:>14.2} {:>14.2} {:>10}",
            s,
            "GnnMincut",
            rm.0 * 100.0,
            rm.1 * 100.0,
            rm.2,
            rm.3,
            succeeded
        );
        println!(
            "{:<6.2} {:<14} {:>11.2}% {:>9.2}% {:>14.2} {:>14.2} {:>10}",
            s,
            "AdaptiveCoherence",
            ra.0 * 100.0,
            ra.1 * 100.0,
            ra.2,
            ra.3,
            succeeded
        );

        all_results.push(SelResult {
            s,
            noisy: rn,
            mincut: rm,
            adaptive: ra,
            n: succeeded,
        });
    }

    // ── acceptance check ──────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(90));
    println!("ACCEPTANCE CHECK");
    println!("{}", "=".repeat(90));

    let at_1_00 = all_results
        .iter()
        .find(|r| (r.s - 1.00).abs() < 1e-6)
        .unwrap();
    let low_sel: Vec<&SelResult> = all_results.iter().filter(|r| r.s <= 0.10 + 1e-6).collect();

    let unfiltered_regression_pp = (at_1_00.mincut.0 - at_1_00.adaptive.0) * 100.0;
    let cond_1_no_regression = unfiltered_regression_pp <= 0.5;
    println!(
        "1) s=1.00 regression vs GnnMincut: {:+.2} pp (budget <= 0.50 pp)  -> {}",
        -unfiltered_regression_pp,
        if cond_1_no_regression { "PASS" } else { "FAIL" }
    );

    for r in &low_sel {
        let gain_pp = (r.adaptive.0 - r.mincut.0) * 100.0;
        println!(
            "   s={:.2}: AdaptiveCoherence - GnnMincut recall@10 = {:+.2} pp (n={})",
            r.s, gain_pp, r.n
        );
    }
    let mean_low_sel_gain_pp: f64 = low_sel
        .iter()
        .map(|r| (r.adaptive.0 - r.mincut.0) * 100.0)
        .sum::<f64>()
        / low_sel.len() as f64;
    let cond_2_low_sel_gain = mean_low_sel_gain_pp >= 3.0;
    println!(
        "2) mean low-selectivity (s<=0.10) recall@10 gain: {mean_low_sel_gain_pp:+.2} pp \
         (threshold >= 3.00 pp) -> {}",
        if cond_2_low_sel_gain { "PASS" } else { "FAIL" }
    );

    let mut cond_3_latency_ok = true;
    for r in &all_results {
        let ratio = r.adaptive.2 / r.mincut.2.max(1e-9);
        if ratio > 2.0 {
            cond_3_latency_ok = false;
        }
        println!(
            "   s={:.2}: AdaptiveCoherence/GnnMincut mean latency ratio = {ratio:.2}x",
            r.s
        );
    }
    println!(
        "3) latency ratio <= 2.0x at every selectivity -> {}",
        if cond_3_latency_ok { "PASS" } else { "FAIL" }
    );

    println!("4) cargo test / clippy status: see CI output captured alongside this run (not checked by this binary).");

    let all_measured_pass = cond_1_no_regression && cond_2_low_sel_gain && cond_3_latency_ok;
    println!("\n{}", "=".repeat(90));
    if all_measured_pass {
        println!("RESULT (conditions 1-3, measured by this binary): ACCEPT-CANDIDATE");
    } else {
        println!("RESULT (conditions 1-3, measured by this binary): REJECT-CANDIDATE");
    }
    println!(
        "Final ACCEPT/REJECT/INCONCLUSIVE determination also requires cargo test + clippy \
         green (condition 4), evaluated outside this binary."
    );
}
