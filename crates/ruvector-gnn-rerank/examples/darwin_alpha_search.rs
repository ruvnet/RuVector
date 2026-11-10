//! Nightly research 2026-10-06: Darwin-evolved `GnnMincutReranker` hyperparameters.
//!
//! The 2026-05-21 nightly hand-set `GnnMincutReranker`'s (alpha,
//! coherence_threshold, k_graph) = (0.60, 0.50, 8) and explicitly flagged
//! "optimal alpha calibration for production embeddings is unknown" /
//! "adaptive alpha tuning via a ruFlo feedback loop" as unimplemented future
//! work. This run builds the first bounded evolutionary search engine in the
//! repository (`ruvector-darwin`, introduced alongside this experiment) and
//! applies it to exactly that gap.
//!
//! ## Anti-overfitting design
//!
//! The 100 deterministic queries from the original benchmark's generator are
//! split into a FITNESS set (first 70, used as the evolutionary search's
//! fitness function) and a disjoint HELD-OUT set (last 30, used only once, at
//! the end, to score the promoted candidate and the baseline). The search
//! never sees the held-out set. This is the acceptance test: an improvement
//! on the fitness set that does not transfer to the held-out set is
//! overfitting, not a real improvement, and must be reported as such.
//!
//! Corpus/query/noise generation and constants are intentionally identical to
//! `src/main.rs` (same N, DIM, N_CLUSTERS, NOISE_SIGMA, SEED) so this run's
//! baseline numbers are directly comparable to the 2026-05-21 nightly's
//! published ones.
//!
//! Run:
//!   cargo run --release -p ruvector-gnn-rerank --example darwin_alpha_search

use std::collections::HashSet;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use ruvector_darwin::{run_evolution, Bound, Evaluation, EvolutionConfig, Genome};
use ruvector_gnn_rerank::{Candidate, CandidateReranker, GnnMincutReranker, RankedResult};

// ── configuration (identical to src/main.rs) ──────────────────────────────────

const N: usize = 5_000;
const DIM: usize = 128;
const N_CLUSTERS: usize = 20;
const CLUSTER_SIGMA: f32 = 0.5;
const N_QUERIES: usize = 100;
const K: usize = 10;
const RETRIEVAL_K: usize = 80;
const NOISE_SIGMA: f32 = 0.40;
const SEED: u64 = 42;

const N_FITNESS: usize = 70;
// Remaining N_QUERIES - N_FITNESS = 30 queries form the held-out set.

// ── data generation (duplicated from src/main.rs; kept self-contained so this
// example has no dependency on the binary target) ────────────────────────────

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

fn gen_queries(corpus: &[Vec<f32>], n_queries: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_queries)
        .map(|_| {
            let base = &corpus[rng.gen_range(0..corpus.len())];
            base.iter()
                .map(|&x| x + rng.gen_range(-0.1_f32..0.1))
                .collect()
        })
        .collect()
}

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn exact_topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> HashSet<usize> {
    let mut dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2sq(query, v)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

fn noisy_retrieve(
    query: &[f32],
    corpus: &[Vec<f32>],
    retrieval_k: usize,
    noise_sigma: f32,
    rng: &mut StdRng,
) -> Vec<Candidate> {
    let noise = Normal::new(0.0_f32, noise_sigma).unwrap();
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let true_l2 = l2sq(query, v).sqrt();
            (i, -true_l2 + noise.sample(rng))
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
        .into_iter()
        .take(retrieval_k)
        .map(|(id, noisy_score)| Candidate {
            id: id as u32,
            vector: corpus[id].clone(),
            noisy_score,
        })
        .collect()
}

fn recall_at_k(results: &[RankedResult], gt: &HashSet<usize>) -> f64 {
    results
        .iter()
        .filter(|r| gt.contains(&(r.id as usize)))
        .count() as f64
        / gt.len() as f64
}

// ── genome <-> reranker mapping ───────────────────────────────────────────────

/// genes = [alpha, coherence_threshold, k_graph (rounded to nearest usize, >= 1)]
fn genome_to_reranker(g: &Genome) -> GnnMincutReranker {
    GnnMincutReranker {
        alpha: g.genes[0] as f32,
        coherence_threshold: g.genes[1] as f32,
        k_graph: (g.genes[2].round() as usize).max(1),
    }
}

/// Mean recall@K and mean per-query latency (µs) of `reranker` over the
/// queries/candidates/ground-truth selected by `idx`.
fn evaluate_subset(
    reranker: &GnnMincutReranker,
    queries: &[Vec<f32>],
    cands_per_query: &[Vec<Candidate>],
    ground_truth: &[HashSet<usize>],
    idx: std::ops::Range<usize>,
) -> (f64, f64) {
    let mut recalls = Vec::with_capacity(idx.len());
    let mut lat_us = Vec::with_capacity(idx.len());
    for qi in idx {
        let t0 = Instant::now();
        let results = reranker
            .rerank(&queries[qi], &cands_per_query[qi], K)
            .expect("rerank failed on well-formed synthetic input");
        lat_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);
        recalls.push(recall_at_k(&results, &ground_truth[qi]));
    }
    let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
    let mean_lat = lat_us.iter().sum::<f64>() / lat_us.len() as f64;
    (mean_recall, mean_lat)
}

fn main() {
    println!("=== Darwin-evolved GnnMincutReranker hyperparameters ===");
    println!("N={N} DIM={DIM} clusters={N_CLUSTERS} queries={N_QUERIES} (fitness={N_FITNESS}, held-out={}) K={K} retrieval_k={RETRIEVAL_K} noise_sigma={NOISE_SIGMA} seed={SEED}", N_QUERIES - N_FITNESS);

    let corpus = gen_corpus(N, DIM, N_CLUSTERS, SEED);
    let queries = gen_queries(&corpus, N_QUERIES, SEED + 1);
    let ground_truth: Vec<HashSet<usize>> =
        queries.iter().map(|q| exact_topk(q, &corpus, K)).collect();
    let mut noise_rng = StdRng::seed_from_u64(SEED + 99);
    let cands_per_query: Vec<Vec<Candidate>> = queries
        .iter()
        .map(|q| noisy_retrieve(q, &corpus, RETRIEVAL_K, NOISE_SIGMA, &mut noise_rng))
        .collect();

    let fitness_range = 0..N_FITNESS;
    let heldout_range = N_FITNESS..N_QUERIES;

    // ── baseline (hand-set defaults from the 2026-05-21 nightly) ─────────────
    let baseline = GnnMincutReranker::default(); // alpha=0.60, coherence_threshold=0.50, k_graph=8
    let (baseline_fit_recall, baseline_fit_lat) = evaluate_subset(
        &baseline,
        &queries,
        &cands_per_query,
        &ground_truth,
        fitness_range.clone(),
    );
    let (baseline_held_recall, baseline_held_lat) = evaluate_subset(
        &baseline,
        &queries,
        &cands_per_query,
        &ground_truth,
        heldout_range.clone(),
    );
    println!("\nBaseline (alpha=0.60, coherence_threshold=0.50, k_graph=8):");
    println!(
        "  fitness-set : recall@{K}={:.2}%  mean_lat={:.1}us",
        baseline_fit_recall * 100.0,
        baseline_fit_lat
    );
    println!(
        "  held-out    : recall@{K}={:.2}%  mean_lat={:.1}us",
        baseline_held_recall * 100.0,
        baseline_held_lat
    );

    // ── bounded evolutionary search (fitness-set ONLY) ────────────────────────
    let parent = Genome::new(vec![0.60, 0.50, 8.0]);
    let bounds = vec![
        Bound::new(0.05, 0.95), // alpha
        Bound::new(0.0, 0.95),  // coherence_threshold
        Bound::new(4.0, 20.0),  // k_graph
    ];
    let config = EvolutionConfig {
        generations: 3,
        candidates_per_generation: 4,
        max_promotions: 1,
        mutation_sigma_frac: 0.20,
        seed: SEED,
    };

    let run_search = || {
        run_evolution(
            parent.clone(),
            bounds.clone(),
            config.clone(),
            |g: &Genome| {
                let reranker = genome_to_reranker(g);
                let (recall, _lat) = evaluate_subset(
                    &reranker,
                    &queries,
                    &cands_per_query,
                    &ground_truth,
                    fitness_range.clone(),
                );
                if recall.is_finite() {
                    Evaluation::Fitness(recall)
                } else {
                    Evaluation::Rejected(format!("non-finite recall: {recall}"))
                }
            },
        )
    };

    let report = run_search();
    // Replay verification (Step 17): re-run the identical search and confirm
    // byte-identical lineage, since this engine's determinism is the basis
    // for treating its evidence as reproducible.
    let report_replay = run_search();
    let replay_verified = report.to_json_pretty() == report_replay.to_json_pretty();

    println!(
        "\nEvolutionary search: {} generations x {} candidates/gen (seed={})",
        config.generations, config.candidates_per_generation, config.seed
    );
    for gen in &report.generations {
        let improved = gen
            .improved_at
            .map(|i| format!("candidate {i} improved running best"))
            .unwrap_or_else(|| "no improvement".to_string());
        println!(
            "  gen {}: running_best_fitness(fitness-set recall)={:.4}  ({improved})",
            gen.index, gen.running_best_fitness
        );
    }
    println!(
        "Replay verified (two independent runs, same seed, identical lineage): {replay_verified}"
    );

    // ── final acceptance: evaluate the promoted candidate on the HELD-OUT set ─
    let (final_recall_held, final_lat_held, final_recall_fit, promoted_genes): (
        f64,
        f64,
        f64,
        Option<Vec<f64>>,
    ) = match &report.promoted {
        Some(candidate) => {
            let reranker = genome_to_reranker(&candidate.genome);
            let (held_recall, held_lat) = evaluate_subset(
                &reranker,
                &queries,
                &cands_per_query,
                &ground_truth,
                heldout_range.clone(),
            );
            let fit_recall = match &candidate.evaluation {
                Evaluation::Fitness(f) => *f,
                Evaluation::Rejected(_) => unreachable!("promoted candidate cannot be rejected"),
            };
            (
                held_recall,
                held_lat,
                fit_recall,
                Some(candidate.genome.genes.clone()),
            )
        }
        None => (
            baseline_held_recall,
            baseline_held_lat,
            baseline_fit_recall,
            None,
        ),
    };

    println!("\n--- Result ---");
    match &promoted_genes {
        Some(genes) => println!(
            "Promoted genome: alpha={:.4} coherence_threshold={:.4} k_graph={}",
            genes[0], genes[1], genes[2].round() as usize
        ),
        None => println!("No genome beat the parent within the search budget; parent (hand-set defaults) retained."),
    }
    println!(
        "Fitness-set recall@{K}: baseline={:.2}%  promoted={:.2}%  (delta {:+.2}pp)",
        baseline_fit_recall * 100.0,
        final_recall_fit * 100.0,
        (final_recall_fit - baseline_fit_recall) * 100.0
    );
    println!(
        "Held-out recall@{K}:    baseline={:.2}%  promoted={:.2}%  (delta {:+.2}pp)",
        baseline_held_recall * 100.0,
        final_recall_held * 100.0,
        (final_recall_held - baseline_held_recall) * 100.0
    );
    println!(
        "Held-out mean latency:  baseline={:.1}us  promoted={:.1}us  (ratio {:.2}x)",
        baseline_held_lat,
        final_lat_held,
        final_lat_held / baseline_held_lat
    );

    let held_out_delta_pp = (final_recall_held - baseline_held_recall) * 100.0;
    let latency_ratio = final_lat_held / baseline_held_lat;
    let overfit = report.promoted.is_some()
        && held_out_delta_pp < 0.5
        && (final_recall_fit - baseline_fit_recall) > 0.0;

    let acceptance = if !replay_verified {
        "INCONCLUSIVE (replay mismatch — search is not reproducible, evidence invalid)"
    } else if report.promoted.is_none() {
        "INCONCLUSIVE (hand-set defaults were not beaten within this bounded search budget/space)"
    } else if held_out_delta_pp < 0.0 {
        "REJECT (promoted genome overfit the fitness set: held-out recall regressed)"
    } else if latency_ratio > 1.5 {
        "REJECT (held-out recall improved but mean latency regressed more than 50%)"
    } else if overfit {
        "INCONCLUSIVE (fitness-set improved but held-out improvement below the 0.5pp acceptance threshold)"
    } else {
        "ACCEPT"
    };
    println!("\nAcceptance: {acceptance}");

    println!("\n--- Full lineage (JSON) ---");
    println!("{}", report.to_json_pretty());
}
