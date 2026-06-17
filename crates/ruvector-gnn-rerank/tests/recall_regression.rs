//! Recall regression guard for the GNN-diffusion reranker (productionizes #479).
//!
//! Reproduces the research regime (N=5000, D=128, σ_noise=0.40, seed=42) with a
//! frozen seed and asserts that `GnnDiffusionReranker` recovers meaningfully more
//! recall@10 than the no-rerank `NoisyScoreReranker` baseline. This turns the
//! benchmark's headline number (+10.4pp in #479) into a CI-runnable guard so the
//! win cannot silently regress. Runs under `cargo test` (no standalone bin).

use std::collections::HashSet;

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_gnn_rerank::{
    Candidate, CandidateReranker, GnnDiffusionReranker, NoisyScoreReranker,
};

const N: usize = 5_000;
const DIM: usize = 128;
const N_CLUSTERS: usize = 20;
const CLUSTER_SIGMA: f32 = 0.5;
const N_QUERIES: usize = 100;
const K: usize = 10;
const RETRIEVAL_K: usize = 80;
const NOISE_SIGMA: f32 = 0.40;
const SEED: u64 = 42;

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
            base.iter().map(|&x| x + rng.gen_range(-0.1_f32..0.1)).collect()
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
    rng: &mut StdRng,
) -> Vec<Candidate> {
    let noise = Normal::new(0.0_f32, NOISE_SIGMA).unwrap();
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, -l2sq(query, v).sqrt() + noise.sample(rng)))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
        .into_iter()
        .take(retrieval_k)
        .map(|(id, noisy_score)| Candidate { id: id as u32, vector: corpus[id].clone(), noisy_score })
        .collect()
}

fn recall_at_k(results: &[ruvector_gnn_rerank::RankedResult], gt: &HashSet<usize>, k: usize) -> f64 {
    let hits = results.iter().take(k).filter(|r| gt.contains(&(r.id as usize))).count();
    hits as f64 / gt.len().min(k) as f64
}

#[test]
fn gnn_diffusion_beats_noisy_baseline() {
    let corpus = gen_corpus(N, DIM, N_CLUSTERS, SEED);
    let queries = gen_queries(&corpus, N_QUERIES, SEED + 1);
    let mut rng = StdRng::seed_from_u64(SEED + 99);

    let noisy = NoisyScoreReranker;
    let gnn = GnnDiffusionReranker::default();

    let (mut noisy_sum, mut gnn_sum) = (0.0_f64, 0.0_f64);
    for q in &queries {
        let gt = exact_topk(q, &corpus, K);
        let cands = noisy_retrieve(q, &corpus, RETRIEVAL_K, &mut rng);
        let nr = noisy.rerank(q, &cands, K).expect("noisy rerank");
        let gr = gnn.rerank(q, &cands, K).expect("gnn rerank");
        noisy_sum += recall_at_k(&nr, &gt, K);
        gnn_sum += recall_at_k(&gr, &gt, K);
    }
    let noisy_recall = noisy_sum / N_QUERIES as f64;
    let gnn_recall = gnn_sum / N_QUERIES as f64;
    let delta = gnn_recall - noisy_recall;
    eprintln!("recall@{K}: noisy={noisy_recall:.3}  gnn={gnn_recall:.3}  delta={delta:+.3}");

    // Research measured +0.104; guard a conservative +0.03 so the win can't silently regress.
    assert!(
        delta >= 0.03,
        "GNN diffusion must beat the noisy baseline by >= 0.03 recall@{K}; got noisy={noisy_recall:.3} gnn={gnn_recall:.3} delta={delta:+.3}"
    );
}
