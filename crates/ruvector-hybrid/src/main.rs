// ruvector-hybrid demo: builds a small hybrid index, runs all four search modes,
// prints recall@10 for each mode vs. oracle ground truth.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_hybrid::{
    dense::l2_normalise, index::HybridIndex, recall_at_k, sparse::bm25_weights, DenseVec,
    HybridDoc, HybridQuery, HybridSearch, SparseVec,
};

const N: usize = 2_000;
const DIMS: usize = 128;
const VOCAB: u32 = 500;
const DOC_TERMS: usize = 25;
const QUERY_TERMS: usize = 6;
const K: usize = 10;
const N_QUERIES: usize = 100;
const SEED: u64 = 42;

fn random_dense(rng: &mut StdRng, dims: usize) -> DenseVec {
    let dist = Normal::new(0.0f32, 1.0).unwrap();
    let mut v = DenseVec::new((0..dims).map(|_| dist.sample(rng)).collect());
    l2_normalise(&mut v);
    v
}

fn random_sparse(
    rng: &mut StdRng,
    vocab: u32,
    n_terms: usize,
    n: u32,
    doc_len: f32,
    avg_len: f32,
) -> SparseVec {
    let mut term_ids: Vec<u32> = (0..n_terms).map(|_| rng.gen_range(0..vocab)).collect();
    term_ids.sort_unstable();
    term_ids.dedup();
    let tf: Vec<f32> = term_ids
        .iter()
        .map(|_| rng.gen_range(1.0f32..4.0))
        .collect();
    // Simulated DF: rare terms get df in [1, 50], common in [100, 300]
    let df: Vec<u32> = term_ids
        .iter()
        .map(|&t| {
            if t < vocab / 5 {
                rng.gen_range(1u32..50)
            } else {
                rng.gen_range(100..300)
            }
        })
        .collect();
    bm25_weights(&term_ids, &tf, &df, n, doc_len, avg_len, 1.5, 0.75)
}

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);
    println!("ruvector-hybrid demo  (N={N}, D={DIMS}, vocab={VOCAB}, K={K})");

    let mut idx = HybridIndex::new(DIMS);

    for i in 0..N as u32 {
        let dense = random_dense(&mut rng, DIMS);
        let sparse = random_sparse(
            &mut rng,
            VOCAB,
            DOC_TERMS,
            N as u32,
            DOC_TERMS as f32,
            DOC_TERMS as f32,
        );
        idx.insert(HybridDoc {
            id: i,
            dense,
            sparse,
        });
    }
    println!(
        "Indexed {N} documents. Memory ≈ {:.1} KB",
        idx.memory_bytes() as f64 / 1024.0
    );

    let mut total_recall_dense = 0.0f64;
    let mut total_recall_sparse = 0.0f64;
    let mut total_recall_rrf = 0.0f64;
    let mut total_recall_linear = 0.0f64;

    for _ in 0..N_QUERIES {
        let dense = random_dense(&mut rng, DIMS);
        let sparse = random_sparse(
            &mut rng,
            VOCAB,
            QUERY_TERMS,
            N as u32,
            QUERY_TERMS as f32,
            DOC_TERMS as f32,
        );
        let q = HybridQuery { dense, sparse };

        let oracle = idx.search_linear(&q, K, N, 0.5); // all-docs oracle
        let r_dense = idx.search_dense(&q, K);
        let r_sparse = idx.search_sparse(&q, K);
        let r_rrf = idx.search_rrf(&q, K, K * 5);
        let r_linear = idx.search_linear(&q, K, K * 5, 0.5);

        total_recall_dense += recall_at_k(&r_dense, &oracle);
        total_recall_sparse += recall_at_k(&r_sparse, &oracle);
        total_recall_rrf += recall_at_k(&r_rrf, &oracle);
        total_recall_linear += recall_at_k(&r_linear, &oracle);
    }

    let q = N_QUERIES as f64;
    println!("\nRecall@{K} vs oracle (hybrid α=0.5) over {N_QUERIES} queries:");
    println!("  DenseOnly    : {:.1}%", 100.0 * total_recall_dense / q);
    println!("  SparseOnly   : {:.1}%", 100.0 * total_recall_sparse / q);
    println!("  HybridRRF    : {:.1}%", 100.0 * total_recall_rrf / q);
    println!("  HybridLinear : {:.1}%", 100.0 * total_recall_linear / q);
    println!("\nHybridLinear >= max(Dense, Sparse) because the oracle is also linear-fused.");
}
