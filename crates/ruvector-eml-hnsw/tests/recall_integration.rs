//! End-to-end recall test — the validation PR #353 was missing.
//!
//! Builds an EmlHnsw on structured (skewed) 128-dim data, runs 100 queries,
//! and compares the top-10 against the brute-force full-cosine ground truth.
//!
//! The production pattern is "reduced index narrows candidates + exact re-rank
//! restores ordering" — so we assert the re-rank recall bar tightly and keep
//! the reduced-only bar loose (it is a candidate filter, not a final ranker).
//! On genuinely structured real data (e.g. SIFT1M) the reduced bar rises
//! substantially; that test is in `sift1m_real.rs` and gated behind env vars.

use ruvector_eml_hnsw::cosine_decomp::cosine_distance_f32;
use ruvector_eml_hnsw::hnsw_integration::{EmlHnsw, EmlMetric};

fn make_skewed(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // Deterministic LCG. Variance concentrated in first 32 dims so the
    // correlation-based selector has signal to find.
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / u32::MAX as f32) - 0.5;
            let scale = if d < 32 { 4.0 } else { 0.3 };
            v.push(u * scale);
        }
        out.push(v);
    }
    out
}

fn brute_force_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i + 1, cosine_distance_f32(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

fn recall_at_k(ground_truth: &[usize], got: &[usize], k: usize) -> f32 {
    let truth: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();
    let hits = got.iter().take(k).filter(|id| truth.contains(id)).count();
    hits as f32 / k as f32
}

#[test]
fn rerank_recall_meets_bar_on_structured_data() {
    const N: usize = 2_000;
    const DIM: usize = 128;
    const K: usize = 10;
    const QUERIES: usize = 100;
    const SELECTED_K: usize = 32;
    const FETCH_K: usize = 50;

    let corpus = make_skewed(N, DIM, 42);
    let queries = make_skewed(QUERIES, DIM, 1337);

    let train: Vec<Vec<f32>> = corpus.iter().take(500).cloned().collect();
    let mut idx = EmlHnsw::train_and_build(
        &train,
        SELECTED_K,
        EmlMetric::Cosine,
        N + 16,
        16,
        200,
    )
    .expect("build succeeds on structured data");
    idx.add_batch(&corpus);
    assert_eq!(idx.len(), N);
    assert_eq!(idx.reduced_dim(), SELECTED_K);

    let mut reduced_sum = 0.0f32;
    let mut rerank_sum = 0.0f32;

    for q in &queries {
        let truth = brute_force_top_k(&corpus, q, K);
        let reduced: Vec<usize> = idx.search(q, K, 64).into_iter().map(|h| h.id).collect();
        let reranked: Vec<usize> = idx
            .search_with_rerank(q, K, FETCH_K, 64)
            .into_iter()
            .map(|h| h.id)
            .collect();
        reduced_sum += recall_at_k(&truth, &reduced, K);
        rerank_sum += recall_at_k(&truth, &reranked, K);
    }
    let reduced_recall = reduced_sum / QUERIES as f32;
    let rerank_recall = rerank_sum / QUERIES as f32;

    eprintln!("reduced recall@10 = {reduced_recall:.4}");
    eprintln!("rerank recall@10  = {rerank_recall:.4}  (fetch_k={FETCH_K})");

    // Reduced-dim cosine is a candidate filter, not a final ranker. The bar
    // only needs to show the filter has signal (i.e. is beating random).
    // Random top-10 of 2000 = 10/2000 = 0.5%, so anything above 10% proves
    // the filter finds relevant candidates in the fetch window.
    assert!(
        reduced_recall > 0.10,
        "reduced recall@10 = {reduced_recall:.3} — no better than random"
    );
    // Re-rank restores near-exact ordering by paying O(fetch_k) full-dim
    // work. This is the production bar.
    assert!(
        rerank_recall >= 0.80,
        "rerank recall@10 = {rerank_recall:.3} < 0.80 — rerank is not recovering truth"
    );
}

#[test]
fn selector_top1_matches_brute_force() {
    let data = make_skewed(500, 128, 77);
    let mut model = ruvector_eml_hnsw::EmlDistanceModel::new(128, 32);
    for chunk in data.chunks(2) {
        if chunk.len() < 2 {
            break;
        }
        let d = cosine_distance_f32(&chunk[0], &chunk[1]);
        model.record(&chunk[0], &chunk[1], d);
    }
    model.train();

    let q = &data[100];
    let truth = brute_force_top_k(&data, q, 1)[0];

    let mut scored: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i + 1, model.selected_distance(q, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    assert_eq!(
        scored[0].0, truth,
        "selected-distance top-1 should equal brute-force top-1 (self-query)"
    );
}

#[test]
fn projection_and_full_distance_are_both_available() {
    // Smoke test that the public API advertised in README / lib.rs actually
    // exports: project, selected_distance, and the HNSW integration type.
    use ruvector_eml_hnsw::{
        cosine_distance_selected, project_vector, sq_euclidean_selected, EmlHnsw, EmlMetric,
    };
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![2.0f32, 2.0, 4.0, 4.0];
    let dims = [0, 2];
    let cos = cosine_distance_selected(&a, &b, &dims);
    let sq = sq_euclidean_selected(&a, &b, &dims);
    let p = project_vector(&a, &dims);
    assert_eq!(p, vec![1.0, 3.0]);
    assert!(cos.is_finite() && (0.0..=2.0).contains(&cos));
    assert!(sq.is_finite() && sq >= 0.0);
    // Existence check for EmlHnsw enum variant
    let _ = EmlMetric::Cosine;
    // Type name exists; no-op.
    let _: Option<EmlHnsw> = None;
}
