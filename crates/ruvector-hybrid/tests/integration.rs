//! Integration tests for ruvector-hybrid.
//!
//! Two dataset regimes:
//! - Random (generate): tokens independent of vectors. Used for structural/API
//!   tests. HybridRRF recall is expected to be lower than DenseFlat because
//!   sparse noise disrupts dense ranking — this is correct behaviour when
//!   lexical and semantic signals are uncorrelated.
//! - Structured (generate_structured): cluster-aligned tokens. Both signals
//!   point to the same documents. Used for recall acceptance tests.

use ruvector_hybrid::{
    dataset::{generate, generate_structured, DatasetConfig},
    recall_at_k, DenseFlat, HybridRrf, HybridSearch, SparseBm25,
};

fn small_cfg() -> DatasetConfig {
    DatasetConfig {
        n_docs: 500,
        dim: 32,
        vocab_size: 100,
        tokens_per_doc: 10,
        n_queries: 50,
        seed: 7,
        k: 10,
    }
}

fn structured_cfg() -> DatasetConfig {
    // Topic model: vocab_size = n_topics (topic IDs are the tokens).
    // Each doc samples k_doc=3 topics; each query samples k_query=2 topics.
    // With 5000 docs and 32 topics: docs sharing the same 2 query topics have
    // cosine ≈ sqrt(2/3) ≈ 0.82 and share both query tokens → high BM25.
    // Docs sharing only 1 topic have cosine ≈ 0.41 and lower BM25.
    // RRF fuses the two signals — both agree on relevance → high recall.
    DatasetConfig {
        n_docs: 5_000,
        dim: 64,
        vocab_size: 32,    // 32 latent topics
        tokens_per_doc: 3, // not used by generate_structured but kept consistent
        n_queries: 100,
        seed: 17,
        k: 10,
    }
}

fn build_and_insert<I: HybridSearch>(
    mut idx: I,
    cfg: &DatasetConfig,
) -> (I, Vec<ruvector_hybrid::dataset::Query>) {
    let (docs, queries) = generate(cfg);
    for d in &docs {
        idx.insert(d.id, &d.vector, &d.tokens);
    }
    (idx, queries)
}

fn build_and_insert_structured<I: HybridSearch>(
    mut idx: I,
    cfg: &DatasetConfig,
) -> (I, Vec<ruvector_hybrid::dataset::Query>) {
    let (docs, queries) = generate_structured(cfg);
    for d in &docs {
        idx.insert(d.id, &d.vector, &d.tokens);
    }
    (idx, queries)
}

// ─── Dense recall (random data) ──────────────────────────────────────────────

#[test]
fn dense_flat_perfect_recall_random() {
    let cfg = small_cfg();
    let (idx, queries) = build_and_insert(DenseFlat::new(cfg.dim), &cfg);
    let total_recall: f32 = queries
        .iter()
        .map(|q| recall_at_k(&idx.search(&q.vector, &q.tokens, cfg.k), &q.ground_truth))
        .sum();
    let mean = total_recall / queries.len() as f32;
    assert!(
        mean >= 0.999,
        "DenseFlat (exact scan) must have 100% recall; got {:.1}%",
        mean * 100.0
    );
}

// ─── Recall on structured (correlated) data ───────────────────────────────────

#[test]
fn hybrid_rrf_recall_structured_above_threshold() {
    let cfg = structured_cfg();
    let (idx, queries) = build_and_insert_structured(HybridRrf::new(cfg.dim), &cfg);
    let total_recall: f32 = queries
        .iter()
        .map(|q| recall_at_k(&idx.search(&q.vector, &q.tokens, cfg.k), &q.ground_truth))
        .sum();
    let mean = total_recall / queries.len() as f32;
    // With the topic model: ~30 docs share both query topics with equal BM25 score.
    // BM25 ranks them randomly → RRF does not perfectly preserve dense order within
    // the topic group.  60% is well above SparseBm25-alone (~33%) and far above the
    // random-data baseline (~30% despite uncorrelated tokens).  The gap between 60%
    // and 100% is explained in the research document and motivates future work on
    // score-weighted fusion.
    assert!(
        mean >= 0.60,
        "HybridRrf on structured data must reach ≥ 60% recall; got {:.1}%",
        mean * 100.0
    );
}

#[test]
fn dense_flat_perfect_recall_structured() {
    let cfg = structured_cfg();
    let (idx, queries) = build_and_insert_structured(DenseFlat::new(cfg.dim), &cfg);
    let total_recall: f32 = queries
        .iter()
        .map(|q| recall_at_k(&idx.search(&q.vector, &q.tokens, cfg.k), &q.ground_truth))
        .sum();
    let mean = total_recall / queries.len() as f32;
    assert!(
        mean >= 0.999,
        "DenseFlat (exact scan) on structured data must have 100% recall; got {:.1}%",
        mean * 100.0
    );
}

// ─── BM25 structural tests ────────────────────────────────────────────────────

#[test]
fn sparse_bm25_returns_relevant_docs() {
    let mut idx = SparseBm25::new();
    idx.insert(0, &[], &[42, 43]);
    idx.insert(1, &[], &[10, 11]);
    idx.insert(2, &[], &[42, 44]);
    let results = idx.search(&[], &[42], 10);
    let ids: Vec<u32> = results.iter().map(|r| r.id).collect();
    assert!(ids.contains(&0), "doc 0 should match token 42");
    assert!(ids.contains(&2), "doc 2 should match token 42");
    assert!(!ids.contains(&1), "doc 1 must NOT match token 42");
}

// ─── API / structural tests (random data fine) ───────────────────────────────

#[test]
fn hybrid_rrf_len_matches_inserts() {
    let cfg = small_cfg();
    let (idx, _) = build_and_insert(HybridRrf::new(cfg.dim), &cfg);
    assert_eq!(idx.len(), cfg.n_docs);
}

#[test]
fn hybrid_rrf_memory_exceeds_dense_alone() {
    let cfg = small_cfg();
    let (hybrid, _) = build_and_insert(HybridRrf::new(cfg.dim), &cfg);
    let (dense, _) = build_and_insert(DenseFlat::new(cfg.dim), &cfg);
    assert!(
        hybrid.memory_bytes() > dense.memory_bytes(),
        "HybridRrf must use more memory than DenseFlat alone"
    );
}

#[test]
fn search_returns_k_or_fewer_results() {
    let cfg = small_cfg();
    let (idx, queries) = build_and_insert(HybridRrf::new(cfg.dim), &cfg);
    for q in &queries {
        let results = idx.search(&q.vector, &q.tokens, cfg.k);
        assert!(results.len() <= cfg.k);
    }
}

#[test]
fn empty_query_tokens_returns_dense_results() {
    let cfg = DatasetConfig {
        n_docs: 50,
        n_queries: 5,
        ..small_cfg()
    };
    let (idx, queries) = build_and_insert(HybridRrf::new(cfg.dim), &cfg);
    for q in &queries {
        let results = idx.search(&q.vector, &[], cfg.k);
        assert!(
            !results.is_empty(),
            "with empty tokens, dense results must still surface"
        );
    }
}

#[test]
fn scores_are_monotone_decreasing() {
    let cfg = small_cfg();
    let (idx, queries) = build_and_insert(HybridRrf::new(cfg.dim), &cfg);
    for q in &queries {
        let results = idx.search(&q.vector, &q.tokens, cfg.k);
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "results must be sorted by descending score"
            );
        }
    }
}
