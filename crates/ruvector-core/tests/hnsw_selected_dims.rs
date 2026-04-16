//! End-to-end test for `HnswIndex::new_with_selected_dims` + `search_with_rerank`.
//!
//! This test exercises the first-class selected-dims constructor on the
//! `HnswIndex` type without reaching into the `ruvector-eml-hnsw` crate. It
//! establishes the minimum acceptance bar from ADR-151 on a tiny synthetic
//! corpus: recall@10 with rerank ≥ 0.80 at fetch_k=200.

use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};

/// Deterministic LCG. Stable across machines.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / u32::MAX as f32) - 0.5
}

/// Build a corpus where the signal is concentrated in the first `signal_dim`
/// dimensions (high variance) and the remaining dims carry noise. This is
/// the canonical setup where a correlation-based selector has something to
/// find.
fn build_skewed_corpus(n: usize, full_dim: usize, signal_dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            (0..full_dim)
                .map(|d| {
                    let v = lcg(&mut s);
                    if d < signal_dim {
                        v * 4.0
                    } else {
                        v * 0.1
                    }
                })
                .collect()
        })
        .collect()
}

fn cosine_distance_full(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-30 {
        1.0
    } else {
        ((1.0 - dot / denom).clamp(0.0, 2.0)) as f32
    }
}

/// Exact top-k indices by full cosine on the corpus.
fn exact_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_distance_full(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[test]
fn selected_dims_rerank_hits_acceptance_bar() {
    // Small but structured corpus.
    let full_dim = 128;
    let corpus = build_skewed_corpus(800, full_dim, 32, 0xABCD_1234);
    // Representative sample disjoint from queries.
    let sample: Vec<Vec<f32>> = corpus[..400].to_vec();
    let queries: Vec<Vec<f32>> = build_skewed_corpus(20, full_dim, 32, 0x1111_2222);

    let config = HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 200,
        max_elements: 10_000,
    };

    let mut index = HnswIndex::new_with_selected_dims(
        full_dim,
        &sample,
        32, // selected_k in the ADR-151 validated band
        DistanceMetric::Cosine,
        config,
    )
    .expect("selector trains on skewed sample");

    // Insert full-dim corpus — we want to measure recall against the same
    // corpus, so all corpus vectors must be indexed.
    let entries: Vec<(String, Vec<f32>)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("id_{i}"), v.clone()))
        .collect();
    index.add_batch(entries).expect("insert corpus");
    assert_eq!(index.len(), corpus.len());

    // Recall@10 with rerank at fetch_k=200.
    let k = 10;
    let fetch_k = 200;
    let mut recall_sum = 0.0f32;
    for q in &queries {
        let gt: std::collections::HashSet<usize> =
            exact_top_k(&corpus, q, k).into_iter().collect();

        let hits = index
            .search_with_rerank(q, k, fetch_k, 200)
            .expect("rerank search");
        let got: std::collections::HashSet<usize> = hits
            .iter()
            .filter_map(|r| r.id.strip_prefix("id_").and_then(|s| s.parse::<usize>().ok()))
            .collect();

        let inter = gt.intersection(&got).count();
        recall_sum += inter as f32 / k as f32;
    }
    let recall = recall_sum / queries.len() as f32;

    assert!(
        recall >= 0.80,
        "recall@10 with rerank must be >= 0.80 (ADR-151 acceptance bar); got {recall}"
    );
}

#[test]
fn selected_dims_construct_errors_on_empty_sample() {
    let res = HnswIndex::new_with_selected_dims(
        128,
        &[],
        32,
        DistanceMetric::Cosine,
        HnswConfig::default(),
    );
    match res {
        Ok(_) => panic!("empty sample must error"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("non-empty") || msg.contains("sample"));
        }
    }
}

#[test]
fn selected_dims_construct_errors_on_bad_selected_k() {
    let sample: Vec<Vec<f32>> = (0..10).map(|_| vec![0.0f32; 16]).collect();
    let res = HnswIndex::new_with_selected_dims(
        16,
        &sample,
        0,
        DistanceMetric::Cosine,
        HnswConfig::default(),
    );
    match res {
        Ok(_) => panic!("selected_k=0 must error"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("selected_k"));
        }
    }

    let res2 = HnswIndex::new_with_selected_dims(
        16,
        &sample,
        64, // > full_dim
        DistanceMetric::Cosine,
        HnswConfig::default(),
    );
    match res2 {
        Ok(_) => panic!("selected_k > full_dim must error"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("selected_k"));
        }
    }
}

#[test]
fn standard_backend_search_with_rerank_falls_through() {
    // Regression: calling search_with_rerank on a Standard HnswIndex should
    // behave like search_with_ef(k). Existing callers of `HnswIndex::new`
    // must see no API changes.
    let mut index =
        HnswIndex::new(16, DistanceMetric::Cosine, HnswConfig::default()).expect("new");
    for i in 0..20u32 {
        let mut v = vec![0.0f32; 16];
        v[(i as usize) % 16] = 1.0;
        v[0] += (i as f32) * 0.01;
        index.add(format!("v_{i}"), v).unwrap();
    }

    let q = {
        let mut v = vec![0.0f32; 16];
        v[0] = 1.0;
        v
    };
    let hits_rerank = index
        .search_with_rerank(&q, 3, 50, 50)
        .expect("rerank fall-through");
    let hits_plain = index.search_with_ef(&q, 3, 50).expect("plain");

    assert_eq!(hits_rerank.len(), hits_plain.len());
    // IDs should match (same search path under the hood).
    let ids_r: Vec<_> = hits_rerank.iter().map(|r| &r.id).collect();
    let ids_p: Vec<_> = hits_plain.iter().map(|r| &r.id).collect();
    assert_eq!(ids_r, ids_p);
}
