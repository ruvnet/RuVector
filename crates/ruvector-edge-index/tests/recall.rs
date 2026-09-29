//! Recall@10 on 10k × 384 clustered synthetic data (the ADR random set is
//! in `recall_random.rs`), held-out queries,
//! against exact f32 ground truth (M2a/M2b acceptance: ≥ 0.95).
//! Run with `--nocapture` to see the numbers.

mod common;

use common::*;
use ruvector_edge_index::{HnswIndex, HnswParams, Metric, QuantFlatIndex, QuantParams, SliceFetch};

/// Quantizer trained on a 1k sample, as the store's reservoir would.
fn trained(metric: Metric, base: &[f32]) -> QuantParams {
    QuantParams::train(metric, DIM, &base[..1000 * DIM], 1).unwrap()
}

fn flat_recall(q: QuantParams, base: &[f32], queries: &[f32], truth: &[Vec<u32>]) -> (f64, f64) {
    let mut idx = QuantFlatIndex::with_capacity(q, N as u32, N).unwrap();
    for (i, v) in base.chunks_exact(DIM).enumerate() {
        idx.upsert(i as u32, v).unwrap();
    }
    let mut fetch = SliceFetch::new(base, DIM);
    let (mut raw, mut rr) = (0.0, 0.0);
    for (qv, t) in queries.chunks_exact(DIM).zip(truth) {
        raw += recall(t, &idx.search(qv, K).unwrap());
        rr += recall(t, &idx.search_rerank(qv, K, 4 * K, &mut fetch).unwrap());
    }
    let n = truth.len() as f64;
    (raw / n, rr / n)
}

fn build_hnsw(q: QuantParams, base: &[f32]) -> HnswIndex {
    let mut idx = HnswIndex::with_capacity(HnswParams::default(), q, N).unwrap();
    for (i, v) in base.chunks_exact(DIM).enumerate() {
        idx.insert(i as u32, v, &mut op_rng(i as u64)).unwrap();
    }
    idx
}

fn hnsw_recall(
    idx: &HnswIndex,
    base: &[f32],
    queries: &[f32],
    truth: &[Vec<u32>],
    ef: usize,
) -> (f64, f64) {
    let mut fetch = SliceFetch::new(base, DIM);
    let (mut raw, mut rr) = (0.0, 0.0);
    for (qv, t) in queries.chunks_exact(DIM).zip(truth) {
        raw += recall(t, &idx.search(qv, K, ef).unwrap());
        rr += recall(t, &idx.search_rerank(qv, K, ef, &mut fetch).unwrap());
    }
    let n = truth.len() as f64;
    (raw / n, rr / n)
}

#[test]
fn int8_flat_with_rerank_recall_at_10() {
    let (base, queries) = data(N, DIM, QUERIES);
    for metric in METRICS {
        let t = truth(metric, &base, &queries, DIM, K);
        let (raw, rr) = flat_recall(trained(metric, &base), &base, &queries, &t);
        eprintln!(
            "flat int8 per-dim  {:6}: codes-only {raw:.4}  rerank(4k) {rr:.4}",
            metric.as_str()
        );
        assert!(rr >= 0.95, "{metric:?} flat rerank recall {rr}");
        if metric == Metric::Cosine {
            let q = QuantParams::cosine_fixed(DIM, 1).unwrap();
            let (raw, rr) = flat_recall(q, &base, &queries, &t);
            eprintln!("flat int8 fixed[-1,1] cosine: codes-only {raw:.4}  rerank(4k) {rr:.4}");
            assert!(rr >= 0.95, "cosine fixed-range flat rerank recall {rr}");
        }
    }
}

#[test]
fn hnsw_over_codes_recall_at_10() {
    let (base, queries) = data(N, DIM, QUERIES);
    for metric in METRICS {
        let t = truth(metric, &base, &queries, DIM, K);
        let idx = build_hnsw(trained(metric, &base), &base);
        for ef in [32, 64, 128] {
            let (raw, rr) = hnsw_recall(&idx, &base, &queries, &t, ef);
            eprintln!(
                "hnsw per-dim {:6} ef={ef:3}: codes-only {raw:.4}  rerank {rr:.4}",
                metric.as_str()
            );
            if ef == 64 {
                assert!(rr >= 0.95, "{metric:?} hnsw recall {rr} at ef 64");
            }
        }
    }
}

#[test]
fn hnsw_cosine_fixed_range_recall_at_10() {
    let (base, queries) = data(N, DIM, QUERIES);
    let t = truth(Metric::Cosine, &base, &queries, DIM, K);
    let idx = build_hnsw(QuantParams::cosine_fixed(DIM, 1).unwrap(), &base);
    for ef in [32, 64, 128] {
        let (raw, rr) = hnsw_recall(&idx, &base, &queries, &t, ef);
        eprintln!("hnsw fixed[-1,1] cosine ef={ef:3}: codes-only {raw:.4}  rerank {rr:.4}");
        if ef == 64 {
            assert!(rr >= 0.95, "cosine fixed-range hnsw recall {rr}");
        }
    }
}
