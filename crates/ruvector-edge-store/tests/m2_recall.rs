//! ADR-351 §15 M2a/M2b acceptance through the full shard path: 10k × 384
//! uniform vectors upserted via `plan_upsert` → `apply_upsert` (with alarm
//! maintenance when due), queried with the default parameters (int8 codes
//! scanned or traversed, the best `max(40, 4k)` reranked exactly from
//! SQLite), recall@10 against an exact f64 brute force ≥ 0.95. Runs on
//! real SQLite (what Durable Objects execute). `--nocapture` prints recall.

mod common;

use common::m2::*;
use common::sqlite::SqliteStore;
use ruvector_edge_store::shard::{IndexConfig, IndexSource, SHARD_RESIDENT_CAP_BYTES};
use ruvector_edge_store::{Metric, VectorShard};

const N: usize = 10_000;
const DIM: usize = 384;
const Q: usize = 50;
const K: usize = 10;

fn run(metric: Metric, index: IndexConfig, tag: u8) -> f64 {
    let st = SqliteStore::default();
    let mut s = VectorShard::open(&st).unwrap();
    let (dm, cfg) = (dm(tag), cfg(DIM as u32, metric, index));
    let base = data(N, DIM, 11);
    ingest(&mut s, &st, &dm, &cfg, &base).unwrap();
    assert_eq!(s.len(), N);
    assert!(s.resident_bytes() <= SHARD_RESIDENT_CAP_BYTES);
    // Codes only: well under the 1536 B/row an f32 slab would need.
    assert!(s.index_bytes() < (N * 600) as u64, "{}", s.index_bytes());
    let queries = data(Q, DIM, 12);
    let mut total = 0.0;
    for (_, q) in &queries {
        let truth = brute(metric, &base, q, K);
        total += recall(&truth, &ranked(&mut s, &st, &dm, &cfg, &req(q, K as u32)));
    }
    let r = total / Q as f64;
    // A cold load (decoded epoch + ≤ 199-op replay) answers identically.
    // Timed natively (store in debug, index crate at opt-level 3): a
    // development figure, not the Worker p95 (size shards with the index
    // crate's DECODE_MS_PER_MIB_WASM).
    let replay = s.pending_ops();
    let t = std::time::Instant::now();
    let mut cold = reopen(&st);
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(cold.index_source(), Some(IndexSource::Current));
    for (_, q) in queries.iter().take(5) {
        assert_eq!(
            ranked(&mut cold, &st, &dm, &cfg, &req(q, K as u32)),
            ranked(&mut s, &st, &dm, &cfg, &req(q, K as u32))
        );
    }
    eprintln!(
        "{:?} {:?}: recall@10 = {r:.3}, index {} B, resident {} B, cold load {load_ms:.0} ms ({replay} ops replayed)",
        index.kind_str(),
        metric,
        s.index_bytes(),
        s.resident_bytes()
    );
    r
}

#[test]
fn flat_int8_rerank_recall_at_10_on_10k_x_384() {
    for (i, metric) in [Metric::Cosine, Metric::L2, Metric::Dot]
        .into_iter()
        .enumerate()
    {
        let r = run(metric, flat(), 20 + i as u8);
        assert!(r >= 0.95, "flat {metric:?}: recall@10 = {r}");
    }
}

#[test]
fn hnsw_recall_at_10_on_10k_x_384() {
    for (i, metric) in [Metric::Cosine, Metric::L2, Metric::Dot]
        .into_iter()
        .enumerate()
    {
        let r = run(metric, hnsw(), 30 + i as u8);
        assert!(r >= 0.95, "hnsw {metric:?}: recall@10 = {r}");
    }
}
