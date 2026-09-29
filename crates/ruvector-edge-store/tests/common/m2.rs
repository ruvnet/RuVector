//! M2 shard fixtures: datasets, ingestion through the full shard path
//! (plan → apply, alarm maintenance when due), queries and brute force.

use super::{tenant, Rng, T0};
use ruvector_edge_store::shard::{Actor, Due, IndexConfig, FLUSH_OPS, HNSW_SYNC_UPSERT};
use ruvector_edge_store::{
    shard_meta_for, ErrorCode, Metric, QueryRequest, ShardConfig, SqlStore, UpsertRow, VectorShard,
};
use ruvector_edge_tenancy::{CollectionUid, DoMeta, ShardIndex};
use serde_json::json;

pub const ACTOR: Actor<'static> = Actor {
    sub: "es1_m2",
    jti: "j",
    family_id: "f",
    act_sub: None,
};

pub fn dm(tag: u8) -> DoMeta {
    shard_meta_for(
        &tenant("org-m2"),
        CollectionUid::from_bytes([tag; 16]),
        ShardIndex::ZERO,
    )
    .unwrap()
}

pub fn cfg(dim: u32, metric: Metric, index: IndexConfig) -> ShardConfig {
    ShardConfig {
        dim,
        metric,
        filterable_keys: vec!["g".into()],
        float_cap: ruvector_edge_store::shard::M2_SHARD_FLOAT_CAP,
        index,
    }
}

pub fn flat() -> IndexConfig {
    IndexConfig::Flat
}

pub fn hnsw() -> IndexConfig {
    IndexConfig::HNSW_DEFAULT
}

/// `n` uniform `[-1, 1)` vectors with ids `v00000…` and a filter key `g`.
pub fn data(n: usize, dim: usize, seed: u64) -> Vec<(String, Vec<f32>)> {
    let mut r = Rng(seed);
    (0..n).map(|i| (format!("v{i:05}"), r.vec(dim))).collect()
}

pub fn rows(d: &[(String, Vec<f32>)]) -> Vec<UpsertRow> {
    d.iter()
        .enumerate()
        .map(|(i, (id, v))| UpsertRow {
            id: id.clone(),
            values: v.clone(),
            metadata: Some(json!({ "g": i % 4 })),
        })
        .collect()
}

/// Upsert `d` in batches (64 for HNSW, 500 for flat), running maintenance
/// whenever it is due now (as the DO alarm would). Every write leaves
/// fewer than `FLUSH_OPS` ops unpersisted (the cold-load replay bound).
pub fn ingest(
    s: &mut VectorShard,
    st: &dyn SqlStore,
    dm: &DoMeta,
    cfg: &ShardConfig,
    d: &[(String, Vec<f32>)],
) -> Result<(), ErrorCode> {
    let batch = match cfg.index {
        IndexConfig::Hnsw { .. } => HNSW_SYNC_UPSERT,
        IndexConfig::Flat => 500,
    };
    for chunk in d.chunks(batch) {
        let plan = s.plan_upsert(dm, cfg, rows(chunk)).map_err(|e| e.code)?;
        s.apply_upsert(st, plan, ACTOR, T0).map_err(|e| e.code)?;
        assert!(s.pending_ops() < FLUSH_OPS, "replay bound");
        if s.maintenance_due() == Some(Due::Now) {
            s.maintain(st).map_err(|e| e.code)?;
        }
    }
    Ok(())
}

pub fn req(q: &[f32], k: u32) -> QueryRequest {
    QueryRequest {
        vector: q.to_vec(),
        top_k: k,
        filter: None,
        include: vec![],
        ef: None,
        rerank: None,
    }
}

/// `(id, exact f64 score bits)` of a query's matches.
pub fn ranked(
    s: &mut VectorShard,
    st: &dyn SqlStore,
    dm: &DoMeta,
    cfg: &ShardConfig,
    r: &QueryRequest,
) -> Vec<(String, u64)> {
    s.query(st, dm, cfg, r)
        .unwrap()
        .matches
        .into_iter()
        .map(|m| (m.id.clone(), m.rank_score().to_bits()))
        .collect()
}

/// Exact top-`k` ids by `(f64 distance, id)`.
pub fn brute(metric: Metric, d: &[(String, Vec<f32>)], q: &[f32], k: usize) -> Vec<String> {
    use ruvector_edge_store::distance::{distance, norm};
    let qn = norm(q);
    let mut all: Vec<(f64, &str)> = d
        .iter()
        .map(|(id, v)| (distance(metric, q, qn, v, norm(v)), id.as_str()))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    all.into_iter()
        .take(k)
        .map(|(_, id)| id.to_string())
        .collect()
}

pub fn recall(truth: &[String], got: &[(String, u64)]) -> f64 {
    let hit = got.iter().filter(|(id, _)| truth.contains(id)).count();
    hit as f64 / truth.len().max(1) as f64
}

/// Flip one body byte of `(epoch, part)` in place.
pub fn corrupt_chunk(st: &dyn SqlStore, epoch: u64, part: u32) {
    use ruvector_edge_store::{schema, Value};
    let e = epoch as i64;
    let rows = st
        .query(
            schema::CHUNK_GET,
            &[e.into(), i64::from(part).into(), 1i64.into()],
        )
        .unwrap();
    let mut bytes = rows[0][1].as_blob().unwrap().to_vec();
    let at = bytes.len() - 1;
    bytes[at] ^= 0x5a;
    st.exec(
        schema::CHUNK_PUT,
        &[e.into(), i64::from(part).into(), Value::Blob(bytes)],
    )
    .unwrap();
}

/// A cold load with its index resident.
pub fn reopen(st: &dyn SqlStore) -> VectorShard {
    let mut s = VectorShard::open(st).unwrap();
    s.load_index(st).unwrap();
    s
}
