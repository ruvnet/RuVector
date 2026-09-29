//! rv-embed (ADR-351 §3, §7.2 `text` on upsert/query): collections created
//! with `embedder: "bge-small-en-v1.5"` (384 dims) accept `text` in place
//! of `values` / `vector`; the gateway embeds through Workers AI
//! (`@cf/baai/bge-small-en-v1.5`, via the snapshot crate's
//! `EmbeddingBatcher`: ≤ 100 texts and ≤ 100 × 512 estimated tokens per
//! model call, ≤ 8 KiB per text) and hands the rewritten body to the M1
//! executor unchanged.
//!
//! Order: authorize (scope ∩ role) → resolve the collection's embedder in
//! the caller's ledger → plan → for upserts, a dry-run of the M1 upsert
//! with placeholder vectors plus a vector-quota check (a body M1 would
//! refuse never reaches the model) → **admit the work units** → call the
//! model. A tenant over quota or without the right never reaches Workers
//! AI. Work units of a failed model call are not refunded.
//! Metering: one work unit per model call plus one per 1,000 estimated
//! tokens.
//!
//! Collections without an embedder refuse `text` (`400`); `text` together
//! with `values` / `vector` is refused.

use crate::m3_ctx::{object, M3};
use crate::m3_ports::{Blob, Queues};
use crate::m3_wire::{kv_get, ledger3, M3Backend, M3LedgerCall, M3LedgerOut, Ns};
use crate::service;
use ruvector_edge_snapshot::{
    embed_all, EmbedError, EmbeddingBatcher, EmbeddingPort, BGE_SMALL_DIM,
};
use ruvector_edge_store::{ErrorCode, Op, OpError};
use serde_json::{json, Map, Value as Json};

/// The one embedder name collections may declare.
pub const EMBEDDER: &str = "bge-small-en-v1.5";

fn embed_err(e: EmbedError) -> OpError {
    match e {
        EmbedError::Port(_) | EmbedError::ResponseCount | EmbedError::BadVector { .. } => {
            OpError::new(ErrorCode::ShardUnavailable, "embedding service")
        }
        EmbedError::TooManyTexts(_) => OpError::new(ErrorCode::PayloadTooLarge, "too many texts"),
        EmbedError::TextTooLong { .. } | EmbedError::TooManyTokens { .. } => {
            OpError::new(ErrorCode::PayloadTooLarge, "text too long")
        }
        EmbedError::EmptyText { .. } => OpError::invalid("empty text"),
    }
}

/// Work units for embedding `texts` (planned before any model call).
pub fn work_units(batcher: &EmbeddingBatcher, texts: &[&str]) -> Result<u64, OpError> {
    let plan = batcher.plan(texts).map_err(embed_err)?;
    Ok(plan
        .iter()
        .map(|b| 1 + (b.est_tokens as u64).div_ceil(1000))
        .sum())
}

/// `POST /v1/collections` with `embedder`: validated, then created and
/// recorded in one ledger turn. `None` when the body has no `embedder`
/// (the M1 path handles it).
pub async fn create<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    body: &[u8],
) -> Result<Option<(u16, Json)>, OpError> {
    let Ok(mut o) = object(body) else {
        return Ok(None);
    };
    let Some(model) = o.remove("embedder") else {
        return Ok(None);
    };
    m.note.scope.set(Some(
        ruvector_edge_auth::Capability::CreateCollection.satisfying_scope(),
    ));
    let a = service::authorize_op(m.b, m.ctx, Op::CollectionCreate).await?;
    m.note.role.set(a.role);
    if model.as_str() != Some(EMBEDDER) {
        return Err(OpError::invalid("unknown embedder"));
    }
    match o.get("dim") {
        None => {
            o.insert("dim".into(), json!(BGE_SMALL_DIM));
        }
        Some(d) if d.as_u64() == Some(BGE_SMALL_DIM as u64) => {}
        Some(_) => {
            return Err(OpError::new(
                ErrorCode::DimensionMismatch,
                "embedder dimension is 384",
            ))
        }
    }
    let call = M3LedgerCall::CreateWithEmbedder {
        spec: Json::Object(o).to_string(),
        model: EMBEDDER.into(),
        sub: m.ctx.sub().to_string(),
        now: m.now_ms / 1000,
    };
    match ledger3(m.b, m.ctx.tenant_key(), call).await? {
        M3LedgerOut::Created { entry } => {
            let mut view = entry.view;
            view["embedder"] = json!(EMBEDDER);
            Ok(Some((201, view)))
        }
        _ => Err(service::unexpected()),
    }
}

/// The collection's embedder check (`400` without one).
async fn require_embedder<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    collection: &str,
) -> Result<(), OpError> {
    let e = m.collection(collection).await?;
    match kv_get(m.b, m.ctx.tenant_key(), Ns::Embedder, &e.uid).await? {
        Some(model) if model == EMBEDDER => Ok(()),
        _ => Err(OpError::invalid("collection has no embedder")),
    }
}

async fn embed<B: M3Backend, R: Blob, Q: Queues, E: EmbeddingPort>(
    m: &M3<'_, B, R, Q>,
    ai: &E,
    texts: &[&str],
) -> Result<(Vec<Vec<f32>>, u64), OpError> {
    let batcher = EmbeddingBatcher::default();
    let wu = work_units(&batcher, texts)?;
    // Admitted before the model is called (§10 layer 4, §17 work units).
    // Not refunded if the model call fails: the M1 ledger has no work-unit
    // correction (`LedgerCall::Adjust` moves quota counters only).
    service::charge(&m.call(), Default::default(), wu).await?;
    m.note.charged(wu);
    let v = embed_all(ai, &batcher, texts).await.map_err(embed_err)?;
    Ok((v, wu))
}

/// Refuse, before any model call or work-unit charge, what the M1 upsert
/// would refuse anyway: batch size, ids, metadata, dimension and shard caps
/// (a dry-run plan with a placeholder unit vector per `text` row) and
/// growth beyond the tenant's remaining vector quota.
async fn precheck_upsert<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    a: service::Access,
    collection: &str,
    o: &Map<String, Json>,
) -> Result<(), OpError> {
    let mut probe = o.clone();
    probe.remove("dry_run");
    if let Some(Json::Array(rows)) = probe.get_mut("vectors") {
        for r in rows.iter_mut().filter_map(Json::as_object_mut) {
            if r.remove("text").is_some() {
                let mut unit = vec![0.0f32; BGE_SMALL_DIM];
                unit[0] = 1.0;
                r.insert("values".into(), json!(unit));
            }
        }
    }
    let body = Json::Object(probe).to_string();
    let (args, _) = crate::rest::op_args(body.as_bytes(), Some(collection), false)?;
    let call = service::Call {
        dry_run: true,
        ..m.call()
    };
    let (plan, _) = service::run(&call, a, Op::VectorUpsert, &args).await?;
    let grow = plan["delta"]["vectors"].as_i64().unwrap_or(0);
    if grow > 0 {
        let dim = BGE_SMALL_DIM as u32;
        let left = crate::ingest::remaining(m.b, m.ctx.tenant_key(), dim, m.now_ms / 1000).await?;
        if grow.unsigned_abs() > left {
            return Err(OpError::new(ErrorCode::QuotaExceeded, "vector quota"));
        }
    }
    Ok(())
}

/// Upsert body with `text` rows → body with `values`; `None` if no row
/// carries `text`. Returns the work units charged.
pub async fn upsert_body<B: M3Backend, R: Blob, Q: Queues, E: EmbeddingPort>(
    m: &M3<'_, B, R, Q>,
    ai: &E,
    collection: &str,
    body: &[u8],
) -> Result<Option<(Vec<u8>, u64)>, OpError> {
    let Ok(mut o) = object(body) else {
        return Ok(None);
    };
    let has_text = o
        .get("vectors")
        .and_then(Json::as_array)
        .is_some_and(|a| a.iter().any(|v| v.get("text").is_some()));
    if !has_text {
        return Ok(None);
    }
    let a = service::authorize_op(m.b, m.ctx, Op::VectorUpsert).await?;
    require_embedder(m, collection).await?;
    let Some(Json::Array(rows)) = o.get("vectors") else {
        return Err(OpError::invalid("vectors"));
    };
    // Per row: `text` or `values`, never both (rows with `values` pass through).
    let mut texts: Vec<String> = Vec::with_capacity(rows.len());
    for r in rows.iter() {
        match (r.get("text"), r.get("values")) {
            (Some(_), Some(_)) => return Err(OpError::invalid("text and values are exclusive")),
            (Some(t), None) => texts.push(t.as_str().ok_or(OpError::invalid("text"))?.to_string()),
            _ => {}
        }
    }
    precheck_upsert(m, a, collection, &o).await?;
    let Some(Json::Array(rows)) = o.get_mut("vectors") else {
        return Err(OpError::invalid("vectors"));
    };
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let (vectors, wu) = embed(m, ai, &refs).await?;
    let texted = rows.iter_mut().filter(|r| r.get("text").is_some());
    for (r, v) in texted.zip(vectors) {
        if let Some(obj) = r.as_object_mut() {
            obj.remove("text");
            obj.insert("values".into(), json!(v));
        }
    }
    Ok(Some((Json::Object(o).to_string().into_bytes(), wu)))
}

/// Query body with `text` → body with `vector`; `None` without `text`.
pub async fn query_body<B: M3Backend, R: Blob, Q: Queues, E: EmbeddingPort>(
    m: &M3<'_, B, R, Q>,
    ai: &E,
    collection: &str,
    body: &[u8],
) -> Result<Option<(Vec<u8>, u64)>, OpError> {
    let Ok(mut o) = object(body) else {
        return Ok(None);
    };
    let Some(text) = o.remove("text") else {
        return Ok(None);
    };
    service::authorize_op(m.b, m.ctx, Op::VectorQuery).await?;
    if o.contains_key("vector") {
        return Err(OpError::invalid("text and vector are exclusive"));
    }
    let text = text.as_str().ok_or(OpError::invalid("text"))?.to_string();
    require_embedder(m, collection).await?;
    let (mut v, wu) = embed(m, ai, &[text.as_str()]).await?;
    let vector = v
        .pop()
        .ok_or(OpError::new(ErrorCode::ShardUnavailable, "embedding"))?;
    let mut out: Map<String, Json> = o;
    out.insert("vector".into(), json!(vector));
    Ok(Some((Json::Object(out).to_string().into_bytes(), wu)))
}
