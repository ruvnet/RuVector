//! `POST /v1/collections/{c}:export` and `GET /v1/exports/{id}` (ADR-351
//! §7.2 M3, read + viewer).
//!
//! The export streams every shard (consistent pages, `409` if a shard
//! moves mid-read) through `RvfExporter` and `finish_witnessed` into an R2
//! multipart upload at `exports/{tenant_key}/{export_id}.rvf`, in equal
//! 8 MiB parts (R2 requires equal parts but the last); memory stays at one
//! part plus one segment pair. `?redact=a,b` drops those metadata keys
//! before rows reach the exporter. The witness names the tenant, the
//! collection and a combined audit head (sha256 over the per-shard heads,
//! each binding the shard state and the tenant's shipped-audit chain head).
//!
//! The reply carries a gateway path, never an R2 or public URL: the file is
//! downloadable only with a bearer token of the same tenant (read +
//! viewer) until `expires_at` (15 min); an R2 lifecycle rule removes
//! `exports/` objects after a day.

use crate::m3_ctx::{dim16, mint, snap_err, snap_metric, M3};
use crate::m3_ports::{Blob, Queues};
use crate::m3_wire::{kv_get, kv_put, M3Backend, Ns};
use crate::service::{count_of, shard_meta};
use crate::snapshots::{page, shard_head};
use ruvector_edge_auth::Capability;
use ruvector_edge_snapshot::{ExportIdentity, RvfExporter, DEFAULT_ROWS_PER_SEGMENT};
use ruvector_edge_store::{ErrorCode, OpError};
use ruvector_edge_tenancy::Role;
use serde_json::{json, Map, Value as Json};

/// R2 multipart part size (every part but the last).
pub const PART_BYTES: usize = 8 << 20;
/// Download window.
pub const EXPORT_TTL_S: u64 = 15 * 60;
/// Largest `redact` list.
pub const MAX_REDACT_KEYS: usize = 32;

/// Parse `?redact=a,b` (keys 1..=64 visible ASCII, no commas).
pub fn redact_keys(query: Option<&str>) -> Result<Vec<String>, OpError> {
    let mut out = Vec::new();
    for pair in query.unwrap_or("").split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        if k != "redact" {
            return Err(OpError::invalid("unknown query parameter"));
        }
        for key in v.split(',').filter(|s| !s.is_empty()) {
            if key.len() > 64 || !key.bytes().all(|b| b.is_ascii_graphic()) {
                return Err(OpError::invalid("redact key"));
            }
            out.push(key.to_string());
        }
    }
    if out.len() > MAX_REDACT_KEYS {
        return Err(OpError::invalid("too many redact keys"));
    }
    Ok(out)
}

fn redact(meta: Option<String>, keys: &[String]) -> Result<Option<String>, OpError> {
    let Some(text) = meta else { return Ok(None) };
    if keys.is_empty() {
        return Ok(Some(text));
    }
    let mut m: Map<String, Json> =
        serde_json::from_str(&text).map_err(|_| OpError::new(ErrorCode::ServerError, "meta"))?;
    for k in keys {
        m.remove(k);
    }
    Ok(Some(Json::Object(m).to_string()))
}

/// Equal-size multipart writer over [`Blob`].
struct Parts<'a, R> {
    blob: &'a R,
    key: String,
    upload: String,
    buf: Vec<u8>,
    done: Vec<(u16, String)>,
    bytes: u64,
}

impl<R: Blob> Parts<'_, R> {
    async fn flush(&mut self, last: bool) -> Result<(), OpError> {
        while self.buf.len() >= PART_BYTES || (last && !self.buf.is_empty()) {
            let n = self.buf.len().min(PART_BYTES);
            let part: Vec<u8> = self.buf.drain(..n).collect();
            let no = u16::try_from(self.done.len() + 1)
                .map_err(|_| OpError::new(ErrorCode::PayloadTooLarge, "export too large"))?;
            self.bytes += part.len() as u64;
            let etag = self.blob.mp_part(&self.key, &self.upload, no, part).await?;
            self.done.push((no, etag));
        }
        Ok(())
    }
}

/// `POST /v1/collections/{c}:export` → `{export_id, download, expires_at}`.
pub async fn export<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    collection: &str,
    query: Option<&str>,
) -> Result<(u16, Json), OpError> {
    m.require(Capability::Read, Role::Viewer).await?;
    let keys = redact_keys(query)?;
    let e = m.collection(collection).await?;
    let shards = count_of(&e)?;
    m.charge(1 + u64::from(shards.get())).await?;
    let tenant = m.ctx.tenant_key().as_str();
    let now = m.now_ms.to_le_bytes();
    let export_id = mint(
        &[
            tenant.as_bytes(),
            e.uid.as_bytes(),
            m.ctx.jti().as_bytes(),
            &now,
        ],
        12,
    );
    let key = format!("exports/{tenant}/{export_id}.rvf");
    let upload = m.blob.mp_begin(&key).await?;
    let mut parts = Parts {
        blob: m.blob,
        key: key.clone(),
        upload,
        buf: Vec::new(),
        done: Vec::new(),
        bytes: 0,
    };
    let r = write(m, &e, &keys, &mut parts).await;
    let (rows, heads) = match r {
        Ok(v) => v,
        Err(err) => {
            let _aborted = m.blob.mp_abort(&parts.key, &parts.upload).await;
            return Err(err);
        }
    };
    let size = m
        .blob
        .mp_complete(&parts.key, &parts.upload, &parts.done)
        .await?;
    let expires_at = m.now_ms / 1000 + EXPORT_TTL_S;
    let rec = json!({
        "export_id": export_id,
        "key": key,
        "collection": e.view["name"],
        "uid": e.uid,
        "rows": rows,
        "bytes": size,
        "audit_head": heads,
        "expires_at": expires_at,
        "created_by": m.ctx.sub(),
    });
    kv_put(
        m.b,
        m.ctx.tenant_key(),
        Ns::Export,
        &export_id,
        rec.to_string(),
    )
    .await?;
    Ok((
        201,
        json!({
            "export_id": export_id,
            "download": format!("/v1/exports/{export_id}"),
            "expires_at": expires_at,
            "rows": rows,
            "bytes": size,
            "redacted": keys,
        }),
    ))
}

async fn write<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    e: &crate::wire::CollectionWire,
    keys: &[String],
    parts: &mut Parts<'_, R>,
) -> Result<(u64, String), OpError> {
    let dim = dim16(e)?;
    let tenant = m.ctx.tenant_key().as_str();
    let mut exp = RvfExporter::new(dim, snap_metric(e.cfg.metric), DEFAULT_ROWS_PER_SEGMENT)
        .map_err(snap_err)?;
    let audit = crate::audit::head(m.b, m.ctx.tenant_key()).await?;
    let mut heads = Vec::new();
    for i in count_of(e)?.indices() {
        let dm = shard_meta(m.ctx, e, i)?;
        let (seq0, _, mut rows) = page(m.b, &dm, None).await?;
        heads.extend_from_slice(&shard_head(tenant, &e.uid, i.get(), seq0, &audit));
        loop {
            let last = rows.last().map(|r| r.id.clone());
            for r in rows {
                let mut row = r.into_row(dim)?;
                row.metadata = redact(row.metadata.take(), keys)?;
                let buf = &mut parts.buf;
                exp.push(&row, &mut |b: &[u8]| buf.extend_from_slice(b))
                    .map_err(snap_err)?;
            }
            parts.flush(false).await?;
            let Some(after) = last else { break };
            let (seq, _, next) = page(m.b, &dm, Some(after)).await?;
            if seq != seq0 {
                return Err(OpError::new(
                    ErrorCode::Conflict,
                    "collection changed during read, retry",
                ));
            }
            rows = next;
        }
    }
    let audit_head = {
        use sha2::{Digest, Sha256};
        let d: [u8; 32] = Sha256::digest(&heads).into();
        d
    };
    let id = ExportIdentity::new(tenant, &e.uid, audit_head).map_err(snap_err)?;
    let buf = &mut parts.buf;
    let summary = exp
        .finish_witnessed(&mut |b: &[u8]| buf.extend_from_slice(b), &id, None)
        .map_err(snap_err)?;
    parts.flush(true).await?;
    Ok((summary.rows, crate::m3_wire::hex32(&audit_head)))
}

/// What `GET /v1/exports/{id}` streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Download {
    /// R2 key.
    pub key: String,
    /// Object size.
    pub size: u64,
    /// Suggested file name.
    pub filename: String,
}

/// `GET /v1/exports/{id}`: the caller's own, unexpired export.
pub async fn download<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    id: &str,
) -> Result<Download, OpError> {
    m.require(Capability::Read, Role::Viewer).await?;
    if id.len() != 24 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(OpError::not_found());
    }
    let rec = kv_get(m.b, m.ctx.tenant_key(), Ns::Export, id)
        .await?
        .ok_or(OpError::not_found())?;
    let rec: Json = serde_json::from_str(&rec).map_err(|_| crate::service::unexpected())?;
    if rec["expires_at"].as_u64().unwrap_or(0) <= m.now_ms / 1000 {
        return Err(OpError::not_found());
    }
    m.charge(1).await?;
    Ok(Download {
        key: rec["key"].as_str().unwrap_or_default().to_string(),
        size: rec["bytes"].as_u64().unwrap_or(0),
        filename: format!("{id}.rvf"),
    })
}
