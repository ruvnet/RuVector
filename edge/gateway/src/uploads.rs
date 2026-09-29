//! Upload sessions for bulk import (ADR-351 §6.3 `staging/`, §7.2
//! `POST /v1/uploads`, write + editor).
//!
//! `POST /v1/uploads {size, sha256}` mints the `upload_id` server-side and
//! opens an R2 multipart upload at `staging/{tenant_key}/{upload_id}`; the
//! session lives in the caller's ledger, so an id resolves only inside its
//! tenant. `POST /v1/uploads/{id}/parts/{n}` carries one part (exactly
//! [`PART_BYTES`], the last one the remainder; part bodies up to 8 MiB are
//! the one exception to the 1 MiB cap), `POST /v1/uploads/{id}:complete`
//! completes it and checks the size. R2 stores no sha256 for a multipart
//! object, so the import job's first delivery hashes the whole upload
//! (streamed) against the declared sha256 before any row is applied.
//!
//! Inline imports (`application/octet-stream` body ≤ 512 KiB on `:import`)
//! skip the session: the body is hashed and PUT with its sha256 so R2
//! stores the checksum.

use crate::m3_ctx::{mint, object, M3};
use crate::m3_ports::{Blob, Queues};
use crate::m3_wire::{kv_get, kv_put, kv_range, unhex32, M3Backend, Ns};
use ruvector_edge_auth::Capability;
use ruvector_edge_store::{ErrorCode, OpError};
use ruvector_edge_tenancy::Role;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

/// Part size (every part but the last). R2 needs ≥ 5 MiB, equal parts.
pub const PART_BYTES: u64 = 8 << 20;
/// Largest upload (the importer's default `max_file_bytes`).
pub const MAX_UPLOAD_BYTES: u64 = 1 << 30;
/// Largest upload part body (R2 multipart parts are [`PART_BYTES`]).
pub const MAX_PART_BODY: usize = 8 << 20;
/// Largest inline import body. The inline path hashes the body in the
/// request (so R2 stores its checksum): 512 KiB keeps that near the
/// Workers Free 10 ms CPU budget (~100 MB/s wasm SHA-256, an estimate);
/// larger imports use an upload session,
/// whose sha256 pass runs in the queue consumer in bounded slices.
pub const MAX_INLINE_BYTES: usize = 512 << 10;

/// A stored upload session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Upload {
    /// Server-minted id.
    pub upload_id: String,
    /// R2 multipart upload id (`None` for inline uploads).
    pub r2_upload: Option<String>,
    /// Declared size.
    pub size: u64,
    /// Declared sha256 (hex).
    pub sha256: String,
    /// Parts expected.
    pub parts: u64,
    /// `open` | `complete` | `consumed`.
    pub state: String,
    /// Creator.
    pub sub: String,
    /// Unix seconds.
    pub created_at: u64,
}

impl Upload {
    /// R2 key (server-derived only).
    pub fn key(&self, tenant: &str) -> String {
        format!("staging/{tenant}/{}", self.upload_id)
    }
}

/// Load the caller's own upload (`404` for foreign or unknown ids).
pub async fn load<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    id: &str,
) -> Result<Upload, OpError> {
    if id.len() != 24 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(OpError::not_found());
    }
    let v = kv_get(m.b, m.ctx.tenant_key(), Ns::Upload, id)
        .await?
        .ok_or(OpError::not_found())?;
    serde_json::from_str(&v).map_err(|_| crate::service::unexpected())
}

/// Persist an upload session.
pub async fn save<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    u: &Upload,
) -> Result<(), OpError> {
    let v = serde_json::to_string(u).map_err(|_| crate::service::unexpected())?;
    kv_put(m.b, m.ctx.tenant_key(), Ns::Upload, &u.upload_id, v).await
}

fn new_id<B: M3Backend, R: Blob, Q: Queues>(m: &M3<'_, B, R, Q>, sha: &str) -> String {
    let t = m.ctx.tenant_key().as_str();
    let now = m.now_ms.to_le_bytes();
    mint(
        &[t.as_bytes(), m.ctx.jti().as_bytes(), sha.as_bytes(), &now],
        12,
    )
}

/// `POST /v1/uploads {size, sha256}` → 201 `{upload_id, part_size, parts}`.
pub async fn create<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    body: &[u8],
) -> Result<(u16, Json), OpError> {
    m.require(Capability::Write, Role::Editor).await?;
    let mut o = object(body)?;
    let size = o.remove("size").and_then(|v| v.as_u64());
    let sha = o
        .remove("sha256")
        .and_then(|v| v.as_str().map(str::to_string));
    let (Some(size), Some(sha), true) = (size, sha, o.is_empty()) else {
        return Err(OpError::invalid("expected {size, sha256}"));
    };
    if size == 0 || size > MAX_UPLOAD_BYTES {
        return Err(OpError::new(ErrorCode::PayloadTooLarge, "upload size"));
    }
    unhex32(&sha.to_ascii_lowercase())?;
    m.charge(1).await?;
    let upload_id = new_id(m, &sha);
    let tenant = m.ctx.tenant_key().as_str();
    let key = format!("staging/{tenant}/{upload_id}");
    let r2 = m.blob.mp_begin(&key).await?;
    let parts = size.div_ceil(PART_BYTES);
    let u = Upload {
        upload_id: upload_id.clone(),
        r2_upload: Some(r2),
        size,
        sha256: sha.to_ascii_lowercase(),
        parts,
        state: "open".into(),
        sub: m.ctx.sub().to_string(),
        created_at: m.now_ms / 1000,
    };
    save(m, &u).await?;
    Ok((
        201,
        json!({ "upload_id": upload_id, "part_size": PART_BYTES, "parts": parts }),
    ))
}

/// `POST /v1/uploads/{id}/parts/{n}` (raw body).
pub async fn part<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    id: &str,
    n: u16,
    body: Vec<u8>,
) -> Result<(u16, Json), OpError> {
    m.require(Capability::Write, Role::Editor).await?;
    let u = load(m, id).await?;
    let r2 = u.r2_upload.as_deref().ok_or(OpError::not_found())?;
    if u.state != "open" {
        return Err(OpError::new(ErrorCode::Conflict, "upload not open"));
    }
    let n64 = u64::from(n);
    if n64 == 0 || n64 > u.parts {
        return Err(OpError::invalid("part number"));
    }
    let want = if n64 < u.parts {
        PART_BYTES
    } else {
        u.size - PART_BYTES * (u.parts - 1)
    };
    if body.len() as u64 != want {
        return Err(OpError::invalid("part size"));
    }
    m.charge(1).await?;
    let tenant = m.ctx.tenant_key().as_str();
    let etag = m.blob.mp_part(&u.key(tenant), r2, n, body).await?;
    let k = format!("{id}/{n:05}");
    kv_put(m.b, m.ctx.tenant_key(), Ns::UploadPart, &k, etag).await?;
    Ok((200, json!({ "upload_id": id, "part": n })))
}

/// `POST /v1/uploads/{id}:complete`.
pub async fn complete<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    id: &str,
) -> Result<(u16, Json), OpError> {
    m.require(Capability::Write, Role::Editor).await?;
    let mut u = load(m, id).await?;
    if u.state != "open" {
        return Err(OpError::new(ErrorCode::Conflict, "upload not open"));
    }
    let r2 = u.r2_upload.clone().ok_or(OpError::not_found())?;
    let items = kv_range(
        m.b,
        m.ctx.tenant_key(),
        Ns::UploadPart,
        format!("{id}/"),
        format!("{id}0"),
        1000,
    )
    .await?;
    let mut parts = Vec::with_capacity(items.len());
    for (k, etag) in items {
        let n: u16 = k
            .rsplit('/')
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or(crate::service::unexpected())?;
        parts.push((n, etag));
    }
    let complete = parts.len() as u64 == u.parts
        && parts
            .iter()
            .enumerate()
            .all(|(i, (n, _))| *n as usize == i + 1);
    if !complete {
        return Err(OpError::new(ErrorCode::Conflict, "upload parts missing"));
    }
    m.charge(1).await?;
    let tenant = m.ctx.tenant_key().as_str();
    let size = m.blob.mp_complete(&u.key(tenant), &r2, &parts).await?;
    if size != u.size {
        let _removed = m.blob.delete(&u.key(tenant)).await;
        return Err(OpError::new(ErrorCode::Conflict, "upload size mismatch"));
    }
    u.state = "complete".into();
    save(m, &u).await?;
    Ok((
        200,
        json!({ "upload_id": id, "size": size, "state": "complete" }),
    ))
}

/// Store an inline body as a complete upload.
pub async fn inline<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    body: Vec<u8>,
) -> Result<Upload, OpError> {
    use sha2::{Digest, Sha256};
    if body.is_empty() || body.len() > MAX_INLINE_BYTES {
        return Err(OpError::new(
            ErrorCode::PayloadTooLarge,
            "inline import size",
        ));
    }
    let digest: [u8; 32] = Sha256::digest(&body).into();
    let sha = crate::m3_wire::hex32(&digest);
    let u = Upload {
        upload_id: new_id(m, &sha),
        r2_upload: None,
        size: body.len() as u64,
        sha256: sha,
        parts: 1,
        state: "complete".into(),
        sub: m.ctx.sub().to_string(),
        created_at: m.now_ms / 1000,
    };
    let tenant = m.ctx.tenant_key().as_str();
    m.blob.put(&u.key(tenant), body, Some(digest)).await?;
    save(m, &u).await?;
    Ok(u)
}
