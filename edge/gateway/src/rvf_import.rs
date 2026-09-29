//! `POST /v1/collections/{c}:import-rvf {package, version, offset?}`
//! (ADR-351 §15 M5): load a registry package into a collection.
//!
//! This is the **snapshot-independent** import path over the registry
//! crate's streaming rvf-wire validator: the pulled blob is re-validated
//! (its SHA-256 must be the manifest's), the live vector records are
//! indexed (deleted ids removed, the last record per id wins, ordered by
//! id), and the vectors are upserted in `MAX_UPSERT_BATCH` batches through
//! the ordinary upsert executor, so quotas, dimension and finiteness checks
//! are exactly those of `vectors:upsert`. RVF vector ids become decimal
//! string ids.
//!
//! **Bounded per request, resumable across requests.** Every upsert batch
//! costs `5 + 2·shards` Durable Object subrequests, so one request imports
//! at most `IMPORT_SUBREQUEST_BUDGET / (5 + 2·shards)` batches, starting at
//! row `offset` of the id-ordered live rows, and answers `next_offset`
//! (`null` once done); the client repeats with it. The order is a pure
//! function of the package, and upserts are idempotent by id, so a retry
//! from any offset is safe. Memory is bounded too: packages up to
//! [`MAX_IMPORT_BYTES`] with at most [`MAX_IMPORT_RECORDS`] vector records;
//! only a 16-byte index entry per record is held, and vectors are decoded
//! for the current window only. Larger packages need M3's chunked
//! bulk-import jobs (`ruvector-edge-snapshot::rvf_import`).
//!
//! Authorization: the collection write is checked first (token scope and
//! ledger role, like `vectors:upsert`), then the package pull (`Read`,
//! visibility; a foreign non-public package is `404`).

use crate::backend::Backend;
use crate::registry_ports::{scope, BlobStore, RegistryRpc};
use crate::registry_routes::target;
use crate::registry_wire::{CallerWire, RvfError, ScopeCall, ScopeOut};
use crate::rvf_upload::{body_json, sha, Deps};
use crate::service::{self, Call};
use ruvector_edge_registry::validate::{validate, VecSegView};
use ruvector_edge_registry::{Caller, RegistryError, ValidatedRvf};
use ruvector_edge_store::{CallerContext, Op};
use ruvector_edge_tenancy::quota::limits::MAX_UPSERT_BATCH;
use ruvector_edge_tenancy::ProblemCode;
use serde::Deserialize;
use serde_json::{json, Value as Json};

/// Largest package imported (the blob is held in the isolate).
pub const MAX_IMPORT_BYTES: u64 = 32 << 20;
/// Most vector records (live or superseded) an imported package may carry.
pub const MAX_IMPORT_RECORDS: u64 = 1_000_000;
/// Durable Object subrequests one import request may spend on upserts
/// (under the Workers per-invocation limit, with room for auth, lookup,
/// pull and the blob read).
pub const IMPORT_SUBREQUEST_BUDGET: u64 = 800;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportBody {
    package: String,
    version: String,
    #[serde(default)]
    offset: u64,
}

/// Rows one request imports for a collection of `shards` shards.
pub fn rows_per_request(budget: u64, shards: u32) -> u64 {
    let per_batch = 5 + 2 * u64::from(shards.max(1));
    (budget / per_batch).max(1) * u64::from(MAX_UPSERT_BATCH)
}

/// One live record: (id, live-segment position, record index).
type Entry = (u64, u32, u32);

fn too_large(detail: &'static str) -> RvfError {
    RvfError::new(ProblemCode::PayloadTooLarge, detail)
}

fn payload<'a>(v: &ValidatedRvf, bytes: &'a [u8], seg: u32) -> Result<&'a [u8], RvfError> {
    let r = v.segments[seg as usize].payload_range();
    usize::try_from(r.start)
        .ok()
        .zip(usize::try_from(r.end).ok())
        .and_then(|(s, e)| bytes.get(s..e))
        .ok_or_else(|| RvfError::new(ProblemCode::ServerError, "segment outside the package"))
}

/// The live VEC_SEG payloads with their record counts; refuses packages
/// with more than `max_records` records before anything is indexed.
fn live_segments<'a>(
    v: &ValidatedRvf,
    bytes: &'a [u8],
    max_records: u64,
) -> Result<Vec<(&'a [u8], usize)>, RvfError> {
    let mut out = Vec::with_capacity(v.live_vec_segments.len());
    let mut total = 0u64;
    for &i in &v.live_vec_segments {
        let p = payload(v, bytes, i)?;
        let n = VecSegView::decode(p, v.dim)
            .map_err(|e| RvfError::from(RegistryError::from(e)))?
            .len();
        total += n as u64;
        if total > max_records {
            return Err(too_large(
                "package has too many vector records for an import",
            ));
        }
        out.push((p, n));
    }
    Ok(out)
}

/// Record `k` of a VEC_SEG payload holding `n` records of `dim`.
fn record(p: &[u8], n: usize, dim: usize, k: usize) -> &[u8] {
    let stride = 8 + 4 * dim;
    let head = p.len() - n * stride;
    &p[head + k * stride..head + (k + 1) * stride]
}

/// Index of the live rows, ordered by id: deleted ids removed, the last
/// record of an id (directory order, then record order) wins.
pub(crate) fn live_index(
    v: &ValidatedRvf,
    bytes: &[u8],
    max_records: u64,
) -> Result<Vec<Entry>, RvfError> {
    let segs = live_segments(v, bytes, max_records)?;
    let dim = usize::from(v.dim);
    let mut idx: Vec<Entry> = Vec::with_capacity(segs.iter().map(|s| s.1).sum());
    for (s, (p, n)) in segs.iter().enumerate() {
        for k in 0..*n {
            let r = record(p, *n, dim, k);
            let id = u64::from_le_bytes(r[..8].try_into().unwrap_or([0; 8]));
            idx.push((id, s as u32, k as u32));
        }
    }
    // Stable by (id, position): the last of each id run is the winner.
    idx.sort_unstable();
    let mut out: Vec<Entry> = Vec::with_capacity(idx.len());
    for e in idx {
        match out.last_mut() {
            Some(last) if last.0 == e.0 => *last = e,
            _ => out.push(e),
        }
    }
    out.retain(|e| v.deleted_ids.binary_search(&e.0).is_err());
    Ok(out)
}

fn values(p: &[u8], n: usize, dim: usize, k: usize) -> Vec<f32> {
    record(p, n, dim, k)[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Import `package@version` into `collection`, one bounded window.
pub async fn import<B: Backend, R: RegistryRpc, S: BlobStore>(
    d: &Deps<'_, B, R, S>,
    ctx: &CallerContext,
    c: &Caller,
    collection: &str,
    body: &[u8],
    now: u64,
) -> Result<(u16, Json), RvfError> {
    import_budgeted(d, ctx, c, collection, body, now, IMPORT_SUBREQUEST_BUDGET).await
}

/// [`import`] with an explicit subrequest budget.
pub(crate) async fn import_budgeted<B: Backend, R: RegistryRpc, S: BlobStore>(
    d: &Deps<'_, B, R, S>,
    ctx: &CallerContext,
    c: &Caller,
    collection: &str,
    body: &[u8],
    now: u64,
    budget: u64,
) -> Result<(u16, Json), RvfError> {
    let ib: ImportBody = body_json(body)?;
    let t = target(&ib.package, &ib.version)?;
    let access = service::authorize_op(d.b, ctx, Op::VectorUpsert).await?;
    let call = Call {
        b: d.b,
        ctx,
        dry_run: false,
        now,
    };
    let entry = service::lookup(&call, collection).await?;
    let pull = ScopeCall::Pull {
        caller: CallerWire::from(c),
        at: t.coords(),
    };
    let ScopeOut::Pull { manifest, blob } = scope(d.r, t.name.scope(), pull).await? else {
        return Err(RvfError::unexpected());
    };
    if u32::from(manifest.dim) != entry.cfg.dim {
        return Err(RvfError::new(
            ProblemCode::DimensionMismatch,
            "package dimension differs from the collection's",
        ));
    }
    if manifest.metric != entry.cfg.metric {
        return Err(RvfError::new(
            ProblemCode::InvalidRequest,
            "package metric differs from the collection's",
        ));
    }
    if manifest.total_size > MAX_IMPORT_BYTES {
        return Err(too_large("package too large for an import"));
    }
    let bytes =
        d.s.get_range(&blob, 0, manifest.total_size)
            .await?
            .ok_or_else(RvfError::storage)?;
    if bytes.len() as u64 != manifest.total_size || sha(&bytes) != manifest.sha256 {
        return Err(RvfError::new(
            ProblemCode::ServerError,
            "stored package does not match its manifest",
        ));
    }
    let v =
        validate(&bytes, d.cfg.validation).map_err(|e| RvfError::from(RegistryError::from(e)))?;
    let index = live_index(&v, &bytes, MAX_IMPORT_RECORDS)?;
    let segs = live_segments(&v, &bytes, MAX_IMPORT_RECORDS)?;
    let total = index.len() as u64;
    let start = ib.offset.min(total);
    let end = start
        .saturating_add(rows_per_request(budget, entry.shard_count))
        .min(total);
    let dim = usize::from(v.dim);
    let mut imported = 0u64;
    for chunk in index[start as usize..end as usize].chunks(MAX_UPSERT_BATCH as usize) {
        let vectors: Vec<Json> = chunk
            .iter()
            .map(|&(id, s, k)| {
                let (p, n) = segs[s as usize];
                json!({ "id": id.to_string(), "values": values(p, n, dim, k as usize) })
            })
            .collect();
        let raw = json!({ "collection": collection, "vectors": vectors }).to_string();
        service::run(&call, access, Op::VectorUpsert, &raw)
            .await
            .map_err(|e| {
                let mut r = RvfError::from(e);
                r.detail = format!("{} (resume with offset {})", r.detail, start + imported);
                r
            })?;
        imported += chunk.len() as u64;
    }
    Ok((
        200,
        json!({
            "collection": collection,
            "package": t.name.to_string(),
            "version": t.version.to_string(),
            "sha256": hex::encode(manifest.sha256),
            "dim": manifest.dim,
            "metric": manifest.metric,
            "total": total,
            "offset": start,
            "imported": imported,
            "next_offset": (end < total).then_some(end),
        }),
    ))
}

/// Every live row decoded through [`live_index`] (tests compare it with
/// the registry crate's reference decoder).
#[cfg(test)]
pub(crate) fn live_rows(
    v: &ValidatedRvf,
    bytes: &[u8],
    max_records: u64,
) -> Result<Vec<(u64, Vec<f32>)>, RvfError> {
    let segs = live_segments(v, bytes, max_records)?;
    let dim = usize::from(v.dim);
    Ok(live_index(v, bytes, max_records)?
        .into_iter()
        .map(|(id, s, k)| {
            let (p, n) = segs[s as usize];
            (id, values(p, n, dim, k as usize))
        })
        .collect())
}
