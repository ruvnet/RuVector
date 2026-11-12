//! rv-snapshot (ADR-351 §3, §6.3, §7.2 M3): `POST …/snapshots` and
//! `GET …/snapshots`; restore is `restore`.
//!
//! One epoch per collection snapshot (ledger counter). Each shard is read
//! in consistent id-ordered pages (the snapshot aborts `409` if its
//! `write_seq` moves mid-read), streamed through `SnapshotWriter`, and every
//! chunk is PUT to R2 the moment it is produced
//! (`snapshots/{tenant}/vector/{uid}/{shard}/{epoch:020}-{id12}/seg-{n}.rvf`);
//! then the manifest, then the ledger appends the manifest root to the
//! tenant's witness chain. The manifest's `audit_head` ([`shard_head`])
//! binds the tenant's shipped-audit chain position (`audit::head`: next
//! object seq and last line hash) to the shard state it was read at
//! (`write_seq`). A failed snapshot still consumes its epoch and may leave
//! unreferenced chunks under `snapshots/`. Signatures:
//! `SignaturePolicy::NotRequired` (no signing key is configured; root +
//! witness chain still bind the bytes). A collection over the synchronous
//! budget ([`crate::sync_budget`]) is refused `413` before the epoch is
//! taken.

use crate::m3_ctx::{dim16, snap_err, snap_metric, M3};
use crate::m3_ports::{Blob, Queues};
use crate::m3_wire::{
    kv_put, kv_range, ledger3, shard3, EntryWire, M3Backend, M3LedgerCall, M3LedgerOut,
    M3ShardCall, M3ShardOut, Ns, RowWire,
};
use crate::service::{count_of, shard_meta, unexpected};
use crate::sync_budget::{self, Meter};
use crate::wire::CollectionWire;
use base64ct::{Base64, Encoding};
use ruvector_edge_auth::Capability;
use ruvector_edge_snapshot::{
    snapshot_key, FixedClock, ShardRef, SnapshotLimits, SnapshotSpec, SnapshotWriter,
};
use ruvector_edge_store::{ErrorCode, OpError};
use ruvector_edge_tenancy::{DoMeta, Role};
use serde_json::{json, Value as Json};

/// Service slug in snapshot keys.
pub const SERVICE: &str = "vector";
/// Rows per shard page.
pub const PAGE_ROWS: u32 = 512;

/// The `audit_head` recorded in a shard's manifest (and, combined, in an
/// export witness): `sha256("rvedge-shard-head-v2" ‖ tenant ‖ uid ‖ shard ‖
/// write_seq ‖ audit next seq ‖ audit chain head)`.
pub fn shard_head(
    tenant: &str,
    uid: &str,
    shard: u32,
    write_seq: u64,
    audit: &(u64, [u8; 32]),
) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"rvedge-shard-head-v2\0");
    for p in [tenant.as_bytes(), uid.as_bytes()] {
        h.update((p.len() as u16).to_le_bytes());
        h.update(p);
    }
    h.update(shard.to_le_bytes());
    h.update(write_seq.to_le_bytes());
    h.update(audit.0.to_le_bytes());
    h.update(audit.1);
    h.finalize().into()
}

/// One consistent page of a shard.
pub async fn page<B: M3Backend>(
    b: &B,
    dm: &DoMeta,
    after: Option<String>,
) -> Result<(u64, u64, Vec<RowWire>), OpError> {
    let call = M3ShardCall::Page {
        after,
        limit: PAGE_ROWS,
    };
    match shard3(b, dm, call).await? {
        M3ShardOut::Page {
            write_seq,
            count,
            rows,
        } => Ok((write_seq, count, rows)),
        _ => Err(unexpected()),
    }
}

fn moved() -> OpError {
    OpError::new(ErrorCode::Conflict, "collection changed during read, retry")
}

pub(crate) fn snap_key(uid: &str, epoch: u64) -> String {
    format!("{uid}/{epoch:020}")
}

async fn snapshot_shard<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    e: &CollectionWire,
    i: ruvector_edge_tenancy::ShardIndex,
    epoch: u64,
    audit: &(u64, [u8; 32]),
    meter: &mut Meter,
) -> Result<(EntryWire, u64, u64), OpError> {
    let tenant = m.ctx.tenant_key().as_str();
    let dm = shard_meta(m.ctx, e, i)?;
    let (seq0, _, mut rows) = page(m.b, &dm, None).await?;
    let spec = SnapshotSpec {
        shard: ShardRef::new(tenant, SERVICE, &e.uid, i.get() as u16).map_err(snap_err)?,
        epoch,
        dim: dim16(e)?,
        metric: snap_metric(e.cfg.metric),
        audit_head: shard_head(tenant, &e.uid, i.get(), seq0, audit),
    };
    let mut w = SnapshotWriter::new(spec, SnapshotLimits::default()).map_err(snap_err)?;
    let mut bytes = 0u64;
    loop {
        meter.add(rows.len())?;
        let last = rows.last().map(|r| r.id.clone());
        for r in rows {
            let row = r.into_row(dim16(e)?)?;
            if let Some(c) = w.push(&row).map_err(snap_err)? {
                bytes += c.bytes.len() as u64;
                m.blob
                    .put(&w.chunk_key(c.index), c.bytes, Some(c.sha256))
                    .await?;
            }
        }
        let Some(after) = last else { break };
        let (seq, _, next) = page(m.b, &dm, Some(after)).await?;
        if seq != seq0 {
            return Err(moved());
        }
        rows = next;
    }
    let manifest_key = w.manifest_key();
    let (tail, sealed) = w.finish(&FixedClock(m.now_ms), None).map_err(snap_err)?;
    for c in tail {
        bytes += c.bytes.len() as u64;
        let key = snapshot_key(&sealed.manifest, c.index);
        m.blob.put(&key, c.bytes, Some(c.sha256)).await?;
    }
    let obj = sealed.to_object();
    m.blob.put(&manifest_key, obj.clone(), None).await?;
    let call = M3LedgerCall::WitnessAppend {
        manifest: Base64::encode_string(&obj),
    };
    match ledger3(m.b, m.ctx.tenant_key(), call).await? {
        M3LedgerOut::Witnessed { entry } => Ok((entry, sealed.manifest.row_count, bytes)),
        _ => Err(unexpected()),
    }
}

/// `POST /v1/collections/{c}/snapshots` (write + editor) → 202.
pub async fn create<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    collection: &str,
) -> Result<(u16, Json), OpError> {
    m.require(Capability::Write, Role::Editor).await?;
    let e = m.collection(collection).await?;
    crate::quant_route::refuse_m3(&e)?;
    let shards = count_of(&e)?;
    m.charge(1 + u64::from(shards.get())).await?;
    // Before the epoch is taken: a refused snapshot burns nothing.
    sync_budget::check_live(m.b, m.ctx, &e).await?;
    let call = M3LedgerCall::NextEpoch { uid: e.uid.clone() };
    let epoch = match ledger3(m.b, m.ctx.tenant_key(), call).await? {
        M3LedgerOut::Epoch { epoch } => epoch,
        _ => return Err(unexpected()),
    };
    let audit = crate::audit::head(m.b, m.ctx.tenant_key()).await?;
    let (mut rows, mut bytes, mut roots) = (0u64, 0u64, Vec::new());
    let mut meter = Meter::new(&e);
    for i in shards.indices() {
        let (entry, n, b) = snapshot_shard(m, &e, i, epoch, &audit, &mut meter).await?;
        rows += n;
        bytes += b;
        roots.push(entry.root);
    }
    let rec = json!({
        "snapshot_id": epoch.to_string(),
        "epoch": epoch,
        "state": "complete",
        "shards": shards.get(),
        "rows": rows,
        "bytes": bytes,
        "roots": roots,
        "created_at": m.now_ms / 1000,
        "created_by": m.ctx.sub(),
    });
    let t = m.ctx.tenant_key();
    kv_put(
        m.b,
        t,
        Ns::Snapshot,
        &snap_key(&e.uid, epoch),
        rec.to_string(),
    )
    .await?;
    Ok((202, rec))
}

/// `GET /v1/collections/{c}/snapshots` (read + viewer).
pub async fn list<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    collection: &str,
) -> Result<(u16, Json), OpError> {
    m.require(Capability::Read, Role::Viewer).await?;
    let e = m.collection(collection).await?;
    m.charge(1).await?;
    let items = kv_range(
        m.b,
        m.ctx.tenant_key(),
        Ns::Snapshot,
        format!("{}/", e.uid),
        format!("{}0", e.uid),
        100,
    )
    .await?;
    let snaps: Vec<Json> = items
        .iter()
        .filter_map(|(_, v)| serde_json::from_str(v).ok())
        .collect();
    Ok((200, json!({ "snapshots": snaps })))
}
