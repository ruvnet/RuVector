//! `POST /v1/collections/{c}/snapshots/{id}:restore` (ADR-351 §6.3, §7.2
//! M3; `ruvector:admin` + owner).
//!
//! The snapshot id resolves only in the caller's own ledger; each shard's
//! manifest is located from the caller's witness entry (never a listing)
//! and `RestoreSession` checks tenant, target, schema, root, the chain up to
//! the ledger's trusted head, quota, then every chunk (size, sha256,
//! segments, order). Rows go to a staging table on each shard (one live
//! `rid` per shard: a new restore drops what an interrupted one left); only
//! when **every** shard staged cleanly is the growth admitted and each
//! shard swapped in one DO turn.
//!
//! **Journal.** The swap is atomic per shard, not across shards, so a
//! restore is journalled in the ledger (`Ns::Restore`, key = uid): written
//! (compare-and-swap, so two restores of one collection cannot overlap)
//! before staging, re-written **in the same DO turn** as the growth
//! admission, and removed in the same DO turn as the final usage
//! correction. Each shard records `(rid, removed, added)` in the turn that
//! commits it. A restore that fails part-way — or is cancelled (client
//! gone, CPU limit) — is settled from those records: by itself on a commit
//! error, else by the next restore of the collection (`409 restore in
//! progress` while an admitted journal is younger than
//! [`RESTORE_LEASE_MS`]): staged rows are dropped and usage is corrected to
//! what the shards really changed, exactly once.

use crate::m3_ctx::{dim16, snap_err, snap_metric, M3};
use crate::m3_ports::{Blob, Queues};
use crate::m3_wire::{
    kv_get, ledger3, shard3, EntryWire, M3Backend, M3LedgerCall, M3LedgerOut, M3ShardCall,
    M3ShardOut, Ns, RowWire,
};
use crate::service::{count_of, shard_meta, unexpected};
use crate::snapshots::{snap_key, SERVICE};
use crate::wire::{ActorWire, CollectionWire, DeltaWire, ShardCall, ShardOut};
use ruvector_edge_auth::Capability;
use ruvector_edge_snapshot::{
    snapshot_key, ChainProof, RestoreQuota, RestoreSession, RestoreTarget, Row, ShardRef,
    SignaturePolicy, WitnessEntry,
};
use ruvector_edge_store::{ErrorCode, OpError};
use ruvector_edge_tenancy::quota::limits::M1_SHARD_FLOAT_CAP;
use ruvector_edge_tenancy::{DoMeta, QuotaDelta, Role};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

/// Rows per staging call.
pub const STAGE_ROWS: usize = 512;
/// Largest restore (chunk bytes per shard).
pub const MAX_RESTORE_BYTES: u64 = 256 << 20;
/// An admitted restore journal younger than this is a live restore.
pub const RESTORE_LEASE_MS: u64 = 15 * 60 * 1000;

/// A restore in flight (ledger record `Ns::Restore/{uid}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Journal {
    /// Staging id on every shard.
    pub rid: String,
    /// Snapshot epoch being restored.
    pub epoch: u64,
    /// Unix ms the restore started.
    pub started_ms: u64,
    /// Growth admitted `(vectors, float_budget)`, once admitted.
    #[serde(default)]
    pub admitted: Option<(i64, i64)>,
}

fn in_progress() -> OpError {
    OpError::new(ErrorCode::Conflict, "restore in progress")
}

async fn swap<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    uid: &str,
    expect: Option<&str>,
    value: Option<&Journal>,
    admit: Option<QuotaDelta>,
    adjust: Option<QuotaDelta>,
) -> Result<Option<String>, OpError> {
    let raw = match value {
        Some(j) => Some(serde_json::to_string(j).map_err(|_| unexpected())?),
        None => None,
    };
    let call = M3LedgerCall::KvSwap {
        ns: Ns::Restore,
        key: uid.to_string(),
        expect: expect.map(str::to_string),
        value: raw.clone(),
        admit,
        adjust,
        now: m.now_ms / 1000,
    };
    ledger3(m.b, m.ctx.tenant_key(), call).await?;
    Ok(raw)
}

/// Settle journal `j` (stored as `raw`): drop its staged rows on every
/// shard, sum what its commits changed, and — in one ledger turn with the
/// journal's removal — correct usage by `applied − admitted`. Idempotent
/// until that last step succeeds.
pub async fn settle<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    e: &CollectionWire,
    raw: &str,
    j: &Journal,
) -> Result<(), OpError> {
    let mut net = DeltaWire::default();
    for i in count_of(e)?.indices() {
        let dm = shard_meta(m.ctx, e, i)?;
        let call = M3ShardCall::Settle { rid: j.rid.clone() };
        match shard3(m.b, &dm, call).await? {
            M3ShardOut::Settled { committed } => {
                if let Some((removed, added)) = committed {
                    net = net.plus(removed).plus(added);
                }
            }
            _ => return Err(unexpected()),
        }
    }
    let adjust = j.admitted.map(|(v, f)| {
        let q = net.quota();
        QuotaDelta {
            vectors: q.vectors - v,
            float_budget: q.float_budget - f,
            bytes: q.bytes,
            ..QuotaDelta::default()
        }
    });
    swap(m, &e.uid, Some(raw), None, None, adjust).await?;
    Ok(())
}

async fn chain<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
) -> Result<(Vec<WitnessEntry>, [u8; 32]), OpError> {
    match ledger3(m.b, m.ctx.tenant_key(), M3LedgerCall::Chain).await? {
        M3LedgerOut::Chain { entries, head } => {
            let es = entries
                .iter()
                .map(EntryWire::to_entry)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((es, crate::m3_wire::unhex32(&head)?))
        }
        _ => Err(unexpected()),
    }
}

/// Validate one shard's snapshot and stage its rows under `rid`.
#[allow(clippy::too_many_arguments)]
async fn stage_shard<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    e: &CollectionWire,
    dm: &DoMeta,
    shard: u16,
    epoch: u64,
    entries: &[WitnessEntry],
    head: [u8; 32],
    rid: &str,
) -> Result<u64, OpError> {
    let tenant = m.ctx.tenant_key().as_str();
    let entry = entries
        .iter()
        .rev()
        .find(|w| w.collection_uid == e.uid && w.shard == shard && w.epoch == epoch)
        .ok_or(OpError::new(ErrorCode::Conflict, "snapshot not witnessed"))?;
    let key = entry.manifest_key(tenant, SERVICE);
    let obj = m
        .blob
        .get(&key)
        .await?
        .ok_or(OpError::new(ErrorCode::Conflict, "snapshot object missing"))?;
    let dim = dim16(e)?;
    let target = RestoreTarget {
        shard: ShardRef::new(tenant, SERVICE, &e.uid, shard).map_err(snap_err)?,
        dim: Some(dim),
        metric: Some(snap_metric(e.cfg.metric)),
    };
    let quota = RestoreQuota {
        max_rows: M1_SHARD_FLOAT_CAP / u64::from(dim.max(1)),
        max_bytes: MAX_RESTORE_BYTES,
    };
    let proof = ChainProof::from_genesis(tenant, entries, head);
    let mut s = RestoreSession::begin(&obj, &target, quota, proof, SignaturePolicy::NotRequired)
        .map_err(snap_err)?;
    let chunks = s.manifest().manifest.chunks.len() as u32;
    let mut rows: Vec<Row> = Vec::new();
    for n in 0..chunks {
        let key = snapshot_key(&s.manifest().manifest, n);
        let bytes = m
            .blob
            .get(&key)
            .await?
            .ok_or(OpError::new(ErrorCode::Conflict, "snapshot chunk missing"))?;
        rows.clear();
        s.accept_chunk(n, &bytes, &mut rows).map_err(snap_err)?;
        for part in rows.chunks(STAGE_ROWS) {
            let call = M3ShardCall::Stage {
                rid: rid.to_string(),
                rows: part.iter().map(RowWire::from_row).collect(),
            };
            shard3(m.b, dm, call).await?;
        }
    }
    Ok(s.finish().map_err(snap_err)?.rows)
}

/// `POST /v1/collections/{c}/snapshots/{id}:restore` (admin + owner).
pub async fn restore<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    collection: &str,
    id: &str,
) -> Result<(u16, Json), OpError> {
    m.require(Capability::Admin, Role::Owner).await?;
    let e = m.collection(collection).await?;
    let epoch: u64 = id.parse().map_err(|_| OpError::not_found())?;
    let t = m.ctx.tenant_key();
    let rec = kv_get(m.b, t, Ns::Snapshot, &snap_key(&e.uid, epoch))
        .await?
        .ok_or(OpError::not_found())?;
    let rec: Json = serde_json::from_str(&rec).map_err(|_| unexpected())?;
    let shards = count_of(&e)?;
    if rec["shards"].as_u64() != Some(u64::from(shards.get())) {
        return Err(OpError::new(ErrorCode::Conflict, "shard count changed"));
    }
    m.charge(1 + u64::from(shards.get())).await?;
    // An earlier restore of this collection that never finished.
    if let Some(raw) = kv_get(m.b, t, Ns::Restore, &e.uid).await? {
        let j: Journal = serde_json::from_str(&raw).map_err(|_| unexpected())?;
        let live = m.now_ms.saturating_sub(j.started_ms) < RESTORE_LEASE_MS;
        if j.admitted.is_some() && live {
            return Err(in_progress());
        }
        settle(m, &e, &raw, &j).await?;
    }
    let (entries, head) = chain(m).await?;
    let j = Journal {
        rid: format!("r{epoch}x{}", m.now_ms),
        epoch,
        started_ms: m.now_ms,
        admitted: None,
    };
    let raw = swap(m, &e.uid, None, Some(&j), None, None)
        .await
        .map_err(|err| match err.code {
            ErrorCode::Conflict => in_progress(),
            _ => err,
        })?
        .unwrap_or_default();
    let mut staged = Vec::new();
    for i in shards.indices() {
        let dm = shard_meta(m.ctx, &e, i)?;
        let shard = i.get() as u16;
        match stage_shard(m, &e, &dm, shard, epoch, &entries, head, &j.rid).await {
            Ok(n) => staged.push((dm, n)),
            Err(err) => {
                let _settled = settle(m, &e, &raw, &j).await;
                return Err(err);
            }
        }
    }
    commit_all(m, &e, &j, &raw, &staged).await
}

async fn commit_all<B: M3Backend, R: Blob, Q: Queues>(
    m: &M3<'_, B, R, Q>,
    e: &CollectionWire,
    j: &Journal,
    raw: &str,
    staged: &[(DoMeta, u64)],
) -> Result<(u16, Json), OpError> {
    let actor = ActorWire {
        sub: m.ctx.sub().to_string(),
        jti: m.ctx.jti().to_string(),
        family_id: m.ctx.family_id().to_string(),
        act_sub: m.ctx.act_sub().map(str::to_string),
    };
    let mut current = 0u64;
    for (dm, _) in staged {
        match crate::backend::shard(m.b, dm, ShardCall::Stats).await {
            Ok(ShardOut::Stats { count, .. }) => current += count,
            Ok(_) => return Err(unexpected()),
            Err(err) => {
                let _settled = settle(m, e, raw, j).await;
                return Err(err.into_op());
            }
        }
    }
    let total: u64 = staged.iter().map(|(_, n)| n).sum();
    let grow =
        i64::try_from(total).unwrap_or(i64::MAX) - i64::try_from(current).unwrap_or(i64::MAX);
    let admitted = QuotaDelta {
        vectors: grow.max(0),
        float_budget: grow.max(0).saturating_mul(i64::from(e.cfg.dim)),
        ..QuotaDelta::default()
    };
    // Admit the growth (limits enforced) and record it, in one ledger turn,
    // before any shard changes.
    let j2 = Journal {
        admitted: Some((admitted.vectors, admitted.float_budget)),
        ..j.clone()
    };
    let raw2 = match swap(m, &e.uid, Some(raw), Some(&j2), Some(admitted), None).await {
        Ok(r) => r.unwrap_or_default(),
        Err(err) => {
            let _settled = settle(m, e, raw, j).await;
            return Err(err);
        }
    };
    let (mut net, mut rows) = (DeltaWire::default(), 0u64);
    for (dm, n) in staged {
        let call = M3ShardCall::Commit {
            rid: j.rid.clone(),
            cfg: e.cfg.clone(),
            rows: *n,
            actor: actor.clone(),
            now: m.now_ms / 1000,
        };
        let err = match shard3(m.b, dm, call).await {
            Ok(M3ShardOut::Committed { removed, added, .. }) => {
                net = net.plus(removed).plus(added);
                rows += n;
                continue;
            }
            Ok(_) => unexpected(),
            Err(err) => err,
        };
        // Usage follows what the shards really changed (their commit
        // records, including a commit whose reply was lost). If this fails
        // too, the next restore of the collection settles it.
        let _settled = settle(m, e, &raw2, &j2).await;
        return Err(err);
    }
    let applied = net.quota();
    let delta = QuotaDelta {
        vectors: applied.vectors - admitted.vectors,
        float_budget: applied.float_budget - admitted.float_budget,
        bytes: applied.bytes,
        ..QuotaDelta::default()
    };
    swap(m, &e.uid, Some(&raw2), None, None, Some(delta)).await?;
    let call = M3LedgerCall::NextEpoch { uid: e.uid.clone() };
    let new_epoch = match ledger3(m.b, m.ctx.tenant_key(), call).await? {
        M3LedgerOut::Epoch { epoch } => epoch,
        _ => return Err(unexpected()),
    };
    Ok((
        200,
        json!({
            "restored": j.epoch.to_string(),
            "rows": rows,
            "epoch": new_epoch,
            "shards": staged.len(),
            "delta": net.to_json(),
        }),
    ))
}
