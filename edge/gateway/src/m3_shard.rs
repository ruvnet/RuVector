//! `VectorShard` M3 core (ADR-351 §6.3 snapshots / restore): consistent
//! id-ordered pages for the snapshot writer and exporter, a staging table
//! for validated restore rows, and the restore commit.
//!
//! Every call first runs the M1 `Stats` call through `shard_core::handle`
//! in the same DO turn: that opens the shard (write-through replay of any
//! torn op), asserts the stored identity (mismatch → 404) and yields the
//! `write_seq` a page was read at. Pages then read `vectors` directly in
//! byte order of `id` (SQLite `BINARY` = the snapshot writer's order).
//!
//! **Commit** replaces the shard's rows with exactly the staged set in one
//! synchronous DO turn (one coalesced commit): ids absent from the snapshot
//! are deleted, then every staged row is upserted in `MAX_UPSERT_BATCH`
//! slices, each planned after the previous one applied — all through the
//! M1 write path, so the op log, filter index, iids and usage totals stay
//! consistent. Every refusal a plan can raise is checked before the first
//! delete (count, float cap), so only a storage failure can stop a commit
//! part-way. The same turn records `(rid, removed, added)` as the shard's
//! last commit, which `Settle` reports so an interrupted restore can correct
//! usage exactly (`restore`); `Settle` also drops the rid's staged rows, and
//! the first `Stage` of a new rid drops any other rid's leftovers.

use crate::m3_wire::{col_text, reply, M3ShardCall, M3ShardOut, M3ShardRequest, RowWire};
use crate::shard_core::{self, ShardHost};
use crate::wire::{ActorWire, CfgWire, DeltaWire, ShardCall, ShardOut, ShardRequest};
use ruvector_edge_store::shard::{UpsertRow, MAX_DELETE_IDS};
use ruvector_edge_store::{ErrorCode, OpError, SqlStore, Value};
use ruvector_edge_tenancy::quota::limits::MAX_UPSERT_BATCH;
use serde_json::Value as Json;
use std::collections::BTreeSet;

const STAGE_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS m3_stage (rid TEXT, id TEXT, f32 BLOB, \
                            metadata TEXT, PRIMARY KEY (rid, id))";
const STAGE_PUT: &str =
    "INSERT OR REPLACE INTO m3_stage (rid, id, f32, metadata) VALUES (?, ?, ?, ?)";
const STAGE_PAGE: &str = "SELECT id, f32, metadata FROM m3_stage WHERE rid = ? AND id > ? \
                          ORDER BY id LIMIT ?";
const STAGE_DROP: &str = "DELETE FROM m3_stage WHERE rid = ?";
const STAGE_ANY: &str = "SELECT id FROM m3_stage WHERE rid = ? LIMIT ?";
const STAGE_DROP_OTHERS: &str = "DELETE FROM m3_stage WHERE rid != ?";
const COMMIT_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS m3_commit (k TEXT PRIMARY KEY, rid TEXT, \
                             removed TEXT, added TEXT)";
const COMMIT_PUT: &str =
    "INSERT OR REPLACE INTO m3_commit (k, rid, removed, added) VALUES (?, ?, ?, ?)";
const COMMIT_GET: &str = "SELECT removed, added FROM m3_commit WHERE k = ? AND rid = ?";
const LAST: &str = "last";
const VEC_PAGE_BY_ID: &str = "SELECT id, f32, metadata FROM vectors WHERE deleted = ? AND id > ? \
                              ORDER BY id LIMIT ?";
const VEC_IDS: &str = "SELECT id FROM vectors WHERE id > ? ORDER BY id LIMIT ?";

/// Largest page.
pub const MAX_PAGE: u32 = 1024;
/// Largest `Stage` call.
pub const MAX_STAGE_ROWS: usize = 1024;

fn db(_: ruvector_edge_store::StoreError) -> OpError {
    OpError::new(ErrorCode::ShardUnavailable, "shard storage")
}

fn rid_ok(r: &str) -> bool {
    !r.is_empty()
        && r.len() <= 64
        && r.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Serve one encoded M3 request (the shard DO's `/m3` path).
pub fn serve(
    host: &mut ShardHost,
    key: &str,
    own_name: Option<&str>,
    store: &dyn SqlStore,
    body: &[u8],
) -> String {
    let r = serde_json::from_slice::<M3ShardRequest>(body)
        .map_err(|_| OpError::invalid("malformed m3 shard call"))
        .and_then(|req| handle(host, key, own_name, store, req));
    reply(r)
}

struct Ctx<'a> {
    host: &'a mut ShardHost,
    key: &'a str,
    own_name: Option<&'a str>,
    store: &'a dyn SqlStore,
    tenant_key: String,
    uid: String,
    shard: u32,
}

impl Ctx<'_> {
    fn m1(&mut self, call: ShardCall) -> Result<ShardOut, OpError> {
        let req = ShardRequest {
            tenant_key: self.tenant_key.clone(),
            uid: self.uid.clone(),
            shard: self.shard,
            call,
        };
        shard_core::handle(self.host, self.key, self.own_name, self.store, req)
    }
}

/// Run one call.
pub fn handle(
    host: &mut ShardHost,
    key: &str,
    own_name: Option<&str>,
    store: &dyn SqlStore,
    req: M3ShardRequest,
) -> Result<M3ShardOut, OpError> {
    let mut c = Ctx {
        host,
        key,
        own_name,
        store,
        tenant_key: req.tenant_key,
        uid: req.uid,
        shard: req.shard,
    };
    // Open + identity assertion + consistent counters, same DO turn.
    let (count, write_seq) = match c.m1(ShardCall::Stats)? {
        ShardOut::Stats {
            count, write_seq, ..
        } => (count, write_seq),
        _ => return Err(crate::service::unexpected()),
    };
    store.exec(STAGE_SCHEMA, &[]).map_err(db)?;
    store.exec(COMMIT_SCHEMA, &[]).map_err(db)?;
    match req.call {
        M3ShardCall::Page { after, limit } => {
            let p = [
                Value::Int(0),
                after.unwrap_or_default().into(),
                Value::Int(i64::from(limit.clamp(1, MAX_PAGE))),
            ];
            let rows = store.query(VEC_PAGE_BY_ID, &p).map_err(db)?;
            let mut out = Vec::with_capacity(rows.len());
            for r in &rows {
                out.push(stored_row(r)?);
            }
            Ok(M3ShardOut::Page {
                write_seq,
                count,
                rows: out,
            })
        }
        M3ShardCall::Stage { rid, rows } => {
            if !rid_ok(&rid) || rows.len() > MAX_STAGE_ROWS {
                return Err(OpError::invalid("stage"));
            }
            // One live restore per shard: the first page of a new `rid` drops
            // whatever an interrupted restore left staged.
            if store
                .query(STAGE_ANY, &[rid.as_str().into(), Value::Int(1)])
                .map_err(db)?
                .is_empty()
            {
                store
                    .exec(STAGE_DROP_OTHERS, &[rid.as_str().into()])
                    .map_err(db)?;
            }
            for r in rows {
                let bytes = r.f32_bytes()?;
                let meta = r.metadata.map_or(Value::Null, Value::Text);
                let p = [rid.as_str().into(), r.id.into(), Value::Blob(bytes), meta];
                store.exec(STAGE_PUT, &p).map_err(db)?;
            }
            Ok(M3ShardOut::Done)
        }
        M3ShardCall::Unstage { rid } => {
            if !rid_ok(&rid) {
                return Err(OpError::invalid("stage"));
            }
            store.exec(STAGE_DROP, &[rid.into()]).map_err(db)?;
            Ok(M3ShardOut::Done)
        }
        M3ShardCall::Commit {
            rid,
            cfg,
            rows,
            actor,
            now,
        } => commit(&mut c, &rid, &cfg, rows, actor, now),
        M3ShardCall::Settle { rid } => {
            if !rid_ok(&rid) {
                return Err(OpError::invalid("stage"));
            }
            store.exec(STAGE_DROP, &[rid.as_str().into()]).map_err(db)?;
            let p = [LAST.into(), rid.as_str().into()];
            let rows = store.query(COMMIT_GET, &p).map_err(db)?;
            let committed = match rows.first() {
                None => None,
                Some(r) => {
                    let bad = |_| OpError::new(ErrorCode::ServerError, "commit record");
                    let removed = serde_json::from_str(&col_text(r, 0)?).map_err(bad)?;
                    let added = serde_json::from_str(&col_text(r, 1)?).map_err(bad)?;
                    Some((removed, added))
                }
            };
            Ok(M3ShardOut::Settled { committed })
        }
    }
}

fn stored_row(r: &[Value]) -> Result<RowWire, OpError> {
    let id = col_text(r, 0)?;
    let blob = r
        .get(1)
        .and_then(Value::as_blob)
        .ok_or(OpError::new(ErrorCode::ServerError, "stored vector"))?;
    let meta = r.get(2).and_then(Value::as_text).map(str::to_string);
    Ok(RowWire::from_parts(id, blob, meta))
}

fn staged(store: &dyn SqlStore, rid: &str, dim: usize) -> Result<Vec<UpsertRow>, OpError> {
    let mut out = Vec::new();
    let mut after = String::new();
    loop {
        let p = [
            rid.into(),
            after.clone().into(),
            Value::Int(i64::from(MAX_PAGE)),
        ];
        let page = store.query(STAGE_PAGE, &p).map_err(db)?;
        for r in &page {
            let w = stored_row(r)?;
            let bytes = w.f32_bytes()?;
            if bytes.len() != dim * 4 {
                return Err(OpError::new(ErrorCode::DimensionMismatch, "staged row"));
            }
            let values = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let metadata = match &w.metadata {
                None => None,
                Some(t) => Some(
                    serde_json::from_str::<Json>(t)
                        .map_err(|_| OpError::invalid("staged metadata"))?,
                ),
            };
            after = w.id.clone();
            out.push(UpsertRow {
                id: w.id,
                values,
                metadata,
            });
        }
        if page.len() < MAX_PAGE as usize {
            return Ok(out);
        }
    }
}

fn current_ids(store: &dyn SqlStore) -> Result<Vec<String>, OpError> {
    let mut out: Vec<String> = Vec::new();
    loop {
        let after: String = out.last().cloned().unwrap_or_default();
        let p = [after.into(), Value::Int(i64::from(MAX_PAGE))];
        let page = store.query(VEC_IDS, &p).map_err(db)?;
        for r in &page {
            out.push(col_text(r, 0)?);
        }
        if page.len() < MAX_PAGE as usize {
            return Ok(out);
        }
    }
}

fn written(out: ShardOut) -> Result<(DeltaWire, u64), OpError> {
    match out {
        ShardOut::Written {
            delta, write_seq, ..
        } => Ok((delta, write_seq)),
        ShardOut::Failed { err, .. } => Err(err.into_op()),
        _ => Err(crate::service::unexpected()),
    }
}

fn commit(
    c: &mut Ctx<'_>,
    rid: &str,
    cfg: &CfgWire,
    expected: u64,
    actor: ActorWire,
    now: u64,
) -> Result<M3ShardOut, OpError> {
    if !rid_ok(rid) {
        return Err(OpError::invalid("stage"));
    }
    let dim = cfg.dim as usize;
    let rows = staged(c.store, rid, dim)?;
    if rows.len() as u64 != expected {
        return Err(OpError::new(ErrorCode::Conflict, "staged rows incomplete"));
    }
    let cap = cfg.to_config().float_cap;
    if (rows.len() as u64).saturating_mul(u64::from(cfg.dim)) > cap {
        return Err(OpError::new(ErrorCode::BudgetExceeded, "shard float cap"));
    }
    let keep: BTreeSet<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    let extras: Vec<String> = current_ids(c.store)?
        .into_iter()
        .filter(|id| !keep.contains(id.as_str()))
        .collect();
    let (mut removed, mut added, mut write_seq) =
        (DeltaWire::default(), DeltaWire::default(), 0u64);
    for ids in extras.chunks(MAX_DELETE_IDS) {
        let call = ShardCall::Delete {
            ids: ids.to_vec(),
            actor: actor.clone(),
            dry_run: false,
            now,
        };
        let (d, s) = written(c.m1(call)?)?;
        removed = removed.plus(d);
        write_seq = s;
    }
    let n = rows.len() as u64;
    let mut rows = rows;
    while !rows.is_empty() {
        let take = rows.len().min(MAX_UPSERT_BATCH as usize);
        let batch: Vec<UpsertRow> = rows.drain(..take).collect();
        let plan = ShardCall::Plan {
            cfg: cfg.clone(),
            rows: batch.clone(),
        };
        let admitted = match c.m1(plan)? {
            ShardOut::Planned { delta } => delta,
            _ => return Err(crate::service::unexpected()),
        };
        let apply = ShardCall::Apply {
            cfg: cfg.clone(),
            rows: batch,
            admitted,
            actor: actor.clone(),
            now,
        };
        let (d, s) = written(c.m1(apply)?)?;
        added = added.plus(d);
        write_seq = s;
    }
    c.store.exec(STAGE_DROP, &[rid.into()]).map_err(db)?;
    // Same DO turn as the swap: an interrupted restore can learn what this
    // shard really changed (`Settle`).
    let enc = |d: &DeltaWire| serde_json::to_string(d).map_err(|_| crate::service::unexpected());
    let p = [
        LAST.into(),
        rid.into(),
        enc(&removed)?.into(),
        enc(&added)?.into(),
    ];
    c.store.exec(COMMIT_PUT, &p).map_err(db)?;
    Ok(M3ShardOut::Committed {
        removed,
        added,
        write_seq,
        rows: n,
    })
}
