//! `TenantLedger` M3 core (ADR-351 §6.2 "jobs", §6.3 witness chain): the
//! per-tenant witness chain over snapshot manifests, snapshot epochs,
//! collection embedders, and small records (snapshots, exports, uploads,
//! jobs) in namespaced key-value rows. Pure and synchronous like
//! `ledger_core`: one call is one DO turn, one coalesced commit.
//!
//! Tables (additive, next to the M1 ledger tables):
//! `m3_meta(k, v)` — `tenant_key` (asserted on every call), `witness_head`,
//! `witness_next`, `epoch/{uid}`; `m3_kv(ns, k, v)`; `m3_witness(seq, …)`.

use crate::ledger_core;
use crate::m3_wire::{
    col_int, col_text, hex32, reply, unhex32, EntryWire, M3LedgerCall, M3LedgerOut,
    M3LedgerRequest, Ns,
};
use crate::wire::{LedgerCall, LedgerOut, LedgerRequest};
use base64ct::{Base64, Encoding};
use ruvector_edge_snapshot::{SealedManifest, WitnessChain};
use ruvector_edge_store::{EntropySource, ErrorCode, OpError, SqlStore, TenantLedger, Value};
use ruvector_edge_tenancy::{QuotaDelta, QuotaLimits, TenantKey};

const SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS m3_meta (k TEXT PRIMARY KEY, v TEXT)",
    "CREATE TABLE IF NOT EXISTS m3_kv (ns TEXT, k TEXT, v TEXT, PRIMARY KEY (ns, k))",
    "CREATE TABLE IF NOT EXISTS m3_witness (seq INTEGER PRIMARY KEY, prev TEXT, root TEXT, \
     uid TEXT, shard INTEGER, epoch INTEGER, audit_head TEXT, hash TEXT)",
];
const META_GET: &str = "SELECT v FROM m3_meta WHERE k = ?";
const META_PUT: &str = "INSERT OR REPLACE INTO m3_meta (k, v) VALUES (?, ?)";
const KV_GET: &str = "SELECT v FROM m3_kv WHERE ns = ? AND k = ?";
const KV_PUT: &str = "INSERT OR REPLACE INTO m3_kv (ns, k, v) VALUES (?, ?, ?)";
const KV_DEL: &str = "DELETE FROM m3_kv WHERE ns = ? AND k = ?";
const KV_RANGE: &str =
    "SELECT k, v FROM m3_kv WHERE ns = ? AND k >= ? AND k < ? ORDER BY k LIMIT ?";
const W_PUT: &str = "INSERT INTO m3_witness (seq, prev, root, uid, shard, epoch, audit_head, \
                     hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";
const W_ALL: &str =
    "SELECT seq, prev, root, uid, shard, epoch, audit_head, hash FROM m3_witness ORDER BY seq";

/// Largest record value (bytes).
pub const MAX_VALUE_BYTES: usize = 64 * 1024;
/// Largest key (bytes).
pub const MAX_KEY_BYTES: usize = 160;
/// Largest `KvRange` page.
pub const MAX_RANGE: u32 = 1000;

pub(crate) fn db(e: ruvector_edge_store::StoreError) -> OpError {
    let _ = e;
    OpError::new(ErrorCode::ShardUnavailable, "ledger storage")
}

pub(crate) fn meta_get(store: &dyn SqlStore, k: &str) -> Result<Option<String>, OpError> {
    let rows = store.query(META_GET, &[k.into()]).map_err(db)?;
    Ok(rows
        .first()
        .and_then(|r| r.first())
        .and_then(Value::as_text)
        .map(str::to_string))
}

pub(crate) fn meta_put(store: &dyn SqlStore, k: &str, v: String) -> Result<(), OpError> {
    store.exec(META_PUT, &[k.into(), v.into()]).map_err(db)?;
    Ok(())
}

/// Create the tables and assert (or record, on first use) the tenant.
fn guard(store: &dyn SqlStore, tenant: &TenantKey) -> Result<(), OpError> {
    for ddl in SCHEMA {
        store.exec(ddl, &[]).map_err(db)?;
    }
    match meta_get(store, "tenant_key")? {
        Some(t) if t == tenant.as_str() => Ok(()),
        Some(_) => Err(OpError::not_found()),
        None => meta_put(store, "tenant_key", tenant.as_str().to_string()),
    }
}

fn key_ok(k: &str) -> bool {
    !k.is_empty() && k.len() <= MAX_KEY_BYTES && k.bytes().all(|b| b.is_ascii_graphic())
}

/// Serve one encoded M3 request (the ledger DO's `/m3` path). `slot` is
/// the same resident M1 ledger the `/rpc` path uses.
pub fn serve(
    slot: &mut Option<TenantLedger>,
    store: &dyn SqlStore,
    limits: QuotaLimits,
    body: &[u8],
    entropy: &dyn EntropySource,
) -> String {
    let r = serde_json::from_slice::<M3LedgerRequest>(body)
        .map_err(|_| OpError::invalid("malformed m3 ledger call"))
        .and_then(|req| handle(slot, store, limits, req, entropy));
    reply(r)
}

fn m1(
    slot: &mut Option<TenantLedger>,
    store: &dyn SqlStore,
    limits: QuotaLimits,
    tenant: &str,
    call: LedgerCall,
    entropy: &dyn EntropySource,
) -> Result<LedgerOut, OpError> {
    let req = LedgerRequest {
        tenant_key: tenant.to_string(),
        call,
    };
    ledger_core::handle(slot, store, limits, req, entropy)
}

/// Run one call.
pub fn handle(
    slot: &mut Option<TenantLedger>,
    store: &dyn SqlStore,
    limits: QuotaLimits,
    req: M3LedgerRequest,
    entropy: &dyn EntropySource,
) -> Result<M3LedgerOut, OpError> {
    let tenant =
        TenantKey::parse(&req.tenant_key).map_err(|_| OpError::invalid("malformed tenant"))?;
    guard(store, &tenant)?;
    let t = tenant.as_str();
    Ok(match req.call {
        M3LedgerCall::KvPut { ns, key, value } => {
            if !key_ok(&key) || value.len() > MAX_VALUE_BYTES {
                return Err(OpError::invalid("record too large"));
            }
            let p = [ns.as_str().into(), key.into(), value.into()];
            store.exec(KV_PUT, &p).map_err(db)?;
            M3LedgerOut::Done
        }
        M3LedgerCall::KvGet { ns, key } => {
            let rows = store
                .query(KV_GET, &[ns.as_str().into(), key.into()])
                .map_err(db)?;
            let value = rows
                .first()
                .and_then(|r| r.first())
                .and_then(Value::as_text)
                .map(str::to_string);
            M3LedgerOut::Value { value }
        }
        M3LedgerCall::KvRange {
            ns,
            from,
            to,
            limit,
        } => {
            let p = [
                ns.as_str().into(),
                from.into(),
                to.into(),
                Value::Int(i64::from(limit.clamp(1, MAX_RANGE))),
            ];
            let rows = store.query(KV_RANGE, &p).map_err(db)?;
            let mut items = Vec::with_capacity(rows.len());
            for r in &rows {
                let k = col_text(r, 0)?;
                let v = col_text(r, 1)?;
                items.push((k, v));
            }
            M3LedgerOut::Values { items }
        }
        M3LedgerCall::CreateWithEmbedder {
            spec,
            model,
            sub,
            now,
        } => {
            // One DO turn: nothing below can interleave with another call.
            let call = LedgerCall::ValidateCreate { spec: spec.clone() };
            m1(slot, store, limits, t, call, entropy)?;
            let one = QuotaDelta {
                ops: 1,
                ..QuotaDelta::default()
            };
            let admit = LedgerCall::Admit {
                delta: one,
                work_units: 1,
                now,
            };
            m1(slot, store, limits, t, admit, entropy)?;
            let create = LedgerCall::CreateCollection { spec, sub, now };
            let entry = match m1(slot, store, limits, t, create, entropy)? {
                LedgerOut::Collections { mut entries } if entries.len() == 1 => entries.remove(0),
                _ => return Err(crate::service::unexpected()),
            };
            let p = [
                Ns::Embedder.as_str().into(),
                entry.uid.clone().into(),
                model.into(),
            ];
            store.exec(KV_PUT, &p).map_err(db)?;
            M3LedgerOut::Created { entry }
        }
        M3LedgerCall::NextEpoch { uid } => {
            if uid.len() != 32 || !uid.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(OpError::invalid("uid"));
            }
            let k = format!("epoch/{uid}");
            let cur = meta_get(store, &k)?.and_then(|v| v.parse::<u64>().ok());
            let epoch = cur.unwrap_or(0) + 1;
            meta_put(store, &k, epoch.to_string())?;
            M3LedgerOut::Epoch { epoch }
        }
        M3LedgerCall::WitnessAppend { manifest } => {
            let obj = Base64::decode_vec(&manifest).map_err(|_| OpError::invalid("manifest"))?;
            let sealed = SealedManifest::from_object(&obj)
                .map_err(|_| OpError::invalid("malformed manifest"))?;
            let mut chain = load_chain(store, t)?;
            let e = chain.append(&sealed).map_err(|e| match e {
                ruvector_edge_snapshot::SnapshotError::TenantMismatch => {
                    OpError::new(ErrorCode::TenantMismatch, "manifest tenant")
                }
                _ => OpError::new(ErrorCode::Conflict, "manifest root"),
            })?;
            let w = EntryWire::from_entry(&e);
            let p = [
                Value::Int(i64::try_from(w.seq).map_err(|_| OpError::invalid("seq"))?),
                w.prev.clone().into(),
                w.root.clone().into(),
                w.uid.clone().into(),
                Value::Int(i64::from(w.shard)),
                Value::Int(i64::try_from(w.epoch).map_err(|_| OpError::invalid("epoch"))?),
                w.audit_head.clone().into(),
                w.hash.clone().into(),
            ];
            store.exec(W_PUT, &p).map_err(db)?;
            meta_put(store, "witness_head", hex32(&chain.head()))?;
            meta_put(store, "witness_next", chain.next_seq().to_string())?;
            M3LedgerOut::Witnessed { entry: w }
        }
        M3LedgerCall::KvSwap {
            ns,
            key,
            expect,
            value,
            admit,
            adjust,
            now,
        } => {
            if !key_ok(&key) || value.as_ref().is_some_and(|v| v.len() > MAX_VALUE_BYTES) {
                return Err(OpError::invalid("record too large"));
            }
            let p = [ns.as_str().into(), key.as_str().into()];
            let rows = store.query(KV_GET, &p).map_err(db)?;
            let cur = rows
                .first()
                .and_then(|r| r.first())
                .and_then(Value::as_text);
            if cur != expect.as_deref() {
                return Err(OpError::new(ErrorCode::Conflict, "record changed"));
            }
            if let Some(delta) = admit {
                let call = LedgerCall::Admit {
                    delta,
                    work_units: 0,
                    now,
                };
                m1(slot, store, limits, t, call, entropy)?;
            }
            if let Some(delta) = adjust {
                m1(
                    slot,
                    store,
                    limits,
                    t,
                    LedgerCall::Adjust { delta, now },
                    entropy,
                )?;
            }
            match value {
                Some(v) => {
                    let p = [ns.as_str().into(), key.into(), v.into()];
                    store.exec(KV_PUT, &p).map_err(db)?;
                }
                None => {
                    let p = [ns.as_str().into(), key.into()];
                    store.exec(KV_DEL, &p).map_err(db)?;
                }
            }
            M3LedgerOut::Done
        }
        M3LedgerCall::AuditAppend { lines, now_ms } => {
            crate::m3_audit_ledger::append(store, t, lines, now_ms)?
        }
        M3LedgerCall::AuditCommit { seqs } => crate::m3_audit_ledger::commit(store, seqs)?,
        M3LedgerCall::AuditHead => crate::m3_audit_ledger::head(store)?,
        M3LedgerCall::Chain => {
            let chain = load_chain(store, t)?;
            let mut entries = Vec::new();
            for r in store.query(W_ALL, &[]).map_err(db)? {
                let int = |i: usize| col_int(&r, i);
                let text = |i: usize| col_text(&r, i);
                entries.push(EntryWire {
                    seq: u64::try_from(int(0)?).map_err(|_| OpError::invalid("seq"))?,
                    prev: text(1)?,
                    root: text(2)?,
                    uid: text(3)?,
                    shard: u16::try_from(int(4)?).map_err(|_| OpError::invalid("shard"))?,
                    epoch: u64::try_from(int(5)?).map_err(|_| OpError::invalid("epoch"))?,
                    audit_head: text(6)?,
                    hash: text(7)?,
                });
            }
            M3LedgerOut::Chain {
                entries,
                head: hex32(&chain.head()),
            }
        }
    })
}

fn load_chain(store: &dyn SqlStore, tenant: &str) -> Result<WitnessChain, OpError> {
    let bad = |_| OpError::new(ErrorCode::ServerError, "witness chain state");
    match (
        meta_get(store, "witness_head")?,
        meta_get(store, "witness_next")?,
    ) {
        (Some(h), Some(n)) => {
            let head = unhex32(&h)?;
            let next = n
                .parse::<u64>()
                .map_err(|_| OpError::new(ErrorCode::ServerError, "witness chain state"))?;
            WitnessChain::resume(tenant, head, next).map_err(bad)
        }
        _ => WitnessChain::new(tenant).map_err(bad),
    }
}
