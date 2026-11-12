//! `TenantLedger` Durable Object core: decode a [`LedgerRequest`], run it
//! against the resident [`TenantLedger`] over the DO's [`SqlStore`], encode
//! the reply. Pure and synchronous, so the Worker class is a thin shell and
//! native tests run the very same code over `MemSqlStore`.

use crate::wire::{CfgWire, CollectionWire, LedgerCall, LedgerOut, LedgerRequest, Reply, WireErr};
use ruvector_edge_store::ledger::{CreateCollection, IdemKey, IdemLookup};
use ruvector_edge_store::{
    ledger_meta_for, CatalogEntry, EntropySource, ErrorCode, OpError, SqlStore, TenantLedger,
};
use ruvector_edge_tenancy::{QuotaLimits, TenantKey};
use serde_json::json;

#[path = "ledger_admin.rs"]
pub mod admin;

/// M1 free-plan limits (ADR-351 §10). `max_vectors` is the §10 free figure;
/// the float budget (250k × 384), bytes and daily-op ceilings are **[U]**
/// placeholders until the plan table lands (M3 control plane).
pub const FREE_PLAN: QuotaLimits = QuotaLimits {
    max_collections: 20,
    max_vectors: 250_000,
    max_float_budget: 96_000_000,
    max_bytes: 1 << 30,
    max_daily_ops: 1_000_000,
};

/// Wire view of a catalog entry.
pub fn collection_wire(e: &CatalogEntry) -> CollectionWire {
    CollectionWire {
        uid: e.uid.to_hex(),
        shard_count: e.shard_count.get(),
        cfg: CfgWire {
            dim: e.dim,
            metric: e.metric,
            filterable_keys: e.filterable_keys.clone(),
            index: e.index,
        },
        view: e.to_json(),
    }
}

/// Serve one encoded request. `slot` is the DO's resident ledger (opened on
/// first use and reopened after a storage error poisoned it). Always
/// returns an encoded [`Reply`].
pub fn serve(
    slot: &mut Option<TenantLedger>,
    store: &dyn SqlStore,
    limits: QuotaLimits,
    body: &[u8],
    entropy: &dyn EntropySource,
) -> String {
    let reply: Reply<LedgerOut> = match serde_json::from_slice::<LedgerRequest>(body) {
        Ok(req) => handle(slot, store, limits, req, entropy).map_err(|e| WireErr::from_op(&e)),
        Err(_) => {
            // Members / deny / drop (ADR-351 §4.2, §5.8) use their own shape.
            if let Some(out) = admin::serve(slot, store, limits, body) {
                return out;
            }
            Err(WireErr::from_op(&OpError::invalid("malformed ledger call")))
        }
    };
    serde_json::to_string(&reply)
        .unwrap_or_else(|_| String::from(r#"{"Err":{"code":"server_error"}}"#))
}

/// The resident ledger, (re)opened when absent or poisoned.
pub(crate) fn open<'a>(
    slot: &'a mut Option<TenantLedger>,
    store: &dyn SqlStore,
    limits: QuotaLimits,
) -> Result<&'a mut TenantLedger, OpError> {
    if slot.as_ref().is_some_and(TenantLedger::is_poisoned) {
        *slot = None;
    }
    if slot.is_none() {
        *slot = Some(TenantLedger::open(store, limits)?);
    }
    slot.as_mut()
        .ok_or(OpError::new(ErrorCode::ServerError, "ledger vanished"))
}

fn idem_out(l: IdemLookup) -> LedgerOut {
    let (replay, conflict, in_flight) = match l {
        IdemLookup::Miss => (None, false, false),
        IdemLookup::Replay(body) => (Some(body), false, false),
        IdemLookup::Conflict => (None, true, false),
        IdemLookup::InFlight => (None, false, true),
    };
    LedgerOut::Idem {
        replay,
        conflict,
        in_flight,
    }
}

fn spec(raw: &str) -> Result<CreateCollection, OpError> {
    serde_json::from_str(raw).map_err(|_| OpError::invalid("malformed args"))
}

/// Run one call.
pub fn handle(
    slot: &mut Option<TenantLedger>,
    store: &dyn SqlStore,
    limits: QuotaLimits,
    req: LedgerRequest,
    entropy: &dyn EntropySource,
) -> Result<LedgerOut, OpError> {
    let tenant =
        TenantKey::parse(&req.tenant_key).map_err(|_| OpError::invalid("malformed tenant"))?;
    let lm = ledger_meta_for(&tenant)?;
    let ledger = open(slot, store, limits)?;
    Ok(match req.call {
        LedgerCall::Access { sub } => LedgerOut::Access {
            role: ledger.role_of(&lm, &sub)?.map(|r| r.as_str().to_string()),
            claimed: ledger.is_claimed(&lm)?,
        },
        LedgerCall::Claim { sub, now } => LedgerOut::Role {
            role: ledger.claim(store, &lm, &sub, now)?.as_str().to_string(),
        },
        LedgerCall::Collection { name } => LedgerOut::Collection {
            entry: ledger.collection(&lm, &name)?.map(collection_wire),
        },
        LedgerCall::Collections => LedgerOut::Collections {
            entries: ledger
                .collections(&lm)?
                .into_iter()
                .map(collection_wire)
                .collect(),
        },
        LedgerCall::ValidateCreate { spec: raw } => {
            let (name, shards, _) = ledger.validate_create(&lm, &spec(&raw)?)?;
            LedgerOut::Validated {
                name,
                shards: shards.get(),
            }
        }
        LedgerCall::CreateCollection {
            spec: raw,
            sub,
            now,
        } => {
            let e = ledger.create_collection(store, &lm, &spec(&raw)?, &sub, entropy, now)?;
            LedgerOut::Collections {
                entries: vec![collection_wire(&e)],
            }
        }
        LedgerCall::Admit {
            delta,
            work_units,
            now,
        } => {
            ledger.admit(store, &lm, delta, work_units, now)?;
            LedgerOut::Done
        }
        LedgerCall::Adjust { delta, now } => {
            ledger.adjust(store, &lm, delta, now)?;
            LedgerOut::Done
        }
        LedgerCall::Usage { now } => LedgerOut::Usage {
            report: json!({
                "usage": ledger.usage(&lm, now)?,
                "limits": ledger.limits(),
                "work_units": ledger.work_units(),
            }),
        },
        LedgerCall::IdemLookup {
            sub,
            key,
            sha256,
            now,
        } => {
            let k = IdemKey {
                sub: &sub,
                key: &key,
                body_sha256: &sha256,
            };
            idem_out(ledger.idem_lookup(store, &lm, k, now)?)
        }
        LedgerCall::IdemReserve {
            sub,
            key,
            sha256,
            now,
        } => {
            let k = IdemKey {
                sub: &sub,
                key: &key,
                body_sha256: &sha256,
            };
            idem_out(ledger.idem_reserve(store, &lm, k, now)?)
        }
        LedgerCall::IdemRelease { sub, key, sha256 } => {
            let k = IdemKey {
                sub: &sub,
                key: &key,
                body_sha256: &sha256,
            };
            ledger.idem_release(store, &lm, k)?;
            LedgerOut::Done
        }
        LedgerCall::IdemStore {
            sub,
            key,
            sha256,
            response,
            now,
        } => {
            let k = IdemKey {
                sub: &sub,
                key: &key,
                body_sha256: &sha256,
            };
            ledger.idem_store(store, &lm, k, &response, now)?;
            LedgerOut::Done
        }
    })
}
