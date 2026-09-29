//! `VectorShard` wipe for a dropped collection (ADR-351 §7.2 `DELETE
//! /v1/collections/{c}`): its own request shape (`{"tenant_key", "uid",
//! "shard", "wipe": true}`), intercepted by the DO shell before
//! `shard_core::serve`, so the core shard wire is untouched.
//!
//! The collection is already `Deleting` in the ledger (no new lookup finds
//! it). The DO checks the stored identity against the request (mismatch →
//! `404`), reports the usage it held (so the gateway releases exactly that
//! from the ledger), deletes every row, index chunks included, and leaves
//! only the `meta.wiped` marker, so it refuses every later request (a
//! write that looked the collection up before the drop included). The
//! Worker shell then evicts the resident index and deletes any pending
//! maintenance alarm; it does **not** `deleteAll` (that would erase the
//! marker). The gateway then purges the uid in the ledger.

use crate::wire::{DeltaWire, Reply, WireErr};
use ruvector_edge_store::schema::{
    CHUNK_DELETE_ALL, FILTER_DELETE_ALL, META_DELETE_ALL, META_PUT, OPS_DELETE_ALL, VEC_DELETE_ALL,
};
use ruvector_edge_store::shard::codec::keys;
use ruvector_edge_store::{shard_meta_for, OpError, SqlStore, VectorShard};
use ruvector_edge_tenancy::{CollectionUid, DoMeta, ShardIndex, TenantKey};
use serde::{Deserialize, Serialize};

/// Wipe one shard of a collection being dropped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WipeRequest {
    /// Tenant key from the verified token.
    pub tenant_key: String,
    /// `collection_uid` (32 hex).
    pub uid: String,
    /// Decimal shard index.
    pub shard: u32,
    /// Always `true` (distinguishes the shape from a `ShardRequest`).
    pub wipe: bool,
}

impl WipeRequest {
    /// The request for the shard `dm` names.
    pub fn for_shard(dm: &DoMeta) -> Self {
        WipeRequest {
            tenant_key: dm.tenant_key().as_str().to_string(),
            uid: dm.collection_uid().to_hex(),
            shard: dm.shard().get(),
            wipe: true,
        }
    }

    fn meta(&self) -> Result<DoMeta, OpError> {
        let tenant =
            TenantKey::parse(&self.tenant_key).map_err(|_| OpError::invalid("malformed tenant"))?;
        let uid = CollectionUid::parse(&self.uid).map_err(|_| OpError::invalid("malformed uid"))?;
        let shard = ShardIndex::parse(&self.shard.to_string())
            .map_err(|_| OpError::invalid("malformed shard"))?;
        shard_meta_for(&tenant, uid, shard)
    }
}

/// Wipe the shard over `store`; the usage it held. A shard that was never
/// written holds nothing (zero); a foreign identity is `404`. Every row
/// goes, and `meta.wiped` is set in the same synchronous call: the DO then
/// answers `404` to every later request and never re-initialises, so a
/// write that raced the drop (looked up before the tombstone) is refused
/// (and refunded by the gateway) instead of charging an unreachable DO.
pub fn handle(
    own_name: Option<&str>,
    store: &dyn SqlStore,
    req: &WipeRequest,
) -> Result<DeltaWire, OpError> {
    let dm = req.meta()?;
    if !req.wipe || own_name.is_some_and(|n| n != dm.do_name().as_str()) {
        return Err(OpError::not_found());
    }
    let shard = VectorShard::open(store)?;
    let released = match shard.identity() {
        None => DeltaWire::default(),
        Some(id) if *id == dm => DeltaWire::from(shard.usage_totals()),
        Some(_) => return Err(OpError::not_found()),
    };
    for sql in [
        VEC_DELETE_ALL,
        FILTER_DELETE_ALL,
        OPS_DELETE_ALL,
        CHUNK_DELETE_ALL,
        META_DELETE_ALL,
    ] {
        store.exec(sql, &[])?;
    }
    store.exec(META_PUT, &[keys::WIPED.into(), "1".into()])?;
    Ok(released)
}

/// Encoded reply for a wipe body (`None`: not a wipe request), plus
/// whether the wipe succeeded (the shell then clears the whole storage).
pub fn serve(own_name: Option<&str>, store: &dyn SqlStore, body: &[u8]) -> Option<(String, bool)> {
    let req = serde_json::from_slice::<WipeRequest>(body).ok()?;
    let reply: Reply<DeltaWire> = handle(own_name, store, &req).map_err(|e| WireErr::from_op(&e));
    let ok = reply.is_ok();
    let text = serde_json::to_string(&reply)
        .unwrap_or_else(|_| String::from(r#"{"Err":{"code":"server_error"}}"#));
    Some((text, ok))
}
