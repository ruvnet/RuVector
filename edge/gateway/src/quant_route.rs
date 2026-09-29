//! Transport routing for `index = rabitq` collections (ADR-351 §3
//! rv-quant): the executors address a shard by its `VectorShard` identity
//! (`DoMeta`, service `vector`); a rabitq collection's shards are instead
//! `QuantShard` DOs named by the same tenant / collection uid / shard with
//! service `quant`. The request body is the same `ShardRequest`, so the
//! upsert / query / fetch / delete / stats / drop paths are unchanged.

use crate::backend::{self, Backend, ShardErr};
use crate::wire::{unavailable, CollectionWire, Reply, ShardCall, ShardOut, ShardRequest};
use ruvector_edge_store::{IndexConfig, OpError};
use ruvector_edge_tenancy::DoMeta;

/// `true` when `e` is served by `QuantShard` DOs.
pub fn is_quant(e: &CollectionWire) -> bool {
    e.cfg.index == IndexConfig::Rabitq
}

/// The `QuantShard` identity of the shard `dm` names.
pub fn quant_meta(dm: &DoMeta) -> Result<DoMeta, OpError> {
    crate::quant_shard::request_meta(
        dm.tenant_key().as_str(),
        &dm.collection_uid().to_hex(),
        dm.shard().get(),
    )
}

/// One shard call for collection `e`, over the transport its index needs.
pub async fn shard_for<B: Backend>(
    b: &B,
    e: &CollectionWire,
    dm: &DoMeta,
    call: ShardCall,
) -> Result<ShardOut, ShardErr> {
    if !is_quant(e) {
        return backend::shard(b, dm, call).await;
    }
    let qm = quant_meta(dm).map_err(ShardErr::Refused)?;
    let req = ShardRequest {
        tenant_key: dm.tenant_key().as_str().to_string(),
        uid: dm.collection_uid().to_hex(),
        shard: dm.shard().get(),
        call,
    };
    let body = serde_json::to_string(&req).map_err(|_| ShardErr::Refused(unavailable()))?;
    let text = b
        .call_quant(&qm.do_name(), body)
        .await
        .map_err(ShardErr::Unknown)?;
    match serde_json::from_str::<Reply<ShardOut>>(&text) {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(e)) => Err(ShardErr::Refused(e.into_op())),
        Err(_) => Err(ShardErr::Unknown(unavailable())),
    }
}

/// Send an encoded drop-wipe body to the shard of `e` that `dm` names.
pub async fn wipe_for<B: Backend>(
    b: &B,
    e: &CollectionWire,
    dm: &DoMeta,
    body: String,
) -> Result<String, OpError> {
    if is_quant(e) {
        b.call_quant(&quant_meta(dm)?.do_name(), body).await
    } else {
        b.call_shard(&dm.do_name(), body).await
    }
}
