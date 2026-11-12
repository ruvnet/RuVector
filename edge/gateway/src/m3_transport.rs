//! Transport of the M3 side channel: the [`M3Backend`] port, its Durable
//! Object implementation (same DOs as M1, path [`M3_PATH`]) and the typed
//! `ledger3` / `shard3` calls. Split out of `m3_wire` (re-exported there).

use crate::backend::Backend;
use crate::durable::{DoBackend, LEDGER_BINDING, SHARD_BINDING};
use crate::m3_wire::{
    M3LedgerCall, M3LedgerOut, M3LedgerRequest, M3ShardCall, M3ShardOut, M3ShardRequest, M3_PATH,
};
use crate::wire::{unavailable, Reply};
use ruvector_edge_store::OpError;
use ruvector_edge_tenancy::{ledger_do_name, DoMeta, DoName, TenantKey};
use serde::de::DeserializeOwned;
use worker::{Headers, Method, Request, RequestInit};

/// Transport for the M3 side channel (same DOs, path [`M3_PATH`]).
#[allow(async_fn_in_trait)] // Workers futures are !Send.
pub trait M3Backend: Backend {
    /// POST an M3 body to the tenant's `TenantLedger`.
    async fn m3_ledger(&self, name: &DoName, body: String) -> Result<String, OpError>;
    /// POST an M3 body to one `VectorShard`.
    async fn m3_shard(&self, name: &DoName, body: String) -> Result<String, OpError>;
}

impl DoBackend<'_> {
    async fn m3_post(&self, binding: &str, name: &DoName, body: String) -> worker::Result<String> {
        let stub = self
            .env
            .durable_object(binding)?
            .id_from_name(name.as_str())?
            .get_stub()?;
        let headers = Headers::new();
        headers.set("Content-Type", "application/json")?;
        let mut init = RequestInit::new();
        init.with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(body.into()));
        let url = format!("https://do.internal{M3_PATH}");
        let req = Request::new_with_init(&url, &init)?;
        let mut resp = stub.fetch_with_request(req).await?;
        if resp.status_code() != 200 {
            return Err(worker::Error::RustError("durable object status".into()));
        }
        resp.text().await
    }
}

impl M3Backend for DoBackend<'_> {
    async fn m3_ledger(&self, name: &DoName, body: String) -> Result<String, OpError> {
        self.m3_post(LEDGER_BINDING, name, body)
            .await
            .map_err(|_| unavailable())
    }

    async fn m3_shard(&self, name: &DoName, body: String) -> Result<String, OpError> {
        self.m3_post(SHARD_BINDING, name, body)
            .await
            .map_err(|_| unavailable())
    }
}

fn decode<T: DeserializeOwned>(text: &str) -> Result<T, OpError> {
    match serde_json::from_str::<Reply<T>>(text) {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(e)) => Err(e.into_op()),
        Err(_) => Err(unavailable()),
    }
}

/// One M3 `TenantLedger` call for `tenant`.
pub async fn ledger3<B: M3Backend>(
    b: &B,
    tenant: &TenantKey,
    call: M3LedgerCall,
) -> Result<M3LedgerOut, OpError> {
    let req = M3LedgerRequest {
        tenant_key: tenant.as_str().to_string(),
        call,
    };
    let body = serde_json::to_string(&req).map_err(|_| unavailable())?;
    decode(&b.m3_ledger(&ledger_do_name(tenant), body).await?)
}

/// One M3 `VectorShard` call for the shard `dm` names.
pub async fn shard3<B: M3Backend>(
    b: &B,
    dm: &DoMeta,
    call: M3ShardCall,
) -> Result<M3ShardOut, OpError> {
    let req = M3ShardRequest {
        tenant_key: dm.tenant_key().as_str().to_string(),
        uid: dm.collection_uid().to_hex(),
        shard: dm.shard().get(),
        call,
    };
    let body = serde_json::to_string(&req).map_err(|_| unavailable())?;
    decode(&b.m3_shard(&dm.do_name(), body).await?)
}
