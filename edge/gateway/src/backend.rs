//! Transport to the two Durable Object classes, plus typed call helpers.
//!
//! [`Backend`] only moves encoded bodies: the Worker implementation
//! addresses `TenantLedger` by `ledger_do_name(tenant)` and `VectorShard` by
//! `DoMeta::do_name()` (both via `idFromName`, derived only from the
//! verified tenant key and the ledger's catalog), and the native test
//! implementation runs the same DO cores in process. Names are never taken
//! from request input.

use crate::wire::{
    unavailable, LedgerCall, LedgerOut, LedgerRequest, Reply, ShardCall, ShardOut, ShardRequest,
};
use ruvector_edge_store::OpError;
use ruvector_edge_tenancy::{ledger_do_name, DoMeta, DoName, TenantKey};
use serde::de::DeserializeOwned;

/// Encoded-body transport to the DOs. `Err` means the call's fate is
/// unknown (the DO may or may not have run it).
#[allow(async_fn_in_trait)] // Workers futures are !Send; no Send bound wanted.
pub trait Backend {
    /// POST `body` to the tenant's `TenantLedger`.
    async fn call_ledger(&self, name: &DoName, body: String) -> Result<String, OpError>;
    /// POST `body` to one `VectorShard`.
    async fn call_shard(&self, name: &DoName, body: String) -> Result<String, OpError>;
}

/// Why a shard call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardErr {
    /// The DO answered with an error before issuing any statement.
    Refused(OpError),
    /// Transport failure or undecodable reply: effect unknown.
    Unknown(OpError),
}

impl ShardErr {
    /// The error to report.
    pub fn into_op(self) -> OpError {
        match self {
            ShardErr::Refused(e) | ShardErr::Unknown(e) => e,
        }
    }
}

fn decode<T: DeserializeOwned>(text: &str) -> Option<Reply<T>> {
    serde_json::from_str(text).ok()
}

/// One `TenantLedger` call for `tenant`.
pub async fn ledger<B: Backend>(
    b: &B,
    tenant: &TenantKey,
    call: LedgerCall,
) -> Result<LedgerOut, OpError> {
    let req = LedgerRequest {
        tenant_key: tenant.as_str().to_string(),
        call,
    };
    let body = serde_json::to_string(&req).map_err(|_| unavailable())?;
    let text = b.call_ledger(&ledger_do_name(tenant), body).await?;
    match decode::<LedgerOut>(&text) {
        Some(Ok(out)) => Ok(out),
        Some(Err(e)) => Err(e.into_op()),
        None => Err(unavailable()),
    }
}

/// One `VectorShard` call for the shard `dm` names.
pub async fn shard<B: Backend>(b: &B, dm: &DoMeta, call: ShardCall) -> Result<ShardOut, ShardErr> {
    let req = ShardRequest {
        tenant_key: dm.tenant_key().as_str().to_string(),
        uid: dm.collection_uid().to_hex(),
        shard: dm.shard().get(),
        call,
    };
    let body = serde_json::to_string(&req).map_err(|_| ShardErr::Refused(unavailable()))?;
    let text = b
        .call_shard(&dm.do_name(), body)
        .await
        .map_err(ShardErr::Unknown)?;
    match decode::<ShardOut>(&text) {
        Some(Ok(out)) => Ok(out),
        Some(Err(e)) => Err(ShardErr::Refused(e.into_op())),
        None => Err(ShardErr::Unknown(unavailable())),
    }
}

/// Native in-process backend running the real DO cores over
/// `MemSqlStore`, JSON-encoding every call exactly as the Worker does.
#[cfg(test)]
pub mod mem {
    use super::Backend;
    use crate::ledger_core;
    use crate::shard_core::{self, ShardHost};
    use ruvector_edge_store::{MemSqlStore, OpError, ResidentRegistry, TenantLedger};
    use ruvector_edge_tenancy::{DoName, EntropySource, QuotaLimits, TenancyError};
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;

    /// Deterministic counter "CSPRNG" (tests only).
    pub struct CounterEntropy(pub Cell<u64>);
    impl EntropySource for CounterEntropy {
        fn fill(&self, out: &mut [u8]) -> Result<(), TenancyError> {
            for b in out.iter_mut() {
                let n = self
                    .0
                    .get()
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.0.set(n);
                *b = (n >> 33) as u8;
            }
            Ok(())
        }
    }

    /// Every DO's storage plus the resident state, in process.
    pub struct MemBackend {
        /// Ledger stores + resident ledgers by DO name.
        pub ledgers: RefCell<BTreeMap<String, (MemSqlStore, Option<TenantLedger>)>>,
        /// Shard stores by DO name.
        pub shards: RefCell<BTreeMap<String, MemSqlStore>>,
        /// The simulated isolate.
        pub host: RefCell<ShardHost>,
        limits: QuotaLimits,
        entropy: CounterEntropy,
        /// When set, every shard call fails in transport.
        pub shard_down: Cell<bool>,
    }

    impl MemBackend {
        /// Free-plan limits, default isolate cap.
        pub fn new() -> Self {
            Self::with(ledger_core::FREE_PLAN, ResidentRegistry::default())
        }

        /// Explicit limits and registry.
        pub fn with(limits: QuotaLimits, registry: ResidentRegistry) -> Self {
            MemBackend {
                ledgers: RefCell::new(BTreeMap::new()),
                shards: RefCell::new(BTreeMap::new()),
                host: RefCell::new(ShardHost::with_registry(registry)),
                limits,
                entropy: CounterEntropy(Cell::new(7)),
                shard_down: Cell::new(false),
            }
        }

        /// Owner `owner` invites `sub` as `role`, directly on the ledger
        /// (the members routes are not part of this surface yet).
        pub fn invite(
            &self,
            tenant: &ruvector_edge_tenancy::TenantKey,
            owner: &str,
            sub: &str,
            role: ruvector_edge_tenancy::Role,
            now: u64,
        ) -> Result<(), OpError> {
            let lm = ruvector_edge_store::ledger_meta_for(tenant)?;
            let name = ruvector_edge_tenancy::ledger_do_name(tenant);
            let mut all = self.ledgers.borrow_mut();
            let (store, slot) = all.entry(name.as_str().to_string()).or_default();
            if slot.is_none() {
                *slot = Some(TenantLedger::open(&*store, self.limits)?);
            }
            let ledger = slot.as_mut().ok_or(crate::wire::unavailable())?;
            ledger.put_member(&*store, &lm, owner, sub, role, now)
        }

        /// Drop every resident state (isolate restart), keep storage.
        pub fn restart(&self) {
            *self.host.borrow_mut() = ShardHost::default();
            for (_, l) in self.ledgers.borrow_mut().values_mut() {
                *l = None;
            }
        }
    }

    impl Backend for MemBackend {
        async fn call_ledger(&self, name: &DoName, body: String) -> Result<String, OpError> {
            let mut all = self.ledgers.borrow_mut();
            let (store, slot) = all.entry(name.as_str().to_string()).or_default();
            Ok(ledger_core::serve(
                slot,
                &*store,
                self.limits,
                body.as_bytes(),
                &self.entropy,
            ))
        }

        async fn call_shard(&self, name: &DoName, body: String) -> Result<String, OpError> {
            if self.shard_down.get() {
                return Err(crate::wire::unavailable());
            }
            let mut all = self.shards.borrow_mut();
            let store = all.entry(name.as_str().to_string()).or_default();
            let key = name.as_str();
            Ok(shard_core::serve(
                &mut self.host.borrow_mut(),
                key,
                Some(key),
                &*store,
                body.as_bytes(),
            ))
        }
    }
}
