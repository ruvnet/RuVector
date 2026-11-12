//! In-process stand-in for the Durable Object namespaces: one store per DO
//! name, resident state opened lazily, shards registered with the isolate
//! [`ResidentRegistry`] and evicted LRU. Names are always derived from the
//! caller's verified tenant key (`ledger_do_name`, `do_name`), never from
//! request input, exactly as the gateway addresses real DOs.
//!
//! A shard is registered the moment it is cold-loaded — before any
//! validation of the request that loaded it — so a failing or dry-run
//! request can never leave untracked resident state behind. A poisoned
//! ledger or shard is dropped and reopened from storage on next use.

use crate::error::OpError;
use crate::ledger::TenantLedger;
use crate::ports::SqlStore;
use crate::resident::ResidentRegistry;
use crate::shard::VectorShard;
use ruvector_edge_tenancy::{ledger_do_name, DoName, QuotaLimits, TenantKey};
use std::collections::BTreeMap;

/// Every DO's storage plus whatever is currently resident.
#[derive(Debug)]
pub struct LocalCluster<S> {
    limits: QuotaLimits,
    stores: BTreeMap<String, S>,
    ledgers: BTreeMap<String, TenantLedger>,
    shards: BTreeMap<String, VectorShard>,
    registry: ResidentRegistry,
}

impl<S: SqlStore + Default> LocalCluster<S> {
    /// Empty cluster with per-tenant `limits` and the default isolate cap.
    pub fn new(limits: QuotaLimits) -> Self {
        Self::with_registry(limits, ResidentRegistry::default())
    }

    /// Empty cluster with an explicit resident registry.
    pub fn with_registry(limits: QuotaLimits, registry: ResidentRegistry) -> Self {
        LocalCluster {
            limits,
            stores: BTreeMap::new(),
            ledgers: BTreeMap::new(),
            shards: BTreeMap::new(),
            registry,
        }
    }

    /// The tenant's ledger (opened on first use, and reopened after a
    /// storage error poisoned it).
    pub fn ledger(&mut self, tenant: &TenantKey) -> Result<(&S, &mut TenantLedger), OpError> {
        let name = ledger_do_name(tenant).as_str().to_string();
        let store = self.stores.entry(name.clone()).or_default();
        if self
            .ledgers
            .get(&name)
            .is_some_and(TenantLedger::is_poisoned)
        {
            self.ledgers.remove(&name);
        }
        let ledger = match self.ledgers.entry(name) {
            std::collections::btree_map::Entry::Occupied(o) => o.into_mut(),
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(TenantLedger::open(store, self.limits)?)
            }
        };
        Ok((store, ledger))
    }

    /// A shard by DO name (opened, i.e. cold-loaded, on first use or after
    /// eviction / poisoning). A freshly loaded shard is registered with the
    /// isolate registry immediately, evicting LRU shards if needed.
    pub fn shard(&mut self, name: &DoName) -> Result<(&S, &mut VectorShard), OpError> {
        let key = name.as_str().to_string();
        if self.shards.get(&key).is_some_and(VectorShard::is_poisoned) {
            self.shards.remove(&key);
            self.registry.remove(&key);
        }
        if !self.shards.contains_key(&key) {
            let store = self.stores.entry(key.clone()).or_default();
            let shard = VectorShard::open(store)?;
            for v in self.registry.touch(&key, shard.resident_bytes()) {
                self.shards.remove(&v);
            }
            self.shards.insert(key.clone(), shard);
        }
        let store = self.stores.entry(key.clone()).or_default();
        let shard = self.shards.get_mut(&key).ok_or(OpError::new(
            crate::ErrorCode::ServerError,
            "shard vanished",
        ))?;
        Ok((store, shard))
    }

    /// Record a shard's current resident size and recency, evicting LRU
    /// shards if the isolate cap is exceeded. Never fails (the shard itself
    /// is never evicted by its own touch). Returns the evicted names.
    pub fn touch(&mut self, name: &DoName) -> Vec<String> {
        let Some(bytes) = self
            .shards
            .get(name.as_str())
            .map(VectorShard::resident_bytes)
        else {
            return Vec::new();
        };
        let evicted = self.registry.touch(name.as_str(), bytes);
        for v in &evicted {
            self.shards.remove(v);
        }
        evicted
    }

    /// Simulate an isolate restart: drop every resident state, keep storage.
    pub fn restart(&mut self) {
        self.ledgers.clear();
        self.shards.clear();
        self.registry = ResidentRegistry::new(self.registry.cap());
    }

    /// Storage of a DO, if it exists.
    pub fn store(&self, name: &str) -> Option<&S> {
        self.stores.get(name)
    }

    /// `true` if the shard is resident and poisoned (a write to it failed
    /// after issuing statements).
    pub fn shard_poisoned(&self, name: &DoName) -> bool {
        self.shards
            .get(name.as_str())
            .is_some_and(VectorShard::is_poisoned)
    }

    /// `true` if the shard's state is resident.
    pub fn is_resident(&self, name: &DoName) -> bool {
        self.shards.contains_key(name.as_str())
    }

    /// Resident bytes across shards (registry view).
    pub fn resident_total(&self) -> u64 {
        self.registry.total()
    }

    /// Number of resident shards (host view; always equal to the number
    /// registered).
    pub fn resident_shards(&self) -> usize {
        self.shards.len()
    }
}
