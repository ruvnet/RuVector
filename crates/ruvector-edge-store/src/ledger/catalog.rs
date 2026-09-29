//! Collection catalog (ADR-351 §4.3, §6.2).
//!
//! Catalog rows are never deleted: a drop tombstones the row, so its
//! `collection_uid` stays reserved forever and a recreated name gets a new
//! uid (new DO, new keys). Allocation retries on a collision with **any**
//! row, live or tombstoned.

use super::TenantLedger;
use crate::distance::Metric;
use crate::error::{ErrorCode, OpError};
use crate::filter::validate_filterable_keys;
use crate::ports::{col_int, col_text, SqlStore, StoreError, Value};
use crate::schema;
use crate::shard::ShardConfig;
use ruvector_edge_tenancy::quota::{dimension_ok, limits::M1_SHARD_FLOAT_CAP};
use ruvector_edge_tenancy::{
    CollectionName, CollectionUid, EntropySource, LedgerMeta, QuotaDelta, ShardCount, UidAllocator,
};
use serde::Deserialize;
use serde_json::{json, Value as Json};

/// Attempts at allocating a non-colliding uid before failing closed.
pub const UID_ALLOC_ATTEMPTS: usize = 8;

/// `POST /v1/collections` body (M1: `index` must be `flat`, no embedder).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateCollection {
    /// `^[a-z0-9][a-z0-9_-]{0,62}$`.
    pub name: String,
    /// `1..=1536`.
    pub dim: u32,
    /// `cosine | l2 | dot`.
    pub metric: Metric,
    /// `flat` (default) at M1.
    #[serde(default)]
    pub index: Option<String>,
    /// M3; must be absent at M1.
    #[serde(default)]
    pub embedder: Option<String>,
    /// ≤ 8 declared metadata keys.
    #[serde(default)]
    pub filterable_keys: Vec<String>,
    /// `1..=6`, default 1.
    #[serde(default)]
    pub shards: Option<u32>,
    /// Provenance tag, e.g. `ruvector-chatgpt` (§16.4).
    #[serde(default)]
    pub origin: Option<String>,
}

/// Catalog row state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionState {
    /// Serving.
    Live,
    /// Tombstoned; uid reserved forever.
    Deleted,
}

impl CollectionState {
    fn as_str(self) -> &'static str {
        match self {
            CollectionState::Live => "live",
            CollectionState::Deleted => "deleted",
        }
    }
}

/// One catalog row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogEntry {
    /// Never-reused uid.
    pub uid: CollectionUid,
    /// Name (unique among live rows).
    pub name: String,
    /// Dimension.
    pub dim: u32,
    /// Metric.
    pub metric: Metric,
    /// Shard count.
    pub shard_count: ShardCount,
    /// State.
    pub state: CollectionState,
    /// Provenance tag.
    pub origin: Option<String>,
    /// Creator `sub`.
    pub created_by: String,
    /// Unix seconds.
    pub created_at: u64,
    /// Declared filterable keys.
    pub filterable_keys: Vec<String>,
}

impl CatalogEntry {
    /// The shard configuration every shard of this collection uses.
    pub fn shard_config(&self) -> ShardConfig {
        ShardConfig {
            dim: self.dim,
            metric: self.metric,
            filterable_keys: self.filterable_keys.clone(),
            float_cap: M1_SHARD_FLOAT_CAP,
        }
    }

    /// Public JSON view.
    pub fn to_json(&self) -> Json {
        json!({
            "name": self.name,
            "collection_uid": self.uid.to_hex(),
            "dim": self.dim,
            "metric": self.metric.as_str(),
            "index": "flat",
            "shards": self.shard_count.get(),
            "filterable_keys": self.filterable_keys,
            "origin": self.origin,
            "created_at": self.created_at,
        })
    }
}

fn corrupt(what: &'static str) -> impl Fn(ruvector_edge_tenancy::TenancyError) -> StoreError {
    move |_| StoreError::Corrupt(what)
}

pub(super) fn load(store: &dyn SqlStore) -> Result<Vec<CatalogEntry>, StoreError> {
    let mut out = Vec::new();
    for r in store.query(schema::CATALOG_SELECT_ALL, &[])? {
        let state = match col_text(&r, 8, "catalog.state")?.as_str() {
            "live" => CollectionState::Live,
            "deleted" => CollectionState::Deleted,
            _ => return Err(StoreError::Corrupt("catalog.state")),
        };
        let int = |i, w| {
            col_int(&r, i, w).and_then(|v| u64::try_from(v).map_err(|_| StoreError::Corrupt(w)))
        };
        out.push(CatalogEntry {
            uid: CollectionUid::parse(&col_text(&r, 0, "catalog.uid")?)
                .map_err(corrupt("catalog.uid"))?,
            name: col_text(&r, 1, "catalog.name")?,
            dim: u32::try_from(int(4, "catalog.dim")?)
                .map_err(|_| StoreError::Corrupt("catalog.dim"))?,
            metric: Metric::parse(&col_text(&r, 5, "catalog.metric")?)
                .ok_or(StoreError::Corrupt("catalog.metric"))?,
            shard_count: ShardCount::new(
                u32::try_from(int(7, "catalog.shard_count")?).unwrap_or(0),
            )
            .map_err(corrupt("catalog.shard_count"))?,
            state,
            origin: r.get(9).and_then(Value::as_text).map(str::to_string),
            created_by: col_text(&r, 10, "catalog.created_by")?,
            created_at: int(11, "catalog.created_at")?,
            filterable_keys: serde_json::from_str(&col_text(&r, 12, "catalog.filterable_keys")?)
                .map_err(|_| StoreError::Corrupt("catalog.filterable_keys"))?,
        });
    }
    Ok(out)
}

fn validate_origin(o: &str) -> Result<(), OpError> {
    let ok = !o.is_empty()
        && o.len() <= 64
        && o.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_' | b'.')
        });
    if ok {
        Ok(())
    } else {
        Err(OpError::invalid("invalid origin"))
    }
}

impl TenantLedger {
    /// Live collection by name (`None` for absent **and** foreign names: a
    /// ledger only ever holds its own tenant's catalog).
    pub fn collection(
        &self,
        expected: &LedgerMeta,
        name: &str,
    ) -> Result<Option<&CatalogEntry>, OpError> {
        self.guard(expected, false)?;
        Ok(self
            .catalog
            .iter()
            .find(|c| c.state == CollectionState::Live && c.name == name))
    }

    /// Live collections, by name.
    pub fn collections(&self, expected: &LedgerMeta) -> Result<Vec<&CatalogEntry>, OpError> {
        self.guard(expected, false)?;
        let mut v: Vec<_> = self
            .catalog
            .iter()
            .filter(|c| c.state == CollectionState::Live)
            .collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(v)
    }

    /// Every uid ever allocated (live and tombstoned).
    pub fn all_uids(&self) -> Vec<CollectionUid> {
        self.catalog.iter().map(|c| c.uid).collect()
    }

    /// Validate a create without writing (also the `dry_run` path): name,
    /// dim, index kind, filterable keys, origin, shard count, live-name
    /// conflict and the collection quota. Returns the normalised name, shard
    /// count and the usage after admission.
    pub fn validate_create(
        &self,
        expected: &LedgerMeta,
        spec: &CreateCollection,
    ) -> Result<(String, ShardCount, ruvector_edge_tenancy::Usage), OpError> {
        self.guard(expected, false)?;
        let name = CollectionName::parse(&spec.name)?.as_str().to_string();
        if !dimension_ok(spec.dim) {
            return Err(OpError::invalid("dim out of range"));
        }
        if spec.index.as_deref().is_some_and(|i| i != "flat") {
            return Err(OpError::invalid("index kind not available"));
        }
        if spec.embedder.is_some() {
            return Err(OpError::invalid("embedder not available"));
        }
        validate_filterable_keys(&spec.filterable_keys)?;
        if let Some(o) = &spec.origin {
            validate_origin(o)?;
        }
        let shard_count = ShardCount::new(spec.shards.unwrap_or(1))?;
        if self
            .catalog
            .iter()
            .any(|c| c.state == CollectionState::Live && c.name == name)
        {
            return Err(OpError::new(ErrorCode::Conflict, "collection exists"));
        }
        let delta = QuotaDelta {
            collections: 1,
            ..QuotaDelta::default()
        };
        let usage = ruvector_edge_tenancy::admit(self.limits(), &self.usage, &delta)?;
        Ok((name, shard_count, usage))
    }

    /// Create a collection: validate, admit `collections + 1`, allocate a
    /// fresh uid (retrying on any catalog collision), persist.
    pub fn create_collection(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        spec: &CreateCollection,
        created_by: &str,
        entropy: &dyn EntropySource,
        now: u64,
    ) -> Result<CatalogEntry, OpError> {
        let check = self.guard(expected, true)?;
        let (name, shard_count, usage) = self.validate_create(expected, spec)?;
        let mut alloc = match &self.uid {
            Some(a) => a.clone(),
            None => UidAllocator::fresh(entropy)?,
        };
        let mut uid = None;
        for _ in 0..UID_ALLOC_ATTEMPTS {
            let u = alloc.allocate(entropy)?;
            if !self.catalog.iter().any(|c| c.uid == u) {
                uid = Some(u);
                break;
            }
        }
        let uid = uid.ok_or(OpError::new(ErrorCode::ServerError, "uid allocation"))?;
        let entry = CatalogEntry {
            uid,
            name,
            dim: spec.dim,
            metric: spec.metric,
            shard_count,
            state: CollectionState::Live,
            origin: spec.origin.clone(),
            created_by: created_by.to_string(),
            created_at: now,
            filterable_keys: spec.filterable_keys.clone(),
        };
        let usage_json = serde_json::to_string(&usage)
            .map_err(|_| OpError::new(ErrorCode::ServerError, "usage"))?;
        let res = self.init_identity(store, check, expected).and_then(|_| {
            store.exec(
                schema::LMETA_PUT,
                &[
                    super::keys::UID_SALT.into(),
                    hex::encode(alloc.salt()).into(),
                ],
            )?;
            store.exec(
                schema::LMETA_PUT,
                &[
                    super::keys::UID_NEXT_SEQ.into(),
                    alloc.next_seq().to_string().into(),
                ],
            )?;
            store.exec(schema::CATALOG_INSERT, &catalog_row(&entry)?)?;
            store.exec(
                schema::LMETA_PUT,
                &[super::keys::USAGE.into(), usage_json.into()],
            )
        });
        if let Err(e) = res {
            return self.poison(e);
        }
        self.commit_identity(check, expected);
        self.uid = Some(alloc);
        self.usage = usage;
        self.catalog.push(entry.clone());
        Ok(entry)
    }

    /// Tombstone a live collection and release its collection slot. The
    /// caller wipes the collection's shards first and releases their vector
    /// usage with [`TenantLedger::admit`].
    pub fn drop_collection(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        name: &str,
    ) -> Result<CatalogEntry, OpError> {
        self.guard(expected, true)?;
        let idx = self
            .catalog
            .iter()
            .position(|c| c.state == CollectionState::Live && c.name == name)
            .ok_or(OpError::not_found())?;
        let delta = QuotaDelta {
            collections: -1,
            ..QuotaDelta::default()
        };
        let usage = ruvector_edge_tenancy::admit(self.limits(), &self.usage, &delta)?;
        let usage_json = serde_json::to_string(&usage)
            .map_err(|_| OpError::new(ErrorCode::ServerError, "usage"))?;
        let uid = self.catalog[idx].uid.to_hex();
        let res = store
            .exec(
                schema::CATALOG_SET_STATE,
                &[CollectionState::Deleted.as_str().into(), uid.into()],
            )
            .and_then(|_| {
                store.exec(
                    schema::LMETA_PUT,
                    &[super::keys::USAGE.into(), usage_json.into()],
                )
            });
        if let Err(e) = res {
            return self.poison(e);
        }
        self.usage = usage;
        self.catalog[idx].state = CollectionState::Deleted;
        Ok(self.catalog[idx].clone())
    }
}

fn catalog_row(e: &CatalogEntry) -> Result<Vec<Value>, StoreError> {
    let ts = i64::try_from(e.created_at).map_err(|_| StoreError::Corrupt("created_at"))?;
    Ok(vec![
        e.uid.to_hex().into(),
        e.name.as_str().into(),
        "vector".into(),
        "flat".into(),
        i64::from(e.dim).into(),
        e.metric.as_str().into(),
        Value::Null,
        i64::from(e.shard_count.get()).into(),
        e.state.as_str().into(),
        e.origin.clone().map_or(Value::Null, Value::Text),
        e.created_by.as_str().into(),
        ts.into(),
        serde_json::to_string(&e.filterable_keys)
            .unwrap_or_else(|_| "[]".into())
            .into(),
    ])
}
