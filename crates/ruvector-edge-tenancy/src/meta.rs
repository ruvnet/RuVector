//! In-object identity assertion (ADR-351 §4.3).
//!
//! On first write each DO stores its identity in its `meta(k TEXT PRIMARY
//! KEY, v TEXT)` table. On **every** call it recomputes the identity the
//! caller expects and compares; any mismatch is `404 not_found`, the same
//! answer a missing resource gets.

use crate::context::TenantContext;
use crate::error::TenancyError;
use crate::names::{do_name, ledger_do_name, DoName, Service, TenantKey};
use crate::shard::ShardIndex;
use crate::uid::CollectionUid;

/// `meta` key for the tenant key.
pub const META_TENANT_KEY: &str = "tenant_key";
/// `meta` key for the service.
pub const META_SERVICE: &str = "service";
/// `meta` key for the collection uid.
pub const META_COLLECTION_UID: &str = "collection_uid";
/// `meta` key for the shard index.
pub const META_SHARD: &str = "shard";

/// Outcome of an identity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityCheck {
    /// Stored identity equals the caller's.
    Matched,
    /// Uninitialised DO and the call is a write: persist
    /// [`DoMeta::to_kv`] (in the same transaction as the write).
    InitializeOnWrite,
    /// Uninitialised DO and the call is a read: serve an empty result and
    /// persist nothing.
    EmptyRead,
}

/// Identity of a data-plane DO (`VectorShard`, `QuantShard`, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoMeta {
    tenant_key: TenantKey,
    service: Service,
    collection_uid: CollectionUid,
    shard: ShardIndex,
}

impl DoMeta {
    /// The identity a caller with `ctx` expects. The tenant comes only from
    /// the verified context, never from request input.
    pub fn expected(
        ctx: &TenantContext,
        service: Service,
        collection_uid: CollectionUid,
        shard: ShardIndex,
    ) -> Self {
        DoMeta {
            tenant_key: ctx.tenant_key().clone(),
            service,
            collection_uid,
            shard,
        }
    }

    /// The DO name this identity maps to.
    pub fn do_name(&self) -> DoName {
        do_name(
            &self.tenant_key,
            self.service,
            &self.collection_uid,
            self.shard,
        )
    }

    /// Rows to persist in `meta` on first write.
    pub fn to_kv(&self) -> [(&'static str, String); 4] {
        [
            (META_TENANT_KEY, self.tenant_key.as_str().to_string()),
            (META_SERVICE, self.service.as_str().to_string()),
            (META_COLLECTION_UID, self.collection_uid.to_hex()),
            (META_SHARD, self.shard.get().to_string()),
        ]
    }

    /// Read the identity from `meta` rows. Rows with other keys are ignored
    /// (the table holds other state). `Ok(None)` when no identity key is
    /// present (a fresh DO); an error when the identity is partial,
    /// duplicated or malformed (fail closed).
    pub fn from_kv<'a, I>(rows: I) -> Result<Option<Self>, TenancyError>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut slots: [Option<&str>; 4] = [None; 4];
        let keys = [
            META_TENANT_KEY,
            META_SERVICE,
            META_COLLECTION_UID,
            META_SHARD,
        ];
        for (k, v) in rows {
            if let Some(i) = keys.iter().position(|key| *key == k) {
                if slots[i].replace(v).is_some() {
                    return Err(TenancyError::MalformedIdentifier("meta duplicate key"));
                }
            }
        }
        match slots {
            [None, None, None, None] => Ok(None),
            [Some(t), Some(s), Some(u), Some(sh)] => Ok(Some(DoMeta {
                tenant_key: TenantKey::parse(t)?,
                service: Service::parse(s)?,
                collection_uid: CollectionUid::parse(u)?,
                shard: ShardIndex::parse(sh)?,
            })),
            _ => Err(TenancyError::MalformedIdentifier("meta partial identity")),
        }
    }

    /// Assert the caller's expected identity against what the DO stored.
    /// Any field mismatch is [`TenancyError::NotFound`].
    pub fn check(
        stored: Option<&DoMeta>,
        expected: &DoMeta,
        is_write: bool,
    ) -> Result<IdentityCheck, TenancyError> {
        match stored {
            Some(s) if s == expected => Ok(IdentityCheck::Matched),
            Some(_) => Err(TenancyError::NotFound),
            None if is_write => Ok(IdentityCheck::InitializeOnWrite),
            None => Ok(IdentityCheck::EmptyRead),
        }
    }

    /// Tenant key.
    pub fn tenant_key(&self) -> &TenantKey {
        &self.tenant_key
    }
    /// Service.
    pub fn service(&self) -> Service {
        self.service
    }
    /// Collection uid.
    pub fn collection_uid(&self) -> CollectionUid {
        self.collection_uid
    }
    /// Shard index.
    pub fn shard(&self) -> ShardIndex {
        self.shard
    }
}

/// Identity of a `TenantLedger` DO: just the tenant key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerMeta {
    tenant_key: TenantKey,
}

impl LedgerMeta {
    /// The ledger identity a caller with `ctx` expects.
    pub fn expected(ctx: &TenantContext) -> Self {
        LedgerMeta {
            tenant_key: ctx.tenant_key().clone(),
        }
    }

    /// The ledger DO name.
    pub fn do_name(&self) -> DoName {
        ledger_do_name(&self.tenant_key)
    }

    /// Row to persist in `meta` on first write.
    pub fn to_kv(&self) -> [(&'static str, String); 1] {
        [(META_TENANT_KEY, self.tenant_key.as_str().to_string())]
    }

    /// Read the identity from `meta` rows (same rules as [`DoMeta::from_kv`]).
    pub fn from_kv<'a, I>(rows: I) -> Result<Option<Self>, TenancyError>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut found: Option<&str> = None;
        for (k, v) in rows {
            if k == META_TENANT_KEY && found.replace(v).is_some() {
                return Err(TenancyError::MalformedIdentifier("meta duplicate key"));
            }
        }
        found
            .map(|t| TenantKey::parse(t).map(|tenant_key| LedgerMeta { tenant_key }))
            .transpose()
    }

    /// Assert the caller's expected ledger identity (mismatch → 404).
    pub fn check(
        stored: Option<&LedgerMeta>,
        expected: &LedgerMeta,
        is_write: bool,
    ) -> Result<IdentityCheck, TenancyError> {
        match stored {
            Some(s) if s == expected => Ok(IdentityCheck::Matched),
            Some(_) => Err(TenancyError::NotFound),
            None if is_write => Ok(IdentityCheck::InitializeOnWrite),
            None => Ok(IdentityCheck::EmptyRead),
        }
    }

    /// Tenant key.
    pub fn tenant_key(&self) -> &TenantKey {
        &self.tenant_key
    }
}
