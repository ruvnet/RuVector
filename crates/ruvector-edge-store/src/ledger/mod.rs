//! `TenantLedger` (ADR-351 §4.2, §6.2): one Durable Object per tenant
//! holding memberships (default-deny), the collection catalog with
//! never-reused `collection_uid`s, quota counters, and the §16.3 `op_id`
//! idempotency store.
//!
//! Like [`crate::shard::VectorShard`], the struct is the resident state;
//! every call takes the DO's [`SqlStore`] and the [`LedgerMeta`] the caller
//! expects (mismatch → `404`). Writes compute new state first, issue the
//! statements, and only then commit to memory; a storage error poisons the
//! ledger until it is reopened.

mod catalog;
mod idem;
mod usage;

pub use catalog::{CatalogEntry, CollectionState, CreateCollection};
pub use idem::{
    IdemKey, IdemLookup, IDEMPOTENCY_TTL_SECS, IDEM_PENDING_TTL_SECS, IDEM_PURGE_BATCH,
    MAX_IDEM_RESPONSE_BYTES,
};

use crate::error::{ErrorCode, OpError};
use crate::ports::{col_int, col_text, read_kv, SqlStore, StoreError};
use crate::schema;
use ruvector_edge_tenancy::validate::validate_edge_subject;
use ruvector_edge_tenancy::{IdentityCheck, LedgerMeta, QuotaLimits, Role, UidAllocator, Usage};
use std::collections::BTreeMap;

/// `ledger_meta` keys.
mod keys {
    pub const UID_SALT: &str = "uid_salt";
    pub const UID_NEXT_SEQ: &str = "uid_next_seq";
    pub const USAGE: &str = "usage";
    pub const USAGE_DAY: &str = "usage_day";
    pub const WORK_UNITS: &str = "work_units";
}

/// Seconds per UTC day (daily-op bucket).
pub const DAY_SECS: u64 = 86_400;

/// A membership row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    /// Role.
    pub role: Role,
    /// Inviter's `sub` (`None` for the claimant).
    pub invited_by: Option<String>,
    /// Unix seconds.
    pub created_at: u64,
}

/// Resident state of one `TenantLedger`.
#[derive(Debug, Clone)]
pub struct TenantLedger {
    identity: Option<LedgerMeta>,
    limits: QuotaLimits,
    members: BTreeMap<String, Member>,
    catalog: Vec<CatalogEntry>,
    usage: Usage,
    usage_day: u64,
    work_units: u64,
    uid: Option<UidAllocator>,
    poisoned: bool,
}

impl TenantLedger {
    /// Create the schema if needed and load everything (a ledger is small:
    /// ≤ 20 live collections plus tombstones and members).
    pub fn open(store: &dyn SqlStore, limits: QuotaLimits) -> Result<Self, StoreError> {
        for ddl in schema::LEDGER_SCHEMA {
            store.exec(ddl, &[])?;
        }
        let kv = read_kv(store, schema::LMETA_SELECT_ALL)?;
        let identity = LedgerMeta::from_kv(kv.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .map_err(|_| StoreError::Corrupt("ledger identity"))?;
        let get = |k: &str| kv.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
        let num = |k: &str| -> Result<u64, StoreError> {
            get(k).map_or(Ok(0), |v| {
                v.parse().map_err(|_| StoreError::Corrupt("ledger counter"))
            })
        };
        let uid = match get(keys::UID_SALT) {
            None => None,
            Some(hex_salt) => {
                let mut salt = [0u8; 32];
                hex::decode_to_slice(hex_salt, &mut salt)
                    .map_err(|_| StoreError::Corrupt("uid_salt"))?;
                Some(UidAllocator::new(salt, num(keys::UID_NEXT_SEQ)?))
            }
        };
        let usage = match get(keys::USAGE) {
            None => Usage::default(),
            Some(s) => serde_json::from_str(s).map_err(|_| StoreError::Corrupt("usage"))?,
        };
        let mut members = BTreeMap::new();
        for r in store.query(schema::MEMBER_SELECT_ALL, &[])? {
            let role = Role::parse(&col_text(&r, 1, "memberships.role")?)
                .ok_or(StoreError::Corrupt("memberships.role"))?;
            let invited_by = r.get(2).and_then(|v| v.as_text()).map(str::to_string);
            let created_at = u64::try_from(col_int(&r, 3, "memberships.created_at")?)
                .map_err(|_| StoreError::Corrupt("memberships.created_at"))?;
            members.insert(
                col_text(&r, 0, "memberships.sub")?,
                Member {
                    role,
                    invited_by,
                    created_at,
                },
            );
        }
        let catalog = catalog::load(store)?;
        Ok(TenantLedger {
            identity,
            limits,
            members,
            catalog,
            usage,
            usage_day: num(keys::USAGE_DAY)?,
            work_units: num(keys::WORK_UNITS)?,
            uid,
            poisoned: false,
        })
    }

    pub(crate) fn guard(
        &self,
        expected: &LedgerMeta,
        is_write: bool,
    ) -> Result<IdentityCheck, OpError> {
        if self.poisoned {
            return Err(OpError::new(
                ErrorCode::ShardUnavailable,
                "ledger must be reopened",
            ));
        }
        Ok(LedgerMeta::check(
            self.identity.as_ref(),
            expected,
            is_write,
        )?)
    }

    /// Write the identity row on first write (same batch as the write).
    pub(crate) fn init_identity(
        &self,
        store: &dyn SqlStore,
        check: IdentityCheck,
        expected: &LedgerMeta,
    ) -> Result<(), StoreError> {
        if check == IdentityCheck::InitializeOnWrite {
            for (k, v) in expected.to_kv() {
                store.exec(schema::LMETA_PUT, &[k.into(), v.into()])?;
            }
        }
        Ok(())
    }

    pub(crate) fn commit_identity(&mut self, check: IdentityCheck, expected: &LedgerMeta) {
        if check == IdentityCheck::InitializeOnWrite {
            self.identity = Some(expected.clone());
        }
    }

    pub(crate) fn poison<T>(&mut self, e: StoreError) -> Result<T, OpError> {
        self.poisoned = true;
        Err(e.into())
    }

    /// `true` after a storage error: the host must drop and reopen the
    /// ledger (resident state may disagree with storage).
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// The caller's role, `None` if not a member (default-deny).
    pub fn role_of(&self, expected: &LedgerMeta, sub: &str) -> Result<Option<Role>, OpError> {
        self.guard(expected, false)?;
        Ok(self.members.get(sub).map(|m| m.role))
    }

    /// `true` once someone has claimed the tenant.
    pub fn is_claimed(&self, expected: &LedgerMeta) -> Result<bool, OpError> {
        self.guard(expected, false)?;
        Ok(!self.members.is_empty())
    }

    /// `tenant:claim`: the first claimant of an unclaimed tenant becomes
    /// owner; afterwards `409 conflict`.
    pub fn claim(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        sub: &str,
        now: u64,
    ) -> Result<Role, OpError> {
        let check = self.guard(expected, true)?;
        validate_edge_subject(sub)?;
        if !self.members.is_empty() {
            return Err(OpError::new(ErrorCode::Conflict, "tenant already claimed"));
        }
        let res = self.init_identity(store, check, expected).and_then(|_| {
            store.exec(
                schema::MEMBER_INSERT,
                &[
                    sub.into(),
                    "owner".into(),
                    crate::ports::Value::Null,
                    ts(now)?.into(),
                ],
            )
        });
        if let Err(e) = res {
            return self.poison(e);
        }
        self.commit_identity(check, expected);
        self.members.insert(
            sub.to_string(),
            Member {
                role: Role::Owner,
                invited_by: None,
                created_at: now,
            },
        );
        Ok(Role::Owner)
    }

    /// Owner adds or changes a member (`viewer` / `editor` only: ownership
    /// transfer is an operator break-glass action, §4.2).
    pub fn put_member(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        actor: &str,
        sub: &str,
        role: Role,
        now: u64,
    ) -> Result<(), OpError> {
        let check = self.guard(expected, true)?;
        self.require_owner(actor)?;
        validate_edge_subject(sub).map_err(|_| OpError::invalid("invalid member sub"))?;
        if role == Role::Owner || sub == actor {
            return Err(OpError::invalid("role not assignable"));
        }
        let res = self.init_identity(store, check, expected).and_then(|_| {
            store.exec(
                schema::MEMBER_PUT,
                &[
                    sub.into(),
                    role.as_str().into(),
                    actor.into(),
                    ts(now)?.into(),
                ],
            )
        });
        if let Err(e) = res {
            return self.poison(e);
        }
        self.commit_identity(check, expected);
        self.members.insert(
            sub.to_string(),
            Member {
                role,
                invited_by: Some(actor.to_string()),
                created_at: now,
            },
        );
        Ok(())
    }

    /// Owner removes a (non-owner) member.
    pub fn remove_member(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        actor: &str,
        sub: &str,
    ) -> Result<(), OpError> {
        self.guard(expected, true)?;
        self.require_owner(actor)?;
        match self.members.get(sub) {
            None => return Err(OpError::not_found()),
            Some(m) if m.role == Role::Owner => {
                return Err(OpError::new(ErrorCode::Conflict, "cannot remove owner"))
            }
            Some(_) => {}
        }
        if let Err(e) = store.exec(schema::MEMBER_DELETE, &[sub.into()]) {
            return self.poison(e);
        }
        self.members.remove(sub);
        Ok(())
    }

    fn require_owner(&self, actor: &str) -> Result<(), OpError> {
        match self.members.get(actor) {
            Some(m) if m.role == Role::Owner => Ok(()),
            _ => Err(OpError::new(ErrorCode::RoleRequired, "owner required")),
        }
    }

    /// Members (for `GET /v1/tenant/members`).
    pub fn members(&self, expected: &LedgerMeta) -> Result<&BTreeMap<String, Member>, OpError> {
        self.guard(expected, false)?;
        Ok(&self.members)
    }
}

fn ts(now: u64) -> Result<i64, StoreError> {
    i64::try_from(now).map_err(|_| StoreError::Corrupt("timestamp"))
}
