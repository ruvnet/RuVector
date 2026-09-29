//! `op_id` idempotency (ADR-351 §7 `Idempotency-Key`, §16.3): remembered
//! 24 h per `(tenant, sub)`, bound to a hash of the request body. The same
//! body replays the stored response verbatim; a different body is `409
//! op_replayed`.
//!
//! Bounded growth (§6.1): every stored row is charged to the tenant's
//! `bytes` usage (admission-checked; over quota the response is simply not
//! remembered), and expired rows are purged in batches of at most
//! [`IDEM_PURGE_BATCH`] through the `expires_at` index, releasing their
//! bytes. Responses larger than [`MAX_IDEM_RESPONSE_BYTES`] are not stored.
//! What gets remembered at all is the dispatcher's decision (successful
//! mutating ops only).

use super::TenantLedger;
use crate::error::OpError;
use crate::ports::{col_text, SqlStore, Value};
use crate::schema;
use ruvector_edge_tenancy::{LedgerMeta, QuotaDelta};

/// Replay window (24 h).
pub const IDEMPOTENCY_TTL_SECS: u64 = 86_400;
/// Expired rows purged per store.
pub const IDEM_PURGE_BATCH: i64 = 100;
/// Largest response remembered (64 KiB).
pub const MAX_IDEM_RESPONSE_BYTES: usize = 64 << 10;
/// Lifetime of an in-flight reservation ([`TenantLedger::idem_reserve`]):
/// a crashed request's key frees itself after this.
pub const IDEM_PENDING_TTL_SECS: u64 = 120;

/// The `(sub, op_id, body hash)` an idempotency row is keyed and bound on.
#[derive(Debug, Clone, Copy)]
pub struct IdemKey<'a> {
    /// Caller subject.
    pub sub: &'a str,
    /// `op_id` / `Idempotency-Key`.
    pub key: &'a str,
    /// sha256 of the request body.
    pub body_sha256: &'a [u8; 32],
}

impl IdemKey<'_> {
    /// Bytes a stored row with `response` is charged.
    pub fn row_bytes(&self, response: &str) -> u64 {
        (self.sub.len() + self.key.len() + self.body_sha256.len() + response.len() + 16) as u64
    }
}

/// Result of an `op_id` lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdemLookup {
    /// Not seen (or expired): execute.
    Miss,
    /// Seen with the same body: return this stored response.
    Replay(String),
    /// Seen with a different body: `409 op_replayed`.
    Conflict,
    /// Reserved by a request with the same body that has not finished:
    /// `409 conflict`, retry shortly.
    InFlight,
}

fn i64_of(v: u64) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

fn bytes_col(r: &[Value], i: usize) -> i64 {
    r.get(i).and_then(Value::as_int).unwrap_or(0).max(0)
}

/// An `IDEM_SELECT` row reserved by [`TenantLedger::idem_reserve`] and not
/// yet completed: its `response` is `NULL`.
fn is_pending(r: &[Value]) -> bool {
    matches!(r.get(1), Some(Value::Null) | None)
}

impl TenantLedger {
    /// Look up `(sub, key)` and, on a miss, reserve it for this request in
    /// the same call (one DO turn, so two concurrent requests with one key
    /// never both miss): a pending row (`response NULL`, uncharged,
    /// [`IDEM_PENDING_TTL_SECS`]) that [`TenantLedger::idem_store`]
    /// replaces and [`TenantLedger::idem_release`] clears. An expired row
    /// the reservation replaces has its charged bytes released.
    pub fn idem_reserve(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        k: IdemKey<'_>,
        now: u64,
    ) -> Result<IdemLookup, OpError> {
        let found = self.idem_lookup(store, expected, k, now)?;
        if found != IdemLookup::Miss {
            return Ok(found);
        }
        let check = self.guard(expected, true)?;
        let stale = store
            .query(schema::IDEM_SELECT, &[k.sub.into(), k.key.into()])?
            .first()
            .map_or(0, |r| bytes_col(r, 3));
        let expires = i64_of(now.saturating_add(IDEM_PENDING_TTL_SECS));
        let res = self.init_identity(store, check, expected).and_then(|_| {
            store.exec(
                schema::IDEM_PUT,
                &[
                    k.sub.into(),
                    k.key.into(),
                    k.body_sha256.to_vec().into(),
                    Value::Null,
                    0i64.into(),
                    expires.into(),
                ],
            )
        });
        if let Err(e) = res {
            return self.poison(e);
        }
        if stale > 0 {
            let release = QuotaDelta {
                bytes: -stale,
                ..Default::default()
            };
            let next = self.plan_adjust(&release, now);
            self.commit_usage(store, check, expected, next, 0, now)?;
        } else {
            self.commit_identity(check, expected);
        }
        Ok(IdemLookup::Miss)
    }

    /// Clear this request's reservation of `(sub, key)` (the op failed or
    /// is not remembered): only a pending row with the same body hash.
    pub fn idem_release(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        k: IdemKey<'_>,
    ) -> Result<(), OpError> {
        self.guard(expected, true)?;
        let rows = store.query(schema::IDEM_SELECT, &[k.sub.into(), k.key.into()])?;
        let ours = rows.first().is_some_and(|r| {
            is_pending(r) && r.first().and_then(Value::as_blob) == Some(k.body_sha256.as_slice())
        });
        if ours {
            if let Err(e) = store.exec(schema::IDEM_DELETE, &[k.sub.into(), k.key.into()]) {
                return self.poison(e);
            }
        }
        Ok(())
    }

    /// Look up `(sub, key)` at `now`.
    pub fn idem_lookup(
        &self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        k: IdemKey<'_>,
        now: u64,
    ) -> Result<IdemLookup, OpError> {
        self.guard(expected, false)?;
        let rows = store.query(schema::IDEM_SELECT, &[k.sub.into(), k.key.into()])?;
        let Some(r) = rows.first() else {
            return Ok(IdemLookup::Miss);
        };
        let expires = r.get(2).and_then(Value::as_int).unwrap_or(0);
        if expires <= i64_of(now) {
            return Ok(IdemLookup::Miss);
        }
        let stored = r.first().and_then(Value::as_blob).unwrap_or(&[]);
        if stored != k.body_sha256.as_slice() {
            return Ok(IdemLookup::Conflict);
        }
        if is_pending(r) {
            return Ok(IdemLookup::InFlight);
        }
        let resp = r.get(1).and_then(Value::as_text).unwrap_or("").to_string();
        Ok(IdemLookup::Replay(resp))
    }

    /// Purge a batch of expired rows and remember `response` for `(sub,
    /// key)` until `now + 24 h`, charging its bytes. Returns `true` if the
    /// response was stored (`false`: too large or over the bytes quota; the
    /// purge still happens). Statements are issued before the counters, so
    /// a torn store never releases bytes twice (it can leave purged bytes
    /// counted, or one stored row of ≤ 64 KiB uncharged).
    pub fn idem_store(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        k: IdemKey<'_>,
        response: &str,
        now: u64,
    ) -> Result<bool, OpError> {
        let check = self.guard(expected, true)?;
        let expired = store.query(
            schema::IDEM_EXPIRED,
            &[i64_of(now).into(), IDEM_PURGE_BATCH.into()],
        )?;
        let mut purge = Vec::with_capacity(expired.len());
        let mut released: i64 = 0;
        for r in &expired {
            let (sub, key) = (col_text(r, 0, "idem.sub")?, col_text(r, 1, "idem.key")?);
            released = released.saturating_add(bytes_col(r, 2));
            purge.push((sub, key));
        }
        // The key being stored may hold an expired row beyond this batch;
        // it is released only if the new row replaces it.
        let (mut existing, mut pending_here) = (0i64, false);
        if !purge.iter().any(|(s, q)| s == k.sub && q == k.key) {
            if let Some(r) = store
                .query(schema::IDEM_SELECT, &[k.sub.into(), k.key.into()])?
                .first()
            {
                existing = bytes_col(r, 3);
                pending_here = is_pending(r);
            }
        }
        let row = i64_of(k.row_bytes(response));
        let release = QuotaDelta {
            bytes: -released,
            ..Default::default()
        };
        let grow = QuotaDelta {
            bytes: row.saturating_sub(released).saturating_sub(existing),
            ..Default::default()
        };
        let (next, keep) = if response.len() > MAX_IDEM_RESPONSE_BYTES {
            (self.plan_adjust(&release, now), false)
        } else if grow.bytes <= 0 {
            (self.plan_adjust(&grow, now), true)
        } else {
            match self.plan_admit(&grow, now) {
                Ok(u) => (u, true),
                Err(_) => (self.plan_adjust(&release, now), false),
            }
        };
        // Not remembered: clear this key's reservation, so a retry
        // re-executes instead of waiting out the pending TTL.
        let clear = !keep && pending_here;
        if purge.is_empty() && !keep && !clear {
            return Ok(false);
        }
        let expires = i64_of(now.saturating_add(IDEMPOTENCY_TTL_SECS));
        let res = self.init_identity(store, check, expected).and_then(|_| {
            for (sub, key) in &purge {
                store.exec(
                    schema::IDEM_DELETE,
                    &[sub.as_str().into(), key.as_str().into()],
                )?;
            }
            if clear {
                store.exec(schema::IDEM_DELETE, &[k.sub.into(), k.key.into()])?;
            }
            if keep {
                store.exec(
                    schema::IDEM_PUT,
                    &[
                        k.sub.into(),
                        k.key.into(),
                        k.body_sha256.to_vec().into(),
                        response.into(),
                        row.into(),
                        expires.into(),
                    ],
                )?;
            }
            Ok(())
        });
        if let Err(e) = res {
            return self.poison(e);
        }
        self.commit_usage(store, check, expected, next, 0, now)?;
        Ok(keep)
    }
}
