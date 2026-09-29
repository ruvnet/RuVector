//! Batch quota leases (ADR-351 §6.1): a `VectorShard` reserves rows, floats
//! and bytes from `TenantLedger` for a bounded time, consumes them locally
//! without a ledger hop per write, and an alarm reconciles the unused part.
//!
//! Pure and clock-injected (`now` is a parameter; no `std::time`), so it is
//! wasm32-safe and unit-testable. All arithmetic is checked.

use crate::quota::{admit, QuotaDelta, QuotaError, QuotaLimits, Usage};
use thiserror::Error;

/// Longest lease the ledger grants (10 min, §6.1).
pub const MAX_LEASE_TTL_SECS: u64 = 600;

/// Amounts of rows (vectors), floats and bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct LeaseAmount {
    pub rows: u64,
    pub floats: u64,
    pub bytes: u64,
}

impl LeaseAmount {
    fn checked_add(self, o: LeaseAmount) -> Option<LeaseAmount> {
        Some(LeaseAmount {
            rows: self.rows.checked_add(o.rows)?,
            floats: self.floats.checked_add(o.floats)?,
            bytes: self.bytes.checked_add(o.bytes)?,
        })
    }

    fn checked_sub(self, o: LeaseAmount) -> Option<LeaseAmount> {
        Some(LeaseAmount {
            rows: self.rows.checked_sub(o.rows)?,
            floats: self.floats.checked_sub(o.floats)?,
            bytes: self.bytes.checked_sub(o.bytes)?,
        })
    }

    fn fits_in(self, o: LeaseAmount) -> bool {
        self.rows <= o.rows && self.floats <= o.floats && self.bytes <= o.bytes
    }
}

/// Why a lease cannot cover a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LeaseError {
    /// `now >= expires_at`: ask the ledger for a new lease.
    #[error("quota lease expired")]
    Expired,
    /// The write needs more than the lease has left.
    #[error("quota lease exhausted")]
    Exhausted,
}

/// A granted reservation. Fields are private so `used <= reserved` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuotaLease {
    reserved: LeaseAmount,
    used: LeaseAmount,
    expires_at: u64,
}

impl QuotaLease {
    /// Reserved amounts.
    pub fn reserved(&self) -> LeaseAmount {
        self.reserved
    }
    /// Consumed so far.
    pub fn used(&self) -> LeaseAmount {
        self.used
    }
    /// Still available (`reserved - used`).
    pub fn remaining(&self) -> LeaseAmount {
        // Invariant used <= reserved; fall back to zero rather than panic.
        self.reserved.checked_sub(self.used).unwrap_or_default()
    }
    /// Unix seconds after which the lease is dead.
    pub fn expires_at(&self) -> u64 {
        self.expires_at
    }
    /// `true` once `now >= expires_at`.
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }

    /// Consume `amount` locally. All-or-nothing: on error nothing changes.
    pub fn consume(&mut self, amount: LeaseAmount, now: u64) -> Result<(), LeaseError> {
        if self.is_expired(now) {
            return Err(LeaseError::Expired);
        }
        match self.used.checked_add(amount) {
            Some(next) if next.fits_in(self.reserved) => {
                self.used = next;
                Ok(())
            }
            _ => Err(LeaseError::Exhausted),
        }
    }
}

fn to_i64(v: u64, over: QuotaError) -> Result<i64, QuotaError> {
    i64::try_from(v).map_err(|_| over)
}

/// Reserve `request` against `limits` at `now` for `ttl_secs` (clamped to
/// [`MAX_LEASE_TTL_SECS`]). Returns the lease and the ledger's new usage
/// (the reservation counts as used until [`reconcile`]). A zero TTL yields an
/// already-expired lease.
pub fn grant_lease(
    limits: &QuotaLimits,
    usage: &Usage,
    request: LeaseAmount,
    ttl_secs: u64,
    now: u64,
) -> Result<(QuotaLease, Usage), QuotaError> {
    let delta = QuotaDelta {
        collections: 0,
        vectors: to_i64(request.rows, QuotaError::Vectors)?,
        float_budget: to_i64(request.floats, QuotaError::FloatBudget)?,
        bytes: to_i64(request.bytes, QuotaError::Bytes)?,
        ops: 0,
    };
    let new_usage = admit(limits, usage, &delta)?;
    let lease = QuotaLease {
        reserved: request,
        used: LeaseAmount::default(),
        expires_at: now.saturating_add(ttl_secs.min(MAX_LEASE_TTL_SECS)),
    };
    Ok((lease, new_usage))
}

/// Return the unused part of `lease` to `usage` (alarm-driven
/// reconciliation). Releases never hit a limit; a release larger than usage
/// is [`QuotaError::Underflow`] (a ledger bug).
pub fn reconcile(usage: &Usage, lease: &QuotaLease) -> Result<Usage, QuotaError> {
    let unused = lease.remaining();
    let delta = QuotaDelta {
        collections: 0,
        vectors: -to_i64(unused.rows, QuotaError::Underflow)?,
        float_budget: -to_i64(unused.floats, QuotaError::Underflow)?,
        bytes: -to_i64(unused.bytes, QuotaError::Underflow)?,
        ops: 0,
    };
    // Limits are irrelevant for pure releases.
    let no_limits = QuotaLimits {
        max_collections: u32::MAX,
        max_vectors: u64::MAX,
        max_float_budget: u64::MAX,
        max_bytes: u64::MAX,
        max_daily_ops: u64::MAX,
    };
    admit(&no_limits, usage, &delta)
}
