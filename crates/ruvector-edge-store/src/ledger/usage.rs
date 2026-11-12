//! Quota counters (ADR-351 §6.2, §10 layer 4): admission with limits for
//! caller-driven growth, and limit-free corrections for refunds,
//! reconciliation and post-commit releases.

use super::{keys, TenantLedger, DAY_SECS};
use crate::error::{ErrorCode, OpError};
use crate::ports::SqlStore;
use crate::schema;
use ruvector_edge_tenancy::{admit, IdentityCheck, LedgerMeta, QuotaDelta, QuotaLimits, Usage};

fn clamp_add(used: u64, delta: i64) -> u64 {
    if delta >= 0 {
        used.saturating_add(delta.unsigned_abs())
    } else {
        used.saturating_sub(delta.unsigned_abs())
    }
}

impl TenantLedger {
    /// Usage with the daily bucket rolled to `now`'s UTC day.
    fn base(&self, now: u64) -> (Usage, u64) {
        let day = now / DAY_SECS;
        let mut base = self.usage;
        if day != self.usage_day {
            base.daily_ops = 0;
        }
        (base, day)
    }

    /// Pure admission check: the usage `delta` would produce (§10 layer 4,
    /// checked arithmetic; growth beyond a limit is `413 quota_exceeded`).
    pub(crate) fn plan_admit(&self, delta: &QuotaDelta, now: u64) -> Result<Usage, OpError> {
        let (base, _) = self.base(now);
        Ok(admit(&self.limits, &base, delta)?)
    }

    /// Persist new counters (identity on first write), then commit them.
    pub(crate) fn commit_usage(
        &mut self,
        store: &dyn SqlStore,
        check: IdentityCheck,
        expected: &LedgerMeta,
        next: Usage,
        work_units: u64,
        now: u64,
    ) -> Result<Usage, OpError> {
        let day = now / DAY_SECS;
        let wu = self
            .work_units
            .checked_add(work_units)
            .ok_or(OpError::new(ErrorCode::QuotaExceeded, "work units"))?;
        let usage_json = serde_json::to_string(&next)
            .map_err(|_| OpError::new(ErrorCode::ServerError, "usage encode"))?;
        let res = self.init_identity(store, check, expected).and_then(|_| {
            store.exec(schema::LMETA_PUT, &[keys::USAGE.into(), usage_json.into()])?;
            store.exec(
                schema::LMETA_PUT,
                &[keys::USAGE_DAY.into(), day.to_string().into()],
            )?;
            store.exec(
                schema::LMETA_PUT,
                &[keys::WORK_UNITS.into(), wu.to_string().into()],
            )
        });
        if let Err(e) = res {
            return self.poison(e);
        }
        self.commit_identity(check, expected);
        self.usage = next;
        self.usage_day = day;
        self.work_units = wu;
        Ok(next)
    }

    /// Admit a usage change (§10 layer 4) with checked arithmetic: growth is
    /// checked against the plan limits (`413 quota_exceeded`), releases always
    /// succeed. `ops` counts toward the daily bucket, which rolls over at UTC
    /// midnight; `work_units` accumulates.
    pub fn admit(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        delta: QuotaDelta,
        work_units: u64,
        now: u64,
    ) -> Result<Usage, OpError> {
        let check = self.guard(expected, true)?;
        let next = self.plan_admit(&delta, now)?;
        self.commit_usage(store, check, expected, next, work_units, now)
    }

    /// The usage an internal correction would produce: signed, no limit
    /// checks, releases clamp at zero (never an underflow error).
    pub(crate) fn plan_adjust(&self, delta: &QuotaDelta, now: u64) -> Usage {
        let (base, _) = self.base(now);
        Usage {
            collections: u32::try_from(clamp_add(
                u64::from(base.collections),
                i64::from(delta.collections),
            ))
            .unwrap_or(u32::MAX),
            vectors: clamp_add(base.vectors, delta.vectors),
            float_budget: clamp_add(base.float_budget, delta.float_budget),
            bytes: clamp_add(base.bytes, delta.bytes),
            daily_ops: base.daily_ops.saturating_add(delta.ops),
        }
    }

    /// Apply an internal correction — a refund of a write that did not
    /// happen, a reconciliation after a torn write, or a release after a
    /// committed delete — without limit checks. A release that would go
    /// below zero clamps (the counter was already short), so a correction
    /// never fails after data has changed.
    pub fn adjust(
        &mut self,
        store: &dyn SqlStore,
        expected: &LedgerMeta,
        delta: QuotaDelta,
        now: u64,
    ) -> Result<Usage, OpError> {
        let check = self.guard(expected, true)?;
        let next = self.plan_adjust(&delta, now);
        self.commit_usage(store, check, expected, next, 0, now)
    }

    /// Current usage (daily ops reported as of `now`'s day).
    pub fn usage(&self, expected: &LedgerMeta, now: u64) -> Result<Usage, OpError> {
        self.guard(expected, false)?;
        Ok(self.base(now).0)
    }

    /// Plan limits.
    pub fn limits(&self) -> &QuotaLimits {
        &self.limits
    }

    /// Accumulated work units.
    pub fn work_units(&self) -> u64 {
        self.work_units
    }
}
