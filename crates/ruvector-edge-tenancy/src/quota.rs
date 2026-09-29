//! Quota types and admission (ADR-351 §6.2 `TenantLedger`, §10).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Plan limits for one tenant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotaLimits {
    /// Maximum live collections.
    pub max_collections: u32,
    /// Maximum stored vectors across collections.
    pub max_vectors: u64,
    /// Maximum `sum(dim * count)` floats held in memory.
    pub max_float_budget: u64,
    /// Maximum stored bytes (vectors + metadata).
    pub max_bytes: u64,
    /// Maximum operations per UTC day.
    pub max_daily_ops: u64,
}

/// Current usage counters (strongly consistent in `TenantLedger`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Usage {
    pub collections: u32,
    pub vectors: u64,
    pub float_budget: u64,
    pub bytes: u64,
    pub daily_ops: u64,
}

/// A requested change to usage (negative deltas are releases).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct QuotaDelta {
    pub collections: i32,
    pub vectors: i64,
    pub float_budget: i64,
    pub bytes: i64,
    pub ops: u64,
}

/// Which limit would be exceeded (maps to `413 quota_exceeded`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[allow(missing_docs)]
pub enum QuotaError {
    #[error("collection quota exceeded")]
    Collections,
    #[error("vector quota exceeded")]
    Vectors,
    #[error("float budget exceeded")]
    FloatBudget,
    #[error("storage quota exceeded")]
    Bytes,
    #[error("daily operation quota exceeded")]
    DailyOps,
    /// Counter would underflow (a release larger than usage): a ledger bug.
    #[error("usage underflow")]
    Underflow,
    /// Scaffold placeholder. Fails closed.
    #[error("quota admission not implemented")]
    NotImplemented,
}

/// Admit `delta` against `limits` given `usage`.
///
/// Contract: pure and total; checked arithmetic (no overflow/underflow
/// panics); returns the new [`Usage`] if every limit holds, else the first
/// violated [`QuotaError`] in field order. Releases always succeed unless they
/// underflow.
pub fn admit(limits: &QuotaLimits, usage: &Usage, delta: &QuotaDelta) -> Result<Usage, QuotaError> {
    let _ = (limits, usage, delta);
    Err(QuotaError::NotImplemented)
}
