//! Quota types and admission (ADR-351 §6.2 `TenantLedger`, §10).
//!
//! All arithmetic is checked: nothing here can panic or wrap.

use crate::problem::ProblemCode;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// ADR §10 M1 defaults that are fixed numbers in the ADR (not `[U]`).
pub mod limits {
    /// Collections per tenant.
    pub const MAX_COLLECTIONS: u32 = 20;
    /// Minimum vector dimension.
    pub const MIN_DIMENSION: u32 = 1;
    /// Maximum vector dimension (fixed per collection).
    pub const MAX_DIMENSION: u32 = 1536;
    /// Per-shard cap in floats at M1 (12 MB of f32).
    pub const M1_SHARD_FLOAT_CAP: u64 = 3_000_000;
    /// Maximum vectors per upsert batch.
    pub const MAX_UPSERT_BATCH: u32 = 500;
    /// Maximum upsert body bytes (1 MiB).
    pub const MAX_UPSERT_BYTES: u64 = 1 << 20;
    /// Maximum `top_k`.
    pub const MAX_TOP_K: u32 = 100;
    /// Maximum metadata bytes per vector (4 KiB).
    pub const MAX_METADATA_BYTES: u64 = 4 << 10;
}

/// `true` if `dim` is in `MIN_DIMENSION..=MAX_DIMENSION`.
pub fn dimension_ok(dim: u32) -> bool {
    (limits::MIN_DIMENSION..=limits::MAX_DIMENSION).contains(&dim)
}

/// Float-budget cost of `count` vectors of dimension `dim` (`dim * count`),
/// or `None` on overflow.
pub fn float_cost(dim: u32, count: u64) -> Option<u64> {
    u64::from(dim).checked_mul(count)
}

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
}

impl QuotaError {
    /// `Underflow` is an internal fault (500); every other variant is
    /// `413 quota_exceeded`.
    pub fn problem_code(self) -> ProblemCode {
        match self {
            QuotaError::Underflow => ProblemCode::ServerError,
            _ => ProblemCode::QuotaExceeded,
        }
    }
}

/// Apply a signed delta to one counter, enforcing its limit only when the
/// delta grows usage. Overflow counts as exceeding the limit.
fn apply_u64(used: u64, delta: i64, limit: u64, over: QuotaError) -> Result<u64, QuotaError> {
    match used.checked_add_signed(delta) {
        None if delta < 0 => Err(QuotaError::Underflow),
        None => Err(over),
        Some(new) if delta > 0 && new > limit => Err(over),
        Some(new) => Ok(new),
    }
}

/// Admit `delta` against `limits` given `usage`.
///
/// Pure and total. Returns the new [`Usage`] if every field is admitted, else
/// the first failure in field order (collections, vectors, float budget,
/// bytes, daily ops). A field whose delta is `<= 0` skips its limit check, so
/// releases always succeed (even when usage is already over a lowered limit)
/// unless they would underflow. A counter that would overflow reports that
/// field's limit error. `ops` is always additive.
pub fn admit(limits: &QuotaLimits, usage: &Usage, delta: &QuotaDelta) -> Result<Usage, QuotaError> {
    let collections = u32::try_from(apply_u64(
        u64::from(usage.collections),
        i64::from(delta.collections),
        u64::from(limits.max_collections),
        QuotaError::Collections,
    )?)
    .map_err(|_| QuotaError::Collections)?;
    let vectors = apply_u64(
        usage.vectors,
        delta.vectors,
        limits.max_vectors,
        QuotaError::Vectors,
    )?;
    let float_budget = apply_u64(
        usage.float_budget,
        delta.float_budget,
        limits.max_float_budget,
        QuotaError::FloatBudget,
    )?;
    let bytes = apply_u64(
        usage.bytes,
        delta.bytes,
        limits.max_bytes,
        QuotaError::Bytes,
    )?;
    let daily_ops = match usage.daily_ops.checked_add(delta.ops) {
        Some(n) if delta.ops == 0 || n <= limits.max_daily_ops => n,
        _ => return Err(QuotaError::DailyOps),
    };
    Ok(Usage {
        collections,
        vectors,
        float_budget,
        bytes,
        daily_ops,
    })
}
