//! Operation error codes (ADR-351 §7 status codes and the §16.3 table).

use crate::ports::StoreError;
use ruvector_edge_tenancy::{QuotaError, TenancyError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stable machine codes returned in `OpResponse.error.code` and REST problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum ErrorCode {
    InvalidToken,
    InsufficientScope,
    RoleRequired,
    NotClaimed,
    TargetMismatch,
    TenantMismatch,
    OpReplayed,
    UnknownOp,
    InvalidRequest,
    DimensionMismatch,
    NonFiniteValue,
    NotFound,
    Conflict,
    QuotaExceeded,
    BudgetExceeded,
    PayloadTooLarge,
    RateLimited,
    JwksUnavailable,
    ShardUnavailable,
    ServerError,
}

impl ErrorCode {
    /// Every code, for exhaustive tests.
    pub const ALL: [ErrorCode; 20] = {
        use ErrorCode::*;
        [
            InvalidToken,
            InsufficientScope,
            RoleRequired,
            NotClaimed,
            TargetMismatch,
            TenantMismatch,
            OpReplayed,
            UnknownOp,
            InvalidRequest,
            DimensionMismatch,
            NonFiniteValue,
            NotFound,
            Conflict,
            QuotaExceeded,
            BudgetExceeded,
            PayloadTooLarge,
            RateLimited,
            JwksUnavailable,
            ShardUnavailable,
            ServerError,
        ]
    };

    /// HTTP status (§16.3 table; `non_finite_value` and `conflict` from §7).
    pub fn status(self) -> u16 {
        use ErrorCode::*;
        match self {
            InvalidToken => 401,
            InsufficientScope | RoleRequired | NotClaimed | TenantMismatch => 403,
            TargetMismatch | UnknownOp | InvalidRequest | DimensionMismatch | NonFiniteValue => 400,
            OpReplayed | Conflict => 409,
            NotFound => 404,
            QuotaExceeded | BudgetExceeded | PayloadTooLarge => 413,
            RateLimited => 429,
            JwksUnavailable | ShardUnavailable => 503,
            ServerError => 500,
        }
    }

    /// Wire string.
    pub fn as_str(self) -> &'static str {
        use ErrorCode::*;
        match self {
            InvalidToken => "invalid_token",
            InsufficientScope => "insufficient_scope",
            RoleRequired => "role_required",
            NotClaimed => "not_claimed",
            TargetMismatch => "target_mismatch",
            TenantMismatch => "tenant_mismatch",
            OpReplayed => "op_replayed",
            UnknownOp => "unknown_op",
            InvalidRequest => "invalid_request",
            DimensionMismatch => "dimension_mismatch",
            NonFiniteValue => "non_finite_value",
            NotFound => "not_found",
            Conflict => "conflict",
            QuotaExceeded => "quota_exceeded",
            BudgetExceeded => "budget_exceeded",
            PayloadTooLarge => "payload_too_large",
            RateLimited => "rate_limited",
            JwksUnavailable => "jwks_unavailable",
            ShardUnavailable => "shard_unavailable",
            ServerError => "server_error",
        }
    }

    /// Retryable outcomes are never stored in the idempotency table.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            ErrorCode::RateLimited
                | ErrorCode::JwksUnavailable
                | ErrorCode::ShardUnavailable
                | ErrorCode::ServerError
        )
    }
}

/// An operation failure. `detail` is a static string: it never echoes input.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{code:?}: {detail}")]
pub struct OpError {
    /// Stable code.
    pub code: ErrorCode,
    /// Static, non-echoing detail.
    pub detail: &'static str,
    /// For `insufficient_scope`: the scope to request.
    pub scope: Option<&'static str>,
}

impl OpError {
    /// Build an error.
    pub fn new(code: ErrorCode, detail: &'static str) -> Self {
        OpError {
            code,
            detail,
            scope: None,
        }
    }
    /// `invalid_request` shorthand.
    pub fn invalid(detail: &'static str) -> Self {
        OpError::new(ErrorCode::InvalidRequest, detail)
    }
    /// `not_found` shorthand.
    pub fn not_found() -> Self {
        OpError::new(ErrorCode::NotFound, "not found")
    }
}

impl From<StoreError> for OpError {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::Backend(_) => OpError::new(ErrorCode::ShardUnavailable, "storage backend"),
            StoreError::Constraint => OpError::new(ErrorCode::Conflict, "constraint"),
            StoreError::Corrupt(_) => OpError::new(ErrorCode::ServerError, "corrupt storage"),
        }
    }
}

impl From<TenancyError> for OpError {
    fn from(e: TenancyError) -> Self {
        match e {
            TenancyError::NotFound => OpError::not_found(),
            TenancyError::InvalidTenantClaim(_) => {
                OpError::new(ErrorCode::InvalidToken, "invalid tenant claim")
            }
            TenancyError::InvalidCollectionName => OpError::invalid("invalid collection name"),
            TenancyError::InvalidVectorId => OpError::invalid("invalid vector id"),
            TenancyError::InvalidShard => OpError::invalid("invalid shard"),
            TenancyError::MalformedIdentifier(_) | TenancyError::UidSequenceExhausted => {
                OpError::new(ErrorCode::ServerError, "malformed stored identifier")
            }
        }
    }
}

/// Tenant quota counters → `413 quota_exceeded` (an underflow is a ledger
/// bug → `500`).
impl From<QuotaError> for OpError {
    fn from(e: QuotaError) -> Self {
        match e {
            QuotaError::Underflow => OpError::new(ErrorCode::ServerError, "usage underflow"),
            _ => OpError::new(ErrorCode::QuotaExceeded, "tenant quota exceeded"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_strings_match_serde_and_statuses_match_adr() {
        for c in ErrorCode::ALL {
            let json = serde_json::to_string(&c).unwrap();
            assert_eq!(json, format!("\"{}\"", c.as_str()));
        }
        assert_eq!(ErrorCode::TargetMismatch.status(), 400);
        assert_eq!(ErrorCode::TenantMismatch.status(), 403);
        assert_eq!(ErrorCode::OpReplayed.status(), 409);
        assert_eq!(ErrorCode::BudgetExceeded.status(), 413);
        assert_eq!(ErrorCode::QuotaExceeded.status(), 413);
        assert_eq!(ErrorCode::ShardUnavailable.status(), 503);
    }
}
