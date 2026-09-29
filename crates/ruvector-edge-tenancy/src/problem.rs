//! RFC 9457 `application/problem+json` bodies with stable codes (ADR-351 §7).

use serde::Serialize;

/// Media type for problem responses.
pub const PROBLEM_CONTENT_TYPE: &str = "application/problem+json";

/// Stable error codes from ADR §7 "Status codes".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum ProblemCode {
    InvalidRequest,
    DimensionMismatch,
    NonFiniteValue,
    InvalidToken,
    InsufficientScope,
    AudienceNotAllowed,
    RoleRequired,
    NotClaimed,
    TenantSuspended,
    NotFound,
    Conflict,
    PayloadTooLarge,
    QuotaExceeded,
    BudgetExceeded,
    IdempotencyMismatch,
    RateLimited,
    JwksUnavailable,
    ShardUnavailable,
    TrustRootMismatch,
    ServerError,
}

impl ProblemCode {
    /// Every code, in §7 order (for exhaustive tests and docs).
    pub const ALL: [ProblemCode; 20] = {
        use ProblemCode::*;
        [
            InvalidRequest,
            DimensionMismatch,
            NonFiniteValue,
            InvalidToken,
            InsufficientScope,
            AudienceNotAllowed,
            RoleRequired,
            NotClaimed,
            TenantSuspended,
            NotFound,
            Conflict,
            PayloadTooLarge,
            QuotaExceeded,
            BudgetExceeded,
            IdempotencyMismatch,
            RateLimited,
            JwksUnavailable,
            ShardUnavailable,
            TrustRootMismatch,
            ServerError,
        ]
    };

    /// `(status, code)` pair.
    pub fn status_and_code(self) -> (u16, &'static str) {
        use ProblemCode::*;
        match self {
            InvalidRequest => (400, "invalid_request"),
            DimensionMismatch => (400, "dimension_mismatch"),
            NonFiniteValue => (400, "non_finite_value"),
            InvalidToken => (401, "invalid_token"),
            InsufficientScope => (403, "insufficient_scope"),
            AudienceNotAllowed => (403, "audience_not_allowed"),
            RoleRequired => (403, "role_required"),
            NotClaimed => (403, "not_claimed"),
            TenantSuspended => (403, "tenant_suspended"),
            NotFound => (404, "not_found"),
            Conflict => (409, "conflict"),
            PayloadTooLarge => (413, "payload_too_large"),
            QuotaExceeded => (413, "quota_exceeded"),
            BudgetExceeded => (413, "budget_exceeded"),
            IdempotencyMismatch => (422, "idempotency_mismatch"),
            RateLimited => (429, "rate_limited"),
            JwksUnavailable => (503, "jwks_unavailable"),
            ShardUnavailable => (503, "shard_unavailable"),
            TrustRootMismatch => (503, "trust_root_mismatch"),
            ServerError => (500, "server_error"),
        }
    }
}

/// Problem document. `detail` must never echo tokens, vectors or payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Problem {
    /// `about:blank` or a docs URL.
    #[serde(rename = "type")]
    pub type_: String,
    /// Short human title.
    pub title: String,
    /// HTTP status.
    pub status: u16,
    /// Stable machine code.
    pub code: String,
    /// Optional safe detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Request id (mirrors `x-request-id`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl Problem {
    /// Build from a code with `type = about:blank` and `title = code`.
    pub fn new(code: ProblemCode) -> Self {
        let (status, c) = code.status_and_code();
        Problem {
            type_: "about:blank".into(),
            title: c.into(),
            status,
            code: c.into(),
            detail: None,
            request_id: None,
        }
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| String::from(r#"{"type":"about:blank","status":500,"code":"server_error","title":"server_error"}"#))
    }
}
