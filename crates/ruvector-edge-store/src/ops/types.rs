//! `/v1/ops` envelope (ADR-351 §16.3).

use crate::error::{ErrorCode, OpError};
use ruvector_edge_tenancy::validate::is_forbidden_char;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value as Json;

/// Envelope version.
pub const OPS_VERSION: u32 = 1;
/// `op_id` length (ULID or a deterministic 26-char id).
pub const OP_ID_LEN: usize = 26;
/// `approval_ref` maximum length.
pub const MAX_APPROVAL_REF: usize = 128;

/// The §16.3 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Op {
    TenantMe,
    CollectionList,
    CollectionCreate,
    VectorUpsert,
    VectorQuery,
    VectorFetch,
    VectorDelete,
    UsageGet,
}

impl Op {
    /// Every op.
    pub const ALL: [Op; 8] = [
        Op::TenantMe,
        Op::CollectionList,
        Op::CollectionCreate,
        Op::VectorUpsert,
        Op::VectorQuery,
        Op::VectorFetch,
        Op::VectorDelete,
        Op::UsageGet,
    ];

    /// Wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Op::TenantMe => "tenant_me",
            Op::CollectionList => "collection_list",
            Op::CollectionCreate => "collection_create",
            Op::VectorUpsert => "vector_upsert",
            Op::VectorQuery => "vector_query",
            Op::VectorFetch => "vector_fetch",
            Op::VectorDelete => "vector_delete",
            Op::UsageGet => "usage_get",
        }
    }

    /// Strict parse.
    pub fn parse(s: &str) -> Option<Op> {
        Op::ALL.into_iter().find(|o| o.as_str() == s)
    }
}

/// The request envelope, strictly parsed (unknown fields rejected).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpRequest {
    /// Must be 1.
    pub v: u32,
    /// 26 chars `[0-9A-Za-z]`; also the `Idempotency-Key`.
    pub op_id: String,
    /// Canonical URL of this route.
    pub target: String,
    /// The adapter's view of the caller's tenant.
    pub tenant_key: String,
    /// Operation name.
    pub op: String,
    /// Operation arguments (the matching REST body plus `collection`), kept
    /// as raw JSON text and parsed once, straight into the per-op type.
    /// Absent means `{}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Box<RawValue>>,
    /// Validate and report the effect without writing.
    #[serde(default)]
    pub dry_run: bool,
    /// Optional approval reference (≤ 128 chars), audited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_ref: Option<String>,
}

impl OpRequest {
    /// Raw `args` JSON text (`{}` when absent).
    pub fn args_text(&self) -> &str {
        self.args.as_deref().map_or("{}", RawValue::get)
    }

    /// Shape checks that need no state.
    pub fn validate_shape(&self) -> Result<(), OpError> {
        if self.v != OPS_VERSION {
            return Err(OpError::invalid("unsupported envelope version"));
        }
        if self.op_id.len() != OP_ID_LEN || !self.op_id.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return Err(OpError::invalid("malformed op_id"));
        }
        if let Some(a) = &self.approval_ref {
            if a.is_empty() || a.len() > MAX_APPROVAL_REF || a.chars().any(is_forbidden_char) {
                return Err(OpError::invalid("malformed approval_ref"));
            }
        }
        if !self.args_text().trim_start().starts_with('{') {
            return Err(OpError::invalid("args must be an object"));
        }
        Ok(())
    }
}

/// Per-call usage report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpUsage {
    /// Work units charged.
    pub work_units: u64,
    /// Rows written, returned or scanned.
    pub rows: u64,
    /// Bytes written (absolute delta).
    pub bytes: u64,
}

/// Wire error body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireError {
    /// Stable code.
    pub code: ErrorCode,
    /// HTTP status.
    pub status: u16,
    /// For `rate_limited`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_s: Option<u32>,
    /// For `insufficient_scope`: the scope to request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

/// The response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpResponse {
    /// 1.
    pub v: u32,
    /// Echoed `op_id` (empty if the envelope did not parse).
    pub op_id: String,
    /// Success flag.
    pub ok: bool,
    /// Result on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Json>,
    /// Usage on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpUsage>,
    /// Error on failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<WireError>,
}

impl OpResponse {
    /// Success.
    pub fn ok(op_id: &str, result: Json, usage: OpUsage) -> Self {
        OpResponse {
            v: OPS_VERSION,
            op_id: op_id.to_string(),
            ok: true,
            result: Some(result),
            usage: Some(usage),
            error: None,
        }
    }

    /// Failure.
    pub fn err(op_id: &str, e: &OpError) -> Self {
        OpResponse {
            v: OPS_VERSION,
            op_id: op_id.to_string(),
            ok: false,
            result: None,
            usage: None,
            error: Some(WireError {
                code: e.code,
                status: e.code.status(),
                retry_after_s: None,
                scope: e.scope.map(str::to_string),
            }),
        }
    }

    /// HTTP status (200 on success).
    pub fn status(&self) -> u16 {
        self.error.as_ref().map_or(200, |e| e.status)
    }
}

/// Deterministic JSON with object keys sorted at every level (independent
/// of serde_json's `preserve_order` feature), for body hashing.
pub fn canonical_json(v: &Json, out: &mut String) {
    match v {
        Json::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Json::String(k.clone()).to_string());
                out.push(':');
                canonical_json(&m[k], out);
            }
            out.push('}');
        }
        Json::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonical_json(x, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// `ErrorCode` for a malformed envelope.
pub(crate) fn malformed() -> OpError {
    OpError::new(ErrorCode::InvalidRequest, "malformed envelope")
}
