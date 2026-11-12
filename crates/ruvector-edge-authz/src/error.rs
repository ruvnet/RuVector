//! OAuth error responses (RFC 6749 §4.1.2.1 / §5.2, RFC 7591 §3.2.2).

use serde::Serialize;
use thiserror::Error;

/// Standard OAuth / DCR error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum OAuthErrorCode {
    InvalidRequest,
    InvalidClient,
    InvalidGrant,
    UnauthorizedClient,
    UnsupportedGrantType,
    UnsupportedResponseType,
    InvalidScope,
    AccessDenied,
    ServerError,
    TemporarilyUnavailable,
    /// RFC 8707.
    InvalidTarget,
    /// RFC 7591.
    InvalidRedirectUri,
    /// RFC 7591.
    InvalidClientMetadata,
}

impl OAuthErrorCode {
    /// HTTP status for token/registration endpoint responses.
    pub fn http_status(self) -> u16 {
        match self {
            OAuthErrorCode::InvalidClient => 401,
            OAuthErrorCode::ServerError => 500,
            OAuthErrorCode::TemporarilyUnavailable => 503,
            _ => 400,
        }
    }
}

/// An OAuth error with a safe, static description (never echoes input).
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize)]
#[error("{error:?}: {error_description}")]
pub struct OAuthError {
    /// Error code.
    pub error: OAuthErrorCode,
    /// Static human-readable description.
    pub error_description: &'static str,
}

impl OAuthError {
    /// Construct.
    pub const fn new(error: OAuthErrorCode, error_description: &'static str) -> Self {
        OAuthError {
            error,
            error_description,
        }
    }

    /// Scaffold placeholder: fails closed as `server_error`.
    pub const fn not_implemented(what: &'static str) -> Self {
        OAuthError {
            error: OAuthErrorCode::ServerError,
            error_description: what,
        }
    }

    /// JSON body.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| String::from(r#"{"error":"server_error"}"#))
    }
}

impl From<crate::ports::StoreError> for OAuthError {
    fn from(_: crate::ports::StoreError) -> Self {
        OAuthError::new(OAuthErrorCode::ServerError, "storage failure")
    }
}
