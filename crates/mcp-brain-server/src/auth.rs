//! API key validation and contributor pseudonym derivation

use axum::{
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use subtle::ConstantTimeEq;

/// Authenticated contributor extracted from request
#[derive(Debug, Clone)]
pub struct AuthenticatedContributor {
    pub pseudonym: String,
    pub api_key_prefix: String,
    pub is_system: bool,
    /// True when the caller presented a well-known placeholder key that
    /// shipped clients send when no key is configured (see
    /// [`SHARED_PLACEHOLDER_KEYS`]). Every such caller hashes to the same
    /// pseudonym, so it must not be treated as an owning identity.
    pub is_shared_identity: bool,
}

/// Placeholder keys sent by shipped clients when the user has not configured
/// a key (`@ruvector/pi-brain` and `mcp-brain` default to `anonymous`; the MCP
/// SSE transport historically defaulted to `mcp-sse-session`). They are public,
/// so they identify nobody: callers using them may read, but may not create,
/// modify or delete attributed content.
pub const SHARED_PLACEHOLDER_KEYS: &[&str] = &["anonymous", "mcp-sse-session"];

/// Whether `api_key` is one of the public placeholder keys (case-insensitive).
pub fn is_shared_placeholder_key(api_key: &str) -> bool {
    let k = api_key.trim();
    SHARED_PLACEHOLDER_KEYS
        .iter()
        .any(|p| k.eq_ignore_ascii_case(p))
}

impl AuthenticatedContributor {
    /// Reject callers whose identity is shared with every other keyless
    /// caller. Use on any handler that attributes content to the caller or
    /// changes content owned by someone else.
    pub fn require_owned_identity(&self) -> Result<(), (StatusCode, String)> {
        if self.is_shared_identity && !self.is_system {
            return Err((
                StatusCode::FORBIDDEN,
                "This operation needs a private API key. The placeholder key is shared by \
                 every keyless client and cannot own content. Generate a key and set \
                 BRAIN_API_KEY (see /.well-known/agent-guide.md)."
                    .into(),
            ));
        }
        Ok(())
    }

    /// Derive pseudonym from API key using SHAKE-256
    pub fn from_api_key(api_key: &str) -> Self {
        let mut hasher = Shake256::default();
        hasher.update(b"ruvector-brain-pseudonym:");
        hasher.update(api_key.as_bytes());
        let mut reader = hasher.finalize_xof();
        let mut buf = [0u8; 16];
        reader.read(&mut buf);
        let pseudonym = hex::encode(buf);
        let prefix = if api_key.len() >= 8 {
            api_key[..8].to_string()
        } else {
            api_key.to_string()
        };
        Self {
            pseudonym,
            api_key_prefix: prefix,
            is_system: false,
            is_shared_identity: is_shared_placeholder_key(api_key),
        }
    }

    pub fn system_seed() -> Self {
        Self {
            pseudonym: "ruvector-seed".to_string(),
            api_key_prefix: "system".to_string(),
            is_system: true,
            is_shared_identity: false,
        }
    }
}

/// Minimum API key length to prevent trivially weak keys.
const MIN_API_KEY_LEN: usize = 8;

/// Cached system key — read from env once, avoids per-request env::var lookup.
/// If BRAIN_SYSTEM_KEY is unset, system key authentication is disabled entirely
/// (no hardcoded fallback).
static SYSTEM_KEY: std::sync::LazyLock<Option<String>> = std::sync::LazyLock::new(|| {
    std::env::var("BRAIN_SYSTEM_KEY")
        .ok()
        .filter(|k| !k.is_empty())
});

#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthenticatedContributor
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let auth_header = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or((StatusCode::UNAUTHORIZED, "Missing Authorization header"))?;

        let api_key = auth_header
            .strip_prefix("Bearer ")
            .ok_or((StatusCode::UNAUTHORIZED, "Invalid Authorization format"))?;

        if api_key.len() < MIN_API_KEY_LEN || api_key.len() > 256 {
            return Err((StatusCode::UNAUTHORIZED, "Invalid API key"));
        }

        // Recognise system-level API keys (cached from BRAIN_SYSTEM_KEY env).
        // If BRAIN_SYSTEM_KEY is not set, system key auth is disabled — no hardcoded fallback.
        if let Some(ref system_key) = *SYSTEM_KEY {
            if api_key.as_bytes().ct_eq(system_key.as_bytes()).into() {
                return Ok(Self {
                    pseudonym: "ruvector-seed".to_string(),
                    api_key_prefix: "system".to_string(),
                    is_system: true,
                    is_shared_identity: false,
                });
            }
        }

        Ok(Self::from_api_key(api_key))
    }
}
