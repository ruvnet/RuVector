//! API key validation and contributor pseudonym derivation

use axum::{
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
};
use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};

/// Authenticated contributor extracted from request
#[derive(Debug, Clone)]
pub struct AuthenticatedContributor {
    pub pseudonym: String,
    pub api_key_prefix: String,
    pub is_system: bool,
}

impl AuthenticatedContributor {
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
        }
    }

    pub fn system_seed() -> Self {
        Self {
            pseudonym: "ruvector-seed".to_string(),
            api_key_prefix: "system".to_string(),
            is_system: true,
        }
    }
}

/// Cached system key — read from env once, avoids per-request env::var lookup.
static SYSTEM_KEY: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    std::env::var("BRAIN_SYSTEM_KEY")
        .unwrap_or_else(|_| "brainpedia-author-key".to_string())
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

        if api_key.is_empty() || api_key.len() > 256 {
            return Err((StatusCode::UNAUTHORIZED, "Invalid API key"));
        }

        // Recognise system-level API keys (cached from BRAIN_SYSTEM_KEY env)
        if api_key == SYSTEM_KEY.as_str() {
            return Ok(Self {
                pseudonym: "ruvector-seed".to_string(),
                api_key_prefix: "system".to_string(),
                is_system: true,
            });
        }

        Ok(Self::from_api_key(api_key))
    }
}
