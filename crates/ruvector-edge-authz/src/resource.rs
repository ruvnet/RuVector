//! RFC 8707 resource allowlist. The minted `aud` is always one of these
//! canonical URLs, e.g.
//! `https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp`.

use crate::error::{OAuthError, OAuthErrorCode};
use ruvector_edge_auth::ResourceUrl;

/// Protected resources this AS will mint tokens for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceAllowlist {
    resources: Vec<ResourceUrl>,
}

impl ResourceAllowlist {
    /// Build from canonical URLs.
    pub fn new(resources: Vec<ResourceUrl>) -> Self {
        ResourceAllowlist { resources }
    }

    /// Parse a comma-separated config value (wrangler `RESOURCE_ALLOWLIST`).
    /// Any invalid entry fails the whole list.
    pub fn from_config(value: &str) -> Result<Self, ruvector_edge_auth::AuthError> {
        let resources = value
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ResourceUrl::parse)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ResourceAllowlist { resources })
    }

    /// All allowed resources.
    pub fn resources(&self) -> &[ResourceUrl] {
        &self.resources
    }

    /// Resolve a requested `resource` parameter.
    ///
    /// Contract: exactly one `resource` value (the caller rejects repeats);
    /// must parse as a [`ResourceUrl`] and be byte-equal to an allowlisted
    /// entry; otherwise `invalid_target`. A missing parameter is also
    /// `invalid_target` (no default audience).
    pub fn resolve(&self, requested: Option<&str>) -> Result<ResourceUrl, OAuthError> {
        let err = OAuthError::new(OAuthErrorCode::InvalidTarget, "resource not allowed");
        let requested = requested.ok_or(err.clone())?;
        let parsed = ResourceUrl::parse(requested).map_err(|_| err.clone())?;
        self.resources
            .iter()
            .find(|r| **r == parsed)
            .cloned()
            .ok_or(err)
    }
}
