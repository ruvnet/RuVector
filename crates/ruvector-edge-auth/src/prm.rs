//! RFC 9728 OAuth 2.0 Protected Resource Metadata and RFC 6750
//! `WWW-Authenticate` challenges (ADR-351 §5.3, amended: the authorization
//! server listed is the edge AS, not `auth.cognitum.one`).

use crate::resource::ResourceUrl;
use serde::Serialize;

/// RFC 9728 well-known path prefix.
pub const PRM_WELL_KNOWN: &str = "/.well-known/oauth-protected-resource";

/// Scopes advertised until M6.
pub const DEFAULT_SCOPES: [&str; 2] = ["mcp:read", "mcp:invoke"];

/// RFC 9728 §2 metadata document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProtectedResourceMetadata {
    /// The protected resource identifier (canonical https URL).
    pub resource: String,
    /// Issuer URLs of accepted authorization servers (the edge AS).
    pub authorization_servers: Vec<String>,
    /// Scopes a client may request for this resource.
    pub scopes_supported: Vec<String>,
    /// Always `["header"]`.
    pub bearer_methods_supported: Vec<String>,
    /// Human documentation URL, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_documentation: Option<String>,
}

impl ProtectedResourceMetadata {
    /// Build the document for `resource` served by `authorization_server`
    /// (the edge AS issuer URL) with [`DEFAULT_SCOPES`].
    pub fn new(resource: &ResourceUrl, authorization_server: &str) -> Self {
        ProtectedResourceMetadata {
            resource: resource.as_str().to_string(),
            authorization_servers: vec![authorization_server.to_string()],
            scopes_supported: DEFAULT_SCOPES.iter().map(|s| (*s).to_string()).collect(),
            bearer_methods_supported: vec!["header".to_string()],
            resource_documentation: None,
        }
    }
}

/// URL of the metadata document for `resource` (RFC 9728 §3.1: the
/// well-known segment is inserted between the origin and the path).
pub fn metadata_url(resource: &ResourceUrl) -> String {
    format!("{}{}{}", resource.origin(), PRM_WELL_KNOWN, resource.path())
}

/// `WWW-Authenticate` value for a 401.
pub fn www_authenticate_invalid_token(resource_metadata_url: &str) -> String {
    format!(r#"Bearer resource_metadata="{resource_metadata_url}", error="invalid_token""#)
}

/// `WWW-Authenticate` value for a 401 with no token at all (RFC 6750 §3.1:
/// no error code when the request lacked credentials).
pub fn www_authenticate_missing(resource_metadata_url: &str) -> String {
    format!(r#"Bearer resource_metadata="{resource_metadata_url}""#)
}

/// `WWW-Authenticate` value for a 403 missing capability.
pub fn www_authenticate_insufficient_scope(resource_metadata_url: &str, scope: &str) -> String {
    format!(
        r#"Bearer resource_metadata="{resource_metadata_url}", error="insufficient_scope", scope="{scope}""#
    )
}
