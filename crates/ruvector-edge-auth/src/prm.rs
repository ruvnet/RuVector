//! RFC 9728 OAuth 2.0 Protected Resource Metadata and RFC 6750
//! `WWW-Authenticate` challenges (ADR-351 §5.7: the authorization server
//! listed is the edge AS, not `auth.cognitum.one`).

use crate::resource::ResourceUrl;
use crate::scopes::Capability;
use serde::Serialize;

/// RFC 9728 well-known path prefix.
pub const PRM_WELL_KNOWN: &str = "/.well-known/oauth-protected-resource";

/// `scopes_supported` for the `/v1` document (§5.7).
pub const REST_SCOPES: [&str; 4] = [
    "ruvector:read",
    "ruvector:write",
    "ruvector:admin",
    "offline_access",
];

/// `scopes_supported` for the `/v1/mcp` document (§5.7: no admin).
pub const MCP_SCOPES: [&str; 3] = ["ruvector:read", "ruvector:write", "offline_access"];

/// Scope named in every 401 challenge (§5.1/§5.7).
pub const CHALLENGE_SCOPE: &str = "ruvector:read offline_access";

/// Scope named in the 403 `insufficient_scope` step-up challenge of a
/// mutating call (§5.3); equals [`step_up_scope`] for
/// [`Capability::Write`] / [`Capability::CreateCollection`].
pub const STEP_UP_SCOPE: &str = "ruvector:read ruvector:write offline_access";

/// `error_description` of the 401 challenge for a token whose `aud` is not
/// this resource (§5.4.7), so clients can tell it from a bad signature.
pub const AUDIENCE_MISMATCH: &str = "audience mismatch";

/// Scope for the 403 `insufficient_scope` challenge when `missing` is the
/// capability the route needs (§5.3): `ruvector:read <scope satisfying
/// missing> offline_access`, e.g. `ruvector:read ruvector:admin
/// offline_access` for an admin route. Never names a scope twice.
pub fn step_up_scope(missing: Capability) -> String {
    let needed = missing.satisfying_scope();
    if needed == "ruvector:read" {
        "ruvector:read offline_access".to_string()
    } else {
        format!("ruvector:read {needed} offline_access")
    }
}

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
    /// (the edge AS issuer URL) with this document's own `scopes`
    /// ([`REST_SCOPES`] or [`MCP_SCOPES`], §5.7).
    pub fn new(resource: &ResourceUrl, authorization_server: &str, scopes: &[&str]) -> Self {
        ProtectedResourceMetadata {
            resource: resource.as_str().to_string(),
            authorization_servers: vec![authorization_server.to_string()],
            scopes_supported: scopes.iter().map(|s| (*s).to_string()).collect(),
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

/// RFC 7230 quoted-string: backslash-escape `"` and `\`; drop control
/// characters, which a quoted-string cannot carry.
fn quoted(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars().filter(|c| !c.is_control()) {
        if c == '"' || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('"');
    out
}

/// `WWW-Authenticate` value for a 401 when a token was presented (§5.7:
/// `resource_metadata`, `scope` — normally [`CHALLENGE_SCOPE`] — and
/// `error="invalid_token"`).
pub fn www_authenticate_invalid_token(resource_metadata_url: &str, scope: &str) -> String {
    www_authenticate_invalid_token_described(resource_metadata_url, scope, None)
}

/// [`www_authenticate_invalid_token`] with an optional RFC 6750 §3
/// `error_description` (e.g. [`AUDIENCE_MISMATCH`], §5.4.7). `None` yields
/// exactly the undescribed challenge.
pub fn www_authenticate_invalid_token_described(
    resource_metadata_url: &str,
    scope: &str,
    description: Option<&str>,
) -> String {
    let mut out = format!(
        r#"Bearer resource_metadata={}, scope={}, error="invalid_token""#,
        quoted(resource_metadata_url),
        quoted(scope)
    );
    if let Some(d) = description {
        out.push_str(", error_description=");
        out.push_str(&quoted(d));
    }
    out
}

/// `WWW-Authenticate` value for a 401 with no token at all (RFC 6750 §3.1:
/// no error code when the request lacked credentials).
pub fn www_authenticate_missing(resource_metadata_url: &str, scope: &str) -> String {
    format!(
        "Bearer resource_metadata={}, scope={}",
        quoted(resource_metadata_url),
        quoted(scope)
    )
}

/// `WWW-Authenticate` value for a 403 missing capability.
pub fn www_authenticate_insufficient_scope(resource_metadata_url: &str, scope: &str) -> String {
    format!(
        r#"Bearer resource_metadata={}, error="insufficient_scope", scope={}"#,
        quoted(resource_metadata_url),
        quoted(scope)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(s: &str) -> ResourceUrl {
        ResourceUrl::parse(s).unwrap()
    }

    #[test]
    fn per_document_scopes() {
        let rest = ProtectedResourceMetadata::new(
            &res("https://gw.example/v1"),
            "https://as",
            &REST_SCOPES,
        );
        assert!(rest
            .scopes_supported
            .contains(&"ruvector:admin".to_string()));
        let mcp = ProtectedResourceMetadata::new(
            &res("https://gw.example/v1/mcp"),
            "https://as",
            &MCP_SCOPES,
        );
        assert_eq!(
            mcp.scopes_supported,
            ["ruvector:read", "ruvector:write", "offline_access"]
        );
        assert!(!REST_SCOPES
            .iter()
            .chain(MCP_SCOPES.iter())
            .any(|s| s.starts_with("mcp:")));
    }

    #[test]
    fn challenges_carry_scope() {
        let url = metadata_url(&res("https://gw.example/v1/mcp"));
        assert_eq!(
            url,
            "https://gw.example/.well-known/oauth-protected-resource/v1/mcp"
        );
        assert_eq!(
            www_authenticate_missing(&url, CHALLENGE_SCOPE),
            format!(r#"Bearer resource_metadata="{url}", scope="ruvector:read offline_access""#)
        );
        assert_eq!(
            www_authenticate_invalid_token(&url, CHALLENGE_SCOPE),
            format!(
                r#"Bearer resource_metadata="{url}", scope="ruvector:read offline_access", error="invalid_token""#
            )
        );
        assert_eq!(
            www_authenticate_insufficient_scope(&url, STEP_UP_SCOPE),
            format!(
                r#"Bearer resource_metadata="{url}", error="insufficient_scope", scope="ruvector:read ruvector:write offline_access""#
            )
        );
    }

    /// Regression (§5.4.7): the audience-mismatch 401 is distinguishable.
    #[test]
    fn invalid_token_can_carry_a_description() {
        let url = metadata_url(&res("https://gw.example/v1"));
        assert_eq!(
            www_authenticate_invalid_token_described(&url, CHALLENGE_SCOPE, None),
            www_authenticate_invalid_token(&url, CHALLENGE_SCOPE)
        );
        assert_eq!(
            www_authenticate_invalid_token_described(
                &url,
                CHALLENGE_SCOPE,
                Some(AUDIENCE_MISMATCH)
            ),
            format!(
                r#"Bearer resource_metadata="{url}", scope="ruvector:read offline_access", error="invalid_token", error_description="audience mismatch""#
            )
        );
    }

    /// Regression (§5.3): the step-up challenge names the scope the missing
    /// capability needs, not a fixed write scope.
    #[test]
    fn step_up_scope_follows_the_missing_capability() {
        assert_eq!(step_up_scope(Capability::Write), STEP_UP_SCOPE);
        assert_eq!(step_up_scope(Capability::CreateCollection), STEP_UP_SCOPE);
        assert_eq!(
            step_up_scope(Capability::Admin),
            "ruvector:read ruvector:admin offline_access"
        );
        assert_eq!(
            step_up_scope(Capability::PublishPublic),
            "ruvector:read ruvector:publish offline_access"
        );
        assert_eq!(
            step_up_scope(Capability::Read),
            "ruvector:read offline_access"
        );
    }

    #[test]
    fn quoted_strings_are_escaped() {
        let v = www_authenticate_missing("https://x/a\"b\\c", "s\"\r\n, error=\"x");
        assert_eq!(
            v,
            r#"Bearer resource_metadata="https://x/a\"b\\c", scope="s\", error=\"x""#
        );
    }
}
