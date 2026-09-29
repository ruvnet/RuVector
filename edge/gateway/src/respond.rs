//! Response helpers: JSON, RFC 9457 problems, auth challenges, CORS.
//!
//! The API is bearer-only (no cookies), so `Access-Control-Allow-Origin: *`
//! is safe and lets browser-based MCP clients read every response —
//! including the `WWW-Authenticate` challenge of a 401/403, which must be
//! exposed explicitly (ADR-351 §5.7).

use ruvector_edge_tenancy::{problem::PROBLEM_CONTENT_TYPE, Problem, ProblemCode};
use serde::Serialize;
use worker::{Headers, Response, Result};

/// Headers on every actual (non-preflight) response.
pub const RESPONSE_CORS: [(&str, &str); 2] = [
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Expose-Headers", "WWW-Authenticate"),
];

/// Headers on a preflight answer.
pub const PREFLIGHT_CORS: [(&str, &str); 4] = [
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"),
    (
        "Access-Control-Allow-Headers",
        "Authorization, Content-Type, MCP-Protocol-Version, Mcp-Session-Id",
    ),
    ("Access-Control-Max-Age", "600"),
];

fn with_cors(headers: &Headers) -> Result<()> {
    for (k, v) in RESPONSE_CORS {
        headers.set(k, v)?;
    }
    Ok(())
}

/// JSON 200 with CORS, and `Cache-Control: no-store` when `credentialed`.
pub fn json<T: Serialize>(value: &T, credentialed: bool) -> Result<Response> {
    let mut resp = Response::from_json(value)?;
    let headers = resp.headers_mut();
    if credentialed {
        headers.set("Cache-Control", "no-store")?;
    }
    with_cors(headers)?;
    Ok(resp)
}

/// Problem response with CORS, optionally with a `WWW-Authenticate`
/// challenge (exposed to browser clients).
pub fn problem(code: ProblemCode, www_authenticate: Option<String>) -> Result<Response> {
    let p = Problem::new(code);
    let headers = Headers::new();
    headers.set("Content-Type", PROBLEM_CONTENT_TYPE)?;
    headers.set("Cache-Control", "no-store")?;
    with_cors(&headers)?;
    if let Some(challenge) = www_authenticate {
        headers.set("WWW-Authenticate", &challenge)?;
    }
    Ok(Response::ok(p.to_json())?
        .with_status(p.status)
        .with_headers(headers))
}

/// Public, cacheable JSON document (RFC 9728 metadata) with CORS, so
/// browser-based MCP clients can discover the authorization server.
pub fn public_json<T: Serialize>(value: &T) -> Result<Response> {
    let mut resp = Response::from_json(value)?;
    let headers = resp.headers_mut();
    headers.set("Cache-Control", "public, max-age=300")?;
    with_cors(headers)?;
    Ok(resp)
}

/// CORS preflight answer (never authenticated).
pub fn preflight() -> Result<Response> {
    let headers = Headers::new();
    for (k, v) in PREFLIGHT_CORS {
        headers.set(k, v)?;
    }
    Ok(Response::empty()?.with_status(204).with_headers(headers))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression (browser discovery blocked): every response exposes the
    /// challenge header cross-origin, and the preflight allows the bearer
    /// header and the MCP methods.
    #[test]
    fn cors_exposes_the_challenge_and_allows_mcp_preflights() {
        let get = |set: &[(&str, &str)], k: &str| {
            set.iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(k))
                .map(|(_, v)| v.to_string())
        };
        assert_eq!(
            get(&RESPONSE_CORS, "access-control-expose-headers").as_deref(),
            Some("WWW-Authenticate")
        );
        assert_eq!(
            get(&RESPONSE_CORS, "access-control-allow-origin").as_deref(),
            Some("*")
        );
        let methods = get(&PREFLIGHT_CORS, "access-control-allow-methods").unwrap();
        assert!(methods.contains("POST") && methods.contains("GET"));
        let headers = get(&PREFLIGHT_CORS, "access-control-allow-headers").unwrap();
        assert!(headers.contains("Authorization"));
    }
}
