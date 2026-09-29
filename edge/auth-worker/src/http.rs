//! Framework-neutral HTTP model for the AS endpoints.
//!
//! Endpoint logic returns a [`Reply`]; only `lib.rs`/`store.rs` convert it
//! into a `worker::Response`. This keeps every endpoint natively testable.

use ruvector_edge_authz::{OAuthError, OAuthErrorCode};
use serde::Serialize;

/// Maximum `application/x-www-form-urlencoded` body (token, revoke).
pub const MAX_FORM_BODY: usize = 8 * 1024;
/// Maximum registration (JSON) body.
pub const MAX_JSON_BODY: usize = 16 * 1024;
/// Maximum raw query string accepted on `/authorize` and `/callback`.
pub const MAX_QUERY_LEN: usize = 8 * 1024;

/// A response produced by endpoint logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    /// HTTP status.
    pub status: u16,
    /// Headers (name, value).
    pub headers: Vec<(&'static str, String)>,
    /// Body bytes.
    pub body: Vec<u8>,
}

impl Reply {
    fn with(status: u16, content_type: Option<&str>, body: Vec<u8>) -> Self {
        let mut headers = vec![("Cache-Control", "no-store".to_string())];
        if let Some(ct) = content_type {
            headers.push(("Content-Type", ct.to_string()));
        }
        Reply {
            status,
            headers,
            body,
        }
    }

    /// JSON body with `Cache-Control: no-store`.
    pub fn json<T: Serialize>(status: u16, value: &T) -> Self {
        match serde_json::to_vec(value) {
            Ok(body) => Reply::with(status, Some("application/json"), body),
            Err(_) => Reply::oauth_error(&OAuthError::new(
                OAuthErrorCode::ServerError,
                "serialization failure",
            )),
        }
    }

    /// RFC 6749 §5.2 JSON error; status from the error code.
    pub fn oauth_error(e: &OAuthError) -> Self {
        Reply::with(
            e.error.http_status(),
            Some("application/json"),
            e.to_json().into_bytes(),
        )
    }

    /// Error shown to the user agent when the redirect target is not
    /// trusted (never redirected). Always 400 unless the code says 5xx.
    pub fn error_page(e: &OAuthError) -> Self {
        let status = match e.error.http_status() {
            s @ 500..=599 => s,
            _ => 400,
        };
        Reply::with(status, Some("application/json"), e.to_json().into_bytes())
    }

    /// 302 to `location`. An empty or non-absolute location is a bug in a
    /// core helper and fails closed as `server_error` (never `Location: `).
    pub fn redirect(location: &str) -> Self {
        let absolute = url::Url::parse(location).is_ok();
        if location.is_empty() || !absolute {
            return Reply::error_page(&OAuthError::new(
                OAuthErrorCode::ServerError,
                "redirect target unavailable",
            ));
        }
        let mut r = Reply::with(302, None, Vec::new());
        r.headers.push(("Location", location.to_string()));
        r
    }

    /// Empty body with `status`.
    pub fn empty(status: u16) -> Self {
        Reply::with(status, None, Vec::new())
    }

    /// Header value by case-insensitive name.
    #[cfg(test)]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Decode a raw query string into pairs. Duplicate detection is the core's
/// job (`Params::from_pairs`); an oversize query is refused here.
pub fn parse_query(query: Option<&str>) -> Result<Vec<(String, String)>, OAuthError> {
    let q = query.unwrap_or("");
    if q.len() > MAX_QUERY_LEN {
        return Err(OAuthError::new(
            OAuthErrorCode::InvalidRequest,
            "query too long",
        ));
    }
    Ok(url::form_urlencoded::parse(q.as_bytes())
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect())
}

/// Media type of a `Content-Type` header, lowercased, parameters dropped.
fn media_type(content_type: Option<&str>) -> String {
    content_type
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

/// Whether the request declared a JSON body.
pub fn is_json(content_type: Option<&str>) -> bool {
    media_type(content_type) == "application/json"
}

/// Decode a form body. Wrong media type -> 415, oversize -> 413, both with
/// an `invalid_request` OAuth body (matching the upstream AS).
pub fn read_form(content_type: Option<&str>, body: &[u8]) -> Result<Vec<(String, String)>, Reply> {
    let bad = |status: u16, desc: &'static str| {
        let mut r = Reply::oauth_error(&OAuthError::new(OAuthErrorCode::InvalidRequest, desc));
        r.status = status;
        r
    };
    if media_type(content_type) != "application/x-www-form-urlencoded" {
        return Err(bad(415, "form-encoded body required"));
    }
    if body.len() > MAX_FORM_BODY {
        return Err(bad(413, "body too large"));
    }
    Ok(url::form_urlencoded::parse(body)
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect())
}

/// Body cap of a stateful POST endpoint: [`MAX_JSON_BODY`] for `/register`,
/// [`MAX_FORM_BODY`] for the form endpoints (`/token`, `/revoke`).
pub fn body_cap(path: &str) -> usize {
    if path == ruvector_edge_authz::metadata::paths::REGISTER {
        MAX_JSON_BODY
    } else {
        MAX_FORM_BODY
    }
}

/// Gate a POST body **before** it is read (Worker front and Durable
/// Object). `Content-Length` is required: without it (a chunked upload) the
/// runtime would buffer an unbounded body into the single global DO, so the
/// request is refused with 411. A declared length above the route cap
/// ([`body_cap`]) is 413. Returns the declared length.
pub fn check_body_length(path: &str, content_length: Option<&str>) -> Result<usize, Reply> {
    let refuse = |status: u16, desc: &'static str| {
        let mut r = Reply::oauth_error(&OAuthError::new(OAuthErrorCode::InvalidRequest, desc));
        r.status = status;
        r
    };
    let declared = content_length
        .map(str::trim)
        .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|v| v.parse::<usize>().ok());
    match declared {
        None => Err(refuse(411, "Content-Length required")),
        Some(n) if n > body_cap(path) => Err(refuse(413, "body too large")),
        Some(n) => Ok(n),
    }
}

/// Permissive CORS for the public, cookie-less AS documents and endpoints
/// (browser-based MCP clients). No credentials are ever honoured.
pub fn cors_headers() -> [(&'static str, &'static str); 3] {
    [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        (
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, MCP-Protocol-Version",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirect_refuses_empty_and_relative_locations() {
        assert_eq!(Reply::redirect("").status, 500);
        assert!(Reply::redirect("").header("Location").is_none());
        assert_eq!(Reply::redirect("/relative").status, 500);
        let ok = Reply::redirect("https://client.example/cb?code=x");
        assert_eq!(ok.status, 302);
        assert_eq!(
            ok.header("location"),
            Some("https://client.example/cb?code=x")
        );
        assert_eq!(ok.header("cache-control"), Some("no-store"));
    }

    #[test]
    fn error_page_is_400_for_client_errors_and_keeps_5xx() {
        let e = OAuthError::new(OAuthErrorCode::InvalidClient, "unknown client_id");
        assert_eq!(Reply::error_page(&e).status, 400);
        let e = OAuthError::new(OAuthErrorCode::TemporarilyUnavailable, "later");
        assert_eq!(Reply::error_page(&e).status, 503);
    }

    #[test]
    fn oauth_error_uses_code_status_and_json() {
        let r = Reply::oauth_error(&OAuthError::new(OAuthErrorCode::InvalidClient, "x"));
        assert_eq!(r.status, 401);
        assert_eq!(r.header("content-type"), Some("application/json"));
        let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["error"], "invalid_client");
    }

    #[test]
    fn parse_query_decodes_and_bounds() {
        let p = parse_query(Some("a=1&b=x%20y&a=2")).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p[1], ("b".into(), "x y".into()));
        assert!(parse_query(None).unwrap().is_empty());
        let long = "a=".to_string() + &"x".repeat(MAX_QUERY_LEN);
        assert!(parse_query(Some(&long)).is_err());
    }

    #[test]
    fn read_form_gates_media_type_and_size() {
        let ct = Some("application/x-www-form-urlencoded; charset=UTF-8");
        assert_eq!(
            read_form(ct, b"a=1").unwrap(),
            vec![("a".into(), "1".into())]
        );
        assert_eq!(
            read_form(Some("application/json"), b"{}")
                .unwrap_err()
                .status,
            415
        );
        assert_eq!(read_form(None, b"a=1").unwrap_err().status, 415);
        let big = vec![b'a'; MAX_FORM_BODY + 1];
        assert_eq!(read_form(ct, &big).unwrap_err().status, 413);
    }

    /// Regression (unbounded chunked bodies into the global DO): a POST with
    /// no or a malformed `Content-Length` is 411, and each route has its own
    /// cap (8 KiB form endpoints, 16 KiB registration).
    #[test]
    fn body_length_gate_requires_content_length_and_uses_route_caps() {
        use ruvector_edge_authz::metadata::paths;
        for bad in [None, Some(""), Some("abc"), Some("-1"), Some("1e3")] {
            let r = check_body_length(paths::TOKEN, bad).unwrap_err();
            assert_eq!(r.status, 411, "{bad:?}");
        }
        for p in [paths::TOKEN, paths::REVOKE] {
            assert_eq!(check_body_length(p, Some("8192")), Ok(MAX_FORM_BODY));
            assert_eq!(check_body_length(p, Some("8193")).unwrap_err().status, 413);
            assert_eq!(
                check_body_length(p, Some(&MAX_JSON_BODY.to_string()))
                    .unwrap_err()
                    .status,
                413
            );
        }
        let json_cap = MAX_JSON_BODY.to_string();
        assert_eq!(
            check_body_length(paths::REGISTER, Some(&json_cap)),
            Ok(MAX_JSON_BODY)
        );
        assert_eq!(
            check_body_length(paths::REGISTER, Some(" 16385 "))
                .unwrap_err()
                .status,
            413
        );
        assert_eq!(
            check_body_length(paths::REGISTER, Some("99999999999999999999999"))
                .unwrap_err()
                .status,
            411
        );
    }

    #[test]
    fn json_media_type_detection() {
        assert!(is_json(Some("Application/JSON; charset=utf-8")));
        assert!(!is_json(Some("text/plain")));
        assert!(!is_json(None));
    }
}
