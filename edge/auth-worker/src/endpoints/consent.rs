//! Consent page and the browser-binding cookie.
//!
//! `/authorize` never auto-redirects upstream: it renders a consent page
//! naming the client, its redirect host, the scopes and the resource, and
//! sets the flow's browser-binding secret as a `__Host-` cookie. Only a
//! user click continues to `auth.cognitum.one`, so a client registered by
//! anyone cannot silently ride an existing upstream session (the core's
//! confused-deputy note in `federation`).

use crate::http::Reply;
use ruvector_edge_authz::authorize::ValidatedAuthorization;
use ruvector_edge_authz::client::ClientRecord;

/// Cookie-name prefix; the name is suffixed with a prefix of the upstream
/// `state` so parallel logins in one browser do not clobber each other.
pub const COOKIE_PREFIX: &str = "__Host-eaf-";
/// Characters of `state` used in the cookie name.
pub const COOKIE_STATE_CHARS: usize = 16;
/// Cookie lifetime (seconds) — matches the flow TTL.
pub const COOKIE_MAX_AGE: u64 = ruvector_edge_authz::federation::FLOW_TTL_SECS;

/// Cookie name for an upstream `state` (base64url only; else `None`).
pub fn cookie_name(state: &str) -> Option<String> {
    let ok = state.len() >= COOKIE_STATE_CHARS
        && state
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    ok.then(|| format!("{COOKIE_PREFIX}{}", &state[..COOKIE_STATE_CHARS]))
}

/// `Set-Cookie` value carrying the browser secret.
pub fn set_cookie(name: &str, secret: &str) -> String {
    format!("{name}={secret}; Path=/; Max-Age={COOKIE_MAX_AGE}; Secure; HttpOnly; SameSite=Lax")
}

/// `Set-Cookie` value deleting the cookie.
pub fn clear_cookie(name: &str) -> String {
    format!("{name}=; Path=/; Max-Age=0; Secure; HttpOnly; SameSite=Lax")
}

/// Value of cookie `name` in a `Cookie` header.
pub fn read_cookie<'a>(header: Option<&'a str>, name: &str) -> Option<&'a str> {
    header?
        .split(';')
        .filter_map(|kv| kv.trim().split_once('='))
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v.trim())
}

/// The upstream `state` inside an authorization URL we built.
pub fn state_of(authorization_url: &str) -> Option<String> {
    url::Url::parse(authorization_url)
        .ok()?
        .query_pairs()
        .find(|(k, _)| k == "state")
        .map(|(_, v)| v.into_owned())
}

/// Minimal HTML escaping for text and double-quoted attributes.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

/// Path of the consent form POST (continues to the upstream login).
pub const CONSENT_PATH: &str = "/authorize/consent";

/// The consent form: `flow` is the upstream `state`, `token` the
/// [`consent_token`](ruvector_edge_authz::authorize::consent_token) bound to
/// the browser cookie, and `upstream_origin` the only other origin the form
/// may lead to (CSP `form-action` also governs the post-submit redirect).
pub struct ConsentForm<'a> {
    /// Upstream `state` of this flow.
    pub flow: &'a str,
    /// Consent form token.
    pub token: &'a str,
    /// Origin of the upstream authorization endpoint.
    pub upstream_origin: &'a str,
    /// This AS's issuer URL (its host is shown in the page footer).
    pub issuer: &'a str,
}

/// Render the consent page ([`super::consent_view::render`]). Continue is a
/// same-origin POST carrying the consent token (a cross-site page cannot
/// compute it without the `__Host-` cookie); `cancel_url` returns
/// `access_denied` to the redirect URI. An unverified redirect host and a
/// non-ASCII client name get explicit warnings (ADR-351 §5.6).
pub fn page(
    client: &ClientRecord,
    auth: &ValidatedAuthorization,
    form: &ConsentForm<'_>,
    cancel_url: &str,
    set_cookie_value: String,
) -> Reply {
    let html = super::consent_view::render(
        client,
        auth,
        &super::consent_view::ViewParams {
            flow: form.flow,
            token: form.token,
            cancel_url,
            issuer: form.issuer,
        },
    );
    let mut r = Reply::empty(200);
    r.body = html.into_bytes();
    r.headers.extend([
        ("Content-Type", "text/html; charset=utf-8".to_string()),
        ("Set-Cookie", set_cookie_value),
        ("X-Frame-Options", "DENY".to_string()),
        (
            "Content-Security-Policy",
            format!(
                "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; \
                 base-uri 'none'; form-action 'self' {}",
                form.upstream_origin
            ),
        ),
        ("Referrer-Policy", "no-referrer".to_string()),
    ]);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cookie_name_requires_base64url_state() {
        let n = cookie_name("abcdefghijklmnopqrstuv").unwrap();
        assert_eq!(n, "__Host-eaf-abcdefghijklmnop");
        assert!(cookie_name("short").is_none());
        assert!(cookie_name("abcdefghijklmnop;=x").is_none());
    }

    #[test]
    fn cookies_are_host_prefixed_secure_httponly() {
        let c = set_cookie("__Host-eaf-x", "s3cret");
        for part in [
            "Path=/",
            "Secure",
            "HttpOnly",
            "SameSite=Lax",
            "Max-Age=600",
        ] {
            assert!(c.contains(part), "{c}");
        }
        assert!(!c.contains("Domain"));
        assert!(clear_cookie("n").contains("Max-Age=0"));
    }

    #[test]
    fn read_cookie_finds_exact_name() {
        let h = Some("a=1; __Host-eaf-x=v2 ;__Host-eaf-xy=v3");
        assert_eq!(read_cookie(h, "__Host-eaf-x"), Some("v2"));
        assert_eq!(read_cookie(h, "__Host-eaf-xy"), Some("v3"));
        assert_eq!(read_cookie(h, "__Host-eaf"), None);
        assert_eq!(read_cookie(None, "a"), None);
    }

    #[test]
    fn escape_neutralises_markup() {
        assert_eq!(
            escape("<a href=\"x\" onclick='y'>&</a>\u{7}"),
            "&lt;a href=&quot;x&quot; onclick=&#x27;y&#x27;&gt;&amp;&lt;/a&gt;"
        );
    }

    #[test]
    fn state_of_reads_query() {
        assert_eq!(
            state_of("https://up.example/a?x=1&state=abc%2Bd").as_deref(),
            Some("abc+d")
        );
        assert!(state_of("not a url").is_none());
    }
}
