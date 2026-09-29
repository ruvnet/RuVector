//! Canonical protected-resource URL, shared by the resource server (exact
//! `aud` match, RFC 9728 `resource`) and the authorization server (RFC 8707
//! `resource` allowlist, minted `aud`). One type on both sides so they can't
//! drift.

use crate::error::AuthError;
use core::fmt;

/// Maximum accepted length of a resource URL.
pub const MAX_RESOURCE_URL_LEN: usize = 512;

/// An absolute `https` URL in canonical form: lowercase host, optional
/// non-443 port, optional path without a trailing `/`, empty segments or
/// dot-segments, no query, no fragment, no userinfo. Equality is byte equality of the
/// canonical string, which is what exact `aud` matching compares.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(try_from = "String", into = "String")]
pub struct ResourceUrl(String);

impl TryFrom<String> for ResourceUrl {
    type Error = AuthError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        ResourceUrl::parse(&s)
    }
}

impl From<ResourceUrl> for String {
    fn from(r: ResourceUrl) -> String {
        r.0
    }
}

impl ResourceUrl {
    /// Parse and canonicalise. Rejects anything that is not already canonical
    /// apart from host case (a resource identifier should be written one way).
    pub fn parse(input: &str) -> Result<Self, AuthError> {
        if input.is_empty() || input.len() > MAX_RESOURCE_URL_LEN {
            return Err(AuthError::InvalidConfig("resource url length"));
        }
        let rest = input
            .strip_prefix("https://")
            .ok_or(AuthError::InvalidConfig("resource url must be https"))?;
        // Only RFC 3986 characters: no non-ASCII (it must be %-encoded), no
        // controls/space, no query/fragment/userinfo, and none of the
        // characters RFC 3986 never allows unencoded (`"`, `<`, `>`, `\`,
        // `^`, `` ` ``, `{`, `|`, `}`), so the value is safe inside a
        // `WWW-Authenticate` quoted-string and compares byte-for-byte.
        if rest.bytes().any(|b| {
            !b.is_ascii()
                || b.is_ascii_control()
                || matches!(
                    b,
                    b' ' | b'?'
                        | b'#'
                        | b'@'
                        | b'\\'
                        | b'"'
                        | b'<'
                        | b'>'
                        | b'^'
                        | b'`'
                        | b'{'
                        | b'|'
                        | b'}'
                )
        }) {
            return Err(AuthError::InvalidConfig(
                "resource url has forbidden characters",
            ));
        }
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (authority, None),
        };
        let host_ok = !host.is_empty()
            && host.len() <= 253
            && host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.')
            && !host.starts_with('.')
            && !host.ends_with('.');
        if !host_ok {
            return Err(AuthError::InvalidConfig("resource url host"));
        }
        if let Some(p) = port {
            if p.is_empty() || p.len() > 5 || !p.bytes().all(|b| b.is_ascii_digit()) || p == "443" {
                return Err(AuthError::InvalidConfig("resource url port"));
            }
        }
        if path.ends_with('/')
            || path.contains("//")
            || path.split('/').any(|s| s == "." || s == "..")
        {
            return Err(AuthError::InvalidConfig("resource url path not canonical"));
        }
        let mut canon = String::with_capacity(input.len());
        canon.push_str("https://");
        canon.push_str(&host.to_ascii_lowercase());
        if let Some(p) = port {
            canon.push(':');
            canon.push_str(p);
        }
        canon.push_str(path);
        Ok(ResourceUrl(canon))
    }

    /// Canonical string form.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// `https://host[:port]` without the path.
    pub fn origin(&self) -> &str {
        let after_scheme = "https://".len();
        match self.0[after_scheme..].find('/') {
            Some(i) => &self.0[..after_scheme + i],
            None => &self.0,
        }
    }

    /// Path component (`""` when the resource is the bare origin).
    pub fn path(&self) -> &str {
        &self.0[self.origin().len()..]
    }
}

impl fmt::Display for ResourceUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_urls_accepted() {
        let r = ResourceUrl::parse("https://GW.Example:8443/v1/mcp").unwrap();
        assert_eq!(r.as_str(), "https://gw.example:8443/v1/mcp");
        assert_eq!(r.origin(), "https://gw.example:8443");
        assert_eq!(r.path(), "/v1/mcp");
        assert!(ResourceUrl::parse("https://gw.example/v1/a%20b").is_ok());
    }

    /// Regression: characters that would break a `WWW-Authenticate`
    /// quoted-string or are not RFC 3986 are refused.
    #[test]
    fn non_rfc3986_bytes_rejected() {
        for bad in [
            "https://gw.example/v1\"x",
            "https://gw.example/v1/caf\u{e9}",
            "https://gw.example/v1/<x>",
            "https://gw.example/v1/a|b",
            "https://gw.example/v1/{c}",
            "https://gw.example/v1/a^b",
            "https://gw.example/v1/a`b",
            "https://gw.example/v1/a\\b",
            "https://gw.example/v1?q=1",
            "https://gw.example/v1#f",
            "https://u@gw.example/v1",
        ] {
            assert!(ResourceUrl::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn non_canonical_forms_rejected() {
        for bad in [
            "http://gw.example/v1",
            "https://gw.example/v1/",
            "https://gw.example//v1",
            "https://gw.example/v1/../x",
            "https://gw.example:443/v1",
            "https://gw.example:/v1",
            "https://.gw.example/v1",
            "",
        ] {
            assert!(ResourceUrl::parse(bad).is_err(), "{bad}");
        }
        let long = format!("https://gw.example/{}", "a".repeat(MAX_RESOURCE_URL_LEN));
        assert!(ResourceUrl::parse(&long).is_err());
    }
}
