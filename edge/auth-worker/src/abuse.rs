//! Abuse controls for the public, unauthenticated AS endpoints (ADR-351
//! §5.6): the per-IP bucket key, the durable DCR rate window, the client
//! idle-expiry horizons and the 429 reply. The Worker front additionally
//! consults optional Workers Rate Limiting bindings before the Durable
//! Object hop (see `lib.rs`).

use crate::http::Reply;
use ruvector_edge_authz::{OAuthError, OAuthErrorCode};

/// Fixed window of the durable per-IP registration counter.
pub const DCR_RATE_WINDOW_SECS: u64 = 3600;
/// Default registrations per IP bucket per window (`DCR_RATE_PER_HOUR`).
pub const DEFAULT_DCR_RATE_PER_HOUR: u64 = 10;
/// A client that never completed a token request is deleted after this.
pub const UNUSED_CLIENT_TTL_SECS: u64 = 24 * 3600;
/// A client idle this long (and holding no live refresh token) is deleted.
pub const CLIENT_IDLE_TTL_SECS: u64 = 30 * 24 * 3600;
/// Domain-separation tag of the IP bucket hash.
const IP_DOMAIN: &str = "ruvector-edge/ip/v1|";
/// Header carrying the client IP on Cloudflare.
pub const CLIENT_IP_HEADER: &str = "CF-Connecting-IP";
/// Optional secret salting the IP hash (never a var).
pub const IP_HASH_SALT_SECRET: &str = "IP_HASH_SALT";

/// Salted, non-reversible bucket key for a client IP. Raw IPs are never
/// stored. A request without the header (never the case behind Cloudflare)
/// shares one `unknown` bucket, so it cannot dodge the limit.
pub fn ip_bucket(ip: Option<&str>, salt: &str) -> String {
    let ip = ip.map(str::trim).filter(|v| !v.is_empty() && v.len() <= 64);
    let input = format!("{IP_DOMAIN}{salt}|{}", ip.unwrap_or("unknown"));
    crate::sql::hex32(&ruvector_edge_authz::secret_hash(&input))[..32].to_string()
}

/// `429 Too Many Requests` with `Retry-After` and an OAuth error body.
pub fn rate_limited(retry_after_secs: u64) -> Reply {
    let mut r = Reply::oauth_error(&OAuthError::new(
        OAuthErrorCode::TemporarilyUnavailable,
        "rate limit exceeded",
    ));
    r.status = 429;
    r.headers
        .push(("Retry-After", retry_after_secs.to_string()));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ip_bucket_is_salted_bounded_and_hides_the_ip() {
        let a = ip_bucket(Some("203.0.113.7"), "s1");
        assert_eq!(a.len(), 32);
        assert!(!a.contains("203"));
        assert_eq!(a, ip_bucket(Some(" 203.0.113.7 "), "s1"));
        assert_ne!(a, ip_bucket(Some("203.0.113.7"), "s2"));
        assert_ne!(a, ip_bucket(Some("203.0.113.8"), "s1"));
        assert_eq!(ip_bucket(None, "s1"), ip_bucket(Some(""), "s1"));
        let long = "x".repeat(65);
        assert_eq!(ip_bucket(Some(&long), "s1"), ip_bucket(None, "s1"));
    }

    #[test]
    fn rate_limited_is_429_with_retry_after() {
        let r = rate_limited(60);
        assert_eq!(r.status, 429);
        assert_eq!(r.header("Retry-After"), Some("60"));
    }
}
