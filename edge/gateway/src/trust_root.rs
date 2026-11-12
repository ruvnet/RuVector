//! Compile-time trust root of the gateway (ADR-351 §5.4 item 5, §8 delta 6).
//!
//! In release builds the public origin (which fixes both audiences), the edge
//! issuer and its JWKS URL, the upstream issuer and its JWKS URL, the
//! first-party audiences and the pinned upstream `kid`s are Rust `const`s: a
//! wrangler `[vars]` edit cannot widen what the gateway trusts. Only the
//! `UPSTREAM_FIRST_PARTY` flag stays a var. Vars may supply the trust root
//! **only** under `#[cfg(feature = "dev-issuer")]`; a release build that
//! sees one of these vars with a different value answers 503
//! `trust_root_mismatch` instead of silently ignoring the drift.

/// Gateway origin; the resources are `<origin>/v1` and `<origin>/v1/mcp`.
pub const PUBLIC_ORIGIN: &str =
    "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev";
/// The edge authorization server (`ruvector-edge-auth`).
pub const EDGE_ISSUER: &str = "https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev";
/// Edge AS JWKS, fetched through the `EDGE_AUTH` Service Binding only.
pub const EDGE_JWKS_URL: &str =
    "https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev/.well-known/jwks.json";
/// Upstream IdP (the §5.5 first-party path and the tenant namespace).
pub const UPSTREAM_ISSUER: &str = "https://auth.cognitum.one";
/// Upstream JWKS (public internet; used only by the §5.5 path).
pub const UPSTREAM_JWKS_URL: &str = "https://auth.cognitum.one/.well-known/jwks.json";
/// Exact first-party client ids accepted as upstream `aud` (§5.5). Empty
/// until a first-party CLI client is registered upstream, so the §5.5 path
/// cannot be enabled by the flag alone (config fails closed).
pub const FIRST_PARTY_AUDS: &[&str] = &[];
/// Pinned upstream signing `kid`s (§5.5). Empty until gate G1
/// (cognitum-one/console#605) records them.
pub const ACCEPTED_UPSTREAM_KIDS: &[&str] = &[];

/// The trust root a [`crate::config::GatewayConfig`] is built from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustRoot {
    /// See [`PUBLIC_ORIGIN`].
    pub public_origin: String,
    /// See [`EDGE_ISSUER`].
    pub edge_issuer: String,
    /// See [`EDGE_JWKS_URL`].
    pub edge_jwks_url: String,
    /// See [`UPSTREAM_ISSUER`].
    pub upstream_issuer: String,
    /// See [`UPSTREAM_JWKS_URL`].
    pub upstream_jwks_url: String,
    /// See [`FIRST_PARTY_AUDS`].
    pub first_party_auds: Vec<String>,
    /// See [`ACCEPTED_UPSTREAM_KIDS`].
    pub accepted_kids: Vec<String>,
}

/// Split a comma/space separated var into non-empty entries.
pub fn list(v: &str) -> Vec<String> {
    v.split([',', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

fn owned(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

impl TrustRoot {
    /// The compiled release trust root.
    #[cfg_attr(feature = "dev-issuer", allow(dead_code))]
    pub fn compiled() -> Self {
        TrustRoot {
            public_origin: PUBLIC_ORIGIN.into(),
            edge_issuer: EDGE_ISSUER.into(),
            edge_jwks_url: EDGE_JWKS_URL.into(),
            upstream_issuer: UPSTREAM_ISSUER.into(),
            upstream_jwks_url: UPSTREAM_JWKS_URL.into(),
            first_party_auds: owned(FIRST_PARTY_AUDS),
            accepted_kids: owned(ACCEPTED_UPSTREAM_KIDS),
        }
    }

    /// Trust root from vars (dev/integration builds and tests only). A
    /// missing var is `None`.
    #[cfg(any(test, feature = "dev-issuer"))]
    pub fn from_vars(get: &dyn Fn(&str) -> Option<String>) -> Option<Self> {
        Some(TrustRoot {
            public_origin: get("PUBLIC_ORIGIN")?,
            edge_issuer: get("EDGE_ISSUER")?,
            edge_jwks_url: get("EDGE_JWKS_URL")?,
            upstream_issuer: get("UPSTREAM_ISSUER")?,
            upstream_jwks_url: get("UPSTREAM_JWKS_URL")?,
            first_party_auds: list(&get("FIRST_PARTY_AUDS")?),
            accepted_kids: list(&get("ACCEPTED_UPSTREAM_KIDS")?),
        })
    }

    /// Release check: every trust-root var that is present must agree with
    /// this (compiled) root — strings byte-equal after trimming, lists as
    /// parsed lists. Returns the name of the first contradicting var.
    #[cfg_attr(feature = "dev-issuer", allow(dead_code))]
    pub fn contradiction(&self, get: &dyn Fn(&str) -> Option<String>) -> Option<&'static str> {
        let scalars: [(&'static str, &str); 5] = [
            ("PUBLIC_ORIGIN", &self.public_origin),
            ("EDGE_ISSUER", &self.edge_issuer),
            ("EDGE_JWKS_URL", &self.edge_jwks_url),
            ("UPSTREAM_ISSUER", &self.upstream_issuer),
            ("UPSTREAM_JWKS_URL", &self.upstream_jwks_url),
        ];
        let lists: [(&'static str, &[String]); 2] = [
            ("FIRST_PARTY_AUDS", &self.first_party_auds),
            ("ACCEPTED_UPSTREAM_KIDS", &self.accepted_kids),
        ];
        scalars
            .into_iter()
            .find(|(name, want)| get(name).is_some_and(|v| v.trim() != *want))
            .map(|(name, _)| name)
            .or_else(|| {
                lists
                    .into_iter()
                    .find(|(name, want)| get(name).is_some_and(|v| list(&v) != *want))
                    .map(|(name, _)| name)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn compiled_root_is_self_consistent() {
        let r = TrustRoot::compiled();
        assert_eq!(
            r.edge_jwks_url,
            format!("{}{}", r.edge_issuer, crate::config::EDGE_JWKS_PATH)
        );
        assert_eq!(
            r.upstream_jwks_url,
            format!("{}{}", r.upstream_issuer, crate::config::EDGE_JWKS_PATH)
        );
        assert_eq!(r.upstream_issuer, ruvector_edge_tenancy::UPSTREAM_ISSUER);
        assert!(r.first_party_auds.is_empty() && r.accepted_kids.is_empty());
    }

    #[test]
    fn contradiction_names_the_drifting_var() {
        let r = TrustRoot::compiled();
        let with = |pairs: &[(&'static str, &str)]| {
            let m: HashMap<&str, String> =
                pairs.iter().map(|(k, v)| (*k, (*v).to_string())).collect();
            r.contradiction(&|k| m.get(k).cloned())
        };
        assert_eq!(with(&[]), None);
        assert_eq!(
            with(&[("EDGE_ISSUER", EDGE_ISSUER), ("FIRST_PARTY_AUDS", "")]),
            None
        );
        assert_eq!(
            with(&[("EDGE_ISSUER", "https://attacker.example")]),
            Some("EDGE_ISSUER")
        );
        assert_eq!(
            with(&[("FIRST_PARTY_AUDS", "cli")]),
            Some("FIRST_PARTY_AUDS")
        );
        assert_eq!(
            with(&[("ACCEPTED_UPSTREAM_KIDS", "kid-1")]),
            Some("ACCEPTED_UPSTREAM_KIDS")
        );
    }
}
