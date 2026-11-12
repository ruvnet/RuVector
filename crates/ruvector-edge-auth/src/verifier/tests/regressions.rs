//! Regression tests for the 2026-09-29 review findings.

use super::*;
use crate::jwks::{Jwk, KeySource};
use crate::VerifyingKey;
use p256::ecdsa::SigningKey;
use std::collections::BTreeSet;

/// Worker-style wrapper denying compromised `kid`s before the signature step.
struct DenyingKeys<'a> {
    inner: &'a StaticKeys,
    denied: BTreeSet<String>,
}

impl KeySource for DenyingKeys<'_> {
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError> {
        if self.denied.contains(kid) {
            return Err(AuthError::Denied);
        }
        self.inner.verifying_key(kid).await
    }
}

/// Finding (high): the verified claims carry the header `kid`, so the §5.6
/// emergency procedure (deny the old `kid`) is implementable.
#[test]
fn verified_claims_expose_header_kid_for_deny_list() {
    let (k1, k2) = (key(1), key(2));
    let (keys, clock) = (StaticKeys::of(&[&k1, &k2]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let claims = run(&v, &edge_token(&k1)).unwrap();
    assert_eq!(claims.kid(), kid(&k1));
    assert_eq!(
        claims.kid(),
        Jwk::from_verifying_key(k1.verifying_key()).kid
    );

    // Post-verify deny-list on kid: a token signed by the compromised key is
    // identified, one signed by the new key is not.
    let denied: BTreeSet<String> = [kid(&k1)].into();
    assert!(denied.contains(claims.kid()));
    let fresh = run(&v, &edge_token(&k2)).unwrap();
    assert!(!denied.contains(fresh.kid()));

    // Pre-signature denial through a wrapping key source.
    let wrapped = DenyingKeys {
        inner: &keys,
        denied,
    };
    let v: Verifier<_, &StaticKeys, _> =
        Verifier::edge_only(&wrapped, &clock, audience(), edge_policy());
    assert_eq!(
        block_on(v.verify(Some(&bearer(&edge_token(&k1))))),
        Err(AuthError::Denied)
    );
    assert!(block_on(v.verify(Some(&bearer(&edge_token(&k2))))).is_ok());
}

fn upstream_token(k: &SigningKey) -> String {
    let h = json!({"alg": "ES256", "kid": kid(k)});
    sign(&h, &upstream_claims(), k)
}

/// Finding (medium): an upstream `kid` outside the compiled pin is refused
/// before any key lookup, even when the key source would trust it.
#[test]
fn upstream_kid_outside_pin_rejected_before_lookup() {
    let (edge_k, pinned, rogue) = (key(1), key(5), key(6));
    let edge_keys = StaticKeys::of(&[&edge_k]);
    let up_keys = StaticKeys::of(&[&pinned, &rogue]);
    let clock = TestClock::at(NOW);
    let v = Verifier::edge_only(
        &edge_keys,
        &clock,
        upstream_audience(vec![kid(&pinned)]),
        edge_policy(),
    )
    .with_upstream(&up_keys, ClaimsPolicy::upstream_access(UPSTREAM_ISS));
    assert_eq!(
        block_on(v.verify(Some(&bearer(&upstream_token(&rogue))))),
        Err(AuthError::UnknownKid)
    );
    assert_eq!(up_keys.calls.get(), 0, "pin is checked before lookup");
    assert!(block_on(v.verify(Some(&bearer(&upstream_token(&pinned))))).is_ok());
    assert_eq!(up_keys.calls.get(), 1);
}

/// Finding (medium): an empty upstream pin fails closed (500), never "trust
/// every key the upstream JWKS publishes".
#[test]
fn upstream_empty_pin_fails_closed() {
    let (edge_k, up_k) = (key(1), key(5));
    let (edge_keys, up_keys) = (StaticKeys::of(&[&edge_k]), StaticKeys::of(&[&up_k]));
    let clock = TestClock::at(NOW);
    let v = Verifier::edge_only(&edge_keys, &clock, upstream_audience(vec![]), edge_policy())
        .with_upstream(&up_keys, ClaimsPolicy::upstream_access(UPSTREAM_ISS));
    let err = block_on(v.verify(Some(&bearer(&upstream_token(&up_k))))).unwrap_err();
    assert!(matches!(err, AuthError::InvalidConfig(_)), "{err:?}");
    assert_eq!(err.http_status(), 500);
    assert_eq!(up_keys.calls.get(), 0);
}

/// Finding (medium): the pin holds even behind an unpinned `JwksCache`
/// (the gateway's upstream cache is built with `with_defaults`).
#[test]
fn upstream_pin_enforced_behind_unpinned_jwks_cache() {
    let (edge_k, pinned, rogue) = (key(1), key(5), key(6));
    let edge_keys = StaticKeys::of(&[&edge_k]);
    let fetch = CountingFetch::serving(jwks_body(&[&pinned, &rogue]));
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, JwksCachePolicy::with_defaults(JWKS_URL));
    let v = Verifier::edge_only(
        &edge_keys,
        &clock,
        upstream_audience(vec![kid(&pinned)]),
        edge_policy(),
    )
    .with_upstream(&cache, ClaimsPolicy::upstream_access(UPSTREAM_ISS));
    assert_eq!(
        block_on(v.verify(Some(&bearer(&upstream_token(&rogue))))),
        Err(AuthError::UnknownKid)
    );
    assert_eq!(fetch.calls.get(), 0);
}

/// Finding (medium): edge tokens are capped at `exp - iat <= 900` whatever
/// the policy says (legacy `with_defaults` allows 3600).
#[test]
fn edge_lifetime_over_900_rejected_even_with_legacy_policy() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    for policy in [edge_policy(), ClaimsPolicy::with_defaults(EDGE_ISS)] {
        let v: EdgeVerifier<'_> = Verifier::edge_only(&keys, &clock, audience(), policy);
        let mut c = edge_claims();
        c["iat"] = json!(NOW - 10);
        c["exp"] = json!(NOW - 10 + 901);
        let t = sign(&edge_header(&k), &c, &k);
        assert_eq!(run(&v, &t), Err(AuthError::LifetimeTooLong));
        c["exp"] = json!(NOW - 10 + 900);
        assert!(run(&v, &sign(&edge_header(&k), &c, &k)).is_ok());
    }
}

/// Finding (medium): upstream tokens need claim `typ == "access"` and
/// `exp - iat <= 3600` even under a policy that forgot both.
#[test]
fn upstream_typ_and_lifetime_enforced_even_with_legacy_policy() {
    let (edge_k, up_k) = (key(1), key(5));
    let (edge_keys, up_keys) = (StaticKeys::of(&[&edge_k]), StaticKeys::of(&[&up_k]));
    let clock = TestClock::at(NOW);
    let loose = ClaimsPolicy {
        max_lifetime_secs: 100_000,
        ..ClaimsPolicy::with_defaults(UPSTREAM_ISS)
    };
    assert_eq!(loose.typ_claim, None);
    let v = Verifier::edge_only(
        &edge_keys,
        &clock,
        upstream_audience(vec![kid(&up_k)]),
        edge_policy(),
    )
    .with_upstream(&up_keys, loose);
    let h = json!({"alg": "ES256", "kid": kid(&up_k)});
    let mut id_token = upstream_claims();
    id_token.as_object_mut().unwrap().remove("typ");
    assert_eq!(
        block_on(v.verify(Some(&bearer(&sign(&h, &id_token, &up_k))))),
        Err(AuthError::InvalidClaim("typ"))
    );
    let mut long = upstream_claims();
    long["exp"] = json!(NOW - 10 + 3601);
    assert_eq!(
        block_on(v.verify(Some(&bearer(&sign(&h, &long, &up_k))))),
        Err(AuthError::LifetimeTooLong)
    );
    assert!(block_on(v.verify(Some(&bearer(&upstream_token(&up_k))))).is_ok());
}

/// Finding (low): edge tokens must carry `upstream_iss` equal to the
/// configured upstream issuer.
#[test]
fn edge_upstream_iss_required_and_exact() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let ok = run(&v, &edge_token(&k)).unwrap();
    assert_eq!(ok.upstream_iss(), Some(UPSTREAM_ISS));
    for bad in [
        json!(null),
        json!("https://evil.example"),
        json!(format!("{UPSTREAM_ISS}/")),
    ] {
        let mut c = edge_claims();
        if bad.is_null() {
            c.as_object_mut().unwrap().remove("upstream_iss");
        } else {
            c["upstream_iss"] = bad.clone();
        }
        let t = sign(&edge_header(&k), &c, &k);
        assert_eq!(
            run(&v, &t),
            Err(AuthError::InvalidClaim("upstream_iss")),
            "{bad}"
        );
    }
}

/// Finding (low): a valid resource URL longer than 256 bytes (up to 512) is
/// accepted as `aud`.
#[test]
fn long_resource_audience_accepted() {
    let k = key(1);
    let long = format!(
        "https://ruvector-edge-gateway.example.workers.dev/v1/{}",
        "a".repeat(400)
    );
    assert!(long.len() > crate::claims::MAX_CLAIM_LEN);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let aud = AudiencePolicy::edge_only(EDGE_ISS, ResourceUrl::parse(&long).unwrap());
    let v: EdgeVerifier<'_> = Verifier::edge_only(&keys, &clock, aud, edge_policy());
    let t = sign(&edge_header(&k), &claims_with("aud", json!(long)), &k);
    assert_eq!(run(&v, &t).unwrap().aud(), long);
}

/// Finding (low): a malformed `Authorization` header is 400
/// `invalid_request`, never a 401 carrying `invalid_request`.
#[test]
fn malformed_authorization_is_400() {
    let (keys, clock) = (StaticKeys::of(&[&key(1)]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    for h in ["Bearer ", "Bearer  abc", "Bearer a b"] {
        let err = block_on(v.verify(Some(h))).unwrap_err();
        assert_eq!(err, AuthError::MalformedAuthorization, "{h:?}");
        assert_eq!(err.http_status(), 400);
        assert_eq!(err.rfc6750_error(), Some("invalid_request"));
    }
}
