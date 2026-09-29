use super::*;
use crate::jwks::{JwksCache, JwksCachePolicy};
use crate::jws::{b64url_encode, MAX_TOKEN_BYTES};
use crate::resource::ResourceUrl;
use crate::test_support::*;
use crate::UpstreamFirstPartyPolicy;
use serde_json::{json, Value};

fn audience() -> AudiencePolicy {
    AudiencePolicy::edge_only(EDGE_ISS, ResourceUrl::parse(RESOURCE).unwrap())
        .with_siblings([ResourceUrl::parse(SIBLING).unwrap()])
}

fn edge_policy() -> ClaimsPolicy {
    ClaimsPolicy::edge(EDGE_ISS, UPSTREAM_ISS)
}

type EdgeVerifier<'a> = Verifier<&'a StaticKeys, &'a StaticKeys, &'a TestClock>;

fn edge_verifier<'a>(keys: &'a StaticKeys, clock: &'a TestClock) -> EdgeVerifier<'a> {
    Verifier::edge_only(keys, clock, audience(), edge_policy())
}

fn run(v: &EdgeVerifier<'_>, token: &str) -> Result<VerifiedClaims, AuthError> {
    block_on(v.verify(Some(&bearer(token))))
}

fn claims_with(k: &str, v: Value) -> Value {
    let mut c = edge_claims();
    c[k] = v;
    c
}

#[test]
fn valid_edge_token() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let claims = run(&v, &edge_token(&k)).unwrap();
    assert_eq!(claims.kind(), TokenKind::EdgeIssued);
    assert_eq!(claims.aud(), RESOURCE);
    assert_eq!(claims.org_id(), Some("org-1"));
    assert_eq!(claims.kid(), kid(&k));
    assert_eq!(claims.upstream_iss(), Some(UPSTREAM_ISS));
    assert_eq!(keys.calls.get(), 1);
}

#[test]
fn missing_or_non_bearer_header() {
    let (keys, clock) = (StaticKeys::of(&[&key(1)]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    assert_eq!(block_on(v.verify(None)), Err(AuthError::MissingToken));
    let t = edge_token(&key(1));
    assert_eq!(
        block_on(v.verify(Some(&format!("Basic {t}")))),
        Err(AuthError::MissingToken)
    );
}

#[test]
fn expired_and_not_yet_valid() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let t = edge_token(&k);
    clock.advance(890 + 60);
    assert_eq!(run(&v, &t), Err(AuthError::Expired));

    let clock = TestClock::at(NOW);
    let v = edge_verifier(&keys, &clock);
    let t = sign(&edge_header(&k), &claims_with("nbf", json!(NOW + 120)), &k);
    assert_eq!(run(&v, &t), Err(AuthError::NotYetValid));
}

#[test]
fn wrong_issuer() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    for iss in [
        json!("https://evil.example"),
        json!(UPSTREAM_ISS),
        json!(null),
    ] {
        let t = sign(&edge_header(&k), &claims_with("iss", iss.clone()), &k);
        assert_eq!(run(&v, &t), Err(AuthError::WrongIssuer), "{iss}");
    }
    assert_eq!(
        keys.calls.get(),
        0,
        "issuer is classified before key lookup"
    );
}

#[test]
fn sibling_audience_is_403_other_audiences_401() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let t = sign(&edge_header(&k), &claims_with("aud", json!(SIBLING)), &k);
    let err = run(&v, &t).unwrap_err();
    assert_eq!(err, AuthError::AudienceNotAllowed);
    assert_eq!(err.http_status(), 403);
    assert_eq!(err.rfc6750_error(), None);
    for aud in [
        json!("https://api.cognitum.one/v1/mcp"),
        json!("edge-client-abc"),
        json!(format!("{RESOURCE}/")),
    ] {
        let t = sign(&edge_header(&k), &claims_with("aud", aud.clone()), &k);
        let err = run(&v, &t).unwrap_err();
        assert_eq!(err, AuthError::InvalidClaim("aud"), "{aud}");
        assert_eq!(err.http_status(), 401);
        assert_eq!(err.rfc6750_error(), Some("invalid_token"));
    }
}

#[test]
fn aud_array_rejected() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    for aud in [
        json!([RESOURCE]),
        json!([RESOURCE, "https://other.example"]),
        json!([]),
        json!(null),
    ] {
        let t = sign(&edge_header(&k), &claims_with("aud", aud.clone()), &k);
        assert_eq!(run(&v, &t), Err(AuthError::InvalidClaim("aud")), "{aud}");
    }
}

#[test]
fn alg_confusion_rejected_before_key_lookup() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let claims = edge_claims();
    for alg in ["none", "HS256", "RS256", "ES384"] {
        let h = json!({"alg": alg, "typ": "at+jwt", "kid": kid(&k)});
        // A 64-byte signature, so only the alg rule can reject it. For HS256
        // this models "HMAC keyed with the public key" confusion.
        let t = with_sig(&h, &claims, &[0x42; 64]);
        assert_eq!(run(&v, &t), Err(AuthError::UnsupportedAlg), "{alg}");
    }
    let h = json!({"alg": "none", "typ": "at+jwt", "kid": kid(&k)});
    let unsigned = format!(
        "{}.{}.",
        b64url_encode(h.to_string().as_bytes()),
        b64url_encode(claims.to_string().as_bytes())
    );
    assert!(run(&v, &unsigned).is_err());
    assert_eq!(keys.calls.get(), 0);
}

#[test]
fn edge_token_requires_at_jwt_typ() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let h = json!({"alg": "ES256", "typ": "JWT", "kid": kid(&k)});
    assert_eq!(
        run(&v, &sign(&h, &edge_claims(), &k)),
        Err(AuthError::BadTyp)
    );
}

#[test]
fn tampered_signature_and_payload() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let t = edge_token(&k);
    let parts: Vec<&str> = t.split('.').collect();
    let mut sig = crate::jws::b64url_decode(parts[2]).unwrap();
    sig[0] ^= 0x80;
    let bad_sig = format!("{}.{}.{}", parts[0], parts[1], b64url_encode(&sig));
    assert_eq!(run(&v, &bad_sig), Err(AuthError::BadSignature));

    let forged = b64url_encode(
        claims_with("scope", json!("mcp:read mcp:invoke brains:contribute"))
            .to_string()
            .as_bytes(),
    );
    let bad_payload = format!("{}.{}.{}", parts[0], forged, parts[2]);
    assert_eq!(run(&v, &bad_payload), Err(AuthError::BadSignature));

    // Right kid, signed by a different key.
    let h = edge_header(&k);
    assert_eq!(
        run(&v, &sign(&h, &edge_claims(), &key(2))),
        Err(AuthError::BadSignature)
    );
}

#[test]
fn oversize_and_malformed_rejected_before_key_lookup() {
    let k = key(1);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let big = sign(
        &edge_header(&k),
        &claims_with("pad", json!("x".repeat(MAX_TOKEN_BYTES))),
        &k,
    );
    assert_eq!(run(&v, &big), Err(AuthError::Malformed("token too large")));

    let t = edge_token(&k);
    let parts: Vec<&str> = t.split('.').collect();
    for bad in [
        format!("{}.{}*.{}", parts[0], parts[1], parts[2]),
        format!("{}=.{}.{}", parts[0], parts[1], parts[2]),
        format!("{}.{}.{}+", parts[0], parts[1], parts[2]),
        format!("{}.{}", parts[0], parts[1]),
        format!("{t}.{}", parts[2]),
    ] {
        assert!(
            matches!(run(&v, &bad), Err(AuthError::Malformed(_))),
            "{bad}"
        );
    }
    // Payload that is valid base64 but not a JSON object.
    let arr = b64url_encode(b"[1,2,3]");
    let t2 = format!("{}.{}.{}", parts[0], arr, parts[2]);
    assert_eq!(
        run(&v, &t2),
        Err(AuthError::Malformed("claims not an object"))
    );
    assert_eq!(keys.calls.get(), 0);
}

#[test]
fn unknown_kid_refresh_through_jwks_cache() {
    let (k1, k2) = (key(1), key(2));
    let fetch = CountingFetch::serving(jwks_body(&[&k1]));
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, JwksCachePolicy::with_defaults(JWKS_URL));
    let v: Verifier<_, &StaticKeys, _> =
        Verifier::edge_only(&cache, &clock, audience(), edge_policy());
    assert!(block_on(v.verify(Some(&bearer(&edge_token(&k1))))).is_ok());
    assert_eq!(fetch.calls.get(), 1);

    // AS rotates to k2; first k2 token inside the refetch interval is refused.
    *fetch.body.borrow_mut() = jwks_body(&[&k1, &k2]);
    clock.advance(5);
    let t2 = edge_token(&k2);
    assert_eq!(
        block_on(v.verify(Some(&bearer(&t2)))),
        Err(AuthError::UnknownKid)
    );
    assert_eq!(fetch.calls.get(), 1);
    clock.advance(25);
    assert!(block_on(v.verify(Some(&bearer(&t2)))).is_ok());
    assert_eq!(fetch.calls.get(), 2);
}

#[test]
fn upstream_path_disabled_by_default() {
    let k = key(5);
    let (keys, clock) = (StaticKeys::of(&[&k]), TestClock::at(NOW));
    let v = edge_verifier(&keys, &clock);
    let h = json!({"alg": "ES256", "typ": "JWT", "kid": kid(&k)});
    let t = sign(&h, &upstream_claims(), &k);
    assert_eq!(run(&v, &t), Err(AuthError::WrongIssuer));
    assert_eq!(keys.calls.get(), 0);
}

/// Audience policy with the upstream path on, pinned to `pinned` kids.
fn upstream_audience(pinned: Vec<String>) -> AudiencePolicy {
    let mut aud = audience();
    aud.upstream = Some(UpstreamFirstPartyPolicy {
        issuer: UPSTREAM_ISS.into(),
        first_party_auds: vec![CLI_CLIENT.into()],
        accepted_kids: pinned,
    });
    aud
}

#[test]
fn upstream_path_when_explicitly_enabled() {
    let (edge_k, up_k) = (key(1), key(5));
    let (edge_keys, up_keys, clock) = (
        StaticKeys::of(&[&edge_k]),
        StaticKeys::of(&[&up_k]),
        TestClock::at(NOW),
    );
    let aud = upstream_audience(vec![kid(&up_k)]);
    // Audience allows upstream but the verifier has no upstream keys: refused.
    let no_keys: Verifier<_, &StaticKeys, _> =
        Verifier::edge_only(&edge_keys, &clock, aud.clone(), edge_policy());
    let h = json!({"alg": "ES256", "kid": kid(&up_k)});
    let t = sign(&h, &upstream_claims(), &up_k);
    assert_eq!(
        block_on(no_keys.verify(Some(&bearer(&t)))),
        Err(AuthError::WrongIssuer)
    );

    let v = Verifier::edge_only(&edge_keys, &clock, aud, edge_policy())
        .with_upstream(&up_keys, ClaimsPolicy::upstream_access(UPSTREAM_ISS));
    let ok = block_on(v.verify(Some(&bearer(&t)))).unwrap();
    assert_eq!(ok.kind(), TokenKind::UpstreamFirstParty);
    assert_eq!(ok.client_id(), CLI_CLIENT);
    assert_eq!(ok.upstream_iss(), Some(UPSTREAM_ISS));
    assert_eq!(ok.kid(), kid(&up_k));

    // Upstream-signed token cannot be accepted by the edge key source.
    let forged_edge = sign(&edge_header(&up_k), &edge_claims(), &up_k);
    assert_eq!(
        block_on(v.verify(Some(&bearer(&forged_edge)))),
        Err(AuthError::UnknownKid)
    );

    // Real-shaped ID token: iss, aud = client_id, no typ claim.
    let mut id_token = upstream_claims();
    id_token.as_object_mut().unwrap().remove("typ");
    let t = sign(&h, &id_token, &up_k);
    let err = block_on(v.verify(Some(&bearer(&t)))).unwrap_err();
    assert_eq!(err, AuthError::InvalidClaim("typ"));
    assert_eq!(err.http_status(), 401);

    // Non-allowlisted aud (a DCR connector id): 401, not 403.
    let mut dcr = upstream_claims();
    dcr["aud"] = json!("dcr-123");
    dcr["client_id"] = json!("dcr-123");
    let t = sign(&h, &dcr, &up_k);
    assert_eq!(
        block_on(v.verify(Some(&bearer(&t)))),
        Err(AuthError::InvalidClaim("aud"))
    );
}

mod regressions;
