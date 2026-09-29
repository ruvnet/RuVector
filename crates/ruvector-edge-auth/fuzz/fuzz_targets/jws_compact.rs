//! Compact JWS parsing and ES256 verification (ADR-351 §5.4, M0 fuzz
//! criterion). Two modes, picked by the first byte:
//!
//! * even: the rest is an `Authorization` header value driven through the
//!   resource-server pipeline (`bearer_token` -> `parse_compact` ->
//!   `payload_bytes` -> claims JSON -> `classify` -> `verify_es256` ->
//!   `check_audience` -> `claims::validate`);
//! * odd: the rest is split into a header and a payload that are encoded and
//!   **validly signed** with a fixed key, so the post-signature audience and
//!   claims checks see arbitrary JSON too (random signatures never verify).
//!
//! Invariant: no panic, and nothing is accepted unless every check passed.
#![no_main]

use libfuzzer_sys::fuzz_target;
use p256::ecdsa::signature::Signer as _;
use p256::ecdsa::{Signature, SigningKey};
use ruvector_edge_auth::claims::validate;
use ruvector_edge_auth::jws::{b64url_decode, b64url_encode};
use ruvector_edge_auth::{
    bearer_token, parse_compact, verify_es256, AudiencePolicy, ClaimsPolicy, RawClaims, ResourceUrl,
};

const ISSUER: &str = "https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev";
const RESOURCE: &str = "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1";
const NOW: u64 = 1_790_000_000;

fn key() -> SigningKey {
    SigningKey::from_slice(&[7u8; 32]).expect("fixed scalar is a valid key")
}

fn pipeline(authorization: &str) {
    let Ok(token) = bearer_token(Some(authorization)) else {
        return;
    };
    let Ok(jws) = parse_compact(token) else {
        return;
    };
    // The parsed header obeys the static policy.
    assert_eq!(jws.header.alg, "ES256");
    assert!(!jws.header.kid.is_empty());
    let Ok(payload) = jws.payload_bytes() else {
        return;
    };
    // Strict base64url: re-encoding reproduces the segment exactly.
    assert_eq!(b64url_encode(&payload), jws.payload_b64);
    let Ok(raw) = serde_json::from_slice::<RawClaims>(&payload) else {
        return;
    };
    let audience = AudiencePolicy::edge_only(ISSUER, ResourceUrl::parse(RESOURCE).unwrap());
    let Ok(kind) = audience.classify(raw.iss.as_deref(), jws.header.typ.as_deref()) else {
        return;
    };
    let vk = *key().verifying_key();
    if verify_es256(&jws, &vk).is_err() {
        return;
    }
    if audience.check_audience(kind, raw.aud.as_ref()).is_err() {
        return;
    }
    let policy = ClaimsPolicy::edge(ISSUER, "https://auth.cognitum.one");
    if let Ok(claims) = validate(raw, &policy, kind, NOW) {
        // Accepted tokens satisfy the §5.2/§5.4 invariants.
        assert_eq!(claims.iss(), ISSUER);
        assert_eq!(claims.aud(), RESOURCE);
        assert!(claims.exp() > claims.iat());
        assert!(claims.exp() - claims.iat() <= 900);
    }
}

fn signed(data: &[u8]) -> String {
    let (len, data) = data
        .split_first()
        .map_or((0, data), |(b, r)| (*b as usize, r));
    let (header, payload) = data.split_at(len.min(data.len()));
    let input = format!("{}.{}", b64url_encode(header), b64url_encode(payload));
    let sig: Signature = key().sign(input.as_bytes());
    format!("Bearer {input}.{}", b64url_encode(&sig.to_bytes()))
}

fuzz_target!(|data: &[u8]| {
    let Some((mode, rest)) = data.split_first() else {
        return;
    };
    if mode % 2 == 0 {
        if let Ok(s) = std::str::from_utf8(rest) {
            pipeline(s);
        }
        // Strict decoder never panics on arbitrary input.
        let _ = b64url_decode(&String::from_utf8_lossy(rest));
    } else {
        pipeline(&signed(rest));
    }
});
