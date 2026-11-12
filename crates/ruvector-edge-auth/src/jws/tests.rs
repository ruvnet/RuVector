use super::*;
use crate::test_support::*;
use serde_json::json;

fn sig64() -> [u8; 64] {
    [7u8; 64]
}

#[test]
fn bearer_token_extraction() {
    assert_eq!(bearer_token(Some("Bearer abc")), Ok("abc"));
    assert_eq!(bearer_token(Some("bearer abc")), Ok("abc"));
    assert_eq!(bearer_token(None), Err(AuthError::MissingToken));
    assert_eq!(
        bearer_token(Some("Basic abc")),
        Err(AuthError::MissingToken)
    );
    assert_eq!(bearer_token(Some("Bearer")), Err(AuthError::MissingToken));
    assert_eq!(
        bearer_token(Some("Bearer ")),
        Err(AuthError::MalformedAuthorization)
    );
    assert_eq!(
        bearer_token(Some("Bearer  abc")),
        Err(AuthError::MalformedAuthorization)
    );
    assert_eq!(
        bearer_token(Some("Bearer a b")),
        Err(AuthError::MalformedAuthorization)
    );
}

#[test]
fn parses_and_verifies_valid_token() {
    let k = key(1);
    let t = edge_token(&k);
    let jws = parse_compact(&t).unwrap();
    assert_eq!(jws.header.alg, "ES256");
    assert_eq!(jws.header.kid, kid(&k));
    assert_eq!(jws.header.typ.as_deref(), Some("at+jwt"));
    assert!(t.starts_with(jws.signing_input));
    let claims: serde_json::Value = serde_json::from_slice(&jws.payload_bytes().unwrap()).unwrap();
    assert_eq!(claims["aud"], RESOURCE);
    assert_eq!(verify_es256(&jws, k.verifying_key()), Ok(()));
}

#[test]
fn oversize_token_rejected() {
    let k = key(1);
    let mut claims = edge_claims();
    claims["pad"] = json!("x".repeat(MAX_TOKEN_BYTES));
    let t = sign(&edge_header(&k), &claims, &k);
    assert!(t.len() > MAX_TOKEN_BYTES);
    assert_eq!(
        parse_compact(&t).unwrap_err(),
        AuthError::Malformed("token too large")
    );
    // Exactly at the limit is allowed through the size gate (fails later).
    let at_limit = "a".repeat(MAX_TOKEN_BYTES);
    assert_ne!(
        parse_compact(&at_limit).unwrap_err(),
        AuthError::Malformed("token too large")
    );
}

#[test]
fn segment_count_enforced() {
    for t in ["a.b", "a.b.c.d", "abc", "a..c", ".b.c", "a.b."] {
        assert!(
            matches!(parse_compact(t), Err(AuthError::Malformed(_))),
            "{t}"
        );
    }
}

#[test]
fn malformed_base64_rejected() {
    let k = key(1);
    let t = edge_token(&k);
    let parts: Vec<&str> = t.split('.').collect();
    let bad = [
        format!("{}=.{}.{}", parts[0], parts[1], parts[2]), // padding
        format!("{}.{}+.{}", parts[0], parts[1], parts[2]), // std alphabet
        format!("{}.{}/x.{}", parts[0], parts[1], parts[2]),
        format!("{}.{}.{}!", parts[0], parts[1], parts[2]),
        format!("{}.{}.{}=", parts[0], parts[1], parts[2]),
        format!("{}.A.{}", parts[0], parts[2]), // len % 4 == 1
    ];
    for t in bad {
        assert_eq!(
            parse_compact(&t).unwrap_err(),
            AuthError::Malformed("base64url"),
            "{t}"
        );
    }
}

#[test]
fn non_canonical_trailing_bits_rejected() {
    // "QQ" is canonical for b"A"; "QR" differs only in the unused low bits.
    assert_eq!(b64url_decode("QQ").unwrap(), b"A");
    assert!(b64url_decode("QR").is_err());
}

#[test]
fn alg_allowlist_is_strict() {
    let claims = edge_claims();
    for alg in [
        json!("none"),
        json!("HS256"),
        json!("HS512"),
        json!("RS256"),
        json!("PS256"),
        json!("ES384"),
        json!("es256"),
        json!(["ES256"]),
        json!(null),
    ] {
        let h = json!({"alg": alg, "typ": "at+jwt", "kid": kid(&key(1))});
        let t = with_sig(&h, &claims, &sig64());
        assert_eq!(
            parse_compact(&t).unwrap_err(),
            AuthError::UnsupportedAlg,
            "{alg}"
        );
    }
    let no_alg = json!({"typ": "at+jwt", "kid": "k"});
    assert_eq!(
        parse_compact(&with_sig(&no_alg, &claims, &sig64())).unwrap_err(),
        AuthError::UnsupportedAlg
    );
    // alg=none with an empty signature: segment rule fires, still refused.
    let none = json!({"alg": "none", "kid": "k"});
    assert!(parse_compact(&with_sig(&none, &claims, &[])).is_err());
}

#[test]
fn forbidden_header_params_rejected() {
    for p in FORBIDDEN_HEADER_PARAMS.iter().chain(["crit"].iter()) {
        let mut h = edge_header(&key(1));
        h[*p] = json!("https://evil.example/jwks");
        let t = with_sig(&h, &edge_claims(), &sig64());
        assert_eq!(
            parse_compact(&t).unwrap_err(),
            AuthError::ForbiddenHeader(p),
            "{p}"
        );
    }
}

#[test]
fn kid_rules() {
    let claims = edge_claims();
    let h = json!({"alg": "ES256", "typ": "at+jwt"});
    assert_eq!(
        parse_compact(&with_sig(&h, &claims, &sig64())).unwrap_err(),
        AuthError::MissingKid
    );
    for bad in [
        json!(5),
        json!(""),
        json!("a/b"),
        json!("x".repeat(MAX_KID_LEN + 1)),
    ] {
        let h = json!({"alg": "ES256", "typ": "at+jwt", "kid": bad});
        assert!(matches!(
            parse_compact(&with_sig(&h, &claims, &sig64())),
            Err(AuthError::Malformed(_))
        ));
    }
    let h = json!({"alg": "ES256", "typ": 1, "kid": "abc"});
    assert_eq!(
        parse_compact(&with_sig(&h, &claims, &sig64())).unwrap_err(),
        AuthError::BadTyp
    );
}

#[test]
fn header_must_be_object_without_duplicates() {
    let claims = b64url_encode(edge_claims().to_string().as_bytes());
    let sig = b64url_encode(&sig64());
    let arr = b64url_encode(br#"["ES256","kid"]"#);
    assert_eq!(
        parse_compact(&format!("{arr}.{claims}.{sig}")).unwrap_err(),
        AuthError::Malformed("header not an object")
    );
    let dup = b64url_encode(br#"{"alg":"ES256","alg":"none","kid":"k"}"#);
    assert_eq!(
        parse_compact(&format!("{dup}.{claims}.{sig}")).unwrap_err(),
        AuthError::Malformed("header json")
    );
}

#[test]
fn non_64_byte_signatures_rejected() {
    let k = key(1);
    // DER-shaped (70-72 bytes) and short signatures.
    for len in [0usize, 63, 65, 70, 72] {
        let t = with_sig(&edge_header(&k), &edge_claims(), &vec![0x30; len]);
        let err = parse_compact(&t).unwrap_err();
        assert!(
            matches!(err, AuthError::BadSignature | AuthError::Malformed(_)),
            "{len}: {err:?}"
        );
    }
}

#[test]
fn tampered_signature_or_payload_fails() {
    let k = key(1);
    let t = edge_token(&k);
    let jws = parse_compact(&t).unwrap();
    let mut flipped = jws.clone();
    flipped.signature[10] ^= 0x01;
    assert_eq!(
        verify_es256(&flipped, k.verifying_key()),
        Err(AuthError::BadSignature)
    );
    let mut zero = jws.clone();
    zero.signature = [0u8; 64];
    assert_eq!(
        verify_es256(&zero, k.verifying_key()),
        Err(AuthError::BadSignature)
    );

    let mut claims = edge_claims();
    claims["sub"] = json!("attacker");
    let forged_payload = b64url_encode(claims.to_string().as_bytes());
    let parts: Vec<&str> = t.split('.').collect();
    let forged = format!("{}.{}.{}", parts[0], forged_payload, parts[2]);
    let jws = parse_compact(&forged).unwrap();
    assert_eq!(
        verify_es256(&jws, k.verifying_key()),
        Err(AuthError::BadSignature)
    );
}

#[test]
fn wrong_key_fails() {
    let t = edge_token(&key(1));
    let jws = parse_compact(&t).unwrap();
    assert_eq!(
        verify_es256(&jws, key(2).verifying_key()),
        Err(AuthError::BadSignature)
    );
}
