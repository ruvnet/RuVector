//! Token request parsing and access-token minting.

use super::assert_code;
use crate::error::OAuthErrorCode as C;
use crate::ports::{MockRng, MockSigner};
use crate::testing::*;
use crate::token::*;
use crate::StoreError;
use p256::ecdsa::{signature::Verifier as _, Signature};
use ruvector_edge_auth::jws::b64url_decode;

const _: () = assert!(ACCESS_TOKEN_TTL_SECS <= 15 * 60);

fn code_form() -> Vec<(&'static str, &'static str)> {
    vec![
        ("grant_type", "authorization_code"),
        ("code", "abc"),
        ("redirect_uri", REDIRECT),
        ("client_id", CLIENT_ID),
        ("code_verifier", VERIFIER),
    ]
}

#[test]
fn parses_authorization_code_request() {
    let mut f = code_form();
    f.push(("resource", RESOURCE));
    f.push(("unknown", "ignored"));
    let r = TokenRequest::from_form(&pairs(&f)).unwrap();
    assert_eq!(
        r,
        TokenRequest::AuthorizationCode {
            code: "abc".into(),
            redirect_uri: REDIRECT.into(),
            client_id: CLIENT_ID.into(),
            code_verifier: VERIFIER.into(),
            resource: Some(RESOURCE.into()),
        }
    );
    assert_eq!(r.client_id(), CLIENT_ID);
}

#[test]
fn parses_refresh_request_and_empty_values_are_omitted() {
    let f = pairs(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", "rt"),
        ("client_id", CLIENT_ID),
        ("scope", ""),
    ]);
    assert_eq!(
        TokenRequest::from_form(&f).unwrap(),
        TokenRequest::RefreshToken {
            refresh_token: "rt".into(),
            client_id: CLIENT_ID.into(),
            scope: None,
            resource: None,
        }
    );
}

#[test]
fn request_rejections() {
    let without = |k: &str| -> Vec<(String, String)> {
        pairs(
            &code_form()
                .into_iter()
                .filter(|(n, _)| *n != k)
                .collect::<Vec<_>>(),
        )
    };
    let with = |extra: (&'static str, &'static str)| {
        let mut f = code_form();
        f.push(extra);
        pairs(&f)
    };
    for k in [
        "grant_type",
        "client_id",
        "code",
        "redirect_uri",
        "code_verifier",
    ] {
        assert_code(TokenRequest::from_form(&without(k)), C::InvalidRequest);
    }
    assert_code(
        TokenRequest::from_form(&with(("client_secret", "s"))),
        C::InvalidClient,
    );
    assert_code(
        TokenRequest::from_form(&with(("client_assertion", "j"))),
        C::InvalidClient,
    );
    assert_code(
        TokenRequest::from_form(&with(("code", "again"))),
        C::InvalidRequest,
    );
    let mut two = code_form();
    two.push(("resource", RESOURCE));
    two.push(("resource", OTHER_RESOURCE));
    assert_code(TokenRequest::from_form(&pairs(&two)), C::InvalidTarget);
    let mut g = code_form();
    g[0] = ("grant_type", "client_credentials");
    assert_code(TokenRequest::from_form(&pairs(&g)), C::UnsupportedGrantType);
    g[0] = ("grant_type", "password");
    assert_code(TokenRequest::from_form(&pairs(&g)), C::UnsupportedGrantType);
    let refresh_missing = pairs(&[("grant_type", "refresh_token"), ("client_id", CLIENT_ID)]);
    assert_code(TokenRequest::from_form(&refresh_missing), C::InvalidRequest);
    let long = "c".repeat(crate::params::MAX_PARAM_LEN + 1);
    let mut f = pairs(&code_form());
    f[1].1 = long;
    assert_code(TokenRequest::from_form(&f), C::InvalidRequest);
}

fn mint(
    signer: &dyn crate::Signer,
    rng: &dyn crate::Rng,
) -> Result<(String, AccessTokenClaims), crate::OAuthError> {
    let (res, id, scopes) = (
        resource(),
        identity(),
        vec!["ruvector:read".to_string(), "ruvector:write".to_string()],
    );
    mint_access_token(
        signer,
        rng,
        &FixedClock::at(T0),
        &MintRequest {
            issuer: ISSUER,
            resource: &res,
            client_id: CLIENT_ID,
            identity: &id,
            family_id: "fam-1",
            scopes: &scopes,
        },
    )
}

#[test]
fn minted_token_is_valid_es256_at_jwt_with_exact_aud() {
    let signer = TestSigner::default();
    let (jwt, claims) = mint(&signer, &SeqRng::default()).unwrap();
    let parts: Vec<&str> = jwt.split('.').collect();
    assert_eq!(parts.len(), 3);
    let header: serde_json::Value =
        serde_json::from_slice(&b64url_decode(parts[0]).unwrap()).unwrap();
    assert_eq!(
        header,
        serde_json::json!({"alg":"ES256","typ":"at+jwt","kid":"test-kid"})
    );
    let payload: serde_json::Value =
        serde_json::from_slice(&b64url_decode(parts[1]).unwrap()).unwrap();
    assert_eq!(payload["iss"], ISSUER);
    assert_eq!(payload["aud"], RESOURCE);
    assert!(payload["aud"].is_string(), "aud must be a single string");
    assert_eq!(payload["sub"], identity().edge_subject());
    assert_eq!(payload["upstream_iss"], "https://auth.cognitum.one");
    assert_eq!(payload["client_id"], CLIENT_ID);
    assert_eq!(payload["org_id"], "org_123");
    assert_eq!(payload["workspace_id"], "ws-456");
    assert_eq!(payload["family_id"], "fam-1");
    assert_eq!(payload["scope"], "ruvector:read ruvector:write");
    assert_eq!(payload["iat"], T0);
    assert_eq!(payload["exp"], T0 + ACCESS_TOKEN_TTL_SECS);
    assert_eq!(payload["jti"].as_str().unwrap().len(), 22);
    assert_eq!(payload["jti"], claims.jti);
    let sig = Signature::from_slice(&b64url_decode(parts[2]).unwrap()).unwrap();
    let input = format!("{}.{}", parts[0], parts[1]);
    signer
        .verifying_key()
        .verify(input.as_bytes(), &sig)
        .unwrap();
    // Tampering with the payload breaks the signature.
    let forged = format!("{input}x");
    assert!(signer
        .verifying_key()
        .verify(forged.as_bytes(), &sig)
        .is_err());
}

#[test]
fn jti_is_unique_per_mint() {
    let (signer, rng) = (TestSigner::default(), SeqRng::default());
    let (_, a) = mint(&signer, &rng).unwrap();
    let (_, b) = mint(&signer, &rng).unwrap();
    assert_ne!(a.jti, b.jti);
}

#[test]
fn mint_failures_are_server_errors() {
    let mut no_kid = MockSigner::new();
    no_kid.expect_kid().returning(String::new);
    no_kid.expect_sign_es256().times(0);
    assert_code(mint(&no_kid, &SeqRng::default()), C::ServerError);

    let mut bad_sign = MockSigner::new();
    bad_sign.expect_kid().returning(|| "k".into());
    bad_sign
        .expect_sign_es256()
        .returning(|_| Err(StoreError("hsm".into())));
    assert_code(mint(&bad_sign, &SeqRng::default()), C::ServerError);

    let mut rng = MockRng::new();
    rng.expect_fill()
        .returning(|_| Err(StoreError("rng".into())));
    let mut signer = MockSigner::new();
    signer.expect_kid().returning(|| "k".into());
    signer.expect_sign_es256().times(0);
    assert_code(mint(&signer, &rng), C::ServerError);
}

#[test]
fn token_response_shape() {
    let r = TokenResponse {
        access_token: "a".into(),
        token_type: "Bearer",
        expires_in: 900,
        refresh_token: None,
        scope: "ruvector:read".into(),
    };
    let v = serde_json::to_value(r).unwrap();
    assert!(v.get("refresh_token").is_none());
    assert_eq!(v["token_type"], "Bearer");
}
