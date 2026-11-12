//! Back-channel leg of the federation to `auth.cognitum.one`: the code
//! exchange (public client, PKCE S256), verification of the upstream access
//! token against the upstream JWKS, the ID-token `nonce` check and the
//! revocation of the upstream refresh token we never use.
//!
//! The upstream **access token** is the identity source: per ADR-351 §1.1 it
//! always carries `sub`, `org_id`, `workspace_id` and `typ = "access"`, with
//! `aud` = our upstream client id. An ID token, if returned, is verified only
//! to bind the flow's `nonce` (ADR-351 §5.6 step 3) and then discarded.

use ruvector_edge_auth::jws::{parse_compact, verify_es256};
use ruvector_edge_auth::{
    AudiencePolicy, AuthError, ClaimsPolicy, Clock, FetchError, HttpResponse, KeySource,
    ResourceUrl, UpstreamFirstPartyPolicy, VerifiedClaims, Verifier, VerifyingKey,
};
use ruvector_edge_authz::federation::UpstreamConfig;
use ruvector_edge_authz::{OAuthError, OAuthErrorCode};
use serde::Deserialize;

/// Maximum upstream token-endpoint response body.
pub const MAX_UPSTREAM_BODY: usize = 64 * 1024;
/// Clock skew accepted on the ID token's time claims.
const ID_TOKEN_SKEW_SECS: u64 = 60;

/// Network port for the token exchange (the Worker implements it with a
/// `POST` via global fetch, redirects not followed).
#[allow(async_fn_in_trait)]
pub trait UpstreamHttp {
    /// `POST url` with an `application/x-www-form-urlencoded` body.
    async fn post_form(&self, url: &str, body: String) -> Result<HttpResponse, FetchError>;
}

/// The upstream token response members we use. `Debug` redacts every token.
#[derive(Clone, PartialEq, Eq, Deserialize)]
pub struct UpstreamTokens {
    /// Upstream ES256 access token.
    pub access_token: String,
    /// Must be `Bearer` (case-insensitive).
    pub token_type: String,
    /// Upstream refresh token: never stored or logged, revoked at once.
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Upstream ID token: verified for its `nonce`, then discarded.
    #[serde(default)]
    pub id_token: Option<String>,
}

impl std::fmt::Debug for UpstreamTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpstreamTokens")
            .field("access_token", &"<redacted>")
            .field("token_type", &self.token_type)
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("id_token", &self.id_token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

fn upstream_failure() -> OAuthError {
    OAuthError::new(
        OAuthErrorCode::ServerError,
        "upstream token exchange failed",
    )
}

impl UpstreamTokens {
    /// Parse and bound a token-endpoint response.
    pub fn from_response(resp: &HttpResponse) -> Result<Self, OAuthError> {
        if resp.status != 200 || resp.body.len() > MAX_UPSTREAM_BODY {
            return Err(upstream_failure());
        }
        let t: UpstreamTokens =
            serde_json::from_slice(&resp.body).map_err(|_| upstream_failure())?;
        let bounded =
            |s: &str| !s.is_empty() && s.len() <= ruvector_edge_auth::jws::MAX_TOKEN_BYTES;
        let ok = t.token_type.eq_ignore_ascii_case("bearer")
            && bounded(&t.access_token)
            && t.refresh_token.as_deref().map_or(true, bounded)
            && t.id_token.as_deref().map_or(true, bounded);
        ok.then_some(t).ok_or_else(upstream_failure)
    }
}

fn form(pairs: &[(&str, &str)]) -> String {
    url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(pairs.iter())
        .finish()
}

/// Exchange an upstream authorization code for tokens (public client, PKCE:
/// the form comes from `federation::upstream_token_form`).
pub async fn exchange<H: UpstreamHttp>(
    http: &H,
    cfg: &UpstreamConfig,
    form_pairs: &[(String, String)],
) -> Result<UpstreamTokens, OAuthError> {
    let body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(form_pairs.iter())
        .finish();
    let resp = http
        .post_form(&cfg.token_endpoint, body)
        .await
        .map_err(|_| upstream_failure())?;
    UpstreamTokens::from_response(&resp)
}

/// Best-effort RFC 7009 revocation of the upstream refresh token (ADR-351
/// §5.6 step 4): the edge AS never refreshes upstream, so no live upstream
/// refresh family may outlive the login. Errors are ignored.
pub async fn revoke_refresh<H: UpstreamHttp>(
    http: &H,
    revocation_endpoint: &str,
    client_id: &str,
    refresh_token: &str,
) {
    let body = form(&[
        ("token", refresh_token),
        ("token_type_hint", "refresh_token"),
        ("client_id", client_id),
    ]);
    let _ = http.post_form(revocation_endpoint, body).await;
}

/// Key source that never yields a key: edge-issued tokens are never accepted
/// on the upstream leg.
pub struct NoKeys;

impl KeySource for NoKeys {
    async fn verifying_key(&self, _kid: &str) -> Result<VerifyingKey, AuthError> {
        Err(AuthError::KeysUnavailable)
    }
}

/// Verify an upstream access token: upstream issuer, upstream JWKS, ES256,
/// `kid` in the pinned `accepted_kids`, `aud == cfg.client_id` exactly,
/// `typ == "access"`, time claims.
///
/// `edge_issuer` is our own issuer; tokens claiming it are classified as
/// edge tokens and fail on [`NoKeys`].
pub async fn verify_access_token<K: KeySource, C: Clock + ?Sized>(
    keys: K,
    clock: &C,
    edge_issuer: &str,
    cfg: &UpstreamConfig,
    accepted_kids: &[String],
    access_token: &str,
) -> Result<VerifiedClaims, AuthError> {
    if cfg.client_id.is_empty() {
        return Err(AuthError::InvalidConfig("UPSTREAM_CLIENT_ID empty"));
    }
    let resource = ResourceUrl::parse(edge_issuer)?;
    let audience = AudiencePolicy {
        edge_issuer: edge_issuer.to_string(),
        resource,
        upstream: Some(UpstreamFirstPartyPolicy {
            issuer: cfg.issuer.clone(),
            first_party_auds: vec![cfg.client_id.clone()],
            accepted_kids: accepted_kids.to_vec(),
        }),
    };
    let verifier = Verifier::edge_only(
        NoKeys,
        clock,
        audience,
        ClaimsPolicy::edge(edge_issuer.to_string(), cfg.issuer.clone()),
    )
    .with_upstream(keys, ClaimsPolicy::upstream_access(cfg.issuer.clone()));
    verifier
        .verify(Some(&format!("Bearer {access_token}")))
        .await
}

/// The ID-token claims the `nonce` check reads.
#[derive(Deserialize)]
struct IdTokenClaims {
    iss: String,
    aud: serde_json::Value,
    exp: u64,
    #[serde(default)]
    iat: Option<u64>,
    #[serde(default)]
    nonce: Option<String>,
}

fn eq_ct(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y))
            == 0
}

/// Verify an upstream ID token for the flow's `nonce` (ADR-351 §5.6 step 3):
/// ES256 over the upstream JWKS with `kid` in `accepted_kids`, `iss` exact,
/// `aud` exactly `cfg.client_id` (string or one-element array), unexpired,
/// and `nonce` present and equal to `expected_nonce` (constant time).
/// Signature, key or claim failures are `server_error`; a missing or wrong
/// `nonce` is `access_denied`.
pub async fn check_id_token<K: KeySource, C: Clock + ?Sized>(
    keys: K,
    clock: &C,
    cfg: &UpstreamConfig,
    accepted_kids: &[String],
    id_token: &str,
    expected_nonce: &str,
) -> Result<(), OAuthError> {
    let bad = || OAuthError::new(OAuthErrorCode::ServerError, "upstream id_token rejected");
    let jws = parse_compact(id_token).map_err(|_| bad())?;
    if !accepted_kids.contains(&jws.header.kid) {
        return Err(bad());
    }
    let key = keys
        .verifying_key(&jws.header.kid)
        .await
        .map_err(|_| bad())?;
    verify_es256(&jws, &key).map_err(|_| bad())?;
    let claims: IdTokenClaims =
        serde_json::from_slice(&jws.payload_bytes().map_err(|_| bad())?).map_err(|_| bad())?;
    let aud_ok = match &claims.aud {
        serde_json::Value::String(a) => *a == cfg.client_id,
        serde_json::Value::Array(v) => v.len() == 1 && v[0] == cfg.client_id.as_str(),
        _ => false,
    };
    let now = clock.now_unix();
    let time_ok = now < claims.exp.saturating_add(ID_TOKEN_SKEW_SECS)
        && claims
            .iat
            .map_or(true, |iat| iat <= now.saturating_add(ID_TOKEN_SKEW_SECS));
    if claims.iss != cfg.issuer || !aud_ok || !time_ok {
        return Err(bad());
    }
    match claims.nonce.as_deref() {
        Some(n) if eq_ct(n, expected_nonce) => Ok(()),
        _ => Err(OAuthError::new(
            OAuthErrorCode::AccessDenied,
            "upstream nonce mismatch",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> UpstreamConfig {
        UpstreamConfig {
            issuer: "https://auth.cognitum.one".into(),
            authorization_endpoint: "https://auth.cognitum.one/oauth/authorize".into(),
            token_endpoint: "https://auth.cognitum.one/oauth/token".into(),
            jwks_url: "https://auth.cognitum.one/.well-known/jwks.json".into(),
            client_id: "dcr-edge".into(),
            redirect_uri: "https://as.example/callback".into(),
            scopes: vec!["openid".into()],
        }
    }

    fn resp(status: u16, body: &str) -> HttpResponse {
        HttpResponse {
            status,
            body: body.as_bytes().to_vec(),
        }
    }

    struct Capture(std::cell::RefCell<Option<(String, String)>>, HttpResponse);

    impl UpstreamHttp for Capture {
        async fn post_form(&self, url: &str, body: String) -> Result<HttpResponse, FetchError> {
            *self.0.borrow_mut() = Some((url.to_string(), body));
            Ok(self.1.clone())
        }
    }

    #[test]
    fn exchange_posts_encoded_form_to_token_endpoint() {
        let http = Capture(
            Default::default(),
            resp(200, r#"{"access_token":"a.b.c","token_type":"Bearer"}"#),
        );
        let form = vec![
            ("grant_type".to_string(), "authorization_code".to_string()),
            ("code".to_string(), "c o&de".to_string()),
        ];
        let t = crate::testutil::block_on(exchange(&http, &cfg(), &form)).unwrap();
        assert_eq!(t.access_token, "a.b.c");
        let (url, body) = http.0.borrow().clone().unwrap();
        assert_eq!(url, "https://auth.cognitum.one/oauth/token");
        assert_eq!(body, "grant_type=authorization_code&code=c+o%26de");
    }

    #[test]
    fn token_response_is_bounded_and_typed() {
        let ok = resp(
            200,
            r#"{"access_token":"a.b.c","token_type":"bearer","id_token":"x"}"#,
        );
        assert_eq!(
            UpstreamTokens::from_response(&ok).unwrap().access_token,
            "a.b.c"
        );
        for bad in [
            resp(400, r#"{"error":"invalid_grant"}"#),
            resp(200, r#"{"access_token":"","token_type":"Bearer"}"#),
            resp(200, r#"{"access_token":"a","token_type":"DPoP"}"#),
            resp(200, "not json"),
            resp(200, &"x".repeat(MAX_UPSTREAM_BODY + 1)),
        ] {
            let e = UpstreamTokens::from_response(&bad).unwrap_err();
            assert_eq!(e.error, OAuthErrorCode::ServerError);
        }
    }

    #[test]
    fn verify_refuses_when_upstream_client_unset() {
        let mut c = cfg();
        c.client_id.clear();
        let r = crate::testutil::block_on(verify_access_token(
            NoKeys,
            &crate::testutil::FixedClock::at(1_000),
            "https://as.example",
            &c,
            &["k1".to_string()],
            "a.b.c",
        ));
        assert!(matches!(r, Err(AuthError::InvalidConfig(_))));
    }

    #[test]
    fn revoke_posts_the_refresh_token_form_and_debug_redacts() {
        let http = Capture(Default::default(), resp(200, ""));
        crate::testutil::block_on(revoke_refresh(
            &http,
            "https://auth.cognitum.one/oauth/revoke",
            "dcr-edge",
            "rt s3cret",
        ));
        let (url, body) = http.0.borrow().clone().unwrap();
        assert_eq!(url, "https://auth.cognitum.one/oauth/revoke");
        assert_eq!(
            body,
            "token=rt+s3cret&token_type_hint=refresh_token&client_id=dcr-edge"
        );
        let t = UpstreamTokens::from_response(&resp(
            200,
            r#"{"access_token":"a.b.c","token_type":"Bearer","refresh_token":"rt","id_token":"i.d.t"}"#,
        ))
        .unwrap();
        assert_eq!(t.refresh_token.as_deref(), Some("rt"));
        let dbg = format!("{t:?}");
        assert!(!dbg.contains("a.b.c") && !dbg.contains("\"rt\"") && !dbg.contains("i.d.t"));
    }

    mod id_token {
        use super::*;
        use crate::signer::tests::test_key;
        use p256::ecdsa::signature::Signer as _;
        use ruvector_edge_auth::jws::b64url_encode;
        use ruvector_edge_auth::Jwk;
        use serde_json::{json, Value};

        const NOW: u64 = 1_800_000_000;

        struct One(VerifyingKey);
        impl KeySource for One {
            async fn verifying_key(&self, _kid: &str) -> Result<VerifyingKey, AuthError> {
                Ok(self.0)
            }
        }

        fn kid() -> String {
            Jwk::from_verifying_key(test_key(5).verifying_key()).kid
        }

        fn sign(claims: Value) -> String {
            let k = test_key(5);
            let h = json!({"alg": "ES256", "typ": "JWT", "kid": kid()});
            let input = format!(
                "{}.{}",
                b64url_encode(h.to_string().as_bytes()),
                b64url_encode(claims.to_string().as_bytes())
            );
            let sig: p256::ecdsa::Signature = k.sign(input.as_bytes());
            format!("{input}.{}", b64url_encode(&sig.to_bytes()))
        }

        fn good() -> Value {
            json!({"iss": "https://auth.cognitum.one", "aud": "dcr-edge", "sub": "u",
                   "iat": NOW, "exp": NOW + 300, "nonce": "n-123"})
        }

        fn run(token: &str, kids: &[String]) -> Result<(), OAuthError> {
            crate::testutil::block_on(check_id_token(
                One(*test_key(5).verifying_key()),
                &crate::testutil::FixedClock::at(NOW),
                &cfg(),
                kids,
                token,
                "n-123",
            ))
        }

        /// Regression (ADR §5.6 step 3 never ran): an ID token must carry
        /// the flow's nonce; any other nonce, or none, is access_denied.
        #[test]
        fn nonce_must_match_the_flow() {
            let kids = vec![kid()];
            assert!(run(&sign(good()), &kids).is_ok());
            let mut array_aud = good();
            array_aud["aud"] = json!(["dcr-edge"]);
            assert!(run(&sign(array_aud), &kids).is_ok());
            for nonce in [json!("n-124"), Value::Null] {
                let mut c = good();
                c["nonce"] = nonce;
                assert_eq!(
                    run(&sign(c), &kids).unwrap_err().error,
                    OAuthErrorCode::AccessDenied
                );
            }
        }

        #[test]
        fn signature_kid_issuer_audience_and_expiry_are_checked() {
            let kids = vec![kid()];
            assert!(run(&sign(good()), &["other".to_string()]).is_err());
            let tampered = sign(good()).replacen('.', ".e30", 1);
            assert!(run(&tampered, &kids).is_err());
            for (k, v) in [
                ("iss", json!("https://evil.example")),
                ("aud", json!("dcr-other")),
                ("aud", json!(["dcr-edge", "dcr-other"])),
                ("exp", json!(NOW - 3600)),
                ("iat", json!(NOW + 3600)),
            ] {
                let mut c = good();
                c[k] = v;
                let e = run(&sign(c), &kids).unwrap_err();
                assert_eq!(e.error, OAuthErrorCode::ServerError, "{k}");
            }
        }
    }
}
