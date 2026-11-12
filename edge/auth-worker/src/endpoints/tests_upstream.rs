//! Upstream-leg hygiene at the callback: the unused upstream refresh token
//! is revoked, and an ID token must carry the flow's nonce (ADR-351 §5.6
//! steps 3-4).

use super::*;
use std::cell::RefCell;

/// Token endpoint answering `body`, recording every POST.
struct Recording {
    body: Value,
    posts: RefCell<Vec<(String, String)>>,
}

impl Recording {
    fn new(body: Value) -> Self {
        Recording {
            body,
            posts: RefCell::new(Vec::new()),
        }
    }
}

impl UpstreamHttp for Recording {
    async fn post_form(&self, url: &str, body: String) -> Result<HttpResponse, FetchError> {
        self.posts.borrow_mut().push((url.to_string(), body));
        Ok(HttpResponse {
            status: 200,
            body: self.body.to_string().into_bytes(),
        })
    }
}

fn finish(w: &World, up: &str, cookie: &str, http: &Recording) -> Reply {
    let q = format!("state={}&code=c", param(up, "state").unwrap());
    block_on(callback(
        &w.ctx(),
        Some(&q),
        Some(cookie),
        http,
        OneKey(*w.upstream_key.verifying_key()),
    ))
}

/// Regression (live 90-day upstream refresh family per login): the
/// upstream refresh token is revoked at the configured endpoint right after
/// the exchange, as a form-encoded RFC 7009 request.
#[test]
fn callback_revokes_the_upstream_refresh_token() {
    let w = World::new();
    let client_id = w.client_id();
    let (up, cookie) = start(&w, &client_id);
    let at = w.upstream_token(w.good_upstream_claims());
    let http = Recording::new(json!({
        "access_token": at, "token_type": "Bearer", "refresh_token": "up-rt-1",
    }));
    let r = finish(&w, &up, &cookie, &http);
    assert_eq!(r.status, 302);
    assert!(param(r.header("Location").unwrap(), "code").is_some());
    let posts = http.posts.borrow();
    let revoke = posts
        .iter()
        .find(|(u, _)| u == "https://auth.cognitum.one/oauth/revoke")
        .expect("upstream refresh token revoked");
    assert!(revoke.1.contains("token=up-rt-1"), "{}", revoke.1);
    assert!(revoke.1.contains("token_type_hint=refresh_token"));
    assert!(revoke.1.contains("client_id=dcr-edge-test"));
}

/// Regression (nonce never checked): an ID token with another flow's nonce
/// is refused with `access_denied` and no code; the flow's own nonce passes.
#[test]
fn callback_requires_the_flow_nonce_in_an_id_token() {
    let w = World::new();
    let client_id = w.client_id();
    let at = w.upstream_token(w.good_upstream_claims());
    let id_token = |nonce: &str| {
        w.upstream_token(json!({
            "iss": "https://auth.cognitum.one", "aud": "dcr-edge-test", "sub": "user-1",
            "iat": T0, "exp": T0 + 300, "nonce": nonce,
        }))
    };
    let (up, cookie) = start(&w, &client_id);
    let http = Recording::new(json!({
        "access_token": at, "token_type": "Bearer", "id_token": id_token("other-flow"),
    }));
    let r = finish(&w, &up, &cookie, &http);
    let loc = r.header("Location").unwrap();
    assert_eq!(param(loc, "error").as_deref(), Some("access_denied"));
    assert!(param(loc, "code").is_none());
    let (up, cookie) = start(&w, &client_id);
    let nonce = param(&up, "nonce").unwrap();
    let http = Recording::new(json!({
        "access_token": at, "token_type": "Bearer", "id_token": id_token(&nonce),
    }));
    let r = finish(&w, &up, &cookie, &http);
    assert!(param(r.header("Location").unwrap(), "code").is_some());
}
