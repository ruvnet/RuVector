//! The gateway side of the cross-crate end-to-end tests: the AS
//! ([`crate::e2e_as::EdgeAs`]) next to a gateway built from the compiled
//! trust root, whose requests go through the production authenticator,
//! route table and handlers over the in-process Durable Object backend.

use crate::api::ratelimit::mem::CountingLimiter;
use crate::api::ratelimit::{class_of, Budget, Class, Limiter};
use crate::api::{caller, data, guard};
use crate::auth::authenticate_full;
use crate::backend::mem::MemBackend;
use crate::config::GatewayConfig;
use crate::e2e_as::{At, EdgeAs};
use crate::rest::{parse, ApiReply};
use crate::routes::{auth_target, classify, Route};
use crate::testkit::{block_on, T0};
use crate::trust_root::TrustRoot;
use p256::ecdsa::VerifyingKey;
use ruvector_edge_auth::prm;
use ruvector_edge_auth::{AuthError, Jwk, KeySource};
use serde_json::{json, Value as Json};
use std::cell::Cell;
use worker::Method;

/// The gateway's JWKS view: exactly the AS's published keys, by thumbprint.
struct Jwks(Vec<VerifyingKey>);
impl KeySource for Jwks {
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError> {
        self.0
            .iter()
            .find(|k| Jwk::from_verifying_key(k).kid == kid)
            .copied()
            .ok_or(AuthError::UnknownKid)
    }
}

/// Rate budgets as the gateway charges them: enforced by a fixed-window
/// counter, or (`None`) admitting everything.
pub struct Budgets(pub Option<CountingLimiter>);
impl Limiter for Budgets {
    async fn allow(&self, budget: &Budget, key: &str) -> bool {
        match &self.0 {
            Some(l) => l.allow(budget, key).await,
            None => true,
        }
    }
}

/// The two Workers side by side.
pub struct World {
    pub edge_as: EdgeAs,
    pub cfg: GatewayConfig,
    pub b: MemBackend,
    /// The rate limiter `api::serve` charges before dispatch.
    pub limiter: Budgets,
    /// `Retry-After` of the last refused request.
    pub retry_after: Cell<Option<u32>>,
}

impl World {
    pub fn new(ceiling: &str) -> Self {
        World {
            edge_as: EdgeAs::new(T0, ceiling),
            cfg: GatewayConfig::from_trust_root(&TrustRoot::compiled(), "false").unwrap(),
            b: MemBackend::new(),
            limiter: Budgets(None),
            retry_after: Cell::new(None),
        }
    }

    /// [`World::new`] with the ADR-351 §10 rate budgets enforced.
    pub fn rate_limited(ceiling: &str) -> Self {
        World {
            limiter: Budgets(Some(CountingLimiter::default())),
            ..Self::new(ceiling)
        }
    }

    /// Start a new rate-limit window.
    pub fn new_window(&self) {
        if let Some(l) = &self.limiter.0 {
            l.reset();
        }
    }

    pub fn v1(&self) -> String {
        self.cfg.rest_resource.as_str().to_string()
    }

    pub fn mcp(&self) -> String {
        self.cfg.mcp_resource.as_str().to_string()
    }

    /// One request, as the router would serve it.
    pub fn send(
        &self,
        token: &str,
        m: Method,
        path: &str,
        body: Json,
        key: Option<&str>,
    ) -> ApiReply {
        self.send_with(token, m, path, body, key, None)
    }

    /// [`World::send`] with an `MCP-Protocol-Version` header (checked after
    /// authentication, before the body, as `api::serve` does).
    pub fn send_with(
        &self,
        token: &str,
        m: Method,
        path: &str,
        body: Json,
        key: Option<&str>,
        mcp_version: Option<&str>,
    ) -> ApiReply {
        let route = classify(&m, path);
        let (resource, surface) = auth_target(route, &self.cfg).expect("authenticated route");
        let header = format!("Bearer {token}");
        let auth = block_on(authenticate_full(
            Some(&header),
            &self.cfg,
            resource,
            surface,
            Jwks(self.edge_as.jwks()),
            || Jwks(Vec::new()),
            At(T0),
        ));
        let a = match auth {
            Ok(a) => a,
            Err(d) => {
                let (status, code) = d.code.status_and_code();
                return ApiReply {
                    status,
                    body: code.to_string(),
                    content_type: "application/problem+json",
                    www_authenticate: d.www_authenticate,
                };
            }
        };
        let c = caller(a);
        // Rate budgets, then the deny list, before anything else
        // (`api::serve`).
        let (class, res) = if route == Route::Mcp {
            (Class::Mcp, &self.cfg.mcp_resource)
        } else {
            (class_of(parse(&m, path).as_ref()), &self.cfg.rest_resource)
        };
        let md = prm::metadata_url(res);
        self.retry_after.set(None);
        if let Err(r) = block_on(guard(&self.b, &self.limiter, class, &c.ctx, T0, &md)) {
            self.retry_after.set(r.retry_after);
            return r.reply;
        }
        let bytes = if body.is_null() {
            Vec::new()
        } else {
            body.to_string().into_bytes()
        };
        if route == Route::Mcp {
            if !crate::mcp::protocol_header_ok(mcp_version) {
                return crate::mcp::bad_protocol_version();
            }
            let md = prm::metadata_url(&self.cfg.mcp_resource);
            return block_on(crate::mcp::handle(&self.b, &c.ctx, &bytes, T0, &md));
        }
        let api = parse(&m, path).expect(path);
        block_on(data(&self.b, &self.cfg, &api, &c, &bytes, key, T0))
    }

    pub fn ok(&self, token: &str, m: Method, path: &str, body: Json) -> Json {
        let r = self.send(token, m, path, body, None);
        assert!(
            (200..300).contains(&r.status),
            "{path}: {} {}",
            r.status,
            r.body
        );
        serde_json::from_str(&r.body).unwrap()
    }

    /// `POST /v1/ops` with `op_id` as the `Idempotency-Key`.
    pub fn op(&self, token: &str, op_id: &str, op: &str, args: Json) -> (u16, Json) {
        let tenant = self.ok(token, Method::Get, "/v1/me", Json::Null)["tenant_key"].clone();
        let env = json!({
            "v": 1, "op_id": op_id, "target": format!("{}/ops", self.v1()),
            "tenant_key": tenant, "op": op, "args": args,
        });
        let r = self.send(token, Method::Post, "/v1/ops", env, Some(op_id));
        (r.status, serde_json::from_str(&r.body).unwrap())
    }
}
