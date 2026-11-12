//! `AuthStore` Durable Object: owns the AS state in SQLite and runs the
//! stateful endpoints (register, authorize, callback, token, revoke) over
//! [`SqlPorts`]. One global instance (`AUTH_STORE_NAME`), so one-time
//! redemption and refresh rotation are serialised by the DO.

use crate::abuse::{ip_bucket, CLIENT_IP_HEADER, IP_HASH_SALT_SECRET};
use crate::config::AuthConfig;
use crate::endpoints::consent::CONSENT_PATH;
use crate::endpoints::{self, Ctx};
use crate::http::{body_cap, check_body_length, Reply};
use crate::platform::{WorkerClock, WorkerFetch, WorkerRng};
use crate::signer::{EnvSigner, SIGNING_KEY_SECRET};
use crate::sql_ports::SqlPorts;
use ruvector_edge_auth::{JwksCache, JwksCachePolicy};
use ruvector_edge_authz::metadata::paths;
use ruvector_edge_authz::{OAuthError, OAuthErrorCode};
use std::cell::OnceCell;
use worker::{
    durable_object, DurableObject, Env, Method, Request, Response, Result, SqlStorage, State,
};

type UpstreamKeys = JwksCache<WorkerFetch, WorkerClock>;

/// Single global instance holding clients, codes, upstream flows and
/// refresh families. The signer and the upstream JWKS cache live as long as
/// the DO instance, so the JWKS TTL spans requests.
#[durable_object]
pub struct AuthStore {
    ports: SqlPorts<SqlStorage>,
    env: Env,
    migrated: bool,
    signer: EnvSigner,
    upstream_keys: OnceCell<UpstreamKeys>,
}

impl DurableObject for AuthStore {
    fn new(state: State, env: Env) -> Self {
        let ports = SqlPorts::new(state.storage().sql());
        let migrated = ports.migrate().is_ok();
        let secret = env.secret(SIGNING_KEY_SECRET).ok().map(|s| s.to_string());
        AuthStore {
            ports,
            env,
            migrated,
            signer: EnvSigner::from_secret(secret.as_deref()),
            upstream_keys: OnceCell::new(),
        }
    }

    async fn fetch(&self, req: Request) -> Result<Response> {
        if !self.migrated {
            return unavailable("storage not ready");
        }
        let cfg = match AuthConfig::from_env(&self.env) {
            Ok(cfg) => cfg,
            Err(e) => return to_response(Reply::oauth_error(&e), false),
        };
        self.dispatch(req, &cfg).await
    }
}

impl AuthStore {
    async fn dispatch(&self, mut req: Request, cfg: &AuthConfig) -> Result<Response> {
        let ctx = Ctx {
            cfg,
            store: &self.ports,
            rng: &WorkerRng,
            clock: &WorkerClock,
            signer: &self.signer,
        };
        let path = req.path();
        let url = req.url()?;
        let query = url.query();
        let header = |name: &str| req.headers().get(name).ok().flatten();
        let content_type = header("Content-Type");
        let client_ip = header(CLIENT_IP_HEADER);
        match (req.method(), path.as_str()) {
            (Method::Get, paths::AUTHORIZE) => {
                to_response(endpoints::authorize::authorize(&ctx, query), false)
            }
            (Method::Post, CONSENT_PATH) => {
                if let Err(r) = check_body_length(CONSENT_PATH, header("Content-Length").as_deref())
                {
                    return to_response(r, false);
                }
                let cookies = header("Cookie");
                let body = req.bytes().await?;
                let reply = endpoints::authorize::consent_submit(
                    &ctx,
                    cookies.as_deref(),
                    content_type.as_deref(),
                    &body,
                );
                to_response(reply, false)
            }
            (Method::Get, paths::CALLBACK) => {
                let cookies = header("Cookie");
                let keys = self.upstream_keys.get_or_init(|| {
                    JwksCache::new(
                        WorkerFetch,
                        WorkerClock,
                        JwksCachePolicy::with_defaults(cfg.upstream.jwks_url.clone())
                            .with_accepted_kids(cfg.upstream_accepted_kids.clone()),
                    )
                });
                let reply = endpoints::callback::callback(
                    &ctx,
                    query,
                    cookies.as_deref(),
                    &WorkerFetch,
                    keys,
                )
                .await;
                to_response(reply, false)
            }
            (Method::Post, p @ (paths::REGISTER | paths::TOKEN | paths::REVOKE)) => {
                // Defence in depth: the Worker front applies the same gate.
                if let Err(r) = check_body_length(p, header("Content-Length").as_deref()) {
                    return to_response(r, true);
                }
                let body = req.bytes().await?;
                if body.len() > body_cap(p) {
                    let mut r = Reply::oauth_error(&OAuthError::new(
                        OAuthErrorCode::InvalidRequest,
                        "body too large",
                    ));
                    r.status = 413;
                    return to_response(r, true);
                }
                let reply = match p {
                    paths::REGISTER => {
                        let salt = self
                            .env
                            .secret(IP_HASH_SALT_SECRET)
                            .map(|s| s.to_string())
                            .unwrap_or_default();
                        let bucket = ip_bucket(client_ip.as_deref(), &salt);
                        endpoints::register::register(&ctx, &bucket, content_type.as_deref(), &body)
                    }
                    paths::TOKEN => endpoints::token::token(&ctx, content_type.as_deref(), &body),
                    _ => endpoints::token::revoke(&ctx, content_type.as_deref(), &body),
                };
                to_response(reply, true)
            }
            _ => Response::error("not found", 404),
        }
    }
}

fn unavailable(desc: &'static str) -> Result<Response> {
    to_response(
        Reply::oauth_error(&OAuthError::new(
            OAuthErrorCode::TemporarilyUnavailable,
            desc,
        )),
        false,
    )
}

/// Convert a [`Reply`] into a Workers response (`append`, so several
/// `Set-Cookie` headers survive), optionally with permissive CORS.
pub(crate) fn to_response(reply: Reply, cors: bool) -> Result<Response> {
    if let Some(line) = &reply.log {
        worker::console_log!("{line}");
    }
    let headers = worker::Headers::new();
    for (k, v) in &reply.headers {
        headers.append(k, v)?;
    }
    if cors {
        for (k, v) in crate::http::cors_headers() {
            headers.set(k, v)?;
        }
    }
    Ok(Response::from_bytes(reply.body)?
        .with_status(reply.status)
        .with_headers(headers))
}
