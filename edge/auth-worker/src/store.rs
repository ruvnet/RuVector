//! `AuthStore` Durable Object: owns the AS state in SQLite and runs the
//! stateful endpoints against `ruvector-edge-authz` with [`SqlPorts`].

use crate::config::AuthConfig;
use crate::platform::{EnvSigner, WorkerClock, WorkerRng};
use crate::sql_ports::SqlPorts;
use ruvector_edge_authz::metadata::paths;
use ruvector_edge_authz::{OAuthError, OAuthErrorCode};
use worker::{durable_object, DurableObject, Env, Method, Request, Response, Result, State};

/// Single global instance (`AUTH_STORE_NAME`) holding clients, codes,
/// upstream flows and refresh families.
#[durable_object]
pub struct AuthStore {
    ports: SqlPorts,
    env: Env,
    migrated: bool,
}

impl DurableObject for AuthStore {
    fn new(state: State, env: Env) -> Self {
        let ports = SqlPorts::new(state.storage().sql());
        let migrated = ports.migrate().is_ok();
        AuthStore {
            ports,
            env,
            migrated,
        }
    }

    /// Dispatch the stateful endpoints.
    ///
    /// Contract per endpoint (implemented in later stages):
    /// - `POST /register`: JSON body <= 16 KiB -> `client::validate_registration`
    ///   -> `ClientStore::insert_client` -> 201 RFC 7591 response.
    /// - `GET /authorize`: `authorize::validate_authorization` ->
    ///   `federation::begin_upstream` -> 302 to auth.cognitum.one.
    /// - `GET /callback`: `federation::complete_upstream` -> upstream code
    ///   exchange + JWKS verification -> `federation::identity_from_upstream`
    ///   -> `code::issue_code` -> 302 `authorize::success_redirect`.
    /// - `POST /token`: `token::TokenRequest::from_form` -> `code::redeem_code`
    ///   or `refresh::rotate_refresh` -> `token::mint_access_token`.
    /// - `POST /revoke`: `revoke::revoke` -> 200.
    async fn fetch(&self, req: Request) -> Result<Response> {
        if !self.migrated {
            return crate::oauth_error(&OAuthError::new(
                OAuthErrorCode::TemporarilyUnavailable,
                "storage not ready",
            ));
        }
        let cfg = match AuthConfig::from_env(&self.env) {
            Ok(cfg) => cfg,
            Err(e) => return crate::oauth_error(&e),
        };
        let _ports = (
            &self.ports,
            WorkerClock,
            WorkerRng,
            EnvSigner::from_env(&self.env),
            &cfg.resources,
            &cfg.upstream,
        );
        match (req.method(), req.path().as_str()) {
            (Method::Post, paths::REGISTER)
            | (Method::Get, paths::AUTHORIZE)
            | (Method::Get, paths::CALLBACK)
            | (Method::Post, paths::TOKEN)
            | (Method::Post, paths::REVOKE) => {
                crate::oauth_error(&OAuthError::not_implemented("endpoint not implemented"))
            }
            _ => Response::error("not found", 404),
        }
    }
}
