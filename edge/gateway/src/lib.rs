//! rv-gateway Worker (ADR-351 M1 router).
//!
//! Thin workers-rs glue over the pure cores: `ruvector-edge-auth` verifies
//! bearer tokens (edge-AS-issued, `aud` = exact resource URL; upstream
//! first-party tokens only behind `UPSTREAM_FIRST_PARTY=true`) and
//! `ruvector-edge-tenancy` derives the tenant. Routes (M1):
//!
//! - `GET /v1/health` -> `{"ok":true}` (anonymous, no version string)
//! - `GET /.well-known/oauth-protected-resource/v1` and `…/v1/mcp` -> RFC 9728
//!   (the bare well-known path is 404, ADR-351 §5.7)
//! - `GET /v1/me` -> tenant identity plus the ledger role, claim state and
//!   effective capabilities (any valid token; edge keys fetched **only**
//!   through the `EDGE_AUTH` service binding, cached isolate-wide)
//! - `POST /v1/claim`, `/v1/usage`, `/v1/collections*` (REST, `rest`),
//!   `POST /v1/ops` (§16.3 envelope, `aud = …/v1` only, `ops`) and
//!   `POST /v1/mcp` (JSON-RPC, MCP resource and surface, `mcp`) all run the
//!   same executors (`service`, `vectors`) over two Durable Object classes,
//!   `TenantLedger` and `VectorShard` (`durable`, cores in `ledger_core` /
//!   `shard_core`, wire in `wire`)
//! - any other `/v1` path -> 401 without a token, otherwise 404
//! - `OPTIONS` on any `/v1` path -> CORS preflight (never authenticated);
//!   every response carries CORS and exposes `WWW-Authenticate`
//!
//! Every 401 carries `WWW-Authenticate: Bearer resource_metadata="..."`.

#![forbid(unsafe_code)]

mod api;
mod auth;
mod backend;
mod config;
mod durable;
mod idem;
mod keys;
mod ledger_core;
// M3 (ADR-351 §15): snapshots, restore, export, uploads, import jobs,
// embeddings, audit shipping; side channel `/m3` into the same DOs.
mod audit;
mod audit_http;
mod embed;
mod export;
mod ingest;
mod jobs;
mod m3_api;
mod m3_audit_ledger;
mod m3_ctx;
mod m3_ledger;
mod m3_ports;
mod m3_shard;
mod m3_transport;
mod m3_wire;
mod mcp;
mod ops;
mod platform;
mod queue_consumer;
mod respond;
mod rest;
mod restore;
mod routes;
mod service;
mod shard_core;
mod snapshots;
mod trust_root;
mod uploads;
mod vectors;
mod wire;

use worker::{event, Context, Env, Request, Response, Result};

/// Worker entry point.
#[event(fetch)]
async fn fetch(req: Request, env: Env, ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    let cfg = match config::GatewayConfig::from_env(&env) {
        Ok(cfg) => cfg,
        Err(config::ConfigError::TrustRootMismatch(_)) => {
            return respond::problem(ruvector_edge_tenancy::ProblemCode::TrustRootMismatch, None)
        }
        Err(config::ConfigError::Invalid(_)) => {
            return respond::problem(ruvector_edge_tenancy::ProblemCode::ServerError, None)
        }
    };
    routes::handle(req, &env, &cfg, &ctx).await
}

#[cfg(test)]
mod e2e_as;
#[cfg(test)]
mod e2e_m1_tests;
#[cfg(test)]
mod e2e_tests;
#[cfg(test)]
mod e2e_world;
#[cfg(test)]
mod m3_embed_audit_tests;
#[cfg(test)]
mod m3_import_tests;
#[cfg(test)]
mod m3_mem;
#[cfg(test)]
mod m3_restore_tests;
#[cfg(test)]
mod m3_review_tests;
#[cfg(test)]
mod m3_snapshot_tests;
#[cfg(test)]
mod mcp_tests;
#[cfg(test)]
mod ops_tests;
#[cfg(test)]
mod rest_tests;
#[cfg(test)]
mod service_tests;
#[cfg(test)]
mod testkit;
