//! Rate limiting, ADR-351 §10 layer 2 (a shield, not a hard limit: only
//! the ledger's admission is consistent).
//!
//! Every authenticated request is charged to two keys of its class — the
//! user `user:{tenant_key}:{sub}` first, then the tenant
//! `org:{tenant_key}` — each against its own Workers Rate Limiting binding
//! (a binding's limit and period are fixed in `wrangler.toml`, so one
//! binding per class × level). Classes have separate budgets: `read`,
//! `write`, `ops` (`/v1/ops`) and `mcp` (`/v1/mcp`). A mutating op on
//! `/v1/ops` or `/v1/mcp` is charged to `write` as well ([`op_class`]),
//! and a query pays one `read` token per shard it fans out to. Over budget
//! → `429 rate_limited` with `Retry-After` = the period.
//!
//! The binding is per Cloudflare location, so the effective limit is per
//! location **[U]** (ADR §10 "Locality"). A missing binding or a limiter
//! error fails **open** (the shield must not become an outage) and is
//! logged (`wrangler tail`); the shipped `wrangler.toml` is tested to
//! declare every binding. Live 429 behaviour is unverified until G1.

use crate::rest::{ApiReply, ApiRoute};
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};

/// Budget class of a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Reads: `/v1/me`, usage, list/get, query, fetch, member list.
    Read,
    /// Writes: claim, create/drop, upsert/delete, members, deny.
    Write,
    /// `POST /v1/ops` (adapter callers).
    Ops,
    /// `POST /v1/mcp`.
    Mcp,
}

/// One Rate Limiting binding (`[[ratelimits]]` in `wrangler.toml`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Binding name.
    pub binding: &'static str,
    /// Requests per period.
    pub limit: u32,
    /// Seconds (Workers allows 10 or 60).
    pub period_s: u32,
}

const fn b(binding: &'static str, limit: u32) -> Budget {
    Budget {
        binding,
        limit,
        period_s: PERIOD_S,
    }
}

/// Window of every budget (seconds); also the `Retry-After` value.
pub const PERIOD_S: u32 = 10;

/// `(class, per-user, per-tenant)` budgets. §10: tenant 100 / 10 s for
/// reads, lower for writes; user 20 / 10 s.
pub const BUDGETS: [(Class, Budget, Budget); 4] = [
    (Class::Read, b("RL_READ_USER", 20), b("RL_READ_ORG", 100)),
    (Class::Write, b("RL_WRITE_USER", 10), b("RL_WRITE_ORG", 50)),
    (Class::Ops, b("RL_OPS_USER", 20), b("RL_OPS_ORG", 100)),
    (Class::Mcp, b("RL_MCP_USER", 20), b("RL_MCP_ORG", 100)),
];

/// The class of a REST route (`None`: unknown path, charged as a read so
/// 404 probing is throttled too).
pub fn class_of(route: Option<&ApiRoute>) -> Class {
    match route {
        Some(ApiRoute::Ops) => Class::Ops,
        Some(
            ApiRoute::Claim
            | ApiRoute::Create
            | ApiRoute::Upsert(_)
            | ApiRoute::Delete(_)
            | ApiRoute::Drop(_)
            | ApiRoute::Invite
            | ApiRoute::RemoveMember(_)
            | ApiRoute::Deny,
        ) => Class::Write,
        Some(
            ApiRoute::Me
            | ApiRoute::Usage
            | ApiRoute::List
            | ApiRoute::Get(_)
            | ApiRoute::Query(_)
            | ApiRoute::Fetch(_)
            | ApiRoute::Members,
        )
        | None => Class::Read,
    }
}

/// Ops that write, by their `/v1/ops` `op` / MCP tool name.
pub const MUTATING_OPS: [&str; 4] = [
    "collection_create",
    "vector_upsert",
    "vector_delete",
    "tenant_claim",
];

/// The class a `/v1/ops` or `/v1/mcp` body is charged **in addition** to
/// its surface class: [`Class::Write`] for a mutating op (`/v1/ops` `op`,
/// MCP `tools/call` tool name), so the §10 "lower for writes" budget holds
/// on every surface, not only REST. `None` for anything else (reads, other
/// JSON-RPC methods, malformed bodies — those fail in the handler).
pub fn op_class(surface: Class, body: &[u8]) -> Option<Class> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let name = match surface {
        Class::Ops => v.get("op")?.as_str()?,
        Class::Mcp if v.get("method")?.as_str()? == "tools/call" => {
            v.get("params")?.get("name")?.as_str()?
        }
        _ => return None,
    };
    MUTATING_OPS.contains(&name).then_some(Class::Write)
}

/// Charge `extra` more tokens of `class` (e.g. the §10 "a query costs
/// `shards_queried` tokens" beyond the one the request already paid);
/// `Err(retry_after_s)` as soon as a budget is exhausted.
pub async fn admit_n<L: Limiter>(
    l: &L,
    class: Class,
    ctx: &CallerContext,
    extra: u32,
) -> Result<(), u32> {
    for _ in 0..extra {
        admit(l, class, ctx).await?;
    }
    Ok(())
}

/// `(user, tenant)` budgets of `class`.
pub fn budgets(class: Class) -> (Budget, Budget) {
    BUDGETS
        .iter()
        .find(|(c, _, _)| *c == class)
        .map(|(_, u, o)| (*u, *o))
        .unwrap_or((BUDGETS[0].1, BUDGETS[0].2))
}

/// `user:{tenant_key}:{sub}`.
pub fn user_key(ctx: &CallerContext) -> String {
    format!("user:{}:{}", ctx.tenant_key().as_str(), ctx.sub())
}

/// `org:{tenant_key}`.
pub fn org_key(ctx: &CallerContext) -> String {
    format!("org:{}", ctx.tenant_key().as_str())
}

/// A rate limiter: `true` = within budget (and charged).
#[allow(async_fn_in_trait)] // Workers futures are !Send.
pub trait Limiter {
    /// Charge one request to `key` on `budget`.
    async fn allow(&self, budget: &Budget, key: &str) -> bool;
}

/// Charge the request; `Err(retry_after_s)` when over budget. The user key
/// is charged first, so one user's flood never spends the tenant budget.
pub async fn admit<L: Limiter>(l: &L, class: Class, ctx: &CallerContext) -> Result<(), u32> {
    let (user, org) = budgets(class);
    if !l.allow(&user, &user_key(ctx)).await {
        return Err(user.period_s);
    }
    if !l.allow(&org, &org_key(ctx)).await {
        return Err(org.period_s);
    }
    Ok(())
}

/// The `429 rate_limited` problem (the caller adds `Retry-After`).
pub fn limited(metadata_url: &str) -> ApiReply {
    let e = OpError::new(ErrorCode::RateLimited, "rate limited");
    ApiReply::problem(&e, metadata_url)
}

/// Workers Rate Limiting bindings.
pub struct EnvLimiter<'a>(pub &'a worker::Env);

impl Limiter for EnvLimiter<'_> {
    async fn allow(&self, budget: &Budget, key: &str) -> bool {
        let rl = match self.0.rate_limiter(budget.binding) {
            Ok(rl) => rl,
            Err(e) => {
                worker::console_warn!("rate limit binding {} unusable: {e}", budget.binding);
                return true;
            }
        };
        match rl.limit(key.to_string()).await {
            Ok(o) => o.success,
            Err(e) => {
                worker::console_warn!("rate limit {} failed: {e}", budget.binding);
                true
            }
        }
    }
}

/// Fixed-window counters (native tests).
#[cfg(test)]
pub mod mem {
    use super::{Budget, Limiter};
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    /// Counts per `(binding, key)`; `reset` starts a new window.
    #[derive(Default)]
    pub struct CountingLimiter {
        /// Charged requests.
        pub seen: RefCell<BTreeMap<(String, String), u32>>,
    }

    impl CountingLimiter {
        /// New window.
        pub fn reset(&self) {
            self.seen.borrow_mut().clear();
        }
    }

    impl Limiter for CountingLimiter {
        async fn allow(&self, budget: &Budget, key: &str) -> bool {
            let mut seen = self.seen.borrow_mut();
            let n = seen
                .entry((budget.binding.to_string(), key.to_string()))
                .or_default();
            *n += 1;
            *n <= budget.limit
        }
    }
}
