//! The per-tenant catalog instance of `GraphStore` ([`CATALOG`], ADR-351
//! §3 rv-graph / rv-mincut, M4): the tenant's graph names (at most
//! [`MAX_GRAPHS`]) and its live min-cut jobs (at most [`MAX_LIVE_JOBS`]).
//!
//! A job row expires at the same instant its `AnalyticsJob` deletes itself
//! (`mincut_job::JOB_TTL_MS` after submit), so the live-job count bounds
//! the job storage a tenant can hold without the job DO ever calling back:
//! expired rows are pruned on the next registration.
//!
//! [`CATALOG`]: crate::graph_wire::CATALOG

use crate::graph_persist as gp;
use crate::graph_wire::{name_ok, GraphOut};
use ruvector_edge_store::{ErrorCode, OpError, SqlStore, Value};
use serde_json::json;

/// Graphs per tenant.
pub const MAX_GRAPHS: usize = 20;
/// Unexpired min-cut jobs per tenant (each holds ≤ 1 MiB of input plus its
/// encoded snapshot until it expires).
pub const MAX_LIVE_JOBS: usize = 20;

const CAT_SCHEMA: &str =
    "CREATE TABLE IF NOT EXISTS gcat (name TEXT PRIMARY KEY, created_at INTEGER, sub TEXT)";
const CAT_ALL: &str = "SELECT name, created_at, sub FROM gcat ORDER BY name";
const CAT_PUT: &str = "INSERT INTO gcat (name, created_at, sub) VALUES (?, ?, ?)";
const CAT_DEL: &str = "DELETE FROM gcat WHERE name = ?";
const JOBS_SCHEMA: &str =
    "CREATE TABLE IF NOT EXISTS gjobs (job_id TEXT PRIMARY KEY, expires_ms INTEGER)";
const JOBS_PRUNE: &str = "DELETE FROM gjobs WHERE expires_ms <= ?";
const JOBS_ALL: &str = "SELECT job_id, expires_ms FROM gjobs";
const JOBS_PUT: &str = "INSERT INTO gjobs (job_id, expires_ms) VALUES (?, ?)";

fn names(store: &dyn SqlStore) -> Result<Vec<Vec<Value>>, OpError> {
    store.exec(CAT_SCHEMA, &[]).map_err(gp::io)?;
    store.query(CAT_ALL, &[]).map_err(gp::io)
}

fn has(rows: &[Vec<Value>], name: &str) -> bool {
    rows.iter()
        .any(|r| matches!(r.first(), Some(Value::Text(n)) if n == name))
}

/// Every graph.
pub fn list(store: &dyn SqlStore) -> Result<GraphOut, OpError> {
    let graphs = names(store)?
        .into_iter()
        .filter_map(|r| match (r.first(), r.get(1)) {
            (Some(Value::Text(n)), Some(Value::Int(t))) => {
                Some(json!({ "name": n, "created_at": t }))
            }
            _ => None,
        })
        .collect();
    Ok(GraphOut::Graphs { graphs })
}

/// Record a graph (`created: false` if it exists; `413` at the limit).
pub fn add(store: &dyn SqlStore, name: &str, sub: &str, now: u64) -> Result<GraphOut, OpError> {
    if !name_ok(name) {
        return Err(OpError::invalid("graph name"));
    }
    let rows = names(store)?;
    if has(&rows, name) {
        return Ok(GraphOut::Added { created: false });
    }
    if rows.len() >= MAX_GRAPHS {
        return Err(OpError::new(ErrorCode::QuotaExceeded, "graph limit"));
    }
    let params = [name.into(), Value::Int(now as i64), sub.into()];
    store.exec(CAT_PUT, &params).map_err(gp::io)?;
    Ok(GraphOut::Added { created: true })
}

/// Forget a graph (`existed: false` if it was not listed).
pub fn remove(store: &dyn SqlStore, name: &str) -> Result<GraphOut, OpError> {
    let existed = has(&names(store)?, name);
    if existed {
        store.exec(CAT_DEL, &[name.into()]).map_err(gp::io)?;
    }
    Ok(GraphOut::Removed { existed })
}

/// Register a job until `expires_ms`, after pruning the expired ones;
/// `413 quota_exceeded` when [`MAX_LIVE_JOBS`] are live at `now_ms`.
pub fn job_add(
    store: &dyn SqlStore,
    job_id: &str,
    now_ms: u64,
    expires_ms: u64,
) -> Result<GraphOut, OpError> {
    store.exec(JOBS_SCHEMA, &[]).map_err(gp::io)?;
    let now = Value::Int(i64::try_from(now_ms).unwrap_or(i64::MAX));
    store.exec(JOBS_PRUNE, &[now]).map_err(gp::io)?;
    let live = store.query(JOBS_ALL, &[]).map_err(gp::io)?;
    if live.len() >= MAX_LIVE_JOBS {
        return Err(OpError::new(
            ErrorCode::QuotaExceeded,
            "live min-cut job limit; retry after a job expires",
        ));
    }
    let exp = Value::Int(i64::try_from(expires_ms).unwrap_or(i64::MAX));
    store
        .exec(JOBS_PUT, &[job_id.into(), exp])
        .map_err(gp::io)?;
    Ok(GraphOut::Added { created: true })
}
