//! Min-cut planning shared by the gateway, `GraphStore` and `AnalyticsJob`
//! (ADR-351 §3 rv-mincut, §10): the Workers-Free inline profile, routing
//! (inline / job / 413), edge-list parsing and the public report.
//!
//! Workers Free gives an invocation 10 ms of CPU. The analytics crate's
//! `Profile::INLINE` (≈ 0.5 s native) is sized for a paid isolate, so the
//! request path uses [`EDGE_INLINE`] (5M work units ≈ 5 ms native, the
//! crate's ≲ 1 ns/unit calibration): e.g. a certified graph of ≈ 10k
//! vertices + edges, or Stoer–Wagner on a few hundred. Anything larger
//! that `Profile::JOB` admits runs as an `AnalyticsJob` in a DO alarm;
//! beyond the job limits the answer is `413`.

use ruvector_edge_analytics::service::{plan, query, Budget, Profile};
use ruvector_edge_analytics::{
    AnalyticsError, CostEstimate, CutReport, EdgeRecord, GraphLimits, GraphUid, QueryMode,
    TenantGraph,
};
use ruvector_edge_store::{ErrorCode, OpError};
use serde_json::{json, Value as Json};

/// Request-path profile on Workers Free (ADR-351 §10 limits, 5 ms budget).
pub const EDGE_INLINE: Profile = Profile {
    limits: GraphLimits::INLINE,
    budget: Budget {
        max_work: 5_000_000,
        max_memory_bytes: 16 << 20,
    },
};

/// Edges above which a request skips building the graph inline (the
/// linear rebuild alone would exceed [`EDGE_INLINE`]: 500 units/item).
pub const INLINE_MAX_EDGES: u64 = EDGE_INLINE.budget.max_work / 500;

/// Where a query runs.
pub enum Plan {
    /// Answered on the request path.
    Inline(CutReport),
    /// Too large inline, admissible as a job.
    Job,
}

/// Map an analytics error (limits and budgets are `413`).
pub fn aerr(e: AnalyticsError) -> OpError {
    OpError::from(e)
}

/// `413` if even a job cannot take `edges` edges (O(1), before parsing).
pub fn job_admissible(edges: u64) -> Result<(), OpError> {
    GraphLimits::JOB.check(0, edges).map_err(aerr)
}

/// Parse `mode` (`exact` | `approximate` + `epsilon`).
pub fn parse_mode(mode: Option<&str>, epsilon: Option<f64>) -> Result<QueryMode, OpError> {
    let m = match mode.unwrap_or("exact") {
        "exact" if epsilon.is_none() => QueryMode::Exact,
        "approximate" => QueryMode::Approximate {
            epsilon: epsilon.ok_or(OpError::invalid("approximate needs epsilon"))?,
        },
        _ => return Err(OpError::invalid("mode must be exact or approximate")),
    };
    if let QueryMode::Approximate { epsilon } = m {
        if !(epsilon.is_finite() && epsilon > 0.0 && epsilon <= 1.0) {
            return Err(OpError::invalid("epsilon must be in (0, 1]"));
        }
    }
    Ok(m)
}

/// Build (validate + canonicalise) and route: inline under
/// [`EDGE_INLINE`], else a job if `Profile::JOB` admits it, else `413`.
pub fn route(
    uid: GraphUid,
    revision: u64,
    edges: &[(u64, u64, f64)],
    mode: &QueryMode,
) -> Result<Plan, OpError> {
    job_admissible(edges.len() as u64)?;
    if edges.len() as u64 > INLINE_MAX_EDGES {
        return Ok(Plan::Job);
    }
    let g = TenantGraph::from_edges(uid, revision, edges, &GraphLimits::JOB).map_err(aerr)?;
    match plan(&g, mode, &EDGE_INLINE) {
        Ok(_) => Ok(Plan::Inline(query(&g, mode, &EDGE_INLINE).map_err(aerr)?)),
        Err(AnalyticsError::BudgetExceeded {
            job_eligible: true, ..
        }) => Ok(Plan::Job),
        Err(AnalyticsError::LimitExceeded { .. }) if plan(&g, mode, &Profile::JOB).is_ok() => {
            Ok(Plan::Job)
        }
        Err(e) => Err(aerr(e)),
    }
}

/// Parse an edge-list JSON `[[u, v], [u, v, w], …]` (`u`, `v` integers).
pub fn parse_edges(v: &Json) -> Result<Vec<(u64, u64, f64)>, OpError> {
    let arr = v
        .as_array()
        .ok_or(OpError::invalid("edges must be an array"))?;
    job_admissible(arr.len() as u64)?;
    let bad = || OpError::invalid("edge must be [u, v] or [u, v, weight]");
    arr.iter()
        .map(|e| {
            let t = e.as_array().ok_or_else(bad)?;
            let id = |i: usize| t.get(i).and_then(Json::as_u64).ok_or_else(bad);
            let w = match t.len() {
                2 => 1.0,
                3 => t[2].as_f64().ok_or_else(bad)?,
                _ => return Err(bad()),
            };
            Ok((id(0)?, id(1)?, w))
        })
        .collect()
}

/// Compact edge-list JSON (no quotes: embedded verbatim in job bodies).
pub fn edges_json(edges: &[(u64, u64, f64)]) -> String {
    let parts: Vec<String> = edges
        .iter()
        .map(|(u, v, w)| format!("[{u},{v},{}]", json!(w)))
        .collect();
    format!("[{}]", parts.join(","))
}

fn mode_json(m: &QueryMode) -> Json {
    match m {
        QueryMode::Exact => json!("exact"),
        QueryMode::Approximate { .. } => json!("approximate"),
    }
}

fn estimate_json(e: &CostEstimate) -> Json {
    serde_json::to_value(e).unwrap_or(Json::Null)
}

/// Public report: always `mode: exact` (approximate requests are answered
/// exactly, which meets any `(1 + ε)` bound); partitions and cut edges in
/// `labels` when the vertices are graph nodes.
pub fn report_json(r: &CutReport, requested: &QueryMode, labels: Option<&[String]>) -> Json {
    let name = |x: u64| -> Json {
        match labels.and_then(|l| l.get(x as usize)) {
            Some(s) => json!(s),
            None => json!(x),
        }
    };
    let side = |s: &[u64]| Json::Array(s.iter().map(|x| name(*x)).collect());
    let edge = |e: &EdgeRecord| json!({ "u": name(e.u), "v": name(e.v), "w": e.w });
    let mut out = json!({
        "mode": mode_json(&r.mode),
        "requested_mode": mode_json(requested),
        "revision": r.revision,
        "value": r.value,
        "partition": r.partition.as_ref().map(|(s, t)| json!([side(s), side(t)])),
        "cut_edges": r.cut_edges.as_ref().map(|c| Json::Array(c.iter().map(edge).collect())),
        "estimate": estimate_json(&r.estimate),
        "digest": r.digest_hex(),
    });
    if let QueryMode::Approximate { epsilon } = requested {
        out["epsilon"] = json!(epsilon);
    }
    out
}

/// Error object for a failed job / tool result.
pub fn error_json(code: ErrorCode, detail: &str) -> Json {
    json!({ "code": code.as_str(), "status": code.status(), "detail": detail })
}
