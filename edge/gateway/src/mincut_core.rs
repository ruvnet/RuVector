//! Min-cut planning shared by the gateway, `GraphStore` and `AnalyticsJob`
//! (ADR-351 §3 rv-mincut, §10): the Workers-Free inline profile, routing
//! (inline / job / 413), edge-list parsing and the public report.
//!
//! Workers Free gives an invocation 10 ms of CPU. The analytics crate's
//! `Profile::INLINE` (≈ 0.5 s native) is sized for a paid isolate, so the
//! request path uses [`EDGE_INLINE`] (5M work units ≈ 5 ms native, the
//! crate's ≲ 1 ns/unit calibration): e.g. a certified graph of ≈ 10k
//! vertices + edges, or Stoer–Wagner on a few hundred. Anything larger
//! that [`EDGE_JOB`] admits runs as an `AnalyticsJob` in a DO alarm;
//! beyond the job limits the answer is `413`.
//!
//! [`EDGE_JOB`] is the crate's `Profile::JOB` with its memory budget cut
//! from 100 MB ("the isolate to itself") to [`JOB_MEMORY_BYTES`]: an
//! `AnalyticsJob` shares its isolate with the resident shards of
//! `VectorShard` / `QuantShard` / `GraphStore` (56 MB of caps), so it may
//! only use the headroom stated in `shard_core` (memory budget). Work stays
//! at `Profile::JOB`: a CPU overrun is contained (the attempt is committed
//! before the solve, at most `MAX_ATTEMPTS`, then `413`), a memory overrun
//! is not (the isolate OOM evicts every co-resident tenant).

use ruvector_edge_analytics::cost::MEM_PER_EDGE;
use ruvector_edge_analytics::service::{plan, query, Budget, Profile};
use ruvector_edge_analytics::{
    AnalyticsError, CostEstimate, CutReport, EdgeRecord, GraphLimits, GraphUid, QueryMode,
    TenantGraph,
};
use ruvector_edge_store::{ErrorCode, OpError};
use serde::de::{self, Deserializer, IgnoredAny, SeqAccess, Visitor};
use serde_json::{json, Value as Json};

/// Request-path profile on Workers Free (ADR-351 §10 limits, 5 ms budget).
pub const EDGE_INLINE: Profile = Profile {
    limits: GraphLimits::INLINE,
    budget: Budget {
        max_work: 5_000_000,
        max_memory_bytes: 16 << 20,
    },
};

/// Peak memory one `AnalyticsJob` turn (admit or solve) may estimate:
/// the isolate headroom left after the resident caps (`shard_core`). The
/// 51k-edge K_320 acceptance job estimates 29.8 MB and fits.
pub const JOB_MEMORY_BYTES: u64 = 32 << 20;

/// The `AnalyticsJob` profile: `Profile::JOB` limits and work, memory
/// bounded to [`JOB_MEMORY_BYTES`].
pub const EDGE_JOB: Profile = Profile {
    limits: GraphLimits::JOB,
    budget: Budget {
        max_work: Profile::JOB.budget.max_work,
        max_memory_bytes: JOB_MEMORY_BYTES,
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

/// `413 budget_exceeded` if a job over `edges` edges certainly exceeds
/// [`JOB_MEMORY_BYTES`]: the estimate is at least `MEM_PER_EDGE` per edge,
/// so this refuses (O(1), before anything is stored) exactly what the
/// admit turn would refuse on the edge term alone.
pub fn job_memory_admissible(edges: u64) -> Result<(), OpError> {
    if edges.saturating_mul(MEM_PER_EDGE) > JOB_MEMORY_BYTES {
        return Err(OpError::new(
            ErrorCode::BudgetExceeded,
            "min-cut job exceeds the job memory budget",
        ));
    }
    Ok(())
}

/// Edges in a raw edge-list text: one `[` per edge plus the outer one
/// (numbers contain none; any other shape fails [`parse_edges`]). O(bytes),
/// no allocation.
pub fn raw_edge_count(raw: &str) -> u64 {
    (raw.bytes().filter(|&c| c == b'[').count() as u64).saturating_sub(1)
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
/// [`EDGE_INLINE`], else a job if [`EDGE_JOB`] admits it, else `413`.
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
        Err(AnalyticsError::LimitExceeded { .. }) if plan(&g, mode, &EDGE_JOB).is_ok() => {
            Ok(Plan::Job)
        }
        Err(e) => Err(aerr(e)),
    }
}

/// One `[u, v]` / `[u, v, w]` edge, deserialized in place (no JSON tree).
struct EdgeIn(u64, u64, f64);

impl<'de> serde::Deserialize<'de> for EdgeIn {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = EdgeIn;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("[u, v] or [u, v, weight]")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> Result<EdgeIn, A::Error> {
                let u = s
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let v = s
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let w = s.next_element()?.unwrap_or(1.0);
                if s.next_element::<IgnoredAny>()?.is_some() {
                    return Err(de::Error::invalid_length(4, &self));
                }
                Ok(EdgeIn(u, v, w))
            }
        }
        d.deserialize_seq(V)
    }
}

/// Parse an edge-list JSON text `[[u, v], [u, v, w], …]` (`u`, `v`
/// integers) straight into edges: 24 bytes per edge, never a
/// `serde_json::Value` tree (≈ 150 bytes per edge).
pub fn parse_edges(raw: &str) -> Result<Vec<(u64, u64, f64)>, OpError> {
    job_admissible(raw_edge_count(raw))?;
    let v: Vec<EdgeIn> = serde_json::from_str(raw)
        .map_err(|_| OpError::invalid("edges must be [[u, v] or [u, v, weight], …]"))?;
    Ok(v.into_iter().map(|EdgeIn(u, v, w)| (u, v, w)).collect())
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
