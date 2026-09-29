//! `POST /v1/mincut` and `GET /v1/mincut/jobs/{id}` (ADR-351 §3 rv-mincut:
//! compute → read; job → write).
//!
//! `POST /v1/mincut` body: exactly one of `"graph": "<name>"` (the graph's
//! edges; weight = the `weight` property) or `"edges": [[u, v], [u, v, w],
//! …]` (integer vertices), plus `"mode": "exact" | "approximate"` and
//! `"epsilon"` for approximate. Answered inline (`200`, read + viewer)
//! when it fits the Workers-Free inline budget; otherwise queued as an
//! `AnalyticsJob` (`202` + `job_id`, which needs write + editor); beyond
//! the job limits `413`. Approximate requests are answered exactly and
//! report `mode: exact`.

use crate::backend::Backend;
use crate::graph_routes::{admit, authorize, graph_call, job_call, Need};
use crate::graph_wire::{job_graph_uid, GraphCall, GraphOut, JobCall};
use crate::mincut_core::{self as mc, Plan};
use ruvector_edge_analytics::QueryMode;
use ruvector_edge_store::{CallerContext, OpError};
use ruvector_edge_tenancy::EntropySource;
use serde_json::value::RawValue;
use serde_json::{json, Value as Json};

/// A fresh job id: `mc_` + 16 hex chars of entropy.
pub fn new_job_id(entropy: &dyn EntropySource) -> Result<String, OpError> {
    let mut b = [0u8; 8];
    entropy
        .fill(&mut b)
        .map_err(|_| OpError::new(ruvector_edge_store::ErrorCode::ServerError, "entropy"))?;
    Ok(format!(
        "mc_{}",
        b.iter().map(|x| format!("{x:02x}")).collect::<String>()
    ))
}

/// `POST /v1/mincut` body. `edges` stays raw text: a job-sized list is
/// counted and handed to the job verbatim (validated in its admit turn),
/// never materialised as a `serde_json::Value` tree on the request path.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Body {
    graph: Option<Json>,
    edges: Option<Box<RawValue>>,
    mode: Option<Json>,
    epsilon: Option<Json>,
}

/// Parse `mode` / `epsilon`.
pub fn mode_of(mode: Option<&Json>, epsilon: Option<&Json>) -> Result<QueryMode, OpError> {
    let mode = match mode {
        None => None,
        Some(Json::String(s)) => Some(s.as_str()),
        Some(_) => return Err(OpError::invalid("mode must be a string")),
    };
    let eps = match epsilon {
        None => None,
        Some(v) => Some(
            v.as_f64()
                .ok_or(OpError::invalid("epsilon must be a number"))?,
        ),
    };
    mc::parse_mode(mode, eps)
}

/// Edges in a raw edge-list text: one `[` per edge plus the outer one
/// (numbers contain none; any other shape fails `parse_edges`). O(bytes),
/// no allocation.
pub fn raw_edge_count(raw: &str) -> u64 {
    (raw.bytes().filter(|&c| c == b'[').count() as u64).saturating_sub(1)
}

async fn submit<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    input: (String, Option<Vec<String>>, Option<String>, u64),
    mode: QueryMode,
    now: u64,
    entropy: &dyn EntropySource,
) -> Result<(u16, Json), OpError> {
    // Compute is read; a job is write (ADR-351 §3).
    authorize(b, ctx, Need::Write).await?;
    let job_id = new_job_id(entropy)?;
    let (edges, labels, graph, revision) = input;
    let call = JobCall::Submit {
        mode,
        edges,
        labels,
        graph,
        revision,
        now_ms: now.saturating_mul(1000),
    };
    let mut view = job_call(b, ctx, &job_id, call).await?;
    view["status_url"] = json!(format!("/v1/mincut/jobs/{job_id}"));
    Ok((202, view))
}

fn no_job() -> OpError {
    OpError::new(
        ruvector_edge_store::ErrorCode::BudgetExceeded,
        "min-cut budget exceeded; submit it as a job (POST /v1/mincut)",
    )
}

/// `POST /v1/mincut`; `entropy: None` answers inline only (the MCP
/// `mincut` tool), refusing a job-sized query with `413`.
pub async fn post<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    body: &[u8],
    now: u64,
    entropy: Option<&dyn EntropySource>,
) -> Result<(u16, Json), OpError> {
    let m: Body = if body.iter().all(u8::is_ascii_whitespace) {
        Body {
            graph: None,
            edges: None,
            mode: None,
            epsilon: None,
        }
    } else {
        serde_json::from_slice(body)
            .map_err(|_| OpError::invalid("body must be a JSON object with known fields"))?
    };
    let mode = mode_of(m.mode.as_ref(), m.epsilon.as_ref())?;
    match (&m.graph, &m.edges) {
        (Some(Json::String(g)), None) => {
            admit(b, ctx, Need::Read, now).await?;
            match graph_call(b, ctx, g, GraphCall::Mincut { mode }).await? {
                GraphOut::Cut { report } => Ok((200, report)),
                GraphOut::JobInput {
                    edges,
                    labels,
                    revision,
                } => {
                    let input = (edges, Some(labels), Some(g.clone()), revision);
                    submit(b, ctx, input, mode, now, entropy.ok_or_else(no_job)?).await
                }
                _ => Err(crate::service::unexpected()),
            }
        }
        (None, Some(raw)) => {
            let raw = raw.get();
            let count = raw_edge_count(raw);
            mc::job_admissible(count)?;
            // Only an inline-sized list is parsed on the request path.
            let inline = if count <= mc::INLINE_MAX_EDGES {
                let v: Json = serde_json::from_str(raw).map_err(|_| OpError::invalid("edges"))?;
                Some(mc::parse_edges(&v)?)
            } else {
                None
            };
            admit(b, ctx, Need::Read, now).await?;
            if let Some(edges) = inline {
                let uid = job_graph_uid(ctx.tenant_key().as_str());
                if let Plan::Inline(r) = mc::route(uid, 0, &edges, &mode)? {
                    return Ok((200, mc::report_json(&r, &mode, None)));
                }
            }
            let entropy = entropy.ok_or_else(no_job)?;
            // Verbatim: the job's admit turn parses and validates it.
            let input = (raw.to_string(), None, None, 0);
            submit(b, ctx, input, mode, now, entropy).await
        }
        _ => Err(OpError::invalid("exactly one of graph (string) or edges")),
    }
}

/// `GET /v1/mincut/jobs/{id}`.
pub async fn get<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    id: &str,
    now: u64,
) -> Result<Json, OpError> {
    admit(b, ctx, Need::Read, now).await?;
    job_call(
        b,
        ctx,
        id,
        JobCall::Get {
            now_ms: now.saturating_mul(1000),
        },
    )
    .await
}
