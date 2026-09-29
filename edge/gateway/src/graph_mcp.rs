//! MCP tools of rv-graph / rv-mincut (ADR-351 §7.2 rv-mcp, M4), served by
//! `mcp::tools_call` for names outside the M1 table:
//!
//! - `graph_list` (read): the tenant's graphs;
//! - `graph_query` (read): a read-only Cypher query (`MATCH` / `RETURN` /
//!   `WITH`); a mutating one is refused (`invalid_request`, use
//!   `graph_mutate`);
//! - `graph_mutate` (write + editor): create a graph, run a mutating
//!   Cypher query, or bulk-insert edges;
//! - `mincut` (read): an inline min-cut; a job-sized one is `413` (submit
//!   it with `POST /v1/mincut`).
//!
//! As for the M1 tools, a missing scope is HTTP 403 with the step-up
//! challenge; every other failure is an `isError` tool result.

use crate::backend::Backend;
use crate::graph_cypher as gc;
use crate::graph_routes::{self as gr, Need};
use crate::graph_wire::{GraphCall, GraphOut};
use crate::mcp::{rpc_error, rpc_result, tool_error};
use crate::rest::{step_up, ApiReply};
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};
use serde_json::{json, Map, Value as Json};

/// Tool names this module serves.
pub const NAMES: [&str; 4] = ["graph_list", "graph_query", "graph_mutate", "mincut"];

/// `tools/list` entries.
pub fn tools() -> Vec<Json> {
    let g = json!({ "type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{0,62}$" });
    let pairs = |max: u64| json!({ "type": "array", "maxItems": max, "items": { "type": "array", "minItems": 2, "maxItems": 3 } });
    let edges = pairs(crate::graph_store::MAX_BULK_EDGES as u64);
    let cut_edges = pairs(crate::mincut_core::INLINE_MAX_EDGES);
    let schema = |props: Json, req: &[&str]| json!({ "type": "object", "properties": props, "required": req, "additionalProperties": false });
    let tool = |name: &str, desc: &str, input: Json, read_only: bool| {
        json!({
            "name": name, "description": desc, "inputSchema": input,
            "annotations": {
                "readOnlyHint": read_only, "destructiveHint": !read_only,
                "idempotentHint": read_only, "openWorldHint": false,
            },
        })
    };
    vec![
        tool("graph_list", "List the tenant's property graphs.", schema(json!({}), &[]), true),
        tool(
            "graph_query",
            "Run a read-only Cypher query (MATCH / RETURN / WITH) on a graph. \
             Budget: 10k steps, 1k rows.",
            schema(json!({ "graph": g, "query": { "type": "string", "maxLength": 4096 } }), &["graph", "query"]),
            true,
        ),
        tool(
            "graph_mutate",
            "Create a graph (create: true), run a mutating Cypher query (CREATE / SET / DELETE), \
             or bulk-insert edges [[from, to, weight?], ...]. Needs ruvector:write and editor.",
            schema(
                json!({
                    "graph": g, "create": { "type": "boolean" },
                    "query": { "type": "string", "maxLength": 4096 },
                    "edges": edges, "type": { "type": "string" }, "label": { "type": "string" },
                    "dry_run": { "type": "boolean", "description": "Validate and authorize without writing." },
                }),
                &["graph"],
            ),
            false,
        ),
        tool(
            "mincut",
            "Exact global minimum cut of a graph (graph) or an edge list (edges: [[u, v, w?], ...]); \
             mode approximate (epsilon) is answered exactly. Inline only.",
            schema(
                json!({
                    "graph": g, "edges": cut_edges,
                    "mode": { "enum": ["exact", "approximate"] }, "epsilon": { "type": "number" },
                }),
                &[],
            ),
            true,
        ),
    ]
}

fn s<'a>(m: &'a Map<String, Json>, k: &str) -> Result<&'a str, OpError> {
    m.get(k)
        .and_then(Json::as_str)
        .ok_or(OpError::invalid("missing string argument"))
}

async fn run<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    name: &str,
    args: Map<String, Json>,
    now: u64,
) -> Result<Json, OpError> {
    match name {
        "graph_list" => gr::list(b, ctx, now).await,
        "graph_query" => {
            let (graph, query) = (s(&args, "graph")?, s(&args, "query")?);
            if !gc::parse(query)?.read_only {
                return Err(OpError::invalid(
                    "graph_query is read-only (MATCH / RETURN / WITH); use graph_mutate",
                ));
            }
            gr::cypher(b, ctx, graph, query, now).await
        }
        "graph_mutate" => {
            let graph = s(&args, "graph")?.to_string();
            // Write + editor for every graph_mutate action.
            gr::authorize(b, ctx, Need::Write).await?;
            let create = args.get("create").and_then(Json::as_bool).unwrap_or(false);
            if args.get("dry_run").and_then(Json::as_bool).unwrap_or(false) {
                return dry_run(&graph, create, &args);
            }
            match (create, args.get("query"), args.get("edges")) {
                (true, None, None) => gr::create(b, ctx, &graph, now).await.map(|(_, v)| v),
                (false, Some(_), None) => gr::cypher(b, ctx, &graph, s(&args, "query")?, now).await,
                (false, None, Some(e)) => {
                    let call = GraphCall::AddEdges {
                        edges: serde_json::from_value(e.clone())
                            .map_err(|_| OpError::invalid("edges"))?,
                        edge_type: args
                            .get("type")
                            .and_then(Json::as_str)
                            .unwrap_or("LINK")
                            .into(),
                        label: args
                            .get("label")
                            .and_then(Json::as_str)
                            .unwrap_or("Node")
                            .into(),
                    };
                    gr::admit(b, ctx, Need::Write, now).await?;
                    match gr::graph_call(b, ctx, &graph, call).await? {
                        GraphOut::Info { view } => Ok(view),
                        _ => Err(crate::service::unexpected()),
                    }
                }
                _ => Err(OpError::invalid("exactly one of create, query or edges")),
            }
        }
        _ => {
            let body = Json::Object(args).to_string();
            crate::mincut_routes::post(b, ctx, body.as_bytes(), now, None)
                .await
                .map(|(_, v)| v)
        }
    }
}

/// `graph_mutate` with `dry_run`: authorized (by the caller) and
/// validated — graph name, query syntax and kind, edge shape and count —
/// without touching the graph.
fn dry_run(graph: &str, create: bool, args: &Map<String, Json>) -> Result<Json, OpError> {
    if !crate::graph_wire::name_ok(graph) {
        return Err(OpError::invalid("graph name"));
    }
    let detail = match (create, args.get("query"), args.get("edges")) {
        (true, None, None) => json!({ "create": graph }),
        (false, Some(Json::String(q)), None) => {
            json!({ "read_only": gc::parse(q)?.read_only })
        }
        (false, None, Some(e)) => {
            let edges: Vec<crate::graph_wire::BulkEdge> =
                serde_json::from_value(e.clone()).map_err(|_| OpError::invalid("edges"))?;
            if edges.len() > crate::graph_store::MAX_BULK_EDGES {
                return Err(OpError::new(ErrorCode::PayloadTooLarge, "too many edges"));
            }
            json!({ "edges": edges.len() })
        }
        _ => return Err(OpError::invalid("exactly one of create, query or edges")),
    };
    Ok(json!({ "dry_run": true, "graph": graph, "valid": detail }))
}

/// `tools/call` for a name outside the M1 table.
pub async fn tools_call<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    id: &Json,
    name: &str,
    params: &Json,
    now: u64,
    metadata_url: &str,
) -> ApiReply {
    if !NAMES.contains(&name) {
        return rpc_error(id, -32602, "unknown tool");
    }
    let args = match params.get("arguments") {
        None | Some(Json::Null) => Map::new(),
        Some(Json::Object(m)) => m.clone(),
        Some(_) => return rpc_error(id, -32602, "arguments must be an object"),
    };
    match run(b, ctx, name, args, now).await {
        Ok(result) => rpc_result(
            id,
            json!({
                "content": [{ "type": "text", "text": result.to_string() }],
                "structuredContent": result,
                "isError": false,
            }),
        ),
        Err(e) if e.code == ErrorCode::InsufficientScope => {
            let mut r = ApiReply::problem(&e, metadata_url);
            r.www_authenticate = step_up(&e, metadata_url);
            r
        }
        Err(e) => rpc_result(id, tool_error(&e)),
    }
}
