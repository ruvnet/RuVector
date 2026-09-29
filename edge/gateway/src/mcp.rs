//! `POST /v1/mcp`: MCP Streamable HTTP, JSON-RPC 2.0, JSON responses only
//! (ADR-351 §3 rv-mcp, §7.2). Tools map 1:1 onto the §16.3 ops (plus
//! `collection_get`), so REST, MCP and `/v1/ops` share one executor and one
//! authorization table.
//!
//! - A missing scope on `tools/call` is **HTTP 403** with the §5.3 step-up
//!   `WWW-Authenticate` challenge naming the `/v1/mcp` metadata, not a
//!   JSON-RPC error, so MCP clients re-authorize with the wider scope.
//! - Every other tool failure (`role_required`, `not_claimed`, `not_found`,
//!   validation, quota) is a tool result with `isError: true`.
//! - `tenant_claim` is the bootstrap tool (ADR-351 §4.2), so a connector
//!   holding only a `/v1/mcp` token can claim its tenant.
//! - Request ids must be a string or an integer; an unsupported
//!   `MCP-Protocol-Version` header is HTTP 400.
//! - Malformed JSON-RPC → `-32700`/`-32600`; unknown method → `-32601`;
//!   unknown tool or bad arguments shape → `-32602`.

use crate::backend::Backend;
use crate::rest::{step_up, ApiReply};
use crate::service::{self, Call};
use ruvector_edge_store::{CallerContext, ErrorCode, Op, OpError};
use serde_json::{json, Map, Value as Json};

/// Protocol revisions this server speaks, newest first. `2025-03-26` is
/// not offered: it requires JSON-RPC batch support (removed in 2025-06-18),
/// and ADR-351 treats 2025-03-26-only clients as unsupported; a client
/// asking for it is answered with the latest revision.
pub const PROTOCOL_VERSIONS: [&str; 2] = ["2025-11-25", "2025-06-18"];

/// `MCP-Protocol-Version` request header check (Streamable HTTP,
/// 2025-06-18+): absent is accepted (the spec's 2025-03-26 fallback, as
/// before `initialize` negotiates), present must be a revision this server
/// speaks, else the request is 400 ([`bad_protocol_version`]).
pub fn protocol_header_ok(header: Option<&str>) -> bool {
    header.map_or(true, |v| PROTOCOL_VERSIONS.contains(&v.trim()))
}

/// The 400 answer to an unsupported `MCP-Protocol-Version` header.
pub fn bad_protocol_version() -> ApiReply {
    let mut r = rpc_error(&Json::Null, -32600, "unsupported MCP-Protocol-Version");
    r.status = 400;
    r
}

/// Largest JSON-RPC body (the §10 1 MiB limit).
pub const MAX_MCP_BODY_BYTES: usize = 1 << 20;

/// What a tool runs.
#[derive(Clone, Copy)]
enum Kind {
    /// A §16.3 op.
    Op(Op),
    /// `collection_get` (catalog view + shard counters).
    Get,
    /// `tenant_claim`: the MCP bootstrap (ADR-351 §4.2).
    Claim,
}

/// Tool definition: name, kind, mutating (dry-run capable unless a claim),
/// input schema, description.
struct Tool {
    name: &'static str,
    op: Kind,
    mutating: bool,
    destructive: bool,
    description: &'static str,
}

const TOOLS: [Tool; 9] = [
    Tool {
        name: "tenant_me",
        op: Kind::Op(Op::TenantMe),
        mutating: false,
        destructive: false,
        description: "Your tenant key, subject, role and whether the tenant is claimed.",
    },
    Tool {
        name: "collection_list",
        op: Kind::Op(Op::CollectionList),
        mutating: false,
        destructive: false,
        description: "List the tenant's vector collections.",
    },
    Tool {
        name: "collection_get",
        op: Kind::Get,
        mutating: false,
        destructive: false,
        description: "One collection's configuration and vector count.",
    },
    Tool {
        name: "vector_query",
        op: Kind::Op(Op::VectorQuery),
        mutating: false,
        destructive: false,
        description:
            "Exact top-k nearest neighbours of a query vector, with an optional metadata filter.",
    },
    Tool {
        name: "vector_fetch",
        op: Kind::Op(Op::VectorFetch),
        mutating: false,
        destructive: false,
        description: "Fetch up to 100 vectors by id.",
    },
    Tool {
        name: "tenant_claim",
        op: Kind::Claim,
        mutating: true,
        destructive: false,
        description: "Claim this tenant: its first claimant becomes owner (needs ruvector:write). \
                      Run once before creating collections; an already claimed tenant is an error.",
    },
    Tool {
        name: "collection_create",
        op: Kind::Op(Op::CollectionCreate),
        mutating: true,
        destructive: false,
        description: "Create a vector collection (flat index). Supports dry_run.",
    },
    Tool {
        name: "vector_upsert",
        op: Kind::Op(Op::VectorUpsert),
        mutating: true,
        destructive: true,
        description: "Insert or overwrite up to 500 vectors by id. Supports dry_run.",
    },
    Tool {
        name: "vector_delete",
        op: Kind::Op(Op::VectorDelete),
        mutating: true,
        destructive: true,
        description: "Delete up to 1000 vectors by id. Supports dry_run.",
    },
];

fn schema(name: &str) -> Json {
    let coll = json!({ "type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{0,62}$" });
    let dry = json!({ "type": "boolean", "description": "Validate and report the effect without writing." });
    let ids = |max: u32| json!({ "type": "array", "items": { "type": "string" }, "minItems": 1, "maxItems": max });
    let (props, required): (Json, &[&str]) = match name {
        "collection_get" => (json!({ "collection": coll }), &["collection"]),
        "vector_query" => (
            json!({
                "collection": coll,
                "vector": { "type": "array", "items": { "type": "number" } },
                "top_k": { "type": "integer", "minimum": 1, "maximum": 100 },
                "filter": { "type": "object" },
                "include": { "type": "array", "items": { "enum": ["metadata", "values"] } },
            }),
            &["collection", "vector", "top_k"],
        ),
        "vector_fetch" => (
            json!({
                "collection": coll, "ids": ids(100), "include_values": { "type": "boolean" },
            }),
            &["collection", "ids"],
        ),
        "collection_create" => (
            json!({
                "name": coll,
                "dim": { "type": "integer", "minimum": 1, "maximum": 1536 },
                "metric": { "enum": ["cosine", "l2", "dot"] },
                "filterable_keys": { "type": "array", "items": { "type": "string" }, "maxItems": 8 },
                "shards": { "type": "integer", "minimum": 1, "maximum": 6 },
                "dry_run": dry,
            }),
            &["name", "dim", "metric"],
        ),
        "vector_upsert" => (
            json!({
                "collection": coll,
                "vectors": { "type": "array", "minItems": 1, "maxItems": 500, "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "values": { "type": "array", "items": { "type": "number" } },
                        "metadata": { "type": "object" },
                    },
                    "required": ["id", "values"],
                    "additionalProperties": false,
                }},
                "dry_run": dry,
            }),
            &["collection", "vectors"],
        ),
        "vector_delete" => (
            json!({ "collection": coll, "ids": ids(1000), "dry_run": dry }),
            &["collection", "ids"],
        ),
        _ => (json!({}), &[]),
    };
    json!({ "type": "object", "properties": props, "required": required, "additionalProperties": false })
}

/// The `tools/list` result.
pub fn tools_list() -> Json {
    let tools: Vec<Json> = TOOLS
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": schema(t.name),
                "annotations": {
                    "readOnlyHint": !t.mutating,
                    "destructiveHint": t.destructive,
                    "idempotentHint": !t.mutating
                        || !matches!(t.name, "collection_create" | "tenant_claim"),
                    "openWorldHint": false,
                },
            })
        })
        .collect();
    json!({ "tools": tools })
}

fn rpc_error(id: &Json, code: i64, message: &str) -> ApiReply {
    ApiReply::json(
        200,
        &json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }),
    )
}

fn rpc_result(id: &Json, result: Json) -> ApiReply {
    ApiReply::json(
        200,
        &json!({ "jsonrpc": "2.0", "id": id, "result": result }),
    )
}

fn tool_error(e: &OpError) -> Json {
    let body = json!({ "code": e.code.as_str(), "status": e.code.status(), "detail": e.detail });
    json!({ "content": [{ "type": "text", "text": body.to_string() }], "isError": true })
}

fn initialize(params: &Json) -> Json {
    let asked = params.get("protocolVersion").and_then(Json::as_str);
    let version = asked
        .and_then(|v| PROTOCOL_VERSIONS.into_iter().find(|s| *s == v))
        .unwrap_or(PROTOCOL_VERSIONS[0]);
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": { "name": "ruvector-edge", "version": env!("CARGO_PKG_VERSION") },
    })
}

/// Handle one JSON-RPC message for a verified caller. `metadata_url` is
/// the `/v1/mcp` RFC 9728 document (step-up challenges).
pub async fn handle<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    body: &[u8],
    now: u64,
    metadata_url: &str,
) -> ApiReply {
    if body.len() > MAX_MCP_BODY_BYTES {
        let e = OpError::new(ErrorCode::PayloadTooLarge, "body too large");
        return ApiReply::problem(&e, metadata_url);
    }
    let Ok(msg) = serde_json::from_slice::<Json>(body) else {
        return rpc_error(&Json::Null, -32700, "parse error");
    };
    let Some(obj) = msg.as_object() else {
        return rpc_error(
            &Json::Null,
            -32600,
            "batches and non-objects are not supported",
        );
    };
    let method = obj.get("method").and_then(Json::as_str);
    if obj.get("jsonrpc").and_then(Json::as_str) != Some("2.0") || method.is_none() {
        // A response or garbage from the client: nothing to answer.
        if obj.contains_key("result") || obj.contains_key("error") {
            return accepted();
        }
        let id = obj.get("id").filter(|i| valid_id(i)).unwrap_or(&Json::Null);
        return rpc_error(id, -32600, "invalid request");
    }
    let method = method.unwrap_or_default();
    let Some(id) = obj.get("id") else {
        // Notifications (e.g. notifications/initialized) get 202, no body.
        return accepted();
    };
    if !valid_id(id) {
        return rpc_error(&Json::Null, -32600, "invalid request id");
    }
    let params = obj.get("params").cloned().unwrap_or_else(|| json!({}));
    match method {
        "initialize" => rpc_result(id, initialize(&params)),
        "ping" => rpc_result(id, json!({})),
        "tools/list" => rpc_result(id, tools_list()),
        "tools/call" => tools_call(b, ctx, id, &params, now, metadata_url).await,
        _ => rpc_error(id, -32601, "method not found"),
    }
}

/// MCP request ids are a string or an integer, never `null` (MCP base
/// protocol, 2025-06-18+).
fn valid_id(id: &Json) -> bool {
    id.is_string() || id.is_i64() || id.is_u64()
}

fn accepted() -> ApiReply {
    ApiReply {
        status: 202,
        body: String::new(),
        content_type: "application/json",
        www_authenticate: None,
    }
}

async fn tools_call<B: Backend>(
    b: &B,
    ctx: &CallerContext,
    id: &Json,
    params: &Json,
    now: u64,
    metadata_url: &str,
) -> ApiReply {
    let Some(name) = params.get("name").and_then(Json::as_str) else {
        return rpc_error(id, -32602, "missing tool name");
    };
    let Some(tool) = TOOLS.iter().find(|t| t.name == name) else {
        return rpc_error(id, -32602, "unknown tool");
    };
    let mut args: Map<String, Json> = match params.get("arguments") {
        None | Some(Json::Null) => Map::new(),
        Some(Json::Object(m)) => m.clone(),
        Some(_) => return rpc_error(id, -32602, "arguments must be an object"),
    };
    let claim = matches!(tool.op, Kind::Claim);
    if claim && !args.is_empty() {
        return rpc_error(id, -32602, "tenant_claim takes no arguments");
    }
    let dry_run = match args.remove("dry_run") {
        None => false,
        Some(Json::Bool(d)) if tool.mutating && !claim => d,
        Some(_) => return rpc_error(id, -32602, "dry_run not accepted"),
    };
    let raw = Json::Object(args).to_string();
    let call = Call {
        b,
        ctx,
        dry_run,
        now,
    };
    let out = match tool.op {
        Kind::Op(op) => service::execute(&call, op, &raw).await,
        Kind::Get => service::collection_get(&call, &raw).await,
        // Missing write/admin scope → the HTTP 403 step-up below; an
        // already claimed tenant (409) → an `isError` result.
        Kind::Claim => service::claim(&call).await,
    };
    match out {
        Ok((result, _usage)) => rpc_result(
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
