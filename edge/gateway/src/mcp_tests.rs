//! `POST /v1/mcp` JSON-RPC over the DO backend.

use crate::backend::mem::MemBackend;
use crate::mcp::{handle, tools_list};
use crate::rest::ApiReply;
use crate::testkit::*;
use ruvector_edge_store::CallerContext;
use ruvector_edge_tenancy::Role;
use serde_json::{json, Value as Json};

fn rpc(b: &MemBackend, c: &CallerContext, msg: Json) -> ApiReply {
    block_on(handle(b, c, msg.to_string().as_bytes(), T0, MCP_MD))
}

fn tool(b: &MemBackend, c: &CallerContext, name: &str, args: Json) -> ApiReply {
    rpc(
        b,
        c,
        json!({ "jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": { "name": name, "arguments": args } }),
    )
}

fn body(r: &ApiReply) -> Json {
    serde_json::from_str(&r.body).unwrap()
}

#[test]
fn tools_list_shape_and_annotations() {
    let list = tools_list();
    let tools = list["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        [
            "tenant_me",
            "collection_list",
            "collection_get",
            "vector_query",
            "vector_fetch",
            "tenant_claim",
            "collection_create",
            "vector_upsert",
            "vector_delete",
            // M4 (`graph_mcp`).
            "graph_list",
            "graph_query",
            "graph_mutate",
            "mincut"
        ]
    );
    for t in tools {
        let n = t["name"].as_str().unwrap();
        assert!(
            t["description"].as_str().is_some_and(|d| !d.is_empty()),
            "{n}"
        );
        assert_eq!(t["inputSchema"]["type"], json!("object"), "{n}");
        let ro = t["annotations"]["readOnlyHint"].as_bool().unwrap();
        let destructive = t["annotations"]["destructiveHint"].as_bool().unwrap();
        let writes = [
            "collection_create",
            "vector_upsert",
            "vector_delete",
            "graph_mutate",
        ]
        .contains(&n);
        assert_eq!(ro, !writes && n != "tenant_claim", "{n}");
        let idem = t["annotations"]["idempotentHint"].as_bool().unwrap();
        assert_eq!(
            idem,
            !["collection_create", "tenant_claim", "graph_mutate"].contains(&n),
            "{n}"
        );
        assert_eq!(
            destructive,
            n == "vector_upsert" || n == "vector_delete" || n == "graph_mutate",
            "{n}"
        );
        let has_dry_run = t["inputSchema"]["properties"].get("dry_run").is_some();
        assert_eq!(has_dry_run, writes, "{n}");
    }
    let b = MemBackend::new();
    let c = ctx(&tenant("org-a"), "alice", ro());
    let r = rpc(
        &b,
        &c,
        json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }),
    );
    assert_eq!(body(&r)["result"], list);
}

#[test]
fn initialize_ping_notifications_and_errors() {
    let b = MemBackend::new();
    let c = ctx(&tenant("org-a"), "alice", ro());
    let init = rpc(
        &b,
        &c,
        json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } } }),
    );
    let v = body(&init);
    assert_eq!(v["result"]["protocolVersion"], json!("2025-06-18"));
    assert!(v["result"]["capabilities"]["tools"].is_object());
    let r = rpc(
        &b,
        &c,
        json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
    );
    assert_eq!((r.status, r.body.as_str()), (202, ""));
    assert_eq!(
        body(&rpc(
            &b,
            &c,
            json!({ "jsonrpc": "2.0", "id": 2, "method": "ping" })
        ))["result"],
        json!({})
    );
    let e = body(&rpc(
        &b,
        &c,
        json!({ "jsonrpc": "2.0", "id": 3, "method": "resources/list" }),
    ));
    assert_eq!(e["error"]["code"], json!(-32601));
    let e = body(&tool(&b, &c, "drop_tenant", json!({})));
    assert_eq!(e["error"]["code"], json!(-32602));
    let r = block_on(handle(&b, &c, b"[1,2]", T0, MCP_MD));
    assert_eq!(body(&r)["error"]["code"], json!(-32600));
}

#[test]
fn write_tools_step_up_with_http_403_and_viewers_get_tool_errors() {
    let b = MemBackend::new();
    let t = tenant("org-a");
    let owner = ctx(&t, "alice", rw());
    block_on(crate::service::claim(&call(&b, &owner))).unwrap();
    let r = tool(
        &b,
        &owner,
        "collection_create",
        json!({ "name": "d", "dim": 2, "metric": "l2" }),
    );
    assert_eq!(body(&r)["result"]["isError"], json!(false), "{}", r.body);
    // Missing ruvector:write: an HTTP 403 challenge, not a JSON-RPC error.
    let owner_ro = ctx(&t, "alice", ro());
    let r = tool(
        &b,
        &owner_ro,
        "vector_upsert",
        json!({ "collection": "d", "vectors": [{ "id": "a", "values": [1.0, 0.0] }], "dry_run": true }),
    );
    assert_eq!(r.status, 403);
    let www = r.www_authenticate.unwrap();
    assert!(www.contains(r#"error="insufficient_scope""#), "{www}");
    assert!(
        www.contains(r#"scope="ruvector:read ruvector:write offline_access""#),
        "{www}"
    );
    assert!(www.contains(MCP_MD), "{www}");
    // With the scope but only a viewer role: a tool error.
    b.invite(
        &t,
        &crate::testkit::sub("alice"),
        &crate::testkit::sub("val"),
        Role::Viewer,
        T0,
    )
    .unwrap();
    let viewer = ctx(&t, "val", rw());
    let r = tool(
        &b,
        &viewer,
        "vector_delete",
        json!({ "collection": "d", "ids": ["a"] }),
    );
    let v = body(&r);
    assert_eq!(
        (r.status, v["result"]["isError"].clone()),
        (200, json!(true))
    );
    assert!(v["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("role_required"));
    // Reads work for the viewer; dry_run is refused on read tools.
    let r = tool(&b, &viewer, "collection_get", json!({ "collection": "d" }));
    assert_eq!(body(&r)["result"]["structuredContent"]["name"], json!("d"));
    let r = tool(
        &b,
        &viewer,
        "vector_query",
        json!({ "collection": "d", "vector": [1.0, 0.0], "top_k": 1, "dry_run": true }),
    );
    assert_eq!(body(&r)["error"]["code"], json!(-32602));
    // Owner dry-run upsert writes nothing.
    let r = tool(
        &b,
        &owner,
        "vector_upsert",
        json!({ "collection": "d", "vectors": [{ "id": "a", "values": [1.0, 0.0] }], "dry_run": true }),
    );
    assert_eq!(
        body(&r)["result"]["structuredContent"]["dry_run"],
        json!(true)
    );
    let r = tool(
        &b,
        &owner,
        "vector_fetch",
        json!({ "collection": "d", "ids": ["a"] }),
    );
    assert_eq!(
        body(&r)["result"]["structuredContent"]["vectors"],
        json!([])
    );
}

#[test]
fn tenant_claim_bootstraps_over_mcp_only() {
    let b = MemBackend::new();
    let t = tenant("org-m");
    let alice = ctx(&t, "alice", rw());
    // Unclaimed: every data tool is `not_claimed`.
    let r = tool(&b, &alice, "collection_list", json!({}));
    assert!(body(&r)["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("not_claimed"));
    // Without ruvector:write: the HTTP 403 step-up challenge.
    let r = tool(&b, &ctx(&t, "alice", ro()), "tenant_claim", json!({}));
    assert_eq!(r.status, 403);
    assert!(r.www_authenticate.unwrap().contains("ruvector:write"));
    // Arguments (dry_run included) are refused.
    for args in [json!({ "dry_run": true }), json!({ "x": 1 })] {
        let e = body(&tool(&b, &alice, "tenant_claim", args));
        assert_eq!(e["error"]["code"], json!(-32602));
    }
    let r = tool(&b, &alice, "tenant_claim", json!({}));
    let v = body(&r);
    assert_eq!(v["result"]["isError"], json!(false), "{}", r.body);
    assert_eq!(v["result"]["structuredContent"]["role"], json!("owner"));
    // A second claim (anyone) is a tool error, not a JSON-RPC error.
    let r = tool(&b, &ctx(&t, "bob", rw()), "tenant_claim", json!({}));
    let v = body(&r);
    assert_eq!(v["result"]["isError"], json!(true));
    assert!(v["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("conflict"));
    let r = tool(&b, &alice, "tenant_me", json!({}));
    assert_eq!(
        body(&r)["result"]["structuredContent"]["role"],
        json!("owner")
    );
}

#[test]
fn protocol_versions_and_request_ids() {
    use crate::mcp::{bad_protocol_version, protocol_header_ok, PROTOCOL_VERSIONS};
    let b = MemBackend::new();
    let c = ctx(&tenant("org-a"), "alice", ro());
    // 2025-03-26 (batches required) is not offered: the latest is.
    assert!(!PROTOCOL_VERSIONS.contains(&"2025-03-26"));
    let init = rpc(
        &b,
        &c,
        json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2025-03-26", "capabilities": {} } }),
    );
    assert_eq!(
        body(&init)["result"]["protocolVersion"],
        json!("2025-11-25")
    );
    // MCP-Protocol-Version: absent or supported passes, anything else 400.
    for ok in [None, Some("2025-06-18"), Some("2025-11-25")] {
        assert!(protocol_header_ok(ok), "{ok:?}");
    }
    for bad in ["1999-01-01", "garbage", "2025-03-26", ""] {
        assert!(!protocol_header_ok(Some(bad)), "{bad}");
    }
    let r = bad_protocol_version();
    assert_eq!(r.status, 400);
    assert_eq!(body(&r)["id"], Json::Null);
    assert_eq!(body(&r)["error"]["code"], json!(-32600));
    // Request ids: string or integer only, never echoed otherwise.
    for id in [json!(null), json!(1.5), json!(true), json!({}), json!([1])] {
        let v = body(&rpc(
            &b,
            &c,
            json!({ "jsonrpc": "2.0", "id": id, "method": "tools/list" }),
        ));
        assert_eq!(v["error"]["code"], json!(-32600), "{id}");
        assert_eq!(v["id"], Json::Null, "{id}");
        assert!(v.get("result").is_none(), "{id}");
    }
    for id in [json!("a-1"), json!(0), json!(-7), json!(u64::MAX)] {
        let v = body(&rpc(
            &b,
            &c,
            json!({ "jsonrpc": "2.0", "id": id, "method": "ping" }),
        ));
        assert_eq!((v["id"].clone(), v["result"].clone()), (id, json!({})));
    }
    // A bad id on an otherwise invalid request is not echoed either.
    let v = body(&rpc(
        &b,
        &c,
        json!({ "jsonrpc": "1.0", "id": {"x": 1}, "method": "ping" }),
    ));
    assert_eq!(
        (v["id"].clone(), v["error"]["code"].clone()),
        (Json::Null, json!(-32600))
    );
}
