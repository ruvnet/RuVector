//! The M5 registry surface next to the M2 one: §10 rate classes of the
//! `/v1/rvf` routes, and `tools/list` carrying both tool sets.

use crate::api::guard;
use crate::api::ratelimit::mem::CountingLimiter;
use crate::api::ratelimit::{budgets, op_class, request_class, Class};
use crate::mcp;
use crate::registry_world::*;
use crate::rvf_mcp::RvfTools;
use crate::testkit::*;
use serde_json::{json, Value as Json};
use worker::Method;

#[test]
fn rvf_routes_are_charged_to_read_or_write() {
    let (get, post, put, del) = (Method::Get, Method::Post, Method::Put, Method::Delete);
    let v = "/v1/rvf/acme/p/1.0.0";
    let cases: [(&Method, String, Class); 16] = [
        (&get, "/v1/rvf/scopes".into(), Class::Read),
        (&get, "/v1/rvf/acme/p".into(), Class::Read),
        (&get, v.into(), Class::Read),
        (&get, format!("{v}/blob"), Class::Read),
        (&post, "/v1/rvf/scopes/acme".into(), Class::Write),
        (&post, format!("{v}/uploads"), Class::Write),
        (&put, format!("{v}/uploads/u1/parts/1"), Class::Write),
        (&post, format!("{v}/uploads/u1:finalize"), Class::Write),
        (&post, format!("{v}:publish"), Class::Write),
        (&post, format!("{v}:yank"), Class::Write),
        (&post, format!("{v}:unyank"), Class::Write),
        (
            &post,
            "/v1/collections/docs:import-rvf".into(),
            Class::Write,
        ),
        // The M2 surface keeps its classes.
        (&get, "/v1/me".into(), Class::Read),
        (&del, "/v1/collections/docs".into(), Class::Write),
        (&post, "/v1/ops".into(), Class::Ops),
        (&get, "/v1/rvf/nope/x/y/z/w".into(), Class::Read),
    ];
    for (m, path, want) in &cases {
        assert_eq!(request_class(false, m, path), *want, "{m:?} {path}");
    }
    assert_eq!(request_class(true, &post, "/v1/mcp"), Class::Mcp);
    // The MCP import tool also pays the write budget; the reads do not.
    let tool = |n: &str| {
        json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": { "name": n } })
            .to_string()
            .into_bytes()
    };
    assert_eq!(
        op_class(Class::Mcp, &tool("rvf_import")),
        Some(Class::Write)
    );
    for n in ["rvf_list", "rvf_get"] {
        assert_eq!(op_class(Class::Mcp, &tool(n)), None, "{n}");
    }
}

#[test]
fn part_uploads_exhaust_the_user_write_budget() {
    let w = World::new();
    let alice = w.owner("org-a", "alice");
    let l = CountingLimiter::default();
    let path = "/v1/rvf/acme/p/1.0.0/uploads/u1/parts/1";
    let class = request_class(false, &Method::Put, path);
    let (user, _) = budgets(class);
    for _ in 0..user.limit {
        assert!(block_on(guard(&w.b, &l, class, &alice, T0, REST_MD)).is_ok());
    }
    let r = block_on(guard(&w.b, &l, class, &alice, T0, REST_MD)).unwrap_err();
    assert_eq!((r.reply.status, r.retry_after), (429, Some(10)));
    // Pulls are a separate (read) budget.
    let pull = request_class(false, &Method::Get, "/v1/rvf/acme/p/1.0.0/blob");
    assert!(block_on(guard(&w.b, &l, pull, &alice, T0, REST_MD)).is_ok());
}

#[test]
fn tools_list_has_the_data_tools_the_graph_tools_and_the_registry_tools() {
    let w = World::new();
    let alice = w.owner("org-a", "alice");
    let d = w.deps();
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }).to_string();
    let r = block_on(mcp::handle_with(
        &w.b,
        &RvfTools(&d),
        &alice,
        body.as_bytes(),
        T0,
        MCP_MD,
    ));
    let v: Json = serde_json::from_str(&r.body).unwrap();
    let tools = v["result"]["tools"].as_array().unwrap();
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
            "graph_list",
            "graph_query",
            "graph_mutate",
            "mincut",
            "rvf_list",
            "rvf_get",
            "rvf_import",
        ]
    );
    // The data and M4 graph tools keep exactly the entries `tools_list`
    // gives them; the registry tools follow.
    assert_eq!(&tools[..13], mcp::tools_list()["tools"].as_array().unwrap());
    for t in &tools[13..] {
        let n = t["name"].as_str().unwrap();
        let a = &t["annotations"];
        let write = n == "rvf_import";
        assert_eq!(a["readOnlyHint"], json!(!write), "{n}");
        assert_eq!(a["destructiveHint"], json!(write), "{n}");
        assert_eq!(a["idempotentHint"], json!(true), "{n}");
        assert_eq!(a["openWorldHint"], json!(false), "{n}");
        assert_eq!(
            t["inputSchema"]["additionalProperties"],
            json!(false),
            "{n}"
        );
    }
}
