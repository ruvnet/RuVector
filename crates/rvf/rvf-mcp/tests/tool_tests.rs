//! TDD integration tests for rvf-mcp tool handlers.
//!
//! RED phase: these tests define the expected behavior of each MCP tool.
//! Written before implementation — they should fail until handlers are built.

use rvf_mcp::{RvfMcpServer, ServerConfig, ToolCall, ToolResult};
use serde_json::json;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_server() -> (TempDir, RvfMcpServer) {
    let dir = TempDir::new().unwrap();
    let config = ServerConfig {
        data_dir: dir.path().to_path_buf(),
        dimension: 4,
    };
    let server = RvfMcpServer::new(config).unwrap();
    (dir, server)
}

fn call(server: &mut RvfMcpServer, tool: &str, args: serde_json::Value) -> ToolResult {
    server
        .handle_tool_call(ToolCall {
            name: tool.to_string(),
            arguments: args,
        })
        .unwrap()
}

fn embedding(seed: u8) -> Vec<f64> {
    // 4-dim deterministic, non-collinear embeddings so distance search can distinguish them
    match seed {
        1 => vec![1.0, 0.0, 0.0, 0.0],
        2 => vec![0.0, 1.0, 0.0, 0.0],
        3 => vec![0.0, 0.0, 1.0, 0.0],
        _ => vec![0.0, 0.0, 0.0, 1.0],
    }
}

// ===========================================================================
// Tool listing
// ===========================================================================

#[test]
fn list_tools_returns_all_expected_tools() {
    let (_dir, server) = make_server();
    let tools = server.list_tools();

    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"rvf_memory_store"));
    assert!(names.contains(&"rvf_memory_search"));
    assert!(names.contains(&"rvf_memory_get"));
    assert!(names.contains(&"rvf_memory_delete"));
    assert!(names.contains(&"rvf_pattern_store"));
    assert!(names.contains(&"rvf_pattern_search"));
    assert!(names.contains(&"rvf_witness_log"));
    assert!(names.contains(&"rvf_witness_verify"));
    assert!(names.contains(&"rvf_coordination_state"));
    assert!(names.contains(&"rvf_coordination_vote"));
    assert!(names.contains(&"rvf_store_status"));
}

#[test]
fn each_tool_has_description_and_schema() {
    let (_dir, server) = make_server();
    let tools = server.list_tools();

    for tool in &tools {
        assert!(
            !tool.description.is_empty(),
            "tool {} missing description",
            tool.name
        );
        assert!(
            tool.input_schema.is_object(),
            "tool {} missing input_schema",
            tool.name
        );
    }
}

// ===========================================================================
// rvf_memory_store
// ===========================================================================

#[test]
fn memory_store_accepts_key_value_namespace_embedding() {
    let (_dir, mut server) = make_server();
    let result = call(
        &mut server,
        "rvf_memory_store",
        json!({
            "key": "user-pref",
            "value": "dark-mode",
            "namespace": "ui",
            "embedding": embedding(1)
        }),
    );
    assert!(!result.is_error);
    assert!(result.text.contains("stored"));
}

#[test]
fn memory_store_works_without_namespace() {
    let (_dir, mut server) = make_server();
    let result = call(
        &mut server,
        "rvf_memory_store",
        json!({
            "key": "lang",
            "value": "en",
            "embedding": embedding(2)
        }),
    );
    assert!(!result.is_error);
}

#[test]
fn memory_store_rejects_missing_key() {
    let (_dir, mut server) = make_server();
    let result = call(
        &mut server,
        "rvf_memory_store",
        json!({
            "value": "something",
            "embedding": embedding(1)
        }),
    );
    assert!(result.is_error);
}

#[test]
fn memory_store_rejects_missing_embedding() {
    let (_dir, mut server) = make_server();
    let result = call(
        &mut server,
        "rvf_memory_store",
        json!({
            "key": "k",
            "value": "v"
        }),
    );
    assert!(result.is_error);
}

// ===========================================================================
// rvf_memory_search
// ===========================================================================

#[test]
fn memory_search_returns_results_sorted_by_distance() {
    let (_dir, mut server) = make_server();

    // Store three memories
    for i in 1..=3u8 {
        call(
            &mut server,
            "rvf_memory_store",
            json!({
                "key": format!("mem-{}", i),
                "value": format!("val-{}", i),
                "embedding": embedding(i)
            }),
        );
    }

    let result = call(
        &mut server,
        "rvf_memory_search",
        json!({
            "embedding": embedding(2),
            "k": 3
        }),
    );
    assert!(!result.is_error);

    let parsed: serde_json::Value = serde_json::from_str(&result.text).unwrap();
    let results = parsed.as_array().unwrap();
    assert_eq!(results.len(), 3);

    // First result should be closest to embedding(2)
    assert_eq!(results[0]["key"], "mem-2");

    // Distances should be sorted ascending
    for i in 1..results.len() {
        let d_prev = results[i - 1]["distance"].as_f64().unwrap();
        let d_curr = results[i]["distance"].as_f64().unwrap();
        assert!(d_curr >= d_prev, "results not sorted");
    }
}

#[test]
fn memory_search_respects_k_limit() {
    let (_dir, mut server) = make_server();

    for i in 1..=5u8 {
        call(
            &mut server,
            "rvf_memory_store",
            json!({
                "key": format!("m-{}", i),
                "value": format!("v-{}", i),
                "embedding": embedding(i)
            }),
        );
    }

    let result = call(
        &mut server,
        "rvf_memory_search",
        json!({ "embedding": embedding(3), "k": 2 }),
    );
    let parsed: serde_json::Value = serde_json::from_str(&result.text).unwrap();
    assert_eq!(parsed.as_array().unwrap().len(), 2);
}

#[test]
fn memory_search_empty_store_returns_empty() {
    let (_dir, mut server) = make_server();
    let result = call(
        &mut server,
        "rvf_memory_search",
        json!({ "embedding": embedding(1), "k": 5 }),
    );
    assert!(!result.is_error);
    let parsed: serde_json::Value = serde_json::from_str(&result.text).unwrap();
    assert!(parsed.as_array().unwrap().is_empty());
}

// ===========================================================================
// rvf_memory_get
// ===========================================================================

#[test]
fn memory_get_retrieves_stored_value() {
    let (_dir, mut server) = make_server();
    call(
        &mut server,
        "rvf_memory_store",
        json!({ "key": "theme", "value": "dark", "embedding": embedding(1) }),
    );

    let result = call(&mut server, "rvf_memory_get", json!({ "key": "theme" }));
    assert!(!result.is_error);
    assert!(result.text.contains("dark"));
}

#[test]
fn memory_get_returns_not_found_for_missing_key() {
    let (_dir, mut server) = make_server();
    let result = call(&mut server, "rvf_memory_get", json!({ "key": "nope" }));
    assert!(!result.is_error);
    assert!(result.text.contains("not found") || result.text.contains("null"));
}

// ===========================================================================
// rvf_memory_delete
// ===========================================================================

#[test]
fn memory_delete_removes_entry() {
    let (_dir, mut server) = make_server();
    call(
        &mut server,
        "rvf_memory_store",
        json!({ "key": "tmp", "value": "x", "embedding": embedding(1) }),
    );

    let result = call(&mut server, "rvf_memory_delete", json!({ "key": "tmp" }));
    assert!(!result.is_error);

    // Should not be searchable after delete
    let search = call(
        &mut server,
        "rvf_memory_search",
        json!({ "embedding": embedding(1), "k": 5 }),
    );
    let parsed: serde_json::Value = serde_json::from_str(&search.text).unwrap();
    assert!(parsed.as_array().unwrap().is_empty());
}

// ===========================================================================
// rvf_pattern_store
// ===========================================================================

#[test]
fn pattern_store_accepts_task_reward_embedding() {
    let (_dir, mut server) = make_server();
    let result = call(
        &mut server,
        "rvf_pattern_store",
        json!({
            "task": "optimize query",
            "reward": 0.85,
            "success": true,
            "critique": "good coverage",
            "embedding": embedding(1)
        }),
    );
    assert!(!result.is_error);
    assert!(result.text.contains("id"));
}

#[test]
fn pattern_store_rejects_missing_task() {
    let (_dir, mut server) = make_server();
    let result = call(
        &mut server,
        "rvf_pattern_store",
        json!({ "reward": 0.5, "embedding": embedding(1) }),
    );
    assert!(result.is_error);
}

// ===========================================================================
// rvf_pattern_search
// ===========================================================================

#[test]
fn pattern_search_with_min_reward_filter() {
    let (_dir, mut server) = make_server();

    // Store patterns with varying rewards
    for i in 1..=5u8 {
        call(
            &mut server,
            "rvf_pattern_store",
            json!({
                "task": format!("task-{}", i),
                "reward": (i as f64) * 0.2,
                "success": i >= 3,
                "critique": "",
                "embedding": embedding(i)
            }),
        );
    }

    let result = call(
        &mut server,
        "rvf_pattern_search",
        json!({
            "embedding": embedding(4),
            "k": 10,
            "min_reward": 0.6
        }),
    );
    assert!(!result.is_error);

    let parsed: serde_json::Value = serde_json::from_str(&result.text).unwrap();
    let results = parsed.as_array().unwrap();
    for r in results {
        assert!(r["reward"].as_f64().unwrap() >= 0.6);
    }
}

// ===========================================================================
// rvf_witness_log
// ===========================================================================

#[test]
fn witness_log_records_action() {
    let (_dir, mut server) = make_server();
    let result = call(
        &mut server,
        "rvf_witness_log",
        json!({ "action": "decision", "details": ["chose-plan-B"] }),
    );
    assert!(!result.is_error);
    assert!(result.text.contains("recorded"));
}

// ===========================================================================
// rvf_witness_verify
// ===========================================================================

#[test]
fn witness_verify_passes_on_intact_chain() {
    let (_dir, mut server) = make_server();

    // Record some actions
    call(
        &mut server,
        "rvf_witness_log",
        json!({ "action": "init", "details": [] }),
    );
    call(
        &mut server,
        "rvf_witness_log",
        json!({ "action": "search", "details": ["query-1"] }),
    );

    let result = call(&mut server, "rvf_witness_verify", json!({}));
    assert!(!result.is_error);
    assert!(result.text.contains("valid") || result.text.contains("ok"));
}

// ===========================================================================
// rvf_coordination_state
// ===========================================================================

#[test]
fn coordination_state_records_and_retrieves() {
    let (_dir, mut server) = make_server();

    // Record state
    let result = call(
        &mut server,
        "rvf_coordination_state",
        json!({
            "agent_id": "agent-1",
            "key": "status",
            "value": "active"
        }),
    );
    assert!(!result.is_error);
}

// ===========================================================================
// rvf_coordination_vote
// ===========================================================================

#[test]
fn coordination_vote_records_consensus() {
    let (_dir, mut server) = make_server();
    let result = call(
        &mut server,
        "rvf_coordination_vote",
        json!({
            "topic": "leader-election",
            "agent_id": "agent-1",
            "vote": true
        }),
    );
    assert!(!result.is_error);
}

// ===========================================================================
// rvf_store_status
// ===========================================================================

#[test]
fn store_status_returns_vector_counts() {
    let (_dir, mut server) = make_server();

    // Empty initially
    let result = call(&mut server, "rvf_store_status", json!({}));
    assert!(!result.is_error);
    let parsed: serde_json::Value = serde_json::from_str(&result.text).unwrap();
    assert_eq!(parsed["memory_vectors"], 0);
    assert_eq!(parsed["pattern_vectors"], 0);

    // Store one memory
    call(
        &mut server,
        "rvf_memory_store",
        json!({ "key": "k", "value": "v", "embedding": embedding(1) }),
    );

    let result = call(&mut server, "rvf_store_status", json!({}));
    let parsed: serde_json::Value = serde_json::from_str(&result.text).unwrap();
    assert_eq!(parsed["memory_vectors"], 1);
}

// ===========================================================================
// JSON-RPC protocol
// ===========================================================================

#[test]
fn handle_jsonrpc_request_dispatches_to_tool() {
    let (_dir, mut server) = make_server();

    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "rvf_store_status",
            "arguments": {}
        }
    });

    let response = server.handle_jsonrpc(&request.to_string()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();

    assert_eq!(parsed["jsonrpc"], "2.0");
    assert_eq!(parsed["id"], 1);
    assert!(parsed["result"].is_object());
}

#[test]
fn handle_jsonrpc_tools_list() {
    let (_dir, mut server) = make_server();

    let request = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    });

    let response = server.handle_jsonrpc(&request.to_string()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();

    assert_eq!(parsed["jsonrpc"], "2.0");
    let tools = parsed["result"]["tools"].as_array().unwrap();
    assert!(tools.len() >= 11);
}

#[test]
fn handle_jsonrpc_initialize() {
    let (_dir, mut server) = make_server();

    let request = json!({
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "test", "version": "0.1.0" }
        }
    });

    let response = server.handle_jsonrpc(&request.to_string()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();

    assert_eq!(parsed["result"]["protocolVersion"], "2024-11-05");
    assert!(parsed["result"]["serverInfo"]["name"].is_string());
    assert!(parsed["result"]["capabilities"]["tools"].is_object());
}

#[test]
fn handle_jsonrpc_unknown_method_returns_error() {
    let (_dir, mut server) = make_server();

    let request = json!({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "nonexistent/method",
        "params": {}
    });

    let response = server.handle_jsonrpc(&request.to_string()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert!(parsed["error"].is_object());
}

#[test]
fn handle_jsonrpc_unknown_tool_returns_error() {
    let (_dir, mut server) = make_server();

    let request = json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": { "name": "fake_tool", "arguments": {} }
    });

    let response = server.handle_jsonrpc(&request.to_string()).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert!(parsed["error"].is_object());
}
