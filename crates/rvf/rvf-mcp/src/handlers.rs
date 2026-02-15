//! Tool handler implementations.
//!
//! Each handler takes the server state + arguments and returns a ToolResult.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rvf_adapter_agentdb::MemoryPattern;

use crate::RvfMcpServer;

/// A tool call with name and JSON arguments.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Result of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub text: String,
    pub is_error: bool,
}

impl ToolResult {
    fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
        }
    }

    fn err(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
        }
    }
}

/// Dispatch a tool call to the appropriate handler.
pub fn dispatch(server: &mut RvfMcpServer, call: ToolCall) -> Result<ToolResult, String> {
    let args = &call.arguments;
    match call.name.as_str() {
        "rvf_memory_store" => memory_store(server, args),
        "rvf_memory_search" => memory_search(server, args),
        "rvf_memory_get" => memory_get(server, args),
        "rvf_memory_delete" => memory_delete(server, args),
        "rvf_pattern_store" => pattern_store(server, args),
        "rvf_pattern_search" => pattern_search(server, args),
        "rvf_witness_log" => witness_log(server, args),
        "rvf_witness_verify" => witness_verify(server, args),
        "rvf_coordination_state" => coordination_state(server, args),
        "rvf_coordination_vote" => coordination_vote(server, args),
        "rvf_store_status" => store_status(server, args),
        _ => Ok(ToolResult::err(format!("unknown tool: {}", call.name))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_embedding(args: &Value) -> Result<Vec<f32>, ToolResult> {
    let arr = args
        .get("embedding")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ToolResult::err("missing or invalid 'embedding' array"))?;

    Ok(arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
}

fn require_str<'a>(args: &'a Value, field: &str) -> Result<&'a str, ToolResult> {
    args.get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolResult::err(format!("missing required field: '{field}'")))
}

// ---------------------------------------------------------------------------
// Memory tools
// ---------------------------------------------------------------------------

fn memory_store(server: &mut RvfMcpServer, args: &Value) -> Result<ToolResult, String> {
    let key = match require_str(args, "key") {
        Ok(k) => k.to_string(),
        Err(e) => return Ok(e),
    };
    let value = match require_str(args, "value") {
        Ok(v) => v.to_string(),
        Err(e) => return Ok(e),
    };
    let namespace = args.get("namespace").and_then(|v| v.as_str()).map(String::from);
    let embedding = match parse_embedding(args) {
        Ok(e) => e,
        Err(e) => return Ok(e),
    };

    let id = server
        .memory_store
        .ingest_memory(&key, &value, namespace.as_deref(), &embedding)
        .map_err(|e| format!("ingest_memory: {e}"))?;

    server
        .key_index
        .insert(key.clone(), (id, value, namespace));

    Ok(ToolResult::ok(
        json!({ "stored": true, "id": id, "key": key }).to_string(),
    ))
}

fn memory_search(server: &mut RvfMcpServer, args: &Value) -> Result<ToolResult, String> {
    let embedding = match parse_embedding(args) {
        Ok(e) => e,
        Err(e) => return Ok(e),
    };
    let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let results = server.memory_store.search_memories(&embedding, k);

    let output: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "key": r.key,
                "value": r.value,
                "namespace": r.namespace,
                "distance": r.distance,
            })
        })
        .collect();

    Ok(ToolResult::ok(serde_json::to_string(&output).unwrap()))
}

fn memory_get(server: &mut RvfMcpServer, args: &Value) -> Result<ToolResult, String> {
    let key = match require_str(args, "key") {
        Ok(k) => k,
        Err(e) => return Ok(e),
    };

    match server.key_index.get(key) {
        Some((_, value, namespace)) => Ok(ToolResult::ok(
            json!({
                "key": key,
                "value": value,
                "namespace": namespace,
            })
            .to_string(),
        )),
        None => Ok(ToolResult::ok(
            json!({ "key": key, "value": null, "not found": true }).to_string(),
        )),
    }
}

fn memory_delete(server: &mut RvfMcpServer, args: &Value) -> Result<ToolResult, String> {
    let key = match require_str(args, "key") {
        Ok(k) => k.to_string(),
        Err(e) => return Ok(e),
    };

    if let Some((id, _, _)) = server.key_index.remove(&key) {
        server
            .memory_store
            .delete_memories(&[id])
            .map_err(|e| format!("delete: {e}"))?;
        Ok(ToolResult::ok(
            json!({ "deleted": true, "key": key }).to_string(),
        ))
    } else {
        Ok(ToolResult::ok(
            json!({ "deleted": false, "key": key, "reason": "not found" }).to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Pattern tools
// ---------------------------------------------------------------------------

fn pattern_store(server: &mut RvfMcpServer, args: &Value) -> Result<ToolResult, String> {
    let task = match require_str(args, "task") {
        Ok(t) => t.to_string(),
        Err(e) => return Ok(e),
    };
    let embedding = match parse_embedding(args) {
        Ok(e) => e,
        Err(e) => return Ok(e),
    };

    let reward = args.get("reward").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let success = args.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
    let critique = args
        .get("critique")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let pattern = MemoryPattern {
        id: 0,
        task,
        reward,
        success,
        critique,
        embedding,
    };

    let id = server
        .pattern_store
        .store_pattern(pattern)
        .map_err(|e| format!("store_pattern: {e}"))?;

    Ok(ToolResult::ok(json!({ "id": id, "stored": true }).to_string()))
}

fn pattern_search(server: &mut RvfMcpServer, args: &Value) -> Result<ToolResult, String> {
    let embedding = match parse_embedding(args) {
        Ok(e) => e,
        Err(e) => return Ok(e),
    };
    let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let min_reward = args.get("min_reward").and_then(|v| v.as_f64()).map(|v| v as f32);

    let results = server
        .pattern_store
        .search_patterns(&embedding, k, min_reward)
        .map_err(|e| format!("search_patterns: {e}"))?;

    let output: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "task": r.task,
                "reward": r.reward,
                "success": r.success,
                "critique": r.critique,
                "distance": r.distance,
            })
        })
        .collect();

    Ok(ToolResult::ok(serde_json::to_string(&output).unwrap()))
}

// ---------------------------------------------------------------------------
// Witness tools
// ---------------------------------------------------------------------------

fn witness_log(server: &mut RvfMcpServer, args: &Value) -> Result<ToolResult, String> {
    let action = match require_str(args, "action") {
        Ok(a) => a,
        Err(e) => return Ok(e),
    };

    let details: Vec<&str> = args
        .get("details")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    server
        .memory_store
        .witness()
        .record_action(action, &details)
        .map_err(|e| format!("witness: {e}"))?;

    Ok(ToolResult::ok(
        json!({ "recorded": true, "action": action }).to_string(),
    ))
}

fn witness_verify(server: &mut RvfMcpServer, _args: &Value) -> Result<ToolResult, String> {
    match server.memory_store.witness_ref().verify() {
        Ok(()) => Ok(ToolResult::ok(
            json!({ "valid": true, "status": "ok" }).to_string(),
        )),
        Err(e) => Ok(ToolResult::ok(
            json!({ "valid": false, "error": format!("{e}") }).to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Coordination tools
// ---------------------------------------------------------------------------

fn coordination_state(server: &mut RvfMcpServer, args: &Value) -> Result<ToolResult, String> {
    let agent_id = match require_str(args, "agent_id") {
        Ok(a) => a,
        Err(e) => return Ok(e),
    };
    let key = match require_str(args, "key") {
        Ok(k) => k,
        Err(e) => return Ok(e),
    };
    let value = match require_str(args, "value") {
        Ok(v) => v,
        Err(e) => return Ok(e),
    };

    server
        .memory_store
        .coordination()
        .record_state(agent_id, key, value)
        .map_err(|e| format!("coordination: {e}"))?;

    Ok(ToolResult::ok(
        json!({ "recorded": true, "agent_id": agent_id, "key": key }).to_string(),
    ))
}

fn coordination_vote(server: &mut RvfMcpServer, args: &Value) -> Result<ToolResult, String> {
    let topic = match require_str(args, "topic") {
        Ok(t) => t,
        Err(e) => return Ok(e),
    };
    let agent_id = match require_str(args, "agent_id") {
        Ok(a) => a,
        Err(e) => return Ok(e),
    };
    let vote = args.get("vote").and_then(|v| v.as_bool()).unwrap_or(false);

    server
        .memory_store
        .coordination()
        .record_consensus_vote(topic, agent_id, vote)
        .map_err(|e| format!("vote: {e}"))?;

    Ok(ToolResult::ok(
        json!({ "recorded": true, "topic": topic, "agent_id": agent_id, "vote": vote }).to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

fn store_status(server: &mut RvfMcpServer, _args: &Value) -> Result<ToolResult, String> {
    let mem_status = server.memory_store.status();
    let pat_stats = server.pattern_store.stats();

    Ok(ToolResult::ok(
        json!({
            "memory_vectors": mem_status.total_vectors,
            "pattern_vectors": pat_stats.vector_count,
            "total_patterns": pat_stats.total_patterns,
            "successful_patterns": pat_stats.successful_patterns,
            "avg_reward": pat_stats.avg_reward,
            "data_dir": server.config.data_dir.to_string_lossy(),
        })
        .to_string(),
    ))
}
