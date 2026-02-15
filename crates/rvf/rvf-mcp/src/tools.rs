//! Tool definitions for the RVF MCP server.

use serde_json::{json, Value};

/// A tool definition exposed via MCP tools/list.
#[derive(Clone, Debug)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

pub fn all_tools() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "rvf_memory_store".into(),
            description: "Store a memory entry with key, value, optional namespace, and embedding vector".into(),
            input_schema: json!({
                "type": "object",
                "required": ["key", "value", "embedding"],
                "properties": {
                    "key": { "type": "string", "description": "Unique memory key" },
                    "value": { "type": "string", "description": "Memory value/content" },
                    "namespace": { "type": "string", "description": "Optional namespace for isolation" },
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Embedding vector" }
                }
            }),
        },
        ToolDef {
            name: "rvf_memory_search".into(),
            description: "Search memories by embedding similarity, returns k nearest neighbors".into(),
            input_schema: json!({
                "type": "object",
                "required": ["embedding", "k"],
                "properties": {
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Query embedding" },
                    "k": { "type": "integer", "description": "Number of results to return" }
                }
            }),
        },
        ToolDef {
            name: "rvf_memory_get".into(),
            description: "Get a specific memory entry by key".into(),
            input_schema: json!({
                "type": "object",
                "required": ["key"],
                "properties": {
                    "key": { "type": "string", "description": "Memory key to retrieve" }
                }
            }),
        },
        ToolDef {
            name: "rvf_memory_delete".into(),
            description: "Delete a memory entry by key".into(),
            input_schema: json!({
                "type": "object",
                "required": ["key"],
                "properties": {
                    "key": { "type": "string", "description": "Memory key to delete" }
                }
            }),
        },
        ToolDef {
            name: "rvf_pattern_store".into(),
            description: "Store a learning pattern with task, reward, success flag, critique, and embedding".into(),
            input_schema: json!({
                "type": "object",
                "required": ["task", "reward", "embedding"],
                "properties": {
                    "task": { "type": "string", "description": "Task description" },
                    "reward": { "type": "number", "description": "Reward score (0.0-1.0)" },
                    "success": { "type": "boolean", "description": "Whether the pattern was successful" },
                    "critique": { "type": "string", "description": "Self-critique notes" },
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "State embedding" }
                }
            }),
        },
        ToolDef {
            name: "rvf_pattern_search".into(),
            description: "Search patterns by embedding similarity with optional min_reward filter".into(),
            input_schema: json!({
                "type": "object",
                "required": ["embedding", "k"],
                "properties": {
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Query embedding" },
                    "k": { "type": "integer", "description": "Number of results" },
                    "min_reward": { "type": "number", "description": "Minimum reward threshold" }
                }
            }),
        },
        ToolDef {
            name: "rvf_witness_log".into(),
            description: "Record an action in the tamper-evident witness chain".into(),
            input_schema: json!({
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": { "type": "string", "description": "Action type (e.g. 'decision', 'search')" },
                    "details": { "type": "array", "items": { "type": "string" }, "description": "Action details" }
                }
            }),
        },
        ToolDef {
            name: "rvf_witness_verify".into(),
            description: "Verify the integrity of the witness chain audit trail".into(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDef {
            name: "rvf_coordination_state".into(),
            description: "Record agent coordination state (agent_id, key, value)".into(),
            input_schema: json!({
                "type": "object",
                "required": ["agent_id", "key", "value"],
                "properties": {
                    "agent_id": { "type": "string", "description": "Agent identifier" },
                    "key": { "type": "string", "description": "State key" },
                    "value": { "type": "string", "description": "State value" }
                }
            }),
        },
        ToolDef {
            name: "rvf_coordination_vote".into(),
            description: "Record a consensus vote for multi-agent coordination".into(),
            input_schema: json!({
                "type": "object",
                "required": ["topic", "agent_id", "vote"],
                "properties": {
                    "topic": { "type": "string", "description": "Vote topic (e.g. 'leader-election')" },
                    "agent_id": { "type": "string", "description": "Voting agent" },
                    "vote": { "type": "boolean", "description": "Vote value" }
                }
            }),
        },
        ToolDef {
            name: "rvf_store_status".into(),
            description: "Get current store status including vector counts and file info".into(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        },
    ]
}
