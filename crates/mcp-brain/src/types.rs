//! Types for MCP Brain server

use serde::{Deserialize, Serialize};

/// Brain memory categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BrainCategory {
    Architecture,
    Pattern,
    Solution,
    Convention,
    Security,
    Performance,
    Tooling,
    Debug,
    Custom(String),
}

/// Brain memory (local representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainMemory {
    pub id: String,
    pub category: BrainCategory,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub code_snippet: Option<String>,
    pub quality_score: f64,
    pub contributor_id: String,
    pub partition_id: Option<u32>,
    pub created_at: String,
    pub updated_at: String,
}

/// Drift report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub domain: Option<String>,
    pub coefficient_of_variation: f64,
    pub is_drifting: bool,
    pub delta_sparsity: f64,
    pub trend: String,
    pub suggested_action: String,
    pub window_size: usize,
}

/// Partition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionResult {
    pub clusters: Vec<KnowledgeCluster>,
    pub cut_value: f64,
    pub total_memories: usize,
}

/// Knowledge cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeCluster {
    pub id: u32,
    pub memory_ids: Vec<String>,
    pub dominant_category: BrainCategory,
    pub size: usize,
    pub coherence: f64,
}

/// Status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusInfo {
    pub total_memories: usize,
    pub total_contributors: usize,
    pub graph_nodes: usize,
    pub graph_edges: usize,
    pub cluster_count: usize,
    pub avg_quality: f64,
    pub drift_status: String,
}

/// Transfer result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferResult {
    pub source_domain: String,
    pub target_domain: String,
    pub acceleration_factor: f64,
    pub transfer_success: bool,
    pub message: String,
}

/// Vote direction
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteDirection {
    Up,
    Down,
}

/// Quality score (Bayesian Beta distribution)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaParams {
    pub alpha: f64,
    pub beta: f64,
}

// ============== JSON-RPC Protocol Types (cloned from mcp-gate) ==============

/// MCP Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// MCP Tool call request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// MCP Tool result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpToolResult {
    Success { content: serde_json::Value },
    Error { error: String },
}

/// JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: serde_json::Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}
