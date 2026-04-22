//! MCP tools for the shared brain

use crate::client::BrainClient;
use crate::embed::BrainEmbedder;
use crate::pipeline::BrainPipeline;
use crate::types::*;
use tracing::info;

/// Error type for brain operations
#[derive(Debug, thiserror::Error)]
pub enum BrainError {
    #[error("Client error: {0}")]
    Client(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Pipeline error: {0}")]
    Pipeline(String),
}

impl BrainError {
    pub fn code(&self) -> i32 {
        match self {
            BrainError::Client(_) => -32001,
            BrainError::InvalidRequest(_) => -32602,
            BrainError::NotFound(_) => -32002,
            BrainError::Pipeline(_) => -32003,
        }
    }
}

/// Brain tools handler
pub struct McpBrainTools {
    client: BrainClient,
    pipeline: BrainPipeline,
    embedder: std::sync::Mutex<BrainEmbedder>,
}

impl McpBrainTools {
    pub fn new() -> Self {
        Self {
            client: BrainClient::new(),
            pipeline: BrainPipeline::new(),
            embedder: std::sync::Mutex::new(BrainEmbedder::new()),
        }
    }

    pub fn with_backend_url(url: String) -> Self {
        Self {
            client: BrainClient::with_url(url),
            pipeline: BrainPipeline::new(),
            embedder: std::sync::Mutex::new(BrainEmbedder::new()),
        }
    }

    /// Get list of all tools (core + Brainpedia + WASM)
    pub fn list_tools() -> Vec<McpTool> {
        vec![
            McpTool {
                name: "brain_share".to_string(),
                description: "Share a learning with the collective brain. Knowledge is PII-stripped, embedded, signed, and stored as an RVF cognitive container.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "category": { "type": "string", "enum": ["architecture", "pattern", "solution", "convention", "security", "performance", "tooling", "debug"], "description": "Knowledge category" },
                        "title": { "type": "string", "description": "Short title (max 200 chars)" },
                        "content": { "type": "string", "description": "Knowledge content (max 10000 chars)" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags (max 10, each max 30 chars)" },
                        "code_snippet": { "type": "string", "description": "Optional code snippet" }
                    },
                    "required": ["category", "title", "content"]
                }),
            },
            McpTool {
                name: "brain_search".to_string(),
                description: "Semantic search across shared knowledge. Returns ranked results with quality scores and drift warnings.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query" },
                        "category": { "type": "string", "description": "Filter by category" },
                        "tags": { "type": "string", "description": "Comma-separated tags to filter" },
                        "limit": { "type": "integer", "description": "Max results (default 10)" },
                        "min_quality": { "type": "number", "description": "Minimum quality score (0-1)" }
                    },
                    "required": ["query"]
                }),
            },
            McpTool {
                name: "brain_get".to_string(),
                description: "Retrieve a specific memory with full provenance including witness chain and quality history.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Memory ID (UUID)" }
                    },
                    "required": ["id"]
                }),
            },
            McpTool {
                name: "brain_vote".to_string(),
                description: "Vote on a memory's quality (Bayesian update). Affects ranking and contributor reputation.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Memory ID" },
                        "direction": { "type": "string", "enum": ["up", "down"], "description": "Vote direction" }
                    },
                    "required": ["id", "direction"]
                }),
            },
            McpTool {
                name: "brain_transfer".to_string(),
                description: "Apply learned priors from one knowledge domain to another. Uses Meta Thompson Sampling with dampened transfer.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "source_domain": { "type": "string", "description": "Source knowledge domain" },
                        "target_domain": { "type": "string", "description": "Target knowledge domain" }
                    },
                    "required": ["source_domain", "target_domain"]
                }),
            },
            McpTool {
                name: "brain_drift".to_string(),
                description: "Check if shared knowledge has drifted from expected distributions. Reports coefficient of variation and trend.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "domain": { "type": "string", "description": "Domain to check (optional, default: global)" },
                        "since": { "type": "string", "description": "ISO timestamp to check from" }
                    }
                }),
            },
            McpTool {
                name: "brain_partition".to_string(),
                description: "Get knowledge partitioned by mincut topology. Shows emergent knowledge clusters with coherence scores.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "domain": { "type": "string", "description": "Domain to partition" },
                        "min_cluster_size": { "type": "integer", "description": "Minimum memories per cluster" }
                    }
                }),
            },
            McpTool {
                name: "brain_list".to_string(),
                description: "List recent shared memories, optionally filtered by category and quality.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "category": { "type": "string", "description": "Filter by category" },
                        "limit": { "type": "integer", "description": "Max results (default 20)" },
                        "min_quality": { "type": "number", "description": "Minimum quality score" }
                    }
                }),
            },
            McpTool {
                name: "brain_delete".to_string(),
                description: "Delete your own contribution. Only the original contributor can delete.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Memory ID to delete" }
                    },
                    "required": ["id"]
                }),
            },
            McpTool {
                name: "brain_status".to_string(),
                description: "Get system health: memory count, contributor count, graph topology, drift status, and quality metrics.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            McpTool {
                name: "brain_sync".to_string(),
                description: "Sync local MicroLoRA weights with the shared brain. Downloads consensus weights, applies locally, exports local deltas, submits to server for federated aggregation.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "direction": { "type": "string", "enum": ["pull", "push", "both"], "description": "Sync direction (default: both)" }
                    }
                }),
            },
            // ── Brainpedia (ADR-062) ───────────────────────────────────
            McpTool {
                name: "brain_page_create".to_string(),
                description: "Create a new Brainpedia page (Draft). Requires reputation >= 0.5. Pages go through Draft → Canonical lifecycle with evidence gating.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "category": { "type": "string", "enum": ["architecture", "pattern", "solution", "convention", "security", "performance", "tooling", "debug"], "description": "Knowledge category" },
                        "title": { "type": "string", "description": "Page title (max 200 chars)" },
                        "content": { "type": "string", "description": "Page content (max 10000 chars)" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags (max 10)" },
                        "code_snippet": { "type": "string", "description": "Optional code snippet" },
                        "evidence_links": { "type": "array", "description": "Initial evidence links" }
                    },
                    "required": ["category", "title", "content"]
                }),
            },
            McpTool {
                name: "brain_page_get".to_string(),
                description: "Get a Brainpedia page with its full delta log, evidence links, and promotion status.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Page ID (UUID)" }
                    },
                    "required": ["id"]
                }),
            },
            McpTool {
                name: "brain_page_delta".to_string(),
                description: "Submit a delta (correction, extension, or deprecation) to an existing Brainpedia page. Requires evidence links.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "page_id": { "type": "string", "description": "Page ID (UUID)" },
                        "delta_type": { "type": "string", "enum": ["correction", "extension", "evidence", "deprecation"], "description": "Type of delta" },
                        "content_diff": { "type": "object", "description": "Content changes" },
                        "evidence_links": { "type": "array", "description": "Supporting evidence" }
                    },
                    "required": ["page_id", "delta_type", "content_diff"]
                }),
            },
            McpTool {
                name: "brain_page_deltas".to_string(),
                description: "List all deltas for a Brainpedia page, showing its modification history.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "page_id": { "type": "string", "description": "Page ID (UUID)" }
                    },
                    "required": ["page_id"]
                }),
            },
            McpTool {
                name: "brain_page_evidence".to_string(),
                description: "Add evidence to a Brainpedia page. Evidence types: test_pass, build_success, metric_improval, peer_review.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "page_id": { "type": "string", "description": "Page ID (UUID)" },
                        "evidence": { "type": "object", "description": "Evidence link with type, description, and verification data" }
                    },
                    "required": ["page_id", "evidence"]
                }),
            },
            McpTool {
                name: "brain_page_promote".to_string(),
                description: "Promote a Draft page to Canonical. Requires: quality >= 0.7, observations >= 5, evidence >= 3 from >= 2 contributors.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "page_id": { "type": "string", "description": "Page ID (UUID)" }
                    },
                    "required": ["page_id"]
                }),
            },
            // ── WASM Executable Nodes (ADR-063) ────────────────────────
            McpTool {
                name: "brain_node_list".to_string(),
                description: "List all published (non-revoked) WASM executable nodes in the brain.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            McpTool {
                name: "brain_node_publish".to_string(),
                description: "Publish a new WASM executable node. V1 ABI requires: memory, malloc, feature_extract_dim, feature_extract exports. Includes conformance test vectors.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Node ID (e.g., 'my-feature-extractor')" },
                        "name": { "type": "string", "description": "Human-readable name" },
                        "version": { "type": "string", "description": "Semver version" },
                        "dim": { "type": "integer", "description": "Output dimension (default 128)" },
                        "exports": { "type": "array", "items": { "type": "string" }, "description": "WASM exports" },
                        "interface": { "type": "object", "description": "Interface specification" },
                        "conformance": { "type": "array", "description": "Conformance test vectors" },
                        "wasm_bytes": { "type": "string", "description": "Base64-encoded WASM binary" },
                        "signature": { "type": "string", "description": "Ed25519 signature (hex)" }
                    },
                    "required": ["id", "name", "version", "exports", "wasm_bytes", "signature"]
                }),
            },
            McpTool {
                name: "brain_node_get".to_string(),
                description: "Get WASM node metadata and conformance test vectors.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Node ID" }
                    },
                    "required": ["id"]
                }),
            },
            McpTool {
                name: "brain_node_wasm".to_string(),
                description: "Download WASM binary for a node. Returns base64-encoded bytes.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Node ID" }
                    },
                    "required": ["id"]
                }),
            },
            McpTool {
                name: "brain_node_revoke".to_string(),
                description: "Revoke a WASM node (original publisher only). Marks as revoked but retains bytes for forensic analysis.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Node ID to revoke" }
                    },
                    "required": ["id"]
                }),
            },
        ]
    }

    /// Handle a tool call
    pub async fn call_tool(&self, call: McpToolCall) -> Result<McpToolResult, BrainError> {
        info!("Calling tool: {}", call.name);
        match call.name.as_str() {
            "brain_share" => self.brain_share(call.arguments).await,
            "brain_search" => self.brain_search(call.arguments).await,
            "brain_get" => self.brain_get(call.arguments).await,
            "brain_vote" => self.brain_vote(call.arguments).await,
            "brain_transfer" => self.brain_transfer(call.arguments).await,
            "brain_drift" => self.brain_drift(call.arguments).await,
            "brain_partition" => self.brain_partition(call.arguments).await,
            "brain_list" => self.brain_list(call.arguments).await,
            "brain_delete" => self.brain_delete(call.arguments).await,
            "brain_status" => self.brain_status(call.arguments).await,
            "brain_sync" => self.brain_sync(call.arguments).await,
            // Brainpedia (ADR-062)
            "brain_page_create" => self.brain_page_create(call.arguments).await,
            "brain_page_get" => self.brain_page_get(call.arguments).await,
            "brain_page_delta" => self.brain_page_delta(call.arguments).await,
            "brain_page_deltas" => self.brain_page_deltas(call.arguments).await,
            "brain_page_evidence" => self.brain_page_evidence(call.arguments).await,
            "brain_page_promote" => self.brain_page_promote(call.arguments).await,
            // WASM Executable Nodes (ADR-063)
            "brain_node_list" => self.brain_node_list(call.arguments).await,
            "brain_node_publish" => self.brain_node_publish(call.arguments).await,
            "brain_node_get" => self.brain_node_get(call.arguments).await,
            "brain_node_wasm" => self.brain_node_wasm(call.arguments).await,
            "brain_node_revoke" => self.brain_node_revoke(call.arguments).await,
            _ => Err(BrainError::InvalidRequest(format!(
                "Unknown tool: {}",
                call.name
            ))),
        }
    }

    async fn brain_share(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let category = args
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("pattern");
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("title required".into()))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("content required".into()))?;
        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let code_snippet = args
            .get("code_snippet")
            .and_then(|v| v.as_str())
            .map(String::from);

        // PII strip all user-provided text
        let clean_title = self.pipeline.strip_pii(title);
        let clean_content = self.pipeline.strip_pii(content);
        let clean_tags: Vec<String> = tags.iter().map(|t| self.pipeline.strip_pii(t)).collect();
        let clean_snippet = code_snippet.as_deref().map(|s| self.pipeline.strip_pii(s));

        // Safety check: reject if PII still detected after stripping
        if self.pipeline.contains_pii(&clean_title) {
            return Err(BrainError::Pipeline(
                "PII detected in title after stripping".into(),
            ));
        }
        if self.pipeline.contains_pii(&clean_content) {
            return Err(BrainError::Pipeline(
                "PII detected in content after stripping".into(),
            ));
        }
        for tag in &clean_tags {
            if self.pipeline.contains_pii(tag) {
                return Err(BrainError::Pipeline(
                    "PII detected in tags after stripping".into(),
                ));
            }
        }
        if let Some(ref s) = clean_snippet {
            if self.pipeline.contains_pii(s) {
                return Err(BrainError::Pipeline(
                    "PII detected in code_snippet after stripping".into(),
                ));
            }
        }

        // Generate embedding via structured hash + MicroLoRA
        let _embedding = if let Ok(mut emb) = self.embedder.lock() {
            emb.embed(&clean_content)
        } else {
            crate::embed::generate_embedding(&clean_content)
        };

        // Build witness chain: pii_strip -> embed -> share
        let mut chain = crate::pipeline::WitnessChain::new();
        chain.append("pii_strip");
        chain.append("embed");
        chain.append("share");
        let _witness_hash = chain.finalize();

        let result = self
            .client
            .share(
                category,
                &clean_title,
                &clean_content,
                &clean_tags,
                clean_snippet.as_deref(),
            )
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::to_value(result).unwrap_or_default(),
        })
    }

    async fn brain_search(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("query required".into()))?;
        let category = args.get("category").and_then(|v| v.as_str());
        let tags = args.get("tags").and_then(|v| v.as_str());
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let min_quality = args.get("min_quality").and_then(|v| v.as_f64());

        // Generate query embedding via structured hash + MicroLoRA
        let _query_embedding = if let Ok(mut emb) = self.embedder.lock() {
            emb.embed(query)
        } else {
            crate::embed::generate_embedding(query)
        };

        let results = self
            .client
            .search(query, category, tags, limit, min_quality)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::to_value(results).unwrap_or_default(),
        })
    }

    async fn brain_get(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("id required".into()))?;

        let memory = self
            .client
            .get(id)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::to_value(memory).unwrap_or_default(),
        })
    }

    async fn brain_vote(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("id required".into()))?;
        let direction = args
            .get("direction")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("direction required".into()))?;

        let result = self
            .client
            .vote(id, direction)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::to_value(result).unwrap_or_default(),
        })
    }

    async fn brain_transfer(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let source = args
            .get("source_domain")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("source_domain required".into()))?;
        let target = args
            .get("target_domain")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("target_domain required".into()))?;

        let result = self
            .client
            .transfer(source, target)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::to_value(result).unwrap_or_default(),
        })
    }

    async fn brain_drift(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let domain = args.get("domain").and_then(|v| v.as_str());
        let since = args.get("since").and_then(|v| v.as_str());

        let report = self
            .client
            .drift(domain, since)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::to_value(report).unwrap_or_default(),
        })
    }

    async fn brain_partition(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let domain = args.get("domain").and_then(|v| v.as_str());
        let min_size = args
            .get("min_cluster_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let result = self
            .client
            .partition(domain, min_size)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::to_value(result).unwrap_or_default(),
        })
    }

    async fn brain_list(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let category = args.get("category").and_then(|v| v.as_str());
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let results = self
            .client
            .list(category, limit)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::to_value(results).unwrap_or_default(),
        })
    }

    async fn brain_delete(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("id required".into()))?;

        self.client
            .delete(id)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::json!({"deleted": true, "id": id}),
        })
    }

    async fn brain_status(&self, _args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let status = self
            .client
            .status()
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::to_value(status).unwrap_or_default(),
        })
    }

    async fn brain_sync(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let direction = args
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("both");
        let mut pulled = false;
        let mut pushed = false;

        // Pull: download consensus weights from server
        if direction == "pull" || direction == "both" {
            match self.client.lora_latest().await {
                Ok(Some(weights)) => {
                    if let Ok(mut emb) = self.embedder.lock() {
                        emb.import_consensus_weights(weights);
                        pulled = true;
                    }
                }
                Ok(None) => {
                    // No consensus weights available yet — that's fine
                }
                Err(e) => {
                    info!("Failed to pull consensus weights: {e}");
                }
            }
        }

        // Push: export local weights and submit to server
        if direction == "push" || direction == "both" {
            let local_weights = if let Ok(emb) = self.embedder.lock() {
                emb.export_local_weights()
            } else {
                None
            };

            if let Some(mut weights) = local_weights {
                weights.clip();
                if weights.validate().is_ok() {
                    match self.client.lora_submit(&weights).await {
                        Ok(_) => {
                            pushed = true;
                        }
                        Err(e) => {
                            info!("Failed to push local weights: {e}");
                        }
                    }
                }
            }
        }

        let embed_count = if let Ok(emb) = self.embedder.lock() {
            emb.embed_count()
        } else {
            0
        };

        Ok(McpToolResult::Success {
            content: serde_json::json!({
                "pulled": pulled,
                "pushed": pushed,
                "direction": direction,
                "local_embed_count": embed_count,
            }),
        })
    }

    // ── Brainpedia (ADR-062) ─────────────────────────────────────────

    async fn brain_page_create(
        &self,
        args: serde_json::Value,
    ) -> Result<McpToolResult, BrainError> {
        let category = args
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("pattern");
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("title required".into()))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("content required".into()))?;
        let tags: Vec<String> = args
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let code_snippet = args.get("code_snippet").and_then(|v| v.as_str());
        let evidence_links = args
            .get("evidence_links")
            .cloned()
            .unwrap_or(serde_json::json!([]));

        let clean_title = self.pipeline.strip_pii(title);
        let clean_content = self.pipeline.strip_pii(content);
        let clean_tags: Vec<String> = tags.iter().map(|t| self.pipeline.strip_pii(t)).collect();

        let embedding = if let Ok(mut emb) = self.embedder.lock() {
            emb.embed(&clean_content)
        } else {
            crate::embed::generate_embedding(&clean_content)
        };

        let mut chain = crate::pipeline::WitnessChain::new();
        chain.append("pii_strip");
        chain.append("embed");
        chain.append("page_create");
        let witness_hash = chain.finalize();

        let body = serde_json::json!({
            "category": category,
            "title": clean_title,
            "content": clean_content,
            "tags": clean_tags,
            "code_snippet": code_snippet,
            "embedding": embedding,
            "evidence_links": evidence_links,
            "witness_hash": hex::encode(witness_hash),
        });

        let result = self
            .client
            .create_page(&body)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success { content: result })
    }

    async fn brain_page_get(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("id required".into()))?;

        let result = self
            .client
            .get_page(id)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success { content: result })
    }

    async fn brain_page_delta(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let page_id = args
            .get("page_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("page_id required".into()))?;
        let delta_type = args
            .get("delta_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("delta_type required".into()))?;
        let content_diff = args
            .get("content_diff")
            .cloned()
            .ok_or_else(|| BrainError::InvalidRequest("content_diff required".into()))?;
        let evidence_links = args
            .get("evidence_links")
            .cloned()
            .unwrap_or(serde_json::json!([]));

        let mut chain = crate::pipeline::WitnessChain::new();
        chain.append("delta_submit");
        let witness_hash = chain.finalize();

        let body = serde_json::json!({
            "delta_type": delta_type,
            "content_diff": content_diff,
            "evidence_links": evidence_links,
            "witness_hash": hex::encode(witness_hash),
        });

        let result = self
            .client
            .submit_delta(page_id, &body)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success { content: result })
    }

    async fn brain_page_deltas(
        &self,
        args: serde_json::Value,
    ) -> Result<McpToolResult, BrainError> {
        let page_id = args
            .get("page_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("page_id required".into()))?;

        let result = self
            .client
            .list_deltas(page_id)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success { content: result })
    }

    async fn brain_page_evidence(
        &self,
        args: serde_json::Value,
    ) -> Result<McpToolResult, BrainError> {
        let page_id = args
            .get("page_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("page_id required".into()))?;
        let evidence = args
            .get("evidence")
            .cloned()
            .ok_or_else(|| BrainError::InvalidRequest("evidence required".into()))?;

        let body = serde_json::json!({ "evidence": evidence });

        let result = self
            .client
            .add_evidence(page_id, &body)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success { content: result })
    }

    async fn brain_page_promote(
        &self,
        args: serde_json::Value,
    ) -> Result<McpToolResult, BrainError> {
        let page_id = args
            .get("page_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("page_id required".into()))?;

        let result = self
            .client
            .promote_page(page_id)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success { content: result })
    }

    // ── WASM Executable Nodes (ADR-063) ───────────────────────────────

    async fn brain_node_list(&self, _args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let result = self
            .client
            .list_nodes()
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success { content: result })
    }

    async fn brain_node_publish(
        &self,
        args: serde_json::Value,
    ) -> Result<McpToolResult, BrainError> {
        let result = self
            .client
            .publish_node(&args)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success { content: result })
    }

    async fn brain_node_get(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("id required".into()))?;

        let result = self
            .client
            .get_node(id)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success { content: result })
    }

    async fn brain_node_wasm(&self, args: serde_json::Value) -> Result<McpToolResult, BrainError> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("id required".into()))?;

        let bytes = self
            .client
            .get_node_wasm(id)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);

        Ok(McpToolResult::Success {
            content: serde_json::json!({
                "id": id,
                "wasm_bytes_b64": b64,
                "size_bytes": bytes.len(),
            }),
        })
    }

    async fn brain_node_revoke(
        &self,
        args: serde_json::Value,
    ) -> Result<McpToolResult, BrainError> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| BrainError::InvalidRequest("id required".into()))?;

        self.client
            .revoke_node(id)
            .await
            .map_err(|e| BrainError::Client(e.to_string()))?;

        Ok(McpToolResult::Success {
            content: serde_json::json!({"revoked": true, "id": id}),
        })
    }
}

impl Default for McpBrainTools {
    fn default() -> Self {
        Self::new()
    }
}
