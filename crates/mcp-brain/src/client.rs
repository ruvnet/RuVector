//! HTTPS client to the Cloud Run brain backend

use crate::embed::{generate_embedding, LoraWeights};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Server error ({status}): {message}")]
    Server { status: u16, message: String },
}

/// Client for the brain.ruv.io backend
pub struct BrainClient {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
}

impl BrainClient {
    pub fn new() -> Self {
        let base_url = std::env::var("BRAIN_URL")
            .unwrap_or_else(|_| "https://ruvbrain-875130704813.us-central1.run.app".to_string());
        let api_key = std::env::var("BRAIN_API_KEY").unwrap_or_else(|_| "anonymous".to_string());
        Self {
            base_url,
            api_key,
            http: reqwest::Client::new(),
        }
    }

    pub fn with_url(url: String) -> Self {
        let api_key = std::env::var("BRAIN_API_KEY").unwrap_or_else(|_| "anonymous".to_string());
        Self {
            base_url: url,
            api_key,
            http: reqwest::Client::new(),
        }
    }

    /// Share a memory
    pub async fn share(
        &self,
        category: &str,
        title: &str,
        content: &str,
        tags: &[String],
        code_snippet: Option<&str>,
    ) -> Result<serde_json::Value, ClientError> {
        let body = serde_json::json!({
            "category": category,
            "title": title,
            "content": content,
            "tags": tags,
            "code_snippet": code_snippet,
            "embedding": generate_embedding(content),
            "witness_hash": hex::encode(sha3_hash(content.as_bytes())),
        });

        self.post("/v1/memories", &body).await
    }

    /// Search memories
    pub async fn search(
        &self,
        query: &str,
        category: Option<&str>,
        tags: Option<&str>,
        limit: Option<usize>,
        min_quality: Option<f64>,
    ) -> Result<serde_json::Value, ClientError> {
        let mut params = vec![("q", query.to_string())];
        if let Some(c) = category {
            params.push(("category", c.to_string()));
        }
        if let Some(t) = tags {
            params.push(("tags", t.to_string()));
        }
        if let Some(l) = limit {
            params.push(("limit", l.to_string()));
        }
        if let Some(q) = min_quality {
            params.push(("min_quality", q.to_string()));
        }

        self.get_with_params("/v1/memories/search", &params).await
    }

    /// Get a memory by ID
    pub async fn get(&self, id: &str) -> Result<serde_json::Value, ClientError> {
        self.get_path(&format!("/v1/memories/{id}")).await
    }

    /// Vote on a memory
    pub async fn vote(&self, id: &str, direction: &str) -> Result<serde_json::Value, ClientError> {
        let body = serde_json::json!({ "direction": direction });
        self.post(&format!("/v1/memories/{id}/vote"), &body).await
    }

    /// Transfer knowledge between domains
    pub async fn transfer(
        &self,
        source: &str,
        target: &str,
    ) -> Result<serde_json::Value, ClientError> {
        let body = serde_json::json!({
            "source_domain": source,
            "target_domain": target,
        });
        self.post("/v1/transfer", &body).await
    }

    /// Get drift report
    pub async fn drift(
        &self,
        domain: Option<&str>,
        since: Option<&str>,
    ) -> Result<serde_json::Value, ClientError> {
        let mut params = Vec::new();
        if let Some(d) = domain {
            params.push(("domain", d.to_string()));
        }
        if let Some(s) = since {
            params.push(("since", s.to_string()));
        }
        self.get_with_params("/v1/drift", &params).await
    }

    /// Get partition topology
    pub async fn partition(
        &self,
        domain: Option<&str>,
        min_size: Option<usize>,
    ) -> Result<serde_json::Value, ClientError> {
        let mut params = Vec::new();
        if let Some(d) = domain {
            params.push(("domain", d.to_string()));
        }
        if let Some(s) = min_size {
            params.push(("min_cluster_size", s.to_string()));
        }
        self.get_with_params("/v1/partition", &params).await
    }

    /// List memories
    pub async fn list(
        &self,
        category: Option<&str>,
        limit: Option<usize>,
    ) -> Result<serde_json::Value, ClientError> {
        let mut params = Vec::new();
        if let Some(c) = category {
            params.push(("category", c.to_string()));
        }
        if let Some(l) = limit {
            params.push(("limit", l.to_string()));
        }
        self.get_with_params("/v1/memories/list", &params).await
    }

    /// Delete a memory
    pub async fn delete(&self, id: &str) -> Result<(), ClientError> {
        let url = format!("{}/v1/memories/{id}", self.base_url);
        let resp = self
            .http
            .delete(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| ClientError::Http(e.to_string()))?;

        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            Err(ClientError::Server {
                status,
                message: msg,
            })
        }
    }

    /// Get system status
    pub async fn status(&self) -> Result<serde_json::Value, ClientError> {
        self.get_path("/v1/status").await
    }

    /// Get latest consensus LoRA weights from server
    pub async fn lora_latest(&self) -> Result<Option<LoraWeights>, ClientError> {
        let result = self.get_path("/v1/lora/latest").await;
        match result {
            Ok(val) => {
                // Server returns {"weights": null} if no consensus yet
                if val.get("weights").map_or(true, |w| w.is_null()) {
                    return Ok(None);
                }
                let weights: LoraWeights =
                    serde_json::from_value(val.get("weights").cloned().unwrap_or_default())
                        .map_err(|e| ClientError::Serialization(e.to_string()))?;
                Ok(Some(weights))
            }
            Err(ClientError::Server { status: 404, .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Submit local LoRA weights for federated aggregation
    pub async fn lora_submit(
        &self,
        weights: &LoraWeights,
    ) -> Result<serde_json::Value, ClientError> {
        let body =
            serde_json::to_value(weights).map_err(|e| ClientError::Serialization(e.to_string()))?;
        self.post("/v1/lora/submit", &body).await
    }

    // ---- Brainpedia (ADR-062) ----

    /// Create a Brainpedia page
    pub async fn create_page(
        &self,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.post("/v1/pages", body).await
    }

    /// Get a Brainpedia page with its delta log and evidence
    pub async fn get_page(&self, id: &str) -> Result<serde_json::Value, ClientError> {
        self.get_path(&format!("/v1/pages/{id}")).await
    }

    /// Submit a delta to a page
    pub async fn submit_delta(
        &self,
        page_id: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.post(&format!("/v1/pages/{page_id}/deltas"), body)
            .await
    }

    /// List deltas for a page
    pub async fn list_deltas(&self, page_id: &str) -> Result<serde_json::Value, ClientError> {
        self.get_path(&format!("/v1/pages/{page_id}/deltas")).await
    }

    /// Add evidence to a page
    pub async fn add_evidence(
        &self,
        page_id: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.post(&format!("/v1/pages/{page_id}/evidence"), body)
            .await
    }

    /// Promote a page from Draft to Canonical
    pub async fn promote_page(&self, page_id: &str) -> Result<serde_json::Value, ClientError> {
        self.post(
            &format!("/v1/pages/{page_id}/promote"),
            &serde_json::json!({}),
        )
        .await
    }

    // ---- WASM Executable Nodes (ADR-063) ----

    /// List all published WASM nodes
    pub async fn list_nodes(&self) -> Result<serde_json::Value, ClientError> {
        self.get_path("/v1/nodes").await
    }

    /// Publish a WASM node
    pub async fn publish_node(
        &self,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.post("/v1/nodes", body).await
    }

    /// Get WASM node metadata
    pub async fn get_node(&self, id: &str) -> Result<serde_json::Value, ClientError> {
        self.get_path(&format!("/v1/nodes/{id}")).await
    }

    /// Download WASM binary
    pub async fn get_node_wasm(&self, id: &str) -> Result<Vec<u8>, ClientError> {
        let url = format!("{}/v1/nodes/{id}.wasm", self.base_url);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| ClientError::Http(e.to_string()))?;

        if resp.status().is_success() {
            resp.bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(|e| ClientError::Http(e.to_string()))
        } else {
            let status = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            Err(ClientError::Server {
                status,
                message: msg,
            })
        }
    }

    /// Revoke a WASM node
    pub async fn revoke_node(&self, id: &str) -> Result<(), ClientError> {
        let url = format!("{}/v1/nodes/{id}/revoke", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({}))
            .send()
            .await
            .map_err(|e| ClientError::Http(e.to_string()))?;

        if resp.status().is_success() || resp.status().as_u16() == 204 {
            Ok(())
        } else {
            let status = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            Err(ClientError::Server {
                status,
                message: msg,
            })
        }
    }

    // ---- HTTP helpers ----

    async fn get_path(&self, path: &str) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| ClientError::Http(e.to_string()))?;

        self.handle_response(resp).await
    }

    async fn get_with_params(
        &self,
        path: &str,
        params: &[(&str, String)],
    ) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.api_key)
            .query(params)
            .send()
            .await
            .map_err(|e| ClientError::Http(e.to_string()))?;

        self.handle_response(resp).await
    }

    async fn post(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| ClientError::Http(e.to_string()))?;

        self.handle_response(resp).await
    }

    async fn handle_response(
        &self,
        resp: reqwest::Response,
    ) -> Result<serde_json::Value, ClientError> {
        let status = resp.status().as_u16();
        if status >= 400 {
            let msg = resp.text().await.unwrap_or_default();
            return Err(ClientError::Server {
                status,
                message: msg,
            });
        }
        resp.json()
            .await
            .map_err(|e| ClientError::Serialization(e.to_string()))
    }
}

impl Default for BrainClient {
    fn default() -> Self {
        Self::new()
    }
}

/// SHAKE-256 hash
fn sha3_hash(data: &[u8]) -> [u8; 32] {
    use sha3::{
        digest::{ExtendableOutput, Update, XofReader},
        Shake256,
    };
    let mut hasher = Shake256::default();
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut buf = [0u8; 32];
    reader.read(&mut buf);
    buf
}
