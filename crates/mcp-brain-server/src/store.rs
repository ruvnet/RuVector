//! Firestore REST API client for metadata storage
//!
//! Architecture: DashMap serves as hot in-memory cache. When `FIRESTORE_URL`
//! is configured, all mutations are written through to Firestore via REST.
//! On startup, `load_from_firestore()` hydrates the cache.
//!
//! When `FIRESTORE_URL` is absent (local dev), operates as in-memory only.
//!
//! On Cloud Run, OAuth2 tokens are automatically fetched from the GCE
//! metadata server and cached with 5-minute pre-expiry refresh.

use crate::graph::cosine_similarity;
use crate::types::*;
use dashmap::DashMap;
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Cached access token with expiry (shared with GCS pattern)
struct TokenCache {
    token: String,
    expires_at: std::time::Instant,
}

/// A preference pair for training data export (Layer A)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreferencePair {
    pub memory_id: Uuid,
    pub category: String,
    pub embedding: Vec<f32>,
    pub direction: String,
    pub quality_before: f64,
    pub quality_after: f64,
    pub voter: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Firestore client with write-through persistence.
///
/// DashMap is the hot cache; Firestore REST is the durable backend.
/// When `base_url` is `None`, operates in local-only mode (dev).
///
/// On Cloud Run, OAuth2 tokens are fetched from the GCE metadata server
/// and cached with 5-minute pre-expiry refresh (same pattern as GcsClient).
pub struct FirestoreClient {
    // ── Hot cache (always populated) ──────────────────────────────────
    memories: DashMap<Uuid, BrainMemory>,
    contributors: DashMap<String, ContributorInfo>,
    vote_log: DashMap<u64, PreferencePair>,
    vote_counter: std::sync::atomic::AtomicU64,
    /// Track votes: (memory_id, voter_pseudonym) → true to prevent duplicates
    vote_tracker: DashMap<(Uuid, String), bool>,
    /// Max vote log entries before FIFO eviction
    vote_log_cap: u64,
    vote_log_start: std::sync::atomic::AtomicU64,
    page_status: DashMap<Uuid, PageStatus>,
    page_deltas: DashMap<Uuid, Vec<PageDelta>>,
    page_evidence: DashMap<Uuid, Vec<EvidenceLink>>,
    wasm_nodes: DashMap<String, WasmNode>,
    wasm_binaries: DashMap<String, Vec<u8>>,

    // ── Firestore REST backend (None = local-only) ───────────────────
    base_url: Option<String>,
    http: reqwest::Client,
    /// Static token from env (local dev) — takes priority over metadata server
    static_token: Option<String>,
    /// Cached metadata server token (auto-refreshed)
    token_cache: RwLock<Option<TokenCache>>,
    /// Whether we're on GCE (metadata server available for token refresh)
    use_metadata_server: bool,
}

impl FirestoreClient {
    pub fn new() -> Self {
        let base_url = std::env::var("FIRESTORE_URL").ok();
        let static_token = std::env::var("FIRESTORE_TOKEN").ok();
        let use_metadata_server = static_token.is_none() && base_url.is_some();

        if let Some(ref url) = base_url {
            if static_token.is_some() {
                tracing::info!("Firestore persistence enabled (static token) at: {url}");
            } else if use_metadata_server {
                tracing::info!("Firestore persistence enabled (metadata server) at: {url}");
            } else {
                tracing::info!("Firestore persistence enabled (no auth) at: {url}");
            }
        } else {
            tracing::info!("Running in local-only mode (no FIRESTORE_URL)");
        }

        Self {
            memories: DashMap::new(),
            contributors: DashMap::new(),
            vote_log: DashMap::new(),
            vote_counter: std::sync::atomic::AtomicU64::new(0),
            vote_tracker: DashMap::new(),
            vote_log_cap: 10_000,
            vote_log_start: std::sync::atomic::AtomicU64::new(0),
            page_status: DashMap::new(),
            page_deltas: DashMap::new(),
            page_evidence: DashMap::new(),
            wasm_nodes: DashMap::new(),
            wasm_binaries: DashMap::new(),
            base_url,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            static_token,
            token_cache: RwLock::new(None),
            use_metadata_server,
        }
    }

    /// Whether Firestore persistence is enabled
    pub fn is_persistent(&self) -> bool {
        self.base_url.is_some()
    }

    /// Rebuild vote tracker from persisted vote data on startup.
    /// Loads brain_votes collection from Firestore to prevent duplicate voting after restart.
    /// Also restores the vote counter for accurate display.
    pub async fn rebuild_vote_tracker(&self) {
        let docs = self.firestore_list("brain_votes").await;
        let mut count = 0usize;
        for doc in docs {
            let memory_id = doc.get("memory_id")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<Uuid>().ok());
            let voter = doc.get("voter")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            if let (Some(mid), Some(v)) = (memory_id, voter) {
                self.vote_tracker.insert((mid, v), true);
                count += 1;
            }
        }
        // Restore vote counter from loaded entries
        self.vote_counter.store(count as u64, std::sync::atomic::Ordering::Relaxed);
        tracing::info!("Vote tracker rebuilt: {} entries from Firestore", count);
    }

    // ── Token management (GCE metadata server) ───────────────────────

    /// Get a valid access token for Firestore REST API.
    /// Priority: static token > cached metadata token > fresh metadata token.
    async fn get_token(&self) -> Option<String> {
        // Static token (env var) takes priority
        if let Some(ref token) = self.static_token {
            return Some(token.clone());
        }

        if !self.use_metadata_server {
            return None;
        }

        // Check cached token
        {
            let cache = self.token_cache.read().await;
            if let Some(ref tc) = *cache {
                // Refresh 5 minutes before expiry
                if tc.expires_at > std::time::Instant::now() + std::time::Duration::from_secs(300) {
                    return Some(tc.token.clone());
                }
            }
        }

        // Refresh from metadata server
        self.refresh_token().await
    }

    /// Fetch a new token from the GCE metadata server
    async fn refresh_token(&self) -> Option<String> {
        let url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
        let resp = self.http
            .get(url)
            .header("Metadata-Flavor", "Google")
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            tracing::warn!("Firestore: GCE metadata token request failed: {}", resp.status());
            return None;
        }

        #[derive(serde::Deserialize)]
        struct TokenResponse {
            access_token: String,
            expires_in: u64,
        }

        let token_resp: TokenResponse = resp.json().await.ok()?;
        let expires_at = std::time::Instant::now()
            + std::time::Duration::from_secs(token_resp.expires_in);

        let token = token_resp.access_token.clone();

        // Cache the new token
        {
            let mut cache = self.token_cache.write().await;
            *cache = Some(TokenCache {
                token: token_resp.access_token,
                expires_at,
            });
        }

        tracing::debug!("Firestore token refreshed, expires in {}s", token_resp.expires_in);
        Some(token)
    }

    /// Build an authenticated request builder for Firestore
    async fn authenticated_request(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let mut builder = self.http.request(method, url);
        if let Some(token) = self.get_token().await {
            builder = builder.bearer_auth(token);
        }
        builder
    }

    // ── Firestore REST helpers ────────────────────────────────────────

    /// Write a document to Firestore REST API (fire-and-forget best-effort).
    /// Wraps JSON body as a single `data` stringValue field for simplicity.
    /// Uses PATCH to create or update documents.
    async fn firestore_put(&self, collection: &str, doc_id: &str, body: &serde_json::Value) {
        let Some(ref base) = self.base_url else { return };
        let url = format!("{base}/{collection}/{doc_id}");
        let json_str = serde_json::to_string(body).unwrap_or_default();
        let firestore_doc = serde_json::json!({
            "fields": {
                "data": { "stringValue": json_str }
            }
        });

        // Retry loop: up to 2 attempts (initial + 1 retry) with token refresh on 401.
        for attempt in 0..2u8 {
            let result = self.authenticated_request(reqwest::Method::PATCH, &url)
                .await
                .json(&firestore_doc)
                .send()
                .await;
            match result {
                Ok(resp) if resp.status().is_success() => {
                    tracing::debug!("Firestore PATCH {collection}/{doc_id} ok");
                    return;
                }
                Ok(resp) if resp.status().as_u16() == 401 && attempt == 0 => {
                    tracing::info!("Firestore PATCH token expired, refreshing for retry...");
                    if self.refresh_token().await.is_none() {
                        tracing::warn!("Firestore PATCH {collection}/{doc_id}: token refresh failed");
                        return;
                    }
                    // Loop will retry with fresh token
                }
                Ok(resp) if resp.status().is_server_error() && attempt == 0 => {
                    let status = resp.status();
                    tracing::warn!("Firestore PATCH {collection}/{doc_id}: {status}, retrying...");
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    // Loop will retry
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    tracing::warn!("Firestore PATCH {collection}/{doc_id}: {status} {body}");
                    return;
                }
                Err(e) if attempt == 0 => {
                    tracing::warn!("Firestore PATCH {collection}/{doc_id} failed: {e}, retrying...");
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                Err(e) => {
                    tracing::warn!("Firestore PATCH {collection}/{doc_id} failed after retry: {e}");
                    return;
                }
            }
        }
    }

    /// Delete a document from Firestore
    async fn firestore_delete(&self, collection: &str, doc_id: &str) {
        let Some(ref base) = self.base_url else { return };
        let url = format!("{base}/{collection}/{doc_id}");
        let result = self.authenticated_request(reqwest::Method::DELETE, &url)
            .await
            .send()
            .await;
        match result {
            Ok(resp) if resp.status().as_u16() == 401 => {
                tracing::info!("Firestore DELETE token expired, refreshing...");
                if let Some(new_token) = self.refresh_token().await {
                    if let Err(e) = self.http.delete(&url).bearer_auth(new_token).send().await {
                        tracing::warn!("Firestore DELETE {collection}/{doc_id} retry failed: {e}");
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Firestore DELETE {collection}/{doc_id} failed: {e}");
            }
            _ => {}
        }
    }

    /// Load all documents from a Firestore collection.
    /// Firestore REST returns `{"documents": [...]}` where each doc has
    /// `{"fields": {"data": {"stringValue": "<json>"}}}`.
    /// We unwrap the `data` field and parse the inner JSON.
    /// Paginates with `pageToken` to fetch all documents.
    /// Maximum number of consecutive page-level errors before aborting pagination.
    const MAX_PAGE_ERRORS: usize = 3;

    async fn firestore_list(&self, collection: &str) -> Vec<serde_json::Value> {
        let Some(ref base) = self.base_url else { return Vec::new() };
        let mut all_docs = Vec::new();
        let mut page_token: Option<String> = None;
        let mut consecutive_errors: usize = 0;

        loop {
            let mut url = format!("{base}/{collection}?pageSize=300");
            if let Some(ref token) = page_token {
                url.push_str(&format!("&pageToken={}", urlencoding::encode(token)));
            }

            let result = self.authenticated_request(reqwest::Method::GET, &url)
                .await
                .send()
                .await;

            let resp = match result {
                Ok(resp) if resp.status().is_success() => {
                    consecutive_errors = 0;
                    resp
                }
                Ok(resp) if resp.status().as_u16() == 401 => {
                    tracing::info!("Firestore LIST token expired, refreshing...");
                    if let Some(new_token) = self.refresh_token().await {
                        match self.http.get(&url).bearer_auth(new_token).send().await {
                            Ok(resp) if resp.status().is_success() => {
                                consecutive_errors = 0;
                                resp
                            }
                            Ok(resp) => {
                                consecutive_errors += 1;
                                tracing::warn!(
                                    "Firestore LIST {collection} retry returned {} (error {}/{})",
                                    resp.status(), consecutive_errors, Self::MAX_PAGE_ERRORS
                                );
                                if consecutive_errors >= Self::MAX_PAGE_ERRORS { break; }
                                continue;
                            }
                            Err(e) => {
                                consecutive_errors += 1;
                                tracing::warn!(
                                    "Firestore LIST {collection} retry failed: {e} (error {}/{})",
                                    consecutive_errors, Self::MAX_PAGE_ERRORS
                                );
                                if consecutive_errors >= Self::MAX_PAGE_ERRORS { break; }
                                continue;
                            }
                        }
                    } else {
                        consecutive_errors += 1;
                        tracing::warn!("Firestore LIST {collection}: token refresh failed (error {}/{})",
                            consecutive_errors, Self::MAX_PAGE_ERRORS);
                        if consecutive_errors >= Self::MAX_PAGE_ERRORS { break; }
                        continue;
                    }
                }
                Ok(resp) => {
                    consecutive_errors += 1;
                    tracing::warn!(
                        "Firestore LIST {collection} returned {} (error {}/{})",
                        resp.status(), consecutive_errors, Self::MAX_PAGE_ERRORS
                    );
                    if consecutive_errors >= Self::MAX_PAGE_ERRORS { break; }
                    continue;
                }
                Err(e) => {
                    consecutive_errors += 1;
                    tracing::warn!(
                        "Firestore LIST {collection} failed: {e} (error {}/{})",
                        consecutive_errors, Self::MAX_PAGE_ERRORS
                    );
                    if consecutive_errors >= Self::MAX_PAGE_ERRORS { break; }
                    continue;
                }
            };

            let body: serde_json::Value = match resp.json().await {
                Ok(v) => v,
                Err(e) => {
                    consecutive_errors += 1;
                    tracing::warn!(
                        "Firestore LIST {collection} parse error: {e} (error {}/{})",
                        consecutive_errors, Self::MAX_PAGE_ERRORS
                    );
                    if consecutive_errors >= Self::MAX_PAGE_ERRORS { break; }
                    continue;
                }
            };

            // Extract documents array
            if let Some(docs) = body.get("documents").and_then(|d| d.as_array()) {
                for doc in docs {
                    // Unwrap: fields.data.stringValue → parse as JSON
                    if let Some(data_str) = doc
                        .get("fields")
                        .and_then(|f| f.get("data"))
                        .and_then(|d| d.get("stringValue"))
                        .and_then(|s| s.as_str())
                    {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data_str) {
                            all_docs.push(parsed);
                        }
                    }
                }
            }

            // Check for next page
            match body.get("nextPageToken").and_then(|t| t.as_str()) {
                Some(token) => page_token = Some(token.to_string()),
                None => break,
            }
        }

        if consecutive_errors > 0 {
            tracing::warn!(
                "Firestore LIST {collection}: loaded {} documents with {} error(s)",
                all_docs.len(), consecutive_errors
            );
        } else {
            tracing::info!("Firestore LIST {collection}: loaded {} documents", all_docs.len());
        }
        all_docs
    }

    /// Hydrate in-memory cache from Firestore on startup.
    /// Silently succeeds with empty cache if Firestore is unavailable.
    pub async fn load_from_firestore(&self) {
        if self.base_url.is_none() {
            return;
        }
        tracing::info!("Loading state from Firestore...");

        // Load memories
        let docs = self.firestore_list("brain_memories").await;
        let mut mem_count = 0usize;
        for doc in docs {
            if let Ok(m) = serde_json::from_value::<BrainMemory>(doc) {
                self.memories.insert(m.id, m);
                mem_count += 1;
            }
        }

        // Load contributors
        let docs = self.firestore_list("brain_contributors").await;
        let mut contrib_count = 0usize;
        for doc in docs {
            if let Ok(c) = serde_json::from_value::<ContributorInfo>(doc) {
                self.contributors.insert(c.pseudonym.clone(), c);
                contrib_count += 1;
            }
        }

        // Load page status
        let docs = self.firestore_list("brain_page_status").await;
        for doc in docs {
            if let (Some(id), Some(status)) = (
                doc.get("id").and_then(|v| v.as_str()).and_then(|s| s.parse::<Uuid>().ok()),
                serde_json::from_value::<PageStatus>(doc.get("status").cloned().unwrap_or_default()).ok(),
            ) {
                self.page_status.insert(id, status);
            }
        }

        // Load WASM nodes
        let docs = self.firestore_list("brain_nodes").await;
        let mut node_count = 0usize;
        for doc in docs {
            if let Ok(n) = serde_json::from_value::<WasmNode>(doc) {
                self.wasm_nodes.insert(n.id.clone(), n);
                node_count += 1;
            }
        }

        tracing::info!(
            "Loaded from Firestore: {mem_count} memories, {contrib_count} contributors, {} pages, {node_count} nodes",
            self.page_status.len()
        );
    }

    /// Public Firestore write for cross-module persistence (e.g., LoRA store)
    pub async fn firestore_put_public(&self, collection: &str, doc_id: &str, body: &serde_json::Value) {
        self.firestore_put(collection, doc_id, body).await;
    }

    /// Public Firestore list for cross-module persistence (e.g., LoRA store)
    pub async fn firestore_list_public(&self, collection: &str) -> Vec<serde_json::Value> {
        self.firestore_list(collection).await
    }

    /// Store a brain memory (cache + Firestore write-through)
    pub async fn store_memory(&self, memory: BrainMemory) -> Result<(), StoreError> {
        let id = memory.id;
        // Write-through to Firestore
        if let Ok(body) = serde_json::to_value(&memory) {
            self.firestore_put("brain_memories", &id.to_string(), &body).await;
        }
        self.memories.insert(id, memory);
        Ok(())
    }

    /// Get a memory by ID
    pub async fn get_memory(&self, id: &Uuid) -> Result<Option<BrainMemory>, StoreError> {
        Ok(self.memories.get(id).map(|m| m.clone()))
    }

    /// Delete a memory (contributor-scoped, cache + Firestore)
    /// Uses atomic remove_if to prevent TOCTOU race
    pub async fn delete_memory(
        &self,
        id: &Uuid,
        contributor: &str,
    ) -> Result<bool, StoreError> {
        // Atomic check-and-remove: no TOCTOU window
        let removed = self.memories.remove_if(id, |_, entry| {
            entry.contributor_id == contributor
        });
        match removed {
            Some(_) => {
                self.firestore_delete("brain_memories", &id.to_string()).await;
                Ok(true)
            }
            None => {
                // Either not found or belongs to another contributor
                if self.memories.contains_key(id) {
                    Err(StoreError::Forbidden(
                        "Can only delete own contributions".into(),
                    ))
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Search memories by embedding similarity, returning (score, memory) pairs
    pub async fn search_memories(
        &self,
        query_embedding: &[f32],
        category: Option<&BrainCategory>,
        tags: Option<&[String]>,
        limit: usize,
        min_quality: f64,
    ) -> Result<Vec<(f64, BrainMemory)>, StoreError> {
        let mut scored: Vec<(f64, BrainMemory)> = self
            .memories
            .iter()
            .filter(|entry| {
                let m = entry.value();
                let quality_ok = m.quality_score.mean() >= min_quality;
                let category_ok = category.map_or(true, |c| &m.category == c);
                let tags_ok = tags.map_or(true, |t| {
                    t.iter().any(|tag| m.tags.contains(tag))
                });
                quality_ok && category_ok && tags_ok
            })
            .map(|entry| {
                let m = entry.value().clone();
                let sim = cosine_similarity(query_embedding, &m.embedding);
                (sim, m)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 {
            scored.truncate(limit);
        }
        // limit=0 means return all (for full-corpus keyword re-ranking)
        Ok(scored)
    }

    /// Keyword-based search fallback when no embedding is provided.
    /// Scores memories by tag match, title/content term overlap, and quality.
    pub async fn keyword_search(
        &self,
        query: &str,
        category: Option<&BrainCategory>,
        tags: Option<&[String]>,
        limit: usize,
        min_quality: f64,
    ) -> Result<Vec<BrainMemory>, StoreError> {
        let query_lower = query.to_lowercase();
        let query_tokens: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 1)
            .collect();

        let mut scored: Vec<(f64, BrainMemory)> = self
            .memories
            .iter()
            .filter(|entry| {
                let m = entry.value();
                let quality_ok = m.quality_score.mean() >= min_quality;
                let category_ok = category.map_or(true, |c| &m.category == c);
                quality_ok && category_ok
            })
            .map(|entry| {
                let m = entry.value().clone();
                let title_lower = m.title.to_lowercase();
                let content_lower = m.content.to_lowercase();

                // Score components
                let mut score = 0.0f64;

                // Tag matches (highest weight)
                for token in &query_tokens {
                    if m.tags.iter().any(|t| t.to_lowercase().contains(token)) {
                        score += 3.0;
                    }
                }

                // Exact tag match from filter
                if let Some(filter_tags) = tags {
                    for ft in filter_tags {
                        if m.tags.iter().any(|t| t == ft) {
                            score += 5.0;
                        }
                    }
                }

                // Title token matches
                for token in &query_tokens {
                    if title_lower.contains(token) {
                        score += 2.0;
                    }
                }

                // Content token matches
                for token in &query_tokens {
                    if content_lower.contains(token) {
                        score += 1.0;
                    }
                }

                // Category match
                if query_tokens.iter().any(|t| m.category.to_string().to_lowercase().contains(t)) {
                    score += 1.5;
                }

                // Quality bonus
                score += m.quality_score.mean() * 0.5;

                (score, m)
            })
            .filter(|(score, _)| *score > 0.5) // Must have at least some relevance
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(_, m)| m).collect())
    }

    /// List memories with pagination, tag filtering, and sorting
    pub async fn list_memories(
        &self,
        category: Option<&BrainCategory>,
        tags: Option<&[String]>,
        limit: usize,
        offset: usize,
        sort: &crate::types::ListSort,
    ) -> Result<(Vec<BrainMemory>, usize), StoreError> {
        let mut memories: Vec<BrainMemory> = self
            .memories
            .iter()
            .filter(|entry| {
                let m = entry.value();
                let category_ok = category.map_or(true, |c| &m.category == c);
                let tags_ok = tags.map_or(true, |t| {
                    t.iter().any(|tag| m.tags.contains(tag))
                });
                category_ok && tags_ok
            })
            .map(|entry| entry.value().clone())
            .collect();

        let total_count = memories.len();

        match sort {
            ListSort::UpdatedAt => {
                memories.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
            }
            ListSort::Quality => {
                memories.sort_by(|a, b| {
                    b.quality_score
                        .mean()
                        .partial_cmp(&a.quality_score.mean())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ListSort::Votes => {
                memories.sort_by(|a, b| {
                    b.quality_score
                        .observations()
                        .partial_cmp(&a.quality_score.observations())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let paginated: Vec<BrainMemory> = memories
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect();

        Ok((paginated, total_count))
    }

    /// Update quality score for a memory and log the preference pair
    /// (cache + Firestore write-through for memory update and vote log)
    ///
    /// Security: prevents self-voting and duplicate votes per (memory, voter) pair.
    pub async fn update_quality(
        &self,
        id: &Uuid,
        direction: &VoteDirection,
        voter: &str,
    ) -> Result<BetaParams, StoreError> {
        // Block self-voting: contributor cannot vote on own memory
        if let Some(entry) = self.memories.get(id) {
            if entry.contributor_id == voter {
                return Err(StoreError::Forbidden(
                    "Cannot vote on your own contribution".into(),
                ));
            }
        } else {
            return Err(StoreError::NotFound(id.to_string()));
        }

        // Block duplicate votes: same voter on same memory (single lookup via entry API)
        let vote_key = (*id, voter.to_string());
        if let dashmap::mapref::entry::Entry::Occupied(_) = self.vote_tracker.entry(vote_key.clone()) {
            return Err(StoreError::Forbidden(
                "Already voted on this memory".into(),
            ));
        }

        let quality_score;
        let vote_doc_id;
        let vote_doc;
        {
            let mut entry = self
                .memories
                .get_mut(id)
                .ok_or_else(|| StoreError::NotFound(id.to_string()))?;
            let quality_before = entry.quality_score.mean();
            match direction {
                VoteDirection::Up => entry.quality_score.upvote(),
                VoteDirection::Down => entry.quality_score.downvote(),
            }
            entry.updated_at = chrono::Utc::now();
            let quality_after = entry.quality_score.mean();
            quality_score = entry.quality_score.clone();

            // Record the vote to prevent future duplicates
            self.vote_tracker.insert(vote_key.clone(), true);

            // Prepare vote persistence doc (written outside borrow block)
            vote_doc_id = format!("{}__{}", id, voter);
            vote_doc = serde_json::json!({
                "memory_id": id.to_string(),
                "voter": voter,
                "direction": match direction {
                    VoteDirection::Up => "up",
                    VoteDirection::Down => "down",
                },
                "timestamp": chrono::Utc::now().to_rfc3339()
            });

            // Record preference pair for training data (Layer A)
            let pair = PreferencePair {
                memory_id: *id,
                category: entry.category.to_string(),
                embedding: entry.embedding.clone(),
                direction: match direction {
                    VoteDirection::Up => "up".to_string(),
                    VoteDirection::Down => "down".to_string(),
                },
                quality_before,
                quality_after,
                voter: voter.to_string(),
                timestamp: chrono::Utc::now(),
            };
            let idx = self.vote_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.vote_log.insert(idx, pair);

            // FIFO eviction: remove oldest entries when over cap
            let start = self.vote_log_start.load(std::sync::atomic::Ordering::Relaxed);
            if idx.saturating_sub(start) >= self.vote_log_cap {
                let evict_to = idx - self.vote_log_cap + 1;
                for old_idx in start..evict_to {
                    self.vote_log.remove(&old_idx);
                }
                self.vote_log_start.store(evict_to, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // Persist vote tracker entry to Firestore (outside borrow block)
        self.firestore_put("brain_votes", &vote_doc_id, &vote_doc).await;

        // Write-through: persist updated memory to Firestore
        if let Some(m) = self.memories.get(id) {
            if let Ok(body) = serde_json::to_value(m.value()) {
                self.firestore_put("brain_memories", &id.to_string(), &body).await;
            }
        }

        Ok(quality_score)
    }

    /// Get preference pairs for training data export (Layer A)
    /// Returns pairs accumulated since the given index
    pub fn get_preference_pairs(&self, since_index: u64, limit: usize) -> (Vec<PreferencePair>, u64) {
        let current = self.vote_counter.load(std::sync::atomic::Ordering::Relaxed);
        let mut pairs = Vec::new();
        for idx in since_index..current {
            if pairs.len() >= limit {
                break;
            }
            if let Some(pair) = self.vote_log.get(&idx) {
                pairs.push(pair.clone());
            }
        }
        let next_index = if pairs.is_empty() {
            since_index
        } else {
            since_index + pairs.len() as u64
        };
        (pairs, next_index)
    }

    /// Get total vote count
    pub fn vote_count(&self) -> u64 {
        self.vote_counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get or create contributor (cache + Firestore write-through)
    pub async fn get_or_create_contributor(
        &self,
        pseudonym: &str,
        is_system: bool,
    ) -> Result<ContributorInfo, StoreError> {
        if let Some(mut c) = self.contributors.get_mut(pseudonym) {
            // Upgrade to system if authenticated as system
            if is_system && !c.is_system {
                c.is_system = true;
                c.reputation = ReputationScore {
                    accuracy: 1.0,
                    uptime: 1.0,
                    stake: 1000.0,
                    composite: 1.0,
                };
                if let Ok(body) = serde_json::to_value(c.value()) {
                    self.firestore_put("brain_contributors", pseudonym, &body).await;
                }
            }
            return Ok(c.clone());
        }
        let info = ContributorInfo {
            pseudonym: pseudonym.to_string(),
            reputation: if is_system {
                ReputationScore {
                    accuracy: 1.0,
                    uptime: 1.0,
                    stake: 1000.0,
                    composite: 1.0,
                }
            } else {
                ReputationScore::cold_start()
            },
            contribution_count: 0,
            created_at: chrono::Utc::now(),
            last_active: chrono::Utc::now(),
            is_system,
        };
        if let Ok(body) = serde_json::to_value(&info) {
            self.firestore_put("brain_contributors", pseudonym, &body).await;
        }
        self.contributors.insert(pseudonym.to_string(), info.clone());
        Ok(info)
    }

    /// Detect the embedding dimension from the first stored memory
    pub fn detect_embedding_dim(&self) -> Option<usize> {
        self.memories.iter().next().map(|e| e.value().embedding.len())
    }

    /// Get all memories (for graph building)
    pub fn all_memories(&self) -> Vec<BrainMemory> {
        self.memories.iter().map(|e| e.value().clone()).collect()
    }

    /// Update a memory's embedding in-place (used during RLM re-embedding on startup)
    pub fn update_embedding(&self, id: &Uuid, embedding: &[f32]) {
        if let Some(mut entry) = self.memories.get_mut(id) {
            entry.embedding = embedding.to_vec();
        }
    }

    /// Get the reputation score for a contributor, if known
    pub fn get_contributor_reputation(&self, pseudonym: &str) -> Option<ReputationScore> {
        self.contributors.get(pseudonym).map(|c| c.reputation.clone())
    }

    /// Record a contribution: increment count, update uptime, recompute composite
    pub async fn record_contribution(&self, pseudonym: &str) {
        if let Some(mut entry) = self.contributors.get_mut(pseudonym) {
            entry.contribution_count += 1;
            entry.last_active = chrono::Utc::now();
            // Grow stake organically through contributions
            entry.reputation.stake += 1.0;
            crate::reputation::ReputationManager::record_activity(
                &mut entry.reputation,
            );
            // Persist updated contributor
            if let Ok(body) = serde_json::to_value(entry.value()) {
                self.firestore_put("brain_contributors", pseudonym, &body).await;
            }
        }
    }

    /// Update contributor reputation based on vote outcome on their content
    pub async fn update_reputation_from_vote(
        &self,
        content_author: &str,
        was_upvoted: bool,
    ) {
        if let Some(mut entry) = self.contributors.get_mut(content_author) {
            crate::reputation::ReputationManager::update_accuracy(
                &mut entry.reputation,
                was_upvoted,
            );
            // Persist updated contributor
            if let Ok(body) = serde_json::to_value(entry.value()) {
                self.firestore_put("brain_contributors", content_author, &body)
                    .await;
            }
        }
    }

    /// Check and apply poisoning penalty if quality is too low after enough votes
    pub async fn check_poisoning(
        &self,
        content_author: &str,
        downvote_count: u32,
        quality: f64,
    ) -> bool {
        let mgr = crate::reputation::ReputationManager::new();
        if let Some(mut entry) = self.contributors.get_mut(content_author) {
            let penalized = mgr.check_poisoning_penalty(
                &mut entry.reputation,
                downvote_count,
                quality,
            );
            if penalized {
                if let Ok(body) = serde_json::to_value(entry.value()) {
                    self.firestore_put("brain_contributors", content_author, &body)
                        .await;
                }
            }
            penalized
        } else {
            false
        }
    }

    /// Get total count
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Get contributor count
    pub fn contributor_count(&self) -> usize {
        self.contributors.len()
    }

    // ──────────────────────────────────────────────────────────────────
    // Brainpedia (ADR-062)
    // ──────────────────────────────────────────────────────────────────

    /// Create a Brainpedia page (cache + Firestore write-through)
    pub async fn create_page(
        &self,
        memory: BrainMemory,
        status: PageStatus,
        evidence: Vec<EvidenceLink>,
    ) -> Result<(), StoreError> {
        let id = memory.id;
        // Persist memory
        if let Ok(body) = serde_json::to_value(&memory) {
            self.firestore_put("brain_memories", &id.to_string(), &body).await;
        }
        // Persist page status
        let status_doc = serde_json::json!({ "id": id.to_string(), "status": status });
        self.firestore_put("brain_page_status", &id.to_string(), &status_doc).await;
        // Cache
        self.memories.insert(id, memory);
        self.page_status.insert(id, status);
        self.page_deltas.insert(id, Vec::new());
        if !evidence.is_empty() {
            self.page_evidence.insert(id, evidence);
        } else {
            self.page_evidence.insert(id, Vec::new());
        }
        Ok(())
    }

    /// List all pages with summary info
    pub fn list_pages(&self, limit: usize, offset: usize) -> (Vec<Uuid>, usize) {
        let all_ids: Vec<Uuid> = self.page_status.iter().map(|e| *e.key()).collect();
        let total = all_ids.len();
        let page_ids: Vec<Uuid> = all_ids.into_iter().skip(offset).take(limit).collect();
        (page_ids, total)
    }

    /// Get page status
    pub fn get_page_status(&self, id: &Uuid) -> Option<PageStatus> {
        self.page_status.get(id).map(|s| s.clone())
    }

    /// Submit a delta to a page
    pub async fn submit_delta(
        &self,
        page_id: &Uuid,
        delta: PageDelta,
    ) -> Result<(), StoreError> {
        if !self.memories.contains_key(page_id) {
            return Err(StoreError::NotFound(page_id.to_string()));
        }

        // Append delta
        self.page_deltas
            .entry(*page_id)
            .or_insert_with(Vec::new)
            .push(delta.clone());

        // If delta carries evidence, add to page evidence
        if !delta.evidence_links.is_empty() {
            self.page_evidence
                .entry(*page_id)
                .or_insert_with(Vec::new)
                .extend(delta.evidence_links);
        }

        // Update memory timestamp
        if let Some(mut entry) = self.memories.get_mut(page_id) {
            entry.updated_at = chrono::Utc::now();
        }

        Ok(())
    }

    /// Add evidence to a page (without a delta)
    pub async fn add_evidence(
        &self,
        page_id: &Uuid,
        evidence: EvidenceLink,
    ) -> Result<u32, StoreError> {
        if !self.memories.contains_key(page_id) {
            return Err(StoreError::NotFound(page_id.to_string()));
        }

        let mut ev = self.page_evidence
            .entry(*page_id)
            .or_insert_with(Vec::new);
        ev.push(evidence);
        Ok(ev.len() as u32)
    }

    /// Get deltas for a page
    pub fn get_deltas(&self, page_id: &Uuid) -> Vec<PageDelta> {
        self.page_deltas
            .get(page_id)
            .map(|d| d.clone())
            .unwrap_or_default()
    }

    /// Get evidence links for a page
    pub fn get_evidence(&self, page_id: &Uuid) -> Vec<EvidenceLink> {
        self.page_evidence
            .get(page_id)
            .map(|e| e.clone())
            .unwrap_or_default()
    }

    /// Get page evidence and delta counts
    pub fn page_counts(&self, page_id: &Uuid) -> (u32, u32) {
        let ev = self.page_evidence.get(page_id).map(|e| e.len()).unwrap_or(0) as u32;
        let dc = self.page_deltas.get(page_id).map(|d| d.len()).unwrap_or(0) as u32;
        (ev, dc)
    }

    /// Check promotion criteria: quality >= 0.7, observations >= 5, evidence >= 3 from >= 2 contributors
    pub fn check_promotion(&self, page_id: &Uuid) -> bool {
        let memory = match self.memories.get(page_id) {
            Some(m) => m,
            None => return false,
        };
        if memory.quality_score.mean() < 0.7 {
            return false;
        }
        if memory.quality_score.observations() < 5.0 {
            return false;
        }
        let evidence = self.get_evidence(page_id);
        if evidence.len() < 3 {
            return false;
        }
        let distinct_contributors: std::collections::HashSet<&str> =
            evidence.iter().map(|e| e.contributor_id.as_str()).collect();
        distinct_contributors.len() >= 2
    }

    /// Promote a page from Draft to Canonical (cache + Firestore)
    pub async fn promote_page(&self, page_id: &Uuid) -> Result<PageStatus, StoreError> {
        let current = self.get_page_status(page_id)
            .ok_or_else(|| StoreError::NotFound(page_id.to_string()))?;
        if current != PageStatus::Draft {
            return Err(StoreError::Storage(format!(
                "Can only promote Draft pages, current status: {current}"
            )));
        }
        if !self.check_promotion(page_id) {
            return Err(StoreError::Storage(
                "Promotion criteria not met: need quality >= 0.7, observations >= 5, evidence >= 3 from >= 2 contributors".into()
            ));
        }
        self.page_status.insert(*page_id, PageStatus::Canonical);
        // Persist status change
        let status_doc = serde_json::json!({ "id": page_id.to_string(), "status": PageStatus::Canonical });
        self.firestore_put("brain_page_status", &page_id.to_string(), &status_doc).await;
        Ok(PageStatus::Canonical)
    }

    /// Get page count by status
    pub fn page_count_by_status(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for entry in self.page_status.iter() {
            *counts.entry(entry.value().to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Get total page count
    pub fn page_count(&self) -> usize {
        self.page_status.len()
    }

    // ──────────────────────────────────────────────────────────────────
    // WASM Executable Nodes (ADR-063)
    // ──────────────────────────────────────────────────────────────────

    /// Publish a WASM node (cache + Firestore write-through for metadata)
    pub async fn publish_node(&self, node: WasmNode, wasm_bytes: Vec<u8>) -> Result<(), StoreError> {
        if self.wasm_nodes.contains_key(&node.id) {
            return Err(StoreError::Storage(format!(
                "Node {} already exists (nodes are immutable, use a new version)",
                node.id
            )));
        }
        if wasm_bytes.len() > 1_048_576 {
            return Err(StoreError::Storage("WASM binary exceeds 1MB limit".into()));
        }
        // Persist node metadata to Firestore (binary goes to GCS)
        if let Ok(body) = serde_json::to_value(&node) {
            self.firestore_put("brain_nodes", &node.id, &body).await;
        }
        self.wasm_binaries.insert(node.id.clone(), wasm_bytes);
        self.wasm_nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Get node metadata
    pub fn get_node(&self, id: &str) -> Option<WasmNode> {
        self.wasm_nodes.get(id).map(|n| n.clone())
    }

    /// Get WASM binary
    pub fn get_node_binary(&self, id: &str) -> Option<Vec<u8>> {
        self.wasm_binaries.get(id).map(|b| b.clone())
    }

    /// List all nodes
    pub fn list_nodes(&self) -> Vec<WasmNode> {
        self.wasm_nodes.iter().map(|e| e.value().clone()).collect()
    }

    /// Revoke a node (marks as revoked, does not delete bytes, cache + Firestore)
    pub async fn revoke_node(&self, id: &str, contributor: &str) -> Result<(), StoreError> {
        {
            let mut node = self.wasm_nodes.get_mut(id)
                .ok_or_else(|| StoreError::NotFound(id.to_string()))?;
            if node.contributor_id != contributor {
                return Err(StoreError::Forbidden("Only original publisher can revoke".into()));
            }
            node.revoked = true;
        }
        // Persist revocation
        if let Some(n) = self.wasm_nodes.get(id) {
            if let Ok(body) = serde_json::to_value(n.value()) {
                self.firestore_put("brain_nodes", id, &body).await;
            }
        }
        Ok(())
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.wasm_nodes.len()
    }
}

impl Default for FirestoreClient {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Forbidden: {0}")]
    Forbidden(String),
    #[error("Storage error: {0}")]
    Storage(String),
}
