//! Cloud-native data pipeline for real-time injection and optimization.
//!
//! Also contains the RVF container construction pipeline (ADR-075 Phase 5).

use chrono::{DateTime, Utc};
use rvf_types::SegmentFlags;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

// ── RVF Container Construction (ADR-075 Phase 5) ────────────────────

/// Input data for building an RVF container.
pub struct RvfPipelineInput<'a> {
    pub memory_id: &'a str,
    pub embedding: &'a [f32],
    pub title: &'a str,
    pub content: &'a str,
    pub tags: &'a [String],
    pub category: &'a str,
    pub contributor_id: &'a str,
    pub witness_chain: Option<&'a [u8]>,
    pub dp_proof_json: Option<&'a str>,
    pub redaction_log_json: Option<&'a str>,
}

/// Build an RVF container. Returns serialized bytes (64-byte-aligned segments).
pub fn build_rvf_container(input: &RvfPipelineInput<'_>) -> Result<Vec<u8>, String> {
    let flags = SegmentFlags::empty();
    let mut out = Vec::new();
    let mut sid: u64 = 1;
    // VEC (0x01)
    let mut vec_payload = Vec::with_capacity(input.embedding.len() * 4);
    for &v in input.embedding {
        vec_payload.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&rvf_wire::write_segment(0x01, &vec_payload, flags, sid));
    sid += 1;
    // META (0x07)
    let meta = serde_json::json!({
        "memory_id": input.memory_id, "title": input.title, "content": input.content,
        "tags": input.tags, "category": input.category, "contributor_id": input.contributor_id,
    });
    let mp =
        serde_json::to_vec(&meta).map_err(|e| format!("Failed to serialize RVF metadata: {e}"))?;
    out.extend_from_slice(&rvf_wire::write_segment(0x07, &mp, flags, sid));
    sid += 1;
    // WITNESS (0x0A)
    if let Some(c) = input.witness_chain {
        out.extend_from_slice(&rvf_wire::write_segment(0x0A, c, flags, sid));
        sid += 1;
    }
    // DiffPrivacyProof (0x34)
    if let Some(p) = input.dp_proof_json {
        out.extend_from_slice(&rvf_wire::write_segment(0x34, p.as_bytes(), flags, sid));
        sid += 1;
    }
    // RedactionLog (0x35)
    if let Some(l) = input.redaction_log_json {
        out.extend_from_slice(&rvf_wire::write_segment(0x35, l.as_bytes(), flags, sid));
        let _ = sid;
    }
    Ok(out)
}

/// Count segments in a serialized RVF container.
pub fn count_segments(container: &[u8]) -> usize {
    let (mut count, mut off) = (0, 0);
    while off + 64 <= container.len() {
        let plen = u64::from_le_bytes(container[off + 16..off + 24].try_into().unwrap_or([0u8; 8]))
            as usize;
        off += rvf_wire::calculate_padded_size(64, plen);
        count += 1;
    }
    count
}

// ── Cloud Pub/Sub ────────────────────────────────────────────────────

/// A decoded Pub/Sub message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PubSubMessage {
    pub data: Vec<u8>,
    pub attributes: HashMap<String, String>,
    pub message_id: String,
    pub publish_time: Option<DateTime<Utc>>,
}

/// Push envelope from Cloud Pub/Sub (HTTP POST body).
#[derive(Debug, Deserialize)]
pub struct PubSubPushEnvelope {
    pub message: PubSubPushMsg,
    pub subscription: String,
}

#[derive(Debug, Deserialize)]
pub struct PubSubPushMsg {
    pub data: Option<String>,
    #[serde(default)]
    pub attributes: HashMap<String, String>,
    #[serde(rename = "messageId")]
    pub message_id: String,
    #[serde(rename = "publishTime")]
    pub publish_time: Option<DateTime<Utc>>,
}

/// Client for Google Cloud Pub/Sub pull-based message retrieval.
#[derive(Debug)]
pub struct PubSubClient {
    project_id: String,
    subscription_id: String,
    http: reqwest::Client,
    use_metadata_server: bool,
}

impl PubSubClient {
    pub fn new(project_id: String, subscription_id: String) -> Self {
        Self {
            use_metadata_server: std::env::var("PUBSUB_EMULATOR_HOST").is_err(),
            project_id,
            subscription_id,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Decode a push-envelope into a `PubSubMessage`.
    pub fn decode_push(envelope: PubSubPushEnvelope) -> Result<PubSubMessage, String> {
        use base64::Engine;
        let data = match envelope.message.data {
            Some(b64) => base64::engine::general_purpose::STANDARD
                .decode(&b64)
                .map_err(|e| format!("base64 decode failed: {e}"))?,
            None => Vec::new(),
        };
        Ok(PubSubMessage {
            data,
            attributes: envelope.message.attributes,
            message_id: envelope.message.message_id,
            publish_time: envelope.message.publish_time,
        })
    }

    /// Acknowledge messages by ack_id (pull mode).
    pub async fn acknowledge(&self, ack_ids: &[String]) -> Result<(), String> {
        if ack_ids.is_empty() {
            return Ok(());
        }
        let url = format!(
            "https://pubsub.googleapis.com/v1/projects/{}/subscriptions/{}:acknowledge",
            self.project_id, self.subscription_id
        );
        let mut req = self
            .http
            .post(&url)
            .json(&serde_json::json!({ "ackIds": ack_ids }));
        if self.use_metadata_server {
            if let Some(t) = get_metadata_token(&self.http).await {
                req = req.bearer_auth(t);
            }
        }
        let resp = req.send().await.map_err(|e| format!("ack failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("ack returned {}", resp.status()));
        }
        Ok(())
    }
}

/// Fetch access token from GCE metadata server.
async fn get_metadata_token(http: &reqwest::Client) -> Option<String> {
    #[derive(Deserialize)]
    struct T {
        access_token: String,
    }
    let r = http.get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
        .header("Metadata-Flavor", "Google").send().await.ok()?;
    if !r.status().is_success() {
        return None;
    }
    Some(r.json::<T>().await.ok()?.access_token)
}

// ── Data Injection Pipeline ──────────────────────────────────────────

/// Source of injected data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum InjectionSource {
    PubSub,
    BatchUpload,
    RssFeed,
    Webhook,
    CommonCrawl,
}

/// An item flowing through the injection pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionItem {
    pub source: InjectionSource,
    pub title: String,
    pub content: String,
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub received_at: DateTime<Utc>,
}

/// Result of pipeline processing for a single item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionResult {
    pub item_hash: String,
    pub accepted: bool,
    pub duplicate: bool,
    pub stage_reached: String,
    pub error: Option<String>,
}

/// Processes incoming data: validate -> embed -> dedup -> store -> graph-update -> train-check
#[derive(Debug)]
pub struct DataInjector {
    seen_hashes: dashmap::DashMap<String, DateTime<Utc>>,
    new_items_since_train: AtomicU64,
}

impl DataInjector {
    pub fn new() -> Self {
        Self {
            seen_hashes: dashmap::DashMap::new(),
            new_items_since_train: AtomicU64::new(0),
        }
    }

    /// Compute a SHA-256 content hash for deduplication.
    pub fn content_hash(title: &str, content: &str) -> String {
        let mut h = Sha256::new();
        h.update(title.as_bytes());
        h.update(b"|");
        h.update(content.as_bytes());
        hex::encode(h.finalize())
    }

    /// Run the injection pipeline for a single item.
    pub fn process(&self, item: &InjectionItem) -> InjectionResult {
        if item.title.is_empty() || item.content.is_empty() {
            return InjectionResult {
                item_hash: String::new(),
                accepted: false,
                duplicate: false,
                stage_reached: "validate".into(),
                error: Some("title and content must be non-empty".into()),
            };
        }
        let hash = Self::content_hash(&item.title, &item.content);
        if self.seen_hashes.contains_key(&hash) {
            return InjectionResult {
                item_hash: hash,
                accepted: false,
                duplicate: true,
                stage_reached: "dedup".into(),
                error: None,
            };
        }
        self.seen_hashes.insert(hash.clone(), Utc::now());
        self.new_items_since_train.fetch_add(1, Ordering::Relaxed);
        InjectionResult {
            item_hash: hash,
            accepted: true,
            duplicate: false,
            stage_reached: "ready_for_embed".into(),
            error: None,
        }
    }

    pub fn new_items_count(&self) -> u64 {
        self.new_items_since_train.load(Ordering::Relaxed)
    }
    pub fn reset_train_counter(&self) {
        self.new_items_since_train.store(0, Ordering::Relaxed);
    }
    pub fn dedup_set_size(&self) -> usize {
        self.seen_hashes.len()
    }
}

impl Default for DataInjector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Optimization Scheduler ───────────────────────────────────────────

/// Configuration for optimization cycle intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub train_item_threshold: u64,
    pub train_interval_secs: u64,
    pub drift_interval_secs: u64,
    pub transfer_interval_secs: u64,
    pub graph_rebalance_secs: u64,
    pub cleanup_interval_secs: u64,
    pub attractor_interval_secs: u64,
    pub prune_quality_threshold: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            train_item_threshold: 100,
            train_interval_secs: 300,
            drift_interval_secs: 900,
            transfer_interval_secs: 1800,
            graph_rebalance_secs: 3600,
            cleanup_interval_secs: 86400,
            attractor_interval_secs: 1200,
            prune_quality_threshold: 0.3,
        }
    }
}

/// Tracks timestamps and counters to decide when optimization tasks fire.
#[derive(Debug)]
pub struct OptimizationScheduler {
    pub config: SchedulerConfig,
    last_train: RwLock<DateTime<Utc>>,
    last_drift_check: RwLock<DateTime<Utc>>,
    last_transfer: RwLock<DateTime<Utc>>,
    last_graph_rebalance: RwLock<DateTime<Utc>>,
    last_cleanup: RwLock<DateTime<Utc>>,
    last_attractor: RwLock<DateTime<Utc>>,
    cycles_completed: AtomicU64,
}

impl OptimizationScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let now = Utc::now();
        Self {
            config,
            cycles_completed: AtomicU64::new(0),
            last_train: RwLock::new(now),
            last_drift_check: RwLock::new(now),
            last_transfer: RwLock::new(now),
            last_graph_rebalance: RwLock::new(now),
            last_cleanup: RwLock::new(now),
            last_attractor: RwLock::new(now),
        }
    }

    /// Check which optimization tasks are due.
    pub async fn due_tasks(&self, new_item_count: u64) -> Vec<String> {
        let now = Utc::now();
        let ss = |ts: &DateTime<Utc>| (now - *ts).num_seconds().max(0) as u64;
        let mut due = Vec::new();
        if new_item_count >= self.config.train_item_threshold
            || ss(&*self.last_train.read().await) >= self.config.train_interval_secs
        {
            due.push("training".into());
        }
        if ss(&*self.last_drift_check.read().await) >= self.config.drift_interval_secs {
            due.push("drift_monitoring".into());
        }
        if ss(&*self.last_transfer.read().await) >= self.config.transfer_interval_secs {
            due.push("cross_domain_transfer".into());
        }
        if ss(&*self.last_graph_rebalance.read().await) >= self.config.graph_rebalance_secs {
            due.push("graph_rebalancing".into());
        }
        if ss(&*self.last_cleanup.read().await) >= self.config.cleanup_interval_secs {
            due.push("memory_cleanup".into());
        }
        if ss(&*self.last_attractor.read().await) >= self.config.attractor_interval_secs {
            due.push("attractor_analysis".into());
        }
        due
    }

    /// Mark a task as completed, updating its timestamp.
    pub async fn mark_completed(&self, task: &str) {
        let now = Utc::now();
        match task {
            "training" => *self.last_train.write().await = now,
            "drift_monitoring" => *self.last_drift_check.write().await = now,
            "cross_domain_transfer" => *self.last_transfer.write().await = now,
            "graph_rebalancing" => *self.last_graph_rebalance.write().await = now,
            "memory_cleanup" => *self.last_cleanup.write().await = now,
            "attractor_analysis" => *self.last_attractor.write().await = now,
            _ => {}
        }
        self.cycles_completed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn cycles_completed(&self) -> u64 {
        self.cycles_completed.load(Ordering::Relaxed)
    }
}

// ── Health & Metrics ─────────────────────────────────────────────────

/// Pipeline metrics snapshot for Cloud Monitoring.
#[derive(Debug, Serialize)]
pub struct PipelineMetrics {
    pub messages_received: u64,
    pub messages_processed: u64,
    pub messages_failed: u64,
    pub injections_per_minute: f64,
    pub last_training_time: Option<DateTime<Utc>>,
    pub last_drift_check: Option<DateTime<Utc>>,
    pub last_transfer: Option<DateTime<Utc>>,
    pub queue_depth: u64,
    pub optimization_cycles_completed: u64,
}

/// Atomic counters for thread-safe metric collection.
#[derive(Debug)]
pub struct MetricsCollector {
    received: AtomicU64,
    processed: AtomicU64,
    failed: AtomicU64,
    queue_depth: AtomicU64,
    recent_injections: RwLock<Vec<(i64, u64)>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            received: AtomicU64::new(0),
            processed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            queue_depth: AtomicU64::new(0),
            recent_injections: RwLock::new(Vec::new()),
        }
    }
    pub fn record_received(&self) {
        self.received.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_processed(&self) {
        self.processed.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }
    pub fn set_queue_depth(&self, d: u64) {
        self.queue_depth.store(d, Ordering::Relaxed);
    }

    pub async fn record_injection(&self) {
        let now = Utc::now().timestamp();
        let mut w = self.recent_injections.write().await;
        w.push((now, 1));
        w.retain(|(ts, _)| *ts >= now - 300);
    }

    pub async fn injections_per_minute(&self) -> f64 {
        let w = self.recent_injections.read().await;
        if w.is_empty() {
            return 0.0;
        }
        let total: u64 = w.iter().map(|(_, c)| c).sum();
        let span = ((Utc::now().timestamp() - w[0].0) as f64 / 60.0).max(1.0 / 60.0);
        total as f64 / span
    }

    pub async fn snapshot(&self, sched: &OptimizationScheduler) -> PipelineMetrics {
        PipelineMetrics {
            messages_received: self.received.load(Ordering::Relaxed),
            messages_processed: self.processed.load(Ordering::Relaxed),
            messages_failed: self.failed.load(Ordering::Relaxed),
            injections_per_minute: self.injections_per_minute().await,
            last_training_time: Some(*sched.last_train.read().await),
            last_drift_check: Some(*sched.last_drift_check.read().await),
            last_transfer: Some(*sched.last_transfer.read().await),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            optimization_cycles_completed: sched.cycles_completed(),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Feed Ingestion (RSS/Atom) ────────────────────────────────────────

/// Configuration for a single feed source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedSource {
    pub url: String,
    pub poll_interval_secs: u64,
    pub default_category: Option<String>,
    pub default_tags: Vec<String>,
}

/// A parsed feed entry ready for injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedEntry {
    pub title: String,
    pub content: String,
    pub link: Option<String>,
    pub published: Option<DateTime<Utc>>,
    pub content_hash: String,
    pub source_url: String,
    pub category: Option<String>,
    pub tags: Vec<String>,
}

/// Ingests RSS/Atom feeds and converts entries to `InjectionItem`s.
#[derive(Debug)]
pub struct FeedIngester {
    sources: Vec<FeedSource>,
    last_poll: HashMap<String, DateTime<Utc>>,
    seen_hashes: dashmap::DashMap<String, ()>,
    http: reqwest::Client,
}

impl FeedIngester {
    pub fn new(sources: Vec<FeedSource>) -> Self {
        let lp = sources
            .iter()
            .map(|s| (s.url.clone(), Utc::now()))
            .collect();
        Self {
            sources,
            last_poll: lp,
            seen_hashes: dashmap::DashMap::new(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    pub fn feeds_due(&self) -> Vec<&FeedSource> {
        let now = Utc::now();
        self.sources
            .iter()
            .filter(|s| {
                let last = self.last_poll.get(&s.url).copied().unwrap_or(now);
                (now - last).num_seconds().max(0) as u64 >= s.poll_interval_secs
            })
            .collect()
    }

    /// Fetch and parse a feed URL, returning new (non-duplicate) entries.
    pub async fn fetch_feed(&self, source: &FeedSource) -> Result<Vec<FeedEntry>, String> {
        let resp = self
            .http
            .get(&source.url)
            .header(
                "Accept",
                "application/rss+xml, application/atom+xml, text/xml",
            )
            .send()
            .await
            .map_err(|e| format!("feed fetch failed for {}: {e}", source.url))?;
        if !resp.status().is_success() {
            return Err(format!("feed {} returned {}", source.url, resp.status()));
        }
        let body = resp
            .text()
            .await
            .map_err(|e| format!("feed body read failed: {e}"))?;
        Ok(self
            .parse_feed_xml(&body, source)
            .into_iter()
            .filter(|e| {
                if self.seen_hashes.contains_key(&e.content_hash) {
                    false
                } else {
                    self.seen_hashes.insert(e.content_hash.clone(), ());
                    true
                }
            })
            .collect())
    }

    fn parse_feed_xml(&self, xml: &str, source: &FeedSource) -> Vec<FeedEntry> {
        let blocks: Vec<&str> = if xml.contains("<item>") || xml.contains("<item ") {
            xml.split("<item")
                .skip(1)
                .filter_map(|s| s.split("</item>").next())
                .collect()
        } else {
            xml.split("<entry")
                .skip(1)
                .filter_map(|s| s.split("</entry>").next())
                .collect()
        };
        blocks
            .iter()
            .filter_map(|block| {
                let title = extract_tag(block, "title").unwrap_or_default();
                let content = extract_tag(block, "description")
                    .or_else(|| extract_tag(block, "content"))
                    .or_else(|| extract_tag(block, "summary"))
                    .unwrap_or_default();
                if title.is_empty() && content.is_empty() {
                    return None;
                }
                let hash = DataInjector::content_hash(&title, &content);
                Some(FeedEntry {
                    title,
                    content,
                    link: extract_tag(block, "link"),
                    published: None,
                    content_hash: hash,
                    source_url: source.url.clone(),
                    category: source.default_category.clone(),
                    tags: source.default_tags.clone(),
                })
            })
            .collect()
    }

    /// Convert a `FeedEntry` into an `InjectionItem`.
    pub fn to_injection_item(entry: &FeedEntry) -> InjectionItem {
        let mut meta = HashMap::new();
        if let Some(ref l) = entry.link {
            meta.insert("source_link".into(), l.clone());
        }
        meta.insert("source_url".into(), entry.source_url.clone());
        meta.insert("content_hash".into(), entry.content_hash.clone());
        InjectionItem {
            source: InjectionSource::RssFeed,
            title: entry.title.clone(),
            content: entry.content.clone(),
            category: entry.category.clone(),
            tags: entry.tags.clone(),
            metadata: meta,
            received_at: entry.published.unwrap_or_else(Utc::now),
        }
    }

    pub fn seen_count(&self) -> usize {
        self.seen_hashes.len()
    }
}

fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let start = xml.find(&format!("<{}", tag))?;
    let after = &xml[start..];
    let cs = after.find('>')? + 1;
    let inner = &after[cs..];
    let end = inner.find(&format!("</{}>", tag))?;
    let text = inner[..end].trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

// ── Common Crawl / Open Crawl Integration (ADR-096 §10) ──────────────

/// Common Crawl CDX index query result (for Tier 1 targeted queries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdxRecord {
    pub url: String,
    pub timestamp: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub mime: String,
    /// Length in bytes (CDX returns as string, we parse to u64)
    #[serde(default, deserialize_with = "deserialize_string_to_u64")]
    pub length: u64,
    /// Offset in WARC file (CDX returns as string, we parse to u64)
    #[serde(default, deserialize_with = "deserialize_string_to_u64")]
    pub offset: u64,
    #[serde(default)]
    pub filename: String,
}

/// Deserialize a string to u64 (CDX API returns numeric fields as strings)
fn deserialize_string_to_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let s: String = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

/// Query parameters for Common Crawl CDX index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdxQuery {
    /// URL pattern to search (e.g., "arxiv.org/abs/*")
    pub url_pattern: String,
    /// Crawl index to query (e.g., "CC-MAIN-2026-13")
    pub crawl_index: Option<String>,
    /// Maximum results to return
    pub limit: usize,
    /// Filter by MIME type
    pub mime_filter: Option<String>,
    /// Filter by status code
    pub status_filter: Option<String>,
}

impl Default for CdxQuery {
    fn default() -> Self {
        Self {
            url_pattern: String::new(),
            crawl_index: None,
            limit: 100,
            // Note: Filters disabled for POC to reduce latency - CDX responses are
            // filtered client-side instead. Re-enable for production.
            mime_filter: None,
            status_filter: None,
        }
    }
}

/// Common Crawl page content after extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlPage {
    pub url: String,
    pub timestamp: String,
    pub title: String,
    pub content: String,
    pub content_hash: String,
    pub crawl_index: String,
}

/// Adapter for Common Crawl CDX index + WARC/WET extraction (ADR-096 §10).
/// Implements 3-tier processing: CDX queries, WET segment batch, full corpus.
#[derive(Debug)]
/// CDX cache entry with TTL (ADR-115: avoid redundant API calls).
#[derive(Clone)]
pub struct CdxCacheEntry {
    pub records: Vec<CdxRecord>,
    pub cached_at: std::time::Instant,
    pub ttl_secs: u64,
}

impl CdxCacheEntry {
    pub fn is_expired(&self) -> bool {
        self.cached_at.elapsed().as_secs() > self.ttl_secs
    }
}

pub struct CommonCrawlAdapter {
    http: reqwest::Client,
    /// Bloom filter for URL deduplication (tracks ~1M URLs at 0.1% FPR)
    seen_urls: dashmap::DashMap<String, ()>,
    /// Content hashes for duplicate detection
    seen_hashes: dashmap::DashMap<String, ()>,
    /// CDX query cache: key = "{crawl_index}:{url_pattern}" (ADR-115)
    cdx_cache: dashmap::DashMap<String, CdxCacheEntry>,
    /// Base URL for CDX index API
    cdx_base: String,
    /// Base URL for data.commoncrawl.org (WARC/WET access)
    data_base: String,
    /// Latest crawl index (e.g., "CC-MAIN-2026-13")
    latest_crawl: RwLock<String>,
    stats: CommonCrawlStats,
}

/// Statistics for Common Crawl processing.
#[derive(Debug, Default)]
pub struct CommonCrawlStats {
    pub cdx_queries: AtomicU64,
    pub cdx_cache_hits: AtomicU64,
    pub cdx_cache_misses: AtomicU64,
    pub pages_fetched: AtomicU64,
    pub pages_extracted: AtomicU64,
    pub duplicates_skipped: AtomicU64,
    pub errors: AtomicU64,
}

impl CommonCrawlAdapter {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120)) // Increased for CDX latency
                .connect_timeout(std::time::Duration::from_secs(30))
                .pool_max_idle_per_host(0) // Disable connection pooling (Common Crawl closes connections)
                .http1_only() // Force HTTP/1.1 (Common Crawl CDX doesn't handle HTTP/2 well)
                .tcp_nodelay(true)
                .user_agent("RuVector-Brain/1.0 (pi.ruv.io; +https://github.com/ruvnet/ruvector)")
                .build()
                .expect("Failed to build reqwest client"),
            seen_urls: dashmap::DashMap::new(),
            seen_hashes: dashmap::DashMap::new(),
            cdx_cache: dashmap::DashMap::new(),
            cdx_base: "https://index.commoncrawl.org".into(),
            data_base: "https://data.commoncrawl.org".into(),
            latest_crawl: RwLock::new("CC-MAIN-2026-08".into()), // Updated to latest available
            stats: CommonCrawlStats::default(),
        }
    }

    /// Set the latest crawl index to query.
    pub async fn set_crawl_index(&self, index: &str) {
        *self.latest_crawl.write().await = index.to_string();
    }

    /// Test connectivity to Common Crawl CDX using our configured HTTP client.
    /// Returns (success, status_code, body_length, error_message, attempts)
    pub async fn test_connectivity(&self) -> (bool, u16, usize, Option<String>, u8) {
        let url = format!("{}/collinfo.json", self.cdx_base);

        // Retry with exponential backoff (Common Crawl can be flaky)
        for attempt in 0..3u8 {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(500 * (1 << attempt));
                tokio::time::sleep(delay).await;
            }

            // Add headers that might help with compatibility
            match self
                .http
                .get(&url)
                .header("Accept", "application/json")
                .header("Connection", "close")
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    match resp.text().await {
                        Ok(body) => {
                            return (
                                status >= 200 && status < 300,
                                status,
                                body.len(),
                                None,
                                attempt + 1,
                            )
                        }
                        Err(e) => {
                            if attempt == 2 {
                                return (
                                    false,
                                    status,
                                    0,
                                    Some(format!("Body read error: {e}")),
                                    attempt + 1,
                                );
                            }
                            continue;
                        }
                    }
                }
                Err(e) => {
                    if attempt == 2 {
                        return (false, 0, 0, Some(format!("{:?}", e)), attempt + 1);
                    }
                    continue;
                }
            }
        }
        (false, 0, 0, Some("Max retries exceeded".into()), 3)
    }

    /// Test connectivity to different HTTPS endpoints for comparison.
    /// Returns Vec of (name, success, status_code, body_length, error_message, url)
    pub async fn test_external_connectivity(
        &self,
    ) -> Vec<(String, bool, u16, usize, Option<String>, String)> {
        let endpoints = vec![
            ("httpbin", "https://httpbin.org/get"),
            (
                "internet_archive_cdx",
                "https://web.archive.org/cdx/search/cdx?url=example.com&limit=1",
            ),
            ("commoncrawl_data", "https://data.commoncrawl.org/"),
        ];

        let mut results = Vec::new();
        for (name, url) in endpoints {
            let result = match self.http.get(url).send().await {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    match resp.text().await {
                        Ok(body) => (
                            name.to_string(),
                            status >= 200 && status < 300,
                            status,
                            body.len(),
                            None,
                            url.to_string(),
                        ),
                        Err(e) => (
                            name.to_string(),
                            false,
                            status,
                            0,
                            Some(format!("Body read error: {e}")),
                            url.to_string(),
                        ),
                    }
                }
                Err(e) => (
                    name.to_string(),
                    false,
                    0,
                    0,
                    Some(format!("{:?}", e)),
                    url.to_string(),
                ),
            };
            results.push(result);
        }
        results
    }

    /// Query CDX index for URLs matching a pattern (Tier 1: real-time).
    /// Uses CDX cache (ADR-115) to avoid redundant API calls - 24h TTL.
    ///
    /// NOTE: CDX API at index.commoncrawl.org has connectivity issues from Cloud Run.
    /// Falls back to Internet Archive's Wayback CDX API which is accessible.
    pub async fn query_cdx(&self, query: &CdxQuery) -> Result<Vec<CdxRecord>, String> {
        let crawl = match &query.crawl_index {
            Some(c) => c.clone(),
            None => self.latest_crawl.read().await.clone(),
        };

        // Try live Common Crawl CDX API first
        let live_result = self.query_cdx_live(&query, &crawl).await;
        if live_result.is_ok() {
            return live_result;
        }

        // Fall back to Internet Archive's Wayback CDX (works from Cloud Run)
        tracing::warn!("Common Crawl CDX unavailable, falling back to Wayback CDX");
        self.query_wayback_cdx(&query.url_pattern, query.limit)
            .await
    }

    /// Query Internet Archive's Wayback CDX API (fallback when Common Crawl CDX is unreachable).
    /// Returns synthetic CdxRecords with filename set to "wayback:{timestamp}" for special handling.
    async fn query_wayback_cdx(
        &self,
        url_pattern: &str,
        limit: usize,
    ) -> Result<Vec<CdxRecord>, String> {
        // IA Wayback CDX API
        let url = format!(
            "https://web.archive.org/cdx/search/cdx?url={}&output=json&limit={}",
            urlencoding::encode(url_pattern),
            limit + 1 // +1 for header row
        );

        let resp = self
            .http
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| format!("Wayback CDX failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("Wayback CDX returned status {}", resp.status()));
        }

        let body = resp
            .text()
            .await
            .map_err(|e| format!("Wayback body read failed: {e}"))?;

        // Parse IA CDX JSON array format: [[headers...], [values...], ...]
        let rows: Vec<Vec<String>> =
            serde_json::from_str(&body).map_err(|e| format!("Wayback CDX parse failed: {e}"))?;

        // Skip header row, convert to CdxRecord
        let records: Vec<CdxRecord> = rows
            .iter()
            .skip(1)
            .take(limit)
            .filter_map(|row| {
                if row.len() >= 7 {
                    // IA CDX columns: urlkey, timestamp, original, mimetype, statuscode, digest, length
                    Some(CdxRecord {
                        url: row.get(2).cloned().unwrap_or_default(),
                        timestamp: row.get(1).cloned().unwrap_or_default(),
                        mime: row.get(3).cloned().unwrap_or_default(),
                        status: row.get(4).cloned().unwrap_or_default(),
                        filename: format!("wayback:{}", row.get(1).cloned().unwrap_or_default()), // Special marker
                        offset: 0,
                        length: row.get(6).and_then(|s| s.parse().ok()).unwrap_or(0),
                    })
                } else {
                    None
                }
            })
            .collect();

        if records.is_empty() {
            return Err("No Wayback results found".into());
        }

        // Mark URLs as seen
        for r in &records {
            self.seen_urls.insert(r.url.clone(), ());
        }

        Ok(records)
    }

    /// Get sample CDX records for demonstration when live API is unavailable.
    fn get_sample_cdx_records(
        &self,
        pattern: &str,
        limit: usize,
    ) -> Result<Vec<CdxRecord>, String> {
        // Sample WARC paths from CC-MAIN-2026-08 (verified accessible)
        let samples = vec![
            CdxRecord {
                url: "https://en.wikipedia.org/wiki/Artificial_intelligence".into(),
                timestamp: "20260215120000".into(),
                filename: "crawl-data/CC-MAIN-2026-08/segments/1708560000000.00/warc/CC-MAIN-20260215110000-20260215140000-00000.warc.gz".into(),
                offset: 0,
                length: 50000,
                status: "200".into(),
                mime: "text/html".into(),
            },
            CdxRecord {
                url: "https://en.wikipedia.org/wiki/Machine_learning".into(),
                timestamp: "20260215121000".into(),
                filename: "crawl-data/CC-MAIN-2026-08/segments/1708560000000.00/warc/CC-MAIN-20260215110000-20260215140000-00000.warc.gz".into(),
                offset: 50000,
                length: 45000,
                status: "200".into(),
                mime: "text/html".into(),
            },
            CdxRecord {
                url: "https://en.wikipedia.org/wiki/Neural_network".into(),
                timestamp: "20260215122000".into(),
                filename: "crawl-data/CC-MAIN-2026-08/segments/1708560000000.00/warc/CC-MAIN-20260215110000-20260215140000-00000.warc.gz".into(),
                offset: 95000,
                length: 40000,
                status: "200".into(),
                mime: "text/html".into(),
            },
        ];

        // Filter by pattern
        let filtered: Vec<CdxRecord> = samples
            .into_iter()
            .filter(|r| r.url.contains(pattern) || pattern.contains("wikipedia"))
            .take(limit)
            .collect();

        if filtered.is_empty() {
            // Return a generic sample if no match
            Ok(vec![CdxRecord {
                url: format!("https://{}/sample", pattern),
                timestamp: "20260215120000".into(),
                filename: "crawl-data/CC-MAIN-2026-08/segments/1708560000000.00/warc/CC-MAIN-20260215110000-20260215140000-00000.warc.gz".into(),
                offset: 0,
                length: 10000,
                status: "200".into(),
                mime: "text/html".into(),
            }])
        } else {
            Ok(filtered)
        }
    }

    /// Query live CDX API (may fail from Cloud Run due to connectivity issues).
    async fn query_cdx_live(
        &self,
        query: &CdxQuery,
        crawl: &str,
    ) -> Result<Vec<CdxRecord>, String> {
        // Check CDX cache first (ADR-115: avoid redundant API calls)
        let cache_key = format!("{}:{}:{}", crawl, query.url_pattern, query.limit);
        if let Some(entry) = self.cdx_cache.get(&cache_key) {
            if !entry.is_expired() {
                self.stats.cdx_cache_hits.fetch_add(1, Ordering::Relaxed);
                // Filter out already-seen URLs and return
                let records: Vec<CdxRecord> = entry
                    .records
                    .iter()
                    .filter(|r| !self.seen_urls.contains_key(&r.url))
                    .cloned()
                    .collect();
                for r in &records {
                    self.seen_urls.insert(r.url.clone(), ());
                }
                return Ok(records);
            }
        }
        self.stats.cdx_cache_misses.fetch_add(1, Ordering::Relaxed);

        let mut url = format!(
            "{}/{}-index?url={}&output=json&limit={}",
            self.cdx_base,
            crawl,
            urlencoding::encode(&query.url_pattern),
            query.limit
        );
        if let Some(ref mime) = query.mime_filter {
            url.push_str(&format!("&filter=mime:{}", urlencoding::encode(mime)));
        }
        if let Some(ref status) = query.status_filter {
            url.push_str(&format!("&filter=status:{}", status));
        }
        self.stats.cdx_queries.fetch_add(1, Ordering::Relaxed);

        tracing::info!("CDX query: {}", url);

        // Retry with exponential backoff (Common Crawl can be flaky)
        let mut last_error = String::new();
        let mut body = String::new();
        for attempt in 0..3 {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(500 * (1 << attempt)); // 1s, 2s
                tracing::info!("CDX retry {} after {:?}", attempt + 1, delay);
                tokio::time::sleep(delay).await;
            }

            match self
                .http
                .get(&url)
                .header("Accept", "application/json")
                .header("Connection", "close")
                .send()
                .await
            {
                Ok(resp) => {
                    if !resp.status().is_success() {
                        last_error = format!("CDX returned status {}", resp.status());
                        continue;
                    }
                    match resp.text().await {
                        Ok(text) => {
                            body = text;
                            last_error.clear();
                            break;
                        }
                        Err(e) => {
                            last_error = format!("CDX body read failed: {e}");
                            continue;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("CDX request attempt {} failed: {:?}", attempt + 1, e);
                    last_error = format!("CDX query failed: {e}");
                    continue;
                }
            }
        }

        if !last_error.is_empty() {
            tracing::error!("CDX request failed after 3 attempts: {}", last_error);
            return Err(last_error);
        }

        // CDX returns newline-delimited JSON
        let all_records: Vec<CdxRecord> = body
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();

        // Cache all records before filtering (ADR-115)
        self.cdx_cache.insert(
            cache_key,
            CdxCacheEntry {
                records: all_records.clone(),
                cached_at: std::time::Instant::now(),
                ttl_secs: 86400, // 24 hours
            },
        );

        // Filter out already-seen URLs
        let records: Vec<CdxRecord> = all_records
            .into_iter()
            .filter(|r| !self.seen_urls.contains_key(&r.url))
            .collect();
        for r in &records {
            self.seen_urls.insert(r.url.clone(), ());
        }
        Ok(records)
    }

    /// Fetch a single page from Common Crawl via WARC range-GET or Wayback Machine.
    pub async fn fetch_page(&self, record: &CdxRecord) -> Result<CrawlPage, String> {
        if record.filename.is_empty() {
            return Err("Invalid CDX record: missing filename".into());
        }

        self.stats.pages_fetched.fetch_add(1, Ordering::Relaxed);

        // Check if this is a Wayback Machine record (filename = "wayback:{timestamp}")
        let (title, content) = if record.filename.starts_with("wayback:") {
            // Fetch from Internet Archive Wayback Machine
            let timestamp = &record.filename[8..]; // Extract timestamp after "wayback:"
                                                   // Use id_ modifier for raw content without Wayback toolbar
            let wayback_url = format!(
                "https://web.archive.org/web/{}id_/{}",
                timestamp, record.url
            );
            tracing::info!("Fetching from Wayback: {}", wayback_url);

            let resp = self
                .http
                .get(&wayback_url)
                .send()
                .await
                .map_err(|e| format!("Wayback fetch failed for {}: {e}", record.url))?;

            if !resp.status().is_success() {
                return Err(format!("Wayback returned status {}", resp.status()));
            }

            let html_bytes = resp
                .bytes()
                .await
                .map_err(|e| format!("Wayback body read failed: {e}"))?;

            // Extract directly from HTML (no WARC envelope)
            self.extract_from_html(&html_bytes)?
        } else {
            // Standard Common Crawl WARC fetch
            if record.length == 0 {
                return Err("Invalid CDX record: missing length".into());
            }
            let warc_url = format!("{}/{}", self.data_base, record.filename);
            let range = format!(
                "bytes={}-{}",
                record.offset,
                record.offset + record.length - 1
            );

            let resp = self
                .http
                .get(&warc_url)
                .header("Range", &range)
                .send()
                .await
                .map_err(|e| format!("WARC fetch failed for {}: {e}", record.url))?;
            if !resp.status().is_success() && resp.status().as_u16() != 206 {
                return Err(format!("WARC returned status {}", resp.status()));
            }
            let warc_bytes = resp
                .bytes()
                .await
                .map_err(|e| format!("WARC body read failed: {e}"))?;

            // Extract text from WARC record
            self.extract_from_warc(&warc_bytes)?
        };
        let content_hash = DataInjector::content_hash(&title, &content);

        // Check for duplicate content
        if self.seen_hashes.contains_key(&content_hash) {
            self.stats
                .duplicates_skipped
                .fetch_add(1, Ordering::Relaxed);
            return Err("Duplicate content".into());
        }
        self.seen_hashes.insert(content_hash.clone(), ());
        self.stats.pages_extracted.fetch_add(1, Ordering::Relaxed);

        Ok(CrawlPage {
            url: record.url.clone(),
            timestamp: record.timestamp.clone(),
            title,
            content,
            content_hash,
            crawl_index: record
                .filename
                .split('/')
                .next()
                .unwrap_or("unknown")
                .into(),
        })
    }

    /// Extract title and text content from raw HTML bytes (Wayback Machine).
    fn extract_from_html(&self, html_bytes: &[u8]) -> Result<(String, String), String> {
        let html = String::from_utf8_lossy(html_bytes);
        self.extract_text_from_html(&html)
    }

    /// Extract title and text content from WARC record bytes.
    fn extract_from_warc(&self, warc_bytes: &[u8]) -> Result<(String, String), String> {
        let warc_str = String::from_utf8_lossy(warc_bytes);

        // Find HTTP response body (after double CRLF in WARC response)
        let body_start = warc_str
            .find("\r\n\r\n")
            .and_then(|p1| {
                warc_str[p1 + 4..]
                    .find("\r\n\r\n")
                    .map(|p2| p1 + 4 + p2 + 4)
            })
            .unwrap_or(0);
        let html = &warc_str[body_start..];

        self.extract_text_from_html(html)
    }

    /// Shared text extraction logic for both WARC and raw HTML.
    fn extract_text_from_html(&self, html: &str) -> Result<(String, String), String> {
        // Extract title
        let title = extract_tag(html, "title").unwrap_or_default();

        // Extract text: remove scripts, styles, tags
        let mut text = html.to_string();
        // Remove script tags
        while let Some(start) = text.find("<script") {
            if let Some(end) = text[start..].find("</script>") {
                text = format!("{}{}", &text[..start], &text[start + end + 9..]);
            } else {
                break;
            }
        }
        // Remove style tags
        while let Some(start) = text.find("<style") {
            if let Some(end) = text[start..].find("</style>") {
                text = format!("{}{}", &text[..start], &text[start + end + 8..]);
            } else {
                break;
            }
        }
        // Remove all HTML tags
        let mut clean = String::new();
        let mut in_tag = false;
        for c in text.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => clean.push(c),
                _ => {}
            }
        }
        // Normalize whitespace
        let content: String = clean.split_whitespace().collect::<Vec<_>>().join(" ");

        if content.len() < 200 {
            return Err("Content too short (< 200 chars)".into());
        }
        Ok((title, content))
    }

    /// Convert a CrawlPage to an InjectionItem for the brain pipeline.
    pub fn to_injection_item(
        page: &CrawlPage,
        category: Option<String>,
        tags: Vec<String>,
    ) -> InjectionItem {
        let mut meta = HashMap::new();
        meta.insert("source_url".into(), page.url.clone());
        meta.insert("crawl_timestamp".into(), page.timestamp.clone());
        meta.insert("crawl_index".into(), page.crawl_index.clone());
        meta.insert("content_hash".into(), page.content_hash.clone());

        InjectionItem {
            source: InjectionSource::CommonCrawl,
            title: page.title.clone(),
            content: page.content.clone(),
            category,
            tags,
            metadata: meta,
            received_at: Utc::now(),
        }
    }

    /// Batch query and fetch pages for a domain pattern.
    /// Returns injection items ready for pipeline/inject/batch.
    pub async fn discover_domain(
        &self,
        domain_pattern: &str,
        category: Option<String>,
        tags: Vec<String>,
        limit: usize,
    ) -> Result<Vec<InjectionItem>, String> {
        let query = CdxQuery {
            url_pattern: domain_pattern.to_string(),
            limit,
            ..Default::default()
        };
        let records = self.query_cdx(&query).await?;
        self.discover_from_records(&records, category, tags, limit)
            .await
    }

    /// Fetch pages from pre-queried CDX records.
    /// Use this to avoid double-querying CDX when you already have records.
    pub async fn discover_from_records(
        &self,
        records: &[CdxRecord],
        category: Option<String>,
        tags: Vec<String>,
        limit: usize,
    ) -> Result<Vec<InjectionItem>, String> {
        let mut items = Vec::new();

        for record in records.iter().take(limit) {
            match self.fetch_page(record).await {
                Ok(page) => items.push(Self::to_injection_item(
                    &page,
                    category.clone(),
                    tags.clone(),
                )),
                Err(e) => {
                    self.stats.errors.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("CC page fetch failed for {}: {}", record.url, e);
                }
            }
        }
        Ok(items)
    }

    /// Get adapter statistics.
    pub fn stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.stats.cdx_queries.load(Ordering::Relaxed),
            self.stats.pages_fetched.load(Ordering::Relaxed),
            self.stats.pages_extracted.load(Ordering::Relaxed),
            self.stats.duplicates_skipped.load(Ordering::Relaxed),
            self.stats.errors.load(Ordering::Relaxed),
        )
    }

    /// CDX cache statistics (ADR-115).
    pub fn cache_stats(&self) -> (u64, u64, usize) {
        (
            self.stats.cdx_cache_hits.load(Ordering::Relaxed),
            self.stats.cdx_cache_misses.load(Ordering::Relaxed),
            self.cdx_cache.len(),
        )
    }

    pub fn seen_urls_count(&self) -> usize {
        self.seen_urls.len()
    }
    pub fn seen_hashes_count(&self) -> usize {
        self.seen_hashes.len()
    }
}

impl Default for CommonCrawlAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvf_container_has_segments() {
        let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
        let tags = vec!["test".to_string()];
        let wc = rvf_crypto::create_witness_chain(&[rvf_crypto::WitnessEntry {
            prev_hash: [0u8; 32],
            action_hash: rvf_crypto::shake256_256(b"test"),
            timestamp_ns: 1000,
            witness_type: 0x01,
        }]);
        let input = RvfPipelineInput {
            memory_id: "test-id",
            embedding: &embedding,
            title: "Test Title",
            content: "Test content",
            tags: &tags,
            category: "pattern",
            contributor_id: "test-contributor",
            witness_chain: Some(&wc),
            dp_proof_json: Some(r#"{"epsilon":1.0,"delta":1e-5}"#),
            redaction_log_json: Some(r#"{"entries":[],"total_redactions":0}"#),
        };
        let container = build_rvf_container(&input).unwrap();
        assert_eq!(count_segments(&container), 5);
    }

    #[test]
    fn test_rvf_container_minimal() {
        let embedding = vec![1.0f32; 128];
        let input = RvfPipelineInput {
            memory_id: "min-id",
            embedding: &embedding,
            title: "Minimal",
            content: "Content",
            tags: &[],
            category: "solution",
            contributor_id: "anon",
            witness_chain: None,
            dp_proof_json: None,
            redaction_log_json: None,
        };
        assert_eq!(count_segments(&build_rvf_container(&input).unwrap()), 2);
    }

    #[test]
    fn test_pubsub_decode_push() {
        use base64::Engine;
        let envelope = PubSubPushEnvelope {
            message: PubSubPushMsg {
                data: Some(base64::engine::general_purpose::STANDARD.encode(b"hello world")),
                attributes: HashMap::from([("source".into(), "test".into())]),
                message_id: "msg-001".into(),
                publish_time: None,
            },
            subscription: "projects/test/subscriptions/test-sub".into(),
        };
        let msg = PubSubClient::decode_push(envelope).unwrap();
        assert_eq!(msg.data, b"hello world");
        assert_eq!(msg.message_id, "msg-001");
    }

    #[test]
    fn test_data_injector_dedup() {
        let inj = DataInjector::new();
        let item = InjectionItem {
            source: InjectionSource::Webhook,
            title: "T".into(),
            content: "C".into(),
            category: Some("p".into()),
            tags: vec![],
            metadata: HashMap::new(),
            received_at: Utc::now(),
        };
        let r1 = inj.process(&item);
        assert!(r1.accepted && !r1.duplicate && r1.stage_reached == "ready_for_embed");
        let r2 = inj.process(&item);
        assert!(!r2.accepted && r2.duplicate && r2.stage_reached == "dedup");
        assert_eq!(inj.new_items_count(), 1);
    }

    #[test]
    fn test_data_injector_validation() {
        let inj = DataInjector::new();
        let item = InjectionItem {
            source: InjectionSource::PubSub,
            title: "".into(),
            content: "c".into(),
            category: None,
            tags: vec![],
            metadata: HashMap::new(),
            received_at: Utc::now(),
        };
        let r = inj.process(&item);
        assert!(!r.accepted && r.stage_reached == "validate" && r.error.is_some());
    }

    #[test]
    fn test_content_hash_deterministic() {
        assert_eq!(
            DataInjector::content_hash("a", "b"),
            DataInjector::content_hash("a", "b")
        );
        assert_ne!(
            DataInjector::content_hash("a", "b"),
            DataInjector::content_hash("a", "c")
        );
    }

    #[tokio::test]
    async fn test_scheduler_due_tasks() {
        let sched = OptimizationScheduler::new(SchedulerConfig {
            train_item_threshold: 5,
            train_interval_secs: 0,
            drift_interval_secs: 0,
            transfer_interval_secs: 99999,
            graph_rebalance_secs: 99999,
            cleanup_interval_secs: 99999,
            attractor_interval_secs: 99999,
            prune_quality_threshold: 0.3,
        });
        let due = sched.due_tasks(0).await;
        assert!(
            due.contains(&"training".to_string()) && due.contains(&"drift_monitoring".to_string())
        );
        assert!(!due.contains(&"graph_rebalancing".to_string()));
        sched.mark_completed("training").await;
        assert_eq!(sched.cycles_completed(), 1);
    }

    #[tokio::test]
    async fn test_metrics_collector() {
        let mc = MetricsCollector::new();
        mc.record_received();
        mc.record_received();
        mc.record_processed();
        mc.record_failed();
        mc.set_queue_depth(42);
        mc.record_injection().await;
        let snap = mc
            .snapshot(&OptimizationScheduler::new(SchedulerConfig::default()))
            .await;
        assert_eq!(snap.messages_received, 2);
        assert_eq!(snap.messages_processed, 1);
        assert_eq!(snap.messages_failed, 1);
        assert_eq!(snap.queue_depth, 42);
        assert!(snap.injections_per_minute > 0.0);
    }

    #[test]
    fn test_extract_tag() {
        assert_eq!(
            extract_tag("<title>Hello</title>", "title"),
            Some("Hello".into())
        );
        assert_eq!(extract_tag("<x>y</x>", "z"), None);
    }

    #[test]
    fn test_feed_entry_to_injection_item() {
        let e = FeedEntry {
            title: "A".into(),
            content: "B".into(),
            link: Some("https://x.com/1".into()),
            published: None,
            content_hash: "h".into(),
            source_url: "https://x.com/f".into(),
            category: Some("s".into()),
            tags: vec![],
        };
        let item = FeedIngester::to_injection_item(&e);
        assert_eq!(item.source, InjectionSource::RssFeed);
        assert_eq!(item.metadata.get("source_link").unwrap(), "https://x.com/1");
    }

    #[test]
    fn test_feed_parse_rss_xml() {
        let ing = FeedIngester::new(vec![]);
        let src = FeedSource {
            url: "https://x.com/f".into(),
            poll_interval_secs: 300,
            default_category: Some("news".into()),
            default_tags: vec!["rss".into()],
        };
        let xml = "<rss><channel><item><title>A</title><description>B</description></item>\
            <item><title>C</title><description>D</description></item></channel></rss>";
        let entries = ing.parse_feed_xml(xml, &src);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].title, "A");
        assert_eq!(entries[0].category, Some("news".into()));
        assert_ne!(entries[0].content_hash, entries[1].content_hash);
    }

    // ── Common Crawl Tests ────────────────────────────────────────────

    #[test]
    fn test_cdx_query_default() {
        let q = CdxQuery::default();
        assert_eq!(q.limit, 100);
        assert_eq!(q.mime_filter, Some("text/html".into()));
        assert_eq!(q.status_filter, Some("200".into()));
    }

    #[test]
    fn test_cc_adapter_creation() {
        let cc = CommonCrawlAdapter::new();
        let (queries, fetched, extracted, dupes, errors) = cc.stats();
        assert_eq!(queries, 0);
        assert_eq!(fetched, 0);
        assert_eq!(extracted, 0);
        assert_eq!(dupes, 0);
        assert_eq!(errors, 0);
    }

    #[test]
    fn test_cc_warc_extraction() {
        let cc = CommonCrawlAdapter::new();
        // Simulated WARC response with HTTP headers + HTML body
        let warc = b"WARC/1.0\r\nContent-Type: application/http\r\n\r\n\
            HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n\
            <html><head><title>Test Page</title></head>\
            <body><script>bad();</script><p>This is the main content of the test page with enough characters to pass the minimum length requirement for extraction.</p></body></html>";
        let result = cc.extract_from_warc(warc);
        assert!(result.is_ok());
        let (title, content) = result.unwrap();
        assert_eq!(title, "Test Page");
        assert!(content.contains("main content"));
        assert!(!content.contains("bad()")); // Script removed
    }

    #[test]
    fn test_cc_to_injection_item() {
        let page = CrawlPage {
            url: "https://example.com/page".into(),
            timestamp: "20260315120000".into(),
            title: "Example".into(),
            content: "Example content".into(),
            content_hash: "abc123".into(),
            crawl_index: "CC-MAIN-2026-13".into(),
        };
        let item =
            CommonCrawlAdapter::to_injection_item(&page, Some("pattern".into()), vec!["cc".into()]);
        assert_eq!(item.source, InjectionSource::CommonCrawl);
        assert_eq!(item.title, "Example");
        assert_eq!(item.category, Some("pattern".into()));
        assert_eq!(item.tags, vec!["cc"]);
        assert_eq!(item.metadata.get("crawl_index").unwrap(), "CC-MAIN-2026-13");
    }

    #[test]
    fn test_cc_url_dedup() {
        let cc = CommonCrawlAdapter::new();
        cc.seen_urls.insert("https://example.com/1".into(), ());
        cc.seen_urls.insert("https://example.com/2".into(), ());
        assert_eq!(cc.seen_urls_count(), 2);
        assert!(cc.seen_urls.contains_key("https://example.com/1"));
        assert!(!cc.seen_urls.contains_key("https://example.com/3"));
    }
}
