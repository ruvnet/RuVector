//! Web memory persistence layer (ADR-094).
//!
//! Extends FirestoreClient with DashMap-backed storage for WebMemory,
//! WebPageDelta, and LinkEdge objects. Uses the same write-through pattern
//! as the core brain memory store (DashMap hot cache + Firestore REST).

use crate::store::FirestoreClient;
use crate::web_memory::*;
use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::Arc;
use uuid::Uuid;

/// Web memory storage layer, backed by DashMap with optional Firestore persistence.
///
/// Designed to sit alongside the core `FirestoreClient` — shares the same
/// write-through architecture. Separated into its own struct to avoid
/// modifying the core store's field layout.
pub struct WebMemoryStore {
    /// Web memories indexed by ID
    memories: DashMap<Uuid, WebMemory>,
    /// Content hashes for deduplication (INV-2)
    content_hashes: DashMap<String, Uuid>,
    /// Page deltas indexed by delta ID
    deltas: DashMap<Uuid, WebPageDelta>,
    /// Page deltas indexed by URL for evolution queries
    url_deltas: DashMap<String, Vec<Uuid>>,
    /// Link edges indexed by source memory ID
    link_edges: DashMap<Uuid, Vec<LinkEdge>>,
    /// Domain statistics cache
    domain_counts: DashMap<String, usize>,
    /// Firestore client for write-through persistence
    firestore: Arc<FirestoreClient>,
}

impl WebMemoryStore {
    pub fn new(firestore: Arc<FirestoreClient>) -> Self {
        Self {
            memories: DashMap::new(),
            content_hashes: DashMap::new(),
            deltas: DashMap::new(),
            url_deltas: DashMap::new(),
            link_edges: DashMap::new(),
            domain_counts: DashMap::new(),
            firestore,
        }
    }

    // ── Memory CRUD ─────────────────────────────────────────────────

    /// Store a web memory (cache + Firestore write-through).
    ///
    /// Also stores the underlying BrainMemory in the core store for
    /// compatibility with existing search/graph infrastructure.
    pub async fn store(&self, mem: WebMemory) {
        let id = mem.base.id;
        let hash = mem.content_hash.clone();
        let domain = mem.domain.clone();
        let _url = mem.source_url.clone();

        // Write-through to Firestore
        if let Ok(body) = serde_json::to_value(&mem) {
            self.firestore
                .firestore_put_public("web_memories", &id.to_string(), &body)
                .await;
        }

        // Store base BrainMemory in core store for search compatibility
        let _ = self.firestore.store_memory(mem.base.clone()).await;

        // Update indexes
        self.content_hashes.insert(hash, id);
        *self.domain_counts.entry(domain).or_insert(0) += 1;

        self.memories.insert(id, mem);
    }

    /// Store a batch of web memories.
    pub async fn store_batch(&self, memories: Vec<WebMemory>) {
        for mem in memories {
            self.store(mem).await;
        }
    }

    /// Get a web memory by ID.
    pub fn get(&self, id: &Uuid) -> Option<WebMemory> {
        self.memories.get(id).map(|m| m.clone())
    }

    /// Get all content hashes for deduplication.
    pub fn content_hashes(&self) -> HashSet<String> {
        self.content_hashes
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    /// Get all embeddings for novelty scoring.
    pub fn all_embeddings(&self) -> Vec<(Uuid, Vec<f32>)> {
        self.memories
            .iter()
            .map(|e| (e.base.id, e.base.embedding.clone()))
            .collect()
    }

    /// Total number of stored web memories.
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    // ── Temporal Deltas ─────────────────────────────────────────────

    /// Store a page delta.
    pub async fn store_delta(&self, delta: WebPageDelta) {
        let delta_id = delta.id;
        let url = delta.page_url.clone();

        if let Ok(body) = serde_json::to_value(&delta) {
            self.firestore
                .firestore_put_public("web_deltas", &delta_id.to_string(), &body)
                .await;
        }

        // Index by URL for evolution queries
        self.url_deltas
            .entry(url)
            .or_insert_with(Vec::new)
            .push(delta_id);

        self.deltas.insert(delta_id, delta);
    }

    /// Get deltas for a URL (evolution tracking).
    pub fn get_deltas_for_url(&self, url: &str) -> Vec<WebPageDelta> {
        self.url_deltas
            .get(url)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.deltas.get(id).map(|d| d.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total number of stored deltas.
    pub fn delta_count(&self) -> usize {
        self.deltas.len()
    }

    // ── Link Edges ──────────────────────────────────────────────────

    /// Store a link edge.
    pub fn store_link_edge(&self, edge: LinkEdge) {
        let source = edge.source_memory_id;
        self.link_edges
            .entry(source)
            .or_insert_with(Vec::new)
            .push(edge);
    }

    /// Get outbound link edges from a memory.
    pub fn get_link_edges(&self, source_id: &Uuid) -> Vec<LinkEdge> {
        self.link_edges
            .get(source_id)
            .map(|edges| edges.clone())
            .unwrap_or_default()
    }

    /// Total number of stored link edges.
    pub fn link_edge_count(&self) -> usize {
        self.link_edges.iter().map(|e| e.value().len()).sum()
    }

    // ── Queries ─────────────────────────────────────────────────────

    /// Search web memories by domain.
    pub fn search_by_domain(&self, domain: &str, limit: usize) -> Vec<WebMemorySummary> {
        self.memories
            .iter()
            .filter(|e| e.domain == domain)
            .take(limit)
            .map(|e| e.to_summary())
            .collect()
    }

    /// Get memories with novelty above threshold.
    pub fn high_novelty(&self, min_novelty: f32, limit: usize) -> Vec<WebMemorySummary> {
        self.memories
            .iter()
            .filter(|e| e.novelty_score >= min_novelty)
            .take(limit)
            .map(|e| e.to_summary())
            .collect()
    }

    /// Get status overview.
    pub fn status(&self) -> WebMemoryStatus {
        let mut tier_dist = TierDistribution {
            full: 0,
            delta_compressed: 0,
            centroid_merged: 0,
            archived: 0,
        };

        let mut domain_map: std::collections::HashMap<String, (usize, f64, f32)> =
            std::collections::HashMap::new();

        for entry in self.memories.iter() {
            match entry.compression_tier {
                CompressionTier::Full => tier_dist.full += 1,
                CompressionTier::DeltaCompressed => tier_dist.delta_compressed += 1,
                CompressionTier::CentroidMerged => tier_dist.centroid_merged += 1,
                CompressionTier::Archived => tier_dist.archived += 1,
            }

            let (count, qual_sum, nov_sum) = domain_map
                .entry(entry.domain.clone())
                .or_insert((0, 0.0, 0.0));
            *count += 1;
            *qual_sum += entry.base.quality_score.mean();
            *nov_sum += entry.novelty_score;
        }

        let total = self.memories.len();
        let compressed =
            tier_dist.delta_compressed + tier_dist.centroid_merged + tier_dist.archived;
        let compression_ratio = if total > 0 {
            compressed as f64 / total as f64
        } else {
            0.0
        };

        let mut top_domains: Vec<DomainStats> = domain_map
            .into_iter()
            .map(|(domain, (count, qual_sum, nov_sum))| DomainStats {
                domain,
                page_count: count,
                avg_quality: if count > 0 {
                    qual_sum / count as f64
                } else {
                    0.0
                },
                avg_novelty: if count > 0 {
                    nov_sum / count as f32
                } else {
                    0.0
                },
            })
            .collect();
        top_domains.sort_by(|a, b| b.page_count.cmp(&a.page_count));
        top_domains.truncate(20);

        WebMemoryStatus {
            total_web_memories: total,
            total_domains: self.domain_counts.len(),
            total_link_edges: self.link_edge_count(),
            total_page_deltas: self.delta_count(),
            compression_ratio,
            tier_distribution: tier_dist,
            top_domains,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BetaParams, BrainCategory, BrainMemory};
    use chrono::Utc;

    fn make_test_web_memory(url: &str, novelty: f32) -> WebMemory {
        let now = Utc::now();
        let id = Uuid::new_v4();
        let tier = CompressionTier::from_novelty(novelty);
        WebMemory {
            base: BrainMemory {
                id,
                category: BrainCategory::Custom("web".to_string()),
                title: format!("Page: {url}"),
                content: "test content".into(),
                tags: vec![],
                code_snippet: None,
                embedding: vec![0.1; 128],
                contributor_id: "web:test".into(),
                quality_score: BetaParams::new(),
                partition_id: None,
                witness_hash: "test_hash".into(),
                rvf_gcs_path: None,
                redaction_log: None,
                dp_proof: None,
                witness_chain: None,
                created_at: now,
                updated_at: now,
            },
            source_url: url.into(),
            domain: crate::web_ingest::extract_domain(url),
            content_hash: format!("hash_{url}"),
            crawl_timestamp: now,
            crawl_source: "test-crawl".into(),
            language: "en".into(),
            outbound_links: vec![],
            compression_tier: tier,
            novelty_score: novelty,
            quantized_embedding: None,
        }
    }

    #[test]
    fn web_memory_to_summary() {
        let mem = make_test_web_memory("https://example.com/page1", 0.8);
        let summary = mem.to_summary();
        assert_eq!(summary.source_url, "https://example.com/page1");
        assert_eq!(summary.domain, "example.com");
        assert!((summary.novelty_score - 0.8).abs() < f32::EPSILON);
        assert!((summary.quality_score - 0.5).abs() < f64::EPSILON); // BetaParams::new() has mean 0.5
    }

    #[test]
    fn store_content_hashes_dedup() {
        let store = WebMemoryStore::new(Arc::new(FirestoreClient::new()));
        let mem1 = make_test_web_memory("https://a.com", 0.9);
        let mem2 = make_test_web_memory("https://b.com", 0.7);

        // Manually insert to test hash tracking (bypassing async store)
        store
            .content_hashes
            .insert(mem1.content_hash.clone(), mem1.base.id);
        store
            .content_hashes
            .insert(mem2.content_hash.clone(), mem2.base.id);

        let hashes = store.content_hashes();
        assert!(hashes.contains(&mem1.content_hash));
        assert!(hashes.contains(&mem2.content_hash));
        assert_eq!(hashes.len(), 2);
    }

    #[test]
    fn store_link_edges() {
        let store = WebMemoryStore::new(Arc::new(FirestoreClient::new()));
        let src = Uuid::new_v4();
        let tgt = Uuid::new_v4();

        let edge = LinkEdge::new(src, tgt, None, vec![0.0; 128], LinkType::Citation, 0.8);
        store.store_link_edge(edge);

        let edges = store.get_link_edges(&src);
        assert_eq!(edges.len(), 1);
        assert!((edges[0].weight - 0.8).abs() < f64::EPSILON);

        // No edges for unknown source
        assert!(store.get_link_edges(&Uuid::new_v4()).is_empty());
    }

    #[test]
    fn status_reports_tier_distribution() {
        let store = WebMemoryStore::new(Arc::new(FirestoreClient::new()));

        // Insert memories with different tiers
        let mems = vec![
            make_test_web_memory("https://a.com/1", 0.9),  // Full
            make_test_web_memory("https://a.com/2", 0.5),  // Full
            make_test_web_memory("https://b.com/1", 0.1),  // DeltaCompressed
            make_test_web_memory("https://c.com/1", 0.01), // CentroidMerged
        ];

        for mem in mems {
            store.domain_counts.entry(mem.domain.clone()).or_insert(0);
            *store.domain_counts.get_mut(&mem.domain).unwrap() += 1;
            store.memories.insert(mem.base.id, mem);
        }

        let status = store.status();
        assert_eq!(status.total_web_memories, 4);
        assert_eq!(status.tier_distribution.full, 2);
        assert_eq!(status.tier_distribution.delta_compressed, 1);
        assert_eq!(status.tier_distribution.centroid_merged, 1);
        assert!(status.compression_ratio > 0.0);
    }
}
