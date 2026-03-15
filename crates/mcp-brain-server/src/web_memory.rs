//! Web memory types for π.ruv.io shared web memory platform (ADR-094).
//!
//! Extends the brain server's memory model with web-sourced memory objects,
//! temporal page deltas, link edges for graph construction, and compression
//! tier management.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{BetaParams, BrainCategory, BrainMemory};

// ── Core Web Memory Types ───────────────────────────────────────────────

/// A web-sourced memory object in the shared memory plane.
/// Extends BrainMemory with provenance, temporal compression, and link metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebMemory {
    /// Core brain memory (embedding, quality, witness, etc.)
    #[serde(flatten)]
    pub base: BrainMemory,
    /// Source URL (canonical, after redirect resolution)
    pub source_url: String,
    /// Domain extracted from source_url
    pub domain: String,
    /// Content hash (SHAKE-256) for deduplication
    pub content_hash: String,
    /// Crawl timestamp (when the content was fetched)
    pub crawl_timestamp: DateTime<Utc>,
    /// Crawl source identifier (e.g., "cc-2026-09")
    pub crawl_source: String,
    /// Language code (ISO 639-1)
    pub language: String,
    /// Outbound link URLs (for graph construction)
    pub outbound_links: Vec<String>,
    /// Temporal compression tier
    pub compression_tier: CompressionTier,
    /// Novelty score relative to existing memory (0.0 = duplicate, 1.0 = entirely new)
    pub novelty_score: f32,
}

/// Temporal compression tiers (ADR-017 alignment).
///
/// Determines how much data is retained for a given memory object based on
/// its novelty relative to existing memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionTier {
    /// Full embedding + content stored (high novelty, first seen)
    Full,
    /// Embedding stored, content as delta from nearest neighbor
    DeltaCompressed,
    /// Only centroid contribution stored (near-duplicate)
    CentroidMerged,
    /// Archived — retrievable from GCS but not in hot memory
    Archived,
}

impl CompressionTier {
    /// Assign tier based on novelty score.
    ///
    /// INV-5: Compression tier assignment matches novelty score deterministically.
    pub fn from_novelty(novelty: f32) -> Self {
        if novelty < 0.05 {
            Self::CentroidMerged
        } else if novelty < 0.20 {
            Self::DeltaCompressed
        } else {
            Self::Full
        }
    }

    /// Whether this tier should be kept in hot (in-memory) storage.
    pub fn is_hot(&self) -> bool {
        matches!(self, Self::Full | Self::DeltaCompressed)
    }
}

// ── Temporal Evolution ──────────────────────────────────────────────────

/// Tracks how a web page changes across crawls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebPageDelta {
    pub id: Uuid,
    /// URL of the page being tracked
    pub page_url: String,
    /// Memory ID of the previous version
    pub previous_memory_id: Uuid,
    /// Memory ID of the current version
    pub current_memory_id: Uuid,
    /// Cosine similarity between previous and current embeddings
    pub embedding_drift: f32,
    /// Content diff summary
    pub content_delta: ContentDelta,
    /// Time between crawls
    #[serde(with = "duration_serde")]
    pub time_delta: Duration,
    /// Whether this delta crossed a mincut partition boundary
    pub boundary_crossing: bool,
    /// Timestamp of this delta
    pub created_at: DateTime<Utc>,
}

/// Structured content change classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Content unchanged (hash match)
    Unchanged,
    /// Minor update (< 5% token change)
    Minor {
        changed_tokens: usize,
        total_tokens: usize,
    },
    /// Major revision (≥ 5% token change)
    Major {
        summary: String,
        changed_tokens: usize,
    },
    /// Complete rewrite (cosine < 0.7)
    Rewrite,
}

impl ContentDelta {
    /// Classify content change based on token counts and embedding similarity.
    pub fn classify(
        changed_tokens: usize,
        total_tokens: usize,
        embedding_cosine: f32,
    ) -> Self {
        if changed_tokens == 0 {
            return Self::Unchanged;
        }
        if embedding_cosine < 0.7 {
            return Self::Rewrite;
        }
        let change_ratio = if total_tokens > 0 {
            changed_tokens as f64 / total_tokens as f64
        } else {
            1.0
        };
        if change_ratio < 0.05 {
            Self::Minor {
                changed_tokens,
                total_tokens,
            }
        } else {
            Self::Major {
                summary: format!("{changed_tokens}/{total_tokens} tokens changed"),
                changed_tokens,
            }
        }
    }

    /// Whether this delta is significant enough to warrant re-embedding.
    pub fn is_significant(&self) -> bool {
        matches!(self, Self::Major { .. } | Self::Rewrite)
    }
}

// ── Graph Construction ──────────────────────────────────────────────────

/// A directed edge from source page to target page in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkEdge {
    pub source_memory_id: Uuid,
    pub target_memory_id: Uuid,
    /// Anchor text embedding (if meaningful anchor text exists)
    pub anchor_embedding: Option<Vec<f32>>,
    /// Link context: surrounding paragraph embedding
    pub context_embedding: Vec<f32>,
    /// Link type classification
    pub link_type: LinkType,
    /// Weight based on semantic relevance (INV-6: clamped to [0.0, 1.0])
    pub weight: f64,
}

impl LinkEdge {
    /// Create a new link edge with weight clamped to [0.0, 1.0] (INV-6).
    pub fn new(
        source: Uuid,
        target: Uuid,
        anchor_embedding: Option<Vec<f32>>,
        context_embedding: Vec<f32>,
        link_type: LinkType,
        weight: f64,
    ) -> Self {
        Self {
            source_memory_id: source,
            target_memory_id: target,
            anchor_embedding,
            context_embedding,
            link_type,
            weight: weight.clamp(0.0, 1.0),
        }
    }
}

/// Classification of link relationships between pages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkType {
    /// Informational reference
    Citation,
    /// Navigational (same-site)
    Navigation,
    /// Supporting evidence
    Evidence,
    /// Contradiction or rebuttal (INV-8: must be symmetric)
    Contradiction,
    /// Unknown / unclassified
    Unknown,
}

// ── API Request/Response Types ──────────────────────────────────────────

/// Batch ingest request for web pages.
#[derive(Debug, Deserialize)]
pub struct WebIngestRequest {
    /// Batch of cleaned pages to ingest
    pub pages: Vec<CleanedPage>,
    /// Crawl source identifier
    pub crawl_source: String,
}

/// A cleaned web page ready for ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanedPage {
    /// Canonical URL
    pub url: String,
    /// Cleaned text content (HTML stripped)
    pub text: String,
    /// Page title
    pub title: String,
    /// Meta description
    #[serde(default)]
    pub meta_description: String,
    /// Outbound link URLs
    #[serde(default)]
    pub links: Vec<String>,
    /// Language code (ISO 639-1)
    #[serde(default = "default_language")]
    pub language: String,
    /// Optional pre-computed embedding (128-dim)
    #[serde(default)]
    pub embedding: Vec<f32>,
    /// Content tags
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_language() -> String {
    "en".to_string()
}

/// Response for batch ingest.
#[derive(Debug, Serialize)]
pub struct WebIngestResponse {
    /// Number of pages accepted
    pub accepted: usize,
    /// Number of pages rejected (duplicates, validation failures)
    pub rejected: usize,
    /// IDs of accepted memories
    pub memory_ids: Vec<Uuid>,
    /// Compression statistics
    pub compression: CompressionStats,
}

/// Compression statistics for an ingest batch.
#[derive(Debug, Serialize)]
pub struct CompressionStats {
    pub full_tier: usize,
    pub delta_compressed: usize,
    pub centroid_merged: usize,
    pub dedup_skipped: usize,
}

/// Query for web search.
#[derive(Debug, Deserialize)]
pub struct WebSearchQuery {
    /// Text query
    pub q: Option<String>,
    /// Pre-computed query embedding
    pub embedding: Option<Vec<f32>>,
    /// Filter by domain
    pub domain: Option<String>,
    /// Filter by language
    pub language: Option<String>,
    /// Filter by crawl source
    pub crawl_source: Option<String>,
    /// Maximum results
    pub limit: Option<usize>,
    /// Minimum novelty score
    pub min_novelty: Option<f32>,
}

/// Query for contradiction detection.
#[derive(Debug, Deserialize)]
pub struct ContradictionQuery {
    /// Topic to search for contradictions
    pub topic: String,
    /// Minimum contradiction strength
    pub min_strength: Option<f64>,
    /// Maximum results
    pub limit: Option<usize>,
}

/// A detected contradiction between two sources.
#[derive(Debug, Serialize)]
pub struct Contradiction {
    pub source_a: WebMemorySummary,
    pub source_b: WebMemorySummary,
    /// Contradiction strength (higher = more contradictory)
    pub strength: f64,
    /// Mincut partition boundary info
    pub partition_boundary: bool,
}

/// Lightweight summary of a web memory (avoids full embedding in responses).
#[derive(Debug, Serialize)]
pub struct WebMemorySummary {
    pub id: Uuid,
    pub title: String,
    pub source_url: String,
    pub domain: String,
    pub novelty_score: f32,
    pub quality_score: f64,
    pub crawl_timestamp: DateTime<Utc>,
}

/// Query for temporal evolution tracking.
#[derive(Debug, Deserialize)]
pub struct EvolutionQuery {
    /// URL or topic to track
    pub url: Option<String>,
    pub topic: Option<String>,
    /// Time range
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

/// Response for evolution tracking.
#[derive(Debug, Serialize)]
pub struct EvolutionResponse {
    pub url: Option<String>,
    pub deltas: Vec<WebPageDelta>,
    pub total_drift: f64,
    pub trend: String,
}

/// Query for novelty detection.
#[derive(Debug, Deserialize)]
pub struct NoveltyQuery {
    /// Only show clusters emerging after this date
    pub since: Option<DateTime<Utc>>,
    /// Minimum cluster size
    pub min_size: Option<usize>,
    pub limit: Option<usize>,
}

/// An emerging knowledge cluster.
#[derive(Debug, Serialize)]
pub struct EmergingCluster {
    pub cluster_id: u32,
    pub dominant_topic: String,
    pub size: usize,
    pub avg_novelty: f32,
    pub first_seen: DateTime<Utc>,
    pub growth_rate: f64,
}

/// Web memory status overview.
#[derive(Debug, Serialize)]
pub struct WebMemoryStatus {
    pub total_web_memories: usize,
    pub total_domains: usize,
    pub total_link_edges: usize,
    pub total_page_deltas: usize,
    pub compression_ratio: f64,
    pub tier_distribution: TierDistribution,
    pub top_domains: Vec<DomainStats>,
}

#[derive(Debug, Serialize)]
pub struct TierDistribution {
    pub full: usize,
    pub delta_compressed: usize,
    pub centroid_merged: usize,
    pub archived: usize,
}

#[derive(Debug, Serialize)]
pub struct DomainStats {
    pub domain: String,
    pub page_count: usize,
    pub avg_quality: f64,
    pub avg_novelty: f32,
}

// ── Duration Serde ──────────────────────────────────────────────────────

mod duration_serde {
    use chrono::Duration;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        duration.num_seconds().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        let secs = i64::deserialize(deserializer)?;
        Ok(Duration::seconds(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_tier_from_novelty() {
        assert_eq!(CompressionTier::from_novelty(0.0), CompressionTier::CentroidMerged);
        assert_eq!(CompressionTier::from_novelty(0.04), CompressionTier::CentroidMerged);
        assert_eq!(CompressionTier::from_novelty(0.05), CompressionTier::DeltaCompressed);
        assert_eq!(CompressionTier::from_novelty(0.19), CompressionTier::DeltaCompressed);
        assert_eq!(CompressionTier::from_novelty(0.20), CompressionTier::Full);
        assert_eq!(CompressionTier::from_novelty(1.0), CompressionTier::Full);
    }

    #[test]
    fn compression_tier_is_hot() {
        assert!(CompressionTier::Full.is_hot());
        assert!(CompressionTier::DeltaCompressed.is_hot());
        assert!(!CompressionTier::CentroidMerged.is_hot());
        assert!(!CompressionTier::Archived.is_hot());
    }

    #[test]
    fn content_delta_classify() {
        assert!(matches!(ContentDelta::classify(0, 100, 1.0), ContentDelta::Unchanged));
        assert!(matches!(ContentDelta::classify(3, 100, 0.99), ContentDelta::Minor { .. }));
        assert!(matches!(ContentDelta::classify(10, 100, 0.85), ContentDelta::Major { .. }));
        assert!(matches!(ContentDelta::classify(50, 100, 0.5), ContentDelta::Rewrite));
    }

    #[test]
    fn link_edge_weight_clamped() {
        let edge = LinkEdge::new(Uuid::new_v4(), Uuid::new_v4(), None, vec![0.0; 128], LinkType::Citation, 1.5);
        assert_eq!(edge.weight, 1.0);

        let edge2 = LinkEdge::new(Uuid::new_v4(), Uuid::new_v4(), None, vec![0.0; 128], LinkType::Citation, -0.5);
        assert_eq!(edge2.weight, 0.0);
    }
}
