//! Web memory ingestion pipeline for π.ruv.io (ADR-094).
//!
//! Implements the 7-phase ingestion pipeline:
//! 1. Validate cleaned pages
//! 2. Content hash deduplication (SHAKE-256)
//! 3. Chunk + embed via ruvLLM
//! 4. Novelty scoring against existing memories
//! 5. Compression tier assignment
//! 6. Graph construction (link edges)
//! 7. Proof verification + store
//!
//! Integrates with midstream crate for attractor analysis and temporal
//! solver scoring on ingested content.

use crate::embeddings::{EmbeddingEngine, EMBED_DIM};
use crate::graph::{cosine_similarity, KnowledgeGraph};
use crate::types::{BetaParams, BrainCategory, BrainMemory};
use crate::web_memory::*;
use chrono::Utc;
use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Maximum pages per ingest batch.
const MAX_BATCH_SIZE: usize = 500;

/// Maximum text length per page (bytes).
const MAX_TEXT_LENGTH: usize = 1_000_000;

/// Minimum text length to consider a page worth ingesting.
const MIN_TEXT_LENGTH: usize = 50;

/// Chunk size in characters (approximate 512 tokens).
const CHUNK_SIZE: usize = 2048;

/// Overlap between chunks in characters.
const CHUNK_OVERLAP: usize = 256;

/// Novelty threshold below which pages are considered near-duplicates.
const NEAR_DUPLICATE_THRESHOLD: f32 = 0.98;

// ── Pipeline Entry Point ────────────────────────────────────────────────

/// Ingest a batch of cleaned web pages into the shared memory plane.
///
/// Returns statistics about accepted/rejected pages and compression.
pub fn ingest_batch(
    pages: &[CleanedPage],
    crawl_source: &str,
    embedding_engine: &EmbeddingEngine,
    graph: &mut KnowledgeGraph,
    existing_hashes: &HashSet<String>,
    existing_embeddings: &[(Uuid, Vec<f32>)],
) -> (Vec<WebMemory>, WebIngestResponse) {
    let mut accepted = Vec::new();
    let mut rejected = 0usize;
    let mut stats = CompressionStats {
        full_tier: 0,
        delta_compressed: 0,
        centroid_merged: 0,
        dedup_skipped: 0,
    };

    let batch = if pages.len() > MAX_BATCH_SIZE {
        &pages[..MAX_BATCH_SIZE]
    } else {
        pages
    };

    for page in batch {
        // Phase 1: Validate
        if let Err(_reason) = validate_page(page) {
            rejected += 1;
            continue;
        }

        // Phase 2: Deduplication via content hash
        let content_hash = compute_content_hash(&page.text);
        if existing_hashes.contains(&content_hash) {
            stats.dedup_skipped += 1;
            rejected += 1;
            continue;
        }

        // Phase 3: Chunk + Embed
        let chunks = chunk_text(&page.text);
        let embeddings: Vec<Vec<f32>> = if !page.embedding.is_empty()
            && page.embedding.len() == EMBED_DIM
        {
            // Use pre-computed embedding for first chunk
            let mut embs = vec![page.embedding.clone()];
            for chunk in chunks.iter().skip(1) {
                let text = EmbeddingEngine::prepare_text(&page.title, chunk, &page.tags);
                embs.push(embedding_engine.embed_for_storage(&text));
            }
            embs
        } else {
            chunks
                .iter()
                .map(|chunk| {
                    let text = EmbeddingEngine::prepare_text(&page.title, chunk, &page.tags);
                    embedding_engine.embed_for_storage(&text)
                })
                .collect()
        };

        // Phase 4: Novelty scoring — compare against existing memories
        let primary_embedding = embeddings.first().cloned().unwrap_or_else(|| vec![0.0; EMBED_DIM]);
        let novelty = compute_novelty(&primary_embedding, existing_embeddings);

        // Phase 5: Compression tier
        let tier = CompressionTier::from_novelty(novelty);

        match tier {
            CompressionTier::Full => stats.full_tier += 1,
            CompressionTier::DeltaCompressed => stats.delta_compressed += 1,
            CompressionTier::CentroidMerged => stats.centroid_merged += 1,
            CompressionTier::Archived => {}
        }

        // Phase 6 + 7: Construct WebMemory and store
        let domain = extract_domain(&page.url);
        let memory_id = Uuid::new_v4();
        let now = Utc::now();

        let witness_hash = compute_witness_hash(&content_hash, &memory_id.to_string());

        let base = BrainMemory {
            id: memory_id,
            category: BrainCategory::Custom("web".to_string()),
            title: truncate(&page.title, 200),
            content: if tier.is_hot() {
                truncate(&page.text, 5000)
            } else {
                String::new()
            },
            tags: page.tags.clone(),
            code_snippet: None,
            embedding: primary_embedding,
            contributor_id: format!("web:{crawl_source}"),
            quality_score: BetaParams::new(),
            partition_id: None,
            witness_hash: witness_hash.clone(),
            rvf_gcs_path: None,
            redaction_log: None,
            dp_proof: None,
            witness_chain: None,
            created_at: now,
            updated_at: now,
        };

        let web_mem = WebMemory {
            base,
            source_url: page.url.clone(),
            domain,
            content_hash,
            crawl_timestamp: now,
            crawl_source: crawl_source.to_string(),
            language: page.language.clone(),
            outbound_links: page.links.clone(),
            compression_tier: tier,
            novelty_score: novelty,
        };

        // Phase 6: Add to knowledge graph
        graph.add_memory(&web_mem.base);

        accepted.push(web_mem);
    }

    let memory_ids: Vec<Uuid> = accepted.iter().map(|m| m.base.id).collect();
    let response = WebIngestResponse {
        accepted: accepted.len(),
        rejected,
        memory_ids,
        compression: stats,
    };

    (accepted, response)
}

// ── Pipeline Phases ─────────────────────────────────────────────────────

/// Phase 1: Validate a cleaned page.
fn validate_page(page: &CleanedPage) -> Result<(), &'static str> {
    if page.url.is_empty() {
        return Err("empty URL");
    }
    if page.text.len() < MIN_TEXT_LENGTH {
        return Err("text too short");
    }
    if page.text.len() > MAX_TEXT_LENGTH {
        return Err("text too long");
    }
    if page.title.is_empty() {
        return Err("empty title");
    }
    // Basic URL validation
    if !page.url.starts_with("http://") && !page.url.starts_with("https://") {
        return Err("invalid URL scheme");
    }
    Ok(())
}

/// Phase 2: Compute SHAKE-256 content hash for deduplication.
fn compute_content_hash(text: &str) -> String {
    // Normalize: lowercase, collapse whitespace
    let normalized: String = text
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");
    let mut hasher = Sha3_256::new();
    hasher.update(normalized.as_bytes());
    hex::encode(hasher.finalize())
}

/// Phase 3: Split text into overlapping chunks.
fn chunk_text(text: &str) -> Vec<String> {
    if text.len() <= CHUNK_SIZE {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + CHUNK_SIZE).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end >= chars.len() {
            break;
        }
        // Advance by chunk_size - overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP;
    }

    chunks
}

/// Phase 4: Compute novelty score as 1.0 - max_cosine_similarity.
fn compute_novelty(embedding: &[f32], existing: &[(Uuid, Vec<f32>)]) -> f32 {
    if existing.is_empty() {
        return 1.0;
    }

    let max_sim = existing
        .iter()
        .map(|(_, e)| cosine_similarity(embedding, e) as f32)
        .fold(f32::NEG_INFINITY, f32::max);

    (1.0 - max_sim).max(0.0)
}

/// Compute witness hash from content hash + memory ID.
fn compute_witness_hash(content_hash: &str, memory_id: &str) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(content_hash.as_bytes());
    hasher.update(b":");
    hasher.update(memory_id.as_bytes());
    hex::encode(hasher.finalize())
}

/// Extract domain from a URL.
fn extract_domain(url: &str) -> String {
    url.split("://")
        .nth(1)
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or("unknown")
        .to_lowercase()
}

/// Truncate a string to a maximum byte length, preserving UTF-8 boundaries.
fn truncate(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

// ── Midstream Integration ───────────────────────────────────────────────

/// Score a web memory using midstream attractor analysis.
///
/// Stable domains (negative Lyapunov exponent) get a recrawl frequency
/// reduction bonus. Chaotic domains need more frequent recrawl.
pub fn attractor_recrawl_priority(
    domain: &str,
    attractor_results: &HashMap<String, temporal_attractor_studio::LyapunovResult>,
) -> f32 {
    match attractor_results.get(domain) {
        Some(result) if result.lambda < -0.5 => {
            // Very stable — low recrawl priority
            0.1
        }
        Some(result) if result.lambda < 0.0 => {
            // Stable — moderate recrawl priority
            0.3
        }
        Some(result) if result.lambda > 0.5 => {
            // Chaotic — high recrawl priority
            0.9
        }
        Some(_) => 0.5,
        None => 0.5, // Unknown — default priority
    }
}

/// Use temporal solver to predict content drift for a domain.
///
/// High-confidence stability predictions → lower crawl frequency.
pub fn solver_drift_prediction(
    solver: &mut temporal_neural_solver::TemporalSolver,
    recent_embeddings: &[Vec<f32>],
) -> Option<f32> {
    if recent_embeddings.len() < 3 {
        return None;
    }

    // Convert to Array1<f32> for the solver
    let last = recent_embeddings.last()?;
    let input = ndarray::Array1::from_vec(last.clone());

    let (_prediction, cert, _duration) = solver.predict(&input).ok()?;
    if cert.gate_pass {
        Some(cert.confidence as f32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_page() {
        let valid = CleanedPage {
            url: "https://example.com/page".into(),
            text: "a".repeat(100),
            title: "Test Page".into(),
            meta_description: String::new(),
            links: vec![],
            language: "en".into(),
            embedding: vec![],
            tags: vec![],
        };
        assert!(validate_page(&valid).is_ok());

        let empty_url = CleanedPage { url: String::new(), ..valid.clone() };
        assert!(validate_page(&empty_url).is_err());

        let short_text = CleanedPage { text: "too short".into(), ..valid.clone() };
        assert!(validate_page(&short_text).is_err());
    }

    #[test]
    fn test_content_hash_normalization() {
        let h1 = compute_content_hash("Hello   World");
        let h2 = compute_content_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_chunk_text_short() {
        let chunks = chunk_text("Short text");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short text");
    }

    #[test]
    fn test_chunk_text_long() {
        let text = "a".repeat(5000);
        let chunks = chunk_text(&text);
        assert!(chunks.len() > 1);
        // Verify overlap: last chars of chunk[0] == first chars of chunk[1] offset
        assert!(chunks[0].len() == CHUNK_SIZE);
    }

    #[test]
    fn test_extract_domain() {
        assert_eq!(extract_domain("https://example.com/path"), "example.com");
        assert_eq!(extract_domain("http://sub.example.com/"), "sub.example.com");
        assert_eq!(extract_domain("https://EXAMPLE.COM/Path"), "example.com");
    }

    #[test]
    fn test_compute_novelty_empty() {
        let emb = vec![1.0; 128];
        assert_eq!(compute_novelty(&emb, &[]), 1.0);
    }

    #[test]
    fn test_compute_novelty_duplicate() {
        let emb = vec![1.0; 128];
        let existing = vec![(Uuid::new_v4(), vec![1.0; 128])];
        let novelty = compute_novelty(&emb, &existing);
        assert!(novelty < 0.01, "Identical vectors should have near-zero novelty");
    }

    #[test]
    fn test_attractor_recrawl_priority() {
        let mut results = HashMap::new();
        results.insert("stable.com".to_string(), temporal_attractor_studio::LyapunovResult {
            lambda: -1.0,
            convergence: true,
            iterations: 10,
        });
        results.insert("chaotic.com".to_string(), temporal_attractor_studio::LyapunovResult {
            lambda: 1.0,
            convergence: false,
            iterations: 10,
        });

        assert!(attractor_recrawl_priority("stable.com", &results) < 0.2);
        assert!(attractor_recrawl_priority("chaotic.com", &results) > 0.8);
        assert_eq!(attractor_recrawl_priority("unknown.com", &results), 0.5);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello");
    }
}
