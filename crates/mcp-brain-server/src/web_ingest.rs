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
pub const MAX_BATCH_SIZE: usize = 500;

/// Maximum text length per page (bytes).
pub const MAX_TEXT_LENGTH: usize = 1_000_000;

/// Minimum text length to consider a page worth ingesting.
pub const MIN_TEXT_LENGTH: usize = 50;

/// Chunk size in characters (approximate 512 tokens).
pub const CHUNK_SIZE: usize = 2048;

/// Overlap between chunks in characters.
pub const CHUNK_OVERLAP: usize = 256;

// ── Pipeline Entry Point ────────────────────────────────────────────────

/// Ingest a batch of cleaned web pages into the shared memory plane.
///
/// Returns accepted `WebMemory` objects and ingest statistics. The caller
/// is responsible for persisting accepted memories to the store.
///
/// Deduplication is performed both against `existing_hashes` (prior state)
/// and within the batch itself to prevent duplicate ingestion.
pub fn ingest_batch(
    pages: &[CleanedPage],
    crawl_source: &str,
    embedding_engine: &EmbeddingEngine,
    graph: &mut KnowledgeGraph,
    existing_hashes: &HashSet<String>,
    existing_embeddings: &[(Uuid, Vec<f32>)],
) -> (Vec<WebMemory>, WebIngestResponse) {
    let mut accepted: Vec<WebMemory> = Vec::new();
    let mut rejected = 0usize;
    let mut stats = CompressionStats {
        full_tier: 0,
        delta_compressed: 0,
        centroid_merged: 0,
        dedup_skipped: 0,
    };

    // Track hashes seen within this batch to prevent intra-batch duplicates
    let mut batch_hashes: HashSet<String> = HashSet::new();

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

        // Phase 2: Deduplication via content hash (SHA3-256)
        let content_hash = compute_content_hash(&page.text);
        if existing_hashes.contains(&content_hash) || batch_hashes.contains(&content_hash) {
            stats.dedup_skipped += 1;
            rejected += 1;
            continue;
        }
        batch_hashes.insert(content_hash.clone());

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
        // and already-accepted memories in this batch
        let primary_embedding = embeddings.first().cloned().unwrap_or_else(|| vec![0.0; EMBED_DIM]);
        let batch_embeddings: Vec<(Uuid, Vec<f32>)> = accepted
            .iter()
            .map(|m| (m.base.id, m.base.embedding.clone()))
            .collect();
        let novelty_existing = compute_novelty(&primary_embedding, existing_embeddings);
        let novelty_batch = compute_novelty(&primary_embedding, &batch_embeddings);
        let novelty = novelty_existing.min(novelty_batch);

        // Phase 5: Compression tier (INV-5: deterministic from novelty)
        let tier = CompressionTier::from_novelty(novelty);

        match tier {
            CompressionTier::Full => stats.full_tier += 1,
            CompressionTier::DeltaCompressed => stats.delta_compressed += 1,
            CompressionTier::CentroidMerged => stats.centroid_merged += 1,
            CompressionTier::Archived => {}
        }

        // Phase 6 + 7: Construct WebMemory with witness hash
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
                // Cold tier: content deferred to GCS, not stored in hot memory
                String::new()
            },
            tags: page.tags.clone(),
            code_snippet: None,
            embedding: primary_embedding,
            contributor_id: format!("web:{crawl_source}"),
            quality_score: BetaParams::new(),
            partition_id: None,
            witness_hash,
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

        // Add to knowledge graph for similarity edge construction
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
pub fn validate_page(page: &CleanedPage) -> Result<(), &'static str> {
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

/// Phase 2: Compute SHA3-256 content hash for deduplication.
///
/// Normalizes text (lowercase, collapse whitespace) before hashing to catch
/// semantically identical pages with minor formatting differences.
pub fn compute_content_hash(text: &str) -> String {
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

/// Phase 3: Split text into overlapping chunks (character-based).
///
/// Uses character count (not byte count) for consistent chunking across
/// multi-byte UTF-8 text. CHUNK_SIZE ≈ 512 tokens at ~4 chars/token.
pub fn chunk_text(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= CHUNK_SIZE {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    let step = CHUNK_SIZE - CHUNK_OVERLAP;

    while start < chars.len() {
        let end = (start + CHUNK_SIZE).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);

        if end >= chars.len() {
            break;
        }
        start += step;
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
pub fn extract_domain(url: &str) -> String {
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
        Some(r) if r.lambda < -0.5 => 0.1,  // Very stable — low recrawl priority
        Some(r) if r.lambda < 0.0 => 0.3,   // Stable — moderate priority
        Some(r) if r.lambda > 0.5 => 0.9,   // Chaotic — high recrawl priority
        Some(_) => 0.5,                       // Marginally chaotic — default
        None => 0.5,                          // Unknown domain — default priority
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

    fn make_page(url: &str, text: &str, title: &str) -> CleanedPage {
        CleanedPage {
            url: url.into(),
            text: text.into(),
            title: title.into(),
            meta_description: String::new(),
            links: vec![],
            language: "en".into(),
            embedding: vec![],
            tags: vec![],
        }
    }

    // ── Validation ──────────────────────────────────────────────────

    #[test]
    fn validate_accepts_valid_page() {
        let page = make_page("https://example.com/page", &"a".repeat(100), "Test Page");
        assert!(validate_page(&page).is_ok());
    }

    #[test]
    fn validate_rejects_empty_url() {
        let page = make_page("", &"a".repeat(100), "Test Page");
        assert_eq!(validate_page(&page).unwrap_err(), "empty URL");
    }

    #[test]
    fn validate_rejects_short_text() {
        let page = make_page("https://example.com", "too short", "Title");
        assert_eq!(validate_page(&page).unwrap_err(), "text too short");
    }

    #[test]
    fn validate_rejects_long_text() {
        let page = make_page("https://example.com", &"a".repeat(MAX_TEXT_LENGTH + 1), "Title");
        assert_eq!(validate_page(&page).unwrap_err(), "text too long");
    }

    #[test]
    fn validate_rejects_empty_title() {
        let page = make_page("https://example.com", &"a".repeat(100), "");
        assert_eq!(validate_page(&page).unwrap_err(), "empty title");
    }

    #[test]
    fn validate_rejects_invalid_scheme() {
        let page = make_page("ftp://example.com", &"a".repeat(100), "Title");
        assert_eq!(validate_page(&page).unwrap_err(), "invalid URL scheme");
    }

    // ── Content Hashing ─────────────────────────────────────────────

    #[test]
    fn hash_normalizes_whitespace_and_case() {
        let h1 = compute_content_hash("Hello   World");
        let h2 = compute_content_hash("hello world");
        let h3 = compute_content_hash("  HELLO\tWORLD  ");
        assert_eq!(h1, h2);
        assert_eq!(h2, h3);
    }

    #[test]
    fn hash_differs_for_different_content() {
        let h1 = compute_content_hash("The quick brown fox");
        let h2 = compute_content_hash("The slow brown fox");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_is_64_hex_chars() {
        let h = compute_content_hash("test");
        assert_eq!(h.len(), 64); // SHA3-256 = 32 bytes = 64 hex chars
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ── Chunking ────────────────────────────────────────────────────

    #[test]
    fn chunk_short_text_single_chunk() {
        let chunks = chunk_text("Short text");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short text");
    }

    #[test]
    fn chunk_long_text_multiple_chunks() {
        let text = "a".repeat(5000);
        let chunks = chunk_text(&text);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].chars().count(), CHUNK_SIZE);
    }

    #[test]
    fn chunk_preserves_overlap() {
        let text = "a".repeat(CHUNK_SIZE + 500);
        let chunks = chunk_text(&text);
        assert_eq!(chunks.len(), 2);
        // Second chunk starts at CHUNK_SIZE - CHUNK_OVERLAP
        let overlap_start = CHUNK_SIZE - CHUNK_OVERLAP;
        let first_tail: String = chunks[0].chars().skip(overlap_start).collect();
        let second_head: String = chunks[1].chars().take(CHUNK_OVERLAP).collect();
        assert_eq!(first_tail, second_head);
    }

    #[test]
    fn chunk_handles_multibyte_utf8() {
        // Each emoji is 4 bytes but 1 char — chunking should be char-based
        let text = "🔥".repeat(CHUNK_SIZE + 100);
        let chunks = chunk_text(&text);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].chars().count(), CHUNK_SIZE);
        // Verify no broken UTF-8
        for chunk in &chunks {
            assert!(chunk.is_ascii() || chunk.chars().count() > 0);
        }
    }

    #[test]
    fn chunk_exactly_at_boundary() {
        let text = "x".repeat(CHUNK_SIZE);
        let chunks = chunk_text(&text);
        assert_eq!(chunks.len(), 1);
    }

    // ── Domain Extraction ───────────────────────────────────────────

    #[test]
    fn extract_domain_basic() {
        assert_eq!(extract_domain("https://example.com/path"), "example.com");
        assert_eq!(extract_domain("http://sub.example.com/"), "sub.example.com");
        assert_eq!(extract_domain("https://EXAMPLE.COM/Path"), "example.com");
    }

    #[test]
    fn extract_domain_with_port() {
        assert_eq!(extract_domain("https://example.com:8080/path"), "example.com:8080");
    }

    #[test]
    fn extract_domain_malformed() {
        assert_eq!(extract_domain("not-a-url"), "not-a-url");
    }

    // ── Novelty Scoring ─────────────────────────────────────────────

    #[test]
    fn novelty_empty_corpus_returns_one() {
        let emb = vec![1.0; 128];
        assert_eq!(compute_novelty(&emb, &[]), 1.0);
    }

    #[test]
    fn novelty_identical_returns_near_zero() {
        let emb = vec![1.0; 128];
        let existing = vec![(Uuid::new_v4(), vec![1.0; 128])];
        let novelty = compute_novelty(&emb, &existing);
        assert!(novelty < 0.01, "novelty={novelty}, expected < 0.01");
    }

    #[test]
    fn novelty_orthogonal_returns_high() {
        // Two orthogonal vectors should have cosine ~0, novelty ~1
        let mut emb_a = vec![0.0; 128];
        emb_a[0] = 1.0;
        let mut emb_b = vec![0.0; 128];
        emb_b[1] = 1.0;
        let existing = vec![(Uuid::new_v4(), emb_b)];
        let novelty = compute_novelty(&emb_a, &existing);
        assert!(novelty > 0.9, "novelty={novelty}, expected > 0.9");
    }

    // ── Witness Hash ────────────────────────────────────────────────

    #[test]
    fn witness_hash_deterministic() {
        let h1 = compute_witness_hash("abc123", "mem-001");
        let h2 = compute_witness_hash("abc123", "mem-001");
        assert_eq!(h1, h2);
    }

    #[test]
    fn witness_hash_differs_by_input() {
        let h1 = compute_witness_hash("abc123", "mem-001");
        let h2 = compute_witness_hash("abc123", "mem-002");
        assert_ne!(h1, h2);
    }

    // ── Truncation ──────────────────────────────────────────────────

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_at_byte_boundary() {
        assert_eq!(truncate("hello world", 5), "hello");
    }

    #[test]
    fn truncate_preserves_utf8_boundary() {
        // "日" is 3 bytes. Truncating at byte 4 should back up to byte 3.
        let s = "日本語";
        let result = truncate(s, 4);
        assert_eq!(result, "日");
    }

    // ── Attractor Integration ───────────────────────────────────────

    #[test]
    fn attractor_recrawl_priority_stable() {
        let mut results = HashMap::new();
        results.insert("stable.com".into(), temporal_attractor_studio::LyapunovResult {
            lambda: -1.0,
            lyapunov_time: 1.0,
            doubling_time: 0.693,
            points_used: 20,
            dimension: 128,
            pairs_found: 10,
        });
        assert_eq!(attractor_recrawl_priority("stable.com", &results), 0.1);
    }

    #[test]
    fn attractor_recrawl_priority_chaotic() {
        let mut results = HashMap::new();
        results.insert("chaotic.com".into(), temporal_attractor_studio::LyapunovResult {
            lambda: 1.0,
            lyapunov_time: 1.0,
            doubling_time: 0.693,
            points_used: 20,
            dimension: 128,
            pairs_found: 10,
        });
        assert_eq!(attractor_recrawl_priority("chaotic.com", &results), 0.9);
    }

    #[test]
    fn attractor_recrawl_priority_unknown() {
        let results = HashMap::new();
        assert_eq!(attractor_recrawl_priority("unknown.com", &results), 0.5);
    }

    #[test]
    fn attractor_recrawl_priority_marginal() {
        let mut results = HashMap::new();
        // lambda=0.3 is > 0 but ≤ 0.5 — hits the Some(_) arm
        results.insert("marginal.com".into(), temporal_attractor_studio::LyapunovResult {
            lambda: 0.3,
            lyapunov_time: 3.33,
            doubling_time: 2.31,
            points_used: 20,
            dimension: 128,
            pairs_found: 10,
        });
        assert_eq!(attractor_recrawl_priority("marginal.com", &results), 0.5);
    }
}
