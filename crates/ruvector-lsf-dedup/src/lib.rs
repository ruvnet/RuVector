//! # ruvector-lsf-dedup
//!
//! Locality-Sensitive Fingerprinting for semantic near-duplicate detection
//! in agent memory stores.
//!
//! Agent memory systems accumulate near-duplicate entries over time. Two
//! observations of the same event, rephrased summaries, or multiple tool
//! calls that retrieve similar facts create redundant vectors that waste
//! capacity, dilute recall, and confuse coherence scoring.
//!
//! This crate provides three fingerprinting strategies to identify and
//! suppress near-duplicates **before** they enter the store, with a
//! tamper-evident proof trail of every dedup decision.
//!
//! | Strategy | Signal | Strength |
//! |----------|--------|----------|
//! | `SimHash` | Random hyperplane sign bits | Fast, cosine-space |
//! | `MinHash` | Jaccard of discretised bins | Set-overlap, quantisation-robust |
//! | `Hybrid`  | SimHash pre-filter + exact cosine verify | High precision, low false-positive |
//!
//! ## References
//! - Charikar 2002 "Similarity Estimation Techniques from Rounding Algorithms" (SimHash)
//! - Broder 1997 "On the Resemblance and Containment of Documents" (MinHash)
//! - Park et al. 2023 "Generative Agents" arXiv:2304.03442
//! - Xu 2026 "Self-Aware Vector Embeddings for RAG" arXiv:2604.20598

pub mod dedup_store;
pub mod minhash;
pub mod simhash;

/// Opaque identifier for a stored memory entry.
pub type MemoryId = u64;

/// A vector memory entry with optional metadata.
#[derive(Clone, Debug)]
pub struct MemoryEntry {
    pub id: MemoryId,
    pub vector: Vec<f32>,
    pub metadata: Option<String>,
}

/// Outcome of a dedup check when inserting a candidate vector.
#[derive(Clone, Debug, PartialEq)]
pub enum DedupResult {
    /// The candidate is unique; it was stored and assigned `id`.
    Unique(MemoryId),
    /// The candidate is a near-duplicate of an existing entry.
    Duplicate {
        /// ID of the existing entry it resembles.
        of: MemoryId,
        /// Estimated similarity in [0, 1].
        similarity: f32,
    },
}

/// One line in the proof trail.
#[derive(Clone, Debug)]
pub struct DedupDecision {
    /// Sequential insert number (0-based).
    pub insert_seq: u64,
    pub result: DedupResult,
    pub method: DedupMethod,
}

/// Which strategy produced the decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DedupMethod {
    SimHash,
    MinHash,
    Hybrid,
}

impl std::fmt::Display for DedupMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DedupMethod::SimHash => write!(f, "SimHash"),
            DedupMethod::MinHash => write!(f, "MinHash"),
            DedupMethod::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Core trait for semantic fingerprinting strategies.
pub trait SemanticFingerprinter: Send + Sync {
    /// The fingerprint type produced by this strategy.
    type Fingerprint: Clone + Send + Sync;
    /// Compute a fingerprint for `vec`.
    fn fingerprint(&self, vec: &[f32]) -> Self::Fingerprint;
    /// Estimate similarity in [0, 1] between two fingerprints.
    fn estimate_similarity(&self, a: &Self::Fingerprint, b: &Self::Fingerprint) -> f32;
}

/// Euclidean L2 distance between two equal-length slices.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity between two vectors (returns NaN if either is zero).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Normalize a vector to unit length in place.
pub fn normalize(v: &mut Vec<f32>) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
