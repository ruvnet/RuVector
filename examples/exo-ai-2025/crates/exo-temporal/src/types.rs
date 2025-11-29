//! Core type definitions for temporal memory

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use uuid::Uuid;

/// Substrate time representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SubstrateTime(DateTime<Utc>);

impl SubstrateTime {
    /// Minimum representable time
    pub const MIN: Self = Self(DateTime::<Utc>::MIN_UTC);

    /// Maximum representable time
    pub const MAX: Self = Self(DateTime::<Utc>::MAX_UTC);

    /// Current substrate time
    pub fn now() -> Self {
        Self(Utc::now())
    }

    /// Create from timestamp
    pub fn from_timestamp(secs: i64, nsecs: u32) -> Option<Self> {
        DateTime::from_timestamp(secs, nsecs).map(Self)
    }

    /// Get inner timestamp
    pub fn timestamp(&self) -> i64 {
        self.0.timestamp()
    }

    /// Get nanoseconds
    pub fn timestamp_nanos(&self) -> i64 {
        self.0.timestamp_nanos_opt().unwrap_or(0)
    }

    /// Duration since another time
    pub fn duration_since(&self, other: &SubstrateTime) -> chrono::Duration {
        self.0 - other.0
    }

    /// Absolute difference between times
    pub fn abs_diff(&self, other: &SubstrateTime) -> chrono::Duration {
        let diff = self.0 - other.0;
        if diff < chrono::Duration::zero() {
            -diff
        } else {
            diff
        }
    }
}

impl Default for SubstrateTime {
    fn default() -> Self {
        Self::now()
    }
}

/// Unique identifier for a pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PatternId(Uuid);

impl PatternId {
    /// Generate new random pattern ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get inner UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for PatternId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for PatternId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Pattern metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Key-value pairs
    pub fields: std::collections::HashMap<String, String>,
}

impl Metadata {
    /// Create empty metadata
    pub fn new() -> Self {
        Self {
            fields: std::collections::HashMap::new(),
        }
    }

    /// Insert field
    pub fn insert(&mut self, key: String, value: String) {
        self.fields.insert(key, value);
    }

    /// Get field
    pub fn get(&self, key: &str) -> Option<&String> {
        self.fields.get(key)
    }
}

impl Default for Metadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern representation in temporal memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Unique identifier
    pub id: PatternId,
    /// Vector embedding
    pub embedding: Vec<f32>,
    /// Metadata
    pub metadata: Metadata,
    /// Temporal origin
    pub timestamp: SubstrateTime,
    /// Causal antecedents (patterns that led to this one)
    pub antecedents: Vec<PatternId>,
    /// Salience score (importance metric)
    pub salience: f32,
    /// Access count
    pub access_count: usize,
    /// Last access time
    pub last_accessed: SubstrateTime,
}

impl Pattern {
    /// Create new pattern
    pub fn new(embedding: Vec<f32>, metadata: Metadata) -> Self {
        let now = SubstrateTime::now();
        Self {
            id: PatternId::new(),
            embedding,
            metadata,
            timestamp: now,
            antecedents: Vec::new(),
            salience: 1.0,
            access_count: 0,
            last_accessed: now,
        }
    }

    /// Create with antecedents
    pub fn with_antecedents(
        embedding: Vec<f32>,
        metadata: Metadata,
        antecedents: Vec<PatternId>,
    ) -> Self {
        let mut pattern = Self::new(embedding, metadata);
        pattern.antecedents = antecedents;
        pattern
    }

    /// Update access tracking
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = SubstrateTime::now();
    }
}

/// Query for pattern retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Query vector embedding
    pub embedding: Vec<f32>,
    /// Origin pattern (for causal queries)
    pub origin: Option<PatternId>,
    /// Number of results requested
    pub k: usize,
}

impl Query {
    /// Create from embedding
    pub fn from_embedding(embedding: Vec<f32>) -> Self {
        Self {
            embedding,
            origin: None,
            k: 10,
        }
    }

    /// Set origin for causal queries
    pub fn with_origin(mut self, origin: PatternId) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Set number of results
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Compute hash for caching
    pub fn hash(&self) -> u64 {
        use ahash::AHasher;
        let mut hasher = AHasher::default();
        for &val in &self.embedding {
            val.to_bits().hash(&mut hasher);
        }
        if let Some(origin) = &self.origin {
            origin.hash(&mut hasher);
        }
        self.k.hash(&mut hasher);
        hasher.finish()
    }
}

/// Result from causal query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalResult {
    /// Retrieved pattern
    pub pattern: Pattern,
    /// Similarity score
    pub similarity: f32,
    /// Causal distance (edges in causal graph)
    pub causal_distance: Option<usize>,
    /// Temporal distance
    pub temporal_distance: chrono::Duration,
    /// Combined relevance score
    pub combined_score: f32,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Pattern ID
    pub id: PatternId,
    /// Pattern
    pub pattern: Pattern,
    /// Similarity score
    pub score: f32,
}

/// Time range for queries
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time (inclusive)
    pub start: SubstrateTime,
    /// End time (inclusive)
    pub end: SubstrateTime,
}

impl TimeRange {
    /// Create new time range
    pub fn new(start: SubstrateTime, end: SubstrateTime) -> Self {
        Self { start, end }
    }

    /// Check if time is within range
    pub fn contains(&self, time: &SubstrateTime) -> bool {
        time >= &self.start && time <= &self.end
    }

    /// Past cone (everything before reference time)
    pub fn past(reference: SubstrateTime) -> Self {
        Self {
            start: SubstrateTime::MIN,
            end: reference,
        }
    }

    /// Future cone (everything after reference time)
    pub fn future(reference: SubstrateTime) -> Self {
        Self {
            start: reference,
            end: SubstrateTime::MAX,
        }
    }
}
