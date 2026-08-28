use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::snapshot;

/// Maximum supported vector dimensionality.
pub const MAX_DIMENSIONS: u32 = 65_536;
/// Maximum number of entries in one index.
pub const MAX_CAPACITY: u32 = 262_144;
/// Maximum aggregate number of stored `f32` values.
pub const MAX_TOTAL_VALUES: u64 = 16_777_216;
/// Maximum number of results returned by a single query.
pub const MAX_SEARCH_RESULTS: usize = 65_536;

/// Exact-search distance metric.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum DistanceMetric {
    /// Cosine similarity. Higher scores are better.
    Cosine = 1,
    /// Negative squared Euclidean distance. Higher scores are better.
    L2 = 2,
    /// Dot-product similarity. Higher scores are better.
    Dot = 3,
}

impl TryFrom<u32> for DistanceMetric {
    type Error = IndexError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Cosine),
            2 => Ok(Self::L2),
            3 => Ok(Self::Dot),
            _ => Err(IndexError::InvalidMetric(value)),
        }
    }
}

/// Validated resource and metric configuration for an index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexConfig {
    /// Number of values required in every vector.
    pub dimensions: u32,
    /// Maximum number of distinct vector IDs.
    pub capacity: u32,
    /// Metric used for all searches.
    pub metric: DistanceMetric,
}

impl IndexConfig {
    /// Validates dimensions, capacity, and the aggregate memory budget.
    pub fn validate(self) -> Result<Self, IndexError> {
        if self.dimensions == 0 || self.dimensions > MAX_DIMENSIONS {
            return Err(IndexError::InvalidDimensions(self.dimensions));
        }
        if self.capacity == 0 || self.capacity > MAX_CAPACITY {
            return Err(IndexError::InvalidCapacity(self.capacity));
        }
        let values = u64::from(self.dimensions) * u64::from(self.capacity);
        if values > MAX_TOTAL_VALUES {
            return Err(IndexError::MemoryBudgetExceeded {
                requested_values: values,
                maximum_values: MAX_TOTAL_VALUES,
            });
        }
        Ok(self)
    }
}

/// Errors returned by the bounded index and snapshot codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndexError {
    /// Dimension count is zero or exceeds the compiled bound.
    InvalidDimensions(u32),
    /// Capacity is zero or exceeds the compiled bound.
    InvalidCapacity(u32),
    /// The requested dimensions and capacity exceed the aggregate value budget.
    MemoryBudgetExceeded {
        /// Requested number of `f32` values.
        requested_values: u64,
        /// Maximum number of `f32` values.
        maximum_values: u64,
    },
    /// A vector did not match the configured dimensions.
    DimensionMismatch {
        /// Expected value count.
        expected: u32,
        /// Actual value count.
        actual: usize,
    },
    /// A vector contained NaN or infinity.
    NonFiniteValue,
    /// Cosine similarity was requested for a zero-norm vector.
    ZeroNormVector,
    /// A new ID could not be inserted because the index is full.
    CapacityExceeded(u32),
    /// A search requested more results than the compiled output bound.
    SearchLimitExceeded(usize),
    /// The supplied metric tag is not supported.
    InvalidMetric(u32),
    /// A snapshot exceeded the codec's maximum byte count.
    SnapshotTooLarge {
        /// Actual or requested byte count.
        actual: u64,
        /// Maximum accepted byte count.
        maximum: u64,
    },
    /// Snapshot bytes failed structural or integrity validation.
    CorruptSnapshot(&'static str),
}

impl Display for IndexError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions(value) => write!(formatter, "invalid dimensions: {value}"),
            Self::InvalidCapacity(value) => write!(formatter, "invalid capacity: {value}"),
            Self::MemoryBudgetExceeded {
                requested_values,
                maximum_values,
            } => write!(
                formatter,
                "value budget exceeded: requested {requested_values}, maximum {maximum_values}"
            ),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    formatter,
                    "dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NonFiniteValue => formatter.write_str("vector contains a non-finite value"),
            Self::ZeroNormVector => formatter.write_str("cosine vectors must have non-zero norm"),
            Self::CapacityExceeded(value) => write!(formatter, "index capacity exceeded: {value}"),
            Self::SearchLimitExceeded(value) => write!(formatter, "search limit exceeded: {value}"),
            Self::InvalidMetric(value) => write!(formatter, "invalid metric tag: {value}"),
            Self::SnapshotTooLarge { actual, maximum } => {
                write!(
                    formatter,
                    "snapshot is {actual} bytes; maximum is {maximum}"
                )
            }
            Self::CorruptSnapshot(reason) => write!(formatter, "corrupt snapshot: {reason}"),
        }
    }
}

impl Error for IndexError {}

/// One exact-search result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    /// Application-defined vector ID.
    pub id: u64,
    /// Metric score. All metrics are ordered with higher scores first.
    pub score: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub(crate) id: u64,
    pub(crate) vector: Vec<f32>,
    norm_squared: f64,
}

/// A bounded in-memory exact vector index.
#[derive(Clone, Debug)]
pub struct ExactVectorIndex {
    config: IndexConfig,
    entries: Vec<Entry>,
    positions: HashMap<u64, usize>,
}

impl ExactVectorIndex {
    /// Creates an empty index after validating all resource bounds.
    pub fn new(config: IndexConfig) -> Result<Self, IndexError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            entries: Vec::new(),
            positions: HashMap::new(),
        })
    }

    /// Returns the validated index configuration.
    pub const fn config(&self) -> IndexConfig {
        self.config
    }

    /// Returns the number of stored vectors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns whether an ID is present.
    pub fn contains(&self, id: u64) -> bool {
        self.positions.contains_key(&id)
    }

    /// Returns a stored vector without transferring ownership.
    pub fn get(&self, id: u64) -> Option<&[f32]> {
        self.positions
            .get(&id)
            .map(|position| self.entries[*position].vector.as_slice())
    }

    /// Inserts a new vector or atomically replaces the vector for an existing ID.
    pub fn upsert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError> {
        let norm_squared = self.validate_vector(vector)?;
        if let Some(position) = self.positions.get(&id).copied() {
            self.entries[position] = Entry {
                id,
                vector: vector.to_vec(),
                norm_squared,
            };
            return Ok(());
        }
        if self.entries.len() >= self.config.capacity as usize {
            return Err(IndexError::CapacityExceeded(self.config.capacity));
        }
        let position = self.entries.len();
        self.entries.push(Entry {
            id,
            vector: vector.to_vec(),
            norm_squared,
        });
        self.positions.insert(id, position);
        Ok(())
    }

    /// Removes an ID, returning whether it existed.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(position) = self.positions.remove(&id) else {
            return false;
        };
        self.entries.swap_remove(position);
        if position < self.entries.len() {
            self.positions.insert(self.entries[position].id, position);
        }
        true
    }

    /// Searches every stored vector and returns the best bounded results.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchHit>, IndexError> {
        if limit > MAX_SEARCH_RESULTS {
            return Err(IndexError::SearchLimitExceeded(limit));
        }
        let query_norm_squared = self.validate_vector(query)?;
        let retained = limit.min(self.entries.len());
        if retained == 0 {
            return Ok(Vec::new());
        }

        let mut heap = BinaryHeap::<Reverse<Candidate>>::with_capacity(retained);
        for entry in &self.entries {
            let score = score(
                self.config.metric,
                query,
                query_norm_squared,
                &entry.vector,
                entry.norm_squared,
            );
            let candidate = Candidate {
                id: entry.id,
                score,
            };
            if heap.len() < retained {
                heap.push(Reverse(candidate));
            } else if candidate > heap.peek().expect("non-empty bounded heap").0 {
                heap.pop();
                heap.push(Reverse(candidate));
            }
        }

        let mut candidates: Vec<_> = heap.into_iter().map(|item| item.0).collect();
        candidates.sort_unstable_by(|left, right| right.cmp(left));
        Ok(candidates
            .into_iter()
            .map(|candidate| SearchHit {
                id: candidate.id,
                score: candidate.score,
            })
            .collect())
    }

    /// Serializes a deterministic, checksummed snapshot.
    pub fn snapshot(&self) -> Result<Vec<u8>, IndexError> {
        snapshot::encode(self)
    }

    /// Restores an index only after validating the complete snapshot.
    pub fn from_snapshot(bytes: &[u8]) -> Result<Self, IndexError> {
        snapshot::decode(bytes)
    }

    pub(crate) fn entries(&self) -> &[Entry] {
        &self.entries
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<f64, IndexError> {
        if vector.len() != self.config.dimensions as usize {
            return Err(IndexError::DimensionMismatch {
                expected: self.config.dimensions,
                actual: vector.len(),
            });
        }
        let mut norm_squared = 0.0_f64;
        for value in vector {
            if !value.is_finite() {
                return Err(IndexError::NonFiniteValue);
            }
            let value = f64::from(*value);
            norm_squared += value * value;
        }
        if self.config.metric == DistanceMetric::Cosine && norm_squared == 0.0 {
            return Err(IndexError::ZeroNormVector);
        }
        Ok(norm_squared)
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    id: u64,
    score: f64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}

fn score(
    metric: DistanceMetric,
    left: &[f32],
    left_norm_squared: f64,
    right: &[f32],
    right_norm_squared: f64,
) -> f64 {
    match metric {
        DistanceMetric::Cosine => {
            let dot = dot_product(left, right);
            (dot / (left_norm_squared * right_norm_squared).sqrt()).clamp(-1.0, 1.0)
        }
        DistanceMetric::L2 => {
            let mut squared_distance = 0.0_f64;
            for (left, right) in left.iter().zip(right) {
                let difference = f64::from(*left) - f64::from(*right);
                squared_distance += difference * difference;
            }
            -squared_distance
        }
        DistanceMetric::Dot => dot_product(left, right),
    }
}

fn dot_product(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}
