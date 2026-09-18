use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Configuration for the TwinKV style repair pass.
#[derive(Debug, Clone, PartialEq)]
pub struct TwinKvConfig {
    /// Cosine similarity above which two nonlocal keys are considered twins.
    pub similarity_threshold: f32,
    /// Adjacent positions within this distance are excluded from twin matching.
    pub local_window: usize,
    /// Leading positions that may never be evicted by the repair pass.
    pub protected_sink_tokens: usize,
    /// Trailing positions that may never be evicted by the repair pass.
    pub protected_recent_tokens: usize,
    /// Optional hard cap on swaps performed by one repair pass.
    pub max_swaps: Option<usize>,
}

impl Default for TwinKvConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            local_window: 32,
            protected_sink_tokens: 4,
            protected_recent_tokens: 64,
            max_swaps: None,
        }
    }
}

/// A single budget preserving replacement made by the repair pass.
#[derive(Debug, Clone, PartialEq)]
pub struct TwinKvSwap {
    pub admitted_orphan: usize,
    pub evicted_donor: usize,
    pub orphan_best_surviving_similarity: f32,
    pub donor_best_surviving_similarity: f32,
}

/// Result of auditing and repairing an existing retained set.
#[derive(Debug, Clone, PartialEq)]
pub struct TwinKvRepair {
    /// Sorted retained positions after applying all swaps.
    pub retained: Vec<usize>,
    pub swaps: Vec<TwinKvSwap>,
    pub orphan_count: usize,
    pub donor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TwinKvError {
    EmptyKeys,
    EmptyRetainedSet,
    EmptyKeyVector {
        index: usize,
    },
    InconsistentKeyDimension {
        index: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteKey {
        index: usize,
    },
    ZeroNormKey {
        index: usize,
    },
    InvalidThreshold,
    RetainedIndexOutOfRange {
        index: usize,
        key_count: usize,
    },
    DuplicateRetainedIndex {
        index: usize,
    },
}

impl Display for TwinKvError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyKeys => write!(f, "keys must not be empty"),
            Self::EmptyRetainedSet => write!(f, "retained set must not be empty"),
            Self::EmptyKeyVector { index } => write!(f, "key vector at index {index} is empty"),
            Self::InconsistentKeyDimension {
                index,
                expected,
                actual,
            } => write!(
                f,
                "key vector at index {index} has dimension {actual}, expected {expected}"
            ),
            Self::NonFiniteKey { index } => {
                write!(f, "key vector at index {index} contains a non finite value")
            }
            Self::ZeroNormKey { index } => {
                write!(f, "key vector at index {index} has zero norm")
            }
            Self::InvalidThreshold => write!(
                f,
                "similarity threshold must be finite and in the interval from negative one to one"
            ),
            Self::RetainedIndexOutOfRange { index, key_count } => write!(
                f,
                "retained index {index} is outside key range 0..{key_count}"
            ),
            Self::DuplicateRetainedIndex { index } => {
                write!(f, "retained index {index} appears more than once")
            }
        }
    }
}

impl Error for TwinKvError {}

/// Repair an arbitrary KV eviction policy without changing its cache budget.
///
/// The input `retained` set is treated as the wrapped policy's immutable
/// decision. For every token we compute its best nonlocal cosine similarity to
/// a surviving retained token. Evicted tokens below the threshold are orphans.
/// Retained, unprotected tokens at or above the threshold are redundant donors.
/// The most severe orphans replace the most redundant donors one for one.
///
/// This implementation computes only similarities against the retained set,
/// reducing the repair specific work from a full pairwise matrix to O(n K d),
/// where K is the retained budget and d is key dimension.
pub fn repair_retained_set(
    keys: &[Vec<f32>],
    retained: &[usize],
    config: &TwinKvConfig,
) -> Result<TwinKvRepair, TwinKvError> {
    if keys.is_empty() {
        return Err(TwinKvError::EmptyKeys);
    }
    if retained.is_empty() {
        return Err(TwinKvError::EmptyRetainedSet);
    }
    if !config.similarity_threshold.is_finite()
        || !(-1.0..=1.0).contains(&config.similarity_threshold)
    {
        return Err(TwinKvError::InvalidThreshold);
    }

    let dimension = keys[0].len();
    if dimension == 0 {
        return Err(TwinKvError::EmptyKeyVector { index: 0 });
    }

    let mut normalized = Vec::with_capacity(keys.len());
    for (index, key) in keys.iter().enumerate() {
        if key.is_empty() {
            return Err(TwinKvError::EmptyKeyVector { index });
        }
        if key.len() != dimension {
            return Err(TwinKvError::InconsistentKeyDimension {
                index,
                expected: dimension,
                actual: key.len(),
            });
        }
        if key.iter().any(|value| !value.is_finite()) {
            return Err(TwinKvError::NonFiniteKey { index });
        }

        let norm_sq = key.iter().map(|value| value * value).sum::<f32>();
        if !norm_sq.is_finite() || norm_sq <= f32::EPSILON {
            return Err(TwinKvError::ZeroNormKey { index });
        }
        let inv_norm = norm_sq.sqrt().recip();
        normalized.push(key.iter().map(|value| value * inv_norm).collect::<Vec<_>>());
    }

    let mut retained_set = BTreeSet::new();
    for &index in retained {
        if index >= keys.len() {
            return Err(TwinKvError::RetainedIndexOutOfRange {
                index,
                key_count: keys.len(),
            });
        }
        if !retained_set.insert(index) {
            return Err(TwinKvError::DuplicateRetainedIndex { index });
        }
    }

    let best_surviving = normalized
        .iter()
        .enumerate()
        .map(|(index, key)| {
            retained_set
                .iter()
                .copied()
                .filter(|&candidate| {
                    candidate != index && index.abs_diff(candidate) > config.local_window
                })
                .map(|candidate| cosine_from_normalized(key, &normalized[candidate]))
                .fold(-1.0_f32, f32::max)
        })
        .collect::<Vec<_>>();

    let mut orphans = (0..keys.len())
        .filter(|index| !retained_set.contains(index))
        .filter(|&index| best_surviving[index] < config.similarity_threshold)
        .map(|index| (index, best_surviving[index]))
        .collect::<Vec<_>>();

    let mut donors = retained_set
        .iter()
        .copied()
        .filter(|&index| !is_protected(index, keys.len(), config))
        .filter(|&index| best_surviving[index] >= config.similarity_threshold)
        .map(|index| (index, best_surviving[index]))
        .collect::<Vec<_>>();

    orphans.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    donors.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let available_swaps = orphans.len().min(donors.len());
    let swap_count = config
        .max_swaps
        .map(|limit| available_swaps.min(limit))
        .unwrap_or(available_swaps);

    let mut repaired = retained_set;
    let mut swaps = Vec::with_capacity(swap_count);
    for ((orphan, orphan_similarity), (donor, donor_similarity)) in orphans
        .iter()
        .take(swap_count)
        .zip(donors.iter().take(swap_count))
    {
        repaired.remove(donor);
        repaired.insert(*orphan);
        swaps.push(TwinKvSwap {
            admitted_orphan: *orphan,
            evicted_donor: *donor,
            orphan_best_surviving_similarity: *orphan_similarity,
            donor_best_surviving_similarity: *donor_similarity,
        });
    }

    debug_assert_eq!(repaired.len(), retained.len());

    Ok(TwinKvRepair {
        retained: repaired.into_iter().collect(),
        swaps,
        orphan_count: orphans.len(),
        donor_count: donors.len(),
    })
}

fn cosine_from_normalized(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>()
}

fn is_protected(index: usize, key_count: usize, config: &TwinKvConfig) -> bool {
    index < config.protected_sink_tokens
        || index >= key_count.saturating_sub(config.protected_recent_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> TwinKvConfig {
        TwinKvConfig {
            similarity_threshold: 0.95,
            local_window: 0,
            protected_sink_tokens: 1,
            protected_recent_tokens: 0,
            max_swaps: None,
        }
    }

    #[test]
    fn swaps_an_orphan_for_a_redundant_donor_without_changing_budget() {
        let keys = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = repair_retained_set(&keys, &[0, 1], &test_config()).unwrap();

        assert_eq!(result.retained, vec![0, 2]);
        assert_eq!(result.swaps.len(), 1);
        assert_eq!(result.swaps[0].evicted_donor, 1);
        assert_eq!(result.swaps[0].admitted_orphan, 2);
        assert_eq!(result.retained.len(), 2);
    }

    #[test]
    fn returns_noop_when_evicted_information_has_a_surviving_twin() {
        let keys = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let result = repair_retained_set(&keys, &[0, 1], &test_config()).unwrap();

        assert_eq!(result.retained, vec![0, 1]);
        assert!(result.swaps.is_empty());
        assert_eq!(result.orphan_count, 0);
    }

    #[test]
    fn excludes_adjacent_similarity_from_redundancy() {
        let keys = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        ];
        let mut config = test_config();
        config.local_window = 1;
        config.protected_sink_tokens = 0;

        let result = repair_retained_set(&keys, &[0, 2], &config).unwrap();
        assert!(result.swaps.is_empty());
    }

    #[test]
    fn never_uses_protected_regions_as_donors() {
        let keys = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        ];
        let config = TwinKvConfig {
            similarity_threshold: 0.95,
            local_window: 0,
            protected_sink_tokens: 2,
            protected_recent_tokens: 0,
            max_swaps: None,
        };

        let result = repair_retained_set(&keys, &[0, 1], &config).unwrap();
        assert_eq!(result.retained, vec![0, 1]);
        assert_eq!(result.donor_count, 0);
    }

    #[test]
    fn obeys_swap_limit_and_is_deterministic() {
        let keys = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, -1.0, 0.0],
        ];
        let config = TwinKvConfig {
            max_swaps: Some(1),
            protected_sink_tokens: 0,
            ..test_config()
        };

        let first = repair_retained_set(&keys, &[0, 1, 2, 3], &config).unwrap();
        let second = repair_retained_set(&keys, &[0, 1, 2, 3], &config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.swaps.len(), 1);
        assert_eq!(first.retained.len(), 4);
    }

    #[test]
    fn rejects_malformed_inputs() {
        assert_eq!(
            repair_retained_set(&[], &[0], &test_config()),
            Err(TwinKvError::EmptyKeys)
        );
        assert_eq!(
            repair_retained_set(&[vec![1.0]], &[], &test_config()),
            Err(TwinKvError::EmptyRetainedSet)
        );
        assert_eq!(
            repair_retained_set(&[vec![0.0, 0.0]], &[0], &test_config()),
            Err(TwinKvError::ZeroNormKey { index: 0 })
        );
        assert_eq!(
            repair_retained_set(&[vec![1.0], vec![1.0, 2.0]], &[0], &test_config()),
            Err(TwinKvError::InconsistentKeyDimension {
                index: 1,
                expected: 1,
                actual: 2,
            })
        );
        assert_eq!(
            repair_retained_set(&[vec![1.0]], &[1], &test_config()),
            Err(TwinKvError::RetainedIndexOutOfRange {
                index: 1,
                key_count: 1,
            })
        );
    }
}
