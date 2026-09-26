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
/// The input `retained` set is the wrapped policy's decision. Two tokens are
/// *twins* when they are more than `local_window` positions apart and their
/// key cosine similarity is at least the threshold. An evicted token with no
/// retained twin is an *orphan*; a retained, unprotected token with at least
/// one retained twin is a *donor*. The most severe orphans (lowest best
/// surviving similarity) replace the most redundant donors one for one.
///
/// Swaps are applied greedily against the *current* retained set, not the
/// input snapshot. A per token count of retained twins is maintained
/// incrementally, and each swap is only committed when, after admitting the
/// orphan:
///
/// * the orphan still has no retained twin (an orphan already covered by an
///   earlier admission is skipped),
/// * the donor still has at least one retained twin (so mutual twins are
///   never both evicted), and
/// * no evicted token loses its last retained twin because of the eviction.
///
/// Consequently every evicted token that had a retained twin before a swap
/// still has one after it, the evicted donor stays represented by a retained
/// twin, and the admitted orphan's information is now retained directly.
/// The pass is a single greedy sweep: a donor rejected by these checks is not
/// reconsidered later, even if a subsequent admission would make it safe.
///
/// Complexity: normalization is O(n d); the initial audit is O(n K d) against
/// the retained set; each committed swap costs O(n d) to update twin counts,
/// and each donor candidate is examined at most once at O(n d). With at most
/// K swaps and K donor candidates the whole pass is O(n K d), where n is the
/// context length, K the retained budget, and d the key dimension. No full
/// pairwise matrix is materialized; extra memory is O(n d + n).
///
/// Swap receipts report best surviving similarities measured against the
/// input retained set; `orphan_count` and `donor_count` describe that initial
/// audit.
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

    let normalized = normalize_keys(keys)?;
    let key_count = keys.len();

    let mut retained_set = BTreeSet::new();
    for &index in retained {
        if index >= key_count {
            return Err(TwinKvError::RetainedIndexOutOfRange { index, key_count });
        }
        if !retained_set.insert(index) {
            return Err(TwinKvError::DuplicateRetainedIndex { index });
        }
    }

    let threshold = config.similarity_threshold;
    let is_twin = |left: usize, right: usize| {
        left.abs_diff(right) > config.local_window
            && cosine_from_normalized(&normalized[left], &normalized[right]) >= threshold
    };

    // Initial audit against the input retained set: O(n K d).
    let mut best_surviving = vec![-1.0_f32; key_count];
    let mut twin_count = vec![0_usize; key_count];
    for (index, key) in normalized.iter().enumerate() {
        for &candidate in &retained_set {
            if index.abs_diff(candidate) <= config.local_window {
                continue;
            }
            let similarity = cosine_from_normalized(key, &normalized[candidate]);
            best_surviving[index] = best_surviving[index].max(similarity);
            if similarity >= threshold {
                twin_count[index] += 1;
            }
        }
    }

    let mut orphans = (0..key_count)
        .filter(|index| !retained_set.contains(index) && twin_count[*index] == 0)
        .map(|index| (index, best_surviving[index]))
        .collect::<Vec<_>>();
    let mut donors = retained_set
        .iter()
        .copied()
        .filter(|&index| !is_protected(index, key_count, config) && twin_count[index] > 0)
        .map(|index| (index, best_surviving[index]))
        .collect::<Vec<_>>();

    orphans.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    donors.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let swap_limit = config.max_swaps.unwrap_or(usize::MAX);
    let mut repaired = retained_set;
    let mut swaps = Vec::new();
    let mut donor_cursor = 0;

    // Increment (admit) or decrement (evict) the twin count of every twin of `pivot`.
    let adjust_twins = |twin_count: &mut [usize], pivot: usize, admit: bool| {
        for (index, count) in twin_count.iter_mut().enumerate() {
            if is_twin(index, pivot) {
                if admit {
                    *count += 1;
                } else {
                    *count -= 1;
                }
            }
        }
    };

    for &(orphan, orphan_similarity) in &orphans {
        if swaps.len() >= swap_limit || donor_cursor >= donors.len() {
            break;
        }
        if twin_count[orphan] > 0 {
            // Covered by an orphan admitted earlier in this pass.
            continue;
        }

        // Tentatively admit the orphan so its coverage counts when judging donors.
        repaired.insert(orphan);
        adjust_twins(&mut twin_count, orphan, true);

        let mut chosen = None;
        while donor_cursor < donors.len() {
            let (donor, donor_similarity) = donors[donor_cursor];
            donor_cursor += 1;
            if twin_count[donor] == 0 {
                continue;
            }
            let strands_evicted = (0..key_count).any(|index| {
                !repaired.contains(&index) && twin_count[index] == 1 && is_twin(index, donor)
            });
            if !strands_evicted {
                chosen = Some((donor, donor_similarity));
                break;
            }
        }

        let Some((donor, donor_similarity)) = chosen else {
            // No safe donor remains; roll back the tentative admission.
            adjust_twins(&mut twin_count, orphan, false);
            repaired.remove(&orphan);
            break;
        };

        repaired.remove(&donor);
        adjust_twins(&mut twin_count, donor, false);
        swaps.push(TwinKvSwap {
            admitted_orphan: orphan,
            evicted_donor: donor,
            orphan_best_surviving_similarity: orphan_similarity,
            donor_best_surviving_similarity: donor_similarity,
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

/// Validate keys and scale each one to unit length.
///
/// The norm is computed after dividing by the largest absolute component, so
/// finite keys never overflow or underflow the squared norm. Only an all zero
/// key is rejected as `ZeroNormKey`.
fn normalize_keys(keys: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, TwinKvError> {
    let dimension = keys[0].len();
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

        let max_abs = key.iter().fold(0.0_f32, |acc, value| acc.max(value.abs()));
        if max_abs == 0.0 {
            return Err(TwinKvError::ZeroNormKey { index });
        }
        let scaled = key.iter().map(|value| value / max_abs).collect::<Vec<_>>();
        // Every scaled component is in [-1, 1] and at least one is +/-1, so
        // the squared norm lies in [1, d] and is always finite and nonzero.
        let inv_norm = scaled
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .recip();
        normalized.push(scaled.into_iter().map(|value| value * inv_norm).collect());
    }
    Ok(normalized)
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

    fn unit(dimension: usize, axis: usize) -> Vec<f32> {
        let mut key = vec![0.0; dimension];
        key[axis] = 1.0;
        key
    }

    #[test]
    fn never_evicts_both_mutual_twins() {
        // keys [x, y, y, z, w]: the two retained y copies are each other's only twin.
        let keys = [0, 1, 1, 2, 3].map(|axis| unit(4, axis)).to_vec();
        let result = repair_retained_set(&keys, &[0, 1, 2], &test_config()).unwrap();

        assert_eq!(result.retained, vec![0, 2, 3]);
        assert_eq!(result.swaps.len(), 1);
        assert_eq!(result.swaps[0].evicted_donor, 1);
    }

    #[test]
    fn skips_orphan_already_covered_by_an_earlier_admission() {
        // keys [x, y, y, u, u, v, v]: both v copies are orphans and twins of each other.
        let keys = [0, 1, 1, 2, 2, 3, 3].map(|axis| unit(4, axis)).to_vec();
        let result = repair_retained_set(&keys, &[0, 1, 2, 3, 4], &test_config()).unwrap();

        assert_eq!(result.orphan_count, 2);
        assert_eq!(result.retained, vec![0, 2, 3, 4, 5]);
        assert_eq!(result.swaps.len(), 1);
    }

    #[test]
    fn never_strands_an_evicted_token_covered_only_by_the_donor() {
        let angle = |degrees: f32| vec![degrees.to_radians().cos(), degrees.to_radians().sin()];
        // t = 0, d = 20, e = 40 degrees: t~d and d~e are twins at 0.9, t~e is not.
        let keys = vec![angle(0.0), angle(20.0), angle(40.0), vec![0.0, -1.0]];
        let config = TwinKvConfig {
            similarity_threshold: 0.9,
            ..test_config()
        };
        let result = repair_retained_set(&keys, &[0, 1], &config).unwrap();

        assert_eq!(result.orphan_count, 1);
        assert_eq!(result.donor_count, 1);
        assert_eq!(result.retained, vec![0, 1]);
        assert!(result.swaps.is_empty());
    }

    #[test]
    fn normalizes_extreme_but_finite_magnitudes() {
        let keys = vec![vec![3.0e38, 3.0e38], vec![1.0e-40, 1.0e-40]];
        let result = repair_retained_set(&keys, &[0], &test_config()).unwrap();
        assert_eq!(result.orphan_count, 0);
    }
}
