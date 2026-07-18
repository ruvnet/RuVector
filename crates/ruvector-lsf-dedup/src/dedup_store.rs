//! Agent memory store with near-duplicate suppression.
//!
//! Three concrete stores are provided, each using a different fingerprinting
//! strategy.  All share the same proof-trail logging so that dedup decisions
//! are auditable.
//!
//! ## Hybrid approach
//! `HybridDedupStore` uses SimHash as a cheap pre-filter, then falls back to
//! exact cosine similarity for any candidate that clears `pre_threshold`.
//! This combination achieves the highest precision while retaining good recall.

use crate::{
    cosine_similarity, DedupDecision, DedupMethod, DedupResult, MemoryEntry, MemoryId,
    SemanticFingerprinter,
};

/// Generic dedup store parameterised over a fingerprinting strategy.
pub struct DedupStore<H: SemanticFingerprinter> {
    hasher: H,
    threshold: f32,
    entries: Vec<MemoryEntry>,
    fingerprints: Vec<H::Fingerprint>,
    decisions: Vec<DedupDecision>,
    insert_count: u64,
    method: DedupMethod,
}

impl<H: SemanticFingerprinter> DedupStore<H> {
    pub fn new(hasher: H, threshold: f32, method: DedupMethod) -> Self {
        Self {
            hasher,
            threshold,
            entries: Vec::new(),
            fingerprints: Vec::new(),
            decisions: Vec::new(),
            insert_count: 0,
            method,
        }
    }

    /// Attempt to insert `vector` into the store.
    ///
    /// If a near-duplicate is found (estimated similarity ≥ threshold) the
    /// vector is rejected and `DedupResult::Duplicate` is returned.
    /// Otherwise it is stored and `DedupResult::Unique` is returned.
    pub fn insert(&mut self, vector: Vec<f32>, metadata: Option<String>) -> DedupResult {
        let seq = self.insert_count;
        self.insert_count += 1;

        let fp = self.hasher.fingerprint(&vector);

        let duplicate = self
            .fingerprints
            .iter()
            .enumerate()
            .find_map(|(i, stored)| {
                let sim = self.hasher.estimate_similarity(&fp, stored);
                if sim >= self.threshold {
                    Some((self.entries[i].id, sim))
                } else {
                    None
                }
            });

        let result = match duplicate {
            Some((existing_id, similarity)) => DedupResult::Duplicate {
                of: existing_id,
                similarity,
            },
            None => {
                let id = self.entries.len() as MemoryId;
                self.fingerprints.push(fp);
                self.entries.push(MemoryEntry {
                    id,
                    vector,
                    metadata,
                });
                DedupResult::Unique(id)
            }
        };

        self.decisions.push(DedupDecision {
            insert_seq: seq,
            result: result.clone(),
            method: self.method.clone(),
        });
        result
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn decisions(&self) -> &[DedupDecision] {
        &self.decisions
    }
}

/// Hybrid store: SimHash pre-filter + exact cosine verify.
///
/// Candidates that pass `pre_threshold` under SimHash get their exact cosine
/// similarity computed against the stored vectors. Only if the exact cosine
/// exceeds `exact_threshold` is the insertion rejected as a duplicate.
pub struct HybridDedupStore<H: SemanticFingerprinter> {
    hasher: H,
    pre_threshold: f32,
    exact_threshold: f32,
    entries: Vec<MemoryEntry>,
    fingerprints: Vec<H::Fingerprint>,
    decisions: Vec<DedupDecision>,
    insert_count: u64,
}

impl<H: SemanticFingerprinter> HybridDedupStore<H> {
    /// `pre_threshold`  — SimHash similarity above which to run exact check
    /// `exact_threshold` — cosine similarity above which to reject as duplicate
    pub fn new(hasher: H, pre_threshold: f32, exact_threshold: f32) -> Self {
        Self {
            hasher,
            pre_threshold,
            exact_threshold,
            entries: Vec::new(),
            fingerprints: Vec::new(),
            decisions: Vec::new(),
            insert_count: 0,
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>, metadata: Option<String>) -> DedupResult {
        let seq = self.insert_count;
        self.insert_count += 1;

        let fp = self.hasher.fingerprint(&vector);

        // Phase 1: SimHash pre-filter
        let candidates: Vec<usize> = self
            .fingerprints
            .iter()
            .enumerate()
            .filter_map(|(i, stored)| {
                let est = self.hasher.estimate_similarity(&fp, stored);
                if est >= self.pre_threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        // Phase 2: Exact cosine check on candidates
        let duplicate = candidates.into_iter().find_map(|i| {
            let exact = cosine_similarity(&vector, &self.entries[i].vector);
            if exact >= self.exact_threshold {
                Some((self.entries[i].id, exact))
            } else {
                None
            }
        });

        let result = match duplicate {
            Some((existing_id, similarity)) => DedupResult::Duplicate {
                of: existing_id,
                similarity,
            },
            None => {
                let id = self.entries.len() as MemoryId;
                self.fingerprints.push(fp);
                self.entries.push(MemoryEntry {
                    id,
                    vector,
                    metadata,
                });
                DedupResult::Unique(id)
            }
        };

        self.decisions.push(DedupDecision {
            insert_seq: seq,
            result: result.clone(),
            method: DedupMethod::Hybrid,
        });
        result
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn decisions(&self) -> &[DedupDecision] {
        &self.decisions
    }
}

/// Evaluation statistics for a dedup run.
#[derive(Default, Debug)]
pub struct DedupStats {
    pub total_inserted: usize,
    pub unique_stored: usize,
    pub duplicates_rejected: usize,
    /// Among `Duplicate` decisions, fraction where exact cosine ≥ exact_threshold.
    pub precision: f32,
    /// Among truly-duplicate pairs in the input, fraction correctly rejected.
    pub recall: f32,
    /// Fraction of unique vectors wrongly rejected.
    pub false_positive_rate: f32,
}

impl DedupStats {
    pub fn compute(decisions: &[DedupDecision], true_labels: &[bool]) -> Self {
        // true_labels[i] = true  → insert[i] is a near-duplicate
        //                = false → insert[i] is unique
        let total = decisions.len().min(true_labels.len());

        let mut tp = 0usize; // true positive: correctly identified duplicate
        let mut fp = 0usize; // false positive: unique wrongly flagged
        let mut fn_ = 0usize; // false negative: duplicate missed
        let mut tn = 0usize; // true negative: unique correctly stored

        for i in 0..total {
            let predicted_dup = matches!(decisions[i].result, DedupResult::Duplicate { .. });
            let actual_dup = true_labels[i];
            match (predicted_dup, actual_dup) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            1.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f32 / (tp + fn_) as f32
        } else {
            1.0
        };
        let fpr = if fp + tn > 0 {
            fp as f32 / (fp + tn) as f32
        } else {
            0.0
        };

        DedupStats {
            total_inserted: total,
            unique_stored: tn + fn_,
            duplicates_rejected: tp + fp,
            precision,
            recall,
            false_positive_rate: fpr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{minhash::MinHasher, normalize, simhash::SimHasher};

    fn make_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state as i64 as f32) / i64::MAX as f32
            })
            .collect();
        normalize(&mut v);
        v
    }

    fn near_dup(v: &[f32], noise: f32, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut out: Vec<f32> = v
            .iter()
            .map(|&x| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(3);
                let n = (state as i64 as f32) / i64::MAX as f32;
                x + noise * n
            })
            .collect();
        normalize(&mut out);
        out
    }

    #[test]
    fn simhash_store_rejects_near_duplicate() {
        let dim = 128;
        let hasher = SimHasher::new(dim, 64, 1);
        let mut store = DedupStore::new(hasher, 0.85, DedupMethod::SimHash);

        let base = make_vec(dim, 100);
        assert!(matches!(
            store.insert(base.clone(), None),
            DedupResult::Unique(_)
        ));

        let dup = near_dup(&base, 0.01, 200);
        let result = store.insert(dup, None);
        assert!(
            matches!(result, DedupResult::Duplicate { .. }),
            "expected duplicate, got {:?}",
            result
        );
    }

    #[test]
    fn minhash_store_rejects_near_duplicate() {
        let dim = 128;
        let hasher = MinHasher::new(dim, 128, 8, 2);
        let mut store = DedupStore::new(hasher, 0.70, DedupMethod::MinHash);

        let base = make_vec(dim, 101);
        assert!(matches!(
            store.insert(base.clone(), None),
            DedupResult::Unique(_)
        ));

        let dup = near_dup(&base, 0.01, 201);
        let result = store.insert(dup, None);
        assert!(
            matches!(result, DedupResult::Duplicate { .. }),
            "expected duplicate, got {:?}",
            result
        );
    }

    #[test]
    fn hybrid_store_accepts_truly_distinct() {
        let dim = 128;
        let hasher = SimHasher::new(dim, 64, 3);
        let mut store = HybridDedupStore::new(hasher, 0.70, 0.92);

        let v1 = make_vec(dim, 102);
        let v2 = make_vec(dim, 99_999);
        store.insert(v1, None);
        let result = store.insert(v2, None);
        assert!(
            matches!(result, DedupResult::Unique(_)),
            "distinct vectors should be unique, got {:?}",
            result
        );
    }

    #[test]
    fn dedup_stats_precision_recall_identity() {
        use crate::{DedupMethod, DedupResult};
        // Simulate 4 decisions: TP, FP, FN, TN
        let decisions = vec![
            DedupDecision {
                insert_seq: 0,
                result: DedupResult::Duplicate {
                    of: 0,
                    similarity: 0.95,
                },
                method: DedupMethod::SimHash,
            },
            DedupDecision {
                insert_seq: 1,
                result: DedupResult::Duplicate {
                    of: 1,
                    similarity: 0.87,
                },
                method: DedupMethod::SimHash,
            },
            DedupDecision {
                insert_seq: 2,
                result: DedupResult::Unique(2),
                method: DedupMethod::SimHash,
            },
            DedupDecision {
                insert_seq: 3,
                result: DedupResult::Unique(3),
                method: DedupMethod::SimHash,
            },
        ];
        let labels = vec![true, false, true, false]; // TP, FP, FN, TN
        let stats = DedupStats::compute(&decisions, &labels);
        assert!((stats.precision - 0.5).abs() < 1e-5);
        assert!((stats.recall - 0.5).abs() < 1e-5);
        assert!((stats.false_positive_rate - 0.5).abs() < 1e-5);
    }
}
