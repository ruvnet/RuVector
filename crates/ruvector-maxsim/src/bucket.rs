//! BucketMaxSim: centroid-filtered approximate MaxSim.
//!
//! Each multi-vector document is summarised by the **centroid** of its token
//! embeddings. At query time we compute the query centroid, find the top-M
//! candidate documents by centroid cosine similarity (a cheap linear scan),
//! then run the exact MaxSim kernel only on those M candidates.
//!
//! This trades recall for speed: missing a candidate document's centroid means
//! its MaxSim score is never evaluated. The `oversampling` factor M > k lets
//! you tune the recall-speed tradeoff.

use std::collections::BinaryHeap;

use crate::{
    error::MaxSimError,
    score::{cosine, maxsim},
    types::{DocId, Embedding, MultiVecDoc, MultiVecQuery, SearchResult},
    MultiVecIndex,
};

/// One stored document: centroid + all token vectors.
struct DocEntry {
    id: DocId,
    centroid: Embedding,
    vecs: Vec<Embedding>,
}

/// Approximate MaxSim via centroid pre-filtering.
pub struct BucketMaxSim {
    entries: Vec<DocEntry>,
    dims: usize,
    /// How many centroid-nearest candidates to rerank with full MaxSim.
    oversampling: usize,
}

impl BucketMaxSim {
    /// Build with a given dimension and oversampling factor.
    ///
    /// A good default for `oversampling` is `k * 4` to `k * 10`.
    pub fn new(dims: usize, oversampling: usize) -> Self {
        Self {
            entries: Vec::new(),
            dims,
            oversampling,
        }
    }

    fn centroid(vecs: &[Embedding]) -> Embedding {
        if vecs.is_empty() {
            return Vec::new();
        }
        let d = vecs[0].len();
        let mut c = vec![0.0_f32; d];
        for v in vecs {
            for (ci, vi) in c.iter_mut().zip(v.iter()) {
                *ci += vi;
            }
        }
        let n = vecs.len() as f32;
        c.iter_mut().for_each(|x| *x /= n);
        c
    }

    fn query_centroid(query: &MultiVecQuery) -> Embedding {
        Self::centroid(&query.vecs)
    }

    /// Approximate memory footprint.
    pub fn memory_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|e| e.centroid.len() * 4 + e.vecs.iter().map(|v| v.len() * 4).sum::<usize>())
            .sum()
    }
}

impl MultiVecIndex for BucketMaxSim {
    fn add(&mut self, doc: MultiVecDoc) -> Result<(), MaxSimError> {
        if let Some(v) = doc.vecs.first() {
            if v.len() != self.dims {
                return Err(MaxSimError::DimensionMismatch {
                    expected: self.dims,
                    got: v.len(),
                });
            }
        }
        let centroid = Self::centroid(&doc.vecs);
        self.entries.push(DocEntry {
            id: doc.id,
            centroid,
            vecs: doc.vecs,
        });
        Ok(())
    }

    fn search(&self, query: &MultiVecQuery, k: usize) -> Result<Vec<SearchResult>, MaxSimError> {
        if self.entries.is_empty() {
            return Ok(Vec::new());
        }
        let qc = Self::query_centroid(query);
        let m = self.oversampling.max(k);

        // Phase 1: centroid-level candidate selection (linear scan over centroids).
        let mut centroid_heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(m + 1);
        for entry in &self.entries {
            let score = cosine(&qc, &entry.centroid);
            centroid_heap.push(SearchResult {
                doc_id: entry.id,
                score,
            });
            if centroid_heap.len() > m {
                centroid_heap.pop();
            }
        }
        let mut candidates: Vec<DocId> = centroid_heap.into_iter().map(|r| r.doc_id).collect();
        candidates.sort_unstable();

        // Phase 2: exact MaxSim on candidates.
        let mut final_heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k + 1);
        for entry in &self.entries {
            if candidates.binary_search(&entry.id).is_ok() {
                let score = maxsim(&query.vecs, &entry.vecs);
                final_heap.push(SearchResult {
                    doc_id: entry.id,
                    score,
                });
                if final_heap.len() > k {
                    final_heap.pop();
                }
            }
        }
        Ok(final_heap.into_sorted_vec())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn dims(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MultiVecQuery;

    #[test]
    fn bucket_finds_best_doc() {
        let mut idx = BucketMaxSim::new(2, 4);
        idx.add(MultiVecDoc {
            id: DocId(1),
            vecs: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        })
        .unwrap();
        idx.add(MultiVecDoc {
            id: DocId(2),
            vecs: vec![vec![0.0, 1.0]],
        })
        .unwrap();
        // Query: topic A (x-axis)
        let q = MultiVecQuery {
            vecs: vec![vec![1.0, 0.0]],
        };
        let res = idx.search(&q, 2).unwrap();
        assert_eq!(res[0].doc_id, DocId(1), "doc 1 covers topic A");
    }

    #[test]
    fn bucket_multi_token_advantage() {
        // Doc 1 has two topic vectors; doc 2 has only one.
        let mut idx = BucketMaxSim::new(2, 10);
        idx.add(MultiVecDoc {
            id: DocId(1),
            vecs: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        })
        .unwrap();
        idx.add(MultiVecDoc {
            id: DocId(2),
            vecs: vec![vec![0.5_f32.sqrt(), 0.5_f32.sqrt()]],
        })
        .unwrap();
        // Query covers both topics → doc 1 should win (score = 2.0 vs ~1.41)
        let q = MultiVecQuery {
            vecs: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let res = idx.search(&q, 2).unwrap();
        assert_eq!(res[0].doc_id, DocId(1));
    }
}
