//! HnswMaxSim: PLAID-inspired approximate MaxSim via inverted token index.
//!
//! Indexes **all** token vectors from **all** documents in a single HNSW-like
//! structure (here: a greedy small-world graph). At query time each query
//! token retrieves the top-M nearest stored token vectors. Retrieved token
//! vectors are grouped by their parent document; the exact MaxSim kernel then
//! scores each candidate document. Only documents with at least one retrieved
//! token are scored — this gives a huge speedup at the cost of occasionally
//! missing documents whose *best* token was not retrieved.
//!
//! Implementation note: rather than pulling in the hnsw_rs workspace dep
//! (which would force a feature flag + large compile surface), we implement a
//! lean greedy insertion graph with:
//!   * fixed M=16 (connections per node at insertion layer 0)
//!   * single-layer flat graph (equivalent to NSW without hierarchy)
//!   * greedy beam search with ef=32 candidates
//!
//! This keeps the crate self-contained and under 500 lines per file.

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::{
    error::MaxSimError,
    score::{cosine, maxsim},
    types::{DocId, Embedding, MultiVecDoc, MultiVecQuery, SearchResult},
    MultiVecIndex,
};

/// Maximum out-edges per node in the NSW graph.
const M: usize = 16;
/// Candidate set size during beam search.
const EF: usize = 32;

/// One indexed token vector.
struct TokenEntry {
    doc_id: DocId,
    vec: Embedding,
    /// Neighbour node indices in the flat NSW graph.
    neighbours: Vec<usize>,
}

/// (score, node_idx) pair for the beam heap.
#[derive(PartialEq)]
struct Candidate {
    score: f32,
    idx: usize,
}

impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Approximate MaxSim via a flat NSW token index.
pub struct HnswMaxSim {
    tokens: Vec<TokenEntry>,
    dims: usize,
    /// How many token neighbours to retrieve per query token before grouping.
    token_candidates: usize,
}

impl HnswMaxSim {
    /// Create with given dimensionality and per-query-token candidate budget.
    ///
    /// `token_candidates` controls the tradeoff: higher → better recall,
    /// more MaxSim evaluations. A good starting point is `32–64`.
    pub fn new(dims: usize, token_candidates: usize) -> Self {
        Self {
            tokens: Vec::new(),
            dims,
            token_candidates,
        }
    }

    /// NSW beam search returning the `ef` nearest token indices to `query`.
    fn search_tokens(&self, query: &[f32], ef: usize) -> Vec<usize> {
        if self.tokens.is_empty() {
            return Vec::new();
        }
        // Entry point: first inserted token.
        let entry = 0_usize;
        let entry_score = cosine(query, &self.tokens[entry].vec);

        // We use a max-heap for "visited" candidates and a min-heap for the
        // result set. Standard NSW search.
        let mut candidates: BinaryHeap<Candidate> = std::iter::once(Candidate {
            score: entry_score,
            idx: entry,
        })
        .collect();
        let mut visited: HashSet<usize> = HashSet::from([entry]);
        let mut result: BinaryHeap<Candidate> = std::iter::once(Candidate {
            score: entry_score,
            idx: entry,
        })
        .collect();

        while let Some(Candidate {
            score: c_score,
            idx: c_idx,
        }) = candidates.pop()
        {
            // Lower bound from result: the worst element in result.
            let worst_in_result = result.iter().map(|c| c.score).fold(f32::INFINITY, f32::min);
            if c_score < worst_in_result && result.len() >= ef {
                break;
            }
            for &nb in &self.tokens[c_idx].neighbours {
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb);
                let nb_score = cosine(query, &self.tokens[nb].vec);
                let worst = result.iter().map(|c| c.score).fold(f32::INFINITY, f32::min);
                if nb_score > worst || result.len() < ef {
                    candidates.push(Candidate {
                        score: nb_score,
                        idx: nb,
                    });
                    result.push(Candidate {
                        score: nb_score,
                        idx: nb,
                    });
                    if result.len() > ef {
                        // Remove worst from result (min-heap trick via re-collection)
                        let mut v: Vec<Candidate> = result.into_iter().collect();
                        v.sort_by(|a, b| {
                            b.score
                                .partial_cmp(&a.score)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        v.truncate(ef);
                        result = v.into_iter().collect();
                    }
                }
            }
        }
        result.into_iter().map(|c| c.idx).collect()
    }

    /// Greedy neighbour selection for a newly inserted node.
    fn select_neighbours(&self, query: &[f32], exclude: usize) -> Vec<usize> {
        if self.tokens.len() <= 1 {
            return Vec::new();
        }
        // Score all existing tokens (excluding self) and pick top-M.
        let mut scored: Vec<(f32, usize)> = self
            .tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != exclude)
            .map(|(i, t)| (cosine(query, &t.vec), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(M);
        scored.into_iter().map(|(_, i)| i).collect()
    }

    /// Approximate memory footprint.
    pub fn memory_bytes(&self) -> usize {
        self.tokens
            .iter()
            .map(|t| t.vec.len() * 4 + t.neighbours.len() * 8)
            .sum()
    }
}

impl MultiVecIndex for HnswMaxSim {
    fn add(&mut self, doc: MultiVecDoc) -> Result<(), MaxSimError> {
        for vec in &doc.vecs {
            if vec.len() != self.dims {
                return Err(MaxSimError::DimensionMismatch {
                    expected: self.dims,
                    got: vec.len(),
                });
            }
        }
        for vec in doc.vecs {
            let new_idx = self.tokens.len();
            // Select neighbours before push (needs self.tokens without new entry).
            let neighbours = self.select_neighbours(&vec, new_idx);
            // Wire back-edges into existing nodes.
            // Note: new_idx is not yet in self.tokens, so we pass `vec` explicitly
            // when pruning to handle the case where new_idx is itself a neighbour.
            for &nb in &neighbours {
                self.tokens[nb].neighbours.push(new_idx);
                if self.tokens[nb].neighbours.len() > M * 2 {
                    let nb_vec: Vec<f32> = self.tokens[nb].vec.clone();
                    let neighbour_list = self.tokens[nb].neighbours.clone();
                    let mut scored: Vec<(f32, usize)> = neighbour_list
                        .iter()
                        .map(|&i| {
                            let score = if i < self.tokens.len() {
                                cosine(&nb_vec, &self.tokens[i].vec)
                            } else {
                                // i == new_idx (not yet inserted); use `vec` directly
                                cosine(&nb_vec, &vec)
                            };
                            (score, i)
                        })
                        .collect();
                    scored
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(M);
                    self.tokens[nb].neighbours = scored.into_iter().map(|(_, i)| i).collect();
                }
            }
            self.tokens.push(TokenEntry {
                doc_id: doc.id,
                vec,
                neighbours,
            });
        }
        Ok(())
    }

    fn search(&self, query: &MultiVecQuery, k: usize) -> Result<Vec<SearchResult>, MaxSimError> {
        if self.tokens.is_empty() {
            return Ok(Vec::new());
        }
        let tc = self.token_candidates.max(k * 2).max(EF);

        // Phase 1: for each query token, retrieve `tc` candidate token indices.
        let mut candidate_docs: HashMap<DocId, ()> = HashMap::new();
        for qvec in &query.vecs {
            for idx in self.search_tokens(qvec, tc) {
                candidate_docs.insert(self.tokens[idx].doc_id, ());
            }
        }

        // Phase 2: group all token vectors per candidate document.
        let mut doc_vecs: HashMap<DocId, Vec<&Embedding>> = HashMap::new();
        for token in &self.tokens {
            if candidate_docs.contains_key(&token.doc_id) {
                doc_vecs.entry(token.doc_id).or_default().push(&token.vec);
            }
        }

        // Phase 3: score each candidate document with the full MaxSim kernel.
        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k + 1);
        for (doc_id, dvecs) in &doc_vecs {
            let owned: Vec<Embedding> = dvecs.iter().map(|v| (*v).clone()).collect();
            let score = maxsim(&query.vecs, &owned);
            heap.push(SearchResult {
                doc_id: *doc_id,
                score,
            });
            if heap.len() > k {
                heap.pop();
            }
        }
        Ok(heap.into_sorted_vec())
    }

    fn len(&self) -> usize {
        // Unique doc count
        let ids: HashSet<DocId> = self.tokens.iter().map(|t| t.doc_id).collect();
        ids.len()
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
    fn hnsw_single_match() {
        let mut idx = HnswMaxSim::new(2, 8);
        idx.add(MultiVecDoc {
            id: DocId(1),
            vecs: vec![vec![1.0, 0.0]],
        })
        .unwrap();
        idx.add(MultiVecDoc {
            id: DocId(2),
            vecs: vec![vec![0.0, 1.0]],
        })
        .unwrap();
        let q = MultiVecQuery {
            vecs: vec![vec![1.0, 0.0]],
        };
        let res = idx.search(&q, 2).unwrap();
        assert_eq!(res[0].doc_id, DocId(1));
    }

    #[test]
    fn hnsw_multi_token_coverage() {
        let mut idx = HnswMaxSim::new(2, 16);
        // Doc 1: two topic vectors; doc 2: one topic vector
        for i in 1..=20u64 {
            idx.add(MultiVecDoc {
                id: DocId(i),
                vecs: vec![vec![(i as f32).cos(), (i as f32).sin()]],
            })
            .unwrap();
        }
        idx.add(MultiVecDoc {
            id: DocId(100),
            vecs: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        })
        .unwrap();
        let q = MultiVecQuery {
            vecs: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let res = idx.search(&q, 5).unwrap();
        assert!(!res.is_empty(), "should return results");
        assert!(res[0].score > 0.0);
    }
}
