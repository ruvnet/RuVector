//! GraphMaxSim: greedy kNN graph on document centroids + beam search + MaxSim rerank.
//!
//! A complementary approximate variant to [`crate::HnswMaxSim`]. Where
//! `HnswMaxSim` builds a navigable graph over **individual token vectors**,
//! `GraphMaxSim` builds a graph over **per-document centroids** (the mean of a
//! document's token vectors). Beam search over the centroid graph selects a
//! candidate document set, which is then rescored with the exact MaxSim kernel.
//!
//! This trades a little recall for a smaller graph (one node per document
//! instead of one per token) and is well suited to corpora with many tokens
//! per document.
//!
//! ## Complexity
//! - Build: `O(N²·D)` greedy kNN construction (offline; suitable for `N ≤ 50K`).
//! - Search: `O(ef·M·D)` beam search + `O(C·Tq·Td·D)` MaxSim rerank.
//!
//! ## Correctness note — beam-search seeding
//! Entry points for the beam are the **first `n_seeds` consecutive** centroids,
//! *not* a strided/step-based sample. Step-based seeding (e.g. `step_by(N/K)`)
//! is catastrophically wrong when the step is a multiple of the underlying
//! cluster count: every seed lands in the same cluster and recall collapses to
//! a few percent. Consecutive seeding of the first `K ≥ N_CLUSTERS` documents
//! guarantees cluster coverage for the common interleaved ordering
//! (`doc i → cluster i % N_CLUSTERS`). See ADR-252.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{
    error::MaxSimError,
    score::{cosine, maxsim},
    types::{Embedding, MultiVecDoc, MultiVecQuery, SearchResult},
    MultiVecIndex,
};

/// Number of consecutive centroids used to seed the beam search.
const N_SEEDS: usize = 40;

// ── Ordered float wrapper for the frontier heap ───────────────────────────────

#[derive(Clone, PartialEq)]
struct OrdF32(f32, usize);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}

/// Mean of a slice of token vectors — the document centroid.
fn centroid(tokens: &[Embedding], dims: usize) -> Embedding {
    if tokens.is_empty() {
        return vec![0.0; dims];
    }
    let n = tokens.len() as f32;
    let mut c = vec![0.0f32; dims];
    for t in tokens {
        for (ci, ti) in c.iter_mut().zip(t.iter()) {
            *ci += ti / n;
        }
    }
    c
}

/// kNN centroid graph + beam search + MaxSim reranking.
///
/// Call [`GraphMaxSim::build`] once after all [`MultiVecIndex::add`] calls and
/// before searching; [`MultiVecIndex::search`] returns an empty result set if
/// the graph has not been built.
pub struct GraphMaxSim {
    docs: Vec<MultiVecDoc>,
    centroids: Vec<Embedding>,
    /// `graph[i]` = neighbour indices, sorted by descending centroid similarity.
    graph: Vec<Vec<usize>>,
    dims: usize,
    /// Neighbours per node in the kNN centroid graph.
    pub m: usize,
    /// Beam width during search (frontier exploration budget multiplier).
    pub ef: usize,
    /// Maximum candidate documents forwarded to the MaxSim reranker.
    pub n_candidates: usize,
    built: bool,
}

impl GraphMaxSim {
    /// Create an empty index.
    ///
    /// * `dims` — embedding dimensionality (validated on `add`).
    /// * `m` — neighbours per centroid node (try `8–16`).
    /// * `ef` — beam exploration budget (try `32–64`).
    /// * `n_candidates` — documents passed to exact MaxSim rerank (try `30–200`).
    pub fn new(dims: usize, m: usize, ef: usize, n_candidates: usize) -> Self {
        Self {
            docs: Vec::new(),
            centroids: Vec::new(),
            graph: Vec::new(),
            dims,
            m,
            ef,
            n_candidates,
            built: false,
        }
    }

    /// Build the kNN graph over all added centroids. `O(N²·D)`.
    ///
    /// Must be called after the final `add` and before `search`.
    pub fn build(&mut self) {
        let n = self.centroids.len();
        let m = self.m;
        self.graph = vec![Vec::new(); n];

        for i in 0..n {
            let mut nbrs: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine(&self.centroids[i], &self.centroids[j])))
                .collect();
            nbrs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            self.graph[i] = nbrs.iter().take(m).map(|(j, _)| *j).collect();
        }
        self.built = true;
    }

    /// Whether [`GraphMaxSim::build`] has been called since the last `add`.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Approximate memory footprint of the centroid graph in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.centroids.iter().map(|c| c.len() * 4).sum::<usize>()
            + self.graph.iter().map(|g| g.len() * 8).sum::<usize>()
    }

    /// Greedy beam search from `N_SEEDS` consecutive entry points. Returns up to
    /// `n_candidates` document indices ordered by centroid cosine similarity.
    fn beam_search(&self, query_centroid: &[f32]) -> Vec<usize> {
        let n = self.centroids.len();
        if n == 0 {
            return Vec::new();
        }

        let mut visited = vec![false; n];
        let mut frontier: BinaryHeap<OrdF32> = BinaryHeap::new();
        let mut found: Vec<(f32, usize)> = Vec::new();

        // Consecutive seeding — see module-level correctness note.
        let n_seeds = N_SEEDS.min(n);
        for (e, c) in self.centroids.iter().enumerate().take(n_seeds) {
            visited[e] = true;
            frontier.push(OrdF32(cosine(query_centroid, c), e));
        }

        let budget = self.ef * 4;
        while let Some(OrdF32(score, curr)) = frontier.pop() {
            found.push((score, curr));
            if found.len() >= budget {
                break;
            }
            for &nbr in &self.graph[curr] {
                if !visited[nbr] {
                    visited[nbr] = true;
                    frontier.push(OrdF32(cosine(query_centroid, &self.centroids[nbr]), nbr));
                }
            }
        }

        found.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        found
            .iter()
            .take(self.n_candidates)
            .map(|(_, idx)| *idx)
            .collect()
    }
}

impl MultiVecIndex for GraphMaxSim {
    fn add(&mut self, doc: MultiVecDoc) -> Result<(), MaxSimError> {
        for vec in &doc.vecs {
            if vec.len() != self.dims {
                return Err(MaxSimError::DimensionMismatch {
                    expected: self.dims,
                    got: vec.len(),
                });
            }
        }
        self.centroids.push(centroid(&doc.vecs, self.dims));
        self.docs.push(doc);
        self.built = false;
        Ok(())
    }

    fn search(&self, query: &MultiVecQuery, k: usize) -> Result<Vec<SearchResult>, MaxSimError> {
        if !self.built || self.docs.is_empty() {
            return Ok(Vec::new());
        }
        for vec in &query.vecs {
            if vec.len() != self.dims {
                return Err(MaxSimError::DimensionMismatch {
                    expected: self.dims,
                    got: vec.len(),
                });
            }
        }

        let qc = centroid(&query.vecs, self.dims);
        let candidates = self.beam_search(&qc);

        let mut results: Vec<SearchResult> = candidates
            .iter()
            .map(|&idx| SearchResult {
                doc_id: self.docs[idx].id,
                score: maxsim(&query.vecs, &self.docs[idx].vecs),
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn dims(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatMaxSim;
    use crate::types::DocId;
    use std::collections::HashSet;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / n).collect()
    }

    #[test]
    fn graph_builds_without_panic() {
        let mut idx = GraphMaxSim::new(8, 4, 16, 20);
        for i in 0..30u64 {
            let vecs: Vec<Embedding> = (0..3)
                .map(|k| vec![i as f32 * 0.1 + k as f32 * 0.01; 8])
                .collect();
            idx.add(MultiVecDoc { id: DocId(i), vecs }).unwrap();
        }
        idx.build();
        assert!(idx.is_built());
    }

    #[test]
    fn graph_search_empty_before_build() {
        let mut idx = GraphMaxSim::new(4, 4, 16, 20);
        for i in 0..10u64 {
            idx.add(MultiVecDoc {
                id: DocId(i),
                vecs: vec![vec![i as f32; 4]],
            })
            .unwrap();
        }
        let q = MultiVecQuery {
            vecs: vec![vec![1.0; 4]],
        };
        // build() not called → empty
        assert!(idx.search(&q, 5).unwrap().is_empty());
    }

    #[test]
    fn graph_empty_index() {
        let idx = GraphMaxSim::new(4, 4, 16, 10);
        let q = MultiVecQuery {
            vecs: vec![vec![1.0; 4]],
        };
        assert!(idx.search(&q, 5).unwrap().is_empty());
    }

    #[test]
    fn graph_dim_mismatch_rejected() {
        let mut idx = GraphMaxSim::new(4, 4, 16, 10);
        let err = idx.add(MultiVecDoc {
            id: DocId(1),
            vecs: vec![vec![1.0; 3]],
        });
        assert!(matches!(err, Err(MaxSimError::DimensionMismatch { .. })));
    }

    #[test]
    fn graph_recall_acceptable_vs_flat() {
        let dim = 16;
        let n = 100u64;
        let k_tokens = 4;
        let n_clusters = 5;

        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|c| {
                unit(
                    (0..dim)
                        .map(|j| if j == c % dim { 1.0f32 } else { 0.0 })
                        .collect(),
                )
            })
            .collect();

        let mut flat = FlatMaxSim::new(dim);
        let mut graph = GraphMaxSim::new(dim, 8, 32, 30);

        for i in 0..n {
            let cluster = (i as usize) % n_clusters;
            let vecs: Vec<Embedding> = (0..k_tokens)
                .map(|t| {
                    (0..dim)
                        .map(|j| {
                            centers[cluster][j] + (i as usize * dim + t * dim + j) as f32 * 0.001
                        })
                        .collect()
                })
                .collect();
            let doc = MultiVecDoc { id: DocId(i), vecs };
            flat.add(doc.clone()).unwrap();
            graph.add(doc).unwrap();
        }
        graph.build();

        let query = MultiVecQuery {
            vecs: vec![centers[0].clone()],
        };
        let flat_ids: HashSet<DocId> = flat
            .search(&query, 10)
            .unwrap()
            .iter()
            .map(|r| r.doc_id)
            .collect();
        let graph_ids: HashSet<DocId> = graph
            .search(&query, 10)
            .unwrap()
            .iter()
            .map(|r| r.doc_id)
            .collect();

        let overlap = flat_ids.intersection(&graph_ids).count();
        let recall = overlap as f32 / flat_ids.len() as f32;
        assert!(
            recall >= 0.5,
            "GraphMaxSim recall@10 ≥ 50% required, got {:.1}%",
            recall * 100.0
        );
    }
}
