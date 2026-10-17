//! # ruvector-bounded-rag
//!
//! Bounded context RAG using MinCut graph partitioning.
//!
//! Standard RAG retrieves top-k chunks by vector similarity. This crate
//! adds two alternative strategies that respect a context *budget* while
//! maximising semantic coherence, measured via graph min-cut.
//!
//! ## Variants
//!
//! | Variant | Strategy | Strength |
//! |---------|----------|----------|
//! | `TopK`  | Cosine rank, no graph | Fastest, baseline |
//! | `GraphBfs` | Builds a dense similarity graph, then expands with a coherence gate | Budget-safe heuristic |
//! | `MinCutBounded` | Max-flow/min-cut flow network over a **dense O(n²) similarity graph**, then top-budget truncation | Coherent-partition heuristic |
//! | `SparseKnnMinCut` | Same flow network, but inter-chunk edges come from an **LSH-bucketed sparse k-NN graph** instead of all pairs (see [`sparse_knn`]) | Coherent partition at lower construction cost |
//!
//! `SparseKnnMinCut` exists because the nightly research that introduced
//! `MinCutBounded` ([ADR-272](https://github.com/ruvnet/ruvector)) measured
//! its dense O(n²·d) graph construction as the dominant cost at scale (over
//! a second at n=3000) and named a pre-built k-NN graph as the named
//! follow-up. [`sparse_knn`] is that follow-up: it reuses the exact same
//! [`flow::source_side_partition`] solver, isolating graph construction as
//! the only variable being measured.
//!
//! ## Quick start
//!
//! ```rust
//! use ruvector_bounded_rag::{Corpus, Query, RetrieverConfig, TopKRetriever, BoundedRetriever};
//!
//! let corpus = Corpus::from_vecs(vec![
//!     vec![1.0_f32, 0.0, 0.0],
//!     vec![0.9, 0.1, 0.0],
//!     vec![0.0, 0.0, 1.0],
//! ]);
//! let query = Query::new(vec![1.0_f32, 0.0, 0.0]);
//! let cfg = RetrieverConfig { budget: 2, ..Default::default() };
//! let retriever = TopKRetriever::new(cfg);
//! let result = retriever.retrieve(&corpus, &query);
//! assert_eq!(result.chunks.len(), 2);
//! ```

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use thiserror::Error;

mod flow;
pub mod sparse_knn;

pub use sparse_knn::{SparseKnnConfig, SparseKnnMinCutRetriever};

// ── public error type ──────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum RagError {
    #[error("empty corpus")]
    EmptyCorpus,
    #[error("dimension mismatch: corpus={0} query={1}")]
    DimMismatch(usize, usize),
    #[error("budget must be >= 1")]
    InvalidBudget,
}

// ── core data types ────────────────────────────────────────────────────────────

/// A single chunk stored in the corpus.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub id: usize,
    pub vector: Vec<f32>,
    /// Optional metadata label (used in precision scoring).
    pub label: Option<u32>,
}

/// Corpus of indexed chunks.
pub struct Corpus {
    pub chunks: Vec<Chunk>,
    pub dim: usize,
}

impl Corpus {
    pub fn from_vecs(vecs: Vec<Vec<f32>>) -> Self {
        let dim = vecs.first().map(|v| v.len()).unwrap_or(0);
        assert!(
            vecs.iter().all(|vector| vector.len() == dim),
            "all corpus vectors must have the same dimension"
        );
        let chunks = vecs
            .into_iter()
            .enumerate()
            .map(|(id, vector)| Chunk {
                id,
                vector,
                label: None,
            })
            .collect();
        Self { chunks, dim }
    }

    pub fn with_labels(mut self, labels: Vec<u32>) -> Self {
        assert_eq!(
            labels.len(),
            self.chunks.len(),
            "label count must match chunk count"
        );
        for (chunk, label) in self.chunks.iter_mut().zip(labels) {
            chunk.label = Some(label);
        }
        self
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Query vector plus ground-truth label set for precision scoring.
pub struct Query {
    pub vector: Vec<f32>,
    pub relevant_labels: HashSet<u32>,
}

impl Query {
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            relevant_labels: HashSet::new(),
        }
    }

    pub fn with_relevant(mut self, labels: impl IntoIterator<Item = u32>) -> Self {
        self.relevant_labels.extend(labels);
        self
    }
}

/// Configuration shared by all retrievers.
#[derive(Clone, Debug)]
pub struct RetrieverConfig {
    /// Maximum chunks to return (context budget).
    pub budget: usize,
    /// Minimum cosine similarity between chunks for a graph edge.
    pub edge_threshold: f32,
    /// Minimum cosine similarity from query to seed chunks.
    pub seed_threshold: f32,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            budget: 10,
            edge_threshold: 0.70,
            seed_threshold: 0.50,
        }
    }
}

/// Output of a retrieval operation.
#[derive(Debug)]
pub struct RetrievalResult {
    pub chunks: Vec<usize>,
    pub scores: Vec<f32>,
    /// Fraction of budget used.
    pub budget_utilisation: f32,
}

impl RetrievalResult {
    /// Precision: fraction of retrieved chunks whose label matches query.relevant_labels.
    pub fn precision(&self, corpus: &Corpus, query: &Query) -> f32 {
        if self.chunks.is_empty() || query.relevant_labels.is_empty() {
            return 0.0;
        }
        let hits = self
            .chunks
            .iter()
            .filter(|&&id| {
                corpus.chunks[id]
                    .label
                    .map(|l| query.relevant_labels.contains(&l))
                    .unwrap_or(false)
            })
            .count();
        hits as f32 / self.chunks.len() as f32
    }
}

// ── trait ──────────────────────────────────────────────────────────────────────

pub trait BoundedRetriever: Send + Sync {
    fn name(&self) -> &'static str;
    fn retrieve(&self, corpus: &Corpus, query: &Query) -> RetrievalResult;
}

// ── helpers ────────────────────────────────────────────────────────────────────

/// L2-normalise a vector in-place.
fn normalise(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Cosine similarity — assumes inputs are already L2-normalised.
#[inline]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

fn validate_config(cfg: &RetrieverConfig) {
    assert!(cfg.budget > 0, "budget must be greater than zero");
    assert!(
        (-1.0..=1.0).contains(&cfg.edge_threshold),
        "edge_threshold must be in [-1, 1]"
    );
    assert!(
        (-1.0..=1.0).contains(&cfg.seed_threshold),
        "seed_threshold must be in [-1, 1]"
    );
}

fn validate_query(corpus: &Corpus, query: &Query) {
    assert_eq!(
        query.vector.len(),
        corpus.dim,
        "query dimension must match corpus dimension"
    );
}

// ── Variant 1: TopK (baseline) ─────────────────────────────────────────────────

/// Pure cosine-rank top-k retriever with no graph reasoning.
pub struct TopKRetriever {
    cfg: RetrieverConfig,
}

impl TopKRetriever {
    pub fn new(cfg: RetrieverConfig) -> Self {
        validate_config(&cfg);
        Self { cfg }
    }
}

impl BoundedRetriever for TopKRetriever {
    fn name(&self) -> &'static str {
        "TopK"
    }

    fn retrieve(&self, corpus: &Corpus, query: &Query) -> RetrievalResult {
        if corpus.is_empty() {
            return RetrievalResult {
                chunks: vec![],
                scores: vec![],
                budget_utilisation: 0.0,
            };
        }
        validate_query(corpus, query);
        let qn = normalise(&query.vector);

        // Score every chunk
        let mut scored: Vec<(usize, f32)> = corpus
            .chunks
            .iter()
            .map(|c| {
                let cn = normalise(&c.vector);
                (c.id, cosine(&qn, &cn))
            })
            .collect();

        // Sort descending
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(self.cfg.budget);

        let n = scored.len();
        let chunks: Vec<usize> = scored.iter().map(|(id, _)| *id).collect();
        let scores: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();
        RetrievalResult {
            budget_utilisation: n as f32 / self.cfg.budget as f32,
            chunks,
            scores,
        }
    }
}

// ── Variant 2: GraphBFS ────────────────────────────────────────────────────────

/// BFS expansion on the chunk similarity graph, bounded by budget and coherence gate.
///
/// Algorithm:
/// 1. Find seed chunks with cosine(query, chunk) >= seed_threshold.
/// 2. BFS: expand to neighbours with edge weight >= edge_threshold.
/// 3. Stop when budget is exhausted.
///
/// Priority queue ensures high-affinity chunks enter first.
///
/// This research implementation rebuilds a dense similarity graph for every
/// query, costing O(n²·d) before the O(V+E) traversal.
pub struct GraphBfsRetriever {
    cfg: RetrieverConfig,
}

impl GraphBfsRetriever {
    pub fn new(cfg: RetrieverConfig) -> Self {
        validate_config(&cfg);
        Self { cfg }
    }

    fn build_adjacency(corpus: &Corpus, threshold: f32) -> Vec<Vec<(usize, f32)>> {
        let n = corpus.len();
        let normed: Vec<Vec<f32>> = corpus.chunks.iter().map(|c| normalise(&c.vector)).collect();
        let mut adj = vec![vec![]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine(&normed[i], &normed[j]);
                if sim >= threshold {
                    adj[i].push((j, sim));
                    adj[j].push((i, sim));
                }
            }
        }
        adj
    }
}

impl BoundedRetriever for GraphBfsRetriever {
    fn name(&self) -> &'static str {
        "GraphBFS"
    }

    fn retrieve(&self, corpus: &Corpus, query: &Query) -> RetrievalResult {
        if corpus.is_empty() {
            return RetrievalResult {
                chunks: vec![],
                scores: vec![],
                budget_utilisation: 0.0,
            };
        }
        validate_query(corpus, query);

        let qn = normalise(&query.vector);
        let normed: Vec<Vec<f32>> = corpus.chunks.iter().map(|c| normalise(&c.vector)).collect();
        let adj = Self::build_adjacency(corpus, self.cfg.edge_threshold);

        // Seeds: chunks with sim >= seed_threshold (capped at budget)
        // Use a max-heap of (score, id)
        #[derive(PartialEq)]
        struct Entry(f32, usize);
        impl Eq for Entry {}
        impl PartialOrd for Entry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Entry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0.total_cmp(&other.0)
            }
        }

        let mut heap: BinaryHeap<Entry> = BinaryHeap::new();
        for (i, nv) in normed.iter().enumerate() {
            let sim = cosine(&qn, nv);
            if sim >= self.cfg.seed_threshold {
                heap.push(Entry(sim, i));
            }
        }
        // Fall back: take top-1 even below threshold
        if heap.is_empty() {
            let (best_id, best_sim) = normed
                .iter()
                .enumerate()
                .map(|(i, nv)| (i, cosine(&qn, nv)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((0, 0.0));
            heap.push(Entry(best_sim, best_id));
        }

        let mut visited = HashSet::new();
        let mut result_chunks = Vec::new();
        let mut result_scores = Vec::new();

        // BFS via priority queue
        while let Some(Entry(score, id)) = heap.pop() {
            if visited.contains(&id) {
                continue;
            }
            if result_chunks.len() >= self.cfg.budget {
                break;
            }
            visited.insert(id);
            result_chunks.push(id);
            result_scores.push(score);

            // Expand neighbours
            for &(nb, edge_w) in &adj[id] {
                if !visited.contains(&nb) {
                    let nb_score = cosine(&qn, &normed[nb]);
                    heap.push(Entry(edge_w.min(nb_score), nb));
                }
            }
        }

        let n = result_chunks.len();
        RetrievalResult {
            budget_utilisation: n as f32 / self.cfg.budget as f32,
            chunks: result_chunks,
            scores: result_scores,
        }
    }
}

// ── Variant 3: MinCut Bounded ──────────────────────────────────────────────────

/// Max-flow / min-cut retriever.
///
/// Constructs a flow network:
/// - Source (node n): edges to each chunk with capacity = cosine(query, chunk)
/// - Sink (node n+1): edges from each chunk with capacity = 1 - cosine(query, chunk)
/// - Inter-chunk edges: capacity = cosine(chunk_i, chunk_j) * edge_scale
///
/// Runs Edmonds-Karp (BFS Ford-Fulkerson) to find max-flow, then BFS from
/// source on residual graph to find the source-side partition — these are the
/// retrieved chunks (capped at budget).
///
/// The min-cut separates "query-coherent" chunks from noise with minimum weight,
/// giving a principled coherence boundary. The partition is subsequently
/// ranked and truncated, so this is not an exact budget-constrained min-cut.
pub struct MinCutRetriever {
    cfg: RetrieverConfig,
    /// Scale factor for inter-chunk edge capacities.
    edge_scale: f32,
}

impl MinCutRetriever {
    pub fn new(cfg: RetrieverConfig) -> Self {
        validate_config(&cfg);
        Self {
            cfg,
            edge_scale: 0.5,
        }
    }

    pub fn with_edge_scale(mut self, scale: f32) -> Self {
        assert!(
            scale.is_finite() && scale >= 0.0,
            "edge scale must be finite and non-negative"
        );
        self.edge_scale = scale;
        self
    }
}

impl BoundedRetriever for MinCutRetriever {
    fn name(&self) -> &'static str {
        "MinCutBounded"
    }

    fn retrieve(&self, corpus: &Corpus, query: &Query) -> RetrievalResult {
        if corpus.is_empty() {
            return RetrievalResult {
                chunks: vec![],
                scores: vec![],
                budget_utilisation: 0.0,
            };
        }
        validate_query(corpus, query);

        let n = corpus.len();
        let qn = normalise(&query.vector);
        let normed: Vec<Vec<f32>> = corpus.chunks.iter().map(|c| normalise(&c.vector)).collect();

        let source_cap: Vec<f32> = (0..n).map(|i| cosine(&qn, &normed[i]).max(0.001)).collect();
        let sink_cap: Vec<f32> = (0..n)
            .map(|i| (1.0 - cosine(&qn, &normed[i])).max(0.001))
            .collect();

        // Dense O(n^2) inter-chunk edge discovery — this is the documented
        // bottleneck that `sparse_knn::SparseKnnMinCutRetriever` replaces
        // with LSH-bucketed candidate generation (see ADR-272 follow-up).
        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine(&normed[i], &normed[j]);
                if sim >= self.cfg.edge_threshold {
                    edges.push((i, j, sim * self.edge_scale));
                }
            }
        }

        let in_source_set = flow::source_side_partition(n, &source_cap, &sink_cap, &edges);

        // Collect chunk indices in source partition, sorted by query similarity
        let mut retrieved: Vec<(usize, f32)> = (0..n)
            .filter(|&i| in_source_set[i])
            .map(|i| (i, cosine(&qn, &normed[i])))
            .collect();
        retrieved.sort_by(|a, b| b.1.total_cmp(&a.1));
        retrieved.truncate(self.cfg.budget);

        let retrieved_n = retrieved.len();
        let chunks: Vec<usize> = retrieved.iter().map(|(id, _)| *id).collect();
        let scores: Vec<f32> = retrieved.iter().map(|(_, s)| *s).collect();

        RetrievalResult {
            budget_utilisation: retrieved_n as f32 / self.cfg.budget as f32,
            chunks,
            scores,
        }
    }
}

// ── evaluation helpers ─────────────────────────────────────────────────────────

/// Mean average precision across a query set.
pub fn mean_precision(retriever: &dyn BoundedRetriever, corpus: &Corpus, queries: &[Query]) -> f32 {
    if queries.is_empty() {
        return 0.0;
    }
    let total: f32 = queries
        .iter()
        .map(|q| retriever.retrieve(corpus, q).precision(corpus, q))
        .sum();
    total / queries.len() as f32
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_corpus() -> Corpus {
        // Two clusters: cluster 0 near (1,0,0), cluster 1 near (0,0,1)
        let vecs = vec![
            vec![1.0_f32, 0.0, 0.0], // 0 – cluster 0
            vec![0.95, 0.05, 0.0],   // 1 – cluster 0
            vec![0.90, 0.10, 0.0],   // 2 – cluster 0
            vec![0.85, 0.15, 0.05],  // 3 – cluster 0
            vec![0.0, 0.0, 1.0],     // 4 – cluster 1
            vec![0.05, 0.05, 0.95],  // 5 – cluster 1
            vec![0.0, 0.1, 0.9],     // 6 – cluster 1
            vec![0.5, 0.5, 0.5],     // 7 – noise
        ];
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 2];
        Corpus::from_vecs(vecs).with_labels(labels)
    }

    fn cluster0_query() -> Query {
        Query::new(vec![1.0_f32, 0.0, 0.0]).with_relevant([0u32])
    }

    #[test]
    fn topk_returns_budget_chunks() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 3,
            ..Default::default()
        };
        let r = TopKRetriever::new(cfg).retrieve(&corpus, &cluster0_query());
        assert_eq!(r.chunks.len(), 3);
    }

    #[test]
    fn topk_highest_similarity_first() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 4,
            ..Default::default()
        };
        let r = TopKRetriever::new(cfg).retrieve(&corpus, &cluster0_query());
        // Chunk 0 has cosine 1.0 — must be first
        assert_eq!(r.chunks[0], 0);
        // Scores should be non-increasing
        for w in r.scores.windows(2) {
            assert!(w[0] >= w[1] - 1e-6, "scores not sorted: {:?}", r.scores);
        }
    }

    #[test]
    fn topk_precision_cluster0() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 4,
            ..Default::default()
        };
        let retriever = TopKRetriever::new(cfg);
        let q = cluster0_query();
        let r = retriever.retrieve(&corpus, &q);
        let p = r.precision(&corpus, &q);
        assert!(p >= 0.75, "TopK precision={p} too low");
    }

    #[test]
    fn graphbfs_respects_budget() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 3,
            edge_threshold: 0.8,
            seed_threshold: 0.7,
        };
        let r = GraphBfsRetriever::new(cfg).retrieve(&corpus, &cluster0_query());
        assert!(r.chunks.len() <= 3);
    }

    #[test]
    fn graphbfs_retrieves_cluster0() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 5,
            edge_threshold: 0.75,
            seed_threshold: 0.60,
        };
        let q = cluster0_query();
        let r = GraphBfsRetriever::new(cfg.clone()).retrieve(&corpus, &q);
        let p = r.precision(&corpus, &q);
        assert!(p >= 0.60, "GraphBFS precision={p} too low");
    }

    #[test]
    fn mincut_respects_budget() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 3,
            edge_threshold: 0.75,
            ..Default::default()
        };
        let r = MinCutRetriever::new(cfg).retrieve(&corpus, &cluster0_query());
        assert!(r.chunks.len() <= 3);
    }

    #[test]
    fn mincut_precision_cluster0() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 6,
            edge_threshold: 0.75,
            seed_threshold: 0.50,
        };
        let q = cluster0_query();
        let r = MinCutRetriever::new(cfg).retrieve(&corpus, &q);
        let p = r.precision(&corpus, &q);
        // MinCut should achieve at least 60% precision on a clustered corpus
        assert!(p >= 0.60, "MinCut precision={p} too low");
    }

    #[test]
    fn mincut_no_cross_cluster_bleed() {
        // Strongly separated clusters — MinCut should not bleed into cluster 1
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 4,
            edge_threshold: 0.85,
            seed_threshold: 0.70,
        };
        let q = Query::new(vec![1.0_f32, 0.0, 0.0]);
        let r = MinCutRetriever::new(cfg).retrieve(&corpus, &q);
        // Chunk 4,5,6 are cluster 1 — they should NOT appear with tight threshold
        for &id in &r.chunks {
            assert!(id < 4 || id == 7, "MinCut bled into cluster 1: chunk={id}");
        }
    }

    #[test]
    fn acceptance_precision_threshold() {
        use rand::{rngs::StdRng, SeedableRng};
        use rand_distr::{Distribution, Normal};

        // 200-chunk corpus, 2 clusters, 30-dim vectors
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 30_usize;
        let n_per_cluster = 100_usize;

        let normal = Normal::new(0.0_f32, 0.1).unwrap();

        let mut vecs: Vec<Vec<f32>> = Vec::new();
        let mut labels: Vec<u32> = Vec::new();

        // Cluster 0: centroid e0
        for _ in 0..n_per_cluster {
            let mut v = vec![0.0_f32; dim];
            v[0] = 1.0;
            for x in v.iter_mut() {
                *x += normal.sample(&mut rng);
            }
            vecs.push(v);
            labels.push(0);
        }
        // Cluster 1: centroid e1
        for _ in 0..n_per_cluster {
            let mut v = vec![0.0_f32; dim];
            v[1] = 1.0;
            for x in v.iter_mut() {
                *x += normal.sample(&mut rng);
            }
            vecs.push(v);
            labels.push(1);
        }

        let corpus = Corpus::from_vecs(vecs).with_labels(labels);

        // Query pointing at cluster 0
        let mut qv = vec![0.0_f32; dim];
        qv[0] = 1.0;
        let query = Query::new(qv).with_relevant([0u32]);

        let cfg = RetrieverConfig {
            budget: 30,
            edge_threshold: 0.70,
            seed_threshold: 0.40,
        };

        let topk = TopKRetriever::new(cfg.clone());
        let bfs = GraphBfsRetriever::new(cfg.clone());
        let mc = MinCutRetriever::new(cfg.clone());

        let p_topk = topk.retrieve(&corpus, &query).precision(&corpus, &query);
        let p_bfs = bfs.retrieve(&corpus, &query).precision(&corpus, &query);
        let p_mc = mc.retrieve(&corpus, &query).precision(&corpus, &query);

        eprintln!("Acceptance precision — TopK:{p_topk:.3} BFS:{p_bfs:.3} MinCut:{p_mc:.3}");

        // All three must achieve > 0.70 precision on this well-separated corpus
        assert!(p_topk >= 0.70, "TopK precision={p_topk:.3} below threshold");
        assert!(p_bfs >= 0.70, "BFS precision={p_bfs:.3} below threshold");
        assert!(p_mc >= 0.70, "MinCut precision={p_mc:.3} below threshold");
    }

    #[test]
    #[should_panic(expected = "budget must be greater than zero")]
    fn zero_budget_is_rejected() {
        let _ = TopKRetriever::new(RetrieverConfig {
            budget: 0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "all corpus vectors must have the same dimension")]
    fn inconsistent_corpus_dimensions_are_rejected() {
        let _ = Corpus::from_vecs(vec![vec![1.0, 0.0], vec![1.0]]);
    }

    #[test]
    #[should_panic(expected = "query dimension must match corpus dimension")]
    fn query_dimension_mismatch_is_rejected() {
        let corpus = toy_corpus();
        let query = Query::new(vec![1.0, 0.0]);
        let _ = TopKRetriever::new(Default::default()).retrieve(&corpus, &query);
    }
}
