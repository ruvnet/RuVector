//! Sparse LSH-bucketed k-NN graph construction for `MinCutBounded` retrieval.
//!
//! # Why this module exists
//!
//! The `MinCutRetriever` variant in the crate root builds its inter-chunk
//! similarity graph by comparing every pair of chunks: O(n²·d). The nightly
//! research that introduced it (2026-07-25, `docs/research/nightly/
//! 2026-07-25-bounded-rag-mincut/`) measured this as the dominant cost at
//! scale — 1.27 seconds at n=3000 on commodity hardware — and named a
//! pre-built k-NN graph as the unimplemented Phase 2 follow-up.
//!
//! This module is that follow-up. It replaces the exhaustive all-pairs scan
//! with random-hyperplane locality-sensitive hashing (SimHash; Charikar,
//! STOC 2002), which buckets near-duplicate-angle vectors together using
//! `O(n·d·L·h)` hashing work (`L` tables, `h` hyperplanes each) instead of
//! `O(n²·d)` comparisons, then only computes exact cosine similarity within
//! each bucket. The discovered edges feed the *same*
//! [`crate::flow::source_side_partition`] solver `MinCutRetriever` uses, so
//! the two variants are an apples-to-apples comparison isolating graph
//! construction as the one changed variable.
//!
//! This is a sparsification, not a correctness-preserving transform: LSH
//! can miss a true edge whose two vectors never land in the same bucket in
//! any table. [`SparseGraphStats`] reports how many candidate pairs were
//! actually compared, so the tradeoff is measurable rather than asserted.

use crate::{cosine, normalise, BoundedRetriever, Corpus, Query, RetrievalResult, RetrieverConfig};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{HashMap, HashSet};

/// Configuration for the LSH sparse graph builder.
///
/// # `max_degree`: why this field exists
///
/// The first version of this module only filtered LSH candidate pairs by
/// `edge_threshold` (an unbounded *threshold graph*), on the assumption
/// that cutting candidate-pair-*discovery* cost from O(n²) to O(n·k) would
/// cut end-to-end latency. Measurement rejected that: on tightly clustered
/// corpora — the exact workload `MinCutBounded` targets — upwards of 90% of
/// same-cluster candidate pairs clear the threshold regardless of how they
/// were discovered, so the resulting flow network stays just as edge-dense
/// as the original O(n²) graph. Cheaper discovery of a dense graph is still
/// a dense graph.
///
/// `max_degree: Some(k)` fixes the actual bottleneck: after threshold
/// filtering, each node's kept neighbours are capped to its `k` nearest
/// (by cosine) among LSH candidates, and an edge is kept only when it is
/// *mutual* — in both endpoints' top-`k` lists. This provably bounds every
/// node's degree to `k`, so the flow network has at most `n·k/2` edges
/// independent of cluster density. `None` keeps the unbounded threshold
/// behaviour (useful as an isolating control to show the above failure
/// mode in benchmarks).
#[derive(Clone, Debug)]
pub struct SparseKnnConfig {
    /// Number of independent hash tables (`L`). More tables raise recall
    /// of true near-duplicate pairs at the cost of more hashing work.
    pub num_tables: usize,
    /// Number of random hyperplanes per table (`h`). More hyperplanes
    /// shrink expected bucket size (fewer, more precise candidate pairs)
    /// at the cost of missing true edges that split across buckets.
    pub hyperplanes: usize,
    /// Seed for the hyperplane RNG — deterministic for a fixed seed.
    pub seed: u64,
    /// When `Some(k)`, bound every node's kept degree to its mutual
    /// `k`-nearest LSH candidates. `None` keeps every candidate pair that
    /// clears `edge_threshold` with no degree bound.
    pub max_degree: Option<usize>,
}

impl Default for SparseKnnConfig {
    fn default() -> Self {
        Self {
            num_tables: 6,
            hyperplanes: 10,
            seed: 0x5EED_CAFE,
            max_degree: None,
        }
    }
}

/// Measured cost of a sparse graph build, for honest benchmark reporting.
#[derive(Clone, Debug, Default)]
pub struct SparseGraphStats {
    /// Total candidate pairs whose exact cosine similarity was computed
    /// (i.e. pairs that landed in the same bucket in at least one table).
    pub candidate_pairs_checked: usize,
    /// Edges that passed `edge_threshold` and were kept.
    pub edges_kept: usize,
}

/// Builds a sparse inter-chunk edge list using random-hyperplane LSH.
///
/// `normed` must already be L2-normalised. Returns `(i, j, weight)` triples
/// with `i < j` and `weight = cosine(i, j)`, plus [`SparseGraphStats`] for
/// the caller to report construction cost honestly.
pub fn build_sparse_edges(
    normed: &[Vec<f32>],
    dim: usize,
    edge_threshold: f32,
    cfg: &SparseKnnConfig,
) -> (Vec<(usize, usize, f32)>, SparseGraphStats) {
    let n = normed.len();
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut checked: HashSet<(usize, usize)> = HashSet::new();
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    let mut candidate_pairs_checked = 0usize;

    for _table in 0..cfg.num_tables {
        // Random hyperplanes: each is a vector of iid standard-normal-ish
        // components; only the sign of the dot product is used (SimHash),
        // so a cheap uniform(-1, 1) draw is sufficient.
        let hyperplanes: Vec<Vec<f32>> = (0..cfg.hyperplanes)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect())
            .collect();

        let mut buckets: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, v) in normed.iter().enumerate() {
            let mut key: u32 = 0;
            for (bit, plane) in hyperplanes.iter().enumerate() {
                let dot: f32 = v.iter().zip(plane.iter()).map(|(a, b)| a * b).sum();
                if dot >= 0.0 {
                    key |= 1 << bit;
                }
            }
            buckets.entry(key).or_default().push(i);
        }

        for members in buckets.values() {
            if members.len() < 2 {
                continue;
            }
            for a in 0..members.len() {
                for b in (a + 1)..members.len() {
                    let (i, j) = (members[a].min(members[b]), members[a].max(members[b]));
                    if !checked.insert((i, j)) {
                        continue; // already compared via an earlier table
                    }
                    candidate_pairs_checked += 1;
                    let sim = cosine(&normed[i], &normed[j]);
                    if sim >= edge_threshold {
                        edges.push((i, j, sim));
                    }
                }
            }
        }
    }

    let Some(k) = cfg.max_degree else {
        let stats = SparseGraphStats {
            candidate_pairs_checked,
            edges_kept: edges.len(),
        };
        return (edges, stats);
    };

    // Degree-capped mode: keep an edge only if it is in the mutual top-k
    // (by cosine) of both endpoints. This bounds every node's final degree
    // to <= k regardless of how many threshold-passing candidates it has.
    let mut neighbours: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for &(i, j, sim) in &edges {
        neighbours[i].push((j, sim));
        neighbours[j].push((i, sim));
    }
    let top_k_sets: Vec<HashSet<usize>> = neighbours
        .into_iter()
        .map(|mut nbrs| {
            nbrs.sort_by(|a, b| b.1.total_cmp(&a.1));
            nbrs.truncate(k);
            nbrs.into_iter().map(|(id, _)| id).collect()
        })
        .collect();

    let capped_edges: Vec<(usize, usize, f32)> = edges
        .into_iter()
        .filter(|&(i, j, _)| top_k_sets[i].contains(&j) && top_k_sets[j].contains(&i))
        .collect();

    let stats = SparseGraphStats {
        candidate_pairs_checked,
        edges_kept: capped_edges.len(),
    };
    (capped_edges, stats)
}

/// Max-flow / min-cut retriever whose inter-chunk graph is built via
/// LSH-bucketed sparse k-NN discovery instead of an exhaustive all-pairs
/// scan. See the module documentation for the hypothesis this tests.
pub struct SparseKnnMinCutRetriever {
    cfg: RetrieverConfig,
    lsh_cfg: SparseKnnConfig,
    edge_scale: f32,
}

impl SparseKnnMinCutRetriever {
    pub fn new(cfg: RetrieverConfig) -> Self {
        assert!(cfg.budget > 0, "budget must be greater than zero");
        Self {
            cfg,
            lsh_cfg: SparseKnnConfig::default(),
            edge_scale: 0.5,
        }
    }

    pub fn with_lsh_config(mut self, lsh_cfg: SparseKnnConfig) -> Self {
        self.lsh_cfg = lsh_cfg;
        self
    }

    pub fn with_edge_scale(mut self, scale: f32) -> Self {
        assert!(
            scale.is_finite() && scale >= 0.0,
            "edge scale must be finite and non-negative"
        );
        self.edge_scale = scale;
        self
    }

    /// Runs retrieval and also returns the sparse graph construction stats,
    /// for benchmark reporting. `retrieve` (the trait method) discards
    /// these; use this directly when the stats are needed.
    pub fn retrieve_with_stats(
        &self,
        corpus: &Corpus,
        query: &Query,
    ) -> (RetrievalResult, SparseGraphStats) {
        if corpus.is_empty() {
            return (
                RetrievalResult {
                    chunks: vec![],
                    scores: vec![],
                    budget_utilisation: 0.0,
                },
                SparseGraphStats::default(),
            );
        }
        assert_eq!(
            query.vector.len(),
            corpus.dim,
            "query dimension must match corpus dimension"
        );

        let n = corpus.len();
        let qn = normalise(&query.vector);
        let normed: Vec<Vec<f32>> = corpus.chunks.iter().map(|c| normalise(&c.vector)).collect();

        let source_cap: Vec<f32> = (0..n).map(|i| cosine(&qn, &normed[i]).max(0.001)).collect();
        let sink_cap: Vec<f32> = (0..n)
            .map(|i| (1.0 - cosine(&qn, &normed[i])).max(0.001))
            .collect();

        let (raw_edges, stats) =
            build_sparse_edges(&normed, corpus.dim, self.cfg.edge_threshold, &self.lsh_cfg);
        let edges: Vec<(usize, usize, f32)> = raw_edges
            .into_iter()
            .map(|(i, j, sim)| (i, j, sim * self.edge_scale))
            .collect();

        let in_source_set = crate::flow::source_side_partition(n, &source_cap, &sink_cap, &edges);

        let mut retrieved: Vec<(usize, f32)> = (0..n)
            .filter(|&i| in_source_set[i])
            .map(|i| (i, cosine(&qn, &normed[i])))
            .collect();
        retrieved.sort_by(|a, b| b.1.total_cmp(&a.1));
        retrieved.truncate(self.cfg.budget);

        let retrieved_n = retrieved.len();
        let chunks: Vec<usize> = retrieved.iter().map(|(id, _)| *id).collect();
        let scores: Vec<f32> = retrieved.iter().map(|(_, s)| *s).collect();

        (
            RetrievalResult {
                budget_utilisation: retrieved_n as f32 / self.cfg.budget as f32,
                chunks,
                scores,
            },
            stats,
        )
    }
}

impl BoundedRetriever for SparseKnnMinCutRetriever {
    fn name(&self) -> &'static str {
        "SparseKnnMinCut"
    }

    fn retrieve(&self, corpus: &Corpus, query: &Query) -> RetrievalResult {
        self.retrieve_with_stats(corpus, query).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_corpus() -> Corpus {
        let vecs = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.95, 0.05, 0.0],
            vec![0.90, 0.10, 0.0],
            vec![0.85, 0.15, 0.05],
            vec![0.0, 0.0, 1.0],
            vec![0.05, 0.05, 0.95],
            vec![0.0, 0.1, 0.9],
            vec![0.5, 0.5, 0.5],
        ];
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 2];
        Corpus::from_vecs(vecs).with_labels(labels)
    }

    #[test]
    fn respects_budget() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 3,
            edge_threshold: 0.75,
            ..Default::default()
        };
        let r = SparseKnnMinCutRetriever::new(cfg).retrieve(
            &corpus,
            &Query::new(vec![1.0_f32, 0.0, 0.0]).with_relevant([0u32]),
        );
        assert!(r.chunks.len() <= 3);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 4,
            edge_threshold: 0.75,
            ..Default::default()
        };
        let q = Query::new(vec![1.0_f32, 0.0, 0.0]).with_relevant([0u32]);
        let a = SparseKnnMinCutRetriever::new(cfg.clone()).retrieve(&corpus, &q);
        let b = SparseKnnMinCutRetriever::new(cfg).retrieve(&corpus, &q);
        assert_eq!(a.chunks, b.chunks);
        assert_eq!(a.scores, b.scores);
    }

    #[test]
    fn no_duplicate_chunks_returned() {
        let corpus = toy_corpus();
        let cfg = RetrieverConfig {
            budget: 8,
            edge_threshold: 0.70,
            seed_threshold: 0.0,
        };
        let r = SparseKnnMinCutRetriever::new(cfg)
            .retrieve(&corpus, &Query::new(vec![1.0_f32, 0.0, 0.0]));
        let unique: HashSet<usize> = r.chunks.iter().copied().collect();
        assert_eq!(unique.len(), r.chunks.len());
    }

    #[test]
    fn precision_on_larger_clustered_corpus() {
        use rand::{rngs::StdRng, SeedableRng};
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(42);
        let dim = 30_usize;
        let n_per_cluster = 100_usize;
        let normal = Normal::new(0.0_f32, 0.1).unwrap();

        let mut vecs: Vec<Vec<f32>> = Vec::new();
        let mut labels: Vec<u32> = Vec::new();
        for (cluster, axis) in [(0u32, 0usize), (1u32, 1usize)] {
            for _ in 0..n_per_cluster {
                let mut v = vec![0.0_f32; dim];
                v[axis] = 1.0;
                for x in v.iter_mut() {
                    *x += normal.sample(&mut rng);
                }
                vecs.push(v);
                labels.push(cluster);
            }
        }
        let corpus = Corpus::from_vecs(vecs).with_labels(labels);

        let mut qv = vec![0.0_f32; dim];
        qv[0] = 1.0;
        let query = Query::new(qv).with_relevant([0u32]);

        let cfg = RetrieverConfig {
            budget: 30,
            edge_threshold: 0.70,
            seed_threshold: 0.40,
        };
        let r = SparseKnnMinCutRetriever::new(cfg).retrieve(&corpus, &query);
        let p = r.precision(&corpus, &query);
        assert!(
            p >= 0.70,
            "SparseKnnMinCut precision={p} below threshold on well-separated corpus"
        );
    }

    #[test]
    fn degree_cap_bounds_degree_on_a_dense_cluster() {
        // A single tight cluster: without a degree cap, an unbounded
        // threshold graph keeps nearly every pair (as measured in the
        // nightly benchmark). With max_degree=Some(k), no node may end up
        // with more than k kept edges.
        use rand::{rngs::StdRng, SeedableRng};
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(3);
        let dim = 16usize;
        let n = 400usize;
        let normal = Normal::new(0.0_f32, 0.05).unwrap();
        let mut vecs = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v = vec![0.0_f32; dim];
            v[0] = 1.0;
            for x in v.iter_mut() {
                *x += normal.sample(&mut rng);
            }
            vecs.push(v);
        }
        let normed: Vec<Vec<f32>> = vecs.iter().map(|v| normalise(v)).collect();

        let unbounded_cfg = SparseKnnConfig::default();
        let (unbounded_edges, _) = build_sparse_edges(&normed, dim, 0.60, &unbounded_cfg);
        let mut unbounded_degree = vec![0usize; n];
        for &(i, j, _) in &unbounded_edges {
            unbounded_degree[i] += 1;
            unbounded_degree[j] += 1;
        }
        assert!(
            *unbounded_degree.iter().max().unwrap() > 20,
            "expected the dense cluster to produce high-degree nodes without a cap"
        );

        let k = 8;
        let capped_cfg = SparseKnnConfig {
            max_degree: Some(k),
            ..SparseKnnConfig::default()
        };
        let (capped_edges, _) = build_sparse_edges(&normed, dim, 0.60, &capped_cfg);
        let mut capped_degree = vec![0usize; n];
        for &(i, j, _) in &capped_edges {
            capped_degree[i] += 1;
            capped_degree[j] += 1;
        }
        assert!(
            capped_degree.iter().all(|&d| d <= k),
            "mutual top-k must bound every node's degree to k={k}, got max={}",
            capped_degree.iter().max().unwrap()
        );
    }

    #[test]
    fn degree_capped_precision_on_larger_clustered_corpus() {
        use rand::{rngs::StdRng, SeedableRng};
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(42);
        let dim = 30_usize;
        let n_per_cluster = 100_usize;
        let normal = Normal::new(0.0_f32, 0.1).unwrap();

        let mut vecs: Vec<Vec<f32>> = Vec::new();
        let mut labels: Vec<u32> = Vec::new();
        for (cluster, axis) in [(0u32, 0usize), (1u32, 1usize)] {
            for _ in 0..n_per_cluster {
                let mut v = vec![0.0_f32; dim];
                v[axis] = 1.0;
                for x in v.iter_mut() {
                    *x += normal.sample(&mut rng);
                }
                vecs.push(v);
                labels.push(cluster);
            }
        }
        let corpus = Corpus::from_vecs(vecs).with_labels(labels);

        let mut qv = vec![0.0_f32; dim];
        qv[0] = 1.0;
        let query = Query::new(qv).with_relevant([0u32]);

        let cfg = RetrieverConfig {
            budget: 30,
            edge_threshold: 0.70,
            seed_threshold: 0.40,
        };
        let retriever = SparseKnnMinCutRetriever::new(cfg).with_lsh_config(SparseKnnConfig {
            max_degree: Some(20),
            ..SparseKnnConfig::default()
        });
        let r = retriever.retrieve(&corpus, &query);
        let p = r.precision(&corpus, &query);
        assert!(
            p >= 0.70,
            "degree-capped SparseKnnMinCut precision={p} below threshold"
        );
    }

    #[test]
    fn sparse_graph_checks_far_fewer_pairs_than_dense_at_scale() {
        use rand::{rngs::StdRng, SeedableRng};
        use rand_distr::{Distribution, Normal};

        let mut rng = StdRng::seed_from_u64(7);
        let dim = 32usize;
        let n = 1500usize;
        let normal = Normal::new(0.0_f32, 0.1).unwrap();
        let mut vecs = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0.0_f32; dim];
            v[i % dim] = 1.0;
            for x in v.iter_mut() {
                *x += normal.sample(&mut rng);
            }
            vecs.push(v);
        }
        let normed: Vec<Vec<f32>> = vecs.iter().map(|v| normalise(v)).collect();
        let (_edges, stats) = build_sparse_edges(&normed, dim, 0.70, &SparseKnnConfig::default());

        let dense_pairs = n * (n - 1) / 2;
        assert!(
            stats.candidate_pairs_checked < dense_pairs / 4,
            "LSH checked {} pairs, expected well under a quarter of the dense {} pairs",
            stats.candidate_pairs_checked,
            dense_pairs
        );
    }
}
