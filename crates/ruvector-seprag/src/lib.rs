//! # ruvector-seprag
//!
//! **SepRAG** — CCH-inspired separator-tree retrieval for hybrid vector + graph
//! memory. This crate is the **M0 correctness gate** described in
//! `docs/plans/seprag-cch-retrieval/M0-correctness-gate.md` and the ADRs
//! 196–199 (`docs/adr/`).
//!
//! It adapts Customizable Contraction Hierarchies — nested dissection, balanced
//! separators, contraction shortcuts, elimination trees, and separator-tree
//! k-NN — to graph-distance retrieval. M0 validates **correctness** on synthetic
//! graphs against a brute-force Dijkstra oracle; it deliberately uses a simple
//! self-contained separator finder. Separator *quality* and real-data scale are
//! M1's concern (where `ruvector-mincut` machinery is swapped in).
//!
//! ## Pipeline
//!
//! ```text
//! Graph ─► nested_dissection ─► contract ─► customize ─► KnnIndex::knn
//!         (order, sep tree)     (G+, elim)  (weights)    (pruned k-NN)
//! ```
//!
//! ## Example
//!
//! ```
//! use ruvector_seprag::{gen, SepRag};
//!
//! let g = gen::sbm(4, 25, 0.30, 0.01, 42);     // 4 communities × 25 vertices
//! let pois: Vec<u32> = (0..100).step_by(3).collect();
//! let sr = SepRag::build(g);
//! let idx = sr.index(&pois);
//! let topk = idx.query(7, 5);                    // 5 nearest POIs to vertex 7
//! assert!(topk.len() <= 5);
//! ```

pub mod ann;
pub mod contraction;
pub mod customize;
pub mod gen;
pub mod graph;
pub mod order;
pub mod query;

pub use contraction::Topology;
pub use customize::Metric;
pub use graph::{Graph, NodeId};
pub use order::{SepNode, SeparatorKind, SepTree};
pub use query::{KnnIndex, QueryStats};

/// A built SepRAG hierarchy: metric-independent topology + one customized metric.
pub struct SepRag {
    pub graph: Graph,
    pub topo: Topology,
    pub metric: Metric,
    pub sep_tree: SepTree,
}

impl SepRag {
    /// Build the full hierarchy from a graph (order → contract → customize) using
    /// the graph's own edge weights as the metric.
    #[must_use]
    pub fn build(graph: Graph) -> Self {
        Self::build_with(graph, SeparatorKind::Balanced)
    }

    /// Build with an explicit separator strategy (for M1 A/B attribution).
    #[must_use]
    pub fn build_with(graph: Graph, kind: SeparatorKind) -> Self {
        let ord = order::nested_dissection_kind(&graph, kind);
        let topo = contraction::contract(&graph, &ord.order);
        let metric = customize::customize(&topo);
        SepRag { graph, topo, metric, sep_tree: ord.sep_tree }
    }

    /// Build a bucket index for a fixed POI set.
    #[must_use]
    pub fn index<'a>(&'a self, pois: &[NodeId]) -> Index<'a> {
        Index {
            inner: KnnIndex::build(&self.topo, &self.metric, pois),
        }
    }

    /// Shortcut-blowup ratio `|G+| / |G_nav|` — the ADR-199 go/no-go metric.
    #[must_use]
    pub fn blowup_ratio(&self) -> f64 {
        let base = self.graph.edges().count().max(1);
        self.topo.arc_count() as f64 / base as f64
    }
}

/// Query handle over a fixed POI set.
pub struct Index<'a> {
    inner: KnnIndex<'a>,
}

impl Index<'_> {
    /// k nearest POIs to `src` by graph distance (pruned branch-and-bound).
    #[must_use]
    pub fn query(&self, src: NodeId, k: usize) -> Vec<(NodeId, f64)> {
        let mut stats = QueryStats::default();
        self.inner.knn(src, k, true, &mut stats)
    }

    /// Same, returning search-space diagnostics alongside the result.
    #[must_use]
    pub fn query_with_stats(&self, src: NodeId, k: usize) -> (Vec<(NodeId, f64)>, QueryStats) {
        let mut stats = QueryStats::default();
        let r = self.inner.knn(src, k, true, &mut stats);
        (r, stats)
    }

    #[doc(hidden)] // exposed for the no-prune correctness oracle in tests
    #[must_use]
    pub fn query_unpruned(&self, src: NodeId, k: usize) -> Vec<(NodeId, f64)> {
        let mut stats = QueryStats::default();
        self.inner.knn(src, k, false, &mut stats)
    }
}
