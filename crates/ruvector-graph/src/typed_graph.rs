//! Schema-validated graph wrapper with a fused vector-search-then-traverse
//! operator (HelixDB-inspired, ADR-252 P2).
//!
//! `TypedGraph` wraps a [`GraphDB`] with an optional [`GraphSchema`]. Mutations
//! are validated against the schema before they touch storage, and the
//! [`TypedGraph::search_then_traverse`] method expresses HelixQL's
//! `SearchV<T>(q, k)::In<Edge>` pattern as a single fused operation: an ANN
//! vector search over a bound node label, immediately traversed into the graph.
//!
//! The vector step has two backends:
//!
//! - **Brute-force** (default): an optimized bounded-top-k scan over the bound
//!   label's nodes — zero-copy borrow scoring, fused cosine, O(n log k) heap,
//!   rayon-parallel above a threshold. Exact, no build step.
//! - **HNSW push-down** (opt-in via [`TypedGraph::build_vector_index`]): an
//!   ANN index ([`HybridIndex`]) per vector type. `search_then_traverse` then
//!   does an ~O(log n) approximate search, **over-fetches**, and **rescores the
//!   candidates exactly** with the schema metric — so results carry identical
//!   higher-is-better score semantics to the brute-force path while skipping the
//!   full-label scan.

use crate::bm25::{Bm25Index, Bm25Params};
use crate::edge::Edge;
use crate::embed::Embedder;
use crate::error::{GraphError, Result};
use crate::graph::GraphDB;
use crate::hybrid::{EmbeddingConfig, HybridIndex, VectorIndexType};
use crate::node::Node;
use crate::schema::{
    extract_vector, reciprocal_rank_fusion, score_property, DistanceMetric, GraphSchema,
    VectorSchema,
};
use crate::types::{NodeId, PropertyValue};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

/// Below this candidate count the serial scan wins (rayon fork/join overhead
/// exceeds the work). Above it, the parallel path engages.
const PARALLEL_SCAN_THRESHOLD: usize = 4_096;

/// Map a schema metric to the core index metric.
fn to_core_metric(m: DistanceMetric) -> ruvector_core::types::DistanceMetric {
    use ruvector_core::types::DistanceMetric as C;
    match m {
        DistanceMetric::Cosine => C::Cosine,
        DistanceMetric::DotProduct => C::DotProduct,
        DistanceMetric::Euclidean => C::Euclidean,
    }
}

type ScoredHeap = BinaryHeap<Reverse<(OrderedFloat<f32>, NodeId)>>;

/// Keep only the top-`k` largest-scored entries in `heap`.
#[inline]
fn trim_to_k(heap: &mut ScoredHeap, k: usize) {
    while heap.len() > k {
        heap.pop(); // Reverse min-heap: pop() drops the smallest score.
    }
}

/// Offer `(score, id)` to a bounded top-`k` heap, cloning `id` only if it wins a
/// slot (avoids an allocation for the common losing candidate).
#[inline]
fn consider(heap: &mut ScoredHeap, k: usize, score: f32, id: &NodeId) {
    if heap.len() < k {
        heap.push(Reverse((OrderedFloat(score), id.clone())));
    } else if let Some(Reverse((min, _))) = heap.peek() {
        if OrderedFloat(score) > *min {
            heap.pop();
            heap.push(Reverse((OrderedFloat(score), id.clone())));
        }
    }
}

/// Traversal direction relative to the matched seed node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Follow edges where the seed is the `from` endpoint (HelixQL `::Out`).
    Out,
    /// Follow edges where the seed is the `to` endpoint (HelixQL `::In`).
    In,
    /// Both directions.
    Both,
}

/// Which edge type to follow, in which direction, optionally filtering targets.
#[derive(Debug, Clone)]
pub struct TraverseSpec {
    pub edge_type: String,
    pub direction: Direction,
    /// If set, only keep target nodes carrying this label.
    pub target_label: Option<String>,
}

impl TraverseSpec {
    pub fn out(edge_type: impl Into<String>) -> Self {
        Self {
            edge_type: edge_type.into(),
            direction: Direction::Out,
            target_label: None,
        }
    }
    pub fn incoming(edge_type: impl Into<String>) -> Self {
        Self {
            edge_type: edge_type.into(),
            direction: Direction::In,
            target_label: None,
        }
    }
    pub fn both(edge_type: impl Into<String>) -> Self {
        Self {
            edge_type: edge_type.into(),
            direction: Direction::Both,
            target_label: None,
        }
    }
    pub fn target_label(mut self, label: impl Into<String>) -> Self {
        self.target_label = Some(label.into());
        self
    }
}

/// A single vector-search hit (seed node) and the nodes reached from it.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub seed_id: NodeId,
    pub score: f32,
    pub connected: Vec<Node>,
}

/// A graph wrapped with an optional, validated schema.
pub struct TypedGraph {
    graph: GraphDB,
    schema: GraphSchema,
    /// Optional ANN index per vector-type name (HNSW push-down). Each index holds
    /// only the bound label's nodes, so searches are naturally label-scoped.
    indexes: HashMap<String, HybridIndex>,
    /// Optional BM25 keyword index per `"label::property"` (snapshot-built).
    text_indexes: HashMap<String, Bm25Index>,
    /// Optional text embedder for inline `embed()` at insert and query.
    embedder: Option<Arc<dyn Embedder>>,
}

impl TypedGraph {
    /// Wrap a graph with a schema. The schema's internal consistency is checked
    /// up front (the HelixQL compile-time check).
    pub fn new(graph: GraphDB, schema: GraphSchema) -> Result<Self> {
        schema.validate_self()?;
        Ok(Self {
            graph,
            schema,
            indexes: HashMap::new(),
            text_indexes: HashMap::new(),
            embedder: None,
        })
    }

    /// Attach a text embedder, enabling inline `embed()` at insert and query
    /// (HelixQL `Embed()`). Builder-style.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Embed `text` with the attached embedder. Errors explicitly if none is
    /// attached — the typed graph never silently falls back (ADR-194).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let e = self.embedder.as_ref().ok_or_else(|| {
            GraphError::SchemaViolation("no embedder attached (call with_embedder)".into())
        })?;
        e.embed(text)
    }

    /// Create a node, embedding `text` into the vector type's bound property
    /// inline (HelixQL `AddN<T>({ embedding: Embed(text) })`). The embedding
    /// dimension is validated against the vector type before the write.
    pub fn create_node_from_text(
        &self,
        mut node: Node,
        vector_type: &str,
        text: &str,
    ) -> Result<NodeId> {
        let vs = self.schema.vector(vector_type).ok_or_else(|| {
            GraphError::SchemaViolation(format!("unknown vector type '{vector_type}'"))
        })?;
        let property = vs.property.clone();
        let dims = vs.dimensions;
        let emb = self.embed(text)?;
        if emb.len() != dims {
            return Err(GraphError::SchemaViolation(format!(
                "embedder produced dimension {} but vector type '{}' expects {}",
                emb.len(),
                vector_type,
                dims
            )));
        }
        node.set_property(property, PropertyValue::FloatArray(emb));
        self.create_node(node)
    }

    /// Search by text: embed the query inline, then run the typed
    /// search-then-traverse (HelixQL `SearchV<T>(Embed(text), k)::...`).
    pub fn search_text(
        &self,
        vector_type: &str,
        text: &str,
        k: usize,
        traverse: &TraverseSpec,
    ) -> Result<Vec<TraversalResult>> {
        let query = self.embed(text)?;
        self.search_then_traverse(vector_type, &query, k, traverse)
    }

    pub fn schema(&self) -> &GraphSchema {
        &self.schema
    }
    pub fn graph(&self) -> &GraphDB {
        &self.graph
    }
    /// Escape hatch to the underlying graph for unvalidated/advanced use.
    pub fn graph_mut(&mut self) -> &mut GraphDB {
        &mut self.graph
    }

    /// Build (or rebuild) an ANN index for a declared vector type, populating it
    /// from the nodes currently carrying that embedding. Subsequent
    /// `create_node` calls keep the index current incrementally. Returns the
    /// number of vectors indexed. Opting in switches `search_then_traverse` for
    /// this vector type onto the HNSW push-down path.
    pub fn build_vector_index(&mut self, vector_type: &str) -> Result<usize> {
        let vs = self
            .schema
            .vector(vector_type)
            .ok_or_else(|| {
                GraphError::SchemaViolation(format!("unknown vector type '{vector_type}'"))
            })?
            .clone();

        let config = EmbeddingConfig {
            dimensions: vs.dimensions,
            metric: to_core_metric(vs.metric),
            embedding_property: vs.property.clone(),
            ..Default::default()
        };
        let index = HybridIndex::new(config)?;
        index.initialize_index(VectorIndexType::Node)?;

        let mut count = 0usize;
        for id in self.graph.node_ids_by_label(&vs.label) {
            let emb = self
                .graph
                .with_node(&id, |n| {
                    n.properties.get(&vs.property).and_then(extract_vector)
                })
                .flatten();
            if let Some(emb) = emb {
                if emb.len() == vs.dimensions {
                    index.add_node_embedding(id, emb)?;
                    count += 1;
                }
            }
        }
        self.indexes.insert(vector_type.to_string(), index);
        Ok(count)
    }

    /// Whether an ANN index is built for `vector_type`.
    pub fn has_vector_index(&self, vector_type: &str) -> bool {
        self.indexes.contains_key(vector_type)
    }

    /// Incrementally add a node to any ANN index whose vector type it matches.
    fn index_node(&self, node: &Node) {
        for (name, index) in &self.indexes {
            let Some(vs) = self.schema.vector(name) else {
                continue;
            };
            if !node.has_label(&vs.label) {
                continue;
            }
            if let Some(emb) = node.properties.get(&vs.property).and_then(extract_vector) {
                if emb.len() == vs.dimensions {
                    // Best-effort: a failed index add must not fail the write.
                    let _ = index.add_node_embedding(node.id.clone(), emb);
                }
            }
        }
    }

    /// Validate a node against the schema, then create it (updating ANN indexes).
    pub fn create_node(&self, node: Node) -> Result<NodeId> {
        self.schema.validate_node(&node)?;
        if !self.indexes.is_empty() {
            self.index_node(&node);
        }
        self.graph.create_node(node)
    }

    /// Validate an edge — including its endpoints' labels — then create it.
    pub fn create_edge(&self, edge: Edge) -> Result<crate::types::EdgeId> {
        let from = self.graph.get_node(&edge.from).ok_or_else(|| {
            GraphError::SchemaViolation(format!("edge from-node '{}' does not exist", edge.from))
        })?;
        let to = self.graph.get_node(&edge.to).ok_or_else(|| {
            GraphError::SchemaViolation(format!("edge to-node '{}' does not exist", edge.to))
        })?;
        let from_labels: Vec<String> = from.labels.iter().map(|l| l.name.clone()).collect();
        let to_labels: Vec<String> = to.labels.iter().map(|l| l.name.clone()).collect();
        self.schema.validate_edge(&edge, &from_labels, &to_labels)?;
        self.graph.create_edge(edge)
    }

    /// Fused vector-search-then-traverse (HelixQL `SearchV<T>(q,k)::In/Out<E>`).
    ///
    /// 1. Resolve `vector_type` to its bound label + property + metric (typed —
    ///    no string/property guessing).
    /// 2. Validate the query dimension.
    /// 3. Find the top-`k` seeds: via the ANN index if one is built for this
    ///    vector type ([`TypedGraph::build_vector_index`]), else a bounded-top-k
    ///    scan. Either way, seeds carry an exact higher-is-better score.
    /// 4. Traverse from each seed along `traverse` and collect target nodes.
    pub fn search_then_traverse(
        &self,
        vector_type: &str,
        query: &[f32],
        k: usize,
        traverse: &TraverseSpec,
    ) -> Result<Vec<TraversalResult>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let vs = self.schema.validate_vector_dims(vector_type, query)?;
        let hits = self.rank_seeds(vs, query, k)?;
        Ok(self.expand(hits, traverse))
    }

    /// Rank the top-`k` seeds for a vector type: ANN index if built, else the
    /// optimized brute-force scan. Returns `(score, id)` descending.
    fn rank_seeds(&self, vs: &VectorSchema, query: &[f32], k: usize) -> Result<Vec<(f32, NodeId)>> {
        let metric = vs.metric;
        let property = vs.property.as_str();
        let query_norm = metric.query_norm(query);
        Ok(match self.indexes.get(&vs.name) {
            Some(index) => self.rank_via_index(index, property, query, query_norm, metric, k)?,
            None => self.rank_via_scan(&vs.label, property, query, query_norm, metric, k),
        })
    }

    /// Traverse from each ranked seed and assemble results.
    fn expand(&self, hits: Vec<(f32, NodeId)>, traverse: &TraverseSpec) -> Vec<TraversalResult> {
        let mut out = Vec::with_capacity(hits.len());
        for (score, seed_id) in hits {
            let connected = self.traverse_from(&seed_id, traverse);
            out.push(TraversalResult {
                seed_id,
                score,
                connected,
            });
        }
        out
    }

    /// Build (snapshot) a BM25 keyword index over a string property of `label`.
    /// Rebuild to reflect later writes. Returns the number of documents indexed.
    pub fn build_text_index(&mut self, label: &str, text_property: &str) -> Result<usize> {
        let mut docs: Vec<(NodeId, String)> = Vec::new();
        for id in self.graph.node_ids_by_label(label) {
            let text = self
                .graph
                .with_node(&id, |n| match n.properties.get(text_property) {
                    Some(PropertyValue::String(s)) => Some(s.clone()),
                    _ => None,
                })
                .flatten();
            if let Some(text) = text {
                docs.push((id, text));
            }
        }
        let count = docs.len();
        let key = format!("{label}::{text_property}");
        self.text_indexes
            .insert(key, Bm25Index::build(docs, Bm25Params::default()));
        Ok(count)
    }

    /// Whether a BM25 index is built for `label::text_property`.
    pub fn has_text_index(&self, label: &str, text_property: &str) -> bool {
        self.text_indexes
            .contains_key(&format!("{label}::{text_property}"))
    }

    /// **Tri-modal hybrid query** (ADR-252 P4): fuse ANN vector similarity, BM25
    /// keyword relevance, and graph traversal in a single typed call.
    ///
    /// The query `text` is embedded (inline `embed()`) for the vector arm and
    /// tokenized for the BM25 arm; the two rankings are fused with Reciprocal
    /// Rank Fusion (`rrf_k`, conventionally 60), and the fused top-`k` seeds are
    /// traversed. Requires both an embedder and a BM25 index
    /// ([`TypedGraph::build_text_index`]); an ANN index is optional (the vector
    /// arm falls back to the exact scan). Result `score` is the fused RRF score.
    pub fn hybrid_search_text(
        &self,
        vector_type: &str,
        text_property: &str,
        text: &str,
        k: usize,
        rrf_k: f32,
        traverse: &TraverseSpec,
    ) -> Result<Vec<TraversalResult>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let vs = self.schema.vector(vector_type).ok_or_else(|| {
            GraphError::SchemaViolation(format!("unknown vector type '{vector_type}'"))
        })?;

        // Over-fetch each arm so fusion has depth to work with.
        let over = k.saturating_mul(4).max(k + 32);

        // Vector arm: embed inline, dimension-check, rank.
        let qvec = self.embed(text)?;
        if qvec.len() != vs.dimensions {
            return Err(GraphError::SchemaViolation(format!(
                "embedder produced dimension {} but vector type '{}' expects {}",
                qvec.len(),
                vector_type,
                vs.dimensions
            )));
        }
        let vec_hits = self.rank_seeds(vs, &qvec, over)?;

        // Keyword arm: BM25 over the text property of the bound label.
        let key = format!("{}::{}", vs.label, text_property);
        let bm = self.text_indexes.get(&key).ok_or_else(|| {
            GraphError::SchemaViolation(format!(
                "BM25 index '{key}' not built (call build_text_index)"
            ))
        })?;
        let kw_hits = bm.search(text, over);

        // Fuse the two rank lists with RRF, then traverse the fused top-k.
        let vec_ids: Vec<NodeId> = vec_hits.into_iter().map(|(_, id)| id).collect();
        let kw_ids: Vec<NodeId> = kw_hits.into_iter().map(|(id, _)| id).collect();
        let fused = reciprocal_rank_fusion(&[vec_ids, kw_ids], rrf_k);
        let hits: Vec<(f32, NodeId)> = fused.into_iter().take(k).map(|(id, s)| (s, id)).collect();
        Ok(self.expand(hits, traverse))
    }

    /// HNSW push-down: approximate ANN search, over-fetch, then rescore the
    /// candidates exactly so the returned scores match the brute-force path.
    fn rank_via_index(
        &self,
        index: &HybridIndex,
        property: &str,
        query: &[f32],
        query_norm: f32,
        metric: DistanceMetric,
        k: usize,
    ) -> Result<Vec<(f32, NodeId)>> {
        // Over-fetch to recover HNSW approximation, then rerank exactly.
        let over = k.saturating_mul(4).max(k + 32);
        let candidates = index.search_similar_nodes(query, over)?;
        let mut scored: Vec<(f32, NodeId)> = candidates
            .into_iter()
            .filter_map(|(id, _approx)| {
                let s = self
                    .graph
                    .with_node(&id, |node| {
                        node.properties
                            .get(property)
                            .and_then(|p| score_property(metric, query, query_norm, p))
                    })
                    .flatten()?;
                Some((s, id))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    /// Brute-force bounded-top-k scan over the bound label, returning seeds in
    /// descending score order. Rayon-parallel above the threshold.
    fn rank_via_scan(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        query_norm: f32,
        metric: DistanceMetric,
        k: usize,
    ) -> Vec<(f32, NodeId)> {
        let ids = self.graph.node_ids_by_label(label);
        // Capture `graph` (not `self`) so the parallel closure stays Send+Sync
        // regardless of the ANN index's thread-safety bounds.
        let graph = &self.graph;
        let score_one = |id: &NodeId| -> Option<f32> {
            graph
                .with_node(id, |node| {
                    node.properties
                        .get(property)
                        .and_then(|prop| score_property(metric, query, query_norm, prop))
                })
                .flatten()
        };

        // Bounded top-k via a min-heap: O(n log k). DashMap allows concurrent
        // reads, so for large candidate sets we fan the scan across cores with
        // per-thread heaps and a bounded merge.
        let heap: ScoredHeap = if ids.len() >= PARALLEL_SCAN_THRESHOLD {
            ids.par_iter()
                .fold(ScoredHeap::new, |mut h, id| {
                    if let Some(score) = score_one(id) {
                        consider(&mut h, k, score, id);
                    }
                    h
                })
                .reduce(ScoredHeap::new, |mut a, b| {
                    a.extend(b);
                    trim_to_k(&mut a, k);
                    a
                })
        } else {
            let mut h = ScoredHeap::new();
            for id in &ids {
                if let Some(score) = score_one(id) {
                    consider(&mut h, k, score, id);
                }
            }
            h
        };

        let mut hits: Vec<(f32, NodeId)> = heap
            .into_iter()
            .map(|Reverse((s, id))| (s.into_inner(), id))
            .collect();
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// Collect nodes reachable from `seed` along the traversal spec.
    fn traverse_from(&self, seed: &NodeId, spec: &TraverseSpec) -> Vec<Node> {
        let mut targets: Vec<NodeId> = Vec::new();
        if matches!(spec.direction, Direction::Out | Direction::Both) {
            for e in self.graph.get_outgoing_edges(seed) {
                if e.edge_type == spec.edge_type {
                    targets.push(e.to);
                }
            }
        }
        if matches!(spec.direction, Direction::In | Direction::Both) {
            for e in self.graph.get_incoming_edges(seed) {
                if e.edge_type == spec.edge_type {
                    targets.push(e.from);
                }
            }
        }

        let mut nodes = Vec::with_capacity(targets.len());
        let mut seen = std::collections::HashSet::new();
        for id in targets {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Some(node) = self.graph.get_node(&id) {
                if let Some(label) = &spec.target_label {
                    if !node.has_label(label) {
                        continue;
                    }
                }
                nodes.push(node);
            }
        }
        nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeBuilder;
    use crate::schema::{
        DistanceMetric, EdgeSchema, NodeSchema, PropertySchema, PropertyType, VectorSchema,
    };
    use crate::types::PropertyValue;

    fn schema() -> GraphSchema {
        let mut s = GraphSchema::new();
        s.add_node(
            NodeSchema::new("Doc")
                .property(PropertySchema::new("title", PropertyType::String).required())
                .property(PropertySchema::new("embedding", PropertyType::Vector)),
        );
        s.add_node(
            NodeSchema::new("Topic").property(PropertySchema::new("name", PropertyType::String)),
        );
        s.add_edge(EdgeSchema::new("ABOUT", "Doc", "Topic"));
        s.add_vector(VectorSchema::new(
            "DocEmb",
            "Doc",
            "embedding",
            3,
            DistanceMetric::Cosine,
        ));
        s
    }

    fn doc(id: &str, title: &str, emb: Vec<f32>) -> Node {
        NodeBuilder::new()
            .id(id)
            .label("Doc")
            .property("title", title)
            .property("embedding", PropertyValue::FloatArray(emb))
            .build()
    }

    #[test]
    fn rejects_invalid_node_and_edge() {
        let tg = TypedGraph::new(GraphDB::new(), schema()).unwrap();
        // Missing required `title`.
        let bad = NodeBuilder::new().id("d0").label("Doc").build();
        assert!(tg.create_node(bad).is_err());

        tg.create_node(doc("d1", "a", vec![1.0, 0.0, 0.0])).unwrap();
        let topic = NodeBuilder::new()
            .id("t1")
            .label("Topic")
            .property("name", "ai")
            .build();
        tg.create_node(topic).unwrap();
        // Wrong direction: Topic -> Doc on an ABOUT edge declared Doc -> Topic.
        let bad_edge = Edge::create("t1".into(), "d1".into(), "ABOUT");
        assert!(tg.create_edge(bad_edge).is_err());
        // Correct direction.
        let good_edge = Edge::create("d1".into(), "t1".into(), "ABOUT");
        assert!(tg.create_edge(good_edge).is_ok());
    }

    #[test]
    fn search_then_traverse_ranks_and_expands() {
        let tg = TypedGraph::new(GraphDB::new(), schema()).unwrap();
        tg.create_node(doc("d1", "near", vec![1.0, 0.0, 0.0]))
            .unwrap();
        tg.create_node(doc("d2", "mid", vec![0.7, 0.7, 0.0]))
            .unwrap();
        tg.create_node(doc("d3", "far", vec![0.0, 0.0, 1.0]))
            .unwrap();
        for t in ["ai", "ml", "db"] {
            tg.create_node(
                NodeBuilder::new()
                    .id(t)
                    .label("Topic")
                    .property("name", t)
                    .build(),
            )
            .unwrap();
        }
        tg.create_edge(Edge::create("d1".into(), "ai".into(), "ABOUT"))
            .unwrap();
        tg.create_edge(Edge::create("d1".into(), "ml".into(), "ABOUT"))
            .unwrap();
        tg.create_edge(Edge::create("d2".into(), "db".into(), "ABOUT"))
            .unwrap();

        let q = [1.0f32, 0.0, 0.0];
        let res = tg
            .search_then_traverse(
                "DocEmb",
                &q,
                2,
                &TraverseSpec::out("ABOUT").target_label("Topic"),
            )
            .unwrap();

        // Top-k respected and ordered by similarity.
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].seed_id, "d1");
        assert!(res[0].score >= res[1].score);
        // d1 expands to its two topics.
        let topics: Vec<&str> = res[0].connected.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(topics.len(), 2);
        assert!(topics.contains(&"ai") && topics.contains(&"ml"));
    }

    #[test]
    fn search_then_traverse_validates_dimension() {
        let tg = TypedGraph::new(GraphDB::new(), schema()).unwrap();
        let err = tg.search_then_traverse("DocEmb", &[1.0, 2.0], 1, &TraverseSpec::out("ABOUT"));
        assert!(err.is_err());
    }

    #[test]
    fn parallel_scan_matches_reference() {
        // Exceed PARALLEL_SCAN_THRESHOLD so the rayon path runs, and check the
        // top-k it returns equals an independent brute-force ranking.
        let tg = TypedGraph::new(GraphDB::new(), schema()).unwrap();
        let n = 5000usize;
        let mut embs: Vec<(String, Vec<f32>)> = Vec::with_capacity(n);
        for i in 0..n {
            // Deterministic spread across the unit-ish cube.
            let v = vec![
                ((i * 7) % 100) as f32 / 100.0,
                ((i * 13) % 100) as f32 / 100.0,
                ((i * 29) % 100) as f32 / 100.0,
            ];
            let id = format!("d{i}");
            tg.create_node(doc(&id, "t", v.clone())).unwrap();
            embs.push((id, v));
        }
        let q = [1.0f32, 0.0, 0.0];
        let k = 10;
        let res = tg
            .search_then_traverse("DocEmb", &q, k, &TraverseSpec::out("ABOUT"))
            .unwrap();

        // Reference: cosine score, sort desc, take k ids.
        let qn = (q[0] * q[0]) as f32;
        let qn = qn.sqrt();
        let mut reference: Vec<(f32, String)> = embs
            .iter()
            .map(|(id, v)| {
                let dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
                let vn: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let s = if qn == 0.0 || vn == 0.0 {
                    0.0
                } else {
                    dot / (qn * vn)
                };
                (s, id.clone())
            })
            .collect();
        reference.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        assert_eq!(res.len(), k);
        for w in res.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        // Ties (identical cosine scores) make exact id order ambiguous, so compare
        // the top-k *scores* against the reference — these must match exactly.
        for (got, want) in res.iter().zip(reference.iter()) {
            assert!(
                (got.score - want.0).abs() < 1e-5,
                "score mismatch: {} vs {}",
                got.score,
                want.0
            );
        }
    }

    #[test]
    fn topk_bounded_heap_returns_exactly_k() {
        let tg = TypedGraph::new(GraphDB::new(), schema()).unwrap();
        for i in 0..50 {
            let v = vec![i as f32 / 50.0, 1.0 - i as f32 / 50.0, 0.0];
            tg.create_node(doc(&format!("d{i}"), "t", v)).unwrap();
        }
        let res = tg
            .search_then_traverse("DocEmb", &[1.0, 0.0, 0.0], 5, &TraverseSpec::out("ABOUT"))
            .unwrap();
        assert_eq!(res.len(), 5);
        // Strictly non-increasing scores.
        for w in res.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn indexed_path_finds_top_result_and_traverses() {
        // HNSW push-down: build an ANN index and confirm it returns the exact
        // winner (over-fetch + exact rescore) with traversal still applied.
        let mut tg = TypedGraph::new(GraphDB::new(), schema()).unwrap();
        // Data on an arc of the unit circle so the nearest neighbour is on the
        // manifold (the realistic, HNSW-friendly case). `winner` sits at angle 0,
        // exactly aligned with the query; the rest fan out to larger angles.
        tg.create_node(doc("winner", "t", vec![1.0, 0.0, 0.0]))
            .unwrap();
        for i in 0..300 {
            let a = 0.2 + (i as f32 / 300.0) * 1.2; // 0.2 .. 1.4 rad
            tg.create_node(doc(&format!("d{i}"), "t", vec![a.cos(), a.sin(), 0.0]))
                .unwrap();
        }
        tg.create_node(
            NodeBuilder::new()
                .id("ai")
                .label("Topic")
                .property("name", "ai")
                .build(),
        )
        .unwrap();
        tg.create_edge(Edge::create("winner".into(), "ai".into(), "ABOUT"))
            .unwrap();

        let built = tg.build_vector_index("DocEmb").unwrap();
        assert_eq!(built, 301);
        assert!(tg.has_vector_index("DocEmb"));

        let res = tg
            .search_then_traverse(
                "DocEmb",
                &[1.0, 0.0, 0.0],
                5,
                &TraverseSpec::out("ABOUT").target_label("Topic"),
            )
            .unwrap();
        assert_eq!(res.len(), 5);
        assert_eq!(res[0].seed_id, "winner");
        assert!((res[0].score - 1.0).abs() < 1e-5);
        for w in res.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        // Traversal still expands the seed.
        assert_eq!(res[0].connected.iter().filter(|n| n.id == "ai").count(), 1);
    }

    #[test]
    fn embed_at_insert_and_query_roundtrip() {
        use crate::embed::HashEmbedder;
        // Schema vector dim must match the embedder dim.
        let mut s = GraphSchema::new();
        s.add_node(
            NodeSchema::new("Doc")
                .property(PropertySchema::new("title", PropertyType::String).required())
                .property(PropertySchema::new("embedding", PropertyType::Vector)),
        );
        s.add_node(NodeSchema::new("Topic"));
        s.add_edge(EdgeSchema::new("ABOUT", "Doc", "Topic"));
        s.add_vector(VectorSchema::new(
            "DocEmb",
            "Doc",
            "embedding",
            128,
            DistanceMetric::Cosine,
        ));
        let tg = TypedGraph::new(GraphDB::new(), s)
            .unwrap()
            .with_embedder(Arc::new(HashEmbedder::new(128)));

        // Inline embed at insert — no caller-supplied vector.
        for (id, text) in [
            ("d1", "machine learning vector database"),
            ("d2", "distributed systems consensus raft"),
            ("d3", "italian pasta cooking recipe"),
        ] {
            let node = NodeBuilder::new()
                .id(id)
                .label("Doc")
                .property("title", text)
                .build();
            tg.create_node_from_text(node, "DocEmb", text).unwrap();
        }

        // Query by text — closest doc to a lexically-overlapping query wins.
        let res = tg
            .search_text(
                "DocEmb",
                "vector database machine learning",
                1,
                &TraverseSpec::out("ABOUT"),
            )
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].seed_id, "d1");
    }

    #[test]
    fn tri_modal_hybrid_fuses_vector_keyword_graph() {
        use crate::embed::HashEmbedder;
        let mut s = GraphSchema::new();
        s.add_node(
            NodeSchema::new("Doc")
                .property(PropertySchema::new("body", PropertyType::String).required())
                .property(PropertySchema::new("embedding", PropertyType::Vector)),
        );
        s.add_node(NodeSchema::new("Topic"));
        s.add_edge(EdgeSchema::new("ABOUT", "Doc", "Topic"));
        s.add_vector(VectorSchema::new(
            "DocEmb",
            "Doc",
            "embedding",
            256,
            DistanceMetric::Cosine,
        ));
        let mut tg = TypedGraph::new(GraphDB::new(), s)
            .unwrap()
            .with_embedder(Arc::new(HashEmbedder::new(256)));

        let docs = [
            ("d1", "vector database for semantic similarity search"),
            ("d2", "graph traversal and relationship queries"),
            ("d3", "machine learning embedding models"),
            ("d4", "italian cooking pasta tomato recipe"),
            ("d5", "approximate nearest neighbour vector search index"),
        ];
        for (id, body) in docs {
            let node = NodeBuilder::new()
                .id(id)
                .label("Doc")
                .property("body", body)
                .build();
            tg.create_node_from_text(node, "DocEmb", body).unwrap();
        }
        // Topic + edge so traversal does work.
        tg.create_node(NodeBuilder::new().id("t-search").label("Topic").build())
            .unwrap();
        tg.create_edge(Edge::create("d1".into(), "t-search".into(), "ABOUT"))
            .unwrap();

        tg.build_text_index("Doc", "body").unwrap();
        assert!(tg.has_text_index("Doc", "body"));

        let res = tg
            .hybrid_search_text(
                "DocEmb",
                "body",
                "vector search",
                3,
                60.0,
                &TraverseSpec::out("ABOUT").target_label("Topic"),
            )
            .unwrap();

        assert_eq!(res.len(), 3);
        // The fused top results must be the vector-search docs, not the recipe.
        let top_ids: Vec<&str> = res.iter().map(|r| r.seed_id.as_str()).collect();
        assert!(top_ids.contains(&"d1") || top_ids.contains(&"d5"));
        assert!(!top_ids.contains(&"d4"));
        // RRF scores are descending.
        for w in res.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        // Traversal applied for d1 if present.
        if let Some(r) = res.iter().find(|r| r.seed_id == "d1") {
            assert_eq!(r.connected.iter().filter(|n| n.id == "t-search").count(), 1);
        }
    }

    #[test]
    fn hybrid_requires_text_index() {
        use crate::embed::HashEmbedder;
        let tg = TypedGraph::new(GraphDB::new(), schema())
            .unwrap()
            .with_embedder(Arc::new(HashEmbedder::new(3)));
        // No build_text_index → explicit error, no silent degradation.
        let err =
            tg.hybrid_search_text("DocEmb", "title", "x", 1, 60.0, &TraverseSpec::out("ABOUT"));
        assert!(err.is_err());
    }

    #[test]
    fn embed_without_embedder_errors() {
        let tg = TypedGraph::new(GraphDB::new(), schema()).unwrap();
        assert!(tg.embed("anything").is_err());
        assert!(tg
            .search_text("DocEmb", "x", 1, &TraverseSpec::out("ABOUT"))
            .is_err());
    }

    #[test]
    fn embed_dimension_mismatch_is_rejected() {
        use crate::embed::HashEmbedder;
        // Embedder dim (64) != schema vector dim (3 from `schema()`).
        let tg = TypedGraph::new(GraphDB::new(), schema())
            .unwrap()
            .with_embedder(Arc::new(HashEmbedder::new(64)));
        let node = NodeBuilder::new()
            .id("d1")
            .label("Doc")
            .property("title", "x")
            .build();
        assert!(tg
            .create_node_from_text(node, "DocEmb", "some text")
            .is_err());
    }

    #[test]
    fn index_incrementally_picks_up_new_nodes() {
        let mut tg = TypedGraph::new(GraphDB::new(), schema()).unwrap();
        tg.create_node(doc("a", "t", vec![0.0, 1.0, 0.0])).unwrap();
        tg.build_vector_index("DocEmb").unwrap();
        // Inserted *after* the index is built — must be indexed incrementally.
        tg.create_node(doc("b", "t", vec![1.0, 0.0, 0.0])).unwrap();
        let res = tg
            .search_then_traverse("DocEmb", &[1.0, 0.0, 0.0], 1, &TraverseSpec::out("ABOUT"))
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].seed_id, "b");
    }
}
