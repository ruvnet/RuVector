//! Schema-validated graph wrapper with a fused vector-search-then-traverse
//! operator (HelixDB-inspired, ADR-252 P2).
//!
//! `TypedGraph` wraps a [`GraphDB`] with an optional [`GraphSchema`]. Mutations
//! are validated against the schema before they touch storage, and the
//! [`TypedGraph::search_then_traverse`] method expresses HelixQL's
//! `SearchV<T>(q, k)::In<Edge>` pattern as a single fused operation: an ANN-style
//! vector search over a bound node label, immediately traversed into the graph.
//!
//! The vector step is a brute-force scan over the bound label's nodes, with an
//! optimized **bounded top-k heap** (O(n log k) instead of O(n log n) sort).
//! Wiring this to the HNSW/`HybridIndex` path for large graphs is ADR-252 future
//! work; the operator's *shape* (typed, single-call, push-down search before
//! join) is the deliverable here.

use crate::edge::Edge;
use crate::error::{GraphError, Result};
use crate::graph::GraphDB;
use crate::node::Node;
use crate::schema::{score_property, GraphSchema};
use crate::types::NodeId;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Below this candidate count the serial scan wins (rayon fork/join overhead
/// exceeds the work). Above it, the parallel path engages.
const PARALLEL_SCAN_THRESHOLD: usize = 4_096;

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
        Self { edge_type: edge_type.into(), direction: Direction::Out, target_label: None }
    }
    pub fn incoming(edge_type: impl Into<String>) -> Self {
        Self { edge_type: edge_type.into(), direction: Direction::In, target_label: None }
    }
    pub fn both(edge_type: impl Into<String>) -> Self {
        Self { edge_type: edge_type.into(), direction: Direction::Both, target_label: None }
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
}

impl TypedGraph {
    /// Wrap a graph with a schema. The schema's internal consistency is checked
    /// up front (the HelixQL compile-time check).
    pub fn new(graph: GraphDB, schema: GraphSchema) -> Result<Self> {
        schema.validate_self()?;
        Ok(Self { graph, schema })
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

    /// Validate a node against the schema, then create it.
    pub fn create_node(&self, node: Node) -> Result<NodeId> {
        self.schema.validate_node(&node)?;
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
    /// 3. Scan the bound label's nodes, scoring with the declared metric, keeping
    ///    the top `k` via a bounded heap.
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
        let metric = vs.metric;
        let property = vs.property.as_str();
        // Hoist the query-side norm out of the per-candidate loop (cosine).
        let query_norm = metric.query_norm(query);
        let ids = self.graph.node_ids_by_label(&vs.label);

        // Score one candidate id; `None` if missing or not vector-shaped.
        let score_one = |id: &NodeId| -> Option<f32> {
            self.graph
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

        // Drain heap into descending-score order.
        let mut hits: Vec<(f32, NodeId)> =
            heap.into_iter().map(|Reverse((s, id))| (s.into_inner(), id)).collect();
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = Vec::with_capacity(hits.len());
        for (score, seed_id) in hits {
            let connected = self.traverse_from(&seed_id, traverse);
            out.push(TraversalResult { seed_id, score, connected });
        }
        Ok(out)
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
    use crate::schema::{DistanceMetric, EdgeSchema, NodeSchema, PropertySchema, PropertyType, VectorSchema};
    use crate::types::PropertyValue;

    fn schema() -> GraphSchema {
        let mut s = GraphSchema::new();
        s.add_node(
            NodeSchema::new("Doc")
                .property(PropertySchema::new("title", PropertyType::String).required())
                .property(PropertySchema::new("embedding", PropertyType::Vector)),
        );
        s.add_node(NodeSchema::new("Topic").property(PropertySchema::new("name", PropertyType::String)));
        s.add_edge(EdgeSchema::new("ABOUT", "Doc", "Topic"));
        s.add_vector(VectorSchema::new("DocEmb", "Doc", "embedding", 3, DistanceMetric::Cosine));
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
        let topic = NodeBuilder::new().id("t1").label("Topic").property("name", "ai").build();
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
        tg.create_node(doc("d1", "near", vec![1.0, 0.0, 0.0])).unwrap();
        tg.create_node(doc("d2", "mid", vec![0.7, 0.7, 0.0])).unwrap();
        tg.create_node(doc("d3", "far", vec![0.0, 0.0, 1.0])).unwrap();
        for t in ["ai", "ml", "db"] {
            tg.create_node(NodeBuilder::new().id(t).label("Topic").property("name", t).build()).unwrap();
        }
        tg.create_edge(Edge::create("d1".into(), "ai".into(), "ABOUT")).unwrap();
        tg.create_edge(Edge::create("d1".into(), "ml".into(), "ABOUT")).unwrap();
        tg.create_edge(Edge::create("d2".into(), "db".into(), "ABOUT")).unwrap();

        let q = [1.0f32, 0.0, 0.0];
        let res = tg
            .search_then_traverse("DocEmb", &q, 2, &TraverseSpec::out("ABOUT").target_label("Topic"))
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
        let res = tg.search_then_traverse("DocEmb", &q, k, &TraverseSpec::out("ABOUT")).unwrap();

        // Reference: cosine score, sort desc, take k ids.
        let qn = (q[0] * q[0]) as f32;
        let qn = qn.sqrt();
        let mut reference: Vec<(f32, String)> = embs
            .iter()
            .map(|(id, v)| {
                let dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
                let vn: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let s = if qn == 0.0 || vn == 0.0 { 0.0 } else { dot / (qn * vn) };
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
}
