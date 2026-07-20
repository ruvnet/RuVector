use crate::{
    dot,
    graph::NswGraph,
    types::{EdgeType, Node},
};
use std::collections::HashSet;

/// Edge-type predicate for `search_edge_constrained`.
pub enum EdgeConstraint {
    /// Accept candidates that have any typed edge.
    Any,
    /// Accept candidates that have at least one edge of this exact type.
    Type(EdgeType),
}

/// Main TENG index: NSW graph + typed semantic edge layer.
pub struct TengIndex {
    pub nodes: Vec<Node>,
    graph: NswGraph,
    ef_search: usize,
}

impl TengIndex {
    /// Build the TENG index from a pre-constructed node list.
    pub fn build(nodes: Vec<Node>, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let mut graph = NswGraph::new(m, ef_construction, ef_search);
        graph.build(&nodes);
        TengIndex { nodes, graph, ef_search }
    }

    // ── Variant 1: VectorOnly ─────────────────────────────────────────────────

    /// Pure NSW beam search. Typed edges are indexed but never consulted.
    pub fn search_vector_only(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut out = self.graph.search(&self.nodes, query, self.ef_search.max(k));
        out.truncate(k);
        out
    }

    // ── Variant 2: EdgeExpand ─────────────────────────────────────────────────

    /// NSW search + typed-edge expansion of the initial candidate set.
    ///
    /// After collecting `ef_search` candidates via NSW, the algorithm walks
    /// typed edges outward from the top-k initial results (weighted by
    /// `edge_factor`). The full extended candidate set is then re-ranked by
    /// pure cosine similarity and the top-k are returned.
    ///
    /// This integrates graph-neighbourhood information into the retrieval
    /// pass without requiring a separate graph-traversal step.
    pub fn search_edge_expand(&self, query: &[f32], k: usize, edge_factor: f32) -> Vec<(usize, f32)> {
        let ef = (self.ef_search * 2).max(k * 3);
        let initial = self.graph.search(&self.nodes, query, ef);

        let mut seen: HashSet<usize> = initial.iter().map(|(id, _)| *id).collect();
        let mut extended: Vec<(usize, f32)> = initial;

        // Walk typed edges from top-k initial results.
        let explore_count = k.min(extended.len());
        for i in 0..explore_count {
            let (base_id, base_sim) = extended[i];
            for edge in &self.nodes[base_id].typed_edges {
                if seen.insert(edge.target) {
                    let vec_sim = dot(query, &self.nodes[edge.target].vector);
                    // Semantic blend: vector sim + edge-following bonus.
                    let blended = vec_sim + edge.weight * edge_factor * base_sim;
                    extended.push((edge.target, blended.min(1.0)));
                }
            }
        }

        // Re-rank full candidate set by pure vector similarity.
        for entry in &mut extended {
            entry.1 = dot(query, &self.nodes[entry.0].vector);
        }
        extended.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        extended.dedup_by_key(|e| e.0);
        extended.truncate(k);
        extended
    }

    // ── Variant 3: EdgeConstrained ────────────────────────────────────────────

    /// NSW search returning only candidates that satisfy an edge-type predicate.
    ///
    /// Runs NSW with a wider ef to compensate for post-hoc filtering.
    /// Useful when the caller wants results that are semantically connected
    /// (e.g. "find similar nodes that are in the same document").
    pub fn search_edge_constrained(
        &self,
        query: &[f32],
        k: usize,
        constraint: &EdgeConstraint,
    ) -> Vec<(usize, f32)> {
        // Over-fetch to compensate for filtering loss.
        let ef = (self.ef_search * 6).max(k * 12);
        let raw = self.graph.search(&self.nodes, query, ef);
        raw.into_iter()
            .filter(|(id, _)| self.node_matches(*id, constraint))
            .take(k)
            .collect()
    }

    fn node_matches(&self, id: usize, constraint: &EdgeConstraint) -> bool {
        let edges = &self.nodes[id].typed_edges;
        match constraint {
            EdgeConstraint::Any => !edges.is_empty(),
            EdgeConstraint::Type(et) => edges.iter().any(|e| &e.edge_type == et),
        }
    }

    // ── Ground truth helpers ──────────────────────────────────────────────────

    /// Brute-force k-NN for recall ground truth.
    pub fn brute_force_knn(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut sims: Vec<(usize, f32)> = self.nodes.iter()
            .map(|n| (n.id, dot(query, &n.vector)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sims.truncate(k);
        sims
    }

    /// Semantic ground truth: brute-force k-NN ∪ their typed-edge neighbours,
    /// ranked by cosine similarity to the query.
    ///
    /// Used for measuring EdgeExpand quality: it captures both vector-near and
    /// graph-adjacent nodes that a complete retrieval should cover.
    pub fn semantic_ground_truth(&self, query: &[f32], k: usize) -> Vec<usize> {
        let knn = self.brute_force_knn(query, k);
        let mut set: HashSet<usize> = knn.iter().map(|(id, _)| *id).collect();
        for (id, _) in &knn {
            for edge in &self.nodes[*id].typed_edges {
                set.insert(edge.target);
            }
        }
        let mut ranked: Vec<(usize, f32)> = set.iter()
            .map(|&id| (id, dot(query, &self.nodes[id].vector)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.into_iter().map(|(id, _)| id).collect()
    }

    /// Filtered ground truth: brute-force k-NN restricted to nodes that match
    /// the edge constraint. Used for EdgeConstrained recall measurement.
    pub fn constrained_ground_truth(
        &self,
        query: &[f32],
        k: usize,
        constraint: &EdgeConstraint,
    ) -> Vec<(usize, f32)> {
        let mut sims: Vec<(usize, f32)> = self.nodes.iter()
            .filter(|n| self.node_matches(n.id, constraint))
            .map(|n| (n.id, dot(query, &n.vector)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sims.truncate(k);
        sims
    }
}
