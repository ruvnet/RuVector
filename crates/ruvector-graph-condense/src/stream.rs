//! Streaming condensation: maintain a growing graph + features and re-condense
//! on demand.
//!
//! The 2024–2026 literature treats streaming/temporal condensation as an open
//! problem (only OpenGC and CaT/PUMA touch it, in restricted settings). This
//! crate takes a deliberately honest stance: it does **lazy re-condensation**,
//! not true incremental region surgery. Edges/features are buffered into a
//! [`DynamicGraph`]; the condensed view is rebuilt when it is read while dirty,
//! or every `rebuild_interval` mutations. The win is amortisation and a stable
//! API for edge pipelines (e.g. condensing a RuView WorldGraph as it grows),
//! not sublinear updates — that remains future work.

use crate::condense::{CondenseConfig, GraphCondenser};
use crate::error::Result;
use crate::features::NodeFeatures;
use crate::node::CondensedGraph;
use ruvector_mincut::{DynamicGraph, VertexId, Weight};

/// A mutable graph + feature store that condenses lazily.
pub struct StreamingCondenser {
    graph: DynamicGraph,
    features: NodeFeatures,
    condenser: GraphCondenser,
    cached: Option<CondensedGraph>,
    dirty: bool,
    ops_since_rebuild: usize,
    rebuild_interval: usize,
}

impl StreamingCondenser {
    /// Create a streaming condenser.
    ///
    /// `rebuild_interval` is the maximum number of mutations tolerated before
    /// [`StreamingCondenser::condensed`] forces a rebuild even if not otherwise
    /// read. Use `0` to rebuild only on explicit reads of a dirty state.
    pub fn new(
        config: CondenseConfig,
        dim: usize,
        num_classes: usize,
        rebuild_interval: usize,
    ) -> Self {
        Self {
            graph: DynamicGraph::new(),
            features: NodeFeatures::new(dim, num_classes),
            condenser: GraphCondenser::new(config),
            cached: None,
            dirty: true,
            ops_since_rebuild: 0,
            rebuild_interval,
        }
    }

    /// Number of vertices currently buffered.
    pub fn num_vertices(&self) -> usize {
        self.graph.num_vertices()
    }

    /// Number of edges currently buffered.
    pub fn num_edges(&self) -> usize {
        self.graph.num_edges()
    }

    /// Borrow the underlying graph (read-only).
    pub fn graph(&self) -> &DynamicGraph {
        &self.graph
    }

    /// Set/replace the embedding (and optional label) for a vertex. Marks the
    /// condensed view dirty.
    ///
    /// # Errors
    /// Propagates dimension validation from [`NodeFeatures`].
    pub fn upsert_feature(
        &mut self,
        vertex: VertexId,
        embedding: Vec<f32>,
        label: Option<usize>,
    ) -> Result<()> {
        self.features.set_embedding(vertex, embedding)?;
        if let Some(l) = label {
            self.features.set_label(vertex, l);
        }
        self.touch();
        Ok(())
    }

    /// Insert an edge. Both endpoints must already have features (call
    /// [`StreamingCondenser::upsert_feature`] first) for a later condense to
    /// succeed. Duplicate edges are ignored (idempotent).
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId, weight: Weight) {
        if self.graph.insert_edge(u, v, weight).is_ok() {
            self.touch();
        }
    }

    /// Update an existing edge's weight (no-op if the edge is absent).
    pub fn update_edge(&mut self, u: VertexId, v: VertexId, weight: Weight) {
        if self.graph.update_edge_weight(u, v, weight).is_ok() {
            self.touch();
        }
    }

    /// Delete an edge (no-op if absent).
    pub fn delete_edge(&mut self, u: VertexId, v: VertexId) {
        if self.graph.delete_edge(u, v).is_ok() {
            self.touch();
        }
    }

    /// Whether the cached condensed view is stale.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get the current condensed view, rebuilding if dirty (or if the rebuild
    /// interval has elapsed). Returns `None` only when the graph is empty.
    ///
    /// # Errors
    /// Propagates condensation errors (e.g. a vertex missing its feature).
    pub fn condensed(&mut self) -> Result<Option<&CondensedGraph>> {
        if self.graph.num_vertices() == 0 {
            self.cached = None;
            self.dirty = false;
            return Ok(None);
        }
        let interval_elapsed =
            self.rebuild_interval > 0 && self.ops_since_rebuild >= self.rebuild_interval;
        if self.dirty || interval_elapsed || self.cached.is_none() {
            self.rebuild()?;
        }
        Ok(self.cached.as_ref())
    }

    /// Force an immediate re-condensation regardless of dirty state.
    ///
    /// # Errors
    /// Propagates condensation errors.
    pub fn rebuild(&mut self) -> Result<()> {
        let condensed = self.condenser.condense(&self.graph, &self.features)?;
        self.cached = Some(condensed);
        self.dirty = false;
        self.ops_since_rebuild = 0;
        Ok(())
    }

    fn touch(&mut self) {
        self.dirty = true;
        self.ops_since_rebuild += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::condense::{CondenseConfig, CondenseMethod};

    fn cfg() -> CondenseConfig {
        CondenseConfig {
            method: CondenseMethod::ConnectedComponents,
            normalize_centroids: false,
        }
    }

    #[test]
    fn empty_returns_none() {
        let mut s = StreamingCondenser::new(cfg(), 1, 0, 0);
        assert!(s.condensed().unwrap().is_none());
    }

    #[test]
    fn condenses_after_growth() {
        let mut s = StreamingCondenser::new(cfg(), 1, 0, 0);
        for v in 0..4u64 {
            s.upsert_feature(v, vec![v as f32], None).unwrap();
        }
        s.insert_edge(0, 1, 1.0);
        s.insert_edge(2, 3, 1.0);
        assert!(s.is_dirty());
        let c = s.condensed().unwrap().unwrap();
        // Two components -> two super-nodes.
        assert_eq!(c.node_count(), 2);
        assert!(!s.is_dirty());
    }

    #[test]
    fn caches_until_mutated() {
        let mut s = StreamingCondenser::new(cfg(), 1, 0, 0);
        s.upsert_feature(0, vec![0.0], None).unwrap();
        s.upsert_feature(1, vec![1.0], None).unwrap();
        s.insert_edge(0, 1, 1.0);
        let n1 = s.condensed().unwrap().unwrap().node_count();
        assert_eq!(n1, 1);
        assert!(!s.is_dirty());
        // Reading again without mutation does not re-dirty.
        let _ = s.condensed().unwrap();
        assert!(!s.is_dirty());

        // A new disconnected vertex+edge splits into a second component.
        s.upsert_feature(2, vec![2.0], None).unwrap();
        s.upsert_feature(3, vec![3.0], None).unwrap();
        s.insert_edge(2, 3, 1.0);
        assert!(s.is_dirty());
        assert_eq!(s.condensed().unwrap().unwrap().node_count(), 2);
    }

    #[test]
    fn interval_forces_rebuild_path() {
        // rebuild_interval=1 exercises the interval branch; result stays correct.
        let mut s = StreamingCondenser::new(cfg(), 1, 0, 1);
        s.upsert_feature(0, vec![0.0], None).unwrap();
        s.upsert_feature(1, vec![1.0], None).unwrap();
        s.insert_edge(0, 1, 1.0);
        assert_eq!(s.condensed().unwrap().unwrap().node_count(), 1);
    }
}
