//! Ingest + edge upsert + antipode binding.

use crate::error::FieldError;
use crate::model::{
    AxisScores, EdgeKind, Embedding, FieldEdge, FieldNode, NodeId, NodeKind, Shell,
};
use crate::scoring::resonance_score;
use crate::storage::TemporalBuckets;
use crate::witness::WitnessEvent;

use super::FieldEngine;

impl FieldEngine {
    /// Ingest a raw interaction into the `Event` shell.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvector_field::prelude::*;
    /// let mut engine = FieldEngine::new();
    /// let id = engine
    ///     .ingest(NodeKind::Interaction, "user reports timeout",
    ///             Embedding::new(vec![0.9, 0.1, 0.0]),
    ///             AxisScores::new(0.7, 0.6, 0.5, 0.8), 0b0001)
    ///     .unwrap();
    /// assert!(engine.node(id).is_some());
    /// ```
    pub fn ingest(
        &mut self,
        kind: NodeKind,
        text: impl Into<String>,
        embedding: Embedding,
        axes: AxisScores,
        policy_mask: u64,
    ) -> Result<NodeId, FieldError> {
        if embedding.values.is_empty() {
            return Err(FieldError::InvalidEmbedding("empty vector"));
        }
        if embedding.values.iter().any(|v| v.is_nan()) {
            return Err(FieldError::InvalidEmbedding("NaN in vector"));
        }
        let id = self.next_node_id();
        let ts = self.now_ns();
        let antipode = embedding.geometric_antipode();
        let semantic_id = self.store.intern(embedding);
        let geometric_id = self.store.intern(antipode);
        let temporal_bucket = TemporalBuckets::bucket_for(ts);

        let mut node = FieldNode {
            id,
            kind,
            semantic_embedding: semantic_id,
            geometric_antipode: geometric_id,
            semantic_antipode: None,
            shell: Shell::Event,
            axes,
            coherence: 0.5,
            continuity: 0.5,
            resonance: 0.0,
            policy_mask,
            witness_ref: None,
            ts_ns: ts,
            temporal_bucket,
            text: text.into(),
            shell_entered_ts: ts,
            promotion_streak: 0,
            promotion_history: Vec::new(),
            selection_count: 0,
            contradiction_hits: 0,
            edges_at_last_tick: 0,
        };
        node.resonance = resonance_score(&node);
        self.nodes.insert(id, node);
        self.index.upsert(id, semantic_id, Shell::Event);
        self.temporal.insert(id, ts);
        self.witness
            .emit(WitnessEvent::FieldNodeCreated { node: id, ts_ns: ts });
        Ok(id)
    }

    /// Insert or upsert an edge between two nodes.
    pub fn add_edge(
        &mut self,
        src: NodeId,
        dst: NodeId,
        kind: EdgeKind,
        weight: f32,
    ) -> Result<(), FieldError> {
        if !self.nodes.contains_key(&src) {
            return Err(FieldError::UnknownNode(src.0));
        }
        if !self.nodes.contains_key(&dst) {
            return Err(FieldError::UnknownNode(dst.0));
        }
        let ts = self.now_ns();
        let clamped = weight.clamp(0.0, 1.0);
        self.edges.push(FieldEdge::new(src, dst, kind, clamped, ts));
        self.witness.emit(WitnessEvent::FieldEdgeUpserted {
            src,
            dst,
            kind,
            weight: clamped,
            ts_ns: ts,
        });
        Ok(())
    }

    /// Bind a semantic antipode between two nodes. Symmetric.
    pub fn bind_semantic_antipode(
        &mut self,
        a: NodeId,
        b: NodeId,
        weight: f32,
    ) -> Result<(), FieldError> {
        if !self.nodes.contains_key(&a) {
            return Err(FieldError::UnknownNode(a.0));
        }
        if !self.nodes.contains_key(&b) {
            return Err(FieldError::UnknownNode(b.0));
        }
        let clamped = weight.clamp(0.0, 1.0);
        if let Some(na) = self.nodes.get_mut(&a) {
            na.semantic_antipode = Some(b);
        }
        if let Some(nb) = self.nodes.get_mut(&b) {
            nb.semantic_antipode = Some(a);
        }
        let ts = self.now_ns();
        self.edges
            .push(FieldEdge::new(a, b, EdgeKind::Contrasts, clamped, ts));
        self.edges
            .push(FieldEdge::new(b, a, EdgeKind::Contrasts, clamped, ts));
        self.witness.emit(WitnessEvent::AntipodeBound {
            a,
            b,
            weight: clamped,
            ts_ns: ts,
        });
        Ok(())
    }
}
