//! Field edges — relational plane between nodes.
//!
//! # Example
//!
//! ```
//! use ruvector_field::model::{FieldEdge, EdgeKind, NodeId};
//! let e = FieldEdge::new(NodeId(1), NodeId(2), EdgeKind::Supports, 0.9, 100);
//! assert_eq!(e.src, NodeId(1));
//! ```

use super::NodeId;

/// Relation type between two field nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// `src` supports / reinforces `dst`.
    Supports,
    /// Explicit contradiction / opposition.
    Contrasts,
    /// `src` refines `dst` into a tighter statement.
    Refines,
    /// Routing hint target.
    RoutesTo,
    /// `src` was derived from `dst` during summarization / promotion.
    DerivedFrom,
    /// Structural adjacency — same partition / locality.
    SharesRegion,
    /// Explicit witness binding.
    BindsWitness,
}

/// Directed, weighted, timestamped edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldEdge {
    /// Source node id.
    pub src: NodeId,
    /// Destination node id.
    pub dst: NodeId,
    /// Kind of relation.
    pub kind: EdgeKind,
    /// Weight in `[0, 1]`.
    pub weight: f32,
    /// Timestamp in nanoseconds.
    pub ts_ns: u64,
}

impl FieldEdge {
    /// Construct a new edge.
    pub fn new(src: NodeId, dst: NodeId, kind: EdgeKind, weight: f32, ts_ns: u64) -> Self {
        Self {
            src,
            dst,
            kind,
            weight: weight.clamp(0.0, 1.0),
            ts_ns,
        }
    }
}
