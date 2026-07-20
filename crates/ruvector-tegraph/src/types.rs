/// Semantic edge types between indexed nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Both nodes belong to the same source document or memory session.
    SameDocument,
    /// Source node explicitly references or cites the target.
    References,
    /// Nodes frequently appear together in retrieval results.
    CoOccurs,
    /// Nodes are adjacent in a time series or event log.
    Temporal,
    /// Source causally depends on or triggers the target.
    Causal,
}

impl EdgeType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::SameDocument => "SameDocument",
            Self::References   => "References",
            Self::CoOccurs     => "CoOccurs",
            Self::Temporal     => "Temporal",
            Self::Causal       => "Causal",
        }
    }
}

/// A directed, typed, weighted edge to another node.
#[derive(Debug, Clone)]
pub struct TypedEdge {
    pub target: usize,
    pub edge_type: EdgeType,
    /// Relationship strength in [0.0, 1.0].
    pub weight: f32,
}

/// A node in the TENG index.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    /// Pre-normalised f32 embedding.
    pub vector: Vec<f32>,
    /// Outgoing typed edges.
    pub typed_edges: Vec<TypedEdge>,
    /// Document cluster id (used for semantic recall evaluation).
    pub doc_id: usize,
}
