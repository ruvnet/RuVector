//! Reference graph construction from parsed declarations.
//!
//! Builds a weighted graph where nodes are declarations and edges are
//! cross-references, suitable for MinCut partitioning.

use std::collections::HashMap;
use std::sync::Arc;

use ruvector_mincut::graph::DynamicGraph;

use crate::types::Declaration;

/// A reference graph built from declarations, ready for MinCut partitioning.
pub struct ReferenceGraph {
    /// The underlying MinCut-compatible dynamic graph.
    pub graph: Arc<DynamicGraph>,
    /// Mapping from graph vertex ID to declaration index.
    pub vertex_to_decl: HashMap<u64, usize>,
    /// Mapping from declaration name to graph vertex ID.
    pub name_to_vertex: HashMap<String, u64>,
    /// The original declarations (indexed by position).
    pub declarations: Vec<Declaration>,
}

/// Build a reference graph from a list of parsed declarations.
///
/// Each declaration becomes a vertex (with ID = index + 1, since MinCut
/// uses 1-based vertex IDs). Edges are weighted by the number of times
/// one declaration references another.
pub fn build_reference_graph(declarations: Vec<Declaration>) -> ReferenceGraph {
    let graph = Arc::new(DynamicGraph::new());
    let mut vertex_to_decl = HashMap::new();
    let mut name_to_vertex = HashMap::new();

    // Create vertices (1-based IDs).
    for (i, decl) in declarations.iter().enumerate() {
        let vid = (i + 1) as u64;
        vertex_to_decl.insert(vid, i);
        name_to_vertex.insert(decl.name.clone(), vid);
    }

    // Create edges from cross-references.
    // Count reference frequency for weighting.
    let mut edge_weights: HashMap<(u64, u64), f64> = HashMap::new();

    for (i, decl) in declarations.iter().enumerate() {
        let src = (i + 1) as u64;
        for ref_name in &decl.references {
            if let Some(&tgt) = name_to_vertex.get(ref_name) {
                if src != tgt {
                    *edge_weights.entry((src, tgt)).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    // Insert edges into the graph.
    for ((src, tgt), weight) in &edge_weights {
        // Ignore insert errors (e.g., duplicate edges from bidirectional refs).
        let _ = graph.insert_edge(*src, *tgt, *weight);
    }

    ReferenceGraph {
        graph,
        vertex_to_decl,
        name_to_vertex,
        declarations,
    }
}

impl ReferenceGraph {
    /// Returns the number of nodes (declarations) in the graph.
    pub fn node_count(&self) -> usize {
        self.declarations.len()
    }

    /// Returns the number of edges (cross-references) in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edges().len()
    }

    /// Get a declaration by its graph vertex ID.
    pub fn declaration_by_vertex(&self, vid: u64) -> Option<&Declaration> {
        self.vertex_to_decl
            .get(&vid)
            .and_then(|&idx| self.declarations.get(idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DeclKind;

    fn make_decl(name: &str, refs: &[&str]) -> Declaration {
        Declaration {
            name: name.to_string(),
            kind: DeclKind::Var,
            byte_range: (0, 10),
            string_literals: vec![],
            property_accesses: vec![],
            references: refs.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_build_graph() {
        let decls = vec![
            make_decl("a", &[]),
            make_decl("b", &["a"]),
            make_decl("c", &["a", "b"]),
        ];

        let graph = build_reference_graph(decls);
        assert_eq!(graph.node_count(), 3);
        // b->a, c->a, c->b = 3 edges
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_empty_refs() {
        let decls = vec![make_decl("a", &[]), make_decl("b", &[])];
        let graph = build_reference_graph(decls);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
    }
}
