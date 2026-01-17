//! Syndrome Graph Construction
//!
//! This module provides functionality to construct graphs from syndrome bitmaps
//! for quantum error correction codes, particularly surface codes.
//!
//! ## Graph Structure
//!
//! Each detector in the syndrome becomes a node in the graph, with edges
//! connecting neighboring detectors. Edge weights are derived from the
//! correlation structure of the error model.

use crate::error::{NeuralDecoderError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in the detector graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique node identifier
    pub id: usize,
    /// Row position in the surface code lattice
    pub row: usize,
    /// Column position in the surface code lattice
    pub col: usize,
    /// Whether this detector is fired (syndrome bit is 1)
    pub fired: bool,
    /// Node type (X-type or Z-type stabilizer)
    pub node_type: NodeType,
    /// Feature vector for this node
    pub features: Vec<f32>,
}

/// Type of stabilizer node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// X-type stabilizer (measures bit flips)
    XStabilizer,
    /// Z-type stabilizer (measures phase flips)
    ZStabilizer,
    /// Boundary node (virtual)
    Boundary,
}

/// An edge in the detector graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node index
    pub from: usize,
    /// Target node index
    pub to: usize,
    /// Edge weight (derived from error probability)
    pub weight: f32,
    /// Edge type
    pub edge_type: EdgeType,
}

/// Type of edge
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Horizontal edge in the lattice
    Horizontal,
    /// Vertical edge in the lattice
    Vertical,
    /// Temporal edge (between measurement rounds)
    Temporal,
    /// Boundary edge (to virtual boundary node)
    Boundary,
}

/// The detector graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorGraph {
    /// All nodes in the graph
    pub nodes: Vec<Node>,
    /// All edges in the graph
    pub edges: Vec<Edge>,
    /// Adjacency list representation
    adjacency: HashMap<usize, Vec<usize>>,
    /// Code distance
    pub distance: usize,
    /// Number of fired detectors
    pub num_fired: usize,
}

impl DetectorGraph {
    /// Create an empty detector graph
    pub fn new(distance: usize) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            distance,
            num_fired: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node) {
        let id = node.id;
        if node.fired {
            self.num_fired += 1;
        }
        self.nodes.push(node);
        self.adjacency.entry(id).or_default();
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: Edge) {
        self.adjacency.entry(edge.from).or_default().push(edge.to);
        self.adjacency.entry(edge.to).or_default().push(edge.from);
        self.edges.push(edge);
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node_id: usize) -> Option<&Vec<usize>> {
        self.adjacency.get(&node_id)
    }

    /// Get the node features as a matrix
    pub fn node_features(&self) -> Array2<f32> {
        if self.nodes.is_empty() {
            return Array2::zeros((0, 1));
        }

        let feature_dim = self.nodes[0].features.len();
        let mut features = Array2::zeros((self.nodes.len(), feature_dim));

        for (i, node) in self.nodes.iter().enumerate() {
            for (j, &f) in node.features.iter().enumerate() {
                features[[i, j]] = f;
            }
        }

        features
    }

    /// Get the adjacency matrix
    pub fn adjacency_matrix(&self) -> Array2<f32> {
        let n = self.nodes.len();
        let mut adj = Array2::zeros((n, n));

        for edge in &self.edges {
            adj[[edge.from, edge.to]] = edge.weight;
            adj[[edge.to, edge.from]] = edge.weight;
        }

        adj
    }

    /// Get edge weights as a vector
    pub fn edge_weights(&self) -> Array1<f32> {
        Array1::from_iter(self.edges.iter().map(|e| e.weight))
    }

    /// Get fired detector indices
    pub fn fired_indices(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| n.fired)
            .map(|n| n.id)
            .collect()
    }

    /// Check if the graph is valid
    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(NeuralDecoderError::EmptyGraph);
        }

        // Check all edge endpoints are valid
        for edge in &self.edges {
            if edge.from >= self.nodes.len() || edge.to >= self.nodes.len() {
                return Err(NeuralDecoderError::InvalidDetector(
                    edge.from.max(edge.to)
                ));
            }
        }

        Ok(())
    }

    /// Get the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Builder for constructing detector graphs
pub struct GraphBuilder {
    distance: usize,
    syndrome: Option<Vec<bool>>,
    node_type_pattern: NodeTypePattern,
    error_rate: f64,
}

/// Pattern for determining node types
#[derive(Debug, Clone, Copy)]
pub enum NodeTypePattern {
    /// Checkerboard pattern (standard surface code)
    Checkerboard,
    /// All X-type
    AllX,
    /// All Z-type
    AllZ,
}

impl GraphBuilder {
    /// Create a builder for a surface code of given distance
    pub fn from_surface_code(distance: usize) -> Self {
        Self {
            distance,
            syndrome: None,
            node_type_pattern: NodeTypePattern::Checkerboard,
            error_rate: 0.001,
        }
    }

    /// Set the syndrome bitmap
    pub fn with_syndrome(mut self, syndrome: &[bool]) -> Result<Self> {
        let expected = self.distance * self.distance;
        if syndrome.len() != expected {
            return Err(NeuralDecoderError::syndrome_dim(
                self.distance,
                syndrome.len(),
                1,
            ));
        }
        self.syndrome = Some(syndrome.to_vec());
        Ok(self)
    }

    /// Set the node type pattern
    pub fn with_node_pattern(mut self, pattern: NodeTypePattern) -> Self {
        self.node_type_pattern = pattern;
        self
    }

    /// Set the error rate (for edge weights)
    pub fn with_error_rate(mut self, rate: f64) -> Self {
        self.error_rate = rate;
        self
    }

    /// Build the detector graph
    pub fn build(self) -> Result<DetectorGraph> {
        let d = self.distance;
        let mut graph = DetectorGraph::new(d);

        // Default syndrome: all zeros
        let syndrome = self.syndrome.unwrap_or_else(|| vec![false; d * d]);

        // Create nodes
        for row in 0..d {
            for col in 0..d {
                let id = row * d + col;
                let fired = syndrome.get(id).copied().unwrap_or(false);

                let node_type = match self.node_type_pattern {
                    NodeTypePattern::Checkerboard => {
                        if (row + col) % 2 == 0 {
                            NodeType::XStabilizer
                        } else {
                            NodeType::ZStabilizer
                        }
                    }
                    NodeTypePattern::AllX => NodeType::XStabilizer,
                    NodeTypePattern::AllZ => NodeType::ZStabilizer,
                };

                // Feature vector: [fired, row_norm, col_norm, node_type_x, node_type_z]
                let features = vec![
                    if fired { 1.0 } else { 0.0 },
                    row as f32 / d as f32,
                    col as f32 / d as f32,
                    if node_type == NodeType::XStabilizer { 1.0 } else { 0.0 },
                    if node_type == NodeType::ZStabilizer { 1.0 } else { 0.0 },
                ];

                graph.add_node(Node {
                    id,
                    row,
                    col,
                    fired,
                    node_type,
                    features,
                });
            }
        }

        // Create edges (grid connectivity)
        let weight = (-self.error_rate.ln()) as f32;

        for row in 0..d {
            for col in 0..d {
                let id = row * d + col;

                // Horizontal edge
                if col + 1 < d {
                    let neighbor = row * d + (col + 1);
                    graph.add_edge(Edge {
                        from: id,
                        to: neighbor,
                        weight,
                        edge_type: EdgeType::Horizontal,
                    });
                }

                // Vertical edge
                if row + 1 < d {
                    let neighbor = (row + 1) * d + col;
                    graph.add_edge(Edge {
                        from: id,
                        to: neighbor,
                        weight,
                        edge_type: EdgeType::Vertical,
                    });
                }
            }
        }

        graph.validate()?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node {
            id: 0,
            row: 0,
            col: 0,
            fired: true,
            node_type: NodeType::XStabilizer,
            features: vec![1.0],
        };
        assert_eq!(node.id, 0);
        assert!(node.fired);
    }

    #[test]
    fn test_edge_creation() {
        let edge = Edge {
            from: 0,
            to: 1,
            weight: 1.5,
            edge_type: EdgeType::Horizontal,
        };
        assert_eq!(edge.from, 0);
        assert_eq!(edge.to, 1);
    }

    #[test]
    fn test_graph_construction_d3() {
        let graph = GraphBuilder::from_surface_code(3)
            .build()
            .unwrap();

        // 3x3 = 9 nodes
        assert_eq!(graph.num_nodes(), 9);

        // Grid edges: 2*3 horizontal + 3*2 vertical = 12 edges
        assert_eq!(graph.num_edges(), 12);
    }

    #[test]
    fn test_graph_construction_d5() {
        let graph = GraphBuilder::from_surface_code(5)
            .build()
            .unwrap();

        // 5x5 = 25 nodes
        assert_eq!(graph.num_nodes(), 25);

        // Grid edges: 4*5 horizontal + 5*4 vertical = 40 edges
        assert_eq!(graph.num_edges(), 40);
    }

    #[test]
    fn test_graph_with_syndrome() {
        let syndrome = vec![true, false, true, false, true, false, false, false, true];
        let graph = GraphBuilder::from_surface_code(3)
            .with_syndrome(&syndrome)
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(graph.num_fired, 4);
        assert_eq!(graph.fired_indices(), vec![0, 2, 4, 8]);
    }

    #[test]
    fn test_graph_syndrome_dimension_mismatch() {
        let syndrome = vec![true, false, true]; // Wrong size
        let result = GraphBuilder::from_surface_code(3)
            .with_syndrome(&syndrome);

        assert!(result.is_err());
    }

    #[test]
    fn test_graph_adjacency() {
        let graph = GraphBuilder::from_surface_code(3)
            .build()
            .unwrap();

        // Corner node (0) should have 2 neighbors
        let neighbors = graph.neighbors(0).unwrap();
        assert_eq!(neighbors.len(), 2);

        // Center node (4) should have 4 neighbors
        let neighbors = graph.neighbors(4).unwrap();
        assert_eq!(neighbors.len(), 4);
    }

    #[test]
    fn test_node_features_matrix() {
        let graph = GraphBuilder::from_surface_code(3)
            .build()
            .unwrap();

        let features = graph.node_features();
        assert_eq!(features.shape(), &[9, 5]);
    }

    #[test]
    fn test_adjacency_matrix() {
        let graph = GraphBuilder::from_surface_code(3)
            .build()
            .unwrap();

        let adj = graph.adjacency_matrix();
        assert_eq!(adj.shape(), &[9, 9]);

        // Matrix should be symmetric
        for i in 0..9 {
            for j in 0..9 {
                assert_eq!(adj[[i, j]], adj[[j, i]]);
            }
        }
    }

    #[test]
    fn test_edge_weights() {
        let graph = GraphBuilder::from_surface_code(3)
            .with_error_rate(0.01)
            .build()
            .unwrap();

        let weights = graph.edge_weights();
        assert_eq!(weights.len(), 12);

        // All weights should be positive
        for w in weights.iter() {
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn test_node_type_pattern_checkerboard() {
        let graph = GraphBuilder::from_surface_code(3)
            .with_node_pattern(NodeTypePattern::Checkerboard)
            .build()
            .unwrap();

        // Check checkerboard pattern
        for node in &graph.nodes {
            let expected = if (node.row + node.col) % 2 == 0 {
                NodeType::XStabilizer
            } else {
                NodeType::ZStabilizer
            };
            assert_eq!(node.node_type, expected);
        }
    }

    #[test]
    fn test_node_type_pattern_all_x() {
        let graph = GraphBuilder::from_surface_code(3)
            .with_node_pattern(NodeTypePattern::AllX)
            .build()
            .unwrap();

        for node in &graph.nodes {
            assert_eq!(node.node_type, NodeType::XStabilizer);
        }
    }

    #[test]
    fn test_empty_syndrome() {
        let syndrome = vec![false; 9];
        let graph = GraphBuilder::from_surface_code(3)
            .with_syndrome(&syndrome)
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(graph.num_fired, 0);
        assert!(graph.fired_indices().is_empty());
    }

    #[test]
    fn test_all_fired_syndrome() {
        let syndrome = vec![true; 9];
        let graph = GraphBuilder::from_surface_code(3)
            .with_syndrome(&syndrome)
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(graph.num_fired, 9);
        assert_eq!(graph.fired_indices().len(), 9);
    }

    #[test]
    fn test_graph_validation() {
        let graph = GraphBuilder::from_surface_code(3)
            .build()
            .unwrap();

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_empty_graph_validation() {
        let graph = DetectorGraph::new(3);
        assert!(graph.validate().is_err());
    }
}
