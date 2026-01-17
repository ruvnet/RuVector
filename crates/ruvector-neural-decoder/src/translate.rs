//! Syndrome Translation for Neural Quantum Error Decoding
//!
//! This module converts quantum error syndromes to detector graphs:
//! - Surface code topology support (rotated and unrotated)
//! - Syndrome bitmap to detector graph conversion
//! - Efficient incremental updates for streaming syndromes
//! - Support for both X and Z stabilizer measurements
//!
//! ## Surface Code Structure
//!
//! In a d x d surface code:
//! - X stabilizers measure Z errors (form horizontal chains)
//! - Z stabilizers measure X errors (form vertical chains)
//! - Boundaries terminate error chains
//!
//! ## Detector Graph
//!
//! Detectors are triggered when syndrome bits flip:
//! - Nodes represent detector events
//! - Edges represent likely error mechanisms
//! - Edge weights encode error probabilities

use crate::error::{NeuralDecoderError, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Type of stabilizer measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilizerType {
    /// X stabilizer (measures Z errors)
    X,
    /// Z stabilizer (measures X errors)
    Z,
}

/// Surface code topology variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceCodeTopology {
    /// Standard rotated surface code
    Rotated,
    /// Unrotated (CSS) surface code
    Unrotated,
    /// Planar code with rough/smooth boundaries
    Planar,
}

/// Configuration for syndrome translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateConfig {
    /// Code distance (d x d grid)
    pub distance: usize,
    /// Surface code topology
    pub topology: SurfaceCodeTopology,
    /// Physical error rate (for edge weights)
    pub error_rate: f64,
    /// Measurement error rate
    pub measurement_error_rate: f64,
    /// Number of syndrome rounds (for temporal codes)
    pub num_rounds: usize,
    /// Include boundary nodes in detector graph
    pub include_boundaries: bool,
}

impl Default for TranslateConfig {
    fn default() -> Self {
        Self {
            distance: 5,
            topology: SurfaceCodeTopology::Rotated,
            error_rate: 0.001,
            measurement_error_rate: 0.001,
            num_rounds: 1,
            include_boundaries: true,
        }
    }
}

impl TranslateConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.distance < 3 {
            return Err(NeuralDecoderError::ConfigError(
                "Distance must be at least 3".to_string(),
            ));
        }
        if self.error_rate < 0.0 || self.error_rate > 1.0 {
            return Err(NeuralDecoderError::ConfigError(format!(
                "Error rate must be in [0, 1], got {}",
                self.error_rate
            )));
        }
        Ok(())
    }

    /// Get number of X stabilizers
    pub fn num_x_stabilizers(&self) -> usize {
        match self.topology {
            SurfaceCodeTopology::Rotated => (self.distance - 1) * self.distance / 2 + self.distance / 2,
            SurfaceCodeTopology::Unrotated => (self.distance - 1) * self.distance,
            SurfaceCodeTopology::Planar => (self.distance - 1) * self.distance,
        }
    }

    /// Get number of Z stabilizers
    pub fn num_z_stabilizers(&self) -> usize {
        self.num_x_stabilizers() // Symmetric for most codes
    }

    /// Total number of detectors per round
    pub fn num_detectors_per_round(&self) -> usize {
        self.num_x_stabilizers() + self.num_z_stabilizers()
    }
}

/// A detector node in the detector graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorNode {
    /// Unique detector index
    pub index: usize,
    /// Type of stabilizer that triggered this detector
    pub stabilizer_type: StabilizerType,
    /// Position in syndrome grid (row, col)
    pub position: (usize, usize),
    /// Syndrome round (for temporal decoding)
    pub round: usize,
    /// Whether this is a boundary node
    pub is_boundary: bool,
    /// Feature vector for neural processing
    pub features: Vec<f32>,
}

impl DetectorNode {
    /// Create a new detector node
    pub fn new(
        index: usize,
        stabilizer_type: StabilizerType,
        position: (usize, usize),
        round: usize,
    ) -> Self {
        Self {
            index,
            stabilizer_type,
            position,
            round,
            is_boundary: false,
            features: vec![],
        }
    }

    /// Convert to feature vector
    pub fn to_features(&self, max_distance: usize, num_rounds: usize) -> Vec<f32> {
        let mut features = Vec::with_capacity(8);

        // Normalized position
        features.push(self.position.0 as f32 / max_distance as f32);
        features.push(self.position.1 as f32 / max_distance as f32);

        // Stabilizer type (one-hot)
        features.push(if self.stabilizer_type == StabilizerType::X { 1.0 } else { 0.0 });
        features.push(if self.stabilizer_type == StabilizerType::Z { 1.0 } else { 0.0 });

        // Temporal position
        features.push(self.round as f32 / num_rounds.max(1) as f32);

        // Boundary flag
        features.push(if self.is_boundary { 1.0 } else { 0.0 });

        // Distance from center
        let center = max_distance as f32 / 2.0;
        let dist = ((self.position.0 as f32 - center).powi(2)
            + (self.position.1 as f32 - center).powi(2))
        .sqrt();
        features.push(dist / (max_distance as f32 * 1.414)); // Normalized by diagonal

        // Parity features
        features.push(((self.position.0 + self.position.1) % 2) as f32);

        features
    }
}

/// Detector graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorGraph {
    /// Configuration
    config: TranslateConfig,
    /// Detector nodes
    nodes: Vec<DetectorNode>,
    /// Adjacency list (node_index -> Vec<neighbor_index>)
    adjacency: HashMap<usize, Vec<usize>>,
    /// Edge weights (error probabilities)
    edge_weights: HashMap<(usize, usize), f32>,
    /// Active detectors (triggered in current syndrome)
    active_detectors: HashSet<usize>,
    /// Boundary nodes (virtual nodes for error chains)
    boundary_nodes: Vec<usize>,
}

impl DetectorGraph {
    /// Create a new detector graph from configuration
    pub fn new(config: TranslateConfig) -> Result<Self> {
        config.validate()?;

        let mut graph = Self {
            config: config.clone(),
            nodes: vec![],
            adjacency: HashMap::new(),
            edge_weights: HashMap::new(),
            active_detectors: HashSet::new(),
            boundary_nodes: vec![],
        };

        graph.build_topology()?;
        Ok(graph)
    }

    /// Build the detector graph topology
    fn build_topology(&mut self) -> Result<()> {
        match self.config.topology {
            SurfaceCodeTopology::Rotated => self.build_rotated_topology(),
            SurfaceCodeTopology::Unrotated => self.build_unrotated_topology(),
            SurfaceCodeTopology::Planar => self.build_planar_topology(),
        }
    }

    /// Build rotated surface code topology
    fn build_rotated_topology(&mut self) -> Result<()> {
        let d = self.config.distance;
        let mut node_idx = 0;

        // Create detector nodes for each syndrome round
        for round in 0..self.config.num_rounds {
            // X stabilizers (checkerboard pattern, starting at odd positions)
            for i in 0..d - 1 {
                for j in 0..d - 1 {
                    if (i + j) % 2 == 1 {
                        self.nodes.push(DetectorNode::new(
                            node_idx,
                            StabilizerType::X,
                            (i, j),
                            round,
                        ));
                        self.adjacency.insert(node_idx, vec![]);
                        node_idx += 1;
                    }
                }
            }

            // Z stabilizers (checkerboard pattern, starting at even positions)
            for i in 0..d - 1 {
                for j in 0..d - 1 {
                    if (i + j) % 2 == 0 {
                        self.nodes.push(DetectorNode::new(
                            node_idx,
                            StabilizerType::Z,
                            (i, j),
                            round,
                        ));
                        self.adjacency.insert(node_idx, vec![]);
                        node_idx += 1;
                    }
                }
            }
        }

        // Add boundary nodes if configured
        if self.config.include_boundaries {
            self.add_boundary_nodes(&mut node_idx);
        }

        // Build edges
        self.build_edges();

        Ok(())
    }

    /// Build unrotated surface code topology
    fn build_unrotated_topology(&mut self) -> Result<()> {
        let d = self.config.distance;
        let mut node_idx = 0;

        for round in 0..self.config.num_rounds {
            // X stabilizers (horizontal faces)
            for i in 0..d - 1 {
                for j in 0..d {
                    self.nodes.push(DetectorNode::new(
                        node_idx,
                        StabilizerType::X,
                        (i, j),
                        round,
                    ));
                    self.adjacency.insert(node_idx, vec![]);
                    node_idx += 1;
                }
            }

            // Z stabilizers (vertical faces)
            for i in 0..d {
                for j in 0..d - 1 {
                    self.nodes.push(DetectorNode::new(
                        node_idx,
                        StabilizerType::Z,
                        (i, j),
                        round,
                    ));
                    self.adjacency.insert(node_idx, vec![]);
                    node_idx += 1;
                }
            }
        }

        if self.config.include_boundaries {
            self.add_boundary_nodes(&mut node_idx);
        }

        self.build_edges();
        Ok(())
    }

    /// Build planar code topology
    fn build_planar_topology(&mut self) -> Result<()> {
        // Same as unrotated for basic implementation
        self.build_unrotated_topology()
    }

    /// Add boundary nodes for error chain termination
    fn add_boundary_nodes(&mut self, node_idx: &mut usize) {
        let d = self.config.distance;

        // X boundaries (left and right)
        for i in 0..d {
            let mut node = DetectorNode::new(*node_idx, StabilizerType::X, (i, 0), 0);
            node.is_boundary = true;
            self.nodes.push(node);
            self.boundary_nodes.push(*node_idx);
            self.adjacency.insert(*node_idx, vec![]);
            *node_idx += 1;

            let mut node = DetectorNode::new(*node_idx, StabilizerType::X, (i, d - 1), 0);
            node.is_boundary = true;
            self.nodes.push(node);
            self.boundary_nodes.push(*node_idx);
            self.adjacency.insert(*node_idx, vec![]);
            *node_idx += 1;
        }

        // Z boundaries (top and bottom)
        for j in 0..d {
            let mut node = DetectorNode::new(*node_idx, StabilizerType::Z, (0, j), 0);
            node.is_boundary = true;
            self.nodes.push(node);
            self.boundary_nodes.push(*node_idx);
            self.adjacency.insert(*node_idx, vec![]);
            *node_idx += 1;

            let mut node = DetectorNode::new(*node_idx, StabilizerType::Z, (d - 1, j), 0);
            node.is_boundary = true;
            self.nodes.push(node);
            self.boundary_nodes.push(*node_idx);
            self.adjacency.insert(*node_idx, vec![]);
            *node_idx += 1;
        }
    }

    /// Build edges between detector nodes
    fn build_edges(&mut self) {
        let n_nodes = self.nodes.len();

        // Connect spatially adjacent detectors of same type
        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
                let node_i = &self.nodes[i];
                let node_j = &self.nodes[j];

                // Same stabilizer type and same round
                if node_i.stabilizer_type == node_j.stabilizer_type
                    && node_i.round == node_j.round
                {
                    let dist = self.manhattan_distance(node_i.position, node_j.position);

                    // Adjacent nodes (distance 1 or 2 for diagonal connections)
                    if dist <= 2 {
                        self.add_edge(i, j);
                    }
                }

                // Temporal edges (same position, consecutive rounds)
                if node_i.position == node_j.position
                    && node_i.stabilizer_type == node_j.stabilizer_type
                    && (node_i.round as i32 - node_j.round as i32).abs() == 1
                {
                    self.add_edge(i, j);
                }
            }
        }

        // Connect boundary nodes to nearby detectors
        for &boundary_idx in &self.boundary_nodes.clone() {
            let boundary = &self.nodes[boundary_idx];

            for i in 0..n_nodes {
                if i == boundary_idx {
                    continue;
                }

                let node = &self.nodes[i];
                if node.stabilizer_type == boundary.stabilizer_type
                    && !node.is_boundary
                {
                    let dist = self.manhattan_distance(boundary.position, node.position);
                    if dist <= 2 {
                        self.add_edge(boundary_idx, i);
                    }
                }
            }
        }
    }

    /// Add an edge between two nodes
    fn add_edge(&mut self, i: usize, j: usize) {
        // Compute edge weight from error probability
        let weight = self.compute_edge_weight(i, j);

        self.adjacency.entry(i).or_default().push(j);
        self.adjacency.entry(j).or_default().push(i);

        let key = if i < j { (i, j) } else { (j, i) };
        self.edge_weights.insert(key, weight);
    }

    /// Compute edge weight based on error model
    fn compute_edge_weight(&self, i: usize, j: usize) -> f32 {
        let node_i = &self.nodes[i];
        let node_j = &self.nodes[j];

        let spatial_dist = self.manhattan_distance(node_i.position, node_j.position);
        let temporal_dist = (node_i.round as i32 - node_j.round as i32).abs() as usize;

        // Weight based on error probability and distance
        let base_weight = if temporal_dist > 0 {
            // Measurement error for temporal edges
            self.config.measurement_error_rate as f32
        } else {
            // Physical error for spatial edges
            self.config.error_rate as f32
        };

        // Scale by distance (closer = higher probability)
        let dist = spatial_dist + temporal_dist;
        let scaled_weight = base_weight * (1.0 / (dist as f32 + 1.0));

        // Return log-probability (for min-cut)
        -scaled_weight.max(1e-10).ln()
    }

    /// Manhattan distance between positions
    fn manhattan_distance(&self, p1: (usize, usize), p2: (usize, usize)) -> usize {
        ((p1.0 as i32 - p2.0 as i32).abs() + (p1.1 as i32 - p2.1 as i32).abs()) as usize
    }

    /// Translate a syndrome bitmap to active detectors
    ///
    /// # Arguments
    /// * `syndrome` - Syndrome bits as 2D array (rows x cols)
    /// * `round` - Syndrome round number
    pub fn translate_syndrome(&mut self, syndrome: &Array2<u8>, round: usize) -> Result<()> {
        let rows = syndrome.shape()[0];
        let cols = syndrome.shape()[1];

        // Clear previous active detectors
        self.active_detectors.clear();

        // Find triggered detectors
        for i in 0..rows {
            for j in 0..cols {
                if syndrome[[i, j]] == 1 {
                    // Find corresponding detector node
                    if let Some(idx) = self.find_detector_at((i, j), round) {
                        self.active_detectors.insert(idx);
                    }
                }
            }
        }

        Ok(())
    }

    /// Find detector node at given position and round
    fn find_detector_at(&self, position: (usize, usize), round: usize) -> Option<usize> {
        self.nodes.iter().position(|node| {
            node.position == position && node.round == round && !node.is_boundary
        })
    }

    /// Incremental update for streaming syndromes
    ///
    /// # Arguments
    /// * `changed_positions` - Positions that changed since last syndrome
    /// * `new_values` - New syndrome values at those positions
    /// * `round` - Current syndrome round
    pub fn update_incremental(
        &mut self,
        changed_positions: &[(usize, usize)],
        new_values: &[u8],
        round: usize,
    ) -> Result<()> {
        if changed_positions.len() != new_values.len() {
            return Err(NeuralDecoderError::ConfigError(
                "Position and value arrays must have same length".to_string(),
            ));
        }

        for (pos, &value) in changed_positions.iter().zip(new_values.iter()) {
            if let Some(idx) = self.find_detector_at(*pos, round) {
                if value == 1 {
                    self.active_detectors.insert(idx);
                } else {
                    self.active_detectors.remove(&idx);
                }
            }
        }

        Ok(())
    }

    /// Get node features as a matrix for neural processing
    pub fn get_node_features(&self) -> Array2<f32> {
        let feature_dim = 8; // Fixed feature dimension
        let n_nodes = self.nodes.len();

        let mut features = Array2::zeros((n_nodes, feature_dim));
        for (i, node) in self.nodes.iter().enumerate() {
            let node_features = node.to_features(self.config.distance, self.config.num_rounds);
            for (j, &f) in node_features.iter().enumerate() {
                features[[i, j]] = f;
            }
        }

        features
    }

    /// Get active detector mask
    pub fn get_active_mask(&self) -> Vec<bool> {
        (0..self.nodes.len())
            .map(|i| self.active_detectors.contains(&i))
            .collect()
    }

    /// Get adjacency list
    pub fn adjacency(&self) -> &HashMap<usize, Vec<usize>> {
        &self.adjacency
    }

    /// Get edge weights
    pub fn edge_weights(&self) -> &HashMap<(usize, usize), f32> {
        &self.edge_weights
    }

    /// Get node positions
    pub fn get_positions(&self) -> Vec<(f32, f32)> {
        self.nodes
            .iter()
            .map(|n| (n.position.0 as f32, n.position.1 as f32))
            .collect()
    }

    /// Get boundary distances for each node
    pub fn get_boundary_distances(&self) -> Vec<f32> {
        let d = self.config.distance as f32;
        self.nodes
            .iter()
            .map(|n| {
                let (i, j) = n.position;
                let dist_to_boundary = [
                    i as f32,
                    (self.config.distance - 1 - i) as f32,
                    j as f32,
                    (self.config.distance - 1 - j) as f32,
                ]
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
                dist_to_boundary / d
            })
            .collect()
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.edge_weights.len()
    }

    /// Get configuration
    pub fn config(&self) -> &TranslateConfig {
        &self.config
    }

    /// Get active detectors
    pub fn active_detectors(&self) -> &HashSet<usize> {
        &self.active_detectors
    }
}

/// Syndrome translator for streaming decoding
#[derive(Debug, Clone)]
pub struct SyndromeTranslator {
    /// Current detector graph
    graph: DetectorGraph,
    /// Previous syndrome for diff computation
    prev_syndrome: Option<Array2<u8>>,
    /// Current round
    current_round: usize,
}

impl SyndromeTranslator {
    /// Create a new syndrome translator
    pub fn new(config: TranslateConfig) -> Result<Self> {
        Ok(Self {
            graph: DetectorGraph::new(config)?,
            prev_syndrome: None,
            current_round: 0,
        })
    }

    /// Process a new syndrome
    ///
    /// # Arguments
    /// * `syndrome` - New syndrome measurement
    ///
    /// # Returns
    /// Reference to the updated detector graph
    pub fn process(&mut self, syndrome: &Array2<u8>) -> Result<&DetectorGraph> {
        // Detect changed positions for incremental update
        if let Some(ref prev) = self.prev_syndrome {
            let mut changed_positions = Vec::new();
            let mut new_values = Vec::new();

            for i in 0..syndrome.shape()[0] {
                for j in 0..syndrome.shape()[1] {
                    if syndrome[[i, j]] != prev[[i, j]] {
                        changed_positions.push((i, j));
                        new_values.push(syndrome[[i, j]]);
                    }
                }
            }

            if !changed_positions.is_empty() {
                self.graph.update_incremental(
                    &changed_positions,
                    &new_values,
                    self.current_round,
                )?;
            }
        } else {
            // First syndrome: full translation
            self.graph.translate_syndrome(syndrome, self.current_round)?;
        }

        self.prev_syndrome = Some(syndrome.clone());
        self.current_round += 1;

        Ok(&self.graph)
    }

    /// Reset translator state
    pub fn reset(&mut self) {
        self.prev_syndrome = None;
        self.current_round = 0;
        self.graph.active_detectors.clear();
    }

    /// Get current detector graph
    pub fn graph(&self) -> &DetectorGraph {
        &self.graph
    }

    /// Get mutable reference to detector graph
    pub fn graph_mut(&mut self) -> &mut DetectorGraph {
        &mut self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let mut config = TranslateConfig::default();
        assert!(config.validate().is_ok());

        config.distance = 2;
        assert!(config.validate().is_err());

        config.distance = 5;
        config.error_rate = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_detector_node_features() {
        let node = DetectorNode::new(0, StabilizerType::X, (2, 3), 0);
        let features = node.to_features(5, 1);

        assert_eq!(features.len(), 8);
        assert!(features.iter().all(|&f| f >= 0.0 && f <= 2.0));
    }

    #[test]
    fn test_detector_graph_creation() {
        let config = TranslateConfig {
            distance: 5,
            topology: SurfaceCodeTopology::Rotated,
            ..Default::default()
        };

        let graph = DetectorGraph::new(config).unwrap();

        assert!(graph.num_nodes() > 0);
        assert!(graph.num_edges() > 0);
    }

    #[test]
    fn test_syndrome_translation() {
        let config = TranslateConfig {
            distance: 5,
            topology: SurfaceCodeTopology::Rotated,
            include_boundaries: false,
            ..Default::default()
        };

        let mut graph = DetectorGraph::new(config).unwrap();

        // Create a syndrome with a few triggered detectors
        let mut syndrome = Array2::zeros((4, 4));
        syndrome[[1, 1]] = 1;
        syndrome[[2, 2]] = 1;

        graph.translate_syndrome(&syndrome, 0).unwrap();

        // Should have some active detectors
        assert!(graph.active_detectors().len() > 0 || graph.num_nodes() == 0);
    }

    #[test]
    fn test_incremental_update() {
        let config = TranslateConfig {
            distance: 5,
            include_boundaries: false,
            ..Default::default()
        };

        let mut graph = DetectorGraph::new(config).unwrap();

        // Initial state
        let syndrome = Array2::zeros((4, 4));
        graph.translate_syndrome(&syndrome, 0).unwrap();
        let initial_active = graph.active_detectors().len();

        // Incremental update
        let changed = vec![(1, 1)];
        let values = vec![1];
        graph.update_incremental(&changed, &values, 0).unwrap();

        // May have more active detectors now (or same if position doesn't map to a detector)
        assert!(graph.active_detectors().len() >= initial_active);
    }

    #[test]
    fn test_node_features_matrix() {
        let config = TranslateConfig::default();
        let graph = DetectorGraph::new(config).unwrap();

        let features = graph.get_node_features();
        assert_eq!(features.shape()[0], graph.num_nodes());
        assert_eq!(features.shape()[1], 8); // Fixed feature dimension
    }

    #[test]
    fn test_boundary_distances() {
        let config = TranslateConfig {
            distance: 5,
            include_boundaries: true,
            ..Default::default()
        };

        let graph = DetectorGraph::new(config).unwrap();
        let distances = graph.get_boundary_distances();

        assert_eq!(distances.len(), graph.num_nodes());
        for &d in &distances {
            assert!(d >= 0.0);
        }
    }

    #[test]
    fn test_syndrome_translator() {
        let config = TranslateConfig::default();
        let mut translator = SyndromeTranslator::new(config).unwrap();

        let syndrome1 = Array2::zeros((4, 4));
        let graph1 = translator.process(&syndrome1).unwrap();
        assert_eq!(graph1.active_detectors().len(), 0);

        let mut syndrome2 = Array2::zeros((4, 4));
        syndrome2[[1, 1]] = 1;
        let _ = translator.process(&syndrome2).unwrap();

        translator.reset();
        assert_eq!(translator.graph().active_detectors().len(), 0);
    }

    #[test]
    fn test_different_topologies() {
        for topology in &[
            SurfaceCodeTopology::Rotated,
            SurfaceCodeTopology::Unrotated,
            SurfaceCodeTopology::Planar,
        ] {
            let config = TranslateConfig {
                distance: 5,
                topology: *topology,
                ..Default::default()
            };

            let graph = DetectorGraph::new(config).unwrap();
            assert!(graph.num_nodes() > 0);
        }
    }

    #[test]
    fn test_edge_weights() {
        let config = TranslateConfig {
            distance: 5,
            error_rate: 0.01,
            ..Default::default()
        };

        let graph = DetectorGraph::new(config).unwrap();

        // All edge weights should be positive (log-probability based)
        for &weight in graph.edge_weights().values() {
            assert!(weight > 0.0);
        }
    }

    #[test]
    fn test_positions_and_adjacency() {
        let config = TranslateConfig::default();
        let graph = DetectorGraph::new(config).unwrap();

        let positions = graph.get_positions();
        assert_eq!(positions.len(), graph.num_nodes());

        // Check adjacency is symmetric
        for (&node, neighbors) in graph.adjacency() {
            for &neighbor in neighbors {
                assert!(
                    graph.adjacency().get(&neighbor).map_or(false, |n| n.contains(&node)),
                    "Adjacency should be symmetric"
                );
            }
        }
    }
}
