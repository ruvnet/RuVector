//! Data lineage tracking for BCBS 239 compliance
//!
//! This module implements comprehensive data lineage tracking to meet regulatory
//! requirements such as BCBS 239 (Basel Committee on Banking Supervision 239).
//!
//! ## BCBS 239 Requirements
//!
//! - **Accuracy**: Data should be accurate and reconcilable
//! - **Completeness**: All material risk data should be captured
//! - **Timeliness**: Data should be available in a timely manner
//! - **Adaptability**: Systems should be adaptable to changing requirements
//! - **Lineage**: Full tracking of data transformations and sources

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Types of data transformations tracked in the lineage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TransformationType {
    /// Direct copy of data
    Copy,

    /// Filtering operation (subset of data)
    Filter,

    /// Aggregation (sum, average, etc.)
    Aggregation,

    /// Join operation (combining multiple sources)
    Join,

    /// Enrichment (adding additional data)
    Enrichment,

    /// Validation/cleaning operation
    Validation,

    /// Calculation or derivation
    Calculation,

    /// Format conversion
    Conversion,

    /// Custom transformation
    Custom(String),
}

impl std::fmt::Display for TransformationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Copy => write!(f, "copy"),
            Self::Filter => write!(f, "filter"),
            Self::Aggregation => write!(f, "aggregation"),
            Self::Join => write!(f, "join"),
            Self::Enrichment => write!(f, "enrichment"),
            Self::Validation => write!(f, "validation"),
            Self::Calculation => write!(f, "calculation"),
            Self::Conversion => write!(f, "conversion"),
            Self::Custom(s) => write!(f, "custom:{}", s),
        }
    }
}

/// An edge in the data flow graph representing a transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFlowEdge {
    /// Source node ID
    pub source: Uuid,

    /// Destination node ID
    pub destination: Uuid,

    /// Type of transformation applied
    pub transformation: TransformationType,

    /// Timestamp when the transformation occurred
    pub timestamp: DateTime<Utc>,

    /// Optional description of the transformation
    pub description: Option<String>,

    /// Metadata about the transformation
    pub metadata: serde_json::Value,
}

impl DataFlowEdge {
    /// Create a new data flow edge
    pub fn new(
        source: Uuid,
        destination: Uuid,
        transformation: TransformationType,
        description: Option<String>,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            source,
            destination,
            transformation,
            timestamp: Utc::now(),
            description,
            metadata,
        }
    }
}

/// Tracks data lineage for regulatory compliance
#[derive(Debug, Clone)]
pub struct LineageTracker {
    /// Map from destination node to list of incoming edges
    inputs: HashMap<Uuid, Vec<DataFlowEdge>>,

    /// Map from source node to list of outgoing edges
    outputs: HashMap<Uuid, Vec<DataFlowEdge>>,

    /// Cache for lineage queries
    lineage_cache: HashMap<Uuid, Vec<Uuid>>,

    /// Cache for impact queries
    impact_cache: HashMap<Uuid, Vec<Uuid>>,
}

impl LineageTracker {
    /// Create a new lineage tracker
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            lineage_cache: HashMap::new(),
            impact_cache: HashMap::new(),
        }
    }

    /// Record a data flow transformation
    pub fn record_flow(
        &mut self,
        source: Uuid,
        destination: Uuid,
        transformation: TransformationType,
        description: Option<String>,
        metadata: serde_json::Value,
    ) {
        let edge = DataFlowEdge::new(
            source,
            destination,
            transformation,
            description,
            metadata,
        );

        // Add to inputs map
        self.inputs
            .entry(destination)
            .or_insert_with(Vec::new)
            .push(edge.clone());

        // Add to outputs map
        self.outputs
            .entry(source)
            .or_insert_with(Vec::new)
            .push(edge);

        // Invalidate caches
        self.lineage_cache.remove(&destination);
        self.impact_cache.remove(&source);
    }

    /// Get all inputs that produced a given output
    /// Answers: "What inputs produced this output?"
    pub fn get_inputs(&self, node_id: &Uuid) -> Vec<&DataFlowEdge> {
        self.inputs
            .get(node_id)
            .map(|edges| edges.iter().collect())
            .unwrap_or_default()
    }

    /// Get all outputs that depend on a given input
    /// Answers: "What outputs depend on this input?"
    pub fn get_outputs(&self, node_id: &Uuid) -> Vec<&DataFlowEdge> {
        self.outputs
            .get(node_id)
            .map(|edges| edges.iter().collect())
            .unwrap_or_default()
    }

    /// Get complete lineage (all ancestors) of a node
    /// This is the full backward trace to all source data
    pub fn get_full_lineage(&mut self, node_id: &Uuid) -> Vec<Uuid> {
        // Check cache first
        if let Some(cached) = self.lineage_cache.get(node_id) {
            return cached.clone();
        }

        let mut lineage = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![*node_id];

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }

            visited.insert(current);
            lineage.push(current);

            // Add all inputs to the stack
            if let Some(edges) = self.inputs.get(&current) {
                for edge in edges {
                    stack.push(edge.source);
                }
            }
        }

        // Cache the result
        self.lineage_cache.insert(*node_id, lineage.clone());

        lineage
    }

    /// Get complete impact (all descendants) of a node
    /// This is the full forward trace to all affected data
    pub fn get_full_impact(&mut self, node_id: &Uuid) -> Vec<Uuid> {
        // Check cache first
        if let Some(cached) = self.impact_cache.get(node_id) {
            return cached.clone();
        }

        let mut impact = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![*node_id];

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }

            visited.insert(current);
            impact.push(current);

            // Add all outputs to the stack
            if let Some(edges) = self.outputs.get(&current) {
                for edge in edges {
                    stack.push(edge.destination);
                }
            }
        }

        // Cache the result
        self.impact_cache.insert(*node_id, impact.clone());

        impact
    }

    /// Get all transformations in the lineage chain
    pub fn get_transformation_chain(&self, node_id: &Uuid) -> Vec<TransformationType> {
        let mut transformations = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![*node_id];

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }

            visited.insert(current);

            if let Some(edges) = self.inputs.get(&current) {
                for edge in edges {
                    transformations.push(edge.transformation.clone());
                    stack.push(edge.source);
                }
            }
        }

        transformations
    }

    /// Generate a lineage report for audit purposes
    pub fn generate_lineage_report(&mut self, node_id: &Uuid) -> LineageReport {
        let full_lineage = self.get_full_lineage(node_id);
        let transformations = self.get_transformation_chain(node_id);

        let mut transformation_counts = HashMap::new();
        for t in &transformations {
            *transformation_counts.entry(t.clone()).or_insert(0) += 1;
        }

        LineageReport {
            node_id: *node_id,
            total_ancestors: full_lineage.len(),
            transformation_counts,
            lineage_depth: self.calculate_depth(node_id),
            generated_at: Utc::now(),
        }
    }

    /// Calculate the maximum depth of the lineage tree
    fn calculate_depth(&self, node_id: &Uuid) -> usize {
        let mut max_depth = 0;
        let mut stack = vec![(*node_id, 0)];
        let mut visited = HashSet::new();

        while let Some((current, depth)) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }

            visited.insert(current);
            max_depth = max_depth.max(depth);

            if let Some(edges) = self.inputs.get(&current) {
                for edge in edges {
                    stack.push((edge.source, depth + 1));
                }
            }
        }

        max_depth
    }

    /// Clear all caches (useful after bulk operations)
    pub fn clear_caches(&mut self) {
        self.lineage_cache.clear();
        self.impact_cache.clear();
    }

    /// Get statistics about tracked lineage
    pub fn stats(&self) -> LineageStats {
        LineageStats {
            total_nodes: self.inputs.len().max(self.outputs.len()),
            total_edges: self.inputs.values().map(|v| v.len()).sum(),
            cached_lineages: self.lineage_cache.len(),
            cached_impacts: self.impact_cache.len(),
        }
    }
}

impl Default for LineageTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Report of data lineage for a specific node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageReport {
    /// The node this report is for
    pub node_id: Uuid,

    /// Total number of ancestor nodes
    pub total_ancestors: usize,

    /// Count of each transformation type
    pub transformation_counts: HashMap<TransformationType, usize>,

    /// Maximum depth of the lineage tree
    pub lineage_depth: usize,

    /// When this report was generated
    pub generated_at: DateTime<Utc>,
}

/// Statistics about the lineage tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageStats {
    /// Total number of nodes being tracked
    pub total_nodes: usize,

    /// Total number of edges (transformations)
    pub total_edges: usize,

    /// Number of cached lineage queries
    pub cached_lineages: usize,

    /// Number of cached impact queries
    pub cached_impacts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_flow() {
        let mut tracker = LineageTracker::new();
        let source = Uuid::new_v4();
        let dest = Uuid::new_v4();

        tracker.record_flow(
            source,
            dest,
            TransformationType::Filter,
            Some("Filter positive values".to_string()),
            serde_json::json!({"condition": "value > 0"}),
        );

        let inputs = tracker.get_inputs(&dest);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].source, source);

        let outputs = tracker.get_outputs(&source);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].destination, dest);
    }

    #[test]
    fn test_full_lineage() {
        let mut tracker = LineageTracker::new();

        // Create a chain: a -> b -> c
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        tracker.record_flow(
            a,
            b,
            TransformationType::Copy,
            None,
            serde_json::json!({}),
        );

        tracker.record_flow(
            b,
            c,
            TransformationType::Filter,
            None,
            serde_json::json!({}),
        );

        let lineage = tracker.get_full_lineage(&c);
        assert_eq!(lineage.len(), 3);
        assert!(lineage.contains(&a));
        assert!(lineage.contains(&b));
        assert!(lineage.contains(&c));
    }

    #[test]
    fn test_impact_analysis() {
        let mut tracker = LineageTracker::new();

        // Create a chain: a -> b -> c
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        tracker.record_flow(
            a,
            b,
            TransformationType::Copy,
            None,
            serde_json::json!({}),
        );

        tracker.record_flow(
            b,
            c,
            TransformationType::Filter,
            None,
            serde_json::json!({}),
        );

        let impact = tracker.get_full_impact(&a);
        assert_eq!(impact.len(), 3);
        assert!(impact.contains(&a));
        assert!(impact.contains(&b));
        assert!(impact.contains(&c));
    }

    #[test]
    fn test_lineage_report() {
        let mut tracker = LineageTracker::new();

        let a = Uuid::new_v4();
        let b = Uuid::new_v4();

        tracker.record_flow(
            a,
            b,
            TransformationType::Aggregation,
            None,
            serde_json::json!({}),
        );

        let report = tracker.generate_lineage_report(&b);
        assert_eq!(report.total_ancestors, 2); // includes self
        assert!(report.transformation_counts.contains_key(&TransformationType::Aggregation));
    }

    #[test]
    fn test_transformation_types() {
        let types = vec![
            TransformationType::Copy,
            TransformationType::Filter,
            TransformationType::Aggregation,
            TransformationType::Join,
            TransformationType::Enrichment,
            TransformationType::Validation,
            TransformationType::Calculation,
            TransformationType::Conversion,
            TransformationType::Custom("test".to_string()),
        ];

        for t in types {
            let s = t.to_string();
            assert!(!s.is_empty());
        }
    }
}
