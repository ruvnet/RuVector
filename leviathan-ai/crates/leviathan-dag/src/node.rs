//! Node types and implementations for the DAG system
//!
//! This module defines different types of nodes that can exist in the audit DAG,
//! each serving a specific purpose in the data lineage tracking system.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Different types of nodes in the DAG
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum NodeType {
    /// A data node representing raw or processed data
    Data(DataNode),

    /// A compute node representing a transformation or calculation
    Compute(ComputeNode),

    /// A validation node representing a verification step
    Validation(ValidationNode),

    /// A checkpoint node for audit snapshots
    Checkpoint(CheckpointNode),
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::Data(_) => write!(f, "Data"),
            NodeType::Compute(_) => write!(f, "Compute"),
            NodeType::Validation(_) => write!(f, "Validation"),
            NodeType::Checkpoint(_) => write!(f, "Checkpoint"),
        }
    }
}

/// A data node containing raw or processed information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataNode {
    /// Size of the data in bytes
    pub size: usize,
}

impl DataNode {
    /// Create a new data node
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

/// A compute node representing a transformation or calculation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComputeNode {
    /// Description of the operation performed
    pub operation: String,
}

impl ComputeNode {
    /// Create a new compute node
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
        }
    }
}

/// A validation node representing a verification step
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationNode {
    /// Type of validation performed
    pub validation_type: ValidationType,

    /// Result of the validation
    pub passed: bool,

    /// Optional error message if validation failed
    pub error_message: Option<String>,
}

impl ValidationNode {
    /// Create a new validation node
    pub fn new(
        validation_type: ValidationType,
        passed: bool,
        error_message: Option<String>,
    ) -> Self {
        Self {
            validation_type,
            passed,
            error_message,
        }
    }

    /// Create a passed validation node
    pub fn passed(validation_type: ValidationType) -> Self {
        Self::new(validation_type, true, None)
    }

    /// Create a failed validation node
    pub fn failed(validation_type: ValidationType, error_message: impl Into<String>) -> Self {
        Self::new(validation_type, false, Some(error_message.into()))
    }
}

/// Types of validation that can be performed
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationType {
    /// Schema validation
    Schema,

    /// Business rules validation
    BusinessRules,

    /// Cryptographic signature validation
    Signature,

    /// Data integrity check
    Integrity,

    /// Compliance check (e.g., BCBS 239)
    Compliance,

    /// Custom validation
    Custom(String),
}

/// A checkpoint node for creating audit snapshots
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointNode {
    /// Name or description of the checkpoint
    pub name: String,

    /// Merkle root of all nodes up to this checkpoint
    pub merkle_root: String,

    /// Number of nodes included in this checkpoint
    pub node_count: usize,
}

impl CheckpointNode {
    /// Create a new checkpoint node
    pub fn new(
        name: impl Into<String>,
        merkle_root: impl Into<String>,
        node_count: usize,
    ) -> Self {
        Self {
            name: name.into(),
            merkle_root: merkle_root.into(),
            node_count,
        }
    }
}

/// Trait for nodes that can be serialized for hashing
pub trait Hashable {
    /// Serialize the node for content-addressable hashing
    fn to_hash_input(&self) -> Vec<u8>;

    /// Compute the BLAKE3 hash of this node
    fn compute_hash(&self) -> String {
        let input = self.to_hash_input();
        blake3::hash(&input).to_hex().to_string()
    }
}

impl Hashable for DataNode {
    fn to_hash_input(&self) -> Vec<u8> {
        // Serialize to JSON for deterministic hashing
        serde_json::to_vec(self).unwrap_or_default()
    }
}

impl Hashable for ComputeNode {
    fn to_hash_input(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
}

impl Hashable for ValidationNode {
    fn to_hash_input(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
}

impl Hashable for CheckpointNode {
    fn to_hash_input(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
}

impl Hashable for NodeType {
    fn to_hash_input(&self) -> Vec<u8> {
        match self {
            NodeType::Data(n) => n.to_hash_input(),
            NodeType::Compute(n) => n.to_hash_input(),
            NodeType::Validation(n) => n.to_hash_input(),
            NodeType::Checkpoint(n) => n.to_hash_input(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_node() {
        let node = DataNode::new(1024);
        assert_eq!(node.size, 1024);

        let hash = node.compute_hash();
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_compute_node() {
        let node = ComputeNode::new("transform");
        assert_eq!(node.operation, "transform");

        let hash = node.compute_hash();
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_validation_node() {
        let passed = ValidationNode::passed(ValidationType::Schema);
        assert!(passed.passed);
        assert!(passed.error_message.is_none());

        let failed = ValidationNode::failed(
            ValidationType::BusinessRules,
            "Invalid amount"
        );
        assert!(!failed.passed);
        assert_eq!(failed.error_message, Some("Invalid amount".to_string()));
    }

    #[test]
    fn test_checkpoint_node() {
        let checkpoint = CheckpointNode::new(
            "Monthly Audit",
            "abc123",
            1000
        );
        assert_eq!(checkpoint.name, "Monthly Audit");
        assert_eq!(checkpoint.node_count, 1000);
    }

    #[test]
    fn test_hashable() {
        let node1 = DataNode::new(100);
        let node2 = DataNode::new(100);
        let node3 = DataNode::new(200);

        // Same data should produce same hash
        assert_eq!(node1.compute_hash(), node2.compute_hash());

        // Different data should produce different hash
        assert_ne!(node1.compute_hash(), node3.compute_hash());
    }

    #[test]
    fn test_node_type_display() {
        let data = NodeType::Data(DataNode::new(100));
        let compute = NodeType::Compute(ComputeNode::new("test"));
        let validation = NodeType::Validation(ValidationNode::passed(ValidationType::Schema));
        let checkpoint = NodeType::Checkpoint(CheckpointNode::new("test", "hash", 10));

        assert_eq!(data.to_string(), "Data");
        assert_eq!(compute.to_string(), "Compute");
        assert_eq!(validation.to_string(), "Validation");
        assert_eq!(checkpoint.to_string(), "Checkpoint");
    }
}
