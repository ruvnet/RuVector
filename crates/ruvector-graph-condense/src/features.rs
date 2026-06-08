//! Per-vertex feature storage (embeddings + optional class labels).
//!
//! Graph condensation needs more than topology: each original vertex carries a
//! feature vector (e.g. a node embedding) and, for supervised settings, a class
//! label. [`NodeFeatures`] is a thin, validated container keyed by the same
//! [`VertexId`](ruvector_mincut::VertexId) used by the min-cut engine's
//! [`DynamicGraph`](ruvector_mincut::DynamicGraph).

use crate::error::{CondenseError, Result};
use ruvector_mincut::VertexId;
use std::collections::HashMap;

/// Feature store mapping graph vertices to embeddings and optional labels.
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    dim: usize,
    num_classes: usize,
    embeddings: HashMap<VertexId, Vec<f32>>,
    labels: HashMap<VertexId, usize>,
}

impl NodeFeatures {
    /// Create an empty feature store for `dim`-dimensional embeddings.
    ///
    /// `num_classes` may be `0` for the unsupervised case (no class
    /// distributions are produced during condensation).
    pub fn new(dim: usize, num_classes: usize) -> Self {
        Self {
            dim,
            num_classes,
            embeddings: HashMap::new(),
            labels: HashMap::new(),
        }
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of distinct classes (`0` if unsupervised).
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Number of vertices with a stored embedding.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Whether any embeddings are stored.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Insert or replace the embedding for `vertex`.
    ///
    /// # Errors
    /// Returns [`CondenseError::DimensionMismatch`] if `embedding.len() != dim`.
    pub fn set_embedding(&mut self, vertex: VertexId, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.dim {
            return Err(CondenseError::DimensionMismatch {
                expected: self.dim,
                got: embedding.len(),
            });
        }
        self.embeddings.insert(vertex, embedding);
        Ok(())
    }

    /// Attach a class label to `vertex`. Labels at or above `num_classes` are
    /// accepted but will be ignored when building class distributions.
    pub fn set_label(&mut self, vertex: VertexId, label: usize) {
        self.labels.insert(vertex, label);
    }

    /// Insert an embedding and label together.
    ///
    /// # Errors
    /// Returns [`CondenseError::DimensionMismatch`] if the embedding dimension
    /// is wrong.
    pub fn set(&mut self, vertex: VertexId, embedding: Vec<f32>, label: usize) -> Result<()> {
        self.set_embedding(vertex, embedding)?;
        self.set_label(vertex, label);
        Ok(())
    }

    /// Borrow the embedding for `vertex`, if present.
    pub fn embedding(&self, vertex: VertexId) -> Option<&[f32]> {
        self.embeddings.get(&vertex).map(Vec::as_slice)
    }

    /// Get the label for `vertex`, if present.
    pub fn label(&self, vertex: VertexId) -> Option<usize> {
        self.labels.get(&vertex).copied()
    }

    /// Borrow the embedding for `vertex` or fail with
    /// [`CondenseError::MissingFeature`].
    pub(crate) fn require(&self, vertex: VertexId) -> Result<&[f32]> {
        self.embedding(vertex)
            .ok_or(CondenseError::MissingFeature(vertex))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_dimension() {
        let mut f = NodeFeatures::new(3, 2);
        assert!(f.set_embedding(1, vec![0.0, 1.0, 2.0]).is_ok());
        let err = f.set_embedding(2, vec![0.0, 1.0]).unwrap_err();
        assert!(matches!(
            err,
            CondenseError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn stores_and_reads_back() {
        let mut f = NodeFeatures::new(2, 3);
        f.set(7, vec![1.0, 2.0], 1).unwrap();
        assert_eq!(f.embedding(7), Some(&[1.0f32, 2.0][..]));
        assert_eq!(f.label(7), Some(1));
        assert_eq!(f.len(), 1);
        assert_eq!(f.num_classes(), 3);
    }

    #[test]
    fn require_reports_missing() {
        let f = NodeFeatures::new(2, 0);
        assert!(matches!(
            f.require(42).unwrap_err(),
            CondenseError::MissingFeature(42)
        ));
    }
}
