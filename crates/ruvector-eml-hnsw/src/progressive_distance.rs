//! Progressive dimensionality: layer-aware distance using fewer dimensions at higher layers.
//!
//! Higher HNSW layers serve as coarse navigation aids -- they only need rough
//! distance estimates. By using fewer dimensions at higher layers, we dramatically
//! speed up the multi-layer traversal.
//!
//! Expected speedup: 5-20x for search due to reduced distance computations
//! in the upper layers where the beam width is 1 (greedy traversal).
//!
//! # Layer-to-Dimensionality Mapping (defaults)
//!
//! - Layer 0 (bottom): full cosine distance
//! - Layer 1: 32-dim EML distance
//! - Layer 2+: 8-dim EML distance

use crate::cosine_decomp::{cosine_distance_f32, EmlDistanceModel};
use serde::{Deserialize, Serialize};

/// Layer-aware distance that uses fewer dimensions at higher HNSW layers.
///
/// # Example
///
/// ```
/// use ruvector_eml_hnsw::ProgressiveDistance;
///
/// let pd = ProgressiveDistance::new(128, 4);
/// let a = vec![0.5f32; 128];
/// let b = vec![0.3f32; 128];
///
/// // Layer 0 always uses full cosine
/// let d0 = pd.distance(&a, &b, 0);
/// // Higher layers use fewer dims (once trained)
/// let d2 = pd.distance(&a, &b, 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveDistance {
    /// Per-layer EML distance models. Index 0 is unused (full distance).
    layer_models: Vec<EmlDistanceModel>,
    /// Full dimensionality for layer 0 (standard cosine).
    full_dim: usize,
    /// Dimensionality schedule: dims[i] = number of dims for layer i.
    dim_schedule: Vec<usize>,
}

impl ProgressiveDistance {
    /// Create with default dimensionality schedule.
    ///
    /// - Layer 0: `full_dim` (standard cosine, no EML)
    /// - Layer 1: `min(32, full_dim)`
    /// - Layer 2+: `min(8, full_dim)`
    pub fn new(full_dim: usize, max_layers: usize) -> Self {
        let mut dim_schedule = Vec::with_capacity(max_layers);
        let mut layer_models = Vec::with_capacity(max_layers);

        for layer in 0..max_layers {
            let dims = match layer {
                0 => full_dim,
                1 => 32.min(full_dim),
                _ => 8.min(full_dim),
            };
            dim_schedule.push(dims);
            layer_models.push(EmlDistanceModel::new(full_dim, dims));
        }

        Self {
            layer_models,
            full_dim,
            dim_schedule,
        }
    }

    /// Create with a custom dimensionality schedule.
    ///
    /// Layer 0 should typically equal `full_dim`.
    pub fn with_schedule(full_dim: usize, schedule: &[usize]) -> Self {
        let mut layer_models = Vec::with_capacity(schedule.len());
        let dim_schedule: Vec<usize> = schedule.iter().map(|&d| d.min(full_dim)).collect();

        for &dims in &dim_schedule {
            layer_models.push(EmlDistanceModel::new(full_dim, dims));
        }

        Self {
            layer_models,
            full_dim,
            dim_schedule,
        }
    }

    /// Compute distance appropriate for the given HNSW layer.
    ///
    /// - Layer 0: full cosine distance.
    /// - Higher layers: EML approximate distance (if trained), otherwise full cosine.
    pub fn distance(&self, a: &[f32], b: &[f32], layer: usize) -> f32 {
        if layer == 0 || layer >= self.layer_models.len() {
            return cosine_distance_f32(a, b);
        }
        let model = &self.layer_models[layer];
        if model.is_trained() {
            model.fast_distance(a, b)
        } else {
            cosine_distance_f32(a, b)
        }
    }

    /// Record a training sample for a specific layer.
    pub fn record(&mut self, layer: usize, a: &[f32], b: &[f32], exact_distance: f32) {
        if layer > 0 && layer < self.layer_models.len() {
            self.layer_models[layer].record(a, b, exact_distance);
        }
    }

    /// Train models for all layers (except layer 0, which uses full distance).
    ///
    /// Returns a vec of bools indicating convergence per layer.
    pub fn train_all(&mut self) -> Vec<bool> {
        let mut results = Vec::with_capacity(self.layer_models.len());
        for (i, model) in self.layer_models.iter_mut().enumerate() {
            if i == 0 {
                results.push(true);
            } else {
                results.push(model.train());
            }
        }
        results
    }

    /// Get the dimensionality schedule.
    pub fn dim_schedule(&self) -> &[usize] {
        &self.dim_schedule
    }

    /// Get the full dimensionality.
    pub fn full_dim(&self) -> usize {
        self.full_dim
    }

    /// Check if a particular layer's model is trained.
    pub fn is_layer_trained(&self, layer: usize) -> bool {
        if layer == 0 {
            return true;
        }
        self.layer_models
            .get(layer)
            .map_or(false, |m| m.is_trained())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_schedule() {
        let pd = ProgressiveDistance::new(128, 4);
        let schedule = pd.dim_schedule();
        assert_eq!(schedule[0], 128);
        assert_eq!(schedule[1], 32);
        assert_eq!(schedule[2], 8);
        assert_eq!(schedule[3], 8);
        assert_eq!(pd.full_dim(), 128);
    }

    #[test]
    fn small_dim_clamping() {
        let pd = ProgressiveDistance::new(4, 3);
        let schedule = pd.dim_schedule();
        assert_eq!(schedule[0], 4);
        assert_eq!(schedule[1], 4); // min(32, 4)
        assert_eq!(schedule[2], 4); // min(8, 4)
    }

    #[test]
    fn custom_schedule() {
        let pd = ProgressiveDistance::with_schedule(64, &[64, 16, 4]);
        let schedule = pd.dim_schedule();
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0], 64);
        assert_eq!(schedule[1], 16);
        assert_eq!(schedule[2], 4);
    }

    #[test]
    fn layer0_uses_full_distance() {
        let pd = ProgressiveDistance::new(8, 3);
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = pd.distance(&a, &b, 0);
        let expected = cosine_distance_f32(&a, &b);
        assert!(
            (d - expected).abs() < 1e-6,
            "layer 0 should use full cosine"
        );
    }

    #[test]
    fn untrained_falls_back() {
        let pd = ProgressiveDistance::new(8, 3);
        let a = vec![1.0f32; 8];
        let b = vec![0.5f32; 8];
        let d = pd.distance(&a, &b, 1);
        let expected = cosine_distance_f32(&a, &b);
        assert!(
            (d - expected).abs() < 1e-6,
            "untrained layer should fall back"
        );
    }

    #[test]
    fn layer_trained_status() {
        let pd = ProgressiveDistance::new(8, 3);
        assert!(pd.is_layer_trained(0));
        assert!(!pd.is_layer_trained(1));
        assert!(!pd.is_layer_trained(2));
        assert!(!pd.is_layer_trained(99));
    }

    #[test]
    fn out_of_range_layer_uses_full() {
        let pd = ProgressiveDistance::new(4, 2);
        let a = vec![0.5f32; 4];
        let b = vec![0.3f32; 4];
        let d = pd.distance(&a, &b, 100);
        let expected = cosine_distance_f32(&a, &b);
        assert!((d - expected).abs() < 1e-6);
    }
}
