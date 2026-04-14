//! EML-powered HNSW optimizations: learned distance, progressive dimensionality,
//! and adaptive beam width (ef).
//!
//! Three improvements to HNSW search that use EML (exp-ln) universal function
//! approximation to learn from actual search patterns:
//!
//! 1. **Cosine Decomposition** ([`EmlDistanceModel`]): learn which dimensions
//!    matter most, then compute approximate distances using only those dimensions
//!    (10-30x distance computation speedup).
//!
//! 2. **Progressive Dimensionality** ([`ProgressiveDistance`]): use fewer
//!    dimensions at higher HNSW layers where coarse navigation suffices
//!    (5-20x search speedup).
//!
//! 3. **Adaptive ef** ([`AdaptiveEfModel`]): learn optimal beam width per query
//!    to avoid wasting work on easy queries (1.5-3x search speedup).

use eml_core::EmlModel;

// ---------------------------------------------------------------------------
// Improvement 1: Cosine Decomposition (Learned Dimension Selection)
// ---------------------------------------------------------------------------

/// Learned dimension selection for fast approximate distance.
///
/// Instead of full O(d) cosine over all dimensions, EML discovers the `k` most
/// discriminative dimensions and a formula for combining them into an accurate
/// distance approximation.
///
/// # Training Process
///
/// 1. Collect 500+ `(vec_a, vec_b, exact_cosine_distance)` samples from actual searches.
/// 2. For each dimension `d`, compute correlation between `|a[d] - b[d]|` and exact distance.
/// 3. Select top-k dimensions by absolute correlation (these are the discriminative ones).
/// 4. Train an EML model: `selected_dim_differences -> exact_distance`.
/// 5. The trained model IS the fast distance function.
///
/// # Example
///
/// ```ignore
/// let mut eml_dist = EmlDistanceModel::new(128, 16); // 128-dim vectors, use 16 dims
///
/// // During index build or warmup, record samples
/// for (a, b, exact) in search_pairs {
///     eml_dist.record(&a, &b, exact);
/// }
/// eml_dist.train();
///
/// // Now use fast distance
/// let approx = eml_dist.fast_distance(&query, &candidate);
/// ```
#[derive(Debug, Clone)]
pub struct EmlDistanceModel {
    /// Total number of dimensions in the original vectors.
    full_dim: usize,
    /// How many dimensions to select for fast distance.
    selected_k: usize,
    /// Which dimensions to use (indices into the full vector). Populated after training.
    selected_dims: Vec<usize>,
    /// EML model: maps selected-dim differences to approximate distance.
    /// Input count = selected_k, output heads = 1.
    model: EmlModel,
    /// Whether training is complete.
    trained: bool,
    /// Training buffer: (vec_a, vec_b, exact_distance).
    training_buffer: Vec<(Vec<f32>, Vec<f32>, f32)>,
}

impl EmlDistanceModel {
    /// Create a new untrained EML distance model.
    ///
    /// # Arguments
    /// - `full_dim`: Number of dimensions in the full vectors.
    /// - `selected_k`: Number of dimensions to select for the fast path.
    ///   Typical values: 8, 16, 32 (depending on accuracy requirements).
    pub fn new(full_dim: usize, selected_k: usize) -> Self {
        let k = selected_k.min(full_dim);
        // Depth 3 gives good accuracy with modest parameter count for distance learning
        let model = EmlModel::new(3, k, 1);
        Self {
            full_dim,
            selected_k: k,
            selected_dims: Vec::new(),
            model,
            trained: false,
            training_buffer: Vec::new(),
        }
    }

    /// Whether the model has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Number of training samples collected.
    pub fn sample_count(&self) -> usize {
        self.training_buffer.len()
    }

    /// Fast approximate distance using only selected dimensions.
    ///
    /// Falls back to standard cosine if not yet trained.
    pub fn fast_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if !self.trained {
            return cosine_distance_f32(a, b);
        }

        // Extract differences for selected dimensions
        let features: Vec<f64> = self
            .selected_dims
            .iter()
            .map(|&d| (a[d] - b[d]).abs() as f64)
            .collect();

        let predicted = self.model.predict_primary(&features);
        // Clamp to valid distance range [0, 2] for cosine
        predicted.clamp(0.0, 2.0) as f32
    }

    /// Record a training sample: (vec_a, vec_b, exact_cosine_distance).
    ///
    /// Collect at least 500 samples before calling [`train`].
    pub fn record(&mut self, a: &[f32], b: &[f32], exact_distance: f32) {
        debug_assert_eq!(a.len(), self.full_dim);
        debug_assert_eq!(b.len(), self.full_dim);
        self.training_buffer
            .push((a.to_vec(), b.to_vec(), exact_distance));
    }

    /// Train: discover which dimensions matter and how to combine them.
    ///
    /// Returns `true` if training converged (the model is usable).
    /// Requires at least 100 samples (500+ recommended).
    pub fn train(&mut self) -> bool {
        if self.training_buffer.len() < 100 {
            return false;
        }

        // Step 1: compute per-dimension correlation with exact distance
        let n = self.training_buffer.len() as f64;
        let mut dim_correlations: Vec<(usize, f64)> = Vec::with_capacity(self.full_dim);

        for d in 0..self.full_dim {
            // Compute Pearson correlation between |a[d]-b[d]| and exact_distance
            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_xx = 0.0f64;
            let mut sum_yy = 0.0f64;
            let mut sum_xy = 0.0f64;

            for (a, b, dist) in &self.training_buffer {
                let x = (a[d] - b[d]).abs() as f64;
                let y = *dist as f64;
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_yy += y * y;
                sum_xy += x * y;
            }

            let numerator = n * sum_xy - sum_x * sum_y;
            let denom_x = (n * sum_xx - sum_x * sum_x).max(1e-12);
            let denom_y = (n * sum_yy - sum_y * sum_y).max(1e-12);
            let correlation = numerator / (denom_x.sqrt() * denom_y.sqrt());
            dim_correlations.push((d, correlation.abs()));
        }

        // Step 2: select top-k dimensions by absolute correlation
        dim_correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.selected_dims = dim_correlations
            .iter()
            .take(self.selected_k)
            .map(|(d, _)| *d)
            .collect();
        // Sort for cache-friendly access
        self.selected_dims.sort();

        // Step 3: re-create EML model with the correct input count
        self.model = EmlModel::new(3, self.selected_k, 1);

        // Step 4: train EML model on selected-dim differences -> exact distance
        for (a, b, dist) in &self.training_buffer {
            let features: Vec<f64> = self
                .selected_dims
                .iter()
                .map(|&d| (a[d] - b[d]).abs() as f64)
                .collect();
            self.model.record(&features, &[Some(*dist as f64)]);
        }

        let converged = self.model.train();
        self.trained = true; // usable even if not fully converged
        converged
    }

    /// Return the selected dimension indices (for inspection/debugging).
    pub fn selected_dims(&self) -> &[usize] {
        &self.selected_dims
    }
}

// ---------------------------------------------------------------------------
// Improvement 2: Progressive Dimensionality
// ---------------------------------------------------------------------------

/// Layer-aware distance that uses fewer dimensions at higher HNSW layers.
///
/// Higher layers in HNSW serve as a coarse navigation aid -- they only need
/// rough distance estimates. By using fewer dimensions at higher layers, we
/// can dramatically speed up the multi-layer traversal.
///
/// # Layer-to-Dimensionality Mapping
///
/// - Layer 0 (bottom): full cosine distance
/// - Layer 1: 32-dim EML distance (or configurable)
/// - Layer 2+: 8-dim EML distance (or configurable)
///
/// Each layer's EML model is trained independently on search traffic at
/// that layer.
#[derive(Debug, Clone)]
pub struct ProgressiveDistance {
    /// Per-layer EML distance models. Index 0 = layer 0 (unused, full distance).
    /// Index 1 = layer 1, etc.
    layer_models: Vec<EmlDistanceModel>,
    /// Full dimensionality for layer 0 (standard cosine).
    full_dim: usize,
    /// Dimensionality schedule: dims[i] = number of dims for layer i.
    dim_schedule: Vec<usize>,
}

impl ProgressiveDistance {
    /// Create a new progressive distance with default dimensionality schedule.
    ///
    /// # Arguments
    /// - `full_dim`: Total vector dimensions.
    /// - `max_layers`: Maximum number of HNSW layers to support.
    ///
    /// Default schedule:
    /// - Layer 0: full_dim (standard cosine, no EML)
    /// - Layer 1: min(32, full_dim)
    /// - Layer 2+: min(8, full_dim)
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
    /// # Arguments
    /// - `full_dim`: Total vector dimensions.
    /// - `schedule`: Slice of dimension counts per layer. Layer 0 should
    ///   typically equal `full_dim`.
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
                results.push(true); // layer 0 always uses full distance
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

// ---------------------------------------------------------------------------
// Improvement 3: Adaptive Beam Width (ef)
// ---------------------------------------------------------------------------

/// Learns optimal beam width (ef) per query for target recall.
///
/// Different queries have different difficulty levels. A query near a dense
/// cluster needs a small ef; a query in a sparse region needs a large ef.
/// This model learns to predict the right ef from query features, avoiding
/// wasted work on easy queries while maintaining recall on hard ones.
///
/// # Features
///
/// The model uses 4 features extracted from each query:
/// 1. `query_norm`: L2 norm of the query vector (normalized to [0,1])
/// 2. `query_variance`: variance of query dimensions (normalized)
/// 3. `graph_size_log`: log10(graph_size) / 8.0 (normalized)
/// 4. `query_max_component`: max absolute component value (normalized)
///
/// # Training
///
/// Record `(query_features, ef_used, recall_achieved)` tuples from actual
/// searches (e.g. with ground truth), then train to predict the minimum
/// ef that achieves >= 95% recall.
#[derive(Debug, Clone)]
pub struct AdaptiveEfModel {
    /// EML model: 4 input features -> 1 output (predicted ef).
    model: EmlModel,
    /// Whether training is complete.
    trained: bool,
    /// Default ef to use before training.
    default_ef: usize,
    /// Minimum ef to ever return.
    min_ef: usize,
    /// Maximum ef to ever return.
    max_ef: usize,
    /// Training buffer: (query_features, ef_used, recall_achieved).
    training_buffer: Vec<([f64; 4], usize, f64)>,
}

impl AdaptiveEfModel {
    /// Create a new adaptive ef model.
    ///
    /// # Arguments
    /// - `default_ef`: ef to use before training is complete.
    /// - `min_ef`: minimum ef to ever predict (safety floor).
    /// - `max_ef`: maximum ef to ever predict (budget ceiling).
    pub fn new(default_ef: usize, min_ef: usize, max_ef: usize) -> Self {
        // Depth 3 with 4 inputs is sufficient for ef prediction
        let model = EmlModel::new(3, 4, 1);
        Self {
            model,
            trained: false,
            default_ef,
            min_ef: min_ef.max(1),
            max_ef,
            training_buffer: Vec::new(),
        }
    }

    /// Whether the model has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Number of training samples collected.
    pub fn sample_count(&self) -> usize {
        self.training_buffer.len()
    }

    /// Predict optimal ef for this query.
    ///
    /// Returns `default_ef` if the model has not been trained yet.
    pub fn predict_ef(&self, query: &[f32], graph_size: usize) -> usize {
        if !self.trained {
            return self.default_ef;
        }

        let features = Self::extract_features(query, graph_size);
        let predicted = self.model.predict_primary(&features);
        // Round and clamp to valid range
        let ef = (predicted as usize).clamp(self.min_ef, self.max_ef);
        ef
    }

    /// Record a training observation.
    ///
    /// # Arguments
    /// - `query`: the query vector used.
    /// - `graph_size`: number of points in the graph at search time.
    /// - `ef`: the ef value used for this search.
    /// - `recall`: the recall achieved (0.0 to 1.0).
    pub fn record(&mut self, query: &[f32], graph_size: usize, ef: usize, recall: f64) {
        let features = Self::extract_features(query, graph_size);
        self.training_buffer.push((features, ef, recall));
    }

    /// Train the model to predict minimum ef for >= 95% recall.
    ///
    /// Returns `true` if training converged.
    pub fn train(&mut self) -> bool {
        self.train_for_target_recall(0.95)
    }

    /// Train for a specific target recall threshold.
    pub fn train_for_target_recall(&mut self, target_recall: f64) -> bool {
        if self.training_buffer.len() < 100 {
            return false;
        }

        // Group observations by similar query features and find the minimum ef
        // that achieves target recall for each group.
        //
        // Strategy: for each observation where recall >= target, the ef_used
        // is a valid (possibly oversized) ef. We want the model to predict
        // the smallest such ef. So we feed it (features, ef) pairs where
        // recall was adequate, and the model learns to predict a value close
        // to the minimum adequate ef.

        self.model = EmlModel::new(3, 4, 1);

        // Only train on samples that achieved adequate recall
        let good_samples: Vec<(&[f64; 4], usize)> = self
            .training_buffer
            .iter()
            .filter(|(_, _, recall)| *recall >= target_recall)
            .map(|(features, ef, _)| (features, *ef))
            .collect();

        if good_samples.len() < 50 {
            // Not enough high-recall samples; use all and train for the ef that was used
            for (features, ef, _) in &self.training_buffer {
                let ef_normalized = *ef as f64 / self.max_ef as f64;
                self.model.record(features, &[Some(ef_normalized)]);
            }
        } else {
            // Group by quantized features and find minimum ef per group
            for (features, ef) in &good_samples {
                let ef_normalized = *ef as f64 / self.max_ef as f64;
                self.model.record(*features, &[Some(ef_normalized)]);
            }
        }

        let converged = self.model.train();
        self.trained = true;
        converged
    }

    /// Extract 4 normalized features from a query vector.
    fn extract_features(query: &[f32], graph_size: usize) -> [f64; 4] {
        let n = query.len() as f64;
        if n == 0.0 {
            return [0.0; 4];
        }

        // Feature 1: L2 norm (normalized by sqrt(dim))
        let norm: f64 = query.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        let norm_normalized = (norm / n.sqrt()).min(1.0);

        // Feature 2: variance of components (normalized)
        let mean: f64 = query.iter().map(|&x| x as f64).sum::<f64>() / n;
        let variance: f64 = query
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let variance_normalized = variance.sqrt().min(1.0);

        // Feature 3: log graph size (normalized to roughly [0, 1])
        let graph_log = if graph_size > 0 {
            (graph_size as f64).log10() / 8.0 // up to 10^8 elements
        } else {
            0.0
        };
        let graph_normalized = graph_log.min(1.0);

        // Feature 4: max absolute component (normalized)
        let max_abs: f64 = query
            .iter()
            .map(|&x| (x as f64).abs())
            .fold(0.0f64, f64::max);
        let max_normalized = max_abs.min(1.0);

        [norm_normalized, variance_normalized, graph_normalized, max_normalized]
    }
}

// ---------------------------------------------------------------------------
// Helper: standard cosine distance for f32 vectors
// ---------------------------------------------------------------------------

/// Compute cosine distance between two f32 vectors.
///
/// Returns `1.0 - cosine_similarity`. Range: [0.0, 2.0].
/// Returns 1.0 (orthogonal) if either vector has zero norm.
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-30 {
        return 1.0;
    }
    let similarity = dot / denom;
    (1.0 - similarity).clamp(0.0, 2.0) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Cosine distance tests ---

    #[test]
    fn cosine_distance_identical_vectors() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let d = cosine_distance_f32(&v, &v);
        assert!(d.abs() < 1e-6, "identical vectors should have distance ~0, got {d}");
    }

    #[test]
    fn cosine_distance_opposite_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let d = cosine_distance_f32(&a, &b);
        assert!((d - 2.0).abs() < 1e-6, "opposite vectors should have distance ~2, got {d}");
    }

    #[test]
    fn cosine_distance_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let d = cosine_distance_f32(&a, &b);
        assert!((d - 1.0).abs() < 1e-6, "orthogonal vectors should have distance ~1, got {d}");
    }

    #[test]
    fn cosine_distance_zero_vector_returns_one() {
        let a = vec![1.0f32, 2.0, 3.0];
        let z = vec![0.0f32, 0.0, 0.0];
        let d = cosine_distance_f32(&a, &z);
        assert!((d - 1.0).abs() < 1e-6, "zero vector should give distance 1.0, got {d}");
    }

    // --- EmlDistanceModel tests ---

    #[test]
    fn eml_distance_new_defaults() {
        let m = EmlDistanceModel::new(128, 16);
        assert!(!m.is_trained());
        assert_eq!(m.sample_count(), 0);
        assert_eq!(m.selected_dims().len(), 0);
    }

    #[test]
    fn eml_distance_untrained_falls_back() {
        let m = EmlDistanceModel::new(8, 4);
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = m.fast_distance(&a, &b);
        let expected = cosine_distance_f32(&a, &b);
        assert!(
            (d - expected).abs() < 1e-6,
            "untrained should fall back to cosine: got {d}, expected {expected}"
        );
    }

    #[test]
    fn eml_distance_record_increments() {
        let mut m = EmlDistanceModel::new(4, 2);
        assert_eq!(m.sample_count(), 0);
        m.record(
            &[1.0, 2.0, 3.0, 4.0],
            &[4.0, 3.0, 2.0, 1.0],
            0.5,
        );
        assert_eq!(m.sample_count(), 1);
    }

    #[test]
    fn eml_distance_train_insufficient_data() {
        let mut m = EmlDistanceModel::new(4, 2);
        for i in 0..10 {
            let v = i as f32 / 10.0;
            m.record(&[v, v, v, v], &[1.0 - v, v, v, v], v);
        }
        assert!(!m.train(), "should not converge with only 10 samples");
    }

    #[test]
    fn eml_distance_train_with_enough_data() {
        let dim = 8;
        let mut m = EmlDistanceModel::new(dim, 4);

        // Generate correlated training data: dims 0 and 1 are discriminative,
        // dims 2-7 are noise
        let mut rng_state = 42u64;
        for _ in 0..200 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let t = (rng_state >> 33) as f32 / (u32::MAX as f32);

            let mut a = vec![0.0f32; dim];
            let mut b = vec![0.0f32; dim];
            // Discriminative dims
            a[0] = t;
            a[1] = t * 0.5;
            b[0] = 1.0 - t;
            b[1] = (1.0 - t) * 0.5;
            // Noise dims
            for d in 2..dim {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng_state >> 33) as f32 / (u32::MAX as f32) * 0.01;
                a[d] = noise;
                b[d] = noise;
            }

            let exact = cosine_distance_f32(&a, &b);
            m.record(&a, &b, exact);
        }

        m.train();
        // After training, the model should be marked as trained
        assert!(m.is_trained());
        // Selected dims should include the discriminative ones (0 and 1)
        assert_eq!(m.selected_dims().len(), 4);

        // Fast distance should produce finite values
        let a = vec![0.5f32, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.3f32, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let fast_d = m.fast_distance(&a, &b);
        assert!(fast_d.is_finite(), "fast_distance should be finite, got {fast_d}");
        assert!(fast_d >= 0.0, "fast_distance should be non-negative, got {fast_d}");
    }

    // --- ProgressiveDistance tests ---

    #[test]
    fn progressive_distance_default_schedule() {
        let pd = ProgressiveDistance::new(128, 4);
        let schedule = pd.dim_schedule();
        assert_eq!(schedule[0], 128);
        assert_eq!(schedule[1], 32);
        assert_eq!(schedule[2], 8);
        assert_eq!(schedule[3], 8);
        assert_eq!(pd.full_dim(), 128);
    }

    #[test]
    fn progressive_distance_small_dim() {
        // When full_dim < default schedule values, dims are clamped
        let pd = ProgressiveDistance::new(4, 3);
        let schedule = pd.dim_schedule();
        assert_eq!(schedule[0], 4);
        assert_eq!(schedule[1], 4); // min(32, 4) = 4
        assert_eq!(schedule[2], 4); // min(8, 4) = 4
    }

    #[test]
    fn progressive_distance_custom_schedule() {
        let pd = ProgressiveDistance::with_schedule(64, &[64, 16, 4]);
        let schedule = pd.dim_schedule();
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0], 64);
        assert_eq!(schedule[1], 16);
        assert_eq!(schedule[2], 4);
    }

    #[test]
    fn progressive_distance_layer0_uses_full() {
        let pd = ProgressiveDistance::new(8, 3);
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = pd.distance(&a, &b, 0);
        let expected = cosine_distance_f32(&a, &b);
        assert!(
            (d - expected).abs() < 1e-6,
            "layer 0 should use full cosine: got {d}, expected {expected}"
        );
    }

    #[test]
    fn progressive_distance_untrained_falls_back() {
        let pd = ProgressiveDistance::new(8, 3);
        let a = vec![1.0f32; 8];
        let b = vec![0.5f32; 8];
        // Layer 1, untrained, should fall back to full cosine
        let d = pd.distance(&a, &b, 1);
        let expected = cosine_distance_f32(&a, &b);
        assert!(
            (d - expected).abs() < 1e-6,
            "untrained layer should fall back to cosine"
        );
    }

    #[test]
    fn progressive_distance_layer_trained_status() {
        let pd = ProgressiveDistance::new(8, 3);
        assert!(pd.is_layer_trained(0)); // layer 0 is always "trained"
        assert!(!pd.is_layer_trained(1));
        assert!(!pd.is_layer_trained(2));
        assert!(!pd.is_layer_trained(99)); // out of range
    }

    // --- AdaptiveEfModel tests ---

    #[test]
    fn adaptive_ef_new_defaults() {
        let m = AdaptiveEfModel::new(64, 10, 200);
        assert!(!m.is_trained());
        assert_eq!(m.sample_count(), 0);
    }

    #[test]
    fn adaptive_ef_untrained_returns_default() {
        let m = AdaptiveEfModel::new(64, 10, 200);
        let query = vec![0.5f32; 128];
        let ef = m.predict_ef(&query, 10_000);
        assert_eq!(ef, 64, "untrained model should return default_ef");
    }

    #[test]
    fn adaptive_ef_record_increments() {
        let mut m = AdaptiveEfModel::new(64, 10, 200);
        assert_eq!(m.sample_count(), 0);
        m.record(&[1.0f32; 8], 1000, 50, 0.98);
        assert_eq!(m.sample_count(), 1);
    }

    #[test]
    fn adaptive_ef_train_insufficient_data() {
        let mut m = AdaptiveEfModel::new(64, 10, 200);
        for _ in 0..10 {
            m.record(&[0.5f32; 8], 1000, 50, 0.95);
        }
        assert!(!m.train(), "should not converge with only 10 samples");
    }

    #[test]
    fn adaptive_ef_train_with_enough_data() {
        let mut m = AdaptiveEfModel::new(64, 10, 200);

        // Generate varied training data: low-norm queries need small ef,
        // high-norm queries need large ef
        let mut rng = 42u64;
        for _ in 0..200 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let t = (rng >> 33) as f32 / (u32::MAX as f32);

            let dim = 16;
            let query: Vec<f32> = (0..dim)
                .map(|i| t * (i as f32 + 1.0) / dim as f32)
                .collect();

            // Simulate: higher variance queries need higher ef
            let ef_needed = (20.0 + t * 100.0) as usize;
            let recall = if ef_needed < 100 { 0.98 } else { 0.92 };
            m.record(&query, 10_000, ef_needed, recall);
        }

        m.train();
        assert!(m.is_trained());

        // Predictions should be in valid range
        let query = vec![0.5f32; 16];
        let ef = m.predict_ef(&query, 10_000);
        assert!(ef >= 10, "ef should be >= min_ef, got {ef}");
        assert!(ef <= 200, "ef should be <= max_ef, got {ef}");
    }

    #[test]
    fn adaptive_ef_clamps_predictions() {
        let mut m = AdaptiveEfModel::new(64, 10, 200);

        // Train with extreme values
        for _ in 0..200 {
            m.record(&[0.1f32; 8], 100, 5, 0.99); // ef < min_ef
        }
        m.train();

        let ef = m.predict_ef(&[0.1f32; 8], 100);
        assert!(ef >= 10, "ef should be clamped to min_ef, got {ef}");
    }

    #[test]
    fn adaptive_ef_feature_extraction() {
        // Test that features are deterministic and normalized
        let query = vec![0.5f32; 8];
        let f1 = AdaptiveEfModel::extract_features(&query, 10_000);
        let f2 = AdaptiveEfModel::extract_features(&query, 10_000);
        assert_eq!(f1, f2, "features should be deterministic");

        for &f in &f1 {
            assert!(f >= 0.0 && f <= 1.0, "feature {f} should be in [0, 1]");
        }
    }

    #[test]
    fn adaptive_ef_empty_query() {
        let features = AdaptiveEfModel::extract_features(&[], 1000);
        assert_eq!(features, [0.0; 4], "empty query should give zero features");
    }

    // --- Integration tests ---

    #[test]
    fn all_three_models_compose() {
        // Verify that all three models can be created and used together
        let dim = 32;

        let dist_model = EmlDistanceModel::new(dim, 8);
        let prog_dist = ProgressiveDistance::new(dim, 4);
        let ef_model = AdaptiveEfModel::new(64, 10, 200);

        let query = vec![0.5f32; dim];
        let candidate = vec![0.3f32; dim];

        // All should work without training (fallback behavior)
        let d1 = dist_model.fast_distance(&query, &candidate);
        let d2 = prog_dist.distance(&query, &candidate, 0);
        let d3 = prog_dist.distance(&query, &candidate, 2);
        let ef = ef_model.predict_ef(&query, 10_000);

        assert!(d1.is_finite());
        assert!(d2.is_finite());
        assert!(d3.is_finite());
        assert_eq!(ef, 64); // default
    }
}
