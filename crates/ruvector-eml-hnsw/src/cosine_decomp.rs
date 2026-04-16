//! Cosine decomposition: learned dimension selection for fast approximate distance.
//!
//! Instead of full O(d) cosine over all dimensions, EML discovers the `k` most
//! discriminative dimensions and a formula for combining them into an accurate
//! distance approximation.
//!
//! # Training Process
//!
//! 1. Collect 500+ `(vec_a, vec_b, exact_cosine_distance)` samples from actual searches.
//! 2. For each dimension `d`, compute correlation between `|a[d] - b[d]|` and exact distance.
//! 3. Select top-k dimensions by absolute correlation (these are the discriminative ones).
//! 4. Train an EML model: `selected_dim_differences -> exact_distance`.
//! 5. The trained model IS the fast distance function.
//!
//! Expected speedup: 10-30x for distance computation on domain-specific data.

use eml_core::EmlModel;
use serde::{Deserialize, Serialize};

use crate::selected_distance::cosine_distance_selected;

/// Learned dimension selection for fast approximate distance.
///
/// # Example
///
/// ```
/// use ruvector_eml_hnsw::EmlDistanceModel;
///
/// let mut model = EmlDistanceModel::new(128, 16);
/// // Record training samples (in practice from real HNSW searches)
/// let a = vec![0.5f32; 128];
/// let b = vec![0.3f32; 128];
/// let exact = ruvector_eml_hnsw::cosine_distance_f32(&a, &b);
/// model.record(&a, &b, exact);
/// // ... record many more ...
/// // model.train();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmlDistanceModel {
    /// Total number of dimensions in the original vectors.
    full_dim: usize,
    /// How many dimensions to select for fast distance.
    selected_k: usize,
    /// Which dimensions to use (indices into the full vector). Populated after training.
    selected_dims: Vec<usize>,
    /// EML model: maps selected-dim differences to approximate distance.
    model: EmlModel,
    /// Whether training is complete.
    trained: bool,
    /// Training buffer: (vec_a, vec_b, exact_distance).
    #[serde(skip)]
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

    /// The selected dimension indices (populated after training).
    pub fn selected_dims(&self) -> &[usize] {
        &self.selected_dims
    }

    /// Fast approximate distance using only selected dimensions.
    ///
    /// Falls back to standard cosine if not yet trained.
    pub fn fast_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if !self.trained {
            return cosine_distance_f32(a, b);
        }

        let features: Vec<f64> = self
            .selected_dims
            .iter()
            .map(|&d| (a[d] - b[d]).abs() as f64)
            .collect();

        let predicted = self.model.predict_primary(&features);
        predicted.clamp(0.0, 2.0) as f32
    }

    /// Record a training sample: (vec_a, vec_b, exact_cosine_distance).
    ///
    /// Collect at least 100 samples before calling [`train`] (500+ recommended).
    pub fn record(&mut self, a: &[f32], b: &[f32], exact_distance: f32) {
        debug_assert_eq!(a.len(), self.full_dim);
        debug_assert_eq!(b.len(), self.full_dim);
        self.training_buffer
            .push((a.to_vec(), b.to_vec(), exact_distance));
    }

    /// Train: discover which dimensions matter and how to combine them.
    ///
    /// Returns `true` if training converged (the model is usable either way
    /// after this call, convergence just indicates accuracy).
    /// Requires at least 100 samples (500+ recommended).
    pub fn train(&mut self) -> bool {
        if self.training_buffer.len() < 100 {
            return false;
        }

        // Step 1: Pearson correlation per dimension against exact distance
        let n = self.training_buffer.len() as f64;
        let mut dim_correlations: Vec<(usize, f64)> = Vec::with_capacity(self.full_dim);

        for d in 0..self.full_dim {
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

        // Step 2: select top-k by absolute correlation
        dim_correlations
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.selected_dims = dim_correlations
            .iter()
            .take(self.selected_k)
            .map(|(d, _)| *d)
            .collect();
        self.selected_dims.sort(); // cache-friendly access

        // Step 3: re-create and train EML model
        self.model = EmlModel::new(3, self.selected_k, 1);

        for (a, b, dist) in &self.training_buffer {
            let features: Vec<f64> = self
                .selected_dims
                .iter()
                .map(|&d| (a[d] - b[d]).abs() as f64)
                .collect();
            self.model.record(&features, &[Some(*dist as f64)]);
        }

        let converged = self.model.train();
        self.trained = true;
        converged
    }

    /// Train the dimension selector by directly optimizing retention of the
    /// exact-cosine top-`target_k` neighbor set.
    ///
    /// This is a retention-objective selector: at each step we add the single
    /// dimension that maximizes mean recall\@`target_k` on the given training
    /// queries against a pre-computed exact full-cosine top-`candidate_pool`
    /// ground truth over the training corpus.
    ///
    /// # Algorithm (greedy forward selection)
    ///
    /// 1. Precompute full-cosine top-`candidate_pool` ground truth for each
    ///    training query against the training corpus.
    /// 2. Start with `selected = []` and `remaining = 0..full_dim`.
    /// 3. At each step, for every candidate dim `d` in `remaining`, form
    ///    `trial = selected ∪ {d}`, compute `cosine_distance_selected` on
    ///    `trial` over the corpus, take the top-`target_k`, and measure
    ///    `recall@target_k` against the ground truth.
    /// 4. Add the single dim whose mean recall over training queries is
    ///    highest (ties broken by lower index for determinism).
    /// 5. Repeat until `|selected| == selected_k` (the field configured at
    ///    construction time).
    ///
    /// ## Why greedy forward (not exhaustive / beam / backward)?
    ///
    /// Exhaustive subset search is combinatorial in `full_dim`. Backward
    /// elimination costs O(full_dim^2) evaluations, each over the whole corpus.
    /// Forward greedy is O(full_dim × selected_k) full-corpus evaluations,
    /// which is the only tractable choice at SIFT1M selector-training scale
    /// (128 × 32 × 500 queries × 1000 corpus ≈ 2e9 inner ops). A beam search
    /// would multiply cost by the beam width for typically sub-1% gain at the
    /// recall levels we are measuring.
    ///
    /// # Arguments
    /// - `corpus`: training corpus (full-dim vectors) to evaluate retention on.
    ///   Must be disjoint from any evaluation corpus to avoid leakage.
    /// - `queries`: training queries (full-dim vectors). Disjoint from
    ///   evaluation queries.
    /// - `target_k`: the k in recall\@k that the selector optimizes for.
    /// - `candidate_pool`: how many ground-truth neighbors to materialize per
    ///   query. Must be ≥ `target_k`. Larger values only change the ground
    ///   truth if `target_k` falls outside the top-`target_k` band, so in
    ///   practice `candidate_pool == target_k` is fine; we take a larger
    ///   pool only so that ties on the boundary do not flip recall.
    ///
    /// # Returns
    /// `true` once the internal EML model has been retrained on the selected
    /// dims using the training-corpus pairs (so `selected_distance` still
    /// works). `false` on argument error (empty corpus / queries, k=0).
    pub fn train_for_retention(
        &mut self,
        corpus: &[Vec<f32>],
        queries: &[Vec<f32>],
        target_k: usize,
        candidate_pool: usize,
    ) -> bool {
        if corpus.is_empty() || queries.is_empty() || target_k == 0 {
            return false;
        }
        if candidate_pool < target_k {
            return false;
        }
        // Defensive: ensure all vectors are the right shape.
        for v in corpus.iter().chain(queries.iter()) {
            if v.len() != self.full_dim {
                return false;
            }
        }
        let k = self.selected_k.min(self.full_dim);
        if k == 0 {
            return false;
        }

        // Step 1: ground truth — top-`candidate_pool` corpus indices per query
        // under exact full-dim cosine.
        let pool = candidate_pool.min(corpus.len());
        let mut gt: Vec<Vec<usize>> = Vec::with_capacity(queries.len());
        for q in queries {
            let mut scored: Vec<(usize, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_distance_f32(q, v)))
                .collect();
            scored.sort_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            gt.push(scored.into_iter().take(pool).map(|(i, _)| i).collect());
        }
        // Truth sets for recall lookup use the top-target_k band.
        let gt_sets: Vec<std::collections::HashSet<usize>> = gt
            .iter()
            .map(|v| v.iter().copied().take(target_k).collect())
            .collect();

        // Step 2: greedy forward selection.
        let mut selected: Vec<usize> = Vec::with_capacity(k);
        let mut remaining: Vec<usize> = (0..self.full_dim).collect();

        while selected.len() < k && !remaining.is_empty() {
            let mut best_dim: Option<usize> = None;
            let mut best_recall: f32 = f32::NEG_INFINITY;

            for (pos, &cand) in remaining.iter().enumerate() {
                // trial = selected ∪ {cand}
                let mut trial = selected.clone();
                trial.push(cand);

                // Mean recall@target_k across training queries.
                let mut recall_sum = 0.0f32;
                for (qi, q) in queries.iter().enumerate() {
                    let mut scored: Vec<(usize, f32)> = corpus
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i, cosine_distance_selected(q, v, &trial)))
                        .collect();
                    // Partial top-target_k via full sort is acceptable at our
                    // corpus sizes (1k). select_nth_unstable would be faster
                    // but complicates tie handling.
                    scored.sort_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let set = &gt_sets[qi];
                    let hits = scored
                        .iter()
                        .take(target_k)
                        .filter(|(i, _)| set.contains(i))
                        .count();
                    recall_sum += hits as f32 / target_k as f32;
                }
                let recall = recall_sum / queries.len() as f32;

                if recall > best_recall {
                    best_recall = recall;
                    best_dim = Some(pos);
                }
            }

            match best_dim {
                Some(pos) => {
                    let chosen = remaining.swap_remove(pos);
                    selected.push(chosen);
                }
                None => break,
            }
        }

        selected.sort(); // cache-friendly access, same convention as `train`
        self.selected_dims = selected;

        // Step 3: retrain the internal EML model on the training-corpus pairs
        // using the chosen dims. This keeps `fast_distance` / `selected_distance`
        // functional after retention-training. We synthesize pairs from the
        // corpus itself (consistent with `train_and_build`).
        self.training_buffer.clear();
        for pair in corpus.chunks(2) {
            if pair.len() < 2 {
                break;
            }
            let d = cosine_distance_f32(&pair[0], &pair[1]);
            self.training_buffer
                .push((pair[0].clone(), pair[1].clone(), d));
        }
        self.model = EmlModel::new(3, self.selected_dims.len(), 1);
        for (a, b, dist) in &self.training_buffer {
            let features: Vec<f64> = self
                .selected_dims
                .iter()
                .map(|&d| (a[d] - b[d]).abs() as f64)
                .collect();
            self.model.record(&features, &[Some(*dist as f64)]);
        }
        let _ = self.model.train();
        self.trained = true;
        true
    }

    /// Serialize the model to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("EmlDistanceModel serialization should not fail")
    }

    /// Deserialize a model from JSON.
    pub fn from_json(json: &str) -> Option<Self> {
        serde_json::from_str(json).ok()
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_distance_identical_vectors() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let d = cosine_distance_f32(&v, &v);
        assert!(d.abs() < 1e-6, "identical vectors: got {d}");
    }

    #[test]
    fn cosine_distance_opposite_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let d = cosine_distance_f32(&a, &b);
        assert!((d - 2.0).abs() < 1e-6, "opposite vectors: got {d}");
    }

    #[test]
    fn cosine_distance_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let d = cosine_distance_f32(&a, &b);
        assert!((d - 1.0).abs() < 1e-6, "orthogonal vectors: got {d}");
    }

    #[test]
    fn cosine_distance_zero_vector() {
        let a = vec![1.0f32, 2.0, 3.0];
        let z = vec![0.0f32; 3];
        let d = cosine_distance_f32(&a, &z);
        assert!((d - 1.0).abs() < 1e-6, "zero vector: got {d}");
    }

    #[test]
    fn eml_distance_new_defaults() {
        let m = EmlDistanceModel::new(128, 16);
        assert!(!m.is_trained());
        assert_eq!(m.sample_count(), 0);
        assert!(m.selected_dims().is_empty());
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
            "untrained should fall back to cosine"
        );
    }

    #[test]
    fn eml_distance_record_increments() {
        let mut m = EmlDistanceModel::new(4, 2);
        assert_eq!(m.sample_count(), 0);
        m.record(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0], 0.5);
        assert_eq!(m.sample_count(), 1);
    }

    #[test]
    fn eml_distance_train_insufficient() {
        let mut m = EmlDistanceModel::new(4, 2);
        for i in 0..10 {
            let v = i as f32 / 10.0;
            m.record(&[v, v, v, v], &[1.0 - v, v, v, v], v);
        }
        assert!(!m.train());
    }

    #[test]
    fn eml_distance_train_with_data() {
        let dim = 8;
        let mut m = EmlDistanceModel::new(dim, 4);
        let mut rng = 42u64;
        for _ in 0..200 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let t = (rng >> 33) as f32 / (u32::MAX as f32);
            let mut a = vec![0.0f32; dim];
            let mut b = vec![0.0f32; dim];
            a[0] = t;
            a[1] = t * 0.5;
            b[0] = 1.0 - t;
            b[1] = (1.0 - t) * 0.5;
            for d in 2..dim {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng >> 33) as f32 / (u32::MAX as f32) * 0.01;
                a[d] = noise;
                b[d] = noise;
            }
            let exact = cosine_distance_f32(&a, &b);
            m.record(&a, &b, exact);
        }

        m.train();
        assert!(m.is_trained());
        assert_eq!(m.selected_dims().len(), 4);

        let a = vec![0.5f32, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.3f32, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let fast_d = m.fast_distance(&a, &b);
        assert!(fast_d.is_finite(), "got {fast_d}");
        assert!(fast_d >= 0.0, "got {fast_d}");
    }

    #[test]
    fn serialization_roundtrip() {
        let m = EmlDistanceModel::new(16, 4);
        let json = m.to_json();
        let m2 = EmlDistanceModel::from_json(&json).expect("should deserialize");
        assert_eq!(m.full_dim, m2.full_dim);
        assert_eq!(m.selected_k, m2.selected_k);
        assert_eq!(m.trained, m2.trained);
    }

    #[test]
    fn train_for_retention_selects_k_dims() {
        // Tiny synthetic test: 8-dim vectors where variance is concentrated
        // in dims 0..4. Retention selector should pick a subset of those.
        let dim = 8;
        let mut rng = 17u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (u32::MAX as f32) - 0.5
        };
        let mk = |n: usize, next: &mut dyn FnMut() -> f32| -> Vec<Vec<f32>> {
            (0..n)
                .map(|_| {
                    (0..dim)
                        .map(|d| if d < 4 { next() * 4.0 } else { next() * 0.1 })
                        .collect()
                })
                .collect()
        };
        let corpus = mk(120, &mut next);
        let queries = mk(20, &mut next);

        let mut m = EmlDistanceModel::new(dim, 3);
        let ok = m.train_for_retention(&corpus, &queries, 5, 10);
        assert!(ok);
        assert!(m.is_trained());
        assert_eq!(m.selected_dims().len(), 3);
    }
}
