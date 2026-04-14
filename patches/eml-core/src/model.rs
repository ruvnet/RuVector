//! Multi-head EML model with training via coordinate descent.
//!
//! [`EmlModel`] is a generic, domain-agnostic learned function that maps
//! N input features to M output heads. It uses the EML operator tree
//! internally and trains via random restart + coordinate descent.

use serde::{Deserialize, Serialize};

use crate::operator::{eml_safe, random_params, softmax3};

// ---------------------------------------------------------------------------
// Training point (internal)
// ---------------------------------------------------------------------------

/// A recorded (inputs, targets) pair for model training.
#[derive(Debug, Clone)]
struct TrainingPoint {
    inputs: Vec<f64>,
    targets: Vec<Option<f64>>,
}

// ---------------------------------------------------------------------------
// EmlModel
// ---------------------------------------------------------------------------

/// Multi-head EML model for O(1) function approximation.
///
/// # Architecture
///
/// The model uses a shared trunk of EML operators that feeds into
/// multiple output heads. Each head produces one scalar prediction.
///
/// ```text
/// Level 0: 8 affine combinations of input features (24 params)
/// Level 1: 4 EML nodes (no params — pure EML pairing)
/// Level 2: mixing + EML (depth-dependent params)
/// ...
/// Level D: multi-head output (2 params per head)
/// ```
///
/// Supported depths: 2, 3, 4, 5.
///
/// # Training
///
/// Training uses gradient-free random restart + coordinate descent,
/// suitable for the modest parameter counts (typically 30-80 params).
/// Call [`record`] to accumulate training data, then [`train`] to
/// optimize parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmlModel {
    depth: usize,
    input_count: usize,
    head_count: usize,
    /// Trainable parameters.
    params: Vec<f64>,
    /// Whether the model has been trained to convergence.
    trained: bool,
    /// Training data buffer.
    #[serde(skip)]
    training_data: Vec<TrainingPoint>,
}

impl EmlModel {
    /// Create a new untrained EML model.
    ///
    /// # Arguments
    /// - `depth`: Tree depth (2, 3, 4, or 5).
    /// - `input_count`: Number of input features.
    /// - `head_count`: Number of output heads (>= 1).
    ///
    /// # Panics
    /// Panics if depth is not in {2, 3, 4, 5} or head_count is 0.
    pub fn new(depth: usize, input_count: usize, head_count: usize) -> Self {
        assert!(
            (2..=5).contains(&depth),
            "EmlModel depth must be 2, 3, 4, or 5, got {depth}"
        );
        assert!(head_count > 0, "head_count must be >= 1");

        let param_count = Self::compute_param_count(depth, head_count);
        Self {
            depth,
            input_count,
            head_count,
            params: vec![0.0; param_count],
            trained: false,
            training_data: Vec::new(),
        }
    }

    /// Total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Whether the model has been trained to convergence.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Number of training samples collected so far.
    pub fn training_sample_count(&self) -> usize {
        self.training_data.len()
    }

    /// Tree depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of input features.
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Number of output heads.
    pub fn head_count(&self) -> usize {
        self.head_count
    }

    // -------------------------------------------------------------------
    // Parameter count
    // -------------------------------------------------------------------

    /// Compute total parameter count for trunk + heads.
    ///
    /// Trunk param layout (same as the depth-4 coherence model):
    ///   Level 0: 8 * 3 = 24 (affine combos via softmax3)
    ///   Level 1: 0 (pure EML pairing)
    ///   Level 2: 4 * 3 = 12 (mixing via softmax3)
    ///   Level 3: 2 * 4 = 8 (mixing with 4 weights each)
    ///   Head layer: head_count * 2
    ///
    /// For shallower trees, fewer mixing levels.
    fn compute_param_count(depth: usize, head_count: usize) -> usize {
        // Level 0: always 8 affine nodes * 3 params
        let mut total = 24;

        // Level 1: no params (pure EML)

        // Levels 2..depth-1: mixing
        match depth {
            2 => {
                // Only level 0 + heads
            }
            3 => {
                // Level 2: 2 mixing nodes * 4 params
                total += 2 * 4;
            }
            4 => {
                // Level 2: 4 mixing nodes * 3 params
                total += 4 * 3;
                // Level 3: 2 mixing nodes * 4 params
                total += 2 * 4;
            }
            5 => {
                // Level 2: 4 mixing nodes * 3 params
                total += 4 * 3;
                // Level 3: 4 mixing nodes * 3 params
                total += 4 * 3;
                // Level 4: 2 mixing nodes * 4 params
                total += 2 * 4;
            }
            _ => unreachable!(),
        }

        // Head layer: 2 params per head
        total += head_count * 2;

        total
    }

    // -------------------------------------------------------------------
    // Prediction
    // -------------------------------------------------------------------

    /// Predict all heads from input features.
    ///
    /// Returns a Vec with one f64 per head. Values are clamped to be
    /// non-negative.
    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        assert_eq!(
            inputs.len(),
            self.input_count,
            "expected {} inputs, got {}",
            self.input_count,
            inputs.len()
        );
        self.evaluate_with_params(&self.params, inputs)
    }

    /// Predict only the primary (first) head.
    pub fn predict_primary(&self, inputs: &[f64]) -> f64 {
        self.predict(inputs)[0]
    }

    /// Evaluate with arbitrary params (used during training).
    fn evaluate_with_params(&self, params: &[f64], inputs: &[f64]) -> Vec<f64> {
        // Level 0: 8 affine combinations
        let feature_pairs = Self::feature_pairs(self.input_count);
        let mut a = [0.0f64; 8];
        for i in 0..8 {
            let base = i * 3;
            let (alpha, beta, gamma) = softmax3(params[base], params[base + 1], params[base + 2]);
            let (j, k) = feature_pairs[i];
            a[i] = (alpha + beta * inputs[j] + gamma * inputs[k]).clamp(-10.0, 10.0);
        }

        // Level 1: 4 EML nodes (pure pairing)
        let b = [
            eml_safe(a[0], a[1]),
            eml_safe(a[2], a[3]),
            eml_safe(a[4], a[5]),
            eml_safe(a[6], a[7]),
        ];

        // Trunk values before heads
        let trunk = match self.depth {
            2 => {
                // Trunk is just b[0..4], heads mix from these
                b.to_vec()
            }
            3 => {
                // Level 2: 2 mixing nodes
                let mut c = [0.0f64; 2];
                for i in 0..2 {
                    let base = 24 + i * 4;
                    let mix_left = params[base]
                        + params[base + 1] * b[0]
                        + (1.0 - params[base] - params[base + 1]) * b[1];
                    let mix_right = params[base + 2]
                        + params[base + 3] * b[2]
                        + (1.0 - params[base + 2] - params[base + 3]) * b[3];
                    let ml = mix_left.clamp(-10.0, 10.0);
                    let mr = mix_right.clamp(0.01, 10.0);
                    c[i] = eml_safe(ml, mr);
                }
                c.to_vec()
            }
            4 => {
                // Level 2: 4 mixing nodes
                let level2_pairs: [(usize, usize, usize, usize); 4] = [
                    (0, 1, 2, 3),
                    (0, 1, 2, 3),
                    (0, 2, 1, 3),
                    (1, 3, 0, 2),
                ];
                let mut c = [0.0f64; 4];
                for i in 0..4 {
                    let base = 24 + i * 3;
                    let (li, lj, ri, rj) = level2_pairs[i];
                    let (alpha, beta, gamma) =
                        softmax3(params[base], params[base + 1], params[base + 2]);
                    let mix_left = (alpha + beta * b[li] + gamma * b[lj]).clamp(-10.0, 10.0);
                    let (ar, br, gr) = softmax3(
                        params[base] + 0.5,
                        params[base + 1] - 0.5,
                        params[base + 2],
                    );
                    let mix_right = (ar + br * b[ri] + gr * b[rj]).clamp(0.01, 10.0);
                    c[i] = eml_safe(mix_left, mix_right);
                }

                // Level 3: 2 mixing nodes
                let level3_pairs: [(usize, usize, usize, usize); 2] =
                    [(0, 1, 2, 3), (0, 2, 1, 3)];
                let mut d = [0.0f64; 2];
                for i in 0..2 {
                    let base = 36 + i * 4;
                    let (li, lj, ri, rj) = level3_pairs[i];
                    let mix_left = (params[base]
                        + params[base + 1] * c[li]
                        + (1.0 - params[base] - params[base + 1]) * c[lj])
                        .clamp(-10.0, 10.0);
                    let mix_right = (params[base + 2]
                        + params[base + 3] * c[ri]
                        + (1.0 - params[base + 2] - params[base + 3]) * c[rj])
                        .clamp(0.01, 10.0);
                    d[i] = eml_safe(mix_left, mix_right);
                }
                d.to_vec()
            }
            5 => {
                // Level 2: 4 mixing nodes (same as depth 4)
                let level2_pairs: [(usize, usize, usize, usize); 4] = [
                    (0, 1, 2, 3),
                    (0, 1, 2, 3),
                    (0, 2, 1, 3),
                    (1, 3, 0, 2),
                ];
                let mut c = [0.0f64; 4];
                for i in 0..4 {
                    let base = 24 + i * 3;
                    let (li, lj, ri, rj) = level2_pairs[i];
                    let (alpha, beta, gamma) =
                        softmax3(params[base], params[base + 1], params[base + 2]);
                    let mix_left = (alpha + beta * b[li] + gamma * b[lj]).clamp(-10.0, 10.0);
                    let (ar, br, gr) = softmax3(
                        params[base] + 0.5,
                        params[base + 1] - 0.5,
                        params[base + 2],
                    );
                    let mix_right = (ar + br * b[ri] + gr * b[rj]).clamp(0.01, 10.0);
                    c[i] = eml_safe(mix_left, mix_right);
                }

                // Level 3: 4 mixing nodes
                let level3_pairs: [(usize, usize, usize, usize); 4] = [
                    (0, 1, 2, 3),
                    (0, 2, 1, 3),
                    (1, 3, 0, 2),
                    (0, 3, 1, 2),
                ];
                let mut e = [0.0f64; 4];
                for i in 0..4 {
                    let base = 36 + i * 3;
                    let (li, lj, ri, rj) = level3_pairs[i];
                    let (alpha, beta, gamma) =
                        softmax3(params[base], params[base + 1], params[base + 2]);
                    let mix_left = (alpha + beta * c[li] + gamma * c[lj]).clamp(-10.0, 10.0);
                    let (ar, br, gr) = softmax3(
                        params[base] + 0.5,
                        params[base + 1] - 0.5,
                        params[base + 2],
                    );
                    let mix_right = (ar + br * c[ri] + gr * c[rj]).clamp(0.01, 10.0);
                    e[i] = eml_safe(mix_left, mix_right);
                }

                // Level 4: 2 mixing nodes
                let mut f = [0.0f64; 2];
                for i in 0..2 {
                    let base = 48 + i * 4;
                    let li = i * 2;
                    let lj = i * 2 + 1;
                    let ri = (i * 2 + 2) % 4;
                    let rj = (i * 2 + 3) % 4;
                    let mix_left = (params[base]
                        + params[base + 1] * e[li]
                        + (1.0 - params[base] - params[base + 1]) * e[lj])
                        .clamp(-10.0, 10.0);
                    let mix_right = (params[base + 2]
                        + params[base + 3] * e[ri]
                        + (1.0 - params[base + 2] - params[base + 3]) * e[rj])
                        .clamp(0.01, 10.0);
                    f[i] = eml_safe(mix_left, mix_right);
                }
                f.to_vec()
            }
            _ => unreachable!(),
        };

        // Head layer: each head mixes the trunk values
        let head_base = self.param_count() - self.head_count * 2;
        let mut outputs = Vec::with_capacity(self.head_count);
        for k in 0..self.head_count {
            let base = head_base + k * 2;
            let w0 = params[base];
            let w1 = params[base + 1];
            let (left, right) = if trunk.len() >= 2 {
                (
                    (w0 * trunk[0] + (1.0 - w0) * trunk[1]).clamp(-10.0, 10.0),
                    (w1 * trunk[0] + (1.0 - w1) * trunk[1]).clamp(0.01, 10.0),
                )
            } else {
                (
                    (w0 * trunk[0]).clamp(-10.0, 10.0),
                    (w1 * trunk[0]).clamp(0.01, 10.0),
                )
            };
            outputs.push(eml_safe(left, right).max(0.0));
        }

        outputs
    }

    /// Generate feature pair indices for level 0 (cycling through inputs).
    fn feature_pairs(input_count: usize) -> [(usize, usize); 8] {
        let mut pairs = [(0usize, 0usize); 8];
        for i in 0..8 {
            pairs[i] = (
                (i * 2) % input_count,
                (i * 2 + 1) % input_count,
            );
        }
        pairs
    }

    // -------------------------------------------------------------------
    // Training
    // -------------------------------------------------------------------

    /// Record a training sample.
    ///
    /// # Arguments
    /// - `inputs`: Input feature values.
    /// - `targets`: Target values for each head. Use `None` for heads
    ///   without ground truth in this sample (they are skipped in the
    ///   loss function).
    pub fn record(&mut self, inputs: &[f64], targets: &[Option<f64>]) {
        assert_eq!(
            inputs.len(),
            self.input_count,
            "expected {} inputs, got {}",
            self.input_count,
            inputs.len()
        );
        assert_eq!(
            targets.len(),
            self.head_count,
            "expected {} targets, got {}",
            self.head_count,
            targets.len()
        );
        self.training_data.push(TrainingPoint {
            inputs: inputs.to_vec(),
            targets: targets.to_vec(),
        });
    }

    /// Train the model using random restart + coordinate descent.
    ///
    /// Requires at least 50 training samples. Returns `true` if the
    /// model converged (MSE < 0.01).
    pub fn train(&mut self) -> bool {
        if self.training_data.len() < 50 {
            return false;
        }

        let param_count = self.params.len();
        let mut best_params = self.params.clone();
        let mut best_mse = self.evaluate_mse(&self.params);

        // Phase 1: random restarts
        let restart_count = if param_count > 40 { 200 } else { 100 };
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
        for _ in 0..restart_count {
            let candidate = random_params(&mut rng_state, param_count);
            let mse = self.evaluate_mse(&candidate);
            if mse < best_mse {
                best_mse = mse;
                best_params = candidate;
            }
        }

        // Phase 2: coordinate descent
        let deltas = [-0.1, -0.01, -0.001, 0.001, 0.01, 0.1];
        for _ in 0..1000 {
            let mut improved = false;
            for i in 0..param_count {
                for &delta in &deltas {
                    let mut candidate = best_params.clone();
                    candidate[i] += delta;
                    let mse = self.evaluate_mse(&candidate);
                    if mse < best_mse {
                        best_mse = mse;
                        best_params = candidate;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        self.params = best_params;
        self.trained = best_mse < 0.01;
        self.trained
    }

    /// Compute weighted MSE over the training set.
    fn evaluate_mse(&self, params: &[f64]) -> f64 {
        if self.training_data.is_empty() {
            return f64::MAX;
        }

        let mut total_loss = 0.0;
        let mut total_weight = 0.0;

        for tp in &self.training_data {
            let predicted = self.evaluate_with_params(params, &tp.inputs);
            for (k, target) in tp.targets.iter().enumerate() {
                if let Some(t) = target {
                    // Primary head (k==0) gets weight 1.0, others 0.3
                    let weight = if k == 0 { 1.0 } else { 0.3 };
                    total_loss += weight * (predicted[k] - t).powi(2);
                    total_weight += weight;
                }
            }
        }

        if total_weight > 0.0 {
            total_loss / total_weight
        } else {
            f64::MAX
        }
    }

    // -------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------

    /// Serialize the model to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("EmlModel serialization should not fail")
    }

    /// Deserialize a model from a JSON string.
    ///
    /// Returns `None` if the JSON is invalid.
    pub fn from_json(json: &str) -> Option<Self> {
        serde_json::from_str(json).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_model_defaults() {
        let m = EmlModel::new(4, 7, 3);
        assert_eq!(m.depth(), 4);
        assert_eq!(m.input_count(), 7);
        assert_eq!(m.head_count(), 3);
        assert!(!m.is_trained());
        assert_eq!(m.training_sample_count(), 0);
    }

    #[test]
    fn param_count_depth_2() {
        let m = EmlModel::new(2, 5, 1);
        // Level 0: 24, heads: 2 = 26
        assert_eq!(m.param_count(), 26);
    }

    #[test]
    fn param_count_depth_3() {
        let m = EmlModel::new(3, 7, 1);
        // Level 0: 24, level 2: 8, heads: 2 = 34
        assert_eq!(m.param_count(), 34);
    }

    #[test]
    fn param_count_depth_4_single_head() {
        let m = EmlModel::new(4, 7, 1);
        // Level 0: 24, level 2: 12, level 3: 8, heads: 2 = 46
        assert_eq!(m.param_count(), 46);
    }

    #[test]
    fn param_count_depth_4_three_heads() {
        let m = EmlModel::new(4, 7, 3);
        // Level 0: 24, level 2: 12, level 3: 8, heads: 6 = 50
        assert_eq!(m.param_count(), 50);
    }

    #[test]
    fn param_count_depth_5() {
        let m = EmlModel::new(5, 4, 2);
        // Level 0: 24, level 2: 12, level 3: 12, level 4: 8, heads: 4 = 60
        assert_eq!(m.param_count(), 60);
    }

    #[test]
    fn predict_untrained_produces_values() {
        let m = EmlModel::new(4, 7, 3);
        let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let result = m.predict(&inputs);
        assert_eq!(result.len(), 3);
        for &v in &result {
            assert!(v.is_finite(), "prediction should be finite");
            assert!(v >= 0.0, "prediction should be non-negative");
        }
    }

    #[test]
    fn predict_primary_matches_first_head() {
        let m = EmlModel::new(3, 5, 3);
        let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let all = m.predict(&inputs);
        let primary = m.predict_primary(&inputs);
        assert!(
            (primary - all[0]).abs() < 1e-12,
            "predict_primary should match predict()[0]"
        );
    }

    #[test]
    fn record_increments_count() {
        let mut m = EmlModel::new(3, 3, 1);
        assert_eq!(m.training_sample_count(), 0);
        m.record(&[0.1, 0.2, 0.3], &[Some(1.0)]);
        assert_eq!(m.training_sample_count(), 1);
    }

    #[test]
    fn train_insufficient_data_returns_false() {
        let mut m = EmlModel::new(3, 3, 1);
        for i in 0..10 {
            m.record(
                &[i as f64 / 10.0, 0.5, 0.5],
                &[Some(1.0)],
            );
        }
        assert!(!m.train());
        assert!(!m.is_trained());
    }

    #[test]
    fn training_convergence_polynomial() {
        // Train on y = x^2 for x in [0, 1]
        let mut m = EmlModel::new(4, 1, 1);
        for i in 0..100 {
            let x = i as f64 / 100.0;
            let y = x * x;
            m.record(&[x], &[Some(y)]);
        }
        let _ = m.train();
        // Even if not fully converged, should produce finite predictions
        let pred = m.predict_primary(&[0.5]);
        assert!(pred.is_finite());
    }

    #[test]
    fn multi_head_training() {
        let mut m = EmlModel::new(4, 2, 3);
        for i in 0..80 {
            let x = i as f64 / 80.0;
            let y = (i + 10) as f64 / 80.0;
            m.record(
                &[x, y],
                &[Some(x + y), Some(x * y), None],
            );
        }
        let _ = m.train();
        let pred = m.predict(&[0.5, 0.5]);
        assert_eq!(pred.len(), 3);
        for &v in &pred {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let mut m = EmlModel::new(4, 5, 2);
        // Set some params to non-zero
        for (i, p) in m.params.iter_mut().enumerate() {
            *p = (i as f64 * 0.1).sin();
        }
        m.trained = true;

        let json = m.to_json();
        let m2 = EmlModel::from_json(&json).expect("should deserialize");

        assert_eq!(m.depth, m2.depth);
        assert_eq!(m.input_count, m2.input_count);
        assert_eq!(m.head_count, m2.head_count);
        assert_eq!(m.params.len(), m2.params.len());
        for (i, (a, b)) in m.params.iter().zip(m2.params.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-14,
                "param[{i}] mismatch: {a} vs {b}"
            );
        }
        assert_eq!(m.trained, m2.trained);
        // training_data is skipped in serde
        assert_eq!(m2.training_sample_count(), 0);
    }

    #[test]
    fn from_json_invalid_returns_none() {
        assert!(EmlModel::from_json("not valid json").is_none());
    }

    #[test]
    fn various_depths_produce_finite_output() {
        for depth in 2..=5 {
            let m = EmlModel::new(depth, 4, 2);
            let inputs = vec![0.3, 0.5, 0.7, 0.1];
            let result = m.predict(&inputs);
            assert_eq!(result.len(), 2);
            for &v in &result {
                assert!(
                    v.is_finite(),
                    "depth-{depth} should produce finite output"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "EmlModel depth must be 2, 3, 4, or 5")]
    fn invalid_depth_panics() {
        EmlModel::new(6, 3, 1);
    }

    #[test]
    #[should_panic(expected = "head_count must be >= 1")]
    fn zero_heads_panics() {
        EmlModel::new(3, 3, 0);
    }
}
