//! Binary logistic head for `noul` (ADR-003): a probability for a predicate,
//! trained on labelled positive/negative examples with a class-balanced loss.
//! Deterministic full-batch gradient descent; no RNG, no time. Platt scaling
//! lives in `calibration`; this module only produces the raw logit.

/// Hyper-parameters for the binary head.
pub(crate) struct LogisticConfig {
    pub lr: f32,
    pub l2: f32,
    pub iters: usize,
}

impl Default for LogisticConfig {
    fn default() -> Self {
        Self {
            lr: 0.8,
            l2: 1e-3,
            iters: 400,
        }
    }
}

/// A trained binary logistic head. `raw(x)` is the pre-sigmoid score; the
/// engine maps it to a probability (directly, or through Platt scaling when
/// a calibration slice is available).
pub(crate) struct BinaryLogistic {
    weights: Vec<f32>,
    bias: f32,
}

impl BinaryLogistic {
    /// Train on `(embedding, label)` with `label` in `{0.0, 1.0}`. The loss is
    /// class-balanced: each class contributes equally regardless of its count,
    /// so a skewed predicate does not collapse to the majority.
    pub fn train(examples: &[(Vec<f32>, f32)], dims: usize, cfg: &LogisticConfig) -> Self {
        let mut weights = vec![0.0f32; dims];
        let mut bias = 0.0f32;

        let n_pos = examples.iter().filter(|(_, y)| *y >= 0.5).count().max(1) as f32;
        let n_neg = examples.iter().filter(|(_, y)| *y < 0.5).count().max(1) as f32;
        let total = examples.len().max(1) as f32;
        // Weights sum to `total` overall, split evenly across the two classes.
        let w_pos = total / (2.0 * n_pos);
        let w_neg = total / (2.0 * n_neg);

        for _ in 0..cfg.iters {
            let mut grad_w = vec![0.0f32; dims];
            let mut grad_b = 0.0f32;

            for (x, y) in examples {
                let p = sigmoid(bias + dot(&weights, x));
                let sample_w = if *y >= 0.5 { w_pos } else { w_neg };
                let err = sample_w * (p - *y);
                grad_b += err;
                for (g, xi) in grad_w.iter_mut().zip(x) {
                    *g += err * xi;
                }
            }

            bias -= cfg.lr * grad_b / total;
            for (w, g) in weights.iter_mut().zip(&grad_w) {
                let gg = g / total + cfg.l2 * *w;
                *w -= cfg.lr * gg;
            }
        }

        Self { weights, bias }
    }

    /// Pre-sigmoid score `w·x + b`.
    pub fn raw(&self, x: &[f32]) -> f32 {
        self.bias + dot(&self.weights, x)
    }

    /// Probability with no calibration layer (used when a Platt slice is absent).
    pub fn prob(&self, x: &[f32]) -> f32 {
        sigmoid(self.raw(x))
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

#[cfg(all(test, feature = "hash-embedder"))]
mod tests {
    use super::*;

    #[test]
    fn learns_a_separable_predicate() {
        let examples = vec![
            (vec![1.0, 0.0], 1.0),
            (vec![0.9, 0.1], 1.0),
            (vec![0.0, 1.0], 0.0),
            (vec![0.1, 0.9], 0.0),
        ];
        let m = BinaryLogistic::train(&examples, 2, &LogisticConfig::default());
        assert!(m.prob(&[1.0, 0.0]) > 0.5);
        assert!(m.prob(&[0.0, 1.0]) < 0.5);
    }

    #[test]
    fn class_balance_survives_imbalance() {
        // One positive, many negatives: a balanced loss must still lift the
        // positive above 0.5.
        let mut examples = vec![(vec![1.0, 0.0], 1.0)];
        for _ in 0..20 {
            examples.push((vec![0.0, 1.0], 0.0));
        }
        let m = BinaryLogistic::train(&examples, 2, &LogisticConfig::default());
        assert!(m.prob(&[1.0, 0.0]) > 0.5);
    }
}
