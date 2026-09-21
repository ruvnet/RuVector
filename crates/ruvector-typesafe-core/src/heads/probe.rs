//! Multinomial logistic regression ("linear probe") over frozen embeddings
//! (ADR-003 `choice` head). Full-batch gradient descent in plain Rust on
//! `Vec<f32>`, L2-regularised, deterministic (fixed iteration count, zero
//! init, examples consumed in stored order — no RNG, no time).

/// Hyper-parameters for the probe (see ADR-003: "a few hundred iterations,
/// milliseconds"). Learning rate is high because features are unit-norm and
/// the weights start at zero, so the gradient magnitude is ~1.
pub(crate) struct ProbeConfig {
    pub lr: f32,
    pub l2: f32,
    pub iters: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            lr: 0.8,
            l2: 1e-3,
            iters: 400,
        }
    }
}

/// A trained multinomial logistic head over `k` classes; `logits(x)[c]` is the
/// pre-softmax score for class `c`. Class indices follow the option-key order
/// the engine passed at training time, so the engine maps them back to keys.
pub(crate) struct MultiProbe {
    weights: Vec<Vec<f32>>, // K x D
    bias: Vec<f32>,         // K
}

impl MultiProbe {
    /// Train on `(embedding, class_index)` pairs over `k` classes. `dims` is
    /// the embedding width.
    pub fn train(examples: &[(Vec<f32>, usize)], k: usize, dims: usize, cfg: &ProbeConfig) -> Self {
        let mut weights = vec![vec![0.0f32; dims]; k];
        let mut bias = vec![0.0f32; k];
        let n = examples.len().max(1) as f32;

        for _ in 0..cfg.iters {
            let mut grad_w = vec![vec![0.0f32; dims]; k];
            let mut grad_b = vec![0.0f32; k];

            for (x, y) in examples {
                let logits: Vec<f32> = weights
                    .iter()
                    .zip(&bias)
                    .map(|(w, b)| b + dot(w, x))
                    .collect();
                let probs = softmax(&logits);
                for (c, (gw, gb)) in grad_w.iter_mut().zip(grad_b.iter_mut()).enumerate() {
                    let err = probs[c] - if c == *y { 1.0 } else { 0.0 };
                    *gb += err;
                    for (g, xi) in gw.iter_mut().zip(x) {
                        *g += err * xi;
                    }
                }
            }

            for (c, (w, b)) in weights.iter_mut().zip(bias.iter_mut()).enumerate() {
                *b -= cfg.lr * grad_b[c] / n;
                for (wi, gi) in w.iter_mut().zip(&grad_w[c]) {
                    let g = gi / n + cfg.l2 * *wi;
                    *wi -= cfg.lr * g;
                }
            }
        }

        Self { weights, bias }
    }

    /// Pre-softmax logits, one per class, in `classes` order.
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .zip(&self.bias)
            .map(|(w, b)| b + dot(w, x))
            .collect()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Numerically stable softmax over a small logit vector.
pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

#[cfg(all(test, feature = "hash-embedder"))]
mod tests {
    use super::*;

    #[test]
    fn separates_two_linearly_separable_classes() {
        // Class 0 has weight in dim 0, class 1 in dim 1.
        let e = |a: f32, b: f32| vec![a, b];
        let examples = vec![
            (e(1.0, 0.0), 0),
            (e(0.9, 0.1), 0),
            (e(0.95, 0.05), 0),
            (e(1.0, 0.02), 0),
            (e(0.0, 1.0), 1),
            (e(0.1, 0.9), 1),
            (e(0.05, 0.95), 1),
            (e(0.02, 1.0), 1),
        ];
        let probe = MultiProbe::train(&examples, 2, 2, &ProbeConfig::default());
        let la = probe.logits(&e(1.0, 0.0));
        assert!(la[0] > la[1], "class 0 should win for a-like input");
        let lb = probe.logits(&e(0.0, 1.0));
        assert!(lb[1] > lb[0], "class 1 should win for b-like input");
    }

    #[test]
    fn is_deterministic() {
        let examples = vec![(vec![1.0, 0.0], 0), (vec![0.0, 1.0], 1)];
        let cfg = ProbeConfig::default();
        let p1 = MultiProbe::train(&examples, 2, 2, &cfg);
        let p2 = MultiProbe::train(&examples, 2, 2, &cfg);
        assert_eq!(p1.logits(&[0.7, 0.3]), p2.logits(&[0.7, 0.3]));
    }
}
