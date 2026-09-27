//! Calibration layers (ADR-003 §3, §noul). Temperature scaling for the
//! `choice`/`score` distribution and Platt scaling for the `noul` head, both
//! fitted on a held-out calibration slice disjoint from the head's training
//! examples. Temperature never changes the argmax; it only makes `confidence`
//! honest. Everything here is deterministic: fixed grid / fixed iteration
//! count, no RNG, no time.

use crate::heads::logistic::sigmoid;
use crate::heads::probe::softmax;

/// Lowest temperature the fitter will return. A near-zero temperature would
/// make the softmax a hard argmax and destroy calibration; the floor keeps
/// `confidence` a probability.
pub(crate) const T_MIN: f32 = 0.1;
pub(crate) const T_MAX: f32 = 10.0;
/// Grid resolution for the 1-D temperature search (log-spaced, deterministic).
const T_GRID: usize = 120;

/// Fit a single temperature `T > 0` minimising the negative log-likelihood of
/// the true labels under `softmax(logits / T)`. A log-spaced grid search: the
/// NLL is smooth and unimodal in `log T`, so a fixed grid is both robust and
/// bit-for-bit reproducible. Returns `1.0` if there is nothing to fit.
pub(crate) fn fit_temperature(logits: &[Vec<f32>], labels: &[usize]) -> f32 {
    if logits.is_empty() || logits.len() != labels.len() {
        return 1.0;
    }
    let ln_min = T_MIN.ln();
    let ln_max = T_MAX.ln();
    let mut best_t = 1.0;
    let mut best_nll = f32::INFINITY;
    for i in 0..T_GRID {
        let frac = i as f32 / (T_GRID - 1) as f32;
        let t = (ln_min + frac * (ln_max - ln_min)).exp();
        let nll = mean_nll(logits, labels, t);
        if nll < best_nll {
            best_nll = nll;
            best_t = t;
        }
    }
    // Grid endpoints can float a hair past the bound (exp(ln(T_MIN)) < T_MIN);
    // clamp so the returned temperature is always a valid one.
    best_t.clamp(T_MIN, T_MAX)
}

fn mean_nll(logits: &[Vec<f32>], labels: &[usize], t: f32) -> f32 {
    let mut sum = 0.0f32;
    for (row, &label) in logits.iter().zip(labels) {
        let scaled: Vec<f32> = row.iter().map(|l| l / t).collect();
        let p = softmax(&scaled);
        let pl = p.get(label).copied().unwrap_or(0.0).max(1e-9);
        sum -= pl.ln();
    }
    sum / logits.len() as f32
}

/// Divide logits by the temperature. Monotone, so the argmax is preserved.
pub(crate) fn apply_temperature(logits: &[f32], t: f32) -> Vec<f32> {
    let t = t.max(T_MIN);
    logits.iter().map(|l| l / t).collect()
}

/// Platt scaling for the binary `noul` head: `p = sigmoid(a * raw + b)`, fitted
/// on `(raw_score, label)` pairs by deterministic gradient descent.
pub(crate) struct Platt {
    a: f32,
    b: f32,
}

impl Platt {
    pub(crate) fn parameters(&self) -> (f32, f32) {
        (self.a, self.b)
    }
    pub fn fit(scores: &[f32], labels: &[f32]) -> Self {
        let mut a = 1.0f32;
        let mut b = 0.0f32;
        let n = scores.len().max(1) as f32;
        for _ in 0..300 {
            let mut ga = 0.0f32;
            let mut gb = 0.0f32;
            for (s, y) in scores.iter().zip(labels) {
                let p = sigmoid(a * s + b);
                let err = p - y;
                ga += err * s;
                gb += err;
            }
            a -= 0.3 * ga / n;
            b -= 0.3 * gb / n;
        }
        Self { a, b }
    }

    pub fn apply(&self, raw: f32) -> f32 {
        sigmoid(self.a * raw + self.b)
    }
}

#[cfg(all(test, feature = "hash-embedder"))]
mod tests {
    use super::*;

    // Small deterministic LCG so "random" inputs carry no RandomState.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        }
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .fold(0, |best, (i, &x)| if x > v[best] { i } else { best })
    }

    #[test]
    fn temperature_never_changes_argmax() {
        let mut rng = Lcg(0x1234_5678);
        for _ in 0..50 {
            let logits: Vec<f32> = (0..8).map(|_| rng.next_f32() * 3.0).collect();
            let t = 0.1 + (rng.next_f32() + 1.0) * 2.5; // in [0.1, 5.1]
            let scaled = apply_temperature(&logits, t);
            assert_eq!(argmax(&logits), argmax(&scaled));
        }
    }

    #[test]
    fn fit_temperature_is_deterministic_and_bounded() {
        let logits = vec![
            vec![2.0, 0.0, -1.0],
            vec![0.0, 3.0, 0.0],
            vec![1.0, 1.0, 2.0],
        ];
        let labels = vec![0usize, 1, 2];
        let t1 = fit_temperature(&logits, &labels);
        let t2 = fit_temperature(&logits, &labels);
        assert_eq!(t1, t2);
        assert!((T_MIN..=T_MAX).contains(&t1));
    }

    #[test]
    fn fit_temperature_softens_overconfident_logits() {
        // Hugely separated logits but the labels disagree half the time -> the
        // fitter should raise T above 1 to stop asserting near-certainty.
        let logits = vec![
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
        ];
        let labels = vec![0usize, 1, 0, 1];
        let t = fit_temperature(&logits, &labels);
        assert!(t > 1.5, "expected softening, got T={t}");
    }

    #[test]
    fn platt_maps_scores_to_calibrated_probabilities() {
        let scores = vec![-3.0, -2.0, 2.0, 3.0];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let p = Platt::fit(&scores, &labels);
        assert!(p.apply(3.0) > 0.5);
        assert!(p.apply(-3.0) < 0.5);
    }
}
