//! Confidence signals and calibration.
//!
//! Adaptive routing is only trustworthy if the confidence driving it is
//! *calibrated* — a stated 0.8 should mean "right about 80% of the time".
//! This module derives a raw confidence from several signals, fits a histogram
//! calibrator (a reliability map) on a training split, and measures Expected
//! Calibration Error (ECE) and abstention precision/recall on the test split.

/// The signals that feed per-query confidence. Each is in `[0, 1]` except
/// `margin_ratio`, which is a normalized top-1/top-2 gap.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConfidenceSignals {
    /// Top fused score (RRF magnitude of the winner).
    pub top_score: f32,
    /// (top1 - top2) / top1 — how decisively the winner leads.
    pub margin_ratio: f32,
    /// Cross-lens min agreement of the winner over the consulted lenses.
    pub cross_lens_agreement: f32,
    /// Source density: best grounding-anchor weight on the winner.
    pub grounding: f32,
    /// Contradiction: the largest below-mean dissent across consulted lenses.
    pub contradiction: f32,
    /// Number of lenses consulted so far.
    pub lens_count: usize,
}

/// Fold the signals into a single raw confidence in `[0, 1]`.
///
/// This need only be *monotonically informative* — the [`Calibrator`] maps it
/// onto empirical accuracy. Agreement and margin push confidence up;
/// contradiction pushes it down; grounding is a mild bonus.
pub fn raw_confidence(s: &ConfidenceSignals) -> f32 {
    let z = 0.55 * s.cross_lens_agreement + 0.30 * s.margin_ratio.clamp(0.0, 1.0)
        + 0.15 * s.grounding
        - 0.40 * s.contradiction;
    z.clamp(0.0, 1.0)
}

/// A histogram (reliability-map) calibrator: bins raw confidence and maps each
/// bin to the empirical accuracy observed in training. Monotone, robust, and
/// interpretable — the reliability diagram made executable.
#[derive(Debug, Clone)]
pub struct Calibrator {
    /// Per-bin calibrated confidence (empirical accuracy).
    bins: Vec<f32>,
}

impl Calibrator {
    /// Identity calibrator (returns the raw value), used for the bootstrap pass.
    pub fn identity(n_bins: usize) -> Self {
        let n = n_bins.max(1);
        let bins = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
        Self { bins }
    }

    /// Fit on `(raw_confidence, was_correct)` samples. Empty bins fall back to
    /// the bin's midpoint so the map stays monotone and defined everywhere.
    pub fn fit(samples: &[(f32, bool)], n_bins: usize) -> Self {
        let n = n_bins.max(1);
        let mut hit = vec![0u32; n];
        let mut tot = vec![0u32; n];
        for &(raw, correct) in samples {
            let b = bin_index(raw, n);
            tot[b] += 1;
            if correct {
                hit[b] += 1;
            }
        }
        let bins = (0..n)
            .map(|b| {
                if tot[b] == 0 {
                    (b as f32 + 0.5) / n as f32
                } else {
                    hit[b] as f32 / tot[b] as f32
                }
            })
            .collect();
        Self { bins }
    }

    /// Map a raw confidence to its calibrated value.
    pub fn calibrate(&self, raw: f32) -> f32 {
        let b = bin_index(raw, self.bins.len());
        self.bins[b]
    }
}

#[inline]
fn bin_index(raw: f32, n_bins: usize) -> usize {
    let n = n_bins.max(1);
    ((raw.clamp(0.0, 0.999_999)) * n as f32) as usize
}

/// Expected Calibration Error over `(confidence, was_correct)` decisions.
///
/// Bins by confidence, then averages `|accuracy - mean_confidence|` weighted by
/// bin population. `0.0` is perfectly calibrated; lower is better.
pub fn expected_calibration_error(samples: &[(f32, bool)], n_bins: usize) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let n = n_bins.max(1);
    let mut sum_conf = vec![0.0f32; n];
    let mut hit = vec![0u32; n];
    let mut tot = vec![0u32; n];
    for &(conf, correct) in samples {
        let b = bin_index(conf, n);
        sum_conf[b] += conf;
        tot[b] += 1;
        if correct {
            hit[b] += 1;
        }
    }
    let total = samples.len() as f32;
    let mut ece = 0.0f32;
    for b in 0..n {
        if tot[b] == 0 {
            continue;
        }
        let acc = hit[b] as f32 / tot[b] as f32;
        let conf = sum_conf[b] / tot[b] as f32;
        ece += (tot[b] as f32 / total) * (acc - conf).abs();
    }
    ece
}

/// Abstention quality. `abstained` and `should_have_abstained` are aligned
/// per-query booleans (true = abstained / true = query was unanswerable).
#[derive(Debug, Clone, Copy)]
pub struct AbstentionQuality {
    pub precision: f32,
    pub recall: f32,
}

/// Precision = fraction of abstentions that were correct (query unanswerable).
/// Recall = fraction of unanswerable queries that were abstained on.
pub fn abstention_quality(abstained: &[bool], unanswerable: &[bool]) -> AbstentionQuality {
    let mut tp = 0u32; // abstained & unanswerable
    let mut abst = 0u32;
    let mut unans = 0u32;
    for (&a, &u) in abstained.iter().zip(unanswerable.iter()) {
        if a {
            abst += 1;
        }
        if u {
            unans += 1;
        }
        if a && u {
            tp += 1;
        }
    }
    AbstentionQuality {
        precision: if abst == 0 { 1.0 } else { tp as f32 / abst as f32 },
        recall: if unans == 0 { 1.0 } else { tp as f32 / unans as f32 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrator_maps_bins_to_empirical_accuracy() {
        // Low-raw samples are 20% correct; high-raw samples 90% correct.
        let mut samples = Vec::new();
        for i in 0..100 {
            samples.push((0.1, i < 20)); // 20% correct in low bin
            samples.push((0.9, i < 90)); // 90% correct in high bin
        }
        let c = Calibrator::fit(&samples, 10);
        assert!((c.calibrate(0.1) - 0.2).abs() < 0.02);
        assert!((c.calibrate(0.9) - 0.9).abs() < 0.02);
    }

    #[test]
    fn ece_zero_when_perfectly_calibrated() {
        // Confidence 0.7 that is correct exactly 70% of the time → ECE ≈ 0.
        let mut samples = Vec::new();
        for i in 0..100 {
            samples.push((0.7, i < 70));
        }
        assert!(expected_calibration_error(&samples, 10) < 0.02);
    }

    #[test]
    fn ece_high_when_overconfident() {
        // Confidence 0.95 but only 40% correct → large ECE.
        let mut samples = Vec::new();
        for i in 0..100 {
            samples.push((0.95, i < 40));
        }
        assert!(expected_calibration_error(&samples, 10) > 0.4);
    }

    #[test]
    fn abstention_precision_recall() {
        let abstained = [true, true, false, false];
        let unanswerable = [true, false, true, false];
        let q = abstention_quality(&abstained, &unanswerable);
        assert!((q.precision - 0.5).abs() < 1e-6); // 1 of 2 abstentions correct
        assert!((q.recall - 0.5).abs() < 1e-6); // caught 1 of 2 unanswerable
    }
}
