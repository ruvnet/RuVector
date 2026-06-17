//! Three drift detector variants implementing [`DriftDetector`].
//!
//! ## CentroidDrift (baseline)
//! Computes the L2 distance between the centroid of the baseline window and
//! the centroid of the latest window. Fast but insensitive to distribution
//! shape; a bimodal shift can have a stable centroid yet very different support.
//!
//! ## PsiDrift (alternative A)
//! Builds a cosine-similarity histogram for each window relative to the window
//! centroid, then computes the Population Stability Index (PSI) between the
//! baseline and latest histograms. PSI is the standard monitoring statistic for
//! input-distribution shift in production ML systems:
//! - PSI < 0.10 → stable
//! - 0.10 ≤ PSI < 0.25 → warning
//! - PSI ≥ 0.25 → significant drift
//!
//! ## CoherenceDrift (alternative B)
//! Combines PSI with the *change in intra-window coherence* (mean pairwise
//! cosine similarity). A cluster that fragments over time loses coherence even
//! if the centroid stays fixed. The combined score is:
//!   combined = (1 - coherence_weight) * PSI_score + coherence_weight * |Δcoherence|

use crate::{
    centroid, cosine, cosine_histogram, l2_sq, mean_pairwise_cosine, psi, DriftConfig,
    DriftDetector, DriftReport, VectorWindow,
};
use std::collections::HashMap;

// ── CentroidDrift ──────────────────────────────────────────────────────────

/// Baseline detector: measures L2 shift between window centroids.
pub struct CentroidDrift {
    cfg: DriftConfig,
    baseline: Option<VectorWindow>,
    latest: Option<VectorWindow>,
}

impl CentroidDrift {
    pub fn new(cfg: DriftConfig) -> Self {
        Self {
            cfg,
            baseline: None,
            latest: None,
        }
    }
}

impl DriftDetector for CentroidDrift {
    fn name(&self) -> &'static str {
        "CentroidDrift"
    }

    fn add_window(&mut self, window_id: u64, vectors: &[Vec<f32>]) {
        let w = VectorWindow::new(window_id, vectors.to_vec());
        if self.baseline.is_none() {
            self.baseline = Some(w);
        } else {
            self.latest = Some(w);
        }
    }

    fn detect(&self) -> DriftReport {
        let (Some(base), Some(latest)) = (self.baseline.as_ref(), self.latest.as_ref()) else {
            return DriftReport::insufficient(0);
        };
        if base.len() < self.cfg.min_window_size || latest.len() < self.cfg.min_window_size {
            return DriftReport::insufficient(latest.window_id);
        }
        let bc = centroid(&base.vectors);
        let lc = centroid(&latest.vectors);
        let shift = l2_sq(&bc, &lc).sqrt();
        // Normalise by sqrt(D) so the threshold is dimension-independent
        let dim = bc.len().max(1) as f32;
        let norm_shift = shift / dim.sqrt();
        let mut details = HashMap::new();
        details.insert("centroid_l2".to_string(), shift);
        details.insert("centroid_l2_norm".to_string(), norm_shift);
        let threshold = self.cfg.centroid_threshold;
        if norm_shift >= threshold {
            DriftReport::drifted(
                latest.window_id,
                norm_shift,
                details,
                norm_shift >= 2.0 * threshold,
            )
        } else {
            DriftReport::stable(latest.window_id, norm_shift)
        }
    }
}

// ── PsiDrift ────────────────────────────────────────────────────────────────

/// Alternative A: Population Stability Index on cosine-similarity histograms.
pub struct PsiDrift {
    cfg: DriftConfig,
    baseline: Option<VectorWindow>,
    latest: Option<VectorWindow>,
}

impl PsiDrift {
    pub fn new(cfg: DriftConfig) -> Self {
        Self {
            cfg,
            baseline: None,
            latest: None,
        }
    }

    /// Compute cosine similarities of each vector in `w` relative to a fixed
    /// `reference` anchor (the baseline centroid). Using a shared reference
    /// instead of each window's own centroid means PSI detects both directional
    /// shift AND spread increase — both are meaningful agent-memory drift events.
    fn cosines_to_reference(w: &VectorWindow, reference: &[f32]) -> Vec<f32> {
        if reference.is_empty() {
            return Vec::new();
        }
        w.vectors.iter().map(|v| cosine(v, reference)).collect()
    }
}

impl DriftDetector for PsiDrift {
    fn name(&self) -> &'static str {
        "PsiDrift"
    }

    fn add_window(&mut self, window_id: u64, vectors: &[Vec<f32>]) {
        let w = VectorWindow::new(window_id, vectors.to_vec());
        if self.baseline.is_none() {
            self.baseline = Some(w);
        } else {
            self.latest = Some(w);
        }
    }

    fn detect(&self) -> DriftReport {
        let (Some(base), Some(latest)) = (self.baseline.as_ref(), self.latest.as_ref()) else {
            return DriftReport::insufficient(0);
        };
        if base.len() < self.cfg.min_window_size || latest.len() < self.cfg.min_window_size {
            return DriftReport::insufficient(latest.window_id);
        }
        // Anchor: baseline centroid shared by both histograms.
        let anchor = centroid(&base.vectors);
        let b_cos = Self::cosines_to_reference(base, &anchor);
        let l_cos = Self::cosines_to_reference(latest, &anchor);
        let b_hist = cosine_histogram(&b_cos, self.cfg.psi_buckets);
        let l_hist = cosine_histogram(&l_cos, self.cfg.psi_buckets);
        let psi_val = psi(&b_hist, &l_hist);
        let mut details = HashMap::new();
        details.insert("psi".to_string(), psi_val);
        let threshold = self.cfg.psi_threshold;
        if psi_val >= threshold {
            DriftReport::drifted(
                latest.window_id,
                psi_val,
                details,
                psi_val >= 2.0 * threshold,
            )
        } else {
            DriftReport::stable(latest.window_id, psi_val)
        }
    }
}

// ── CoherenceDrift ──────────────────────────────────────────────────────────

/// Alternative B: PSI combined with change in intra-window coherence.
///
/// Intra-coherence = mean pairwise cosine similarity across all pairs in the
/// window. A fragmenting cluster shows low coherence even if PSI is moderate.
/// Combined score = (1-w)*PSI + w*|Δcoherence|, where w = `coherence_weight`.
///
/// O(N²) per window — suitable for windows up to ~500 vectors; use sub-sampling
/// for larger windows in production.
pub struct CoherenceDrift {
    cfg: DriftConfig,
    baseline: Option<(VectorWindow, f32)>,
    latest: Option<(VectorWindow, f32)>,
}

impl CoherenceDrift {
    pub fn new(cfg: DriftConfig) -> Self {
        Self {
            cfg,
            baseline: None,
            latest: None,
        }
    }
}

impl DriftDetector for CoherenceDrift {
    fn name(&self) -> &'static str {
        "CoherenceDrift"
    }

    fn add_window(&mut self, window_id: u64, vectors: &[Vec<f32>]) {
        // Sub-sample to 200 vectors for O(N²) coherence cost
        let sample: Vec<Vec<f32>> = if vectors.len() > 200 {
            vectors
                .iter()
                .step_by(vectors.len() / 200 + 1)
                .cloned()
                .collect()
        } else {
            vectors.to_vec()
        };
        let coh = mean_pairwise_cosine(&sample);
        let w = VectorWindow::new(window_id, vectors.to_vec());
        if self.baseline.is_none() {
            self.baseline = Some((w, coh));
        } else {
            self.latest = Some((w, coh));
        }
    }

    fn detect(&self) -> DriftReport {
        let (Some((base, base_coh)), Some((latest, lat_coh))) =
            (self.baseline.as_ref(), self.latest.as_ref())
        else {
            return DriftReport::insufficient(0);
        };
        if base.len() < self.cfg.min_window_size || latest.len() < self.cfg.min_window_size {
            return DriftReport::insufficient(latest.window_id);
        }

        // PSI component — use baseline centroid as shared anchor.
        let anchor = centroid(&base.vectors);
        let b_cos: Vec<f32> = base.vectors.iter().map(|v| cosine(v, &anchor)).collect();
        let l_cos: Vec<f32> = latest.vectors.iter().map(|v| cosine(v, &anchor)).collect();
        let b_hist = cosine_histogram(&b_cos, self.cfg.psi_buckets);
        let l_hist = cosine_histogram(&l_cos, self.cfg.psi_buckets);
        let psi_val = psi(&b_hist, &l_hist);

        // Coherence delta component
        let coh_delta = (lat_coh - base_coh).abs();

        let w = self.cfg.coherence_weight.clamp(0.0, 1.0);
        let combined = (1.0 - w) * psi_val + w * coh_delta;

        let mut details = HashMap::new();
        details.insert("psi".to_string(), psi_val);
        details.insert("coherence_baseline".to_string(), *base_coh);
        details.insert("coherence_latest".to_string(), *lat_coh);
        details.insert("coherence_delta".to_string(), coh_delta);
        details.insert("combined_score".to_string(), combined);

        // Threshold: PSI scale (0.25) weighted
        let effective_threshold = (1.0 - w) * self.cfg.psi_threshold + w * self.cfg.psi_threshold;
        if combined >= effective_threshold {
            DriftReport::drifted(
                latest.window_id,
                combined,
                details,
                combined >= 2.0 * effective_threshold,
            )
        } else {
            DriftReport::stable(latest.window_id, combined)
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn gaussian_cluster(n: usize, d: usize, mean: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| mean + rng.gen_range(-0.3f32..0.3)).collect())
            .collect()
    }

    // ── CentroidDrift tests ──────────────────────────────────────────────

    #[test]
    fn centroid_no_drift_same_distribution() {
        let cfg = DriftConfig::default();
        let mut det = CentroidDrift::new(cfg);
        det.add_window(0, &gaussian_cluster(200, 64, 0.0, 42));
        det.add_window(1, &gaussian_cluster(200, 64, 0.0, 43));
        let r = det.detect();
        assert!(
            !r.is_drifted,
            "same cluster → no drift, score={}",
            r.drift_score
        );
    }

    #[test]
    fn centroid_detects_large_shift() {
        let cfg = DriftConfig::default();
        let mut det = CentroidDrift::new(cfg);
        det.add_window(0, &gaussian_cluster(200, 64, 0.0, 42));
        det.add_window(1, &gaussian_cluster(200, 64, 10.0, 99));
        let r = det.detect();
        assert!(
            r.is_drifted,
            "large mean shift must be flagged, score={}",
            r.drift_score
        );
    }

    #[test]
    fn centroid_insufficient_windows() {
        let cfg = DriftConfig::default();
        let mut det = CentroidDrift::new(cfg);
        det.add_window(0, &gaussian_cluster(200, 64, 0.0, 42));
        // Only one window — should return insufficient
        let r = det.detect();
        // No panic; score = 0
        assert_eq!(r.drift_score, 0.0);
    }

    // ── PsiDrift tests ───────────────────────────────────────────────────

    #[test]
    fn psi_no_drift_same_cluster() {
        let cfg = DriftConfig::default();
        let mut det = PsiDrift::new(cfg);
        // Use non-zero mean so the anchor centroid is stable and well-defined.
        // Near-zero mean causes high cosine-similarity variance with finite n.
        det.add_window(0, &gaussian_cluster(500, 64, 1.0, 1));
        det.add_window(1, &gaussian_cluster(500, 64, 1.0, 2));
        let r = det.detect();
        assert!(
            !r.is_drifted,
            "same cluster → PSI < 0.25, score={}",
            r.drift_score
        );
    }

    #[test]
    fn psi_detects_distribution_shift() {
        let mut cfg = DriftConfig::default();
        cfg.psi_buckets = 10;
        let mut det = PsiDrift::new(cfg);
        // Tight cluster: all vectors pointing in the same direction
        // → cosine similarity to centroid ≈ 1.0 for all (narrow histogram)
        let tight: Vec<Vec<f32>> = (0..300).map(|_| vec![1.0f32; 32]).collect();
        // Fragmented: half the vectors point in the exact opposite direction
        // → bimodal distribution of cosine similarities (+1 and -1)
        let bimodal: Vec<Vec<f32>> = (0..300)
            .map(|i| if i < 150 { vec![1.0f32; 32] } else { vec![-1.0f32; 32] })
            .collect();
        det.add_window(0, &tight);
        det.add_window(1, &bimodal);
        let r = det.detect();
        assert!(
            r.is_drifted,
            "unimodal→bimodal shift must be flagged, score={}",
            r.drift_score
        );
    }

    // ── CoherenceDrift tests ─────────────────────────────────────────────

    #[test]
    fn coherence_stable_same_cluster() {
        let cfg = DriftConfig::default();
        let mut det = CoherenceDrift::new(cfg);
        det.add_window(0, &gaussian_cluster(200, 32, 1.0, 7));
        det.add_window(1, &gaussian_cluster(200, 32, 1.0, 8));
        let r = det.detect();
        assert!(
            !r.is_drifted,
            "same cluster → stable, score={}",
            r.drift_score
        );
    }

    #[test]
    fn coherence_detects_fragmentation() {
        let cfg = DriftConfig::default();
        let mut det = CoherenceDrift::new(cfg);
        // Baseline: tight cluster
        let tight: Vec<Vec<f32>> = (0..200).map(|_| vec![1.0f32; 32]).collect();
        // Latest: two opposing clusters
        let fragmented: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                if i < 100 {
                    vec![1.0f32; 32]
                } else {
                    vec![-1.0f32; 32]
                }
            })
            .collect();
        det.add_window(0, &tight);
        det.add_window(1, &fragmented);
        let r = det.detect();
        assert!(
            r.is_drifted,
            "tight→fragmented must be flagged, score={}",
            r.drift_score
        );
    }

    #[test]
    fn coherence_details_populated() {
        let cfg = DriftConfig::default();
        let mut det = CoherenceDrift::new(cfg);
        let tight: Vec<Vec<f32>> = (0..200).map(|_| vec![1.0f32; 32]).collect();
        let fragmented: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                if i < 100 {
                    vec![1.0f32; 32]
                } else {
                    vec![-1.0f32; 32]
                }
            })
            .collect();
        det.add_window(0, &tight);
        det.add_window(1, &fragmented);
        let r = det.detect();
        assert!(r.details.contains_key("psi"), "must include PSI sub-score");
        assert!(
            r.details.contains_key("coherence_delta"),
            "must include coherence delta"
        );
    }
}
