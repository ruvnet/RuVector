//! Differential-detection readout (the accuracy-per-line lever, ADR-260).
//!
//! Plain optical classifiers read one intensity integral per class and take the
//! argmax. Differential detection instead reads **two** regions per class and
//! scores `class k = I+_k - I-_k` — the same trick that lifts diffractive-net
//! MNIST from ~91-92% to ~97-98% in the literature (Li/Ozcan, arXiv:1906.03417)
//! for only +10 detector regions and a subtraction.
//!
//! Both readouts here operate on the *same* propagated `OpticalFrame`, so an
//! ablation that swaps only the readout (plain vs differential) on one trained
//! mask isolates the lever exactly. The readout is the entire digital backend:
//! `K` (=10) or `2K` (=20) region integrals and an argmax — no learned decoder
//! parameters, so any accuracy difference is attributable to optics + readout.

use photonlayer_core::detector::OpticalFrame;

/// A rectangular detector region on the sensor grid (inclusive `x0..x1`).
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize, // exclusive
    pub y1: usize, // exclusive
}

impl Region {
    /// Integrate intensity over this region of a row-major frame.
    fn integrate(&self, frame: &OpticalFrame) -> f32 {
        let mut acc = 0.0f32;
        for y in self.y0..self.y1.min(frame.height) {
            for x in self.x0..self.x1.min(frame.width) {
                acc += frame.intensity[y * frame.width + x];
            }
        }
        acc
    }
}

/// Fixed differential-detection region layout for `num_classes` classes.
///
/// The sensor is tiled into a `rows x cols` grid of equal cells (enough cells
/// for `2 * num_classes` of them). Class `k` is assigned cell `2k` as its
/// positive region and cell `2k+1` as its negative region. The layout is
/// deterministic and mask-independent, so the learned phase mask — not the
/// readout — is what routes class-specific energy into the right cells.
#[derive(Clone, Debug)]
pub struct DiffDetector {
    pub num_classes: usize,
    /// `pos[k]` and `neg[k]` regions for class `k`.
    pub pos: Vec<Region>,
    pub neg: Vec<Region>,
    /// Number of distinct sensor regions actually read (the digital readout
    /// size that the compression ratio is measured against).
    pub readout_regions: usize,
}

impl DiffDetector {
    /// Lay out `2 * num_classes` equal tiles over a `width x height` sensor.
    ///
    /// Tiles fill a near-square `rows x cols` grid in row-major order; any
    /// trailing cells beyond `2 * num_classes` are simply unused. Panics only
    /// if the sensor is too small to hold the required tiles (caller controls
    /// the grid, so this is a programming error, not a runtime input error).
    pub fn new(num_classes: usize, width: usize, height: usize) -> Self {
        let needed = 2 * num_classes;
        // Choose a tiling close to square.
        let cols = (needed as f32).sqrt().ceil() as usize;
        let rows = needed.div_ceil(cols);
        assert!(
            cols <= width && rows <= height,
            "sensor {width}x{height} too small for {needed} differential tiles ({rows}x{cols})"
        );
        let tile_w = width / cols;
        let tile_h = height / rows;
        let cell = |idx: usize| -> Region {
            let r = idx / cols;
            let c = idx % cols;
            Region {
                x0: c * tile_w,
                y0: r * tile_h,
                x1: (c + 1) * tile_w,
                y1: (r + 1) * tile_h,
            }
        };
        let mut pos = Vec::with_capacity(num_classes);
        let mut neg = Vec::with_capacity(num_classes);
        for k in 0..num_classes {
            pos.push(cell(2 * k));
            neg.push(cell(2 * k + 1));
        }
        Self {
            num_classes,
            pos,
            neg,
            readout_regions: needed,
        }
    }

    /// Per-class positive-region integrals `I+_k` (the plain readout vector).
    pub fn positive_scores(&self, frame: &OpticalFrame) -> Vec<f32> {
        self.pos.iter().map(|r| r.integrate(frame)).collect()
    }

    /// Per-class differential scores `I+_k - I-_k`.
    pub fn differential_scores(&self, frame: &OpticalFrame) -> Vec<f32> {
        self.pos
            .iter()
            .zip(&self.neg)
            .map(|(p, n)| p.integrate(frame) - n.integrate(frame))
            .collect()
    }

    /// Raw `2K` region integrals as a feature vector, interleaved
    /// `[I+_0, I-_0, I+_1, I-_1, ...]`. This is the differential readout's full
    /// information (before the per-class subtraction) and is what a small
    /// trainable decoder consumes. `plain_features` exposes only the `K`
    /// positive integrals so an ablation can keep the decoder identical and
    /// vary only the feature set.
    pub fn diff_features(&self, frame: &OpticalFrame) -> Vec<f32> {
        let mut f = Vec::with_capacity(2 * self.num_classes);
        for (p, n) in self.pos.iter().zip(&self.neg) {
            f.push(p.integrate(frame));
            f.push(n.integrate(frame));
        }
        f
    }

    /// The `K` positive-region integrals only (plain readout feature set).
    pub fn plain_features(&self, frame: &OpticalFrame) -> Vec<f32> {
        self.positive_scores(frame)
    }

    /// Plain prediction: argmax of the positive-region integrals only.
    /// Reads `num_classes` regions.
    pub fn predict_plain(&self, frame: &OpticalFrame) -> usize {
        argmax(&self.positive_scores(frame))
    }

    /// Differential prediction: argmax of `I+_k - I-_k`.
    /// Reads `2 * num_classes` regions.
    pub fn predict_differential(&self, frame: &OpticalFrame) -> usize {
        argmax(&self.differential_scores(frame))
    }
}

/// Index of the maximum element (first on ties). Empty -> 0.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use photonlayer_core::detector::OpticalFrame;

    fn frame_from(width: usize, height: usize, fill: impl Fn(usize, usize) -> f32) -> OpticalFrame {
        // Build an OpticalFrame via a captured field would be heavy; instead use
        // the detector capture on a hand-built field. Simpler: construct the
        // intensity directly through the public capture path is unnecessary for
        // a readout unit test, so we exercise integration via a tiny field.
        use photonlayer_core::config::DetectorConfig;
        use photonlayer_core::detector::capture_with;
        use photonlayer_core::field::{InputImage, OpticalField};
        let px: Vec<f32> = (0..width * height)
            .map(|i| fill(i % width, i / width))
            .collect();
        let img = InputImage::from_norm_f32(width, height, px).unwrap();
        let field = OpticalField::from_image(&img, width, height).unwrap();
        // Amplitude = sqrt(intensity), so |field|^2 recovers the original px.
        capture_with(&field, &DetectorConfig::default(), 0)
    }

    #[test]
    fn layout_reads_two_regions_per_class() {
        let d = DiffDetector::new(10, 32, 32);
        assert_eq!(d.pos.len(), 10);
        assert_eq!(d.neg.len(), 10);
        assert_eq!(d.readout_regions, 20);
    }

    #[test]
    fn differential_score_is_pos_minus_neg() {
        // Put all energy into class-0's positive tile.
        let d = DiffDetector::new(2, 8, 8);
        let p0 = d.pos[0];
        let frame = frame_from(8, 8, |x, y| {
            if x >= p0.x0 && x < p0.x1 && y >= p0.y0 && y < p0.y1 {
                1.0
            } else {
                0.0
            }
        });
        let diff = d.differential_scores(&frame);
        assert!(diff[0] > diff[1], "class 0 should win: {diff:?}");
        assert_eq!(d.predict_differential(&frame), 0);
        assert_eq!(d.predict_plain(&frame), 0);
    }
}
