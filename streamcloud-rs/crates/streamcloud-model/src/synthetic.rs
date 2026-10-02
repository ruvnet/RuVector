//! Pure-Rust deterministic reconstructor (the ADR-0006 synthetic fallback).
//!
//! This is **not** the trained network. It derives a plausible depth field from
//! monocular cues (shading + a ground-plane vertical gradient), projects it
//! through a pinhole camera to a colored 3D point cloud, and folds retrieved
//! anchor context in as a small drift-correction term. Its purpose is to make
//! the entire streaming system — memory retrieval, point-cloud generation,
//! PNG/MP4 export, native + WebGPU demos — runnable and testable end-to-end
//! without the 4.63 GB checkpoint.

use crate::{Frame, KeyframeFeatures, Point, PointCloud, Reconstructor};
use streamcloud_tensor::ModelConfig;

/// Deterministic monocular reconstructor.
pub struct SyntheticReconstructor {
    config: ModelConfig,
    /// Pixel stride for point-cloud sampling (1 = every pixel).
    sample_step: usize,
    /// Base camera depth (scene placed in front of the camera).
    base_depth: f32,
}

impl SyntheticReconstructor {
    /// New reconstructor. `sample_step` controls point density (>=1).
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            sample_step: 6,
            base_depth: 4.0,
        }
    }

    /// Override the point-cloud sampling stride.
    pub fn with_sample_step(mut self, step: usize) -> Self {
        self.sample_step = step.max(1);
        self
    }

    /// Average-pool the frame's luminance into a fixed-length, L2-normalized
    /// feature vector. Structurally similar frames map to similar vectors, so
    /// cosine retrieval in `streamcloud-memory` behaves sensibly.
    pub fn encode(&self, frame: &Frame) -> KeyframeFeatures {
        let dim = self.config.keyframe_feature_dim.max(1);
        let cols = (dim as f32).sqrt().ceil() as usize;
        let rows = dim.div_ceil(cols);
        let mut feats = vec![0.0f32; dim];

        for (cell, feat) in feats.iter_mut().enumerate() {
            let cx = cell % cols;
            let cy = cell / cols;
            // Cell pixel range.
            let x0 = cx * frame.width / cols;
            let x1 = ((cx + 1) * frame.width / cols).max(x0 + 1).min(frame.width);
            let y0 = cy * frame.height / rows;
            let y1 = ((cy + 1) * frame.height / rows)
                .max(y0 + 1)
                .min(frame.height);
            let mut sum = 0.0f32;
            let mut n = 0u32;
            let mut yy = y0;
            while yy < y1 {
                let mut xx = x0;
                while xx < x1 {
                    sum += frame.luma(xx, yy);
                    n += 1;
                    xx += 1;
                }
                yy += 1;
            }
            *feat = if n > 0 { sum / n as f32 } else { 0.0 };
        }

        // L2 normalize (unit vector → cosine similarity is well-behaved).
        let norm: f32 = feats.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut feats {
                *v /= norm;
            }
        }
        KeyframeFeatures(feats)
    }

    /// Mean feature value across anchors → a small scalar drift-correction term.
    fn drift_correction(&self, anchors: &[KeyframeFeatures]) -> f32 {
        if anchors.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0f32;
        let mut n = 0u32;
        for a in anchors {
            for &v in &a.0 {
                acc += v;
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            // Bounded, subtle correction in world units.
            (acc / n as f32).clamp(-1.0, 1.0) * 0.25
        }
    }
}

impl Reconstructor for SyntheticReconstructor {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn encode(&self, frame: &Frame) -> KeyframeFeatures {
        SyntheticReconstructor::encode(self, frame)
    }

    fn reconstruct(&self, frame: &Frame, anchors: &[KeyframeFeatures]) -> PointCloud {
        let correction = self.drift_correction(anchors);

        let w = frame.width as f32;
        let h = frame.height as f32;
        let cx = w * 0.5;
        let cy = h * 0.5;
        let focal = w; // ~53° HFOV pinhole

        let mut points = Vec::new();
        let step = self.sample_step;
        let mut y = 0usize;
        while y < frame.height {
            let mut x = 0usize;
            while x < frame.width {
                let luma = frame.luma(x, y);
                // Monocular depth cue: darker + lower-in-frame → nearer ground;
                // brighter → farther. Plus anchor-based drift correction.
                let vertical = y as f32 / h; // 0 top .. 1 bottom
                let depth = self.base_depth + (1.0 - luma) * 3.0 - vertical * 1.5 + correction;
                let depth = depth.max(0.2);

                let xf = (x as f32 - cx) / focal * depth;
                let yf = (cy - y as f32) / focal * depth; // flip: image y down → world y up
                points.push(Point {
                    pos: [xf, yf, depth],
                    color: frame.rgb(x, y),
                });
                x += step;
            }
            y += step;
        }

        PointCloud { points }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(w: usize, h: usize, seed: u8) -> Frame {
        let mut px = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let on = ((x / 4 + y / 4 + seed as usize) % 2) == 0;
                let v = if on { 220 } else { 40 };
                px.extend_from_slice(&[v, v, v, 255]);
            }
        }
        Frame::new(w, h, px)
    }

    #[test]
    fn encode_dim_and_normalized() {
        let cfg = ModelConfig {
            keyframe_feature_dim: 256,
            ..Default::default()
        };
        let r = SyntheticReconstructor::new(cfg);
        let f = checker(64, 48, 0);
        let kf = r.encode(&f);
        assert_eq!(kf.dim(), 256);
        let norm: f32 = kf.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn encode_is_deterministic() {
        let r = SyntheticReconstructor::new(ModelConfig::default());
        let f = checker(80, 60, 3);
        assert_eq!(r.encode(&f), r.encode(&f));
    }

    #[test]
    fn produces_points_with_finite_bounds() {
        let mut r = SyntheticReconstructor::new(ModelConfig::default()).with_sample_step(4);
        let f = checker(96, 72, 1);
        let (_kf, pc) = r.process_frame(0, &f, &[]);
        assert!(pc.len() > 100);
        let (min, max) = pc.bounds().unwrap();
        for a in 0..3 {
            assert!(min[a].is_finite() && max[a].is_finite());
        }
        assert!(min[2] > 0.0, "all depths must be in front of camera");
    }

    #[test]
    fn anchors_change_reconstruction() {
        let mut r = SyntheticReconstructor::new(ModelConfig::default()).with_sample_step(8);
        let f = checker(64, 64, 2);
        let (_k, pc0) = r.process_frame(1, &f, &[]);
        // Non-trivial anchor → drift correction shifts depth.
        let anchor = KeyframeFeatures(vec![1.0; r.config().keyframe_feature_dim]);
        let (_k2, pc1) = r.process_frame(1, &f, std::slice::from_ref(&anchor));
        assert_ne!(pc0.points[0].pos[2], pc1.points[0].pos[2]);
    }
}
