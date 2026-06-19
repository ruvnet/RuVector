//! Compact digital backend: feature extraction + nearest-centroid classifier.
//!
//! This is the small electronic network that "reads" the optical measurement
//! (ADR-260 §4). Nearest-centroid is chosen because it is deterministic,
//! parameter-counted exactly (`classes * feature_dim`), and has no training
//! randomness — so any accuracy difference is attributable to the optics.

use photonlayer_core::detector::OpticalFrame;

/// Average-pool a row-major grid to `out_dim x out_dim`, then L2-normalize.
pub fn pool_features(values: &[f32], w: usize, h: usize, out_dim: usize) -> Vec<f32> {
    let mut feat = vec![0.0f32; out_dim * out_dim];
    for oy in 0..out_dim {
        for ox in 0..out_dim {
            let x0 = ox * w / out_dim;
            let x1 = ((ox + 1) * w / out_dim).max(x0 + 1).min(w);
            let y0 = oy * h / out_dim;
            let y1 = ((oy + 1) * h / out_dim).max(y0 + 1).min(h);
            let mut acc = 0.0;
            let mut cnt = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    acc += values[y * w + x];
                    cnt += 1.0;
                }
            }
            feat[oy * out_dim + ox] = if cnt > 0.0 { acc / cnt } else { 0.0 };
        }
    }
    l2_normalize(&mut feat);
    feat
}

/// Feature vector from a detector frame.
pub fn frame_features(frame: &OpticalFrame, out_dim: usize) -> Vec<f32> {
    pool_features(&frame.intensity, frame.width, frame.height, out_dim)
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// A trained nearest-centroid classifier.
#[derive(Clone, Debug)]
pub struct NearestCentroid {
    pub centroids: Vec<Vec<f32>>,
    pub feature_dim: usize,
}

impl NearestCentroid {
    /// Fit one centroid per class from labeled feature vectors.
    pub fn fit(features: &[Vec<f32>], labels: &[usize], num_classes: usize) -> Self {
        let dim = features.first().map(|f| f.len()).unwrap_or(0);
        let mut sums = vec![vec![0.0f32; dim]; num_classes];
        let mut counts = vec![0usize; num_classes];
        for (f, &lab) in features.iter().zip(labels) {
            counts[lab] += 1;
            for (s, &x) in sums[lab].iter_mut().zip(f) {
                *s += x;
            }
        }
        let centroids = sums
            .into_iter()
            .zip(&counts)
            .map(|(mut s, &c)| {
                if c > 0 {
                    let inv = 1.0 / c as f32;
                    for v in &mut s {
                        *v *= inv;
                    }
                }
                s
            })
            .collect();
        Self {
            centroids,
            feature_dim: dim,
        }
    }

    /// Predict the class of a single feature vector (min L2 distance).
    pub fn predict(&self, feat: &[f32]) -> usize {
        let mut best = 0;
        let mut best_d = f32::INFINITY;
        for (c, centroid) in self.centroids.iter().enumerate() {
            let d: f32 = centroid
                .iter()
                .zip(feat)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best
    }

    /// Total learnable parameter count (ADR-260 §16.2 backend size metric).
    pub fn param_count(&self) -> usize {
        self.centroids.len() * self.feature_dim
    }

    /// Classification accuracy over a labeled feature set.
    pub fn accuracy(&self, features: &[Vec<f32>], labels: &[usize]) -> f32 {
        if features.is_empty() {
            return 0.0;
        }
        let correct = features
            .iter()
            .zip(labels)
            .filter(|(f, &l)| self.predict(f) == l)
            .count();
        correct as f32 / features.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separable_features_classify() {
        let feats = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        let labels = vec![0, 0, 1, 1];
        let ncc = NearestCentroid::fit(&feats, &labels, 2);
        assert_eq!(ncc.accuracy(&feats, &labels), 1.0);
        assert_eq!(ncc.param_count(), 4);
    }
}
