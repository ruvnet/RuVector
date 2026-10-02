//! WASM bindings for the StreamCloud browser demo.
//!
//! Exposes the streaming reconstruction pipeline to JavaScript: each call to
//! [`StreamCloudDemo::next_frame`] advances the synthetic scene, encodes a keyframe,
//! retrieves anchor context, reconstructs a 3D point cloud, and returns it as a
//! flat `Float32Array` of `[x, y, z, r, g, b]` (colors in 0..1) for the WebGPU
//! renderer in `demo/`.
//!
//! ## Memory layer note
//!
//! The native pipeline uses `streamcloud-memory`'s lock-free **HNSW** index
//! (`ruvector-core`). That stack pulls threads/rayon, which don't cross-compile
//! cleanly to `wasm32-unknown-unknown`, so the browser build uses a portable
//! **brute-force cosine** top-K over recent keyframes instead. The retrieval
//! *semantics* (top-K structurally similar past keyframes → drift correction)
//! are identical; only the index differs. See ADR-0002 / ADR-0004.

use js_sys::Float32Array;
use streamcloud_model::{
    scene, KeyframeFeatures, ModelConfig, Reconstructor, SyntheticReconstructor,
};
use wasm_bindgen::prelude::*;

/// Streaming reconstruction demo, driven frame-by-frame from JS.
#[wasm_bindgen]
pub struct StreamCloudDemo {
    model: SyntheticReconstructor,
    history: Vec<KeyframeFeatures>,
    width: usize,
    height: usize,
    top_k: usize,
    frame_index: u64,
    buf: Vec<f32>,
    last_anchors: usize,
}

#[wasm_bindgen]
impl StreamCloudDemo {
    /// Create a demo for a `width`x`height` synthetic scene retrieving `top_k`
    /// anchors per frame.
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, top_k: usize) -> StreamCloudDemo {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        let cfg = ModelConfig {
            keyframe_feature_dim: 256,
            anchor_top_k: top_k,
            ..Default::default()
        };
        StreamCloudDemo {
            model: SyntheticReconstructor::new(cfg).with_sample_step(4),
            history: Vec::new(),
            width,
            height,
            top_k,
            frame_index: 0,
            buf: Vec::new(),
            last_anchors: 0,
        }
    }

    /// Advance one frame and fill the internal point buffer (native-safe; no JS
    /// types). Use [`StreamCloudDemo::next_frame`] from JS to get the data out.
    pub fn advance(&mut self) {
        let frame = scene::synthetic_frame(self.frame_index, self.width, self.height);
        let feats = self.model.encode(&frame);
        let anchors = self.retrieve(&feats);
        self.last_anchors = anchors.len();
        let pc = self.model.reconstruct(&frame, &anchors);
        self.history.push(feats);
        self.frame_index += 1;

        self.buf.clear();
        self.buf.reserve(pc.points.len() * 6);
        for p in &pc.points {
            self.buf.push(p.pos[0]);
            self.buf.push(p.pos[1]);
            self.buf.push(p.pos[2]);
            self.buf.push(p.color[0] as f32 / 255.0);
            self.buf.push(p.color[1] as f32 / 255.0);
            self.buf.push(p.color[2] as f32 / 255.0);
        }
    }

    /// Advance one frame and return interleaved `[x,y,z,r,g,b]` point data.
    pub fn next_frame(&mut self) -> Float32Array {
        self.advance();
        Float32Array::from(self.buf.as_slice())
    }

    /// Points produced by the last `next_frame`.
    pub fn point_count(&self) -> usize {
        self.buf.len() / 6
    }

    /// Anchors retrieved for the last frame.
    pub fn anchors_used(&self) -> usize {
        self.last_anchors
    }

    /// Frames processed so far.
    pub fn frame_index(&self) -> u32 {
        self.frame_index as u32
    }

    /// Keyframes held in (brute-force) memory.
    pub fn keyframes(&self) -> usize {
        self.history.len()
    }
}

impl StreamCloudDemo {
    fn retrieve(&self, query: &KeyframeFeatures) -> Vec<KeyframeFeatures> {
        if self.history.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f32, usize)> = self
            .history
            .iter()
            .enumerate()
            .map(|(i, f)| (cosine(query.as_slice(), f.as_slice()), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(self.top_k)
            .map(|(_, i)| self.history[i].clone())
            .collect()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-8);
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_points_and_anchors() {
        let mut demo = StreamCloudDemo::new(64, 48, 8);
        for _ in 0..10 {
            demo.advance();
            assert!(demo.point_count() > 0);
        }
        assert_eq!(demo.keyframes(), 10);
        assert!(demo.anchors_used() > 0);
    }
}
