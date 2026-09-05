//! # streamcloud-pipeline
//!
//! The streaming reconstruction loop that ties the port together:
//!
//! ```text
//! frame ─▶ model.encode ─▶ memory.retrieve_context(top_k) ─▶ anchor features
//!                                                              │
//!         model.reconstruct(frame, anchors) ◀─────────────────┘
//!                     │
//!                     ├─▶ memory.insert_keyframe(frame_id, features)
//!                     └─▶ point cloud ─▶ renderer ─▶ FrameSink (PNG/MP4)
//! ```
//!
//! The per-frame anchor retrieval is exactly the KV-cache replacement described
//! in ADR-0002: instead of attending to a VRAM-bound paged cache, each frame
//! pulls the top-K structurally similar past keyframes from the HNSW index,
//! regardless of how far back they occurred.

pub mod render;

pub use render::SoftwareRenderer;
pub use streamcloud_model::scene;

use streamcloud_memory::{StreamingMemory, StreamingMemoryConfig};
use streamcloud_model::{Frame, KeyframeFeatures, PointCloud, Reconstructor};

/// Errors from the pipeline.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// Memory layer error.
    #[error("memory: {0}")]
    Memory(#[from] streamcloud_memory::MemoryError),
    /// I/O / encoding error.
    #[error("io: {0}")]
    Io(#[from] streamcloud_io::IoError),
}

/// Per-frame result.
#[derive(Debug, Clone)]
pub struct FrameResult {
    /// Frame id (sequential).
    pub frame_id: u64,
    /// Number of anchor keyframes retrieved and used for drift correction.
    pub anchors_used: usize,
    /// Number of 3D points produced.
    pub points: usize,
}

/// Aggregate stats over a run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunStats {
    /// Frames processed.
    pub frames: u64,
    /// Total anchors retrieved across all frames.
    pub total_anchors: u64,
    /// Total points produced.
    pub total_points: u64,
    /// Keyframes resident in memory at the end.
    pub memory_len: usize,
}

/// Streaming reconstruction over any [`Reconstructor`] backend.
pub struct StreamingReconstructor<R: Reconstructor> {
    model: R,
    memory: StreamingMemory,
    history: Vec<KeyframeFeatures>,
    top_k: usize,
    frame_index: u64,
}

impl<R: Reconstructor> StreamingReconstructor<R> {
    /// Build a pipeline. Memory dimensionality follows the model's keyframe
    /// feature dimension.
    pub fn new(model: R, top_k: usize) -> Result<Self, PipelineError> {
        let dim = model.config().keyframe_feature_dim;
        let memory = StreamingMemory::new(StreamingMemoryConfig::new(dim))?;
        Ok(Self {
            model,
            memory,
            history: Vec::new(),
            top_k,
            frame_index: 0,
        })
    }

    /// Borrow the memory layer (e.g. to inspect `len`).
    pub fn memory(&self) -> &StreamingMemory {
        &self.memory
    }

    /// Process one frame: encode → retrieve anchors → reconstruct → store.
    pub fn process(&mut self, frame: &Frame) -> Result<(PointCloud, FrameResult), PipelineError> {
        let features = self.model.encode(frame);

        // Retrieve anchor context (skip on the very first frame).
        let anchor_feats: Vec<KeyframeFeatures> = if self.memory.is_empty() {
            Vec::new()
        } else {
            let matches = self
                .memory
                .retrieve_context(features.as_slice(), self.top_k)?;
            matches
                .into_iter()
                // Don't anchor to ourselves; map id → stored features.
                .filter(|m| m.frame_id != self.frame_index)
                .filter_map(|m| self.history.get(m.frame_id as usize).cloned())
                .collect()
        };

        let pc = self.model.reconstruct(frame, &anchor_feats);

        // Persist this keyframe.
        self.memory
            .insert_keyframe(self.frame_index, features.as_slice())?;
        self.history.push(features);

        let result = FrameResult {
            frame_id: self.frame_index,
            anchors_used: anchor_feats.len(),
            points: pc.len(),
        };
        self.frame_index += 1;
        Ok((pc, result))
    }
}

/// Drive a whole stream: process each frame, render it from an orbiting camera,
/// and push the rendered RGBA to `sink`. Returns aggregate stats.
pub fn run_stream<R, I, S>(
    pipeline: &mut StreamingReconstructor<R>,
    frames: I,
    renderer: &SoftwareRenderer,
    sink: &mut S,
    orbit_per_frame: f32,
) -> Result<RunStats, PipelineError>
where
    R: Reconstructor,
    I: IntoIterator<Item = Frame>,
    S: streamcloud_io::FrameSink,
{
    let mut stats = RunStats::default();
    for frame in frames {
        let (pc, res) = pipeline.process(&frame)?;
        let angle = res.frame_id as f32 * orbit_per_frame;
        let rgba = renderer.render(&pc, angle);
        sink.push(&rgba, renderer.width, renderer.height)?;

        stats.frames += 1;
        stats.total_anchors += res.anchors_used as u64;
        stats.total_points += res.points as u64;
    }
    sink.finish()?;
    stats.memory_len = pipeline.memory().len();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use streamcloud_io::NullSink;
    use streamcloud_model::{ModelConfig, SyntheticReconstructor};

    fn small_model() -> SyntheticReconstructor {
        let cfg = ModelConfig {
            keyframe_feature_dim: 128,
            anchor_top_k: 8,
            ..Default::default()
        };
        SyntheticReconstructor::new(cfg).with_sample_step(8)
    }

    #[test]
    fn streams_and_retrieves_anchors() {
        let mut pipe = StreamingReconstructor::new(small_model(), 8).unwrap();
        let renderer = SoftwareRenderer::new(128, 96);
        let mut sink = NullSink::default();
        let stats = run_stream(
            &mut pipe,
            scene::synthetic_stream(40, 128, 96),
            &renderer,
            &mut sink,
            0.05,
        )
        .unwrap();

        assert_eq!(stats.frames, 40);
        assert_eq!(stats.memory_len, 40);
        assert_eq!(sink.count, 40);
        assert!(stats.total_points > 0);
        // After the first frame, retrieval should surface anchors.
        assert!(
            stats.total_anchors > 0,
            "expected anchor retrieval across the stream"
        );
    }

    #[test]
    fn first_frame_has_no_anchors() {
        let mut pipe = StreamingReconstructor::new(small_model(), 8).unwrap();
        let f = scene::synthetic_frame(0, 64, 48);
        let (_pc, res) = pipe.process(&f).unwrap();
        assert_eq!(res.frame_id, 0);
        assert_eq!(res.anchors_used, 0);
    }
}
