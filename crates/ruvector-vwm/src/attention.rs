//! Four-level attention pipeline for query-first rendering.
//!
//! Implements ADR-021's four attention stages that progressively narrow the
//! set of Gaussians considered for rendering:
//!
//! 1. **View attention** – frustum culling based on camera pose.
//! 2. **Temporal attention** – filter by activity in the current time window.
//! 3. **Semantic attention** – filter by embedding similarity to a query.
//! 4. **Write attention** – commit gating based on coherence/mincut score.
//!
//! The [`AttentionPipeline`] chains all four stages and returns the surviving
//! Gaussian indices along with per-stage statistics.

use crate::gaussian::Gaussian4D;
use crate::streaming::ActiveMask;

/// Result of running the attention pipeline on a set of Gaussians.
#[derive(Clone, Debug)]
pub struct AttentionResult {
    /// Indices of Gaussians that passed all attention stages.
    pub surviving_indices: Vec<u32>,
    /// Per-stage statistics.
    pub stats: AttentionStats,
}

/// Per-stage statistics from the attention pipeline.
#[derive(Clone, Debug, Default)]
pub struct AttentionStats {
    /// Input count.
    pub input_count: u32,
    /// Passed view attention (frustum).
    pub after_view: u32,
    /// Passed temporal attention.
    pub after_temporal: u32,
    /// Passed semantic attention.
    pub after_semantic: u32,
    /// Passed write attention (final output).
    pub after_write: u32,
}

/// View attention: frustum culling against 6 clip planes.
///
/// Accepts Gaussians whose center (at time `t`) lies inside the frustum
/// defined by the view-projection matrix. Uses a simplified point-in-frustum
/// test (not accounting for Gaussian radius).
pub struct ViewAttention {
    /// Extracted frustum planes as [a, b, c, d] (normal pointing inward).
    planes: [[f32; 4]; 6],
}

impl ViewAttention {
    /// Construct from a column-major 4×4 view-projection matrix.
    pub fn from_view_proj(vp: &[f32; 16]) -> Self {
        Self {
            planes: extract_frustum_planes(vp),
        }
    }

    /// Test if the Gaussian's position at time `t` is inside the frustum.
    #[inline]
    pub fn test(&self, g: &Gaussian4D, t: f32) -> bool {
        let pos = g.position_at(t);
        for plane in &self.planes {
            let d = plane[0] * pos[0] + plane[1] * pos[1] + plane[2] * pos[2] + plane[3];
            if d < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Temporal attention: filter by time-range activity.
pub struct TemporalAttention {
    /// Current evaluation time.
    pub time: f32,
}

impl TemporalAttention {
    /// Create temporal attention for a specific time.
    pub fn new(time: f32) -> Self {
        Self { time }
    }

    /// Test if the Gaussian is active at the configured time.
    #[inline]
    pub fn test(&self, g: &Gaussian4D) -> bool {
        g.is_active_at(self.time)
    }
}

/// Semantic attention: cosine-similarity filter against a query embedding.
pub struct SemanticAttention {
    /// Normalized query embedding.
    query: Vec<f32>,
    /// Minimum similarity threshold (0.0 to 1.0).
    pub threshold: f32,
}

impl SemanticAttention {
    /// Create semantic attention with the given query embedding and threshold.
    ///
    /// The query is normalized internally.
    pub fn new(query: &[f32], threshold: f32) -> Self {
        let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();
        let query = if norm > 0.0 {
            query.iter().map(|v| v / norm).collect()
        } else {
            query.to_vec()
        };
        Self { query, threshold }
    }

    /// Compute cosine similarity between the query and the given embedding.
    #[inline]
    pub fn similarity(&self, embedding: &[f32]) -> f32 {
        if self.query.len() != embedding.len() || self.query.is_empty() {
            return 0.0;
        }
        let dot: f32 = self.query.iter().zip(embedding).map(|(a, b)| a * b).sum();
        let norm_b: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_b > 0.0 {
            dot / norm_b
        } else {
            0.0
        }
    }

    /// Test if the embedding exceeds the similarity threshold.
    #[inline]
    pub fn test(&self, embedding: &[f32]) -> bool {
        self.similarity(embedding) >= self.threshold
    }
}

/// Write attention: commit gating based on a coherence score.
///
/// Gaussians with a coherence score below the threshold are rejected
/// (preventing uncommitted or conflicted data from reaching the renderer).
pub struct WriteAttention {
    /// Minimum coherence score to pass.
    pub min_coherence: f32,
}

impl WriteAttention {
    /// Create write attention with the given minimum coherence threshold.
    pub fn new(min_coherence: f32) -> Self {
        Self { min_coherence }
    }

    /// Test if a coherence score passes the gate.
    #[inline]
    pub fn test(&self, coherence_score: f32) -> bool {
        coherence_score >= self.min_coherence
    }
}

/// Chained four-stage attention pipeline.
///
/// Runs View → Temporal → Semantic → Write in order, collecting statistics.
/// Semantic and write stages are optional; when `None`, the stage passes all.
pub struct AttentionPipeline {
    pub view: ViewAttention,
    pub temporal: TemporalAttention,
    pub semantic: Option<SemanticAttention>,
    pub write: Option<WriteAttention>,
}

impl AttentionPipeline {
    /// Run the full pipeline on a slice of Gaussians.
    ///
    /// `embeddings` maps Gaussian index → embedding slice (used by semantic
    /// stage). If `None` or shorter than the input, semantic attention is
    /// skipped for those Gaussians.
    ///
    /// `coherence_scores` maps Gaussian index → score (used by write stage).
    /// If `None`, write attention is skipped.
    pub fn execute(
        &self,
        gaussians: &[Gaussian4D],
        embeddings: Option<&[Vec<f32>]>,
        coherence_scores: Option<&[f32]>,
    ) -> AttentionResult {
        let input_count = gaussians.len() as u32;
        let mut indices: Vec<u32> = (0..input_count).collect();

        // Stage 1: View attention (frustum)
        indices.retain(|&i| self.view.test(&gaussians[i as usize], self.temporal.time));
        let after_view = indices.len() as u32;

        // Stage 2: Temporal attention
        indices.retain(|&i| self.temporal.test(&gaussians[i as usize]));
        let after_temporal = indices.len() as u32;

        // Stage 3: Semantic attention
        let after_semantic = if let Some(ref sem) = self.semantic {
            if let Some(embs) = embeddings {
                indices.retain(|&i| {
                    let idx = i as usize;
                    if idx < embs.len() {
                        sem.test(&embs[idx])
                    } else {
                        true // no embedding → pass
                    }
                });
            }
            indices.len() as u32
        } else {
            indices.len() as u32
        };

        // Stage 4: Write attention (coherence gate)
        let after_write = if let Some(ref write) = self.write {
            if let Some(scores) = coherence_scores {
                indices.retain(|&i| {
                    let idx = i as usize;
                    if idx < scores.len() {
                        write.test(scores[idx])
                    } else {
                        true
                    }
                });
            }
            indices.len() as u32
        } else {
            indices.len() as u32
        };

        AttentionResult {
            surviving_indices: indices,
            stats: AttentionStats {
                input_count,
                after_view,
                after_temporal,
                after_semantic,
                after_write,
            },
        }
    }
}

/// Build an [`ActiveMask`] from attention pipeline results.
///
/// Sets bits for the surviving indices over a total of `total_count` Gaussians.
pub fn attention_result_to_mask(result: &AttentionResult, total_count: u32) -> ActiveMask {
    let mut mask = ActiveMask::new(total_count);
    for &idx in &result.surviving_indices {
        mask.set(idx, true);
    }
    mask
}

// ---- Frustum plane extraction ----

/// Extract six frustum planes from a column-major 4×4 view-projection matrix.
///
/// Each plane is `[a, b, c, d]` where `ax + by + cz + d >= 0` means inside.
/// Planes are normalized so `sqrt(a² + b² + c²) = 1`.
fn extract_frustum_planes(m: &[f32; 16]) -> [[f32; 4]; 6] {
    // Column-major: m[col*4+row]
    // Row i of the matrix: m[i], m[4+i], m[8+i], m[12+i]
    let row = |r: usize| -> [f32; 4] {
        [m[r], m[4 + r], m[8 + r], m[12 + r]]
    };
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    let mut planes = [
        // Left:   row3 + row0
        [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]],
        // Right:  row3 - row0
        [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]],
        // Bottom: row3 + row1
        [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]],
        // Top:    row3 - row1
        [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]],
        // Near:   row3 + row2
        [r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]],
        // Far:    row3 - row2
        [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]],
    ];

    for plane in &mut planes {
        let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
        if len > 0.0 {
            plane[0] /= len;
            plane[1] /= len;
            plane[2] /= len;
            plane[3] /= len;
        }
    }

    planes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::Gaussian4D;

    fn identity_vp() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn make_gaussian(pos: [f32; 3], id: u32, time: [f32; 2]) -> Gaussian4D {
        let mut g = Gaussian4D::new(pos, id);
        g.time_range = time;
        g
    }

    // -- ViewAttention --

    #[test]
    fn test_view_attention_identity_passes_origin() {
        let va = ViewAttention::from_view_proj(&identity_vp());
        let g = make_gaussian([0.0, 0.0, 0.0], 0, [f32::NEG_INFINITY, f32::INFINITY]);
        assert!(va.test(&g, 0.0));
    }

    #[test]
    fn test_view_attention_identity_rejects_outside() {
        let va = ViewAttention::from_view_proj(&identity_vp());
        // Point well outside NDC cube
        let g = make_gaussian([10.0, 0.0, 0.0], 0, [f32::NEG_INFINITY, f32::INFINITY]);
        assert!(!va.test(&g, 0.0));
    }

    // -- TemporalAttention --

    #[test]
    fn test_temporal_passes_active() {
        let ta = TemporalAttention::new(5.0);
        let g = make_gaussian([0.0, 0.0, 0.0], 0, [0.0, 10.0]);
        assert!(ta.test(&g));
    }

    #[test]
    fn test_temporal_rejects_inactive() {
        let ta = TemporalAttention::new(15.0);
        let g = make_gaussian([0.0, 0.0, 0.0], 0, [0.0, 10.0]);
        assert!(!ta.test(&g));
    }

    #[test]
    fn test_temporal_unbounded_always_active() {
        let ta = TemporalAttention::new(999.0);
        let g = make_gaussian([0.0, 0.0, 0.0], 0, [f32::NEG_INFINITY, f32::INFINITY]);
        assert!(ta.test(&g));
    }

    // -- SemanticAttention --

    #[test]
    fn test_semantic_identical_vectors() {
        let sa = SemanticAttention::new(&[1.0, 0.0, 0.0], 0.9);
        assert!(sa.test(&[1.0, 0.0, 0.0]));
        let sim = sa.similarity(&[1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_semantic_orthogonal_vectors() {
        let sa = SemanticAttention::new(&[1.0, 0.0, 0.0], 0.5);
        let sim = sa.similarity(&[0.0, 1.0, 0.0]);
        assert!(sim.abs() < 1e-5);
        assert!(!sa.test(&[0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_semantic_empty_embedding() {
        let sa = SemanticAttention::new(&[1.0, 0.0], 0.5);
        assert_eq!(sa.similarity(&[]), 0.0);
    }

    #[test]
    fn test_semantic_dimension_mismatch() {
        let sa = SemanticAttention::new(&[1.0, 0.0, 0.0], 0.5);
        assert_eq!(sa.similarity(&[1.0, 0.0]), 0.0);
    }

    // -- WriteAttention --

    #[test]
    fn test_write_passes_above_threshold() {
        let wa = WriteAttention::new(0.7);
        assert!(wa.test(0.8));
        assert!(wa.test(0.7));
    }

    #[test]
    fn test_write_rejects_below_threshold() {
        let wa = WriteAttention::new(0.7);
        assert!(!wa.test(0.69));
    }

    // -- AttentionPipeline --

    #[test]
    fn test_pipeline_all_pass() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0, [0.0, 10.0]),
            make_gaussian([0.1, 0.0, 0.0], 1, [0.0, 10.0]),
        ];
        let pipeline = AttentionPipeline {
            view: ViewAttention::from_view_proj(&identity_vp()),
            temporal: TemporalAttention::new(5.0),
            semantic: None,
            write: None,
        };
        let result = pipeline.execute(&gaussians, None, None);
        assert_eq!(result.surviving_indices.len(), 2);
        assert_eq!(result.stats.input_count, 2);
        assert_eq!(result.stats.after_view, 2);
        assert_eq!(result.stats.after_temporal, 2);
    }

    #[test]
    fn test_pipeline_temporal_filters() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0, [0.0, 3.0]),
            make_gaussian([0.1, 0.0, 0.0], 1, [0.0, 10.0]),
        ];
        let pipeline = AttentionPipeline {
            view: ViewAttention::from_view_proj(&identity_vp()),
            temporal: TemporalAttention::new(5.0),
            semantic: None,
            write: None,
        };
        let result = pipeline.execute(&gaussians, None, None);
        assert_eq!(result.stats.after_temporal, 1);
        assert_eq!(result.surviving_indices, vec![1]);
    }

    #[test]
    fn test_pipeline_semantic_filters() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0, [f32::NEG_INFINITY, f32::INFINITY]),
            make_gaussian([0.1, 0.0, 0.0], 1, [f32::NEG_INFINITY, f32::INFINITY]),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0], // similar to query
            vec![0.0, 1.0, 0.0], // orthogonal to query
        ];
        let pipeline = AttentionPipeline {
            view: ViewAttention::from_view_proj(&identity_vp()),
            temporal: TemporalAttention::new(0.0),
            semantic: Some(SemanticAttention::new(&[1.0, 0.0, 0.0], 0.5)),
            write: None,
        };
        let result = pipeline.execute(&gaussians, Some(&embeddings), None);
        assert_eq!(result.stats.after_semantic, 1);
        assert_eq!(result.surviving_indices, vec![0]);
    }

    #[test]
    fn test_pipeline_write_filters() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0, [f32::NEG_INFINITY, f32::INFINITY]),
            make_gaussian([0.1, 0.0, 0.0], 1, [f32::NEG_INFINITY, f32::INFINITY]),
        ];
        let scores = vec![0.9, 0.3];
        let pipeline = AttentionPipeline {
            view: ViewAttention::from_view_proj(&identity_vp()),
            temporal: TemporalAttention::new(0.0),
            semantic: None,
            write: Some(WriteAttention::new(0.5)),
        };
        let result = pipeline.execute(&gaussians, None, Some(&scores));
        assert_eq!(result.stats.after_write, 1);
        assert_eq!(result.surviving_indices, vec![0]);
    }

    // -- attention_result_to_mask --

    #[test]
    fn test_result_to_mask() {
        let result = AttentionResult {
            surviving_indices: vec![0, 3, 7],
            stats: AttentionStats::default(),
        };
        let mask = attention_result_to_mask(&result, 10);
        assert!(mask.is_active(0));
        assert!(!mask.is_active(1));
        assert!(mask.is_active(3));
        assert!(mask.is_active(7));
        assert_eq!(mask.active_count(), 3);
    }
}
