//! Query-first rendering: retrieve → select → render.
//!
//! Implements ADR-022's query-driven pattern where scene queries specify what
//! to render rather than rendering everything and filtering afterward.
//!
//! A [`SceneQuery`] describes the camera, time, optional semantic filter, and
//! coherence threshold. [`execute_query`] runs the full attention pipeline and
//! returns a [`QueryResult`] with surviving Gaussian indices and statistics.

use crate::attention::{
    AttentionPipeline, AttentionStats, SemanticAttention, TemporalAttention,
    ViewAttention, WriteAttention,
};
use crate::gaussian::Gaussian4D;
use crate::streaming::ActiveMask;

/// A scene query describing what to render.
#[derive(Clone, Debug)]
pub struct SceneQuery {
    /// Column-major 4×4 view-projection matrix for frustum culling.
    pub view_proj: [f32; 16],
    /// Time at which to evaluate Gaussian positions and activity.
    pub time: f32,
    /// Optional semantic query embedding for similarity filtering.
    pub semantic_query: Option<Vec<f32>>,
    /// Similarity threshold for semantic filtering (default 0.5).
    pub semantic_threshold: f32,
    /// Optional minimum coherence score for write-attention gating.
    pub min_coherence: Option<f32>,
    /// Maximum number of Gaussians to return (0 = unlimited).
    pub max_results: u32,
}

impl Default for SceneQuery {
    fn default() -> Self {
        Self {
            view_proj: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            time: 0.0,
            semantic_query: None,
            semantic_threshold: 0.5,
            min_coherence: None,
            max_results: 0,
        }
    }
}

/// Result of a scene query.
#[derive(Clone, Debug)]
pub struct QueryResult {
    /// Indices of Gaussians that passed all stages.
    pub indices: Vec<u32>,
    /// Active mask for the surviving Gaussians.
    pub active_mask: ActiveMask,
    /// Per-stage attention statistics.
    pub stats: AttentionStats,
}

/// Execute a scene query against a set of Gaussians.
///
/// Constructs an [`AttentionPipeline`] from the query parameters and runs it.
/// Optionally truncates results to `max_results`.
pub fn execute_query(
    query: &SceneQuery,
    gaussians: &[Gaussian4D],
    embeddings: Option<&[Vec<f32>]>,
    coherence_scores: Option<&[f32]>,
) -> QueryResult {
    let semantic = query.semantic_query.as_ref().map(|q| {
        SemanticAttention::new(q, query.semantic_threshold)
    });

    let write = query.min_coherence.map(WriteAttention::new);

    let pipeline = AttentionPipeline {
        view: ViewAttention::from_view_proj(&query.view_proj),
        temporal: TemporalAttention::new(query.time),
        semantic,
        write,
    };

    let mut result = pipeline.execute(gaussians, embeddings, coherence_scores);

    // Truncate if max_results is set
    if query.max_results > 0 && result.surviving_indices.len() > query.max_results as usize {
        result.surviving_indices.truncate(query.max_results as usize);
    }

    let total = gaussians.len() as u32;
    let mut mask = ActiveMask::new(total);
    for &idx in &result.surviving_indices {
        mask.set(idx, true);
    }

    QueryResult {
        indices: result.surviving_indices,
        active_mask: mask,
        stats: result.stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::Gaussian4D;

    fn make_gaussian(pos: [f32; 3], id: u32, time: [f32; 2]) -> Gaussian4D {
        let mut g = Gaussian4D::new(pos, id);
        g.time_range = time;
        g
    }

    fn identity_vp() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    #[test]
    fn test_default_query() {
        let q = SceneQuery::default();
        assert_eq!(q.time, 0.0);
        assert!(q.semantic_query.is_none());
        assert!(q.min_coherence.is_none());
        assert_eq!(q.max_results, 0);
    }

    #[test]
    fn test_execute_query_all_pass() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0, [f32::NEG_INFINITY, f32::INFINITY]),
            make_gaussian([0.1, 0.0, 0.0], 1, [f32::NEG_INFINITY, f32::INFINITY]),
        ];
        let query = SceneQuery {
            view_proj: identity_vp(),
            time: 0.0,
            ..SceneQuery::default()
        };
        let result = execute_query(&query, &gaussians, None, None);
        assert_eq!(result.indices.len(), 2);
        assert_eq!(result.stats.input_count, 2);
        assert_eq!(result.active_mask.active_count(), 2);
    }

    #[test]
    fn test_execute_query_temporal_filter() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0, [0.0, 3.0]),
            make_gaussian([0.1, 0.0, 0.0], 1, [0.0, 10.0]),
        ];
        let query = SceneQuery {
            view_proj: identity_vp(),
            time: 5.0,
            ..SceneQuery::default()
        };
        let result = execute_query(&query, &gaussians, None, None);
        assert_eq!(result.indices.len(), 1);
        assert_eq!(result.indices[0], 1);
    }

    #[test]
    fn test_execute_query_max_results() {
        let gaussians: Vec<Gaussian4D> = (0..10)
            .map(|i| make_gaussian([i as f32 * 0.05, 0.0, 0.0], i, [f32::NEG_INFINITY, f32::INFINITY]))
            .collect();
        let query = SceneQuery {
            view_proj: identity_vp(),
            time: 0.0,
            max_results: 3,
            ..SceneQuery::default()
        };
        let result = execute_query(&query, &gaussians, None, None);
        assert_eq!(result.indices.len(), 3);
    }

    #[test]
    fn test_execute_query_with_semantic() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0, [f32::NEG_INFINITY, f32::INFINITY]),
            make_gaussian([0.1, 0.0, 0.0], 1, [f32::NEG_INFINITY, f32::INFINITY]),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let query = SceneQuery {
            view_proj: identity_vp(),
            time: 0.0,
            semantic_query: Some(vec![1.0, 0.0, 0.0]),
            semantic_threshold: 0.5,
            ..SceneQuery::default()
        };
        let result = execute_query(&query, &gaussians, Some(&embeddings), None);
        assert_eq!(result.indices.len(), 1);
        assert_eq!(result.indices[0], 0);
    }

    #[test]
    fn test_execute_query_with_coherence() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0, [f32::NEG_INFINITY, f32::INFINITY]),
            make_gaussian([0.1, 0.0, 0.0], 1, [f32::NEG_INFINITY, f32::INFINITY]),
        ];
        let scores = vec![0.9, 0.3];
        let query = SceneQuery {
            view_proj: identity_vp(),
            time: 0.0,
            min_coherence: Some(0.5),
            ..SceneQuery::default()
        };
        let result = execute_query(&query, &gaussians, None, Some(&scores));
        assert_eq!(result.indices.len(), 1);
        assert_eq!(result.indices[0], 0);
    }

    #[test]
    fn test_execute_query_empty_gaussians() {
        let query = SceneQuery::default();
        let result = execute_query(&query, &[], None, None);
        assert!(result.indices.is_empty());
        assert_eq!(result.stats.input_count, 0);
    }
}
