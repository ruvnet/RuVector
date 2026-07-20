//! # ruvector-tegraph
//!
//! **Typed-Edge Navigable Graph (TENG)** — hybrid vector + knowledge-graph
//! retrieval for RuVector agent memory and graph RAG workloads.
//!
//! ## Core idea
//!
//! Current graph-RAG systems (GraphRAG, HippoRAG, LightRAG) keep the vector
//! index and knowledge graph as **separate structures**: ANN retrieves
//! candidates, then a second graph-traversal pass re-ranks them.  TENG
//! eliminates the two-pass overhead by weaving typed semantic edges into
//! the NSW navigation beam itself.
//!
//! ## Three measured retrieval variants
//!
//! | Variant | Edges used during navigation | Best for |
//! |---------|------------------------------|----------|
//! | [`variants::TengIndex::search_vector_only`] | Vector-proximity only | Baseline ANN |
//! | [`variants::TengIndex::search_edge_expand`] | Vector + semantic edge walk | High semantic recall |
//! | [`variants::TengIndex::search_edge_constrained`] | Vector, filtered by edge predicate | Access-controlled retrieval |
//!
//! ## Supported edge types
//!
//! [`types::EdgeType`]: SameDocument · References · CoOccurs · Temporal · Causal

pub mod dataset;
pub mod graph;
pub mod types;
pub mod variants;

pub use types::{EdgeType, Node, TypedEdge};
pub use variants::{EdgeConstraint, TengIndex};

// ─── distance / recall utilities ─────────────────────────────────────────────

/// Inner-product similarity (assumes pre-normalised unit vectors).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalise a mutable f32 slice to unit length.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() { *x /= norm; }
    }
}

/// Recall\@k: fraction of `ground_truth`'s first k entries that appear in
/// the first k entries of `returned`.
pub fn recall_at_k(returned: &[(usize, f32)], ground_truth: &[usize], k: usize) -> f32 {
    let ret: std::collections::HashSet<usize> =
        returned.iter().take(k).map(|r| r.0).collect();
    let hits = ground_truth.iter().take(k).filter(|id| ret.contains(*id)).count();
    let denom = k.min(ground_truth.len());
    if denom == 0 { return 1.0; }
    hits as f32 / denom as f32
}

/// Same as `recall_at_k` but ground truth is already `(id, score)` pairs.
pub fn recall_at_k_pairs(returned: &[(usize, f32)], gt: &[(usize, f32)], k: usize) -> f32 {
    let gt_ids: Vec<usize> = gt.iter().map(|(id, _)| *id).collect();
    recall_at_k(returned, &gt_ids, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_perfect() {
        let returned = vec![(0, 0.9), (1, 0.8), (2, 0.7)];
        let gt       = vec![0, 1, 2];
        assert!((recall_at_k(&returned, &gt, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_zero() {
        let returned = vec![(3, 0.5), (4, 0.4)];
        let gt       = vec![0, 1, 2];
        assert!(recall_at_k(&returned, &gt, 3) < 1e-6);
    }

    #[test]
    fn dot_unit_vectors() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        assert!((dot(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_makes_unit() {
        let mut v = vec![3.0_f32, 4.0];
        normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
