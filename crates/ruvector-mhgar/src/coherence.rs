//! Coherence scoring for multi-hop stopping decisions.
//!
//! Coherence measures how well a set of candidate vectors "agree" on a
//! common direction relative to the query.  High coherence → the current
//! candidate set likely contains the true answers; low coherence → expanding
//! further via graph traversal may recover missed entities.

use crate::vector::cosine_distance;

/// Compute the mean cosine distance between the query and all candidate vectors.
/// Lower = candidates are on average closer to the query direction.
pub fn mean_query_distance(query: &[f32], candidate_vecs: &[Vec<f32>]) -> f32 {
    if candidate_vecs.is_empty() {
        return 1.0;
    }
    let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let total: f32 = candidate_vecs
        .iter()
        .map(|v| cosine_distance(query, v, q_norm))
        .sum();
    total / candidate_vecs.len() as f32
}

/// Pairwise mean cosine distance within the candidate set.
/// Low inter-candidate distance = high coherence = tight cluster.
pub fn intra_set_coherence(candidate_vecs: &[Vec<f32>]) -> f32 {
    let n = candidate_vecs.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut count = 0u32;
    for i in 0..n {
        let a = &candidate_vecs[i];
        let a_norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        for b in candidate_vecs.iter().skip(i + 1) {
            total += cosine_distance(a, b, a_norm);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total / count as f32 }
}

/// Decision: should we expand another hop?
///
/// Returns `true` (expand) when the candidate set's mean query-distance
/// is still above `distance_threshold`, meaning there may be better-matching
/// entities reachable via graph edges.
pub fn should_expand(
    query: &[f32],
    candidate_vecs: &[Vec<f32>],
    current_hop: u32,
    max_hops: u32,
    distance_threshold: f32,
) -> bool {
    if current_hop >= max_hops {
        return false;
    }
    if candidate_vecs.is_empty() {
        return true;
    }
    mean_query_distance(query, candidate_vecs) > distance_threshold
}
