//! Per-vector coherence scoring, reusing `ruvector-attn-mincut`'s dynamic
//! min-cut gate as a self-similarity graph pruner instead of an attention
//! operator.
//!
//! For a block of vectors (here: one synthetic memory cluster), we build a
//! directed self-similarity graph (cosine similarity as edge weight, self
//! loops removed) ordered so that node 0 is the point closest to the block
//! centroid and node `m-1` is the point farthest from it. Running
//! [`ruvector_attn_mincut::dynamic_min_cut`] on that graph with that
//! source/sink choice asks exactly the question we want answered: "how much
//! of this point's similarity structure survives the cut that separates the
//! block's core from its fringe?"
//!
//! `coherence(i) = kept_incident_edges(i) / total_incident_edges(i)`, in
//! `[0, 1]`. High coherence => `i` sits in the dense, redundant interior of
//! its cluster (safe to quantize aggressively). Low coherence => `i` is a
//! boundary/outlier point whose nearest neighbors mostly got cut away (needs
//! precision kept).

use ruvector_attn_mincut::{dynamic_min_cut, MinCutConfig};

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

/// Coherence score per vector in `block`, aligned with `block`'s input
/// order. `centroid` is used only to pick a meaningful source/sink pair for
/// the global min-cut, not for distance scoring.
pub fn block_coherence(block: &[Vec<f32>], centroid: &[f32], cfg: &MinCutConfig) -> Vec<f32> {
    let m = block.len();
    if m < 3 {
        return vec![1.0; m];
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        dist2(&block[a], centroid)
            .partial_cmp(&dist2(&block[b], centroid))
            .unwrap()
    });

    let mut logits = vec![0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            if i != j {
                logits[i * m + j] = cosine(&block[order[i]], &block[order[j]]);
            }
        }
    }

    let gating = dynamic_min_cut(&logits, m, cfg.lambda, cfg.tau, cfg.eps);

    let mut kept = vec![0u32; m];
    let mut total = vec![0u32; m];
    for i in 0..m {
        for j in 0..m {
            if i == j || logits[i * m + j] <= cfg.eps {
                continue;
            }
            total[i] += 1;
            total[j] += 1;
            if gating.keep_mask[i * m + j] {
                kept[i] += 1;
                kept[j] += 1;
            }
        }
    }

    let mut coherence = vec![0f32; m];
    for i in 0..m {
        coherence[order[i]] = if total[i] == 0 {
            0.0
        } else {
            kept[i] as f32 / total[i] as f32
        };
    }
    coherence
}

/// Score every vector in `vectors`, grouped into contiguous blocks by
/// `block_of[i]` (e.g. the cluster id each vector belongs to). Blocks are
/// processed independently, which keeps the O(V^2) min-cut bounded by block
/// size instead of corpus size — the realistic regime for a sharded /
/// streaming agent-memory store.
pub fn corpus_coherence(
    vectors: &[Vec<f32>],
    block_of: &[u32],
    centroids: &[Vec<f32>],
    cfg: &MinCutConfig,
) -> Vec<f32> {
    let n = vectors.len();
    let mut scores = vec![0f32; n];
    let n_blocks = centroids.len();
    for b in 0..n_blocks {
        let idxs: Vec<usize> = (0..n).filter(|&i| block_of[i] as usize == b).collect();
        if idxs.is_empty() {
            continue;
        }
        let block: Vec<Vec<f32>> = idxs.iter().map(|&i| vectors[i].clone()).collect();
        let block_scores = block_coherence(&block, &centroids[b], cfg);
        for (k, &i) in idxs.iter().enumerate() {
            scores[i] = block_scores[k];
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_block_defaults_to_coherent() {
        let block = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let scores = block_coherence(&block, &[0.5, 0.5], &MinCutConfig::default());
        assert_eq!(scores, vec![1.0, 1.0]);
    }

    #[test]
    fn deterministic_across_runs() {
        let mut rng = crate::rng::SplitMix64(7);
        let block: Vec<Vec<f32>> = (0..40).map(|_| rng.gaussian_vec(16)).collect();
        let centroid = vec![0.0f32; 16];
        let cfg = MinCutConfig::default();
        let a = block_coherence(&block, &centroid, &cfg);
        let b = block_coherence(&block, &centroid, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn scores_are_bounded() {
        let mut rng = crate::rng::SplitMix64(11);
        let block: Vec<Vec<f32>> = (0..40).map(|_| rng.gaussian_vec(16)).collect();
        let centroid = vec![0.0f32; 16];
        let scores = block_coherence(&block, &centroid, &MinCutConfig::default());
        assert_eq!(scores.len(), 40);
        assert!(scores.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }
}
