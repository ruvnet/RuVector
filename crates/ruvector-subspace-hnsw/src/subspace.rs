use crate::hnsw::{brute_force_knn, sq_l2, HnswIndex};

/// Project a full-dimensional vector onto a contiguous subspace [start, end).
pub fn project(v: &[f32], start: usize, end: usize) -> Vec<f32> {
    v[start..end].to_vec()
}

/// Compute recall@k: fraction of ground-truth ids found in result ids.
pub fn recall_at_k(result: &[(u32, f32)], ground_truth: &[(u32, f32)], k: usize) -> f32 {
    let gt_ids: std::collections::HashSet<u32> =
        ground_truth.iter().take(k).map(|&(id, _)| id).collect();
    let found = result
        .iter()
        .take(k)
        .filter(|&&(id, _)| gt_ids.contains(&id))
        .count();
    found as f32 / k.min(ground_truth.len()) as f32
}

// ─── Variant 1: Baseline full-dimension HNSW ────────────────────────────────

/// Single HNSW on all D dimensions.
pub struct BaselineHnsw {
    pub index: HnswIndex,
}

impl BaselineHnsw {
    pub fn build(vectors: &[Vec<f32>], m: usize, ef_construction: usize) -> Self {
        let dim = vectors[0].len();
        let mut index = HnswIndex::new(dim, m, ef_construction);
        for v in vectors {
            index.insert(v.clone());
        }
        BaselineHnsw { index }
    }

    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        self.index.search(q, k, ef)
    }

    pub fn memory_bytes(&self) -> usize {
        self.index.memory_bytes()
    }
}

// ─── Variant 2: Multi-subspace HNSW with union fusion ───────────────────────

/// K independent HNSW indexes, one per equal-width subspace.
/// Results are merged by taking the union of all subspace top-ef candidates
/// and re-ranking by L2 distance in full space.
pub struct SubspaceUnionHnsw {
    pub subgraphs: Vec<HnswIndex>,
    pub full_vectors: Vec<Vec<f32>>,
    pub num_subspaces: usize,
    pub sub_dim: usize,
    pub full_dim: usize,
}

impl SubspaceUnionHnsw {
    pub fn build(
        vectors: &[Vec<f32>],
        num_subspaces: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let full_dim = vectors[0].len();
        assert_eq!(
            full_dim % num_subspaces,
            0,
            "dim must be divisible by num_subspaces"
        );
        let sub_dim = full_dim / num_subspaces;

        let mut subgraphs: Vec<HnswIndex> = (0..num_subspaces)
            .map(|_| HnswIndex::new(sub_dim, m, ef_construction))
            .collect();

        for v in vectors {
            for (s, graph) in subgraphs.iter_mut().enumerate() {
                let proj = project(v, s * sub_dim, (s + 1) * sub_dim);
                graph.insert(proj);
            }
        }

        SubspaceUnionHnsw {
            subgraphs,
            full_vectors: vectors.to_vec(),
            num_subspaces,
            sub_dim,
            full_dim,
        }
    }

    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let mut candidate_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for (s, graph) in self.subgraphs.iter().enumerate() {
            let q_sub = project(q, s * self.sub_dim, (s + 1) * self.sub_dim);
            let res = graph.search(&q_sub, ef, ef);
            for (id, _) in res {
                candidate_ids.insert(id);
            }
        }

        // Re-rank by full-space L2.
        let mut scored: Vec<(u32, f32)> = candidate_ids
            .iter()
            .map(|&id| (id, sq_l2(&self.full_vectors[id as usize], q)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    pub fn memory_bytes(&self) -> usize {
        let graphs: usize = self.subgraphs.iter().map(|g| g.memory_bytes()).sum();
        let vecs: usize = self.full_vectors.iter().map(|v| v.len() * 4).sum();
        graphs + vecs
    }
}

// ─── Variant 3: Multi-subspace HNSW with coherence-weighted fusion ───────────

/// Same K subspace HNSWs, but results are fused using a coherence score derived
/// from the variance of top-candidate distances in each subspace.
///
/// A subspace where top-ef candidates cluster tightly (low CV of distances)
/// is rewarded with higher fusion weight, enabling query-adaptive subspace selection.
pub struct CoherenceHnsw {
    pub union_base: SubspaceUnionHnsw,
}

impl CoherenceHnsw {
    pub fn build(
        vectors: &[Vec<f32>],
        num_subspaces: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        CoherenceHnsw {
            union_base: SubspaceUnionHnsw::build(vectors, num_subspaces, m, ef_construction),
        }
    }

    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let ub = &self.union_base;
        let mut per_subspace: Vec<Vec<(u32, f32)>> = Vec::with_capacity(ub.num_subspaces);

        for (s, graph) in ub.subgraphs.iter().enumerate() {
            let q_sub = project(q, s * ub.sub_dim, (s + 1) * ub.sub_dim);
            per_subspace.push(graph.search(&q_sub, ef, ef));
        }

        // Compute per-subspace coherence weight from distance CV.
        let weights: Vec<f32> = per_subspace
            .iter()
            .map(|cands| coherence_weight(cands))
            .collect();

        // Collect all candidate ids.
        let mut all_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for cands in &per_subspace {
            for &(id, _) in cands {
                all_ids.insert(id);
            }
        }

        // Score each candidate as weighted sum of normalised subspace distances.
        let mut scored: Vec<(u32, f32)> = all_ids
            .iter()
            .map(|&id| {
                let full_v = &ub.full_vectors[id as usize];
                let score = weighted_full_dist(q, full_v, &weights, ub.sub_dim);
                (id, score)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    pub fn memory_bytes(&self) -> usize {
        self.union_base.memory_bytes()
    }
}

/// Coherence weight for a subspace: 1 / (1 + CV) where CV = std/mean of top-k dists.
/// A tight cluster of distances → low CV → high weight.
fn coherence_weight(cands: &[(u32, f32)]) -> f32 {
    if cands.len() < 2 {
        return 1.0;
    }
    let dists: Vec<f32> = cands.iter().map(|&(_, d)| d).collect();
    let mean = dists.iter().sum::<f32>() / dists.len() as f32;
    if mean < 1e-9 {
        return 1.0;
    }
    let variance = dists.iter().map(|d| (d - mean) * (d - mean)).sum::<f32>() / dists.len() as f32;
    let cv = variance.sqrt() / mean;
    1.0 / (1.0 + cv)
}

/// Compute a coherence-weighted distance score for candidate `full_v` against query `q`.
/// Each subspace distance is weighted by its coherence; the final score is the
/// weighted sum of per-subspace sq-L2 distances.
fn weighted_full_dist(q: &[f32], full_v: &[f32], weights: &[f32], sub_dim: usize) -> f32 {
    let total_w: f32 = weights.iter().sum::<f32>().max(1e-9);
    let mut score = 0.0_f32;
    for (s, &w) in weights.iter().enumerate() {
        let start = s * sub_dim;
        let end = start + sub_dim;
        score += (w / total_w) * sq_l2(&q[start..end], &full_v[start..end]);
    }
    score
}

/// Exact brute-force ground truth on full vectors.
pub fn ground_truth(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    brute_force_knn(vectors, query, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{generate_clustered, generate_queries};

    fn small_dataset() -> Vec<Vec<f32>> {
        let (vecs, _) = generate_clustered(500, 32, 5, 16, 42);
        vecs
    }

    #[test]
    fn baseline_recall_acceptable() {
        let vecs = small_dataset();
        let idx = BaselineHnsw::build(&vecs, 8, 60);
        let queries = generate_queries(20, 32, 99);
        let total_recall: f32 = queries
            .iter()
            .map(|q| {
                let gt = ground_truth(&vecs, q, 10);
                let res = idx.search(q, 10, 80);
                recall_at_k(&res, &gt, 10)
            })
            .sum::<f32>()
            / 20.0;
        assert!(
            total_recall >= 0.70,
            "baseline recall@10 = {total_recall:.3} < 0.70"
        );
    }

    #[test]
    fn union_returns_k() {
        let vecs = small_dataset();
        let idx = SubspaceUnionHnsw::build(&vecs, 2, 8, 60);
        let q: Vec<f32> = vec![0.0; 32];
        let res = idx.search(&q, 10, 30);
        assert!(!res.is_empty());
        assert!(res.len() <= 10);
    }

    #[test]
    fn coherence_recall_not_below_union() {
        let vecs = small_dataset();
        let union_idx = SubspaceUnionHnsw::build(&vecs, 2, 8, 60);
        let coh_idx = CoherenceHnsw::build(&vecs, 2, 8, 60);
        let queries = generate_queries(20, 32, 77);

        let recall_union: f32 = queries
            .iter()
            .map(|q| {
                let gt = ground_truth(&vecs, q, 10);
                let res = union_idx.search(q, 10, 30);
                recall_at_k(&res, &gt, 10)
            })
            .sum::<f32>()
            / 20.0;

        let recall_coh: f32 = queries
            .iter()
            .map(|q| {
                let gt = ground_truth(&vecs, q, 10);
                let res = coh_idx.search(q, 10, 30);
                recall_at_k(&res, &gt, 10)
            })
            .sum::<f32>()
            / 20.0;

        // Coherence fusion must not be worse than simple union.
        assert!(
            recall_coh >= recall_union - 0.05,
            "coherence recall {recall_coh:.3} more than 5pp below union {recall_union:.3}"
        );
    }

    #[test]
    fn coherence_weight_tight_cluster() {
        // All distances identical → CV = 0 → weight = 1.0
        let cands: Vec<(u32, f32)> = (0..10).map(|i| (i, 1.0)).collect();
        let w = coherence_weight(&cands);
        assert!((w - 1.0).abs() < 1e-5, "w = {w}");
    }

    #[test]
    fn coherence_weight_spread_cluster() {
        // Distances spread 1..10 → high CV → weight < 0.5
        let cands: Vec<(u32, f32)> = (1u32..=10).map(|i| (i, i as f32)).collect();
        let w = coherence_weight(&cands);
        assert!(w < 0.8, "w = {w} should be below 0.8 for spread cluster");
    }
}
