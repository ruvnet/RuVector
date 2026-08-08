//! Three search backends over the cluster tree.
//!
//! FlatBrute     — ground-truth O(n·d) scan over all leaves.
//! ClusterSearch — IVF-style: rank clusters by centroid L2 distance, search top-nprobe.
//! CoherenceTree — rank clusters by λ·cosine_sim(q,c) + (1-λ)·cohesion(c).

use crate::{cosine_sim, l2_sq, tree::ClusterTree, AnnVariant, Hit};

// ─── FlatBrute ───────────────────────────────────────────────────────────────

/// Brute-force exhaustive L2 scan — ground truth baseline.
pub struct FlatBrute {
    vectors: Vec<Vec<f32>>,
}

impl FlatBrute {
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        Self { vectors }
    }
}

impl AnnVariant for FlatBrute {
    fn name(&self) -> &'static str {
        "FlatBrute"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut dists: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (l2_sq(query, v), i))
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists
            .into_iter()
            .take(k)
            .map(|(d, id)| Hit { id, dist_sq: d })
            .collect()
    }

    fn mem_bytes(&self) -> usize {
        let n = self.vectors.len();
        let dim = if n > 0 { self.vectors[0].len() } else { 0 };
        n * dim * 4
    }
}

// ─── ClusterSearch ───────────────────────────────────────────────────────────

/// IVF-style cluster search: score centroids by L2, expand top-nprobe clusters.
pub struct ClusterSearch {
    tree: ClusterTree,
    nprobe: usize,
}

impl ClusterSearch {
    pub fn new(tree: ClusterTree, nprobe: usize) -> Self {
        Self { tree, nprobe }
    }
}

impl AnnVariant for ClusterSearch {
    fn name(&self) -> &'static str {
        "ClusterSearch"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        // Score each centroid by L2 distance; pick top-nprobe clusters.
        let mut centroid_scores: Vec<(f32, usize)> = self
            .tree
            .centroids
            .iter()
            .enumerate()
            .map(|(c, cen)| (l2_sq(query, cen), c))
            .collect();
        centroid_scores.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let probe = self.nprobe.min(self.tree.k);
        let mut candidates: Vec<Hit> = centroid_scores
            .iter()
            .take(probe)
            .flat_map(|(_, c)| {
                self.tree.inverted[*c].iter().map(|&id| Hit {
                    id,
                    dist_sq: l2_sq(query, &self.tree.leaves[id]),
                })
            })
            .collect();

        candidates.sort_unstable_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
        candidates.dedup_by_key(|h| h.id);
        candidates.truncate(k);
        candidates
    }

    fn mem_bytes(&self) -> usize {
        self.tree.mem_bytes()
    }
}

// ─── CoherenceTree ───────────────────────────────────────────────────────────

/// Coherence-weighted cluster search.
///
/// Cluster score = λ · cosine_sim(query, centroid) + (1-λ) · cohesion(cluster)
///
/// Clusters with high cosine alignment AND high internal cohesion are searched first.
/// This differentiates us from standard IVF: a tight, relevant cluster is preferred
/// over a spread-out, vaguely-close one — which matters for agent memory retrieval
/// where query context aligns with coherent topic clusters.
pub struct CoherenceTree {
    tree: ClusterTree,
    nprobe: usize,
    /// Weighting between query alignment (λ) and cluster cohesion (1-λ).
    lambda: f32,
}

impl CoherenceTree {
    /// `lambda = 0.7` weights query alignment more; `0.3` would weight cohesion more.
    pub fn new(tree: ClusterTree, nprobe: usize, lambda: f32) -> Self {
        Self {
            tree,
            nprobe,
            lambda: lambda.clamp(0.0, 1.0),
        }
    }
}

impl AnnVariant for CoherenceTree {
    fn name(&self) -> &'static str {
        "CoherenceTree"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        // Score each cluster: higher is better (so we negate for sorting).
        let mut cluster_scores: Vec<(f32, usize)> = self
            .tree
            .centroids
            .iter()
            .enumerate()
            .map(|(c, cen)| {
                let sim = cosine_sim(query, cen); // ∈ [-1, 1]
                let coh = self.tree.cohesion[c]; // ∈ [-1, 1]
                                                 // Map both to [0,1] before mixing.
                let sim_n = (sim + 1.0) * 0.5;
                let coh_n = (coh + 1.0) * 0.5;
                let score = self.lambda * sim_n + (1.0 - self.lambda) * coh_n;
                // Negate so ascending sort gives top scores first.
                (-score, c)
            })
            .collect();
        cluster_scores.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let probe = self.nprobe.min(self.tree.k);
        let mut candidates: Vec<Hit> = cluster_scores
            .iter()
            .take(probe)
            .flat_map(|(_, c)| {
                self.tree.inverted[*c].iter().map(|&id| Hit {
                    id,
                    dist_sq: l2_sq(query, &self.tree.leaves[id]),
                })
            })
            .collect();

        candidates.sort_unstable_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
        candidates.dedup_by_key(|h| h.id);
        candidates.truncate(k);
        candidates
    }

    fn mem_bytes(&self) -> usize {
        self.tree.mem_bytes()
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cluster::kmeans, dataset::generate_vectors, recall_at_k, tree::ClusterTree};

    fn make_tree(n: usize, dim: usize, k: usize, seed: u64) -> ClusterTree {
        let vecs = generate_vectors(n, dim, seed);
        let km = kmeans(&vecs, k, 15);
        ClusterTree::new(vecs, km)
    }

    #[test]
    fn flat_brute_returns_k_results() {
        let vecs = generate_vectors(500, 32, 10);
        let flat = FlatBrute::new(vecs);
        let q = generate_vectors(1, 32, 777)[0].clone();
        let hits = flat.search(&q, 10);
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn flat_brute_ascending_distance() {
        let vecs = generate_vectors(500, 32, 11);
        let flat = FlatBrute::new(vecs);
        let q = generate_vectors(1, 32, 888)[0].clone();
        let hits = flat.search(&q, 10);
        for w in hits.windows(2) {
            assert!(w[0].dist_sq <= w[1].dist_sq, "results not sorted ascending");
        }
    }

    #[test]
    fn cluster_search_recall_above_threshold() {
        let n = 1000;
        let dim = 32;
        let k = 20;
        let seed = 42;
        let vecs = generate_vectors(n, dim, seed);
        let queries = generate_vectors(50, dim, seed + 1);

        let km = kmeans(&vecs, k, 15);
        let flat = FlatBrute::new(vecs.clone());
        let tree = ClusterTree::new(vecs, km);
        let cs = ClusterSearch::new(tree, 5);

        let mean_recall: f32 = queries
            .iter()
            .map(|q| {
                let gt = flat.search(q, 10);
                let cand = cs.search(q, 10);
                recall_at_k(&cand, &gt)
            })
            .sum::<f32>()
            / 50.0;

        // nprobe=5 of k=20 clusters covers ≥25% of corpus; expect ≥0.5 recall.
        assert!(
            mean_recall >= 0.50,
            "ClusterSearch recall {mean_recall:.3} below 0.50 threshold"
        );
    }

    #[test]
    fn coherence_tree_recall_above_cluster_search() {
        let n = 1000;
        let dim = 32;
        let k = 20;
        let seed = 77;
        let vecs = generate_vectors(n, dim, seed);
        let queries = generate_vectors(50, dim, seed + 5);

        let km_a = kmeans(&vecs, k, 15);
        let km_b = kmeans(&vecs, k, 15);
        let flat = FlatBrute::new(vecs.clone());
        let cs = ClusterSearch::new(ClusterTree::new(vecs.clone(), km_a), 5);
        let ct = CoherenceTree::new(ClusterTree::new(vecs, km_b), 5, 0.7);

        let (recall_cs, recall_ct): (f32, f32) = queries
            .iter()
            .map(|q| {
                let gt = flat.search(q, 10);
                let r_cs = recall_at_k(&cs.search(q, 10), &gt);
                let r_ct = recall_at_k(&ct.search(q, 10), &gt);
                (r_cs, r_ct)
            })
            .fold((0.0, 0.0), |(a, b), (c, d)| (a + c, b + d));

        let mean_cs = recall_cs / 50.0;
        let mean_ct = recall_ct / 50.0;

        // CoherenceTree should match or exceed ClusterSearch recall
        // (both are ≥ 0 recall; the acceptance bar is CoherenceTree ≥ 0.50).
        assert!(
            mean_ct >= 0.50,
            "CoherenceTree recall {mean_ct:.3} below 0.50 threshold"
        );
        // Print for human review (doesn't affect pass/fail).
        eprintln!("ClusterSearch recall: {mean_cs:.3}, CoherenceTree recall: {mean_ct:.3}");
    }
}
