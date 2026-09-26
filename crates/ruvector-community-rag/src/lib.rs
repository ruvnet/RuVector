//! MinCut-Partitioned Community Graph-RAG for Agent Memory Coherence
//!
//! Agent memory graphs develop community structure as agents work on related tasks.
//! Standard ANN is community-blind: it retrieves semantically close vectors
//! regardless of coherent task context. By partitioning memory with a graph
//! connectivity threshold (analogous to a mincut boundary), we can:
//!
//! 1. Identify coherent task communities at index time.
//! 2. During retrieval, surface vectors that share the query's community context.
//! 3. Trade ANN recall for community precision when context coherence matters.
//!
//! Three variants implement the [`CommunitySearch`] trait:
//!
//! | Variant        | Strategy                          | Best for          |
//! |----------------|-----------------------------------|-------------------|
//! | `FlatScan`     | Exact L2 brute-force              | Ground truth      |
//! | `GraphHop`     | k-NN scan + 1-hop expansion       | Cross-community   |
//! | `CommunityRAG` | Community centroid + member rerank| Intra-community   |

pub mod community;
pub mod community_rag;
pub mod flat_scan;
pub mod graph_hop;

/// Minimal 64-bit LCG: Knuth's multiplicative constants.
/// Deterministic, no-std, no external deps.
pub struct Lcg64(u64);

impl Lcg64 {
    pub fn new(seed: u64) -> Self { Self(seed ^ 0x9e37_79b9_7f4a_7c15) }

    /// Next value in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Take top 24 bits for mantissa precision.
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Next value in [lo, hi).
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

/// A retrieved candidate from search.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// Index of the vector in the database.
    pub id: usize,
    /// Approximate L2 distance to the query.
    pub distance: f32,
}

/// Metadata attached to each stored vector.
#[derive(Debug, Clone)]
pub struct VectorMeta {
    /// Index into the corpus.
    pub id: usize,
    /// True community label (ground truth cluster, 0-indexed).
    pub community: usize,
}

/// Unified trait for all community-aware search backends.
pub trait CommunitySearch {
    /// Insert a vector with its ground-truth community label.
    fn insert(&mut self, vector: &[f32], community: usize);

    /// Finalise the index (build graph, detect communities, etc.).
    /// Must be called after all inserts and before any search.
    fn build(&mut self);

    /// Return the top-k approximate nearest neighbours for `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;

    /// Estimated heap memory used by this index, in bytes.
    fn memory_bytes(&self) -> usize;

    /// Human-readable variant name for reporting.
    fn name(&self) -> &'static str;
}

/// Euclidean (L2) distance squared between two equal-length slices.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine similarity between two equal-length slices.
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Synthetic dataset: K tight Gaussian clusters in D dimensions.
///
/// Each cluster is centred at a random unit vector scaled to 5.0 units.
/// Vectors within each cluster are sampled by adding uniform noise in [-σ, σ].
/// Returns `(vectors, community_labels)`.
pub fn generate_clustered_dataset(
    n: usize,
    dims: usize,
    k_communities: usize,
    sigma: f32,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = Lcg64::new(seed);
    let per = (n + k_communities - 1) / k_communities;

    // Cluster centres: random unit vectors scaled to radius 5.0.
    let centres: Vec<Vec<f32>> = (0..k_communities)
        .map(|_| {
            let raw: Vec<f32> = (0..dims).map(|_| rng.range_f32(-1.0, 1.0)).collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            raw.into_iter().map(|x| x / norm * 5.0).collect()
        })
        .collect();

    let mut vectors = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for c in 0..k_communities {
        let count = if c < k_communities - 1 { per } else { n.saturating_sub(vectors.len()) };
        for _ in 0..count {
            let v: Vec<f32> = centres[c]
                .iter()
                .map(|&cx| cx + rng.range_f32(-sigma, sigma))
                .collect();
            vectors.push(v);
            labels.push(c);
        }
    }

    (vectors, labels)
}

/// Compute recall@k: fraction of ground-truth top-k ids present in retrieved ids.
pub fn recall_at_k(retrieved: &[Hit], ground_truth: &[Hit], k: usize) -> f32 {
    let k = k.min(retrieved.len()).min(ground_truth.len());
    if k == 0 { return 0.0; }
    let gt_ids: std::collections::HashSet<usize> = ground_truth[..k].iter().map(|h| h.id).collect();
    let matches = retrieved[..k].iter().filter(|h| gt_ids.contains(&h.id)).count();
    matches as f32 / k as f32
}

/// Community precision: among top-k retrieved, fraction in the same true community as the query.
pub fn community_precision(
    retrieved: &[Hit],
    metas: &[VectorMeta],
    query_community: usize,
    k: usize,
) -> f32 {
    let k = k.min(retrieved.len());
    if k == 0 { return 0.0; }
    let same = retrieved[..k]
        .iter()
        .filter(|h| metas[h.id].community == query_community)
        .count();
    same as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat_scan::FlatScan;
    use crate::graph_hop::GraphHop;
    use crate::community_rag::CommunityRAG;

    #[test]
    fn lcg_produces_values_in_range() {
        let mut rng = Lcg64::new(0);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "LCG out of [0,1): {v}");
        }
    }

    #[test]
    fn l2_sq_zero_for_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        assert_eq!(l2_sq(&a, &a), 0.0);
    }

    #[test]
    fn cosine_sim_one_for_identical() {
        let a = vec![1.0f32, 0.0, 0.0];
        let diff = (cosine_sim(&a, &a) - 1.0).abs();
        assert!(diff < 1e-6, "cosine_sim of identical vectors: {}", cosine_sim(&a, &a));
    }

    #[test]
    fn cosine_sim_zero_for_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_sim(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn dataset_has_correct_size_and_labels() {
        let (vecs, labels) = generate_clustered_dataset(100, 8, 5, 0.3, 1);
        assert_eq!(vecs.len(), 100);
        assert_eq!(labels.len(), 100);
        assert!(labels.iter().all(|&l| l < 5));
        assert!(vecs.iter().all(|v| v.len() == 8));
    }

    #[test]
    fn recall_at_k_perfect() {
        let gt = vec![Hit { id: 0, distance: 0.1 }, Hit { id: 1, distance: 0.2 }];
        let ret = vec![Hit { id: 0, distance: 0.1 }, Hit { id: 1, distance: 0.2 }];
        assert!((recall_at_k(&ret, &gt, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_at_k_zero() {
        let gt = vec![Hit { id: 0, distance: 0.1 }];
        let ret = vec![Hit { id: 99, distance: 5.0 }];
        assert!((recall_at_k(&ret, &gt, 1)).abs() < 1e-6);
    }

    #[test]
    fn flat_scan_returns_exact_nearest() {
        let (vecs, labels) = generate_clustered_dataset(200, 16, 4, 0.2, 42);
        let mut idx = FlatScan::new();
        for (v, l) in vecs.iter().zip(labels.iter()) { idx.insert(v, *l); }
        idx.build();
        let query = &vecs[0];
        let results = idx.search(query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0, "Nearest neighbour of a vector is itself");
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn graph_hop_recall_high_for_clustered_data() {
        let (vecs, labels) = generate_clustered_dataset(300, 16, 5, 0.2, 7);
        let mut flat = FlatScan::new();
        let mut hop = GraphHop::new(6, 30);
        for (v, l) in vecs.iter().zip(labels.iter()) {
            flat.insert(v, *l);
            hop.insert(v, *l);
        }
        flat.build();
        hop.build();
        let query = &vecs[150];
        let gt = flat.search(query, 10);
        let res = hop.search(query, 10);
        let r = recall_at_k(&res, &gt, 10);
        assert!(r >= 0.7, "GraphHop recall@10 too low: {r:.3}");
    }

    #[test]
    fn community_rag_high_precision_for_clustered_data() {
        let (vecs, labels) = generate_clustered_dataset(400, 16, 8, 0.25, 99);
        let mut flat = FlatScan::new();
        let mut rag = CommunityRAG::new(0.75);
        for (v, l) in vecs.iter().zip(labels.iter()) {
            flat.insert(v, *l);
            rag.insert(v, *l);
        }
        flat.build();
        rag.build();
        let metas = flat.metas();
        // Pick a query in the middle of the corpus.
        let qidx = 200;
        let query = &vecs[qidx];
        let qtrue = labels[qidx];
        let res = rag.search(query, 10);
        let prec = community_precision(&res, metas, qtrue, 10);
        assert!(prec >= 0.5, "CommunityRAG community_precision too low: {prec:.3}");
    }

    #[test]
    fn community_rag_detects_multiple_communities() {
        let (vecs, labels) = generate_clustered_dataset(200, 16, 5, 0.2, 55);
        let mut rag = CommunityRAG::new(0.80);
        for (v, l) in vecs.iter().zip(labels.iter()) { rag.insert(v, *l); }
        rag.build();
        assert!(rag.num_communities() >= 2, "Expected ≥2 communities, got {}", rag.num_communities());
    }

    #[test]
    fn memory_estimates_are_positive() {
        let (vecs, labels) = generate_clustered_dataset(100, 8, 3, 0.3, 1);
        let mut flat = FlatScan::new();
        let mut hop = GraphHop::new(4, 20);
        let mut rag = CommunityRAG::new(0.70);
        for (v, l) in vecs.iter().zip(labels.iter()) {
            flat.insert(v, *l);
            hop.insert(v, *l);
            rag.insert(v, *l);
        }
        flat.build(); hop.build(); rag.build();
        assert!(flat.memory_bytes() > 0);
        assert!(hop.memory_bytes() > flat.memory_bytes(), "GraphHop should use more memory");
        assert!(rag.memory_bytes() > 0);
    }
}
