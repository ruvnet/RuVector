/// Page-coherent agent memory: vector stores organized into coherent pages.
///
/// Instead of retrieving individual vectors, this library groups vectors into
/// "pages" of semantically related items. An agent loads an entire coherent page
/// into its context window, improving context quality and enabling page-level
/// prefetch on SSD-backed storage (DiskANN-style).
///
/// Three implementations are provided:
///   - FlatStore:          all vectors in a single page (baseline, exhaustive)
///   - CentroidPageStore:  k-means style centroid clustering into pages
///   - GreedyCoherenceStore: greedy similarity-based page construction
pub mod centroid;
pub mod flat;
pub mod greedy;

use std::collections::HashSet;

/// A single coherent page of vectors.
#[derive(Clone)]
pub struct VecPage {
    /// (vector_id, vector_data) pairs in this page.
    pub vecs: Vec<(usize, Vec<f32>)>,
    /// Unit-normalized centroid of this page's vectors.
    pub centroid: Vec<f32>,
    /// Average pairwise cosine similarity within the page (0..1 for unit vecs).
    pub coherence: f32,
}

impl VecPage {
    /// Build a page from a set of vectors, computing centroid and coherence.
    pub fn from_vecs(vecs: Vec<(usize, Vec<f32>)>) -> Self {
        assert!(!vecs.is_empty(), "page must not be empty");
        let dim = vecs[0].1.len();
        let n = vecs.len();

        let mut centroid = vec![0.0f32; dim];
        for (_, v) in &vecs {
            for (c, x) in centroid.iter_mut().zip(v.iter()) {
                *c += x / n as f32;
            }
        }
        let len: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if len > 1e-9 {
            for c in &mut centroid {
                *c /= len;
            }
        }

        // Sample up to 10 vectors for coherence to keep O(1) per page.
        let sample = n.min(10);
        let coherence = if sample < 2 {
            1.0
        } else {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for i in 0..sample {
                for j in (i + 1)..sample {
                    sum += dot(&vecs[i].1, &vecs[j].1);
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f32
            } else {
                0.0
            }
        };

        Self {
            vecs,
            centroid,
            coherence,
        }
    }
}

/// Statistics returned after building a store.
pub struct BuildStats {
    pub page_count: usize,
    pub avg_coherence: f32,
    pub build_ms: u128,
}

/// Results from a single search.
pub struct SearchResult {
    pub ids: Vec<usize>,
    pub scores: Vec<f32>,
    /// Number of pages probed during this search.
    pub pages_probed: usize,
}

/// Core trait every page store must implement.
pub trait PageStore: Send + Sync {
    fn name(&self) -> &str;
    /// Consume a batch of (id, vector) pairs and build internal page structure.
    fn build_from(&mut self, data: Vec<(usize, Vec<f32>)>) -> BuildStats;
    /// Return the top-k results probing at most `probe` pages.
    fn search(&self, query: &[f32], k: usize, probe: usize) -> SearchResult;
    fn page_count(&self) -> usize;
    fn avg_page_coherence(&self) -> f32;
}

// ─── utility ──────────────────────────────────────────────────────────────────

#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn normalize(v: &mut [f32]) {
    let len: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if len > 1e-9 {
        for x in v.iter_mut() {
            *x /= len;
        }
    }
}

/// Minimal LCG for deterministic dataset generation (no external deps).
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    pub fn next_usize(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as usize % n
    }
}

/// Generate `n` random unit vectors of dimension `dim`.
pub fn gen_unit_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            normalize(&mut v);
            v
        })
        .collect()
}

/// Exhaustive brute-force search returning top-k ids by cosine similarity.
pub fn brute_force(data: &[(usize, Vec<f32>)], query: &[f32], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = data.iter().map(|(id, v)| (*id, dot(query, v))).collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores.into_iter().map(|(id, _)| id).collect()
}

/// Recall@k: fraction of truth ids found in `got`.
pub fn recall(got: &[usize], truth: &[usize]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_set: HashSet<usize> = truth.iter().copied().collect();
    let hits = got.iter().filter(|id| truth_set.contains(id)).count();
    hits as f32 / truth.len() as f32
}

/// Compute p-th percentile of a sorted slice (0..100).
pub fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{centroid::CentroidPageStore, flat::FlatStore, greedy::GreedyCoherenceStore};

    fn make_data(n: usize, dim: usize) -> Vec<(usize, Vec<f32>)> {
        gen_unit_vecs(n, dim, 42).into_iter().enumerate().collect()
    }

    #[test]
    fn flat_store_recall_perfect() {
        let data = make_data(500, 32);
        let mut store = FlatStore::default();
        store.build_from(data.clone());
        let query = &data[0].1.clone();
        let res = store.search(query, 10, 1);
        let truth = brute_force(&data, query, 10);
        assert_eq!(
            recall(&res.ids, &truth),
            1.0,
            "flat store must have perfect recall"
        );
    }

    #[test]
    fn centroid_store_builds_correct_pages() {
        let data = make_data(400, 32);
        let mut store = CentroidPageStore::new(10, 4);
        let stats = store.build_from(data);
        assert_eq!(stats.page_count, 10);
        assert!(stats.avg_coherence > -1.0 && stats.avg_coherence <= 1.0);
        assert!(store.avg_page_coherence() >= 0.0);
    }

    #[test]
    fn greedy_store_recall_above_threshold() {
        let data = make_data(500, 32);
        let queries = gen_unit_vecs(20, 32, 99);
        let mut store = GreedyCoherenceStore::new(50, 5);
        store.build_from(data.clone());
        let full_data: Vec<(usize, Vec<f32>)> = data;
        let mut total_recall = 0.0f32;
        for q in &queries {
            let res = store.search(q, 10, 5);
            let truth = brute_force(&full_data, q, 10);
            total_recall += recall(&res.ids, &truth);
        }
        let avg = total_recall / queries.len() as f32;
        // Greedy coherence with 5/10 pages probed (50%) should achieve >=35% recall.
        // Greedy maximizes local coherence (not k-means cluster quality), so recall
        // is lower than centroid paging at the same probe fraction.
        assert!(avg >= 0.35, "greedy coherence recall {avg:.3} < 0.35");
    }

    #[test]
    fn centroid_store_recall_above_threshold() {
        let data = make_data(500, 32);
        let queries = gen_unit_vecs(20, 32, 77);
        let mut store = CentroidPageStore::new(10, 4);
        store.build_from(data.clone());
        let full_data: Vec<(usize, Vec<f32>)> = data;
        let mut total_recall = 0.0f32;
        for q in &queries {
            let res = store.search(q, 10, 4);
            let truth = brute_force(&full_data, q, 10);
            total_recall += recall(&res.ids, &truth);
        }
        let avg = total_recall / queries.len() as f32;
        // Centroid paging probing 4/10 pages should achieve >=35% recall on random unit vecs
        assert!(avg >= 0.35, "centroid store recall {avg:.3} < 0.35");
    }

    #[test]
    fn page_coherence_greedy_beats_flat() {
        let data = make_data(400, 32);
        let mut greedy = GreedyCoherenceStore::new(40, 4);
        let s_g = greedy.build_from(data.clone());
        let mut flat = FlatStore::default();
        flat.build_from(data);
        let flat_coherence = flat.avg_page_coherence();
        // Greedy pages should be more coherent than the flat (whole-dataset) average
        assert!(
            s_g.avg_coherence > flat_coherence,
            "greedy coherence {:.4} not > flat {:.4}",
            s_g.avg_coherence,
            flat_coherence
        );
    }

    #[test]
    fn dot_and_normalize_correct() {
        let mut v = vec![3.0f32, 4.0];
        normalize(&mut v);
        assert!(
            (dot(&v, &v) - 1.0).abs() < 1e-6,
            "normalized vector should have unit dot"
        );
    }
}
