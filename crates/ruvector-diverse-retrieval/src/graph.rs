//! Graph-cut-inspired partitioning for candidate sets.
//!
//! Given a set of candidate vector indices, builds an approximate connectivity
//! graph (edge iff L2 distance < threshold) and extracts connected components
//! via union-find.  The resulting partition labels are used by
//! [`PartitionMmrRetriever`][crate::partition_mmr::PartitionMmrRetriever] to
//! penalise same-cluster selections during greedy MMR search.

use crate::l2;

/// Union-Find (Disjoint Set Union) with path compression and union by rank.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Create a new UF structure with `n` singleton sets.
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find canonical root of `x` with path compression.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Merge the sets containing `x` and `y`.
    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }

    /// Return a compact partition label in `0..num_partitions` for each of `n` elements.
    pub fn labels(&mut self, n: usize) -> Vec<usize> {
        let roots: Vec<usize> = (0..n).map(|i| self.find(i)).collect();
        let mut map = std::collections::HashMap::new();
        let mut next = 0usize;
        roots
            .iter()
            .map(|&r| {
                *map.entry(r).or_insert_with(|| {
                    let v = next;
                    next += 1;
                    v
                })
            })
            .collect()
    }

    /// Number of distinct sets currently in the structure.
    pub fn num_components(&mut self, n: usize) -> usize {
        let roots: std::collections::HashSet<usize> = (0..n).map(|i| self.find(i)).collect();
        roots.len()
    }
}

/// Assign partition labels to `candidates` (indices into `vectors`).
///
/// Two candidates are connected—and thus share a partition—when their L2
/// distance is strictly less than `threshold`.  The resulting labels are
/// compact: `0..num_partitions`.
pub fn partition_candidates(
    candidates: &[usize],
    vectors: &[Vec<f32>],
    threshold: f32,
) -> Vec<usize> {
    let n = candidates.len();
    let mut uf = UnionFind::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            let d = l2(&vectors[candidates[i]], &vectors[candidates[j]]);
            if d < threshold {
                uf.union(i, j);
            }
        }
    }

    uf.labels(n)
}

/// Estimate a connectivity threshold from the first `sample` candidates.
///
/// Uses `fraction` of the mean pairwise distance as the threshold.
/// A lower `fraction` → more, smaller partitions (stricter isolation).
/// A higher `fraction` → fewer, larger partitions (looser isolation).
pub fn estimate_threshold(
    candidates: &[usize],
    vectors: &[Vec<f32>],
    fraction: f32,
    sample: usize,
) -> f32 {
    let n = candidates.len().min(sample);
    if n < 2 {
        return f32::INFINITY;
    }
    let mut total = 0.0f32;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            total += l2(&vectors[candidates[i]], &vectors[candidates[j]]);
            count += 1;
        }
    }
    if count == 0 {
        return f32::INFINITY;
    }
    (total / count as f32) * fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_find_singletons() {
        let mut uf = UnionFind::new(3);
        let labels = uf.labels(3);
        // all singletons: three distinct labels
        let distinct: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert_eq!(distinct.len(), 3);
    }

    #[test]
    fn union_find_two_components() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(2, 3);
        let labels = uf.labels(4);
        assert_eq!(labels[0], labels[1], "0 and 1 should share a label");
        assert_eq!(labels[2], labels[3], "2 and 3 should share a label");
        assert_ne!(labels[0], labels[2], "groups should differ");
    }

    #[test]
    fn partition_nearby_vectors() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0f32, 0.0],
            vec![0.1f32, 0.0], // near [0]
            vec![10.0f32, 0.0],
            vec![10.1f32, 0.0], // near [2]
        ];
        let candidates = vec![0, 1, 2, 3];
        let labels = partition_candidates(&candidates, &vectors, 1.0);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn estimate_threshold_basic() {
        let vectors: Vec<Vec<f32>> = vec![vec![0.0f32], vec![2.0f32], vec![4.0f32]];
        let candidates = vec![0, 1, 2];
        let t = estimate_threshold(&candidates, &vectors, 0.5, 10);
        // mean pairwise distance: (2 + 4 + 2) / 3 = 8/3 ≈ 2.667; * 0.5 ≈ 1.333
        assert!(t > 1.0 && t < 2.0, "threshold={t}");
    }
}
