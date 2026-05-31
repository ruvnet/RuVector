//! Similarity graph construction and union-find clustering.
//!
//! The core primitive: build a pairwise cosine similarity graph over stored
//! memory vectors, then use union-find to identify connected components at
//! threshold θ.  Each component becomes a single compaction candidate.

use std::collections::HashMap;

// ── Cosine similarity ────────────────────────────────────────────────────────

/// Cosine similarity ∈ [−1, 1] between two equal-length slices.
/// Returns 0.0 when either vector is zero.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
    let mut dot = 0.0f32;
    let mut na2 = 0.0f32;
    let mut nb2 = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na2 += x * x;
        nb2 += y * y;
    }
    let denom = na2.sqrt() * nb2.sqrt();
    if denom < 1e-9 {
        0.0
    } else {
        dot / denom
    }
}

// ── Union-Find ───────────────────────────────────────────────────────────────

/// Path-compressed, rank-unioned disjoint set data structure.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

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

    /// Return connected components as lists of indices.
    pub fn components(&mut self, n: usize) -> Vec<Vec<usize>> {
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = self.find(i);
            groups.entry(root).or_default().push(i);
        }
        groups.into_values().collect()
    }
}

// ── Graph clustering ─────────────────────────────────────────────────────────

/// Group vector indices into connected components where every pair within a
/// component has cosine similarity ≥ `threshold`.
///
/// Time: O(n² · d) where n = number of vectors, d = dimension.
/// Space: O(n) for the union-find structure.
pub fn cluster_by_similarity(vectors: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
    let n = vectors.len();
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if cosine_sim(&vectors[i], &vectors[j]) >= threshold {
                uf.union(i, j);
            }
        }
    }
    uf.components(n)
}

// ── Centroid ─────────────────────────────────────────────────────────────────

/// Arithmetic mean of a non-empty slice of vectors.
/// All vectors must have the same length.
pub fn centroid(vectors: &[&Vec<f32>]) -> Vec<f32> {
    assert!(!vectors.is_empty(), "cannot compute centroid of empty set");
    let dim = vectors[0].len();
    let mut c = vec![0.0f32; dim];
    for v in vectors {
        for (ci, &x) in c.iter_mut().zip(v.iter()) {
            *ci += x;
        }
    }
    let n = vectors.len() as f32;
    c.iter_mut().for_each(|x| *x /= n);
    c
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn union_find_two_components() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(2, 3);
        let comps = uf.components(4);
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn cluster_two_groups() {
        // Two tight clusters, far apart.
        let g1: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0, 0.01, 0.0]).collect();
        let g2: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0, 0.01, 1.0]).collect();
        let vecs: Vec<Vec<f32>> = g1.into_iter().chain(g2).collect();
        let clusters = cluster_by_similarity(&vecs, 0.9);
        assert_eq!(
            clusters.len(),
            2,
            "expected 2 clusters, got {}",
            clusters.len()
        );
    }

    #[test]
    fn centroid_of_two_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let c = centroid(&[&a, &b]);
        assert!((c[0] - 0.5).abs() < 1e-6);
        assert!((c[1] - 0.5).abs() < 1e-6);
    }
}
