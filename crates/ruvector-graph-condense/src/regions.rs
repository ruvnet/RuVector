//! Region formation and per-region summarisation helpers.
//!
//! Split out from [`crate::condense`] to keep the orchestration small: this
//! module owns *how a region is detected and summarised* (weak-boundary
//! components, coverage, centroid/medoid, class histograms), while `condense`
//! owns the pipeline that wires them together.

use crate::error::Result;
use crate::features::NodeFeatures;
use ruvector_mincut::{DynamicGraph, VertexId};
use std::collections::HashMap;

/// Regions = connected components of the graph after removing edges lighter than
/// `relative_threshold * mean_weight`. Isolated / weak-only vertices fall out as
/// singletons. Deterministic for a fixed graph. Single edge pass + union-find,
/// so it scales near-linearly.
pub(crate) fn weak_boundary_regions(
    graph: &DynamicGraph,
    relative_threshold: f64,
) -> Vec<Vec<VertexId>> {
    let vertices = graph.vertices();
    let edges = graph.edges();

    // Index vertices contiguously for the union-find.
    let mut index: HashMap<VertexId, usize> = HashMap::with_capacity(vertices.len());
    for (i, &v) in vertices.iter().enumerate() {
        index.insert(v, i);
    }
    let mut uf = UnionFind::new(vertices.len());

    let threshold = if edges.is_empty() {
        0.0
    } else {
        let mean = edges.iter().map(|e| e.weight).sum::<f64>() / edges.len() as f64;
        relative_threshold * mean
    };

    for e in &edges {
        if e.weight >= threshold {
            uf.union(index[&e.source], index[&e.target]);
        }
    }

    // Group vertices by their union-find root.
    let mut groups: HashMap<usize, Vec<VertexId>> = HashMap::new();
    for (i, &v) in vertices.iter().enumerate() {
        groups.entry(uf.find(i)).or_default().push(v);
    }
    groups.into_values().collect()
}

/// Append singleton regions for any graph vertex not already covered by the
/// partitioner (some partitioners drop isolated or unsplittable vertices).
pub(crate) fn ensure_coverage(regions: &mut Vec<Vec<VertexId>>, vertices: &[VertexId]) {
    let mut seen: std::collections::HashSet<VertexId> =
        std::collections::HashSet::with_capacity(vertices.len());
    for r in regions.iter() {
        for &v in r {
            seen.insert(v);
        }
    }
    for &v in vertices {
        if seen.insert(v) {
            regions.push(vec![v]);
        }
    }
}

/// Mean embedding and medoid (member closest to the mean) of a region.
/// `members` must be non-empty.
pub(crate) fn centroid_and_medoid(
    members: &[VertexId],
    features: &NodeFeatures,
    dim: usize,
) -> Result<(Vec<f32>, VertexId)> {
    let mut centroid = vec![0f32; dim];
    for &v in members {
        let emb = features.require(v)?;
        for (c, &x) in centroid.iter_mut().zip(emb.iter()) {
            *c += x;
        }
    }
    let inv = 1.0 / members.len() as f32;
    for c in &mut centroid {
        *c *= inv;
    }

    let mut best = members[0];
    let mut best_dist = f32::INFINITY;
    for &v in members {
        let emb = features.require(v)?;
        let d = l2_sq(&centroid, emb);
        if d < best_dist {
            best_dist = d;
            best = v;
        }
    }
    Ok((centroid, best))
}

/// Normalised class histogram over `num_classes`, or empty when unsupervised.
pub(crate) fn class_distribution(
    members: &[VertexId],
    features: &NodeFeatures,
    num_classes: usize,
) -> Vec<f32> {
    if num_classes == 0 {
        return Vec::new();
    }
    let mut hist = vec![0f32; num_classes];
    let mut counted = 0f32;
    for &v in members {
        if let Some(label) = features.label(v) {
            if label < num_classes {
                hist[label] += 1.0;
                counted += 1.0;
            }
        }
    }
    if counted > 0.0 {
        let inv = 1.0 / counted;
        for h in &mut hist {
            *h *= inv;
        }
    }
    hist
}

/// Squared Euclidean distance.
pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// L2-normalise in place (no-op for a zero vector).
pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v {
            *x *= inv;
        }
    }
}

/// Minimal union-find with path compression and union by size.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_boundary_splits_on_light_edges() {
        let g = DynamicGraph::new();
        // Heavy clique {0,1,2}, heavy clique {3,4,5}, light bridge 2-3.
        for &(u, v, w) in &[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 3, 1.0),
            (2, 3, 0.05),
        ] {
            g.insert_edge(u, v, w).unwrap();
        }
        let mut regions = weak_boundary_regions(&g, 0.5);
        for r in &mut regions {
            r.sort_unstable();
        }
        regions.sort_by_key(|r| r[0]);
        assert_eq!(regions, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn ensure_coverage_adds_missing() {
        let mut regions = vec![vec![0u64, 1]];
        ensure_coverage(&mut regions, &[0, 1, 2, 3]);
        let singletons: usize = regions.iter().filter(|r| r.len() == 1).count();
        assert_eq!(singletons, 2); // 2 and 3 added
    }

    #[test]
    fn centroid_mean_and_medoid() {
        let mut f = NodeFeatures::new(1, 0);
        f.set_embedding(0, vec![0.0]).unwrap();
        f.set_embedding(1, vec![2.0]).unwrap();
        f.set_embedding(2, vec![4.0]).unwrap();
        let (centroid, medoid) = centroid_and_medoid(&[0, 1, 2], &f, 1).unwrap();
        assert!((centroid[0] - 2.0).abs() < 1e-6);
        assert_eq!(medoid, 1);
    }

    #[test]
    fn class_dist_normalises() {
        let mut f = NodeFeatures::new(1, 3);
        f.set(0, vec![0.0], 0).unwrap();
        f.set(1, vec![0.0], 0).unwrap();
        f.set(2, vec![0.0], 2).unwrap();
        let d = class_distribution(&[0, 1, 2], &f, 3);
        assert!((d[0] - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(d[1], 0.0);
        assert!((d[2] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_unit_length() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6);
    }
}
