//! Lightweight community detection via threshold-based graph connectivity.
//!
//! Two-step process:
//! 1. Build a sparse similarity graph: add an edge between two nodes when
//!    their cosine similarity exceeds `edge_threshold`.
//! 2. Run Union-Find to collect connected components → communities.
//!
//! This approximates the partition produced by a global mincut with the
//! same threshold: communities are the strongly-connected subgraphs separated
//! by cuts weaker than (1 - edge_threshold) in cosine distance.
//!
//! The approach is O(N²) to build (same as k-NN graph), O(N·α(N)) to label,
//! where α is the inverse Ackermann function. Both are fast in practice for
//! the PoC scale of N ≤ 50k.

/// Union-Find with path compression and union-by-rank.
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

    pub fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb { return; }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

/// Detected community structure for a vector corpus.
pub struct Communities {
    /// For each vector id, its community label (0-indexed, contiguous).
    pub labels: Vec<usize>,
    /// Number of distinct communities found.
    pub num_communities: usize,
    /// Centroid of each community (mean of member vectors).
    pub centroids: Vec<Vec<f32>>,
}

impl Communities {
    /// Detect communities from raw vectors using cosine-similarity threshold.
    ///
    /// Two vectors are in the same community when `cosine_sim > edge_threshold`.
    /// A higher threshold → smaller, more coherent communities.
    /// A lower threshold → larger, loosely connected communities.
    pub fn detect(vectors: &[Vec<f32>], edge_threshold: f32) -> Self {
        let n = vectors.len();
        let dims = vectors.first().map(|v| v.len()).unwrap_or(0);
        let mut uf = UnionFind::new(n);

        // Build edges: O(N²) — acceptable for PoC sizes.
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_sim(&vectors[i], &vectors[j]);
                if sim > edge_threshold {
                    uf.union(i, j);
                }
            }
        }

        // Collect labels and renumber contiguously.
        let roots: Vec<usize> = (0..n).map(|i| uf.find(i)).collect();
        let mut root_to_label = std::collections::HashMap::new();
        let mut next_label = 0usize;
        let labels: Vec<usize> = roots
            .iter()
            .map(|&r| {
                *root_to_label.entry(r).or_insert_with(|| {
                    let l = next_label;
                    next_label += 1;
                    l
                })
            })
            .collect();
        let num_communities = next_label;

        // Compute centroids.
        let mut sums = vec![vec![0.0f32; dims]; num_communities];
        let mut counts = vec![0usize; num_communities];
        for (i, &lab) in labels.iter().enumerate() {
            for d in 0..dims {
                sums[lab][d] += vectors[i][d];
            }
            counts[lab] += 1;
        }
        let centroids: Vec<Vec<f32>> = sums
            .into_iter()
            .zip(counts.iter())
            .map(|(s, &c)| {
                let c = c.max(1) as f32;
                s.into_iter().map(|x| x / c).collect()
            })
            .collect();

        Communities { labels, num_communities, centroids }
    }

    /// Find the nearest community centroid for a query vector.
    /// Returns (community_label, distance_to_centroid).
    pub fn nearest_community(&self, query: &[f32]) -> (usize, f32) {
        self.centroids
            .iter()
            .enumerate()
            .map(|(c, cent)| {
                let d: f32 = query.iter().zip(cent.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                (c, d)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, f32::INFINITY))
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}
