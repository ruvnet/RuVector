//! ruvector-soar — Spilling Orthogonal Anti-correlated Refinement (SOAR) for IVF.
//!
//! Reference: Sun, Simhadri, Guo, Kumar, "SOAR: Improved Indexing for Approximate
//! Nearest Neighbor Search" (NeurIPS 2024). This crate provides a pure-Rust IVF
//! index with three pluggable assignment strategies — `Single`, `Spillover`, and
//! `Soar { lambda }` — so you can reproduce the paper's recall improvement on
//! synthetic and real workloads without unsafe code.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod kmeans;

pub use kmeans::{kmeans_pp_init, lloyd_refine};

use std::cmp::Ordering;

/// How database vectors are written into the inverted-file posting lists.
#[derive(Debug, Clone, Copy)]
pub enum Assignment {
    /// Each vector is assigned to its single nearest centroid (classic IVF).
    Single,
    /// Each vector is assigned to its top-2 nearest centroids (2x spillover).
    Spillover,
    /// SOAR — primary = nearest centroid; secondary minimizes
    /// `||x - c||^2 + lambda * ((x - c) . r_hat)^2`
    /// where `r_hat` is the unit residual after primary assignment.
    /// `lambda = 0` reduces to plain spillover; larger values prefer
    /// secondaries whose residual is orthogonal to the primary residual.
    Soar {
        /// Anti-correlation penalty. Paper recommends ~1.0–4.0; we default to 1.5.
        lambda: f32,
    },
}

impl Assignment {
    /// Number of centroids each vector is written to (replication factor).
    pub fn replication(&self) -> usize {
        match self {
            Assignment::Single => 1,
            Assignment::Spillover | Assignment::Soar { .. } => 2,
        }
    }
}

/// Errors produced while building or querying a SOAR/IVF index.
#[derive(Debug, thiserror::Error)]
pub enum SoarError {
    /// At least one input vector did not match the index dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// Expected dim
        expected: usize,
        /// Actual dim
        got: usize,
    },
    /// `n_centroids` was zero or larger than the dataset.
    #[error("invalid centroid count {n_centroids} for {n_vectors} vectors")]
    BadCentroidCount {
        /// Requested centroid count
        n_centroids: usize,
        /// Vector count
        n_vectors: usize,
    },
    /// The dataset was empty.
    #[error("empty dataset")]
    Empty,
}

/// IVF index over `f32` vectors with pluggable assignment.
#[derive(Debug, Clone)]
pub struct IvfIndex {
    dim: usize,
    centroids: Vec<Vec<f32>>,
    /// `posting_lists[c]` holds the ids of vectors assigned to centroid `c`.
    posting_lists: Vec<Vec<u32>>,
    vectors: Vec<Vec<f32>>,
    assignment: Assignment,
}

impl IvfIndex {
    /// Build an IVF index. Runs deterministic k-means (k-means++ init + Lloyd
    /// refinement) and writes posting lists according to `assignment`.
    pub fn build(
        vectors: Vec<Vec<f32>>,
        n_centroids: usize,
        assignment: Assignment,
        seed: u64,
    ) -> Result<Self, SoarError> {
        if vectors.is_empty() {
            return Err(SoarError::Empty);
        }
        if n_centroids == 0 || n_centroids > vectors.len() {
            return Err(SoarError::BadCentroidCount {
                n_centroids,
                n_vectors: vectors.len(),
            });
        }
        let dim = vectors[0].len();
        for v in &vectors {
            if v.len() != dim {
                return Err(SoarError::DimMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }

        let mut centroids = kmeans_pp_init(&vectors, n_centroids, seed);
        lloyd_refine(&vectors, &mut centroids, 12);

        let mut posting_lists = vec![Vec::<u32>::new(); n_centroids];
        for (vid, v) in vectors.iter().enumerate() {
            let assigned = assign_vector(v, &centroids, assignment);
            for c in assigned {
                posting_lists[c].push(vid as u32);
            }
        }

        Ok(Self {
            dim,
            centroids,
            posting_lists,
            vectors,
            assignment,
        })
    }

    /// Top-`k` vector ids and squared L2 distances using `n_probe` cells.
    /// Returned vector is sorted ascending by distance, deduplicated by id.
    pub fn search(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dim mismatch");

        // 1) probe nearest centroids
        let mut centroid_d: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, sq_l2(c, query)))
            .collect();
        centroid_d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let probes = centroid_d.iter().take(n_probe.min(self.centroids.len()));

        // 2) collect candidate ids (dedup — a vector may live in 2 cells)
        let mut seen = vec![false; self.vectors.len()];
        let mut hits: Vec<(u32, f32)> = Vec::new();
        for (cid, _) in probes {
            for &vid in &self.posting_lists[*cid] {
                let i = vid as usize;
                if seen[i] {
                    continue;
                }
                seen[i] = true;
                let d = sq_l2(&self.vectors[i], query);
                hits.push((vid, d));
            }
        }

        // 3) partial-sort to top-k
        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        hits.truncate(k);
        hits
    }

    /// Total number of (vector, centroid) entries across all posting lists.
    /// `Single` ≈ N, `Spillover`/`Soar` ≈ 2N.
    pub fn posting_entries(&self) -> usize {
        self.posting_lists.iter().map(|p| p.len()).sum()
    }

    /// Centroid count.
    pub fn n_centroids(&self) -> usize {
        self.centroids.len()
    }

    /// Dataset size.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Returns true iff the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Which assignment strategy this index was built with.
    pub fn assignment(&self) -> Assignment {
        self.assignment
    }

    /// Average secondary-vs-primary correlation (cosine of residual angle)
    /// across the dataset. Lower magnitude means more orthogonal coverage —
    /// the SOAR objective drives this toward 0.
    /// Returns `None` for `Single`.
    pub fn mean_residual_correlation(&self) -> Option<f32> {
        if matches!(self.assignment, Assignment::Single) {
            return None;
        }
        let mut sum = 0.0_f32;
        let mut n = 0usize;
        for (vid, v) in self.vectors.iter().enumerate() {
            let assigned = assign_vector(v, &self.centroids, self.assignment);
            if assigned.len() < 2 {
                continue;
            }
            let r1 = sub(v, &self.centroids[assigned[0]]);
            let r2 = sub(v, &self.centroids[assigned[1]]);
            let n1 = dot(&r1, &r1).sqrt();
            let n2 = dot(&r2, &r2).sqrt();
            if n1 > 1e-12 && n2 > 1e-12 {
                sum += dot(&r1, &r2) / (n1 * n2);
                n += 1;
                let _ = vid;
            }
        }
        if n == 0 {
            None
        } else {
            Some(sum / n as f32)
        }
    }
}

/// Pick centroid ids for a single vector under the given `assignment`.
fn assign_vector(v: &[f32], centroids: &[Vec<f32>], assignment: Assignment) -> Vec<usize> {
    // Ranked centroid distances (we always need at least the top-2)
    let mut d: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, sq_l2(c, v)))
        .collect();
    d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    match assignment {
        Assignment::Single => vec![d[0].0],
        Assignment::Spillover => {
            if centroids.len() == 1 {
                vec![d[0].0]
            } else {
                vec![d[0].0, d[1].0]
            }
        }
        Assignment::Soar { lambda } => {
            if centroids.len() == 1 {
                return vec![d[0].0];
            }
            let primary = d[0].0;
            let r = sub(v, &centroids[primary]);
            let r_norm = dot(&r, &r).sqrt();
            // Degenerate: vector exactly at centroid → fallback to spillover.
            if r_norm < 1e-12 {
                return vec![primary, d[1].0];
            }
            let r_hat: Vec<f32> = r.iter().map(|x| x / r_norm).collect();

            let mut best = (usize::MAX, f32::INFINITY);
            for (cid, base_sq) in d.iter().skip(1) {
                let err = sub(v, &centroids[*cid]);
                let par = dot(&err, &r_hat);
                let score = base_sq + lambda * par * par;
                if score < best.1 {
                    best = (*cid, score);
                }
            }
            vec![primary, best.0]
        }
    }
}

#[inline]
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s += d * d;
    }
    s
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

#[inline]
fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Brute-force top-`k` (squared L2). Used for ground truth.
pub fn brute_force_topk(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, sq_l2(v, query)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    all.truncate(k);
    all
}

/// Recall@k: fraction of `truth` ids present in `retrieved`.
pub fn recall(retrieved: &[(u32, f32)], truth: &[(u32, f32)]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let mut hits = 0usize;
    for (id, _) in truth {
        if retrieved.iter().any(|(rid, _)| rid == id) {
            hits += 1;
        }
    }
    hits as f32 / truth.len() as f32
}
