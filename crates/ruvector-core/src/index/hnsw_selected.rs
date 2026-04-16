//! Selected-dimension HNSW backend — learned dimension subset for fast
//! approximate search with exact re-rank.
//!
//! This module is a deliberate, self-contained port of the selected-dims
//! machinery from `ruvector-eml-hnsw`. It intentionally does **not** depend
//! on `eml-core` or `ruvector-eml-hnsw`, so `ruvector-core` keeps its
//! lightweight dependency graph and there is no inverted crate dependency.
//!
//! The runtime hot path is just:
//!   1. project query onto `k` selected dims  (1 alloc, O(k))
//!   2. search a reduced-dim HNSW             (graph traversal at k dims)
//!   3. optionally re-rank candidates with exact full-dim cosine
//!
//! The selector is trained **once**, offline, from a representative sample
//! of full-dim vectors. It discovers which dimensions discriminate best on
//! the caller's data distribution via per-dimension Pearson correlation
//! against exact full-dim cosine distance. This mirrors
//! [`ruvector-eml-hnsw::EmlDistanceModel::train`] but without the EML tree —
//! at query time we only need the dimension indices, not a learned
//! regression over them.
//!
//! # Evidence and acceptance bar
//!
//! See ADR-151 for SIFT1M measurements. Empirically validated operating
//! points:
//!   - `selected_k ∈ [32, 48]`
//!   - `fetch_k ≥ 500`
//!   - recall\@10 with rerank ≥ 0.80 on SIFT1M
//!
//! # Scope
//!
//! `ruvector-eml-hnsw` retains the richer research API (EML tree,
//! progressive cascade, PQ integration, retention-objective selector). This
//! module ships the stable single-stage "project + rerank" pipeline only.

use crate::distance::cosine_distance;
use crate::error::{Result, RuvectorError};
use crate::types::{DistanceMetric, HnswConfig};
use hnsw_rs::prelude::*;

/// Learned subset of dimensions for projected cosine distance.
///
/// Trained offline from a representative sample. At runtime we only use
/// `selected_dims` — no per-query model evaluation.
#[derive(Debug, Clone)]
pub(super) struct SelectedDimsSelector {
    #[allow(dead_code)] // exposed via full_dim() for future introspection
    full_dim: usize,
    selected_dims: Vec<usize>,
}

impl SelectedDimsSelector {
    /// Train a selector by Pearson-correlating `|a[d] - b[d]|` against exact
    /// cosine distance, then taking the top-`selected_k` dimensions.
    ///
    /// Follows the `EmlDistanceModel::train` algorithm in `ruvector-eml-hnsw`
    /// but stores only the dimension indices — no EML tree evaluation is
    /// needed at query time.
    pub fn train(sample: &[Vec<f32>], selected_k: usize) -> Result<Self> {
        if sample.is_empty() {
            return Err(RuvectorError::InvalidInput(
                "selector training requires a non-empty sample".into(),
            ));
        }
        let full_dim = sample[0].len();
        if full_dim == 0 {
            return Err(RuvectorError::InvalidInput(
                "selector training sample has zero-dim vectors".into(),
            ));
        }
        for v in sample {
            if v.len() != full_dim {
                return Err(RuvectorError::DimensionMismatch {
                    expected: full_dim,
                    actual: v.len(),
                });
            }
        }
        let k = selected_k.min(full_dim).max(1);

        // Synthesize (a, b, exact_cosine) training pairs from adjacent sample
        // vectors. This mirrors `EmlHnsw::train_and_build`.
        let mut pairs: Vec<(Vec<f32>, Vec<f32>, f32)> = Vec::new();
        for chunk in sample.chunks(2) {
            if chunk.len() < 2 {
                break;
            }
            let d = cosine_distance(&chunk[0], &chunk[1]);
            pairs.push((chunk[0].clone(), chunk[1].clone(), d));
        }
        // Need at least a handful of pairs for meaningful correlation; top up
        // with strided cross-pairs if pass one was too thin.
        if pairs.len() < 50 && sample.len() > 4 {
            let stride = (sample.len() / 8).max(1);
            let mut i = 0;
            while pairs.len() < 50 && i + stride < sample.len() {
                let d = cosine_distance(&sample[i], &sample[i + stride]);
                pairs.push((sample[i].clone(), sample[i + stride].clone(), d));
                i += 1;
            }
        }
        if pairs.is_empty() {
            return Err(RuvectorError::InvalidInput(
                "selector training sample produced zero pairs".into(),
            ));
        }

        // Per-dimension Pearson correlation between |a[d] - b[d]| and exact
        // distance. Higher |correlation| → more discriminative dim.
        let n = pairs.len() as f64;
        let mut dim_corr: Vec<(usize, f64)> = Vec::with_capacity(full_dim);
        for d in 0..full_dim {
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sxx = 0.0f64;
            let mut syy = 0.0f64;
            let mut sxy = 0.0f64;
            for (a, b, dist) in &pairs {
                let x = (a[d] - b[d]).abs() as f64;
                let y = *dist as f64;
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }
            let num = n * sxy - sx * sy;
            let dx = (n * sxx - sx * sx).max(1e-12);
            let dy = (n * syy - sy * sy).max(1e-12);
            let r = num / (dx.sqrt() * dy.sqrt());
            dim_corr.push((d, r.abs()));
        }

        dim_corr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<usize> =
            dim_corr.into_iter().take(k).map(|(d, _)| d).collect();
        selected.sort(); // cache-friendly access

        Ok(Self {
            full_dim,
            selected_dims: selected,
        })
    }

    #[allow(dead_code)]
    #[inline]
    pub fn full_dim(&self) -> usize {
        self.full_dim
    }

    #[inline]
    pub fn selected_dims(&self) -> &[usize] {
        &self.selected_dims
    }

    #[inline]
    pub fn reduced_dim(&self) -> usize {
        self.selected_dims.len()
    }
}

/// Project a full-dim vector onto the selector's chosen dimensions.
#[inline]
pub(super) fn project_vector(full: &[f32], dims: &[usize]) -> Vec<f32> {
    dims.iter().map(|&i| full[i]).collect()
}

/// Selected-dims HNSW backend. Owned by `HnswInner::Backend::SelectedDims`.
///
/// Holds:
///   - the trained selector (dim indices + cached reduced_dim)
///   - a reduced-dim HNSW using `hnsw_rs::DistCosine` on the projection
///   - the full-dim vectors (for exact rerank + API symmetry)
///   - our own running `next_idx`, mirroring the standard backend
pub(super) struct SelectedDimsBackend {
    pub selector: SelectedDimsSelector,
    pub reduced_hnsw: Hnsw<'static, f32, DistCosine>,
    pub full_store: Vec<Vec<f32>>,
}

impl SelectedDimsBackend {
    /// Construct the backend from a trained selector and HNSW config.
    pub fn new(selector: SelectedDimsSelector, config: &HnswConfig) -> Self {
        let reduced_dim = selector.reduced_dim();
        let reduced_hnsw = Hnsw::<f32, DistCosine>::new(
            config.m,
            config.max_elements,
            reduced_dim.max(1),
            config.ef_construction,
            DistCosine,
        );
        Self {
            selector,
            reduced_hnsw,
            full_store: Vec::new(),
        }
    }

    /// Insert one vector. Returns the internal 0-based index assigned.
    pub fn insert(&mut self, vector: &[f32]) -> usize {
        let idx = self.full_store.len();
        let reduced = project_vector(vector, self.selector.selected_dims());
        self.reduced_hnsw.insert((&reduced, idx));
        self.full_store.push(vector.to_vec());
        idx
    }

    /// Search the reduced-dim HNSW. Returns (internal_idx, distance) pairs;
    /// distance is cosine over the projection.
    pub fn search_reduced(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Vec<(usize, f32)> {
        let reduced = project_vector(query, self.selector.selected_dims());
        let neighbors = self.reduced_hnsw.search(&reduced, k, ef_search);
        neighbors
            .into_iter()
            .map(|n| (n.d_id, n.distance))
            .collect()
    }

    /// Approx-then-exact search: pull `fetch_k` candidates, rerank with
    /// exact full-dim distance, return top-`k`.
    pub fn search_with_rerank(
        &self,
        query: &[f32],
        k: usize,
        fetch_k: usize,
        ef_search: usize,
        metric: DistanceMetric,
    ) -> Vec<(usize, f32)> {
        let fetch = fetch_k.max(k);
        let mut cands = self.search_reduced(query, fetch, ef_search);
        for c in cands.iter_mut() {
            let stored = &self.full_store[c.0];
            c.1 = crate::distance::distance(query, stored, metric).unwrap_or(f32::MAX);
        }
        cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(k);
        cands
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.full_store.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_vector_basic() {
        let v = vec![10.0f32, 20.0, 30.0, 40.0];
        let p = project_vector(&v, &[0, 3]);
        assert_eq!(p, vec![10.0, 40.0]);
    }

    #[test]
    fn selector_trains_on_skewed_data() {
        // Variance concentrated in dims 0..4; selector should pick from there.
        let dim = 16;
        let mut seed = 17u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / u32::MAX as f32 - 0.5
        };
        let sample: Vec<Vec<f32>> = (0..200)
            .map(|_| {
                (0..dim)
                    .map(|d| if d < 4 { next() * 4.0 } else { next() * 0.1 })
                    .collect()
            })
            .collect();
        let sel = SelectedDimsSelector::train(&sample, 4).expect("train");
        assert_eq!(sel.reduced_dim(), 4);
        assert_eq!(sel.full_dim(), dim);
        // Most picks should land in the high-variance band.
        let in_band = sel
            .selected_dims()
            .iter()
            .filter(|&&d| d < 4)
            .count();
        assert!(
            in_band >= 3,
            "selector should prefer high-variance dims, got {:?}",
            sel.selected_dims()
        );
    }
}
