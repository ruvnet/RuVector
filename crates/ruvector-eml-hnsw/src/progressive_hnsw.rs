//! Progressive multi-level HNSW: approximates per-layer distance
//! dimensionality by building one HNSW per dim budget in a schedule and
//! searching all of them coarsest → finest, then reranking with exact
//! full-dim cosine.
//!
//! `hnsw_rs` has no public hook for per-layer distance functions
//! (`search_layer` is private and consumes a single `DistCosine`). We
//! approximate the per-layer-dimensionality idea of [`ProgressiveDistance`]
//! at a different granularity: instead of one HNSW with different dims per
//! layer, we build an ordered stack of HNSWs — each indexing the whole
//! corpus but projected down to a different dimensionality (e.g. 8, 32, 128).
//!
//! Search cascade:
//! 1. Search the coarsest HNSW (e.g. 8-dim) — cheap distance, fast traversal.
//!    Pull `fetch_coarse = max(2 * k, ef_search)` candidates.
//! 2. Search each finer HNSW independently for the same or fewer candidates.
//!    These act as a refinement at a better fidelity, reducing the
//!    approximation error of any single level.
//! 3. Take the UNION of ids surfaced across all levels. Rerank with exact
//!    full-dim cosine and return the top-k.
//!
//! The task description suggested feeding coarse candidates as a "seed set"
//! into the next HNSW. `hnsw_rs::search_filter` can restrict a search to
//! allowed ids, but on a large index with a tiny allow-list (dozens of ids
//! out of tens of thousands) the HNSW traversal collapses into near-random
//! walk and both recall and latency regress hard — we measured 115 ms/query
//! at n=50k before switching to union+rerank (4–10× slower than a single
//! full HNSW). The union+rerank path gives the exact-rerank guarantee the
//! task asks for at the end of the cascade, and each per-level HNSW search
//! is a normal, well-tuned HNSW search.
//!
//! Build time scales roughly with the schedule length: three levels ≈ 3× a
//! single EmlHnsw build. The SIFT1M 50k A/B harness reports the exact ratio.

use crate::cosine_decomp::{cosine_distance_f32, EmlDistanceModel};
use crate::hnsw_integration::{EmlMetric, EmlSearchResult};
use crate::progressive_distance::ProgressiveDistance;
use crate::selected_distance::project_vector;
use hnsw_rs::prelude::{DistCosine, Hnsw};

/// One cascade level: the trained dim selection and an HNSW indexing the
/// projected corpus.
struct Level {
    selected_dims: Vec<usize>,
    hnsw: Hnsw<'static, f32, DistCosine>,
}

/// Multi-index HNSW cascade that approximates [`ProgressiveDistance`] using
/// one HNSW per dim budget plus exact full-dim rerank on the union of hits.
pub struct ProgressiveEmlHnsw {
    /// Ordered coarsest → finest.
    levels: Vec<Level>,
    /// The progressive-distance schedule, retained for metadata.
    schedule: ProgressiveDistance,
    /// Full-dim store indexed by (global id - 1).
    full_store: Vec<Vec<f32>>,
    metric: EmlMetric,
}

impl ProgressiveEmlHnsw {
    /// Train per-level selectors on `samples` and build one HNSW per
    /// schedule entry.
    ///
    /// `schedule` is coarsest → finest, e.g. `[8, 32, 128]`. Each value is
    /// clamped to full-dim and at least 1.
    ///
    /// `max_elements`, `m`, `ef_construction` are applied uniformly to every
    /// sub-HNSW.
    pub fn train_and_build(
        samples: &[Vec<f32>],
        schedule: &[usize],
        metric: EmlMetric,
        max_elements: usize,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self, &'static str> {
        if samples.len() < 100 {
            return Err("need at least 100 samples to train selectors");
        }
        if schedule.is_empty() {
            return Err("schedule must have at least one entry");
        }
        let full_dim = samples[0].len();

        let mut levels = Vec::with_capacity(schedule.len());
        for &dims in schedule {
            let dims = dims.min(full_dim).max(1);
            let mut selector = EmlDistanceModel::new(full_dim, dims);
            // Mirror EmlHnsw::train_and_build's pairing scheme so a single
            // schedule entry at full-dim reproduces the baseline selector.
            for chunk in samples.chunks(2) {
                if chunk.len() < 2 {
                    break;
                }
                let a = &chunk[0];
                let b = &chunk[1];
                let d = cosine_distance_f32(a, b);
                selector.record(a, b, d);
            }
            let needed = 100usize.saturating_sub(selector.sample_count());
            if needed > 0 {
                let stride = (samples.len() / (needed + 2)).max(1);
                let mut recorded = 0;
                let mut i = 0;
                while recorded < needed && i + stride < samples.len() {
                    let a = &samples[i];
                    let b = &samples[i + stride];
                    let d = cosine_distance_f32(a, b);
                    selector.record(a, b, d);
                    recorded += 1;
                    i += 1;
                }
            }
            let _ = selector.train();
            if !selector.is_trained() || selector.selected_dims().is_empty() {
                return Err("per-level selector failed to train");
            }
            let hnsw = Hnsw::<f32, DistCosine>::new(
                m,
                max_elements,
                16,
                ef_construction,
                DistCosine,
            );
            levels.push(Level {
                selected_dims: selector.selected_dims().to_vec(),
                hnsw,
            });
        }

        let schedule_model = ProgressiveDistance::with_schedule(full_dim, schedule);

        Ok(Self {
            levels,
            schedule: schedule_model,
            full_store: Vec::with_capacity(max_elements),
            metric,
        })
    }

    /// Number of cascade levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Per-level dim schedule (coarsest → finest).
    pub fn schedule(&self) -> &[usize] {
        self.schedule.dim_schedule()
    }

    /// Selected dim indices at a given level.
    pub fn level_dims(&self, level: usize) -> Option<&[usize]> {
        self.levels.get(level).map(|l| l.selected_dims.as_slice())
    }

    /// Insert one full-dim vector. Projects into every level and inserts
    /// into every sub-HNSW under the same global id. Returns the 1-based id.
    pub fn add(&mut self, full: &[f32]) -> usize {
        let id = self.full_store.len() + 1;
        self.full_store.push(full.to_vec());
        for level in &self.levels {
            let proj = project_vector(full, &level.selected_dims);
            level.hnsw.insert((&proj, id));
        }
        id
    }

    /// Bulk insert. Returns assigned ids in insertion order.
    pub fn add_batch(&mut self, fulls: &[Vec<f32>]) -> Vec<usize> {
        let mut ids = Vec::with_capacity(fulls.len());
        for v in fulls {
            ids.push(self.add(v));
        }
        ids
    }

    /// Number of vectors indexed.
    pub fn len(&self) -> usize {
        self.full_store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.full_store.is_empty()
    }

    /// Cascading search coarsest → finest, then exact full-dim rerank on the
    /// union of all levels' candidates.
    ///
    /// - Coarsest level pulls `max(2 * k, ef_search)` ids — cheap distance.
    /// - Finer levels each pull `max(k, ef_search / 2)` ids as refinement.
    /// - We union, exact-cosine rerank, and return the top-k.
    pub fn search(
        &self,
        query_full: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Vec<EmlSearchResult> {
        if self.levels.is_empty() || self.full_store.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut union_ids: Vec<usize> =
            Vec::with_capacity(self.levels.len() * (2 * k).max(ef_search));

        for (i, level) in self.levels.iter().enumerate() {
            let qi = project_vector(query_full, &level.selected_dims);
            // Coarsest level fetches widest so the rerank has headroom;
            // finer levels just need to re-confirm / correct coarse picks.
            let fetch = if i == 0 {
                (2 * k).max(ef_search)
            } else {
                k.max(ef_search / 2)
            };
            let hits = level.hnsw.search(&qi, fetch, ef_search);
            for h in hits {
                union_ids.push(h.d_id);
            }
        }

        // Dedupe the union before paying for full-dim distance.
        union_ids.sort_unstable();
        union_ids.dedup();
        if union_ids.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<EmlSearchResult> = union_ids
            .into_iter()
            .map(|id| {
                let stored = &self.full_store[id - 1];
                let dist = match self.metric {
                    EmlMetric::Cosine => cosine_distance_f32(query_full, stored),
                };
                EmlSearchResult { id, distance: dist }
            })
            .collect();

        scored.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skewed(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for d in 0..dim {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f32 / u32::MAX as f32) - 0.5;
                let scale = if d < 32 { 4.0 } else { 0.3 };
                v.push(u * scale);
            }
            out.push(v);
        }
        out
    }

    #[test]
    fn build_and_search_cascade_returns_nearest() {
        let data = make_skewed(400, 128, 17);
        let mut idx = ProgressiveEmlHnsw::train_and_build(
            &data[..200],
            &[8, 32, 128],
            EmlMetric::Cosine,
            1024,
            16,
            100,
        )
        .expect("build");
        idx.add_batch(&data);
        let q = &data[42];
        let hits = idx.search(q, 5, 64);
        assert!(!hits.is_empty());
        // Self must be in top-k with ~0 distance.
        let self_pos = hits.iter().position(|r| r.id == 43);
        assert!(self_pos.is_some(), "self not in top-5: {:?}", hits);
        assert!(hits[self_pos.unwrap()].distance < 1e-4);
    }

    #[test]
    fn schedule_accessors_reflect_config() {
        let data = make_skewed(300, 64, 11);
        let idx = ProgressiveEmlHnsw::train_and_build(
            &data,
            &[4, 16, 64],
            EmlMetric::Cosine,
            512,
            12,
            64,
        )
        .expect("build");
        assert_eq!(idx.num_levels(), 3);
        assert_eq!(idx.schedule(), &[4, 16, 64]);
        assert_eq!(idx.level_dims(0).unwrap().len(), 4);
        assert_eq!(idx.level_dims(1).unwrap().len(), 16);
        assert_eq!(idx.level_dims(2).unwrap().len(), 64);
    }

    #[test]
    fn two_level_schedule_works() {
        let data = make_skewed(300, 64, 77);
        let mut idx = ProgressiveEmlHnsw::train_and_build(
            &data[..150],
            &[8, 64],
            EmlMetric::Cosine,
            512,
            12,
            64,
        )
        .expect("build");
        idx.add_batch(&data);
        let q = &data[7];
        let hits = idx.search(q, 3, 32);
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, 8);
    }
}
