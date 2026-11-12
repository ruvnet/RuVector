//! RaBitQ two-stage query path for the runtime.
//!
//! Maintains an in-memory RaBitQ code book (1-bit sign codes + correction
//! scalars, ~32x smaller than the f32 vectors) over the store's vectors.
//! Queries that opt in via [`crate::options::QueryOptions::rabitq`] are
//! served in two stages:
//!
//! 1. **Candidate scan** — every live code is scored with the asymmetric
//!    RaBitQ estimator and the best `oversample * k` candidates are
//!    collected with deterministic `(distance, id)` tie-breaking.
//! 2. **Exact rescore** — candidates are re-ranked with full-precision
//!    f32 distances in the store's true metric, so the final top-k
//!    ordering is exact for the candidate set.
//!
//! Like the HNSW index, the state is built lazily on first use and kept
//! in sync as new vectors are ingested. All three metrics are supported:
//!
//! - **L2** — the RaBitQ squared-L2 estimator, unchanged.
//! - **Cosine** — vectors and queries are normalized to the unit sphere
//!   before encoding, where `||a - b||^2 = 2 - 2 cos(a, b)`, so the L2
//!   estimate orders candidates exactly as cosine distance would.
//! - **InnerProduct** — `<v, q> = <v-c, q-c> + <v, c> + <q, c> - <c, c>`
//!   for the quantizer centroid `c`. RaBitQ already estimates the
//!   centered term; `<v, c>` is kept per vector (one f32 next to each
//!   code, runtime-only, no wire change) and the other two are per-query
//!   constants, so the estimate has the same error as the L2 one.

use std::collections::HashMap;

use rvf_quant::rabitq::{RabitqCode, RabitqQuantizer, RabitqQuery, DEFAULT_ROUNDS};

use crate::options::DistanceMetric;
use crate::read_path::VectorData;

/// Fixed seed for the deterministic rotation, so codes are reproducible
/// across rebuilds of the same store contents.
const RABITQ_SEED: u64 = 0x5EED_4AB1_7B17_C0DE;

/// Floor applied to the stage-1 candidate pool. Measured on 10k random
/// 128-dim vectors (fixed seed, the hardest case for 1-bit codes since
/// the intrinsic dimension equals the full dimension): recall@10 is 0.49
/// at 40 candidates, 0.83 at 160, 0.92 at 320, 0.97 at 640, and 0.99 at
/// 1280. The floor keeps the two-stage path above the >= 0.95 recall@10
/// contract; larger `rabitq_oversample * k` pools are still honored.
/// Real-world embeddings (lower intrinsic dimension) need far fewer.
pub(crate) const RABITQ_MIN_CANDIDATES: usize = 640;

/// One encoded vector plus `<v, c>` (used only by the inner-product
/// estimator).
struct Entry {
    code: RabitqCode,
    dot_centroid: f32,
}

/// In-memory RaBitQ codes over the store's vectors.
pub(crate) struct RabitqState {
    quantizer: RabitqQuantizer,
    metric: DistanceMetric,
    codes: HashMap<u64, Entry>,
}

/// Unit-length copy of `v` (zero vectors stay zero).
fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Query state prepared once per query.
struct Prepared {
    query: RabitqQuery,
    /// `<q, c> - <c, c>` (inner product only).
    ip_offset: f32,
}

impl RabitqState {
    /// Train a quantizer (global mean centroid of the metric's encoding
    /// space) over all current vectors and encode every vector.
    pub(crate) fn build(vectors: &VectorData, metric: DistanceMetric) -> Option<Self> {
        if vectors.len() == 0 {
            return None;
        }
        let dim = vectors.dimension as usize;
        let mut sum = vec![0.0f64; dim];
        let mut count = 0usize;
        for &id in vectors.ids() {
            if let Some(v) = vectors.get(id) {
                let v = if metric == DistanceMetric::Cosine {
                    normalized(v)
                } else {
                    v.to_vec()
                };
                for (acc, &x) in sum.iter_mut().zip(&v) {
                    *acc += x as f64;
                }
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        let centroid: Vec<f32> = sum.iter().map(|&s| (s / count as f64) as f32).collect();
        let quantizer = RabitqQuantizer::with_centroid(dim, centroid, RABITQ_SEED, DEFAULT_ROUNDS);
        let mut state = Self {
            quantizer,
            metric,
            codes: HashMap::with_capacity(vectors.len()),
        };
        state.sync_missing(vectors);
        Some(state)
    }

    fn encode(&self, v: &[f32]) -> Entry {
        match self.metric {
            DistanceMetric::Cosine => Entry {
                code: self.quantizer.encode_code(&normalized(v)),
                dot_centroid: 0.0,
            },
            DistanceMetric::L2 => Entry {
                code: self.quantizer.encode_code(v),
                dot_centroid: 0.0,
            },
            DistanceMetric::InnerProduct => Entry {
                code: self.quantizer.encode_code(v),
                dot_centroid: dot(v, &self.quantizer.centroid),
            },
        }
    }

    fn prepare(&self, query: &[f32]) -> Prepared {
        match self.metric {
            DistanceMetric::Cosine => Prepared {
                query: self.quantizer.prepare_query(&normalized(query)),
                ip_offset: 0.0,
            },
            DistanceMetric::L2 => Prepared {
                query: self.quantizer.prepare_query(query),
                ip_offset: 0.0,
            },
            DistanceMetric::InnerProduct => {
                let c = &self.quantizer.centroid;
                Prepared {
                    query: self.quantizer.prepare_query(query),
                    ip_offset: dot(query, c) - dot(c, c),
                }
            }
        }
    }

    /// Stage-1 ranking key (ascending = better): estimated squared L2 for
    /// L2 and cosine (on the unit sphere), estimated negated inner product
    /// for inner product.
    fn estimate(&self, prepared: &Prepared, entry: &Entry) -> f32 {
        let l2 = self.quantizer.estimate_l2_sq(&prepared.query, &entry.code);
        match self.metric {
            DistanceMetric::L2 | DistanceMetric::Cosine => l2,
            DistanceMetric::InnerProduct => {
                let norm = entry.code.norm;
                let centered_ip = 0.5 * (norm * norm + prepared.query.norm_sq - l2);
                -(centered_ip + entry.dot_centroid + prepared.ip_offset)
            }
        }
    }

    /// Encode any store vectors that do not have a code yet (e.g. vectors
    /// ingested after the state was built). The centroid is intentionally
    /// not retrained: the estimator does not require an exact mean.
    pub(crate) fn sync_missing(&mut self, vectors: &VectorData) {
        if self.codes.len() >= vectors.len() {
            return;
        }
        for &id in vectors.ids() {
            if !self.codes.contains_key(&id) {
                if let Some(v) = vectors.get(id) {
                    let entry = self.encode(v);
                    self.codes.insert(id, entry);
                }
            }
        }
    }

    /// Returns true if `id` has an encoded code.
    pub(crate) fn contains(&self, id: u64) -> bool {
        self.codes.contains_key(&id)
    }

    /// Stage 1: scan all live codes with the asymmetric estimator and
    /// return the best `k_fetch` candidates as `(id, estimate)` sorted
    /// ascending by `(estimate, id)` (deterministic regardless of HashMap
    /// iteration order).
    pub(crate) fn candidates(
        &self,
        query: &[f32],
        k_fetch: usize,
        is_live: impl Fn(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if k_fetch == 0 {
            return Vec::new();
        }
        let prepared = self.prepare(query);

        // Bounded max-heap keyed by (estimate, id): the worst candidate
        // is on top and evicted when a better one arrives.
        let mut heap: std::collections::BinaryHeap<(OrderedF32, u64)> =
            std::collections::BinaryHeap::with_capacity(k_fetch + 1);
        for (&id, entry) in &self.codes {
            if !is_live(id) {
                continue;
            }
            let est = self.estimate(&prepared, entry);
            if heap.len() < k_fetch {
                heap.push((OrderedF32(est), id));
            } else if let Some(&(OrderedF32(worst), worst_id)) = heap.peek() {
                if est < worst || (est == worst && id < worst_id) {
                    heap.pop();
                    heap.push((OrderedF32(est), id));
                }
            }
        }

        let mut out: Vec<(u64, f32)> = heap
            .into_iter()
            .map(|(OrderedF32(d), id)| (id, d))
            .collect();
        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        out
    }
}

/// `f32` with a total ordering for use in heaps.
#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: u64, dim: usize, seed: u64) -> VectorData {
        let mut data = VectorData::new(dim as u16);
        let mut x = seed;
        for id in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
            }
            data.insert(id, v);
        }
        data
    }

    #[test]
    fn build_is_deterministic_and_syncs() {
        let mut vectors = make_vectors(100, 16, 7);
        let mut state = RabitqState::build(&vectors, DistanceMetric::L2).expect("non-empty store");
        assert_eq!(state.codes.len(), 100);

        // New vector picked up by sync.
        let extra = make_vectors(1, 16, 99);
        vectors.insert(500, extra.get(0).unwrap().to_vec());
        state.sync_missing(&vectors);
        assert!(state.contains(500));

        // Candidate order is deterministic across states built from the
        // same contents (same training set -> same centroid and codes).
        let state2 = RabitqState::build(&vectors, DistanceMetric::L2).unwrap();
        let state3 = RabitqState::build(&vectors, DistanceMetric::L2).unwrap();
        let q = extra.get(0).unwrap();
        let b = state2.candidates(q, 10, |_| true);
        let c = state3.candidates(q, 10, |_| true);
        assert_eq!(b, c);
        // The exact-match vector must surface among the candidates.
        let a = state.candidates(q, 10, |_| true);
        assert!(a.iter().any(|&(id, _)| id == 500));
        assert!(b.iter().any(|&(id, _)| id == 500));
    }

    #[test]
    fn empty_store_yields_no_state() {
        let vectors = VectorData::new(8);
        assert!(RabitqState::build(&vectors, DistanceMetric::L2).is_none());
    }

    #[test]
    fn deleted_ids_are_filtered() {
        let vectors = make_vectors(50, 8, 3);
        let state = RabitqState::build(&vectors, DistanceMetric::L2).unwrap();
        let q = vectors.get(10).unwrap().to_vec();
        let cands = state.candidates(&q, 50, |id| id != 10);
        assert!(cands.iter().all(|&(id, _)| id != 10));
        assert_eq!(cands.len(), 49);
    }

    /// Mean recall@10 of the two-stage path (stage-1 pool of
    /// `RABITQ_MIN_CANDIDATES`, exact rescore) against brute force.
    fn two_stage_recall(metric: DistanceMetric) -> f64 {
        use std::collections::HashSet;
        // Offset, varying-norm vectors so L2, cosine and IP disagree.
        let raw = make_vectors(5000, 64, 11);
        let mut vectors = VectorData::new(64);
        for &id in raw.ids() {
            let scale = 0.5 + (id % 7) as f32 * 0.25;
            let v: Vec<f32> = raw
                .get(id)
                .unwrap()
                .iter()
                .map(|x| (x + 0.2) * scale)
                .collect();
            vectors.insert(id, v);
        }
        let exact = |q: &[f32], v: &[f32]| match metric {
            DistanceMetric::L2 => q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum(),
            DistanceMetric::InnerProduct => -dot(q, v),
            DistanceMetric::Cosine => {
                1.0 - dot(q, v) / (dot(q, q).sqrt() * dot(v, v).sqrt()).max(f32::EPSILON)
            }
        };
        let state = RabitqState::build(&vectors, metric).unwrap();
        let queries = make_vectors(20, 64, 777);
        let mut total = 0.0;
        for qid in 0..20u64 {
            let q = queries.get(qid).unwrap();
            let rank = |ids: Vec<u64>| -> HashSet<u64> {
                let mut scored: Vec<(u64, f32)> = ids
                    .into_iter()
                    .map(|id| (id, exact(q, vectors.get(id).unwrap())))
                    .collect();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                scored.into_iter().take(10).map(|(id, _)| id).collect()
            };
            let truth = rank(vectors.ids().copied().collect());
            let cands = state.candidates(q, RABITQ_MIN_CANDIDATES, |_| true);
            let got = rank(cands.into_iter().map(|(id, _)| id).collect());
            total += truth.intersection(&got).count() as f64 / 10.0;
        }
        total / 20.0
    }

    #[test]
    fn two_stage_recall_holds_for_every_metric() {
        for metric in [
            DistanceMetric::L2,
            DistanceMetric::Cosine,
            DistanceMetric::InnerProduct,
        ] {
            let recall = two_stage_recall(metric);
            assert!(recall >= 0.95, "{metric:?} recall@10 {recall:.3}");
        }
    }

    #[test]
    fn cosine_codes_ignore_vector_scale() {
        // Under cosine, a vector and a scaled copy are the same point.
        let mut vectors = VectorData::new(8);
        let v = make_vectors(1, 8, 5).get(0).unwrap().to_vec();
        vectors.insert(1, v.clone());
        vectors.insert(2, v.iter().map(|x| x * 9.0).collect());
        vectors.insert(3, make_vectors(1, 8, 6).get(0).unwrap().to_vec());
        let state = RabitqState::build(&vectors, DistanceMetric::Cosine).unwrap();
        let (a, b) = (&state.codes[&1].code, &state.codes[&2].code);
        assert_eq!(a.bits, b.bits);
        assert!((a.norm - b.norm).abs() < 1e-5);
    }
}
