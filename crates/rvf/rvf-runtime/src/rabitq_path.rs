//! RaBitQ two-stage query path for the runtime.
//!
//! Maintains an in-memory RaBitQ code book (1-bit sign codes + correction
//! scalars, ~32x smaller than the f32 vectors) over the store's vectors.
//! Queries that opt in via [`crate::options::QueryOptions::rabitq`] are
//! served in two stages:
//!
//! 1. **Candidate scan** — every live code is scored with the asymmetric
//!    RaBitQ L2 estimator and the best `oversample * k` candidates are
//!    collected with deterministic `(distance, id)` tie-breaking.
//! 2. **Exact rescore** — candidates are re-ranked with full-precision
//!    f32 distances, so the final top-k ordering is exact for the
//!    candidate set.
//!
//! Like the HNSW index, the state is built lazily on first use and kept
//! in sync as new vectors are ingested. v1 supports the L2 metric (the
//! estimator needs only the residual norm; inner-product and cosine
//! would require an extra correction scalar per vector).

use std::collections::HashMap;

use rvf_quant::rabitq::{RabitqCode, RabitqQuantizer};

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

/// In-memory RaBitQ codes over the store's vectors.
pub(crate) struct RabitqState {
    quantizer: RabitqQuantizer,
    codes: HashMap<u64, RabitqCode>,
}

impl RabitqState {
    /// Train a quantizer (global mean centroid) over all current vectors
    /// and encode every vector.
    pub(crate) fn build(vectors: &VectorData) -> Option<Self> {
        if vectors.len() == 0 {
            return None;
        }
        let refs: Vec<&[f32]> = vectors.ids().filter_map(|&id| vectors.get(id)).collect();
        let quantizer = RabitqQuantizer::train(&refs, RABITQ_SEED);
        let mut state = Self {
            quantizer,
            codes: HashMap::with_capacity(vectors.len()),
        };
        state.sync_missing(vectors);
        Some(state)
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
                    self.codes.insert(id, self.quantizer.encode_code(v));
                }
            }
        }
    }

    /// Returns true if `id` has an encoded code.
    pub(crate) fn contains(&self, id: u64) -> bool {
        self.codes.contains_key(&id)
    }

    /// Stage 1: scan all live codes with the asymmetric estimator and
    /// return the best `k_fetch` candidates as `(id, estimated_distance)`
    /// sorted ascending by `(distance, id)` (deterministic regardless of
    /// HashMap iteration order).
    pub(crate) fn candidates(
        &self,
        query: &[f32],
        k_fetch: usize,
        is_live: impl Fn(u64) -> bool,
    ) -> Vec<(u64, f32)> {
        if k_fetch == 0 {
            return Vec::new();
        }
        let prepared = self.quantizer.prepare_query(query);

        // Bounded max-heap keyed by (estimate, id): the worst candidate
        // is on top and evicted when a better one arrives.
        let mut heap: std::collections::BinaryHeap<(OrderedF32, u64)> =
            std::collections::BinaryHeap::with_capacity(k_fetch + 1);
        for (&id, code) in &self.codes {
            if !is_live(id) {
                continue;
            }
            let est = self.quantizer.estimate_l2_sq(&prepared, code);
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
        let mut state = RabitqState::build(&vectors).expect("non-empty store");
        assert_eq!(state.codes.len(), 100);

        // New vector picked up by sync.
        let extra = make_vectors(1, 16, 99);
        vectors.insert(500, extra.get(0).unwrap().to_vec());
        state.sync_missing(&vectors);
        assert!(state.contains(500));

        // Candidate order is deterministic across states built from the
        // same contents (same training set -> same centroid and codes).
        let state2 = RabitqState::build(&vectors).unwrap();
        let state3 = RabitqState::build(&vectors).unwrap();
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
        assert!(RabitqState::build(&vectors).is_none());
    }

    #[test]
    fn deleted_ids_are_filtered() {
        let vectors = make_vectors(50, 8, 3);
        let state = RabitqState::build(&vectors).unwrap();
        let q = vectors.get(10).unwrap().to_vec();
        let cands = state.candidates(&q, 50, |id| id != 10);
        assert!(cands.iter().all(|&(id, _)| id != 10));
        assert_eq!(cands.len(), 49);
    }
}
