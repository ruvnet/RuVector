//! Disagreement as a query primitive.
//!
//! A single-embedding store can only answer "what is *similar* to X". Because a
//! constellation keeps every lens as a distinct slot, it can answer a question
//! no flat vector DB can express: **"where do two lenses disagree?"** — which
//! records look alike through one lens but different through another (the "code
//! comment says one thing, the code does another" detector).
//!
//! Two flavours:
//! - **intrinsic** — a record whose lens-A neighbourhood and lens-B
//!   neighbourhood barely overlap sits on a boundary between the two views.
//! - **query-relative** — given a query, records that score high under lens A
//!   but low under lens B (or vice-versa).

use std::collections::BTreeMap;

use crate::constellation::ConstellationStore;
use crate::scoring::cosine_sim;

/// The `k` nearest neighbours of record `id` under `lens` (excluding itself).
pub fn neighbor_set(store: &ConstellationStore, lens: &str, id: u64, k: usize) -> Vec<u64> {
    let Some(rec) = store.get(id) else {
        return Vec::new();
    };
    let Some(slot) = rec.slot(lens) else {
        return Vec::new();
    };
    store
        .search_lens(lens, slot, k + 1)
        .into_iter()
        .filter(|&(nid, _)| nid != id)
        .take(k)
        .map(|(nid, _)| nid)
        .collect()
}

/// Jaccard similarity of two id lists.
fn jaccard(a: &[u64], b: &[u64]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let sa: std::collections::BTreeSet<u64> = a.iter().copied().collect();
    let sb: std::collections::BTreeSet<u64> = b.iter().copied().collect();
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    if union == 0 {
        1.0
    } else {
        inter as f32 / union as f32
    }
}

/// Intrinsic cross-lens disagreement of record `id`: `1 − Jaccard` of its
/// lens-A and lens-B neighbourhoods. `1.0` = the two lenses place the record in
/// entirely different company.
pub fn intrinsic_disagreement(
    store: &ConstellationStore,
    lens_a: &str,
    lens_b: &str,
    id: u64,
    k: usize,
) -> f32 {
    let na = neighbor_set(store, lens_a, id, k);
    let nb = neighbor_set(store, lens_b, id, k);
    1.0 - jaccard(&na, &nb)
}

/// Find the records where `lens_a` and `lens_b` most disagree, best-first.
pub fn find_conflicts(
    store: &ConstellationStore,
    lens_a: &str,
    lens_b: &str,
    k: usize,
    top_n: usize,
) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = store
        .records()
        .iter()
        .map(|r| (r.id, intrinsic_disagreement(store, lens_a, lens_b, r.id, k)))
        .collect();
    scored.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.0.cmp(&y.0))
    });
    scored.truncate(top_n);
    scored
}

/// Query-relative disagreement: records scored by `sim_A(q,·) − sim_B(q,·)`.
/// Large positive = "looks right through lens A, wrong through lens B".
/// Returns `(id, signed_gap)` sorted by descending absolute gap.
pub fn query_disagreement(
    store: &ConstellationStore,
    query: &BTreeMap<String, Vec<f32>>,
    lens_a: &str,
    lens_b: &str,
    top_n: usize,
) -> Vec<(u64, f32)> {
    let (Some(qa), Some(qb)) = (query.get(lens_a), query.get(lens_b)) else {
        return Vec::new();
    };
    let mut scored: Vec<(u64, f32)> = store
        .records()
        .iter()
        .filter_map(|r| {
            let (sa, sb) = (r.slot(lens_a)?, r.slot(lens_b)?);
            Some((r.id, cosine_sim(qa, sa) - cosine_sim(qb, sb)))
        })
        .collect();
    scored.sort_by(|x, y| {
        y.1.abs()
            .partial_cmp(&x.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.0.cmp(&y.0))
    });
    scored.truncate(top_n);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constellation::Constellation;
    use crate::lens::{LensKind, LensManifest};

    fn store() -> ConstellationStore {
        let lenses = vec![
            LensManifest::new("a", "v1", LensKind::DenseSemantic, 2, 1.0, 1.0),
            LensManifest::new("b", "v1", LensKind::Structural, 2, 1.0, 1.0),
        ];
        let mut s = ConstellationStore::new(lenses);
        // Records 1,2,3 agree on both lenses (cluster X). Record 4 is the
        // conflict: lens-a like cluster X, lens-b like cluster Y (record 5,6).
        s.insert(Constellation::new(1).with_slot("a", vec![1.0, 0.0]).with_slot("b", vec![1.0, 0.0])).unwrap();
        s.insert(Constellation::new(2).with_slot("a", vec![0.98, 0.02]).with_slot("b", vec![0.98, 0.02])).unwrap();
        s.insert(Constellation::new(3).with_slot("a", vec![0.96, 0.04]).with_slot("b", vec![0.97, 0.03])).unwrap();
        s.insert(Constellation::new(5).with_slot("a", vec![0.0, 1.0]).with_slot("b", vec![0.0, 1.0])).unwrap();
        s.insert(Constellation::new(6).with_slot("a", vec![0.02, 0.98]).with_slot("b", vec![0.03, 0.97])).unwrap();
        // The conflict record: lens-a near cluster X, lens-b near cluster Y.
        s.insert(Constellation::new(4).with_slot("a", vec![0.97, 0.03]).with_slot("b", vec![0.03, 0.97])).unwrap();
        s
    }

    #[test]
    fn conflict_record_ranks_first() {
        let s = store();
        let conflicts = find_conflicts(&s, "a", "b", 2, 3);
        assert_eq!(conflicts[0].0, 4, "the planted conflict should rank first");
        assert!(conflicts[0].1 > 0.5);
    }
}
