//! Regressions for manifest-directory amplification and unchecked
//! `total_vectors`: stores whose content hashes are all correct (so the
//! directory logic is actually reached) but whose directory repeats,
//! reorders or over-counts segments.

mod common;

use common::*;
use proptest::prelude::*;
use ruvector_edge_registry::validate::{validate, ValidatedRvf, ValidationError, ValidationLimits};
use rvf_types::SegmentType;

/// `k` VEC_SEGs (packed runtime layout, correct legacy hashes), then one
/// MANIFEST_SEG whose directory lists the VEC_SEGs at `picks` (indices, in
/// the given order, repeats allowed) and declares `total`.
fn store_with_directory(counts: &[u64], picks: &[usize], total: u64) -> Vec<u8> {
    let dim = 2u16;
    let mut out = Vec::new();
    let mut heads = Vec::new();
    for (i, n) in counts.iter().enumerate() {
        let p = vec_payload(dim, &rows(*n, dim, i as u64));
        heads.push((i as u64 + 1, out.len() as u64, p.len() as u64));
        out.extend(Seg::new(SegmentType::Vec, i as u64 + 1).build(&p, &[]));
    }
    let dir: Vec<_> = picks
        .iter()
        .map(|&i| {
            let (id, off, len) = heads[i];
            (id, off, len, SegmentType::Vec as u8)
        })
        .collect();
    let man = manifest_payload(dim, total, 0, &dir, &[]);
    out.extend(Seg::new(SegmentType::Manifest, 99).build(&man, &[]));
    out
}

fn reason(r: Result<ValidatedRvf, ValidationError>) -> &'static str {
    match r {
        Err(ValidationError::MalformedManifest { reason, .. }) => reason,
        other => panic!("expected MalformedManifest, got {other:?}"),
    }
}

/// The reported proof of concept, scaled down: one VEC_SEG named by many
/// directory entries used to validate with `live_vec_segments.len()` equal
/// to the entry count (42 GiB of range reads for a 4 MB upload).
#[test]
fn duplicated_directory_entries_are_refused() {
    let bytes = store_with_directory(&[3], &[0; 64], 3);
    assert_eq!(
        reason(validate(&bytes, ValidationLimits::default())),
        "directory has more entries than segments"
    );
    let bytes = store_with_directory(&[3, 3], &[0, 0], 3);
    assert_eq!(
        reason(validate(&bytes, ValidationLimits::default())),
        "directory offsets are not strictly increasing"
    );
    let bytes = store_with_directory(&[3, 3], &[1, 0], 3);
    assert_eq!(
        reason(validate(&bytes, ValidationLimits::default())),
        "directory offsets are not strictly increasing"
    );
    let ok = validate(
        &store_with_directory(&[3, 3], &[0, 1], 6),
        ValidationLimits::default(),
    )
    .unwrap();
    assert_eq!(ok.live_vec_segments, vec![0, 1]);
}

#[test]
fn total_vectors_is_bounded_by_live_records() {
    let bytes = store_with_directory(&[3, 4], &[0, 1], 8);
    assert_eq!(
        reason(validate(&bytes, ValidationLimits::default())),
        "total_vectors exceeds the live vector records"
    );
    let bytes = store_with_directory(&[3, 4], &[1], 4);
    assert_eq!(
        validate(&bytes, ValidationLimits::default())
            .unwrap()
            .total_vectors,
        4
    );
    let bytes = store_with_directory(&[3], &[0], u64::MAX);
    assert!(validate(&bytes, ValidationLimits::default()).is_err());
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// Structure-aware: arbitrary directories over correctly hashed stores.
    /// Accepted iff the picks are strictly increasing and `total` fits; an
    /// accepted result never names a segment twice.
    #[test]
    fn directory_mutations_with_valid_hashes(
        counts in proptest::collection::vec(0u64..5, 1..6),
        raw_picks in proptest::collection::vec(any::<usize>(), 0..10),
        extra in 0u64..4,
        over in any::<bool>(),
    ) {
        let picks: Vec<usize> = raw_picks.iter().map(|p| p % counts.len()).collect();
        let cap: u64 = picks.iter().map(|&i| counts[i]).sum();
        let total = if over { cap + extra } else { cap.saturating_sub(extra) };
        let bytes = store_with_directory(&counts, &picks, total);
        let got = validate(&bytes, ValidationLimits::default());
        let increasing = picks.windows(2).all(|w| w[0] < w[1]);
        prop_assert_eq!(got.is_ok(), increasing && total <= cap, "{:?}", got);
        if let Ok(v) = got {
            prop_assert!(v.live_vec_segments.windows(2).all(|w| w[0] < w[1]));
            let live = v.live_vectors(&bytes).unwrap();
            prop_assert!(live.len() as u64 <= cap);
            prop_assert!(v.total_vectors <= cap);
        }
    }
}
