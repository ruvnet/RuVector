//! Property tests for the validator (the proptest twin of the cargo-fuzz
//! target): chunking never changes the verdict, arbitrary input never
//! panics, and valid stores round-trip their live vectors.

mod common;

use common::*;
use proptest::prelude::*;
use ruvector_edge_registry::validate::{
    validate, StreamValidator, ValidatedRvf, ValidationError, ValidationLimits,
};

fn chunked(
    bytes: &[u8],
    cuts: &[usize],
    limits: ValidationLimits,
) -> Result<ValidatedRvf, ValidationError> {
    let mut v = StreamValidator::new(limits);
    let mut points: Vec<usize> = cuts.iter().map(|c| c % (bytes.len() + 1)).collect();
    points.sort_unstable();
    let mut prev = 0;
    for p in points.into_iter().chain([bytes.len()]) {
        // Errors are sticky; keep pushing to exercise that path too.
        let _ = v.push(&bytes[prev..p]);
        prev = p;
    }
    v.finish()
}

/// `(object bytes, rows written, deleted ids)`.
type Store = (Vec<u8>, Vec<(u64, Vec<f32>)>, Vec<u64>);

fn store() -> impl Strategy<Value = Store> {
    (
        1u16..9,
        0u64..20,
        any::<u64>(),
        any::<bool>(),
        any::<bool>(),
        proptest::collection::vec(1u64..25, 0..4),
    )
        .prop_map(|(dim, n, seed, wire, root, deleted)| {
            let data = rows(n, dim, seed);
            let bytes = if wire {
                wire_store(dim, &data, &deleted, root)
            } else {
                packed_store(dim, &data, &deleted, (seed % 3) as u8)
            };
            (bytes, data, deleted)
        })
}

#[derive(Debug, Clone)]
enum Mutation {
    Flip(usize, u8),
    Truncate(usize),
    Insert(usize, Vec<u8>),
    Overwrite(usize, Vec<u8>),
}

fn mutation() -> impl Strategy<Value = Mutation> {
    prop_oneof![
        (any::<usize>(), 1u8..=255).prop_map(|(i, x)| Mutation::Flip(i, x)),
        any::<usize>().prop_map(Mutation::Truncate),
        (
            any::<usize>(),
            proptest::collection::vec(any::<u8>(), 1..80)
        )
            .prop_map(|(i, b)| Mutation::Insert(i, b)),
        (
            any::<usize>(),
            proptest::collection::vec(any::<u8>(), 1..16)
        )
            .prop_map(|(i, b)| Mutation::Overwrite(i, b)),
    ]
}

fn apply(mut b: Vec<u8>, m: &Mutation) -> Vec<u8> {
    if b.is_empty() {
        return b;
    }
    match m {
        Mutation::Flip(i, x) => {
            let n = b.len();
            b[i % n] ^= x;
        }
        Mutation::Truncate(i) => b.truncate(i % b.len()),
        Mutation::Insert(i, s) => {
            let at = i % (b.len() + 1);
            b.splice(at..at, s.iter().copied());
        }
        Mutation::Overwrite(i, s) => {
            let at = i % b.len();
            for (k, x) in s.iter().enumerate() {
                if let Some(slot) = b.get_mut(at + k) {
                    *slot = *x;
                }
            }
        }
    }
    b
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn valid_stores_round_trip((bytes, data, deleted) in store(), cuts in proptest::collection::vec(any::<usize>(), 0..8)) {
        let v = validate(&bytes, ValidationLimits::default()).unwrap();
        prop_assert_eq!(chunked(&bytes, &cuts, ValidationLimits::default()), Ok(v.clone()));
        let want: Vec<_> = data.into_iter().filter(|(id, _)| !deleted.contains(id)).collect();
        prop_assert_eq!(v.live_vectors(&bytes).unwrap(), want);
        prop_assert_eq!(v.total_size, bytes.len() as u64);
    }

    #[test]
    fn mutated_stores_never_panic_and_chunking_is_irrelevant(
        (bytes, _, _) in store(),
        muts in proptest::collection::vec(mutation(), 1..4),
        cuts in proptest::collection::vec(any::<usize>(), 0..8),
        tiny in any::<bool>(),
    ) {
        let bytes = muts.iter().fold(bytes, apply);
        let limits = if tiny {
            ValidationLimits { max_total_bytes: 700, max_segments: 3, max_manifest_payload: 64, ..Default::default() }
        } else {
            ValidationLimits::default()
        };
        let one = validate(&bytes, limits);
        prop_assert_eq!(chunked(&bytes, &cuts, limits), one.clone());
        if let Ok(v) = one {
            // Anything accepted is internally consistent.
            let _ = v.live_vectors(&bytes).unwrap();
            prop_assert!(v.segments.iter().all(|s| s.offset + s.length <= v.total_size));
        }
    }

    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..2048), cuts in proptest::collection::vec(any::<usize>(), 0..4)) {
        let one = validate(&bytes, ValidationLimits::default());
        prop_assert_eq!(chunked(&bytes, &cuts, ValidationLimits::default()), one);
    }
}
