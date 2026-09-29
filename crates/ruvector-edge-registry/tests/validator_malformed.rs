//! Malformed, truncated, oversized and segment-bomb inputs are refused with
//! typed errors, without allocating from declared lengths.

mod common;

use common::*;
use ruvector_edge_registry::validate::{
    validate, StreamValidator, ValidationError as E, ValidationLimits,
};
use rvf_types::SegmentType;

fn lim() -> ValidationLimits {
    ValidationLimits::default()
}

fn err(bytes: &[u8]) -> E {
    validate(bytes, lim()).unwrap_err()
}

fn store_with_extra_first(seg: Vec<u8>) -> Vec<u8> {
    // `seg` followed by a valid packed store shifted by its length.
    let data = rows(2, 3, 1);
    let vp = vec_payload(3, &data);
    let base = seg.len() as u64;
    let mut out = seg;
    out.extend(Seg::new(SegmentType::Vec, 1).build(&vp, &[]));
    let man = manifest_payload(3, 2, 0, &[(1, base, vp.len() as u64, 1)], &[]);
    out.extend(Seg::new(SegmentType::Manifest, 2).build(&man, &[]));
    out
}

#[test]
fn baseline_stores_validate() {
    validate(&packed_store(3, &rows(4, 3, 1), &[], 0), lim()).unwrap();
    validate(&wire_store(3, &rows(4, 3, 1), &[], true), lim()).unwrap();
    let meta = Seg::new(SegmentType::Meta, 9).build(b"k=v", &[]);
    validate(&store_with_extra_first(meta), lim()).unwrap();
}

#[test]
fn empty_and_every_truncation_are_refused() {
    assert_eq!(err(&[]), E::Empty);
    for bytes in [
        packed_store(3, &rows(3, 3, 1), &[], 0),
        wire_store(3, &rows(3, 3, 1), &[2], true),
    ] {
        let root_at = bytes.len().checked_sub(4096).filter(|_| bytes.len() > 4096);
        for cut in 1..bytes.len() {
            let r = validate(&bytes[..cut], lim());
            if Some(cut) == root_at {
                // The segment stream without its optional root page is complete.
                assert!(r.is_ok());
                continue;
            }
            let e = r.unwrap_err();
            assert!(
                matches!(e, E::Truncated { .. } | E::NoManifest),
                "cut {cut}: {e:?}"
            );
        }
    }
    let b = packed_store(3, &rows(1, 3, 1), &[], 0);
    assert_eq!(
        err(&b[..10]),
        E::Truncated {
            offset: 0,
            what: "segment header"
        }
    );
    assert_eq!(
        err(&b[..70]),
        E::Truncated {
            offset: 0,
            what: "segment payload"
        }
    );
}

#[test]
fn header_field_violations_have_typed_errors() {
    let p = b"payload";
    let cases: Vec<(Seg, E)> = vec![
        (
            Seg {
                magic: 0xDEAD_BEEF,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::BadMagic { offset: 0 },
        ),
        (
            Seg {
                version: 2,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::BadVersion { offset: 0 },
        ),
        (
            Seg {
                seg_type: 0x00,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::UnknownSegmentType {
                offset: 0,
                seg_type: 0,
            },
        ),
        (
            Seg {
                seg_type: 0x40,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::UnknownSegmentType {
                offset: 0,
                seg_type: 0x40,
            },
        ),
        (
            Seg {
                flags: 0x8000,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::UnknownFlags { offset: 0 },
        ),
        (
            Seg {
                flags: 0x0001,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::UnsupportedCompression { offset: 0 },
        ),
        (
            Seg {
                compression: 1,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::UnsupportedCompression { offset: 0 },
        ),
        (
            Seg {
                reserved0: 1,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::ReservedNonZero {
                offset: 0,
                field: "reserved_0",
            },
        ),
        (
            Seg {
                reserved1: 1,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::ReservedNonZero {
                offset: 0,
                field: "reserved_1",
            },
        ),
        (
            Seg {
                uncompressed: 5,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::ReservedNonZero {
                offset: 0,
                field: "uncompressed_len",
            },
        ),
        (
            Seg {
                pad: 64,
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::BadPadding { offset: 0 },
        ),
        (
            Seg {
                algo: 3,
                hash: Some([0; 16]),
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::UnsupportedChecksum { offset: 0, algo: 3 },
        ),
        (
            Seg {
                hash: Some([7; 16]),
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::ChecksumMismatch { offset: 0 },
        ),
        (
            Seg {
                algo: 2,
                hash: Some(legacy_hash(p)),
                ..Seg::new(SegmentType::Meta, 1)
            },
            E::ChecksumMismatch { offset: 0 },
        ),
        (
            Seg::new(SegmentType::Kernel, 1),
            E::ExecutableSegment { offset: 0 },
        ),
        (
            Seg::new(SegmentType::Ebpf, 1),
            E::ExecutableSegment { offset: 0 },
        ),
        (
            Seg::new(SegmentType::Wasm, 1),
            E::ExecutableSegment { offset: 0 },
        ),
    ];
    for (seg, want) in cases {
        assert_eq!(
            err(&store_with_extra_first(seg.build(p, &[]))),
            want,
            "{seg:?}"
        );
    }
    // Non-zero padding bytes.
    let mut padded = Seg {
        pad: 9,
        ..Seg::new(SegmentType::Meta, 1)
    }
    .build(p, &[]);
    *padded.last_mut().unwrap() = 1;
    assert_eq!(
        err(&store_with_extra_first(padded)),
        E::BadPadding { offset: 0 }
    );
    // Correct padding and every accepted algorithm validate.
    for algo in 0..=2 {
        let ok = Seg {
            pad: 9,
            algo,
            ..Seg::new(SegmentType::Meta, 1)
        }
        .build(p, &[]);
        validate(&store_with_extra_first(ok), lim()).unwrap();
    }
    let lenient = ValidationLimits {
        allow_executable: true,
        ..lim()
    };
    validate(
        &store_with_extra_first(Seg::new(SegmentType::Kernel, 1).build(p, &[])),
        lenient,
    )
    .unwrap();
}

#[test]
fn signature_footers_are_bounded_and_consistent() {
    let signed = |f: &[u8]| {
        Seg {
            flags: 0x0004,
            ..Seg::new(SegmentType::Meta, 1)
        }
        .build(b"x", f)
    };
    validate(&store_with_extra_first(signed(&footer(64))), lim()).unwrap();
    assert_eq!(
        err(&store_with_extra_first(signed(&footer(0)))),
        E::BadFooter { offset: 0 }
    );
    let mut wrong_len = footer(64);
    let n = wrong_len.len();
    wrong_len[n - 4..].copy_from_slice(&99u32.to_le_bytes());
    assert_eq!(
        err(&store_with_extra_first(signed(&wrong_len))),
        E::BadFooter { offset: 0 }
    );
    let small = ValidationLimits {
        max_signature_len: 32,
        ..lim()
    };
    let e = validate(&store_with_extra_first(signed(&footer(64))), small).unwrap_err();
    assert_eq!(e, E::BadFooter { offset: 0 });
}

#[test]
fn oversized_inputs_are_refused_before_reading_payloads() {
    // A header declaring an absurd payload: refused from the header alone.
    let huge = Seg {
        payload_len: Some(u64::MAX),
        ..Seg::new(SegmentType::Meta, 1)
    }
    .build(&[], &[]);
    assert!(matches!(
        err(&huge),
        E::SegmentTooLarge { len: u64::MAX, .. }
    ));
    let big = Seg {
        payload_len: Some(200 << 20),
        ..Seg::new(SegmentType::Meta, 1)
    }
    .build(&[], &[]);
    let small = ValidationLimits {
        max_total_bytes: 1 << 20,
        ..lim()
    };
    assert_eq!(
        validate(&big, small).unwrap_err(),
        E::TooLarge { limit: 1 << 20 }
    );
    // Streaming total over the limit.
    let bytes = packed_store(8, &rows(100, 8, 2), &[], 0);
    let tight = ValidationLimits {
        max_total_bytes: bytes.len() as u64 - 1,
        ..lim()
    };
    assert!(matches!(
        validate(&bytes, tight).unwrap_err(),
        E::TooLarge { .. }
    ));
    // Oversized manifest payload.
    let tiny_manifest = ValidationLimits {
        max_manifest_payload: 16,
        ..lim()
    };
    assert!(matches!(
        validate(&packed_store(3, &rows(1, 3, 1), &[], 0), tiny_manifest).unwrap_err(),
        E::ManifestTooLarge { limit: 16, .. }
    ));
}

#[test]
fn segment_bomb_is_bounded() {
    let limits = ValidationLimits {
        max_segments: 64,
        ..lim()
    };
    let mut bomb = Vec::new();
    for i in 0..10_000u64 {
        bomb.extend(Seg::new(SegmentType::Meta, i).build(&[], &[]));
    }
    let mut v = StreamValidator::new(limits);
    assert_eq!(v.push(&bomb).unwrap_err(), E::TooManySegments { limit: 64 });
    // Sticky: later pushes and finish keep reporting it.
    assert_eq!(
        v.push(b"more").unwrap_err(),
        E::TooManySegments { limit: 64 }
    );
    assert_eq!(v.finish().unwrap_err(), E::TooManySegments { limit: 64 });
}

#[test]
fn manifest_and_vector_semantics_are_enforced() {
    let data = rows(2, 3, 1);
    let vp = vec_payload(3, &data);
    let vec_seg = Seg::new(SegmentType::Vec, 1).build(&vp, &[]);
    let with_manifest = |man: Vec<u8>| {
        let mut b = vec_seg.clone();
        b.extend(Seg::new(SegmentType::Manifest, 2).build(&man, &[]));
        b
    };
    let dir = |off, id, len, t| vec![(id, off, len, t)];
    let n = vp.len() as u64;
    assert_eq!(err(&vec_seg), E::NoManifest);
    assert_eq!(
        err(&with_manifest(manifest_payload(
            3,
            2,
            0,
            &dir(64, 1, n, 1),
            &[]
        ))),
        E::DirectoryMismatch {
            offset: vec_seg.len() as u64,
            target: 64
        }
    );
    assert!(matches!(
        err(&with_manifest(manifest_payload(
            3,
            2,
            0,
            &dir(0, 7, n, 1),
            &[]
        ))),
        E::DirectoryMismatch { .. }
    ));
    assert!(matches!(
        err(&with_manifest(manifest_payload(
            3,
            2,
            0,
            &dir(0, 1, n, 7),
            &[]
        ))),
        E::DirectoryMismatch { .. }
    ));
    assert_eq!(
        err(&with_manifest(manifest_payload(
            4,
            2,
            0,
            &dir(0, 1, n, 1),
            &[]
        ))),
        E::DimensionMismatch {
            expected: 4,
            found: 3
        }
    );
    assert_eq!(
        err(&with_manifest(manifest_payload(0, 2, 0, &[], &[]))),
        E::BadDimension { dim: 0 }
    );
    let narrow = ValidationLimits {
        max_dimension: 2,
        ..lim()
    };
    assert_eq!(
        validate(&with_manifest(manifest_payload(3, 2, 0, &[], &[])), narrow).unwrap_err(),
        E::BadDimension { dim: 3 }
    );
    assert_eq!(
        err(&with_manifest(manifest_payload(3, 2, 9, &[], &[]))),
        E::UnknownMetric(9)
    );
    let mut trailing = manifest_payload(3, 2, 0, &[], &[]);
    trailing.push(0);
    assert!(matches!(
        err(&with_manifest(trailing)),
        E::MalformedManifest { .. }
    ));
    // VEC_SEG whose count disagrees with its length.
    let mut bad_vec = vp.clone();
    bad_vec[2] = 3;
    assert!(matches!(
        err(&Seg::new(SegmentType::Vec, 1).build(&bad_vec, &[])),
        E::VecSegMalformed { .. }
    ));
    assert!(matches!(
        err(&Seg::new(SegmentType::Vec, 1).build(&vp[..4], &[])),
        E::VecSegMalformed { .. }
    ));
}

#[test]
fn root_page_must_be_valid_and_final() {
    let bytes = wire_store(3, &rows(2, 3, 1), &[], true);
    let mut corrupt = bytes.clone();
    let n = corrupt.len();
    corrupt[n - 100] ^= 1;
    assert!(matches!(err(&corrupt), E::RootManifestInvalid { .. }));
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(
        err(&trailing),
        E::TrailingBytes {
            offset: bytes.len() as u64
        }
    );
    // A root page with a conflicting dimension.
    let mut other_dim = wire_store(3, &rows(2, 3, 1), &[], false);
    let root = rvf_wire::manifest_codec::Level0Root {
        dimension: 5,
        ..Default::default()
    };
    other_dim.extend_from_slice(&rvf_wire::manifest_codec::write_root_manifest(&root));
    assert_eq!(
        err(&other_dim),
        E::DimensionMismatch {
            expected: 3,
            found: 5
        }
    );
    // A root page cannot open the stream.
    let lone = rvf_wire::manifest_codec::write_root_manifest(&root);
    assert_eq!(err(&lone), E::BadMagic { offset: 0 });
}

#[test]
fn status_codes_map_limits_to_413() {
    assert_eq!(E::TooLarge { limit: 1 }.http_status(), 413);
    assert_eq!(
        E::SegmentTooLarge {
            offset: 0,
            len: 2,
            limit: 1
        }
        .http_status(),
        413
    );
    assert_eq!(E::NoManifest.http_status(), 400);
}
