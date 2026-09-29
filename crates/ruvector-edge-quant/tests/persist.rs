//! Persist v2: round-trip identity, frame limits, every corruption shape
//! → typed error, and a forged oversized header → 413 before allocation.

#![allow(clippy::field_reassign_with_default)]

mod common;

use common::*;
use ruvector_edge_quant::persist::{self, format, load_frames, load_from, save_frames, save_to};
use ruvector_edge_quant::*;

fn build(
    n: usize,
    dim: usize,
    metric: Metric,
    kind: RandomRotationKind,
) -> (QuantShard, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (data, qs) = clustered(n, 25, dim, 10, 0.35, 11);
    let mut cfg = QuantConfig::new(dim, metric, 0xC0FFEE);
    cfg.rotation = kind;
    let mut s = QuantShard::new(cfg).unwrap();
    s.upsert(&rows(&data)).unwrap();
    s.delete(&[3, 17]); // exercise non-contiguous keys
    (s, data, qs)
}

fn answers(s: &QuantShard, data: &[Vec<f32>], qs: &[Vec<f32>]) -> Vec<Vec<(u64, u32, bool)>> {
    let mut src = MemRows::from_data(data);
    let mut out = Vec::new();
    for (i, q) in qs.iter().enumerate() {
        let opts = QueryOptions {
            top_k: 10,
            rerank_factor: 5,
        };
        let r = if i % 2 == 0 {
            s.query(q, &opts, Some(&mut src)).unwrap()
        } else {
            s.query(q, &opts, None).unwrap()
        };
        out.push(
            r.hits
                .iter()
                .map(|h| (h.key, h.distance.to_bits(), h.exact))
                .collect(),
        );
    }
    out
}

#[test]
fn round_trip_gives_bit_identical_state_and_results() {
    for (metric, kind) in [
        (Metric::Cosine, RandomRotationKind::HaarDense),
        (Metric::L2, RandomRotationKind::HadamardSigned),
        (Metric::Dot, RandomRotationKind::HaarDense),
    ] {
        let (s, data, qs) = build(3_000, 100, metric, kind);
        let before = answers(&s, &data, &qs);
        for mfb in [format::MIN_FRAME_BYTES, 4096, persist::MAX_FRAME_BYTES] {
            let frames = save_frames(&s, mfb).unwrap();
            assert!(frames.iter().all(|f| f.len() <= mfb), "frame over {mfb}");
            let l = load_frames(frames.iter().map(|f| f.as_slice()), Budget::default()).unwrap();
            assert_eq!(l.keys(), s.keys());
            assert_eq!(l.norms(), s.norms());
            assert_eq!(l.packed(), s.packed());
            assert_eq!(l.config(), s.config());
            assert_eq!(answers(&l, &data, &qs), before, "{metric:?} {kind:?} {mfb}");

            let mut buf = Vec::new();
            save_to(&s, &mut buf, mfb).unwrap();
            assert_eq!(buf, frames.concat());
            let l2 = load_from(&mut buf.as_slice(), Budget::default()).unwrap();
            assert_eq!(answers(&l2, &data, &qs), before);
        }
    }
}

#[test]
fn loaded_shard_keeps_accepting_writes() {
    let (s, data, _) = build(500, 64, Metric::L2, RandomRotationKind::HaarDense);
    let frames = save_frames(&s, 4096).unwrap();
    let mut l = load_frames(frames.iter().map(|f| f.as_slice()), Budget::default()).unwrap();
    l.upsert(&[(3, data[3].as_slice())]).unwrap();
    assert!(l.contains(3));
    let mut fresh = QuantShard::new(*s.config()).unwrap();
    fresh.upsert(&[(3, data[3].as_slice())]).unwrap();
    let n = l.n_words();
    let pos = l.keys().iter().position(|&k| k == 3).unwrap();
    assert_eq!(&l.packed()[pos * n..(pos + 1) * n], fresh.packed());
}

#[test]
fn empty_shard_round_trips() {
    let s = QuantShard::new(QuantConfig::new(384, Metric::Cosine, 5)).unwrap();
    let frames = save_frames(&s, persist::MAX_FRAME_BYTES).unwrap();
    assert_eq!(frames.len(), 2);
    let l = load_frames(frames.iter().map(|f| f.as_slice()), Budget::default()).unwrap();
    assert!(l.is_empty());
}

fn load_concat(buf: &[u8]) -> Result<QuantShard> {
    load_from(&mut &buf[..], Budget::default())
}

fn kind(r: Result<QuantShard>) -> CorruptKind {
    match r {
        Err(QuantError::Corrupt(k)) => k,
        Err(e) => panic!("expected Corrupt, got {e:?}"),
        Ok(_) => panic!("expected Corrupt, got Ok"),
    }
}

#[test]
fn corruption_is_a_typed_error() {
    let (s, _, _) = build(400, 96, Metric::L2, RandomRotationKind::HaarDense);
    let frames = save_frames(&s, 1024).unwrap();
    let whole = frames.concat();
    let hdr = format::HEADER_LEN;

    let mut b = whole.clone();
    b[0] ^= 1;
    assert_eq!(kind(load_concat(&b)), CorruptKind::BadMagic);

    let mut b = whole.clone();
    b[8] = 3;
    assert_eq!(kind(load_concat(&b)), CorruptKind::UnsupportedVersion(3));

    let mut b = whole.clone();
    b[20] ^= 0x40; // n
    assert_eq!(kind(load_concat(&b)), CorruptKind::HeaderChecksum);

    // A flipped payload byte in every data frame is caught by its CRC.
    let mut off = hdr;
    for (i, f) in frames[1..frames.len() - 1].iter().enumerate() {
        let mut b = whole.clone();
        b[off + format::FRAME_PREFIX_LEN + f.len() / 3 % (f.len() - 16)] ^= 0x10;
        assert_eq!(
            kind(load_concat(&b)),
            CorruptKind::FrameChecksum(i as u32),
            "frame {i}"
        );
        off += f.len();
    }

    // Frame prefix: index / section / length fields.
    let mut b = whole.clone();
    b[hdr] = 7;
    assert_eq!(
        kind(load_concat(&b)),
        CorruptKind::FrameSequence {
            expected: 0,
            got: 7
        }
    );
    let mut b = whole.clone();
    b[hdr + 4] = format::SECTION_CODES;
    assert!(matches!(kind(load_concat(&b)), CorruptKind::Malformed(_)));
    let mut b = whole.clone();
    b[hdr + 8] ^= 1;
    assert!(matches!(kind(load_concat(&b)), CorruptKind::Malformed(_)));

    // Swapped frames.
    let mut fs = frames.clone();
    fs.swap(1, 2);
    let r = load_frames(fs.iter().map(|f| f.as_slice()), Budget::default());
    assert_eq!(
        kind(r),
        CorruptKind::FrameSequence {
            expected: 0,
            got: 1
        }
    );

    // Footer.
    let n = whole.len();
    let mut b = whole.clone();
    b[n - 1] ^= 1;
    assert_eq!(kind(load_concat(&b)), CorruptKind::FooterChecksum);
    let mut b = whole.clone();
    b[n - 16] ^= 1;
    assert!(matches!(kind(load_concat(&b)), CorruptKind::Malformed(_)));

    // Truncation anywhere, and trailing bytes.
    for cut in [4, hdr - 1, hdr + 10, n / 2, n - 16, n - 1] {
        assert_eq!(
            kind(load_concat(&whole[..cut])),
            CorruptKind::Truncated,
            "cut {cut}"
        );
    }
    let r = load_frames(
        frames[..frames.len() - 1].iter().map(|f| f.as_slice()),
        Budget::default(),
    );
    assert_eq!(kind(r), CorruptKind::Truncated);
    let mut b = whole.clone();
    b.push(0);
    assert_eq!(kind(load_concat(&b)), CorruptKind::TrailingData);
    let mut fs = frames.clone();
    fs.push(vec![0; 16]);
    let r = load_frames(fs.iter().map(|f| f.as_slice()), Budget::default());
    assert_eq!(kind(r), CorruptKind::TrailingData);
}

/// Rewrite a header field and fix its CRC, as a forger (or a writer bug)
/// with a valid-checksum but inconsistent header would.
fn forge_header(frame: &[u8], edit: impl FnOnce(&mut format::Header)) -> Vec<u8> {
    let mut h = format::Header::decode(frame).unwrap();
    edit(&mut h);
    h.encode().to_vec()
}

/// Re-seal a data frame after editing its payload so only content checks fire.
fn reseal(frames: &mut [Vec<u8>], i: usize, edit: impl FnOnce(&mut [u8])) {
    edit(&mut frames[i][format::FRAME_PREFIX_LEN..]);
    let f = &mut frames[i];
    let mut c = format::Crc32::new();
    c.update(&f[0..12]);
    c.update(&f[format::FRAME_PREFIX_LEN..]);
    let crc = c.finish();
    f[12..16].copy_from_slice(&crc.to_le_bytes());
    let mut all = format::Crc32::new();
    for g in &frames[1..frames.len() - 1] {
        all.update(&g[12..16]);
    }
    let last = frames.len() - 1;
    frames[last][12..16].copy_from_slice(&all.finish().to_le_bytes());
}

fn load_vec(fs: &[Vec<u8>]) -> Result<QuantShard> {
    load_frames(fs.iter().map(|f| f.as_slice()), Budget::default())
}

#[test]
fn checksum_valid_but_inconsistent_content_is_rejected() {
    let (s, _, _) = build(200, 70, Metric::L2, RandomRotationKind::HaarDense);
    let frames = save_frames(&s, persist::MAX_FRAME_BYTES).unwrap(); // hdr, keys, norms, codes, footer
    assert_eq!(frames.len(), 5);

    let mut fs = frames.clone();
    fs[0] = forge_header(&fs[0], |h| h.seed ^= 1);
    assert_eq!(kind(load_vec(&fs)), CorruptKind::RotationMismatch);

    let mut fs = frames.clone();
    fs[0] = forge_header(&fs[0], |h| h.rotation = RandomRotationKind::HadamardSigned);
    assert_eq!(kind(load_vec(&fs)), CorruptKind::RotationMismatch);

    let mut fs = frames.clone();
    fs[0] = forge_header(&fs[0], |h| h.n += 1);
    assert!(matches!(kind(load_vec(&fs)), CorruptKind::Malformed(_)));

    let mut fs = frames.clone();
    reseal(&mut fs, 1, |p| p.copy_within(0..8, 8));
    assert_eq!(kind(load_vec(&fs)), CorruptKind::DuplicateKey);

    let mut fs = frames.clone();
    reseal(&mut fs, 2, |p| {
        p[0..4].copy_from_slice(&f32::NAN.to_le_bytes())
    });
    assert_eq!(kind(load_vec(&fs)), CorruptKind::InvalidNorm);
    let mut fs = frames.clone();
    reseal(&mut fs, 2, |p| {
        p[4..8].copy_from_slice(&(-1.0f32).to_le_bytes())
    });
    assert_eq!(kind(load_vec(&fs)), CorruptKind::InvalidNorm);

    // dim 70 → 2 words, 58 padding bits in word 1: set the lowest.
    let mut fs = frames.clone();
    reseal(&mut fs, 3, |p| p[8] |= 1);
    assert_eq!(kind(load_vec(&fs)), CorruptKind::PaddingBits);
}

#[test]
fn oversized_header_is_413_before_allocation() {
    let s = QuantShard::new(QuantConfig::new(384, Metric::Cosine, 1)).unwrap();
    let frames = save_frames(&s, persist::MAX_FRAME_BYTES).unwrap();
    // A self-consistent header claiming 10M rows (≈ 900 MB resident): only
    // the 64-byte header is supplied, so any allocation-first loader would
    // have to reserve the rows before noticing.
    let forged = forge_header(&frames[0], |h| {
        h.n = 10_000_000;
        let lens = format::section_lens(h.n, 384).unwrap();
        h.payload_bytes = lens.iter().sum();
        h.frames =
            format::frame_count(&lens, format::payload_cap(h.max_frame_bytes as usize)) as u32;
    });
    let r = persist::Decoder::new(&forged, Budget::default());
    match r {
        Err(e @ QuantError::BudgetExceeded { .. }) => assert_eq!(e.status(), 413),
        Err(e) => panic!("expected 413, got {e:?}"),
        Ok(_) => panic!("expected 413"),
    }
    let r = load_from(&mut forged.as_slice(), Budget::default());
    assert_eq!(r.err().unwrap().status(), 413);

    // A real snapshot over a tighter budget is refused the same way.
    let (s, _, _) = build(1_000, 64, Metric::L2, RandomRotationKind::HaarDense);
    let frames = save_frames(&s, 4096).unwrap();
    let mut b = Budget::default();
    b.max_vectors = 500;
    let r = load_frames(frames.iter().map(|f| f.as_slice()), b);
    assert!(matches!(
        r,
        Err(QuantError::BudgetExceeded {
            resource: BudgetResource::Vectors,
            ..
        })
    ));
}

#[test]
fn header_row_count_overflow_is_malformed_not_a_panic() {
    let s = QuantShard::new(QuantConfig::new(384, Metric::Cosine, 1)).unwrap();
    let frames = save_frames(&s, persist::MAX_FRAME_BYTES).unwrap();
    // Each section length fits u64 but their sum does not (n × 60 > 2^64).
    let forged = forge_header(&frames[0], |h| h.n = 350_000_000_000_000_000);
    let r = load_from(&mut forged.as_slice(), Budget::default());
    assert!(matches!(kind(r), CorruptKind::Malformed(_)));
}
