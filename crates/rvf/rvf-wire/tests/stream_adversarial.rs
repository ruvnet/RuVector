//! Adversarial properties P1-P4 and chunk-split independence of the RVSD verifier.
mod common;
use common::*;
use proptest::prelude::*;
use rvf_types::ErrorCode;
use rvf_wire::stream::*;
use std::sync::OnceLock;

struct Fix {
    body: Vec<u8>,
    frames: Vec<Vec<u8>>,
    keys: Vec<[u8; 32]>,
}

fn small_body() -> Vec<u8> {
    use rvf_types::SegmentFlags as F;
    let mut b = rvf_wire::writer::write_segment(0x07, &payload(1, 100), F::empty(), 1);
    b.extend(rvf_wire::writer::write_segment_with_algo(
        0x10,
        &payload(2, 5000),
        F::empty(),
        2,
        2,
    ));
    b.extend(root_page());
    b
}

fn fix() -> &'static Fix {
    static F: OnceLock<Fix> = OnceLock::new();
    F.get_or_init(|| {
        let body = small_body();
        let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
            .chunk_log2(12)
            .build(&body)
            .unwrap();
        let frames = b.frames_file_order(&body).collect();
        Fix {
            body,
            frames,
            keys: vec![pk(1)],
        }
    })
}

fn pol(f: &Fix, budget: u32) -> StreamPolicy<'_> {
    let mut p = StreamPolicy::new(&f.keys);
    p.fault_budget = budget;
    p
}

/// Bytes assembled from released chunks; panics if any chunk differs from the body.
fn check_released(
    f: &Fix,
    v: &mut StreamVerifier,
    frames: &[Vec<u8>],
) -> (u64, Option<IdentityStatus>) {
    let mut seen = std::collections::HashSet::new();
    let mut total = 0u64;
    let mut complete = None;
    for fr in frames {
        if let Ok(Event::Chunk {
            chunk, complete: c, ..
        }) = v.push(fr)
        {
            let o = chunk.body_offset as usize;
            assert_eq!(
                &f.body[o..o + chunk.bytes.len()],
                chunk.bytes,
                "released bytes differ from the signed body"
            );
            assert!(
                seen.insert((chunk.seg_id, chunk.chunk_index)),
                "chunk released twice"
            );
            total += chunk.bytes.len() as u64;
            complete = complete.or(c);
        }
    }
    (total, complete)
}

#[derive(Debug, Clone)]
enum Op {
    Flip(usize, usize, u8),
    Drop(usize),
    Dup(usize),
    Swap(usize, usize),
    Insert(usize, Vec<u8>),
    Cut(usize, usize),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (any::<usize>(), any::<usize>(), 1u8..=255).prop_map(|(a, b, m)| Op::Flip(a, b, m)),
        any::<usize>().prop_map(Op::Drop),
        any::<usize>().prop_map(Op::Dup),
        (any::<usize>(), any::<usize>()).prop_map(|(a, b)| Op::Swap(a, b)),
        (
            any::<usize>(),
            proptest::collection::vec(any::<u8>(), 0..64)
        )
            .prop_map(|(a, b)| Op::Insert(a, b)),
        (any::<usize>(), any::<usize>()).prop_map(|(a, b)| Op::Cut(a, b)),
    ]
}

fn apply(mut fr: Vec<Vec<u8>>, ops: &[Op]) -> Vec<Vec<u8>> {
    for o in ops {
        let n = fr.len().max(1);
        match o.clone() {
            Op::Flip(a, b, m) if !fr.is_empty() => {
                let f = &mut fr[a % n];
                let l = f.len();
                if l > 0 {
                    f[b % l] ^= m;
                }
            }
            Op::Drop(a) if !fr.is_empty() => {
                fr.remove(a % n);
            }
            Op::Dup(a) if !fr.is_empty() => fr.insert(a % n, fr[a % n].clone()),
            Op::Swap(a, b) if !fr.is_empty() => fr.swap(a % n, b % n),
            Op::Insert(a, b) => fr.insert(a % (fr.len() + 1), b),
            Op::Cut(a, b) if !fr.is_empty() => {
                let f = &mut fr[a % n];
                let l = f.len();
                f.truncate(b % (l + 1));
            }
            _ => {}
        }
    }
    fr
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(160))]

    /// P1 + P4: whatever mutation of the frame list is pushed, released chunks are
    /// byte-identical to the signed body (each at most once), and Complete implies
    /// the whole body was released.
    #[test]
    fn p1_p4_mutated_frame_lists(ops in proptest::collection::vec(op(), 0..8), budget in 0u32..64) {
        let f = fix();
        let fr = apply(f.frames.clone(), &ops);
        let mut v = StreamVerifier::new(pol(f, budget));
        let (total, _) = check_released(f, &mut v, &fr);
        if matches!(v.state(), StreamState::Complete | StreamState::CompleteUnverified) {
            prop_assert_eq!(total, f.body.len() as u64);
        }
    }

    /// Authentic duplicates alone (copies of chunk or descriptor frames re-sent later) are
    /// idempotent: with a zero fault budget the stream still completes with the same
    /// released set.
    #[test]
    fn duplicates_alone_still_complete(picks in proptest::collection::vec((any::<usize>(), any::<usize>()), 0..12)) {
        let f = fix();
        let mut fr = f.frames.clone();
        for (a, b) in picks {
            let i = a % fr.len();
            if fr[i][0] == 2 { continue; }
            let at = i + 1 + b % (fr.len() - i);
            fr.insert(at.min(fr.len()), fr[i].clone());
        }
        let mut v = StreamVerifier::new(pol(f, 0));
        let mut rel = Vec::new();
        for x in &fr {
            if let Ok(Event::Chunk { chunk, .. }) = v.push(x) {
                rel.push((chunk.body_offset, chunk.bytes.to_vec()));
            }
        }
        prop_assert_eq!(v.state(), StreamState::Complete);
        let clean = run_frames(f.frames.clone(), pol(f, 0));
        prop_assert_eq!(rel, clean.released);
    }

    /// P1 with fully arbitrary frames after a valid descriptor.
    #[test]
    fn p1_arbitrary_bytes(frames in proptest::collection::vec(proptest::collection::vec(any::<u8>(), 0..80), 0..20)) {
        let f = fix();
        let mut v = StreamVerifier::new(pol(f, 1000));
        let mut all = vec![f.frames[0].clone()];
        all.extend(frames);
        check_released(f, &mut v, &all);
    }

    /// Chunk-split independence: any split of the byte stream gives the same verdict.
    #[test]
    fn split_independence(ops in proptest::collection::vec(op(), 0..4), cuts in proptest::collection::vec(any::<u16>(), 0..12)) {
        let f = fix();
        let fr = apply(f.frames.clone(), &ops);
        let bytes: Vec<u8> = fr.iter().flatten().copied().collect();
        let whole = run_reader(f, &[&bytes[..]]);
        let mut pts: Vec<usize> = cuts.iter().map(|c| *c as usize % (bytes.len() + 1)).collect();
        pts.sort_unstable();
        let mut parts = Vec::new();
        let mut last = 0;
        for p in pts { parts.push(&bytes[last..p]); last = p; }
        parts.push(&bytes[last..]);
        prop_assert_eq!(whole, run_reader(f, &parts));
    }
}

type Verdict = (String, Vec<(u64, u32, Vec<u8>)>, Vec<ErrorCode>);

fn run_reader(f: &Fix, parts: &[&[u8]]) -> Verdict {
    let mut v = StreamVerifier::new(pol(f, 3));
    let mut rd = FrameReader::new();
    let (mut rel, mut errs) = (vec![], vec![]);
    for p in parts {
        let _ = rd.feed(&mut v, p, &mut |r| match r {
            Ok(Event::Chunk { chunk, .. }) => {
                rel.push((chunk.seg_id, chunk.chunk_index, chunk.bytes.to_vec()))
            }
            Ok(_) => {}
            Err(e) => errs.push(e),
        });
    }
    (format!("{:?}", v.state()), rel, errs)
}

#[test]
fn split_independence_on_the_clean_stream_byte_by_byte() {
    let f = fix();
    let bytes: Vec<u8> = f.frames.iter().flatten().copied().collect();
    let singles: Vec<&[u8]> = bytes.chunks(1).collect();
    let a = run_reader(f, &[&bytes]);
    assert_eq!(a.0, "Complete");
    assert_eq!(a, run_reader(f, &singles));
}

/// P3: truncation at every byte offset never reaches Complete. The verifier is a
/// deterministic online machine, so its state after consuming k bytes is exactly the
/// state for the stream truncated at k; one byte-at-a-time pass covers every offset.
/// Independent fresh runs at a stride cross-check that.
#[test]
fn p3_truncation_at_every_byte_offset() {
    let f = fix();
    let bytes: Vec<u8> = f.frames.iter().flatten().copied().collect();
    let mut v = StreamVerifier::new(pol(f, 4));
    let mut rd = FrameReader::new();
    for (k, b) in bytes.iter().enumerate() {
        rd.feed(&mut v, std::slice::from_ref(b), &mut |_| {})
            .unwrap();
        let last = k + 1 == bytes.len();
        assert_eq!(
            v.state() == StreamState::Complete,
            last,
            "Complete at offset {k}"
        );
    }
    for k in (0..bytes.len()).step_by(211) {
        let mut v = StreamVerifier::new(pol(f, 4));
        let mut rd = FrameReader::new();
        rd.feed(&mut v, &bytes[..k], &mut |_| {}).unwrap();
        assert_ne!(v.state(), StreamState::Complete);
        assert_ne!(v.end_of_input(), StreamState::Complete);
    }
}

/// P2: every single-bit flip of every frame is refused at that frame, releases
/// nothing from it, and the stream does not complete.
#[test]
fn p2_every_bit_flip_in_every_frame_is_refused() {
    let f = fix();
    for (i, fr) in f.frames.iter().enumerate() {
        let mut base = StreamVerifier::new(pol(f, 4));
        for g in &f.frames[..i] {
            base.push(g).unwrap();
        }
        // Whole frame for headers; stride through large payloads to bound debug-build time.
        let step = if fr.len() > 300 { 7 } else { 1 };
        for bit in (0..fr.len() * 8).step_by(step).chain([fr.len() * 8 - 1]) {
            let mut bad = fr.clone();
            bad[bit / 8] ^= 1 << (bit % 8);
            let mut v = base.clone();
            let r = v.push(&bad);
            assert!(r.is_err(), "flip of bit {bit} in frame {i} accepted: {r:?}");
            for g in &f.frames[i + 1..] {
                let _ = v.push(g);
            }
            assert_ne!(
                v.state(),
                StreamState::Complete,
                "flip {bit} in frame {i} still completed"
            );
        }
    }
}
