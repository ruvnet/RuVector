//! Adversarial review repros for the RVSD verifier (project Ofer review).
//! Finding ids: F-04 .. F-08. Tests that assert *current behaviour that is a
//! defect* are named so; the oracle test (F-08) is expected to pass.
mod common;
use common::*;
use rvf_types::ErrorCode;
use rvf_wire::stream::*;

fn bundle(stages: &[u8]) -> (Vec<u8>, StreamBundle) {
    let body = sample_body(1);
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(12)
        .stages(stages)
        .build(&body)
        .unwrap();
    (body, b)
}

/// F-04: a stream that reached `Complete` is flipped to `Poisoned` by a single
/// trailing junk byte (unknown frame type) fed through the FrameReader, and by
/// `poison()`. A consumer that checks `state()` at end of transfer discards a
/// fully verified body ("revoke, terminate, drop layers").
#[test]
fn f04_trailing_junk_does_not_change_complete() {
    let (body, b) = bundle(&[0]);
    let keys = [pk(1)];
    let mut v = StreamVerifier::new(policy(&keys));
    let mut fr = FrameReader::new();
    let mut n = 0;
    for f in b.frames_file_order(&body) {
        fr.feed(&mut v, &f, &mut |r| {
            if r.is_ok() {
                n += 1
            }
        })
        .unwrap();
    }
    assert_eq!(v.state(), StreamState::Complete);
    // attacker (or a sloppy transport) appends one byte after the last frame
    let r = fr.feed(&mut v, &[0xFF], &mut |_| {});
    assert_eq!(r, Err(ErrorCode::StreamBadState));
    assert_eq!(
        v.state(),
        StreamState::Complete,
        "trailing bytes must not change Complete"
    );
    assert_eq!(
        v.poison(ErrorCode::StreamUnexpectedFrame),
        ErrorCode::StreamUnexpectedFrame
    );
    assert_eq!(v.state(), StreamState::Complete);
}

/// F-05: with `identity_present` and any delivery order other than file order
/// the stream reaches `Complete` with `Unverifiable`; nothing ever checks
/// `rvf_identity` unless the caller remembers `finish_identity`.
#[test]
fn f05_stage_order_completes_with_unverifiable_identity() {
    // stages: reverse-ish so stage order != file order
    let (body, b) = bundle(&[3, 0, 2, 1, 0, 0, 0]);
    let keys = [pk(1)];
    let r = run_frames(b.frames_stage_order(&body), policy(&keys));
    // The unsafe path is a distinct terminal state, not Complete.
    assert_eq!(r.state, StreamState::CompleteUnverified);
    assert_eq!(r.complete, Some(IdentityStatus::Unverifiable));
}

/// F-06: equal epoch + pin without a descriptor hash accepts a *different*
/// signed descriptor for the same file_id (no conflict detection).
#[test]
fn f06_equal_epoch_requires_matching_hash_and_pins_bind_signer() {
    let body = sample_body(1);
    let keys = [pk(1)];
    let a = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .epoch(5)
        .build(&body)
        .unwrap();
    let mut body2 = body.clone();
    body2[100] ^= 1; // different content, same file_id/epoch
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .epoch(5)
        .build(&body2)
        .unwrap();
    assert_ne!(a.descriptor_hash, b.descriptor_hash);
    // The hash is mandatory: there is no way to express "equal epoch, any content".
    let sid = rvf_crypto::signer_key_id(&pk(1));
    let pin = move |s: &[u8; 8], f: &[u8; 16]| {
        (*s == sid && *f == FILE_ID).then_some(EpochPin {
            epoch: 5,
            descriptor_hash: a.descriptor_hash,
        })
    };
    let mut p = policy(&keys);
    p.epoch_pin = Some(&pin);
    assert_eq!(
        decode_descriptor(&b.descriptor, &p).unwrap_err(),
        ErrorCode::StreamEpochConflict
    );
    // A pin bound to another signer does not constrain this lineage.
    let other = move |s: &[u8; 8], _: &[u8; 16]| {
        (*s != sid).then_some(EpochPin {
            epoch: 99,
            descriptor_hash: [0; 32],
        })
    };
    p.epoch_pin = Some(&other);
    assert!(decode_descriptor(&b.descriptor, &p).is_ok());
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// F-08 (oracle, expected to pass): random frame-level and byte-level
/// mutations, duplications, drops, cross-segment swaps and arbitrary splits.
/// Invariant: no panic; every released chunk equals the signed body slice;
/// `Complete` only if every body byte was released exactly once.
#[test]
fn f08_random_mutation_oracle() {
    let (body, b) = bundle(&[1, 0, 2, 2, 3, 3, 0]);
    let keys = [pk(1)];
    let base: Vec<Vec<u8>> = b.frames_stage_order(&body).collect();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let iters: usize = std::env::var("OFER_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let mut completes = 0usize;
    for _ in 0..iters {
        let mut frames = base.clone();
        for _ in 0..1 + rng.below(4) {
            match rng.below(7) {
                0 if frames.len() > 1 => {
                    let i = 1 + rng.below(frames.len() - 1);
                    let l = frames[i].len();
                    let j = rng.below(l);
                    frames[i][j] ^= 1 << rng.below(8);
                }
                1 => {
                    let i = rng.below(frames.len());
                    let f = frames[i].clone();
                    frames.insert(rng.below(frames.len() + 1), f);
                }
                2 if frames.len() > 1 => {
                    frames.remove(1 + rng.below(frames.len() - 1));
                }
                3 => {
                    let (i, j) = (rng.below(frames.len()), rng.below(frames.len()));
                    frames.swap(i, j);
                }
                4 => {
                    let i = rng.below(frames.len());
                    let l = frames[i].len();
                    frames[i].truncate(rng.below(l));
                }
                5 => {
                    let i = rng.below(frames.len());
                    let extra = rng.below(40);
                    for _ in 0..extra {
                        frames[i].push(rng.next() as u8);
                    }
                }
                _ => {
                    // splice a chunk frame body from a different segment
                    let i = rng.below(frames.len());
                    let j = rng.below(frames.len());
                    let (a, c) = (frames[i].clone(), frames[j].clone());
                    if a.len() > 17 && c.len() > 17 && a[0] == 1 && c[0] == 1 {
                        let mut m = a[..13].to_vec();
                        m.extend_from_slice(&c[13..]);
                        frames[i] = m;
                    }
                }
            }
        }
        let stream: Vec<u8> = frames.concat();
        let mut p = policy(&keys);
        p.fault_budget = rng.below(8) as u32;
        let mut v = StreamVerifier::new(p);
        let mut fr = FrameReader::new();
        let mut covered = vec![0u8; body.len()];
        let mut pos = 0;
        while pos < stream.len() {
            let n = 1 + rng.below(stream.len() - pos).min(1 + rng.below(9000));
            let end = (pos + n).min(stream.len());
            let _ = fr.feed(&mut v, &stream[pos..end], &mut |r| {
                if let Ok(Event::Chunk { chunk, .. }) = r {
                    let o = chunk.body_offset as usize;
                    assert_eq!(&body[o..o + chunk.bytes.len()], chunk.bytes);
                    for c in &mut covered[o..o + chunk.bytes.len()] {
                        assert_eq!(*c, 0, "byte released twice");
                        *c = 1;
                    }
                }
            });
            pos = end;
        }
        if v.state() == StreamState::Complete {
            completes += 1;
            assert!(covered.iter().all(|&c| c == 1), "Complete with gaps");
        }
    }
    eprintln!("{completes}/{iters} mutated streams still completed (benign mutations)");
}

/// F-09: the portable SHA-256 and sha2 agree (the commitments depend on it).
#[test]
fn f09_portable_sha256_matches_sha2() {
    use sha2::Digest;
    let mut rng = Rng(77);
    for len in (0..400).chain([1023, 1024, 1025, 4095, 4096, 4097, 65535, 65536, 65537]) {
        let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        let want: [u8; 32] = sha2::Sha256::digest(&data).into();
        assert_eq!(rvf_types::sha256::sha256(&data), want, "len {len}");
        let mut h = rvf_types::sha256::Sha256::new();
        let mut i = 0;
        while i < data.len() {
            let n = 1 + rng.below(130);
            let e = (i + n).min(data.len());
            h.update(&data[i..e]);
            i = e;
        }
        assert_eq!(h.finalize(), want, "split len {len}");
    }
}

/// F-10 (oracle): every single-bit flip, every truncation and every one-byte
/// extension of a signed descriptor is refused, for chunked and segment-
/// granular profiles.
#[test]
fn f10_descriptor_bitflip_truncate_extend_all_refused() {
    let body = sample_body(1);
    let keys = [pk(1)];
    for cl in [12u8, 16, 0xFF] {
        let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
            .chunk_log2(cl)
            .stages(&[1, 0, 2, 2, 3, 3, 0])
            .requires_lower_stages(true)
            .build(&body)
            .unwrap();
        let p = policy(&keys);
        assert!(decode_descriptor(&b.descriptor, &p).is_ok());
        for i in 0..b.descriptor.len() {
            for bit in 0..8 {
                let mut d = b.descriptor.clone();
                d[i] ^= 1 << bit;
                assert!(
                    decode_descriptor(&d, &p).is_err(),
                    "cl {cl:#x}: flip byte {i} bit {bit} accepted"
                );
            }
        }
        for n in 0..b.descriptor.len() {
            assert!(decode_descriptor(&b.descriptor[..n], &p).is_err());
        }
        let mut d = b.descriptor.clone();
        d.push(0);
        assert!(decode_descriptor(&d, &p).is_err());
    }
}

/// F-11 (32-bit only, info): `(hdr + payload) as usize` wraps for a 0xFFFF_FFF0-byte
/// frame; observed result is a runt-frame push and Poisoned (fail closed).
/// Run with: cargo test --target armv7-unknown-linux-musleabihf.
#[cfg(target_pointer_width = "32")]
#[test]
fn f11_frame_total_wraps_on_32bit() {
    use rvf_types::stream::*;
    let keys = [pk(1)];
    // Hand-built one-segment descriptor, segment-granular.
    let big = 0xFFFF_FFF0u64;
    let e = StreamEntry {
        seg_id: 1,
        seg_type: 0x01,
        stage: 0,
        flags: 0,
        requires_mask: 0,
        body_offset: 0,
        stored_len: big,
        uncompressed_len: 0,
        chunk_table_hash: [7; 32],
    };
    let h = StreamHeader {
        flags: 0,
        chunk_log2: STREAM_CHUNK_LOG2_SEGMENT,
        file_id: FILE_ID,
        epoch: 1,
        body_len: big,
        rvf_identity: [0; 32],
        policy_hash: [0; 32],
        seg_count: 1,
        table_len: 72,
        expires_ns: 0,
        signer_key_id: rvf_crypto::signer_key_id(&pk(1)),
    };
    let mut d = h.to_bytes().to_vec();
    d.extend_from_slice(&e.to_bytes());
    d.extend_from_slice(&64u16.to_le_bytes());
    let sig = rvf_crypto::sign_descriptor(&d, &key(1));
    d.extend_from_slice(&sig);
    let mut p = policy(&keys);
    p.max_segment_bytes = u32::MAX as u64;
    p.max_body_len = u32::MAX as u64;
    p.max_class_bytes = [u32::MAX as u64; 4];
    let mut v = StreamVerifier::from_descriptor(&d, p).expect("descriptor accepted");
    let mut fr = FrameReader::new();
    let mut hdr = vec![0x01u8];
    hdr.extend_from_slice(&1u64.to_le_bytes());
    hdr.extend_from_slice(&0u32.to_le_bytes());
    hdr.extend_from_slice(&(big as u32).to_le_bytes());
    let mut sink = |_r: Result<Event<'_>, ErrorCode>| {};
    // Header split across feeds: must not panic or buffer without bound.
    let r1 = fr.feed(&mut v, &hdr[..9], &mut sink);
    assert!(r1.is_ok());
    let r2 = fr.feed(&mut v, &hdr[9..], &mut sink);
    assert_eq!(r2, Err(ErrorCode::StreamFrameSizeLie));
    assert_eq!(
        v.state(),
        StreamState::Poisoned(ErrorCode::StreamFrameSizeLie)
    );
    assert!(fr.buffered() <= 17, "buffered {}", fr.buffered());
    let r3 = fr.feed(&mut v, &[0u8; 4096], &mut sink);
    assert!(r3.is_err());
    assert!(fr.buffered() <= 17);
}
