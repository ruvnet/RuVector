//! A malicious mirror must never get unverified data published, must end the
//! stream in a terminal or suspended state with the right reason, and must
//! never panic the client. (HTTP-level lies about Content-Length / Range are
//! exercised in rvf-server's tests.)

#[path = "support/stream_fixture.rs"]
mod fx;

use fx::*;
use rvf_runtime::progressive_stream::client::{ClientConfig, MemTransport, RunEnd, StreamClient};
use rvf_runtime::progressive_stream::StreamWitness;
use rvf_types::ErrorCode;
use rvf_wire::stream::{EpochPin, StreamPolicy, StreamState};

fn run_mirror<'p>(
    f: &Fixture,
    bytes: Vec<u8>,
    policy: StreamPolicy<'p>,
) -> (RunEnd, StreamClient<'p, MemTransport>) {
    let mut c = StreamClient::new(
        policy,
        MemTransport::new(bytes),
        ClientConfig {
            max_stalls: 2,
            frame_retries: 3,
        },
    );
    let end = c.run(&mut |ing| assert_authentic(ing, &f.ds));
    assert_authentic(&c.ingest, &f.ds);
    (end, c)
}

fn not_complete(end: RunEnd, c: &StreamClient<'_, MemTransport>) {
    assert!(!end.all_verified());
    assert!(!matches!(
        c.ingest.state(),
        StreamState::Complete | StreamState::CompleteUnverified
    ));
}

#[test]
fn truncation_at_many_offsets_never_completes() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    for cut in [0, 7, 100, full.len() / 5, full.len() / 2, full.len() - 1] {
        let (end, c) = run_mirror(&f, full[..cut].to_vec(), StreamPolicy::new(&keys));
        not_complete(end, &c);
        assert_eq!(end, RunEnd::Suspended, "cut {cut}");
    }
}

#[test]
fn reordered_frames_are_refused() {
    let f = fixture();
    let keys = [f.public];
    let mut fr = frames(&f);
    // Within a segment: swap two chunk frames (descriptor + first segment).
    let i = fr.iter().position(|x| x[0] == 0x01 && x[9] == 1).unwrap();
    fr.swap(i, i + 1);
    let (end, c) = run_mirror(&f, fr.concat(), StreamPolicy::new(&keys));
    not_complete(end, &c);

    // Across stages: the last frame (stage 4) delivered first, right after
    // the descriptor, must be refused (`requires_mask` unmet).
    let mut fr = frames(&f);
    let last = fr.pop().unwrap();
    fr.insert(1, last);
    let (end, c) = run_mirror(&f, fr.concat(), StreamPolicy::new(&keys));
    not_complete(end, &c);
    assert!(
        c.ingest.vectors().is_empty(),
        "nothing released before its stage"
    );
    assert!(!c.ingest.layers().layer_a);
}

#[test]
fn duplicated_and_bit_flipped_frames_are_refused() {
    let f = fixture();
    let keys = [f.public];
    let fr = frames(&f);
    let i = fr.iter().position(|x| x[0] == 0x01 && x[9] == 2).unwrap();
    for flip in [17usize, 100, 2000] {
        let mut bad = fr.clone();
        bad[i][flip] ^= 0x04;
        let (end, c) = run_mirror(&f, bad.concat(), StreamPolicy::new(&keys));
        not_complete(end, &c);
        assert!(matches!(end, RunEnd::Suspended | RunEnd::Poisoned(_)));
    }
    // A flipped descriptor bit is rejected outright.
    let mut bad = fr.clone();
    bad[0][5 + 0x40] ^= 1;
    let (end, c) = run_mirror(&f, bad.concat(), StreamPolicy::new(&keys));
    assert!(matches!(end, RunEnd::Poisoned(_)));
    assert!(matches!(c.ingest.state(), StreamState::Rejected(_)));
    assert!(c.ingest.vectors().is_empty());
}

#[test]
fn replayed_old_epoch_is_rejected_with_a_witness() {
    let f = fixture_with(2600, 16, 8, 11, 1);
    let keys = [f.public];
    // Pins are bound to (signer_key_id, file_id) and always carry the accepted descriptor hash.
    let pin = |_signer: &[u8; 8], id: &[u8; 16]| {
        (id == &f.file_id).then_some(EpochPin {
            epoch: 2,
            descriptor_hash: [0x5a; 32],
        })
    };
    let mut policy = StreamPolicy::new(&keys);
    policy.epoch_pin = Some(&pin);
    let (end, c) = run_mirror(&f, frames(&f).concat(), policy);
    assert_eq!(end, RunEnd::Poisoned(ErrorCode::StreamStaleEpoch));
    assert!(matches!(
        c.ingest.state(),
        StreamState::Rejected(ErrorCode::StreamStaleEpoch)
    ));
    assert!(c
        .ingest
        .witnesses()
        .contains(&StreamWitness::DescriptorRejected(
            ErrorCode::StreamStaleEpoch
        )));
    assert!(c.ingest.vectors().is_empty() && !c.ingest.layers().layer_a);
}

#[test]
fn chunks_spliced_from_another_file_signed_by_the_same_key_are_refused() {
    let a = fixture_with(2600, 16, 8, 11, 1);
    let b = fixture_with(2600, 16, 8, 12, 1); // same key, other file_id and data
    assert_ne!(a.file_id, b.file_id);
    let keys = [a.public];
    let fa = frames(&a);
    let fb = frames(&b);
    let i = fa.iter().position(|x| x[0] == 0x01 && x[9] == 3).unwrap();
    assert_eq!(fa[i][..13], fb[i][..13], "same seg_id and chunk_index");
    let mut spliced = fa.clone();
    spliced[i] = fb[i].clone();
    let (end, c) = run_mirror(&a, spliced.concat(), StreamPolicy::new(&keys));
    not_complete(end, &c);
    assert!(c.stats.refusals >= 1);
}

#[test]
fn size_lies_poison_before_allocating_and_unpublish_everything() {
    let f = fixture();
    let keys = [f.public];
    // Let layers A and B publish, then lie about a late chunk's length.
    let mut fr = frames(&f);
    let late = fr.len() - 3;
    assert_eq!(fr[late][0], 0x01);
    fr[late][13..17].copy_from_slice(&0xFFFF_FF00u32.to_le_bytes());
    let mut c = StreamClient::new(
        StreamPolicy::new(&keys),
        MemTransport::new(fr.concat()),
        ClientConfig::default(),
    );
    let mut published_a = false;
    let end = c.run(&mut |ing| published_a |= ing.layers().layer_a);
    assert!(published_a, "layer A was searchable before the lie");
    assert_eq!(end, RunEnd::Poisoned(ErrorCode::StreamFrameSizeLie));
    let l = c.ingest.layers();
    assert!(
        !l.layer_a && !l.layer_b && !l.layer_c,
        "poison drops every layer"
    );
    assert!(
        c.ingest.vectors().is_empty(),
        "poison unpublishes every vector"
    );
    let a = c.ingest.query(f.ds.vector(0), 5, 64);
    assert!(a.envelope.results.is_empty() && !a.sealed);
    assert!(matches!(
        c.ingest.witnesses().last(),
        Some(StreamWitness::StreamPoisoned {
            reason: ErrorCode::StreamFrameSizeLie,
            ..
        })
    ));
    // The reassembly buffer never grew toward the declared 4 GiB.
    assert!(c.ingest.offset() < 1 << 28);
}

#[test]
fn transient_corruption_is_retried_to_completion() {
    use rvf_runtime::progressive_stream::client::{FetchEnd, Transport};
    struct Flaky {
        good: Vec<u8>,
        corrupt_left: u32,
        seed: u64,
    }
    impl Transport for Flaky {
        fn fetch(
            &mut self,
            off: u64,
            sink: &mut dyn FnMut(&[u8]) -> bool,
        ) -> std::io::Result<FetchEnd> {
            let mut data = self.good[off as usize..].to_vec();
            if self.corrupt_left > 0 && data.len() > 5000 {
                self.corrupt_left -= 1;
                self.seed = self
                    .seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let at = 1000 + (self.seed >> 33) as usize % (data.len() - 1000);
                data[at] ^= 0x10;
            }
            for c in data.chunks(8192) {
                if !sink(c) {
                    return Ok(FetchEnd::Aborted);
                }
            }
            Ok(FetchEnd::Eof)
        }
    }
    let f = fixture();
    let keys = [f.public];
    let mut policy = StreamPolicy::new(&keys);
    policy.fault_budget = 100;
    let mut c = StreamClient::new(
        policy,
        Flaky {
            good: frames(&f).concat(),
            corrupt_left: 6,
            seed: 1,
        },
        ClientConfig {
            max_stalls: 4,
            frame_retries: 5,
        },
    );
    let end = c.run(&mut |ing| assert_authentic(ing, &f.ds));
    // Stage order is not file order: all chunks verified, identity unconfirmed.
    assert_eq!(end, RunEnd::CompleteUnverified);
    assert!(c.stats.refusals >= 1, "{:?}", c.stats);
    assert_eq!(c.ingest.vectors().len(), f.ds.len());
}

/// Authentic duplicates (chunk and descriptor frames) are idempotent: nothing
/// is applied twice, the witness trail is unchanged, the stream still
/// completes. A flood of them is capped and poisons.
#[test]
fn authentic_duplicates_never_double_apply_and_floods_poison() {
    let f = fixture();
    let keys = [f.public];
    let fr = frames(&f);
    let (base_end, base) = run_mirror(&f, fr.concat(), StreamPolicy::new(&keys));
    assert_eq!(base_end, RunEnd::CompleteUnverified);
    let base_w = base.ingest.witnesses().to_vec();
    let published = |w: &[StreamWitness], l: u8| {
        w.iter()
            .filter(|x| **x == StreamWitness::LayerPublished(l))
            .count()
    };

    let mut dup = fr.clone();
    let chunk_frames: Vec<usize> = (1..fr.len()).filter(|&i| fr[i][0] == 0x01).collect();
    // Duplicate the first chunk of each stage, an index chunk, and the descriptor.
    for &i in chunk_frames
        .iter()
        .rev()
        .step_by(chunk_frames.len() / 6 + 1)
    {
        dup.insert(i + 1, fr[i].clone());
    }
    dup.insert(1, fr[0].clone());
    let (end, c) = run_mirror(&f, dup.concat(), StreamPolicy::new(&keys));
    assert_eq!(end, RunEnd::CompleteUnverified);
    assert_eq!(c.ingest.witnesses(), &base_w[..], "witness trail unchanged");
    for l in [b'A', b'B', b'C'] {
        assert_eq!(
            published(c.ingest.witnesses(), l),
            1,
            "layer {} once",
            l as char
        );
    }
    assert_eq!(c.ingest.vectors().len(), f.ds.len());

    // Flood: the same authentic chunk over and over exceeds the cap.
    let i = chunk_frames[0];
    let mut flood = fr.clone();
    for _ in 0..50 {
        flood.insert(i + 1, fr[i].clone());
    }
    let mut policy = StreamPolicy::new(&keys);
    policy.max_duplicate_frames = Some(10);
    let (end, c) = run_mirror(&f, flood.concat(), policy);
    assert!(
        matches!(end, RunEnd::Poisoned(ErrorCode::StreamDuplicateFlood)),
        "{end:?}"
    );
    assert!(c.ingest.vectors().is_empty() && !c.ingest.layers().layer_a);
}

/// RV2-02: `sealed` is a derived release and re-checks the epoch pin: a session
/// superseded after completion stops claiming `sealed = true`.
#[test]
fn sealed_requires_the_pin_to_still_hold() {
    use rvf_wire::stream::{MemoryPinStore, PinStore};
    let f = fixture_with(2600, 16, 8, 11, 1);
    let keys = [f.public];
    let store = MemoryPinStore::new();
    let mut policy = StreamPolicy::new(&keys);
    policy.pins = Some(&store);
    let (_end, mut c) = run_mirror(&f, frames(&f).concat(), policy);
    assert_eq!(c.ingest.state(), StreamState::CompleteUnverified);
    let body = f.body.clone();
    c.ingest
        .finish_identity(|o, b| {
            b.copy_from_slice(&body[o as usize..o as usize + b.len()]);
            true
        })
        .unwrap();
    let q = f.ds.vector(0).to_vec();
    assert!(c.ingest.query(&q, 3, 16).sealed);
    let d = c.ingest.verifier().descriptor().unwrap().clone();
    store
        .advance(
            &d.header.signer_key_id,
            &d.header.file_id,
            None,
            EpochPin {
                epoch: d.header.epoch + 1,
                descriptor_hash: [7; 32],
            },
        )
        .unwrap();
    assert_eq!(
        c.ingest.recheck_pin(&store),
        Err(ErrorCode::StreamStaleEpoch)
    );
    assert!(
        !c.ingest.query(&q, 3, 16).sealed,
        "superseded session must not be sealed"
    );
}

/// F2-01: INDEX/JOURNAL segments are buffered whole; an ingest-level budget
/// caps that (on top of the signed table's per-segment sizes).
#[test]
fn index_buffer_budget_poisons_instead_of_buffering() {
    use rvf_runtime::progressive_stream::StreamIngest;
    let f = fixture();
    let keys = [f.public];
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    ing.set_buffer_budget(1000);
    let mut refused = None;
    for fr in frames(&f) {
        let out = ing.feed(&fr);
        if out.refused.is_some() {
            refused = out.refused;
            break;
        }
    }
    let _ = refused;
    assert!(matches!(
        ing.state(),
        StreamState::Poisoned(ErrorCode::StreamLimitExceeded)
    ));
    assert!(ing.vectors().is_empty() && !ing.layers().layer_a);
}

/// F2-03: routine witnesses are capped with a dropped count; terminal ones never.
#[test]
fn terminal_witnesses_survive_the_routine_cap() {
    use rvf_runtime::progressive_stream::StreamIngest;
    let f = fixture();
    let keys = [f.public];
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    let d_len = 5 + f.bundle.descriptor.len();
    ing.feed(&frames(&f).concat()[..d_len]);
    for _ in 0..3000 {
        ing.end_of_input();
        ing.resume();
    }
    assert!(ing.dropped_witnesses() > 0);
    ing.feed(&[0xEE]);
    assert!(matches!(
        ing.witnesses().last(),
        Some(StreamWitness::StreamPoisoned { .. })
    ));
}
