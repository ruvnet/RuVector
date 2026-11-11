//! RV2-01/02/03: pin store, completion re-check, identity re-verification.
mod common;
use common::*;
use rvf_types::ErrorCode as E;
use rvf_wire::stream::*;

fn pin(epoch: u64, h: u8) -> EpochPin {
    EpochPin {
        epoch,
        descriptor_hash: [h; 32],
    }
}

#[test]
fn memory_pin_store_is_a_never_backwards_compare_and_set() {
    let s = MemoryPinStore::new();
    let (sg, f) = ([1u8; 8], [2u8; 16]);
    assert_eq!(s.get(&sg, &f), None);
    assert!(s.advance(&sg, &f, None, pin(5, 1)).is_ok());
    // lost race: caller thought the pin was absent
    assert_eq!(
        s.advance(&sg, &f, None, pin(6, 2)),
        Err(PinConflict {
            current: Some(pin(5, 1))
        })
    );
    // backwards and equal-epoch-different-hash are refused even with the right expectation
    assert!(s.advance(&sg, &f, Some(pin(5, 1)), pin(4, 9)).is_err());
    assert!(s.advance(&sg, &f, Some(pin(5, 1)), pin(5, 9)).is_err());
    // idempotent re-record and forward moves work
    assert!(s.advance(&sg, &f, Some(pin(5, 1)), pin(5, 1)).is_ok());
    assert!(s.advance(&sg, &f, Some(pin(5, 1)), pin(7, 3)).is_ok());
    assert_eq!(s.get(&sg, &f), Some(pin(7, 3)));
    // keys are (signer, file): another signer or file is untouched
    assert_eq!(s.get(&[9u8; 8], &f), None);
    assert_eq!(s.get(&sg, &[3u8; 16]), None);
}

fn stream(seed: u8, epoch: u64, body: &[u8], stages: &[u8]) -> StreamBundle {
    StreamDescriptorBuilder::new(FILE_ID, key(seed))
        .epoch(epoch)
        .chunk_log2(12)
        .stages(stages)
        .build(body)
        .unwrap()
}

#[test]
fn a_second_signer_cannot_overwrite_or_raise_another_signers_pin() {
    let body = sample_body(1);
    let keys = [pk(1), pk(2)];
    let store = MemoryPinStore::new();
    let mut pol = StreamPolicy::new(&keys);
    pol.pins = Some(&store);

    // signer 2 (also trusted) completes a huge-epoch stream for the same file_id
    let b2 = stream(2, u64::MAX, &body, &[0]);
    let mut v2 = StreamVerifier::new(pol);
    for f in b2.frames_file_order(&body) {
        v2.push(&f).unwrap();
    }
    assert_eq!(v2.state(), StreamState::Complete);
    v2.commit_pin(&store).unwrap();
    assert_eq!(
        store
            .get(&rvf_crypto::signer_key_id(&pk(2)), &FILE_ID)
            .unwrap()
            .epoch,
        u64::MAX
    );

    // signer 1's lineage is unaffected and still verifies at a low epoch
    let b1 = stream(1, 3, &body, &[0]);
    let mut v1 = StreamVerifier::new(pol);
    for f in b1.frames_file_order(&body) {
        v1.push(&f).unwrap();
    }
    assert_eq!(v1.state(), StreamState::Complete);
    v1.commit_pin(&store).unwrap();
    assert_eq!(
        store
            .get(&rvf_crypto::signer_key_id(&pk(1)), &FILE_ID)
            .unwrap()
            .epoch,
        3
    );
    // commit_pin is only available for a finished, identity-confirmed stream
    let mut early = StreamVerifier::new(pol);
    early
        .push(&b1.frames_file_order(&body).next().unwrap())
        .unwrap();
    assert_eq!(early.commit_pin(&store), Err(E::StreamBadState));
}

#[test]
fn a_session_superseded_while_running_ends_stale_not_verified() {
    let body = sample_body(1);
    let keys = [pk(1)];
    let store = MemoryPinStore::new();
    let mut pol = StreamPolicy::new(&keys);
    pol.pins = Some(&store);
    let old = stream(1, 5, &body, &[0]);
    let newer = stream(1, 6, &body, &[0]);

    let frames: Vec<Vec<u8>> = old.frames_file_order(&body).collect();
    let mut v = StreamVerifier::new(pol); // admitted: store empty
    let (last, rest) = frames.split_last().unwrap();
    for f in rest {
        v.push(f).unwrap();
    }
    // meanwhile a newer epoch completes elsewhere and is recorded
    let mut other = StreamVerifier::new(pol);
    for f in newer.frames_file_order(&body) {
        other.push(&f).unwrap();
    }
    other.commit_pin(&store).unwrap();
    // the old session's final chunk must not produce Complete
    assert_eq!(v.push(last).err(), Some(E::StreamStaleEpoch));
    assert_eq!(v.state(), StreamState::Poisoned(E::StreamStaleEpoch));
    assert_eq!(v.commit_pin(&store), Err(E::StreamBadState));
}

#[test]
fn recheck_pin_and_commit_cas_detect_supersession_after_completion() {
    let body = sample_body(1);
    let keys = [pk(1)];
    let store = MemoryPinStore::new();
    let mut pol = StreamPolicy::new(&keys);
    pol.pins = Some(&store);
    let old = stream(1, 5, &body, &[0]);
    let mut v = StreamVerifier::new(pol);
    for f in old.frames_file_order(&body) {
        v.push(&f).unwrap();
    }
    assert_eq!(v.state(), StreamState::Complete);
    assert!(v.recheck_pin(&store).is_ok());
    store
        .advance(
            &rvf_crypto::signer_key_id(&pk(1)),
            &FILE_ID,
            None,
            pin(9, 7),
        )
        .unwrap();
    // state is untouched, but consumers get the typed error before publishing
    assert_eq!(v.recheck_pin(&store), Err(E::StreamStaleEpoch));
    assert_eq!(v.state(), StreamState::Complete);
    assert_eq!(v.commit_pin(&store), Err(E::StreamStaleEpoch));
    // equal epoch, different descriptor
    let b = stream(1, 9, &body, &[0]);
    let mut w = StreamVerifier::new(StreamPolicy::new(&keys));
    for f in b.frames_file_order(&body) {
        w.push(&f).unwrap();
    }
    assert_eq!(w.recheck_pin(&store), Err(E::StreamEpochConflict));
}

#[test]
fn stale_session_cannot_become_complete_through_finish_identity() {
    let body = sample_body(1);
    let keys = [pk(1)];
    let store = MemoryPinStore::new();
    let mut pol = StreamPolicy::new(&keys);
    pol.pins = Some(&store);
    let old = stream(1, 5, &body, &[3, 0, 2, 1, 0, 0, 0]);
    let mut v = StreamVerifier::new(pol);
    for f in old.frames_stage_order(&body) {
        v.push(&f).unwrap();
    }
    assert_eq!(v.state(), StreamState::CompleteUnverified);
    store
        .advance(
            &rvf_crypto::signer_key_id(&pk(1)),
            &FILE_ID,
            None,
            pin(6, 1),
        )
        .unwrap();
    let r = v.finish_identity(|o, b| {
        b.copy_from_slice(&body[o as usize..o as usize + b.len()]);
        true
    });
    assert_eq!(r, Err(E::StreamStaleEpoch));
    assert_eq!(v.state(), StreamState::Poisoned(E::StreamStaleEpoch));
}

/// RV2-03: a descriptor (lying signer) claims another body's SHA-256 as identity and the
/// host supplies that other body. The old identity-only check would have passed.
#[test]
fn finish_identity_reverifies_the_persisted_body_against_the_commitments() {
    let body_a = sample_body(1);
    let mut body_b = body_a.clone();
    body_b[5000] ^= 0x40; // same length, different content
    let keys = [pk(1)];
    let honest = stream(1, 1, &body_a, &[1, 1, 0, 0, 0, 0, 0]);
    // lying descriptor: commitments of A, identity of B
    let lying = resign(&honest, &key(1), |_, h| {
        h.rvf_identity = rvf_types::sha256(&body_b)
    });
    let mut v = StreamVerifier::new(StreamPolicy::new(&keys));
    v.push(&descriptor_frame(&lying)).unwrap();
    for f in honest.frames_stage_order(&body_a).skip(1) {
        v.push(&f).unwrap();
    }
    assert_eq!(v.state(), StreamState::CompleteUnverified);
    let mut reads = 0usize;
    let r = v.finish_identity(|o, b| {
        reads += 1;
        b.copy_from_slice(&body_b[o as usize..o as usize + b.len()]);
        true
    });
    assert_eq!(r, Err(E::StreamChunkHashMismatch));
    assert_eq!(v.state(), StreamState::Poisoned(E::StreamChunkHashMismatch));
    assert!(reads > 0);

    // the same lying descriptor with the true body A: chunks pass, identity does not
    let mut v = StreamVerifier::new(StreamPolicy::new(&keys));
    v.push(&descriptor_frame(&lying)).unwrap();
    for f in honest.frames_stage_order(&body_a).skip(1) {
        v.push(&f).unwrap();
    }
    let r = v.finish_identity(|o, b| {
        b.copy_from_slice(&body_a[o as usize..o as usize + b.len()]);
        true
    });
    assert_eq!(r, Err(E::StreamIdentityMismatch));

    // honest descriptor and body: Complete, read in bounded pieces
    let mut v = StreamVerifier::new(StreamPolicy::new(&keys));
    for f in honest.frames_stage_order(&body_a) {
        v.push(&f).unwrap();
    }
    let mut max_read = 0usize;
    let r = v.finish_identity(|o, b| {
        max_read = max_read.max(b.len());
        b.copy_from_slice(&body_a[o as usize..o as usize + b.len()]);
        true
    });
    assert_eq!(r, Ok(()));
    assert_eq!(v.state(), StreamState::Complete);
    assert!(max_read <= 65536);
}

/// F2-09: the default table cap is derived from the default body limit, so a
/// legitimate body at the limit (4 KiB chunks, the smallest) cannot exceed it.
#[test]
fn default_table_cap_covers_the_default_body_limit() {
    use rvf_wire::stream::StreamPolicy;
    let keys: [[u8; 32]; 0] = [];
    let p = StreamPolicy::new(&keys);
    let worst_tables = p.max_body_len / 4096 * 32;
    assert!(
        p.max_table_bytes >= worst_tables,
        "{} < {worst_tables}",
        p.max_table_bytes
    );
}
