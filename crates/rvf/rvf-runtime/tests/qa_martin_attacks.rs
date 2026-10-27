//! Extra adversarial cases from independent QA (Martin, 2026-10-27).
#[path = "support/stream_fixture.rs"]
mod fx;
use fx::*;
use rvf_runtime::progressive_stream::StreamIngest;
use rvf_wire::stream::{StreamPolicy, StreamState};

fn rng(seed: u64) -> impl FnMut() -> u64 { let mut x = seed; move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x } }
fn complete(s: StreamState) -> bool { matches!(s, StreamState::Complete | StreamState::CompleteUnverified) }

/// Feed everything in the given pieces; returns the ingest after end_of_input.
fn feed_pieces<'p>(policy: StreamPolicy<'p>, bytes: &[u8], mut piece: impl FnMut() -> usize) -> StreamIngest<'p> {
    let mut ing = StreamIngest::new(policy);
    let mut at = 0;
    while at < bytes.len() {
        let n = piece().max(1).min(bytes.len() - at);
        let out = ing.feed(&bytes[at..at + n]);
        if out.refused.is_some() { break; }
        at += n;
    }
    ing.end_of_input();
    ing
}

#[test]
fn random_split_points_complete_with_every_vector() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    for seed in [1u64, 2, 3] {
        let mut r = rng(seed);
        let ing = feed_pieces(StreamPolicy::new(&keys), &full, || (r() % 3000) as usize);
        assert!(complete(ing.state()), "seed {seed}: {:?}", ing.state());
        assert_eq!(ing.vectors().len(), f.ds.len());
        assert_authentic(&ing, &f.ds);
    }
    let ing = feed_pieces(StreamPolicy::new(&keys), &full, || 1);
    assert!(complete(ing.state()), "byte-at-a-time: {:?}", ing.state());
    assert_eq!(ing.vectors().len(), f.ds.len());
}

#[test]
fn hundreds_of_random_single_bit_flips_never_complete_or_publish_fakes() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    let mut r = rng(0xBEEF);
    for _ in 0..400 {
        let mut b = full.clone();
        let at = (r() as usize) % b.len();
        b[at] ^= 1 << (r() % 8);
        let ing = feed_pieces(StreamPolicy::new(&keys), &b, || 4096);
        assert_authentic(&ing, &f.ds);
        assert!(!complete(ing.state()), "flip at byte {at} still reached {:?}", ing.state());
    }
}

#[test]
fn wrong_key_and_no_keys_publish_nothing() {
    let f = fixture();
    let full = frames(&f).concat();
    let other = [[0x11u8; 32]];
    let ing = feed_pieces(StreamPolicy::new(&other), &full, || 4096);
    assert!(!complete(ing.state()));
    assert!(ing.vectors().is_empty(), "untrusted signer published {} vectors", ing.vectors().len());
    let none: [[u8; 32]; 0] = [];
    let ing = feed_pieces(StreamPolicy::new(&none), &full, || 4096);
    assert!(!complete(ing.state()));
    assert!(ing.vectors().is_empty());
}

#[test]
fn garbage_before_the_descriptor_publishes_nothing() {
    let f = fixture();
    let keys = [f.public];
    let mut b: Vec<u8> = (0..37u8).collect();
    b.extend(frames(&f).concat());
    let ing = feed_pieces(StreamPolicy::new(&keys), &b, || 4096);
    assert!(!complete(ing.state()));
    assert!(ing.vectors().is_empty());
}

#[test]
fn a_missing_middle_frame_never_completes() {
    let f = fixture();
    let keys = [f.public];
    let mut fr = frames(&f);
    let mid = fr.len() / 2;
    fr.remove(mid);
    let ing = feed_pieces(StreamPolicy::new(&keys), &fr.concat(), || 4096);
    assert_authentic(&ing, &f.ds);
    assert!(!complete(ing.state()), "{:?}", ing.state());
}

#[test]
fn truncate_suspend_then_resume_from_the_reported_offset_completes() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    for cut in [full.len() / 3, full.len() / 2 + 7, full.len() - 3] {
        let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
        let out = ing.feed(&full[..cut]);
        assert!(out.refused.is_none());
        assert_eq!(ing.end_of_input(), StreamState::Suspended, "cut {cut}");
        assert!(ing.resume(), "resume refused at cut {cut}");
        let from = ing.offset() as usize;
        assert!(from <= cut, "resume offset {from} past the cut {cut}");
        ing.feed(&full[from..]);
        ing.end_of_input();
        assert!(complete(ing.state()), "cut {cut} resumed at {from}: {:?}", ing.state());
        assert_eq!(ing.vectors().len(), f.ds.len());
        assert_authentic(&ing, &f.ds);
    }
}

#[test]
fn mid_download_answers_are_unsealed_and_only_verified_ids() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    let q = f.ds.vector(7).to_vec();
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    ing.feed(&full[..full.len() / 2]);
    let a = ing.query(&q, 10, 64);
    assert!(!a.sealed);
    assert!(a.coverage_vectors < 1.0 && a.coverage_bytes < 1.0, "{} {}", a.coverage_vectors, a.coverage_bytes);
    let visible: std::collections::HashSet<u64> = ing.vectors().iter().map(|(id, _)| id).collect();
    for r in &a.envelope.results { assert!(visible.contains(&r.id), "answer id {} is not a verified vector", r.id); }
    ing.feed(&full[full.len() / 2..]);
    ing.end_of_input();
    // Stage-order delivery ends CompleteUnverified: answers stay unsealed until the identity is settled.
    assert_eq!(ing.state(), StreamState::CompleteUnverified);
    assert!(!ing.query(&q, 10, 64).sealed);
    let body = f.body.clone();
    ing.finish_identity(|off, buf| { let o = off as usize; if o + buf.len() > body.len() { return false; } buf.copy_from_slice(&body[o..o + buf.len()]); true }).unwrap();
    let a = ing.query(&q, 10, 64);
    assert!(a.sealed, "complete stream must seal; state {:?}", ing.state());
    eprintln!("DEBUG visible={} vectors={} ds={} cov_v={} cov_b={} path={:?} results={}", a.visible_vectors, ing.vectors().len(), f.ds.len(), a.coverage_vectors, a.coverage_bytes, a.path, a.envelope.results.len());
    assert_eq!(a.coverage_vectors, 1.0, "docs: 1.0 once all arrived");
}

#[test]
fn identity_settlement_over_a_tampered_persisted_body_does_not_seal() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    let mut r = rng(9);
    for _ in 0..20 {
        let ing0 = feed_pieces(StreamPolicy::new(&keys), &full, || 4096);
        assert_eq!(ing0.state(), StreamState::CompleteUnverified);
        let mut ing = ing0;
        let mut body = f.body.clone();
        let at = (r() as usize) % body.len();
        body[at] ^= 1 << (r() % 8);
        let res = ing.finish_identity(|off, buf| { let o = off as usize; if o + buf.len() > body.len() { return false; } buf.copy_from_slice(&body[o..o + buf.len()]); true });
        assert!(res.is_err() || ing.state() != StreamState::Complete, "tampered byte {at} in the persisted body still settled identity");
        assert!(!ing.query(&f.ds.vector(1).to_vec(), 5, 32).sealed, "tampered byte {at}: answer sealed");
    }
    // A short read is refused too.
    let mut ing = feed_pieces(StreamPolicy::new(&keys), &full, || 4096);
    assert!(ing.finish_identity(|_, _| false).is_err() || ing.state() != StreamState::Complete);
}

/// An index segment that overruns the ingest buffer budget must poison the stream ("Exceeding it poisons
/// the stream"). If that segment's chunk is the last one delivered, the verifier is already Complete,
/// poison() is a no-op, and the stream seals with the layer silently dropped.
#[test]
fn budget_overrun_on_the_final_segment_still_poisons() {
    let mut found = None;
    'search: for n in [2600usize, 6000, 12000] { for dim in [4usize, 8, 16, 64] { for clusters in [1usize, 4, 32] {
        let f = fixture_with(n, dim, clusters, 11, 1);
        let e = &f.bundle.entries;
        let last = e.iter().map(|x| x.stage).max().unwrap();
        let Some(t) = (0..e.len()).filter(|&i| e[i].seg_type == 0x02 && e[i].stage == last).max_by_key(|&i| e[i].stored_len) else { continue };
        let others = (0..e.len()).filter(|&i| i != t && matches!(e[i].seg_type, 0x02 | 0x04)).map(|i| e[i].stored_len).max().unwrap_or(0);
        if e[t].stored_len > others { eprintln!("fixture n={n} dim={dim} clusters={clusters}: final-stage index {} > others {}", e[t].stored_len, others); found = Some((f, t, others)); break 'search; }
    }}}
    let Some((f, target, _others)) = found else { panic!("no fixture shape with a final-stage index segment larger than the others") };
    let keys = [f.public];
    let e = &f.bundle.entries;
    // One byte under the segment: the overrun lands on its final chunk, the stream's last byte.
    let budget = e[target].stored_len - 1;
    let mut order: Vec<usize> = (0..e.len()).collect();
    order.sort_by_key(|&i| (e[i].stage, i));
    let p = order.iter().position(|&i| i == target).unwrap();
    order.remove(p);
    order.push(target);
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    ing.set_buffer_budget(budget);
    for fr in f.bundle.frames_in_order(&f.body, order) { if ing.feed(&fr).refused.is_some() { break; } }
    ing.end_of_input();
    eprintln!("after the stream: state {:?} layers {:?}", ing.state(), ing.layers());
    let body = f.body.clone();
    let _ = ing.finish_identity(|off, buf| { let o = off as usize; if o + buf.len() > body.len() { return false; } buf.copy_from_slice(&body[o..o + buf.len()]); true });
    let a = ing.query(&f.ds.vector(0).to_vec(), 5, 32);
    eprintln!("after finish_identity: state {:?} sealed {} layers {:?}", ing.state(), a.sealed, ing.layers());
    assert!(matches!(ing.state(), StreamState::Poisoned(_)), "budget overrun on the last segment did not poison: {:?}", ing.state());
    assert!(!a.sealed, "sealed with an index layer dropped by an ignored poison");
}

/// Corrupts one byte on the first fetch only, then serves clean bytes (a transient network fault).
struct FlakyOnce { bytes: Vec<u8>, at: usize, done: bool }
impl rvf_runtime::progressive_stream::client::Transport for FlakyOnce {
    fn fetch(&mut self, offset: u64, sink: &mut dyn FnMut(&[u8]) -> bool) -> std::io::Result<rvf_runtime::progressive_stream::client::FetchEnd> {
        let mut b = self.bytes.clone();
        if !self.done { b[self.at] ^= 0x10; self.done = true; }
        let mut at = offset as usize;
        while at < b.len() {
            let z = (at + 16 * 1024).min(b.len());
            if !sink(&b[at..z]) { return Ok(rvf_runtime::progressive_stream::client::FetchEnd::Aborted); }
            at = z;
        }
        Ok(rvf_runtime::progressive_stream::client::FetchEnd::Eof)
    }
}

/// A refused frame that began in an earlier 16 KiB slice cuts the log back with set_len, which does not
/// move the write cursor: the next bytes landed past a hole of zeros and recover() stopped there.
#[test]
fn the_frame_log_stays_an_exact_prefix_after_a_refused_frame_across_slices() {
    use rvf_runtime::progressive_stream::client::{recover, ClientConfig, StreamClient};
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("frames.log");
    // A byte just past the first slice boundary, inside a chunk frame that started before it.
    let t = FlakyOnce { bytes: full.clone(), at: 16 * 1024 + 8, done: false };
    let mut c = StreamClient::new(StreamPolicy::new(&keys), t, ClientConfig::default()).with_log(&log).unwrap();
    let end = c.run(&mut |_| {});
    assert!(end.all_verified(), "{end:?}");
    let written = std::fs::read(&log).unwrap();
    assert_eq!(written.len(), full.len(), "log length differs from the stream");
    assert!(written == full, "log is not the exact stream (first difference at {:?})", written.iter().zip(&full).position(|(a, b)| a != b));
    let (r, _) = recover(StreamPolicy::new(&keys), &log).unwrap();
    assert!(complete(r.state()), "recover() from the log: {:?}", r.state());
}
