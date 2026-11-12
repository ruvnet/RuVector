//! Regression tests for the Worker upload flows: per-attempt R2 part
//! numbers, idempotent finalize, bounded finalize work and begin cleanup.

use crate::registry_ports::{scope, BlobStore};
use crate::registry_routes::{registry_caller, RvfReply};
use crate::registry_upload_core::ATTEMPTS_PER_PART;
use crate::registry_wire::{CallerWire, Coords, ScopeCall, ScopeOut};
use crate::registry_world::*;
use crate::testkit::block_on;
use ruvector_edge_registry::upload::PartRecord;
use ruvector_edge_registry::Scope;
use ruvector_edge_store::CallerContext;
use worker::Method;

fn fixture() -> Vec<u8> {
    rvf_export(40, 8, 7, &[]).0
}

fn world() -> (World, CallerContext) {
    let w = World::new();
    let alice = w.owner("org-a", "alice");
    assert_eq!(w.claim(&alice, "acme").0, 200);
    (w, alice)
}

fn at() -> Coords {
    Coords {
        name: "@acme/pkg".into(),
        version: "1.0.0".into(),
    }
}

fn acme() -> Scope {
    Scope::parse("acme").unwrap()
}

fn begin(w: &World, c: &CallerContext, bytes: &[u8]) -> String {
    let (s, b) = w.begin(c, "acme/pkg", "1.0.0", "tenant", bytes);
    assert_eq!(s, 201, "{b}");
    b["upload_id"].as_str().unwrap().to_string()
}

fn target(w: &World, cw: &CallerWire, id: &str) -> Result<(String, String, u16), u16> {
    let call = ScopeCall::PartTarget {
        caller: cw.clone(),
        at: at(),
        upload_id: id.into(),
        number: 1,
    };
    match block_on(scope(&w.r, &acme(), call)) {
        Ok(ScopeOut::PartTarget {
            staging,
            multipart,
            r2_part,
        }) => Ok((staging, multipart, r2_part)),
        Ok(_) => panic!("unexpected reply"),
        Err(e) => Err(e.status),
    }
}

fn record(w: &World, cw: &CallerWire, id: &str, bytes: &[u8], r2: u16, etag: String) -> u16 {
    let part = PartRecord {
        number: 1,
        size: bytes.len() as u64,
        sha256: sha(bytes),
    };
    let call = ScopeCall::RecordPart {
        caller: cw.clone(),
        at: at(),
        upload_id: id.into(),
        part,
        r2_part: r2,
        etag,
    };
    block_on(scope(&w.r, &acme(), call)).map_or_else(|e| e.status, |_| 200)
}

/// F6: two uploads of one part racing (a client retry) reach R2 and the
/// index in opposite orders; each attempt has its own R2 part number, so
/// the recorded attempt is always the bytes R2 completes with.
#[test]
fn reordered_part_retries_still_finalize_the_recorded_bytes() {
    let (w, alice) = world();
    let cw = CallerWire::from(&block_on(registry_caller(&w.b, &alice)).unwrap());
    let bytes = fixture();
    let id = begin(&w, &alice, &bytes);
    let (p1, p2) = bytes.split_at(bytes.len().div_ceil(2));
    let path = format!("/v1/rvf/acme/pkg/1.0.0/uploads/{id}/parts/2");
    assert!(matches!(w.call(&alice, Method::Put, &path, p2), RvfReply::Api(r) if r.status == 200));
    let (staging, mp, ra) = target(&w, &cw, &id).unwrap();
    let (_, _, rb) = target(&w, &cw, &id).unwrap();
    assert_ne!(ra, rb, "each attempt gets its own R2 part number");
    // R2 takes attempt A (the right bytes), then B (other bytes)...
    let ea = block_on(w.s.mp_part(&staging, &mp, ra, p1.to_vec())).unwrap();
    let junk = vec![0x5a; p1.len()];
    let eb = block_on(w.s.mp_part(&staging, &mp, rb, junk.clone())).unwrap();
    // ...while the index records B first and A last.
    assert_eq!(record(&w, &cw, &id, &junk, rb, eb), 200);
    assert_eq!(record(&w, &cw, &id, p1, ra, ea), 200);
    let (s, m) = w.finalize(&alice, "acme/pkg", "1.0.0", &id);
    assert_eq!(s, 201, "{m}");
    assert_eq!(w.pull(&alice, "acme/pkg", "1.0.0"), Ok((bytes, false)));
}

#[test]
fn part_attempts_are_bounded_and_must_have_been_handed_out() {
    let (w, alice) = world();
    let cw = CallerWire::from(&block_on(registry_caller(&w.b, &alice)).unwrap());
    let id = begin(&w, &alice, &fixture());
    let mut seen = Vec::new();
    for _ in 0..ATTEMPTS_PER_PART {
        seen.push(target(&w, &cw, &id).unwrap().2);
    }
    seen.dedup();
    assert_eq!(seen.len(), ATTEMPTS_PER_PART as usize);
    assert_eq!(target(&w, &cw, &id), Err(409));
    // An R2 part number this session never handed out for part 1.
    let stray = seen.iter().max().unwrap() + 1;
    assert_eq!(record(&w, &cw, &id, b"x", stray, "e".into()), 400);
}

/// F7: a finalize retried after it committed answers with the version.
#[test]
fn a_retried_finalize_returns_the_committed_version() {
    let (w, alice) = world();
    let bytes = fixture();
    let id = begin(&w, &alice, &bytes);
    w.parts(&alice, "acme/pkg", "1.0.0", &id, &bytes, 2);
    let (s, first) = w.finalize(&alice, "acme/pkg", "1.0.0", &id);
    assert_eq!(s, 201, "{first}");
    let (s, again) = w.finalize(&alice, "acme/pkg", "1.0.0", &id);
    assert_eq!((s, &again), (201, &first));
}

/// F8: reading stops at the first invalid part; a valid single-part push
/// takes R2's put result as evidence instead of re-reading the blob.
#[test]
fn finalize_stops_at_the_first_invalid_part_and_skips_rereads() {
    let (w, alice) = world();
    let junk = vec![0xab; 3000];
    let id = begin(&w, &alice, &junk);
    w.parts(&alice, "acme/pkg", "1.0.0", &id, &junk, 3);
    w.s.reads.set((0, 0));
    let (s, b) = w.finalize(&alice, "acme/pkg", "1.0.0", &id);
    assert_eq!(s, 400, "{b}");
    assert!(
        !b["detail"].as_str().unwrap().contains("parts changed"),
        "{b}"
    );
    assert_eq!(w.s.reads.get(), (1, 0), "one part read, no measure");
    assert_eq!(w.s.open_uploads(), 0, "blob multipart aborted");
    assert!(w.s.objects.borrow().keys().all(|k| !k.starts_with("rvf/")));
    let bytes = fixture();
    w.s.reads.set((0, 0));
    assert_eq!(
        w.push(&alice, "acme/pkg", "2.0.0", "tenant", &bytes, 1).0,
        201
    );
    assert_eq!(w.s.reads.get(), (1, 0), "single part: no re-measure");
    assert_eq!(w.pull(&alice, "acme/pkg", "2.0.0"), Ok((bytes, false)));
}

/// F10: a begin whose attach fails aborts the multipart it created.
#[test]
fn begin_aborts_its_multipart_when_attach_fails() {
    let (w, alice) = world();
    let bytes = fixture();
    // The next scope call (Begin) passes, the one after (Attach) fails.
    w.r.fail_scope_call_in.set(Some(1));
    let (s, b) = w.begin(&alice, "acme/pkg", "1.0.0", "tenant", &bytes);
    assert_eq!((s, b["code"].as_str()), (503, Some("shard_unavailable")));
    assert_eq!(w.s.open_uploads(), 0);
    assert_eq!(w.s.aborted.borrow().len(), 1);
}
