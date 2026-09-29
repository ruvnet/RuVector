//! Stepped finalize (`rvf_finalize_step`): large uploads are validated in
//! bounded steps inside the scope's DO, each `:finalize` answering `202`
//! with progress until the commit's `201`.

use crate::registry_kv::SqlKv;
use crate::registry_world::*;
use crate::rvf_finalize_step::k_cursor;
use ruvector_edge_registry::keys::registry_do_name;
use ruvector_edge_registry::ports::KvStore;
use ruvector_edge_registry::Scope;
use ruvector_edge_store::CallerContext;
use serde_json::Value as Json;
use worker::Method;

const PARTS: usize = 8;

/// A world where every multi-part upload is stepped, `per_step` parts of
/// `bytes` at a time.
fn world(bytes: &[u8], per_step: usize) -> (World, CallerContext) {
    let w = World::new();
    let alice = w.owner("org-a", "alice");
    assert_eq!(w.claim(&alice, "acme").0, 200);
    w.inline_finalize.set(0);
    let part = bytes.len().div_ceil(PARTS) as u64;
    w.r.step_bytes.set(part * per_step as u64);
    (w, alice)
}

fn fixture() -> Vec<u8> {
    rvf_export(300, 8, 11, &[]).0
}

fn do_name() -> String {
    registry_do_name(&Scope::parse("acme").unwrap())
}

fn upload(w: &World, c: &CallerContext, bytes: &[u8]) -> String {
    let (s, b) = w.begin(c, "acme/pkg", "1.0.0", "tenant", bytes);
    assert_eq!(s, 201, "{b}");
    let id = b["upload_id"].as_str().unwrap().to_string();
    w.parts(c, "acme/pkg", "1.0.0", &id, bytes, PARTS);
    id
}

fn fin(w: &World, c: &CallerContext, id: &str) -> (u16, Json) {
    w.finalize(c, "acme/pkg", "1.0.0", id)
}

/// Finalize until it stops answering `202`; returns the progress seen and
/// the final reply.
fn drive(w: &World, c: &CallerContext, id: &str) -> (Vec<Json>, (u16, Json)) {
    let mut seen = Vec::new();
    for _ in 0..64 {
        let r = fin(w, c, id);
        if r.0 != 202 {
            return (seen, r);
        }
        seen.push(r.1);
    }
    panic!("finalize never finished");
}

fn cursor(w: &World, id: &str) -> Option<Vec<u8>> {
    let scopes = w.r.scopes.borrow();
    let store = scopes.get(&do_name()).unwrap();
    SqlKv::open(store).unwrap().get(&k_cursor(id)).unwrap()
}

#[test]
fn large_upload_finalizes_in_bounded_steps_then_commits() {
    let bytes = fixture();
    let (w, alice) = world(&bytes, 2);
    let id = upload(&w, &alice, &bytes);
    let (seen, (s, m)) = drive(&w, &alice, &id);
    assert_eq!(s, 201, "{m}");
    assert_eq!(m["version"], "1.0.0");
    // 8 parts, 2 per step: three progress replies, then the commit.
    assert_eq!(seen.len(), 3, "{seen:?}");
    let done: Vec<u64> = seen
        .iter()
        .map(|p| p["bytes_validated"].as_u64().unwrap())
        .collect();
    assert!(done.windows(2).all(|p| p[0] < p[1]), "{done:?}");
    for p in &seen {
        assert_eq!(p["status"], "finalizing");
        assert_eq!(p["size"].as_u64().unwrap(), bytes.len() as u64);
        assert_eq!(p["restarts"].as_u64().unwrap(), 0);
    }
    // The blob is the uploaded bytes (R2-checked copy), staging and the
    // cursor are gone, no job is left in memory.
    assert_eq!(w.pull(&alice, "acme/pkg", "1.0.0").unwrap().0, bytes);
    assert!(w
        .s
        .objects
        .borrow()
        .keys()
        .all(|k| !k.starts_with("staging/")));
    assert!(cursor(&w, &id).is_none());
    assert_eq!(w.r.scope_jobs(&do_name()).live(), 0);
    // A retry after the commit answers the same version.
    let (s, again) = fin(&w, &alice, &id);
    assert_eq!((s, &again["sha256"]), (201, &m["sha256"]));
}

#[test]
fn evicted_object_restarts_validation_from_byte_zero() {
    let bytes = fixture();
    let (w, alice) = world(&bytes, 3);
    let id = upload(&w, &alice, &bytes);
    let (s, p) = fin(&w, &alice, &id);
    assert_eq!(s, 202, "{p}");
    assert!(cursor(&w, &id).is_some());
    // The DO lost its memory (the live validator) but kept the cursor.
    w.r.scope_jobs(&do_name()).evict();
    let (s, p2) = fin(&w, &alice, &id);
    assert_eq!(s, 202, "{p2}");
    assert_eq!(p2["restarts"].as_u64().unwrap(), 1);
    assert_eq!(p2["bytes_validated"], p["bytes_validated"]);
    let (_, (s, m)) = drive(&w, &alice, &id);
    assert_eq!(s, 201, "{m}");
    assert_eq!(w.pull(&alice, "acme/pkg", "1.0.0").unwrap().0, bytes);
}

#[test]
fn sweep_holding_the_object_defers_a_step_without_losing_progress() {
    let bytes = fixture();
    let (w, alice) = world(&bytes, 2);
    let id = upload(&w, &alice, &bytes);
    let (_, p) = fin(&w, &alice, &id);
    w.r.sweeping.set(true);
    let (s, _) = fin(&w, &alice, &id);
    assert_eq!(s, 503);
    w.r.sweeping.set(false);
    let (s, p2) = fin(&w, &alice, &id);
    assert_eq!(s, 202);
    assert_eq!(p2["restarts"].as_u64().unwrap(), 0);
    assert!(p2["bytes_validated"].as_u64() > p["bytes_validated"].as_u64());
    assert_eq!(drive(&w, &alice, &id).1 .0, 201);
}

#[test]
fn invalid_package_fails_the_session_at_the_first_bad_step() {
    // Declared correctly (size and SHA-256 match) but not an RVF file.
    let bytes: Vec<u8> = (0..8192u32).map(|i| (i * 31 % 251) as u8).collect();
    let (w, alice) = world(&bytes, 2);
    let id = upload(&w, &alice, &bytes);
    let (seen, (s, e)) = drive(&w, &alice, &id);
    assert!(seen.is_empty(), "stops at the first invalid part: {seen:?}");
    assert_eq!(
        (s, &e["detail"]),
        (400, &Json::from("bad magic at offset 0"))
    );
    assert!(cursor(&w, &id).is_none());
    // Terminal: the retry is refused too, nothing was published.
    assert_ne!(fin(&w, &alice, &id).0, 202);
    assert!(w.pull(&alice, "acme/pkg", "1.0.0").is_err());
}

#[test]
fn staging_changed_after_freeze_fails_as_parts_changed() {
    let bytes = fixture();
    let (w, alice) = world(&bytes, 2);
    *w.s.tamper_prefix.borrow_mut() = Some("staging/".into());
    let id = upload(&w, &alice, &bytes);
    let (_, (s, e)) = drive(&w, &alice, &id);
    assert_eq!(s, 400, "{e}");
    assert!(
        e["detail"].as_str().unwrap().contains("parts changed"),
        "{e}"
    );
    assert!(w.pull(&alice, "acme/pkg", "1.0.0").is_err());
}

#[test]
fn small_uploads_keep_the_one_request_finalize() {
    let bytes = fixture();
    let (w, alice) = world(&bytes, 1);
    w.inline_finalize.set(bytes.len() as u64);
    let id = upload(&w, &alice, &bytes);
    let (s, m) = fin(&w, &alice, &id);
    assert_eq!(s, 201, "{m}");
    assert_eq!(w.r.scope_jobs(&do_name()).live(), 0);
    // Single part: always inline, whatever the threshold.
    w.inline_finalize.set(0);
    let one = rvf_export(50, 8, 5, &[]).0;
    let (s, _) = w.push(&alice, "acme/one", "1.0.0", "tenant", &one, 1);
    assert_eq!(s, 201);
    let path = "/v1/rvf/acme/one/1.0.0";
    assert_eq!(w.json(&alice, Method::Get, path, Json::Null).0, 200);
}

#[test]
fn staging_removed_mid_job_is_a_retryable_storage_error() {
    let bytes = fixture();
    let (w, alice) = world(&bytes, 2);
    let id = upload(&w, &alice, &bytes);
    assert_eq!(fin(&w, &alice, &id).0, 202);
    // The staging object disappears under a live job (not a commit).
    w.s.objects
        .borrow_mut()
        .retain(|k, _| !k.starts_with("staging/"));
    let (s, e) = fin(&w, &alice, &id);
    assert_eq!(s, 503, "{e}");
    assert!(cursor(&w, &id).is_none());
    assert!(w.pull(&alice, "acme/pkg", "1.0.0").is_err());
}
