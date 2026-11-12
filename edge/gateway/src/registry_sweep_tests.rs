//! Regression tests for the `RegistryScope` sweep: the durable R2 queue,
//! the per-alarm operation budget, the sweep cursor, the session cap and
//! the sweep gate.

use crate::registry_kv::SqlKv;
use crate::registry_pending::{PA, PD};
use crate::registry_ports::OPS_PER_ALARM;
use crate::registry_routes::RvfReply;
use crate::registry_sweep::{sweep_scope, SweepGate, K_CURSOR, SWEEP_DEADLINE_MS};
use crate::registry_upload_core::MAX_SESSIONS_PER_SCOPE;
use crate::registry_world::*;
use ruvector_edge_registry::keys::registry_do_name;
use ruvector_edge_registry::mem::CounterEntropy;
use ruvector_edge_registry::ports::KvStore;
use ruvector_edge_registry::Scope;
use ruvector_edge_store::{CallerContext, MemSqlStore};
use serde_json::Value as Json;
use std::cell::Cell;
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

/// Run `f` on `scope`'s live index.
fn with_index<T>(w: &World, scope: &str, f: impl FnOnce(&MemSqlStore) -> T) -> T {
    let name = registry_do_name(&Scope::parse(scope).unwrap());
    let mut all = w.r.scopes.borrow_mut();
    f(all.entry(name).or_default())
}

fn queued(w: &World) -> usize {
    with_index(w, "acme", |s| {
        let kv = SqlKv::open(s).unwrap();
        kv.list(PA, None, 100_000).unwrap().len() + kv.list(PD, None, 100_000).unwrap().len()
    })
}

fn tenant_blob(w: &World) -> String {
    w.s.objects
        .borrow()
        .keys()
        .find(|k| k.starts_with("rvf/"))
        .cloned()
        .unwrap()
}

/// Begin a session and upload its first part (left open).
fn stale_session(w: &World, c: &CallerContext, pkg: &str, bytes: &[u8]) -> String {
    let (s, b) = w.begin(c, pkg, "1.0.0", "tenant", bytes);
    assert_eq!(s, 201, "{b}");
    let id = b["upload_id"].as_str().unwrap().to_string();
    let path = format!("/v1/rvf/{pkg}/1.0.0/uploads/{id}/parts/1");
    let half = &bytes[..bytes.len().div_ceil(2)];
    assert!(matches!(w.call(c, Method::Put, &path, half), RvfReply::Api(r) if r.status == 200));
    id
}

/// F1/F4: an R2 failure during the sweep leaves every delete and abort
/// queued (staging objects included); the next alarm completes them.
#[test]
fn failed_sweep_deletes_and_aborts_stay_queued_until_r2_confirms() {
    let (w, alice) = world();
    let bytes = fixture();
    let id = stale_session(&w, &alice, "acme/stale", &bytes);
    // The finalize's own best-effort staging delete fails too.
    w.s.fail_sweep_ops.set(true);
    assert_eq!(
        w.push(&alice, "acme/pkg", "1.0.0", "tenant", &bytes, 2).0,
        201
    );
    let staging: Vec<String> =
        w.s.objects
            .borrow()
            .keys()
            .filter(|k| k.starts_with("staging/"))
            .cloned()
            .collect();
    assert_eq!(staging.len(), 1, "finalized staging object left behind");
    let blob = tenant_blob(&w);
    let (s, _) = w.json(
        &alice,
        Method::Post,
        "/v1/rvf/acme/pkg/1.0.0:publish",
        Json::Null,
    );
    assert_eq!(s, 200);
    w.r.clock.advance(24 * 3600 + 1);
    let work = w.r.alarm("acme", &w.s);
    assert!(work.delete.is_empty() && work.abort.is_empty(), "{work:?}");
    assert_eq!(work.expired, vec![id.clone()]);
    assert!(work.remaining);
    assert!(
        queued(&w) >= 4,
        "abort + staging deletes + blob delete queued"
    );
    assert!(w.s.get(&staging[0]).is_some() && w.s.get(&blob).is_some());
    assert_eq!(w.s.open_uploads(), 1);
    // R2 recovers: the next alarm drains the queue first.
    w.s.fail_sweep_ops.set(false);
    let work = w.r.alarm("acme", &w.s);
    assert!(!work.remaining, "{work:?}");
    assert_eq!(queued(&w), 0);
    assert!(w.s.get(&staging[0]).is_none(), "staging object reclaimed");
    assert!(w.s.get(&blob).is_none(), "released blob reclaimed");
    assert_eq!(w.s.open_uploads(), 0, "stale multipart aborted");
    assert!(
        w.pull(&alice, "acme/pkg", "1.0.0").is_ok(),
        "public copy stays"
    );
}

/// The queue's safety check: a released blob pinned again (a new version
/// of the same bytes) after its delete was queued is kept.
#[test]
fn a_queued_blob_delete_skips_a_blob_pinned_again() {
    let (w, alice) = world();
    let bytes = fixture();
    assert_eq!(
        w.push(&alice, "acme/pkg", "1.0.0", "tenant", &bytes, 2).0,
        201
    );
    let blob = tenant_blob(&w);
    let (s, _) = w.json(
        &alice,
        Method::Post,
        "/v1/rvf/acme/pkg/1.0.0:publish",
        Json::Null,
    );
    assert_eq!(s, 200);
    w.s.fail_sweep_ops.set(true);
    let work = w.r.alarm("acme", &w.s);
    assert!(work.remaining && work.delete.is_empty(), "{work:?}");
    assert_eq!(
        w.push(&alice, "acme/pkg", "2.0.0", "tenant", &bytes, 2).0,
        201
    );
    w.s.fail_sweep_ops.set(false);
    let work = w.r.alarm("acme", &w.s);
    assert!(!work.delete.contains(&blob), "{work:?}");
    assert_eq!(queued(&w), 0);
    assert!(w.s.get(&blob).is_some(), "re-pinned blob kept");
    assert_eq!(w.pull(&alice, "acme/pkg", "2.0.0"), Ok((bytes, false)));
}

/// F4: an alarm that stops half way (deadline, limits, eviction) loses
/// nothing: the index already queued every key.
#[test]
fn an_alarm_cut_short_leaves_the_rest_queued() {
    let (w, alice) = world();
    let bytes = fixture();
    for i in 0..5 {
        stale_session(&w, &alice, &format!("acme/s{i}"), &bytes);
    }
    w.r.clock.advance(24 * 3600 + 1);
    let chunks = Cell::new(0);
    let work = w.r.alarm_while("acme", &w.s, &|| {
        chunks.set(chunks.get() + 1);
        chunks.get() <= 2
    });
    assert_eq!(work.expired.len(), 5);
    assert!(work.remaining && queued(&w) > 0);
    let work = w.r.alarm("acme", &w.s);
    assert!(!work.remaining);
    assert_eq!(queued(&w), 0);
    assert_eq!(w.s.open_uploads(), 0);
    assert_eq!(w.s.aborted.borrow().len(), 5);
}

/// F5 + F9: one alarm issues at most `OPS_PER_ALARM` R2 operations and
/// re-arms while work is left; the cursor walks every session; a scope
/// caps its sessions and frees them once swept.
#[test]
fn sweeps_are_budgeted_resume_from_a_cursor_and_sessions_are_capped() {
    let (w, alice) = world();
    let bytes = fixture();
    for i in 0..MAX_SESSIONS_PER_SCOPE {
        let (s, b) = w.begin(&alice, &format!("acme/p{i}"), "1.0.0", "tenant", &bytes);
        assert_eq!(s, 201, "{b}");
    }
    assert_eq!(w.s.open_uploads(), MAX_SESSIONS_PER_SCOPE);
    let (s, b) = w.begin(&alice, "acme/one-more", "1.0.0", "tenant", &bytes);
    assert_eq!((s, b["code"].as_str()), (429, Some("rate_limited")));
    w.r.clock.advance(24 * 3600 + 1);
    let mut alarms = 0;
    loop {
        let before = w.s.sweep_ops.get();
        let work = w.r.alarm("acme", &w.s);
        assert!(w.s.sweep_ops.get() - before <= OPS_PER_ALARM);
        alarms += 1;
        let rows = with_index(&w, "acme", |s| {
            SqlKv::open(s)
                .unwrap()
                .count_keys("upload/", 10_000)
                .unwrap()
        });
        if !work.remaining && rows == 0 {
            break;
        }
        assert!(alarms < 100, "sweep does not converge");
    }
    assert!(alarms > 1, "the work took several budgeted alarms");
    assert_eq!(w.s.open_uploads(), 0);
    assert_eq!(w.s.aborted.borrow().len(), MAX_SESSIONS_PER_SCOPE);
    assert_eq!(
        w.begin(&alice, "acme/one-more", "1.0.0", "tenant", &bytes)
            .0,
        201
    );
}

/// F9: the walk over `upload/` resumes where the last alarm stopped and
/// wraps at the end, so sessions late in key order are reached.
#[test]
fn sweep_cursor_is_persisted_and_wraps() {
    let (w, alice) = world();
    let bytes = fixture();
    let ids: Vec<String> = (0..4)
        .map(|i| stale_session(&w, &alice, &format!("acme/c{i}"), &bytes))
        .collect();
    w.r.clock.advance(24 * 3600 + 1);
    let mut seen = Vec::new();
    for _ in 0..4 {
        let expired = with_index(&w, "acme", |s| {
            sweep_scope(s, &w.r.clock, &CounterEntropy::default(), w.r.cfg(), 1, 1).unwrap()
        });
        assert_eq!(expired.len(), 1, "one page per pass");
        seen.extend(expired);
    }
    seen.sort();
    let mut want = ids.clone();
    want.sort();
    assert_eq!(seen, want, "every session reached once");
    let cursor = || {
        with_index(&w, "acme", |s| {
            SqlKv::open(s).unwrap().get(K_CURSOR).unwrap()
        })
    };
    assert!(cursor().is_some(), "the walk stopped mid-way");
    let expired = with_index(&w, "acme", |s| {
        sweep_scope(s, &w.r.clock, &CounterEntropy::default(), w.r.cfg(), 1, 1).unwrap()
    });
    assert!(expired.is_empty());
    assert!(cursor().is_none(), "cursor wrapped at the end");
}

/// F2: the gate is released by the guard's drop, never by a newer sweep's
/// older guard, and stops refusing past its deadline.
#[test]
fn sweep_gate_releases_on_drop_and_expires() {
    let gate = SweepGate::default();
    assert!(!gate.busy(1_000));
    {
        let g = gate.enter(1_000);
        assert!(gate.busy(1_001) && g.live(1_001));
        // An abandoned sweep: the flag would stay set forever without the
        // deadline.
        assert!(!gate.busy(1_000 + SWEEP_DEADLINE_MS));
        assert!(!g.live(1_000 + SWEEP_DEADLINE_MS));
    }
    assert!(!gate.busy(1_002), "drop released the gate");
    let old = gate.enter(5_000);
    let new = gate.enter(6_000);
    assert!(!old.live(6_001) && new.live(6_001));
    drop(old);
    assert!(
        gate.busy(6_001),
        "an older guard does not release a newer sweep"
    );
    drop(new);
    assert!(!gate.busy(6_001));
}
