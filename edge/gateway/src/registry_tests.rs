//! M5 registry routes end to end over the real ledger and registry DO
//! cores, an in-memory R2 and rvf-CLI-exported packages.

use crate::registry_routes::{parse, RvfReply, RvfRoute};
use crate::registry_world::*;
use crate::testkit::*;
use ruvector_edge_auth::Capability;
use ruvector_edge_tenancy::Role;
use serde_json::{json, Value as Json};
use worker::Method;

fn fixture() -> Vec<u8> {
    rvf_export(40, 8, 7, &[]).0
}

fn world_with_acme() -> (World, ruvector_edge_store::CallerContext) {
    let w = World::new();
    let alice = w.owner("org-a", "alice");
    assert_eq!(w.claim(&alice, "acme").0, 200);
    (w, alice)
}

#[test]
fn routes_parse_scope_spellings_actions_and_import() {
    let p = |m: Method, s: &str| parse(&m, s);
    for s in ["acme", "@acme", "%40acme"] {
        assert_eq!(
            p(Method::Get, &format!("/v1/rvf/{s}/x/1.0.0")),
            Some(RvfRoute::Get("@acme/x".into(), "1.0.0".into()))
        );
    }
    assert_eq!(p(Method::Get, "/v1/rvf/scopes"), Some(RvfRoute::ListScopes));
    assert_eq!(
        p(Method::Post, "/v1/rvf/scopes/@acme"),
        Some(RvfRoute::ClaimScope("acme".into()))
    );
    assert_eq!(
        p(Method::Post, "/v1/rvf/acme/x/1.0.0:publish"),
        Some(RvfRoute::Publish("@acme/x".into(), "1.0.0".into()))
    );
    assert_eq!(p(Method::Post, "/v1/rvf/acme/x/1.0.0:delete"), None);
    assert_eq!(
        p(Method::Put, "/v1/rvf/acme/x/1.0.0/uploads/up_x/parts/3"),
        Some(RvfRoute::Part(
            "@acme/x".into(),
            "1.0.0".into(),
            "up_x".into(),
            "3".into()
        ))
    );
    assert_eq!(
        p(Method::Post, "/v1/collections/docs:import-rvf"),
        Some(RvfRoute::Import("docs".into()))
    );
    assert_eq!(p(Method::Get, "/v1/collections/docs:import-rvf"), None);
    assert_eq!(p(Method::Get, "/v1/rvf//x"), None);
    assert_eq!(p(Method::Get, "/v1/collections/docs"), None);
}

#[test]
fn scope_claims_refuse_squatting_and_lookalikes() {
    let (w, alice) = world_with_acme();
    // Idempotent for the same tenant and exact scope.
    assert_eq!(w.claim(&alice, "@acme").0, 200);
    let bob = w.owner("org-b", "bob");
    let (s, b) = w.claim(&bob, "acme");
    assert_eq!((s, b["code"].as_str()), (409, Some("conflict")));
    // A confusable of a claimed scope is squatting too.
    assert_eq!(w.claim(&bob, "acrne").0, 409);
    assert_eq!(w.claim(&alice, "acrne").0, 409);
    // Claiming needs admin: an editor's token is refused with a step-up.
    let ed = w.member(&alice, "ed", Role::Editor, rw());
    let r = match w.call(&ed, Method::Post, "/v1/rvf/scopes/other", b"") {
        RvfReply::Api(r) => r,
        RvfReply::Blob { .. } => unreachable!(),
    };
    assert_eq!(r.status, 403);
    assert!(r.www_authenticate.unwrap().contains("ruvector:admin"));
    // Reserved / brand scopes never parse.
    assert_eq!(w.claim(&bob, "ruvector").0, 400);
    let (s, b) = w.json(&alice, Method::Get, "/v1/rvf/scopes", Json::Null);
    assert_eq!((s, b), (200, json!({ "scopes": ["acme"] })));
    // Another tenant cannot push into the claimed scope.
    let bytes = fixture();
    let (s, b) = w.begin(&bob, "acme/pkg", "1.0.0", "tenant", &bytes);
    assert_eq!((s, b["code"].as_str()), (403, Some("role_required")));
    // An unclaimed scope refuses pushes.
    assert_eq!(
        w.begin(&bob, "nobody/pkg", "1.0.0", "tenant", &bytes).0,
        403
    );
}

#[test]
fn upload_finalize_pull_round_trip_through_multipart_parts() {
    let (w, alice) = world_with_acme();
    let bytes = fixture();
    let (s, m) = w.push(&alice, "acme/pkg", "1.0.0", "tenant", &bytes, 3);
    assert_eq!(s, 201, "{m}");
    assert_eq!(
        (m["dim"].as_u64(), m["metric"].as_str()),
        (Some(8), Some("cosine"))
    );
    assert_eq!(m["total_vectors"], json!(40));
    assert_eq!(m["sha256"], json!(hex::encode(sha(&bytes))));
    assert_eq!(m["owner"].as_str(), Some(alice.tenant_key().as_str()));
    // Staging multipart completed and the staging object cleaned up.
    assert_eq!(w.s.open_uploads(), 0);
    assert!(w
        .s
        .objects
        .borrow()
        .keys()
        .all(|k| !k.starts_with("staging/")));
    // A viewer in the tenant pulls the exact bytes.
    let viewer = w.member(&alice, "vic", Role::Viewer, ro());
    assert_eq!(
        w.pull(&viewer, "acme/pkg", "1.0.0"),
        Ok((bytes.clone(), false))
    );
    let (s, g) = w.json(&viewer, Method::Get, "/v1/rvf/acme/pkg/1.0.0", Json::Null);
    assert_eq!((s, &g["sha256"]), (200, &m["sha256"]));
    let (s, page) = w.json(&viewer, Method::Get, "/v1/rvf/acme/pkg", Json::Null);
    assert_eq!(
        (s, page["items"][0]["version"].as_str()),
        (200, Some("1.0.0"))
    );
    // Versions are immutable.
    assert_eq!(
        w.begin(&alice, "acme/pkg", "1.0.0", "tenant", &bytes).0,
        409
    );
    // Declared digest must match the bytes: a lying declaration fails.
    let body = json!({ "size": bytes.len(), "sha256": hex::encode([7u8; 32]) });
    let (_, b) = w.json(&alice, Method::Post, "/v1/rvf/acme/pkg/2.0.0/uploads", body);
    let id = b["upload_id"].as_str().unwrap().to_string();
    w.parts(&alice, "acme/pkg", "2.0.0", &id, &bytes, 2);
    let (s, b) = w.finalize(&alice, "acme/pkg", "2.0.0", &id);
    assert_eq!((s, b["detail"].as_str()), (400, Some("digest mismatch")));
    // Invalid RVF bytes are refused with the validator's error.
    let junk = vec![0x42u8; 300];
    let (s, _) = w.push(&alice, "acme/pkg", "3.0.0", "tenant", &junk, 1);
    assert_eq!(s, 400);
    assert_eq!(w.pull(&alice, "acme/pkg", "3.0.0"), Err(404));
}

#[test]
fn parts_changed_after_freeze_fail_the_session() {
    let (w, alice) = world_with_acme();
    let bytes = fixture();
    *w.s.tamper_prefix.borrow_mut() = Some("staging/".into());
    let (s, b) = w.begin(&alice, "acme/pkg", "1.0.0", "tenant", &bytes);
    assert_eq!(s, 201);
    let id = b["upload_id"].as_str().unwrap().to_string();
    w.parts(&alice, "acme/pkg", "1.0.0", &id, &bytes, 3);
    let (s, b) = w.finalize(&alice, "acme/pkg", "1.0.0", &id);
    assert_eq!(
        (s, b["detail"].as_str()),
        (400, Some("upload parts changed after finalize_plan"))
    );
    // Terminal: a retry is refused, nothing was published or written.
    assert_eq!(w.finalize(&alice, "acme/pkg", "1.0.0", &id).0, 409);
    assert_eq!(w.pull(&alice, "acme/pkg", "1.0.0"), Err(404));
    assert!(w.s.objects.borrow().keys().all(|k| !k.starts_with("rvf/")));
}

#[test]
fn blob_evidence_mismatch_fails_the_finalize() {
    for parts in [1, 3] {
        let (w, alice) = world_with_acme();
        let bytes = fixture();
        *w.s.corrupt_prefix.borrow_mut() = Some("rvf/".into());
        let (s, b) = w.push(&alice, "acme/pkg", "1.0.0", "tenant", &bytes, parts);
        assert_eq!(
            (s, b["detail"].as_str()),
            (400, Some("blob evidence mismatch"))
        );
        assert_eq!(w.pull(&alice, "acme/pkg", "1.0.0"), Err(404));
    }
}

#[test]
fn cross_tenant_private_and_tenant_pulls_are_404() {
    let (w, alice) = world_with_acme();
    let bytes = fixture();
    assert_eq!(
        w.push(&alice, "acme/priv", "1.0.0", "private", &bytes, 2).0,
        201
    );
    assert_eq!(
        w.push(&alice, "acme/team", "1.0.0", "tenant", &bytes, 2).0,
        201
    );
    let bob = w.owner("org-b", "bob");
    for pkg in ["acme/priv", "acme/team"] {
        assert_eq!(w.pull(&bob, pkg, "1.0.0"), Err(404));
        let (s, _) = w.json(
            &bob,
            Method::Get,
            &format!("/v1/rvf/{pkg}/1.0.0"),
            Json::Null,
        );
        assert_eq!(s, 404);
        // Listing reveals nothing either.
        let (s, _) = w.json(&bob, Method::Get, &format!("/v1/rvf/{pkg}"), Json::Null);
        assert_eq!(s, 404);
    }
    // Private: the uploader and admins only, even inside the tenant.
    let viewer = w.member(&alice, "vic", Role::Viewer, ro());
    assert_eq!(w.pull(&viewer, "acme/priv", "1.0.0"), Err(404));
    assert!(w.pull(&viewer, "acme/team", "1.0.0").is_ok());
    // A token without read gets the step-up, not the bytes.
    let blind = w.member(&alice, "blind", Role::Viewer, caps(&[Capability::Write]));
    assert_eq!(w.pull(&blind, "acme/team", "1.0.0"), Err(403));
}

#[test]
fn public_publish_is_write_once_and_shared() {
    let (w, alice) = world_with_acme();
    let bytes = fixture();
    assert_eq!(
        w.push(&alice, "acme/pkg", "1.0.0", "tenant", &bytes, 2).0,
        201
    );
    // Needs ruvector:publish: an editor gets the step-up challenge.
    let ed = w.member(&alice, "ed", Role::Editor, rw());
    let r = match w.call(&ed, Method::Post, "/v1/rvf/acme/pkg/1.0.0:publish", b"") {
        RvfReply::Api(r) => r,
        RvfReply::Blob { .. } => unreachable!(),
    };
    assert_eq!(r.status, 403);
    assert!(r.www_authenticate.unwrap().contains("ruvector:publish"));
    let (s, m) = w.json(
        &alice,
        Method::Post,
        "/v1/rvf/acme/pkg/1.0.0:publish",
        Json::Null,
    );
    assert_eq!((s, m["visibility"].as_str()), (200, Some("public")));
    let public = format!("public/blobs/sha256/{}", hex::encode(sha(&bytes)));
    assert_eq!(w.s.get(&public), Some(bytes.clone()));
    // Idempotent.
    assert_eq!(
        w.json(
            &alice,
            Method::Post,
            "/v1/rvf/acme/pkg/1.0.0:publish",
            Json::Null
        )
        .0,
        200
    );
    // Any tenant can now pull it, without seeing tenant identities.
    let bob = w.owner("org-b", "bob");
    assert_eq!(
        w.pull(&bob, "acme/pkg", "1.0.0"),
        Ok((bytes.clone(), false))
    );
    let (_, v) = w.json(&bob, Method::Get, "/v1/rvf/acme/pkg/1.0.0", Json::Null);
    assert!(v.get("owner").is_none() && v.get("created_by").is_none());
    // A poisoned public key is never overwritten and never backs a version.
    let other = rvf_export(12, 8, 99, &[]).0;
    let poisoned = format!("public/blobs/sha256/{}", hex::encode(sha(&other)));
    w.s.put(&poisoned, b"not the package".to_vec());
    assert_eq!(
        w.push(&alice, "acme/other", "1.0.0", "tenant", &other, 1).0,
        201
    );
    let (s, b) = w.json(
        &alice,
        Method::Post,
        "/v1/rvf/acme/other/1.0.0:publish",
        Json::Null,
    );
    assert_eq!(
        (s, b["detail"].as_str()),
        (400, Some("blob evidence mismatch"))
    );
    assert_eq!(w.s.get(&poisoned), Some(b"not the package".to_vec()));
    let (_, v) = w.json(&alice, Method::Get, "/v1/rvf/acme/other/1.0.0", Json::Null);
    assert_eq!(v["visibility"].as_str(), Some("tenant"));
    assert_eq!(w.pull(&bob, "acme/other", "1.0.0"), Err(404));
}

#[test]
fn yank_keeps_exact_pulls_and_guards_unyank() {
    let (w, alice) = world_with_acme();
    let carol = w.member(&alice, "carol", Role::Editor, rw());
    let bytes = fixture();
    assert_eq!(
        w.push(&carol, "acme/pkg", "1.0.0", "tenant", &bytes, 2).0,
        201
    );
    let reason = json!({ "reason": "bad embeddings" });
    let (s, m) = w.json(&alice, Method::Post, "/v1/rvf/acme/pkg/1.0.0:yank", reason);
    assert_eq!(
        (s, m["yanked"]["reason"].as_str()),
        (200, Some("bad embeddings"))
    );
    // Exact-version pulls keep working, flagged.
    assert_eq!(w.pull(&carol, "acme/pkg", "1.0.0"), Ok((bytes, true)));
    let (_, page) = w.json(&carol, Method::Get, "/v1/rvf/acme/pkg", Json::Null);
    assert_eq!(page["items"][0]["yanked"], json!(true));
    // The uploader cannot undo an admin's yank; the admin can.
    let (s, b) = w.json(
        &carol,
        Method::Post,
        "/v1/rvf/acme/pkg/1.0.0:unyank",
        Json::Null,
    );
    assert_eq!((s, b["code"].as_str()), (403, Some("role_required")));
    let (s, m) = w.json(
        &alice,
        Method::Post,
        "/v1/rvf/acme/pkg/1.0.0:unyank",
        Json::Null,
    );
    assert_eq!((s, &m["yanked"]), (200, &Json::Null));
    // A yanked version cannot be published.
    w.json(
        &alice,
        Method::Post,
        "/v1/rvf/acme/pkg/1.0.0:yank",
        Json::Null,
    );
    assert_eq!(
        w.json(
            &alice,
            Method::Post,
            "/v1/rvf/acme/pkg/1.0.0:publish",
            Json::Null
        )
        .0,
        400
    );
    // Another tenant cannot yank.
    let bob = w.owner("org-b", "bob");
    assert_eq!(
        w.json(
            &bob,
            Method::Post,
            "/v1/rvf/acme/pkg/1.0.0:yank",
            Json::Null
        )
        .0,
        404
    );
}

#[test]
fn sweep_aborts_stale_staging_and_keeps_repinned_blobs() {
    let (w, alice) = world_with_acme();
    let bytes = fixture();
    // An abandoned session with one part uploaded.
    let (_, b) = w.begin(&alice, "acme/stale", "1.0.0", "tenant", &bytes);
    let id = b["upload_id"].as_str().unwrap().to_string();
    let chunk = bytes.len().div_ceil(2);
    let path = format!("/v1/rvf/acme/stale/1.0.0/uploads/{id}/parts/1");
    assert!(
        matches!(w.call(&alice, Method::Put, &path, &bytes[..chunk]), RvfReply::Api(r) if r.status == 200)
    );
    assert_eq!(w.s.open_uploads(), 1);
    // Publish releases the tenant blob (queued), then a new version of the
    // same bytes re-pins it before any sweep.
    assert_eq!(
        w.push(&alice, "acme/pkg", "1.0.0", "tenant", &bytes, 2).0,
        201
    );
    assert_eq!(
        w.json(
            &alice,
            Method::Post,
            "/v1/rvf/acme/pkg/1.0.0:publish",
            Json::Null
        )
        .0,
        200
    );
    assert_eq!(
        w.push(&alice, "acme/pkg", "2.0.0", "tenant", &bytes, 2).0,
        201
    );
    // Nothing is due yet.
    let work = w.r.alarm("acme", &w.s);
    assert!(work.abort.is_empty() && work.expired.is_empty(), "{work:?}");
    // A day later the stale session expires: its multipart is aborted.
    w.r.clock.advance(24 * 3600 + 1);
    let work = w.r.alarm("acme", &w.s);
    assert_eq!(work.expired, vec![id.clone()]);
    assert_eq!(work.abort.len(), 1);
    assert_eq!(w.s.aborted.borrow().len(), 1);
    assert_eq!(w.s.open_uploads(), 0);
    assert!(work.delete.iter().any(|k| k.ends_with(&id)));
    assert_eq!(w.finalize(&alice, "acme/stale", "1.0.0", &id).0, 409);
    // The re-pinned tenant blob survived; both versions still pull.
    assert_eq!(
        w.pull(&alice, "acme/pkg", "2.0.0"),
        Ok((bytes.clone(), false))
    );
    assert_eq!(w.pull(&alice, "acme/pkg", "1.0.0"), Ok((bytes, false)));
}

#[test]
fn released_blob_is_deleted_by_the_sweep() {
    let (w, alice) = world_with_acme();
    let bytes = fixture();
    assert_eq!(
        w.push(&alice, "acme/pkg", "1.0.0", "tenant", &bytes, 2).0,
        201
    );
    let tenant_blob =
        w.s.objects
            .borrow()
            .keys()
            .find(|k| k.starts_with("rvf/"))
            .cloned()
            .unwrap();
    assert_eq!(
        w.json(
            &alice,
            Method::Post,
            "/v1/rvf/acme/pkg/1.0.0:publish",
            Json::Null
        )
        .0,
        200
    );
    // Queued, not deleted, by the publish.
    assert!(w.s.get(&tenant_blob).is_some());
    let work = w.r.alarm("acme", &w.s);
    assert!(work.delete.contains(&tenant_blob));
    assert!(w.s.get(&tenant_blob).is_none());
    assert!(w.pull(&alice, "acme/pkg", "1.0.0").is_ok());
}

#[test]
fn finalize_after_a_storage_outage_is_retryable() {
    for parts in [1, 3] {
        let (w, alice) = world_with_acme();
        let bytes = fixture();
        let (_, b) = w.begin(&alice, "acme/pkg", "1.0.0", "tenant", &bytes);
        let id = b["upload_id"].as_str().unwrap().to_string();
        w.parts(&alice, "acme/pkg", "1.0.0", &id, &bytes, parts);
        w.s.fail_blob_writes.set(true);
        let (s, b) = w.finalize(&alice, "acme/pkg", "1.0.0", &id);
        assert_eq!((s, b["code"].as_str()), (503, Some("shard_unavailable")));
        // The session stayed frozen; the staging object is already complete.
        w.s.fail_blob_writes.set(false);
        let (s, m) = w.finalize(&alice, "acme/pkg", "1.0.0", &id);
        assert_eq!(s, 201, "{m}");
        assert_eq!(w.pull(&alice, "acme/pkg", "1.0.0"), Ok((bytes, false)));
    }
}
