//! M3 acceptance (ADR-351 §15): kill-and-restore returns byte-identical
//! results; cross-tenant and tampered snapshots are refused; exports carry
//! a witness that verifies offline and honour `redact`.

use crate::m3_mem::*;
use crate::testkit::Rng;
use ruvector_edge_auth::Capability;
use ruvector_edge_snapshot::{verify_export, SignaturePolicy};
use ruvector_edge_store::ErrorCode;
use serde_json::{json, Value as Json};
use worker::Method;

const DIM: usize = 16;

fn seed(w: &World, c: &crate::rest::Caller, name: &str, n: usize) {
    let body = json!({ "name": name, "dim": DIM, "metric": "cosine", "shards": 3,
                       "filterable_keys": ["kind"] });
    let (s, v) = w.req(c, Method::Post, "/v1/collections", body);
    assert_eq!(s, 201, "{v}");
    let mut rng = Rng(42);
    for chunk in (0..n).collect::<Vec<_>>().chunks(200) {
        let rows: Vec<Json> = chunk
            .iter()
            .map(|i| {
                json!({ "id": format!("v{i:05}"), "values": rng.vec(DIM),
                        "metadata": { "kind": if i % 2 == 0 { "a" } else { "b" }, "n": i,
                                      "secret": format!("s{i}") } })
            })
            .collect();
        let path = format!("/v1/collections/{name}/vectors:upsert");
        let (s, v) = w.req(c, Method::Post, &path, json!({ "vectors": rows }));
        assert_eq!(s, 200, "{v}");
    }
}

/// Query results as exact JSON text (bit-identical floats and metadata).
fn results(w: &World, c: &crate::rest::Caller, name: &str) -> Vec<String> {
    let mut rng = Rng(7);
    let mut out = Vec::new();
    for k in 0..6 {
        let mut q = json!({ "vector": rng.vec(DIM), "top_k": 20,
                            "include": ["metadata", "values"] });
        if k % 2 == 1 {
            q["filter"] = json!({ "kind": "a" });
        }
        let (s, v) = w.req(c, Method::Post, &format!("/v1/collections/{name}/query"), q);
        assert_eq!(s, 200, "{v}");
        out.push(v["matches"].to_string());
    }
    let ids: Vec<String> = (0..100).map(|i| format!("v{:05}", i * 3)).collect();
    let body = json!({ "ids": ids, "include_values": true });
    let (s, v) = w.req(
        c,
        Method::Post,
        &format!("/v1/collections/{name}/vectors:fetch"),
        body,
    );
    assert_eq!(s, 200);
    out.push(v.to_string());
    out
}

fn usage_vectors(w: &World, c: &crate::rest::Caller) -> u64 {
    let (s, v) = w.req(c, Method::Get, "/v1/usage", Json::Null);
    assert_eq!(s, 200);
    v["usage"]["vectors"].as_u64().unwrap()
}

fn snapshot(w: &World, c: &crate::rest::Caller, name: &str) -> String {
    let (s, v) = w.req(
        c,
        Method::Post,
        &format!("/v1/collections/{name}/snapshots"),
        Json::Null,
    );
    assert_eq!(s, 202, "{v}");
    v["snapshot_id"].as_str().unwrap().to_string()
}

fn restore(w: &World, c: &crate::rest::Caller, name: &str, id: &str) -> (u16, Json) {
    let path = format!("/v1/collections/{name}/snapshots/{id}:restore");
    w.req(c, Method::Post, &path, Json::Null)
}

fn mutate(w: &World, c: &crate::rest::Caller) {
    let ids: Vec<String> = (0..60).map(|i| format!("v{:05}", i * 5)).collect();
    let (s, _) = w.req(
        c,
        Method::Post,
        "/v1/collections/docs/vectors:delete",
        json!({ "ids": ids }),
    );
    assert_eq!(s, 200);
    let mut rng = Rng(99);
    let mut rows: Vec<Json> = (0..40)
        .map(|i| json!({ "id": format!("v{:05}", i * 7 + 1), "values": rng.vec(DIM) }))
        .collect();
    rows.extend((0..25).map(|i| json!({ "id": format!("new{i}"), "values": rng.vec(DIM) })));
    let (s, v) = w.req(
        c,
        Method::Post,
        "/v1/collections/docs/vectors:upsert",
        json!({ "vectors": rows }),
    );
    assert_eq!(s, 200, "{v}");
}

#[test]
fn kill_and_restore_returns_byte_identical_results() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    seed(&w, &o, "docs", 900);
    let before = results(&w, &o, "docs");
    let id = snapshot(&w, &o, "docs");
    assert!(!w.blob.keys("snapshots/").is_empty());
    let (s, v) = w.req(
        &o,
        Method::Get,
        "/v1/collections/docs/snapshots",
        Json::Null,
    );
    assert_eq!((s, v["snapshots"].as_array().unwrap().len()), (200, 1));
    assert_eq!(v["snapshots"][0]["rows"], 900);

    // Diverge, then kill the isolate (every resident shard and ledger dropped).
    mutate(&w, &o);
    assert_ne!(results(&w, &o, "docs"), before);
    w.b.restart();
    let (s, v) = restore(&w, &o, "docs", &id);
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["rows"], 900);
    assert_eq!(v["epoch"], 2, "restore lands in a new epoch");
    assert_eq!(results(&w, &o, "docs"), before);
    assert_eq!(usage_vectors(&w, &o), 900);

    // Disaster: every shard's storage is lost; R2 + the ledger chain suffice.
    w.b.shards.borrow_mut().clear();
    w.b.restart();
    let (s, v) = restore(&w, &o, "docs", &id);
    assert_eq!(s, 200, "{v}");
    assert_eq!(results(&w, &o, "docs"), before);
    // A later snapshot takes the next epoch.
    // Epochs: snapshot 1, restores 2 and 3, the next snapshot 4.
    assert_eq!(snapshot(&w, &o, "docs"), "4");
}

#[test]
fn restore_needs_admin_scope_and_owner_role() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    seed(&w, &o, "docs", 50);
    let id = snapshot(&w, &o, "docs");
    let editor = caller("org-a", "alice", &[Capability::Read, Capability::Write]);
    let (s, v) = restore(&w, &editor, "docs", &id);
    assert_eq!(s, 403);
    assert!(is(&v, ErrorCode::InsufficientScope));
    let stranger = caller("org-a", "mallory", ALL);
    let (s, v) = restore(&w, &stranger, "docs", &id);
    assert_eq!(s, 403);
    assert!(is(&v, ErrorCode::RoleRequired));
    let (s, _) = restore(&w, &o, "docs", "99");
    assert_eq!(s, 404);
}

#[test]
fn cross_tenant_restore_is_refused() {
    let w = World::default();
    let a = w.owner("org-a", "alice");
    let b = w.owner("org-b", "bob");
    seed(&w, &a, "docs", 120);
    seed(&w, &b, "docs", 60);
    let a_id = snapshot(&w, &a, "docs");
    let b_before = results(&w, &b, "docs");
    // B has no snapshot with A's id in its own ledger.
    let (s, _) = restore(&w, &b, "docs", &a_id);
    assert_eq!(s, 404);
    // B snapshots, then an attacker replaces B's snapshot objects with A's.
    let b_id = snapshot(&w, &b, "docs");
    let a_keys = w
        .blob
        .keys(&format!("snapshots/{}/", a.ctx.tenant_key().as_str()));
    let b_keys = w
        .blob
        .keys(&format!("snapshots/{}/", b.ctx.tenant_key().as_str()));
    let a_manifests: Vec<_> = a_keys
        .iter()
        .filter(|k| k.ends_with("manifest.rvf"))
        .collect();
    let b_manifests: Vec<_> = b_keys
        .iter()
        .filter(|k| k.ends_with("manifest.rvf"))
        .collect();
    assert_eq!((a_manifests.len(), b_manifests.len()), (3, 3));
    for (ak, bk) in a_manifests.iter().zip(&b_manifests) {
        let obj = w.blob.bytes(ak).unwrap();
        w.blob
            .objects
            .borrow_mut()
            .insert((*bk).clone(), (obj, None));
    }
    let (s, v) = restore(&w, &b, "docs", &b_id);
    assert_eq!(s, 403, "{v}");
    assert!(is(&v, ErrorCode::TenantMismatch));
    assert_eq!(results(&w, &b, "docs"), b_before, "B's data untouched");
}

#[test]
fn tampered_snapshot_is_refused_and_nothing_changes() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    seed(&w, &o, "docs", 600);
    let id = snapshot(&w, &o, "docs");
    mutate(&w, &o);
    let now = results(&w, &o, "docs");
    let tenant = o.ctx.tenant_key().as_str().to_string();
    let chunk = w
        .blob
        .keys(&format!("snapshots/{tenant}/"))
        .into_iter()
        .find(|k| k.ends_with("seg-0.rvf"))
        .unwrap();
    let good = w.blob.bytes(&chunk).unwrap();
    let mut bad = good.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0x01;
    w.blob
        .objects
        .borrow_mut()
        .insert(chunk.clone(), (bad, None));
    let (s, v) = restore(&w, &o, "docs", &id);
    assert_eq!(s, 409, "{v}");
    assert!(is(&v, ErrorCode::Conflict));
    assert_eq!(
        results(&w, &o, "docs"),
        now,
        "a refused restore changes nothing"
    );
    // Truncation is refused too.
    w.blob
        .objects
        .borrow_mut()
        .insert(chunk.clone(), (good[..good.len() - 64].to_vec(), None));
    assert_eq!(restore(&w, &o, "docs", &id).0, 409);
    // A rewritten manifest (root no longer in the witness chain) is refused.
    w.blob.objects.borrow_mut().insert(chunk, (good, None));
    let manifest = w
        .blob
        .keys(&format!("snapshots/{tenant}/"))
        .into_iter()
        .find(|k| k.ends_with("manifest.rvf"))
        .unwrap();
    let mut m = w.blob.bytes(&manifest).unwrap();
    m[100] ^= 0x80;
    w.blob.objects.borrow_mut().insert(manifest, (m, None));
    assert_eq!(restore(&w, &o, "docs", &id).0, 409);
    assert_eq!(results(&w, &o, "docs"), now);
    // No staging rows are left behind.
    for store in w.b.shards.borrow().values() {
        use ruvector_edge_store::SqlStore;
        let left = store.query("SELECT id FROM m3_stage", &[]).unwrap();
        assert!(left.is_empty());
    }
}

#[test]
fn export_is_witnessed_redacted_and_tenant_scoped() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    seed(&w, &o, "docs", 300);
    let (s, v) = w.req(
        &o,
        Method::Post,
        "/v1/collections/docs:export?redact=secret",
        Json::Null,
    );
    assert_eq!(s, 201, "{v}");
    assert_eq!(v["rows"], 300);
    let id = v["export_id"].as_str().unwrap().to_string();
    let download = v["download"].as_str().unwrap().to_string();
    assert_eq!(download, format!("/v1/exports/{id}"));
    assert!(!download.contains("r2") && !download.starts_with("http"));
    let (s, d) = w.req(&o, Method::Get, &download, Json::Null);
    assert_eq!(s, 200);
    let file = w.blob.bytes(d["key"].as_str().unwrap()).unwrap();
    let tenant = o.ctx.tenant_key().as_str();
    let wit = verify_export(&file, tenant, SignaturePolicy::NotRequired).unwrap();
    assert_eq!(wit.rows, 300);
    assert!(
        !file.windows(7).any(|x| x == b"\"secret"),
        "redacted key absent"
    );
    assert!(file.windows(4).any(|x| x == b"kind"));
    // Another tenant cannot fetch it; nor can anyone after expiry.
    let b = w.owner("org-b", "bob");
    assert_eq!(w.req(&b, Method::Get, &download, Json::Null).0, 404);
    let other = verify_export(
        &file,
        b.ctx.tenant_key().as_str(),
        SignaturePolicy::NotRequired,
    );
    assert!(other.is_err());
    w.now_ms.set(w.now_ms.get() + 16 * 60 * 1000);
    assert_eq!(w.req(&o, Method::Get, &download, Json::Null).0, 404);
}

#[test]
fn m3_requests_emit_audit_events_without_payloads() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    seed(&w, &o, "docs", 20);
    w.q.take(crate::m3_ports::QueueName::Audit);
    snapshot(&w, &o, "docs");
    let evs = w.q.take(crate::m3_ports::QueueName::Audit);
    assert_eq!(evs.len(), 1);
    assert_eq!(evs[0]["route"], "snapshot.create");
    assert_eq!(evs[0]["outcome"], 202);
    assert_eq!(evs[0]["tenant_key"], o.ctx.tenant_key().as_str());
    let text = evs[0].to_string();
    assert!(!text.contains("values") && !text.contains("metadata"));
}
