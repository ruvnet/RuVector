//! Restore journal regressions: a restore that fails, or dies, between
//! two shard commits leaves usage exact and no staged rows behind.

use crate::backend::mem::MemBackend;
use crate::backend::Backend;
use crate::m3_ctx::M3;
use crate::m3_mem::*;
use crate::m3_wire::M3Backend;
use crate::testkit::{block_on, Rng};
use ruvector_edge_store::{OpError, SqlStore};
use ruvector_edge_tenancy::DoName;
use serde_json::{json, Value as Json};
use std::cell::Cell;
use worker::Method;

/// `MemBackend` whose shard commit `fail_at` is refused (not applied), or
/// whose isolate dies right after commit `die_after` applied (reply lost,
/// every later call fails — including the restore's own cleanup).
struct Flaky<'a> {
    b: &'a MemBackend,
    commits: Cell<u32>,
    fail_at: Option<u32>,
    die_after: Option<u32>,
    dead: Cell<bool>,
}

impl<'a> Flaky<'a> {
    fn new(b: &'a MemBackend, fail_at: Option<u32>, die_after: Option<u32>) -> Self {
        Flaky {
            b,
            commits: Cell::new(0),
            fail_at,
            die_after,
            dead: Cell::new(false),
        }
    }
    fn gone(&self) -> Result<(), OpError> {
        if self.dead.get() {
            return Err(crate::wire::unavailable());
        }
        Ok(())
    }
}

impl Backend for Flaky<'_> {
    async fn call_ledger(&self, name: &DoName, body: String) -> Result<String, OpError> {
        self.gone()?;
        self.b.call_ledger(name, body).await
    }
    async fn call_shard(&self, name: &DoName, body: String) -> Result<String, OpError> {
        self.gone()?;
        self.b.call_shard(name, body).await
    }
}

impl M3Backend for Flaky<'_> {
    async fn m3_ledger(&self, name: &DoName, body: String) -> Result<String, OpError> {
        self.gone()?;
        self.b.m3_ledger(name, body).await
    }
    async fn m3_shard(&self, name: &DoName, body: String) -> Result<String, OpError> {
        self.gone()?;
        if !body.contains("\"call\":\"commit\"") {
            return self.b.m3_shard(name, body).await;
        }
        let n = self.commits.get() + 1;
        self.commits.set(n);
        if self.fail_at == Some(n) {
            return Err(crate::wire::unavailable());
        }
        let out = self.b.m3_shard(name, body).await;
        if self.die_after == Some(n) {
            self.dead.set(true);
        }
        self.gone()?;
        out
    }
}

const DIM: usize = 8;

fn seed(w: &World, c: &crate::rest::Caller, n: usize, salt: u64) {
    let mut rng = Rng(salt);
    for chunk in (0..n).collect::<Vec<_>>().chunks(250) {
        let rows: Vec<Json> = chunk
            .iter()
            .map(|i| json!({ "id": format!("v{i:05}"), "values": rng.vec(DIM) }))
            .collect();
        let path = "/v1/collections/docs/vectors:upsert";
        assert_eq!(
            w.req(c, Method::Post, path, json!({ "vectors": rows })).0,
            200
        );
    }
}

fn usage(w: &World, c: &crate::rest::Caller) -> u64 {
    w.req(c, Method::Get, "/v1/usage", Json::Null).1["usage"]["vectors"]
        .as_u64()
        .unwrap()
}

fn count(w: &World, c: &crate::rest::Caller) -> u64 {
    let v = w.req(c, Method::Get, "/v1/collections/docs", Json::Null).1;
    v["count"].as_u64().unwrap()
}

fn staged_rows(w: &World) -> usize {
    let shards = w.b.shards.borrow();
    shards
        .values()
        .map(|s| {
            let q = s.query("SELECT id FROM m3_stage WHERE rid != ?", &["".into()]);
            q.map(|r| r.len()).unwrap_or(0)
        })
        .sum()
}

/// 3 shards, 900 rows snapshotted, then 600 deleted: restoring grows it
/// (the growth is admitted before the first shard swaps).
fn world() -> (World, crate::rest::Caller, String) {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let body = json!({ "name": "docs", "dim": DIM, "metric": "cosine", "shards": 3 });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", body).0, 201);
    seed(&w, &o, 900, 1);
    let (s, v) = w.req(
        &o,
        Method::Post,
        "/v1/collections/docs/snapshots",
        Json::Null,
    );
    assert_eq!(s, 202, "{v}");
    let ids: Vec<String> = (0..600).map(|i| format!("v{i:05}")).collect();
    let path = "/v1/collections/docs/vectors:delete";
    assert_eq!(w.req(&o, Method::Post, path, json!({ "ids": ids })).0, 200);
    assert_eq!((usage(&w, &o), count(&w, &o)), (300, 300));
    (w, o, v["snapshot_id"].as_str().unwrap().to_string())
}

fn flaky_restore(w: &World, o: &crate::rest::Caller, id: &str, f: &Flaky<'_>) -> OpError {
    let m = M3 {
        b: f,
        blob: &w.blob,
        queues: &w.q,
        ctx: &o.ctx,
        now_ms: w.now_ms.get(),
        note: crate::audit::Note::default(),
    };
    block_on(crate::restore::restore(&m, "docs", id)).unwrap_err()
}

#[test]
fn a_commit_error_between_shards_settles_usage_at_once() {
    let (w, o, id) = world();
    let f = Flaky::new(&w.b, Some(2), None);
    flaky_restore(&w, &o, &id, &f);
    // Shard 1 swapped, shards 2-3 untouched: usage is what really is.
    let now = count(&w, &o);
    assert!(now > 300 && now < 900, "{now}");
    assert_eq!(usage(&w, &o), now);
    assert_eq!(staged_rows(&w), 0);
    let path = format!("/v1/collections/docs/snapshots/{id}:restore");
    let (s, v) = w.req(&o, Method::Post, &path, Json::Null);
    assert_eq!(s, 200, "{v}");
    assert_eq!((usage(&w, &o), count(&w, &o)), (900, 900));
}

#[test]
fn a_restore_that_dies_mid_commit_is_settled_by_the_next_one() {
    let (w, o, id) = world();
    let f = Flaky::new(&w.b, None, Some(1));
    flaky_restore(&w, &o, &id, &f);
    // Died after one shard: its cleanup never ran.
    let partial = count(&w, &o);
    assert!(partial > 300 && partial < 900, "{partial}");
    assert!(staged_rows(&w) > 0, "leftovers of the dead restore");
    let path = format!("/v1/collections/docs/snapshots/{id}:restore");
    let (s, v) = w.req(&o, Method::Post, &path, Json::Null);
    assert_eq!(s, 409, "a young admitted journal is a live restore: {v}");
    w.now_ms.set(w.now_ms.get() + 16 * 60 * 1000);
    let (s, v) = w.req(&o, Method::Post, &path, Json::Null);
    assert_eq!(s, 200, "{v}");
    assert_eq!((usage(&w, &o), count(&w, &o)), (900, 900));
    assert_eq!(staged_rows(&w), 0);
}

#[test]
fn a_refused_first_commit_returns_usage_to_where_it_was() {
    let (w, o, id) = world();
    let f = Flaky::new(&w.b, Some(1), None);
    flaky_restore(&w, &o, &id, &f);
    // The admitted growth is released; staged rows and the journal are gone.
    assert_eq!(
        (usage(&w, &o), count(&w, &o), staged_rows(&w)),
        (300, 300, 0)
    );
    let path = format!("/v1/collections/docs/snapshots/{id}:restore");
    let (s, v) = w.req(&o, Method::Post, &path, Json::Null);
    assert_eq!(s, 200, "no journal left behind: {v}");
    assert_eq!((usage(&w, &o), count(&w, &o)), (900, 900));
}
