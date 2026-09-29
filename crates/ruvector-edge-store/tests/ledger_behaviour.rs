//! TenantLedger: claim, members, collection_uid allocation, quotas,
//! idempotency TTL, identity.

mod common;

use common::*;
use ruvector_edge_store::ledger::{IdemKey, IdemLookup};
use ruvector_edge_store::{
    ledger_meta_for, schema, CreateCollection, ErrorCode, MemSqlStore, Metric, SqlStore,
    TenantLedger,
};
use ruvector_edge_tenancy::{CollectionUid, QuotaDelta, QuotaLimits, Role};

fn spec(name: &str) -> CreateCollection {
    CreateCollection {
        name: name.into(),
        dim: 8,
        metric: Metric::Cosine,
        index: None,
        hnsw: None,
        embedder: None,
        filterable_keys: vec![],
        shards: None,
        origin: None,
    }
}

fn open(st: &MemSqlStore) -> TenantLedger {
    TenantLedger::open(st, limits()).unwrap()
}

#[test]
fn claim_once_members_default_deny() {
    let st = MemSqlStore::new();
    let mut l = open(&st);
    let lm = ledger_meta_for(&tenant("org-a")).unwrap();
    assert!(!l.is_claimed(&lm).unwrap());
    assert_eq!(l.role_of(&lm, &sub("alice")).unwrap(), None);
    assert_eq!(l.claim(&st, &lm, &sub("alice"), T0).unwrap(), Role::Owner);
    assert_eq!(
        l.claim(&st, &lm, &sub("bob"), T0).unwrap_err().code,
        ErrorCode::Conflict
    );
    assert_eq!(
        l.role_of(&lm, &sub("bob")).unwrap(),
        None,
        "non-member has no role"
    );
    // Only the owner invites; owner role is not assignable.
    assert_eq!(
        l.put_member(&st, &lm, &sub("bob"), &sub("carol"), Role::Viewer, T0)
            .unwrap_err()
            .code,
        ErrorCode::RoleRequired
    );
    assert_eq!(
        l.put_member(&st, &lm, &sub("alice"), &sub("bob"), Role::Owner, T0)
            .unwrap_err()
            .code,
        ErrorCode::InvalidRequest
    );
    assert_eq!(
        l.put_member(&st, &lm, &sub("alice"), "not-an-edge-sub", Role::Viewer, T0)
            .unwrap_err()
            .code,
        ErrorCode::InvalidRequest
    );
    l.put_member(&st, &lm, &sub("alice"), &sub("bob"), Role::Editor, T0)
        .unwrap();
    assert_eq!(l.role_of(&lm, &sub("bob")).unwrap(), Some(Role::Editor));
    // Persisted: a reopened ledger sees the same memberships.
    let l2 = open(&st);
    assert_eq!(l2.role_of(&lm, &sub("bob")).unwrap(), Some(Role::Editor));
    assert_eq!(l2.role_of(&lm, &sub("alice")).unwrap(), Some(Role::Owner));
    l.remove_member(&st, &lm, &sub("alice"), &sub("bob"))
        .unwrap();
    assert_eq!(l.role_of(&lm, &sub("bob")).unwrap(), None);
    assert_eq!(
        l.remove_member(&st, &lm, &sub("alice"), &sub("alice"))
            .unwrap_err()
            .code,
        ErrorCode::Conflict
    );
    // Another tenant's identity is refused (404), not served.
    let other = ledger_meta_for(&tenant("org-b")).unwrap();
    assert_eq!(
        l.role_of(&other, &sub("alice")).unwrap_err().code,
        ErrorCode::NotFound
    );
}

#[test]
fn collection_uid_never_reused_across_drop_recreate_and_rollback() {
    let st = MemSqlStore::new();
    let mut l = open(&st);
    let lm = ledger_meta_for(&tenant("org-a")).unwrap();
    let e = CounterEntropy::new(1);
    l.create_collection(&st, &lm, &spec("seed"), &sub("o"), &e, T0)
        .unwrap();
    let before = st.clone(); // snapshot: salt persisted, next_seq = 1
    let a1 = l
        .create_collection(&st, &lm, &spec("docs"), &sub("o"), &e, T0)
        .unwrap();
    assert_eq!(
        l.create_collection(&st, &lm, &spec("docs"), &sub("o"), &e, T0)
            .unwrap_err()
            .code,
        ErrorCode::Conflict
    );
    l.drop_collection(&st, &lm, "docs").unwrap();
    assert!(l.collection(&lm, "docs").unwrap().is_none());
    // While its shards are wiped the name stays reserved; a retried drop
    // returns the same uid; the purge is by uid and idempotent.
    let busy = l
        .create_collection(&st, &lm, &spec("docs"), &sub("o"), &e, T0)
        .unwrap_err();
    assert_eq!(busy.code, ErrorCode::Conflict);
    assert_eq!(l.drop_collection(&st, &lm, "docs").unwrap().uid, a1.uid);
    l.purge_collection(&st, &lm, a1.uid).unwrap();
    l.purge_collection(&st, &lm, a1.uid).unwrap();
    assert_eq!(
        l.drop_collection(&st, &lm, "docs").unwrap_err().code,
        ErrorCode::NotFound
    );
    let a2 = l
        .create_collection(&st, &lm, &spec("docs"), &sub("o"), &e, T0)
        .unwrap();
    assert_ne!(a1.uid, a2.uid);
    // A stale purge of the old uid never touches the new collection.
    l.purge_collection(&st, &lm, a1.uid).unwrap();
    assert_eq!(
        l.purge_collection(&st, &lm, a2.uid).unwrap_err().code,
        ErrorCode::NotFound
    );
    assert_eq!(l.collection(&lm, "docs").unwrap().unwrap().uid, a2.uid);
    // Tombstone survives a reopen; uid set only grows.
    let l2 = open(&st);
    assert!(l2.all_uids().contains(&a1.uid) && l2.all_uids().contains(&a2.uid));
    // Ledger rollback to the snapshot: same salt and seq as `a1`, but the
    // fresh nonce still yields a different uid.
    let mut rolled = open(&before);
    let a3 = rolled
        .create_collection(&before, &lm, &spec("docs"), &sub("o"), &e, T0)
        .unwrap();
    assert!(a3.uid != a1.uid && a3.uid != a2.uid);
}

#[test]
fn uid_collision_with_any_catalog_row_is_retried() {
    let st = MemSqlStore::new();
    let mut l = open(&st);
    let lm = ledger_meta_for(&tenant("org-a")).unwrap();
    let e = ConstEntropy(9);
    let a = l
        .create_collection(&st, &lm, &spec("a"), &sub("o"), &e, T0)
        .unwrap();
    assert_eq!(a.uid, CollectionUid::derive(&[9; 32], 0, &[9; 16]));
    // Plant a tombstoned row holding the uid the next allocation would get.
    let planted = CollectionUid::derive(&[9; 32], 1, &[9; 16]);
    st.exec(
        schema::CATALOG_INSERT,
        &[
            planted.to_hex().into(),
            "ghost".into(),
            "vector".into(),
            "flat".into(),
            8i64.into(),
            "cosine".into(),
            ruvector_edge_store::Value::Null,
            1i64.into(),
            "deleted".into(),
            ruvector_edge_store::Value::Null,
            sub("o").into(),
            0i64.into(),
            "[]".into(),
        ],
    )
    .unwrap();
    let mut l = open(&st);
    let b = l
        .create_collection(&st, &lm, &spec("b"), &sub("o"), &e, T0)
        .unwrap();
    assert_eq!(b.uid, CollectionUid::derive(&[9; 32], 2, &[9; 16]));
}

#[test]
fn quotas_are_413_and_checked() {
    let st = MemSqlStore::new();
    let lim = QuotaLimits {
        max_collections: 2,
        max_vectors: 10,
        max_float_budget: 100,
        max_bytes: 1000,
        max_daily_ops: 3,
    };
    let mut l = TenantLedger::open(&st, lim).unwrap();
    let lm = ledger_meta_for(&tenant("org-a")).unwrap();
    let e = CounterEntropy::new(3);
    l.create_collection(&st, &lm, &spec("a"), &sub("o"), &e, T0)
        .unwrap();
    l.create_collection(&st, &lm, &spec("b"), &sub("o"), &e, T0)
        .unwrap();
    assert_eq!(
        l.create_collection(&st, &lm, &spec("c"), &sub("o"), &e, T0)
            .unwrap_err()
            .code,
        ErrorCode::QuotaExceeded
    );
    let grow = |v| QuotaDelta {
        vectors: v,
        float_budget: v * 8,
        bytes: v * 40,
        ..QuotaDelta::default()
    };
    l.admit(&st, &lm, grow(10), 1, T0).unwrap();
    assert_eq!(
        l.admit(&st, &lm, grow(1), 1, T0).unwrap_err().code,
        ErrorCode::QuotaExceeded
    );
    l.admit(&st, &lm, grow(-10), 0, T0).unwrap();
    assert_eq!(
        l.admit(&st, &lm, grow(-1), 0, T0).unwrap_err().code,
        ErrorCode::ServerError,
        "underflow is a bug, never wraps"
    );
    let op = QuotaDelta {
        ops: 1,
        ..QuotaDelta::default()
    };
    for _ in 0..3 {
        l.admit(&st, &lm, op, 1, T0).unwrap();
    }
    assert_eq!(
        l.admit(&st, &lm, op, 1, T0).unwrap_err().code,
        ErrorCode::QuotaExceeded
    );
    // The daily bucket rolls over at the next UTC day.
    l.admit(&st, &lm, op, 1, T0 + 86_400).unwrap();
    let u = open_with(&st, lim).usage(&lm, T0 + 86_400).unwrap();
    assert_eq!((u.collections, u.vectors, u.daily_ops), (2, 0, 1));
}

fn open_with(st: &MemSqlStore, lim: QuotaLimits) -> TenantLedger {
    TenantLedger::open(st, lim).unwrap()
}

#[test]
fn create_validation() {
    let st = MemSqlStore::new();
    let mut l = open(&st);
    let lm = ledger_meta_for(&tenant("org-a")).unwrap();
    let e = CounterEntropy::new(5);
    let bad = [
        CreateCollection {
            name: "Bad".into(),
            ..spec("x")
        },
        CreateCollection {
            dim: 0,
            ..spec("x")
        },
        CreateCollection {
            dim: 1537,
            ..spec("x")
        },
        CreateCollection {
            index: Some("rabitq".into()),
            ..spec("x")
        },
        CreateCollection {
            embedder: Some("bge".into()),
            ..spec("x")
        },
        CreateCollection {
            shards: Some(7),
            ..spec("x")
        },
        CreateCollection {
            filterable_keys: (0..9).map(|i| format!("k{i}")).collect(),
            ..spec("x")
        },
        CreateCollection {
            origin: Some("Bad Origin".into()),
            ..spec("x")
        },
    ];
    for b in bad {
        let err = l
            .create_collection(&st, &lm, &b, &sub("o"), &e, T0)
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidRequest, "{b:?}");
    }
    assert!(l.collections(&lm).unwrap().is_empty());
    let ok = CreateCollection {
        dim: 1536,
        shards: Some(6),
        origin: Some("ruvector-chatgpt".into()),
        ..spec("x")
    };
    assert_eq!(
        l.create_collection(&st, &lm, &ok, &sub("o"), &e, T0)
            .unwrap()
            .shard_count
            .get(),
        6
    );
}

#[test]
fn idempotency_ttl_and_body_binding() {
    let st = MemSqlStore::new();
    let mut l = open(&st);
    let lm = ledger_meta_for(&tenant("org-a")).unwrap();
    let (h1, h2) = ([1u8; 32], [2u8; 32]);
    let k = |h| IdemKey {
        sub: "es1_s",
        key: "K",
        body_sha256: h,
    };
    assert_eq!(
        l.idem_lookup(&st, &lm, k(&h1), T0).unwrap(),
        IdemLookup::Miss
    );
    l.idem_store(&st, &lm, k(&h1), "{\"ok\":true}", T0).unwrap();
    assert_eq!(
        l.idem_lookup(&st, &lm, k(&h1), T0 + 10).unwrap(),
        IdemLookup::Replay("{\"ok\":true}".into())
    );
    assert_eq!(
        l.idem_lookup(&st, &lm, k(&h2), T0 + 10).unwrap(),
        IdemLookup::Conflict
    );
    let other = IdemKey {
        sub: "es1_other",
        key: "K",
        body_sha256: &h2,
    };
    assert_eq!(
        l.idem_lookup(&st, &lm, other, T0).unwrap(),
        IdemLookup::Miss,
        "scoped per sub"
    );
    assert_eq!(
        l.idem_lookup(&st, &lm, k(&h2), T0 + 86_400).unwrap(),
        IdemLookup::Miss,
        "expired after 24 h"
    );
    // Expired rows are purged by the next store.
    l.idem_store(&st, &lm, other, "{}", T0 + 86_400).unwrap();
    assert_eq!(st.row_count("idempotency"), 1);
}
