//! ADR-351 §4.3 in-object assertion (`DoMeta` / `LedgerMeta`). The
//! `TenantContext::from_verified` tests live in `context_claims.rs`.

mod common;

use ruvector_edge_auth::RouteSurface;
use ruvector_edge_tenancy::meta::{META_COLLECTION_UID, META_SERVICE, META_SHARD, META_TENANT_KEY};
use ruvector_edge_tenancy::{
    ledger_do_name, CollectionUid, DoMeta, IdentityCheck, LedgerMeta, ProblemCode, Service,
    ShardCount, ShardIndex, TenancyError, TenantContext,
};

fn ctx(org: &'static str) -> TenantContext {
    let mut t = common::edge();
    t.org = org;
    TenantContext::from_verified(t.verify(), None, RouteSurface::Rest).unwrap()
}

fn uid(n: u64) -> CollectionUid {
    CollectionUid::derive(&[1; 32], n, &[0; 16])
}

#[test]
fn do_meta_round_trip_through_kv() {
    let c = ctx("org-a");
    let shard = ShardIndex::new(2, ShardCount::new(3).unwrap()).unwrap();
    let m = DoMeta::expected(&c, Service::Vector, uid(0), shard);
    let kv = m.to_kv();
    let mut rows: Vec<(&str, &str)> = kv.iter().map(|(k, v)| (*k, v.as_str())).collect();
    rows.push(("quant_epoch", "7")); // unrelated meta keys are ignored
    let back = DoMeta::from_kv(rows).unwrap().unwrap();
    assert_eq!(back, m);
    assert_eq!(back.do_name(), m.do_name());
    assert_eq!(back.tenant_key(), c.tenant_key());
    assert_eq!(back.service(), Service::Vector);
    assert_eq!(back.collection_uid(), uid(0));
    assert_eq!(back.shard(), shard);
}

#[test]
fn identity_check_outcomes() {
    let a = DoMeta::expected(&ctx("org-a"), Service::Vector, uid(0), ShardIndex::ZERO);
    assert_eq!(
        DoMeta::check(Some(&a), &a, false),
        Ok(IdentityCheck::Matched)
    );
    assert_eq!(
        DoMeta::check(Some(&a), &a, true),
        Ok(IdentityCheck::Matched)
    );
    assert_eq!(
        DoMeta::check(None, &a, true),
        Ok(IdentityCheck::InitializeOnWrite)
    );
    assert_eq!(DoMeta::check(None, &a, false), Ok(IdentityCheck::EmptyRead));
}

#[test]
fn foreign_tenant_is_404_indistinguishable_from_missing() {
    let stored = DoMeta::expected(&ctx("org-a"), Service::Vector, uid(0), ShardIndex::ZERO);
    let intruder = DoMeta::expected(&ctx("org-b"), Service::Vector, uid(0), ShardIndex::ZERO);
    for write in [false, true] {
        let err = DoMeta::check(Some(&stored), &intruder, write).unwrap_err();
        assert_eq!(err, TenancyError::NotFound);
        assert_eq!(err.problem_code(), ProblemCode::NotFound);
    }
    // Any single-field mismatch is also 404.
    let c = ctx("org-a");
    let s1 = ShardIndex::new(1, ShardCount::new(2).unwrap()).unwrap();
    for other in [
        DoMeta::expected(&c, Service::Quant, uid(0), ShardIndex::ZERO),
        DoMeta::expected(&c, Service::Vector, uid(1), ShardIndex::ZERO),
        DoMeta::expected(&c, Service::Vector, uid(0), s1),
    ] {
        assert_eq!(
            DoMeta::check(Some(&stored), &other, false),
            Err(TenancyError::NotFound)
        );
    }
}

#[test]
fn do_meta_from_kv_fails_closed() {
    assert_eq!(DoMeta::from_kv(Vec::<(&str, &str)>::new()), Ok(None));
    assert_eq!(DoMeta::from_kv([("unrelated", "x")]), Ok(None));
    let m = DoMeta::expected(&ctx("org-a"), Service::Graph, uid(3), ShardIndex::ZERO);
    let kv = m.to_kv();
    let rows: Vec<(&str, &str)> = kv.iter().map(|(k, v)| (*k, v.as_str())).collect();
    // Partial identity.
    assert!(DoMeta::from_kv(rows[..3].to_vec()).is_err());
    // Duplicate key.
    let mut dup = rows.clone();
    dup.push((META_SHARD, "0"));
    assert!(DoMeta::from_kv(dup).is_err());
    // Malformed fields.
    for (key, bad) in [
        (META_TENANT_KEY, "NOT-A-KEY"),
        (META_SERVICE, "Vector"),
        (META_COLLECTION_UID, "zz"),
        (META_SHARD, "7"),
    ] {
        let tampered: Vec<(&str, &str)> = rows
            .iter()
            .map(|(k, v)| if *k == key { (*k, bad) } else { (*k, *v) })
            .collect();
        let err = DoMeta::from_kv(tampered).unwrap_err();
        assert!(matches!(err, TenancyError::MalformedIdentifier(_)));
        assert_eq!(err.problem_code(), ProblemCode::ServerError);
    }
}

#[test]
fn ledger_meta_identity() {
    let a = LedgerMeta::expected(&ctx("org-a"));
    let b = LedgerMeta::expected(&ctx("org-b"));
    assert_eq!(a.do_name(), ledger_do_name(ctx("org-a").tenant_key()));
    let kv = a.to_kv();
    let back = LedgerMeta::from_kv(kv.iter().map(|(k, v)| (*k, v.as_str())))
        .unwrap()
        .unwrap();
    assert_eq!(back, a);
    assert_eq!(LedgerMeta::from_kv([("x", "y")]), Ok(None));
    assert!(LedgerMeta::from_kv([(META_TENANT_KEY, "bad")]).is_err());
    let t = a.tenant_key().as_str().to_string();
    assert!(
        LedgerMeta::from_kv([(META_TENANT_KEY, t.as_str()), (META_TENANT_KEY, t.as_str())])
            .is_err()
    );
    assert_eq!(
        LedgerMeta::check(Some(&a), &a, false),
        Ok(IdentityCheck::Matched)
    );
    assert_eq!(
        LedgerMeta::check(Some(&a), &b, true),
        Err(TenancyError::NotFound)
    );
    assert_eq!(
        LedgerMeta::check(None, &a, true),
        Ok(IdentityCheck::InitializeOnWrite)
    );
    assert_eq!(
        LedgerMeta::check(None, &a, false),
        Ok(IdentityCheck::EmptyRead)
    );
}
