//! Dedupe domains, public publish, cross-tenant visibility, paginated
//! listings with bound cursors, scope ownership and version caps.

mod common;

use common::*;
use ruvector_edge_auth::Capability;
use ruvector_edge_registry::keys::blob_key;
use ruvector_edge_registry::registry::PageRequest;
use ruvector_edge_registry::{BeginUpload, RegistryError, Visibility};

#[test]
fn dedupe_is_per_scope_for_private_and_global_for_public() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let carol = caller('b', "es1_carol", &ALL_CAPS);
    let bytes = small_store(7);
    push(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &bytes,
    )
    .unwrap();
    let acme = name("@acme/pkg");
    let key_a = blob_key(&a.tenant, acme.scope(), Visibility::Tenant, &sha(&bytes));
    assert!(key_a.as_str().contains("/acme/blobs/sha256/"));
    // Same scope, same bytes: the plan says the bytes are stored.
    let id = begin(&reg, &a, "1.0.1", &bytes);
    stage(&reg, &r2, &a, &id, &bytes, 1);
    let plan = reg.finalize_plan(&a, &id).unwrap();
    assert!(!plan.copy_required && plan.blob == key_a);
    finish(&reg, &r2, &a, &id).unwrap();
    assert_eq!(reg.blob_ref(&key_a).unwrap().unwrap().refs, 2);
    // Another tenant (another scope), same bytes: its own key, no oracle.
    adopt(&reg, &carol, "@other/pkg");
    let id = reg
        .begin_upload(
            &carol,
            BeginUpload {
                name: name("@other/pkg"),
                version: ver("1.0.0"),
                visibility: Visibility::Private,
                size: bytes.len() as u64,
                sha256: sha(&bytes),
            },
        )
        .unwrap()
        .id;
    stage(&reg, &r2, &carol, &id, &bytes, 1);
    let plan = reg.finalize_plan(&carol, &id).unwrap();
    assert!(plan.copy_required && plan.blob != key_a);
    // Publishing both versions: one global public object, never counted
    // per scope; the tenant blob is queued once nothing references it.
    let p1 = reg.publish_plan(&a, &acme, &ver("1.0.0")).unwrap();
    assert!(p1.to.as_str().starts_with("public/blobs/sha256/"));
    let out = publish(&reg, &r2, &a, "@acme/pkg", "1.0.0").unwrap();
    assert_eq!(
        out.queued_for_gc, None,
        "1.0.1 still references the tenant blob"
    );
    let out = publish(&reg, &r2, &a, "@acme/pkg", "1.0.1").unwrap();
    assert_eq!(out.queued_for_gc, Some(key_a.clone()));
    assert_eq!(reg.blob_ref(&p1.to).unwrap(), None);
    // Idempotent.
    let out = publish(&reg, &r2, &a, "@acme/pkg", "1.0.1").unwrap();
    assert_eq!(out.queued_for_gc, None);
    // The sweep hands out the released tenant blob exactly once.
    let swept = reg.sweep(None, 100).unwrap();
    assert!(swept.delete.contains(&key_a));
    assert!(!reg.sweep(None, 100).unwrap().delete.contains(&key_a));
}

#[test]
fn public_packages_are_pullable_and_listable_across_tenants() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let carol = caller('b', "es1_carol", &[Capability::Read]);
    push(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &small_store(1),
    )
    .unwrap();
    push(
        &reg,
        &r2,
        &a,
        "@acme/secret",
        "1.0.0",
        Visibility::Private,
        &small_store(2),
    )
    .unwrap();
    let n = name("@acme/pkg");
    let page = reg
        .list_packages(&carol, "@acme/", &PageRequest::default())
        .unwrap();
    assert!(page.items.is_empty());
    assert_eq!(
        reg.list_packages(&a, "@acme/", &PageRequest::default())
            .unwrap()
            .items
            .len(),
        2
    );
    publish(&reg, &r2, &a, "@acme/pkg", "1.0.0").unwrap();
    let page = reg
        .list_packages(&carol, "@acme/", &PageRequest::default())
        .unwrap();
    assert_eq!(
        page.items
            .iter()
            .map(|p| p.name.to_string())
            .collect::<Vec<_>>(),
        vec!["@acme/pkg"]
    );
    let t = reg.pull(&carol, &n, &ver("1.0.0")).unwrap();
    assert!(t.blob.as_str().starts_with("public/"));
    // The outsider can read but never write.
    let rw = caller('b', "es1_carol", &ALL_CAPS);
    assert_eq!(
        push(
            &reg,
            &r2,
            &rw,
            "@acme/pkg",
            "1.0.1",
            Visibility::Tenant,
            &small_store(3)
        )
        .unwrap_err(),
        RegistryError::NotOwner
    );
    assert_eq!(
        reg.yank(&rw, &n, &ver("1.0.0"), "").unwrap_err(),
        RegistryError::NotOwner
    );
    assert!(reg
        .list_packages(&caller('b', "x", &[]), "@acme/", &PageRequest::default())
        .is_err());
}

#[test]
fn version_listing_paginates_in_semver_order_with_bound_cursors() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let n = name("@acme/pkg");
    let mut all = Vec::new();
    for i in 0..25u64 {
        let v = format!("1.{}.{}", i % 5, i / 5);
        let vis = if i % 7 == 0 {
            Visibility::Private
        } else {
            Visibility::Tenant
        };
        push(&reg, &r2, &a, "@acme/pkg", &v, vis, &small_store(i)).unwrap();
        all.push((ver(&v), vis));
    }
    push(
        &reg,
        &r2,
        &a,
        "@acme/other",
        "1.0.0",
        Visibility::Tenant,
        &small_store(99),
    )
    .unwrap();
    let bob = caller('a', "es1_bob", &[Capability::Read]);
    let mut seen = Vec::new();
    let mut req = PageRequest {
        cursor: None,
        limit: 10,
    };
    loop {
        let page = reg.list_versions(&bob, &n, &req).unwrap();
        assert!(page.items.len() <= 10);
        seen.extend(page.items.iter().map(|s| s.version.clone()));
        match page.next_cursor {
            Some(c) => req.cursor = Some(c),
            None => break,
        }
    }
    let mut want: Vec<_> = all
        .iter()
        .filter(|(_, v)| *v != Visibility::Private)
        .map(|(v, _)| v.clone())
        .collect();
    want.sort_by(|x, y| y.cmp(x));
    assert_eq!(
        seen, want,
        "bob sees tenant versions only, newest first, no repeats"
    );
    // Cursors are bound to their listing.
    let c = reg
        .list_versions(
            &a,
            &n,
            &PageRequest {
                cursor: None,
                limit: 3,
            },
        )
        .unwrap()
        .next_cursor
        .unwrap();
    let other = name("@acme/other");
    assert_eq!(
        reg.list_versions(
            &a,
            &other,
            &PageRequest {
                cursor: Some(c.clone()),
                limit: 3
            }
        )
        .unwrap_err(),
        RegistryError::InvalidCursor
    );
    assert_eq!(
        reg.list_packages(
            &a,
            "@acme/",
            &PageRequest {
                cursor: Some(c),
                limit: 3
            }
        )
        .unwrap_err(),
        RegistryError::InvalidCursor
    );
    for bad in ["", "c1.zz", "c2.00", "c1.00"] {
        assert_eq!(
            reg.list_versions(
                &a,
                &n,
                &PageRequest {
                    cursor: Some(bad.into()),
                    limit: 3
                }
            )
            .unwrap_err(),
            RegistryError::InvalidCursor,
            "{bad}"
        );
    }
}

#[test]
fn package_listing_paginates_and_validates_prefixes() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    for i in 0..12 {
        push(
            &reg,
            &r2,
            &a,
            &format!("@acme/emb-{i:02}"),
            "1.0.0",
            Visibility::Tenant,
            &small_store(i),
        )
        .unwrap();
    }
    push(
        &reg,
        &r2,
        &a,
        "@acme/zzz",
        "1.0.0",
        Visibility::Tenant,
        &small_store(50),
    )
    .unwrap();
    let mut names = Vec::new();
    let mut req = PageRequest {
        cursor: None,
        limit: 5,
    };
    loop {
        let page = reg.list_packages(&a, "@acme/emb", &req).unwrap();
        names.extend(page.items.into_iter().map(|p| p.name.to_string()));
        match page.next_cursor {
            Some(c) => req.cursor = Some(c),
            None => break,
        }
    }
    assert_eq!(
        names,
        (0..12)
            .map(|i| format!("@acme/emb-{i:02}"))
            .collect::<Vec<_>>()
    );
    for bad in [
        "acme/",
        "@acme",
        "@ACME/",
        "@acme/Emb",
        "@acme/a/b",
        "@ruvector/",
    ] {
        assert!(
            reg.list_packages(&a, bad, &PageRequest::default()).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn foreign_scopes_are_refused_and_version_caps_hold() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let carol = caller('b', "es1_carol", &ALL_CAPS);
    push(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &small_store(1),
    )
    .unwrap();
    assert_eq!(
        push(
            &reg,
            &r2,
            &carol,
            "@acme/new",
            "1.0.0",
            Visibility::Tenant,
            &small_store(2)
        )
        .unwrap_err(),
        RegistryError::NotOwner
    );
    let writer_without_read = caller('a', "es1_dan", &[Capability::Write]);
    push(
        &reg,
        &r2,
        &writer_without_read,
        "@acme/dan",
        "1.0.0",
        Visibility::Private,
        &small_store(3),
    )
    .unwrap();
    let mut cfg = config();
    cfg.max_versions_per_package = 2;
    let reg = ruvector_edge_registry::Registry::new(
        ruvector_edge_registry::mem::MemKv::new(),
        ruvector_edge_registry::mem::FixedClock::at(T0),
        ruvector_edge_registry::mem::CounterEntropy::default(),
        cfg,
    );
    push(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &small_store(1),
    )
    .unwrap();
    push(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.1",
        Visibility::Tenant,
        &small_store(2),
    )
    .unwrap();
    let e = push(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.2",
        Visibility::Tenant,
        &small_store(3),
    )
    .unwrap_err();
    assert_eq!(e, RegistryError::TooManyVersions);
}
