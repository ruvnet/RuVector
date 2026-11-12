//! Upload sessions: the happy path, digest / size / validation refusals,
//! immutability (also under races and after yanking), yank semantics,
//! session privacy, part checks and expiry.

mod common;

use common::*;
use ruvector_edge_registry::keys::blob_key;
use ruvector_edge_registry::registry::PageRequest;
use ruvector_edge_registry::upload::{UploadId, UploadState};
use ruvector_edge_registry::validate::{validate, ValidationError, ValidationLimits};
use ruvector_edge_registry::{BeginUpload, PartRecord, RegistryError, Visibility};

#[test]
fn happy_path_records_manifest_blob_and_scope() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(1);
    let m = push_with(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Private,
        &bytes,
        3,
        Declare::default(),
    )
    .unwrap();
    assert_eq!(
        (m.dim, m.total_size, m.sha256),
        (4, bytes.len() as u64, sha(&bytes))
    );
    assert_eq!(m.created_by, "es1_alice");
    assert_eq!(m.segments.len(), 2);
    let t = reg.pull(&a, &name("@acme/pkg"), &ver("1.0.0")).unwrap();
    let n = name("@acme/pkg");
    assert_eq!(
        t.blob,
        blob_key(&a.tenant, n.scope(), Visibility::Private, &m.sha256)
    );
    assert!(t
        .blob
        .as_str()
        .starts_with(&format!("rvf/{}/acme/blobs/sha256/", a.tenant.as_str())));
    assert_eq!(r2.get(t.blob.as_str()).unwrap(), bytes);
    assert_eq!(reg.blob_ref(&t.blob).unwrap().unwrap().refs, 1);
    // The manifest serializes and parses back identically.
    let j = serde_json::to_string(&m).unwrap();
    assert_eq!(
        serde_json::from_str::<ruvector_edge_registry::PackageManifest>(&j).unwrap(),
        m
    );
}

#[test]
fn digest_mismatch_is_refused_and_terminal() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(1);
    let lie = Declare {
        sha256: Some(sha(b"something else")),
        size: None,
    };
    let e = push_with(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &bytes,
        2,
        lie,
    )
    .unwrap_err();
    assert_eq!(e, RegistryError::DigestMismatch);
    assert_eq!(e.http_status(), 400);
    assert_eq!(
        reg.get(&a, &name("@acme/pkg"), &ver("1.0.0")).unwrap_err(),
        RegistryError::NotFound
    );
    // Nothing was written to the declared blob key; the pin was released.
    assert!(!r2.objects.borrow().keys().any(|k| k.contains("/blobs/")));
    // Scopes are claimed explicitly, so another tenant still cannot push.
    let other = caller('b', "es1_carol", &ALL_CAPS);
    assert_eq!(
        push(
            &reg,
            &r2,
            &other,
            "@acme/pkg",
            "1.0.0",
            Visibility::Tenant,
            &bytes
        )
        .unwrap_err(),
        RegistryError::NotOwner
    );
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
}

#[test]
fn failed_session_cannot_be_retried() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(1);
    let id = begin(&reg, &a, "1.0.0", &bytes);
    reg.record_part(
        &a,
        &id,
        PartRecord {
            number: 1,
            size: bytes.len() as u64,
            sha256: sha(&bytes),
        },
    )
    .unwrap();
    let wrong = validate(&small_store(2), ValidationLimits::default());
    assert_eq!(
        finalize_raw(&reg, &a, &id, wrong).unwrap_err(),
        RegistryError::DigestMismatch
    );
    let s = reg.upload(&a, &id).unwrap();
    assert_eq!(
        s.state,
        UploadState::Failed {
            reason: "digest_mismatch".into()
        }
    );
    let right = validate(&bytes, ValidationLimits::default());
    assert_eq!(
        finalize_raw(&reg, &a, &id, right).unwrap_err(),
        RegistryError::UploadNotOpen
    );
    let _ = r2;
}

#[test]
fn size_mismatch_and_invalid_rvf_fail_the_session() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(1);
    let short = Declare {
        sha256: None,
        size: Some(bytes.len() as u64 + 1),
    };
    // Parts cannot sum to a size that was not uploaded.
    let e = push_with(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &bytes,
        1,
        short,
    )
    .unwrap_err();
    assert!(matches!(e, RegistryError::PartsIncomplete(_)));
    let mut junk = bytes.clone();
    junk[0] ^= 0xff;
    let e = push(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &junk,
    )
    .unwrap_err();
    assert_eq!(
        e,
        RegistryError::Validation(ValidationError::BadMagic { offset: 0 })
    );
    let id = begin(&reg, &a, "1.0.0", &bytes);
    reg.record_part(
        &a,
        &id,
        PartRecord {
            number: 1,
            size: bytes.len() as u64,
            sha256: sha(&bytes),
        },
    )
    .unwrap();
    let mut other = validate(&bytes, ValidationLimits::default()).unwrap();
    other.total_size += 1;
    assert_eq!(
        finalize_raw(&reg, &a, &id, Ok(other)).unwrap_err(),
        RegistryError::SizeMismatch
    );
}

#[test]
fn versions_are_immutable_even_when_yanked_or_raced() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let (b1, b2) = (small_store(1), small_store(2));
    let (x, y) = (begin(&reg, &a, "1.0.0", &b1), begin(&reg, &a, "1.0.0", &b2));
    for (id, b) in [(&x, &b1), (&y, &b2)] {
        reg.record_part(
            &a,
            id,
            PartRecord {
                number: 1,
                size: b.len() as u64,
                sha256: sha(b),
            },
        )
        .unwrap();
    }
    let first = finalize_raw(&reg, &a, &x, validate(&b1, ValidationLimits::default())).unwrap();
    let e = finalize_raw(&reg, &a, &y, validate(&b2, ValidationLimits::default())).unwrap_err();
    assert_eq!(e, RegistryError::VersionExists);
    assert_eq!(e.http_status(), 409);
    let n = name("@acme/pkg");
    reg.yank(&a, &n, &ver("1.0.0"), "broken").unwrap();
    assert_eq!(
        push(&reg, &r2, &a, "@acme/pkg", "1.0.0", Visibility::Tenant, &b2).unwrap_err(),
        RegistryError::VersionExists
    );
    let now = reg.get(&a, &n, &ver("1.0.0")).unwrap();
    assert_eq!(now.content.sha256, first.sha256);
    assert_eq!(now.content.segments, first.segments);
    assert!(now.yanked.is_some());
}

#[test]
fn yank_semantics() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let n = name("@acme/pkg");
    for (i, v) in ["1.0.0", "1.1.0", "2.0.0-rc.1"].iter().enumerate() {
        push(
            &reg,
            &r2,
            &a,
            "@acme/pkg",
            v,
            Visibility::Tenant,
            &small_store(i as u64),
        )
        .unwrap();
    }
    assert_eq!(
        reg.resolve_latest(&a, &n, false).unwrap().version,
        ver("1.1.0")
    );
    assert_eq!(
        reg.resolve_latest(&a, &n, true).unwrap().version,
        ver("2.0.0-rc.1")
    );
    let y = reg.yank(&a, &n, &ver("1.1.0"), "regression").unwrap();
    let again = reg.yank(&a, &n, &ver("1.1.0"), "different").unwrap();
    assert_eq!(
        again.yanked, y.yanked,
        "yank is idempotent and keeps the first record"
    );
    assert_eq!(
        reg.resolve_latest(&a, &n, false).unwrap().version,
        ver("1.0.0")
    );
    // Still pullable by exact version, flagged.
    assert!(reg
        .pull(&a, &n, &ver("1.1.0"))
        .unwrap()
        .manifest
        .yanked
        .is_some());
    let page = reg.list_versions(&a, &n, &PageRequest::default()).unwrap();
    assert!(page
        .items
        .iter()
        .any(|s| s.version == ver("1.1.0") && s.yanked));
    reg.unyank(&a, &n, &ver("1.1.0")).unwrap();
    assert_eq!(
        reg.resolve_latest(&a, &n, false).unwrap().version,
        ver("1.1.0")
    );
    assert!(matches!(
        reg.yank(&a, &n, &ver("1.1.0"), &"x".repeat(201)),
        Err(RegistryError::InvalidRequest(_))
    ));
    assert!(matches!(
        reg.yank(&a, &n, &ver("1.1.0"), "a\nb"),
        Err(RegistryError::InvalidRequest(_))
    ));
    // A yanked version cannot be made public.
    reg.yank(&a, &n, &ver("1.0.0"), "").unwrap();
    assert!(matches!(
        reg.publish_plan(&a, &n, &ver("1.0.0")),
        Err(RegistryError::InvalidRequest(_))
    ));
}

#[test]
fn sessions_are_private_expire_and_check_parts() {
    let (reg, _, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(1);
    let id = begin(&reg, &a, "1.0.0", &bytes);
    let bob = caller('a', "es1_bob", &ALL_CAPS);
    let carol = caller('b', "es1_carol", &ALL_CAPS);
    assert_eq!(reg.upload(&bob, &id).unwrap_err(), RegistryError::NotFound);
    assert_eq!(
        reg.upload(&carol, &id).unwrap_err(),
        RegistryError::NotFound
    );
    let part = |n, size| PartRecord {
        number: n,
        size,
        sha256: [0; 32],
    };
    assert!(matches!(
        reg.record_part(&a, &id, part(0, 1)),
        Err(RegistryError::UploadLimit(_))
    ));
    assert!(matches!(
        reg.record_part(&a, &id, part(1, 0)),
        Err(RegistryError::UploadLimit(_))
    ));
    assert!(matches!(
        reg.record_part(&a, &id, part(1, bytes.len() as u64 + 1)),
        Err(RegistryError::UploadLimit(_))
    ));
    reg.record_part(&a, &id, part(2, 10)).unwrap();
    assert!(matches!(
        reg.finalize_plan(&a, &id),
        Err(RegistryError::PartsIncomplete(_))
    ));
    reg.record_part(&a, &id, part(1, bytes.len() as u64 - 10))
        .unwrap();
    assert_eq!(reg.finalize_plan(&a, &id).unwrap().parts.len(), 2);
    let id2 = begin(&reg, &a, "1.0.1", &bytes);
    assert_eq!(
        reg.abort_upload(&a, &id2).unwrap().state,
        UploadState::Aborted
    );
    assert_eq!(
        reg.abort_upload(&a, &id2).unwrap_err(),
        RegistryError::UploadNotOpen
    );
    assert!(UploadId::parse("up_../../etc").is_err());
    assert!(matches!(
        reg.begin_upload(
            &a,
            BeginUpload {
                name: name("@acme/p"),
                version: ver("1.0.0"),
                visibility: Visibility::Public,
                size: 1,
                sha256: [0; 32]
            }
        ),
        Err(RegistryError::InvalidRequest(_))
    ));
}

#[test]
fn expiry_uses_the_clock_port() {
    use ruvector_edge_registry::mem::{CounterEntropy, FixedClock, MemKv};
    let clock = FixedClock::at(T0);
    let reg = ruvector_edge_registry::Registry::new(
        MemKv::new(),
        &clock,
        CounterEntropy::default(),
        config(),
    );
    let a = alice();
    reg.adopt_scope(&claim_for(&a, "acme")).unwrap();
    let bytes = small_store(1);
    let s = reg
        .begin_upload(
            &a,
            BeginUpload {
                name: name("@acme/pkg"),
                version: ver("1.0.0"),
                visibility: Visibility::Tenant,
                size: bytes.len() as u64,
                sha256: sha(&bytes),
            },
        )
        .unwrap();
    assert_eq!(s.expires_at, T0 + 24 * 3600);
    reg.record_part(
        &a,
        &s.id,
        PartRecord {
            number: 1,
            size: bytes.len() as u64,
            sha256: sha(&bytes),
        },
    )
    .unwrap();
    clock.advance(24 * 3600);
    let e = reg.finalize_plan(&a, &s.id).unwrap_err();
    assert_eq!(e, RegistryError::UploadExpired);
    assert_eq!(e.http_status(), 404);
    let part = PartRecord {
        number: 2,
        size: 1,
        sha256: [0; 32],
    };
    assert_eq!(
        reg.record_part(&a, &s.id, part).unwrap_err(),
        RegistryError::UploadExpired
    );
    assert_eq!(
        reg.finalize_plan(&a, &s.id).unwrap_err(),
        RegistryError::UploadExpired
    );
}
