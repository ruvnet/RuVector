//! Blob integrity regressions: the refcount domain equals the dedupe domain,
//! pins close the plan/finalize window, sessions freeze at plan time, and
//! public objects are write-once and evidence-checked.

mod common;

use common::*;
use ruvector_edge_registry::mem::{CounterEntropy, FixedClock, MemKv};
use ruvector_edge_registry::upload::UploadState;
use ruvector_edge_registry::validate::{validate, ValidationLimits};
use ruvector_edge_registry::{
    BeginUpload, FinalizeReport, ObjectEvidence, PartRecord, Registry, RegistryError, Visibility,
};

/// Two scopes of one tenant are two registry DOs (two stores). Publishing
/// one scope's version must not release the other scope's bytes.
#[test]
fn two_scopes_of_one_tenant_never_share_a_counted_blob() {
    let (r2, a) = (FakeR2::default(), alice());
    let (reg_a, reg_b) = (registry(), registry());
    let bytes = small_store(4);
    let ma = push(
        &reg_a,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &bytes,
    )
    .unwrap();
    let mb = push(
        &reg_b,
        &r2,
        &a,
        "@beta/pkg",
        "1.0.0",
        Visibility::Private,
        &bytes,
    )
    .unwrap();
    assert_eq!(ma.sha256, mb.sha256);
    let ta = reg_a.pull(&a, &name("@acme/pkg"), &ver("1.0.0")).unwrap();
    let tb = reg_b.pull(&a, &name("@beta/pkg"), &ver("1.0.0")).unwrap();
    assert_ne!(ta.blob, tb.blob, "the dedupe domain is one scope");
    let out = publish(&reg_a, &r2, &a, "@acme/pkg", "1.0.0").unwrap();
    assert_eq!(out.queued_for_gc, Some(ta.blob.clone()));
    for k in reg_a.sweep(None, 100).unwrap().delete {
        r2.delete(k.as_str());
    }
    assert_eq!(
        r2.get(tb.blob.as_str()).unwrap(),
        bytes,
        "@beta keeps its bytes"
    );
    assert_eq!(reg_b.blob_ref(&tb.blob).unwrap().unwrap().refs, 1);
    assert!(reg_b.sweep(None, 100).unwrap().delete.is_empty());
}

/// Plan says "stored, skip the copy"; a concurrent publish drops the only
/// version reference. The pin keeps the blob alive through finalize.
#[test]
fn a_pinned_blob_survives_a_concurrent_release() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(5);
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
    let id = begin(&reg, &a, "1.0.1", &bytes);
    stage(&reg, &r2, &a, &id, &bytes, 1);
    let plan = reg.finalize_plan(&a, &id).unwrap();
    assert!(!plan.copy_required);
    let out = publish(&reg, &r2, &a, "@acme/pkg", "1.0.0").unwrap();
    assert_eq!(out.queued_for_gc, None, "the pin still references the blob");
    assert!(reg
        .sweep(None, 100)
        .unwrap()
        .delete
        .iter()
        .all(|k| *k != plan.blob));
    finish(&reg, &r2, &a, &id).unwrap();
    assert_eq!(r2.get(plan.blob.as_str()).unwrap(), bytes);
    assert_eq!(reg.blob_ref(&plan.blob).unwrap().unwrap().refs, 1);
}

/// Two in-flight uploads of new bytes: a pin alone is not "stored", so both
/// are told to write the blob.
#[test]
fn concurrent_first_uploads_both_copy() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(6);
    let (x, y) = (
        begin(&reg, &a, "1.0.0", &bytes),
        begin(&reg, &a, "1.0.1", &bytes),
    );
    stage(&reg, &r2, &a, &x, &bytes, 1);
    stage(&reg, &r2, &a, &y, &bytes, 1);
    assert!(reg.finalize_plan(&a, &x).unwrap().copy_required);
    assert!(reg.finalize_plan(&a, &y).unwrap().copy_required);
    finish(&reg, &r2, &a, &x).unwrap();
    finish(&reg, &r2, &a, &y).unwrap();
    assert_eq!(
        reg.finalize_plan(&a, &x).unwrap_err(),
        RegistryError::UploadNotOpen,
        "finalized sessions stay closed"
    );
}

#[test]
fn finalize_plan_freezes_the_session() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(7);
    let id = begin(&reg, &a, "1.0.0", &bytes);
    stage(&reg, &r2, &a, &id, &bytes, 2);
    let plan = reg.finalize_plan(&a, &id).unwrap();
    assert!(matches!(
        reg.upload(&a, &id).unwrap().state,
        UploadState::Finalizing { .. }
    ));
    assert_eq!(reg.finalize_plan(&a, &id).unwrap(), plan, "idempotent");
    let swap = PartRecord {
        number: 2,
        size: plan.parts[1].size,
        sha256: [9; 32],
    };
    assert_eq!(
        reg.record_part(&a, &id, swap.clone()).unwrap_err(),
        RegistryError::UploadNotOpen
    );
    assert_eq!(
        reg.abort_upload(&a, &id).unwrap_err(),
        RegistryError::UploadNotOpen
    );
    // Bytes that are not the frozen parts: terminal, pin released.
    let mut streamed = plan.parts.clone();
    streamed[1] = swap;
    let e = reg
        .finalize(
            &a,
            &id,
            FinalizeReport {
                validated: validate(&bytes, ValidationLimits::default()),
                streamed,
                copied: Some(evidence(&bytes)),
                provenance: None,
            },
        )
        .unwrap_err();
    assert_eq!(e, RegistryError::PartsChanged);
    assert_eq!(
        reg.upload(&a, &id).unwrap().state,
        UploadState::Failed {
            reason: "parts_changed".into()
        }
    );
    assert_eq!(reg.blob_ref(&plan.blob).unwrap(), None);
    assert!(reg.sweep(None, 100).unwrap().delete.contains(&plan.blob));
}

#[test]
fn copy_evidence_is_required_and_checked() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(8);
    for (v, copied) in [
        ("1.0.0", None),
        ("1.0.1", Some(evidence(b"not the object"))),
        (
            "1.0.2",
            Some(ObjectEvidence {
                sha256: sha(&bytes),
                size: 1,
            }),
        ),
    ] {
        let id = begin(&reg, &a, v, &bytes);
        stage(&reg, &r2, &a, &id, &bytes, 1);
        let plan = reg.finalize_plan(&a, &id).unwrap();
        assert!(plan.copy_required);
        let e = reg
            .finalize(
                &a,
                &id,
                FinalizeReport {
                    validated: validate(&bytes, ValidationLimits::default()),
                    streamed: plan.parts,
                    copied,
                    provenance: None,
                },
            )
            .unwrap_err();
        assert_eq!(e, RegistryError::EvidenceMismatch, "{v}");
    }
}

/// A Worker that froze a session and never came back: the sweep fails it,
/// releases its pin and hands out the staging object and the orphan blob.
#[test]
fn expired_finalizing_sessions_release_their_pins() {
    let clock = FixedClock::at(T0);
    let reg = Registry::new(MemKv::new(), &clock, CounterEntropy::default(), config());
    let a = alice();
    reg.adopt_scope(&claim_for(&a, "acme")).unwrap();
    let bytes = small_store(9);
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
    let part = PartRecord {
        number: 1,
        size: bytes.len() as u64,
        sha256: sha(&bytes),
    };
    reg.record_part(&a, &s.id, part.clone()).unwrap();
    let plan = reg.finalize_plan(&a, &s.id).unwrap();
    assert_eq!(reg.blob_ref(&plan.blob).unwrap().unwrap().refs, 1);
    assert!(
        reg.sweep(None, 100).unwrap().delete.is_empty(),
        "not yet expired"
    );
    clock.advance(24 * 3600);
    let report = reg.sweep(None, 100).unwrap();
    assert_eq!(report.expired, vec![s.id.clone()]);
    assert!(report.delete.contains(&plan.staging));
    assert!(report.delete.contains(&plan.blob));
    assert_eq!(reg.blob_ref(&plan.blob).unwrap(), None);
    let e = reg
        .finalize(
            &a,
            &s.id,
            FinalizeReport {
                validated: validate(&bytes, ValidationLimits::default()),
                streamed: vec![part],
                copied: Some(evidence(&bytes)),
                provenance: None,
            },
        )
        .unwrap_err();
    assert_eq!(e, RegistryError::UploadNotOpen);
    // Terminal and expired: the next sweep drops the session record.
    reg.sweep(None, 100).unwrap();
    assert_eq!(reg.upload(&a, &s.id).unwrap_err(), RegistryError::NotFound);
}

/// The public key is shared by every tenant: never overwritten, and a
/// public version is only committed against a measured, matching object.
#[test]
fn public_objects_are_write_once_and_evidence_checked() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(10);
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
    let (n, v) = (name("@acme/pkg"), ver("1.0.0"));
    let plan = reg.publish_plan(&a, &n, &v).unwrap();
    assert_eq!((plan.sha256, plan.size), (sha(&bytes), bytes.len() as u64));
    // A poisoned object already sits at the public key.
    r2.put(plan.to.as_str(), b"poison".to_vec());
    let e = publish(&reg, &r2, &a, "@acme/pkg", "1.0.0").unwrap_err();
    assert_eq!(e, RegistryError::EvidenceMismatch);
    assert_eq!(r2.get(plan.to.as_str()).unwrap(), b"poison", "create-only");
    let m = reg.get(&a, &n, &v).unwrap();
    assert_eq!(m.visibility, Visibility::Tenant, "still not public");
    // With the correct object in place the commit goes through; the public
    // blob is never refcounted by a scope.
    r2.put(plan.to.as_str(), bytes.clone());
    publish(&reg, &r2, &a, "@acme/pkg", "1.0.0").unwrap();
    assert_eq!(reg.get(&a, &n, &v).unwrap().visibility, Visibility::Public);
    assert_eq!(reg.blob_ref(&plan.to).unwrap(), None);
    let other = caller('b', "es1_carol", &ALL_CAPS);
    let t = reg.pull(&other, &n, &v).unwrap();
    assert_eq!(r2.get(t.blob.as_str()).unwrap(), bytes);
}
