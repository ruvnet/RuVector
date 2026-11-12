//! Content addressing and the five release-publication rules.

mod common;

use common::{Fixture, NOW};
use rvforge_registry::objects::{
    CapabilityManifest, Release, SignatureRole, SubjectType, TrustLevel,
};
use rvforge_registry::{key_id_for, testkit, ExecutionStatus, Registry, RegistryErrorCode};

// --- content addressing ---

#[test]
fn content_address_round_trips_through_the_store() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");

    let fetched: Release = fx.registry.store().get(&id).unwrap();
    assert_eq!(fetched.release_id, id);
    assert_eq!(
        rvforge_registry::compute_id(&fetched).unwrap(),
        id,
        "a stored object must still hash to the name it is filed under"
    );
}

#[test]
fn two_releases_differing_in_one_field_get_different_ids() {
    let fx = Fixture::new();
    let a = fx.draft("example-agent", "1.0.0");
    let mut b = a.clone();
    b.rvf_size += 1;
    assert_ne!(
        rvforge_registry::compute_id(&a).unwrap(),
        rvforge_registry::compute_id(&b).unwrap()
    );
}

#[test]
fn a_declared_id_that_does_not_match_the_content_is_refused() {
    let fx = Fixture::new();
    let mut release = fx.signed("example-agent", "1.0.0");
    release.release_id = format!("sha256:{}", "0".repeat(64));

    let err = fx.registry.publish_release(release, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::IdMismatch);
}

#[test]
fn a_countersignature_does_not_change_the_release_id() {
    let fx = Fixture::new();
    let published = fx.publish("example-agent", "1.0.0");

    let release: Release = fx.registry.store().get(&published).unwrap();
    let countersigned = testkit::sign(release, &fx.registry_key, SignatureRole::Registry).unwrap();
    let stored = fx.registry.store().put(countersigned).unwrap();

    assert_eq!(stored.release_id, published);
    assert_eq!(stored.signatures.len(), 2);
}

// --- the publish happy path ---

#[test]
fn publishing_a_well_formed_release_succeeds() {
    let fx = Fixture::new();
    let outcome = fx
        .registry
        .publish_release(fx.signed("example-agent", "1.0.0"), NOW)
        .unwrap();

    let id = &outcome.release.release_id;
    assert!(fx.registry.is_published(id).unwrap());
    assert!(fx.registry.is_executable(id).unwrap());
    assert_eq!(
        fx.registry.effective_trust_level(id).unwrap(),
        TrustLevel::Published
    );
    assert_eq!(outcome.entry.object, *id);
    assert_eq!(outcome.receipt.subject, *id);

    let index = fx
        .registry
        .releases_for(&fx.publisher.publisher_id, "example-agent")
        .unwrap();
    assert_eq!(index.len(), 1);
    assert_eq!(index[0].version, "1.0.0");
}

#[test]
fn a_release_only_in_the_store_is_not_published() {
    let fx = Fixture::new();
    let stored = fx
        .registry
        .store()
        .put(fx.signed("example-agent", "1.0.0"))
        .unwrap();

    assert!(!fx.registry.is_published(&stored.release_id).unwrap());
    assert_eq!(
        fx.registry.execution_status(&stored.release_id).unwrap(),
        ExecutionStatus::Unpublished
    );
}

// --- one test per release rule ---

#[test]
fn an_unsigned_release_is_refused() {
    let fx = Fixture::new();
    let err = fx
        .registry
        .publish_release(fx.draft("example-agent", "1.0.0"), NOW)
        .unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::SignatureInvalid);
}

#[test]
fn a_signature_over_the_wrong_content_is_refused() {
    let fx = Fixture::new();
    // Sign version 1.0.0, then publish 1.0.1 carrying that signature: the id
    // the signature covers is no longer the id of the object.
    let signed = fx.signed("example-agent", "1.0.0");
    let mut tampered = signed.clone();
    tampered.version = "1.0.1".into();
    tampered.release_id = String::new();

    let err = fx.registry.publish_release(tampered, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::SignatureInvalid);
}

#[test]
fn a_signature_from_a_key_the_publisher_does_not_list_is_refused() {
    let fx = Fixture::new();
    let stranger = testkit::signing_key(0x77);
    let release = testkit::sign(
        fx.draft("example-agent", "1.0.0"),
        &stranger,
        SignatureRole::Publisher,
    )
    .unwrap();

    let err = fx.registry.publish_release(release, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::KeyRevoked);
    assert!(err.to_string().contains("not listed"), "{err}");
}

#[test]
fn a_key_marked_revoked_in_the_publisher_record_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let registry = Registry::open(dir.path()).unwrap();
    let key = testkit::signing_key(0x02);

    let mut record = testkit::publisher_record("Example Inc", &[&key]);
    record.public_keys[0].revoked_at = Some("2026-07-01T00:00:00Z".into());
    let publisher = registry.put_publisher(record, NOW).unwrap();
    let manifest = registry
        .put_capability_manifest(testkit::capability_manifest())
        .unwrap();

    let release = testkit::sign(
        testkit::release(
            &publisher.publisher_id,
            "example-agent",
            "1.0.0",
            &manifest.manifest_id,
        ),
        &key,
        SignatureRole::Publisher,
    )
    .unwrap();

    let err = registry.publish_release(release, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::KeyRevoked);
    assert!(
        err.to_string().contains("revoked in the publisher record"),
        "{err}"
    );
}

#[test]
fn a_key_the_registry_revoked_is_refused_for_new_releases() {
    let fx = Fixture::new();
    let key_id = key_id_for(&fx.publisher_key.verifying_key());
    fx.registry
        .revoke(fx.revocation(&key_id, SubjectType::PublisherKey), NOW)
        .unwrap();

    let err = fx
        .registry
        .publish_release(fx.signed("example-agent", "1.0.0"), NOW)
        .unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::KeyRevoked);
    assert!(err.to_string().contains("revoked by the registry"), "{err}");
}

#[test]
fn a_missing_publisher_record_is_a_missing_dependency() {
    let fx = Fixture::new();
    let mut draft = fx.draft("example-agent", "1.0.0");
    draft.package.publisher_id = testkit::digest("no-such-publisher");
    let release = testkit::sign(draft, &fx.publisher_key, SignatureRole::Publisher).unwrap();

    let err = fx.registry.publish_release(release, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::MissingDependency);
    assert!(err.to_string().contains("publisherId"), "{err}");
}

#[test]
fn a_missing_capability_manifest_is_a_missing_dependency() {
    let fx = Fixture::new();
    let mut draft = fx.draft("example-agent", "1.0.0");
    draft.capability_manifest = testkit::digest("no-such-manifest");
    let release = testkit::sign(draft, &fx.publisher_key, SignatureRole::Publisher).unwrap();

    let err = fx.registry.publish_release(release, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::MissingDependency);
    assert!(err.to_string().contains("capabilityManifest"), "{err}");
}

#[test]
fn a_predecessor_of_the_same_package_chains() {
    let fx = Fixture::new();
    let first = fx.publish("example-agent", "1.0.0");

    let mut draft = fx.draft("example-agent", "1.1.0");
    draft.predecessor = Some(first.clone());
    let second = fx
        .registry
        .publish_release(
            testkit::sign(draft, &fx.publisher_key, SignatureRole::Publisher).unwrap(),
            NOW,
        )
        .unwrap();

    assert_eq!(second.release.predecessor, Some(first));
    let index = fx
        .registry
        .releases_for(&fx.publisher.publisher_id, "example-agent")
        .unwrap();
    assert_eq!(index.len(), 2);
}

#[test]
fn a_predecessor_from_another_package_is_refused() {
    let fx = Fixture::new();
    let other = fx.publish("other-agent", "1.0.0");

    let mut draft = fx.draft("example-agent", "1.1.0");
    draft.predecessor = Some(other);
    let release = testkit::sign(draft, &fx.publisher_key, SignatureRole::Publisher).unwrap();

    let err = fx.registry.publish_release(release, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::BadPredecessor);
    assert!(err.to_string().contains("belongs to"), "{err}");
}

#[test]
fn a_predecessor_that_does_not_exist_is_refused() {
    let fx = Fixture::new();
    let mut draft = fx.draft("example-agent", "1.1.0");
    draft.predecessor = Some(testkit::digest("ghost"));
    let release = testkit::sign(draft, &fx.publisher_key, SignatureRole::Publisher).unwrap();

    let err = fx.registry.publish_release(release, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::BadPredecessor);
    assert!(err.to_string().contains("no such release"), "{err}");
}

#[test]
fn a_predecessor_that_was_never_published_is_refused() {
    let fx = Fixture::new();
    let unpublished = fx
        .registry
        .store()
        .put(fx.signed("example-agent", "1.0.0"))
        .unwrap();

    let mut draft = fx.draft("example-agent", "1.1.0");
    draft.predecessor = Some(unpublished.release_id);
    let release = testkit::sign(draft, &fx.publisher_key, SignatureRole::Publisher).unwrap();

    let err = fx.registry.publish_release(release, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::BadPredecessor);
    assert!(err.to_string().contains("never published"), "{err}");
}

#[test]
fn a_manifest_with_a_broad_scope_and_no_review_trigger_is_refused() {
    let fx = Fixture::new();
    let mut manifest = testkit::capability_manifest();
    manifest.requests[0].scope = "all-files".into();

    let err = fx.registry.put_capability_manifest(manifest).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::InvalidObject);
    assert!(err.to_string().contains("manualReviewTriggers"), "{err}");
}

#[test]
fn a_broad_scope_is_allowed_once_review_is_declared() {
    let fx = Fixture::new();
    let mut manifest = testkit::capability_manifest();
    manifest.requests[0].scope = "all-files".into();
    manifest.manual_review_triggers = vec!["unrestricted-filesystem".into()];

    let stored: CapabilityManifest = fx.registry.put_capability_manifest(manifest).unwrap();
    assert_eq!(stored.broad_requests().len(), 1);
}
