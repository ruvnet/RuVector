//! The trust ladder and revocation semantics.

mod common;

use common::{Fixture, NOW};
use rvforge_registry::objects::{PublicKeyEntry, Release, SignatureRole, SubjectType, TrustLevel};
use rvforge_registry::{key_id_for, testkit, ExecutionStatus, Registry, RegistryErrorCode};

// --- the trust ladder ---

#[test]
fn a_publisher_may_not_claim_a_trust_level() {
    let fx = Fixture::new();
    let mut draft = fx.draft("example-agent", "1.0.0");
    draft.trust_level = TrustLevel::Reviewed;
    let release = testkit::sign(draft, &fx.publisher_key, SignatureRole::Publisher).unwrap();

    let err = fx.registry.publish_release(release, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::TrustLevel);
}

#[test]
fn a_registry_signed_update_raises_the_trust_level() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");

    fx.registry
        .raise_trust_level(fx.trust_update(&id, TrustLevel::Tested), NOW)
        .unwrap();
    assert_eq!(
        fx.registry.effective_trust_level(&id).unwrap(),
        TrustLevel::Tested
    );

    fx.registry
        .raise_trust_level(fx.trust_update(&id, TrustLevel::Reviewed), NOW)
        .unwrap();
    assert_eq!(
        fx.registry.effective_trust_level(&id).unwrap(),
        TrustLevel::Reviewed
    );

    // The release object itself is untouched: it is immutable.
    let release: Release = fx.registry.store().get(&id).unwrap();
    assert_eq!(release.trust_level, TrustLevel::Published);
}

#[test]
fn an_unsigned_trust_update_is_refused() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");

    let err = fx
        .registry
        .raise_trust_level(fx.unsigned_trust_update(&id, TrustLevel::Reviewed), NOW)
        .unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::SignatureInvalid);
    assert_eq!(
        fx.registry.effective_trust_level(&id).unwrap(),
        TrustLevel::Published
    );
}

#[test]
fn a_trust_update_signed_by_the_publisher_is_refused() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");

    let forged = testkit::sign(
        fx.unsigned_trust_update(&id, TrustLevel::EnterpriseApproved),
        &fx.publisher_key,
        SignatureRole::Registry,
    )
    .unwrap();

    let err = fx.registry.raise_trust_level(forged, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::SignatureInvalid);
    assert!(err.to_string().contains("not the registry key"), "{err}");
}

#[test]
fn a_registry_without_a_configured_key_cannot_raise_trust() {
    let dir = tempfile::tempdir().unwrap();
    let registry = Registry::open(dir.path()).unwrap();
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");

    let err = registry
        .raise_trust_level(fx.trust_update(&id, TrustLevel::Tested), NOW)
        .unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::SignatureInvalid);
    assert!(
        err.to_string().contains("no registry key is configured"),
        "{err}"
    );
}

#[test]
fn trust_may_not_be_lowered_or_repeated() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    fx.registry
        .raise_trust_level(fx.trust_update(&id, TrustLevel::Reviewed), NOW)
        .unwrap();

    for level in [
        TrustLevel::Published,
        TrustLevel::Tested,
        TrustLevel::Reviewed,
    ] {
        let err = fx
            .registry
            .raise_trust_level(fx.trust_update(&id, level), NOW)
            .unwrap_err();
        assert_eq!(err.code(), RegistryErrorCode::TrustLevel, "{level:?}");
    }
    assert_eq!(
        fx.registry.effective_trust_level(&id).unwrap(),
        TrustLevel::Reviewed
    );
}

#[test]
fn trust_cannot_be_raised_on_an_unpublished_release() {
    let fx = Fixture::new();
    let stored = fx
        .registry
        .store()
        .put(fx.signed("example-agent", "1.0.0"))
        .unwrap();

    let err = fx
        .registry
        .raise_trust_level(fx.trust_update(&stored.release_id, TrustLevel::Tested), NOW)
        .unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::TrustLevel);
}

// --- revocation ---

#[test]
fn revoking_a_release_blocks_execution_but_preserves_reads() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    let before: Release = fx.registry.store().get(&id).unwrap();
    let before_bytes = fx.registry.store().get_bytes(&id).unwrap();

    fx.registry
        .revoke(fx.revocation(&id, SubjectType::Release), NOW)
        .unwrap();

    assert!(fx.registry.is_revoked(&id).unwrap());
    assert!(!fx.registry.is_executable(&id).unwrap());
    assert_eq!(
        fx.registry.execution_status(&id).unwrap(),
        ExecutionStatus::Revoked {
            subject: id.clone()
        }
    );

    // Export and forensic access survive: the object is byte-identical and the
    // log entry it was published under is still there (ADR-294 §9, P12).
    assert_eq!(fx.registry.store().get::<Release>(&id).unwrap(), before);
    assert_eq!(fx.registry.store().get_bytes(&id).unwrap(), before_bytes);
    assert!(fx.registry.is_published(&id).unwrap());
    assert!(fx.registry.inclusion_proof(&id).is_ok());
}

#[test]
fn revoking_one_release_leaves_its_siblings_alone() {
    let fx = Fixture::new();
    let first = fx.publish("example-agent", "1.0.0");
    let second = fx.publish("example-agent", "2.0.0");

    fx.registry
        .revoke(fx.revocation(&first, SubjectType::Release), NOW)
        .unwrap();

    assert!(!fx.registry.is_executable(&first).unwrap());
    assert!(fx.registry.is_executable(&second).unwrap());
}

#[test]
fn revoking_a_publisher_blocks_all_of_its_releases() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");

    fx.registry
        .revoke(
            fx.revocation(&fx.publisher.publisher_id, SubjectType::Publisher),
            NOW,
        )
        .unwrap();

    assert!(!fx.registry.is_executable(&id).unwrap());
    assert_eq!(
        fx.registry.execution_status(&id).unwrap(),
        ExecutionStatus::Revoked {
            subject: fx.publisher.publisher_id.clone()
        }
    );

    let err = fx
        .registry
        .publish_release(fx.signed("example-agent", "2.0.0"), NOW)
        .unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::Revoked);
}

#[test]
fn revoking_a_key_blocks_the_releases_it_signed() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");
    let key_id = key_id_for(&fx.publisher_key.verifying_key());

    fx.registry
        .revoke(fx.revocation(&key_id, SubjectType::PublisherKey), NOW)
        .unwrap();

    assert_eq!(
        fx.registry.execution_status(&id).unwrap(),
        ExecutionStatus::Revoked { subject: key_id }
    );
}

#[test]
fn an_unsigned_revocation_is_refused() {
    let fx = Fixture::new();
    let id = fx.publish("example-agent", "1.0.0");

    let mut unsigned = fx.revocation(&id, SubjectType::Release);
    unsigned.signatures.clear();
    unsigned.revocation_id = String::new();

    let err = fx.registry.revoke(unsigned, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::SignatureInvalid);
    assert!(fx.registry.is_executable(&id).unwrap());
}

#[test]
fn revoking_an_unknown_release_is_not_found() {
    let fx = Fixture::new();
    let err = fx
        .registry
        .revoke(
            fx.revocation(&testkit::digest("ghost"), SubjectType::Release),
            NOW,
        )
        .unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::NotFound);
}

#[test]
fn a_publisher_key_revocation_needs_a_key_id_not_a_digest() {
    let fx = Fixture::new();
    let err = fx
        .registry
        .revoke(
            fx.revocation(&testkit::digest("x"), SubjectType::PublisherKey),
            NOW,
        )
        .unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::InvalidObject);
}

#[test]
fn a_publisher_record_with_no_keys_is_refused() {
    let fx = Fixture::new();
    let mut record = testkit::publisher_record("Empty Inc", &[]);
    record.public_keys = Vec::<PublicKeyEntry>::new();

    let err = fx.registry.put_publisher(record, NOW).unwrap_err();
    assert_eq!(err.code(), RegistryErrorCode::InvalidObject);
}
