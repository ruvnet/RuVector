//! Shared fixtures: a registry with a configured registry key, one publisher,
//! and one capability manifest.
//!
//! Each integration test binary compiles this module separately, so a helper
//! only one of them uses looks dead to the other.
#![allow(dead_code)]

use ed25519_dalek::SigningKey;
use rvforge_registry::objects::{
    Actor, ActorKind, CapabilityManifest, PublisherRecord, Release, Revocation, RevocationReason,
    SignatureRole, SubjectType, TrustEvidence, TrustLevel, TrustUpdate,
};
use rvforge_registry::{key_id_for, testkit, Registry};

/// A fixed instant; the registry never reads a clock.
pub const NOW: &str = "2026-08-03T00:00:00Z";

pub struct Fixture {
    pub _dir: tempfile::TempDir,
    pub registry: Registry,
    pub registry_key: SigningKey,
    pub publisher_key: SigningKey,
    pub publisher: PublisherRecord,
    pub manifest: CapabilityManifest,
}

impl Fixture {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let registry_key = testkit::signing_key(0xAA);
        let registry = Registry::open(dir.path()).unwrap().with_registry_key(
            key_id_for(&registry_key.verifying_key()),
            registry_key.verifying_key(),
        );

        let publisher_key = testkit::signing_key(0x01);
        let publisher = registry
            .put_publisher(
                testkit::publisher_record("Example Inc", &[&publisher_key]),
                NOW,
            )
            .unwrap();
        let manifest = registry
            .put_capability_manifest(testkit::capability_manifest())
            .unwrap();

        Self {
            _dir: dir,
            registry,
            registry_key,
            publisher_key,
            publisher,
            manifest,
        }
    }

    /// An unsigned release of `name`.
    pub fn draft(&self, name: &str, version: &str) -> Release {
        testkit::release(
            &self.publisher.publisher_id,
            name,
            version,
            &self.manifest.manifest_id,
        )
    }

    /// A release signed by the fixture's publisher key.
    pub fn signed(&self, name: &str, version: &str) -> Release {
        testkit::sign(
            self.draft(name, version),
            &self.publisher_key,
            SignatureRole::Publisher,
        )
        .unwrap()
    }

    /// Publish a signed release and return its id.
    pub fn publish(&self, name: &str, version: &str) -> String {
        self.registry
            .publish_release(self.signed(name, version), NOW)
            .unwrap()
            .release
            .release_id
    }

    /// A registry-signed revocation of `subject`.
    pub fn revocation(&self, subject: &str, subject_type: SubjectType) -> Revocation {
        testkit::sign(
            Revocation {
                subject: subject.into(),
                subject_type,
                reason: RevocationReason::Compromise,
                issued_at: NOW.into(),
                ..Default::default()
            },
            &self.registry_key,
            SignatureRole::Registry,
        )
        .unwrap()
    }

    /// A registry-signed trust update.
    pub fn trust_update(&self, subject: &str, level: TrustLevel) -> TrustUpdate {
        testkit::sign(
            self.unsigned_trust_update(subject, level),
            &self.registry_key,
            SignatureRole::Registry,
        )
        .unwrap()
    }

    /// A trust update with no signature attached.
    pub fn unsigned_trust_update(&self, subject: &str, level: TrustLevel) -> TrustUpdate {
        TrustUpdate {
            subject: subject.into(),
            trust_level: level,
            evidence: TrustEvidence {
                method: "automated-suite".into(),
                reference: "run-4711".into(),
            },
            issued_at: NOW.into(),
            ..Default::default()
        }
    }
}

/// A reader-shaped actor for verification receipts.
pub fn reader() -> Actor {
    Actor {
        kind: ActorKind::Reader,
        id: "rvforge-reader/test".into(),
    }
}
