//! Publication, revocation, and trust-level raises.
//!
//! The release rules of `registry-model.md` are enforced here in the order they
//! are stated: publisher signature, `rvfIdentity`, capability manifest,
//! predecessor. Each violation gets its own error code so a publisher CLI can
//! say which rule was broken rather than "rejected".

use super::witness::ReceiptRequest;
use super::{ExecutionStatus, Registry};
use crate::crypto::verify_over_id;
use crate::error::{RegistryError, Result};
use crate::objects::{
    compute_id, validate, Actor, ActorKind, CapabilityManifest, LogObjectType, Outcome,
    PublisherRecord, Release, Revocation, SubjectType, TransparencyLogEntry, TrustLevel,
    TrustUpdate, WitnessEvent, WitnessReceipt,
};

/// What a successful publication produced.
#[derive(Clone, Debug)]
pub struct PublishOutcome {
    /// The stored release, with its id set.
    pub release: Release,
    /// Its transparency-log entry.
    pub entry: TransparencyLogEntry,
    /// The witness receipt for the publication.
    pub receipt: WitnessReceipt,
}

/// What a successful revocation produced.
#[derive(Clone, Debug)]
pub struct RevokeOutcome {
    /// The stored revocation.
    pub revocation: Revocation,
    /// Its transparency-log entry.
    pub entry: TransparencyLogEntry,
    /// The witness receipt for the revocation.
    pub receipt: WitnessReceipt,
}

/// What a successful trust raise produced.
#[derive(Clone, Debug)]
pub struct TrustOutcome {
    /// The stored trust update.
    pub update: TrustUpdate,
    /// Its transparency-log entry.
    pub entry: TransparencyLogEntry,
    /// The witness receipt for the raise.
    pub receipt: WitnessReceipt,
}

impl Registry {
    /// Store a capability manifest so releases can reference it.
    ///
    /// Manifests are not logged: the log covers what is *published*, and a
    /// manifest reaches the public record through the release that names it.
    ///
    /// # Errors
    ///
    /// As [`crate::store::ObjectStore::put`].
    pub fn put_capability_manifest(
        &self,
        manifest: CapabilityManifest,
    ) -> Result<CapabilityManifest> {
        self.store().put(manifest)
    }

    /// Apply every release rule, then store, index, log, and witness it.
    ///
    /// # Errors
    ///
    /// One code per broken rule: [`RegistryError::TrustLevel`] for a claimed
    /// trust level, [`RegistryError::SignatureInvalid`] for a bad or absent
    /// publisher signature, [`RegistryError::KeyRevoked`] for a signing key
    /// that is revoked or unknown to the publisher record,
    /// [`RegistryError::MissingDependency`] for an unresolvable publisher
    /// record or capability manifest, [`RegistryError::BadPredecessor`] for a
    /// predecessor that is missing, unpublished, or from another package, and
    /// [`RegistryError::Revoked`] if the publisher itself is revoked.
    pub fn publish_release(&self, release: Release, at: &str) -> Result<PublishOutcome> {
        validate(&release)?;
        let release_id = compute_id(&release)?;
        if !release.release_id.is_empty() && release.release_id != release_id {
            return Err(RegistryError::IdMismatch {
                declared: release.release_id.clone(),
                computed: release_id,
            });
        }

        // (0) Trust is evidence, not a publisher assertion (ADR-294 §7).
        if release.trust_level != TrustLevel::INITIAL {
            return Err(RegistryError::TrustLevel {
                detail: format!(
                    "a publisher may only submit trustLevel {:?}; {:?} must come from a \
                     registry-signed trust update",
                    TrustLevel::INITIAL,
                    release.trust_level
                ),
            });
        }

        // (a) The publisher signature verifies against a live key in the
        //     referenced publisher record.
        self.check_publisher_signature(&release, &release_id)?;

        // (b) rvfIdentity is present. Its shape is checked by validate(); that
        //     it matches the uploaded artifact is the caller's to assert, since
        //     the registry never holds the bytes.

        // (c) The capability manifest resolves.
        if self
            .store()
            .get_optional::<CapabilityManifest>(&release.capability_manifest)?
            .is_none()
        {
            return Err(RegistryError::MissingDependency {
                kind: "capabilityManifest".into(),
                id: release.capability_manifest.clone(),
            });
        }

        // (d) The predecessor is null or a published release of this package.
        self.check_predecessor(&release)?;

        let stored = self.store().put(release)?;
        self.append_package_index(&stored)?;
        let entry = self
            .log()
            .append(&stored.release_id, LogObjectType::Release, at)?;
        let receipt = self.emit_receipt(&ReceiptRequest {
            subject: &stored.release_id,
            event: WitnessEvent::Publish,
            outcome: Outcome::Pass,
            actor: Actor {
                kind: ActorKind::Registry,
                id: "rvforge-registry".into(),
            },
            details: format!(
                "release {} of {} accepted at log index {}",
                stored.version, stored.package.name, entry.index
            ),
            timestamp: at,
        })?;

        Ok(PublishOutcome {
            release: stored,
            entry,
            receipt,
        })
    }

    /// Record a revocation. Nothing is deleted: the revoked object stays
    /// readable through [`Registry::store`] so export and forensic access are
    /// preserved (ADR-294 §9, requirements P12).
    ///
    /// # Errors
    ///
    /// [`RegistryError::SignatureInvalid`] without a valid registry
    /// countersignature, [`RegistryError::NotFound`] if a release or publisher
    /// subject does not resolve.
    pub fn revoke(&self, revocation: Revocation, at: &str) -> Result<RevokeOutcome> {
        validate(&revocation)?;
        let revocation_id = compute_id(&revocation)?;
        self.require_registry_signature(&revocation, &revocation_id)?;

        match revocation.subject_type {
            SubjectType::Release | SubjectType::Publisher => {
                if !self.store().contains(&revocation.subject)? {
                    return Err(RegistryError::NotFound {
                        kind: format!("{:?} subject", revocation.subject_type).to_lowercase(),
                        id: revocation.subject.clone(),
                    });
                }
            }
            SubjectType::PublisherKey => {}
        }

        let stored = self.store().put(revocation)?;
        let entry = self
            .log()
            .append(&stored.revocation_id, LogObjectType::Revocation, at)?;
        let receipt = self.emit_receipt(&ReceiptRequest {
            subject: &stored.subject,
            event: WitnessEvent::Revoke,
            outcome: Outcome::Denied,
            actor: Actor {
                kind: ActorKind::Registry,
                id: "rvforge-registry".into(),
            },
            details: format!(
                "{:?} revoked, reason {:?}; execution blocked, reads preserved",
                stored.subject_type, stored.reason
            ),
            timestamp: at,
        })?;

        Ok(RevokeOutcome {
            revocation: stored,
            entry,
            receipt,
        })
    }

    /// Raise a release's trust level.
    ///
    /// The update must carry a registry-role signature that verifies against
    /// the configured registry key, and it may only move the level up.
    ///
    /// # Errors
    ///
    /// [`RegistryError::SignatureInvalid`] without a valid registry signature,
    /// [`RegistryError::TrustLevel`] if the subject is unpublished or the new
    /// level is not strictly higher than the current one.
    pub fn raise_trust_level(&self, update: TrustUpdate, at: &str) -> Result<TrustOutcome> {
        validate(&update)?;
        let update_id = compute_id(&update)?;
        self.require_registry_signature(&update, &update_id)?;

        if !self.is_published(&update.subject)? {
            return Err(RegistryError::TrustLevel {
                detail: format!(
                    "{} is not a published release, so its trust level cannot be raised",
                    update.subject
                ),
            });
        }
        let current = self.effective_trust_level(&update.subject)?;
        if update.trust_level <= current {
            return Err(RegistryError::TrustLevel {
                detail: format!(
                    "trust level may only be raised: {} is already {current:?}, update asks for {:?}",
                    update.subject, update.trust_level
                ),
            });
        }

        let stored = self.store().put(update)?;
        let entry = self
            .log()
            .append(&stored.update_id, LogObjectType::TrustUpdate, at)?;
        let receipt = self.emit_receipt(&ReceiptRequest {
            subject: &stored.subject,
            event: WitnessEvent::Verify,
            outcome: Outcome::Pass,
            actor: Actor {
                kind: ActorKind::Registry,
                id: "rvforge-registry".into(),
            },
            details: format!(
                "trust level raised to {:?} on {} evidence ({})",
                stored.trust_level, stored.evidence.method, stored.evidence.reference
            ),
            timestamp: at,
        })?;

        Ok(TrustOutcome {
            update: stored,
            entry,
            receipt,
        })
    }

    fn check_publisher_signature(&self, release: &Release, release_id: &str) -> Result<()> {
        let sig = release
            .publisher_signature()
            .ok_or_else(|| RegistryError::SignatureInvalid {
                detail: "release carries no publisher-role signature".into(),
            })?;

        let publisher: PublisherRecord = self
            .store()
            .get_optional(&release.package.publisher_id)?
            .ok_or_else(|| RegistryError::MissingDependency {
                kind: "package.publisherId".into(),
                id: release.package.publisher_id.clone(),
            })?;

        if self.is_revoked(&release.package.publisher_id)? {
            return Err(RegistryError::Revoked {
                subject: release.package.publisher_id.clone(),
                detail: "the publisher is revoked and may not publish".into(),
            });
        }

        let entry = publisher
            .live_key(&sig.key_id)
            .ok_or_else(|| RegistryError::KeyRevoked {
                key_id: sig.key_id.clone(),
                detail: if publisher.knows_key(&sig.key_id) {
                    "revoked in the publisher record".into()
                } else {
                    format!("not listed in publisher {}", publisher.publisher_id)
                },
            })?;

        if self.is_revoked(&sig.key_id)? {
            return Err(RegistryError::KeyRevoked {
                key_id: sig.key_id.clone(),
                detail: "revoked by the registry".into(),
            });
        }

        let key = crate::crypto::verifying_key_from_b64(&entry.public_key)?;
        verify_over_id(release_id, &sig.sig, &key)
    }

    fn check_predecessor(&self, release: &Release) -> Result<()> {
        let Some(predecessor) = &release.predecessor else {
            return Ok(());
        };
        let previous: Release = self.store().get_optional(predecessor)?.ok_or_else(|| {
            RegistryError::BadPredecessor {
                predecessor: predecessor.clone(),
                detail: "no such release".into(),
            }
        })?;
        if !previous.same_package_as(release) {
            return Err(RegistryError::BadPredecessor {
                predecessor: predecessor.clone(),
                detail: format!(
                    "belongs to {}/{}, not {}/{}",
                    previous.package.publisher_id,
                    previous.package.name,
                    release.package.publisher_id,
                    release.package.name
                ),
            });
        }
        if matches!(
            self.execution_status(predecessor)?,
            ExecutionStatus::Unpublished
        ) {
            return Err(RegistryError::BadPredecessor {
                predecessor: predecessor.clone(),
                detail: "absent from the transparency log, so it was never published".into(),
            });
        }
        Ok(())
    }
}
