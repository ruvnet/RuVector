//! Push by `upload_id`:
//!
//! 1. [`Registry::begin_upload`]: the scope must be adopted and owned by the
//!    caller's tenant.
//! 2. [`Registry::record_part`] per part (size and SHA-256 measured by the
//!    Worker as it streamed the part to staging).
//! 3. [`Registry::finalize_plan`] **freezes** the session (`Finalizing`:
//!    parts can no longer change, no abort) and pins the blob. Idempotent,
//!    so a Worker retry gets the same plan.
//! 4. The Worker streams the frozen parts, in order, through a
//!    [`StreamValidator`](crate::validate::StreamValidator), measuring each
//!    part's size and SHA-256 again on this read. Only if validation passed
//!    and the object's SHA-256 equals the plan's, and `copy_required`, it
//!    writes the blob from those same bytes (R2 put with the expected
//!    SHA-256, so R2 itself rejects different bytes) and measures what it
//!    wrote.
//! 5. [`Registry::finalize`] with a [`FinalizeReport`]: the re-measured
//!    parts must equal the frozen ones, the validator's digest and size the
//!    declared ones, and the copy evidence the object's digest and size.

use super::{k_pkg, k_ver, PackageRecord, Registry};
use crate::authz::{authorize_scope_push, Caller};
use crate::error::{RegistryError, Result};
use crate::keys::{blob_key, staging_key, BlobKey};
use crate::manifest::{PackageManifest, Provenance, MAX_WITNESS_ROOT};
use crate::ports::{Clock, EntropySource, KvStore};
use crate::upload::{
    parts_digest, BeginUpload, ObjectEvidence, PartRecord, UploadId, UploadSession, UploadState,
};
use crate::validate::{ValidatedRvf, ValidationError, ValidationLimits};
use ruvector_edge_auth::Capability;

/// Everything the Worker needs to finish an upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizePlan {
    /// The session.
    pub upload_id: UploadId,
    /// R2 staging object (multipart).
    pub staging: BlobKey,
    /// Frozen parts to stream, in order.
    pub parts: Vec<PartRecord>,
    /// Where the bytes go once validated and digest-checked.
    pub blob: BlobKey,
    /// The bytes are not known to be stored: write `blob` (after validation
    /// and the digest check) and report [`FinalizeReport::copied`].
    pub copy_required: bool,
    /// Limits to construct the validator with.
    pub limits: ValidationLimits,
    /// Declared size.
    pub size: u64,
    /// Declared SHA-256.
    pub sha256: [u8; 32],
}

/// What the Worker observed while finishing an upload.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizeReport {
    /// The validator's result over the staged bytes.
    pub validated: core::result::Result<ValidatedRvf, ValidationError>,
    /// Each part as streamed into the validator (number, size, SHA-256
    /// measured on this read, not copied from the plan).
    pub streamed: Vec<PartRecord>,
    /// When the plan said `copy_required`: the measured blob object.
    pub copied: Option<ObjectEvidence>,
    /// Optional witness root.
    pub provenance: Option<Provenance>,
}

pub(crate) fn k_upload(id: &UploadId) -> String {
    format!("upload/{}", id.as_str())
}

pub(crate) fn session_blob(s: &UploadSession) -> BlobKey {
    blob_key(
        &s.tenant,
        s.target.name.scope(),
        s.target.visibility,
        &s.target.sha256,
    )
}

fn require_write(caller: &Caller) -> Result<()> {
    if caller.caps.contains(Capability::Write) {
        Ok(())
    } else {
        Err(RegistryError::Forbidden(Capability::Write))
    }
}

impl<S: KvStore, C: Clock, E: EntropySource> Registry<S, C, E> {
    fn authorize_push(&self, caller: &Caller, req: &BeginUpload) -> Result<()> {
        let scope = self
            .scope_record(req.name.scope())?
            .ok_or(RegistryError::ScopeUnclaimed)?;
        Ok(authorize_scope_push(caller, Some(&scope.owner))?)
    }

    /// `POST /v1/rvf/{name}/{version}`: open an upload session.
    pub fn begin_upload(&self, caller: &Caller, req: BeginUpload) -> Result<UploadSession> {
        self.authorize_push(caller, &req)?;
        if req.size > self.config.validation.max_total_bytes {
            return Err(RegistryError::UploadLimit(
                "declared size exceeds the object limit",
            ));
        }
        if self.store.get(&k_ver(&req.name, &req.version))?.is_some() {
            return Err(RegistryError::VersionExists);
        }
        let id = UploadId::mint(&self.entropy)?;
        let s = UploadSession::new(
            id,
            caller.tenant.clone(),
            caller.sub.clone(),
            req,
            self.now(),
            &self.config.upload,
        )?;
        self.save(&k_upload(&s.id), &s)?;
        Ok(s)
    }

    /// Load a session the caller owns; anyone else gets `NotFound`.
    pub fn upload(&self, caller: &Caller, id: &UploadId) -> Result<UploadSession> {
        let s: UploadSession = self.load(&k_upload(id))?.ok_or(RegistryError::NotFound)?;
        if s.tenant != caller.tenant || s.created_by != caller.sub {
            return Err(RegistryError::NotFound);
        }
        Ok(s)
    }

    /// `PUT …/uploads/{upload_id}/parts/{n}`: record a part. Open sessions
    /// only; needs `Write`.
    pub fn record_part(
        &self,
        caller: &Caller,
        id: &UploadId,
        part: PartRecord,
    ) -> Result<UploadSession> {
        require_write(caller)?;
        let mut s = self.upload(caller, id)?;
        s.record_part(part, self.now(), &self.config.upload)?;
        self.save(&k_upload(id), &s)?;
        Ok(s)
    }

    /// Cancel an open session (not once frozen by `finalize_plan`).
    pub fn abort_upload(&self, caller: &Caller, id: &UploadId) -> Result<UploadSession> {
        require_write(caller)?;
        let mut s = self.upload(caller, id)?;
        if s.state != UploadState::Open {
            return Err(RegistryError::UploadNotOpen);
        }
        s.state = UploadState::Aborted;
        self.save(&k_upload(id), &s)?;
        Ok(s)
    }

    /// Freeze the session and plan the finalize (idempotent).
    pub fn finalize_plan(&self, caller: &Caller, id: &UploadId) -> Result<FinalizePlan> {
        require_write(caller)?;
        let mut s = self.upload(caller, id)?;
        let now = self.now();
        let blob = session_blob(&s);
        let copy_required = match s.state {
            UploadState::Open => {
                s.check_unexpired(now)?;
                let digest = parts_digest(s.check_parts(&self.config.upload)?);
                let copy_required = self.pin_blob(&blob, s.target.size)?;
                s.state = UploadState::Finalizing {
                    parts_digest: digest,
                    copy_required,
                };
                self.save(&k_upload(id), &s)?;
                copy_required
            }
            UploadState::Finalizing { copy_required, .. } => {
                s.check_unexpired(now)?;
                copy_required
            }
            _ => return Err(RegistryError::UploadNotOpen),
        };
        Ok(FinalizePlan {
            upload_id: s.id.clone(),
            staging: staging_key(&s.tenant, &s.id),
            parts: s.parts.clone(),
            blob,
            copy_required,
            limits: self.config.validation,
            size: s.target.size,
            sha256: s.target.sha256,
        })
    }

    /// End a session unsuccessfully, releasing its pin if it holds one.
    pub(crate) fn fail_session(&self, mut s: UploadSession, reason: &str) -> Result<()> {
        if s.is_finalizing() {
            self.release_blob(&session_blob(&s))?;
        }
        s.state = UploadState::Failed {
            reason: reason.to_string(),
        };
        self.save(&k_upload(&s.id), &s)
    }

    fn fail(&self, s: UploadSession, reason: &str, err: RegistryError) -> Result<PackageManifest> {
        self.fail_session(s, reason)?;
        Err(err)
    }

    /// Commit the upload as an immutable version. A validation error, parts
    /// other than the frozen ones, a size or digest other than the
    /// declaration, missing or wrong copy evidence, or an existing version
    /// fails the session (terminal); nothing is published.
    pub fn finalize(
        &self,
        caller: &Caller,
        id: &UploadId,
        report: FinalizeReport,
    ) -> Result<PackageManifest> {
        let s = self.upload(caller, id)?;
        let now = self.now();
        let UploadState::Finalizing {
            parts_digest: frozen,
            copy_required,
        } = s.state
        else {
            return Err(RegistryError::UploadNotOpen);
        };
        if s.check_unexpired(now).is_err() {
            return self.fail(s, "expired", RegistryError::UploadExpired);
        }
        if report
            .provenance
            .as_ref()
            .is_some_and(|p| p.witness_root.len() > MAX_WITNESS_ROOT)
        {
            return Err(RegistryError::InvalidRequest("witness root too long"));
        }
        authorize_scope_push(
            caller,
            self.scope_record(s.target.name.scope())?
                .map(|r| r.owner)
                .as_ref(),
        )?;
        if parts_digest(&report.streamed) != frozen || report.streamed != s.parts {
            return self.fail(s, "parts_changed", RegistryError::PartsChanged);
        }
        let v = match report.validated {
            Ok(v) => v,
            Err(e) => return self.fail(s, "invalid_rvf", e.into()),
        };
        if v.total_size != s.target.size {
            return self.fail(s, "size_mismatch", RegistryError::SizeMismatch);
        }
        if v.sha256 != s.target.sha256 {
            return self.fail(s, "digest_mismatch", RegistryError::DigestMismatch);
        }
        let expected = ObjectEvidence {
            sha256: v.sha256,
            size: v.total_size,
        };
        let evidence_ok = match report.copied {
            Some(ev) => ev == expected,
            None => !copy_required,
        };
        if !evidence_ok {
            return self.fail(s, "evidence_mismatch", RegistryError::EvidenceMismatch);
        }
        let name = s.target.name.clone();
        let version = s.target.version.clone();
        if self.store.get(&k_ver(&name, &version))?.is_some() {
            return self.fail(s, "version_exists", RegistryError::VersionExists);
        }
        let mut pkg: PackageRecord = self.load(&k_pkg(&name))?.unwrap_or(PackageRecord {
            name: name.clone(),
            owner: s.tenant.clone(),
            created_at: now,
            versions: 0,
            public_versions: 0,
        });
        if pkg.versions >= self.config.max_versions_per_package {
            return self.fail(s, "too_many_versions", RegistryError::TooManyVersions);
        }
        pkg.versions += 1;
        let manifest = PackageManifest::from_validated(
            name.clone(),
            version,
            &v,
            s.tenant.clone(),
            s.created_by.clone(),
            now,
            s.target.visibility,
            report.provenance,
        );
        // All checks passed: writes, back to back. The session's pin becomes
        // the version's reference.
        self.write_version(&manifest)?;
        self.save(&k_pkg(&name), &pkg)?;
        self.mark_stored(&session_blob(&s))?;
        let mut s = s;
        s.state = UploadState::Finalized;
        self.save(&k_upload(id), &s)?;
        Ok(manifest)
    }
}
