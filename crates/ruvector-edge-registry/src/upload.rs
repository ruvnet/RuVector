//! Upload sessions: `POST /v1/rvf/{name}/{version}` mints a server-side
//! `upload_id`; the client sends numbered parts (an R2 multipart upload
//! under [`crate::keys::staging_key`]); finalize streams the parts, in order,
//! through the validator and refuses unless the object's SHA-256 and size
//! equal what was declared at begin.
//!
//! Sessions are personal (tenant **and** uploader must match, otherwise
//! `404`), expire after `ttl_secs`, and end in exactly one terminal state.

use crate::error::{RegistryError, Result};
use crate::hexser;
use crate::manifest::{tenant_key_serde, Visibility};
use crate::name::PackageName;
use crate::ports::EntropySource;
use crate::version::Version;
use ruvector_edge_tenancy::TenantKey;
use serde::{Deserialize, Serialize};

/// `up_` + 26 base32 characters (130 bits).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct UploadId(String);

const UPLOAD_PREFIX: &str = "up_";
const UPLOAD_BODY: usize = 26;

impl UploadId {
    /// Mint from 16 bytes of entropy.
    pub fn mint(entropy: &dyn EntropySource) -> Result<Self> {
        let mut b = [0u8; 17];
        entropy.fill(&mut b[..16]).map_err(|_| {
            RegistryError::Store(crate::ports::StoreError::Backend("entropy".into()))
        })?;
        let enc = data_encoding::BASE32_NOPAD.encode(&b).to_ascii_lowercase();
        Ok(UploadId(format!("{UPLOAD_PREFIX}{}", &enc[..UPLOAD_BODY])))
    }

    /// Strict parse of a client-supplied id.
    pub fn parse(s: &str) -> Result<Self> {
        let ok = s.len() == UPLOAD_PREFIX.len() + UPLOAD_BODY
            && s.starts_with(UPLOAD_PREFIX)
            && s[UPLOAD_PREFIX.len()..]
                .bytes()
                .all(|b| b.is_ascii_lowercase() || (b'2'..=b'7').contains(&b));
        if ok {
            Ok(UploadId(s.to_string()))
        } else {
            Err(RegistryError::InvalidRequest("malformed upload_id"))
        }
    }

    /// The id text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for UploadId {
    type Error = RegistryError;
    fn try_from(s: String) -> Result<Self> {
        UploadId::parse(&s)
    }
}

impl From<UploadId> for String {
    fn from(u: UploadId) -> String {
        u.0
    }
}

/// Upload limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadLimits {
    /// Most parts per session.
    pub max_parts: u32,
    /// Largest part.
    pub max_part_size: u64,
    /// Smallest part except the last (R2 multipart requires 5 MiB).
    pub min_part_size: u64,
    /// Session lifetime.
    pub ttl_secs: u64,
}

impl Default for UploadLimits {
    fn default() -> Self {
        UploadLimits {
            max_parts: 10_000,
            max_part_size: 100 << 20,
            min_part_size: 5 << 20,
            ttl_secs: 24 * 3600,
        }
    }
}

/// What the client declares at begin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeginUpload {
    /// Target package.
    pub name: PackageName,
    /// Target version.
    pub version: Version,
    /// `private` or `tenant` (public needs a separate publish).
    pub visibility: Visibility,
    /// Object size in bytes.
    pub size: u64,
    /// Object SHA-256.
    #[serde(with = "hexser")]
    pub sha256: [u8; 32],
}

/// One received part (size and hash computed by the Worker as it streamed
/// the part to R2, never taken from the client).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartRecord {
    /// 1-based part number.
    pub number: u32,
    /// Bytes.
    pub size: u64,
    /// SHA-256 of the part.
    #[serde(with = "hexser")]
    pub sha256: [u8; 32],
}

/// What the Worker measured of an R2 object it wrote or read back: the
/// SHA-256 and size of the bytes actually in the object, never the digest
/// the client declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectEvidence {
    /// SHA-256 of the object's bytes.
    #[serde(with = "hexser")]
    pub sha256: [u8; 32],
    /// Object size.
    pub size: u64,
}

/// SHA-256 over the ordered part records (number, size, part SHA-256): the
/// fingerprint that ties the bytes a finalize validated to the parts the
/// session froze.
pub fn parts_digest(parts: &[PartRecord]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for p in parts {
        h.update(p.number.to_le_bytes());
        h.update(p.size.to_le_bytes());
        h.update(p.sha256);
    }
    h.finalize().into()
}

/// Session lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum UploadState {
    /// Accepting parts.
    Open,
    /// Frozen by `finalize_plan`: parts can no longer change, the session
    /// can no longer be aborted, and the blob is pinned (its reference count
    /// includes this session) until finalize, failure or expiry.
    Finalizing {
        /// [`parts_digest`] of the frozen parts.
        #[serde(with = "hexser")]
        parts_digest: [u8; 32],
        /// The plan told the Worker to write the blob object.
        copy_required: bool,
    },
    /// Committed as a version.
    Finalized,
    /// Refused at finalize (digest, size, validation, conflict).
    Failed {
        /// Stable reason code.
        reason: String,
    },
    /// Cancelled by the uploader.
    Aborted,
}

/// A persisted upload session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UploadSession {
    /// Server-minted id.
    pub id: UploadId,
    /// Owning tenant.
    #[serde(with = "tenant_key_serde")]
    pub tenant: TenantKey,
    /// Uploader.
    pub created_by: String,
    /// What was declared.
    pub target: BeginUpload,
    /// Received parts, ascending by number, unique.
    pub parts: Vec<PartRecord>,
    /// Creation time (seconds).
    pub created_at: u64,
    /// Expiry (seconds).
    pub expires_at: u64,
    /// Lifecycle state.
    pub state: UploadState,
}

impl UploadSession {
    /// A fresh open session.
    pub fn new(
        id: UploadId,
        tenant: TenantKey,
        created_by: String,
        target: BeginUpload,
        now: u64,
        limits: &UploadLimits,
    ) -> Result<Self> {
        if target.visibility == Visibility::Public {
            return Err(RegistryError::InvalidRequest(
                "upload as private or tenant, then publish",
            ));
        }
        if target.size == 0 {
            return Err(RegistryError::InvalidRequest("empty upload"));
        }
        Ok(UploadSession {
            id,
            tenant,
            created_by,
            target,
            parts: Vec::new(),
            created_at: now,
            expires_at: now.saturating_add(limits.ttl_secs),
            state: UploadState::Open,
        })
    }

    /// `Ok` if open and unexpired.
    pub fn check_open(&self, now: u64) -> Result<()> {
        if self.state != UploadState::Open {
            return Err(RegistryError::UploadNotOpen);
        }
        self.check_unexpired(now)
    }

    /// `Ok` if not yet expired.
    pub fn check_unexpired(&self, now: u64) -> Result<()> {
        if now >= self.expires_at {
            return Err(RegistryError::UploadExpired);
        }
        Ok(())
    }

    /// `true` while the session holds a blob pin.
    pub fn is_finalizing(&self) -> bool {
        matches!(self.state, UploadState::Finalizing { .. })
    }

    /// `true` once finalized, failed or aborted.
    pub fn is_terminal(&self) -> bool {
        !matches!(
            self.state,
            UploadState::Open | UploadState::Finalizing { .. }
        )
    }

    /// Record (or replace) part `number`.
    pub fn record_part(&mut self, part: PartRecord, now: u64, limits: &UploadLimits) -> Result<()> {
        self.check_open(now)?;
        if part.number == 0 || part.number > limits.max_parts {
            return Err(RegistryError::UploadLimit("part number out of range"));
        }
        if part.size == 0 || part.size > limits.max_part_size {
            return Err(RegistryError::UploadLimit("part size out of range"));
        }
        match self.parts.binary_search_by_key(&part.number, |p| p.number) {
            Ok(i) => self.parts[i] = part,
            Err(i) => self.parts.insert(i, part),
        }
        let received: u64 = self.parts.iter().map(|p| p.size).sum();
        if received > self.target.size {
            return Err(RegistryError::UploadLimit("parts exceed the declared size"));
        }
        Ok(())
    }

    /// The parts to stream, in order, if they are complete: numbered
    /// `1..=n` without gaps, every part but the last at least
    /// `min_part_size`, summing to the declared size.
    pub fn complete_parts(&self, now: u64, limits: &UploadLimits) -> Result<&[PartRecord]> {
        self.check_open(now)?;
        self.check_parts(limits)
    }

    /// The completeness rules of [`Self::complete_parts`], without the
    /// state check.
    pub fn check_parts(&self, limits: &UploadLimits) -> Result<&[PartRecord]> {
        if self.parts.is_empty() {
            return Err(RegistryError::PartsIncomplete("no parts"));
        }
        if self
            .parts
            .iter()
            .enumerate()
            .any(|(i, p)| p.number as usize != i + 1)
        {
            return Err(RegistryError::PartsIncomplete("part numbers are not 1..=n"));
        }
        let last = self.parts.len() - 1;
        if self.parts[..last]
            .iter()
            .any(|p| p.size < limits.min_part_size)
        {
            return Err(RegistryError::PartsIncomplete(
                "non-final part below the minimum size",
            ));
        }
        if self.parts.iter().map(|p| p.size).sum::<u64>() != self.target.size {
            return Err(RegistryError::PartsIncomplete(
                "parts do not sum to the declared size",
            ));
        }
        Ok(&self.parts)
    }
}
