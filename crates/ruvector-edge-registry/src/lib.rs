//! # ruvector-edge-registry
//!
//! Core of the rv-registry service (ADR-351 §3, M5): `/v1/rvf/{name}/{version}*`.
//!
//! - [`name`], [`version`]: `@scope/name` package names (lowercase, bounded,
//!   reserved and brand look-alike scopes refused) and strict SemVer
//!   versions (no build metadata).
//! - [`scopes`]: the global scope directory (one root DO): explicit claims,
//!   one owner per confusable skeleton, a per-tenant cap.
//! - [`mod@validate`]: the streaming strict RVF validator over untrusted bytes
//!   (rvf-wire header / root-page readers, size and segment-count limits,
//!   per-segment content hashes, runtime manifest and VEC_SEG layouts).
//! - [`manifest`]: the package manifest (dim, metric, segments with SHA-256,
//!   size, uploader, visibility, opaque witness-root provenance, yank).
//! - [`keys`]: content-addressed R2 blob keys (tenant- and scope-scoped for
//!   private and tenant-visible objects, global write-once for public) and
//!   the registry DO name.
//! - [`upload`]: server-minted `upload_id` sessions with numbered parts,
//!   expiry, a freeze at plan time and a digest-checked finalize.
//! - [`view`]: what reads return; foreign tenants never see tenant keys or
//!   subjects.
//! - [`authz`]: visibility × capability matrix. `private` = uploader and
//!   admins; `tenant` = every member with read; `public` = any tenant with
//!   read. Cross-tenant access to a non-public version is always `404`.
//! - [`registry`]: the service over a [`ports::KvStore`] index (a registry
//!   Durable Object per scope; no D1), with immutable versions, pinned blob
//!   reference counts, yank / unyank, evidence-checked public publish,
//!   cursor-paginated listings and a maintenance sweep.
//! - [`mem`]: in-memory store, fixed clock and test entropy.
//!
//! Pure Rust, `wasm32-unknown-unknown` clean: no `std::time`, no `getrandom`,
//! no filesystem. Time, randomness and storage come in through [`ports`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod authz;
pub mod error;
pub mod keys;
pub mod manifest;
pub mod mem;
pub mod name;
pub mod ports;
pub mod registry;
pub mod scopes;
pub mod upload;
pub mod validate;
pub mod version;
pub mod view;

pub use authz::{authorize, Action, Caller, Denial};
pub use error::RegistryError;
pub use keys::{blob_key, BlobKey};
pub use manifest::{PackageManifest, Provenance, Visibility};
pub use name::{PackageName, Scope};
pub use registry::{FinalizeReport, Registry, RegistryConfig};
pub use scopes::{ScopeClaim, ScopeDirectory};
pub use upload::{BeginUpload, ObjectEvidence, PartRecord, UploadId, UploadSession};
pub use validate::{validate, StreamValidator, ValidatedRvf, ValidationError, ValidationLimits};
pub use version::Version;
pub use view::{ManifestView, PublicManifest};

pub(crate) mod hexser {
    //! Serde helper: `[u8; 32]` as lowercase hex.
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(v))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        let mut out = [0u8; 32];
        hex::decode_to_slice(&s, &mut out).map_err(serde::de::Error::custom)?;
        Ok(out)
    }
}
