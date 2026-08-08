//! RVForge packaging and verification library (ADR-283, ADR-290, ADR-291).
//!
//! One canonical `.rvf` becomes installable packages for Windows, macOS,
//! Linux, and RVM. This crate is the part of that pipeline that authors,
//! reads, verifies, and describes the artifact; it does not build installers,
//! and it never runs the agent inside one.
//!
//! # The six operations
//!
//! | Function | Produces | Purpose |
//! |---|---|---|
//! | [`ContainerBuilder`] | [`AuthoredContainer`] | A valid, signed container built from segments |
//! | [`inspect`] | [`ForgeInspection`] | Identity, segment inventory, declared capabilities, signature metadata |
//! | [`verify`] | [`VerificationReport`] | Root-manifest and per-segment checks, one witness record each |
//! | [`build_manifest`] | [`CanonicalBuildManifest`] | The deterministic, canonical-JSON build request |
//! | [`provenance`] | [`ProvenanceRecord`] | The trace from output back to input, runtime, revision, builder, and signer |
//! | [`checksums`] | [`Sha256Manifest`] | SHA-256 of every produced artifact |
//!
//! Authoring is the newest of the six and the only one that writes. It exists
//! because the rest of the pipeline consumes an artifact that, until it was
//! added, nothing in the toolchain could produce; see [`author`].
//!
//! # What this crate will not do
//!
//! Three refusals are load-bearing rather than incidental, and each traces to
//! a stated invariant:
//!
//! - **Nothing here executes an RVF.** Build workers must never run the
//!   submitted artifact (ADR-283 invariant 2, ADR-290 §1). Inspection reads
//!   64-byte segment headers and byte ranges; the only time an executable
//!   segment's payload is touched at all is to hash it. Authoring is held to
//!   the same line: it writes payload bytes without interpreting them.
//! - **Unsigned executable segments are refused by default.** ADR-283
//!   invariant 3 and the RVM loading requirements §4.3. The single opt-out,
//!   [`VerifyOptions::allow_unsigned_executable`], exists for the unsigned
//!   development builds of ADR-290 §5, which provenance labels as such.
//! - **Unvalidated targets are refused before any compute is spent.** ADR-291
//!   §2 gates at manifest-generation time, so [`build_manifest`] rejects a
//!   target absent from the compatibility matrix rather than approximating,
//!   downgrading, or substituting one.
//!
//! # Determinism
//!
//! Manifests, provenance records, and verification reports are hashed and
//! compared across the three workers of a single build, so every one of them
//! is a pure function of its inputs. Nothing here reads the clock, the
//! environment, or the network; canonical JSON fixes key order and refuses
//! floats (see [`canonical`]); and callers pass timestamps in rather than
//! having them sampled.
//!
//! # Relationship to the RVF format crates
//!
//! The container format is not reimplemented here. `rvf-types`, `rvf-wire`,
//! and `rvf-crypto` — from the nested workspace at `crates/rvf` — supply the
//! segment header layout, the reader, the content-hash algorithms, the
//! signature footer codec, and Ed25519 segment signing. This crate adds the
//! Forge-level policy on top of them.
//!
//! # Example
//!
//! ```
//! use rvf_forge_core::{
//!     build_manifest, inspect_bytes, verify_bytes, AppMetadata, BuildSpec,
//!     CapabilityPolicy, PackagingMode, RuntimeProfile, Target, VerifyOptions,
//! };
//! # use rvf_forge_core::testkit::minimal_container;
//! let rvf = minimal_container("network");
//!
//! let inspection = inspect_bytes(&rvf)?;
//! assert!(inspection.all_executables_signed());
//!
//! let report = verify_bytes(&rvf, &VerifyOptions::default())?;
//! assert!(report.ok);
//!
//! let manifest = build_manifest(&BuildSpec {
//!     rvf_identity: inspection.rvf_identity.clone(),
//!     app: AppMetadata {
//!         name: "Example Agent".into(),
//!         version: "1.0.0".into(),
//!         publisher: "Example Inc".into(),
//!         license: None,
//!         description: None,
//!     },
//!     targets: vec![Target::WindowsX64, Target::MacosUniversal],
//!     packaging_mode: PackagingMode::Embedded,
//!     runtime_profile: RuntimeProfile::Wasm,
//!     capability_policy: CapabilityPolicy::deny_all(),
//!     rvm_version: "0.4.0".into(),
//!     rvm_commit: "abc123".into(),
//!     compatibility_matrix_revision: None,
//! })?;
//!
//! // The seven-field contract every generated package embeds (ADR-291 §1).
//! assert_eq!(manifest.runtime_contract().rvf_identity, inspection.rvf_identity);
//! # Ok::<(), rvf_forge_core::ForgeError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod author;
pub mod canonical;
pub mod capability;
pub mod checksums;
mod container;
pub mod error;
pub mod inspect;
pub mod manifest;
pub mod provenance;
pub mod target;
pub mod testkit;
pub mod verify;

pub use author::{AuthoredContainer, ContainerBuilder, DEFAULT_CHECKSUM_ALGO};
pub use canonical::{canonical_sha256_hex, hex_lower, to_canonical_json};
pub use capability::{CapabilityClass, CapabilityGrant, CapabilityPolicy};
pub use checksums::{
    checksums, resolve_artifact, verify_checksums, ChecksumVerification, Sha256Entry,
    Sha256Manifest, CHECKSUM_SCHEMA,
};
pub use error::{ErrorCode, ForgeError, Result};
pub use inspect::{
    inspect, inspect_bytes, ForgeInspection, SegmentSummary, SignatureAlgoSummary,
    CAPABILITY_DECLARATION_KEY, INSPECTION_SCHEMA,
};
pub use manifest::{
    build_manifest, AppMetadata, BuildSpec, CanonicalBuildManifest, PackagingMode, RuntimeContract,
    RuntimeProfile, BUILD_MANIFEST_SCHEMA, RUNTIME_CONTRACT_SCHEMA, STATE_SCHEMA_VERSION,
    WITNESS_SCHEMA_VERSION,
};
pub use provenance::{
    provenance, BuilderIdentity, BuilderKind, ProvenanceInputs, ProvenanceRecord, SigningIdentity,
    SigningPosture, PROVENANCE_SCHEMA,
};
/// The segment type discriminator, re-exported so callers of
/// [`author::ContainerBuilder`] need not depend on `rvf-types` directly.
pub use rvf_types::SegmentType;
pub use target::Target;
pub use verify::{
    verify, verify_bytes, CheckKind, Outcome, VerificationRecord, VerificationReport,
    VerifyOptions, VERIFICATION_SCHEMA,
};

/// The version of this crate, recorded in provenance records.
pub const FORGE_CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_identifiers_are_distinct_and_versioned() {
        let schemas = [
            INSPECTION_SCHEMA,
            VERIFICATION_SCHEMA,
            BUILD_MANIFEST_SCHEMA,
            RUNTIME_CONTRACT_SCHEMA,
            PROVENANCE_SCHEMA,
            CHECKSUM_SCHEMA,
        ];
        let unique: std::collections::BTreeSet<_> = schemas.iter().collect();
        assert_eq!(unique.len(), schemas.len());
        for s in schemas {
            assert!(s.starts_with("rvforge."), "{s}");
            assert!(s.ends_with("/1"), "{s}");
        }
    }

    #[test]
    fn adr_291_schema_versions_are_one() {
        assert_eq!(STATE_SCHEMA_VERSION, 1);
        assert_eq!(WITNESS_SCHEMA_VERSION, 1);
    }
}
