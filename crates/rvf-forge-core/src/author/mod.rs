//! Authoring: build a valid RVF container from segments and sign it.
//!
//! Every other module in this crate reads an artifact someone else produced.
//! This one writes, and it is the only one that does. It exists because a
//! publisher who has just run `rvforge init --keygen` has a key, a config, and
//! nothing to sign: without a supported way to *make* an `.rvf`, the first
//! documented command after `init` cannot run.
//!
//! # What authoring is not
//!
//! Writing bytes is not running them. Core invariant 2 (ADR-283 §2) holds here
//! exactly as it does in [`crate::inspect`]: a payload handed to
//! [`ContainerBuilder::segment`] is copied, length-checked, and hashed. It is
//! never parsed as code, validated as WASM, linked, or executed, and that is
//! true whatever [`SegmentType`] the caller labels it with.
//!
//! # The shape produced
//!
//! ```text
//! [ META segment: capability declaration ]   (when any are declared)
//! [ caller segments, in the order supplied ]
//! [ MANIFEST segment: the root manifest    ]
//! [ 4096-byte Level-0 root manifest page   ]  <- at EOF, always
//! ```
//!
//! The trailing page is what makes the output a *file* rather than a bare
//! segment stream: `rvf-manifest` puts a fixed 4096-byte Level-0 root at EOF,
//! and readers discover an RVF from its tail. [`crate::container`] trims it
//! before walking.
//!
//! # Determinism
//!
//! Identical input yields identical bytes. Nothing here reads the clock or the
//! environment: segment timestamps and the root page's created/modified fields
//! are zero, segment ids are assigned by position, and the file id is supplied
//! by the caller rather than generated. Two publishers building the same agent
//! from the same inputs get the same artifact, so the SHA-256 that identifies
//! it is reproducible.
//!
//! # Example
//!
//! ```
//! use rvf_forge_core::{author::ContainerBuilder, inspect_bytes, CapabilityClass};
//!
//! let built = ContainerBuilder::new()
//!     .declare_capability(CapabilityClass::Network)
//!     .segment(rvf_forge_core::SegmentType::Vec, b"embeddings".to_vec())
//!     .build()?;
//!
//! let inspection = inspect_bytes(&built.bytes)?;
//! assert_eq!(inspection.declared_capabilities, vec![CapabilityClass::Network]);
//! # Ok::<(), rvf_forge_core::ForgeError>(())
//! ```

mod root_manifest;
mod segment;

pub use root_manifest::{
    root_manifest_page_of, root_manifest_signing_bytes, verify_root_manifest_signature,
};

use crate::capability::CapabilityClass;
use crate::error::{ForgeError, Result};
use crate::inspect::CAPABILITY_DECLARATION_KEY;
use ed25519_dalek::SigningKey;
use root_manifest::root_manifest_page;
use rvf_types::{SegmentType, MAX_SEGMENT_PAYLOAD};
pub(crate) use segment::write_segment_bytes;
use std::collections::BTreeSet;

/// Content-hash algorithm authored containers use: SHAKE-256 truncated to 128
/// bits (`checksum_algo` 2).
///
/// The format's default is XXH3-128, which is faster. SHAKE-256 is chosen here
/// because authoring is the one operation a *non-Rust* implementation has to
/// reproduce byte for byte — the TypeScript CLI writes the same containers —
/// and SHAKE-256 is reachable from any standard crypto library, where XXH3-128
/// would mean shipping a hand-ported hash on the other side. It is also the
/// cryptographic option, which is the right default for a signing-adjacent
/// tool.
pub const DEFAULT_CHECKSUM_ALGO: u8 = 2;

/// Byte length of an Ed25519 signature.
const ED25519_SIGNATURE_LENGTH: usize = 64;

/// A container produced by [`ContainerBuilder::build`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthoredContainer {
    /// The complete file, ready to write to disk.
    pub bytes: Vec<u8>,
    /// The 16-byte file id recorded in the root manifest page.
    pub file_id: [u8; 16],
    /// How many segments were written, including the generated capability and
    /// root-manifest segments.
    pub segment_count: usize,
    /// Whether a signing key was supplied, and so whether the root manifest
    /// page and the signable segments carry signatures.
    pub signed: bool,
}

/// Builds a valid RVF container.
///
/// The builder owns two format obligations the caller should not have to know
/// about: every container ends with a `MANIFEST` segment (verification refuses
/// one without a root manifest), and every container ends with a Level-0 root
/// manifest page.
#[derive(Default)]
pub struct ContainerBuilder {
    capabilities: BTreeSet<CapabilityClass>,
    segments: Vec<(SegmentType, Vec<u8>)>,
    manifest_payload: Option<Vec<u8>>,
    file_id: [u8; 16],
    checksum_algo: Option<u8>,
    signing_key: Option<SigningKey>,
}

impl ContainerBuilder {
    /// A builder with no segments, no declared capabilities, and no key.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the 16-byte file id recorded in the root manifest page.
    ///
    /// Taken from the caller rather than generated so that authoring stays a
    /// pure function of its inputs.
    pub fn file_id(mut self, file_id: [u8; 16]) -> Self {
        self.file_id = file_id;
        self
    }

    /// Declare one capability class in the container's `META` segment.
    ///
    /// Declaring is not granting. The runtime is default-deny (ADR-286 §1);
    /// this records what the artifact *asks* for, which is what
    /// [`crate::inspect`] reports and what a reviewer reads.
    pub fn declare_capability(mut self, class: CapabilityClass) -> Self {
        self.capabilities.insert(class);
        self
    }

    /// Declare several capability classes at once.
    pub fn declare_capabilities(
        mut self,
        classes: impl IntoIterator<Item = CapabilityClass>,
    ) -> Self {
        self.capabilities.extend(classes);
        self
    }

    /// Append a segment. Payload bytes are stored, never interpreted.
    pub fn segment(mut self, seg_type: SegmentType, payload: impl Into<Vec<u8>>) -> Self {
        self.segments.push((seg_type, payload.into()));
        self
    }

    /// Override the root `MANIFEST` segment's payload.
    ///
    /// One is always written; this replaces its contents.
    pub fn manifest_payload(mut self, payload: impl Into<Vec<u8>>) -> Self {
        self.manifest_payload = Some(payload.into());
        self
    }

    /// Use a specific `checksum_algo` instead of [`DEFAULT_CHECKSUM_ALGO`].
    pub fn checksum_algo(mut self, algo: u8) -> Self {
        self.checksum_algo = Some(algo);
        self
    }

    /// Sign with `key`: the root manifest page, the `MANIFEST` segment, and
    /// every executable segment get Ed25519 signatures.
    ///
    /// Without a key the container is unsigned, which
    /// [`crate::verify`] refuses under default options as soon as it holds an
    /// executable segment (ADR-283 invariant 3).
    pub fn sign_with(mut self, key: SigningKey) -> Self {
        self.signing_key = Some(key);
        self
    }

    /// Assemble the container.
    ///
    /// # Errors
    ///
    /// [`ForgeError::InvalidRvf`] if a payload exceeds [`MAX_SEGMENT_PAYLOAD`].
    pub fn build(self) -> Result<AuthoredContainer> {
        let algo = self.checksum_algo.unwrap_or(DEFAULT_CHECKSUM_ALGO);
        let key = self.signing_key.as_ref();

        let mut planned: Vec<(SegmentType, Vec<u8>)> = Vec::new();
        if !self.capabilities.is_empty() {
            planned.push((
                SegmentType::Meta,
                self.capability_declaration().into_bytes(),
            ));
        }
        planned.extend(self.segments);
        planned.push((
            SegmentType::Manifest,
            self.manifest_payload.unwrap_or_else(|| b"root".to_vec()),
        ));

        let segment_count = planned.len();
        let mut body: Vec<u8> = Vec::new();
        let mut manifest_span = (0usize, 0usize);

        for (index, (seg_type, payload)) in planned.into_iter().enumerate() {
            if payload.len() as u64 > MAX_SEGMENT_PAYLOAD {
                return Err(ForgeError::invalid_rvf(format!(
                    "segment {index} carries {} payload bytes, over the \
                     {MAX_SEGMENT_PAYLOAD}-byte maximum",
                    payload.len()
                )));
            }
            let signer = key.filter(|_| is_signable(seg_type));
            let offset = body.len();
            // Segment ids are 1-based and assigned by position, so the same
            // inputs always produce the same ids.
            let encoded = write_segment_bytes(seg_type, &payload, index as u64 + 1, algo, signer);
            if seg_type == SegmentType::Manifest {
                manifest_span = (offset, encoded.len());
            }
            body.extend_from_slice(&encoded);
        }

        let page = root_manifest_page(
            &self.file_id,
            manifest_span.0 as u64,
            manifest_span.1 as u64,
            key,
        );
        body.extend_from_slice(&page);

        Ok(AuthoredContainer {
            bytes: body,
            file_id: self.file_id,
            segment_count,
            signed: key.is_some(),
        })
    }

    /// The `META` payload line declaring the requested capability classes.
    fn capability_declaration(&self) -> String {
        let names: Vec<&str> = self.capabilities.iter().map(|c| c.as_str()).collect();
        format!("{CAPABILITY_DECLARATION_KEY}={}\n", names.join(","))
    }
}

/// Whether a segment of this type carries a signature footer when a key is
/// supplied.
///
/// Executable segments must be signed for a runtime to load them (ADR-283
/// invariant 3). The root manifest is signed because it is the thing every
/// other check is anchored to. Signing inert data segments as well would cost
/// 128 bytes each and prove nothing the root manifest signature does not.
fn is_signable(seg_type: SegmentType) -> bool {
    matches!(
        seg_type,
        SegmentType::Kernel | SegmentType::Ebpf | SegmentType::Wasm | SegmentType::Manifest
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TestKeypair;
    use crate::{inspect_bytes, verify_bytes, VerifyOptions};
    use rvf_types::SEGMENT_HEADER_SIZE;

    #[test]
    fn a_minimal_authored_container_inspects_and_verifies() {
        let built = ContainerBuilder::new().build().unwrap();
        let inspection = inspect_bytes(&built.bytes).unwrap();
        assert_eq!(inspection.segments.len(), 1);
        assert_eq!(
            inspection.root_manifest().unwrap().segment_type_name,
            "manifest"
        );

        let report = verify_bytes(&built.bytes, &VerifyOptions::default()).unwrap();
        assert!(report.ok, "{:?}", report.failures());
    }

    #[test]
    fn declared_capabilities_round_trip_through_inspection() {
        let built = ContainerBuilder::new()
            .declare_capabilities([CapabilityClass::Network, CapabilityClass::Filesystem])
            .build()
            .unwrap();
        let inspection = inspect_bytes(&built.bytes).unwrap();
        assert_eq!(
            inspection.declared_capabilities,
            vec![CapabilityClass::Filesystem, CapabilityClass::Network]
        );
    }

    #[test]
    fn an_unsigned_executable_is_refused_by_default_and_allowed_by_opt_in() {
        let built = ContainerBuilder::new()
            .segment(SegmentType::Wasm, b"\0asm\x01\0\0\0".to_vec())
            .build()
            .unwrap();

        let inspection = inspect_bytes(&built.bytes).unwrap();
        assert_eq!(inspection.executable_segment_count, 1);
        assert!(!inspection.all_executables_signed());

        let strict = verify_bytes(&built.bytes, &VerifyOptions::default()).unwrap();
        assert!(!strict.ok);

        let permissive = verify_bytes(
            &built.bytes,
            &VerifyOptions {
                allow_unsigned_executable: true,
                ..VerifyOptions::default()
            },
        )
        .unwrap();
        assert!(permissive.ok, "{:?}", permissive.failures());
    }

    #[test]
    fn a_signed_executable_verifies_against_the_signing_key() {
        let kp = TestKeypair::deterministic(11);
        let built = ContainerBuilder::new()
            .segment(SegmentType::Wasm, b"\0asm\x01\0\0\0".to_vec())
            .sign_with(kp.signing.clone())
            .build()
            .unwrap();

        assert!(built.signed);
        assert!(inspect_bytes(&built.bytes)
            .unwrap()
            .all_executables_signed());

        let report = verify_bytes(
            &built.bytes,
            &VerifyOptions::with_trusted_keys(vec![kp.public]),
        )
        .unwrap();
        assert!(report.ok, "{:?}", report.failures());
    }

    #[test]
    fn tampering_with_any_byte_breaks_verification() {
        let kp = TestKeypair::deterministic(14);
        let built = ContainerBuilder::new()
            .segment(SegmentType::Wasm, b"\0asm\x01\0\0\0".to_vec())
            .sign_with(kp.signing.clone())
            .build()
            .unwrap();

        let mut tampered = built.bytes.clone();
        // Flip a byte inside the WASM payload, past the first segment header.
        let target = SEGMENT_HEADER_SIZE + 4;
        tampered[target] ^= 0xff;
        let report = verify_bytes(
            &tampered,
            &VerifyOptions::with_trusted_keys(vec![kp.public]),
        )
        .unwrap();
        assert!(!report.ok);
    }

    #[test]
    fn identical_input_yields_identical_bytes() {
        let build = || {
            ContainerBuilder::new()
                .file_id([7u8; 16])
                .declare_capability(CapabilityClass::Network)
                .segment(SegmentType::Vec, b"payload".to_vec())
                .sign_with(TestKeypair::deterministic(3).signing)
                .build()
                .unwrap()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn capability_order_does_not_change_the_bytes() {
        let one = ContainerBuilder::new()
            .declare_capabilities([CapabilityClass::Network, CapabilityClass::Clock])
            .build()
            .unwrap();
        let two = ContainerBuilder::new()
            .declare_capabilities([CapabilityClass::Clock, CapabilityClass::Network])
            .build()
            .unwrap();
        assert_eq!(one.bytes, two.bytes);
    }

    #[test]
    fn segments_appear_in_the_order_supplied_with_the_manifest_last() {
        let built = ContainerBuilder::new()
            .declare_capability(CapabilityClass::Network)
            .segment(SegmentType::Vec, b"a".to_vec())
            .segment(SegmentType::Index, b"b".to_vec())
            .build()
            .unwrap();
        let names: Vec<String> = inspect_bytes(&built.bytes)
            .unwrap()
            .segments
            .into_iter()
            .map(|s| s.segment_type_name)
            .collect();
        assert_eq!(names, ["meta", "vec", "index", "manifest"]);
        assert_eq!(built.segment_count, 4);
    }

    #[test]
    fn an_empty_payload_is_legal_and_round_trips() {
        let built = ContainerBuilder::new()
            .segment(SegmentType::Vec, Vec::new())
            .build()
            .unwrap();
        let segments = inspect_bytes(&built.bytes).unwrap().segments;
        assert_eq!(segments[0].payload_length, 0);
        assert!(
            verify_bytes(&built.bytes, &VerifyOptions::default())
                .unwrap()
                .ok
        );
    }

    #[test]
    fn authored_containers_use_shake256_content_hashes() {
        let built = ContainerBuilder::new()
            .segment(SegmentType::Vec, b"hash me".to_vec())
            .build()
            .unwrap();
        let vec_segment = &inspect_bytes(&built.bytes).unwrap().segments[0];
        assert_eq!(vec_segment.checksum_algo, DEFAULT_CHECKSUM_ALGO);
        assert_eq!(
            vec_segment.content_hash,
            crate::hex_lower(&rvf_wire::hash::compute_shake256_128(b"hash me"))
        );
    }
}
