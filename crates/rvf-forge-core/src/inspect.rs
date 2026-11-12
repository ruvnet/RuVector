//! Inspection: read an RVF's identity, segment inventory, declared
//! capabilities, and signature metadata without executing anything.
//!
//! This is the operation a build worker performs on a submitted artifact.
//! ADR-283 core invariant 2 and ADR-290 §1 both state it plainly: packaging,
//! validation, and scanning treat the RVF as inert data. Nothing here maps an
//! executable segment, resolves an entry point, or interprets a payload. The
//! only bytes read from an executable segment are read to hash them.

use crate::canonical::hex_lower;
use crate::capability::CapabilityClass;
use crate::container::{self, ParsedSegment};
use crate::error::{ForgeError, Result};
use rvf_types::{SegmentType, SEGMENT_HEADER_SIZE};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

/// Schema identifier carried by every [`ForgeInspection`].
pub const INSPECTION_SCHEMA: &str = "rvforge.inspection/1";

/// What an RVF declares about one segment, as observed from its header.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentSummary {
    /// Zero-based position in the segment stream.
    pub index: usize,
    /// Byte offset of the segment header within the container.
    pub offset: u64,
    /// The segment's ordinal, from its header.
    pub segment_id: u64,
    /// Raw type discriminator.
    pub segment_type: u8,
    /// Human-readable type name, e.g. `wasm`, `manifest`, `unknown(0xee)`.
    pub segment_type_name: String,
    /// Declared payload length in bytes.
    pub payload_length: u64,
    /// Raw header flags bitfield.
    pub flags: u16,
    /// Whether a signature footer follows the payload.
    pub signed: bool,
    /// Whether the payload is declared encrypted.
    pub encrypted: bool,
    /// Whether this segment's payload is code a runtime would execute.
    pub executable: bool,
    /// The header's 128-bit content hash, lowercase hex.
    pub content_hash: String,
    /// Content-hash algorithm: 0 CRC32C (deprecated), 1 XXH3-128, 2 SHAKE-256.
    pub checksum_algo: u8,
    /// Signature algorithm from the footer, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature_algo: Option<SignatureAlgoSummary>,
}

/// Signature algorithm named by a segment's footer.
///
/// A mirror of `rvf_types::SignatureAlgo` that is serializable and that keeps
/// an unrecognized discriminator rather than failing inspection: reporting
/// "this file claims algorithm 9" is more useful to an operator than refusing
/// to describe the file at all.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignatureAlgoSummary {
    /// Raw algorithm discriminator.
    pub algo: u16,
    /// Name: `ed25519`, `ml-dsa-65`, `slh-dsa-128s`, or `unknown(N)`.
    pub name: String,
    /// Signature length in bytes, as declared by the footer.
    pub signature_length: u16,
}

/// Everything inspection can learn about an RVF without running it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForgeInspection {
    /// Schema identifier; always [`INSPECTION_SCHEMA`].
    pub schema: String,
    /// SHA-256 of the whole container, lowercase hex. This is `rvfIdentity`
    /// in the runtime contract (ADR-291 §1) and must be identical across every
    /// package produced from this artifact.
    pub rvf_identity: String,
    /// Container size in bytes.
    pub byte_length: u64,
    /// Every segment, in stream order.
    pub segments: Vec<SegmentSummary>,
    /// Offset of the root manifest segment, when one was found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_manifest_offset: Option<u64>,
    /// Capability classes the artifact declares. Empty means default-deny with
    /// nothing re-opened (ADR-286 §1).
    pub declared_capabilities: Vec<CapabilityClass>,
    /// Count of executable segments (kernel, eBPF, WASM).
    pub executable_segment_count: usize,
    /// `segment_id` of every executable segment that carries no signature.
    /// Non-empty means [`crate::verify`] will refuse the artifact under its
    /// default options.
    pub unsigned_executable_segments: Vec<u64>,
}

impl ForgeInspection {
    /// Whether every executable segment carries a signature footer.
    ///
    /// This reports only the *presence* of signatures. Whether they verify
    /// against a trusted key is [`crate::verify`]'s job.
    pub fn all_executables_signed(&self) -> bool {
        self.unsigned_executable_segments.is_empty()
    }

    /// The root manifest segment, when the container has one.
    pub fn root_manifest(&self) -> Option<&SegmentSummary> {
        self.segments
            .iter()
            .find(|s| s.segment_type == SegmentType::Manifest as u8)
    }
}

/// Inspect the RVF at `path`.
///
/// # Errors
///
/// [`ForgeError::Io`] if the file cannot be read, [`ForgeError::InvalidRvf`]
/// if it is not a well-formed container.
pub fn inspect(path: impl AsRef<Path>) -> Result<ForgeInspection> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| ForgeError::io(path, &e))?;
    inspect_bytes(&data)
}

/// Inspect an in-memory RVF container.
///
/// # Errors
///
/// [`ForgeError::InvalidRvf`] if `data` is not a well-formed container.
pub fn inspect_bytes(data: &[u8]) -> Result<ForgeInspection> {
    let segments = container::walk(data)?;

    let mut summaries = Vec::with_capacity(segments.len());
    let mut unsigned_executables = Vec::new();
    let mut executable_count = 0usize;

    for seg in &segments {
        if seg.is_executable() {
            executable_count += 1;
            if !seg.is_signed() {
                unsigned_executables.push(seg.header.segment_id);
            }
        }
        summaries.push(summarize(data, seg));
    }

    let root_manifest_offset = segments
        .iter()
        .rev()
        .find(|s| s.header.seg_type == SegmentType::Manifest as u8)
        .map(|s| s.offset as u64);

    Ok(ForgeInspection {
        schema: INSPECTION_SCHEMA.to_string(),
        rvf_identity: hex_lower(&rvf_types::sha256(data)),
        byte_length: data.len() as u64,
        segments: summaries,
        root_manifest_offset,
        declared_capabilities: declared_capabilities(data, &segments),
        executable_segment_count: executable_count,
        unsigned_executable_segments: unsigned_executables,
    })
}

fn summarize(data: &[u8], seg: &ParsedSegment) -> SegmentSummary {
    let signature_algo = seg.footer.as_ref().and_then(|range| {
        rvf_crypto::decode_signature_footer(&data[range.clone()])
            .ok()
            .map(|f| SignatureAlgoSummary {
                algo: f.sig_algo,
                name: signature_algo_name(f.sig_algo),
                signature_length: f.sig_length,
            })
    });

    SegmentSummary {
        index: seg.index,
        offset: seg.offset as u64,
        segment_id: seg.header.segment_id,
        segment_type: seg.header.seg_type,
        segment_type_name: container::segment_type_name(seg.header.seg_type),
        payload_length: seg.header.payload_length,
        flags: seg.header.flags,
        signed: seg.is_signed(),
        encrypted: seg.is_encrypted(),
        executable: seg.is_executable(),
        content_hash: hex_lower(&seg.header.content_hash),
        checksum_algo: seg.header.checksum_algo,
        signature_algo,
    }
}

fn signature_algo_name(algo: u16) -> String {
    match rvf_types::SignatureAlgo::try_from(algo) {
        Ok(rvf_types::SignatureAlgo::Ed25519) => "ed25519".into(),
        Ok(rvf_types::SignatureAlgo::MlDsa65) => "ml-dsa-65".into(),
        Ok(rvf_types::SignatureAlgo::SlhDsa128s) => "slh-dsa-128s".into(),
        Err(raw) => format!("unknown({raw})"),
    }
}

/// Capability classes declared by the artifact.
///
/// Declarations live in `META` segments as a newline-separated list under the
/// well-known key [`CAPABILITY_DECLARATION_KEY`]. Reading them is byte
/// scanning over inert metadata; no other segment kind is consulted, and in
/// particular an executable segment is never parsed to infer what it might do.
///
/// An unrecognized class name is skipped rather than failing inspection. It
/// cannot widen the granted set: the runtime is default-deny, so a class the
/// runtime does not know is a class the runtime does not grant (ADR-286 §1).
fn declared_capabilities(data: &[u8], segments: &[ParsedSegment]) -> Vec<CapabilityClass> {
    let mut found = BTreeSet::new();
    for seg in segments {
        if seg.header.seg_type != SegmentType::Meta as u8 || seg.is_encrypted() {
            continue;
        }
        let payload = &data[seg.payload.clone()];
        let Ok(text) = std::str::from_utf8(payload) else {
            continue;
        };
        for line in text.lines() {
            let Some(rest) = line.strip_prefix(CAPABILITY_DECLARATION_KEY) else {
                continue;
            };
            let Some(value) = rest.strip_prefix('=') else {
                continue;
            };
            for name in value.split(',') {
                if let Some(class) = CapabilityClass::from_wire(name.trim()) {
                    found.insert(class);
                }
            }
        }
    }
    found.into_iter().collect()
}

/// The `META` segment key under which an RVF declares its capability classes,
/// as `rvf.capabilities=network,filesystem`.
pub const CAPABILITY_DECLARATION_KEY: &str = "rvf.capabilities";

/// The fixed size of a segment header, re-exported for callers that compute
/// offsets against an inspection result.
pub const HEADER_SIZE: usize = SEGMENT_HEADER_SIZE;

#[cfg(test)]
mod tests {
    use super::*;
    use rvf_types::SegmentFlags;
    use rvf_wire::write_segment;

    fn meta(body: &str) -> Vec<u8> {
        write_segment(
            SegmentType::Meta as u8,
            body.as_bytes(),
            SegmentFlags::empty(),
            7,
        )
    }

    #[test]
    fn identity_is_sha256_of_the_whole_container() {
        let data = meta("hello");
        let got = inspect_bytes(&data).unwrap();
        assert_eq!(got.rvf_identity, hex_lower(&rvf_types::sha256(&data)));
        assert_eq!(got.byte_length, data.len() as u64);
    }

    #[test]
    fn identity_changes_when_any_byte_changes() {
        let data = meta("hello");
        let mut tampered = data.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0x01;
        assert_ne!(
            inspect_bytes(&data).unwrap().rvf_identity,
            inspect_bytes(&tampered).unwrap().rvf_identity
        );
    }

    #[test]
    fn inventories_every_segment_in_order() {
        let mut data = meta("m");
        data.extend(write_segment(
            SegmentType::Vec as u8,
            b"vectors",
            SegmentFlags::empty(),
            8,
        ));
        let got = inspect_bytes(&data).unwrap();
        assert_eq!(got.segments.len(), 2);
        assert_eq!(got.segments[0].segment_type_name, "meta");
        assert_eq!(got.segments[0].segment_id, 7);
        assert_eq!(got.segments[1].segment_type_name, "vec");
        assert_eq!(got.segments[1].payload_length, 7);
        assert_eq!(got.segments[1].index, 1);
    }

    #[test]
    fn unsigned_wasm_segment_is_flagged() {
        let mut data = meta("m");
        data.extend(write_segment(
            SegmentType::Wasm as u8,
            b"\0asm not really",
            SegmentFlags::empty(),
            9,
        ));
        let got = inspect_bytes(&data).unwrap();
        assert_eq!(got.executable_segment_count, 1);
        assert_eq!(got.unsigned_executable_segments, vec![9]);
        assert!(!got.all_executables_signed());
    }

    #[test]
    fn non_executable_segments_are_not_counted() {
        let got = inspect_bytes(&meta("m")).unwrap();
        assert_eq!(got.executable_segment_count, 0);
        assert!(got.all_executables_signed());
    }

    #[test]
    fn capability_declarations_are_read_from_meta() {
        let data = meta("name=agent\nrvf.capabilities=network, filesystem,clock\n");
        let got = inspect_bytes(&data).unwrap();
        assert_eq!(
            got.declared_capabilities,
            vec![
                CapabilityClass::Filesystem,
                CapabilityClass::Network,
                CapabilityClass::Clock
            ]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn unknown_capability_names_are_skipped() {
        let got = inspect_bytes(&meta("rvf.capabilities=network,teleportation")).unwrap();
        assert_eq!(got.declared_capabilities, vec![CapabilityClass::Network]);
    }

    #[test]
    fn absent_declaration_means_default_deny() {
        let got = inspect_bytes(&meta("name=agent")).unwrap();
        assert!(got.declared_capabilities.is_empty());
    }

    #[test]
    fn root_manifest_offset_is_reported() {
        let mut data = meta("m");
        let manifest_offset = data.len();
        data.extend(write_segment(
            SegmentType::Manifest as u8,
            b"root",
            SegmentFlags::empty(),
            99,
        ));
        let got = inspect_bytes(&data).unwrap();
        assert_eq!(got.root_manifest_offset, Some(manifest_offset as u64));
        assert_eq!(got.root_manifest().unwrap().segment_id, 99);
    }

    #[test]
    fn malformed_container_is_rejected() {
        assert_eq!(
            inspect_bytes(b"not an rvf").unwrap_err().code(),
            crate::ErrorCode::InvalidRvf
        );
    }

    #[test]
    fn inspection_serializes_to_json() {
        let json = serde_json::to_string(&inspect_bytes(&meta("m")).unwrap()).unwrap();
        assert!(json.contains("\"rvfIdentity\""), "{json}");
        assert!(json.contains("\"segmentTypeName\":\"meta\""), "{json}");
    }
}
