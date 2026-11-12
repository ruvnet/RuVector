//! Snapshot manifest: identity, shape, per-chunk sha256 and the manifest root.
//!
//! The manifest body has one canonical binary encoding; the root is
//! `sha256("rvedge-snapshot-root-v1\0" ‖ body)`. The stored object
//! (`manifest.rvf`) is a single rvf-wire `MANIFEST_SEG` whose payload is
//! `body ‖ root ‖ signature block`. The segment's XXH3 content hash is only a
//! transport check; the root (and the witness chain / signature over it) is
//! the integrity check.

use crate::bytes::{put_str16, Reader};
use crate::error::SnapshotError;
use crate::types::{
    sha256, validate_collection_uid, validate_service, validate_tenant_key, Metric,
};
use crate::types::{MAX_DIM, MAX_SERVICE_LEN, TENANT_KEY_LEN};
use rvf_types::{SegmentFlags, SegmentType};

/// Manifest format version written by this build.
pub const FORMAT_VERSION: u16 = 1;
/// Row schema versions this build can restore.
pub const KNOWN_SCHEMA_VERSIONS: &[u32] = &[1];
/// Row schema version written by this build.
pub const SCHEMA_VERSION: u32 = 1;
/// Upper bound on a manifest object (checked before parsing).
pub const MAX_MANIFEST_BYTES: usize = 4 << 20;
/// Upper bound on chunks per snapshot.
pub const MAX_CHUNKS: usize = 65_536;
/// Upper bound on a signing key id.
pub const MAX_KEY_ID_LEN: usize = 128;
/// Upper bound on rows per chunk (writer default and restore refusal). A
/// decoded [`crate::Row`] costs ~72 bytes of struct plus 2–3 heap blocks on
/// top of its encoded size, so at small dimensions an 8 MiB chunk of
/// ~500k tiny rows would decode to 50+ MB; this cap keeps one decoded chunk
/// within ~8 MiB + 32k × ~120 B ≈ 12 MB whatever the dimension.
pub const MAX_ROWS_PER_CHUNK: u64 = 32_768;

const KEY_ID_TAG: &[u8] = b"rvedge-snapshot-key-v1\0";

const BODY_MAGIC: &[u8; 8] = b"RVSNAPM1";
const ROOT_TAG: &[u8] = b"rvedge-snapshot-root-v1\0";
const SIG_TAG: &[u8] = b"rvedge-snapshot-sig-v1\0";

/// One stored chunk object (`seg-{n}.rvf`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRef {
    /// Exact byte length.
    pub size: u64,
    /// Rows encoded in the chunk.
    pub rows: u64,
    /// sha256 of the chunk bytes.
    pub sha256: [u8; 32],
}

/// The manifest body (everything covered by the root).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Manifest format version.
    pub format_ver: u16,
    /// Row schema version.
    pub schema_ver: u32,
    /// Owning tenant.
    pub tenant_key: String,
    /// Service slug.
    pub service: String,
    /// Collection uid (hex).
    pub collection_uid: String,
    /// Shard.
    pub shard: u16,
    /// Snapshot epoch (the shard's `snapshot_seq`).
    pub epoch: u64,
    /// Vector dimension.
    pub dim: u16,
    /// Distance metric.
    pub metric: Metric,
    /// Total rows.
    pub row_count: u64,
    /// Creation time from the [`crate::Clock`] port.
    pub created_at_ms: u64,
    /// Head of the shard's audit hash chain at snapshot time (ADR-351 §6.3).
    pub audit_head: [u8; 32],
    /// Chunks in order.
    pub chunks: Vec<ChunkRef>,
}

/// Optional signature over the root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureBlock {
    /// Key identifier (resolved by the verifier).
    pub key_id: String,
    /// Ed25519 signature over [`signing_message`].
    pub signature: [u8; 64],
}

/// A manifest with its root and optional signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedManifest {
    /// Body.
    pub manifest: Manifest,
    /// Root as stored.
    pub root: [u8; 32],
    /// Optional signature.
    pub signature: Option<SignatureBlock>,
}

/// The exact bytes a signer signs for `root`.
pub fn signing_message(root: &[u8; 32]) -> Vec<u8> {
    let mut m = Vec::with_capacity(SIG_TAG.len() + 32);
    m.extend_from_slice(SIG_TAG);
    m.extend_from_slice(root);
    m
}

impl Manifest {
    /// Canonical body encoding.
    pub fn encode_body(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(160 + self.chunks.len() * 48);
        out.extend_from_slice(BODY_MAGIC);
        out.extend_from_slice(&self.format_ver.to_le_bytes());
        out.extend_from_slice(&self.schema_ver.to_le_bytes());
        put_str16(&mut out, &self.tenant_key);
        put_str16(&mut out, &self.service);
        put_str16(&mut out, &self.collection_uid);
        out.extend_from_slice(&self.shard.to_le_bytes());
        out.extend_from_slice(&self.epoch.to_le_bytes());
        out.extend_from_slice(&self.dim.to_le_bytes());
        out.push(self.metric.code());
        out.extend_from_slice(&self.row_count.to_le_bytes());
        out.extend_from_slice(&self.created_at_ms.to_le_bytes());
        out.extend_from_slice(&self.audit_head);
        out.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());
        for c in &self.chunks {
            out.extend_from_slice(&c.size.to_le_bytes());
            out.extend_from_slice(&c.rows.to_le_bytes());
            out.extend_from_slice(&c.sha256);
        }
        out
    }

    /// Recompute the root over the canonical body.
    pub fn compute_root(&self) -> [u8; 32] {
        sha256(&[ROOT_TAG, &self.encode_body()])
    }

    /// The R2 key prefix of this snapshot, see [`key_prefix`].
    pub fn key_prefix(&self) -> String {
        key_prefix(
            &self.tenant_key,
            &self.service,
            &self.collection_uid,
            self.shard,
            self.epoch,
            &self.audit_head,
        )
    }

    /// Sum of chunk sizes.
    pub fn total_bytes(&self) -> u64 {
        self.chunks
            .iter()
            .fold(0u64, |a, c| a.saturating_add(c.size))
    }

    fn decode_body(r: &mut Reader<'_>) -> Result<Manifest, SnapshotError> {
        let bad = SnapshotError::Malformed;
        if r.take(8).ok_or(bad("magic"))? != BODY_MAGIC {
            return Err(bad("magic"));
        }
        let format_ver = r.u16().ok_or(bad("format_ver"))?;
        if format_ver != FORMAT_VERSION {
            return Err(SnapshotError::UnknownFormat(format_ver));
        }
        let schema_ver = r.u32().ok_or(bad("schema_ver"))?;
        let tenant_key = r
            .str16(TENANT_KEY_LEN)
            .ok_or(bad("tenant_key"))?
            .to_string();
        let service = r.str16(MAX_SERVICE_LEN).ok_or(bad("service"))?.to_string();
        let collection_uid = r.str16(64).ok_or(bad("collection_uid"))?.to_string();
        let shard = r.u16().ok_or(bad("shard"))?;
        let epoch = r.u64().ok_or(bad("epoch"))?;
        let dim = r.u16().ok_or(bad("dim"))?;
        let metric = Metric::from_code(r.u8().ok_or(bad("metric"))?).ok_or(bad("metric"))?;
        let row_count = r.u64().ok_or(bad("row_count"))?;
        let created_at_ms = r.u64().ok_or(bad("created_at"))?;
        let audit_head = r.arr32().ok_or(bad("audit_head"))?;
        let n = r.u32().ok_or(bad("chunk_count"))? as usize;
        if n > MAX_CHUNKS || n.saturating_mul(48) > r.remaining() {
            return Err(bad("chunk_count"));
        }
        let mut chunks = Vec::with_capacity(n);
        for _ in 0..n {
            chunks.push(ChunkRef {
                size: r.u64().ok_or(bad("chunk"))?,
                rows: r.u64().ok_or(bad("chunk"))?,
                sha256: r.arr32().ok_or(bad("chunk"))?,
            });
        }
        if dim == 0 || dim > MAX_DIM {
            return Err(bad("dim"));
        }
        // Identifiers are re-validated so a crafted manifest cannot smuggle
        // path components into derived R2 keys.
        validate_tenant_key(&tenant_key).map_err(|_| bad("tenant_key"))?;
        validate_service(&service).map_err(|_| bad("service"))?;
        validate_collection_uid(&collection_uid).map_err(|_| bad("collection_uid"))?;
        Ok(Manifest {
            format_ver,
            schema_ver,
            tenant_key,
            service,
            collection_uid,
            shard,
            epoch,
            dim,
            metric,
            row_count,
            created_at_ms,
            audit_head,
            chunks,
        })
    }
}

impl SealedManifest {
    /// Encode as the `manifest.rvf` object (one rvf-wire segment).
    pub fn to_object(&self) -> Vec<u8> {
        let mut payload = self.manifest.encode_body();
        payload.extend_from_slice(&self.root);
        match &self.signature {
            None => payload.push(0),
            Some(s) => {
                payload.push(1);
                put_str16(&mut payload, &s.key_id);
                payload.extend_from_slice(&s.signature);
            }
        }
        rvf_wire::write_segment(
            SegmentType::Manifest as u8,
            &payload,
            SegmentFlags::empty(),
            0,
        )
    }

    /// Strictly parse a `manifest.rvf` object. Does **not** check the root;
    /// use [`SealedManifest::root_matches`] (restore does it in order).
    pub fn from_object(obj: &[u8]) -> Result<SealedManifest, SnapshotError> {
        let bad = SnapshotError::Malformed;
        if obj.len() > MAX_MANIFEST_BYTES {
            return Err(SnapshotError::QuotaExceeded {
                what: "manifest bytes",
                limit: MAX_MANIFEST_BYTES as u64,
                requested: obj.len() as u64,
            });
        }
        // Bound the declared length before slicing: rvf_wire::read_segment adds
        // it to 64 unchecked, which a crafted length would overflow.
        let hdr = rvf_wire::read_segment_header(obj).map_err(|_| bad("segment header"))?;
        if hdr.payload_length > (obj.len() - 64) as u64 {
            return Err(bad("truncated"));
        }
        let payload = &obj[64..64 + hdr.payload_length as usize];
        if hdr.seg_type != SegmentType::Manifest as u8 {
            return Err(bad("segment type"));
        }
        rvf_wire::validate_segment(&hdr, payload).map_err(|_| SnapshotError::ManifestChecksum)?;
        let padded = rvf_wire::calculate_padded_size(64, payload.len());
        if obj.len() != padded || obj[64 + payload.len()..].iter().any(|&b| b != 0) {
            return Err(bad("trailing bytes"));
        }
        let mut r = Reader::new(payload);
        let manifest = Manifest::decode_body(&mut r)?;
        let root = r.arr32().ok_or(bad("root"))?;
        let signature = match r.u8().ok_or(bad("sig flag"))? {
            0 => None,
            1 => Some(SignatureBlock {
                key_id: r.str16(MAX_KEY_ID_LEN).ok_or(bad("key_id"))?.to_string(),
                signature: r
                    .take(64)
                    .and_then(|s| s.try_into().ok())
                    .ok_or(bad("signature"))?,
            }),
            _ => return Err(bad("sig flag")),
        };
        if r.remaining() != 0 {
            return Err(bad("trailing payload"));
        }
        let sealed = SealedManifest {
            manifest,
            root,
            signature,
        };
        // Header fields outside the content hash (flags, segment id,
        // timestamp, …) must be canonical too: one manifest, one encoding.
        if sealed.to_object() != obj {
            return Err(bad("non-canonical encoding"));
        }
        Ok(sealed)
    }

    /// `true` when the stored root equals the recomputed root.
    pub fn root_matches(&self) -> bool {
        let computed = self.manifest.compute_root();
        // Constant-time compare is not needed (roots are public), but be strict.
        computed == self.root
    }
}

/// R2 key prefix (ADR-351 §6.3):
/// `snapshots/{tenant_key}/{service}/{collection_uid}/{shard}/{epoch:020}-{id12}`.
///
/// `id12` is the first 12 hex chars of
/// `sha256("rvedge-snapshot-key-v1\0" ‖ tenant ‖ service ‖ uid ‖ shard ‖ epoch ‖ audit_head)`.
/// Every input is known when the writer starts (and is recorded in the
/// manifest), so each chunk can be PUT to its final key the moment it is
/// produced; the manifest root — which only exists after the last chunk —
/// is deliberately not part of the key. Identifiers must already be
/// validated (the writer's [`crate::ShardRef`] and the manifest parser do),
/// so no component can contain `/` or `..`.
pub fn key_prefix(
    tenant_key: &str,
    service: &str,
    collection_uid: &str,
    shard: u16,
    epoch: u64,
    audit_head: &[u8; 32],
) -> String {
    let mut buf = Vec::with_capacity(128);
    put_str16(&mut buf, tenant_key);
    put_str16(&mut buf, service);
    put_str16(&mut buf, collection_uid);
    buf.extend_from_slice(&shard.to_le_bytes());
    buf.extend_from_slice(&epoch.to_le_bytes());
    buf.extend_from_slice(audit_head);
    let id = sha256(&[KEY_ID_TAG, &buf]);
    format!(
        "snapshots/{tenant_key}/{service}/{collection_uid}/{shard}/{epoch:020}-{}",
        hex::encode(&id[..6])
    )
}
