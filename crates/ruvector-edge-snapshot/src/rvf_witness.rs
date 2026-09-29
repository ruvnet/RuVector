//! Export witness (ADR-351 §7 `:export` "streamed `.rvf` with witness
//! manifest", §17): a `PROFILE_SEG` with magic `RVEW`, written by
//! [`crate::RvfExporter::finish_witnessed`] directly before the runtime
//! manifest (and listed in its directory), so the file stays readable by
//! `rvf inspect` / `rvf query` and by [`crate::RvfImporter`], which both skip
//! unknown `PROFILE` payloads.
//!
//! ```text
//! "RVEW" | ver u8 = 1 | tenant_key str16 | collection_uid str16 |
//! audit_head [32] | dim u16 | metric u8 | rows u64 | manifest_sha256 [32] |
//! n u32 | n × sha256(segment header ‖ payload) | root [32] |
//! sig u8 (0 | 1 key_id str16 sig [64])
//! ```
//!
//! `root = sha256("rvedge-export-root-v1\0" ‖ everything before root)`; it
//! covers the identity, the audit-chain head, every preceding segment
//! (sidecars and vectors, byte for byte) and the payload of the runtime
//! manifest that follows. The optional Ed25519 signature is over
//! `"rvedge-export-sig-v1\0" ‖ root`. [`verify_export`] checks all of it
//! offline against the expected tenant.

use crate::bytes::{put_str16, Reader};
use crate::error::SnapshotError;
use crate::manifest::MAX_KEY_ID_LEN;
use crate::restore::SignaturePolicy;
use crate::rvf_format::{self as fmt, SEG_MANIFEST, SEG_PROFILE};
use crate::types::{sha256, validate_collection_uid, validate_tenant_key, Metric};
use crate::types::{COLLECTION_UID_HEX_LEN, TENANT_KEY_LEN};

const MAGIC: &[u8; 4] = b"RVEW";
const VERSION: u8 = 1;
const ROOT_TAG: &[u8] = b"rvedge-export-root-v1\0";
const SIG_TAG: &[u8] = b"rvedge-export-sig-v1\0";

/// Who the export belongs to (server-derived).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportIdentity {
    /// Owning tenant.
    pub tenant_key: String,
    /// Exported collection.
    pub collection_uid: String,
    /// The shard's audit-chain head at export time.
    pub audit_head: [u8; 32],
}

impl ExportIdentity {
    /// Validate at the boundary.
    pub fn new(
        tenant_key: &str,
        collection_uid: &str,
        audit_head: [u8; 32],
    ) -> Result<Self, SnapshotError> {
        validate_tenant_key(tenant_key)?;
        validate_collection_uid(collection_uid)?;
        Ok(ExportIdentity {
            tenant_key: tenant_key.to_string(),
            collection_uid: collection_uid.to_string(),
            audit_head,
        })
    }
}

/// A verified export witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportWitness {
    /// Identity.
    pub identity: ExportIdentity,
    /// Dimension.
    pub dim: u16,
    /// Metric.
    pub metric: Metric,
    /// Rows exported.
    pub rows: u64,
    /// sha256 of the runtime manifest payload that follows the witness.
    pub manifest_sha256: [u8; 32],
    /// sha256 of every preceding segment, in file order.
    pub segment_sha256: Vec<[u8; 32]>,
    /// Root.
    pub root: [u8; 32],
    /// `(key_id, signature)`, if signed.
    pub signature: Option<(String, [u8; 64])>,
}

/// The bytes a signer signs for an export `root`.
pub fn export_signing_message(root: &[u8; 32]) -> Vec<u8> {
    [SIG_TAG, root.as_slice()].concat()
}

/// Encoded payload length for `n` segments and a key id of `key_len` bytes.
pub(crate) fn payload_len(id: &ExportIdentity, n: usize, key_len: Option<usize>) -> usize {
    let body = 4 + 1 + 2 + id.tenant_key.len() + 2 + id.collection_uid.len() + 32 + 2 + 1 + 8;
    body + 32 + 4 + 32 * n + 32 + key_len.map_or(1, |k| 1 + 2 + k + 64)
}

/// Encode the body (everything the root covers).
pub(crate) fn encode_body(w: &ExportWitness) -> Vec<u8> {
    let id = &w.identity;
    let mut o = Vec::with_capacity(payload_len(
        id,
        w.segment_sha256.len(),
        Some(MAX_KEY_ID_LEN),
    ));
    o.extend_from_slice(MAGIC);
    o.push(VERSION);
    put_str16(&mut o, &id.tenant_key);
    put_str16(&mut o, &id.collection_uid);
    o.extend_from_slice(&id.audit_head);
    o.extend_from_slice(&w.dim.to_le_bytes());
    o.push(w.metric.code());
    o.extend_from_slice(&w.rows.to_le_bytes());
    o.extend_from_slice(&w.manifest_sha256);
    o.extend_from_slice(&(w.segment_sha256.len() as u32).to_le_bytes());
    for h in &w.segment_sha256 {
        o.extend_from_slice(h);
    }
    o
}

/// Root over an encoded body.
pub(crate) fn root_of(body: &[u8]) -> [u8; 32] {
    sha256(&[ROOT_TAG, body])
}

/// Full payload: body ‖ root ‖ signature block.
pub(crate) fn encode_payload(w: &ExportWitness) -> Vec<u8> {
    let mut p = encode_body(w);
    p.extend_from_slice(&w.root);
    match &w.signature {
        None => p.push(0),
        Some((k, s)) => {
            p.push(1);
            put_str16(&mut p, k);
            p.extend_from_slice(s);
        }
    }
    p
}

fn decode(payload: &[u8]) -> Result<(ExportWitness, usize), SnapshotError> {
    let bad = SnapshotError::Malformed;
    let mut r = Reader::new(payload);
    if r.take(4) != Some(MAGIC.as_slice()) || r.u8() != Some(VERSION) {
        return Err(bad("export witness header"));
    }
    let tenant = r.str16(TENANT_KEY_LEN).ok_or(bad("witness tenant"))?;
    let uid = r
        .str16(COLLECTION_UID_HEX_LEN)
        .ok_or(bad("witness collection"))?;
    let audit_head = r.arr32().ok_or(bad("witness audit head"))?;
    let identity = ExportIdentity::new(tenant, uid, audit_head).map_err(|_| bad("witness id"))?;
    let dim = r.u16().ok_or(bad("witness dim"))?;
    let metric = Metric::from_code(r.u8().ok_or(bad("witness metric"))?).ok_or(bad("metric"))?;
    let rows = r.u64().ok_or(bad("witness rows"))?;
    let manifest_sha256 = r.arr32().ok_or(bad("witness manifest"))?;
    let n = r.u32().ok_or(bad("witness count"))? as usize;
    if n.saturating_mul(32) > r.remaining() {
        return Err(bad("witness count"));
    }
    let segment_sha256 = (0..n).map(|_| r.arr32().unwrap_or([0; 32])).collect();
    let body_len = r.pos();
    let root = r.arr32().ok_or(bad("witness root"))?;
    let signature = match r.u8().ok_or(bad("witness sig"))? {
        0 => None,
        1 => {
            let k = r.str16(MAX_KEY_ID_LEN).ok_or(bad("witness key id"))?;
            let s = r
                .take(64)
                .and_then(|s| s.try_into().ok())
                .ok_or(bad("sig"))?;
            Some((k.to_string(), s))
        }
        _ => return Err(bad("witness sig")),
    };
    if r.remaining() != 0 {
        return Err(bad("witness trailing bytes"));
    }
    let w = ExportWitness {
        identity,
        dim,
        metric,
        rows,
        manifest_sha256,
        segment_sha256,
        root,
        signature,
    };
    Ok((w, body_len))
}

/// Verify a complete exported file offline: every segment before the
/// witness hashes to the listed digest, the witness is followed by exactly
/// the runtime manifest it names (which ends the file), the root recomputes,
/// the tenant is `tenant_key`, and the signature satisfies `sig`.
pub fn verify_export(
    file: &[u8],
    tenant_key: &str,
    sig: SignaturePolicy<'_>,
) -> Result<ExportWitness, SnapshotError> {
    let bad = SnapshotError::Malformed;
    let mut pos = 0usize;
    let mut hashes = Vec::new();
    let witness = loop {
        let rest = file.get(pos..).filter(|r| r.len() >= fmt::HEADER);
        let h = fmt::parse_header(rest.ok_or(bad("no export witness"))?).map_err(bad)?;
        let len = (h.payload_length as usize).saturating_add(h.alignment_pad as usize);
        let end = pos
            .checked_add(fmt::HEADER + len)
            .filter(|&e| e <= file.len() && h.payload_length <= (file.len() - pos) as u64)
            .ok_or(bad("segment length"))?;
        let seg = &file[pos..end];
        let payload = &seg[fmt::HEADER..fmt::HEADER + h.payload_length as usize];
        if h.seg_type == SEG_PROFILE && payload.starts_with(MAGIC) {
            break (decode(payload)?, pos + fmt::HEADER, end);
        }
        hashes.push(sha256(&[seg]));
        pos = end;
    };
    let ((w, body_len), body_at, after) = witness;
    let mrest = &file[after..];
    let mh = fmt::parse_header(mrest).map_err(bad)?;
    let mlen = mh.payload_length;
    if mh.seg_type != SEG_MANIFEST || mlen.saturating_add(fmt::HEADER as u64) != mrest.len() as u64
    {
        return Err(bad("witness not followed by the final manifest"));
    }
    if sha256(&[&mrest[fmt::HEADER..]]) != w.manifest_sha256 {
        return Err(SnapshotError::ManifestChecksum);
    }
    if let Some(i) = (0..hashes.len().max(w.segment_sha256.len()))
        .find(|&i| hashes.get(i) != w.segment_sha256.get(i))
    {
        return Err(SnapshotError::ChunkChecksum { index: i as u32 });
    }
    if root_of(&file[body_at..body_at + body_len]) != w.root {
        return Err(SnapshotError::ManifestChecksum);
    }
    if w.identity.tenant_key != tenant_key {
        return Err(SnapshotError::TenantMismatch);
    }
    if let SignaturePolicy::Required(v) = sig {
        let (k, s) = w
            .signature
            .as_ref()
            .ok_or(SnapshotError::SignatureMissing)?;
        if !v.verify(tenant_key, k, &export_signing_message(&w.root), s) {
            return Err(SnapshotError::SignatureInvalid);
        }
    }
    Ok(w)
}
