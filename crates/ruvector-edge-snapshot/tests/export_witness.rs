//! `:export` witness: offline verification, tamper / tenant / signature
//! refusals, and the witnessed file still imports.

mod common;
use common::*;
use ruvector_edge_snapshot::rvf_format::legacy_content_hash;
use ruvector_edge_snapshot::*;

fn identity(tenant: &str) -> ExportIdentity {
    ExportIdentity::new(tenant, UID, [4; 32]).unwrap()
}

fn export(tenant: &str, data: &[Row], dim: u16, signer: Option<&dyn Signer>) -> Vec<u8> {
    let mut file = Vec::new();
    let mut sink = |b: &[u8]| file.extend_from_slice(b);
    let mut x = RvfExporter::new(dim, Metric::Cosine, 16).unwrap();
    for r in data {
        x.push(r, &mut sink).unwrap();
    }
    let s = x
        .finish_witnessed(&mut sink, &identity(tenant), signer)
        .unwrap();
    assert_eq!(s.bytes, file.len() as u64);
    file
}

/// Flip one payload byte of the segment at `seg_offset` and re-seal its
/// legacy CRC, as an attacker who knows the format would.
fn tamper(file: &[u8], seg_offset: usize, at: usize) -> Vec<u8> {
    let mut f = file.to_vec();
    let len = u64::from_le_bytes(f[seg_offset + 16..seg_offset + 24].try_into().unwrap()) as usize;
    f[seg_offset + 64 + at] ^= 0x40;
    let h = legacy_content_hash(&f[seg_offset + 64..seg_offset + 64 + len]);
    f[seg_offset + 0x28..seg_offset + 0x38].copy_from_slice(&h);
    f
}

fn seg_offsets(file: &[u8]) -> Vec<(usize, u8)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < file.len() {
        let len = u64::from_le_bytes(file[pos + 16..pos + 24].try_into().unwrap()) as usize;
        out.push((pos, file[pos + 5]));
        pos += 64 + len;
    }
    out
}

#[test]
fn witnessed_export_verifies_offline_and_still_imports() {
    let dim = 6;
    let data = rows(100, dim, 31);
    let signer = TestSigner::fixed(21);
    let file = export(TENANT_A, &data, dim, Some(&signer));

    let w = verify_export(&file, TENANT_A, SignaturePolicy::NotRequired).unwrap();
    assert_eq!((w.dim, w.metric, w.rows), (dim, Metric::Cosine, 100));
    assert_eq!(w.identity, identity(TENANT_A));
    // 7 (sidecar, vec) pairs precede the witness.
    assert_eq!(w.segment_sha256.len(), 14);
    let mut v = Ed25519Verifier::new();
    v.add_tenant_key(TENANT_A, "snap-k1", signer.public())
        .unwrap();
    assert!(verify_export(&file, TENANT_A, SignaturePolicy::Required(&v)).is_ok());

    // Layout: pairs, then the witness (PROFILE), then the manifest last.
    let segs = seg_offsets(&file);
    let types: Vec<u8> = segs.iter().rev().take(2).map(|s| s.1).collect();
    assert_eq!(types, [0x05, 0x0B]);

    // The importer skips the witness; the summary lists it in the directory.
    let summary = inspect_tail(&file, file.len() as u64, &ImportLimits::default()).unwrap();
    let spec = ImportSpec {
        dim,
        metric: Metric::Cosine,
        limits: ImportLimits::default(),
    };
    let mut imp = RvfImporter::new(spec, summary, ImportCursor::default()).unwrap();
    let mut back = Vec::new();
    for p in file.chunks(777) {
        imp.feed(p).unwrap();
        while let Some(b) = imp.next_batch().unwrap() {
            back.extend(b.rows);
        }
    }
    imp.finish().unwrap();
    assert_rows_identical(&data, &back);
}

#[test]
fn tampered_foreign_or_unsigned_exports_are_refused() {
    let dim = 4;
    let signer = TestSigner::fixed(22);
    let file = export(TENANT_B, &rows(40, dim, 5), dim, Some(&signer));
    let segs = seg_offsets(&file);

    // A vector value changed (CRC re-sealed so the runtime would accept it).
    let vec_seg = segs.iter().position(|s| s.1 == 0x01).unwrap();
    let bad = tamper(&file, segs[vec_seg].0, 20);
    assert_eq!(
        verify_export(&bad, TENANT_B, SignaturePolicy::NotRequired).unwrap_err(),
        SnapshotError::ChunkChecksum {
            index: vec_seg as u32
        }
    );
    // An id in a sidecar.
    let bad = tamper(&file, segs[0].0, 12);
    assert_eq!(
        verify_export(&bad, TENANT_B, SignaturePolicy::NotRequired).unwrap_err(),
        SnapshotError::ChunkChecksum { index: 0 }
    );
    // The runtime manifest after the witness (e.g. a metric flip).
    let last = segs.last().unwrap().0;
    let bad = tamper(&file, last, 19);
    assert_eq!(
        verify_export(&bad, TENANT_B, SignaturePolicy::NotRequired).unwrap_err(),
        SnapshotError::ManifestChecksum
    );
    // A witness field (the row count) — the root no longer matches.
    let wit = segs[segs.len() - 2].0;
    let rows_at = 4 + 1 + 2 + 26 + 2 + 32 + 32 + 2 + 1;
    let bad = tamper(&file, wit, rows_at);
    assert_eq!(
        verify_export(&bad, TENANT_B, SignaturePolicy::NotRequired).unwrap_err(),
        SnapshotError::ManifestChecksum
    );
    // Presented as another tenant's export.
    assert_eq!(
        verify_export(&file, TENANT_A, SignaturePolicy::NotRequired).unwrap_err(),
        SnapshotError::TenantMismatch
    );
    // Tenant A's key does not vouch for tenant B's export.
    let mut a_only = Ed25519Verifier::new();
    a_only
        .add_tenant_key(TENANT_A, "snap-k1", signer.public())
        .unwrap();
    assert_eq!(
        verify_export(&file, TENANT_B, SignaturePolicy::Required(&a_only)).unwrap_err(),
        SnapshotError::SignatureInvalid
    );
    // Unsigned export under a signature requirement.
    let unsigned = export(TENANT_B, &rows(40, dim, 5), dim, None);
    let mut v = Ed25519Verifier::new();
    v.add_key("snap-k1", signer.public()).unwrap();
    assert_eq!(
        verify_export(&unsigned, TENANT_B, SignaturePolicy::Required(&v)).unwrap_err(),
        SnapshotError::SignatureMissing
    );
    // A plain (unwitnessed) export has no witness at all.
    let mut plain = Vec::new();
    let mut sink = |b: &[u8]| plain.extend_from_slice(b);
    let mut x = RvfExporter::new(dim, Metric::Cosine, 16).unwrap();
    x.push(&rows(1, dim, 1)[0], &mut sink).unwrap();
    x.finish(&mut sink);
    assert!(matches!(
        verify_export(&plain, TENANT_B, SignaturePolicy::NotRequired),
        Err(SnapshotError::Malformed(_))
    ));
}
