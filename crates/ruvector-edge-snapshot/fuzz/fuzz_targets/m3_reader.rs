//! M3 readers (ADR-351 §6.3, §15 M3): every byte a restore or an import
//! reads from R2 is attacker-influenced (uploads, tampered snapshots).
//!
//! Input: `mode ‖ rest`; `mode % 5` picks the reader:
//!
//! * 0 — `SealedManifest::from_object` on raw bytes;
//! * 1 — restore of one raw chunk: `rest = dim u16 ‖ rows u16 ‖ chunk`; a
//!   manifest naming exactly that chunk (size, sha256, rows) is sealed and
//!   witnessed, so `RestoreSession::accept_chunk` parses the rvf-wire
//!   segments and row blocks of arbitrary bytes;
//! * 2 — as 1, but the chunk is built from `u16`-length-prefixed payloads
//!   each wrapped in a valid rvf-wire `VEC_SEG` (ids 0, 1, …), so the
//!   row-block decoder sees arbitrary payloads past the content hash;
//! * 3 — RVF import of a raw file: `inspect_tail` + `RvfImporter` fed in
//!   pieces;
//! * 4 — RVF import of a built file: `dim u16 ‖ metric u8 ‖ piece u8 ‖`
//!   segments (`type u8 ‖ len u16 ‖ payload`) written with valid legacy
//!   content hashes, then a runtime manifest listing them at their true
//!   offsets, so the importer's record parser sees arbitrary payloads.
//!
//! Invariants: no panic; a parsed manifest re-encodes to the same object;
//! restored rows match the declared count, have `dim` values and strictly
//! increasing ids; import batches respect the batch size and dimension,
//! and a finished import consumed exactly the file.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ruvector_edge_snapshot::manifest::{FORMAT_VERSION, SCHEMA_VERSION};
use ruvector_edge_snapshot::rvf_format::{
    legacy_content_hash, DirEntry, RvfManifest, HEADER, SEG_MANIFEST, SEG_PROFILE, SEG_VEC,
};
use ruvector_edge_snapshot::{
    inspect_tail, ChainProof, ChunkRef, ImportCursor, ImportLimits, ImportSpec, Manifest, Metric,
    RestoreQuota, RestoreSession, RestoreTarget, RvfImporter, SealedManifest, ShardRef,
    SignaturePolicy, WitnessChain, MAX_ROWS_PER_CHUNK,
};
use rvf_types::{SegmentFlags, SegmentType, SEGMENT_MAGIC, SEGMENT_VERSION};

const TENANT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaa";
const UID: &str = "0123456789abcdef0123456789abcdef";
const MAX_BATCH: usize = 64;

fn sha256(b: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    Sha256::digest(b).into()
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    b.get(at..at + 2)
        .map_or(0, |s| u16::from_le_bytes([s[0], s[1]]))
}

/// `u16`-length-prefixed parts (lengths clamped to the input).
fn parts(mut r: &[u8], max: usize) -> Vec<&[u8]> {
    let mut out = Vec::new();
    while r.len() >= 2 && out.len() < max {
        let len = usize::from(u16_at(r, 0)).min(r.len() - 2);
        out.push(&r[2..2 + len]);
        r = &r[2 + len..];
    }
    out
}

fn manifest(obj: &[u8]) {
    if let Ok(m) = SealedManifest::from_object(obj) {
        assert_eq!(m.to_object(), obj, "canonical encoding");
        let _ = m.root_matches();
        let _ = m.manifest.key_prefix();
        let _ = m.manifest.total_bytes();
    }
}

fn restore(dim: u16, rows: u64, chunk: &[u8]) {
    let dim = dim % 64 + 1;
    let rows = rows.min(MAX_ROWS_PER_CHUNK);
    let m = Manifest {
        format_ver: FORMAT_VERSION,
        schema_ver: SCHEMA_VERSION,
        tenant_key: TENANT.into(),
        service: "vector".into(),
        collection_uid: UID.into(),
        shard: 0,
        epoch: 1,
        dim,
        metric: Metric::Cosine,
        row_count: rows,
        created_at_ms: 1_790_000_000_000,
        audit_head: [9; 32],
        chunks: vec![ChunkRef {
            size: chunk.len() as u64,
            rows,
            sha256: sha256(chunk),
        }],
    };
    let sealed = SealedManifest {
        root: m.compute_root(),
        manifest: m,
        signature: None,
    };
    let mut chain = WitnessChain::new(TENANT).unwrap();
    let entry = chain.append(&sealed).unwrap();
    let entries = [entry];
    let proof = ChainProof::from_genesis(TENANT, &entries, chain.head());
    let target = RestoreTarget {
        shard: ShardRef::new(TENANT, "vector", UID, 0).unwrap(),
        dim: Some(dim),
        metric: Some(Metric::Cosine),
    };
    let quota = RestoreQuota {
        max_rows: 1 << 20,
        max_bytes: 1 << 30,
    };
    let mut s = RestoreSession::begin(
        &sealed.to_object(),
        &target,
        quota,
        proof,
        SignaturePolicy::NotRequired,
    )
    .expect("a freshly sealed, witnessed manifest is accepted");
    let mut out = Vec::new();
    let r = s.accept_chunk(0, chunk, &mut out);
    if r.is_err() {
        assert!(out.is_empty(), "a refused chunk appends nothing");
        assert!(s.finish().is_err());
        return;
    }
    assert_eq!(out.len() as u64, rows);
    for w in out.windows(2) {
        assert!(w[0].id < w[1].id, "ids strictly increasing");
    }
    for row in &out {
        assert_eq!(row.values.len(), usize::from(dim));
    }
    let summary = s.finish().expect("all chunks accepted");
    assert_eq!(summary.rows, rows);
}

fn vec_chunk(payloads: &[&[u8]]) -> Vec<u8> {
    let mut chunk = Vec::new();
    for (i, p) in payloads.iter().enumerate() {
        chunk.extend(rvf_wire::write_segment(
            SegmentType::Vec as u8,
            p,
            SegmentFlags::empty(),
            i as u64,
        ));
    }
    chunk
}

fn import(file: &[u8], piece: usize) {
    let limits = ImportLimits {
        max_batch_rows: MAX_BATCH,
        ..ImportLimits::default()
    };
    let Ok(summary) = inspect_tail(file, file.len() as u64, &limits) else {
        return;
    };
    assert!(summary.manifest_offset < summary.file_len);
    let spec = ImportSpec {
        dim: summary.dim,
        metric: summary.metric,
        limits,
    };
    let Ok(mut imp) = RvfImporter::new(spec, summary, ImportCursor::default()) else {
        return;
    };
    let mut rows = 0u64;
    let mut seq = 0u64;
    for p in file.chunks(piece.max(1)) {
        if imp.feed(p).is_err() {
            return;
        }
        loop {
            match imp.next_batch() {
                Ok(Some(b)) => {
                    assert_eq!(b.seq, seq);
                    seq += 1;
                    assert!(!b.rows.is_empty() && b.rows.len() <= MAX_BATCH);
                    for r in &b.rows {
                        assert_eq!(r.values.len(), usize::from(spec.dim));
                    }
                    rows += b.rows.len() as u64;
                    assert_eq!(b.cursor_after.rows_done, rows);
                    assert_eq!(b.cursor_after.batch_seq, seq);
                }
                Ok(None) => break,
                Err(_) => return,
            }
        }
    }
    if let Ok(t) = imp.finish() {
        assert_eq!(t.rows, rows);
        assert_eq!(t.batches, seq);
        assert_eq!(t.bytes, file.len() as u64);
        assert_eq!(t.sha256, Some(sha256(file)));
    }
}

fn segment(out: &mut Vec<u8>, seg_type: u8, seg_id: u64, payload: &[u8]) {
    let mut h = [0u8; HEADER];
    h[0..4].copy_from_slice(&SEGMENT_MAGIC.to_le_bytes());
    h[4] = SEGMENT_VERSION;
    h[5] = seg_type;
    h[8..16].copy_from_slice(&seg_id.to_le_bytes());
    h[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    h[0x28..0x38].copy_from_slice(&legacy_content_hash(payload));
    out.extend_from_slice(&h);
    out.extend_from_slice(payload);
}

fn built_file(rest: &[u8]) -> Option<(Vec<u8>, usize)> {
    let (head, body) = (rest.get(..4)?, &rest[4..]);
    let dim = u16_at(head, 0) % 64 + 1;
    let metric_id = head[2] % 4;
    let piece = usize::from(head[3]) * 16 + 1;
    let mut file = Vec::new();
    let mut dir = Vec::new();
    let mut r = body;
    while r.len() >= 3 && dir.len() < 32 {
        let seg_type = match r[0] % 4 {
            0 | 1 => SEG_VEC,
            2 => SEG_PROFILE,
            _ => r[0],
        };
        let len = usize::from(u16_at(r, 1)).min(r.len() - 3);
        let payload = &r[3..3 + len];
        r = &r[3 + len..];
        let seg_id = dir.len() as u64;
        dir.push(DirEntry {
            seg_id,
            offset: file.len() as u64,
            payload_len: payload.len() as u64,
            seg_type,
        });
        segment(&mut file, seg_type, seg_id, payload);
    }
    let deleted = r
        .chunks_exact(8)
        .take(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()));
    let m = RvfManifest {
        epoch: 1,
        dim,
        total_vectors: 0,
        profile: 0,
        metric_id,
        dir,
        deleted: deleted.collect(),
    };
    let seg_id = m.dir.len() as u64;
    segment(&mut file, SEG_MANIFEST, seg_id, &m.encode());
    Some((file, piece))
}

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else {
        return;
    };
    match mode % 5 {
        0 => manifest(rest),
        1 if rest.len() >= 4 => restore(u16_at(rest, 0), u64::from(u16_at(rest, 2)), &rest[4..]),
        2 if rest.len() >= 4 => {
            let payloads = parts(&rest[4..], 16);
            restore(
                u16_at(rest, 0),
                u64::from(u16_at(rest, 2)),
                &vec_chunk(&payloads),
            );
        }
        3 => import(rest, usize::from(mode / 5) * 64 + 1),
        4 => {
            if let Some((file, piece)) = built_file(rest) {
                import(&file, piece);
            }
        }
        _ => {}
    }
});
