//! #1289: `derive` records the derivation type it is given (lineage record
//! segment + witness entry), and `merge` records several parents.
//! #1290: typed application segments (append / read / list).
//!
//! Round trips against the published 0.5.3 runtime live in the standalone
//! `crates/rvf/tests/format-compat` crate (its own workspace and lockfile).

use rvf_crypto::{lineage_record_to_bytes, shake256_256, verify_witness_chain};
use rvf_runtime::{CustomSegmentType, QueryOptions, RvfOptions, RvfStore, LINEAGE_SEG_TYPE};

use rvf_types::{DerivationType, ErrorCode, RvfError, WITNESS_DERIVATION};
use std::path::Path;

const DIM: u16 = 4;

fn opts() -> RvfOptions {
    RvfOptions {
        dimension: DIM,
        ..Default::default()
    }
}

fn store_with(path: &Path, ids: &[u64]) -> RvfStore {
    let mut s = RvfStore::create(path, opts()).unwrap();
    ingest(&mut s, ids);
    s
}

fn ingest(s: &mut RvfStore, ids: &[u64]) {
    let rows: Vec<Vec<f32>> = ids.iter().map(|&i| vec![i as f32, 1.0, 0.0, 0.5]).collect();
    let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    s.ingest_batch(&refs, ids, None).unwrap();
}

fn code(e: RvfError) -> ErrorCode {
    match e {
        RvfError::Code(c) => c,
        other => panic!("unexpected error {other:?}"),
    }
}

fn failure<T>(r: Result<T, RvfError>) -> ErrorCode {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => code(e),
    }
}

/// `(type, payload)` of every wire segment in file order.
fn segments(bytes: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 64 <= bytes.len() && bytes[pos..pos + 4] == 0x5256_4653u32.to_le_bytes() {
        let plen = u64::from_le_bytes(bytes[pos + 0x10..pos + 0x18].try_into().unwrap()) as usize;
        out.push((bytes[pos + 5], bytes[pos + 64..pos + 64 + plen].to_vec()));
        pos = (pos + 64 + plen + 63) & !63;
    }
    out
}

fn witness_chain(path: &Path) -> Vec<u8> {
    segments(&std::fs::read(path).unwrap())
        .into_iter()
        .filter(|(t, _)| *t == 0x0A)
        .flat_map(|(_, p)| p)
        .collect()
}

// ---- #1289: derive records its type ---------------------------------------------

const SINGLE_PARENT_TYPES: [DerivationType; 8] = [
    DerivationType::Clone,
    DerivationType::Filter,
    DerivationType::Quantize,
    DerivationType::Reindex,
    DerivationType::Transform,
    DerivationType::Snapshot,
    DerivationType::UserDefined,
    // Still accepted by derive() (no breakage): recorded with its one parent.
    DerivationType::Merge,
];

#[test]
fn derive_records_the_requested_type_and_it_survives_reopen() {
    let d = tempfile::tempdir().unwrap();
    let parent = store_with(&d.path().join("p.rvf"), &[1, 2]);
    for (i, ty) in SINGLE_PARENT_TYPES.into_iter().enumerate() {
        let child_path = d.path().join(format!("c{i}.rvf"));
        let child = parent.derive(&child_path, ty, None).unwrap();
        assert_eq!(child.derivation_type().unwrap(), Some(ty), "{ty:?}");
        let records = child.lineage_records().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].file_id, *child.file_id());
        assert_eq!(records[0].parent_id, *parent.file_id());
        assert_eq!(records[0].parent_hash, child.file_identity().parent_hash);
        child.close().unwrap();

        let reopened = RvfStore::open_readonly(&child_path).unwrap();
        assert_eq!(
            reopened.derivation_type().unwrap(),
            Some(ty),
            "{ty:?} reopen"
        );
        assert_eq!(reopened.parent_id(), parent.file_id());
    }
}

#[test]
fn derive_chains_a_witness_entry_over_the_lineage_record() {
    let d = tempfile::tempdir().unwrap();
    let parent = store_with(&d.path().join("p.rvf"), &[1]);
    let child_path = d.path().join("c.rvf");
    let child = parent
        .derive(&child_path, DerivationType::Snapshot, None)
        .unwrap();
    let record = child.lineage_records().unwrap().remove(0);
    child.close().unwrap();

    let entries = verify_witness_chain(&witness_chain(&child_path)).expect("chain verifies");
    let derivation: Vec<_> = entries
        .iter()
        .filter(|e| e.witness_type == WITNESS_DERIVATION)
        .collect();
    assert_eq!(derivation.len(), 1);
    assert_eq!(
        derivation[0].action_hash,
        shake256_256(&lineage_record_to_bytes(&record)),
        "witness commits to the readable record"
    );
    // The record sits in its own segment type, listed once.
    let lineage_segs = segments(&std::fs::read(&child_path).unwrap())
        .into_iter()
        .filter(|(t, _)| *t == LINEAGE_SEG_TYPE)
        .count();
    assert_eq!(lineage_segs, 1);
}

#[test]
fn single_parent_merge_is_recorded_and_branch_records_clone() {
    let d = tempfile::tempdir().unwrap();
    let parent = store_with(&d.path().join("p.rvf"), &[1, 2, 3]);
    let m = parent
        .derive(&d.path().join("m.rvf"), DerivationType::Merge, None)
        .unwrap();
    assert_eq!(m.derivation_type().unwrap(), Some(DerivationType::Merge));
    assert_eq!(m.lineage_records().unwrap().len(), 1);
    assert_eq!(m.parent_id(), parent.file_id());

    let branch = parent.branch(&d.path().join("b.rvf")).unwrap();
    assert_eq!(
        branch.derivation_type().unwrap(),
        Some(DerivationType::Clone)
    );
    assert_eq!(
        branch.status().total_vectors,
        0,
        "COW child holds no local rows"
    );
    let hits = branch
        .query(&[2.0, 1.0, 0.0, 0.5], 1, &QueryOptions::default())
        .unwrap();
    assert_eq!(hits[0].id, 2, "branch still reads through to its parent");
}

#[test]
fn root_files_record_no_derivation() {
    let d = tempfile::tempdir().unwrap();
    let root = store_with(&d.path().join("r.rvf"), &[1]);
    assert_eq!(root.derivation_type().unwrap(), None);
    assert!(root.lineage_records().unwrap().is_empty());
}

#[test]
fn lineage_record_survives_compaction() {
    let d = tempfile::tempdir().unwrap();
    let parent = store_with(&d.path().join("p.rvf"), &[1]);
    let child_path = d.path().join("c.rvf");
    let mut child = parent
        .derive(&child_path, DerivationType::Transform, None)
        .unwrap();
    ingest(&mut child, &[10, 11, 12]);
    child.delete(&[11]).unwrap();
    child.compact().unwrap();
    child.close().unwrap();
    let child = RvfStore::open(&child_path).unwrap();
    assert_eq!(
        child.derivation_type().unwrap(),
        Some(DerivationType::Transform)
    );
    assert_eq!(child.status().total_vectors, 2);
}

// ---- #1289: merge records every parent --------------------------------------------

#[test]
fn merge_records_every_parent_in_declared_order() {
    let d = tempfile::tempdir().unwrap();
    let a = store_with(&d.path().join("a.rvf"), &[1]);
    let b_root = store_with(&d.path().join("b0.rvf"), &[2]);
    let b = b_root
        .derive(&d.path().join("b.rvf"), DerivationType::Filter, None)
        .unwrap(); // depth 1
    let c = store_with(&d.path().join("c.rvf"), &[3]);

    let path = d.path().join("m.rvf");
    let m = RvfStore::merge(&[&b, &a, &c], &path, None).unwrap();
    let check = |m: &RvfStore| {
        assert_eq!(m.derivation_type().unwrap(), Some(DerivationType::Merge));
        let parents: Vec<[u8; 16]> = m
            .lineage_records()
            .unwrap()
            .iter()
            .map(|r| r.parent_id)
            .collect();
        assert_eq!(parents, vec![*b.file_id(), *a.file_id(), *c.file_id()]);
        assert_eq!(m.parent_id(), b.file_id(), "FileIdentity names parents[0]");
        assert_eq!(m.lineage_depth(), 2, "1 + max(parent depth)");
    };
    check(&m);
    m.close().unwrap();
    check(&RvfStore::open_readonly(&path).unwrap());

    let entries = verify_witness_chain(&witness_chain(&path)).unwrap();
    assert_eq!(
        entries
            .iter()
            .filter(|e| e.witness_type == WITNESS_DERIVATION)
            .count(),
        3,
        "one witness entry per parent"
    );
}

#[test]
fn merge_rejects_bad_parent_sets() {
    let d = tempfile::tempdir().unwrap();
    let a = store_with(&d.path().join("a.rvf"), &[1]);
    let b = store_with(&d.path().join("b.rvf"), &[2]);
    let wide = RvfStore::create(
        &d.path().join("w.rvf"),
        RvfOptions {
            dimension: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let m = |name: &str, parents: &[&RvfStore]| {
        failure(RvfStore::merge(parents, &d.path().join(name), None))
    };
    assert_eq!(m("1.rvf", &[&a]), ErrorCode::LineageBroken);
    assert_eq!(m("2.rvf", &[&a, &a]), ErrorCode::LineageBroken);
    assert_eq!(m("3.rvf", &[&a, &b, &wide]), ErrorCode::DimensionMismatch);
    assert!(
        !d.path().join("3.rvf").exists(),
        "nothing created on rejection"
    );
}

// ---- #1290: application segments -------------------------------------------------

fn app(v: u8) -> CustomSegmentType {
    CustomSegmentType::new(v).unwrap()
}

#[test]
fn custom_segment_type_is_limited_to_the_application_range() {
    for v in [0x00u8, 0x01, 0x24, 0xBF, 0xF0, 0xFE, 0xFF] {
        assert_eq!(
            code(CustomSegmentType::new(v).unwrap_err()),
            ErrorCode::UnknownSegmentType
        );
    }
    assert_eq!(app(0xC0).get(), 0xC0);
    assert_eq!(app(0xEF).get(), 0xEF);
}

#[test]
fn append_read_list_latest_wins_and_survives_reopen() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("s.rvf");
    let mut s = store_with(&path, &[1]);
    let notes = app(0xC1);
    let other = app(0xC2);
    s.append_segment(notes, b"doc-1", b"v1").unwrap();
    s.append_segment(notes, b"doc-2", b"x").unwrap();
    s.append_segment(other, b"doc-1", b"other-type").unwrap();
    s.append_segment(notes, b"doc-1", b"v2").unwrap();
    s.append_segment(notes, b"", b"empty key").unwrap();

    let check = |s: &RvfStore| {
        assert_eq!(
            s.read_segment(notes, b"doc-1").unwrap().as_deref(),
            Some(&b"v2"[..])
        );
        assert_eq!(
            s.read_segment(other, b"doc-1").unwrap().as_deref(),
            Some(&b"other-type"[..])
        );
        assert_eq!(s.read_segment(notes, b"missing").unwrap(), None);
        assert_eq!(
            s.list_segments(notes).unwrap(),
            vec![
                (b"doc-2".to_vec(), b"x".to_vec()),
                (b"doc-1".to_vec(), b"v2".to_vec()),
                (Vec::new(), b"empty key".to_vec()),
            ]
        );
    };
    check(&s);
    s.close().unwrap();
    check(&RvfStore::open_readonly(&path).unwrap());

    let mut s = RvfStore::open(&path).unwrap();
    assert_eq!(
        code(s.append_segment(notes, &[0u8; 256], b"").unwrap_err()),
        ErrorCode::SegmentTooLarge
    );
    s.close().unwrap();
    let mut ro = RvfStore::open_readonly(&path).unwrap();
    assert_eq!(
        code(ro.append_segment(notes, b"k", b"v").unwrap_err()),
        ErrorCode::ReadOnly
    );
}

#[test]
fn custom_segment_append_is_witnessed() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("s.rvf");
    let mut s = RvfStore::create(&path, opts()).unwrap();
    let seg_id = s.append_segment(app(0xD0), b"k", b"value").unwrap();
    s.close().unwrap();
    let entries = verify_witness_chain(&witness_chain(&path)).unwrap();
    let mut action = b"rvf.custom_segment\0".to_vec();
    action.push(0xD0);
    action.extend_from_slice(&seg_id.to_le_bytes());
    action.extend_from_slice(&[1, 0]);
    action.extend_from_slice(b"kvalue");
    assert_eq!(entries.last().unwrap().action_hash, shake256_256(&action));
}

#[test]
fn custom_segments_survive_compaction() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("s.rvf");
    let mut s = store_with(&path, &[1, 2, 3]);
    let ty = app(0xE0);
    s.append_segment(ty, b"schema", b"v1").unwrap();
    s.append_segment(ty, b"schema", b"v2").unwrap();
    s.delete(&[2]).unwrap();
    s.compact().unwrap();
    assert_eq!(
        s.read_segment(ty, b"schema").unwrap().as_deref(),
        Some(&b"v2"[..])
    );
    s.close().unwrap();
    let s = RvfStore::open(&path).unwrap();
    assert_eq!(
        s.read_segment(ty, b"schema").unwrap().as_deref(),
        Some(&b"v2"[..])
    );
    assert_eq!(s.status().total_vectors, 2);
}

#[cfg(feature = "ed25519")]
#[test]
fn custom_segments_are_covered_by_the_signature_and_survive_signed_compaction() {
    use rvf_types::ed25519::Ed25519Keypair;
    use rvf_types::security::SecurityPolicy;

    let secret = [0x11u8; 32];
    let signed_opts = |trusted: Vec<[u8; 32]>| RvfOptions {
        dimension: DIM,
        trusted_signers: trusted,
        freshness: Some(rvf_runtime::Freshness::Unchecked),
        ..Default::default()
    };
    let public = Ed25519Keypair::from_secret(&secret).public_key();
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("signed.rvf");
    let ty = app(0xC5);
    {
        let mut s = RvfStore::create_signed(&path, signed_opts(vec![]), &secret).unwrap();
        ingest(&mut s, &[1, 2, 3]);
        s.append_segment(ty, b"note", b"signed value").unwrap();
        s.delete(&[2]).unwrap();
        s.compact().unwrap();
        s.close().unwrap();
    }
    let s = RvfStore::open_with_policy(&path, signed_opts(vec![public]), SecurityPolicy::Strict)
        .expect("Strict accepts the signed file");
    assert_eq!(
        s.read_segment(ty, b"note").unwrap().as_deref(),
        Some(&b"signed value"[..])
    );
    drop(s);

    // Tamper with the value bytes: Strict refuses the read (or the open).
    let mut bytes = std::fs::read(&path).unwrap();
    let needle = b"signed value";
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .unwrap();
    bytes[at] ^= 0x20;
    std::fs::write(&path, &bytes).unwrap();
    let tampered =
        RvfStore::open_with_policy(&path, signed_opts(vec![public]), SecurityPolicy::Strict)
            .and_then(|s| s.read_segment(ty, b"note"));
    assert!(tampered.is_err(), "altered custom segment must not verify");
}
