//! Format round trips with the published rvf-runtime 0.5.3 for the lineage
//! record segment (#1289) and application segments (#1290):
//! - 0.5.3 opens, writes and compacts files the new runtime wrote, and its
//!   compaction carries the new segments;
//! - the new runtime opens files 0.5.3 wrote (no lineage record: `None`).
//!
//! Run with `cargo test --manifest-path crates/rvf/tests/format-compat/Cargo.toml`.

use rvf_runtime::{CustomSegmentType, RvfOptions, RvfStore};
use rvf_runtime_0_5_3 as old;
use rvf_types::DerivationType;
use std::path::Path;

const DIM: u16 = 4;

fn opts() -> RvfOptions {
    RvfOptions {
        dimension: DIM,
        ..Default::default()
    }
}

fn row(i: u64) -> Vec<f32> {
    vec![i as f32, 1.0, 0.0, 0.5]
}

fn ingest(s: &mut RvfStore, ids: &[u64]) {
    let rows: Vec<Vec<f32>> = ids.iter().map(|&i| row(i)).collect();
    let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    s.ingest_batch(&refs, ids, None).unwrap();
}

/// 0.5.3 session: open, ingest one row, delete one, compact, close.
fn old_rewrite(path: &Path, add: u64, delete: u64) {
    let mut o = old::RvfStore::open(path).expect("0.5.3 opens the file");
    o.ingest_batch(&[&row(add)[..]], &[add], None).unwrap();
    o.delete(&[delete]).unwrap();
    o.compact().unwrap();
    o.close().unwrap();
}

#[test]
fn old_runtime_opens_a_new_derived_file_and_keeps_its_record_through_compaction() {
    let d = tempfile::tempdir().unwrap();
    let mut parent = RvfStore::create(&d.path().join("p.rvf"), opts()).unwrap();
    ingest(&mut parent, &[1]);
    let child_path = d.path().join("c.rvf");
    let mut child = parent
        .derive(&child_path, DerivationType::Quantize, None)
        .unwrap();
    ingest(&mut child, &[5, 6, 7]);
    child.close().unwrap();

    {
        let o = old::RvfStore::open_readonly(&child_path).expect("0.5.3 opens the new file");
        assert_eq!(o.status().total_vectors, 3);
        assert_eq!(o.parent_id(), parent.file_id());
        assert_eq!(o.lineage_depth(), 1);
    }
    old_rewrite(&child_path, 8, 5);

    let child = RvfStore::open(&child_path).unwrap();
    assert_eq!(
        child.derivation_type().unwrap(),
        Some(DerivationType::Quantize),
        "0.5.3 compaction carried the lineage segment"
    );
    assert_eq!(child.status().total_vectors, 3);
}

#[test]
fn old_runtime_opens_a_new_merge_child() {
    let d = tempfile::tempdir().unwrap();
    let mut a = RvfStore::create(&d.path().join("a.rvf"), opts()).unwrap();
    ingest(&mut a, &[1]);
    let mut b = RvfStore::create(&d.path().join("b.rvf"), opts()).unwrap();
    ingest(&mut b, &[2]);
    let path = d.path().join("m.rvf");
    RvfStore::merge(&[&a, &b], &path, None)
        .unwrap()
        .close()
        .unwrap();

    let o = old::RvfStore::open_readonly(&path).expect("0.5.3 opens a merge child");
    assert_eq!(o.parent_id(), a.file_id(), "older readers see parents[0]");
    assert_eq!(o.lineage_depth(), 1);
    drop(o);
    let m = RvfStore::open_readonly(&path).unwrap();
    assert_eq!(m.lineage_records().unwrap().len(), 2);
}

#[test]
fn new_runtime_opens_an_old_derived_file() {
    let d = tempfile::tempdir().unwrap();
    let old_child_path = d.path().join("c.rvf");
    let old_parent_id = {
        let mut p = old::RvfStore::create(
            &d.path().join("p.rvf"),
            old::RvfOptions {
                dimension: DIM,
                ..Default::default()
            },
        )
        .unwrap();
        p.ingest_batch(&[&row(1)[..]], &[1], None).unwrap();
        p.derive(
            &old_child_path,
            rvf_types_0_3_1::DerivationType::Filter,
            None,
        )
        .unwrap()
        .close()
        .unwrap();
        let id = *p.file_id();
        p.close().unwrap();
        id
    };

    let mut child = RvfStore::open(&old_child_path).expect("opens a 0.5.3 file");
    assert_eq!(child.parent_id(), &old_parent_id);
    assert_eq!(child.lineage_depth(), 1);
    assert_eq!(
        child.derivation_type().unwrap(),
        None,
        "0.5.3 recorded no derivation type"
    );
    ingest(&mut child, &[2]);
    let grandchild = child
        .derive(&d.path().join("g.rvf"), DerivationType::Reindex, None)
        .unwrap();
    assert_eq!(
        grandchild.derivation_type().unwrap(),
        Some(DerivationType::Reindex)
    );
    assert_eq!(grandchild.lineage_depth(), 2);
    grandchild.close().unwrap();
    child.close().unwrap();
    // And back: 0.5.3 still opens what the new runtime appended to its file.
    let o = old::RvfStore::open_readonly(&old_child_path).unwrap();
    assert_eq!(o.status().total_vectors, 1);
}

#[test]
fn custom_segments_survive_old_runtime_compaction() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("s.rvf");
    let ty = CustomSegmentType::new(0xE0).unwrap();
    {
        let mut s = RvfStore::create(&path, opts()).unwrap();
        ingest(&mut s, &[1, 2, 3]);
        s.append_segment(ty, b"schema", b"v1").unwrap();
        s.append_segment(ty, b"schema", b"v2").unwrap();
        s.append_segment(ty, b"owner", b"app").unwrap();
        s.close().unwrap();
    }
    old_rewrite(&path, 9, 1);

    let s = RvfStore::open(&path).unwrap();
    assert_eq!(
        s.read_segment(ty, b"schema").unwrap().as_deref(),
        Some(&b"v2"[..]),
        "latest value still wins after 0.5.3 compaction"
    );
    assert_eq!(
        s.list_segments(ty).unwrap(),
        vec![
            (b"schema".to_vec(), b"v2".to_vec()),
            (b"owner".to_vec(), b"app".to_vec()),
        ]
    );
    assert_eq!(s.status().total_vectors, 3);
}
