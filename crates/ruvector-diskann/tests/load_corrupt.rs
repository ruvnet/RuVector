//! `DiskAnnIndex::load` treats the on-disk files as untrusted input: a
//! truncated or crafted file must return `Err`, never panic or attempt an
//! unbounded allocation. Through the NAPI binding a panic aborts the whole
//! Node.js process, so each of these used to be a one-file denial of service.

use ruvector_diskann::{DiskAnnConfig, DiskAnnError, DiskAnnIndex};
use std::fs;
use std::path::Path;

const DIM: usize = 4;

fn config(pq_subspaces: usize) -> DiskAnnConfig {
    DiskAnnConfig {
        dim: DIM,
        max_degree: 8,
        build_beam: 16,
        search_beam: 16,
        pq_subspaces,
        ..Default::default()
    }
}

fn populated(n: usize, pq_subspaces: usize) -> DiskAnnIndex {
    let mut index = DiskAnnIndex::new(config(pq_subspaces));
    for i in 0..n {
        let base = i as f32;
        index
            .insert(format!("v{i}"), vec![base, base + 1.0, base * 0.5, -base])
            .unwrap();
    }
    index
}

fn saved_built(dir: &Path, pq_subspaces: usize) {
    let mut index = populated(40, pq_subspaces);
    index.build().unwrap();
    index.save(dir).unwrap();
}

fn assert_load_err(dir: &Path) {
    assert!(
        DiskAnnIndex::load(dir).is_err(),
        "load must reject corrupt files"
    );
    assert!(
        DiskAnnIndex::load_mmap(dir).is_err(),
        "load_mmap must reject corrupt files"
    );
}

fn edit_graph(dir: &Path, edit: impl FnOnce(&mut Vec<u8>)) {
    let path = dir.join("graph.bin");
    let mut bytes = fs::read(&path).unwrap();
    edit(&mut bytes);
    fs::write(path, bytes).unwrap();
}

#[test]
fn save_before_build_round_trips_as_unbuilt() {
    let dir = tempfile::tempdir().unwrap();
    populated(20, 0).save(dir.path()).unwrap();

    let mut loaded = DiskAnnIndex::load(dir.path()).unwrap();
    assert_eq!(loaded.count(), 20);
    assert!(matches!(
        loaded.search(&[0.0; DIM], 3),
        Err(DiskAnnError::NotBuilt)
    ));

    loaded.build().unwrap();
    let results = loaded.search(&[3.0, 4.0, 1.5, -3.0], 3).unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, "v3");
}

#[test]
fn rejects_truncated_graph() {
    let dir = tempfile::tempdir().unwrap();
    saved_built(dir.path(), 0);
    edit_graph(dir.path(), |b| b.truncate(b.len() - 3));
    assert_load_err(dir.path());
    edit_graph(dir.path(), |b| b.truncate(5));
    assert_load_err(dir.path());
}

#[test]
fn rejects_out_of_range_neighbor_and_medoid() {
    let dir = tempfile::tempdir().unwrap();
    saved_built(dir.path(), 0);
    // First neighbor of node 0 sits after the header (16) and its degree (4).
    edit_graph(dir.path(), |b| {
        b[20..24].copy_from_slice(&0xffff_fff0u32.to_le_bytes())
    });
    assert_load_err(dir.path());

    saved_built(dir.path(), 0);
    edit_graph(dir.path(), |b| {
        b[0..8].copy_from_slice(&1000u64.to_le_bytes())
    });
    assert_load_err(dir.path());
}

#[test]
fn rejects_huge_node_count_and_degree_without_allocating() {
    let dir = tempfile::tempdir().unwrap();
    saved_built(dir.path(), 0);
    edit_graph(dir.path(), |b| {
        b[8..16].copy_from_slice(&(u64::MAX / 2).to_le_bytes())
    });
    assert_load_err(dir.path());

    saved_built(dir.path(), 0);
    edit_graph(dir.path(), |b| {
        b[16..20].copy_from_slice(&u32::MAX.to_le_bytes())
    });
    assert_load_err(dir.path());
}

#[test]
fn rejects_config_missing_required_fields() {
    let dir = tempfile::tempdir().unwrap();
    saved_built(dir.path(), 0);
    let path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    config.as_object_mut().unwrap().remove("dim");
    fs::write(&path, config.to_string()).unwrap();
    assert_load_err(dir.path());
}

#[test]
fn rejects_truncated_pq_codes() {
    let dir = tempfile::tempdir().unwrap();
    saved_built(dir.path(), 2);
    let path = dir.path().join("pq_codes.bin");
    let codes = fs::read(&path).unwrap();
    fs::write(&path, &codes[..codes.len() / 2]).unwrap();
    assert_load_err(dir.path());
}

#[test]
fn rejects_pq_with_huge_length_prefix() {
    let dir = tempfile::tempdir().unwrap();
    saved_built(dir.path(), 2);
    // bincode standard: m=2, dsub=2, dim=4 as varints, then the centroids Vec
    // length as a u64 varint (tag 253) claiming ~2^56 elements.
    let mut crafted = vec![2u8, 2, 4, 253];
    crafted.extend_from_slice(&(1u64 << 56).to_le_bytes());
    fs::write(dir.path().join("pq.bin"), crafted).unwrap();
    assert_load_err(dir.path());
}

#[test]
fn rejects_pq_inconsistent_with_dimension() {
    let dir = tempfile::tempdir().unwrap();
    saved_built(dir.path(), 2);
    // m=1, dsub=8, dim=8: self-consistent, but not this index's dim of 4.
    fs::write(dir.path().join("pq.bin"), [1u8, 8, 8, 0, 1]).unwrap();
    assert_load_err(dir.path());
}

#[test]
fn search_returns_k_after_heavy_deletes() {
    let mut index = populated(200, 0);
    index.build().unwrap();
    for i in 0..150 {
        assert!(index.delete(&format!("v{i}")).unwrap());
    }
    let results = index.search(&[0.0; DIM], 10).unwrap();
    assert_eq!(results.len(), 10);
    assert!(results
        .iter()
        .all(|r| r.id[1..].parse::<usize>().unwrap() >= 150));
}
