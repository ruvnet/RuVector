//! Replacements and deletes must expose only current live HNSW rows.
#![cfg(feature = "hnsw")]
use ruvector_core::index::{hnsw::HnswIndex, VectorIndex};
use ruvector_core::types::{DistanceMetric, HnswConfig};

fn index() -> HnswIndex {
    HnswIndex::new(
        2,
        DistanceMetric::Euclidean,
        HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 100,
            max_elements: 100,
        },
    )
    .unwrap()
}

#[test]
fn replacement_returns_only_the_latest_vector_score() {
    let mut index = index();
    index.add("changed".into(), vec![1.0, 0.0]).unwrap();
    index.add("changed".into(), vec![100.0, 0.0]).unwrap();
    index.add("keeper".into(), vec![2.0, 0.0]).unwrap();
    let hits = index.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(hits.len(), 2, "one result per live caller ID");
    assert_eq!(hits[0].id, "keeper");
    assert_eq!(hits[0].score, 1.0);
    assert_eq!(hits[1].id, "changed");
    assert_eq!(hits[1].score, 99.0);
    assert_eq!(index.len(), 2);
}

#[test]
fn deleting_a_replaced_id_removes_all_searchable_versions() {
    let mut index = index();
    index.add("changed".into(), vec![1.0, 0.0]).unwrap();
    index.add("changed".into(), vec![100.0, 0.0]).unwrap();
    index.add("keeper".into(), vec![2.0, 0.0]).unwrap();
    assert!(index.remove(&"changed".into()).unwrap());
    let hits = index.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "keeper");
    assert!(!index.remove(&"changed".into()).unwrap());
    assert_eq!(index.len(), 1);
}

#[test]
fn repeated_ids_in_a_batch_keep_only_the_final_version() {
    let mut index = index();
    index
        .add_batch(vec![
            ("changed".into(), vec![1.0, 0.0]),
            ("changed".into(), vec![100.0, 0.0]),
            ("keeper".into(), vec![2.0, 0.0]),
        ])
        .unwrap();
    let hits = index.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
        vec!["keeper", "changed"]
    );
    assert_eq!(hits[1].score, 99.0);
    assert_eq!(index.len(), 2);
}

#[test]
fn deleted_nearby_graph_nodes_do_not_underfill_live_top_k() {
    let mut index = index();
    index
        .add_batch(
            (0..8)
                .map(|i| (i.to_string(), vec![i as f32, 0.0]))
                .collect(),
        )
        .unwrap();
    for i in 0..6 {
        assert!(index.remove(&i.to_string()).unwrap());
    }
    let hits = index.search_with_ef(&[0.0, 0.0], 2, 1).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
        vec!["6", "7"]
    );
    assert_eq!(
        hits.iter().map(|hit| hit.score).collect::<Vec<_>>(),
        vec![6.0, 7.0]
    );
}

#[test]
fn replacement_roundtrip_keeps_unique_latest_ids() {
    let mut index = index();
    index.add("changed".into(), vec![1.0, 0.0]).unwrap();
    index.add("changed".into(), vec![100.0, 0.0]).unwrap();
    index.add("keeper".into(), vec![2.0, 0.0]).unwrap();
    let restored = HnswIndex::deserialize(&index.serialize().unwrap()).unwrap();
    let hits = restored.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
        vec!["keeper", "changed"]
    );
    assert_eq!(hits[1].score, 99.0);
}

#[test]
fn ordinary_search_and_empty_live_index_keep_existing_behavior() {
    let mut index = index();
    assert!(index.search(&[0.0, 0.0], 3).unwrap().is_empty());
    index
        .add_batch(vec![
            ("near".into(), vec![1.0, 0.0]),
            ("far".into(), vec![2.0, 0.0]),
        ])
        .unwrap();
    assert_eq!(index.search(&[0.0, 0.0], 1).unwrap()[0].id, "near");
    assert!(index.search(&[0.0, 0.0], 0).unwrap().is_empty());
    assert!(index.search(&[0.0], 1).is_err());
    assert!(index.remove(&"near".into()).unwrap());
    assert!(index.remove(&"far".into()).unwrap());
    assert!(index.search(&[0.0, 0.0], 3).unwrap().is_empty());
}

#[test]
fn legacy_replacement_snapshot_restores_only_canonical_live_rows() {
    let mut restored =
        HnswIndex::deserialize(include_bytes!("fixtures/hnsw-legacy-replacement.bin")).unwrap();
    let hits = restored.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
        vec!["keeper", "changed"]
    );
    assert_eq!(hits[1].score, 99.0);
    assert!(restored.remove(&"changed".into()).unwrap());
    restored.add("new".into(), vec![3.0, 0.0]).unwrap();
    let hits = restored.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
        vec!["keeper", "new"]
    );
    assert_eq!(restored.len(), 2);
}

#[test]
fn sparse_deleted_labels_survive_restore_and_new_insertion() {
    let mut index = index();
    for i in 0..8 {
        index.add(i.to_string(), vec![i as f32, 0.0]).unwrap();
    }
    for i in 0..6 {
        assert!(index.remove(&i.to_string()).unwrap());
    }
    // Restore inserts only labels 6 and 7 at dense positions 0 and 1.
    let mut restored = HnswIndex::deserialize(&index.serialize().unwrap()).unwrap();
    let hits = restored.search(&[0.0, 0.0], 2).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
        vec!["6", "7"]
    );
    assert_eq!(
        hits.iter().map(|hit| hit.score).collect::<Vec<_>>(),
        vec![6.0, 7.0]
    );
    restored.add("new".into(), vec![0.0, 0.0]).unwrap();
    let again = HnswIndex::deserialize(&restored.serialize().unwrap()).unwrap();
    let hits = again.search(&[0.0, 0.0], 3).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
        vec!["new", "6", "7"]
    );
}
