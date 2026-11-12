//! Generated IDs must not replace caller-owned rows.
use ruvector_core::{
    types::{DbOptions, DistanceMetric, SearchQuery, VectorEntry},
    vector_db::VectorDB,
};
fn database() -> VectorDB {
    VectorDB::new(DbOptions {
        dimensions: 2,
        distance_metric: DistanceMetric::Euclidean,
        storage_path: "memory://generated-ids".into(),
        hnsw_config: None,
        quantization: None,
    })
    .unwrap()
}
fn entry(id: Option<&str>, x: f32) -> VectorEntry {
    VectorEntry {
        id: id.map(str::to_owned),
        vector: vec![x, 0.0],
        metadata: Some(std::collections::HashMap::from([(
            "marker".into(),
            serde_json::json!(x),
        )])),
    }
}
#[test]
fn default_insert_preserves_a_live_caller_id_and_its_metadata() {
    let db = database();
    db.insert(entry(Some("vec_0"), 1.0)).unwrap();
    let generated = db.insert(entry(None, 2.0)).unwrap();
    assert_ne!(generated, "vec_0");
    assert_eq!(db.len().unwrap(), 2);
    let kept = db.get("vec_0").unwrap().unwrap();
    assert_eq!(kept.vector, vec![1.0, 0.0]);
    assert_eq!(kept.metadata.unwrap()["marker"], serde_json::json!(1.0));
    let hits = db
        .search(SearchQuery {
            vector: vec![0.0, 0.0],
            k: 2,
            filter: None,
            ef_search: None,
        })
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_ne!(hits[0].id, hits[1].id);
}
#[test]
fn batch_defaults_preserve_preexisting_caller_rows() {
    let db = database();
    db.insert(entry(Some("vec_0"), 1.0)).unwrap();
    let ids = db
        .insert_batch([entry(None, 2.0), entry(None, 3.0)])
        .unwrap();
    assert!(!ids.iter().any(|id| id == "vec_0"));
    assert_ne!(ids[0], ids[1]);
    assert_eq!(db.len().unwrap(), 3);
    assert_eq!(db.get("vec_0").unwrap().unwrap().vector, vec![1.0, 0.0]);
}
#[test]
fn generated_batch_names_do_not_collide_with_later_declared_ids() {
    let db = database();
    let ids = db
        .insert_batch([entry(None, 1.0), entry(Some("vec_0"), 2.0)])
        .unwrap();
    assert_ne!(ids[0], ids[1]);
    assert_eq!(ids[1], "vec_0");
    assert_eq!(db.len().unwrap(), 2);
    assert_eq!(db.get(&ids[0]).unwrap().unwrap().vector, vec![1.0, 0.0]);
}
#[test]
fn explicit_upsert_and_normal_default_allocation_remain_supported() {
    let db = database();
    let id = db.insert(entry(None, 1.0)).unwrap();
    assert_eq!(id, "vec_0");
    assert_eq!(db.insert(entry(Some(&id), 2.0)).unwrap(), id);
    assert_eq!(db.len().unwrap(), 1);
    assert_eq!(db.get(&id).unwrap().unwrap().vector, vec![2.0, 0.0]);
    assert_eq!(db.insert(entry(None, 3.0)).unwrap(), "vec_1");
}
#[test]
fn concurrent_generated_inserts_preserve_preexisting_ids() {
    let storage =
        std::sync::Arc::new(ruvector_core::storage_memory::MemoryStorage::new(2).unwrap());
    for i in 0..8 {
        storage
            .insert(&entry(Some(&format!("vec_{i}")), i as f32))
            .unwrap();
    }
    let threads = (0..16)
        .map(|i| {
            let storage = storage.clone();
            std::thread::spawn(move || storage.insert(&entry(None, 100.0 + i as f32)).unwrap())
        })
        .collect::<Vec<_>>();
    let ids = threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(ids.len(), 16);
    assert_eq!(storage.len().unwrap(), 24);
    for i in 0..8 {
        assert_eq!(
            storage.get(&format!("vec_{i}")).unwrap().unwrap().vector,
            vec![i as f32, 0.0]
        );
    }
}
