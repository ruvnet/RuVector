//! Dimension errors must not leave memory storage ahead of the search index.
use ruvector_core::{
    types::{DbOptions, DistanceMetric, SearchQuery, VectorEntry},
    vector_db::VectorDB,
};

fn database() -> VectorDB {
    VectorDB::new(DbOptions {
        dimensions: 2,
        distance_metric: DistanceMetric::Euclidean,
        storage_path: "memory://batch-validation".into(),
        hnsw_config: None,
        quantization: None,
    })
    .unwrap()
}

fn entry(id: Option<&str>, vector: Vec<f32>) -> VectorEntry {
    VectorEntry {
        id: id.map(str::to_owned),
        vector,
        metadata: None,
    }
}

fn search_ids(db: &VectorDB) -> Vec<String> {
    db.search(SearchQuery {
        vector: vec![1.0, 0.0],
        k: 10,
        filter: None,
        ef_search: None,
    })
    .unwrap()
    .into_iter()
    .map(|hit| hit.id)
    .collect()
}

#[test]
fn rejected_batch_does_not_leave_a_storage_only_prefix() {
    let db = database();
    db.insert(entry(Some("seed"), vec![1.0, 0.0])).unwrap();
    let result = db.insert_batch([
        entry(Some("prefix"), vec![0.0, 1.0]),
        entry(Some("invalid"), vec![1.0]),
    ]);
    assert!(result.is_err());
    assert!(
        db.get("prefix").unwrap().is_none(),
        "rejected batch must not store its valid prefix"
    );
    assert_eq!(db.len().unwrap(), 1);
    assert_eq!(search_ids(&db), vec!["seed"]);
    assert!(db.get("invalid").unwrap().is_none());
}

#[test]
fn rejected_batch_preserves_an_existing_vector_and_metadata() {
    let db = database();
    let mut seed = entry(Some("seed"), vec![1.0, 0.0]);
    seed.metadata = Some(std::collections::HashMap::from([(
        "version".into(),
        serde_json::json!("old"),
    )]));
    db.insert(seed).unwrap();
    let mut replacement = entry(Some("seed"), vec![0.0, 1.0]);
    replacement.metadata = Some(std::collections::HashMap::from([(
        "version".into(),
        serde_json::json!("new"),
    )]));
    assert!(db
        .insert_batch([replacement, entry(Some("invalid"), vec![1.0])])
        .is_err());
    let stored = db.get("seed").unwrap().unwrap();
    assert_eq!(
        stored.vector,
        vec![1.0, 0.0],
        "rejected batch must preserve its pre-existing vector"
    );
    assert_eq!(
        stored.metadata.unwrap()["version"],
        serde_json::json!("old")
    );
    assert_eq!(search_ids(&db), vec!["seed"]);
}

#[test]
fn rejected_batch_does_not_consume_generated_ids() {
    let db = database();
    assert!(db
        .insert_batch([entry(None, vec![1.0, 0.0]), entry(None, vec![1.0])])
        .is_err());
    assert_eq!(
        db.insert(entry(None, vec![1.0, 0.0])).unwrap(),
        "vec_0",
        "rejected batch must not allocate ids"
    );
    assert_eq!(db.len().unwrap(), 1);
}

#[test]
fn invalid_first_entry_and_empty_batch_preserve_storage() {
    let db = database();
    assert!(db
        .insert_batch([entry(None, vec![1.0]), entry(None, vec![1.0, 0.0])])
        .is_err());
    assert_eq!(db.len().unwrap(), 0);
    assert!(db
        .insert_batch(Vec::<VectorEntry>::new())
        .unwrap()
        .is_empty());
    assert_eq!(db.len().unwrap(), 0);
}

#[test]
fn valid_batch_remains_available_to_storage_and_search() {
    let db = database();
    let ids = db
        .insert_batch([entry(None, vec![1.0, 0.0]), entry(None, vec![0.0, 1.0])])
        .unwrap();
    assert_eq!(ids, vec!["vec_0", "vec_1"]);
    assert_eq!(db.len().unwrap(), 2);
    assert_eq!(search_ids(&db), ids);
    assert_eq!(db.get("vec_1").unwrap().unwrap().vector, vec![0.0, 1.0]);
}
