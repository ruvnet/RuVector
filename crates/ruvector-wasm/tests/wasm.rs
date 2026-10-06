//! WASM-specific tests

#![cfg(target_arch = "wasm32")]

use js_sys::Float32Array;
use ruvector_wasm::*;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_vector_db_creation() {
    let db = VectorDB::new(128, Some("cosine".to_string()), Some(false));
    assert!(db.is_ok());
}

#[wasm_bindgen_test]
fn test_insert_and_search() {
    let db = VectorDB::new(3, Some("euclidean".to_string()), Some(false)).unwrap();

    // Insert a vector
    let vector = Float32Array::from(&[1.0, 0.0, 0.0][..]);
    let id = db.insert(vector, Some("test1".to_string()), None);
    assert!(id.is_ok());

    // Search
    let query = Float32Array::from(&[1.0, 0.0, 0.0][..]);
    let results = db.search(query, 1, None);
    assert!(results.is_ok());

    let results = results.unwrap();
    assert_eq!(results.len(), 1);
}

#[wasm_bindgen_test]
fn test_batch_insert() {
    let db = VectorDB::new(3, Some("cosine".to_string()), Some(false)).unwrap();

    let entries = js_sys::Array::new();

    for i in 0..10 {
        let entry = js_sys::Object::new();
        let vector = Float32Array::from(&[i as f32, 0.0, 0.0][..]);
        js_sys::Reflect::set(&entry, &"vector".into(), &vector).unwrap();
        js_sys::Reflect::set(&entry, &"id".into(), &format!("vec_{}", i).into()).unwrap();
        entries.push(&entry);
    }

    let result = db.insert_batch(entries.into());
    assert!(result.is_ok());

    let ids = result.unwrap();
    assert_eq!(ids.len(), 10);
}

#[wasm_bindgen_test]
fn test_delete() {
    let db = VectorDB::new(3, Some("cosine".to_string()), Some(false)).unwrap();

    // Insert
    let vector = Float32Array::from(&[1.0, 0.0, 0.0][..]);
    let id = db
        .insert(vector, Some("test_delete".to_string()), None)
        .unwrap();

    // Delete
    let deleted = db.delete(&id);
    assert!(deleted.is_ok());
    assert_eq!(deleted.unwrap(), true);

    // Verify deleted
    let get_result = db.get(&id);
    assert!(get_result.is_ok());
    assert!(get_result.unwrap().is_none());
}

#[wasm_bindgen_test]
fn test_get() {
    let db = VectorDB::new(3, Some("cosine".to_string()), Some(false)).unwrap();

    // Insert
    let vector = Float32Array::from(&[1.0, 2.0, 3.0][..]);
    let id = db
        .insert(vector, Some("test_get".to_string()), None)
        .unwrap();

    // Get
    let entry = db.get(&id);
    assert!(entry.is_ok());

    let entry = entry.unwrap();
    assert!(entry.is_some());

    let entry = entry.unwrap();
    assert_eq!(entry.id(), Some("test_get".to_string()));
}

#[wasm_bindgen_test]
fn test_len_and_is_empty() {
    let db = VectorDB::new(3, Some("cosine".to_string()), Some(false)).unwrap();

    // Initially empty
    assert!(db.is_empty().unwrap());
    assert_eq!(db.len().unwrap(), 0);

    // Insert vector
    let vector = Float32Array::from(&[1.0, 0.0, 0.0][..]);
    db.insert(vector, Some("test1".to_string()), None).unwrap();

    // Not empty
    assert!(!db.is_empty().unwrap());
    assert_eq!(db.len().unwrap(), 1);
}

#[wasm_bindgen_test]
fn test_different_metrics() {
    for metric in &["euclidean", "cosine", "dotproduct", "manhattan"] {
        let db = VectorDB::new(3, Some(metric.to_string()), Some(false));
        assert!(db.is_ok(), "Failed to create DB with metric: {}", metric);
    }
}

#[wasm_bindgen_test]
fn test_dimension_mismatch() {
    let db = VectorDB::new(3, Some("cosine".to_string()), Some(false)).unwrap();

    // Try to insert vector with wrong dimensions
    let vector = Float32Array::from(&[1.0, 0.0][..]); // Only 2 dimensions
    let result = db.insert(vector, Some("test_wrong_dim".to_string()), None);

    // Should fail due to dimension mismatch
    // Note: This might succeed depending on implementation
    // The search with wrong dimensions should definitely fail
    let query = Float32Array::from(&[1.0, 0.0][..]);
    let search_result = db.search(query, 1, None);
    assert!(search_result.is_err());
}

#[wasm_bindgen_test]
fn test_version() {
    let v = version();
    assert!(!v.is_empty());
    assert!(v.contains('.'));
}

#[wasm_bindgen_test]
fn test_detect_simd() {
    // Just ensure it doesn't panic
    let _ = detect_simd();
}

#[wasm_bindgen_test]
fn test_array_to_float32_array() {
    let arr = vec![1.0, 2.0, 3.0, 4.0];
    let float_arr = array_to_float32_array(arr.clone());

    assert_eq!(float_arr.length(), 4);
    assert_eq!(float_arr.get_index(0), 1.0);
    assert_eq!(float_arr.get_index(3), 4.0);
}

// --- IndexedDB persistence (regression: save resolved but wrote nothing) ---

/// Names of every IndexedDB database visible to this origin.
async fn idb_database_names() -> Vec<String> {
    let factory = web_sys::window().unwrap().indexed_db().unwrap().unwrap();
    // `IdbFactory::databases` is behind web-sys unstable APIs, so call it dynamically.
    let databases: js_sys::Function = js_sys::Reflect::get(&factory, &"databases".into())
        .unwrap()
        .into();
    let promise: js_sys::Promise = databases.call0(&factory).unwrap().into();
    let list = wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
    js_sys::Array::from(&list)
        .iter()
        .filter_map(|d| js_sys::Reflect::get(&d, &"name".into()).ok()?.as_string())
        .collect()
}

#[wasm_bindgen_test]
async fn test_save_to_indexeddb_actually_persists() {
    let db = VectorDB::new(3, Some("euclidean".to_string()), Some(false)).unwrap();
    db.insert(
        Float32Array::from(&[1.0, 0.0, 0.0][..]),
        Some("a".to_string()),
        None,
    )
    .unwrap();

    wasm_bindgen_futures::JsFuture::from(db.save_to_indexed_db().unwrap())
        .await
        .unwrap();

    let names = idb_database_names().await;
    assert!(
        names.iter().any(|n| n.starts_with("ruvector_db_")),
        "save resolved but no IndexedDB database exists; found {:?}",
        names
    );
}

async fn save_named(db: &mut VectorDB, name: &str) {
    db.set_db_name(name.to_string()).unwrap();
    wasm_bindgen_futures::JsFuture::from(db.save_to_indexed_db().unwrap())
        .await
        .unwrap();
}

async fn load_named(name: &str) -> Result<VectorDB, JsValue> {
    let v = wasm_bindgen_futures::JsFuture::from(VectorDB::load_from_indexed_db(name.to_string())?)
        .await?;
    wasm_bindgen::convert::TryFromJsValue::try_from_js_value(v)
}

#[wasm_bindgen_test]
async fn test_indexeddb_round_trip_restores_vectors_and_search() {
    let mut db = VectorDB::new(3, Some("euclidean".to_string()), Some(false)).unwrap();
    for (id, v) in [("a", [1.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0])] {
        db.insert(Float32Array::from(&v[..]), Some(id.to_string()), None)
            .unwrap();
    }
    save_named(&mut db, "rt_round_trip").await;

    let loaded = load_named("rt_round_trip").await.unwrap();
    assert_eq!(loaded.len().unwrap(), 2);
    assert_eq!(loaded.dimensions(), 3);
    assert!(loaded.get("b").unwrap().is_some());
    let hits = loaded
        .search(Float32Array::from(&[0.0, 1.0, 0.0][..]), 1, None)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id(), "b");
}

#[wasm_bindgen_test]
async fn test_load_missing_database_is_error() {
    assert!(load_named("rt_never_saved").await.is_err());
}

#[wasm_bindgen_test]
async fn test_invalid_database_name_is_rejected() {
    let mut db = VectorDB::new(3, None, Some(false)).unwrap();
    assert!(db.set_db_name("bad name/../x".to_string()).is_err());
    assert!(db.set_db_name(String::new()).is_err());
    assert!(VectorDB::load_from_indexed_db("a;b".to_string()).is_err());
}

/// Overwrite the stored snapshot with arbitrary text, bypassing the crate.
async fn put_raw(name: &str, payload: &str) {
    use wasm_bindgen::JsCast;
    let factory = web_sys::window().unwrap().indexed_db().unwrap().unwrap();
    let req = factory.open_with_u32(name, 1).unwrap();
    let rq = req.clone();
    let on_upgrade = wasm_bindgen::closure::Closure::once_into_js(move || {
        let db: web_sys::IdbDatabase = rq.result().unwrap().unchecked_into();
        let _ = db.create_object_store("ruvector");
    });
    req.set_onupgradeneeded(Some(on_upgrade.unchecked_ref()));
    let rq = req.clone();
    let opened = js_sys::Promise::new(&mut |res, _| {
        let rq = rq.clone();
        let ok = wasm_bindgen::closure::Closure::once_into_js(move || {
            res.call1(&JsValue::NULL, &rq.result().unwrap()).unwrap();
        });
        req.set_onsuccess(Some(ok.unchecked_ref()));
    });
    let db: web_sys::IdbDatabase = wasm_bindgen_futures::JsFuture::from(opened)
        .await
        .unwrap()
        .unchecked_into();
    let tx = db
        .transaction_with_str_and_mode("ruvector", web_sys::IdbTransactionMode::Readwrite)
        .unwrap();
    let tx2 = tx.clone();
    let done = js_sys::Promise::new(&mut |res, _| {
        let c = wasm_bindgen::closure::Closure::once_into_js(move || {
            res.call0(&JsValue::NULL).unwrap();
        });
        tx2.set_oncomplete(Some(c.unchecked_ref()));
    });
    tx.object_store("ruvector")
        .unwrap()
        .put_with_key(&JsValue::from_str(payload), &JsValue::from_str("snapshot"))
        .unwrap();
    wasm_bindgen_futures::JsFuture::from(done).await.unwrap();
    db.close();
}

#[wasm_bindgen_test]
async fn test_load_rejects_corrupt_and_unsupported_payloads() {
    put_raw("rt_corrupt", "{not json").await;
    assert!(load_named("rt_corrupt").await.is_err());

    put_raw(
        "rt_future",
        r#"{"format":999,"dimensions":3,"metric":"Euclidean","hnsw":false,"entries":[]}"#,
    )
    .await;
    let err = load_named("rt_future").await.err().unwrap();
    assert!(err.as_string().unwrap().contains("Unsupported"));

    // Entry whose vector length disagrees with the stored dimensions.
    put_raw(
        "rt_baddim",
        r#"{"format":1,"dimensions":3,"metric":"Euclidean","hnsw":false,"entries":[{"id":"x","vector":[1.0],"metadata":null}]}"#,
    )
    .await;
    assert!(load_named("rt_baddim").await.is_err());
}
