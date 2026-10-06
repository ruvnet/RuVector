//! IndexedDB persistence for `VectorDB` snapshots.
//!
//! A snapshot is one JSON record stored under a fixed key, so a save is a single
//! atomic `put`. A save only reports success after the transaction fires
//! `complete`; `error` and `abort` (including quota exceeded) become `Err`.

use js_sys::{Function, Promise};
use ruvector_core::types::{DistanceMetric, VectorEntry};
use serde::{Deserialize, Serialize};
use std::{cell::Cell, collections::HashSet, rc::Rc};
use wasm_bindgen::{closure::Closure, JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbDatabase, IdbFactory, IdbOpenDbRequest, IdbRequest, IdbTransactionMode};

/// Bumped whenever the snapshot layout changes; a mismatch fails loudly on load.
pub(crate) const FORMAT_VERSION: u32 = 1;
const IDB_VERSION: u32 = 1;
const STORE: &str = "ruvector";
const KEY: &str = "snapshot";
const MAX_NAME_LEN: usize = 128;
/// Upper bound on an untrusted payload before it is parsed.
const MAX_SNAPSHOT_CHARS: usize = 256 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
pub(crate) struct Snapshot {
    pub format: u32,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub hnsw: bool,
    pub entries: Vec<VectorEntry>,
}

/// Reject names that are empty, oversized or contain anything but `[A-Za-z0-9_.-]`.
pub(crate) fn validate_name(name: &str) -> Result<(), JsValue> {
    let ok = !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
    if ok {
        Ok(())
    } else {
        Err(JsValue::from_str(
            "Invalid database name: use 1-128 chars of [A-Za-z0-9_.-]",
        ))
    }
}

pub(crate) fn parse_snapshot(json: &str) -> Result<Snapshot, JsValue> {
    if json.len() > MAX_SNAPSHOT_CHARS {
        return Err(JsValue::from_str("Saved database is too large to load"));
    }
    let snap: Snapshot = serde_json::from_str(json)
        .map_err(|e| JsValue::from_str(&format!("Corrupt saved database: {e}")))?;
    if snap.format != FORMAT_VERSION {
        return Err(JsValue::from_str(&format!(
            "Unsupported saved database format {} (this build reads {})",
            snap.format, FORMAT_VERSION
        )));
    }
    // Duplicate ids would make `insert_batch` rebuild the index once per duplicate.
    let mut seen = HashSet::with_capacity(snap.entries.len());
    for entry in &snap.entries {
        if let Some(id) = &entry.id {
            if !seen.insert(id.as_str()) {
                return Err(JsValue::from_str(&format!(
                    "Corrupt saved database: duplicate id '{id}'"
                )));
            }
        }
    }
    Ok(snap)
}

fn dom_error(err: Option<web_sys::DomException>, fallback: &str) -> JsValue {
    match err {
        Some(e) => JsValue::from_str(&format!("IndexedDB {}: {}", e.name(), e.message())),
        None => JsValue::from_str(fallback),
    }
}

/// A promise that settles with the request's result or error.
fn request_promise(req: &IdbRequest) -> Promise {
    Promise::new(&mut |resolve: Function, reject: Function| {
        let r = req.clone();
        let ok = Closure::once_into_js(move || {
            let _ = resolve.call1(&JsValue::NULL, &r.result().unwrap_or(JsValue::UNDEFINED));
        });
        req.set_onsuccess(Some(ok.unchecked_ref()));
        let r = req.clone();
        let err = Closure::once_into_js(move || {
            let _ = reject.call1(
                &JsValue::NULL,
                &dom_error(r.error().ok().flatten(), "IndexedDB request failed"),
            );
        });
        req.set_onerror(Some(err.unchecked_ref()));
    })
}

/// Open `name`. With `create == false` a database that does not exist yet is
/// not created: the upgrade is aborted and `Ok(None)` is returned.
async fn open(name: &str, create: bool) -> Result<Option<IdbDatabase>, JsValue> {
    validate_name(name)?;
    let factory: IdbFactory = js_sys::Reflect::get(&js_sys::global(), &"indexedDB".into())?
        .dyn_into()
        .map_err(|_| JsValue::from_str("IndexedDB is not available in this environment"))?;
    let req: IdbOpenDbRequest = factory.open_with_u32(name, IDB_VERSION)?;
    let missing = Rc::new(Cell::new(false));
    let (upgrade_req, upgrade_missing) = (req.clone(), missing.clone());
    let upgrade = Closure::once_into_js(move || {
        if create {
            if let Ok(db) = upgrade_req
                .result()
                .and_then(|r| r.dyn_into::<IdbDatabase>())
            {
                let _ = db.create_object_store(STORE);
            }
        } else if let Some(tx) = upgrade_req.transaction() {
            upgrade_missing.set(true);
            let _ = tx.abort();
        }
    });
    req.set_onupgradeneeded(Some(upgrade.unchecked_ref()));
    let blocked = Closure::once_into_js(|| {});
    req.set_onblocked(Some(blocked.unchecked_ref()));
    match JsFuture::from(request_promise(&req)).await {
        Ok(v) => v
            .dyn_into::<IdbDatabase>()
            .map(Some)
            .map_err(|_| JsValue::from_str("IndexedDB open returned an unexpected value")),
        Err(_) if missing.get() => Ok(None),
        Err(e) => Err(e),
    }
}

/// Write the snapshot and resolve only once the transaction has committed.
pub(crate) async fn put(name: &str, json: String) -> Result<(), JsValue> {
    let db = open(name, true)
        .await?
        .ok_or_else(|| JsValue::from_str("IndexedDB open returned no database"))?;
    let result = commit(&db, json).await;
    db.close();
    result
}

async fn commit(db: &IdbDatabase, json: String) -> Result<(), JsValue> {
    let tx = db.transaction_with_str_and_mode(STORE, IdbTransactionMode::Readwrite)?;
    let store = tx.object_store(STORE)?;
    store.put_with_key(&JsValue::from_str(&json), &JsValue::from_str(KEY))?;
    let done = Promise::new(&mut |resolve: Function, reject: Function| {
        let complete = Closure::once_into_js(move || {
            let _ = resolve.call0(&JsValue::NULL);
        });
        tx.set_oncomplete(Some(complete.unchecked_ref()));
        for abort_like in [true, false] {
            let (rej, t) = (reject.clone(), tx.clone());
            let handler = Closure::once_into_js(move || {
                let _ = rej.call1(
                    &JsValue::NULL,
                    &dom_error(t.error(), "IndexedDB transaction aborted"),
                );
            });
            if abort_like {
                tx.set_onabort(Some(handler.unchecked_ref()));
            } else {
                tx.set_onerror(Some(handler.unchecked_ref()));
            }
        }
    });
    JsFuture::from(done).await.map(|_| ())
}

/// Read the stored snapshot JSON, or `None` when nothing was saved under `name`.
pub(crate) async fn get(name: &str) -> Result<Option<String>, JsValue> {
    let Some(db) = open(name, false).await? else {
        return Ok(None);
    };
    let result = read(&db).await;
    db.close();
    result
}

async fn read(db: &IdbDatabase) -> Result<Option<String>, JsValue> {
    let tx = db.transaction_with_str_and_mode(STORE, IdbTransactionMode::Readonly)?;
    let req = tx.object_store(STORE)?.get(&JsValue::from_str(KEY))?;
    let value = JsFuture::from(request_promise(&req)).await?;
    if value.is_undefined() || value.is_null() {
        return Ok(None);
    }
    value
        .as_string()
        .map(Some)
        .ok_or_else(|| JsValue::from_str("Corrupt saved database: payload is not a string"))
}
