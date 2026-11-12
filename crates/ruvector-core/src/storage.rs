//! Storage layer with redb for metadata and memory-mapped vectors
//!
//! This module is only available when the "storage" feature is enabled.
//! For WASM builds, use the in-memory storage backend instead.

#[cfg(feature = "storage")]
use crate::error::{Result, RuvectorError};
#[cfg(feature = "storage")]
use crate::types::{DbOptions, VectorEntry, VectorId};
#[cfg(feature = "storage")]
use bincode::config;
#[cfg(feature = "storage")]
use once_cell::sync::Lazy;
#[cfg(feature = "storage")]
use parking_lot::Mutex;
#[cfg(feature = "storage")]
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
#[cfg(feature = "storage")]
use serde_json;
#[cfg(feature = "storage")]
use std::collections::HashMap;
#[cfg(feature = "storage")]
use std::path::{Path, PathBuf};
#[cfg(feature = "storage")]
use std::sync::{Arc, Weak};

#[cfg(feature = "storage")]
const VECTORS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("vectors");
const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");
const CONFIG_TABLE: TableDefinition<&str, &str> = TableDefinition::new("config");

/// Key used to store database configuration in CONFIG_TABLE
const DB_CONFIG_KEY: &str = "__ruvector_db_config__";

/// Per-path guard around the pooled database handle.
///
/// The `Weak` (rather than a strong `Arc`) is what stops an erased-and-recreated
/// path from handing out a `Database` that still points at the unlinked inode:
/// the pool never keeps a database alive on its own. Opening and closing a
/// database both happen while this guard is held, so a close (redb's final write
/// transaction, fsync, and file-lock release, all of which run in
/// `Database::drop`) is never observed half-finished by a concurrent open.
type PathSlot = Mutex<Option<Weak<Database>>>;

// Global database connection pool to allow multiple VectorDB instances
// to share the same underlying database file. The global lock only guards the
// map itself; the slow work happens under the per-path slot lock so an fsync on
// one path never blocks opens of unrelated paths.
static DB_POOL: Lazy<Mutex<HashMap<PathBuf, Arc<PathSlot>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// The pooled database and the guard that owns its lifecycle.
struct Pooled {
    db: Arc<Database>,
    slot: Arc<PathSlot>,
}

/// Storage backend for vector database
pub struct VectorStorage {
    /// Always `Some` for a live handle; only `Drop` takes it.
    pooled: Option<Pooled>,
    path: PathBuf,
    dimensions: usize,
}

impl VectorStorage {
    /// Create or open a vector storage at the given path
    ///
    /// This method uses a global connection pool to allow multiple VectorDB
    /// instances to share the same underlying database file, fixing the
    /// "Database already open. Cannot acquire lock" error.
    pub fn new<P: AsRef<Path>>(path: P, dimensions: usize) -> Result<Self> {
        // SECURITY: Validate path to prevent directory traversal attacks
        let path_ref = path.as_ref();

        // Create parent directories if they don't exist (needed for canonicalize)
        if let Some(parent) = path_ref.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    RuvectorError::InvalidPath(format!("Failed to create directory: {}", e))
                })?;
            }
        }

        // Convert to absolute path first, then validate
        let path_buf = if path_ref.is_absolute() {
            path_ref.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|e| RuvectorError::InvalidPath(format!("Failed to get cwd: {}", e)))?
                .join(path_ref)
        };

        // SECURITY: Check for path traversal attempts (e.g., "../../../etc/passwd")
        // Only reject paths that contain ".." components trying to escape
        let path_str = path_ref.to_string_lossy();
        if path_str.contains("..") {
            // Verify the resolved path doesn't escape intended boundaries
            // For absolute paths, we allow them as-is (user explicitly specified)
            // For relative paths with "..", check they don't escape cwd
            if !path_ref.is_absolute() {
                if let Ok(cwd) = std::env::current_dir() {
                    // Normalize the path by resolving .. components
                    let mut normalized = cwd.clone();
                    for component in path_ref.components() {
                        match component {
                            std::path::Component::ParentDir => {
                                if !normalized.pop() || !normalized.starts_with(&cwd) {
                                    return Err(RuvectorError::InvalidPath(
                                        "Path traversal attempt detected".to_string(),
                                    ));
                                }
                            }
                            std::path::Component::Normal(c) => normalized.push(c),
                            _ => {}
                        }
                    }
                }
            }
        }

        // The database may not exist yet, so canonicalize its existing parent
        // before using the path as a pool key. This also resolves symlinked
        // directories and redundant components to the same redb file.
        let parent = path_buf
            .parent()
            .ok_or_else(|| RuvectorError::InvalidPath("Database path has no parent".to_string()))?;
        let file_name = path_buf.file_name().ok_or_else(|| {
            RuvectorError::InvalidPath("Database path has no file name".to_string())
        })?;
        let path_buf = parent
            .canonicalize()
            .map_err(|e| RuvectorError::InvalidPath(format!("Failed to resolve directory: {}", e)))?
            .join(file_name);

        // Claim this path's slot. Slot handles are only ever cloned while the
        // pool lock is held, which is what makes the reference-count check in
        // `release_slot_if_unused` sound.
        let slot = {
            let mut pool = DB_POOL.lock();
            Arc::clone(pool.entry(path_buf.clone()).or_default())
        };

        let db = match Self::open_pooled(&slot, &path_buf) {
            Ok(db) => db,
            Err(e) => {
                // Nothing was pooled, so this slot may now be garbage.
                Self::release_slot_if_unused(&path_buf, slot);
                return Err(e);
            }
        };

        Ok(Self {
            pooled: Some(Pooled { db, slot }),
            path: path_buf,
            dimensions,
        })
    }

    /// Reuse this path's pooled database, or open a fresh one under its guard.
    fn open_pooled(slot: &Arc<PathSlot>, path: &Path) -> Result<Arc<Database>> {
        let mut guard = slot.lock();

        if let Some(existing_db) = guard.as_ref().and_then(Weak::upgrade) {
            // Reuse existing database connection
            return Ok(existing_db);
        }

        // Create new database and publish it to the pool. On any failure below,
        // `new_db` is dropped before `guard`, so the file lock is released while
        // the slot is still held.
        let new_db = Arc::new(Database::create(path)?);

        // Initialize tables
        let write_txn = new_db.begin_write()?;
        {
            let _ = write_txn.open_table(VECTORS_TABLE)?;
            let _ = write_txn.open_table(METADATA_TABLE)?;
            let _ = write_txn.open_table(CONFIG_TABLE)?;
        }
        write_txn.commit()?;

        *guard = Some(Arc::downgrade(&new_db));
        Ok(new_db)
    }

    /// Drop this path's pool entry once nothing can reach it any more.
    ///
    /// Leaving the entry in place would leak one map entry per opened path.
    /// `slot` is consumed and released *under* the pool lock, so a remaining
    /// strong count of one proves the pool's own entry is the last handle: new
    /// handles are only ever cloned under this same lock, so no other thread can
    /// be mid-`new` for this path.
    fn release_slot_if_unused(path: &Path, slot: Arc<PathSlot>) {
        let mut pool = DB_POOL.lock();
        let is_ours = matches!(pool.get(path), Some(pooled) if Arc::ptr_eq(pooled, &slot));
        drop(slot);

        if is_ours && matches!(pool.get(path), Some(pooled) if Arc::strong_count(pooled) == 1) {
            pool.remove(path);
        }
    }

    /// Borrow the pooled database.
    ///
    /// Handing out `&Database` rather than the `Arc` keeps the strong count
    /// under this handle's control, which `Drop` relies on to decide whether it
    /// is closing the last reference.
    #[inline]
    fn db(&self) -> &Database {
        &self
            .pooled
            .as_ref()
            .expect("database handle used after drop")
            .db
    }

    /// Insert a vector entry
    pub fn insert(&self, entry: &VectorEntry) -> Result<VectorId> {
        if entry.vector.len() != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: entry.vector.len(),
            });
        }

        let id = entry
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let write_txn = self.db().begin_write()?;
        {
            let mut table = write_txn.open_table(VECTORS_TABLE)?;

            // Serialize vector data
            let vector_data = bincode::encode_to_vec(&entry.vector, config::standard())
                .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;

            table.insert(id.as_str(), vector_data.as_slice())?;

            // Store metadata if present
            if let Some(metadata) = &entry.metadata {
                let mut meta_table = write_txn.open_table(METADATA_TABLE)?;
                let metadata_json = serde_json::to_string(metadata)
                    .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;
                meta_table.insert(id.as_str(), metadata_json.as_str())?;
            }
        }
        write_txn.commit()?;

        Ok(id)
    }

    /// Insert multiple vectors in a batch
    pub fn insert_batch(&self, entries: &[VectorEntry]) -> Result<Vec<VectorId>> {
        let write_txn = self.db().begin_write()?;
        let mut ids = Vec::with_capacity(entries.len());

        {
            let mut table = write_txn.open_table(VECTORS_TABLE)?;
            let mut meta_table = write_txn.open_table(METADATA_TABLE)?;

            for entry in entries {
                if entry.vector.len() != self.dimensions {
                    return Err(RuvectorError::DimensionMismatch {
                        expected: self.dimensions,
                        actual: entry.vector.len(),
                    });
                }

                let id = entry
                    .id
                    .clone()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                // Serialize and insert vector
                let vector_data = bincode::encode_to_vec(&entry.vector, config::standard())
                    .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;
                table.insert(id.as_str(), vector_data.as_slice())?;

                // Insert metadata if present
                if let Some(metadata) = &entry.metadata {
                    let metadata_json = serde_json::to_string(metadata)
                        .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;
                    meta_table.insert(id.as_str(), metadata_json.as_str())?;
                }

                ids.push(id);
            }
        }

        write_txn.commit()?;
        Ok(ids)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(VECTORS_TABLE)?;

        let Some(vector_data) = table.get(id)? else {
            return Ok(None);
        };

        let (vector, _): (Vec<f32>, usize) =
            bincode::decode_from_slice(vector_data.value(), config::standard())
                .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;

        // Try to get metadata
        let meta_table = read_txn.open_table(METADATA_TABLE)?;
        let metadata = if let Some(meta_data) = meta_table.get(id)? {
            let meta_str = meta_data.value();
            Some(
                serde_json::from_str(meta_str)
                    .map_err(|e| RuvectorError::SerializationError(e.to_string()))?,
            )
        } else {
            None
        };

        Ok(Some(VectorEntry {
            id: Some(id.to_string()),
            vector,
            metadata,
        }))
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        let write_txn = self.db().begin_write()?;
        let deleted;

        {
            let mut table = write_txn.open_table(VECTORS_TABLE)?;
            deleted = table.remove(id)?.is_some();

            let mut meta_table = write_txn.open_table(METADATA_TABLE)?;
            let _ = meta_table.remove(id)?;
        }

        write_txn.commit()?;
        Ok(deleted)
    }

    /// Get the number of vectors stored
    pub fn len(&self) -> Result<usize> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(VECTORS_TABLE)?;
        Ok(table.len()? as usize)
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Get all vector IDs
    pub fn all_ids(&self) -> Result<Vec<VectorId>> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(VECTORS_TABLE)?;

        let mut ids = Vec::new();
        let iter = table.iter()?;
        for item in iter {
            let (key, _) = item?;
            ids.push(key.value().to_string());
        }

        Ok(ids)
    }

    /// Save database configuration to persistent storage
    pub fn save_config(&self, options: &DbOptions) -> Result<()> {
        let config_json = serde_json::to_string(options)
            .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;

        let write_txn = self.db().begin_write()?;
        {
            let mut table = write_txn.open_table(CONFIG_TABLE)?;
            table.insert(DB_CONFIG_KEY, config_json.as_str())?;
        }
        write_txn.commit()?;

        Ok(())
    }

    /// Load database configuration from persistent storage
    pub fn load_config(&self) -> Result<Option<DbOptions>> {
        let read_txn = self.db().begin_read()?;

        // Try to open config table - may not exist in older databases
        let table = match read_txn.open_table(CONFIG_TABLE) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };

        let Some(config_data) = table.get(DB_CONFIG_KEY)? else {
            return Ok(None);
        };

        let config: DbOptions = serde_json::from_str(config_data.value())
            .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;

        Ok(Some(config))
    }

    /// Store an arbitrary provenance value alongside the vectors themselves.
    ///
    /// Lives in the same file as the vectors on purpose: provenance that a
    /// caller can delete without deleting the data it describes is not a
    /// safety property. See [`AgenticDB::with_embedding_provider`](crate::AgenticDB::with_embedding_provider).
    pub fn save_config_value(&self, key: &str, value: &str) -> Result<()> {
        let write_txn = self.db().begin_write()?;
        {
            let mut table = write_txn.open_table(CONFIG_TABLE)?;
            table.insert(key, value)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Read a value written by [`save_config_value`](Self::save_config_value).
    /// A database created before the key existed returns `None`.
    pub fn load_config_value(&self, key: &str) -> Result<Option<String>> {
        let read_txn = self.db().begin_read()?;
        let table = match read_txn.open_table(CONFIG_TABLE) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        Ok(table.get(key)?.map(|v| v.value().to_string()))
    }

    /// Get the stored dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl Drop for VectorStorage {
    fn drop(&mut self) {
        let Some(Pooled { db, slot }) = self.pooled.take() else {
            return;
        };

        {
            let mut guard = slot.lock();

            // Evicting under the guard is what keeps a concurrent `new` for this
            // path correct: `Weak::upgrade` starts failing the moment the strong
            // count hits zero, but redb only commits, fsyncs, and unlocks the
            // file later, inside `Database::drop`. Anyone racing us blocks on the
            // guard instead of calling `Database::create` against a held lock.
            if Arc::strong_count(&db) == 1 {
                *guard = None;
            }
            drop(db);
        }

        Self::release_slot_if_unused(&self.path, slot);
    }
}

// Add uuid dependency
use uuid;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_insert_and_get() -> Result<()> {
        let dir = tempdir().unwrap();
        let storage = VectorStorage::new(dir.path().join("test.db"), 3)?;

        let entry = VectorEntry {
            id: Some("test1".to_string()),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        };

        let id = storage.insert(&entry)?;
        assert_eq!(id, "test1");

        let retrieved = storage.get("test1")?;
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.vector, vec![1.0, 2.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_batch_insert() -> Result<()> {
        let dir = tempdir().unwrap();
        let storage = VectorStorage::new(dir.path().join("test.db"), 3)?;

        let entries = vec![
            VectorEntry {
                id: None,
                vector: vec![1.0, 2.0, 3.0],
                metadata: None,
            },
            VectorEntry {
                id: None,
                vector: vec![4.0, 5.0, 6.0],
                metadata: None,
            },
        ];

        let ids = storage.insert_batch(&entries)?;
        assert_eq!(ids.len(), 2);
        assert_eq!(storage.len()?, 2);

        Ok(())
    }

    #[test]
    fn test_delete() -> Result<()> {
        let dir = tempdir().unwrap();
        let storage = VectorStorage::new(dir.path().join("test.db"), 3)?;

        let entry = VectorEntry {
            id: Some("test1".to_string()),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        };

        storage.insert(&entry)?;
        assert_eq!(storage.len()?, 1);

        let deleted = storage.delete("test1")?;
        assert!(deleted);
        assert_eq!(storage.len()?, 0);

        Ok(())
    }

    #[test]
    fn test_multiple_instances_same_path() -> Result<()> {
        // This test verifies the fix for the database locking bug
        // Multiple VectorStorage instances should be able to share the same database file
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("shared.db");

        // Create first instance
        let storage1 = VectorStorage::new(&db_path, 3)?;

        // Insert data with first instance
        storage1.insert(&VectorEntry {
            id: Some("test1".to_string()),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        })?;

        // Create second instance with SAME path - this should NOT fail
        let storage2 = VectorStorage::new(&db_path, 3)?;

        // Both instances should see the same data
        assert_eq!(storage1.len()?, 1);
        assert_eq!(storage2.len()?, 1);

        // Insert with second instance
        storage2.insert(&VectorEntry {
            id: Some("test2".to_string()),
            vector: vec![4.0, 5.0, 6.0],
            metadata: None,
        })?;

        // Both instances should see both records
        assert_eq!(storage1.len()?, 2);
        assert_eq!(storage2.len()?, 2);

        // Verify data integrity
        let retrieved1 = storage1.get("test1")?;
        assert!(retrieved1.is_some());

        let retrieved2 = storage2.get("test2")?;
        assert!(retrieved2.is_some());

        Ok(())
    }

    #[test]
    fn different_spellings_of_one_database_share_the_pool() -> Result<()> {
        let dir = tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let db_path = dir.path().join("shared.db");
        let alias = dir.path().join("sub/../shared.db");

        let first = VectorStorage::new(&db_path, 3)?;
        let second = VectorStorage::new(&alias, 3)?;
        assert!(Arc::ptr_eq(
            &first.pooled.as_ref().unwrap().db,
            &second.pooled.as_ref().unwrap().db
        ));
        first.insert(&VectorEntry {
            id: Some("shared".to_string()),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        })?;
        assert!(second.get("shared")?.is_some());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_directory_shares_the_pool() -> Result<()> {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let alias = dir.path().join("alias");
        symlink(&real, &alias).unwrap();

        let first = VectorStorage::new(real.join("shared.db"), 3)?;
        let second = VectorStorage::new(alias.join("shared.db"), 3)?;
        assert!(Arc::ptr_eq(
            &first.pooled.as_ref().unwrap().db,
            &second.pooled.as_ref().unwrap().db
        ));
        first.insert(&VectorEntry {
            id: Some("shared".to_string()),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        })?;
        assert!(second.get("shared")?.is_some());
        Ok(())
    }

    #[test]
    fn relative_parent_traversal_is_still_rejected() {
        assert!(matches!(
            VectorStorage::new("../escape.db", 3),
            Err(RuvectorError::InvalidPath(message)) if message == "Path traversal attempt detected"
        ));
    }

    #[test]
    fn reopening_after_the_last_handle_drops_preserves_data() -> Result<()> {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("reopen.db");

        {
            let storage = VectorStorage::new(&db_path, 3)?;
            storage.insert(&VectorEntry {
                id: Some("kept".to_string()),
                vector: vec![1.0, 2.0, 3.0],
                metadata: None,
            })?;
        }

        // The pool entry must be reaped once the last handle is gone, otherwise
        // long-running processes accumulate one dead entry per opened path.
        assert!(
            !DB_POOL.lock().contains_key(&db_path),
            "pool kept a dead entry for {}",
            db_path.display()
        );

        let reopened = VectorStorage::new(&db_path, 3)?;
        assert_eq!(reopened.len()?, 1);
        assert_eq!(
            reopened.get("kept")?.expect("entry survived reopen").vector,
            vec![1.0, 2.0, 3.0]
        );

        Ok(())
    }

    #[test]
    fn concurrently_dropping_two_handles_still_reaps_the_pool_entry() {
        use std::sync::Barrier;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("double-drop.db");

        for _ in 0..50 {
            let first = VectorStorage::new(&db_path, 3).expect("first open");
            let second = VectorStorage::new(&db_path, 3).expect("second open");
            let barrier = Arc::new(Barrier::new(2));

            let threads: Vec<_> = [first, second]
                .into_iter()
                .map(|handle| {
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        drop(handle);
                    })
                })
                .collect();

            for thread in threads {
                thread.join().unwrap();
            }

            assert!(
                !DB_POOL.lock().contains_key(&db_path),
                "pool kept a dead entry after both handles dropped"
            );
        }
    }

    #[test]
    fn concurrent_drop_and_open_never_race_the_file_lock() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Barrier;

        // The unfixed pool failed this race on ~97.5% of attempts (195/200
        // when measured), so a handful of iterations already makes a
        // regression a near-certainty to catch: missing it 40 times running
        // is about 0.025^40. Each iteration is a real `Database::create` plus
        // redb's fsync-bearing drop, so this is kept small deliberately —
        // 200 iterations cost ~26s and would dominate the crate's suite.
        const ITERATIONS: usize = 40;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("race.db");
        let failures = Arc::new(AtomicUsize::new(0));
        let last_error = Arc::new(Mutex::new(None::<String>));

        for _ in 0..ITERATIONS {
            // The only live handle for this path; dropping it releases the
            // underlying redb file lock.
            let holder = VectorStorage::new(&db_path, 3).expect("initial open");
            let barrier = Arc::new(Barrier::new(2));

            let dropper = {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    drop(holder);
                })
            };

            let opener = {
                let barrier = Arc::clone(&barrier);
                let path = db_path.clone();
                let failures = Arc::clone(&failures);
                let last_error = Arc::clone(&last_error);
                std::thread::spawn(move || {
                    barrier.wait();
                    match VectorStorage::new(&path, 3) {
                        Ok(storage) => drop(storage),
                        Err(e) => {
                            failures.fetch_add(1, Ordering::Relaxed);
                            *last_error.lock() = Some(e.to_string());
                        }
                    }
                })
            };

            dropper.join().unwrap();
            opener.join().unwrap();
        }

        let failed = failures.load(Ordering::Relaxed);
        assert_eq!(
            failed,
            0,
            "{failed}/{ITERATIONS} concurrent opens raced the drop of the last \
             handle; last error: {:?}",
            last_error.lock().as_deref()
        );
    }
}
