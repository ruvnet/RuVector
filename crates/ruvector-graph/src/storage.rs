//! Persistent storage layer with redb and memory-mapped vectors
//!
//! Provides ACID-compliant storage for graph nodes, edges, and hyperedges

#[cfg(feature = "storage")]
use crate::edge::Edge;
#[cfg(feature = "storage")]
use crate::hyperedge::{Hyperedge, HyperedgeId};
#[cfg(feature = "storage")]
use crate::node::Node;
#[cfg(feature = "storage")]
use crate::types::{EdgeId, NodeId};
#[cfg(feature = "storage")]
use anyhow::Result;
#[cfg(feature = "storage")]
use bincode::config;
#[cfg(feature = "storage")]
use once_cell::sync::Lazy;
#[cfg(feature = "storage")]
use parking_lot::Mutex;
#[cfg(feature = "storage")]
use redb::{Database, ReadableTable, TableDefinition};
#[cfg(feature = "storage")]
use std::collections::HashMap;
#[cfg(feature = "storage")]
use std::path::{Path, PathBuf};
#[cfg(feature = "storage")]
use std::sync::{Arc, Weak};

#[cfg(feature = "storage")]
// Table definitions
const NODES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("nodes");
#[cfg(feature = "storage")]
const EDGES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("edges");
#[cfg(feature = "storage")]
const HYPEREDGES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("hyperedges");
#[cfg(feature = "storage")]
const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");

#[cfg(feature = "storage")]
/// Per-path guard around the pooled database handle.
///
/// The `Weak` (rather than a strong `Arc`) is what stops an erased-and-recreated
/// path from handing out a `Database` that still points at the unlinked inode —
/// the strong-`Arc` shape this pool had before #907 kept "erased" rows alive
/// across delete-and-recreate. Opening and closing a database both happen while
/// this guard is held, so a close (redb's final write transaction, fsync, and
/// file-lock release, all of which run in `Database::drop`) is never observed
/// half-finished by a concurrent open. The same shape lives in
/// `ruvector-core::storage`, which is the reference implementation.
type PathSlot = Mutex<Option<Weak<Database>>>;

#[cfg(feature = "storage")]
// Global database connection pool to allow multiple GraphStorage instances
// to share the same underlying database file. The global lock only guards the
// map itself; the slow work happens under the per-path slot lock so an fsync on
// one path never blocks opens of unrelated paths.
static DB_POOL: Lazy<Mutex<HashMap<PathBuf, Arc<PathSlot>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "storage")]
/// The pooled database and the guard that owns its lifecycle.
struct Pooled {
    db: Arc<Database>,
    slot: Arc<PathSlot>,
}

#[cfg(feature = "storage")]
/// Storage backend for graph database
pub struct GraphStorage {
    /// Always `Some` for a live handle; only `Drop` takes it.
    pooled: Option<Pooled>,
    path: PathBuf,
}

#[cfg(feature = "storage")]
impl GraphStorage {
    /// Create or open a graph storage at the given path
    ///
    /// Uses a global connection pool to allow multiple GraphStorage
    /// instances to share the same underlying database file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();

        // Create parent directories if they don't exist
        if let Some(parent) = path_ref.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Convert to absolute path
        let path_buf = if path_ref.is_absolute() {
            path_ref.to_path_buf()
        } else {
            std::env::current_dir()?.join(path_ref)
        };

        // SECURITY: Check for path traversal attempts
        let path_str = path_ref.to_string_lossy();
        if path_str.contains("..") && !path_ref.is_absolute() {
            if let Ok(cwd) = std::env::current_dir() {
                let mut normalized = cwd.clone();
                for component in path_ref.components() {
                    match component {
                        std::path::Component::ParentDir => {
                            if !normalized.pop() || !normalized.starts_with(&cwd) {
                                anyhow::bail!("Path traversal attempt detected");
                            }
                        }
                        std::path::Component::Normal(c) => normalized.push(c),
                        _ => {}
                    }
                }
            }
        }

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
            let _ = write_txn.open_table(NODES_TABLE)?;
            let _ = write_txn.open_table(EDGES_TABLE)?;
            let _ = write_txn.open_table(HYPEREDGES_TABLE)?;
            let _ = write_txn.open_table(METADATA_TABLE)?;
        }
        write_txn.commit()?;

        *guard = Some(Arc::downgrade(&new_db));
        Ok(new_db)
    }

    /// Drop this path's pool entry once nothing can reach it any more.
    ///
    /// Leaving the entry in place would leak one map entry per opened path —
    /// the unbounded-growth half of #907. `slot` is consumed and released
    /// *under* the pool lock, so a remaining strong count of one proves the
    /// pool's own entry is the last handle: new handles are only ever cloned
    /// under this same lock, so no other thread can be mid-`new` for this path.
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

    // Node operations

    /// Insert a node
    pub fn insert_node(&self, node: &Node) -> Result<NodeId> {
        let write_txn = self.db().begin_write()?;
        {
            let mut table = write_txn.open_table(NODES_TABLE)?;

            // Serialize node data
            let node_data = bincode::encode_to_vec(node, config::standard())?;
            table.insert(node.id.as_str(), node_data.as_slice())?;
        }
        write_txn.commit()?;

        Ok(node.id.clone())
    }

    /// Insert multiple nodes in a batch
    pub fn insert_nodes_batch(&self, nodes: &[Node]) -> Result<Vec<NodeId>> {
        let write_txn = self.db().begin_write()?;
        let mut ids = Vec::with_capacity(nodes.len());

        {
            let mut table = write_txn.open_table(NODES_TABLE)?;

            for node in nodes {
                let node_data = bincode::encode_to_vec(node, config::standard())?;
                table.insert(node.id.as_str(), node_data.as_slice())?;
                ids.push(node.id.clone());
            }
        }

        write_txn.commit()?;
        Ok(ids)
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &str) -> Result<Option<Node>> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(NODES_TABLE)?;

        let Some(node_data) = table.get(id)? else {
            return Ok(None);
        };

        let (node, _): (Node, usize) =
            bincode::decode_from_slice(node_data.value(), config::standard())?;
        Ok(Some(node))
    }

    /// Delete a node by ID
    pub fn delete_node(&self, id: &str) -> Result<bool> {
        let write_txn = self.db().begin_write()?;
        let deleted;
        {
            let mut table = write_txn.open_table(NODES_TABLE)?;
            let result = table.remove(id)?;
            deleted = result.is_some();
        }
        write_txn.commit()?;
        Ok(deleted)
    }

    /// Get all node IDs
    pub fn all_node_ids(&self) -> Result<Vec<NodeId>> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(NODES_TABLE)?;

        let mut ids = Vec::new();
        let iter = table.iter()?;
        for item in iter {
            let (key, _) = item?;
            ids.push(key.value().to_string());
        }

        Ok(ids)
    }

    // Edge operations

    /// Insert an edge
    pub fn insert_edge(&self, edge: &Edge) -> Result<EdgeId> {
        let write_txn = self.db().begin_write()?;
        {
            let mut table = write_txn.open_table(EDGES_TABLE)?;

            // Serialize edge data
            let edge_data = bincode::encode_to_vec(edge, config::standard())?;
            table.insert(edge.id.as_str(), edge_data.as_slice())?;
        }
        write_txn.commit()?;

        Ok(edge.id.clone())
    }

    /// Insert multiple edges in a batch
    pub fn insert_edges_batch(&self, edges: &[Edge]) -> Result<Vec<EdgeId>> {
        let write_txn = self.db().begin_write()?;
        let mut ids = Vec::with_capacity(edges.len());

        {
            let mut table = write_txn.open_table(EDGES_TABLE)?;

            for edge in edges {
                let edge_data = bincode::encode_to_vec(edge, config::standard())?;
                table.insert(edge.id.as_str(), edge_data.as_slice())?;
                ids.push(edge.id.clone());
            }
        }

        write_txn.commit()?;
        Ok(ids)
    }

    /// Get an edge by ID
    pub fn get_edge(&self, id: &str) -> Result<Option<Edge>> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(EDGES_TABLE)?;

        let Some(edge_data) = table.get(id)? else {
            return Ok(None);
        };

        let (edge, _): (Edge, usize) =
            bincode::decode_from_slice(edge_data.value(), config::standard())?;
        Ok(Some(edge))
    }

    /// Delete an edge by ID
    pub fn delete_edge(&self, id: &str) -> Result<bool> {
        let write_txn = self.db().begin_write()?;
        let deleted;
        {
            let mut table = write_txn.open_table(EDGES_TABLE)?;
            let result = table.remove(id)?;
            deleted = result.is_some();
        }
        write_txn.commit()?;
        Ok(deleted)
    }

    pub fn delete_edges_batch(&self, ids: &[impl AsRef<str>]) -> Result<usize> {
        let write_txn = self.db().begin_write()?;
        let mut deleted = 0;
        {
            let mut table = write_txn.open_table(EDGES_TABLE)?;
            for id in ids {
                if table.remove(id.as_ref())?.is_some() {
                    deleted += 1;
                }
            }
        }

        write_txn.commit()?;
        Ok(deleted)
    }

    /// Get all edge IDs
    pub fn all_edge_ids(&self) -> Result<Vec<EdgeId>> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(EDGES_TABLE)?;

        let mut ids = Vec::new();
        let iter = table.iter()?;
        for item in iter {
            let (key, _) = item?;
            ids.push(key.value().to_string());
        }

        Ok(ids)
    }

    // Hyperedge operations

    /// Insert a hyperedge
    pub fn insert_hyperedge(&self, hyperedge: &Hyperedge) -> Result<HyperedgeId> {
        let write_txn = self.db().begin_write()?;
        {
            let mut table = write_txn.open_table(HYPEREDGES_TABLE)?;

            // Serialize hyperedge data
            let hyperedge_data = bincode::encode_to_vec(hyperedge, config::standard())?;
            table.insert(hyperedge.id.as_str(), hyperedge_data.as_slice())?;
        }
        write_txn.commit()?;

        Ok(hyperedge.id.clone())
    }

    /// Insert multiple hyperedges in a batch
    pub fn insert_hyperedges_batch(&self, hyperedges: &[Hyperedge]) -> Result<Vec<HyperedgeId>> {
        let write_txn = self.db().begin_write()?;
        let mut ids = Vec::with_capacity(hyperedges.len());

        {
            let mut table = write_txn.open_table(HYPEREDGES_TABLE)?;

            for hyperedge in hyperedges {
                let hyperedge_data = bincode::encode_to_vec(hyperedge, config::standard())?;
                table.insert(hyperedge.id.as_str(), hyperedge_data.as_slice())?;
                ids.push(hyperedge.id.clone());
            }
        }

        write_txn.commit()?;
        Ok(ids)
    }

    /// Get a hyperedge by ID
    pub fn get_hyperedge(&self, id: &str) -> Result<Option<Hyperedge>> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(HYPEREDGES_TABLE)?;

        let Some(hyperedge_data) = table.get(id)? else {
            return Ok(None);
        };

        let (hyperedge, _): (Hyperedge, usize) =
            bincode::decode_from_slice(hyperedge_data.value(), config::standard())?;
        Ok(Some(hyperedge))
    }

    /// Delete a hyperedge by ID
    pub fn delete_hyperedge(&self, id: &str) -> Result<bool> {
        let write_txn = self.db().begin_write()?;
        let deleted;
        {
            let mut table = write_txn.open_table(HYPEREDGES_TABLE)?;
            let result = table.remove(id)?;
            deleted = result.is_some();
        }
        write_txn.commit()?;
        Ok(deleted)
    }

    /// Get all hyperedge IDs
    pub fn all_hyperedge_ids(&self) -> Result<Vec<HyperedgeId>> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(HYPEREDGES_TABLE)?;

        let mut ids = Vec::new();
        let iter = table.iter()?;
        for item in iter {
            let (key, _) = item?;
            ids.push(key.value().to_string());
        }

        Ok(ids)
    }

    // Metadata operations

    /// Set metadata
    pub fn set_metadata(&self, key: &str, value: &str) -> Result<()> {
        let write_txn = self.db().begin_write()?;
        {
            let mut table = write_txn.open_table(METADATA_TABLE)?;
            table.insert(key, value)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Result<Option<String>> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(METADATA_TABLE)?;

        let value = table.get(key)?.map(|v| v.value().to_string());
        Ok(value)
    }

    // Statistics

    /// Get the number of nodes
    pub fn node_count(&self) -> Result<usize> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(NODES_TABLE)?;
        Ok(table.iter()?.count())
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> Result<usize> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(EDGES_TABLE)?;
        Ok(table.iter()?.count())
    }

    /// Get the number of hyperedges
    pub fn hyperedge_count(&self) -> Result<usize> {
        let read_txn = self.db().begin_read()?;
        let table = read_txn.open_table(HYPEREDGES_TABLE)?;
        Ok(table.iter()?.count())
    }
}

#[cfg(feature = "storage")]
impl Drop for GraphStorage {
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
            // A naive Arc→Weak swap without this ordering fails ~97% of
            // concurrent drop-vs-open probes with "Database already open" (#907).
            if Arc::strong_count(&db) == 1 {
                *guard = None;
            }
            drop(db);
        }

        Self::release_slot_if_unused(&self.path, slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeBuilder;
    use crate::hyperedge::HyperedgeBuilder;
    use crate::node::NodeBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_node_storage() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::new(dir.path().join("test.db"))?;

        let node = NodeBuilder::new()
            .label("Person")
            .property("name", "Alice")
            .build();

        let id = storage.insert_node(&node)?;
        assert_eq!(id, node.id);

        let retrieved = storage.get_node(&id)?;
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, node.id);
        assert!(retrieved.has_label("Person"));

        Ok(())
    }

    #[test]
    fn test_edge_storage() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::new(dir.path().join("test.db"))?;

        let edge = EdgeBuilder::new("n1".to_string(), "n2".to_string(), "KNOWS")
            .property("since", 2020i64)
            .build();

        let id = storage.insert_edge(&edge)?;
        assert_eq!(id, edge.id);

        let retrieved = storage.get_edge(&id)?;
        assert!(retrieved.is_some());

        Ok(())
    }

    #[test]
    fn test_batch_insert() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::new(dir.path().join("test.db"))?;

        let nodes = vec![
            NodeBuilder::new().label("Person").build(),
            NodeBuilder::new().label("Person").build(),
        ];

        let ids = storage.insert_nodes_batch(&nodes)?;
        assert_eq!(ids.len(), 2);
        assert_eq!(storage.node_count()?, 2);

        Ok(())
    }

    #[test]
    fn erased_path_recreation_never_reuses_the_unlinked_database() -> Result<()> {
        // Regression test for #907: the strong-Arc pool kept a Database alive
        // after its file was deleted, so a recreated path was handed the old
        // handle pointing at the unlinked inode — "erased" rows reappeared.
        let dir = tempdir()?;
        let db_path = dir.path().join("erase.db");

        {
            let storage = GraphStorage::new(&db_path)?;
            let node = NodeBuilder::new().label("Erased").build();
            storage.insert_node(&node)?;
            assert_eq!(storage.node_count()?, 1);
        }

        std::fs::remove_file(&db_path)?;

        let recreated = GraphStorage::new(&db_path)?;
        assert_eq!(
            recreated.node_count()?,
            0,
            "recreated path served rows from the unlinked database"
        );
        Ok(())
    }

    #[test]
    fn reopening_after_the_last_handle_drops_preserves_data_and_reaps_the_pool() -> Result<()> {
        let dir = tempdir()?;
        let db_path = dir.path().join("reopen.db");

        {
            let storage = GraphStorage::new(&db_path)?;
            let node = NodeBuilder::new().label("Kept").build();
            storage.insert_node(&node)?;
        }

        // The pool entry must be reaped once the last handle is gone,
        // otherwise long-running processes accumulate one entry per opened
        // path — the unbounded-growth half of #907.
        assert!(
            !DB_POOL.lock().contains_key(&db_path),
            "pool kept a dead entry for {}",
            db_path.display()
        );

        let reopened = GraphStorage::new(&db_path)?;
        assert_eq!(reopened.node_count()?, 1);
        Ok(())
    }

    #[test]
    fn concurrent_drop_and_open_never_race_the_file_lock() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Barrier;

        // A naive Arc→Weak swap without Drop-under-the-slot-guard eviction
        // fails this probe ~97% of the time with "Database already open"
        // (measured on the equivalent ruvector-core pool, #902/#907), so a
        // few dozen iterations make a regression a near-certainty to catch
        // while keeping the suite fast.
        const ITERATIONS: usize = 40;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("race.db");
        let failures = Arc::new(AtomicUsize::new(0));
        let last_error = Arc::new(Mutex::new(None::<String>));

        for _ in 0..ITERATIONS {
            let holder = GraphStorage::new(&db_path).expect("initial open");
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
                    match GraphStorage::new(&path) {
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

    #[test]
    fn test_hyperedge_storage() -> Result<()> {
        let dir = tempdir()?;
        let storage = GraphStorage::new(dir.path().join("test.db"))?;

        let hyperedge = HyperedgeBuilder::new(
            vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
            "MEETING",
        )
        .description("Team meeting")
        .build();

        let id = storage.insert_hyperedge(&hyperedge)?;
        assert_eq!(id, hyperedge.id);

        let retrieved = storage.get_hyperedge(&id)?;
        assert!(retrieved.is_some());

        Ok(())
    }
}
