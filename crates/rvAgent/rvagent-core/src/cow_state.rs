//! Copy-on-Write State Backend (ADR-103 B2).
//!
//! Provides efficient state forking for subagent spawning via copy-on-write semantics.
//! Fork operations are O(1) via Arc cloning, with mutations triggering lazy copies.

use parking_lot::Mutex;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::error::{Result, RvAgentError};

/// Copy-on-Write state backend for efficient forking.
///
/// # Architecture
///
/// - Each `CowStateBackend` has a unique `branch_id` for tracking lineage.
/// - `data` is wrapped in `RefCell<Arc<StateData>>` for true COW semantics.
/// - `parent` is an optional reference to the parent branch, forming a parent-chain.
/// - On `get`, we check local data first, then walk up the parent chain.
/// - On `set`, we clone the Arc if strong_count > 1 (true COW).
/// - `fork_for_subagent` creates a new branch with O(1) cost.
/// - `merge_from` allows merging changes from a child branch back into the parent.
///
/// # Example
///
/// ```
/// use rvagent_core::cow_state::CowStateBackend;
///
/// let parent = CowStateBackend::new();
/// parent.set("key1", b"parent_value".to_vec()).unwrap();
///
/// let child = parent.fork_for_subagent();
/// child.set("key2", b"child_value".to_vec()).unwrap();
///
/// // Parent sees key1, child sees both.
/// assert_eq!(parent.get("key1"), Some(b"parent_value".to_vec()));
/// assert!(parent.get("key2").is_none());
/// assert_eq!(child.get("key1"), Some(b"parent_value".to_vec()));
/// assert_eq!(child.get("key2"), Some(b"child_value".to_vec()));
///
/// // Merge child changes back to parent.
/// parent.merge_from(&child).unwrap();
/// assert_eq!(parent.get("key2"), Some(b"child_value".to_vec()));
/// ```
pub struct CowStateBackend {
    /// Current branch data (RefCell allows interior mutability for COW).
    data: RefCell<Arc<StateData>>,
    /// Modified keys (thread-safe tracking).
    modified: Arc<Mutex<Vec<String>>>,
    /// Branch ID for tracking lineage.
    branch_id: u32,
    /// Parent branch (for COW semantics).
    parent: Option<Arc<CowStateBackend>>,
}

/// Internal state data for a COW branch (immutable once shared).
#[derive(Debug, Clone, Default)]
struct StateData {
    /// Key-value store.
    values: HashMap<String, Vec<u8>>,
    /// Tombstones for deleted keys (to override parent values).
    deleted_keys: HashSet<String>,
    /// Version counter incremented on each mutation.
    version: u64,
}

impl CowStateBackend {
    /// Create a new root-level COW state backend.
    pub fn new() -> Self {
        Self {
            data: RefCell::new(Arc::new(StateData::default())),
            modified: Arc::new(Mutex::new(Vec::new())),
            branch_id: 0,
            parent: None,
        }
    }

    /// Fork a new subagent branch with O(1) cost.
    ///
    /// The child branch shares the parent's data via Arc, but writes are
    /// local to the child. Walking the parent chain on reads allows
    /// inheriting parent state without copying.
    pub fn fork_for_subagent(&self) -> Self {
        Self {
            data: RefCell::new(Arc::new(StateData::default())),
            modified: Arc::new(Mutex::new(Vec::new())),
            branch_id: self.branch_id + 1,
            parent: Some(Arc::new(Self {
                data: RefCell::new(Arc::clone(&self.data.borrow())),
                modified: Arc::clone(&self.modified),
                branch_id: self.branch_id,
                parent: self.parent.clone(),
            })),
        }
    }

    /// Get a value by key, walking the parent chain if not found locally.
    ///
    /// # Returns
    ///
    /// `Some(value)` if found in this branch or any ancestor, `None` otherwise.
    /// Returns `None` if the key was explicitly deleted in this branch (tombstone).
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        // Check local data first.
        {
            let data = self.data.borrow();

            // Check if key was explicitly deleted (tombstone).
            if data.deleted_keys.contains(key) {
                return None;
            }

            if let Some(val) = data.values.get(key) {
                return Some(val.clone());
            }
        }

        // Walk parent chain.
        let mut current_parent = self.parent.as_ref();
        while let Some(parent_arc) = current_parent {
            let parent_data = parent_arc.data.borrow();

            // Check for tombstone in parent.
            if parent_data.deleted_keys.contains(key) {
                return None;
            }

            if let Some(val) = parent_data.values.get(key) {
                return Some(val.clone());
            }
            current_parent = parent_arc.parent.as_ref();
        }

        None
    }

    /// Make data mutable (COW: clone if Arc is shared).
    fn make_mut(&self) -> std::cell::RefMut<'_, Arc<StateData>> {
        let mut data_ref = self.data.borrow_mut();

        // If Arc is shared (strong_count > 1), clone to make independent.
        if Arc::strong_count(&*data_ref) > 1 {
            *data_ref = Arc::new((**data_ref).clone());
        }

        data_ref
    }

    /// Set a value in the current branch (copy-on-write).
    ///
    /// This mutation is local to the current branch and does not affect
    /// the parent or any siblings.
    pub fn set(&self, key: impl Into<String>, value: Vec<u8>) -> Result<()> {
        let key = key.into();

        // Trigger COW if needed and get mutable reference.
        let mut data_ref = self.make_mut();
        let data = Arc::make_mut(&mut *data_ref);

        // Remove tombstone if key was previously deleted.
        data.deleted_keys.remove(&key);

        // Track modification for merge operations.
        {
            let mut modified = self.modified.lock();
            if !modified.contains(&key) {
                modified.push(key.clone());
            }
        }

        data.values.insert(key, value);
        data.version += 1;

        Ok(())
    }

    /// Delete a key from the current branch.
    ///
    /// For COW semantics: we add a tombstone to mark the key as deleted,
    /// which overrides any parent values.
    ///
    /// # Returns
    ///
    /// `true` if the key existed (locally or in parent chain) and was deleted, `false` otherwise.
    pub fn delete(&self, key: &str) -> bool {
        // Check if key exists anywhere (local or parent chain).
        let existed = self.get(key).is_some();

        if existed {
            let mut data_ref = self.make_mut();
            let data = Arc::make_mut(&mut *data_ref);

            // Remove from local values if present.
            data.values.remove(key);

            // Add tombstone to mark as deleted.
            data.deleted_keys.insert(key.to_string());

            // Mark as modified/deleted for merge operations.
            let mut modified = self.modified.lock();
            modified.push(key.to_string());

            data.version += 1;
        }

        existed
    }

    /// Snapshot the current state (O(1) via Arc clone).
    ///
    /// The snapshot shares data with the current backend until either is mutated,
    /// at which point copy-on-write triggers.
    pub fn snapshot(&self) -> Self {
        Self {
            data: RefCell::new(Arc::clone(&self.data.borrow())),
            modified: Arc::new(Mutex::new(Vec::new())),
            branch_id: self.branch_id,
            parent: self.parent.clone(),
        }
    }

    /// Merge changes from a child branch back into this branch.
    ///
    /// # Strategy
    ///
    /// - For each key modified in the child, copy its value to the parent.
    /// - If a key exists in both parent and child (conflict), child wins.
    /// - Keys deleted in the child are deleted in the parent.
    ///
    /// # Errors
    ///
    /// Returns `MergeConflict` if the child is not a direct descendant of this branch.
    pub fn merge_from(&self, child: &Self) -> Result<()> {
        // Verify lineage: child must be a direct descendant.
        if !self.is_ancestor_of(child) {
            return Err(RvAgentError::state(
                "Cannot merge from non-descendant branch",
            ));
        }

        let child_data = child.data.borrow();
        let child_modified = child.modified.lock();

        let mut parent_data_ref = self.make_mut();
        let parent_data = Arc::make_mut(&mut *parent_data_ref);

        for key in child_modified.iter() {
            // Check if key was deleted in child (tombstone).
            if child_data.deleted_keys.contains(key) {
                // Propagate deletion to parent.
                parent_data.values.remove(key);
                parent_data.deleted_keys.insert(key.clone());

                let mut parent_modified = self.modified.lock();
                if !parent_modified.contains(key) {
                    parent_modified.push(key.clone());
                }
            } else if let Some(val) = child_data.values.get(key) {
                // Key was set in child - propagate value to parent.
                parent_data.values.insert(key.clone(), val.clone());
                parent_data.deleted_keys.remove(key);

                let mut parent_modified = self.modified.lock();
                if !parent_modified.contains(key) {
                    parent_modified.push(key.clone());
                }
            }
        }

        parent_data.version += 1;

        Ok(())
    }

    /// Check if this branch is an ancestor of the given branch.
    fn is_ancestor_of(&self, other: &Self) -> bool {
        let mut current_parent = other.parent.as_ref();
        while let Some(parent_arc) = current_parent {
            if self.branch_id == parent_arc.branch_id {
                return true;
            }
            current_parent = parent_arc.parent.as_ref();
        }
        false
    }

    /// Return the branch ID of this backend.
    pub fn branch_id(&self) -> u32 {
        self.branch_id
    }

    /// Return the current version number (incremented on each mutation).
    pub fn version(&self) -> u64 {
        self.data.borrow().version
    }

    /// Return the number of keys stored locally (excluding parent chain).
    pub fn local_key_count(&self) -> usize {
        self.data.borrow().values.len()
    }

    /// Return all keys stored locally (excluding parent chain).
    pub fn local_keys(&self) -> Vec<String> {
        self.data.borrow().values.keys().cloned().collect()
    }

    /// Return all keys modified in this branch.
    pub fn modified_keys(&self) -> Vec<String> {
        self.modified.lock().clone()
    }

    /// Clear all local data (does not affect parent chain).
    pub fn clear(&self) {
        let mut data_ref = self.make_mut();
        let data = Arc::make_mut(&mut *data_ref);
        data.values.clear();
        data.deleted_keys.clear();
        data.version += 1;

        let mut modified = self.modified.lock();
        modified.clear();
    }
}

impl Default for CowStateBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CowStateBackend {
    fn clone(&self) -> Self {
        Self {
            data: RefCell::new(Arc::clone(&self.data.borrow())),
            modified: Arc::clone(&self.modified),
            branch_id: self.branch_id,
            parent: self.parent.clone(),
        }
    }
}

impl std::fmt::Debug for CowStateBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.borrow();
        let modified = self.modified.lock();
        f.debug_struct("CowStateBackend")
            .field("branch_id", &self.branch_id)
            .field("version", &data.version)
            .field("local_keys", &data.values.len())
            .field("modified_keys", &modified.len())
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_backend_is_empty() {
        let backend = CowStateBackend::new();
        assert_eq!(backend.branch_id(), 0);
        assert_eq!(backend.version(), 0);
        assert_eq!(backend.local_key_count(), 0);
        assert!(backend.get("any_key").is_none());
    }

    #[test]
    fn test_set_and_get() {
        let backend = CowStateBackend::new();
        backend.set("key1", b"value1".to_vec()).unwrap();

        assert_eq!(backend.get("key1"), Some(b"value1".to_vec()));
        assert_eq!(backend.local_key_count(), 1);
        assert_eq!(backend.version(), 1);
    }

    #[test]
    fn test_set_multiple_keys() {
        let backend = CowStateBackend::new();
        backend.set("a", b"1".to_vec()).unwrap();
        backend.set("b", b"2".to_vec()).unwrap();
        backend.set("c", b"3".to_vec()).unwrap();

        assert_eq!(backend.local_key_count(), 3);
        assert_eq!(backend.get("a"), Some(b"1".to_vec()));
        assert_eq!(backend.get("b"), Some(b"2".to_vec()));
        assert_eq!(backend.get("c"), Some(b"3".to_vec()));
    }

    #[test]
    fn test_overwrite_key() {
        let backend = CowStateBackend::new();
        backend.set("key", b"v1".to_vec()).unwrap();
        assert_eq!(backend.version(), 1);

        backend.set("key", b"v2".to_vec()).unwrap();
        assert_eq!(backend.get("key"), Some(b"v2".to_vec()));
        assert_eq!(backend.version(), 2);
        assert_eq!(backend.local_key_count(), 1);
    }

    #[test]
    fn test_delete_key() {
        let backend = CowStateBackend::new();
        backend.set("key", b"val".to_vec()).unwrap();

        let existed = backend.delete("key");
        assert!(existed);
        assert!(backend.get("key").is_none());
        assert_eq!(backend.local_key_count(), 0);
    }

    #[test]
    fn test_delete_nonexistent_key() {
        let backend = CowStateBackend::new();
        let existed = backend.delete("nonexistent");
        assert!(!existed);
    }

    #[test]
    fn test_fork_for_subagent() {
        let parent = CowStateBackend::new();
        parent.set("p_key", b"p_val".to_vec()).unwrap();

        let child = parent.fork_for_subagent();

        assert_eq!(child.branch_id(), 1);
        assert!(child.parent.is_some());
        assert_eq!(child.local_key_count(), 0); // Child has no local data yet.

        // Child inherits parent keys.
        assert_eq!(child.get("p_key"), Some(b"p_val".to_vec()));
    }

    #[test]
    fn test_child_writes_do_not_affect_parent() {
        let parent = CowStateBackend::new();
        parent.set("shared", b"parent_version".to_vec()).unwrap();

        let child = parent.fork_for_subagent();
        child.set("shared", b"child_version".to_vec()).unwrap();
        child.set("child_only", b"child_data".to_vec()).unwrap();

        // Parent still sees original value for "shared".
        assert_eq!(parent.get("shared"), Some(b"parent_version".to_vec()));
        assert!(parent.get("child_only").is_none());

        // Child sees its own version.
        assert_eq!(child.get("shared"), Some(b"child_version".to_vec()));
        assert_eq!(child.get("child_only"), Some(b"child_data".to_vec()));
    }

    #[test]
    fn test_snapshot_is_o1() {
        let backend = CowStateBackend::new();
        backend.set("key", b"val".to_vec()).unwrap();

        let snapshot = backend.snapshot();

        // Snapshot shares the same Arc.
        assert!(Arc::ptr_eq(
            &*backend.data.borrow(),
            &*snapshot.data.borrow()
        ));
        assert_eq!(snapshot.branch_id(), backend.branch_id());
        assert_eq!(snapshot.version(), backend.version());
        assert_eq!(snapshot.get("key"), Some(b"val".to_vec()));
    }

    #[test]
    fn test_snapshot_cow_on_mutation() {
        let backend = CowStateBackend::new();
        backend.set("original", b"v1".to_vec()).unwrap();

        let snapshot = backend.snapshot();
        // Snapshot shares the same Arc initially.
        assert!(Arc::ptr_eq(
            &*backend.data.borrow(),
            &*snapshot.data.borrow()
        ));

        // Mutate backend triggers copy-on-write.
        backend.set("original", b"v2".to_vec()).unwrap();

        // Now they have different data Arcs.
        assert!(!Arc::ptr_eq(
            &*backend.data.borrow(),
            &*snapshot.data.borrow()
        ));

        // Snapshot still has old value.
        assert_eq!(snapshot.get("original"), Some(b"v1".to_vec()));
        assert_eq!(backend.get("original"), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_merge_from_child() {
        let parent = CowStateBackend::new();
        parent.set("p1", b"parent_val1".to_vec()).unwrap();

        let child = parent.fork_for_subagent();
        child.set("c1", b"child_val1".to_vec()).unwrap();
        child.set("c2", b"child_val2".to_vec()).unwrap();

        parent.merge_from(&child).unwrap();

        // Parent now has child keys.
        assert_eq!(parent.get("c1"), Some(b"child_val1".to_vec()));
        assert_eq!(parent.get("c2"), Some(b"child_val2".to_vec()));
        assert_eq!(parent.get("p1"), Some(b"parent_val1".to_vec()));
    }

    #[test]
    fn test_merge_conflict_child_wins() {
        let parent = CowStateBackend::new();
        parent.set("shared", b"parent_version".to_vec()).unwrap();

        let child = parent.fork_for_subagent();
        child.set("shared", b"child_version".to_vec()).unwrap();

        parent.merge_from(&child).unwrap();

        // Child version wins.
        assert_eq!(parent.get("shared"), Some(b"child_version".to_vec()));
    }

    #[test]
    fn test_merge_from_non_descendant_fails() {
        let parent = CowStateBackend::new();
        let unrelated = CowStateBackend::new();

        let result = parent.merge_from(&unrelated);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-descendant"));
    }

    #[test]
    fn test_merge_deleted_key() {
        let parent = CowStateBackend::new();
        parent.set("to_delete", b"value".to_vec()).unwrap();

        let child = parent.fork_for_subagent();
        child.delete("to_delete");

        parent.merge_from(&child).unwrap();

        // Key should be deleted in parent.
        assert!(parent.get("to_delete").is_none());
    }

    #[test]
    fn test_modified_keys_tracking() {
        let backend = CowStateBackend::new();
        backend.set("a", b"1".to_vec()).unwrap();
        backend.set("b", b"2".to_vec()).unwrap();
        backend.set("a", b"1_updated".to_vec()).unwrap();

        let modified = backend.modified_keys();
        assert!(modified.contains(&"a".to_string()));
        assert!(modified.contains(&"b".to_string()));
        // Each key appears only once (we check before adding).
        assert_eq!(modified.len(), 2);
        assert_eq!(modified.iter().filter(|k| *k == "a").count(), 1);
    }

    #[test]
    fn test_clear() {
        let backend = CowStateBackend::new();
        backend.set("k1", b"v1".to_vec()).unwrap();
        backend.set("k2", b"v2".to_vec()).unwrap();

        backend.clear();

        assert_eq!(backend.local_key_count(), 0);
        assert!(backend.get("k1").is_none());
        assert!(backend.get("k2").is_none());
    }

    #[test]
    fn test_local_keys() {
        let backend = CowStateBackend::new();
        backend.set("a", b"1".to_vec()).unwrap();
        backend.set("b", b"2".to_vec()).unwrap();

        let mut keys = backend.local_keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_parent_chain_walk() {
        let root = CowStateBackend::new();
        root.set("root_key", b"root_val".to_vec()).unwrap();

        let child1 = root.fork_for_subagent();
        child1.set("child1_key", b"child1_val".to_vec()).unwrap();

        let child2 = child1.fork_for_subagent();
        child2.set("child2_key", b"child2_val".to_vec()).unwrap();

        // child2 should see all three keys by walking the chain.
        assert_eq!(child2.get("root_key"), Some(b"root_val".to_vec()));
        assert_eq!(child2.get("child1_key"), Some(b"child1_val".to_vec()));
        assert_eq!(child2.get("child2_key"), Some(b"child2_val".to_vec()));

        // child1 should not see child2's key.
        assert!(child1.get("child2_key").is_none());

        // root should not see any child keys.
        assert!(root.get("child1_key").is_none());
        assert!(root.get("child2_key").is_none());
    }

    #[test]
    fn test_debug_format() {
        let backend = CowStateBackend::new();
        backend.set("key", b"val".to_vec()).unwrap();

        let debug = format!("{:?}", backend);
        assert!(debug.contains("CowStateBackend"));
        assert!(debug.contains("branch_id"));
        assert!(debug.contains("version"));
    }

    #[test]
    fn test_multiple_forks() {
        let parent = CowStateBackend::new();
        parent.set("shared", b"parent".to_vec()).unwrap();

        let child_a = parent.fork_for_subagent();
        let child_b = parent.fork_for_subagent();

        child_a.set("a_key", b"a_val".to_vec()).unwrap();
        child_b.set("b_key", b"b_val".to_vec()).unwrap();

        // Each child sees parent data but not sibling data.
        assert_eq!(child_a.get("shared"), Some(b"parent".to_vec()));
        assert_eq!(child_a.get("a_key"), Some(b"a_val".to_vec()));
        assert!(child_a.get("b_key").is_none());

        assert_eq!(child_b.get("shared"), Some(b"parent".to_vec()));
        assert_eq!(child_b.get("b_key"), Some(b"b_val".to_vec()));
        assert!(child_b.get("a_key").is_none());
    }

    #[test]
    fn test_large_dataset() {
        let backend = CowStateBackend::new();

        // Insert 1000 keys.
        for i in 0..1000 {
            backend
                .set(format!("key_{}", i), format!("value_{}", i).into_bytes())
                .unwrap();
        }

        assert_eq!(backend.local_key_count(), 1000);

        // Fork and verify child sees all keys.
        let child = backend.fork_for_subagent();
        for i in 0..1000 {
            assert_eq!(
                child.get(&format!("key_{}", i)),
                Some(format!("value_{}", i).into_bytes())
            );
        }

        // Child adds 100 more keys.
        for i in 1000..1100 {
            child
                .set(format!("key_{}", i), format!("value_{}", i).into_bytes())
                .unwrap();
        }

        // Merge back to parent.
        backend.merge_from(&child).unwrap();
        assert_eq!(backend.local_key_count(), 1100);
    }

    #[test]
    fn test_version_increments() {
        let backend = CowStateBackend::new();
        assert_eq!(backend.version(), 0);

        backend.set("k1", b"v1".to_vec()).unwrap();
        assert_eq!(backend.version(), 1);

        backend.set("k2", b"v2".to_vec()).unwrap();
        assert_eq!(backend.version(), 2);

        backend.delete("k1");
        assert_eq!(backend.version(), 3);

        backend.clear();
        assert_eq!(backend.version(), 4);
    }

    #[test]
    fn test_is_ancestor_of() {
        let root = CowStateBackend::new();
        let child = root.fork_for_subagent();
        let grandchild = child.fork_for_subagent();

        assert!(root.is_ancestor_of(&child));
        assert!(root.is_ancestor_of(&grandchild));
        assert!(child.is_ancestor_of(&grandchild));

        assert!(!child.is_ancestor_of(&root));
        assert!(!grandchild.is_ancestor_of(&root));
        assert!(!grandchild.is_ancestor_of(&child));
    }

    #[test]
    fn test_snapshot_after_fork() {
        let parent = CowStateBackend::new();
        parent.set("p", b"parent".to_vec()).unwrap();

        let child = parent.fork_for_subagent();
        child.set("c", b"child".to_vec()).unwrap();

        let snapshot = child.snapshot();

        // Snapshot shares child's data.
        assert!(Arc::ptr_eq(&*child.data.borrow(), &*snapshot.data.borrow()));

        // Mutate child to trigger COW.
        child.set("c", b"child_updated".to_vec()).unwrap();

        // Snapshot still has original value.
        assert_eq!(snapshot.get("c"), Some(b"child".to_vec()));
        assert_eq!(child.get("c"), Some(b"child_updated".to_vec()));
    }
}
