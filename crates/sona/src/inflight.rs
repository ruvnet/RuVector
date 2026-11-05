//! Bounded store of in-flight items keyed by a small integer handle.
//!
//! Backs the trajectory handles of the Node.js binding: JS can call
//! `beginTrajectory` without ever calling `endTrajectory`, so the open
//! builders must not grow without bound. Handles are `u32` because that is
//! what the binding exposes to JS (a plain `number`); the counter wraps, and a
//! handle still in flight is never reissued.

use std::collections::{BTreeMap, HashMap};

/// Maps `u32` handles to items, holding at most `capacity` of them.
/// When full, inserting evicts the oldest item.
pub(crate) struct InflightStore<T> {
    capacity: usize,
    items: HashMap<u32, (u64, T)>,
    /// Insertion sequence -> handle, oldest first
    order: BTreeMap<u64, u32>,
    next_handle: u32,
    next_seq: u64,
}

impl<T> InflightStore<T> {
    /// Create a store holding at most `capacity` items (at least 1).
    pub(crate) fn new(capacity: usize) -> Self {
        Self::with_start(capacity, 0)
    }

    fn with_start(capacity: usize, first_handle: u32) -> Self {
        // The handle space must stay larger than the store, or a free handle
        // could not always be found.
        let capacity = capacity.clamp(1, u32::MAX as usize);
        Self {
            capacity,
            items: HashMap::new(),
            order: BTreeMap::new(),
            next_handle: first_handle,
            next_seq: 0,
        }
    }

    /// Store `item` and return its handle, plus the oldest item if one had
    /// to be evicted to make room.
    pub(crate) fn insert(&mut self, item: T) -> (u32, Option<T>) {
        let evicted = if self.items.len() >= self.capacity {
            self.pop_oldest()
        } else {
            None
        };

        // At most `capacity` (< 2^32) handles are taken, so this terminates.
        let mut handle = self.next_handle;
        while self.items.contains_key(&handle) {
            handle = handle.wrapping_add(1);
        }
        self.next_handle = handle.wrapping_add(1);

        let seq = self.next_seq;
        self.next_seq += 1;
        self.items.insert(handle, (seq, item));
        self.order.insert(seq, handle);
        (handle, evicted)
    }

    pub(crate) fn get_mut(&mut self, handle: u32) -> Option<&mut T> {
        self.items.get_mut(&handle).map(|(_, item)| item)
    }

    pub(crate) fn remove(&mut self, handle: u32) -> Option<T> {
        let (seq, item) = self.items.remove(&handle)?;
        self.order.remove(&seq);
        Some(item)
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    fn pop_oldest(&mut self) -> Option<T> {
        let (_, handle) = self.order.pop_first()?;
        self.items.remove(&handle).map(|(_, item)| item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn capacity_is_enforced_by_evicting_oldest() {
        let mut store = InflightStore::new(3);
        let (a, _) = store.insert("a");
        let (b, _) = store.insert("b");
        let (c, _) = store.insert("c");
        let (d, evicted) = store.insert("d");

        assert_eq!(evicted, Some("a"));
        assert_eq!(store.len(), 3);
        assert!(store.get_mut(a).is_none());
        assert_eq!(store.remove(b), Some("b"));
        assert_eq!(store.remove(c), Some("c"));
        assert_eq!(store.remove(d), Some("d"));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn eviction_follows_insertion_order_after_removals() {
        let mut store = InflightStore::new(2);
        let (a, _) = store.insert(1);
        let (_b, _) = store.insert(2);
        store.remove(a);
        store.insert(3);
        // 2 is now the oldest
        let (_, evicted) = store.insert(4);
        assert_eq!(evicted, Some(2));
    }

    #[test]
    fn unbounded_begins_stay_bounded() {
        let mut store = InflightStore::new(100);
        for i in 0..100_000u32 {
            store.insert(i);
        }
        assert_eq!(store.len(), 100);
        assert_eq!(store.order.len(), 100);
    }

    #[test]
    fn handles_are_unique_across_many_inserts() {
        let mut store = InflightStore::new(50_000);
        let mut seen = HashSet::new();
        for i in 0..50_000u32 {
            let (h, evicted) = store.insert(i);
            assert!(evicted.is_none());
            assert!(seen.insert(h), "handle {h} reissued");
        }
    }

    #[test]
    fn wrap_skips_handles_still_in_flight() {
        let mut store = InflightStore::with_start(8, u32::MAX - 1);
        let (h0, _) = store.insert("held-0"); // u32::MAX - 1
        let (h1, _) = store.insert("held-1"); // u32::MAX
        let (h2, _) = store.insert("held-2"); // wraps to 0
        assert_eq!((h0, h1, h2), (u32::MAX - 1, u32::MAX, 0));

        // Drive the counter back around to the held handles.
        store.next_handle = u32::MAX - 1;
        let (h3, _) = store.insert("new");
        assert_eq!(h3, 1, "in-flight handles must be skipped after a wrap");
        assert_eq!(store.remove(h0), Some("held-0"));
        assert_eq!(store.remove(h1), Some("held-1"));
        assert_eq!(store.remove(h2), Some("held-2"));
        assert_eq!(store.remove(h3), Some("new"));
    }

    #[test]
    fn unknown_and_ended_handles_are_ignored() {
        let mut store: InflightStore<u8> = InflightStore::new(4);
        assert!(store.get_mut(7).is_none());
        let (h, _) = store.insert(1);
        assert_eq!(store.remove(h), Some(1));
        assert_eq!(store.remove(h), None);
    }
}
