//! In-memory [`KvStore`] (tests, local tooling), plus fixed clock and
//! counter-based entropy for deterministic tests.

use crate::ports::{Clock, EntropySource, KvStore, StoreError};
use core::cell::{Cell, RefCell};
use ruvector_edge_tenancy::TenancyError;
use std::collections::BTreeMap;

/// `BTreeMap`-backed store.
#[derive(Debug, Default)]
pub struct MemKv {
    map: RefCell<BTreeMap<String, Vec<u8>>>,
}

impl MemKv {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.map.borrow().len()
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.map.borrow().is_empty()
    }
}

impl KvStore for MemKv {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self.map.borrow().get(key).cloned())
    }
    fn put(&self, key: &str, value: &[u8]) -> Result<(), StoreError> {
        self.map
            .borrow_mut()
            .insert(key.to_string(), value.to_vec());
        Ok(())
    }
    fn delete(&self, key: &str) -> Result<(), StoreError> {
        self.map.borrow_mut().remove(key);
        Ok(())
    }
    fn list(
        &self,
        prefix: &str,
        start_after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Vec<u8>)>, StoreError> {
        use core::ops::Bound;
        let lower = match start_after {
            Some(s) if s >= prefix => Bound::Excluded(s.to_string()),
            _ => Bound::Included(prefix.to_string()),
        };
        Ok(self
            .map
            .borrow()
            .range((lower, Bound::Unbounded))
            .take_while(|(k, _)| k.starts_with(prefix))
            .take(limit)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }
}

/// A settable clock.
#[derive(Debug, Default)]
pub struct FixedClock(pub Cell<u64>);

impl FixedClock {
    /// At `now` seconds.
    pub fn at(now: u64) -> Self {
        FixedClock(Cell::new(now))
    }
    /// Move forward.
    pub fn advance(&self, secs: u64) {
        self.0.set(self.0.get() + secs);
    }
}

impl Clock for FixedClock {
    fn now_unix(&self) -> u64 {
        self.0.get()
    }
}

/// Deterministic, **non-secret** entropy (a counter). Tests only.
#[derive(Debug, Default)]
pub struct CounterEntropy(Cell<u64>);

impl EntropySource for CounterEntropy {
    fn fill(&self, out: &mut [u8]) -> Result<(), TenancyError> {
        let n = self.0.get() + 1;
        self.0.set(n);
        for (i, b) in out.iter_mut().enumerate() {
            *b = (n >> ((i % 8) * 8)) as u8 ^ (i as u8).wrapping_mul(31);
        }
        Ok(())
    }
}
