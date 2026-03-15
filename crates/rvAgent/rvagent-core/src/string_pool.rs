//! String interning for repeated strings (ADR-103 A8).
//!
//! Tool names, field names, and other frequently repeated strings are
//! interned into a shared pool backed by `DashMap` for lock-free concurrent
//! reads.

use dashmap::DashMap;
use std::sync::Arc;

/// Thread-safe string interner that deduplicates common strings.
///
/// Interned strings are returned as `Arc<str>`, which is cheap to clone
/// (pointer-sized + atomic increment) compared to allocating a new `String`
/// each time.
///
/// # Example
///
/// ```
/// use rvagent_core::string_pool::StringPool;
///
/// let pool = StringPool::new();
/// let a = pool.intern("read_file");
/// let b = pool.intern("read_file");
/// assert!(Arc::ptr_eq(&a, &b));
/// # use std::sync::Arc;
/// ```
pub struct StringPool {
    pool: DashMap<String, Arc<str>>,
}

impl StringPool {
    /// Create a new, empty string pool.
    pub fn new() -> Self {
        Self {
            pool: DashMap::new(),
        }
    }

    /// Intern the string `s`, returning a shared reference.
    ///
    /// If `s` has been interned before, the existing `Arc<str>` is returned
    /// (no new allocation). Otherwise `s` is stored and a new `Arc<str>` is
    /// created.
    pub fn intern(&self, s: &str) -> Arc<str> {
        // Fast path: already interned.
        if let Some(entry) = self.pool.get(s) {
            return Arc::clone(entry.value());
        }

        // Slow path: insert.
        let arc: Arc<str> = Arc::from(s);
        self.pool
            .entry(s.to_string())
            .or_insert_with(|| Arc::clone(&arc))
            .clone()
    }

    /// Number of unique strings currently interned.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Returns `true` if no strings have been interned.
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }
}

impl Default for StringPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_dedup() {
        let pool = StringPool::new();
        let a = pool.intern("hello");
        let b = pool.intern("hello");
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_intern_different() {
        let pool = StringPool::new();
        let _a = pool.intern("foo");
        let _b = pool.intern("bar");
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_empty_pool() {
        let pool = StringPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }
}
