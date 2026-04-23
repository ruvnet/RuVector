//! RaBitQ-compressed cache — witness-addressed.
//!
//! Wraps `ruvector_rabitq::RabitqPlusIndex`. Cache entries are keyed by
//! the [`RuLakeBundle`](crate::RuLakeBundle) SHAKE-256 witness, NOT by
//! `(backend_id, collection)`. Two backends serving the same logical
//! dataset — same `data_ref`, same rotation seed, same rerank factor,
//! same generation — produce the same witness and share one compressed
//! cache entry. This implements the reviewer's "use the RVF witness
//! chain hash as cache-key anchor" fix for cache-invalidation drift
//! (see ADR-155 §Decision 6).
//!
//! Callers still search by `(backend, collection)`; a secondary pointer
//! map resolves that to a witness. When the backend reports a new
//! witness (generation bump, seed change, data_ref change), the pointer
//! moves and — if the old entry has no remaining pointers — it is
//! garbage-collected.
//!
//! ## Coherence model
//!
//! On every search the router asks the backend for its current bundle
//! and compares its witness with the cached pointer. On mismatch the
//! pointer updates and a fresh pull+prime runs (unless the target
//! witness is already cached under another pointer — then we just
//! swap the pointer for free). Under `Consistency::Eventual` the
//! witness check is skipped for up to `ttl_ms` after the last
//! successful check.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ruvector_rabitq::{AnnIndex, RabitqPlusIndex};

use crate::backend::{BackendId, CollectionId, PulledBatch};

/// How strictly the cache checks freshness before answering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Consistency {
    /// Consult the backend's current bundle on every search. Default.
    #[default]
    Fresh,
    /// Trust the cache for up to `ttl_ms` milliseconds between checks.
    /// Higher QPS; backend updates may be ignored for up to ttl.
    Eventual { ttl_ms: u64 },
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub primes: u64,
    pub invalidations: u64,
    /// Incremented when a pointer move found the target witness
    /// already cached under another pointer — zero prime work done.
    pub shared_hits: u64,
}

/// External lookup key: `(backend_id, collection_id)`.
pub type CacheKey = (BackendId, CollectionId);

/// Internal content-addressed key: the RuLakeBundle witness (SHAKE-256
/// hex). Two bundles with the same witness are interchangeable; the
/// cache deduplicates them.
pub type WitnessKey = String;

struct CacheEntry {
    index: RabitqPlusIndex,
    dim: usize,
    #[allow(dead_code)] // kept for diagnostics
    generation_hint: Option<u64>,
    last_checked: Instant,
    /// Last time any search hit this entry — used by LRU eviction.
    last_used: Instant,
    /// internal-position → external id.
    pos_to_id: Vec<u64>,
    /// How many external pointers currently resolve to this witness.
    /// An entry with `refcount > 0` is ineligible for LRU eviction.
    refcount: u32,
}

pub struct VectorCache {
    inner: Arc<Mutex<CacheState>>,
    rerank_factor: usize,
    rotation_seed: u64,
    /// LRU cap on the number of distinct compressed entries. `None`
    /// means unbounded (the MVP default). Evicts the least-recently-used
    /// *refcount-0* entry on over-cap; refcount>0 entries are never
    /// evicted (a live pointer needs them). Note: in the current API all
    /// active pointers refcount their entries, so `max_entries` only has
    /// effect when pointers have been removed via `invalidate()` but the
    /// witness entry is still in the pool awaiting GC.
    max_entries: Option<usize>,
}

struct CacheState {
    /// witness → compressed index
    entries: HashMap<WitnessKey, CacheEntry>,
    /// (backend, collection) → witness
    pointers: HashMap<CacheKey, WitnessKey>,
    /// cache-key → last time the witness check ran (for Eventual mode)
    last_checked: HashMap<CacheKey, Instant>,
    stats: CacheStats,
}

impl VectorCache {
    pub fn new(rerank_factor: usize, rotation_seed: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheState {
                entries: HashMap::new(),
                pointers: HashMap::new(),
                last_checked: HashMap::new(),
                stats: CacheStats::default(),
            })),
            rerank_factor,
            rotation_seed,
            max_entries: None,
        }
    }

    /// Cap the cache at `n` distinct compressed entries. Evicts the
    /// least-recently-used *unpinned* (refcount == 0) entry when the
    /// pool exceeds `n`. Pinned entries (refcount > 0) are never
    /// evicted; if the cap is reached and every entry is pinned,
    /// `prime()` succeeds anyway and the cap is temporarily exceeded —
    /// correctness over strict bounds.
    pub fn with_max_entries(mut self, n: usize) -> Self {
        self.max_entries = Some(n.max(1));
        self
    }

    pub fn rerank_factor(&self) -> usize {
        self.rerank_factor
    }
    pub fn rotation_seed(&self) -> u64 {
        self.rotation_seed
    }

    pub fn stats(&self) -> CacheStats {
        self.inner.lock().unwrap().stats.clone()
    }

    /// Compress a pulled batch into a RaBitQ index and associate it with
    /// the target witness. Bookkeeping happens under the lock; the
    /// heavy O(n·D) compression runs BEFORE acquiring the lock.
    pub fn prime(
        &self,
        key: CacheKey,
        witness: WitnessKey,
        batch: PulledBatch,
    ) -> crate::Result<()> {
        // Fast path: target witness already cached — just point and return.
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.entries.contains_key(&witness) {
                return self.inner_install_pointer_unlocked(&mut inner, key, witness, true);
            }
        }

        // Slow path: build the index lock-free.
        let dim = batch.dim;
        let generation = batch.generation;
        let mut idx = RabitqPlusIndex::new(dim, self.rotation_seed, self.rerank_factor);
        let mut pos_to_id = Vec::with_capacity(batch.ids.len());
        for (pos, v) in batch.vectors.into_iter().enumerate() {
            idx.add(pos, v)?;
            pos_to_id.push(batch.ids[pos]);
        }
        let entry = CacheEntry {
            index: idx,
            dim,
            generation_hint: Some(generation),
            last_checked: Instant::now(),
            last_used: Instant::now(),
            pos_to_id,
            refcount: 0, // install_pointer bumps it
        };

        let mut inner = self.inner.lock().unwrap();
        // Another thread might have raced us and installed the witness
        // in the meantime — if so, drop our work and take the shared
        // entry (the two builds produce identical codes by determinism).
        if inner.entries.contains_key(&witness) {
            return self.inner_install_pointer_unlocked(&mut inner, key, witness, true);
        }
        inner.entries.insert(witness.clone(), entry);
        inner.stats.primes += 1;
        let rc = self.inner_install_pointer_unlocked(&mut inner, key, witness, false);
        // Opportunistic LRU eviction: only runs when a cap is set,
        // and only trims unpinned (refcount==0) entries, so live
        // pointers are never orphaned.
        if let Some(cap) = self.max_entries {
            self.evict_lru_if_over(&mut inner, cap);
        }
        rc
    }

    /// Evict the least-recently-used unpinned entry until we're at or
    /// below `cap`. Pinned entries are skipped; in the worst case every
    /// entry is pinned and we can't evict anyone — that's by design.
    fn evict_lru_if_over(&self, inner: &mut CacheState, cap: usize) {
        while inner.entries.len() > cap {
            // Find the oldest unpinned entry.
            let victim = inner
                .entries
                .iter()
                .filter(|(_, e)| e.refcount == 0)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(w, _)| w.clone());
            match victim {
                Some(w) => {
                    inner.entries.remove(&w);
                    inner.stats.invalidations += 1;
                }
                None => break, // every entry pinned
            }
        }
    }

    /// Core pointer-install logic — must be called with the lock held.
    /// If `shared`, we bump the `shared_hits` stat (the caller saved a
    /// full prime by resolving to an already-cached witness).
    fn inner_install_pointer_unlocked(
        &self,
        inner: &mut CacheState,
        key: CacheKey,
        witness: WitnessKey,
        shared: bool,
    ) -> crate::Result<()> {
        // If this key already points somewhere, decrement the old entry.
        if let Some(old_w) = inner.pointers.remove(&key) {
            if let Some(e) = inner.entries.get_mut(&old_w) {
                e.refcount = e.refcount.saturating_sub(1);
                if e.refcount == 0 {
                    inner.entries.remove(&old_w);
                    inner.stats.invalidations += 1;
                }
            }
        }
        inner.pointers.insert(key.clone(), witness.clone());
        if let Some(e) = inner.entries.get_mut(&witness) {
            e.refcount = e.refcount.saturating_add(1);
            e.last_checked = Instant::now();
        }
        inner.last_checked.insert(key, Instant::now());
        if shared {
            inner.stats.shared_hits += 1;
        }
        Ok(())
    }

    /// Drop the pointer for a given key (used by explicit invalidation).
    /// The underlying entry is garbage-collected when its last pointer
    /// goes.
    pub fn invalidate(&self, key: &CacheKey) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(old_w) = inner.pointers.remove(key) {
            if let Some(e) = inner.entries.get_mut(&old_w) {
                e.refcount = e.refcount.saturating_sub(1);
                if e.refcount == 0 {
                    inner.entries.remove(&old_w);
                }
            }
            inner.stats.invalidations += 1;
        }
        inner.last_checked.remove(key);
    }

    pub fn has(&self, key: &CacheKey) -> bool {
        self.inner.lock().unwrap().pointers.contains_key(key)
    }

    /// What witness currently resolves from this key? `None` if unprimed.
    pub fn witness_of(&self, key: &CacheKey) -> Option<WitnessKey> {
        self.inner.lock().unwrap().pointers.get(key).cloned()
    }

    /// How many external pointers resolve to this witness? (diagnostic)
    pub fn refcount_of(&self, witness: &str) -> u32 {
        self.inner
            .lock()
            .unwrap()
            .entries
            .get(witness)
            .map(|e| e.refcount)
            .unwrap_or(0)
    }

    /// How many distinct compressed-index entries exist in the cache?
    /// Differs from `pointers.len()` when witnesses are shared.
    pub fn entry_count(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    pub fn dim_of(&self, key: &CacheKey) -> Option<usize> {
        let inner = self.inner.lock().unwrap();
        let w = inner.pointers.get(key)?;
        inner.entries.get(w).map(|e| e.dim)
    }

    pub(crate) fn mark_hit(&self) {
        self.inner.lock().unwrap().stats.hits += 1;
    }
    pub(crate) fn mark_miss(&self) {
        self.inner.lock().unwrap().stats.misses += 1;
    }

    /// Run the search against the cached entry for `key`. Caller must
    /// ensure freshness first.
    pub fn search_cached(
        &self,
        key: &CacheKey,
        query: &[f32],
        k: usize,
    ) -> crate::Result<Vec<(u64, f32)>> {
        let mut inner = self.inner.lock().unwrap();
        let witness = inner
            .pointers
            .get(key)
            .ok_or_else(|| crate::RuLakeError::UnknownCollection {
                backend: key.0.clone(),
                collection: key.1.clone(),
            })?
            .clone();
        let entry = inner.entries.get_mut(&witness).ok_or_else(|| {
            crate::RuLakeError::UnknownCollection {
                backend: key.0.clone(),
                collection: key.1.clone(),
            }
        })?;
        if query.len() != entry.dim {
            return Err(crate::RuLakeError::DimensionMismatch {
                expected: entry.dim,
                actual: query.len(),
            });
        }
        // LRU timestamp bump — unconditional; cheaper than the branch.
        entry.last_used = Instant::now();
        let hits = entry.index.search(query, k)?;
        let pos_to_id = entry.pos_to_id.clone();
        drop(inner);
        Ok(hits
            .into_iter()
            .map(|r| (pos_to_id[r.id], r.score))
            .collect())
    }

    pub fn touch(&self, key: &CacheKey) {
        let mut inner = self.inner.lock().unwrap();
        inner.last_checked.insert(key.clone(), Instant::now());
    }

    pub fn can_skip_check(&self, key: &CacheKey, consistency: Consistency) -> bool {
        match consistency {
            Consistency::Fresh => false,
            Consistency::Eventual { ttl_ms } => {
                let inner = self.inner.lock().unwrap();
                match inner.last_checked.get(key) {
                    Some(t) => t.elapsed().as_millis() < ttl_ms as u128,
                    None => false,
                }
            }
        }
    }
}
