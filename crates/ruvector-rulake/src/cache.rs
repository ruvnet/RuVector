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
//!
//! ## Key interning (memory-audit finding #1)
//!
//! The hot path used to clone `(String, String)` keys across every
//! `mark_hit` / `mark_miss` / `per_backend_mut` call — ≈96 B/query at
//! federated fan-out with 3 K calls per query. We now intern the
//! backend id and collection id into `Arc<str>` once per incoming
//! `RuLake` call ([`InternedKey`]); every downstream op moves through
//! cheap refcount bumps instead of `String::clone` + hashmap rehashing.
//!
//! The **public** `CacheKey = (String, String)` alias is unchanged so
//! existing callers / tests compile untouched; the `Arc` world is a
//! strict implementation detail of the cache + router hot path.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ruvector_rabitq::{AnnIndex, RabitqPlusIndex};

use crate::backend::{BackendId, CollectionId, PulledBatch};

/// How strictly the cache checks freshness before answering.
///
/// This is the product knob surfaced by the strategic review: Fresh
/// for compliance, Eventual for recall, Frozen for audit. It lets a
/// single ruLake deployment expose per-collection staleness SLAs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Consistency {
    /// Consult the backend's current bundle on every search. Default.
    /// Use for compliance, finance, policy-enforced workloads where
    /// any stale answer is worse than a slower answer.
    #[default]
    Fresh,
    /// Trust the cache for up to `ttl_ms` milliseconds between checks.
    /// Higher QPS; backend updates may be ignored for up to ttl. Use
    /// for search, AI retrieval, recommendation, RAG — where a small
    /// staleness window is a trade every customer accepts.
    Eventual { ttl_ms: u64 },
    /// Caller asserts the bundle at this `(backend, collection)` key
    /// is immutable for the cache's lifetime — never re-check the
    /// backend, never invalidate on generation bump. Designed for
    /// witness-sealed historical snapshots: the audit tier.
    ///
    /// Use when you have a materialized bundle whose `data_ref` points
    /// at a content-addressed (CA) artifact and the `rvf_witness` is
    /// already verifiable end-to-end. An explicit `refresh_from_bundle_dir`
    /// call still invalidates — the guarantee is about automatic
    /// coherence checks, not about whether the cache can be swapped
    /// by the operator.
    Frozen,
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
    /// Sum of prime durations in milliseconds. Paired with `primes` to
    /// compute the mean via `avg_prime_ms`.
    pub total_prime_ms: f64,
    /// Most-recent prime duration in milliseconds (useful to detect
    /// drift between warm primes and the very first miss).
    pub last_prime_ms: f64,
    /// Incremented each time a pre-built index is installed via
    /// [`VectorCache::install_prebuilt`] — i.e. a warm-from-disk
    /// rehydrate that did NOT round-trip to the backend. Deliberately
    /// separate from `primes` so operators can tell cold-prime cost
    /// (pull+compress) apart from warm-restart cost (mmap+install).
    pub warm_installs: u64,
}

impl CacheStats {
    /// Cache hit rate over `hits + misses`. Returns `None` when no
    /// coherence-checked searches have run yet.
    ///
    /// This is the primary KPI for cache-first operation: an
    /// `hit_rate >= 0.95` means 95% of queries never pull from the
    /// backend, which is the acceptance bar for the cache-first reframe
    /// in ADR-155.
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.hits + self.misses;
        if total == 0 {
            None
        } else {
            Some(self.hits as f64 / total as f64)
        }
    }

    /// Mean prime duration in milliseconds, or `None` when no primes
    /// have run. Use to budget the cold-start cost of entering reality
    /// on a warm serving process.
    pub fn avg_prime_ms(&self) -> Option<f64> {
        if self.primes == 0 {
            None
        } else {
            Some(self.total_prime_ms / self.primes as f64)
        }
    }
}

/// Per-backend counters. Lets operators see which backend is hot
/// (high hit_rate) vs cold (high miss+prime cost) without having to
/// attribute global stats back to individual backends.
///
/// Used by [`VectorCache::stats_by_backend`] / [`RuLake::cache_stats_by_backend`].
#[derive(Debug, Clone, Default)]
pub struct PerBackendStats {
    pub hits: u64,
    pub misses: u64,
    pub primes: u64,
    pub invalidations: u64,
    pub shared_hits: u64,
}

impl PerBackendStats {
    /// Cache hit rate over `hits + misses` for this backend. `None`
    /// when no coherence-checked searches have run against it yet.
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.hits + self.misses;
        if total == 0 {
            None
        } else {
            Some(self.hits as f64 / total as f64)
        }
    }
}

/// External lookup key: `(backend_id, collection_id)`.
///
/// Kept as `(String, String)` for public API stability — callers pass
/// owned tuples by reference at rare diagnostic call sites
/// (`cache_witness_of`, `invalidate_cache`, …) and we intern them into
/// [`InternedKey`] on the boundary. The hot router path never allocates
/// an owned `(String, String)` — see [`InternedKey`].
pub type CacheKey = (BackendId, CollectionId);

/// Build a [`CacheKey`] from two `&str` without sprinkling
/// `.to_string()` across call sites. The public API-compat name from
/// the memory-audit follow-up.
#[inline]
pub fn make_key(backend: &str, collection: &str) -> CacheKey {
    (backend.to_string(), collection.to_string())
}

/// Internal, refcount-cheap `(backend_id, collection_id)` key.
///
/// The router interns once per incoming call and hands `Arc<str>` refs
/// down through the cache — every subsequent `mark_hit` /
/// `per_backend_mut` / pointer-map lookup does a refcount bump instead
/// of a `String` clone. That closes memory-audit finding #1.
pub(crate) type InternedKey = (Arc<str>, Arc<str>);

/// Intern a `(backend, collection)` pair into [`InternedKey`].
///
/// One allocation per `Arc<str>` up front; every downstream op is a
/// cheap refcount bump.
#[inline]
pub(crate) fn intern_key(backend: &str, collection: &str) -> InternedKey {
    (Arc::from(backend), Arc::from(collection))
}

/// Intern an external [`CacheKey`] into [`InternedKey`]. Used by the
/// rare public-facing entrypoints that still take `&CacheKey`
/// (`invalidate_cache`, `witness_of`, …) to cross from the owned-string
/// world into the Arc world before talking to the cache.
#[inline]
pub(crate) fn intern_cache_key(key: &CacheKey) -> InternedKey {
    intern_key(&key.0, &key.1)
}

/// Internal content-addressed key: the RuLakeBundle witness (SHAKE-256
/// hex). Two bundles with the same witness are interchangeable; the
/// cache deduplicates them.
pub type WitnessKey = String;

struct CacheEntry {
    /// `Arc`-wrapped so reader threads can `clone` it under the lock
    /// and run the RaBitQ scan *without* the cache mutex held.
    /// Eliminates the scan-serializes-on-mutex behavior the original
    /// implementation showed under concurrent clients — see
    /// `BENCHMARK.md` "concurrent clients" block.
    index: Arc<RabitqPlusIndex>,
    dim: usize,
    #[allow(dead_code)] // kept for diagnostics
    generation_hint: Option<u64>,
    last_checked: Instant,
    /// Last time any search hit this entry — used by LRU eviction.
    last_used: Instant,
    /// internal-position → external id. Arc so we can share-clone
    /// without copying the vector on every search.
    ///
    /// Memory-audit finding #2 (HIGH) — this duplicates
    /// `RabitqPlusIndex.ids: Vec<u32>` inside `ruvector-rabitq`. A dedup
    /// would require widening rabitq's internal `ids: Vec<u32>` to
    /// `Vec<u64>` (loses cache-line density) OR exposing a position
    /// iterator from rabitq. Both are cross-crate changes and the
    /// memory-audit follow-up explicitly scoped us to `ruvector-rulake`
    /// only — so we keep the split and document it here. At n=1 M with
    /// u64 ids this is 8 MB/entry; entry counts are bounded by
    /// `max_entries`, so the waste is capped, not unbounded.
    pos_to_id: Arc<Vec<u64>>,
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
    /// (backend, collection) → witness. Uses `Arc<str>` keys so that
    /// the hot router path never has to own a `(String, String)`.
    pointers: HashMap<InternedKey, WitnessKey>,
    /// cache-key → last time the witness check ran (for Eventual mode).
    last_checked: HashMap<InternedKey, Instant>,
    /// Per-backend counters. Populated lazily on the first event per
    /// backend id.
    stats: CacheStats,
    per_backend: HashMap<Arc<str>, PerBackendStats>,
    /// Per-`(backend, collection)` counters — same events as
    /// `per_backend`, attributed one level finer.
    per_collection: HashMap<InternedKey, PerBackendStats>,
}

impl CacheState {
    /// Look up (or insert) the per-backend counter by borrowing the
    /// existing `Arc<str>` — only creates a new `Arc<str>` allocation
    /// when the backend is seen for the first time.
    fn per_backend_mut(&mut self, backend: &Arc<str>) -> &mut PerBackendStats {
        if !self.per_backend.contains_key(backend) {
            self.per_backend
                .insert(Arc::clone(backend), PerBackendStats::default());
        }
        self.per_backend.get_mut(backend).unwrap()
    }
    /// Look up (or insert) the per-collection counter. Same pattern:
    /// the `Arc` pair is cloned (refcount bumps × 2) on first-insert
    /// only — subsequent events reuse the existing map entry.
    fn per_collection_mut(&mut self, key: &InternedKey) -> &mut PerBackendStats {
        if !self.per_collection.contains_key(key) {
            self.per_collection.insert(
                (Arc::clone(&key.0), Arc::clone(&key.1)),
                PerBackendStats::default(),
            );
        }
        self.per_collection.get_mut(key).unwrap()
    }
}

impl VectorCache {
    pub fn new(rerank_factor: usize, rotation_seed: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheState {
                entries: HashMap::new(),
                pointers: HashMap::new(),
                last_checked: HashMap::new(),
                stats: CacheStats::default(),
                per_backend: HashMap::new(),
                per_collection: HashMap::new(),
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

    /// Per-backend counters. Populated lazily on first activity
    /// against a given backend id — empty backends are not in the
    /// map. Use to see which backend is hot vs cold without having
    /// to attribute global counters manually.
    ///
    /// Converts the internal `Arc<str>` keys back to `String` on the
    /// way out so the public surface matches the pre-intern API
    /// (cold-path call: one clone per entry, run once per diagnostic
    /// read).
    pub fn stats_by_backend(&self) -> HashMap<BackendId, PerBackendStats> {
        self.inner
            .lock()
            .unwrap()
            .per_backend
            .iter()
            .map(|(k, v)| (k.as_ref().to_string(), v.clone()))
            .collect()
    }

    /// Per-`(backend, collection)` counters. One level finer than
    /// `stats_by_backend`: operators who want to know "which
    /// collection is hot?" get exact attribution.
    pub fn stats_by_collection(&self) -> HashMap<CacheKey, PerBackendStats> {
        self.inner
            .lock()
            .unwrap()
            .per_collection
            .iter()
            .map(|((b, c), v)| ((b.as_ref().to_string(), c.as_ref().to_string()), v.clone()))
            .collect()
    }

    /// Compress a pulled batch into a RaBitQ index and associate it with
    /// the target witness. Bookkeeping happens under the lock; the
    /// heavy O(n·D) compression runs BEFORE acquiring the lock.
    pub(crate) fn prime_interned(
        &self,
        key: InternedKey,
        witness: WitnessKey,
        batch: PulledBatch,
    ) -> crate::Result<()> {
        // Defense-in-depth: reject hostile / corrupt batches before any
        // allocation. A malicious backend claiming n=u64::MAX / dim=2^30
        // would otherwise OOM the host during prime.
        crate::backend::validate_pulled_batch(&batch)?;
        // Fast path: target witness already cached — just point and return.
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.entries.contains_key(&witness) {
                return self.inner_install_pointer_unlocked(&mut inner, key, witness, true);
            }
        }

        // Slow path: build the index lock-free, time it for prime stats.
        //
        // We pick between serial `add` and rayon-parallel
        // `from_vectors_parallel` based on batch size. The parallel
        // path amortizes rayon's task-queue overhead only when the
        // D×D rotation work is large enough — below ~1k vectors the
        // serial loop wins. 1024 was picked from a sweep on D=128;
        // workloads with much larger D will benefit from parallel
        // sooner and the threshold can be tuned.
        let prime_start = Instant::now();
        let dim = batch.dim;
        let generation = batch.generation;
        const PARALLEL_PRIME_THRESHOLD: usize = 1024;
        let pos_to_id: Vec<u64> = batch.ids.clone();
        let idx = if batch.vectors.len() >= PARALLEL_PRIME_THRESHOLD {
            let items: Vec<(usize, Vec<f32>)> = batch.vectors.into_iter().enumerate().collect();
            RabitqPlusIndex::from_vectors_parallel(
                dim,
                self.rotation_seed,
                self.rerank_factor,
                items,
            )?
        } else {
            let mut idx = RabitqPlusIndex::new(dim, self.rotation_seed, self.rerank_factor);
            for (pos, v) in batch.vectors.into_iter().enumerate() {
                idx.add(pos, v)?;
            }
            idx
        };
        let entry = CacheEntry {
            index: Arc::new(idx),
            dim,
            generation_hint: Some(generation),
            last_checked: Instant::now(),
            last_used: Instant::now(),
            pos_to_id: Arc::new(pos_to_id),
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
        inner.per_backend_mut(&key.0).primes += 1;
        inner.per_collection_mut(&key).primes += 1;
        let prime_ms = prime_start.elapsed().as_secs_f64() * 1000.0;
        inner.stats.total_prime_ms += prime_ms;
        inner.stats.last_prime_ms = prime_ms;
        let rc = self.inner_install_pointer_unlocked(&mut inner, key, witness, false);
        // Opportunistic LRU eviction: only runs when a cap is set,
        // and only trims unpinned (refcount==0) entries, so live
        // pointers are never orphaned.
        if let Some(cap) = self.max_entries {
            self.evict_lru_if_over(&mut inner, cap);
        }
        rc
    }

    /// Public entrypoint kept for API compatibility. Interns the
    /// `(String, String)` key once, then delegates to the hot path.
    pub fn prime(
        &self,
        key: CacheKey,
        witness: WitnessKey,
        batch: PulledBatch,
    ) -> crate::Result<()> {
        let interned = intern_cache_key(&key);
        self.prime_interned(interned, witness, batch)
    }

    /// Install a pre-built `RabitqPlusIndex` under `witness` and install
    /// the pointer `key → witness` — the warm-from-disk counterpart to
    /// [`prime`]. No backend round-trip, no RaBitQ compression: the
    /// caller supplies an already-compressed index (typically loaded via
    /// `ruvector_rabitq::persist::load_index`) together with the
    /// `pos_to_id` mapping the cache would otherwise derive from
    /// `PulledBatch::ids`.
    ///
    /// Semantics:
    ///
    /// - If `witness` is already cached (another pointer brought it in),
    ///   the supplied `idx` is dropped and the existing entry is shared
    ///   — this is the same "witness already present" fast path `prime`
    ///   uses, so two operators warming the same bundle from different
    ///   sidecars see one compressed entry with refcount 2.
    /// - If `witness` is new, the entry is inserted, `warm_installs` is
    ///   bumped, and `primes` / prime-duration counters are left alone
    ///   (this is not a prime — no compression ran).
    /// - Either way the pointer `key → witness` is installed and its
    ///   refcount bumped.
    ///
    /// The LRU cap honours warm installs identically to prime installs:
    /// an oversize cache still evicts unpinned entries.
    pub fn install_prebuilt(
        &self,
        key: CacheKey,
        witness: WitnessKey,
        idx: Arc<RabitqPlusIndex>,
        pos_to_id: Arc<Vec<u64>>,
    ) -> crate::Result<()> {
        let interned = intern_cache_key(&key);
        self.install_prebuilt_interned(interned, witness, idx, pos_to_id)
    }

    pub(crate) fn install_prebuilt_interned(
        &self,
        key: InternedKey,
        witness: WitnessKey,
        idx: Arc<RabitqPlusIndex>,
        pos_to_id: Arc<Vec<u64>>,
    ) -> crate::Result<()> {
        // Defensive consistency check before we touch state: the caller
        // must hand us a `pos_to_id` whose length matches the index.
        // A mismatch means the sidecar and the .rbpx drifted, and
        // serving through that would map positions to the wrong ids
        // without any visible error until a query returns garbage.
        if pos_to_id.len() != idx.len() {
            return Err(crate::RuLakeError::InvalidParameter(format!(
                "install_prebuilt: pos_to_id.len()={} but index.len()={}",
                pos_to_id.len(),
                idx.len()
            )));
        }
        let mut inner = self.inner.lock().unwrap();
        // Fast path: target witness already cached — just point and
        // bookkeep as a shared install. `shared=true` bumps
        // `shared_hits` — but this is a warm install, not a coherence
        // event, so we route through `inner_install_pointer_unlocked`
        // with `shared=false` to avoid polluting coherence stats.
        if inner.entries.contains_key(&witness) {
            return self.inner_install_pointer_unlocked(&mut inner, key, witness, false);
        }
        let dim = idx.dim();
        let entry = CacheEntry {
            index: idx,
            dim,
            generation_hint: None,
            last_checked: Instant::now(),
            last_used: Instant::now(),
            pos_to_id,
            refcount: 0, // install_pointer bumps it
        };
        inner.entries.insert(witness.clone(), entry);
        inner.stats.warm_installs += 1;
        // NOTE: we intentionally do NOT bump `primes` / prime timers —
        // a warm install did no compression work, so conflating the
        // two would hide cold-start cost from operators.
        let rc = self.inner_install_pointer_unlocked(&mut inner, key, witness, false);
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
        key: InternedKey,
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
                    inner.per_backend_mut(&key.0).invalidations += 1;
                    inner.per_collection_mut(&key).invalidations += 1;
                }
            }
        }
        inner.pointers.insert(key.clone(), witness.clone());
        if let Some(e) = inner.entries.get_mut(&witness) {
            e.refcount = e.refcount.saturating_add(1);
            e.last_checked = Instant::now();
        }
        inner.last_checked.insert(key.clone(), Instant::now());
        if shared {
            inner.stats.shared_hits += 1;
            inner.per_backend_mut(&key.0).shared_hits += 1;
            inner.per_collection_mut(&key).shared_hits += 1;
        }
        Ok(())
    }

    /// Drop the pointer for a given key (used by explicit invalidation).
    /// The underlying entry is garbage-collected when its last pointer
    /// goes.
    pub fn invalidate(&self, key: &CacheKey) {
        let interned = intern_cache_key(key);
        self.invalidate_interned(&interned);
    }

    pub(crate) fn invalidate_interned(&self, key: &InternedKey) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(old_w) = inner.pointers.remove(key) {
            if let Some(e) = inner.entries.get_mut(&old_w) {
                e.refcount = e.refcount.saturating_sub(1);
                if e.refcount == 0 {
                    inner.entries.remove(&old_w);
                }
            }
            inner.stats.invalidations += 1;
            inner.per_backend_mut(&key.0).invalidations += 1;
            inner.per_collection_mut(key).invalidations += 1;
        }
        inner.last_checked.remove(key);
    }

    pub fn has(&self, key: &CacheKey) -> bool {
        let interned = intern_cache_key(key);
        self.inner.lock().unwrap().pointers.contains_key(&interned)
    }

    /// What witness currently resolves from this key? `None` if unprimed.
    pub fn witness_of(&self, key: &CacheKey) -> Option<WitnessKey> {
        let interned = intern_cache_key(key);
        self.witness_of_interned(&interned)
    }

    pub(crate) fn witness_of_interned(&self, key: &InternedKey) -> Option<WitnessKey> {
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

    /// Clone out the `Arc<RabitqPlusIndex>` backing `witness` and its
    /// `Arc<Vec<u64>>` pos→id map — used by the `save_cache_to_dir`
    /// path in `RuLake` to serialize a primed entry without exposing
    /// `CacheEntry` publicly. Returns `None` when the witness is not
    /// currently cached.
    pub(crate) fn index_and_ids_of(
        &self,
        witness: &str,
    ) -> Option<(Arc<RabitqPlusIndex>, Arc<Vec<u64>>)> {
        let inner = self.inner.lock().unwrap();
        inner
            .entries
            .get(witness)
            .map(|e| (Arc::clone(&e.index), Arc::clone(&e.pos_to_id)))
    }

    pub fn dim_of(&self, key: &CacheKey) -> Option<usize> {
        let interned = intern_cache_key(key);
        let inner = self.inner.lock().unwrap();
        let w = inner.pointers.get(&interned)?;
        inner.entries.get(w).map(|e| e.dim)
    }

    pub(crate) fn mark_hit(&self, key: &InternedKey) {
        let mut inner = self.inner.lock().unwrap();
        inner.stats.hits += 1;
        inner.per_backend_mut(&key.0).hits += 1;
        inner.per_collection_mut(key).hits += 1;
    }
    pub(crate) fn mark_miss(&self, key: &InternedKey) {
        let mut inner = self.inner.lock().unwrap();
        inner.stats.misses += 1;
        inner.per_backend_mut(&key.0).misses += 1;
        inner.per_collection_mut(key).misses += 1;
    }

    /// Run the search against the cached entry for `key`. Caller must
    /// ensure freshness first. Uses the cache's default rerank factor.
    pub fn search_cached(
        &self,
        key: &CacheKey,
        query: &[f32],
        k: usize,
    ) -> crate::Result<Vec<(u64, f32)>> {
        self.search_cached_with_rerank(key, query, k, None)
    }

    /// As [`search_cached`], but allows per-call override of the rerank
    /// factor. `None` uses the cache's default; `Some(n)` passes `n` to
    /// the underlying `RabitqPlusIndex::search_with_rerank`. Used by
    /// `RuLake::search_federated` to divide the global rerank cost
    /// across K shards instead of paying K× for it.
    pub fn search_cached_with_rerank(
        &self,
        key: &CacheKey,
        query: &[f32],
        k: usize,
        rerank_factor_override: Option<usize>,
    ) -> crate::Result<Vec<(u64, f32)>> {
        let interned = intern_cache_key(key);
        self.search_cached_with_rerank_interned(&interned, query, k, rerank_factor_override)
    }

    pub(crate) fn search_cached_with_rerank_interned(
        &self,
        key: &InternedKey,
        query: &[f32],
        k: usize,
        rerank_factor_override: Option<usize>,
    ) -> crate::Result<Vec<(u64, f32)>> {
        // Look up the entry under the lock, clone the Arcs, drop the
        // lock, then run the scan unlocked. Concurrent readers on
        // different or same keys all proceed in parallel — the scan
        // is CPU-bound and needs no shared state beyond the index
        // (which is Arc-owned; no `&mut` required).
        let (index, pos_to_id, dim) = {
            let mut inner = self.inner.lock().unwrap();
            let witness = inner
                .pointers
                .get(key)
                .ok_or_else(|| crate::RuLakeError::UnknownCollection {
                    backend: key.0.as_ref().to_string(),
                    collection: key.1.as_ref().to_string(),
                })?
                .clone();
            let entry = inner.entries.get_mut(&witness).ok_or_else(|| {
                crate::RuLakeError::UnknownCollection {
                    backend: key.0.as_ref().to_string(),
                    collection: key.1.as_ref().to_string(),
                }
            })?;
            if query.len() != entry.dim {
                return Err(crate::RuLakeError::DimensionMismatch {
                    expected: entry.dim,
                    actual: query.len(),
                });
            }
            entry.last_used = Instant::now();
            (
                Arc::clone(&entry.index),
                Arc::clone(&entry.pos_to_id),
                entry.dim,
            )
        };
        // `dim` check was above under the lock so the clone is safe.
        let _ = dim;
        let hits = match rerank_factor_override {
            None => index.search(query, k)?,
            Some(rf) => index.search_with_rerank(query, k, rf)?,
        };
        Ok(hits
            .into_iter()
            .map(|r| (pos_to_id[r.id], r.score))
            .collect())
    }

    /// Batch form of [`search_cached_with_rerank`]: all `queries` hit
    /// the same cached entry, so we take the mutex once and look up
    /// the witness + pos_to_id mapping once. This is the surface a
    /// GPU / SIMD kernel needs to amortize per-query setup; today's
    /// CPU impl already saves the repeated mutex acquisition and
    /// eliminates per-query `ensure_fresh` calls from the caller.
    ///
    /// Preserves the query order: `result[i]` is the top-k for
    /// `queries[i]`.
    pub fn search_cached_batch(
        &self,
        key: &CacheKey,
        queries: &[Vec<f32>],
        k: usize,
        rerank_factor_override: Option<usize>,
    ) -> crate::Result<Vec<Vec<(u64, f32)>>> {
        let interned = intern_cache_key(key);
        self.search_cached_batch_interned(&interned, queries, k, rerank_factor_override)
    }

    pub(crate) fn search_cached_batch_interned(
        &self,
        key: &InternedKey,
        queries: &[Vec<f32>],
        k: usize,
        rerank_factor_override: Option<usize>,
    ) -> crate::Result<Vec<Vec<(u64, f32)>>> {
        // Lock-once pattern: validate + clone Arcs, drop the mutex,
        // run the N scans unlocked. Concurrent batches against the
        // same or different keys parallelize — the scan is pure CPU.
        let (index, pos_to_id) = {
            let mut inner = self.inner.lock().unwrap();
            let witness = inner
                .pointers
                .get(key)
                .ok_or_else(|| crate::RuLakeError::UnknownCollection {
                    backend: key.0.as_ref().to_string(),
                    collection: key.1.as_ref().to_string(),
                })?
                .clone();
            let entry = inner.entries.get_mut(&witness).ok_or_else(|| {
                crate::RuLakeError::UnknownCollection {
                    backend: key.0.as_ref().to_string(),
                    collection: key.1.as_ref().to_string(),
                }
            })?;
            let dim = entry.dim;
            for q in queries {
                if q.len() != dim {
                    return Err(crate::RuLakeError::DimensionMismatch {
                        expected: dim,
                        actual: q.len(),
                    });
                }
            }
            entry.last_used = Instant::now();
            (Arc::clone(&entry.index), Arc::clone(&entry.pos_to_id))
        };
        let mut raw: Vec<Vec<ruvector_rabitq::SearchResult>> = Vec::with_capacity(queries.len());
        for q in queries {
            let r = match rerank_factor_override {
                None => index.search(q, k)?,
                Some(rf) => index.search_with_rerank(q, k, rf)?,
            };
            raw.push(r);
        }
        Ok(raw
            .into_iter()
            .map(|v| v.into_iter().map(|r| (pos_to_id[r.id], r.score)).collect())
            .collect())
    }

    pub fn touch(&self, key: &CacheKey) {
        let interned = intern_cache_key(key);
        self.touch_interned(&interned);
    }

    pub(crate) fn touch_interned(&self, key: &InternedKey) {
        let mut inner = self.inner.lock().unwrap();
        inner.last_checked.insert(key.clone(), Instant::now());
    }

    pub fn can_skip_check(&self, key: &CacheKey, consistency: Consistency) -> bool {
        let interned = intern_cache_key(key);
        self.can_skip_check_interned(&interned, consistency)
    }

    pub(crate) fn can_skip_check_interned(
        &self,
        key: &InternedKey,
        consistency: Consistency,
    ) -> bool {
        match consistency {
            Consistency::Fresh => false,
            Consistency::Eventual { ttl_ms } => {
                let inner = self.inner.lock().unwrap();
                match inner.last_checked.get(key) {
                    Some(t) => t.elapsed().as_millis() < ttl_ms as u128,
                    None => false,
                }
            }
            // Frozen skips the coherence check iff the pointer is
            // already installed — we still need the first prime. After
            // that the caller has asserted immutability, so we never
            // round-trip to the backend again.
            Consistency::Frozen => {
                let inner = self.inner.lock().unwrap();
                inner.pointers.contains_key(key)
            }
        }
    }
}
