//! The public `RuLake` entry point — registers backends, routes searches.
//!
//! v1 fans out across registered backends, runs RaBitQ-cache searches,
//! and merges the results by score. v2 will push-down to backends that
//! support native vector ops.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ruvector_rabitq::AnnIndex;

use crate::backend::{BackendAdapter, BackendId};
use crate::cache::{intern_key, CacheKey, Consistency, InternedKey, VectorCache};
use crate::error::{Result, RuLakeError};

/// Canonical filename for the persisted RaBitQ index written by
/// [`RuLake::save_cache_to_dir`] and read back by
/// [`RuLake::warm_from_dir`]. Sits alongside `table.rulake.json` in the
/// same directory; the pair is the portable "primed collection
/// snapshot" operators hand from warm-shutdown to warm-restart.
const PERSISTED_INDEX_FILENAME: &str = "index.rbpx";

/// Result from a search — the external id and its estimated L2² score.
/// Includes the backend that produced the hit so callers can audit.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub backend: BackendId,
    pub collection: String,
    pub id: u64,
    pub score: f32,
}

/// Outcome of [`RuLake::refresh_from_bundle_dir`]. A cache sidecar
/// daemon decides what to log / alert on based on which variant it
/// sees — `Invalidated` is the normal "bundle just rotated" signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefreshResult {
    /// The cache pointer's witness matches the on-disk bundle.
    UpToDate,
    /// Witnesses differed; the cache entry for this key was dropped.
    Invalidated,
    /// No sidecar was found at the target directory.
    BundleMissing,
}

/// ruLake entry point. Cheap to clone (everything is behind `Arc`).
#[derive(Clone)]
pub struct RuLake {
    backends: Arc<std::sync::RwLock<HashMap<BackendId, Arc<dyn BackendAdapter>>>>,
    cache: Arc<VectorCache>,
    consistency: Consistency,
}

impl RuLake {
    /// Build a fresh ruLake. `rerank_factor` controls the RaBitQ cache
    /// precision (20 → 100% recall@10 on clustered D=128 at n ≥ 50k per
    /// `ruvector-rabitq::BENCHMARK.md`). `rotation_seed` is shared across
    /// all cached collections so the compression is deterministic
    /// (important for the reproducibility + witness story).
    pub fn new(rerank_factor: usize, rotation_seed: u64) -> Self {
        Self {
            backends: Arc::new(std::sync::RwLock::new(HashMap::new())),
            cache: Arc::new(VectorCache::new(rerank_factor, rotation_seed)),
            consistency: Consistency::default(),
        }
    }

    /// Set the cache consistency mode. Defaults to `Consistency::Fresh`.
    pub fn with_consistency(mut self, c: Consistency) -> Self {
        self.consistency = c;
        self
    }

    /// Cap the cache at `n` distinct compressed entries (LRU eviction
    /// over *unpinned* entries). Unbounded by default. Useful in
    /// serving processes where memory is finite.
    pub fn with_max_cache_entries(self, n: usize) -> Self {
        Self {
            cache: Arc::new(
                VectorCache::new(self.cache.rerank_factor(), self.cache.rotation_seed())
                    .with_max_entries(n),
            ),
            backends: self.backends,
            consistency: self.consistency,
        }
    }

    /// Register a backend under its `id()`. Returns an error if a backend
    /// with the same id already exists.
    pub fn register_backend(&self, backend: Arc<dyn BackendAdapter>) -> Result<()> {
        let mut map = self.backends.write().unwrap();
        let id = backend.id().to_string();
        if map.contains_key(&id) {
            return Err(RuLakeError::InvalidParameter(format!(
                "backend {id} already registered"
            )));
        }
        map.insert(id, backend);
        Ok(())
    }

    pub fn backend_ids(&self) -> Vec<BackendId> {
        self.backends.read().unwrap().keys().cloned().collect()
    }

    /// Access the cache stats for diagnostics / benchmarking.
    pub fn cache_stats(&self) -> crate::CacheStats {
        self.cache.stats()
    }

    /// Per-backend cache stats. Empty backends are not in the map.
    /// Lets operators see which backend is hot (high hit_rate) vs
    /// cold (high miss+prime) without having to attribute global
    /// counters manually. Useful for capacity planning and kernel
    /// dispatch decisions (ADR-157).
    pub fn cache_stats_by_backend(
        &self,
    ) -> std::collections::HashMap<crate::backend::BackendId, crate::cache::PerBackendStats> {
        self.cache.stats_by_backend()
    }

    /// Per-`(backend, collection)` cache stats. One level finer than
    /// `cache_stats_by_backend`. Operators use this to identify hot
    /// collections for per-collection LRU pinning or eviction.
    pub fn cache_stats_by_collection(
        &self,
    ) -> std::collections::HashMap<crate::cache::CacheKey, crate::cache::PerBackendStats> {
        self.cache.stats_by_collection()
    }

    /// What witness resolves from a `(backend, collection)` pair?
    /// Useful for diagnostics and cross-backend cache-sharing tests.
    pub fn cache_witness_of(&self, key: &CacheKey) -> Option<String> {
        self.cache.witness_of(key)
    }

    /// How many distinct compressed-index entries live in the cache?
    /// Smaller than the pointer count when multiple backends share a
    /// witness.
    pub fn cache_entry_count(&self) -> usize {
        self.cache.entry_count()
    }

    /// How many external pointers currently resolve to this witness?
    /// Returns 0 for unknown witnesses.
    pub fn cache_refcount_of(&self, witness: &str) -> u32 {
        self.cache.refcount_of(witness)
    }

    /// Drop the cache pointer for a given `(backend, collection)` pair.
    /// If this was the last pointer at the witness, the underlying
    /// compressed entry is garbage-collected.
    pub fn invalidate_cache(&self, key: &CacheKey) {
        self.cache.invalidate(key);
    }

    /// Publish the current bundle for `(backend, collection)` to `dir`
    /// as `table.rulake.json`. Pairs with `RuLakeBundle::read_from_dir`:
    /// a cache sidecar daemon can watch the target directory and hand
    /// the updated bundle to another ruLake instance, enabling
    /// warehouse-driven cache refreshes without the reader having to
    /// poll the backend directly.
    ///
    /// Returns the path of the written sidecar so callers can log /
    /// audit the publish step.
    pub fn publish_bundle(
        &self,
        key: &CacheKey,
        dir: impl AsRef<std::path::Path>,
    ) -> Result<std::path::PathBuf> {
        let backend = self.get_backend(&key.0)?;
        let bundle = backend.current_bundle(
            &key.1,
            self.cache.rotation_seed(),
            self.cache.rerank_factor(),
        )?;
        bundle.write_to_dir(dir)
    }

    /// Refresh the cache for `key` against a published bundle sidecar in
    /// `dir`. This is the reader side of the sidecar protocol: a cache
    /// daemon watches a publish directory (e.g. a GCS prefix mounted
    /// locally), and when a fresh `table.rulake.json` lands, calls this
    /// to invalidate stale entries so the next search primes against
    /// the new generation.
    ///
    /// Returns:
    /// - `RefreshResult::UpToDate` when the on-disk bundle's witness
    ///   matches what the cache currently points at — nothing to do.
    /// - `RefreshResult::Invalidated` when the witnesses differ; the
    ///   cache pointer for `key` has been dropped and the next search
    ///   will re-prime.
    /// - `RefreshResult::BundleMissing` when the sidecar isn't there —
    ///   caller decides whether that's expected (not yet published) or
    ///   an error.
    ///
    /// Rejects a tampered sidecar with `InvalidParameter` so a corrupt
    /// publish doesn't silently poison the cache.
    pub fn refresh_from_bundle_dir(
        &self,
        key: &CacheKey,
        dir: impl AsRef<std::path::Path>,
    ) -> Result<RefreshResult> {
        let dir = dir.as_ref();
        let path = dir.join(crate::RuLakeBundle::SIDECAR_FILENAME);
        if !path.exists() {
            return Ok(RefreshResult::BundleMissing);
        }
        let bundle = crate::RuLakeBundle::read_from_dir(dir)?;
        let current = self.cache.witness_of(key);
        match current.as_deref() {
            Some(w) if w == bundle.rvf_witness.as_str() => Ok(RefreshResult::UpToDate),
            Some(_) => {
                // Live cache pointer, but it disagrees with the
                // published bundle → rotate it out.
                self.cache.invalidate(key);
                Ok(RefreshResult::Invalidated)
            }
            None => {
                // Cache pointer is empty. Nothing to invalidate; the
                // next search will prime transparently. Report
                // UpToDate so daemons don't re-fire for every poll
                // between "we invalidated" and "somebody queried".
                Ok(RefreshResult::UpToDate)
            }
        }
    }

    /// Snapshot a primed cache entry to `dir/index.rbpx` plus the
    /// companion bundle sidecar `dir/table.rulake.json`. Pairs with
    /// [`Self::warm_from_dir`] to give operators a warm-restart path
    /// that skips the backend round-trip + RaBitQ compression on boot.
    ///
    /// Flow:
    ///
    /// 1. Resolve the witness currently held for `key` — errors with
    ///    `UnknownCollection` if nothing is primed (the cache-first
    ///    operator story assumes the entry exists; we never
    ///    opportunistically prime on save).
    /// 2. Look up the compressed `Arc<RabitqPlusIndex>` out of the
    ///    cache (fails `UnknownCollection` if the pointer is dangling
    ///    — shouldn't happen but is checked defensively).
    /// 3. `export_items()` → call `ruvector_rabitq::persist::save_index`
    ///    with `(idx, cache.rotation_seed(), items)`. The seed is
    ///    pulled from the cache because the on-disk format's witness
    ///    chain requires it for deterministic rebuild.
    /// 4. Atomically rename the temp file into place so readers never
    ///    observe a half-written index.
    /// 5. Emit `publish_bundle(key, dir)` so the bundle sidecar
    ///    accompanies the `.rbpx` — `warm_from_dir` cross-checks the
    ///    two before installing.
    ///
    /// Returns the path of the written `.rbpx` file; the sidecar lives
    /// next to it at `dir/table.rulake.json`. Directory creation is
    /// transitive, mirroring `publish_bundle`'s behaviour.
    ///
    /// Errors:
    /// - `UnknownBackend` if the key's backend isn't registered
    ///   (required for the bundle publish step).
    /// - `UnknownCollection` if nothing is primed under `key`.
    /// - `InvalidParameter` on any filesystem or serialization failure.
    pub fn save_cache_to_dir(&self, key: &CacheKey, dir: impl AsRef<Path>) -> Result<PathBuf> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| {
            RuLakeError::InvalidParameter(format!(
                "save_cache_to_dir: mkdir {}: {e}",
                dir.display()
            ))
        })?;

        // Step 1 — resolve witness.
        let witness = self
            .cache
            .witness_of(key)
            .ok_or_else(|| RuLakeError::UnknownCollection {
                backend: key.0.clone(),
                collection: key.1.clone(),
            })?;

        // Step 2 — clone the Arc<RabitqPlusIndex> out of the cache.
        let (idx, _pos_to_id) = self.cache.index_and_ids_of(&witness).ok_or_else(|| {
            RuLakeError::UnknownCollection {
                backend: key.0.clone(),
                collection: key.1.clone(),
            }
        })?;

        // Step 3 — serialize via rabitq's persist API.
        let items = idx.export_items();
        let final_path = dir.join(PERSISTED_INDEX_FILENAME);
        let tmp_path = dir.join(format!(
            ".{PERSISTED_INDEX_FILENAME}.tmp.{}",
            std::process::id()
        ));
        {
            let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
                RuLakeError::InvalidParameter(format!(
                    "save_cache_to_dir: create {}: {e}",
                    tmp_path.display()
                ))
            })?;
            let mut buf = std::io::BufWriter::new(&mut f);
            ruvector_rabitq::persist::save_index(
                idx.as_ref(),
                self.cache.rotation_seed(),
                &items,
                &mut buf,
            )?;
            use std::io::Write;
            buf.flush().map_err(|e| {
                RuLakeError::InvalidParameter(format!("save_cache_to_dir: flush: {e}"))
            })?;
            drop(buf);
            f.sync_all().map_err(|e| {
                RuLakeError::InvalidParameter(format!("save_cache_to_dir: fsync: {e}"))
            })?;
        }
        // Step 4 — atomic rename.
        std::fs::rename(&tmp_path, &final_path).map_err(|e| {
            RuLakeError::InvalidParameter(format!(
                "save_cache_to_dir: rename {} → {}: {e}",
                tmp_path.display(),
                final_path.display()
            ))
        })?;

        // Step 5 — companion sidecar. Reuses the existing publish path
        // so the on-disk format is identical to the "cache sidecar
        // daemon publishes to GCS" flow.
        self.publish_bundle(key, dir)?;

        Ok(final_path)
    }

    /// Inverse of [`Self::save_cache_to_dir`]. Reads the bundle sidecar
    /// plus the `index.rbpx` at `dir`, verifies the witness, and installs
    /// the loaded index into the cache under the pointer for `key`. No
    /// backend round-trip, no RaBitQ compression — the primed entry pops
    /// back into reality in O(file-read) time.
    ///
    /// Returns the number of vectors loaded (equal to the
    /// `RabitqPlusIndex::len()` of the installed entry). Operators use
    /// this to confirm the snapshot they expected is the one that
    /// landed.
    ///
    /// Flow:
    ///
    /// 1. `RuLakeBundle::read_from_dir` — opens + witness-verifies
    ///    `table.rulake.json`. A tampered sidecar is rejected here
    ///    before we touch the index.
    /// 2. `ruvector_rabitq::persist::load_index(dir/index.rbpx)`.
    ///    Malformed / oversize inputs are rejected by the persist
    ///    layer.
    /// 3. Cross-check: the loaded index's `dim()` must match the
    ///    bundle's `dim`, and its `rerank_factor()` must match the
    ///    bundle's `rerank_factor`. The witness covers these fields,
    ///    but a belt-and-braces check catches a future format that
    ///    decouples them.
    /// 4. Build `pos_to_id` from `idx.ids()` (each u32 widens to u64).
    /// 5. `cache.install_prebuilt` under the bundle's `rvf_witness`.
    ///    The cache's shared-witness fast path means two operators
    ///    warming the same bundle share one compressed entry.
    ///
    /// Deliberately does *not* require the backend to be registered —
    /// this is the warm-restart path, and the operator might warm the
    /// cache before the backend controller is wired. Any subsequent
    /// `Consistency::Fresh` search against `key` will fail with
    /// `UnknownBackend` at that point, which is the same behaviour as
    /// a cold cache with a missing backend. `Consistency::Frozen`
    /// serves out of the installed entry without ever asking.
    ///
    /// Errors:
    /// - `InvalidParameter` if the sidecar or index file is missing,
    ///   malformed, or carries a mismatched witness/dim/rerank_factor.
    /// - `Rabitq` on any RaBitQ-layer failure (propagated from
    ///   `persist::load_index`).
    pub fn warm_from_dir(&self, key: &CacheKey, dir: impl AsRef<Path>) -> Result<usize> {
        let dir = dir.as_ref();

        // Step 1 — sidecar (witness-verified inside read_from_dir).
        let bundle = crate::RuLakeBundle::read_from_dir(dir)?;

        // Step 2 — load the compressed index.
        let index_path = dir.join(PERSISTED_INDEX_FILENAME);
        if !index_path.exists() {
            return Err(RuLakeError::InvalidParameter(format!(
                "warm_from_dir: missing {}",
                index_path.display()
            )));
        }
        let file = std::fs::File::open(&index_path).map_err(|e| {
            RuLakeError::InvalidParameter(format!(
                "warm_from_dir: open {}: {e}",
                index_path.display()
            ))
        })?;
        let mut reader = std::io::BufReader::new(file);
        let idx = ruvector_rabitq::persist::load_index(&mut reader)?;

        // Step 3 — bundle vs index cross-check. The witness chain
        // already proves the bundle is self-consistent, and the
        // persist format checks `(dim, seed, rerank_factor)` against
        // the items it saved — but nothing binds the persisted
        // index's seed to the *bundle's* seed without this check,
        // and a mismatched seed would silently serve wrong neighbours.
        if idx.dim() != bundle.dim {
            return Err(RuLakeError::InvalidParameter(format!(
                "warm_from_dir: index dim {} != bundle dim {}",
                idx.dim(),
                bundle.dim
            )));
        }
        if idx.rerank_factor() != bundle.rerank_factor {
            return Err(RuLakeError::InvalidParameter(format!(
                "warm_from_dir: index rerank_factor {} != bundle rerank_factor {}",
                idx.rerank_factor(),
                bundle.rerank_factor
            )));
        }

        // Step 4 — pos_to_id from the loaded index's internal ids.
        //
        // Wave-6 close: RabitqPlusIndex::ids_u64() now widens the
        // preserved u32 ids to u64 and returns them in position order,
        // so non-dense external ids (e.g. `[7, 42, 99, 2000]`) round-
        // trip faithfully through the persist format. The previous
        // `(0..n)` workaround is gone.
        let pos_to_id: Vec<u64> = idx.ids_u64();
        let n = pos_to_id.len();

        // Step 5 — install into the cache via the interned path.
        self.cache.install_prebuilt_interned(
            intern_key(&key.0, &key.1),
            bundle.rvf_witness,
            Arc::new(idx),
            Arc::new(pos_to_id),
        )?;

        Ok(n)
    }

    /// Search a single (backend, collection) pair. Handles cache
    /// miss / staleness transparently.
    pub fn search_one(
        &self,
        backend: &str,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        // Intern once per query — the memory-audit hot-path fix. Every
        // downstream cache op takes `Arc<str>` refcount bumps instead
        // of cloning `String`s (memory-audit finding #1).
        let key = intern_key(backend, collection);
        self.ensure_fresh(&key)?;
        let hits = self
            .cache
            .search_cached_with_rerank_interned(&key, query, k, None)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                backend: backend.to_string(),
                collection: collection.to_string(),
                id,
                score,
            })
            .collect())
    }

    /// Floor for per-shard rerank factor under adaptive federation.
    /// Below this, the rerank candidate set gets too small for exact
    /// L2² rerank to meaningfully separate near ties.
    const MIN_PER_SHARD_RERANK: usize = 5;

    /// Federated search: fan out to every `(backend, collection)` pair
    /// in `targets` in parallel, merge by score, return global top-k.
    ///
    /// Uses an **adaptive per-shard rerank factor** of
    /// `max(MIN_PER_SHARD_RERANK, global_rerank / K)`. Before this, a
    /// K-shard federated search paid K× the rerank cost because each
    /// shard reranked its own `rerank_factor × k` candidates — see
    /// `BENCHMARK.md` "concurrent clients × shard count". The adaptive
    /// default keeps the total pre-merge rerank budget roughly constant
    /// in K while relying on the merge step to produce the globally
    /// correct top-k.
    ///
    /// Callers who need byte-exact parity with the single-shard path
    /// should use [`Self::search_federated_with_rerank`] to pass
    /// `Some(self.cache.rerank_factor())` explicitly.
    pub fn search_federated(
        &self,
        targets: &[(&str, &str)],
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.search_federated_with_rerank(targets, query, k, None)
    }

    /// As [`Self::search_federated`], but with explicit per-shard rerank
    /// override. `None` → adaptive default (`global / K`, floored).
    /// `Some(rf)` → that exact rerank factor on every shard.
    pub fn search_federated_with_rerank(
        &self,
        targets: &[(&str, &str)],
        query: &[f32],
        k: usize,
        per_shard_rerank: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        use rayon::prelude::*;
        let shards = targets.len().max(1);
        let rerank_override = per_shard_rerank.or_else(|| {
            if shards <= 1 {
                None // single shard: no reason to override.
            } else {
                let global = self.cache.rerank_factor();
                Some((global / shards).max(Self::MIN_PER_SHARD_RERANK))
            }
        });
        // Per-shard over-request (2026-04-23 SOTA-federation agent):
        //   k' = k + ceil(sqrt(k * ln(S)))
        // Closes the data-skew gap Weaviate / Elasticsearch leave with
        // naive k-per-shard fan-out. At k=10, S=4 this gives k'=13;
        // at k=10, S=16, k'=16. Cheap insurance against one shard
        // holding disproportionately more of the true top-K.
        let k_per_shard = Self::over_request_k(k, shards);
        let shard_hits: Result<Vec<Vec<SearchResult>>> = targets
            .par_iter()
            .map(|(backend, collection)| {
                self.search_one_with_rerank(
                    backend,
                    collection,
                    query,
                    k_per_shard,
                    rerank_override,
                )
            })
            .collect();
        let mut merged: Vec<SearchResult> = shard_hits?.into_iter().flatten().collect();
        merged.sort_by(|a, b| a.score.total_cmp(&b.score));
        merged.truncate(k);
        Ok(merged)
    }

    /// Per-shard over-request for federated search:
    /// `k' = k + ceil(sqrt(k * ln(S)))`, clamped to `[k, 4k]`.
    ///
    /// The formula is the folklore rule cited in the 2024-2025
    /// federated-ANN literature (see SPIRE, HARMONY, OpenSearch
    /// recall guide). Over-requests enough to cover data-skew across
    /// shards without pulling so many rows that rerank cost explodes.
    /// Single-shard (S=1) returns k unchanged.
    #[inline]
    fn over_request_k(k: usize, shards: usize) -> usize {
        if shards <= 1 || k == 0 {
            return k;
        }
        let bonus = ((k as f64) * (shards as f64).ln()).sqrt().ceil() as usize;
        (k + bonus).min(k.saturating_mul(4))
    }

    /// Like [`search_one`] but with an optional per-call rerank override.
    /// The federated path uses this to fan out with a reduced rerank
    /// budget per shard.
    fn search_one_with_rerank(
        &self,
        backend: &str,
        collection: &str,
        query: &[f32],
        k: usize,
        rerank_override: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        let key = intern_key(backend, collection);
        self.ensure_fresh(&key)?;
        let hits =
            self.cache
                .search_cached_with_rerank_interned(&key, query, k, rerank_override)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                backend: backend.to_string(),
                collection: collection.to_string(),
                id,
                score,
            })
            .collect())
    }

    /// Batched single-collection search. All `queries` run against the
    /// same `(backend, collection)` key, so coherence is checked once
    /// and the cache mutex is held for a single round of N scans
    /// instead of N separate acquires. Preserves query order:
    /// `result[i]` is the top-k for `queries[i]`.
    ///
    /// This is also the plug-point for the future `VectorKernel` trait
    /// (ADR-157): GPU / SIMD kernels cross over CPU only at batch
    /// sizes above their `min_batch`, so a per-query API would never
    /// let dispatch pick them. The batch API makes the dispatch
    /// decision tractable.
    pub fn search_batch(
        &self,
        backend: &str,
        collection: &str,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let key = intern_key(backend, collection);
        self.ensure_fresh(&key)?;
        let raw = self
            .cache
            .search_cached_batch_interned(&key, queries, k, None)?;
        Ok(raw
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .map(|(id, score)| SearchResult {
                        backend: backend.to_string(),
                        collection: collection.to_string(),
                        id,
                        score,
                    })
                    .collect()
            })
            .collect())
    }

    /// Coherence check: ask the backend for its current bundle and
    /// compare its witness with whatever the cache currently points at.
    ///
    /// Four cases:
    ///
    /// 1. Fast path, Eventual within TTL → skip check entirely.
    /// 2. Witness matches the cache pointer → hit, nothing to do.
    /// 3. Witness mismatch, but target witness is already in the
    ///    entry pool (cached under another pointer) → just move the
    ///    pointer, zero prime work. This is the cross-backend share.
    /// 4. Witness not in the pool → pull + prime.
    fn ensure_fresh(&self, key: &InternedKey) -> Result<()> {
        // The hot path already owns Arc<str> clones of (backend,
        // collection) — every downstream cache op is a refcount bump,
        // never a String::clone. Memory-audit finding #1.
        if self.cache.can_skip_check_interned(key, self.consistency) {
            self.cache.mark_hit(key);
            return Ok(());
        }

        let backend = self.get_backend(&key.0)?;
        let bundle = backend.current_bundle(
            &key.1,
            self.cache.rotation_seed(),
            self.cache.rerank_factor(),
        )?;
        let target_witness = bundle.rvf_witness.clone();

        if self.cache.witness_of_interned(key).as_deref() == Some(target_witness.as_str()) {
            // Case 2: pointer up-to-date.
            self.cache.mark_hit(key);
            self.cache.touch_interned(key);
            return Ok(());
        }

        // Cases 3 + 4 are handled in `prime_interned`: it reuses an
        // existing entry for the target witness if present, or builds
        // a new one.
        self.cache.mark_miss(key);
        let batch = backend.pull_vectors(&key.1)?;
        // Clone the Arcs (refcount bumps) to hand the cache an owned
        // InternedKey — no String allocation.
        let owned_key: InternedKey = (Arc::clone(&key.0), Arc::clone(&key.1));
        self.cache
            .prime_interned(owned_key, target_witness, batch)?;
        Ok(())
    }

    fn get_backend(&self, id: &str) -> Result<Arc<dyn BackendAdapter>> {
        self.backends
            .read()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| RuLakeError::UnknownBackend(id.to_string()))
    }
}
