//! Main RvfStore API — the primary user-facing interface.
//!
//! Ties together the write path, read path, indexing, deletion, and
//! compaction into a single cohesive store.

use std::collections::BinaryHeap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use rvf_types::dashboard::{DashboardHeader, DASHBOARD_MAGIC, DASHBOARD_MAX_SIZE};
use rvf_types::ebpf::{EbpfHeader, EBPF_MAGIC};
use rvf_types::kernel::{KernelHeader, KERNEL_MAGIC};
use rvf_types::kernel_binding::KernelBinding;
use rvf_types::wasm_bootstrap::{WasmHeader, WasmRole, WASM_MAGIC};
use rvf_types::{
    DomainProfile, ErrorCode, FileIdentity, RvfError, SegmentType, SEGMENT_HEADER_SIZE,
    SEGMENT_MAGIC,
};

use crate::cow::{CowEngine, CowStats};
use crate::deletion::DeletionBitmap;
use crate::filter::{self, metadata_value_to_filter, FilterExpr, FilterValue, MetadataStore};
use crate::index_path::{
    VectorIndex, INDEX_MAX_DELETED_FRACTION, INDEX_MIN_EF_SEARCH, INDEX_MIN_VECTORS,
};
use crate::locking::WriterLock;
use crate::membership::MembershipFilter;
use crate::options::*;
use crate::rabitq_path::RabitqState;
use crate::read_path::{self, VectorData};
use crate::status::{CompactionState, StoreStatus};
use crate::write_path::SegmentWriter;

/// Helper to convert any error into an RvfError with the given code.
fn err(code: ErrorCode) -> RvfError {
    RvfError::Code(code)
}

/// Witness type discriminators matching rvf-crypto's WitnessType.
/// Kept here to avoid a hard dependency on rvf-crypto in the runtime.
mod witness_types {
    /// Data provenance witness (tracks data origin and lineage).
    pub const DATA_PROVENANCE: u8 = 0x00;
    /// Computation witness (tracks processing / transform operations).
    pub const COMPUTATION: u8 = 0x01;
}

/// The main RVF store handle.
///
/// Provides create, open, ingest, query, delete, compact, and close.
pub struct RvfStore {
    path: PathBuf,
    options: RvfOptions,
    file: File,
    seg_writer: Option<SegmentWriter>,
    writer_lock: Option<WriterLock>,
    vectors: VectorData,
    deletion_bitmap: DeletionBitmap,
    metadata: MetadataStore,
    epoch: u32,
    segment_dir: Vec<(u64, u64, u64, u8)>,
    read_only: bool,
    last_compaction_time: u64,
    file_identity: FileIdentity,
    /// COW engine for branched/snapshot stores (None for root stores).
    cow_engine: Option<CowEngine>,
    /// Membership filter for branch-level vector visibility (None if unused).
    membership_filter: Option<MembershipFilter>,
    /// Path to the parent file (for COW reads that need parent data).
    parent_path: Option<PathBuf>,
    /// Hash of the last witness entry, used to chain-link successive witnesses.
    /// All zeros when no witness has been written yet (genesis).
    last_witness_hash: [u8; 32],
    /// In-memory HNSW index over the stored vectors (None until loaded
    /// from an INDEX_SEG or built on the first eligible query). Guarded by
    /// a Mutex so `query(&self)` can build/maintain it lazily.
    index: Mutex<Option<VectorIndex>>,
    /// True while one query thread is (re)building the HNSW index OUTSIDE
    /// the `index` mutex. Other query threads fall back to the exact scan
    /// instead of blocking behind an O(N log N) build (audit finding 5:
    /// overwrite invalidation must not cause head-of-line blocking).
    index_building: AtomicBool,
    /// In-memory RaBitQ codes for the opt-in two-stage query path (None
    /// until the first `QueryOptions::rabitq` query builds it lazily).
    rabitq: Mutex<Option<RabitqState>>,
    /// Same single-builder gate as `index_building`, for the RaBitQ code
    /// book (its lazy build is an O(N) scan + encode).
    rabitq_building: AtomicBool,
}

/// Clears an `AtomicBool` on drop, so a panicking index build can never
/// leave the "building" gate latched (queries would silently fall back to
/// the exact scan forever).
struct ClearOnDrop<'a>(&'a AtomicBool);

impl Drop for ClearOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl RvfStore {
    /// Create a new RVF store at the given path.
    pub fn create(path: &Path, options: RvfOptions) -> Result<Self, RvfError> {
        if options.dimension == 0 {
            return Err(err(ErrorCode::InvalidManifest));
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|_| err(ErrorCode::FsyncFailed))?;

        let writer_lock = WriterLock::acquire(path).map_err(|_| err(ErrorCode::LockHeld))?;

        // Generate a random file_id from path hash + timestamp
        let file_id = generate_file_id(path);

        // Detect domain profile from file extension
        let domain_profile = path
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(DomainProfile::from_extension)
            .unwrap_or(options.domain_profile);

        let mut opts = options.clone();
        opts.domain_profile = domain_profile;

        let mut store = Self {
            path: path.to_path_buf(),
            options: opts,
            file,
            seg_writer: Some(SegmentWriter::new(1)),
            writer_lock: Some(writer_lock),
            vectors: VectorData::new(options.dimension),
            deletion_bitmap: DeletionBitmap::new(),
            metadata: MetadataStore::new(),
            epoch: 0,
            segment_dir: Vec::new(),
            read_only: false,
            last_compaction_time: 0,
            file_identity: FileIdentity::new_root(file_id),
            cow_engine: None,
            membership_filter: None,
            parent_path: None,
            last_witness_hash: [0u8; 32],
            index: Mutex::new(None),
            index_building: AtomicBool::new(false),
            rabitq: Mutex::new(None),
            rabitq_building: AtomicBool::new(false),
        };

        store.write_manifest()?;
        Ok(store)
    }

    /// Open an existing RVF store for read-write access.
    pub fn open(path: &Path) -> Result<Self, RvfError> {
        if !path.exists() {
            return Err(err(ErrorCode::ManifestNotFound));
        }

        let writer_lock = WriterLock::acquire(path).map_err(|_| err(ErrorCode::LockHeld))?;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|_| err(ErrorCode::InvalidManifest))?;

        // Detect domain profile from extension
        let domain_profile = path
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(DomainProfile::from_extension)
            .unwrap_or(DomainProfile::Generic);

        let opts = RvfOptions {
            domain_profile,
            ..Default::default()
        };

        let mut store = Self {
            path: path.to_path_buf(),
            options: opts,
            file,
            seg_writer: None,
            writer_lock: Some(writer_lock),
            vectors: VectorData::new(0),
            deletion_bitmap: DeletionBitmap::new(),
            metadata: MetadataStore::new(),
            epoch: 0,
            segment_dir: Vec::new(),
            read_only: false,
            last_compaction_time: 0,
            file_identity: FileIdentity::zeroed(),
            cow_engine: None,
            membership_filter: None,
            parent_path: None,
            last_witness_hash: [0u8; 32],
            index: Mutex::new(None),
            index_building: AtomicBool::new(false),
            rabitq: Mutex::new(None),
            rabitq_building: AtomicBool::new(false),
        };

        store.boot()?;
        Ok(store)
    }

    /// Open an existing RVF store for read-only access (no lock required).
    pub fn open_readonly(path: &Path) -> Result<Self, RvfError> {
        if !path.exists() {
            return Err(err(ErrorCode::ManifestNotFound));
        }

        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|_| err(ErrorCode::InvalidManifest))?;

        let domain_profile = path
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(DomainProfile::from_extension)
            .unwrap_or(DomainProfile::Generic);

        let opts = RvfOptions {
            domain_profile,
            ..Default::default()
        };

        let mut store = Self {
            path: path.to_path_buf(),
            options: opts,
            file,
            seg_writer: None,
            writer_lock: None,
            vectors: VectorData::new(0),
            deletion_bitmap: DeletionBitmap::new(),
            metadata: MetadataStore::new(),
            epoch: 0,
            segment_dir: Vec::new(),
            read_only: true,
            last_compaction_time: 0,
            file_identity: FileIdentity::zeroed(),
            cow_engine: None,
            membership_filter: None,
            parent_path: None,
            last_witness_hash: [0u8; 32],
            index: Mutex::new(None),
            index_building: AtomicBool::new(false),
            rabitq: Mutex::new(None),
            rabitq_building: AtomicBool::new(false),
        };

        store.boot()?;
        Ok(store)
    }

    /// Ingest a batch of vectors into the store.
    pub fn ingest_batch(
        &mut self,
        vectors: &[&[f32]],
        ids: &[u64],
        metadata: Option<&[MetadataEntry]>,
    ) -> Result<IngestResult, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }
        if vectors.len() != ids.len() {
            return Err(err(ErrorCode::DimensionMismatch));
        }

        let dim = self.options.dimension as usize;
        let mut accepted = 0u64;
        let mut rejected = 0u64;

        let mut valid_vectors: Vec<&[f32]> = Vec::with_capacity(vectors.len());
        let mut valid_ids: Vec<u64> = Vec::with_capacity(ids.len());

        for (i, &vec_data) in vectors.iter().enumerate() {
            if vec_data.len() != dim {
                rejected += 1;
                continue;
            }
            valid_vectors.push(vec_data);
            valid_ids.push(ids[i]);
            accepted += 1;
        }

        if valid_vectors.is_empty() {
            self.epoch += 1;
            return Ok(IngestResult {
                accepted: 0,
                rejected,
                epoch: self.epoch,
            });
        }

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;

        let (vec_seg_id, vec_seg_offset) = {
            let mut buf_writer = BufWriter::with_capacity(256 * 1024, &self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            writer
                .write_vec_seg(
                    &mut buf_writer,
                    &valid_vectors,
                    &valid_ids,
                    self.options.dimension,
                )
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        let bytes_per_vec = (self.options.dimension as usize) * 4;
        let vec_payload_len = (2 + 4 + valid_vectors.len() * (8 + bytes_per_vec)) as u64;

        self.segment_dir.push((
            vec_seg_id,
            vec_seg_offset,
            vec_payload_len,
            SegmentType::Vec as u8,
        ));

        for (vec_data, &vec_id) in valid_vectors.iter().zip(valid_ids.iter()) {
            self.vectors.insert_slice(vec_id, vec_data);
        }

        // Keep the in-memory HNSW index in sync with the new vectors.
        {
            let mut guard = self.index.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(idx) = guard.as_mut() {
                if valid_ids.iter().any(|&id| idx.contains(id)) {
                    // An existing ID was overwritten: the graph's edges for
                    // that node are stale. Drop the index (it is rebuilt
                    // lazily) and unlink any persisted INDEX_SEG so a
                    // reopen does not load the stale graph.
                    *guard = None;
                    self.segment_dir
                        .retain(|&(_, _, _, stype)| stype != SegmentType::Index as u8);
                } else {
                    let mut new_ids = valid_ids.clone();
                    new_ids.sort_unstable();
                    idx.insert_ids(&new_ids, &self.vectors, self.options.metric);
                }
            }
        }

        // RaBitQ codes for overwritten IDs are stale: drop the state (it
        // is rebuilt lazily). New IDs are encoded lazily by sync_missing.
        {
            let mut guard = self.rabitq.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(state) = guard.as_ref() {
                if valid_ids.iter().any(|&id| state.contains(id)) {
                    *guard = None;
                }
            }
        }

        if let Some(meta_entries) = metadata {
            let entries_per_id = meta_entries.len() / valid_ids.len().max(1);
            if entries_per_id > 0 {
                for (i, &vid) in valid_ids.iter().enumerate() {
                    let start = i * entries_per_id;
                    let end = ((i + 1) * entries_per_id).min(meta_entries.len());
                    let fields: Vec<(u16, FilterValue)> = meta_entries[start..end]
                        .iter()
                        .map(|e| (e.field_id, metadata_value_to_filter(&e.value)))
                        .collect();
                    self.metadata.insert(vid, fields);
                }
            }
        }

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;

        self.epoch += 1;

        // Append a witness entry recording this ingest operation.
        if self.options.witness.witness_ingest {
            let action = format!("ingest:count={},epoch={}", accepted, self.epoch);
            self.append_witness(witness_types::COMPUTATION, action.as_bytes())?;
        }

        self.write_manifest()?;

        Ok(IngestResult {
            accepted,
            rejected,
            epoch: self.epoch,
        })
    }

    /// Query the store for the k nearest neighbors of the given vector.
    ///
    /// Routing: stores with at least `INDEX_MIN_VECTORS` vectors are served
    /// by the HNSW index (loaded from an INDEX_SEG at open time, or built
    /// on the first eligible query and maintained incrementally). Smaller
    /// stores, filtered queries, COW/membership stores, and stores with a
    /// high soft-deleted fraction use the exact brute-force scan.
    pub fn query(
        &self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
    ) -> Result<Vec<SearchResult>, RvfError> {
        self.query_routed(vector, k, options).map(|(r, _)| r)
    }

    /// Internal query entry point. Returns the results plus whether the
    /// HNSW index actually served the query (for honest evidence reporting
    /// in [`Self::query_with_envelope`]).
    fn query_routed(
        &self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
    ) -> Result<(Vec<SearchResult>, bool), RvfError> {
        let dim = self.options.dimension as usize;
        if vector.len() != dim {
            return Err(err(ErrorCode::DimensionMismatch));
        }

        if self.vectors.len() == 0 {
            return Ok((Vec::new(), false));
        }

        // Opt-in RaBitQ two-stage path (binary candidate scan + exact
        // rescore). Not an HNSW-index serve, so the flag stays false.
        if self.rabitq_eligible(options) {
            if let Some(results) = self.query_via_rabitq(vector, k, options) {
                return Ok((results, false));
            }
        }

        if self.index_eligible(options) {
            if let Some(results) = self.query_via_index(vector, k, options) {
                return Ok((results, true));
            }
        }

        Ok((self.query_exact(vector, k, options), false))
    }

    /// Whether this query can be served by the opt-in RaBitQ two-stage
    /// path. v1 supports the L2 metric; filtered queries and COW /
    /// membership stores use the default routing.
    fn rabitq_eligible(&self, options: &QueryOptions) -> bool {
        options.rabitq
            && !options.force_exact
            && options.filter.is_none()
            && self.membership_filter.is_none()
            && self.cow_engine.is_none()
            && self.options.metric == DistanceMetric::L2
    }

    /// Serve a query through the RaBitQ two-stage path, building the code
    /// book on first use.
    ///
    /// Stage 1 scans the 1-bit codes with the asymmetric estimator and
    /// collects `rabitq_oversample * k` live candidates; stage 2 rescores
    /// them with exact f32 distances. Returns `None` when the candidate
    /// set cannot supply `k` live results (caller falls back).
    fn query_via_rabitq(
        &self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
    ) -> Option<Vec<SearchResult>> {
        let mut guard = self.rabitq.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            // Build OUTSIDE the lock so concurrent queries are not blocked
            // behind the O(N) encode; only one thread builds, the others
            // fall back to the default routing (audit finding 5 pattern).
            drop(guard);
            if self.rabitq_building.swap(true, Ordering::AcqRel) {
                return None;
            }
            let _clear = ClearOnDrop(&self.rabitq_building);
            let built = RabitqState::build(&self.vectors);
            guard = self.rabitq.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                *guard = built;
            }
        }
        let state = guard.as_mut()?;
        // Encode vectors ingested since the state was built.
        state.sync_missing(&self.vectors);

        let oversample = (options.rabitq_oversample.max(1)) as usize;
        // Oversample for estimator error (floored so the candidate pool
        // meets the >= 0.95 recall@10 contract, see RABITQ_MIN_CANDIDATES),
        // plus headroom for soft-deleted hits the same way the HNSW path
        // compensates.
        let deleted = self.deletion_bitmap.count();
        let k_fetch = k
            .saturating_mul(oversample)
            .max(crate::rabitq_path::RABITQ_MIN_CANDIDATES)
            .saturating_add(deleted.min(2 * k + 16));

        let candidates =
            state.candidates(vector, k_fetch, |id| !self.deletion_bitmap.is_deleted(id));

        // Stage 2: exact f32 rescore of the candidate set.
        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|(id, _est)| {
                self.vectors.get(id).map(|v| SearchResult {
                    id,
                    distance: compute_distance(vector, v, &self.options.metric, 0.0),
                    retrieval_quality: rvf_types::quality::RetrievalQuality::Full,
                })
            })
            .collect();
        results.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(k);

        let live = self.vectors.len().saturating_sub(deleted);
        if results.len() < k.min(live) {
            return None;
        }
        Some(results)
    }

    /// Whether this query can be served by the HNSW index path.
    ///
    /// Filtered queries, COW/membership stores, small stores, and stores
    /// with a high deleted fraction always use the exact scan (which is
    /// both correct and faster in those regimes).
    fn index_eligible(&self, options: &QueryOptions) -> bool {
        if options.force_exact || options.filter.is_some() {
            return false;
        }
        if self.membership_filter.is_some() || self.cow_engine.is_some() {
            return false;
        }
        let total = self.vectors.len();
        if total < INDEX_MIN_VECTORS {
            return false;
        }
        let deleted = self.deletion_bitmap.count();
        (deleted as f64) <= (total as f64) * INDEX_MAX_DELETED_FRACTION
    }

    /// Serve a query through the HNSW index, building it on first use.
    ///
    /// Returns `None` when the index cannot supply `k` live results; the
    /// caller then falls back to the exact scan.
    fn query_via_index(
        &self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
    ) -> Option<Vec<SearchResult>> {
        let mut guard = self.index.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            // Audit finding 5: the O(N log N) build must not run under the
            // query mutex (head-of-line blocking after an overwrite drops
            // the index). Exactly one thread builds into a local index
            // with NO lock held, then swaps it in; concurrent queries see
            // `index_building` set and fall back to the exact scan, so
            // queries keep serving throughout the rebuild.
            drop(guard);
            if self.index_building.swap(true, Ordering::AcqRel) {
                return None;
            }
            let _clear = ClearOnDrop(&self.index_building);
            let built = VectorIndex::build(
                &self.vectors,
                self.options.metric,
                (self.options.m.max(2)) as usize,
                (self.options.ef_construction.max(16)) as usize,
            );
            guard = self.index.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                *guard = Some(built);
            }
        }
        let idx = guard.as_mut()?;
        // Insert vectors ingested since the index was built/loaded.
        idx.sync_missing(&self.vectors, self.options.metric);

        // Oversample to compensate for soft-deleted entries that may
        // appear among the nearest graph hits.
        let deleted = self.deletion_bitmap.count();
        let live = self.vectors.len().saturating_sub(deleted);
        let k_fetch = k.saturating_add(deleted.min(2 * k + 16));
        // Floor ef_search so the index path meets the >=0.95 recall@10
        // contract (see INDEX_MIN_EF_SEARCH); larger values are honored.
        let ef = (options.ef_search as usize)
            .max(k_fetch)
            .max(INDEX_MIN_EF_SEARCH);

        let hits = idx.search(vector, k_fetch, ef, &self.vectors, self.options.metric);
        let mut results: Vec<SearchResult> = hits
            .into_iter()
            .filter(|&(id, _)| !self.deletion_bitmap.is_deleted(id))
            .take(k)
            .map(|(id, distance)| SearchResult {
                id,
                distance,
                retrieval_quality: rvf_types::quality::RetrievalQuality::Full,
            })
            .collect();
        if results.len() < k.min(live) {
            // Not enough live hits after deletion filtering; let the exact
            // scan answer instead of returning an under-filled result set.
            return None;
        }
        // Preserve deterministic (distance, id) result ordering.
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        Some(results)
    }

    /// Exact brute-force k-nearest-neighbor scan over all live vectors.
    fn query_exact(&self, vector: &[f32], k: usize, options: &QueryOptions) -> Vec<SearchResult> {
        // Max-heap: peek() returns the largest (farthest) distance in our k set.
        // When a closer vector is found, evict the farthest.
        let mut heap: BinaryHeap<(OrderedFloat, u64)> = BinaryHeap::new();

        // Precompute the query's squared norm once for the cosine metric
        // instead of recomputing it for every stored vector in the scan.
        let query_norm_sq = match self.options.metric {
            DistanceMetric::Cosine => {
                let mut norm = 0.0f32;
                for x in vector {
                    norm += x * x;
                }
                norm
            }
            _ => 0.0,
        };

        // Scan the contiguous slab in ordinal order (cache-friendly: rows
        // are adjacent in memory, no per-vector pointer chase).
        for (vec_id, stored_vec) in self.vectors.iter() {
            if self.deletion_bitmap.is_deleted(vec_id) {
                continue;
            }
            if let Some(ref filter_expr) = options.filter {
                if !filter::evaluate(filter_expr, vec_id, &self.metadata) {
                    continue;
                }
            }
            let dist = compute_distance(vector, stored_vec, &self.options.metric, query_norm_sq);
            if heap.len() < k {
                heap.push((OrderedFloat(dist), vec_id));
            } else if let Some(&(OrderedFloat(worst), worst_id)) = heap.peek() {
                // Tie-break equal distances by smaller id so the selected
                // k-set is independent of storage iteration order.
                if dist < worst || (dist == worst && vec_id < worst_id) {
                    heap.pop();
                    heap.push((OrderedFloat(dist), vec_id));
                }
            }
        }

        // Drain the max-heap into sorted results (closest first).
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|(OrderedFloat(dist), id)| SearchResult {
                id,
                distance: dist,
                retrieval_quality: rvf_types::quality::RetrievalQuality::Full,
            })
            .collect();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        results
    }

    /// Query the store and return a full QualityEnvelope (ADR-033 §2.4).
    ///
    /// This is the preferred query API. The QualityEnvelope is the mandatory
    /// outer return type — consumers MUST inspect the `quality` field before
    /// using results.
    pub fn query_with_envelope(
        &self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
    ) -> Result<QualityEnvelope, RvfError> {
        use rvf_types::quality::*;
        use std::time::Instant;

        let start = Instant::now();
        let dim = self.options.dimension as usize;
        if vector.len() != dim {
            return Err(err(ErrorCode::DimensionMismatch));
        }

        // Determine effective budget based on quality preference.
        let budget = match options.quality_preference {
            QualityPreference::PreferQuality => options.safety_net_budget.extended_4x(),
            QualityPreference::PreferLatency => SafetyNetBudget::DISABLED,
            _ => options.safety_net_budget,
        };

        // Execute the base query, tracking whether the HNSW index served it.
        let (results, index_used) = self.query_routed(vector, k, options)?;
        let hnsw_candidate_count = results.len() as u32;

        // Determine if safety net should activate.
        let needs_safety_net = crate::safety_net::should_activate_safety_net(results.len(), k)
            && !budget.is_disabled();

        let mut all_results = results;
        let mut safety_net_candidate_count = 0u32;
        let mut budget_report = BudgetReport::default();
        let mut degradation: Option<DegradationReport> = None;

        if needs_safety_net && self.vectors.len() > 0 {
            // Build vector refs for safety net scan (slab rows, no copies).
            let vec_refs: Vec<(u64, &[f32])> = self
                .vectors
                .iter()
                .filter(|&(id, _)| !self.deletion_bitmap.is_deleted(id))
                .collect();

            let base_results: Vec<crate::options::SearchResult> = all_results.clone();
            let scan_result = crate::safety_net::selective_safety_net_scan(
                vector,
                k,
                &base_results,
                &vec_refs,
                &budget,
                self.vectors.len() as u64,
            );

            safety_net_candidate_count = scan_result.candidates.len() as u32;
            budget_report = scan_result.budget_report;
            degradation = scan_result.degradation;

            // Merge safety net candidates into results.
            for candidate in scan_result.candidates {
                all_results.push(SearchResult {
                    id: candidate.id,
                    distance: candidate.distance,
                    retrieval_quality: RetrievalQuality::BruteForceBudgeted,
                });
            }

            // Re-sort and take top-k.
            all_results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
            all_results.truncate(k);
        }

        let elapsed_us = start.elapsed().as_micros() as u64;
        budget_report.total_us = elapsed_us;

        // Derive response quality from all candidate qualities.
        let retrieval_qualities: Vec<RetrievalQuality> =
            all_results.iter().map(|r| r.retrieval_quality).collect();
        let quality = derive_response_quality(&retrieval_qualities);

        // Honest evidence: index layers are reported as used only when the
        // HNSW index actually served the query (not on brute-force scans).
        let evidence = SearchEvidenceSummary {
            layers_used: IndexLayersUsed {
                layer_a: index_used,
                layer_b: false,
                layer_c: index_used,
                hot_cache: needs_safety_net,
            },
            n_probe_effective: 0,
            degenerate_detected: false,
            centroid_distance_cv: 0.0,
            hnsw_candidate_count,
            safety_net_candidate_count,
        };

        let envelope = QualityEnvelope {
            results: all_results,
            quality,
            evidence,
            budgets: budget_report,
            degradation,
        };

        // Enforce quality threshold policy.
        if matches!(
            quality,
            ResponseQuality::Degraded | ResponseQuality::Unreliable
        ) && !matches!(
            options.quality_preference,
            QualityPreference::AcceptDegraded
        ) {
            return Err(RvfError::QualityBelowThreshold {
                quality,
                reason: "result quality below threshold; set AcceptDegraded to use partial results",
            });
        }

        Ok(envelope)
    }

    /// Query the store with optional audit witness.
    ///
    /// Behaves identically to [`query`] but, when `audit_queries` is enabled
    /// in the store's `WitnessConfig`, appends a WITNESS_SEG recording the
    /// query operation. Requires `&mut self` due to the file write.
    pub fn query_audited(
        &mut self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
    ) -> Result<Vec<SearchResult>, RvfError> {
        let results = self.query(vector, k, options)?;

        if self.options.witness.audit_queries && !self.read_only {
            let action = format!(
                "query:k={},results={},epoch={}",
                k,
                results.len(),
                self.epoch
            );
            self.append_witness(witness_types::COMPUTATION, action.as_bytes())?;
            // Flush the witness to disk but skip a full manifest rewrite
            // to keep query overhead minimal.
            self.file
                .sync_all()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
        }

        Ok(results)
    }

    /// Soft-delete vectors by ID.
    pub fn delete(&mut self, ids: &[u64]) -> Result<DeleteResult, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let epoch = self.epoch + 1;

        let (journal_seg_id, journal_offset) = {
            let mut buf_writer = BufWriter::new(&self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            writer
                .write_journal_seg(&mut buf_writer, ids, epoch)
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        let journal_payload_len = (16 + ids.len() * 12) as u64;
        self.segment_dir.push((
            journal_seg_id,
            journal_offset,
            journal_payload_len,
            SegmentType::Journal as u8,
        ));

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;

        let mut deleted = 0u64;
        for &id in ids {
            if self.vectors.get(id).is_some() && !self.deletion_bitmap.is_deleted(id) {
                self.deletion_bitmap.delete(id);
                deleted += 1;
            }
        }

        self.epoch = epoch;

        // Append a witness entry recording this delete operation.
        if self.options.witness.witness_delete {
            let action = format!("delete:count={},epoch={}", deleted, self.epoch);
            self.append_witness(witness_types::DATA_PROVENANCE, action.as_bytes())?;
        }

        self.write_manifest()?;

        Ok(DeleteResult {
            deleted,
            epoch: self.epoch,
        })
    }

    /// Soft-delete vectors matching a filter expression.
    pub fn delete_by_filter(&mut self, filter_expr: &FilterExpr) -> Result<DeleteResult, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }

        let matching_ids: Vec<u64> = self
            .vectors
            .ids()
            .filter(|&&id| {
                !self.deletion_bitmap.is_deleted(id)
                    && filter::evaluate(filter_expr, id, &self.metadata)
            })
            .copied()
            .collect();

        if matching_ids.is_empty() {
            return Ok(DeleteResult {
                deleted: 0,
                epoch: self.epoch,
            });
        }

        self.delete(&matching_ids)
    }

    /// Get the current store status.
    pub fn status(&self) -> StoreStatus {
        let total_vectors =
            (self.vectors.len() as u64).saturating_sub(self.deletion_bitmap.count() as u64);
        let file_size = self.file.metadata().map(|m| m.len()).unwrap_or(0);
        let dead_space_ratio = {
            let total = self.vectors.len() as f64;
            let deleted = self.deletion_bitmap.count() as f64;
            if total > 0.0 {
                deleted / total
            } else {
                0.0
            }
        };

        StoreStatus {
            total_vectors,
            total_segments: self.segment_dir.len() as u32,
            file_size,
            current_epoch: self.epoch,
            profile_id: self.options.profile,
            compaction_state: CompactionState::Idle,
            dead_space_ratio,
            read_only: self.read_only,
        }
    }

    /// Run compaction to reclaim dead space.
    ///
    /// Preserves all non-Vec, non-Manifest, non-Journal segments byte-for-byte
    /// to maintain forward compatibility with segment types this version does
    /// not understand (e.g., future Kernel, Ebpf, or vendor-extension segments).
    pub fn compact(&mut self) -> Result<CompactionResult, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }

        let deleted_ids = self.deletion_bitmap.to_sorted_ids();
        for &id in &deleted_ids {
            self.vectors.remove(id);
        }
        // Reclaim the tombstoned slab slots (the only point where slots
        // are reused, so live rows never move between mutations).
        self.vectors.compact_in_place();
        self.metadata.remove_ids(&deleted_ids);

        // Compaction removes vectors, so the in-memory HNSW index is
        // invalidated (rebuilt lazily). Persisted INDEX_SEGs are likewise
        // dropped from the rewritten file below. The RaBitQ code book
        // references removed IDs too, so it is dropped alongside.
        *self.index.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.rabitq.lock().unwrap_or_else(|e| e.into_inner()) = None;

        let segments_compacted = deleted_ids.len() as u32;
        let bytes_reclaimed = (deleted_ids.len() as u64) * (self.options.dimension as u64) * 4;

        self.deletion_bitmap.clear();

        // Read the entire original file into memory so we can scan for segments
        // that may not be in the manifest (e.g., unknown types appended by newer tools).
        let original_bytes = {
            let mut reader = BufReader::new(&self.file);
            reader
                .seek(SeekFrom::Start(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            let mut buf = Vec::new();
            reader
                .read_to_end(&mut buf)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            buf
        };

        let temp_path = self.path.with_extension("rvf.compact.tmp");
        let mut new_segment_dir = Vec::new();
        let mut seg_writer = SegmentWriter::new(1);
        {
            let temp_file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)
                .map_err(|_| err(ErrorCode::DiskFull))?;

            let mut temp_writer = BufWriter::new(&temp_file);

            let mut live_ids: Vec<u64> = Vec::with_capacity(self.vectors.len());
            let mut vec_refs: Vec<&[f32]> = Vec::with_capacity(self.vectors.len());
            for (id, row) in self.vectors.iter() {
                live_ids.push(id);
                vec_refs.push(row);
            }

            if !live_ids.is_empty() {
                let (seg_id, offset) = seg_writer
                    .write_vec_seg(
                        &mut temp_writer,
                        &vec_refs,
                        &live_ids,
                        self.options.dimension,
                    )
                    .map_err(|_| err(ErrorCode::FsyncFailed))?;

                let bytes_per_vec = (self.options.dimension as usize) * 4;
                let payload_len = (2 + 4 + live_ids.len() * (8 + bytes_per_vec)) as u64;
                new_segment_dir.push((seg_id, offset, payload_len, SegmentType::Vec as u8));
            }

            // Preserve non-Vec, non-Manifest, non-Journal segments from the
            // original file. This includes both segments recorded in the old
            // manifest and segments appended after it (e.g., unknown types from
            // newer format versions).
            let preserved = scan_preservable_segments(&original_bytes);
            for (orig_offset, seg_id, payload_len, seg_type) in &preserved {
                // Drop INDEX_SEGs: compaction changes the vector set, so a
                // preserved index would be stale. It is rebuilt from the
                // live vectors on the next eligible query.
                if *seg_type == SegmentType::Index as u8 {
                    continue;
                }
                // Use checked arithmetic for bounds safety.
                let total_bytes = match (*payload_len as usize).checked_add(SEGMENT_HEADER_SIZE) {
                    Some(t) => t,
                    None => continue, // skip segment with implausible size
                };
                let end = match orig_offset.checked_add(total_bytes) {
                    Some(e) if e <= original_bytes.len() => e,
                    _ => continue, // skip out-of-bounds segment
                };
                let src = &original_bytes[*orig_offset..end];

                // Flush the BufWriter so stream_position reflects the true offset.
                temp_writer
                    .flush()
                    .map_err(|_| err(ErrorCode::FsyncFailed))?;
                let new_offset = temp_writer
                    .stream_position()
                    .map_err(|_| err(ErrorCode::FsyncFailed))?;

                temp_writer
                    .write_all(src)
                    .map_err(|_| err(ErrorCode::FsyncFailed))?;

                // Ensure the seg_writer's next_seg_id stays above any preserved ID.
                while seg_writer.next_id() <= *seg_id {
                    seg_writer.alloc_seg_id();
                }

                new_segment_dir.push((*seg_id, new_offset, *payload_len, *seg_type));
            }

            self.epoch += 1;
            let total_vectors = live_ids.len() as u64;
            let empty_dels: Vec<u64> = Vec::new();
            let fi = if self.file_identity.file_id != [0u8; 16] {
                Some(&self.file_identity)
            } else {
                None
            };
            // Flush before writing manifest so offsets are accurate.
            temp_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            seg_writer
                .write_manifest_seg_with_identity(
                    &mut temp_writer,
                    self.epoch,
                    self.options.dimension,
                    total_vectors,
                    self.options.profile,
                    &new_segment_dir,
                    &empty_dels,
                    fi,
                )
                .map_err(|_| err(ErrorCode::FsyncFailed))?;

            temp_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            temp_file
                .sync_all()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
        }

        fs::rename(&temp_path, &self.path).map_err(|_| err(ErrorCode::FsyncFailed))?;

        // Sync parent directory to make rename durable
        if let Some(parent) = self.path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }

        self.file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|_| err(ErrorCode::InvalidManifest))?;

        self.segment_dir = new_segment_dir;
        self.seg_writer = Some(seg_writer);
        self.last_compaction_time = now_secs();

        // Reset witness chain after compaction (the file has been rewritten).
        self.last_witness_hash = [0u8; 32];

        // Append a witness entry recording this compact operation.
        if self.options.witness.witness_compact {
            let action = format!(
                "compact:segments_compacted={},bytes_reclaimed={},epoch={}",
                segments_compacted, bytes_reclaimed, self.epoch
            );
            self.append_witness(witness_types::COMPUTATION, action.as_bytes())?;
            self.file
                .sync_all()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
        }

        Ok(CompactionResult {
            segments_compacted,
            bytes_reclaimed,
            epoch: self.epoch,
        })
    }

    /// Close the store, releasing the writer lock.
    ///
    /// If the in-memory HNSW index changed since it was last persisted,
    /// it is written out as an INDEX_SEG so the next open can load it
    /// instead of rebuilding from vectors.
    pub fn close(mut self) -> Result<(), RvfError> {
        self.persist_index()?;

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;

        if let Some(lock) = self.writer_lock {
            lock.release().map_err(|_| err(ErrorCode::LockHeld))?;
        }

        Ok(())
    }

    /// True when an HNSW index is resident in memory (loaded from an
    /// INDEX_SEG at open time or built by a previous query).
    pub fn index_ready(&self) -> bool {
        self.index
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Persist the in-memory HNSW index as an INDEX_SEG if it has changed
    /// since it was last persisted (or since it was built/loaded).
    fn persist_index(&mut self) -> Result<(), RvfError> {
        if self.read_only {
            return Ok(());
        }
        let payload = {
            let mut guard = self.index.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_mut() {
                Some(idx) if idx.is_dirty() && idx.node_count() > 0 => {
                    let payload = idx.encode_payload();
                    idx.mark_clean();
                    payload
                }
                _ => return Ok(()),
            }
        };

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let (seg_id, seg_offset) = {
            let mut buf_writer = BufWriter::with_capacity(256 * 1024, &self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            writer
                .write_index_seg(&mut buf_writer, &payload)
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        // Newer INDEX_SEGs supersede older ones: keep only the latest
        // entry in the manifest (orphaned bytes are reclaimed by compact).
        self.segment_dir
            .retain(|&(_, _, _, stype)| stype != SegmentType::Index as u8);
        self.segment_dir.push((
            seg_id,
            seg_offset,
            payload.len() as u64,
            SegmentType::Index as u8,
        ));

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        self.epoch += 1;
        self.write_manifest()
    }

    // -- Kernel / eBPF embedding API --

    /// Embed a kernel image into this RVF file as a KERNEL_SEG.
    ///
    /// Builds a 128-byte KernelHeader, serializes it, then delegates to
    /// the write path. Returns the segment_id of the new KERNEL_SEG.
    pub fn embed_kernel(
        &mut self,
        arch: u8,
        kernel_type: u8,
        kernel_flags: u32,
        kernel_image: &[u8],
        api_port: u16,
        cmdline: Option<&str>,
    ) -> Result<u64, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }

        let image_hash = simple_shake256_256(kernel_image);
        let header = KernelHeader {
            kernel_magic: KERNEL_MAGIC,
            header_version: 1,
            arch,
            kernel_type,
            kernel_flags,
            min_memory_mb: 0,
            entry_point: 0,
            image_size: kernel_image.len() as u64,
            compressed_size: kernel_image.len() as u64,
            compression: 0,
            api_transport: 0,
            api_port,
            api_version: 1,
            image_hash,
            build_id: [0u8; 16],
            build_timestamp: 0,
            vcpu_count: 0,
            reserved_0: 0,
            cmdline_offset: 128,
            cmdline_length: cmdline.map_or(0, |s| s.len() as u32),
            reserved_1: 0,
        };
        let header_bytes = header.to_bytes();

        let cmdline_bytes = cmdline.map(|s| s.as_bytes());

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let (seg_id, seg_offset) = {
            let mut buf_writer = BufWriter::new(&self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            writer
                .write_kernel_seg(&mut buf_writer, &header_bytes, kernel_image, cmdline_bytes)
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        let cmdline_len = cmdline_bytes.map_or(0, |c| c.len());
        let payload_len = (128 + kernel_image.len() + cmdline_len) as u64;
        self.segment_dir
            .push((seg_id, seg_offset, payload_len, SegmentType::Kernel as u8));

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        self.epoch += 1;
        self.write_manifest()?;

        Ok(seg_id)
    }

    /// Embed a kernel image with a KernelBinding footer.
    ///
    /// The new KERNEL_SEG wire format is:
    ///   KernelHeader (128B) || KernelBinding (128B) || cmdline || kernel_image
    ///
    /// The KernelBinding ties the manifest root hash to the kernel, preventing
    /// segment-swap attacks.
    pub fn embed_kernel_with_binding(
        &mut self,
        arch: u8,
        kernel_type: u8,
        kernel_flags: u32,
        kernel_image: &[u8],
        api_port: u16,
        cmdline: Option<&str>,
        binding: &KernelBinding,
    ) -> Result<u64, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }

        let image_hash = simple_shake256_256(kernel_image);
        let cmdline_len = cmdline.map_or(0u32, |s| s.len() as u32);
        let header = KernelHeader {
            kernel_magic: KERNEL_MAGIC,
            header_version: 1,
            arch,
            kernel_type,
            kernel_flags,
            min_memory_mb: 0,
            entry_point: 0,
            image_size: kernel_image.len() as u64,
            compressed_size: kernel_image.len() as u64,
            compression: 0,
            api_transport: 0,
            api_port,
            api_version: 1,
            image_hash,
            build_id: [0u8; 16],
            build_timestamp: 0,
            vcpu_count: 0,
            reserved_0: 0,
            // cmdline_offset now accounts for KernelBinding (128 + 128 = 256)
            cmdline_offset: 128 + 128,
            cmdline_length: cmdline_len,
            reserved_1: 0,
        };
        let header_bytes = header.to_bytes();
        let binding_bytes = binding.to_bytes();

        // Build the combined payload: header(128) + binding(128) + cmdline + image
        let cmdline_data = cmdline.map(|s| s.as_bytes());
        let cmdline_slice = cmdline_data.unwrap_or(&[]);

        let mut payload = Vec::with_capacity(128 + 128 + cmdline_slice.len() + kernel_image.len());
        payload.extend_from_slice(&header_bytes);
        payload.extend_from_slice(&binding_bytes);
        payload.extend_from_slice(cmdline_slice);
        payload.extend_from_slice(kernel_image);

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;

        let (seg_id, seg_offset) = {
            let mut buf_writer = BufWriter::new(&self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // Write as raw kernel segment: the write_kernel_seg expects
            // header_bytes separately, but we need to include binding in
            // the "image" portion to keep the wire format correct.
            // So we pass the full payload minus the header as "image".
            writer
                .write_kernel_seg(
                    &mut buf_writer,
                    &header_bytes,
                    &payload[128..], // binding + cmdline + image
                    None,            // cmdline already included above
                )
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        let total_payload_len = payload.len() as u64;
        self.segment_dir.push((
            seg_id,
            seg_offset,
            total_payload_len,
            SegmentType::Kernel as u8,
        ));

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        self.epoch += 1;
        self.write_manifest()?;

        Ok(seg_id)
    }

    /// Extract the kernel image from this RVF file.
    ///
    /// Scans the segment directory for a KERNEL_SEG (type 0x0E) and returns
    /// the first 128 bytes (serialized KernelHeader) plus the remainder
    /// (kernel image bytes). Returns None if no KERNEL_SEG is present.
    ///
    /// For files with KernelBinding (ADR-031), the remainder includes the
    /// 128-byte binding followed by optional cmdline and the kernel image.
    /// Use `extract_kernel_binding` to parse the binding separately.
    #[allow(clippy::type_complexity)]
    pub fn extract_kernel(&self) -> Result<Option<(Vec<u8>, Vec<u8>)>, RvfError> {
        let entry = self
            .segment_dir
            .iter()
            .find(|&&(_, _, _, stype)| stype == SegmentType::Kernel as u8);

        let entry = match entry {
            Some(e) => e,
            None => return Ok(None),
        };

        let (_header, payload) = {
            let mut reader = BufReader::new(&self.file);
            read_path::read_segment_payload(&mut reader, entry.1)
                .map_err(|_| err(ErrorCode::InvalidChecksum))?
        };

        if payload.len() < 128 {
            return Err(err(ErrorCode::TruncatedSegment));
        }

        let kernel_header = payload[..128].to_vec();
        let kernel_image = payload[128..].to_vec();

        Ok(Some((kernel_header, kernel_image)))
    }

    /// Extract the KernelBinding from a KERNEL_SEG, if present.
    ///
    /// Returns `None` if no KERNEL_SEG exists or if the payload is too short
    /// to contain a KernelBinding (backward-compatible with old format).
    pub fn extract_kernel_binding(&self) -> Result<Option<KernelBinding>, RvfError> {
        let result = self.extract_kernel()?;
        match result {
            None => Ok(None),
            Some((_header_bytes, remainder)) => {
                if remainder.len() < 128 {
                    // Old format: no KernelBinding present
                    return Ok(None);
                }
                let mut binding_data = [0u8; 128];
                binding_data.copy_from_slice(&remainder[..128]);
                let binding = KernelBinding::from_bytes(&binding_data);
                // Check if this looks like a real binding (version > 0)
                if binding.binding_version == 0 {
                    return Ok(None);
                }
                Ok(Some(binding))
            }
        }
    }

    /// Embed an eBPF program into this RVF file as an EBPF_SEG.
    ///
    /// Builds a 64-byte EbpfHeader, serializes it, then delegates to
    /// the write path. Returns the segment_id of the new EBPF_SEG.
    pub fn embed_ebpf(
        &mut self,
        program_type: u8,
        attach_type: u8,
        max_dimension: u16,
        program_bytecode: &[u8],
        btf_data: Option<&[u8]>,
    ) -> Result<u64, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }

        let program_hash = simple_shake256_256(program_bytecode);
        let header = EbpfHeader {
            ebpf_magic: EBPF_MAGIC,
            header_version: 1,
            program_type,
            attach_type,
            program_flags: 0,
            insn_count: (program_bytecode.len() / 8) as u16,
            max_dimension,
            program_size: program_bytecode.len() as u64,
            map_count: 0,
            btf_size: btf_data.map_or(0, |b| b.len() as u32),
            program_hash,
        };
        let header_bytes = header.to_bytes();

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let (seg_id, seg_offset) = {
            let mut buf_writer = BufWriter::new(&self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            writer
                .write_ebpf_seg(&mut buf_writer, &header_bytes, program_bytecode, btf_data)
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        let btf_len = btf_data.map_or(0, |b| b.len());
        let payload_len = (64 + program_bytecode.len() + btf_len) as u64;
        self.segment_dir
            .push((seg_id, seg_offset, payload_len, SegmentType::Ebpf as u8));

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        self.epoch += 1;
        self.write_manifest()?;

        Ok(seg_id)
    }

    /// Extract eBPF program bytecode from this RVF file.
    ///
    /// Scans the segment directory for an EBPF_SEG (type 0x0F) and returns
    /// the first 64 bytes (serialized EbpfHeader) plus the remainder
    /// (program bytecode + optional BTF). Returns None if no EBPF_SEG.
    #[allow(clippy::type_complexity)]
    pub fn extract_ebpf(&self) -> Result<Option<(Vec<u8>, Vec<u8>)>, RvfError> {
        let entry = self
            .segment_dir
            .iter()
            .find(|&&(_, _, _, stype)| stype == SegmentType::Ebpf as u8);

        let entry = match entry {
            Some(e) => e,
            None => return Ok(None),
        };

        let (_header, payload) = {
            let mut reader = BufReader::new(&self.file);
            read_path::read_segment_payload(&mut reader, entry.1)
                .map_err(|_| err(ErrorCode::InvalidChecksum))?
        };

        if payload.len() < 64 {
            return Err(err(ErrorCode::TruncatedSegment));
        }

        let ebpf_header = payload[..64].to_vec();
        let ebpf_bytecode = payload[64..].to_vec();

        Ok(Some((ebpf_header, ebpf_bytecode)))
    }

    /// Embed a web dashboard bundle into this RVF file as a DASHBOARD_SEG.
    ///
    /// Builds a 64-byte DashboardHeader, serializes it, then delegates to
    /// the write path. Returns the segment_id of the new DASHBOARD_SEG.
    ///
    /// The `bundle_data` should contain: `[entry_path_bytes | file_table | file_data...]`
    /// as described in `DashboardHeader` documentation.
    pub fn embed_dashboard(
        &mut self,
        ui_framework: u8,
        bundle_data: &[u8],
        entry_path: &str,
    ) -> Result<u64, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }

        if bundle_data.len() as u64 > DASHBOARD_MAX_SIZE {
            return Err(err(ErrorCode::InvalidManifest));
        }

        let content_hash = simple_shake256_256(bundle_data);
        let header = DashboardHeader {
            dashboard_magic: DASHBOARD_MAGIC,
            header_version: 1,
            ui_framework,
            compression: 0,
            bundle_size: bundle_data.len() as u64,
            file_count: 0, // Caller encodes file count in bundle_data
            entry_path_len: entry_path.len() as u16,
            reserved: 0,
            build_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            content_hash,
        };
        let header_bytes = header.to_bytes();

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let (seg_id, seg_offset) = {
            let mut buf_writer = BufWriter::new(&self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            writer
                .write_dashboard_seg(&mut buf_writer, &header_bytes, bundle_data)
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        let payload_len = (64 + bundle_data.len()) as u64;
        self.segment_dir.push((
            seg_id,
            seg_offset,
            payload_len,
            SegmentType::Dashboard as u8,
        ));

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        self.epoch += 1;
        self.write_manifest()?;

        Ok(seg_id)
    }

    /// Extract dashboard bundle from this RVF file.
    ///
    /// Scans the segment directory for a DASHBOARD_SEG (type 0x11) and returns
    /// the first 64 bytes (serialized DashboardHeader) plus the remainder
    /// (bundle data). Returns None if no DASHBOARD_SEG.
    #[allow(clippy::type_complexity)]
    pub fn extract_dashboard(&self) -> Result<Option<(Vec<u8>, Vec<u8>)>, RvfError> {
        let entry = self
            .segment_dir
            .iter()
            .find(|&&(_, _, _, stype)| stype == SegmentType::Dashboard as u8);

        let entry = match entry {
            Some(e) => e,
            None => return Ok(None),
        };

        let (_header, payload) = {
            let mut reader = BufReader::new(&self.file);
            read_path::read_segment_payload(&mut reader, entry.1)
                .map_err(|_| err(ErrorCode::InvalidChecksum))?
        };

        if payload.len() < 64 {
            return Err(err(ErrorCode::TruncatedSegment));
        }

        let dashboard_header = payload[..64].to_vec();
        let bundle_data = payload[64..].to_vec();

        Ok(Some((dashboard_header, bundle_data)))
    }

    /// Embed a WASM module into this RVF file as a WASM_SEG.
    ///
    /// Builds a 64-byte WasmHeader, serializes it, then delegates to
    /// the write path. Returns the segment_id of the new WASM_SEG.
    ///
    /// For self-bootstrapping, embed two WASM_SEGs:
    /// 1. `role = Interpreter` (a minimal WASM interpreter, ~50 KB)
    /// 2. `role = Microkernel` (the RVF query engine, ~5.5 KB)
    ///
    /// The file then carries both its runtime and its data.
    pub fn embed_wasm(
        &mut self,
        role: u8,
        target: u8,
        required_features: u16,
        wasm_bytecode: &[u8],
        export_count: u16,
        bootstrap_priority: u8,
        interpreter_type: u8,
    ) -> Result<u64, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }

        let bytecode_hash = simple_shake256_256(wasm_bytecode);
        let header = WasmHeader {
            wasm_magic: WASM_MAGIC,
            header_version: 1,
            role,
            target,
            required_features,
            export_count,
            bytecode_size: wasm_bytecode.len() as u32,
            compressed_size: 0,
            compression: 0,
            min_memory_pages: 2,
            max_memory_pages: 0,
            table_count: 0,
            bytecode_hash,
            bootstrap_priority,
            interpreter_type,
            reserved: [0; 6],
        };
        let header_bytes = header.to_bytes();

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let (seg_id, seg_offset) = {
            let mut buf_writer = BufWriter::new(&self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            writer
                .write_wasm_seg(&mut buf_writer, &header_bytes, wasm_bytecode)
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        let payload_len = (64 + wasm_bytecode.len()) as u64;
        self.segment_dir
            .push((seg_id, seg_offset, payload_len, SegmentType::Wasm as u8));

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        self.epoch += 1;
        self.write_manifest()?;

        Ok(seg_id)
    }

    /// Extract the first WASM module from this RVF file.
    ///
    /// Scans the segment directory for a WASM_SEG (type 0x10) and returns
    /// the first 64 bytes (serialized WasmHeader) plus the remainder
    /// (WASM bytecode). Returns None if no WASM_SEG.
    #[allow(clippy::type_complexity)]
    pub fn extract_wasm(&self) -> Result<Option<(Vec<u8>, Vec<u8>)>, RvfError> {
        let entry = self
            .segment_dir
            .iter()
            .find(|&&(_, _, _, stype)| stype == SegmentType::Wasm as u8);

        let entry = match entry {
            Some(e) => e,
            None => return Ok(None),
        };

        let (_header, payload) = {
            let mut reader = BufReader::new(&self.file);
            read_path::read_segment_payload(&mut reader, entry.1)
                .map_err(|_| err(ErrorCode::InvalidChecksum))?
        };

        if payload.len() < 64 {
            return Err(err(ErrorCode::TruncatedSegment));
        }

        let wasm_header = payload[..64].to_vec();
        let wasm_bytecode = payload[64..].to_vec();

        Ok(Some((wasm_header, wasm_bytecode)))
    }

    /// Extract all WASM modules from this RVF file, ordered by bootstrap_priority.
    ///
    /// Returns a vector of (WasmHeader bytes, bytecode) tuples for each WASM_SEG,
    /// sorted by the `bootstrap_priority` field (lowest first). This ordering
    /// determines the bootstrap chain: interpreter first, then microkernel.
    pub fn extract_wasm_all(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>, RvfError> {
        let entries: Vec<_> = self
            .segment_dir
            .iter()
            .filter(|&&(_, _, _, stype)| stype == SegmentType::Wasm as u8)
            .collect();

        let mut results = Vec::with_capacity(entries.len());

        for entry in entries {
            let (_header, payload) = {
                let mut reader = BufReader::new(&self.file);
                read_path::read_segment_payload(&mut reader, entry.1)
                    .map_err(|_| err(ErrorCode::InvalidChecksum))?
            };

            if payload.len() < 64 {
                return Err(err(ErrorCode::TruncatedSegment));
            }

            let wasm_header = payload[..64].to_vec();
            let wasm_bytecode = payload[64..].to_vec();
            results.push((wasm_header, wasm_bytecode));
        }

        // Sort by bootstrap_priority (byte offset 0x38 in WasmHeader)
        results.sort_by_key(|(hdr, _)| hdr[0x38]);

        Ok(results)
    }

    /// Check if this RVF file is self-bootstrapping.
    ///
    /// A file is self-bootstrapping if it contains at least one WASM_SEG
    /// with role=Interpreter or role=Combined. This means the file carries
    /// its own execution runtime and can run on any host with raw compute.
    pub fn is_self_bootstrapping(&self) -> bool {
        for &(_, offset, _, stype) in &self.segment_dir {
            if stype != SegmentType::Wasm as u8 {
                continue;
            }
            // Read the WasmHeader role byte (offset 0x06 within the payload)
            let result = (|| -> Result<bool, RvfError> {
                let mut reader = BufReader::new(&self.file);
                let (_header, payload) = read_path::read_segment_payload(&mut reader, offset)
                    .map_err(|_| err(ErrorCode::InvalidChecksum))?;
                if payload.len() < 64 {
                    return Ok(false);
                }
                let role = payload[0x06];
                Ok(role == WasmRole::Interpreter as u8 || role == WasmRole::Combined as u8)
            })();
            if result.unwrap_or(false) {
                return true;
            }
        }
        false
    }

    /// Get the segment directory.
    pub fn segment_dir(&self) -> &[(u64, u64, u64, u8)] {
        &self.segment_dir
    }

    /// Get the store's vector dimensionality.
    pub fn dimension(&self) -> u16 {
        self.options.dimension
    }

    /// Iterate every live `(id, &vector)` pair currently materialized in the store.
    ///
    /// Lazy and zero-copy: borrows the in-memory vector store and yields one
    /// entry per non-deleted vector, in arbitrary order. Deleted vectors (per
    /// the deletion bitmap) are skipped, matching [`query`](Self::query)
    /// visibility semantics.
    ///
    /// Motivation: `query` returns only `(id, distance)` ([`SearchResult`]),
    /// and there was previously no public way to recover the vector payloads.
    /// Downstream caches (e.g. an external `BackendAdapter` priming a quantized
    /// index) need to read every `(id, vector)` pair without re-deriving it.
    /// The reader existed internally but was `pub(crate)`.
    pub fn iter_vectors(&self) -> impl Iterator<Item = (u64, &[f32])> + '_ {
        let vectors = &self.vectors;
        let deletion_bitmap = &self.deletion_bitmap;
        vectors
            .ids()
            .filter(move |&&id| !deletion_bitmap.is_deleted(id))
            .filter_map(move |&id| vectors.get(id).map(|v| (id, v)))
    }

    /// Collect every live `(id, vector)` pair into an owned `Vec`.
    ///
    /// Convenience over [`iter_vectors`](Self::iter_vectors) for callers that
    /// want owned data. For very large stores, prefer `iter_vectors` and batch
    /// at the call site to avoid materializing the whole set at once.
    pub fn read_all_vectors(&self) -> Vec<(u64, Vec<f32>)> {
        self.iter_vectors()
            .map(|(id, v)| (id, v.to_vec()))
            .collect()
    }

    /// Get the file identity (lineage metadata) for this store.
    pub fn file_identity(&self) -> &FileIdentity {
        &self.file_identity
    }

    /// Get this file's unique identifier.
    pub fn file_id(&self) -> &[u8; 16] {
        &self.file_identity.file_id
    }

    /// Get the parent file's identifier (all zeros if root).
    pub fn parent_id(&self) -> &[u8; 16] {
        &self.file_identity.parent_id
    }

    /// Get the lineage depth (0 for root files).
    pub fn lineage_depth(&self) -> u32 {
        self.file_identity.lineage_depth
    }

    /// Create a COW branch from this store.
    ///
    /// Creates a new child file that inherits all vectors from the parent via
    /// COW references. Writes to the child only allocate local clusters as
    /// needed. The parent should be frozen first to ensure immutability.
    pub fn branch(&self, child_path: &Path) -> Result<Self, RvfError> {
        // Compute cluster geometry from the vector data
        let dim = self.options.dimension as u32;
        let bytes_per_vec = dim * 4; // f32
        let vectors_per_cluster = if bytes_per_vec > 0 {
            (4096 / bytes_per_vec).max(1)
        } else {
            64
        };
        let cluster_size = vectors_per_cluster * bytes_per_vec;
        let total_vecs = self.vectors.len() as u64;
        let cluster_count = if vectors_per_cluster > 0 {
            total_vecs.div_ceil(vectors_per_cluster as u64) as u32
        } else {
            0
        };

        // Derive the child via the standard lineage path
        let mut child = self.derive(
            child_path,
            rvf_types::DerivationType::Clone,
            Some(self.options.clone()),
        )?;

        // Initialize COW engine on the child with all clusters pointing to parent
        child.cow_engine = Some(CowEngine::from_parent(
            cluster_count,
            cluster_size,
            vectors_per_cluster,
            bytes_per_vec,
        ));

        // Initialize membership filter with all parent vectors visible
        let mut filter = MembershipFilter::new_include(total_vecs);
        for &vid in self.vectors.ids() {
            if !self.deletion_bitmap.is_deleted(vid) {
                filter.add(vid);
            }
        }
        child.membership_filter = Some(filter);

        Ok(child)
    }

    /// Freeze (snapshot) this store. Prevents further writes to this generation.
    pub fn freeze(&mut self) -> Result<(), RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }

        if let Some(ref mut engine) = self.cow_engine {
            engine.freeze(self.epoch)?;
        }

        // Set read_only to prevent further mutations
        self.read_only = true;
        Ok(())
    }

    /// Check if this store is a COW child (has a parent).
    pub fn is_cow_child(&self) -> bool {
        self.cow_engine.is_some()
    }

    /// Get COW statistics, if this store uses COW.
    pub fn cow_stats(&self) -> Option<CowStats> {
        self.cow_engine.as_ref().map(|e| e.stats())
    }

    /// Get the membership filter, if present.
    pub fn membership_filter(&self) -> Option<&MembershipFilter> {
        self.membership_filter.as_ref()
    }

    /// Get a mutable reference to the membership filter.
    pub fn membership_filter_mut(&mut self) -> Option<&mut MembershipFilter> {
        self.membership_filter.as_mut()
    }

    /// Get the parent file path, if this is a COW child.
    pub fn parent_path(&self) -> Option<&Path> {
        self.parent_path.as_deref()
    }

    /// Derive a child store from this parent.
    ///
    /// Creates a new RVF file at `child_path` that records this store as its
    /// parent. The child gets a new file_id, inherits dimensions and options,
    /// and records the parent's manifest hash for provenance verification.
    pub fn derive(
        &self,
        child_path: &Path,
        _derivation_type: rvf_types::DerivationType,
        child_options: Option<RvfOptions>,
    ) -> Result<Self, RvfError> {
        let opts = child_options.unwrap_or_else(|| self.options.clone());

        let child_file_id = generate_file_id(child_path);

        // Compute parent manifest hash from the file on disk
        let parent_hash = self.compute_own_manifest_hash()?;

        let new_depth = self
            .file_identity
            .lineage_depth
            .checked_add(1)
            .ok_or_else(|| err(ErrorCode::LineageBroken))?;

        let child_identity = FileIdentity {
            file_id: child_file_id,
            parent_id: self.file_identity.file_id,
            parent_hash,
            lineage_depth: new_depth,
        };

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(child_path)
            .map_err(|_| err(ErrorCode::FsyncFailed))?;

        let writer_lock = WriterLock::acquire(child_path).map_err(|_| err(ErrorCode::LockHeld))?;

        // Detect domain profile from child extension
        let domain_profile = child_path
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(DomainProfile::from_extension)
            .unwrap_or(opts.domain_profile);

        let mut child_opts = opts;
        child_opts.domain_profile = domain_profile;

        let mut store = Self {
            path: child_path.to_path_buf(),
            options: child_opts,
            file,
            seg_writer: Some(SegmentWriter::new(1)),
            writer_lock: Some(writer_lock),
            vectors: VectorData::new(self.options.dimension),
            deletion_bitmap: DeletionBitmap::new(),
            metadata: MetadataStore::new(),
            epoch: 0,
            segment_dir: Vec::new(),
            read_only: false,
            last_compaction_time: 0,
            file_identity: child_identity,
            cow_engine: None,
            membership_filter: None,
            parent_path: Some(self.path.clone()),
            last_witness_hash: [0u8; 32],
            index: Mutex::new(None),
            index_building: AtomicBool::new(false),
            rabitq: Mutex::new(None),
            rabitq_building: AtomicBool::new(false),
        };

        store.write_manifest()?;
        Ok(store)
    }

    /// Compute a hash of this file's content for use as parent_hash in derivation.
    fn compute_own_manifest_hash(&self) -> Result<[u8; 32], RvfError> {
        use std::io::Read;
        let file_len = self
            .file
            .metadata()
            .map_err(|_| err(ErrorCode::InvalidManifest))?
            .len();
        if file_len == 0 {
            return Ok([0u8; 32]);
        }
        // Hash up to 64KB from the end of the file (covers manifest segments)
        let read_len = file_len.min(65536) as usize;
        let mut reader = BufReader::new(&self.file);
        reader
            .seek(SeekFrom::End(-(read_len as i64)))
            .map_err(|_| err(ErrorCode::InvalidManifest))?;
        let mut buf = vec![0u8; read_len];
        reader
            .read_exact(&mut buf)
            .map_err(|_| err(ErrorCode::InvalidManifest))?;
        Ok(simple_shake256_256(&buf))
    }

    /// Return the hash of the last witness entry (for external verification).
    pub fn last_witness_hash(&self) -> &[u8; 32] {
        &self.last_witness_hash
    }

    /// Get a reference to the store's configuration options.
    pub fn options(&self) -> &RvfOptions {
        &self.options
    }

    /// Get the distance metric used by this store.
    pub fn metric(&self) -> DistanceMetric {
        self.options.metric
    }

    /// Get the current manifest epoch.
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    // ── Internal methods ──────────────────────────────────────────────

    /// Append a witness segment to the file and update the witness chain.
    ///
    /// `witness_type` is one of the `witness_types::*` constants.
    /// `action` is a human-readable action description encoded as bytes.
    ///
    /// The witness entry is chain-linked to the previous witness via
    /// `last_witness_hash` using `simple_shake256_256`.
    fn append_witness(&mut self, witness_type: u8, action: &[u8]) -> Result<(), RvfError> {
        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;

        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let (seg_id, seg_offset) = {
            let mut buf_writer = BufWriter::new(&self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            writer
                .write_witness_seg(
                    &mut buf_writer,
                    witness_type,
                    timestamp_ns,
                    action,
                    &self.last_witness_hash,
                )
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        // Compute the payload length for the segment directory.
        let payload_len = (1 + 8 + 4 + action.len() + 32) as u64;
        self.segment_dir
            .push((seg_id, seg_offset, payload_len, SegmentType::Witness as u8));

        // Build the serialized witness entry bytes and hash them to update
        // the chain. This mirrors the payload layout exactly so that
        // external verifiers can reconstruct the chain from raw segments.
        let mut entry_bytes = Vec::with_capacity(1 + 8 + 4 + action.len() + 32);
        entry_bytes.push(witness_type);
        entry_bytes.extend_from_slice(&timestamp_ns.to_le_bytes());
        entry_bytes.extend_from_slice(&(action.len() as u32).to_le_bytes());
        entry_bytes.extend_from_slice(action);
        entry_bytes.extend_from_slice(&self.last_witness_hash);
        self.last_witness_hash = simple_shake256_256(&entry_bytes);

        Ok(())
    }

    fn boot(&mut self) -> Result<(), RvfError> {
        let manifest = {
            let mut reader = BufReader::new(&self.file);
            read_path::find_latest_manifest(&mut reader)
                .map_err(|_| err(ErrorCode::ManifestNotFound))?
        };

        let manifest = match manifest {
            Some(m) => m,
            None => return Err(err(ErrorCode::ManifestNotFound)),
        };

        self.epoch = manifest.epoch;
        self.options.dimension = manifest.dimension;
        self.options.profile = manifest.profile_id;
        // Pre-size the slab from the manifest so the cold-open load does a
        // single allocation instead of growing through repeated doublings.
        self.vectors = VectorData::with_capacity(
            manifest.dimension,
            usize::try_from(manifest.total_vectors).unwrap_or(0),
        );
        self.deletion_bitmap = DeletionBitmap::from_ids(&manifest.deleted_ids);

        self.segment_dir = manifest
            .segment_dir
            .iter()
            .map(|e| (e.seg_id, e.offset, e.payload_length, e.seg_type))
            .collect();

        let vec_seg_entries: Vec<_> = manifest
            .segment_dir
            .iter()
            .filter(|e| e.seg_type == SegmentType::Vec as u8)
            .collect();

        for entry in vec_seg_entries {
            let (_header, payload) = {
                let mut reader = BufReader::new(&self.file);
                read_path::read_segment_payload(&mut reader, entry.offset)
                    .map_err(|_| err(ErrorCode::InvalidChecksum))?
            };

            // Fast path: bulk-copy the segment's rows straight into the
            // contiguous slab (no per-vector allocation). Falls back to
            // the legacy parser for segments whose dimension differs from
            // the manifest dimension (such rows are skipped by the slab,
            // matching the layout invariant).
            if self.vectors.load_from_vec_seg(&payload).is_none() {
                if let Some(vec_entries) = read_path::read_vec_seg_payload(&payload) {
                    for (vec_id, vec_data) in vec_entries {
                        self.vectors.insert(vec_id, vec_data);
                    }
                }
            }
        }

        // Restore FileIdentity from manifest if present
        if let Some(fi) = manifest.file_identity {
            self.file_identity = fi;
        }

        // Load the most recently persisted HNSW index, if any. A stale or
        // corrupt INDEX_SEG is ignored; the index is then rebuilt from
        // vectors on the first eligible query.
        let index_entry = self
            .segment_dir
            .iter()
            .rev()
            .find(|&&(_, _, _, stype)| stype == SegmentType::Index as u8)
            .copied();
        if let Some((_, offset, _, _)) = index_entry {
            let payload = {
                let mut reader = BufReader::new(&self.file);
                read_path::read_segment_payload(&mut reader, offset)
                    .ok()
                    .map(|(_, p)| p)
            };
            if let Some(payload) = payload {
                if let Some(idx) = VectorIndex::decode_payload(&payload, &self.vectors) {
                    *self.index.lock().unwrap_or_else(|e| e.into_inner()) = Some(idx);
                }
            }
        }

        if !self.read_only {
            let max_seg_id = self
                .segment_dir
                .iter()
                .map(|&(id, _, _, _)| id)
                .max()
                .unwrap_or(0);
            self.seg_writer = Some(SegmentWriter::new(max_seg_id + 1));
        }

        Ok(())
    }

    fn write_manifest(&mut self) -> Result<(), RvfError> {
        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;

        let total_vectors = self.vectors.len() as u64;
        let deleted_ids = self.deletion_bitmap.to_sorted_ids();

        // Include FileIdentity if this file has a non-zero file_id
        let fi = if self.file_identity.file_id != [0u8; 16] {
            Some(&self.file_identity)
        } else {
            None
        };

        let (manifest_seg_id, manifest_offset) = {
            let mut buf_writer = BufWriter::new(&self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            writer
                .write_manifest_seg_with_identity(
                    &mut buf_writer,
                    self.epoch,
                    self.options.dimension,
                    total_vectors,
                    self.options.profile,
                    &self.segment_dir,
                    &deleted_ids,
                    fi,
                )
                .map_err(|_| err(ErrorCode::FsyncFailed))?
        };

        let mut manifest_payload_len =
            (22 + self.segment_dir.len() * 25 + 4 + deleted_ids.len() * 8) as u64;
        if fi.is_some() {
            manifest_payload_len += 4 + 68; // FIDI marker + FileIdentity
        }
        self.segment_dir.push((
            manifest_seg_id,
            manifest_offset,
            manifest_payload_len,
            SegmentType::Manifest as u8,
        ));

        self.file
            .sync_all()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        Ok(())
    }
}

/// Compute the distance between query `a` and stored vector `b`.
///
/// `a_norm_sq` is the precomputed squared norm of `a`, used only by the
/// cosine metric so the query norm is not recomputed per stored vector.
fn compute_distance(a: &[f32], b: &[f32], metric: &DistanceMetric, a_norm_sq: f32) -> f32 {
    match metric {
        DistanceMetric::L2 => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum(),
        DistanceMetric::InnerProduct => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            -dot
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0f32;
            let mut norm_b = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                dot += x * y;
                norm_b += y * y;
            }
            let denom = (a_norm_sq * norm_b).sqrt();
            if denom < f32::EPSILON {
                1.0
            } else {
                1.0 - dot / denom
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Generate a file_id from path + timestamp using `simple_shake256_256`.
///
/// Previous implementation used XOR mixing which has very poor distribution
/// (e.g. paths differing in a single byte could collide). Now we hash the
/// concatenation of path bytes and nanosecond timestamp through
/// `simple_shake256_256` and take the first 16 bytes for much better
/// collision resistance.
fn generate_file_id(path: &Path) -> [u8; 16] {
    let path_str = path.to_string_lossy();
    let path_bytes = path_str.as_bytes();

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let ts_bytes = ts.to_le_bytes();

    // Concatenate path + timestamp, then hash for uniform distribution
    let mut input = Vec::with_capacity(path_bytes.len() + 8);
    input.extend_from_slice(path_bytes);
    input.extend_from_slice(&ts_bytes);

    let digest = simple_shake256_256(&input);
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

/// Minimal SHAKE-256 hash without depending on rvf-crypto.
/// Uses a simple XOR-fold for a 32-byte digest.
pub(crate) fn simple_shake256_256(data: &[u8]) -> [u8; 32] {
    // We use a simple non-cryptographic hash here since rvf-runtime
    // doesn't depend on rvf-crypto. For production lineage verification,
    // use rvf_crypto::compute_manifest_hash.
    let mut out = [0u8; 32];
    for (i, &b) in data.iter().enumerate() {
        out[i % 32] = out[i % 32].wrapping_add(b);
        // Avalanche
        let j = (i + 13) % 32;
        out[j] = out[j].wrapping_add(out[i % 32].rotate_left(3));
    }
    out
}

/// Scan raw file bytes for segment headers whose type should be preserved
/// during compaction. Returns `(file_offset, seg_id, payload_len, seg_type)`
/// for every segment that is NOT Vec (0x01), Manifest (0x05), or Journal (0x04).
///
/// This ensures forward compatibility: segment types unknown to this version
/// of the runtime (e.g., Kernel, Ebpf, or vendor extensions) survive a
/// compact/rewrite cycle byte-for-byte.
fn scan_preservable_segments(file_bytes: &[u8]) -> Vec<(usize, u64, u64, u8)> {
    let magic_bytes = SEGMENT_MAGIC.to_le_bytes();
    let mut results = Vec::new();

    if file_bytes.len() < SEGMENT_HEADER_SIZE {
        return results;
    }

    let last_possible = file_bytes.len() - SEGMENT_HEADER_SIZE;
    let mut i = 0;
    while i <= last_possible {
        if file_bytes[i..i + 4] == magic_bytes {
            let seg_type = file_bytes[i + 5];
            let seg_id = u64::from_le_bytes([
                file_bytes[i + 0x08],
                file_bytes[i + 0x09],
                file_bytes[i + 0x0A],
                file_bytes[i + 0x0B],
                file_bytes[i + 0x0C],
                file_bytes[i + 0x0D],
                file_bytes[i + 0x0E],
                file_bytes[i + 0x0F],
            ]);
            let payload_len = u64::from_le_bytes([
                file_bytes[i + 0x10],
                file_bytes[i + 0x11],
                file_bytes[i + 0x12],
                file_bytes[i + 0x13],
                file_bytes[i + 0x14],
                file_bytes[i + 0x15],
                file_bytes[i + 0x16],
                file_bytes[i + 0x17],
            ]);

            // Use checked arithmetic to prevent overflow on crafted payload_len.
            let total = match (payload_len as usize).checked_add(SEGMENT_HEADER_SIZE) {
                Some(t) if payload_len <= file_bytes.len() as u64 => t,
                _ => {
                    // Payload length is implausibly large; skip this byte.
                    i += 1;
                    continue;
                }
            };

            // Skip Vec, Manifest, and Journal segments -- these are
            // reconstructed by the compaction logic itself.
            if seg_type != SegmentType::Vec as u8
                && seg_type != SegmentType::Manifest as u8
                && seg_type != SegmentType::Journal as u8
            {
                // Only include if the full segment fits in the file.
                if i.checked_add(total)
                    .is_some_and(|end| end <= file_bytes.len())
                {
                    results.push((i, seg_id, payload_len, seg_type));
                }
            }

            // Advance past this segment (header + payload) to avoid
            // false magic matches inside payload data.
            if total > 0 {
                match i.checked_add(total) {
                    Some(next) if next > i => i = next,
                    _ => i += 1,
                }
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    results
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterValue;
    use tempfile::TempDir;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        let mut x = seed;
        for _ in 0..dim {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
        }
        v
    }

    #[test]
    fn read_all_vectors_round_trips_and_excludes_deleted() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("read_all.rvf");

        let options = RvfOptions {
            dimension: 8,
            metric: DistanceMetric::L2,
            ..Default::default()
        };
        let mut store = RvfStore::create(&path, options).unwrap();

        let ids = [10u64, 20, 30];
        let vecs: Vec<Vec<f32>> = ids.iter().map(|&i| random_vector(8, i)).collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        store.ingest_batch(&vec_refs, &ids, None).unwrap();

        // read_all_vectors returns every ingested (id, vector) pair.
        let mut got = store.read_all_vectors();
        got.sort_by_key(|(id, _)| *id);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].0, 10);
        assert_eq!(got[0].1, vecs[0]);
        assert_eq!(got[2].0, 30);
        assert_eq!(got[2].1, vecs[2]);

        // iter_vectors yields the same ids, lazily and zero-copy.
        let mut iter_ids: Vec<u64> = store.iter_vectors().map(|(id, _)| id).collect();
        iter_ids.sort_unstable();
        assert_eq!(iter_ids, vec![10, 20, 30]);

        // Deleted vectors are excluded, matching query() visibility.
        store.delete(&[20]).unwrap();
        let after: Vec<u64> = store.iter_vectors().map(|(id, _)| id).collect();
        assert!(!after.contains(&20));
        assert_eq!(after.len(), 2);

        store.close().unwrap();
    }

    #[test]
    fn create_ingest_query() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rvf");

        let options = RvfOptions {
            dimension: 8,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let dim = 8;
        let vecs: Vec<Vec<f32>> = (0..100).map(|i| random_vector(dim, i)).collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (0..100).collect();

        let result = store.ingest_batch(&vec_refs, &ids, None).unwrap();
        assert_eq!(result.accepted, 100);
        assert_eq!(result.rejected, 0);

        let query_vec = random_vector(dim, 42);
        let results = store
            .query(&query_vec, 10, &QueryOptions::default())
            .unwrap();
        assert_eq!(results.len(), 10);

        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }

        assert_eq!(results[0].id, 42);
        assert!(results[0].distance < f32::EPSILON);

        store.close().unwrap();
    }

    #[test]
    fn open_existing_store() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("reopen.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        {
            let mut store = RvfStore::create(&path, options.clone()).unwrap();
            let v1 = vec![1.0, 0.0, 0.0, 0.0];
            let v2 = vec![0.0, 1.0, 0.0, 0.0];
            let vecs: Vec<&[f32]> = vec![&v1, &v2];
            let ids = vec![10, 20];
            store.ingest_batch(&vecs, &ids, None).unwrap();
            store.close().unwrap();
        }

        {
            let store = RvfStore::open(&path).unwrap();
            let query = vec![1.0, 0.0, 0.0, 0.0];
            let results = store.query(&query, 2, &QueryOptions::default()).unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].id, 10);
            assert!(results[0].distance < f32::EPSILON);
            store.close().unwrap();
        }
    }

    #[test]
    fn reopen_with_manifest_beyond_64kb_tail_window() {
        // Regression test: find_latest_manifest used to scan only the final
        // 64 KB of the file. A manifest larger than that (here, via a large
        // deletion bitmap; the same happens after ~870 ingest batches as the
        // segment directory grows) pushed the manifest header beyond the
        // window and made the store unreadable on reopen.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("big_manifest.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        {
            let mut store = RvfStore::create(&path, options).unwrap();

            let vecs: Vec<Vec<f32>> = (0..10_000).map(|i| random_vector(4, i)).collect();
            let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            let ids: Vec<u64> = (0..10_000).collect();
            store.ingest_batch(&vec_refs, &ids, None).unwrap();

            // Deleting 9,000 vectors puts 72,000 bytes of deleted IDs into
            // every subsequent manifest, so the latest manifest header sits
            // more than 64 KB before EOF.
            let del_ids: Vec<u64> = (0..9_000).collect();
            let del_result = store.delete(&del_ids).unwrap();
            assert_eq!(del_result.deleted, 9_000);

            // One more small ingest so the file ends with a fresh large manifest.
            let extra = random_vector(4, 99_999);
            store
                .ingest_batch(&[extra.as_slice()], &[20_000], None)
                .unwrap();

            store.close().unwrap();
        }

        // Sanity-check the premise: no manifest header exists within the
        // final 64 KB of the file, so the old fixed-window scan would fail.
        {
            let data = std::fs::read(&path).unwrap();
            assert!(data.len() > 65_536);
            let tail = &data[data.len() - 65_536..];
            let magic = SEGMENT_MAGIC.to_le_bytes();
            let found = tail
                .windows(6)
                .any(|w| w[0..4] == magic && w[5] == SegmentType::Manifest as u8);
            assert!(!found, "manifest header unexpectedly within 64 KB tail");
        }

        {
            let store = RvfStore::open(&path).unwrap();
            assert_eq!(store.status().total_vectors, 1_001);

            let query = random_vector(4, 9_500);
            let results = store.query(&query, 1, &QueryOptions::default()).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, 9_500);
            store.close().unwrap();
        }
    }

    #[test]
    fn delete_vectors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("delete.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];
        let vecs: Vec<&[f32]> = vec![&v1, &v2, &v3];
        let ids = vec![1, 2, 3];
        store.ingest_batch(&vecs, &ids, None).unwrap();

        let del_result = store.delete(&[2]).unwrap();
        assert_eq!(del_result.deleted, 1);

        let query = vec![0.0, 1.0, 0.0, 0.0];
        let results = store.query(&query, 10, &QueryOptions::default()).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != 2));

        store.close().unwrap();
    }

    #[test]
    fn filter_query() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("filter.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];
        let vecs: Vec<&[f32]> = vec![&v1, &v2, &v3];
        let ids = vec![1, 2, 3];
        let metadata = vec![
            MetadataEntry {
                field_id: 0,
                value: MetadataValue::String("cat_a".into()),
            },
            MetadataEntry {
                field_id: 0,
                value: MetadataValue::String("cat_b".into()),
            },
            MetadataEntry {
                field_id: 0,
                value: MetadataValue::String("cat_a".into()),
            },
        ];
        store.ingest_batch(&vecs, &ids, Some(&metadata)).unwrap();

        let query = vec![0.5, 0.5, 0.5, 0.0];
        let query_opts = QueryOptions {
            filter: Some(FilterExpr::Eq(0, FilterValue::String("cat_a".into()))),
            ..Default::default()
        };
        let results = store.query(&query, 10, &query_opts).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id == 1 || r.id == 3));

        store.close().unwrap();
    }

    #[test]
    fn status_reports() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("status.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let status = store.status();
        assert_eq!(status.total_vectors, 0);
        assert!(!status.read_only);

        let v1 = [1.0, 0.0, 0.0, 0.0];
        store.ingest_batch(&[&v1[..]], &[1], None).unwrap();

        let status = store.status();
        assert_eq!(status.total_vectors, 1);
        assert!(status.file_size > 0);

        store.close().unwrap();
    }

    #[test]
    fn compact_reclaims_space() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("compact.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let vecs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (0..10).collect();
        store.ingest_batch(&vec_refs, &ids, None).unwrap();

        store.delete(&[0, 2, 4, 6, 8]).unwrap();

        let status = store.status();
        assert_eq!(status.total_vectors, 5);
        assert!(status.dead_space_ratio > 0.0);

        let compact_result = store.compact().unwrap();
        assert_eq!(compact_result.segments_compacted, 5);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = store.query(&query, 10, &QueryOptions::default()).unwrap();
        assert_eq!(results.len(), 5);
        let result_ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        for id in &[1, 3, 5, 7, 9] {
            assert!(result_ids.contains(id));
        }

        store.close().unwrap();
    }

    #[test]
    fn lock_prevents_two_writers() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("locked.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let _store1 = RvfStore::create(&path, options.clone()).unwrap();

        let result = RvfStore::open(&path);
        assert!(result.is_err());
    }

    #[test]
    fn readonly_open() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("readonly.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        {
            let mut store = RvfStore::create(&path, options).unwrap();
            let v1 = [1.0, 0.0, 0.0, 0.0];
            store.ingest_batch(&[&v1[..]], &[1], None).unwrap();
            store.close().unwrap();
        }

        let store = RvfStore::open_readonly(&path).unwrap();
        let status = store.status();
        assert!(status.read_only);
        assert_eq!(status.total_vectors, 1);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = store.query(&query, 1, &QueryOptions::default()).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn delete_by_filter_works() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("del_filter.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];
        let vecs: Vec<&[f32]> = vec![&v1, &v2, &v3];
        let ids = vec![1, 2, 3];
        let metadata = vec![
            MetadataEntry {
                field_id: 0,
                value: MetadataValue::U64(10),
            },
            MetadataEntry {
                field_id: 0,
                value: MetadataValue::U64(20),
            },
            MetadataEntry {
                field_id: 0,
                value: MetadataValue::U64(30),
            },
        ];
        store.ingest_batch(&vecs, &ids, Some(&metadata)).unwrap();

        let filter = FilterExpr::Gt(0, FilterValue::U64(15));
        let del_result = store.delete_by_filter(&filter).unwrap();
        assert_eq!(del_result.deleted, 2);

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = store.query(&query, 10, &QueryOptions::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);

        store.close().unwrap();
    }

    #[test]
    fn embed_extract_kernel_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("kernel_rt.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let kernel_image = b"fake-compressed-kernel-image-0123456789abcdef";
        let seg_id = store
            .embed_kernel(
                1,    // arch: x86_64
                0,    // kernel_type: unikernel
                0x01, // kernel_flags
                kernel_image,
                8080, // api_port
                Some("console=ttyS0 quiet"),
            )
            .unwrap();
        assert!(seg_id > 0);

        let result = store.extract_kernel().unwrap();
        assert!(result.is_some());
        let (header_bytes, image_bytes) = result.unwrap();
        assert_eq!(header_bytes.len(), 128);

        // Verify the image portion matches what we embedded
        // (image_bytes includes the cmdline appended after the kernel)
        assert!(image_bytes.starts_with(kernel_image));

        // Verify magic in the header
        let magic = u32::from_le_bytes([
            header_bytes[0],
            header_bytes[1],
            header_bytes[2],
            header_bytes[3],
        ]);
        assert_eq!(magic, KERNEL_MAGIC);

        // Verify arch (offset 0x06)
        assert_eq!(header_bytes[0x06], 1);

        // Verify api_port (offset 0x2A, big-endian)
        let port = u16::from_be_bytes([header_bytes[0x2A], header_bytes[0x2B]]);
        assert_eq!(port, 8080);

        store.close().unwrap();
    }

    #[test]
    fn embed_extract_ebpf_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ebpf_rt.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let bytecode = b"ebpf-program-instructions-here";
        let btf = b"btf-type-information";
        let seg_id = store
            .embed_ebpf(
                2,    // program_type: XDP
                1,    // attach_type
                1024, // max_dimension
                bytecode,
                Some(btf),
            )
            .unwrap();
        assert!(seg_id > 0);

        let result = store.extract_ebpf().unwrap();
        assert!(result.is_some());
        let (header_bytes, payload_bytes) = result.unwrap();
        assert_eq!(header_bytes.len(), 64);

        // Payload should be bytecode + btf
        assert_eq!(payload_bytes.len(), bytecode.len() + btf.len());
        assert_eq!(&payload_bytes[..bytecode.len()], bytecode);
        assert_eq!(&payload_bytes[bytecode.len()..], btf);

        // Verify magic
        let magic = u32::from_le_bytes([
            header_bytes[0],
            header_bytes[1],
            header_bytes[2],
            header_bytes[3],
        ]);
        assert_eq!(magic, EBPF_MAGIC);

        // Verify program_type (offset 0x06)
        assert_eq!(header_bytes[0x06], 2);

        // Verify max_dimension (offset 0x0E)
        let dim = u16::from_le_bytes([header_bytes[0x0E], header_bytes[0x0F]]);
        assert_eq!(dim, 1024);

        store.close().unwrap();
    }

    #[test]
    fn embed_kernel_persists_through_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("kernel_persist.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let kernel_image = b"persistent-kernel-image-data";

        {
            let mut store = RvfStore::create(&path, options).unwrap();
            store
                .embed_kernel(
                    2, // arch: aarch64
                    1, // kernel_type
                    0, // flags
                    kernel_image,
                    9090,
                    None,
                )
                .unwrap();
            store.close().unwrap();
        }

        {
            let store = RvfStore::open_readonly(&path).unwrap();
            let result = store.extract_kernel().unwrap();
            assert!(result.is_some());
            let (header_bytes, image_bytes) = result.unwrap();
            assert_eq!(header_bytes.len(), 128);
            assert_eq!(image_bytes, kernel_image);

            // Verify arch (offset 0x06)
            assert_eq!(header_bytes[0x06], 2);

            // Verify api_port (offset 0x2A, big-endian)
            let port = u16::from_be_bytes([header_bytes[0x2A], header_bytes[0x2B]]);
            assert_eq!(port, 9090);
        }
    }

    #[test]
    fn extract_returns_none_when_no_segment() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("no_kernel.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let store = RvfStore::create(&path, options).unwrap();
        assert!(store.extract_kernel().unwrap().is_none());
        assert!(store.extract_ebpf().unwrap().is_none());
        store.close().unwrap();
    }

    // ── Witness integration tests ────────────────────────────────────

    /// Helper: count how many WITNESS_SEG entries exist in the segment directory.
    fn count_witness_segments(store: &RvfStore) -> usize {
        store
            .segment_dir()
            .iter()
            .filter(|&&(_, _, _, stype)| stype == SegmentType::Witness as u8)
            .count()
    }

    #[test]
    fn test_ingest_creates_witness() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("witness_ingest.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        // Before ingest: no witness segments.
        assert_eq!(count_witness_segments(&store), 0);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let vecs: Vec<&[f32]> = vec![&v1, &v2];
        let ids = vec![1, 2];
        store.ingest_batch(&vecs, &ids, None).unwrap();

        // After ingest: exactly 1 witness segment.
        assert_eq!(count_witness_segments(&store), 1);

        // The last_witness_hash should be non-zero now.
        assert_ne!(store.last_witness_hash(), &[0u8; 32]);

        store.close().unwrap();
    }

    #[test]
    fn test_delete_creates_witness() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("witness_delete.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        store
            .ingest_batch(&[&v1[..], &v2[..]], &[1, 2], None)
            .unwrap();

        // 1 witness from ingest.
        assert_eq!(count_witness_segments(&store), 1);

        store.delete(&[1]).unwrap();

        // 2 witnesses: 1 from ingest + 1 from delete.
        assert_eq!(count_witness_segments(&store), 2);

        store.close().unwrap();
    }

    #[test]
    fn test_compact_creates_witness() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("witness_compact.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let vecs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (0..5).collect();
        store.ingest_batch(&vec_refs, &ids, None).unwrap();
        store.delete(&[0, 2]).unwrap();

        // Before compact: 1 witness from ingest + 1 witness from delete = 2.
        assert_eq!(count_witness_segments(&store), 2);

        store.compact().unwrap();

        // After compaction the file is rewritten. Witness segments from
        // before compaction are preserved (they are non-Vec/non-Manifest/
        // non-Journal) plus the new compact witness is appended: 2 + 1 = 3.
        assert_eq!(count_witness_segments(&store), 3);

        // Verify the last witness hash is non-zero.
        assert_ne!(store.last_witness_hash(), &[0u8; 32]);

        store.close().unwrap();
    }

    #[test]
    fn test_witness_chain_integrity() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("witness_chain.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        // Perform 3 operations to build a chain of 3 witnesses.
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];

        store.ingest_batch(&[&v1[..]], &[1], None).unwrap();
        let hash_after_first = *store.last_witness_hash();
        assert_ne!(hash_after_first, [0u8; 32]);

        store.ingest_batch(&[&v2[..]], &[2], None).unwrap();
        let hash_after_second = *store.last_witness_hash();
        // Each successive hash must be different (chain progresses).
        assert_ne!(hash_after_second, hash_after_first);
        assert_ne!(hash_after_second, [0u8; 32]);

        store.ingest_batch(&[&v3[..]], &[3], None).unwrap();
        let hash_after_third = *store.last_witness_hash();
        assert_ne!(hash_after_third, hash_after_second);
        assert_ne!(hash_after_third, hash_after_first);

        // Total witness segments should be 3.
        assert_eq!(count_witness_segments(&store), 3);

        store.close().unwrap();
    }

    #[test]
    fn test_witness_disabled_produces_no_segments() {
        use crate::options::WitnessConfig;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("witness_off.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            witness: WitnessConfig {
                witness_ingest: false,
                witness_delete: false,
                witness_compact: false,
                audit_queries: false,
            },
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        store.ingest_batch(&[&v1[..]], &[1], None).unwrap();
        store.delete(&[1]).unwrap();

        // No witness segments should have been created.
        assert_eq!(count_witness_segments(&store), 0);
        assert_eq!(store.last_witness_hash(), &[0u8; 32]);

        store.close().unwrap();
    }

    #[test]
    fn test_query_audited_creates_witness() {
        use crate::options::WitnessConfig;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("witness_query.rvf");

        let options = RvfOptions {
            dimension: 4,
            metric: DistanceMetric::L2,
            witness: WitnessConfig {
                witness_ingest: false, // disable ingest witness to isolate query
                witness_delete: false,
                witness_compact: false,
                audit_queries: true,
            },
            ..Default::default()
        };

        let mut store = RvfStore::create(&path, options).unwrap();

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        store.ingest_batch(&[&v1[..]], &[1], None).unwrap();

        // Regular query should NOT create a witness (immutable &self).
        let _results = store
            .query(&[1.0, 0.0, 0.0, 0.0], 1, &QueryOptions::default())
            .unwrap();
        assert_eq!(count_witness_segments(&store), 0);

        // Audited query SHOULD create a witness.
        let _results = store
            .query_audited(&[1.0, 0.0, 0.0, 0.0], 1, &QueryOptions::default())
            .unwrap();
        assert_eq!(count_witness_segments(&store), 1);
        assert_ne!(store.last_witness_hash(), &[0u8; 32]);

        store.close().unwrap();
    }

    // ── Audit finding 5: index rebuild must not block queries ─────────

    /// While one thread holds the `index_building` gate, other queries must
    /// be served by the exact scan instead of blocking (and must not build
    /// the index themselves).
    #[test]
    fn query_falls_back_to_exact_scan_while_index_is_building() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("building_fallback.rvf");

        let options = RvfOptions {
            dimension: 8,
            metric: DistanceMetric::L2,
            ..Default::default()
        };
        let mut store = RvfStore::create(&path, options).unwrap();

        let n = (crate::index_path::INDEX_MIN_VECTORS + 64) as u64;
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| random_vector(8, i)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (0..n).collect();
        store.ingest_batch(&refs, &ids, None).unwrap();

        // Simulate another thread mid-build: the gate is held.
        store.index_building.store(true, Ordering::Release);
        let q = random_vector(8, 17);
        let results = store.query(&q, 10, &QueryOptions::default()).unwrap();
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].id, 17);
        // Nobody built the index under the latched gate.
        assert!(!store.index_ready());

        // Gate released: the next query builds and installs the index.
        store.index_building.store(false, Ordering::Release);
        let results = store.query(&q, 10, &QueryOptions::default()).unwrap();
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].id, 17);
        assert!(store.index_ready());

        store.close().unwrap();
    }

    /// Interleaved overwrites (which drop the HNSW index) and concurrent
    /// queries: queries must keep serving — no panic, no deadlock, full
    /// result sets — while rebuilds happen outside the query mutex.
    /// Timing is deliberately not asserted to avoid flakes; a deadlock
    /// fails via the harness timeout.
    #[test]
    fn overwrite_invalidation_keeps_queries_serving() {
        use std::sync::RwLock;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("hot_overwrite.rvf");

        let options = RvfOptions {
            dimension: 8,
            metric: DistanceMetric::L2,
            ..Default::default()
        };
        let mut store = RvfStore::create(&path, options).unwrap();

        let n = (crate::index_path::INDEX_MIN_VECTORS + 200) as u64;
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| random_vector(8, i)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (0..n).collect();
        store.ingest_batch(&refs, &ids, None).unwrap();

        // Build the index once so the overwrites below actually invalidate it.
        let warm = random_vector(8, 7_777);
        store.query(&warm, 10, &QueryOptions::default()).unwrap();
        assert!(store.index_ready());

        let store = RwLock::new(store);
        std::thread::scope(|scope| {
            // Reader threads: concurrent queries throughout the overwrites.
            for t in 0..4u64 {
                let store = &store;
                scope.spawn(move || {
                    for i in 0..40u64 {
                        let q = random_vector(8, 100_000 + t * 1_000 + i);
                        let guard = store.read().unwrap();
                        let results = guard.query(&q, 10, &QueryOptions::default()).unwrap();
                        assert_eq!(results.len(), 10);
                        for w in results.windows(2) {
                            assert!(w[0].distance <= w[1].distance);
                        }
                    }
                });
            }
            // Writer thread: repeatedly overwrite existing ids, each one
            // dropping the in-memory index.
            let store = &store;
            scope.spawn(move || {
                for i in 0..10u64 {
                    let v = random_vector(8, 555_000 + i);
                    let mut guard = store.write().unwrap();
                    guard
                        .ingest_batch(&[v.as_slice()], &[i % 50], None)
                        .unwrap();
                    drop(guard);
                    std::thread::yield_now();
                }
            });
        });

        // Queries still serve after the dust settles, and the index can
        // be rebuilt (possibly by this very query).
        let store = store.into_inner().unwrap();
        let results = store.query(&warm, 10, &QueryOptions::default()).unwrap();
        assert_eq!(results.len(), 10);
        let _ = store.query(&warm, 10, &QueryOptions::default()).unwrap();
        store.close().unwrap();
    }

    /// Overwriting an existing id must still invalidate stale index state
    /// and produce correct nearest-neighbor results afterwards.
    #[test]
    fn overwrite_returns_fresh_vector_via_index_path() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("overwrite_fresh.rvf");

        let options = RvfOptions {
            dimension: 8,
            metric: DistanceMetric::L2,
            ..Default::default()
        };
        let mut store = RvfStore::create(&path, options).unwrap();

        let n = (crate::index_path::INDEX_MIN_VECTORS + 16) as u64;
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| random_vector(8, i)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (0..n).collect();
        store.ingest_batch(&refs, &ids, None).unwrap();

        store
            .query(&random_vector(8, 1), 5, &QueryOptions::default())
            .unwrap();
        assert!(store.index_ready());

        // Overwrite id 3 with a far-away vector.
        let far = vec![100.0f32; 8];
        store.ingest_batch(&[far.as_slice()], &[3], None).unwrap();
        assert!(!store.index_ready(), "overwrite must drop the stale index");

        // Query at the new location must find the overwritten id first.
        let results = store.query(&far, 1, &QueryOptions::default()).unwrap();
        assert_eq!(results[0].id, 3);
        assert!(results[0].distance < f32::EPSILON);

        // And the old location must NOT return id 3 anymore.
        let old_q = random_vector(8, 3);
        let results = store.query(&old_q, 5, &QueryOptions::default()).unwrap();
        assert!(results.iter().all(|r| r.id != 3 || r.distance > 1.0));

        store.close().unwrap();
    }
}
