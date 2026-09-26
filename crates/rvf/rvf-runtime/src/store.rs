//! Main RvfStore API — the primary user-facing interface.
//!
//! Ties together the write path, read path, indexing, deletion, and
//! compaction into a single cohesive store.

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use rvf_types::cow_map::{CowMapHeader, MapFormat, COWMAP_MAGIC};
use rvf_types::dashboard::{DashboardHeader, DASHBOARD_MAGIC, DASHBOARD_MAX_SIZE};
use rvf_types::ebpf::{EbpfHeader, EBPF_MAGIC};
use rvf_types::kernel::{KernelHeader, KERNEL_MAGIC};
use rvf_types::kernel_binding::KernelBinding;
use rvf_types::membership::MembershipHeader;
use rvf_types::metadata::{
    FileMetadataRecord as WireFileMetadata, MetadataRecord as WireMetadataRecord,
    MetadataRecordOp as WireMetadataRecordOp, MetadataSchemaEntry, MetadataSegment, MetadataType,
    MetadataValue as WireMetadataValue,
};
use rvf_types::wasm_bootstrap::{WasmHeader, WasmRole, WASM_MAGIC};
use rvf_types::{
    DomainProfile, ErrorCode, FileIdentity, RvfError, SegmentType, SEGMENT_HEADER_SIZE,
    SEGMENT_MAGIC,
};

use crate::compaction::{evaluate_triggers, CompactionDecision, CompactionThresholds};
use crate::cow::{CowEngine, CowStats};
use crate::cow_map::CowMap;
use crate::deletion::DeletionBitmap;
use crate::filter::{self, FilterExpr, MetadataStore};
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

const COW_STATE_FIXED_SIZE: usize = 64 + 4 + 1 + 4 + 4 + 4;
const MAX_COW_LINEAGE_DEPTH: u32 = 1024;
/// Bound dense membership bitmaps so a sparse, attacker-controlled vector ID
/// cannot turn branch creation into an unbounded allocation.
const MAX_MEMBERSHIP_FILTER_BYTES: u64 = 64 * 1024 * 1024;
/// Delta metadata generations written before the next full snapshot.
///
/// ADR-280 §5 caps replay at [`rvf_types::metadata::MAX_META_DELTAS`] deltas;
/// materializing well inside that ceiling keeps a reopen bounded and leaves
/// headroom for the torn-generation fallback in `restore_metadata`.
const META_SNAPSHOT_INTERVAL: u64 = 32;

/// Position of the committed metadata chain, rolled back as a unit when a
/// mutation fails after appending its `META_SEG`.
#[derive(Clone, Copy, Default)]
struct MetadataChainState {
    /// Newest committed generation; 0 when nothing is committed.
    generation: u64,
    /// Generation of the newest full snapshot in the chain.
    snapshot_generation: u64,
    /// Encoded bytes from that snapshot through `generation`. Open replays the
    /// whole chain, so writers hold this under the same decoded-byte ceiling
    /// readers enforce (ADR-280 §5).
    chain_bytes: usize,
    /// Set when the chain on disk cannot be safely extended by a delta, so the
    /// next generation must re-anchor it with a full snapshot.
    needs_snapshot: bool,
}

impl MetadataChainState {
    /// Delta generations written since the last full snapshot.
    fn deltas(&self) -> u64 {
        self.generation.saturating_sub(self.snapshot_generation)
    }
}

enum IngestMetadata<'a> {
    None,
    Legacy(&'a [MetadataEntry]),
    Records(&'a [VectorMetadata]),
}

fn relative_parent_reference(child_path: &Path, parent_path: &Path) -> Result<PathBuf, RvfError> {
    let child_dir = child_path
        .parent()
        .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
    let child_dir = if child_dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        child_dir
    };
    let child_dir = fs::canonicalize(child_dir).map_err(|_| err(ErrorCode::ParentNotFound))?;
    let parent = fs::canonicalize(parent_path).map_err(|_| err(ErrorCode::ParentNotFound))?;
    let base_parts: Vec<Component<'_>> = child_dir.components().collect();
    let target_parts: Vec<Component<'_>> = parent.components().collect();
    let common = base_parts
        .iter()
        .zip(&target_parts)
        .take_while(|(left, right)| left == right)
        .count();
    if common == 0 {
        return Err(err(ErrorCode::ParentChainBroken));
    }

    let mut relative = PathBuf::new();
    for _ in common..base_parts.len() {
        relative.push("..");
    }
    for component in &target_parts[common..] {
        relative.push(component.as_os_str());
    }
    if relative.as_os_str().is_empty() {
        return Err(err(ErrorCode::LineageCyclic));
    }
    Ok(relative)
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
    file_metadata: BTreeMap<String, MetadataValue>,
    metadata_chain: MetadataChainState,
    /// The metadata the committed `META_SEG` chain actually contains, which is
    /// the base every new generation is encoded against. It is not the same as
    /// `metadata`: records for identifiers the store does not yet hold are kept
    /// in memory but excluded from the chain, and must be emitted in full once
    /// those identifiers become committed. Advanced only after a mutation
    /// publishes its manifest.
    committed_metadata: MetadataStore,
    committed_file_metadata: BTreeMap<String, MetadataValue>,
    /// Generations dropped by open because the chain could not be replayed
    /// past them (see [`Self::metadata_recovery`]).
    dropped_metadata_generations: u64,
    /// Records the truncated replay resurrected and open therefore discarded.
    dropped_metadata_records: u64,
    /// `META_SEG` offsets a truncated replay could not reach. They are dropped
    /// from the segment directory by the same write that re-anchors the chain,
    /// never on their own -- see `restore_metadata`.
    orphaned_metadata_offsets: Vec<u64>,
    embedding_compatibility: EmbeddingCompatibility,
    metadata_filter_scanned_records: AtomicU64,
    metadata_filter_scanned_bytes: AtomicU64,
    metadata_filter_aborts: AtomicU64,
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
    /// Lazily-opened, read-only handle to the parent store, used for COW
    /// ANN dual-graph merge and exact parent read-through. `None` for root
    /// stores and until the first COW query on a child. Boxed to break the
    /// recursive type; `Mutex` so `query(&self)` can populate it lazily.
    parent_store: Mutex<Option<Box<RvfStore>>>,
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
            file_metadata: BTreeMap::new(),
            metadata_chain: MetadataChainState::default(),
            committed_metadata: MetadataStore::new(),
            committed_file_metadata: BTreeMap::new(),
            dropped_metadata_generations: 0,
            dropped_metadata_records: 0,
            orphaned_metadata_offsets: Vec::new(),
            embedding_compatibility: EmbeddingCompatibility::Unchecked,
            metadata_filter_scanned_records: AtomicU64::new(0),
            metadata_filter_scanned_bytes: AtomicU64::new(0),
            metadata_filter_aborts: AtomicU64::new(0),
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
            parent_store: Mutex::new(None),
        };

        store.write_manifest()?;
        Ok(store)
    }

    /// Open an existing RVF store for read-write access.
    ///
    /// Opening succeeds even when the committed metadata chain is damaged: the
    /// longest replayable prefix is served rather than failing the open
    /// (ADR-280 §5). That means a successful open does **not** by itself mean
    /// all committed metadata is present. Callers that care -- anything
    /// user-facing, and anything that reports on artifact health -- should
    /// check [`Self::metadata_recovery`] and surface a warning when it reports
    /// dropped generations or records. The next metadata write re-anchors the
    /// chain on the recovered state, and `compact()` reclaims the unreachable
    /// bytes.
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
            file_metadata: BTreeMap::new(),
            metadata_chain: MetadataChainState::default(),
            committed_metadata: MetadataStore::new(),
            committed_file_metadata: BTreeMap::new(),
            dropped_metadata_generations: 0,
            dropped_metadata_records: 0,
            orphaned_metadata_offsets: Vec::new(),
            embedding_compatibility: EmbeddingCompatibility::Unchecked,
            metadata_filter_scanned_records: AtomicU64::new(0),
            metadata_filter_scanned_bytes: AtomicU64::new(0),
            metadata_filter_aborts: AtomicU64::new(0),
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
            parent_store: Mutex::new(None),
        };

        store.boot()?;
        Ok(store)
    }

    /// Open an existing RVF store for read-only access (no lock required).
    ///
    /// Damaged metadata is recovered rather than fatal; see [`Self::open`] and
    /// check [`Self::metadata_recovery`].
    pub fn open_readonly(path: &Path) -> Result<Self, RvfError> {
        let mut ancestry = HashSet::new();
        Self::open_readonly_with_ancestry(path, &mut ancestry)
    }

    fn open_readonly_with_ancestry(
        path: &Path,
        ancestry: &mut HashSet<PathBuf>,
    ) -> Result<Self, RvfError> {
        if !path.exists() {
            return Err(err(ErrorCode::ManifestNotFound));
        }
        let canonical_path = fs::canonicalize(path).map_err(|_| err(ErrorCode::InvalidManifest))?;
        if !ancestry.insert(canonical_path.clone()) {
            return Err(err(ErrorCode::LineageCyclic));
        }

        let file = OpenOptions::new()
            .read(true)
            .open(&canonical_path)
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
            path: canonical_path.clone(),
            options: opts,
            file,
            seg_writer: None,
            writer_lock: None,
            vectors: VectorData::new(0),
            deletion_bitmap: DeletionBitmap::new(),
            metadata: MetadataStore::new(),
            file_metadata: BTreeMap::new(),
            metadata_chain: MetadataChainState::default(),
            committed_metadata: MetadataStore::new(),
            committed_file_metadata: BTreeMap::new(),
            dropped_metadata_generations: 0,
            dropped_metadata_records: 0,
            orphaned_metadata_offsets: Vec::new(),
            embedding_compatibility: EmbeddingCompatibility::Unchecked,
            metadata_filter_scanned_records: AtomicU64::new(0),
            metadata_filter_scanned_bytes: AtomicU64::new(0),
            metadata_filter_aborts: AtomicU64::new(0),
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
            parent_store: Mutex::new(None),
        };

        let boot_result = store.boot_with_ancestry(ancestry);
        ancestry.remove(&canonical_path);
        boot_result?;
        Ok(store)
    }

    /// Ingest a batch of vectors into the store.
    pub fn ingest_batch(
        &mut self,
        vectors: &[&[f32]],
        ids: &[u64],
        metadata: Option<&[MetadataEntry]>,
    ) -> Result<IngestResult, RvfError> {
        self.ingest_batch_internal(
            vectors,
            ids,
            metadata.map_or(IngestMetadata::None, IngestMetadata::Legacy),
        )
    }

    /// Atomically ingest vectors with explicitly associated metadata records.
    ///
    /// Record field counts may differ. Metadata IDs are validated against the
    /// resulting vector snapshot (existing live vectors plus this batch).
    pub fn ingest_batch_with_metadata(
        &mut self,
        vectors: &[&[f32]],
        ids: &[u64],
        metadata: &[VectorMetadata],
    ) -> Result<IngestResult, RvfError> {
        self.ingest_batch_internal(vectors, ids, IngestMetadata::Records(metadata))
    }

    fn ingest_batch_internal(
        &mut self,
        vectors: &[&[f32]],
        ids: &[u64],
        metadata: IngestMetadata<'_>,
    ) -> Result<IngestResult, RvfError> {
        if self.read_only {
            return Err(err(ErrorCode::ReadOnly));
        }
        if self.embedding_compatibility == EmbeddingCompatibility::VectorReadOnly {
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
        if matches!(
            metadata,
            IngestMetadata::Legacy(entries)
                if !entries.is_empty() && entries.len() % valid_ids.len() != 0
        ) {
            return Err(err(ErrorCode::InvalidManifest));
        }
        let unique_ids: HashSet<u64> = valid_ids.iter().copied().collect();
        if unique_ids.len() != valid_ids.len() {
            return Err(err(ErrorCode::InvalidManifest));
        }

        let old_vectors = self.vectors.clone();
        let old_metadata = self.metadata.clone();
        let old_chain = self.metadata_chain;
        let old_directory = self.segment_dir.clone();
        let old_epoch = self.epoch;
        let old_witness_hash = self.last_witness_hash;

        let candidate_metadata = self.build_ingest_metadata(&metadata, &valid_ids)?;
        let has_metadata = !matches!(metadata, IngestMetadata::None);

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;

        let (vec_seg_id, vec_seg_offset) = {
            let mut buf_writer = BufWriter::with_capacity(256 * 1024, &self.file);
            buf_writer
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            let written = writer
                .write_vec_seg(
                    &mut buf_writer,
                    &valid_vectors,
                    &valid_ids,
                    self.options.dimension,
                )
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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

        if has_metadata {
            self.metadata = candidate_metadata;
            if let Err(error) = self.write_metadata_generation() {
                self.metadata = old_metadata;
                self.metadata_chain = old_chain;
                self.vectors = old_vectors;
                self.segment_dir = old_directory;
                *self.index.lock().unwrap_or_else(|e| e.into_inner()) = None;
                *self.rabitq.lock().unwrap_or_else(|e| e.into_inner()) = None;
                return Err(error);
            }
        }

        if self.file.sync_all().is_err() {
            self.vectors = old_vectors;
            self.metadata = old_metadata;
            self.metadata_chain = old_chain;
            self.segment_dir = old_directory;
            *self.index.lock().unwrap_or_else(|e| e.into_inner()) = None;
            *self.rabitq.lock().unwrap_or_else(|e| e.into_inner()) = None;
            return Err(err(ErrorCode::FsyncFailed));
        }

        self.epoch += 1;

        // Append a witness entry recording this ingest operation.
        if self.options.witness.witness_ingest {
            let action = format!("ingest:count={},epoch={}", accepted, self.epoch);
            if let Err(error) = self.append_witness(witness_types::COMPUTATION, action.as_bytes()) {
                self.vectors = old_vectors;
                self.metadata = old_metadata;
                self.metadata_chain = old_chain;
                self.segment_dir = old_directory;
                self.epoch = old_epoch;
                self.last_witness_hash = old_witness_hash;
                *self.index.lock().unwrap_or_else(|e| e.into_inner()) = None;
                *self.rabitq.lock().unwrap_or_else(|e| e.into_inner()) = None;
                return Err(error);
            }
        }

        if let Err(error) = self.write_manifest() {
            self.vectors = old_vectors;
            self.metadata = old_metadata;
            self.metadata_chain = old_chain;
            self.segment_dir = old_directory;
            self.epoch = old_epoch;
            self.last_witness_hash = old_witness_hash;
            *self.index.lock().unwrap_or_else(|e| e.into_inner()) = None;
            *self.rabitq.lock().unwrap_or_else(|e| e.into_inner()) = None;
            return Err(error);
        }

        if has_metadata {
            self.commit_metadata_state();
        }

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
        let (mut results, _) = self.query_routed(vector, k, options)?;
        self.attach_metadata(&mut results, options);
        Ok(results)
    }

    /// Populate [`SearchResult::metadata`] on each hit when the query opted
    /// in via [`QueryOptions::include_metadata`] (ADR-280 §6).
    ///
    /// Metadata is resolved from `self.metadata`, the same committed,
    /// COW-resolved store that backs [`Self::get_metadata`], so child field
    /// overrides and record tombstones are reflected identically and hits
    /// come from the same committed snapshot as the hit list. This is a
    /// no-op (and allocates nothing) when `include_metadata` is `false`.
    fn attach_metadata(&self, results: &mut [SearchResult], options: &QueryOptions) {
        if !options.include_metadata {
            return;
        }
        for result in results.iter_mut() {
            result.metadata = self.get_metadata(result.id);
        }
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

        // COW children may have zero child-side vectors but still need parent
        // read-through; only skip early for non-COW empty stores.
        if self.vectors.len() == 0 && self.cow_engine.is_none() {
            return Ok((Vec::new(), false));
        }

        // Opt-in RaBitQ two-stage path (binary candidate scan + exact
        // rescore). Not an HNSW-index serve, so the flag stays false.
        if self.rabitq_eligible(options) {
            if let Some(results) = self.query_via_rabitq(vector, k, options) {
                return Ok((results, false));
            }
        }

        // COW ANN path: dual-graph merge over the child's own HNSW (or small
        // exact scan) and the parent's HNSW.  Approximate but sub-linear in
        // parent size — the parent HNSW is not rebuilt per branch.
        if self.cow_ann_eligible(options) {
            if let Some(results) = self.query_via_index_cow(vector, k, options) {
                return Ok((results, true));
            }
        }

        if self.index_eligible(options) {
            if let Some(results) = self.query_via_index(vector, k, options) {
                return Ok((results, true));
            }
        }

        Ok((self.query_exact(vector, k, options)?, false))
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
                    metadata: None,
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

    /// Whether a COW dual-graph ANN query is eligible.
    ///
    /// Requires: COW child with parent path, no metadata filter, not forced exact.
    /// The fast dual-graph path is skipped for filtered and force-exact queries,
    /// which fall through to `query_exact` (with parent read-through).
    fn cow_ann_eligible(&self, options: &QueryOptions) -> bool {
        self.cow_engine.is_some()
            && self.parent_path.is_some()
            && !options.force_exact
            && options.filter.is_none()
    }

    /// COW dual-graph ANN merge.
    ///
    /// Queries the child's own HNSW (or exact scan for small child slabs) AND
    /// the parent's HNSW (lazily opened, cached in `parent_store`), then merges
    /// the candidate pools with child-wins semantics:
    ///
    /// - Tombstoned IDs (removed from `membership_filter` by a child `delete`)
    ///   are silently dropped.
    /// - IDs present in the child slab (overrides) use the child's distance;
    ///   the parent's entry for the same ID is discarded.
    /// - Remaining parent candidates are included as-is.
    ///
    /// The candidate pool is over-fetched by `COW_ANN_OVERFETCH`× from each
    /// arm so the merged set can absorb tombstones and overrides and still
    /// supply `k` results.  Returns `None` when the child HNSW is still
    /// building (caller falls back to the exact scan).
    ///
    /// Approximation note: dual-graph merge is sub-linear in parent size but
    /// slightly approximate — recall@10 measured at ≥0.97 with C=4 on 1 200-
    /// vector L2 datasets with up to 5 % tombstones (see integration test
    /// `cow_ann_recall_vs_exact`).
    fn query_via_index_cow(
        &self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
    ) -> Option<Vec<SearchResult>> {
        /// Over-fetch multiplier per arm.  Each arm fetches k′ = k × C
        /// candidates so the merged pool can absorb tombstones and overrides
        /// and still supply k results.  C = 4 achieves recall@10 ≥ 0.97.
        const COW_ANN_OVERFETCH: usize = 4;
        let k_prime = k.saturating_mul(COW_ANN_OVERFETCH).max(k + 16);

        // Merged (id -> distance) map; child distances take priority.
        let mut merged: HashMap<u64, f32> = HashMap::with_capacity(k_prime * 2);

        // ── Child arm ────────────────────────────────────────────────────
        // Build / reuse child HNSW for its own vectors.  Fall back to an
        // exact scan of the (small) child slab when below the HNSW floor.
        if self.vectors.len() >= INDEX_MIN_VECTORS {
            let mut guard = self.index.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                // Exactly one thread builds; others return None so the
                // caller falls back to the exact scan (audit finding 5).
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
            idx.sync_missing(&self.vectors, self.options.metric);
            let ef = (options.ef_search as usize)
                .max(k_prime)
                .max(INDEX_MIN_EF_SEARCH);
            let hits = idx.search(vector, k_prime, ef, &self.vectors, self.options.metric);
            for (id, dist) in hits {
                if !self.deletion_bitmap.is_deleted(id) {
                    merged.insert(id, dist);
                }
            }
        } else {
            // Child too small for HNSW: exact scan of the child slab.
            let query_norm_sq = if self.options.metric == DistanceMetric::Cosine {
                vector.iter().map(|x| x * x).sum()
            } else {
                0.0f32
            };
            for (id, v) in self.vectors.iter() {
                if !self.deletion_bitmap.is_deleted(id) {
                    let d = compute_distance(vector, v, &self.options.metric, query_norm_sq);
                    merged.insert(id, d);
                }
            }
        }

        // ── Parent arm ───────────────────────────────────────────────────
        // Lazily open the parent store (read-only, cached), then query its
        // HNSW.  The parent's own HNSW is built on first query and cached
        // inside the parent store handle — no rebuild per branch.
        {
            let mut guard = self.parent_store.lock().unwrap_or_else(|e| e.into_inner());
            if guard.is_none() {
                // Open the parent read-only so we don't need its write lock.
                if let Some(ref parent_path) = self.parent_path {
                    if let Ok(p) = RvfStore::open_readonly(parent_path) {
                        *guard = Some(Box::new(p));
                    }
                    // On open failure (race / disk): skip parent arm silently.
                    // The child arm still provides approximate results.
                }
            }
            if let Some(ref parent) = *guard {
                // Pass ef_search through for tuning quality vs latency.
                let parent_opts = QueryOptions {
                    ef_search: options.ef_search,
                    ..Default::default()
                };
                if let Ok(parent_results) = parent.query(vector, k_prime, &parent_opts) {
                    // child_ids: IDs in the child slab that override the parent.
                    let child_ids: HashSet<u64> = self.vectors.ids().copied().collect();
                    for res in parent_results {
                        // Tombstone check: ID must still be visible in the
                        // membership_filter (delete() removes it on child-side
                        // deletion of an inherited parent vector).
                        if let Some(ref mf) = self.membership_filter {
                            if !mf.contains(res.id) {
                                continue;
                            }
                        }
                        // Override check: child's own vector wins; don't insert
                        // the parent's stale distance for an overridden ID.
                        if child_ids.contains(&res.id) {
                            continue;
                        }
                        // entry().or_insert: child candidates from the child arm
                        // (inserted above) are never overwritten by a parent hit
                        // for the same ID.  (Should be unreachable given the
                        // child_ids check, but guard for safety.)
                        merged.entry(res.id).or_insert(res.distance);
                    }
                }
            }
        }

        if merged.is_empty() {
            return None;
        }

        // Re-rank merged candidates by distance (ascending), take top-k.
        let mut results: Vec<SearchResult> = merged
            .into_iter()
            .map(|(id, distance)| SearchResult {
                id,
                distance,
                retrieval_quality: rvf_types::quality::RetrievalQuality::Full,
                metadata: None,
            })
            .collect();
        results.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(k);
        Some(results)
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
                metadata: None,
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
    fn query_exact(
        &self,
        vector: &[f32],
        k: usize,
        options: &QueryOptions,
    ) -> Result<Vec<SearchResult>, RvfError> {
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
        let filter_started = std::time::Instant::now();
        let mut filter_records = 0u64;
        let mut filter_bytes = 0u64;
        for (vec_id, stored_vec) in self.vectors.iter() {
            if self.deletion_bitmap.is_deleted(vec_id) {
                continue;
            }
            if let Some(ref filter_expr) = options.filter {
                self.check_metadata_filter_budget(
                    vec_id,
                    options,
                    filter_started,
                    &mut filter_records,
                    &mut filter_bytes,
                )?;
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

        // COW parent read-through: for a COW child (created via `branch()`),
        // also scan parent vectors that are visible in the membership filter
        // and not overridden by the child's own slab.  This makes `query_exact`
        // the correct ground-truth for recall comparison against the ANN path.
        if self.cow_engine.is_some() {
            self.cow_exact_parent_scan(
                vector,
                query_norm_sq,
                k,
                options,
                filter_started,
                &mut filter_records,
                &mut filter_bytes,
                &mut heap,
            )?;
        }

        // Drain the max-heap into sorted results (closest first).
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|(OrderedFloat(dist), id)| SearchResult {
                id,
                distance: dist,
                retrieval_quality: rvf_types::quality::RetrievalQuality::Full,
                metadata: None,
            })
            .collect();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(results)
    }

    /// Extend the `query_exact` result heap with parent vectors visible in the
    /// COW child's membership filter.
    ///
    /// Called from [`query_exact`] when `self.cow_engine.is_some()`.  Iterates
    /// the parent store's vector slab directly (O(parent_size) — the expected
    /// fallback when the ANN path returns `None` or is disabled).
    ///
    /// Parent vectors that are:
    /// - not in the membership filter (tombstoned by child `delete`)
    /// - overridden by the child's own slab (same ID exists in `self.vectors`)
    /// - soft-deleted in the parent itself
    ///
    /// …are silently skipped.
    fn cow_exact_parent_scan(
        &self,
        vector: &[f32],
        query_norm_sq: f32,
        k: usize,
        options: &QueryOptions,
        filter_started: std::time::Instant,
        filter_records: &mut u64,
        filter_bytes: &mut u64,
        heap: &mut BinaryHeap<(OrderedFloat, u64)>,
    ) -> Result<(), RvfError> {
        let parent_path = match self.parent_path.as_ref() {
            Some(p) => p,
            None => return Ok(()),
        };

        let mut guard = self.parent_store.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            if let Ok(p) = RvfStore::open_readonly(parent_path) {
                *guard = Some(Box::new(p));
            } else {
                return Ok(()); // parent unreadable; skip silently
            }
        }

        let parent = match guard.as_ref() {
            Some(p) => p,
            None => return Ok(()),
        };

        // IDs in the child slab override their parent counterpart.
        let child_ids: HashSet<u64> = self.vectors.ids().copied().collect();

        for (vid, stored_vec) in parent.vectors.iter() {
            // Tombstone check: ID must be visible in the membership filter.
            if let Some(ref mf) = self.membership_filter {
                if !mf.contains(vid) {
                    continue;
                }
            }
            // Override: child has its own version of this ID.
            if child_ids.contains(&vid) {
                continue;
            }
            // Parent-soft-deleted.
            if parent.deletion_bitmap.is_deleted(vid) {
                continue;
            }
            if let Some(ref filter_expr) = options.filter {
                self.check_metadata_filter_budget(
                    vid,
                    options,
                    filter_started,
                    filter_records,
                    filter_bytes,
                )?;
                if !filter::evaluate(filter_expr, vid, &self.metadata) {
                    continue;
                }
            }

            let dist = compute_distance(vector, stored_vec, &self.options.metric, query_norm_sq);
            if heap.len() < k {
                heap.push((OrderedFloat(dist), vid));
            } else if let Some(&(OrderedFloat(worst), worst_id)) = heap.peek() {
                if dist < worst || (dist == worst && vid < worst_id) {
                    heap.pop();
                    heap.push((OrderedFloat(dist), vid));
                }
            }
        }
        Ok(())
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
                    metadata: None,
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

        // Attach committed metadata to the final hit set on the same code
        // path as `query`, after safety-net merge and truncation so only the
        // returned hits pay the decode cost (ADR-280 §6).
        self.attach_metadata(&mut all_results, options);

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
        if self.embedding_compatibility == EmbeddingCompatibility::VectorReadOnly {
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
            let written = writer
                .write_journal_seg(&mut buf_writer, ids, epoch)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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

        // Everything below mutates visible state, so snapshot it first: a
        // failed delete that leaves tombstones behind in memory would let a
        // later commit publish a manifest whose deleted_ids contradict the
        // newest META_SEG, which open rejects outright.
        let old_metadata = self.metadata.clone();
        let old_chain = self.metadata_chain;
        let old_directory = self.segment_dir.clone();
        let old_deletion_bitmap = self.deletion_bitmap.clone();
        let old_membership_filter = self.membership_filter.clone();
        let old_epoch = self.epoch;
        let old_witness_hash = self.last_witness_hash;

        let mut deleted = 0u64;
        for &id in ids {
            if self.vectors.get(id).is_some() && !self.deletion_bitmap.is_deleted(id) {
                self.deletion_bitmap.delete(id);
                deleted += 1;
            }
        }

        // COW child: also tombstone parent-inherited IDs from the membership
        // filter.  Parent IDs are not in `self.vectors`, so the loop above
        // does not mark them.  Removing them from the membership filter makes
        // `cow_exact_parent_scan` and `query_via_index_cow` correctly exclude
        // them without an extra deletion_bitmap entry.
        if let Some(ref mut mf) = self.membership_filter {
            for &id in ids {
                if self.vectors.get(id).is_none() && mf.contains(id) {
                    deleted += 1;
                }
                mf.remove(id);
            }
        }
        self.metadata.remove_ids(ids);

        let result = (|| {
            // Data before pointer: the metadata payload must be durable before
            // the manifest that publishes it (ADR-280 §4, steps 2-3).
            self.write_metadata_generation()?;
            self.file
                .sync_all()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            self.epoch = epoch;
            if self.options.witness.witness_delete {
                let action = format!("delete:count={},epoch={}", deleted, self.epoch);
                self.append_witness(witness_types::DATA_PROVENANCE, action.as_bytes())?;
            }
            self.write_manifest()
        })();

        if let Err(error) = result {
            self.metadata = old_metadata;
            self.metadata_chain = old_chain;
            self.segment_dir = old_directory;
            self.deletion_bitmap = old_deletion_bitmap;
            self.membership_filter = old_membership_filter;
            self.epoch = old_epoch;
            self.last_witness_hash = old_witness_hash;
            return Err(error);
        }
        self.commit_metadata_state();

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

    /// Evaluate the documented compaction triggers against this store's live state.
    ///
    /// Spec 10 §7 defines when compaction *should* run — dead space > 20%, segment
    /// count > 32, at least 60s since the last run, with an emergency path above 70%.
    /// [`crate::compaction::evaluate_triggers`] implements exactly that and was unit
    /// tested, but had no caller outside its own tests, so the policy could not affect
    /// anything. Every input it needs was already tracked here: `dead_space_ratio` and
    /// `total_segments` are computed by [`Self::status`], and `last_compaction_time` is
    /// stamped at the end of [`Self::compact`].
    ///
    /// This is a read-only query — it never compacts. Use it to decide, or call
    /// [`Self::compact_if_needed`] to decide and act in one step.
    pub fn should_compact(&self) -> CompactionDecision {
        self.should_compact_with(&CompactionThresholds::default())
    }

    /// [`Self::should_compact`] against caller-supplied thresholds.
    pub fn should_compact_with(&self, thresholds: &CompactionThresholds) -> CompactionDecision {
        let status = self.status();
        // Saturating: a clock that moved backwards must not wrap into a huge interval
        // and make an over-eager compaction look policy-approved.
        let secs_since_last = now_secs().saturating_sub(self.last_compaction_time);
        evaluate_triggers(
            status.dead_space_ratio,
            status.total_segments,
            secs_since_last,
            thresholds,
        )
    }

    /// Compact only if [`Self::should_compact`] says the policy is met.
    ///
    /// Returns `Ok(None)` when no compaction was needed — distinct from
    /// `Ok(Some(result))` with zero reclaimed, which means compaction ran and found
    /// nothing. Callers that want to compact regardless should call [`Self::compact`]
    /// directly; that remains unconditional, because an explicit request is not a
    /// policy decision.
    pub fn compact_if_needed(&mut self) -> Result<Option<CompactionResult>, RvfError> {
        match self.should_compact() {
            CompactionDecision::None => Ok(None),
            CompactionDecision::Normal | CompactionDecision::Emergency => self.compact().map(Some),
        }
    }

    /// Run compaction to reclaim dead space.
    ///
    /// Unconditional by design: this is an explicit request, not a scheduling
    /// decision. For the documented policy, see [`Self::compact_if_needed`].
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
        let keeps_metadata = !self.metadata.is_empty() || !self.file_metadata.is_empty();
        let mut compacted_metadata_bytes = 0usize;
        let compacted_generation = self
            .metadata_chain
            .generation
            .checked_add(1)
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
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
                // Drop META_SEGs: compaction removes vectors, so a preserved
                // generation would reference identifiers the compacted file no
                // longer holds. The complete live snapshot is rewritten below,
                // which is also what reclaims superseded metadata history
                // (ADR-280 §1).
                if *seg_type == SegmentType::Meta as u8 {
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

            // Rewrite the live metadata as a single full snapshot, replacing
            // every superseded generation dropped above.
            if keeps_metadata {
                let payload = self.encode_metadata_generation(compacted_generation, None)?;
                let (seg_id, offset) = seg_writer
                    .write_meta_seg(&mut temp_writer, &payload)
                    .map_err(|_| err(ErrorCode::FsyncFailed))?;
                compacted_metadata_bytes = payload.len();
                new_segment_dir.push((
                    seg_id,
                    offset,
                    payload.len() as u64,
                    SegmentType::Meta as u8,
                ));
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
                    self.options.metric.to_id(),
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

        // Sync the parent directory so the rename itself is durable. A failure
        // here leaves the rename possibly unpersisted, so it is reported rather
        // than discarded (ADR-280 §4, step 7).
        if let Some(parent) = self.path.parent() {
            let dir = std::fs::File::open(parent).map_err(|_| err(ErrorCode::FsyncFailed))?;
            dir.sync_all().map_err(|_| err(ErrorCode::FsyncFailed))?;
        }

        self.file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|_| err(ErrorCode::InvalidManifest))?;

        self.metadata_chain = if keeps_metadata {
            MetadataChainState {
                generation: compacted_generation,
                snapshot_generation: compacted_generation,
                chain_bytes: compacted_metadata_bytes,
                needs_snapshot: false,
            }
        } else {
            MetadataChainState::default()
        };
        self.commit_metadata_state();
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
            let written = writer
                .write_index_seg(&mut buf_writer, &payload)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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
            let written = writer
                .write_kernel_seg(&mut buf_writer, &header_bytes, kernel_image, cmdline_bytes)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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
            let written = writer
                .write_kernel_seg(
                    &mut buf_writer,
                    &header_bytes,
                    &payload[128..], // binding + cmdline + image
                    None,            // cmdline already included above
                )
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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
            let written = writer
                .write_ebpf_seg(&mut buf_writer, &header_bytes, program_bytecode, btf_data)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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
            let written = writer
                .write_dashboard_seg(&mut buf_writer, &header_bytes, bundle_data)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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
            let written = writer
                .write_wasm_seg(&mut buf_writer, &header_bytes, wasm_bytecode)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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

    /// Return one vector's committed metadata, preserving null and binary
    /// values. An empty record is distinct from no record.
    pub fn get_metadata(&self, vector_id: u64) -> Option<Vec<MetadataEntry>> {
        self.metadata.get(vector_id).map(|fields| {
            fields
                .into_iter()
                .map(|(field_id, value)| MetadataEntry { field_id, value })
                .collect()
        })
    }

    /// Return a file-level metadata value.
    pub fn get_file_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.file_metadata.get(key)
    }

    /// Persist a file-level metadata value in the authoritative META_SEG.
    pub fn set_file_metadata(&mut self, key: String, value: MetadataValue) -> Result<(), RvfError> {
        if self.read_only || self.embedding_compatibility == EmbeddingCompatibility::VectorReadOnly
        {
            return Err(err(ErrorCode::ReadOnly));
        }
        if key.starts_with("rvf.") {
            return Err(err(ErrorCode::InvalidManifest));
        }
        self.set_file_metadata_inner(key, value)
    }

    /// Persist an implementation-owned `rvf.*` value such as canonical
    /// embedding identity JSON or its immutable space ID.
    pub fn set_system_file_metadata(
        &mut self,
        key: String,
        value: MetadataValue,
    ) -> Result<(), RvfError> {
        if !key.starts_with("rvf.") {
            return Err(err(ErrorCode::InvalidManifest));
        }
        self.set_file_metadata_inner(key, value)
    }

    /// Compare the caller's immutable embedding-space ID with the artifact.
    /// A mismatch is sticky and permits vector-only reads, inspection, and
    /// export while rejecting subsequent corpus mutation.
    pub fn verify_embedding_space(&mut self, expected_space_id: &str) -> EmbeddingCompatibility {
        self.embedding_compatibility = match self.file_metadata.get("rvf.embedding.space_id") {
            Some(MetadataValue::String(actual)) if actual == expected_space_id => {
                EmbeddingCompatibility::Compatible
            }
            Some(MetadataValue::Bytes(actual))
                if actual.as_slice() == expected_space_id.as_bytes() =>
            {
                EmbeddingCompatibility::Compatible
            }
            _ => EmbeddingCompatibility::VectorReadOnly,
        };
        self.embedding_compatibility
    }

    pub fn embedding_compatibility(&self) -> EmbeddingCompatibility {
        self.embedding_compatibility
    }

    /// What open was able to replay from the committed `META_SEG` chain.
    ///
    /// `dropped_generations` is non-zero when damage forced replay to stop
    /// short of the newest generation on disk; the metadata served is then the
    /// state as of `generation`, and `compact()` rewrites the file around it.
    pub fn metadata_recovery(&self) -> MetadataRecovery {
        MetadataRecovery {
            generation: self.metadata_chain.generation,
            dropped_generations: self.dropped_metadata_generations,
            dropped_records: self.dropped_metadata_records,
        }
    }

    /// Cumulative observability counters for bounded linear metadata scans.
    pub fn metadata_filter_stats(&self) -> MetadataFilterStats {
        MetadataFilterStats {
            scanned_records: self.metadata_filter_scanned_records.load(Ordering::Relaxed),
            scanned_bytes: self.metadata_filter_scanned_bytes.load(Ordering::Relaxed),
            aborted_scans: self.metadata_filter_aborts.load(Ordering::Relaxed),
        }
    }

    fn build_ingest_metadata(
        &self,
        metadata: &IngestMetadata<'_>,
        batch_ids: &[u64],
    ) -> Result<MetadataStore, RvfError> {
        let mut candidate = self.metadata.clone();
        match metadata {
            IngestMetadata::None => {}
            IngestMetadata::Legacy(entries) => {
                if entries.iter().any(
                    |entry| matches!(entry.value, MetadataValue::F64(value) if !value.is_finite()),
                ) {
                    return Err(err(ErrorCode::InvalidMetadata));
                }
                let entries_per_id = entries.len() / batch_ids.len().max(1);
                for (index, &vector_id) in batch_ids.iter().enumerate() {
                    let start = index * entries_per_id;
                    let end = start.saturating_add(entries_per_id).min(entries.len());
                    let fields = entries[start..end]
                        .iter()
                        .map(|entry| (entry.field_id, entry.value.clone()))
                        .collect();
                    candidate.insert(vector_id, fields);
                }
            }
            IngestMetadata::Records(records) => {
                let mut record_ids = HashSet::with_capacity(records.len());
                let batch_ids: HashSet<u64> = batch_ids.iter().copied().collect();
                for record in *records {
                    if !record_ids.insert(record.vector_id) {
                        return Err(err(ErrorCode::InvalidMetadata));
                    }
                    let existing_live = self.vectors.get(record.vector_id).is_some()
                        && !self.deletion_bitmap.is_deleted(record.vector_id);
                    let inherited_live = self
                        .membership_filter
                        .as_ref()
                        .is_some_and(|membership| membership.contains(record.vector_id));
                    if !batch_ids.contains(&record.vector_id) && !existing_live && !inherited_live {
                        return Err(err(ErrorCode::InvalidMetadata));
                    }
                    let mut field_ids = HashSet::with_capacity(record.fields.len());
                    for field in &record.fields {
                        if !field_ids.insert(field.field_id)
                            || matches!(field.value, MetadataValue::F64(value) if !value.is_finite())
                        {
                            return Err(err(ErrorCode::InvalidMetadata));
                        }
                    }
                    if record.delete_record {
                        if !record.fields.is_empty() {
                            return Err(err(ErrorCode::InvalidMetadata));
                        }
                        candidate.remove_ids(&[record.vector_id]);
                    } else {
                        candidate.insert(
                            record.vector_id,
                            record
                                .fields
                                .iter()
                                .map(|field| (field.field_id, field.value.clone()))
                                .collect(),
                        );
                    }
                }
            }
        }
        Ok(candidate)
    }

    fn check_metadata_filter_budget(
        &self,
        vector_id: u64,
        options: &QueryOptions,
        started: std::time::Instant,
        records: &mut u64,
        bytes: &mut u64,
    ) -> Result<(), RvfError> {
        *records = records.saturating_add(1);
        let record_bytes = self.metadata.decoded_size(vector_id) as u64;
        *bytes = bytes.saturating_add(record_bytes);
        self.metadata_filter_scanned_records
            .fetch_add(1, Ordering::Relaxed);
        self.metadata_filter_scanned_bytes
            .fetch_add(record_bytes, Ordering::Relaxed);
        let cancelled = options
            .metadata_filter_cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire));
        let deadline_exceeded = options.timeout_ms != 0
            && started.elapsed() >= std::time::Duration::from_millis(options.timeout_ms as u64);
        if cancelled
            || deadline_exceeded
            || *records > options.metadata_filter_max_records
            || *bytes > options.metadata_filter_max_bytes
        {
            self.metadata_filter_aborts.fetch_add(1, Ordering::Relaxed);
            return Err(err(if deadline_exceeded {
                ErrorCode::Timeout
            } else {
                ErrorCode::FilterBudgetExceeded
            }));
        }
        Ok(())
    }

    fn set_file_metadata_inner(
        &mut self,
        key: String,
        value: MetadataValue,
    ) -> Result<(), RvfError> {
        if self.read_only || self.embedding_compatibility == EmbeddingCompatibility::VectorReadOnly
        {
            return Err(err(ErrorCode::ReadOnly));
        }
        let old_file_metadata = self.file_metadata.clone();
        let old_chain = self.metadata_chain;
        let old_epoch = self.epoch;
        let old_directory = self.segment_dir.clone();
        self.file_metadata.insert(key, value);
        let result = (|| {
            self.write_metadata_generation()?;
            // Data before pointer: the metadata payload must be durable before
            // the manifest that publishes it (ADR-280 §4, steps 2-3).
            self.file
                .sync_all()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            self.epoch = self
                .epoch
                .checked_add(1)
                .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
            self.write_manifest()
        })();
        if result.is_err() {
            self.file_metadata = old_file_metadata;
            self.metadata_chain = old_chain;
            self.epoch = old_epoch;
            self.segment_dir = old_directory;
        } else {
            self.commit_metadata_state();
        }
        result
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
        let cluster_size = vectors_per_cluster
            .checked_mul(bytes_per_vec)
            .ok_or_else(|| err(ErrorCode::CowMapCorrupt))?
            .max(4096)
            .checked_next_power_of_two()
            .ok_or_else(|| err(ErrorCode::CowMapCorrupt))?;
        let total_vecs = self.vectors.len() as u64;
        // MembershipFilter is keyed by vector ID, not by slab ordinal. Sizing
        // it from `len()` silently drops ID == len (the normal Node string-ID
        // case, which starts at 1) and every sparse ID. Size from the highest
        // live ID instead, while bounding the dense bitmap to fail closed
        // rather than risk an allocation DoS.
        let membership_capacity = self
            .vectors
            .ids()
            .copied()
            .max()
            .map(|max_id| {
                max_id
                    .checked_add(1)
                    .ok_or_else(|| err(ErrorCode::MembershipInvalid))
            })
            .transpose()?
            .unwrap_or(0);
        let membership_bytes = membership_capacity
            .div_ceil(64)
            .checked_mul(8)
            .ok_or_else(|| err(ErrorCode::MembershipInvalid))?;
        if membership_bytes > MAX_MEMBERSHIP_FILTER_BYTES {
            return Err(err(ErrorCode::MembershipInvalid));
        }
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
        let mut filter = MembershipFilter::new_include(membership_capacity);
        for &vid in self.vectors.ids() {
            if !self.deletion_bitmap.is_deleted(vid) {
                filter.add(vid);
            }
        }
        child.membership_filter = Some(filter);
        // `derive` writes an initial lineage-only manifest. Commit a second
        // manifest after the COW state is initialized so a returned branch is
        // durable even if the caller immediately closes it.
        child.write_manifest()?;

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

        self.write_manifest()?;
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
            metadata: self.metadata.clone(),
            file_metadata: self.file_metadata.clone(),
            // The child file carries its own `META_SEG` generation sequence and
            // starts empty, so its first commit is a full snapshot rather than
            // a delta against a base generation only the parent file holds.
            metadata_chain: MetadataChainState::default(),
            committed_metadata: MetadataStore::new(),
            committed_file_metadata: BTreeMap::new(),
            dropped_metadata_generations: 0,
            dropped_metadata_records: 0,
            orphaned_metadata_offsets: Vec::new(),
            embedding_compatibility: self.embedding_compatibility,
            metadata_filter_scanned_records: AtomicU64::new(0),
            metadata_filter_scanned_bytes: AtomicU64::new(0),
            metadata_filter_aborts: AtomicU64::new(0),
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
            parent_store: Mutex::new(None),
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
            let written = writer
                .write_witness_seg(
                    &mut buf_writer,
                    witness_type,
                    timestamp_ns,
                    action,
                    &self.last_witness_hash,
                )
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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
        let own_path = fs::canonicalize(&self.path).map_err(|_| err(ErrorCode::InvalidManifest))?;
        let mut ancestry = HashSet::from([own_path]);
        self.boot_with_ancestry(&mut ancestry)
    }

    fn boot_with_ancestry(&mut self, ancestry: &mut HashSet<PathBuf>) -> Result<(), RvfError> {
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
        // Restore the distance metric persisted in the manifest header (byte
        // [19], previously a reserved zero).  Old stores read 0x00 there and
        // boot as L2 — the correct backward-compatible default.  Without this
        // restore, COW dual-graph queries open the parent via open_readonly()
        // which goes through boot() and was silently resetting the metric to
        // L2, breaking cosine queries (recall@10 ≈ 0.10 → ≈ 1.0 after fix).
        self.options.metric = manifest.metric;
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

        self.restore_metadata()?;
        self.restore_cow_state(ancestry)?;

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

    fn restore_cow_state(&mut self, ancestry: &mut HashSet<PathBuf>) -> Result<(), RvfError> {
        let cow_entry = self
            .segment_dir
            .iter()
            .rev()
            .find(|&&(_, _, _, segment_type)| segment_type == SegmentType::CowMap as u8)
            .copied();
        let membership_entry = self
            .segment_dir
            .iter()
            .rev()
            .find(|&&(_, _, _, segment_type)| segment_type == SegmentType::Membership as u8)
            .copied();

        let (cow_entry, membership_entry) = match (cow_entry, membership_entry) {
            (None, None) => return Ok(()),
            (Some(cow), Some(membership)) => (cow, membership),
            _ => return Err(err(ErrorCode::CowMapCorrupt)),
        };

        let cow_payload = {
            let mut reader = BufReader::new(&self.file);
            read_path::read_segment_payload(&mut reader, cow_entry.1)
                .map_err(|_| err(ErrorCode::CowMapCorrupt))?
                .1
        };
        if cow_payload.len() < COW_STATE_FIXED_SIZE {
            return Err(err(ErrorCode::CowMapCorrupt));
        }

        let mut header_bytes = [0u8; 64];
        header_bytes.copy_from_slice(&cow_payload[..64]);
        let header =
            CowMapHeader::from_bytes(&header_bytes).map_err(|_| err(ErrorCode::CowMapCorrupt))?;
        if header.version != 1 {
            return Err(err(ErrorCode::CowMapCorrupt));
        }

        let bytes_per_vector =
            u32::from_le_bytes(cow_payload[64..68].try_into().expect("fixed slice"));
        let frozen = match cow_payload[68] {
            0 => false,
            1 => true,
            _ => return Err(err(ErrorCode::CowMapCorrupt)),
        };
        let snapshot_epoch =
            u32::from_le_bytes(cow_payload[69..73].try_into().expect("fixed slice"));
        let parent_len =
            u32::from_le_bytes(cow_payload[73..77].try_into().expect("fixed slice")) as usize;
        let map_len =
            u32::from_le_bytes(cow_payload[77..81].try_into().expect("fixed slice")) as usize;
        let expected_len = COW_STATE_FIXED_SIZE
            .checked_add(parent_len)
            .and_then(|length| length.checked_add(map_len))
            .ok_or_else(|| err(ErrorCode::CowMapCorrupt))?;
        if expected_len != cow_payload.len() || parent_len == 0 {
            return Err(err(ErrorCode::CowMapCorrupt));
        }

        let parent_ref = std::str::from_utf8(
            &cow_payload[COW_STATE_FIXED_SIZE..COW_STATE_FIXED_SIZE + parent_len],
        )
        .map_err(|_| err(ErrorCode::CowMapCorrupt))?;
        let map_bytes = &cow_payload[COW_STATE_FIXED_SIZE + parent_len..expected_len];
        let map_format =
            MapFormat::try_from(header.map_format).map_err(|_| err(ErrorCode::CowMapCorrupt))?;
        let cow_map = CowMap::deserialize(map_bytes, map_format)
            .map_err(|_| err(ErrorCode::CowMapCorrupt))?;

        let unresolved_parent = PathBuf::from(parent_ref);
        let parent_path = if unresolved_parent.is_absolute() {
            unresolved_parent
        } else {
            self.path
                .parent()
                .ok_or_else(|| err(ErrorCode::ParentChainBroken))?
                .join(unresolved_parent)
        };
        let parent_path =
            fs::canonicalize(parent_path).map_err(|_| err(ErrorCode::ParentNotFound))?;
        let own_path = fs::canonicalize(&self.path).map_err(|_| err(ErrorCode::InvalidManifest))?;
        if parent_path == own_path {
            return Err(err(ErrorCode::LineageCyclic));
        }

        if self.file_identity.lineage_depth == 0
            || self.file_identity.lineage_depth > MAX_COW_LINEAGE_DEPTH
            || ancestry.len() >= MAX_COW_LINEAGE_DEPTH as usize
        {
            return Err(err(ErrorCode::LineageBroken));
        }
        let parent = RvfStore::open_readonly_with_ancestry(&parent_path, ancestry).map_err(
            |open_error| match open_error {
                RvfError::Code(ErrorCode::LineageCyclic) => open_error,
                _ => err(ErrorCode::ParentChainBroken),
            },
        )?;
        let parent_metadata = parent.metadata.clone();
        let parent_file_metadata = parent.file_metadata.clone();
        if parent.file_identity.lineage_depth.checked_add(1)
            != Some(self.file_identity.lineage_depth)
        {
            return Err(err(ErrorCode::LineageBroken));
        }
        if header.magic != COWMAP_MAGIC
            || header.base_file_id != parent.file_identity.file_id
            || header.base_file_id != self.file_identity.parent_id
        {
            return Err(err(ErrorCode::LineageBroken));
        }
        let actual_parent_hash = parent.compute_own_manifest_hash()?;
        if header.base_file_hash != actual_parent_hash
            || header.base_file_hash != self.file_identity.parent_hash
        {
            return Err(err(ErrorCode::ParentHashMismatch));
        }
        let expected_bytes_per_vector = u32::from(self.options.dimension) * 4;
        let minimum_cluster_size = header
            .vectors_per_cluster
            .checked_mul(bytes_per_vector)
            .ok_or_else(|| err(ErrorCode::CowMapCorrupt))?;
        if bytes_per_vector != expected_bytes_per_vector
            || header.vectors_per_cluster == 0
            || header.cluster_size_bytes < minimum_cluster_size
        {
            return Err(err(ErrorCode::CowMapCorrupt));
        }

        let membership_payload = {
            let mut reader = BufReader::new(&self.file);
            read_path::read_segment_payload(&mut reader, membership_entry.1)
                .map_err(|_| err(ErrorCode::MembershipInvalid))?
                .1
        };
        if membership_payload.len() < 96 {
            return Err(err(ErrorCode::MembershipInvalid));
        }
        let mut membership_header_bytes = [0u8; 96];
        membership_header_bytes.copy_from_slice(&membership_payload[..96]);
        let membership_header = MembershipHeader::from_bytes(&membership_header_bytes)
            .map_err(|_| err(ErrorCode::MembershipInvalid))?;
        let bitmap_start = usize::try_from(membership_header.filter_offset)
            .map_err(|_| err(ErrorCode::MembershipInvalid))?;
        let bitmap_end = bitmap_start
            .checked_add(membership_header.filter_size as usize)
            .ok_or_else(|| err(ErrorCode::MembershipInvalid))?;
        if bitmap_start != 96 || bitmap_end != membership_payload.len() {
            return Err(err(ErrorCode::MembershipInvalid));
        }
        let bitmap = &membership_payload[bitmap_start..bitmap_end];
        if simple_shake256_256(bitmap) != membership_header.filter_hash {
            return Err(err(ErrorCode::MembershipInvalid));
        }
        let membership_filter = MembershipFilter::deserialize(bitmap, &membership_header)
            .map_err(|_| err(ErrorCode::MembershipInvalid))?;
        if membership_filter.member_count() != membership_header.member_count {
            return Err(err(ErrorCode::MembershipInvalid));
        }

        self.cow_engine = Some(CowEngine::from_persisted(
            cow_map,
            header.cluster_size_bytes,
            header.vectors_per_cluster,
            bytes_per_vector,
            frozen,
            snapshot_epoch,
        )?);
        self.membership_filter = Some(membership_filter);
        self.parent_path = Some(parent_path);
        let has_local_metadata_snapshot = self
            .segment_dir
            .iter()
            .any(|entry| entry.3 == SegmentType::Meta as u8);
        if !has_local_metadata_snapshot {
            self.metadata = parent_metadata;
            self.file_metadata = parent_file_metadata;
        }
        let membership = self
            .membership_filter
            .as_ref()
            .ok_or_else(|| err(ErrorCode::MembershipInvalid))?;
        // Same rule as the root-store check in `restore_metadata`: an intact
        // chain that references an invisible identifier is a real
        // inconsistency, but a truncated one may simply have dropped the
        // generation that removed the record.
        let truncated = self.dropped_metadata_generations > 0;
        let mut resurrected: Vec<u64> = Vec::new();
        for vector_id in self.metadata.ids() {
            let local_live = self.vectors.get(vector_id).is_some()
                && !self.deletion_bitmap.is_deleted(vector_id);
            if !local_live && !membership.contains(vector_id) {
                if !truncated {
                    return Err(err(ErrorCode::InvalidMetadata));
                }
                resurrected.push(vector_id);
            }
        }
        self.metadata.remove_ids(&resurrected);
        self.dropped_metadata_records = self
            .dropped_metadata_records
            .saturating_add(resurrected.len() as u64);
        self.committed_metadata = self.metadata.clone();
        Ok(())
    }

    fn write_cow_state_segments(&mut self) -> Result<(), RvfError> {
        let engine = self
            .cow_engine
            .as_ref()
            .ok_or_else(|| err(ErrorCode::CowMapCorrupt))?;
        let membership = self
            .membership_filter
            .as_ref()
            .ok_or_else(|| err(ErrorCode::MembershipInvalid))?;
        let parent_path = self
            .parent_path
            .as_ref()
            .ok_or_else(|| err(ErrorCode::ParentChainBroken))?;
        let relative_parent = relative_parent_reference(&self.path, parent_path)?;
        let parent_bytes = relative_parent
            .to_str()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?
            .as_bytes();
        let parent_len =
            u32::try_from(parent_bytes.len()).map_err(|_| err(ErrorCode::InvalidManifest))?;
        let map_bytes = engine.cow_map().serialize();
        let map_len = u32::try_from(map_bytes.len()).map_err(|_| err(ErrorCode::CowMapCorrupt))?;
        let stats = engine.stats();
        let header = CowMapHeader {
            magic: COWMAP_MAGIC,
            version: 1,
            map_format: engine.cow_map().format() as u8,
            compression_policy: 0,
            cluster_size_bytes: stats.cluster_size,
            vectors_per_cluster: stats.vectors_per_cluster,
            base_file_id: self.file_identity.parent_id,
            base_file_hash: self.file_identity.parent_hash,
        };
        let mut cow_payload =
            Vec::with_capacity(COW_STATE_FIXED_SIZE + parent_bytes.len() + map_bytes.len());
        cow_payload.extend_from_slice(&header.to_bytes());
        cow_payload.extend_from_slice(&engine.bytes_per_vector().to_le_bytes());
        cow_payload.push(u8::from(stats.frozen));
        cow_payload.extend_from_slice(&stats.snapshot_epoch.to_le_bytes());
        cow_payload.extend_from_slice(&parent_len.to_le_bytes());
        cow_payload.extend_from_slice(&map_len.to_le_bytes());
        cow_payload.extend_from_slice(parent_bytes);
        cow_payload.extend_from_slice(&map_bytes);

        let membership_header = membership.to_header();
        let membership_bitmap = membership.serialize();
        let mut membership_payload = Vec::with_capacity(96 + membership_bitmap.len());
        membership_payload.extend_from_slice(&membership_header.to_bytes());
        membership_payload.extend_from_slice(&membership_bitmap);

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let mut buf_writer = BufWriter::new(&self.file);
        buf_writer
            .seek(SeekFrom::End(0))
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        let (cow_id, cow_offset) = writer
            .write_cow_map_seg(&mut buf_writer, &cow_payload)
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        let (membership_id, membership_offset) = writer
            .write_membership_seg(&mut buf_writer, &membership_payload)
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        buf_writer
            .flush()
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        self.segment_dir.push((
            cow_id,
            cow_offset,
            cow_payload.len() as u64,
            SegmentType::CowMap as u8,
        ));
        self.segment_dir.push((
            membership_id,
            membership_offset,
            membership_payload.len() as u64,
            SegmentType::Membership as u8,
        ));
        Ok(())
    }

    fn write_manifest(&mut self) -> Result<(), RvfError> {
        if self.cow_engine.is_some() {
            self.write_cow_state_segments()?;
        }
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
            let written = writer
                .write_manifest_seg_with_identity(
                    &mut buf_writer,
                    self.epoch,
                    self.options.dimension,
                    total_vectors,
                    self.options.profile,
                    self.options.metric.to_id(),
                    &self.segment_dir,
                    &deleted_ids,
                    fi,
                )
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            // A dropped `BufWriter` swallows flush errors, so an acknowledged
            // write could be lost before the fsync that follows. Surface them.
            buf_writer
                .flush()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            written
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

    /// True when `vector_id` belongs to this store's committed vector state and
    /// may therefore carry a metadata record (ADR-280 §4).
    ///
    /// Mirrors the acceptance rule in [`Self::build_ingest_metadata`]: a record
    /// is valid for a locally held live vector, or for a parent vector a COW
    /// child still exposes through its membership filter. A lineage child
    /// created by [`Self::derive`] holds neither, so it persists no record it
    /// cannot resolve on reopen.
    fn metadata_id_is_committed(&self, vector_id: u64) -> bool {
        if self.deletion_bitmap.is_deleted(vector_id) {
            return false;
        }
        self.vectors.get(vector_id).is_some()
            || self
                .membership_filter
                .as_ref()
                .is_some_and(|membership| membership.contains(vector_id))
    }

    /// Encode the metadata generation `generation` describing the current
    /// in-memory state relative to `base`.
    ///
    /// `base` is the state the previous generation committed. When it is
    /// `None` the result is a full snapshot; otherwise it is a delta carrying
    /// only the records and fields that changed, which is what keeps commit
    /// cost proportional to the mutation rather than to the live snapshot
    /// (ADR-280 §5, and the rejected "write a complete metadata snapshot on
    /// every mutation" alternative).
    fn encode_metadata_generation(
        &self,
        generation: u64,
        delta: Option<()>,
    ) -> Result<Vec<u8>, RvfError> {
        let (base_metadata, base_file_metadata) = match delta {
            Some(()) => (
                Some(&self.committed_metadata),
                Some(&self.committed_file_metadata),
            ),
            None => (None, None),
        };

        let mut file_metadata: Vec<WireFileMetadata> = Vec::new();
        for (key, value) in &self.file_metadata {
            if base_file_metadata.is_some_and(|base| base.get(key) == Some(value)) {
                continue;
            }
            file_metadata.push(WireFileMetadata {
                key: key.clone(),
                value: runtime_to_wire(value),
            });
        }
        if let Some(base) = base_file_metadata {
            for key in base.keys() {
                if !self.file_metadata.contains_key(key) {
                    file_metadata.push(WireFileMetadata {
                        key: key.clone(),
                        value: WireMetadataValue::DeleteField,
                    });
                }
            }
        }
        file_metadata.sort_by(|left, right| left.key.cmp(&right.key));

        let mut records: Vec<WireMetadataRecord> = Vec::new();
        for vector_id in self.metadata.ids() {
            if !self.metadata_id_is_committed(vector_id) {
                continue;
            }
            let Some(fields) = self.metadata.fields(vector_id) else {
                continue;
            };
            let previous = base_metadata.and_then(|base| base.fields(vector_id));
            let mut wire_fields: Vec<(u16, WireMetadataValue)> = Vec::new();
            for (&field_id, value) in fields {
                if previous.is_some_and(|previous| previous.get(&field_id) == Some(value)) {
                    continue;
                }
                wire_fields.push((field_id, runtime_to_wire(value)));
            }
            if let Some(previous) = previous {
                for &field_id in previous.keys() {
                    if !fields.contains_key(&field_id) {
                        wire_fields.push((field_id, WireMetadataValue::DeleteField));
                    }
                }
            }
            // An unchanged record is absent from a delta and inherited from the
            // base; in a full snapshot every live record is written out.
            if previous.is_some() && wire_fields.is_empty() {
                continue;
            }
            wire_fields.sort_by_key(|&(field_id, _)| field_id);
            records.push(WireMetadataRecord {
                vector_id,
                operation: WireMetadataRecordOp::Upsert(wire_fields),
            });
        }
        if let Some(base) = base_metadata {
            for vector_id in base.ids() {
                if self.metadata.fields(vector_id).is_none()
                    || !self.metadata_id_is_committed(vector_id)
                {
                    records.push(WireMetadataRecord {
                        vector_id,
                        operation: WireMetadataRecordOp::DeleteRecord,
                    });
                }
            }
        }
        records.sort_by_key(|record| record.vector_id);

        let schema = build_metadata_schema(&records)?;
        MetadataSegment {
            generation,
            base_generation: delta.is_some().then(|| generation.saturating_sub(1)),
            full_snapshot: delta.is_none(),
            schema,
            file_metadata,
            records,
        }
        .encode()
        .map_err(|_| err(ErrorCode::InvalidManifest))
    }

    /// Append the next authoritative `META_SEG` generation.
    ///
    /// The generation is encoded against `committed_metadata` -- what the
    /// published chain actually contains -- so a record held only in memory
    /// (because its identifier was not yet part of the committed vector state)
    /// is emitted in full the first time it becomes committable, rather than
    /// comparing equal to an in-memory base and being silently dropped.
    ///
    /// A full snapshot is written for the first generation, again before the
    /// delta chain reaches [`META_SNAPSHOT_INTERVAL`], and again whenever the
    /// chain would otherwise grow past the decoded-byte ceiling that open
    /// enforces -- a delta must never be able to commit a chain that cannot be
    /// replayed (ADR-280 §5).
    fn write_metadata_generation(&mut self) -> Result<(), RvfError> {
        let generation = self
            .metadata_chain
            .generation
            .checked_add(1)
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let re_anchoring = self.metadata_chain.needs_snapshot;
        let mut full_snapshot = self.metadata_chain.generation == 0
            || re_anchoring
            || self.metadata_chain.deltas() >= META_SNAPSHOT_INTERVAL;
        let mut payload =
            self.encode_metadata_generation(generation, (!full_snapshot).then_some(()))?;
        if !full_snapshot
            && self
                .metadata_chain
                .chain_bytes
                .saturating_add(payload.len())
                > rvf_types::metadata::MAX_META_DECODED_BYTES
        {
            full_snapshot = true;
            payload = self.encode_metadata_generation(generation, None)?;
        }
        // A full snapshot larger than the ceiling could never be reopened, so
        // the commit fails rather than producing an unopenable artifact.
        if payload.len() > rvf_types::metadata::MAX_META_DECODED_BYTES {
            return Err(err(ErrorCode::MetadataReplayLimitExceeded));
        }

        let writer = self
            .seg_writer
            .as_mut()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let (segment_id, offset) = {
            let mut buffered = BufWriter::new(&self.file);
            buffered
                .seek(SeekFrom::End(0))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            let result = writer
                .write_meta_seg(&mut buffered, &payload)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            buffered.flush().map_err(|_| err(ErrorCode::FsyncFailed))?;
            result
        };
        // A re-anchoring snapshot supersedes everything a truncated replay
        // could not reach, so the orphans leave the directory in the very same
        // write. Doing it here rather than at recovery time keeps the pruning
        // and the new anchor in one commit, and keeps manifest-only mutations
        // from publishing the pruning by itself.
        if re_anchoring {
            let orphans = self.orphaned_metadata_offsets.clone();
            self.segment_dir
                .retain(|&(_, segment_offset, _, segment_type)| {
                    segment_type != SegmentType::Meta as u8 || !orphans.contains(&segment_offset)
                });
        }
        self.segment_dir.push((
            segment_id,
            offset,
            payload.len() as u64,
            SegmentType::Meta as u8,
        ));
        self.metadata_chain.generation = generation;
        self.metadata_chain.needs_snapshot = false;
        if full_snapshot {
            self.metadata_chain.snapshot_generation = generation;
            self.metadata_chain.chain_bytes = payload.len();
        } else {
            self.metadata_chain.chain_bytes = self
                .metadata_chain
                .chain_bytes
                .saturating_add(payload.len());
        }
        Ok(())
    }

    /// Adopt the in-memory metadata as the published chain contents.
    ///
    /// Called only once a mutation's manifest is durable: until then the
    /// appended `META_SEG` is orphaned and the previous chain is still
    /// authoritative, so the delta base must not move.
    fn commit_metadata_state(&mut self) {
        self.committed_metadata = self.committed_metadata_view();
        self.committed_file_metadata = self.file_metadata.clone();
        // The manifest that just published no longer references the orphans.
        self.orphaned_metadata_offsets.clear();
        self.dropped_metadata_generations = 0;
        self.dropped_metadata_records = 0;
    }

    /// The subset of `metadata` a `META_SEG` written now would contain.
    fn committed_metadata_view(&self) -> MetadataStore {
        let mut view = self.metadata.clone();
        view.retain_ids(|vector_id| self.metadata_id_is_committed(vector_id));
        view
    }

    /// Restore the authoritative metadata snapshot (ADR-280 §1, §5).
    ///
    /// Only the newest committed generation is authoritative; every older
    /// generation it does not name as an ancestor is superseded history. The
    /// scan therefore walks `META_SEG` entries newest-first and stops as soon
    /// as the chain reaches a full snapshot, so open cost tracks the live
    /// snapshot and its delta chain rather than the number of commits ever
    /// made.
    ///
    /// Corruption never bricks the artifact. Because every generation is
    /// applied on top of the one before it, a chain damaged anywhere still has
    /// a well-defined committed state: the newest snapshot plus the
    /// consecutive valid deltas that follow it. Replay therefore serves that
    /// longest complete prefix and reports what it had to drop through
    /// [`Self::metadata_recovery`], rather than refusing to open because one
    /// byte of one delta went bad. Only a chain with no readable snapshot at
    /// all is an error, since then nothing committed can be reconstructed.
    fn restore_metadata(&mut self) -> Result<(), RvfError> {
        let meta_offsets: Vec<u64> = self
            .segment_dir
            .iter()
            .filter(|&&(_, _, _, segment_type)| segment_type == SegmentType::Meta as u8)
            .map(|&(_, offset, _, _)| offset)
            .collect();
        if meta_offsets.is_empty() {
            return Ok(());
        }

        // Walk newest-first and stop at the first full snapshot: that is the
        // newest one, and everything older than it is superseded history.
        // Generations that cannot be read or decoded are collected as gaps.
        let mut decoded_bytes = 0usize;
        let mut newest_first: Vec<(u64, usize, MetadataSegment)> = Vec::new();
        let mut visited: HashSet<u64> = HashSet::new();
        let mut damaged = 0u64;
        let mut found_snapshot = false;

        for &offset in meta_offsets.iter().rev() {
            if newest_first.len() > rvf_types::metadata::MAX_META_DELTAS {
                return Err(err(ErrorCode::MetadataReplayLimitExceeded));
            }
            let payload = {
                let mut reader = BufReader::new(&self.file);
                read_path::read_segment_payload(&mut reader, offset)
                    .ok()
                    .map(|(_, payload)| payload)
            };
            let Some(payload) = payload else {
                damaged += 1;
                continue;
            };
            decoded_bytes = decoded_bytes
                .checked_add(payload.len())
                .ok_or_else(|| err(ErrorCode::MetadataReplayLimitExceeded))?;
            if decoded_bytes > rvf_types::metadata::MAX_META_DECODED_BYTES {
                return Err(err(ErrorCode::MetadataReplayLimitExceeded));
            }
            let Ok(segment) = MetadataSegment::decode(&payload) else {
                damaged += 1;
                continue;
            };
            // A repeated generation is manifest corruption; the newer copy is
            // the one already recorded, so ignore the older duplicate.
            if !visited.insert(segment.generation) {
                damaged += 1;
                continue;
            }
            found_snapshot = segment.full_snapshot;
            newest_first.push((offset, payload.len(), segment));
            if found_snapshot {
                break;
            }
        }
        if !found_snapshot {
            // Without a readable snapshot there is no base to apply deltas to.
            return Err(err(ErrorCode::InvalidMetadata));
        }

        // Apply forward from the snapshot, stopping at the first generation
        // that is missing or does not name its predecessor as its base.
        newest_first.reverse();
        let mut chain = newest_first.into_iter();
        let (snapshot_offset, snapshot_bytes, snapshot) = chain
            .next()
            .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
        let snapshot_generation = snapshot.generation;
        let mut chain_bytes = snapshot_bytes;
        let mut applied_offsets: HashSet<u64> = HashSet::from([snapshot_offset]);
        let mut applied = vec![snapshot];
        let mut remaining = chain;
        for (offset, bytes, segment) in remaining.by_ref() {
            let expected = applied
                .last()
                .and_then(|previous: &MetadataSegment| previous.generation.checked_add(1));
            if Some(segment.generation) != expected
                || segment.base_generation != applied.last().map(|p| p.generation)
            {
                damaged += 1;
                break;
            }
            chain_bytes = chain_bytes.saturating_add(bytes);
            applied_offsets.insert(offset);
            applied.push(segment);
        }
        // Everything after the break is unreachable too, so it is dropped as
        // well and must be counted -- the CLI reports this number verbatim.
        damaged = damaged.saturating_add(remaining.count() as u64);
        let latest = applied
            .last()
            .map(|segment| segment.generation)
            .unwrap_or(snapshot_generation);

        let mut metadata = MetadataStore::new();
        let mut file_metadata = BTreeMap::new();
        for segment in &applied {
            if segment.full_snapshot {
                metadata = MetadataStore::new();
                file_metadata.clear();
            }
            for record in &segment.file_metadata {
                match wire_to_runtime(&record.value)? {
                    MetadataValue::DeleteField => {
                        file_metadata.remove(&record.key);
                    }
                    value => {
                        file_metadata.insert(record.key.clone(), value);
                    }
                }
            }
            for record in &segment.records {
                match &record.operation {
                    WireMetadataRecordOp::DeleteRecord => metadata.remove_ids(&[record.vector_id]),
                    WireMetadataRecordOp::Upsert(fields) => {
                        let fields = fields
                            .iter()
                            .map(|(id, value)| Ok((*id, wire_to_runtime(value)?)))
                            .collect::<Result<Vec<_>, RvfError>>()?;
                        metadata.insert(record.vector_id, fields);
                    }
                }
            }
        }
        let is_cow = self
            .segment_dir
            .iter()
            .any(|entry| entry.3 == SegmentType::CowMap as u8);
        // Deletion state comes from the newest manifest but metadata now comes
        // from a possibly older prefix, so a dropped generation that carried a
        // deletion resurrects its record. On an intact chain that mismatch
        // means the artifact really is inconsistent and is fatal; on a
        // truncated one it is a known consequence of the damage, so the
        // resurrected records are discarded and counted instead.
        let truncated = damaged > 0;
        let mut resurrected: Vec<u64> = Vec::new();
        for vector_id in metadata.ids() {
            if (!is_cow && self.vectors.get(vector_id).is_none())
                || self.deletion_bitmap.is_deleted(vector_id)
            {
                if !truncated {
                    return Err(err(ErrorCode::InvalidMetadata));
                }
                resurrected.push(vector_id);
            }
        }
        metadata.remove_ids(&resurrected);

        self.metadata = metadata;
        self.file_metadata = file_metadata;
        self.metadata_chain = MetadataChainState {
            generation: latest,
            snapshot_generation,
            chain_bytes,
            // A truncated replay leaves generations on disk that the chain no
            // longer reaches, and may have discarded resurrected records that
            // the applied prefix still contains. Re-anchoring the next write as
            // a full snapshot makes the file converge on the served state
            // instead of extending a chain that disagrees with it.
            needs_snapshot: truncated,
        };
        self.dropped_metadata_generations = damaged;
        self.dropped_metadata_records = resurrected.len() as u64;
        // Record which generations the chain no longer reaches, but do NOT
        // prune them here. Pruning edits `segment_dir`, which any manifest
        // write persists, while the re-anchoring snapshot is written only by
        // `write_metadata_generation`. Splitting them lets a manifest-only
        // mutation commit the pruning alone, after which the damage is no
        // longer discoverable, the chain reads as intact, and the resurrected
        // records above become a fatal inconsistency. The two happen together
        // in `write_metadata_generation` instead.
        self.orphaned_metadata_offsets = if truncated {
            self.segment_dir
                .iter()
                .filter(|&&(_, offset, _, segment_type)| {
                    segment_type == SegmentType::Meta as u8 && !applied_offsets.contains(&offset)
                })
                .map(|&(_, offset, _, _)| offset)
                .collect()
        } else {
            Vec::new()
        };
        // The served state is what the next generation is encoded against. It
        // can differ from the applied prefix when records were discarded above,
        // which is safe only because that next generation is a full snapshot.
        self.committed_metadata = self.metadata.clone();
        self.committed_file_metadata = self.file_metadata.clone();
        Ok(())
    }
}

/// Declare the schema entries a `META_SEG` payload needs to be self-describing.
///
/// The wire format requires every field referenced by a record to be declared
/// in the same segment. The legacy field-id API carries no schema, so types are
/// inferred from the values present and a field seen only as null is
/// conservatively declared nullable UTF-8.
fn build_metadata_schema(
    records: &[WireMetadataRecord],
) -> Result<Vec<MetadataSchemaEntry>, RvfError> {
    let mut schema_by_id: BTreeMap<u16, MetadataType> = BTreeMap::new();
    let mut nullable: HashSet<u16> = HashSet::new();
    // Fields named only by a null or a deletion carry no type of their own, so
    // they are typed after the pass in case a real value appears elsewhere.
    let mut untyped: HashSet<u16> = HashSet::new();
    for record in records {
        let WireMetadataRecordOp::Upsert(fields) = &record.operation else {
            continue;
        };
        for (field_id, value) in fields {
            let value_type = match value {
                WireMetadataValue::Null => {
                    nullable.insert(*field_id);
                    untyped.insert(*field_id);
                    continue;
                }
                // A field deletion names a field without carrying a value, so
                // it still needs a declaration but constrains no type.
                WireMetadataValue::DeleteField => {
                    untyped.insert(*field_id);
                    continue;
                }
                WireMetadataValue::String(_) => MetadataType::String,
                WireMetadataValue::Bytes(_) => MetadataType::Bytes,
                WireMetadataValue::I64(_) => MetadataType::I64,
                WireMetadataValue::U64(_) => MetadataType::U64,
                WireMetadataValue::F64(_) => MetadataType::F64,
                WireMetadataValue::Bool(_) => MetadataType::Bool,
            };
            if schema_by_id
                .insert(*field_id, value_type)
                .is_some_and(|previous| previous != value_type)
            {
                return Err(err(ErrorCode::InvalidManifest));
            }
        }
    }
    for &field_id in &untyped {
        schema_by_id.entry(field_id).or_insert(MetadataType::String);
    }
    Ok(schema_by_id
        .into_iter()
        .map(|(field_id, value_type)| MetadataSchemaEntry {
            field_id,
            value_type,
            nullable: nullable.contains(&field_id),
            name: format!("field.{field_id}"),
        })
        .collect())
}

fn runtime_to_wire(value: &MetadataValue) -> WireMetadataValue {
    match value {
        MetadataValue::Null => WireMetadataValue::Null,
        MetadataValue::String(value) => WireMetadataValue::String(value.clone()),
        MetadataValue::Bytes(value) => WireMetadataValue::Bytes(value.clone()),
        MetadataValue::I64(value) => WireMetadataValue::I64(*value),
        MetadataValue::U64(value) => WireMetadataValue::U64(*value),
        MetadataValue::F64(value) => WireMetadataValue::F64(*value),
        MetadataValue::Bool(value) => WireMetadataValue::Bool(*value),
        MetadataValue::DeleteField => WireMetadataValue::DeleteField,
    }
}

fn wire_to_runtime(value: &WireMetadataValue) -> Result<MetadataValue, RvfError> {
    Ok(match value {
        WireMetadataValue::Null => MetadataValue::Null,
        WireMetadataValue::String(value) => MetadataValue::String(value.clone()),
        WireMetadataValue::Bytes(value) => MetadataValue::Bytes(value.clone()),
        WireMetadataValue::I64(value) => MetadataValue::I64(*value),
        WireMetadataValue::U64(value) => MetadataValue::U64(*value),
        WireMetadataValue::F64(value) if value.is_finite() => MetadataValue::F64(*value),
        WireMetadataValue::F64(_) => return Err(err(ErrorCode::InvalidManifest)),
        WireMetadataValue::Bool(value) => MetadataValue::Bool(*value),
        WireMetadataValue::DeleteField => MetadataValue::DeleteField,
    })
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

    #[test]
    fn metadata_and_embedding_identity_survive_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("durable_metadata.rvf");
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let vectors = [vec![1.0, 0.0], vec![0.0, 1.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let metadata = vec![
            MetadataEntry {
                field_id: 1,
                value: MetadataValue::String("first".into()),
            },
            MetadataEntry {
                field_id: 2,
                value: MetadataValue::Null,
            },
            MetadataEntry {
                field_id: 1,
                value: MetadataValue::String("second".into()),
            },
            MetadataEntry {
                field_id: 2,
                value: MetadataValue::Bool(true),
            },
        ];
        store
            .ingest_batch(&refs, &[10, 20], Some(&metadata))
            .unwrap();
        store
            .set_file_metadata(
                "application.owner".into(),
                MetadataValue::String("team-a".into()),
            )
            .unwrap();
        store
            .set_system_file_metadata(
                "rvf.embedding.identity".into(),
                MetadataValue::Bytes(br#"{"model_id":"fixture"}"#.to_vec()),
            )
            .unwrap();
        store
            .set_system_file_metadata(
                "rvf.embedding.space_id".into(),
                MetadataValue::String("space-a".into()),
            )
            .unwrap();
        drop(store);

        let mut reopened = RvfStore::open(&path).unwrap();
        assert_eq!(
            reopened.get_metadata(10).unwrap(),
            vec![
                MetadataEntry {
                    field_id: 1,
                    value: MetadataValue::String("first".into())
                },
                MetadataEntry {
                    field_id: 2,
                    value: MetadataValue::Null
                },
            ]
        );
        assert_eq!(
            reopened.get_file_metadata("application.owner"),
            Some(&MetadataValue::String("team-a".into()))
        );
        assert_eq!(
            reopened.verify_embedding_space("space-b"),
            EmbeddingCompatibility::VectorReadOnly
        );
        assert_eq!(reopened.read_all_vectors().len(), 2);
        assert!(matches!(
            reopened.ingest_batch(&[&[0.5f32, 0.5]], &[30], None),
            Err(RvfError::Code(ErrorCode::ReadOnly))
        ));
    }

    #[test]
    fn metadata_is_removed_with_vector_delete_and_stays_removed() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("metadata_delete.rvf");
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: 2,
                ..Default::default()
            },
        )
        .unwrap();
        store
            .ingest_batch(
                &[&[1.0, 0.0]],
                &[7],
                Some(&[MetadataEntry {
                    field_id: 3,
                    value: MetadataValue::U64(42),
                }]),
            )
            .unwrap();
        store.delete(&[7]).unwrap();
        assert!(store.get_metadata(7).is_none());
        drop(store);
        let reopened = RvfStore::open_readonly(&path).unwrap();
        assert!(reopened.get_metadata(7).is_none());
    }

    #[test]
    fn metadata_filter_budget_and_cancellation_fail_closed_with_stats() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("metadata_budget.rvf");
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: 2,
                ..Default::default()
            },
        )
        .unwrap();
        store
            .ingest_batch(
                &[&[1.0, 0.0], &[0.0, 1.0]],
                &[1, 2],
                Some(&[
                    MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::Bool(true),
                    },
                    MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::Bool(true),
                    },
                ]),
            )
            .unwrap();
        let mut options = QueryOptions {
            filter: Some(FilterExpr::Eq(1, FilterValue::Bool(true))),
            metadata_filter_max_records: 1,
            ..Default::default()
        };
        assert!(matches!(
            store.query(&[1.0, 0.0], 2, &options),
            Err(RvfError::Code(ErrorCode::FilterBudgetExceeded))
        ));
        assert_eq!(store.metadata_filter_stats().aborted_scans, 1);

        let cancelled = std::sync::Arc::new(AtomicBool::new(true));
        options.metadata_filter_max_records = 10;
        options.metadata_filter_cancel = Some(cancelled);
        assert!(matches!(
            store.query(&[1.0, 0.0], 2, &options),
            Err(RvfError::Code(ErrorCode::FilterBudgetExceeded))
        ));
        assert_eq!(store.metadata_filter_stats().aborted_scans, 2);
    }

    #[test]
    fn cow_metadata_delete_does_not_resurrect_after_reopen() {
        let dir = TempDir::new().unwrap();
        let parent_path = dir.path().join("parent.rvf");
        let child_path = dir.path().join("child.rvf");
        let mut parent = RvfStore::create(
            &parent_path,
            RvfOptions {
                dimension: 2,
                ..Default::default()
            },
        )
        .unwrap();
        parent
            .ingest_batch(
                &[&[1.0, 0.0]],
                &[9],
                Some(&[MetadataEntry {
                    field_id: 1,
                    value: MetadataValue::String("inherited".into()),
                }]),
            )
            .unwrap();
        let mut child = parent.branch(&child_path).unwrap();
        assert!(child.get_metadata(9).is_some());
        child.delete(&[9]).unwrap();
        assert!(child.get_metadata(9).is_none());
        drop(child);

        let reopened = RvfStore::open_readonly(&child_path).unwrap();
        assert!(reopened.get_metadata(9).is_none());
    }

    #[test]
    fn record_metadata_ingest_supports_unequal_fields_and_tombstones() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("record_metadata.rvf");
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: 2,
                ..Default::default()
            },
        )
        .unwrap();
        store
            .ingest_batch_with_metadata(
                &[&[1.0, 0.0], &[0.0, 1.0]],
                &[10, 20],
                &[
                    VectorMetadata {
                        vector_id: 10,
                        fields: vec![
                            MetadataEntry {
                                field_id: 1,
                                value: MetadataValue::String("ten".into()),
                            },
                            MetadataEntry {
                                field_id: 2,
                                value: MetadataValue::Bool(true),
                            },
                        ],
                        delete_record: false,
                    },
                    VectorMetadata {
                        vector_id: 20,
                        fields: vec![MetadataEntry {
                            field_id: 1,
                            value: MetadataValue::String("twenty".into()),
                        }],
                        delete_record: false,
                    },
                ],
            )
            .unwrap();
        assert_eq!(store.get_metadata(10).unwrap().len(), 2);
        assert_eq!(store.get_metadata(20).unwrap().len(), 1);

        store
            .ingest_batch_with_metadata(
                &[&[1.0, 0.0]],
                &[10],
                &[VectorMetadata {
                    vector_id: 10,
                    fields: vec![MetadataEntry {
                        field_id: 2,
                        value: MetadataValue::DeleteField,
                    }],
                    delete_record: false,
                }],
            )
            .unwrap();
        assert_eq!(
            store.get_metadata(10).unwrap(),
            vec![MetadataEntry {
                field_id: 1,
                value: MetadataValue::String("ten".into()),
            }]
        );

        store
            .ingest_batch_with_metadata(
                &[&[0.0, 1.0]],
                &[20],
                &[VectorMetadata {
                    vector_id: 20,
                    fields: vec![],
                    delete_record: true,
                }],
            )
            .unwrap();
        assert!(store.get_metadata(20).is_none());
        drop(store);
        let reopened = RvfStore::open_readonly(&path).unwrap();
        assert_eq!(reopened.get_metadata(10).unwrap().len(), 1);
        assert!(reopened.get_metadata(20).is_none());
    }

    #[test]
    fn record_metadata_rejection_is_atomic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("record_metadata_atomic.rvf");
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let epoch = store.status().current_epoch;
        let result = store.ingest_batch_with_metadata(
            &[&[1.0, 0.0]],
            &[30],
            &[VectorMetadata {
                vector_id: 999,
                fields: vec![MetadataEntry {
                    field_id: 1,
                    value: MetadataValue::String("unknown".into()),
                }],
                delete_record: false,
            }],
        );
        assert!(matches!(
            result,
            Err(RvfError::Code(ErrorCode::InvalidMetadata))
        ));
        assert!(store.read_all_vectors().is_empty());
        assert_eq!(store.status().current_epoch, epoch);

        let duplicate_fields = store.ingest_batch_with_metadata(
            &[&[1.0, 0.0]],
            &[30],
            &[VectorMetadata {
                vector_id: 30,
                fields: vec![
                    MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(1),
                    },
                    MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(2),
                    },
                ],
                delete_record: false,
            }],
        );
        assert!(matches!(
            duplicate_fields,
            Err(RvfError::Code(ErrorCode::InvalidMetadata))
        ));
        assert!(store.read_all_vectors().is_empty());
        drop(store);
        assert!(RvfStore::open_readonly(&path)
            .unwrap()
            .read_all_vectors()
            .is_empty());
    }

    /// Criterion 2 (gap): bytes, signed-integer, and finite-float per-vector
    /// values round-trip through a full store close/reopen. The existing
    /// round-trip coverage exercised strings, nulls, and booleans only.
    #[test]
    fn bytes_i64_f64_values_round_trip_through_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("scalar_value_roundtrip.rvf");
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let fields = vec![
            MetadataEntry {
                field_id: 1,
                value: MetadataValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            },
            MetadataEntry {
                field_id: 2,
                value: MetadataValue::I64(-1_234_567_890),
            },
            MetadataEntry {
                field_id: 3,
                value: MetadataValue::F64(core::f64::consts::PI),
            },
        ];
        store
            .ingest_batch_with_metadata(
                &[&[1.0, 0.0]],
                &[42],
                &[VectorMetadata {
                    vector_id: 42,
                    fields: fields.clone(),
                    delete_record: false,
                }],
            )
            .unwrap();
        assert_eq!(store.get_metadata(42).unwrap(), fields);
        drop(store);

        let reopened = RvfStore::open_readonly(&path).unwrap();
        assert_eq!(
            reopened.get_metadata(42).unwrap(),
            fields,
            "bytes, i64, and finite f64 values must survive close/reopen byte-for-byte"
        );
    }

    /// Criterion 7 (gap): a single atomic batch that both tombstones and
    /// upserts metadata for the same vector is rejected as a whole, and the
    /// store is left exactly as it was (no vector, no metadata, no epoch bump).
    #[test]
    fn same_batch_vector_delete_and_metadata_upsert_fails_atomically() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("batch_delete_upsert_atomic.rvf");
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let epoch = store.status().current_epoch;
        let result = store.ingest_batch_with_metadata(
            &[&[1.0, 0.0]],
            &[5],
            &[
                // Tombstone the record for vector 5 ...
                VectorMetadata {
                    vector_id: 5,
                    fields: vec![],
                    delete_record: true,
                },
                // ... while also upserting metadata for the same vector 5 in
                // the same batch. The two operations contradict, so the whole
                // batch must be refused rather than partially applied.
                VectorMetadata {
                    vector_id: 5,
                    fields: vec![MetadataEntry {
                        field_id: 1,
                        value: MetadataValue::U64(1),
                    }],
                    delete_record: false,
                },
            ],
        );
        assert!(matches!(
            result,
            Err(RvfError::Code(ErrorCode::InvalidMetadata))
        ));
        assert!(store.read_all_vectors().is_empty());
        assert!(store.get_metadata(5).is_none());
        assert_eq!(store.status().current_epoch, epoch);
        drop(store);
        let reopened = RvfStore::open_readonly(&path).unwrap();
        assert!(reopened.read_all_vectors().is_empty());
        assert!(reopened.get_metadata(5).is_none());
    }

    /// Criterion 5: metadata-bearing search hits return the committed values,
    /// opt-in and matching `get_metadata`, both before and after reopen. With
    /// the flag off (the default) hits carry no metadata.
    #[test]
    fn include_metadata_returns_committed_values_on_hits() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("include_metadata.rvf");
        let mut store = RvfStore::create(
            &path,
            RvfOptions {
                dimension: 2,
                ..Default::default()
            },
        )
        .unwrap();
        store
            .ingest_batch_with_metadata(
                &[&[1.0, 0.0], &[0.0, 1.0]],
                &[10, 20],
                &[
                    VectorMetadata {
                        vector_id: 10,
                        fields: vec![MetadataEntry {
                            field_id: 1,
                            value: MetadataValue::String("ten".into()),
                        }],
                        delete_record: false,
                    },
                    // A present but empty record: hits must report Some(vec![]).
                    VectorMetadata {
                        vector_id: 20,
                        fields: vec![],
                        delete_record: false,
                    },
                ],
            )
            .unwrap();

        let query = [1.0f32, 0.0];
        // Default query: no metadata attached, zero cost.
        let plain = store.query(&query, 2, &QueryOptions::default()).unwrap();
        assert!(plain.iter().all(|hit| hit.metadata.is_none()));

        let with_meta = QueryOptions {
            include_metadata: true,
            ..Default::default()
        };
        let assert_hits = |store: &RvfStore| {
            let hits = store.query(&query, 2, &with_meta).unwrap();
            assert_eq!(hits.len(), 2);
            for hit in &hits {
                assert_eq!(
                    hit.metadata,
                    Some(store.get_metadata(hit.id).unwrap_or_default()),
                    "hit metadata must match the committed get_metadata view"
                );
            }
            let ten = hits.iter().find(|h| h.id == 10).unwrap();
            assert_eq!(
                ten.metadata,
                Some(vec![MetadataEntry {
                    field_id: 1,
                    value: MetadataValue::String("ten".into()),
                }])
            );
            let twenty = hits.iter().find(|h| h.id == 20).unwrap();
            assert_eq!(
                twenty.metadata,
                Some(vec![]),
                "an empty-but-present record surfaces as Some(vec![])"
            );
        };
        assert_hits(&store);
        drop(store);
        let reopened = RvfStore::open_readonly(&path).unwrap();
        assert_hits(&reopened);
    }
}
