//! HNSW PostgreSQL Access Method Implementation
//!
//! This module implements HNSW as a proper PostgreSQL index access method,
//! storing the graph structure in PostgreSQL pages for persistence.

use pgrx::pg_sys::{
    self, bytea, BlockNumber, Buffer, Cost, Datum, IndexAmRoutine, IndexBuildResult,
    IndexBulkDeleteCallback, IndexBulkDeleteResult, IndexInfo, IndexPath, IndexScanDesc,
    IndexUniqueCheck, IndexVacuumInfo, ItemPointer, ItemPointerData, NodeTag, Page, PageHeaderData,
    PlannerInfo, Relation, ScanDirection, ScanKey, Selectivity, Size, TIDBitmap,
};
use pgrx::prelude::*;
use pgrx::Internal;
use std::mem::size_of;
use std::ptr;

use crate::distance::{distance, DistanceMetric};
use crate::index::HnswConfig;

// ============================================================================
// Page Layout Constants
// ============================================================================

/// Magic number for HNSW index pages (ASCII "HNSW")
const HNSW_MAGIC: u32 = 0x484E5357;

/// Page type identifiers
const HNSW_PAGE_META: u8 = 0;
const HNSW_PAGE_NODE: u8 = 1;
#[allow(dead_code)]
const HNSW_PAGE_DELETED: u8 = 2;

/// Maximum neighbors per node (aligned with default M)
#[allow(dead_code)]
const MAX_NEIGHBORS_L0: usize = 32; // 2*M for layer 0
#[allow(dead_code)]
const MAX_NEIGHBORS: usize = 16; // M for other layers
#[allow(dead_code)]
const MAX_LAYERS: usize = 16; // Maximum graph layers

/// P_NEW equivalent for allocating new pages
const P_NEW_BLOCK: BlockNumber = pg_sys::InvalidBlockNumber;

// ============================================================================
// Page Structures
// ============================================================================

/// Metadata page (page 0)
#[repr(C)]
#[derive(Copy, Clone)]
struct HnswMetaPage {
    magic: u32,
    version: u32,
    dimensions: u32,
    m: u16,
    m0: u16,
    ef_construction: u32,
    entry_point: BlockNumber,
    max_layer: u16,
    metric: u8,
    _padding: u8,
    node_count: u64,
    next_block: BlockNumber,
}

impl Default for HnswMetaPage {
    fn default() -> Self {
        Self {
            magic: HNSW_MAGIC,
            version: 1,
            dimensions: 0,
            m: 16,
            m0: 32,
            ef_construction: 64,
            entry_point: pg_sys::InvalidBlockNumber,
            max_layer: 0,
            metric: 0, // L2 by default
            _padding: 0,
            node_count: 0,
            next_block: 1, // First node page
        }
    }
}

/// Node page header
#[repr(C)]
#[derive(Copy, Clone)]
struct HnswNodePageHeader {
    page_type: u8,
    #[allow(dead_code)]
    max_layer: u8,
    _padding: [u8; 2],
    item_id: ItemPointerData, // TID of the heap tuple
}

/// Neighbor entry in the graph
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
struct HnswNeighbor {
    block_num: BlockNumber,
    distance: f32,
}

// ============================================================================
// Index Scan State
// ============================================================================

/// State for scanning an HNSW index
#[allow(dead_code)]
struct HnswScanState {
    query_vector: Vec<f32>,
    k: usize,
    ef_search: usize,
    metric: DistanceMetric,
    dimensions: usize,
    results: Vec<(BlockNumber, ItemPointerData, f32)>,
    current_pos: usize,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get metadata page from index relation
/// Returns (page pointer, buffer)
/// Note: Page in pgrx is already a pointer type (*mut i8)
unsafe fn get_meta_page(index_rel: Relation) -> (Page, Buffer) {
    let buffer = pg_sys::ReadBuffer(index_rel, 0);
    pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buffer);
    (page, buffer)
}

/// Get or create metadata page
/// Returns (page pointer, buffer)
/// For new indexes, uses P_NEW to allocate the first page
unsafe fn get_or_create_meta_page(index_rel: Relation, for_write: bool) -> (Page, Buffer) {
    // Check if the relation has any blocks
    // Use MAIN_FORKNUM (0) for the main relation fork
    let nblocks =
        pg_sys::RelationGetNumberOfBlocksInFork(index_rel, pg_sys::ForkNumber::MAIN_FORKNUM);

    let buffer = if nblocks == 0 {
        // New index - allocate first page using P_NEW (InvalidBlockNumber)
        pg_sys::ReadBuffer(index_rel, P_NEW_BLOCK)
    } else {
        // Existing index - read block 0
        pg_sys::ReadBuffer(index_rel, 0)
    };

    if for_write {
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    } else {
        pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
    }
    let page = pg_sys::BufferGetPage(buffer);
    (page, buffer)
}

/// Read metadata from page
unsafe fn read_metadata(page: Page) -> HnswMetaPage {
    let header = page as *const PageHeaderData;
    let data_ptr = (header as *const u8).add(std::mem::size_of::<PageHeaderData>());
    ptr::read(data_ptr as *const HnswMetaPage)
}

/// Write metadata to page
unsafe fn write_metadata(page: Page, meta: &HnswMetaPage) {
    let header = page as *mut PageHeaderData;
    let data_ptr =
        (header as *mut u8).add(std::mem::size_of::<PageHeaderData>()) as *mut HnswMetaPage;
    ptr::write(data_ptr, *meta);
}

/// Allocate a new node page
#[allow(dead_code)]
unsafe fn allocate_node_page(
    index_rel: Relation,
    vector: &[f32],
    tid: ItemPointerData,
    max_layer: usize,
) -> BlockNumber {
    // Get a new buffer using InvalidBlockNumber (equivalent to P_NEW)
    let buffer = pg_sys::ReadBuffer(index_rel, P_NEW_BLOCK);
    let block = pg_sys::BufferGetBlockNumber(buffer);

    pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let page = pg_sys::BufferGetPage(buffer);

    // Initialize page
    pg_sys::PageInit(page, pg_sys::BLCKSZ as Size, 0);

    // Write node header
    let header = page as *mut PageHeaderData;
    let data_ptr = (header as *mut u8).add(std::mem::size_of::<PageHeaderData>());
    let node_header = HnswNodePageHeader {
        page_type: HNSW_PAGE_NODE,
        max_layer: max_layer as u8,
        _padding: [0; 2],
        item_id: tid,
    };
    ptr::write(data_ptr as *mut HnswNodePageHeader, node_header);

    // Write vector data after header
    let vector_ptr = data_ptr.add(std::mem::size_of::<HnswNodePageHeader>()) as *mut f32;
    for (i, &val) in vector.iter().enumerate() {
        ptr::write(vector_ptr.add(i), val);
    }

    // Mark buffer dirty and unlock
    pg_sys::MarkBufferDirty(buffer);
    pg_sys::UnlockReleaseBuffer(buffer);

    block
}

/// Read vector from node page
#[allow(dead_code)]
unsafe fn read_vector(
    index_rel: Relation,
    block: BlockNumber,
    dimensions: usize,
) -> Option<Vec<f32>> {
    if block == pg_sys::InvalidBlockNumber {
        return None;
    }

    let buffer = pg_sys::ReadBuffer(index_rel, block);
    pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buffer);

    let header = page as *const PageHeaderData;
    let data_ptr = (header as *const u8).add(std::mem::size_of::<PageHeaderData>());
    let vector_ptr = data_ptr.add(std::mem::size_of::<HnswNodePageHeader>()) as *const f32;

    let mut vector = Vec::with_capacity(dimensions);
    for i in 0..dimensions {
        vector.push(ptr::read(vector_ptr.add(i)));
    }

    pg_sys::UnlockReleaseBuffer(buffer);
    Some(vector)
}

/// Calculate distance between query and node
#[allow(dead_code)]
unsafe fn calculate_distance(
    index_rel: Relation,
    query: &[f32],
    block: BlockNumber,
    dimensions: usize,
    metric: DistanceMetric,
) -> f32 {
    match read_vector(index_rel, block, dimensions) {
        Some(vec) => distance(query, &vec, metric),
        None => f32::MAX,
    }
}

// ============================================================================
// Access Method Callbacks
// ============================================================================

/// Build callback - builds the HNSW index from scratch
///
/// Extracts vector dimensions from the indexed column's type modifier.
/// The column must be declared with explicit dimensions, e.g., `ruvector(384)`.
/// Returns an error if dimensions are not specified.
#[pg_guard]
unsafe extern "C" fn hnsw_build(
    _heap: Relation,
    index: Relation,
    _index_info: *mut IndexInfo,
) -> *mut IndexBuildResult {
    pgrx::log!("HNSW: Starting index build");

    // Extract dimensions from the indexed column's type modifier
    // When user defines ruvector(384), typmod = 384
    let dimensions = {
        // RelationGetDescr(index) -> (*index).rd_att
        let index_desc = (*index).rd_att;
        if index_desc.is_null() || (*index_desc).natts < 1 {
            pgrx::error!("HNSW: Cannot build index - no indexed columns found");
        }

        // TupleDescAttr(desc, 0) -> (*desc).attrs.as_ptr().add(0)
        let attr = (*index_desc).attrs.as_ptr().add(0);
        let typmod = (*attr).atttypmod;

        if typmod > 0 {
            typmod as u32
        } else {
            // typmod = -1 means dimensions not specified in type declaration
            // This happens with: CREATE TABLE t (v ruvector) instead of ruvector(384)
            pgrx::error!(
                "HNSW: Vector column must have dimensions specified. \
                 Use ruvector(dimensions) instead of ruvector, e.g., ruvector(384)"
            );
        }
    };

    pgrx::log!("HNSW: Building index with {} dimensions", dimensions);
    let config = HnswConfig::default();

    // Initialize metadata page
    let (page, buffer) = get_or_create_meta_page(index, true);
    pg_sys::PageInit(page, pg_sys::BLCKSZ as Size, 0);

    let meta = HnswMetaPage {
        dimensions: dimensions as u32,
        m: config.m as u16,
        m0: config.m0 as u16,
        ef_construction: config.ef_construction as u32,
        metric: match config.metric {
            DistanceMetric::Euclidean => 0,
            DistanceMetric::Cosine => 1,
            DistanceMetric::InnerProduct => 2,
            _ => 0,
        },
        ..Default::default()
    };

    write_metadata(page, &meta);
    pg_sys::MarkBufferDirty(buffer);
    pg_sys::UnlockReleaseBuffer(buffer);

    // Scan heap and build index
    // This is a simplified version - full implementation would use IndexBuildHeapScan
    let tuple_count = 0.0;

    pgrx::log!(
        "HNSW: Index build complete, {} tuples indexed",
        tuple_count as u64
    );

    // Return build result
    let mut result = PgBox::<IndexBuildResult>::alloc0();
    result.heap_tuples = tuple_count;
    result.index_tuples = tuple_count;
    result.into_pg()
}

/// Build empty index callback
///
/// Creates an empty HNSW index with proper dimensions from the column's type modifier.
#[pg_guard]
unsafe extern "C" fn hnsw_buildempty(index: Relation) {
    pgrx::log!("HNSW: Building empty index");

    // Extract dimensions from the indexed column's type modifier
    let dimensions = {
        // RelationGetDescr(index) -> (*index).rd_att
        let index_desc = (*index).rd_att;
        if !index_desc.is_null() && (*index_desc).natts >= 1 {
            // TupleDescAttr(desc, 0) -> (*desc).attrs.as_ptr().add(0)
            let attr = (*index_desc).attrs.as_ptr().add(0);
            let typmod = (*attr).atttypmod;
            if typmod > 0 {
                typmod as u32
            } else {
                0
            }
        } else {
            0
        }
    };

    // Initialize metadata page only
    let (page, buffer) = get_or_create_meta_page(index, true);
    pg_sys::PageInit(page, pg_sys::BLCKSZ as Size, 0);

    let meta = HnswMetaPage {
        dimensions,
        ..Default::default()
    };
    write_metadata(page, &meta);

    pg_sys::MarkBufferDirty(buffer);
    pg_sys::UnlockReleaseBuffer(buffer);
}

/// Insert callback - insert a single tuple into the index
#[pg_guard]
unsafe extern "C" fn hnsw_insert(
    index: Relation,
    _values: *mut Datum,
    isnull: *mut bool,
    _heap_tid: ItemPointer,
    _heap: Relation,
    _check_unique: IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut IndexInfo,
) -> bool {
    // Check for null
    if *isnull {
        return false;
    }

    // Get metadata
    let (meta_page, meta_buffer) = get_meta_page(index);
    let _meta = read_metadata(meta_page);
    pg_sys::UnlockReleaseBuffer(meta_buffer);

    // TODO: Extract vector from datum and insert into graph
    // For now, just return success
    true
}

/// Bulk delete callback
#[pg_guard]
unsafe extern "C" fn hnsw_bulkdelete(
    _info: *mut IndexVacuumInfo,
    stats: *mut IndexBulkDeleteResult,
    _callback: IndexBulkDeleteCallback,
    _callback_state: *mut ::std::os::raw::c_void,
) -> *mut IndexBulkDeleteResult {
    pgrx::log!("HNSW: Bulk delete called");

    // Return stats (simplified implementation)
    if stats.is_null() {
        let new_stats = PgBox::<IndexBulkDeleteResult>::alloc0();
        new_stats.into_pg()
    } else {
        stats
    }
}

/// Vacuum cleanup callback
#[pg_guard]
unsafe extern "C" fn hnsw_vacuumcleanup(
    _info: *mut IndexVacuumInfo,
    stats: *mut IndexBulkDeleteResult,
) -> *mut IndexBulkDeleteResult {
    pgrx::log!("HNSW: Vacuum cleanup called");

    if stats.is_null() {
        let new_stats = PgBox::<IndexBulkDeleteResult>::alloc0();
        new_stats.into_pg()
    } else {
        stats
    }
}

/// Cost estimate callback
#[pg_guard]
unsafe extern "C" fn hnsw_costestimate(
    _root: *mut PlannerInfo,
    path: *mut IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut Cost,
    index_total_cost: *mut Cost,
    index_selectivity: *mut Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    // Simplified cost estimation
    // HNSW has logarithmic search complexity
    let tuples = if let Some(info) = (*path).indexinfo.as_ref() {
        (*info).tuples
    } else {
        1000.0
    };

    // Startup cost is minimal
    *index_startup_cost = 0.0;

    // Total cost is O(log n) for HNSW
    let log_tuples = tuples.max(1.0).ln();
    *index_total_cost = log_tuples * 10.0; // Scale factor for page accesses

    // HNSW provides good selectivity for top-k queries
    *index_selectivity = 0.01; // Typically returns ~1% of tuples
    *index_correlation = 0.0; // No correlation with physical order
    *index_pages = (tuples / 100.0).max(1.0); // Rough estimate
}

/// Get tuple callback (for index scans)
#[pg_guard]
unsafe extern "C" fn hnsw_gettuple(_scan: IndexScanDesc, _direction: ScanDirection::Type) -> bool {
    pgrx::log!("HNSW: Get tuple called");

    // TODO: Implement actual index scan
    // For now, return false (no more tuples)
    false
}

/// Get bitmap callback (for bitmap scans)
#[pg_guard]
unsafe extern "C" fn hnsw_getbitmap(_scan: IndexScanDesc, _tbm: *mut TIDBitmap) -> i64 {
    pgrx::log!("HNSW: Get bitmap called");

    // TODO: Implement bitmap scan
    // Return number of tuples
    0
}

/// Begin scan callback
#[pg_guard]
unsafe extern "C" fn hnsw_beginscan(
    index: Relation,
    nkeys: ::std::os::raw::c_int,
    norderbys: ::std::os::raw::c_int,
) -> IndexScanDesc {
    pgrx::log!("HNSW: Begin scan");

    let scan = pg_sys::RelationGetIndexScan(index, nkeys, norderbys);
    scan
}

/// Rescan callback
#[pg_guard]
unsafe extern "C" fn hnsw_rescan(
    _scan: IndexScanDesc,
    _keys: ScanKey,
    _nkeys: ::std::os::raw::c_int,
    _orderbys: ScanKey,
    _norderbys: ::std::os::raw::c_int,
) {
    pgrx::log!("HNSW: Rescan");
    // Reset scan state
}

/// End scan callback
#[pg_guard]
unsafe extern "C" fn hnsw_endscan(_scan: IndexScanDesc) {
    pgrx::log!("HNSW: End scan");
    // Clean up scan state
}

/// Can return callback - indicates if index can return indexed data
#[pg_guard]
unsafe extern "C" fn hnsw_canreturn(_index: Relation, attno: ::std::os::raw::c_int) -> bool {
    // HNSW can return the vector column
    attno == 1
}

/// Options callback - parse index options
#[pg_guard]
unsafe extern "C" fn hnsw_options(_reloptions: Datum, _validate: bool) -> *mut bytea {
    pgrx::log!("HNSW: Parsing options");

    // TODO: Parse m, ef_construction, metric from reloptions
    // For now, return null (use defaults)
    ptr::null_mut()
}

// ============================================================================
// Access Method Handler
// ============================================================================

/// Static IndexAmRoutine template for HNSW
/// This is copied into a palloc'd structure when the handler is called
static HNSW_AM_HANDLER: IndexAmRoutine = IndexAmRoutine {
    type_: NodeTag::T_IndexAmRoutine,

    // Index structure capabilities
    amstrategies: 1, // One strategy: nearest neighbor
    amsupport: 1,    // One support function: distance
    amoptsprocnum: 0,
    amcanorder: false,
    amcanorderbyop: true, // Supports ORDER BY with distance operators
    amcanbackward: false,
    amcanunique: false,
    amcanmulticol: false, // Single column only (vector)
    amoptionalkey: true,
    amsearcharray: false,
    amsearchnulls: false,
    amstorage: false,
    amclusterable: false,
    ampredlocks: false,
    amcanparallel: false,
    amcaninclude: false,
    amusemaintenanceworkmem: true,
    amsummarizing: false,
    amparallelvacuumoptions: 0,

    // Key type
    amkeytype: pg_sys::ANYELEMENTOID,

    // Callbacks - set to None, will be filled in at runtime
    ambuild: None,
    ambuildempty: None,
    aminsert: None,
    ambulkdelete: None,
    amvacuumcleanup: None,
    amcanreturn: None,
    amcostestimate: None,
    amoptions: None,
    amproperty: None,
    ambuildphasename: None,
    amvalidate: None,
    amadjustmembers: None,
    ambeginscan: None,
    amrescan: None,
    amgettuple: None,
    amgetbitmap: None,
    amendscan: None,
    ammarkpos: None,
    amrestrpos: None,
    amestimateparallelscan: None,
    aminitparallelscan: None,
    amparallelrescan: None,
    // PG17 additions
    #[cfg(any(feature = "pg17"))]
    amcanbuildparallel: false,
    #[cfg(any(feature = "pg17"))]
    aminsertcleanup: None,
};

/// Main handler function for HNSW index access method
#[pg_extern(sql = "
CREATE OR REPLACE FUNCTION hnsw_handler(internal) RETURNS index_am_handler
AS 'MODULE_PATHNAME', 'hnsw_handler_wrapper' LANGUAGE C STRICT;
")]
fn hnsw_handler(_fcinfo: pg_sys::FunctionCallInfo) -> Internal {
    unsafe {
        // Allocate IndexAmRoutine in PostgreSQL memory context
        let am_routine = pg_sys::palloc0(size_of::<IndexAmRoutine>()) as *mut IndexAmRoutine;

        // Copy template into allocated memory
        ptr::copy_nonoverlapping(&HNSW_AM_HANDLER, am_routine, 1);

        // Set callback function pointers
        (*am_routine).ambuild = Some(hnsw_build);
        (*am_routine).ambuildempty = Some(hnsw_buildempty);
        (*am_routine).aminsert = Some(hnsw_insert);
        (*am_routine).ambulkdelete = Some(hnsw_bulkdelete);
        (*am_routine).amvacuumcleanup = Some(hnsw_vacuumcleanup);
        (*am_routine).ambeginscan = Some(hnsw_beginscan);
        (*am_routine).amrescan = Some(hnsw_rescan);
        (*am_routine).amgettuple = Some(hnsw_gettuple);
        (*am_routine).amgetbitmap = Some(hnsw_getbitmap);
        (*am_routine).amendscan = Some(hnsw_endscan);
        (*am_routine).amcostestimate = Some(hnsw_costestimate);
        (*am_routine).amoptions = Some(hnsw_options);
        (*am_routine).amcanreturn = Some(hnsw_canreturn);

        // Return as Internal datum
        Internal::from(Some(Datum::from(am_routine)))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_page_size() {
        assert!(std::mem::size_of::<HnswMetaPage>() < 8192);
    }

    #[test]
    fn test_node_header_size() {
        assert!(std::mem::size_of::<HnswNodePageHeader>() < 100);
    }
}
