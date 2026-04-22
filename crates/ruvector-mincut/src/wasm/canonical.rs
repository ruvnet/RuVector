//! WASM bindings for source-anchored canonical minimum cut (ADR-117).
//!
//! Provides `#[no_mangle] extern "C"` functions for use from WASM
//! host environments. All data crosses the boundary via flat arrays
//! and the `CanonicalMinCutResult` repr(C) struct.
//!
//! # Thread safety
//!
//! State is guarded by a `Mutex` for safe testing. In actual WASM
//! (single-threaded), the mutex is uncontended and zero-cost.

use crate::canonical::source_anchored::{
    canonical_mincut, CanonicalMinCutResult, SourceAnchoredConfig, SourceAnchoredCut,
};
use crate::graph::DynamicGraph;

use std::sync::Mutex;

/// Global state for WASM canonical cut operations.
struct WasmState {
    graph: Option<DynamicGraph>,
    last_cut: Option<SourceAnchoredCut>,
    last_result: CanonicalMinCutResult,
}

static WASM_STATE: Mutex<WasmState> = Mutex::new(WasmState {
    graph: None,
    last_cut: None,
    last_result: CanonicalMinCutResult {
        lambda_raw: 0,
        source_vertex: 0,
        first_separable_vertex: 0,
        side_size: 0,
        priority_sum: 0,
        cut_edge_count: 0,
        cut_hash: [0u8; 32],
    },
});

/// Initialize the WASM graph with a given number of vertices.
///
/// Must be called before any other WASM canonical cut function.
/// Returns 0 on success, -1 if `num_vertices` exceeds the limit.
#[no_mangle]
pub extern "C" fn canonical_init(num_vertices: u32) -> i32 {
    if num_vertices > 10_000 {
        return -1;
    }

    let g = DynamicGraph::with_capacity(num_vertices as usize, num_vertices as usize * 2);
    for i in 0..num_vertices as u64 {
        g.add_vertex(i);
    }

    let mut state = WASM_STATE.lock().unwrap();
    state.graph = Some(g);
    state.last_cut = None;
    0
}

/// Add an edge to the WASM graph.
///
/// `weight_fixed` is a 32.32 fixed-point weight (e.g. `1u64 << 32` = 1.0).
/// Returns 0 on success, -1 if graph not initialized, -2 if edge invalid.
#[no_mangle]
pub extern "C" fn canonical_add_edge(u: u64, v: u64, weight_fixed: u64) -> i32 {
    let state = WASM_STATE.lock().unwrap();
    let graph = match state.graph.as_ref() {
        Some(g) => g,
        None => return -1,
    };

    if u == v {
        return -2;
    }
    if !graph.has_vertex(u) || !graph.has_vertex(v) {
        return -2;
    }

    let weight = (weight_fixed as f64) / (1u64 << 32) as f64;
    match graph.insert_edge(u, v, weight) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

/// Compute the canonical minimum cut on the current WASM graph.
///
/// Pass `u64::MAX` for `source` to use the default (smallest vertex ID).
/// Returns 0 on success, -1 if graph not initialized, -2 if cut not found.
///
/// After a successful call, use `canonical_get_result` to read the result.
#[no_mangle]
pub extern "C" fn canonical_compute(source: u64) -> i32 {
    let mut state = WASM_STATE.lock().unwrap();
    let graph = match state.graph.as_ref() {
        Some(g) => g,
        None => return -1,
    };

    let config = SourceAnchoredConfig {
        source: if source == u64::MAX {
            None
        } else {
            Some(source)
        },
        vertex_order: None,
        vertex_priorities: None,
    };

    match canonical_mincut(graph, &config) {
        Some(cut) => {
            state.last_result = CanonicalMinCutResult::from(&cut);
            state.last_cut = Some(cut);
            0
        }
        None => -2,
    }
}

/// Get the result of the last canonical computation.
///
/// Returns a pointer to the `CanonicalMinCutResult` struct.
/// The pointer is valid until the next `canonical_compute` or `canonical_init`.
///
/// Returns null if no computation has been performed.
///
/// # Safety
///
/// The returned pointer must not be used after the next mutation call.
#[no_mangle]
pub extern "C" fn canonical_get_result() -> *const CanonicalMinCutResult {
    let state = WASM_STATE.lock().unwrap();
    if state.last_cut.is_some() {
        &state.last_result as *const CanonicalMinCutResult
    } else {
        std::ptr::null()
    }
}

/// Get the cut hash from the last computation.
///
/// Copies 32 bytes into the provided buffer. Returns 0 on success,
/// -1 if no cut has been computed.
///
/// # Safety
///
/// `out_buf` must point to at least 32 bytes of writable memory.
#[no_mangle]
pub unsafe extern "C" fn canonical_get_hash(out_buf: *mut u8) -> i32 {
    if out_buf.is_null() {
        return -1;
    }
    let state = WASM_STATE.lock().unwrap();
    match state.last_cut.as_ref() {
        Some(cut) => {
            std::ptr::copy_nonoverlapping(cut.cut_hash.as_ptr(), out_buf, 32);
            0
        }
        None => -1,
    }
}

/// Get the side vertices from the last computation.
///
/// Writes vertex IDs into the provided buffer. Returns the number of
/// vertices written, or -1 if no cut has been computed.
///
/// # Safety
///
/// `out_buf` must point to a buffer of at least `buf_len` u64 values.
#[no_mangle]
pub unsafe extern "C" fn canonical_get_side(out_buf: *mut u64, buf_len: u32) -> i32 {
    if out_buf.is_null() {
        return -1;
    }
    let state = WASM_STATE.lock().unwrap();
    match state.last_cut.as_ref() {
        Some(cut) => {
            let count = cut.side_vertices.len().min(buf_len as usize);
            for i in 0..count {
                *out_buf.add(i) = cut.side_vertices[i];
            }
            count as i32
        }
        None => -1,
    }
}

/// Get the cut edges from the last computation.
///
/// Writes edge pairs (u, v) interleaved: [u0, v0, u1, v1, ...].
/// Returns the number of edges written, or -1 if no cut computed.
///
/// # Safety
///
/// `out_buf` must point to a buffer of at least `buf_len * 2` u64 values.
#[no_mangle]
pub unsafe extern "C" fn canonical_get_cut_edges(out_buf: *mut u64, buf_len: u32) -> i32 {
    if out_buf.is_null() {
        return -1;
    }
    let state = WASM_STATE.lock().unwrap();
    match state.last_cut.as_ref() {
        Some(cut) => {
            let count = cut.cut_edges.len().min(buf_len as usize);
            for i in 0..count {
                *out_buf.add(i * 2) = cut.cut_edges[i].0;
                *out_buf.add(i * 2 + 1) = cut.cut_edges[i].1;
            }
            count as i32
        }
        None => -1,
    }
}

/// Free the WASM graph and any cached results.
#[no_mangle]
pub extern "C" fn canonical_free() {
    let mut state = WASM_STATE.lock().unwrap();
    state.graph = None;
    state.last_cut = None;
}

// ---------------------------------------------------------------------------
// Dynamic MinCut WASM bindings (Tier 3)
// ---------------------------------------------------------------------------

use crate::canonical::dynamic::{DynamicMinCut, DynamicMinCutConfig, EdgeMutation};

/// Global state for WASM dynamic canonical cut operations.
struct DynamicWasmState {
    engine: Option<DynamicMinCut>,
    last_result: CanonicalMinCutResult,
}

static DYNAMIC_STATE: Mutex<DynamicWasmState> = Mutex::new(DynamicWasmState {
    engine: None,
    last_result: CanonicalMinCutResult {
        lambda_raw: 0,
        source_vertex: 0,
        first_separable_vertex: 0,
        side_size: 0,
        priority_sum: 0,
        cut_edge_count: 0,
        cut_hash: [0u8; 32],
    },
});

/// Initialize a new dynamic canonical min-cut engine.
///
/// `staleness_threshold`: number of incremental updates before forcing
/// a full recomputation. Pass 0 to disable.
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn dynamic_init(staleness_threshold: u64) -> i32 {
    let config = DynamicMinCutConfig {
        canonical_config: SourceAnchoredConfig::default(),
        staleness_threshold,
    };
    let engine = DynamicMinCut::with_config(config);

    let mut state = DYNAMIC_STATE.lock().unwrap();
    state.engine = Some(engine);
    0
}

/// Add an edge to the dynamic engine.
///
/// Returns 0 on success, -1 if not initialized, -2 on error.
#[no_mangle]
pub extern "C" fn dynamic_add_edge(u: u64, v: u64, weight_fixed: u64) -> i32 {
    let mut state = DYNAMIC_STATE.lock().unwrap();
    let engine = match state.engine.as_mut() {
        Some(e) => e,
        None => return -1,
    };

    let weight = (weight_fixed as f64) / (1u64 << 32) as f64;
    match engine.add_edge(u, v, weight) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

/// Remove an edge from the dynamic engine.
///
/// Returns 0 on success, -1 if not initialized, -2 on error.
#[no_mangle]
pub extern "C" fn dynamic_remove_edge(u: u64, v: u64) -> i32 {
    let mut state = DYNAMIC_STATE.lock().unwrap();
    let engine = match state.engine.as_mut() {
        Some(e) => e,
        None => return -1,
    };

    match engine.remove_edge(u, v) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

/// Compute the canonical cut on the dynamic engine.
///
/// Returns 0 on success, -1 if not initialized, -2 if cut not found.
#[no_mangle]
pub extern "C" fn dynamic_compute() -> i32 {
    let mut state = DYNAMIC_STATE.lock().unwrap();
    let engine = match state.engine.as_mut() {
        Some(e) => e,
        None => return -1,
    };

    match engine.canonical_cut() {
        Some(cut) => {
            state.last_result = CanonicalMinCutResult::from(&cut);
            0
        }
        None => -2,
    }
}

/// Get the current epoch of the dynamic engine.
///
/// Returns the epoch, or u64::MAX if not initialized.
#[no_mangle]
pub extern "C" fn dynamic_epoch() -> u64 {
    let state = DYNAMIC_STATE.lock().unwrap();
    match state.engine.as_ref() {
        Some(e) => e.epoch(),
        None => u64::MAX,
    }
}

/// Check if the dynamic engine's cached cut is stale.
///
/// Returns 1 if stale, 0 if not, -1 if not initialized.
#[no_mangle]
pub extern "C" fn dynamic_is_stale() -> i32 {
    let state = DYNAMIC_STATE.lock().unwrap();
    match state.engine.as_ref() {
        Some(e) => {
            if e.is_stale() {
                1
            } else {
                0
            }
        }
        None => -1,
    }
}

/// Force a full recomputation on the dynamic engine.
///
/// Returns 0 on success, -1 if not initialized.
#[no_mangle]
pub extern "C" fn dynamic_force_recompute() -> i32 {
    let mut state = DYNAMIC_STATE.lock().unwrap();
    let engine = match state.engine.as_mut() {
        Some(e) => e,
        None => return -1,
    };

    engine.force_recompute();
    0
}

/// Get the result of the last dynamic computation.
///
/// Returns a pointer to the `CanonicalMinCutResult` struct, or null.
#[no_mangle]
pub extern "C" fn dynamic_get_result() -> *const CanonicalMinCutResult {
    let state = DYNAMIC_STATE.lock().unwrap();
    if state.engine.is_some() && state.last_result.lambda_raw > 0 {
        &state.last_result as *const CanonicalMinCutResult
    } else {
        std::ptr::null()
    }
}

/// Free the dynamic engine.
#[no_mangle]
pub extern "C" fn dynamic_free() {
    let mut state = DYNAMIC_STATE.lock().unwrap();
    state.engine = None;
}

/// Verify that two cut hashes are equal using constant-time comparison.
///
/// Returns 1 if equal, 0 if not equal, -1 if either pointer is null.
///
/// # Safety
///
/// Both pointers must point to at least 32 bytes of readable memory.
#[no_mangle]
pub unsafe extern "C" fn canonical_hashes_equal(a: *const u8, b: *const u8) -> i32 {
    if a.is_null() || b.is_null() {
        return -1;
    }
    let sa = std::slice::from_raw_parts(a, 32);
    let sb = std::slice::from_raw_parts(b, 32);

    // Constant-time comparison to prevent timing side channels
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= sa[i] ^ sb[i];
    }
    if diff == 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each test acquires the global lock via canonical_init/canonical_free,
    // so they are safe to run in parallel (mutex serializes access).

    #[test]
    fn test_wasm_init_and_compute() {
        assert_eq!(canonical_init(3), 0);
        assert_eq!(canonical_add_edge(0, 1, 1u64 << 32), 0);
        assert_eq!(canonical_add_edge(1, 2, 1u64 << 32), 0);
        assert_eq!(canonical_add_edge(2, 0, 1u64 << 32), 0);

        let rc = canonical_compute(u64::MAX);
        assert_eq!(rc, 0);

        let ptr = canonical_get_result();
        assert!(!ptr.is_null());

        let state = WASM_STATE.lock().unwrap();
        assert_eq!(state.last_result.source_vertex, 0);
        assert_eq!(state.last_result.first_separable_vertex, 1);
        drop(state);

        canonical_free();
    }

    #[test]
    fn test_wasm_init_too_large() {
        assert_eq!(canonical_init(100_000), -1);
    }

    #[test]
    fn test_wasm_add_edge_no_init() {
        canonical_free();
        assert_eq!(canonical_add_edge(0, 1, 1u64 << 32), -1);
    }

    #[test]
    fn test_wasm_self_loop_rejected() {
        assert_eq!(canonical_init(3), 0);
        assert_eq!(canonical_add_edge(0, 0, 1u64 << 32), -2);
        canonical_free();
    }

    #[test]
    fn test_wasm_hash_comparison() {
        assert_eq!(canonical_init(3), 0);
        assert_eq!(canonical_add_edge(0, 1, 1u64 << 32), 0);
        assert_eq!(canonical_add_edge(1, 2, 1u64 << 32), 0);
        assert_eq!(canonical_add_edge(2, 0, 1u64 << 32), 0);

        assert_eq!(canonical_compute(u64::MAX), 0);

        let mut hash = [0u8; 32];
        let rc = unsafe { canonical_get_hash(hash.as_mut_ptr()) };
        assert_eq!(rc, 0);

        // Compare with self
        let equal = unsafe { canonical_hashes_equal(hash.as_ptr(), hash.as_ptr()) };
        assert_eq!(equal, 1);

        // Compare with zeros
        let zeros = [0u8; 32];
        let not_equal = unsafe { canonical_hashes_equal(hash.as_ptr(), zeros.as_ptr()) };
        assert_eq!(not_equal, 0);

        canonical_free();
    }

    #[test]
    fn test_wasm_get_side_vertices() {
        assert_eq!(canonical_init(3), 0);
        assert_eq!(canonical_add_edge(0, 1, 1u64 << 32), 0);
        assert_eq!(canonical_add_edge(1, 2, 1u64 << 32), 0);
        assert_eq!(canonical_add_edge(2, 0, 1u64 << 32), 0);

        assert_eq!(canonical_compute(u64::MAX), 0);

        let mut buf = [0u64; 16];
        let count = unsafe { canonical_get_side(buf.as_mut_ptr(), 16) };
        assert!(count > 0);

        canonical_free();
    }

    #[test]
    fn test_wasm_null_safety() {
        let rc = unsafe { canonical_get_hash(std::ptr::null_mut()) };
        assert_eq!(rc, -1);

        let rc = unsafe { canonical_get_side(std::ptr::null_mut(), 10) };
        assert_eq!(rc, -1);

        let rc = unsafe { canonical_get_cut_edges(std::ptr::null_mut(), 10) };
        assert_eq!(rc, -1);

        let rc = unsafe { canonical_hashes_equal(std::ptr::null(), std::ptr::null()) };
        assert_eq!(rc, -1);
    }

    // -------------------------------------------------------------------
    // Dynamic WASM tests
    // -------------------------------------------------------------------

    #[test]
    fn test_dynamic_wasm_init_and_compute() {
        assert_eq!(dynamic_init(100), 0);

        // Add edges to build a triangle
        assert_eq!(dynamic_add_edge(0, 1, 1u64 << 32), 0);
        assert_eq!(dynamic_add_edge(1, 2, 1u64 << 32), 0);
        assert_eq!(dynamic_add_edge(2, 0, 1u64 << 32), 0);

        let rc = dynamic_compute();
        assert_eq!(rc, 0);

        assert_eq!(dynamic_epoch(), 3); // 3 edge additions

        dynamic_free();
    }

    #[test]
    fn test_dynamic_wasm_add_remove() {
        assert_eq!(dynamic_init(50), 0);

        assert_eq!(dynamic_add_edge(0, 1, 1u64 << 32), 0);
        assert_eq!(dynamic_add_edge(1, 2, 1u64 << 32), 0);
        assert_eq!(dynamic_add_edge(2, 0, 1u64 << 32), 0);

        assert_eq!(dynamic_compute(), 0);

        // Add another edge
        assert_eq!(dynamic_add_edge(0, 3, 1u64 << 32), 0);
        assert_eq!(dynamic_add_edge(3, 1, 1u64 << 32), 0);

        // Remove an edge
        assert_eq!(dynamic_remove_edge(0, 3), 0);

        assert_eq!(dynamic_epoch(), 6);

        dynamic_free();
    }

    #[test]
    fn test_dynamic_wasm_stale_check() {
        assert_eq!(dynamic_init(100), 0);

        assert_eq!(dynamic_add_edge(0, 1, 1u64 << 32), 0);
        assert_eq!(dynamic_add_edge(1, 2, 1u64 << 32), 0);

        // Before compute, should be stale
        assert_eq!(dynamic_is_stale(), 1);

        dynamic_free();
    }

    #[test]
    fn test_dynamic_wasm_not_initialized() {
        dynamic_free();
        assert_eq!(dynamic_add_edge(0, 1, 1u64 << 32), -1);
        assert_eq!(dynamic_remove_edge(0, 1), -1);
        assert_eq!(dynamic_compute(), -1);
        assert_eq!(dynamic_is_stale(), -1);
        assert_eq!(dynamic_force_recompute(), -1);
        assert_eq!(dynamic_epoch(), u64::MAX);
        assert!(dynamic_get_result().is_null());
    }

    #[test]
    fn test_dynamic_wasm_force_recompute() {
        assert_eq!(dynamic_init(100), 0);

        assert_eq!(dynamic_add_edge(0, 1, 1u64 << 32), 0);
        assert_eq!(dynamic_add_edge(1, 2, 1u64 << 32), 0);
        assert_eq!(dynamic_add_edge(2, 0, 1u64 << 32), 0);

        assert_eq!(dynamic_force_recompute(), 0);

        dynamic_free();
    }
}
