//! Minimal WASM binary for pi.ruv.io brain node publication (ADR-063).
//!
//! Re-exports the canonical min-cut `extern "C"` functions from
//! `ruvector-mincut::wasm::canonical` and provides V1 ABI stub exports
//! (`memory`, `malloc`, `feature_extract_dim`, `feature_extract`) required
//! by the brain server's node publish endpoint.
//!
//! The V1 stubs are no-ops since this is a graph algorithm node, not an
//! embedding/feature-extraction node.

// Pull in the canonical WASM module so its #[no_mangle] functions are linked.
pub use ruvector_mincut::wasm::canonical::*;

// ── V1 ABI stubs ─────────────────────────────────────────────────────
//
// The brain server (pi.ruv.io) requires these four exports for all WASM
// nodes.  For graph algorithm nodes they are unused stubs.
//
// `memory` is automatically exported by the WASM linker (linear memory).
// We only need to provide the three function exports.

/// Allocate `size` bytes in linear memory.  Returns pointer.
///
/// Minimal bump allocator for V1 ABI conformance.
#[no_mangle]
pub extern "C" fn malloc(size: u32) -> u32 {
    // Simple bump allocator using a static pointer.
    // Safe for single-threaded WASM; no free() needed for V1 stubs.
    static BUMP: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

    if BUMP.load(core::sync::atomic::Ordering::Relaxed) == 0 {
        // Start after 64KB to avoid clobbering stack / data
        BUMP.store(65536, core::sync::atomic::Ordering::Relaxed);
    }

    let ptr = BUMP.fetch_add(size, core::sync::atomic::Ordering::Relaxed);
    ptr
}

/// Return the output embedding dimension.  Returns 0 for graph algorithm nodes.
#[no_mangle]
pub extern "C" fn feature_extract_dim() -> u32 {
    0 // Not an embedding node
}

/// Feature extraction stub.  No-op for graph algorithm nodes.
///
/// # Safety
///
/// Pointers are unused; the function returns immediately.
#[no_mangle]
pub unsafe extern "C" fn feature_extract(
    _input_ptr: u32,
    _input_len: u32,
    _output_ptr: u32,
    _output_len: u32,
) -> i32 {
    -1 // Unsupported: this is a graph algorithm node, not an embedding node
}
