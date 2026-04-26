//! WASM bindings for ruvector-rabitq.
//!
//! Exposes [`RabitqIndex`] as a JavaScript-friendly class for use in
//! browsers and edge runtimes (Cloudflare Workers, Deno, Bun).
//! Single-threaded — the underlying `from_vectors_parallel` falls back
//! to sequential iteration on wasm32 (output is bit-identical because
//! rotation is deterministic).
//!
//! ```ignore
//! import init, { RabitqIndex } from "ruvector-rabitq";
//! await init();
//!
//! const dim = 768;
//! const n = 10_000;
//! const vectors = new Float32Array(n * dim);  // populate
//! const idx = RabitqIndex.build(vectors, dim, 42, 20);
//! const query = new Float32Array(dim);  // populate
//! const results = idx.search(query, 10);  // [{id, distance}, ...]
//! ```

#![allow(clippy::new_without_default)]

use ruvector_rabitq::{AnnIndex, RabitqPlusIndex};
use wasm_bindgen::prelude::*;

/// Initialize panic hook for clearer error messages in the browser
/// console. Called once at module import.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Search result — single nearest-neighbor hit.
///
/// Mirrors the structure used by the Python SDK's `RabitqIndex.search`
/// so callers porting code between languages get identical shapes.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    /// Caller-supplied vector id (the position passed to `build`).
    #[wasm_bindgen(readonly)]
    pub id: u32,
    /// Approximate L2² distance after RaBitQ rerank.
    #[wasm_bindgen(readonly)]
    pub distance: f32,
}

/// 1-bit quantized vector index. Builds in O(n × dim) memory + O(n × dim)
/// time; searches in O(n) hamming distance + O(rerank_factor × k × dim)
/// exact-L2² rerank.
#[wasm_bindgen]
pub struct RabitqIndex {
    inner: RabitqPlusIndex,
}

#[wasm_bindgen]
impl RabitqIndex {
    /// Build an index from a flat Float32Array of length `n * dim`.
    ///
    /// `seed` controls the random rotation matrix; the same `(seed,
    /// dim, vectors)` triple produces bit-identical codes (ADR-154
    /// determinism guarantee). `rerank_factor` is the multiplier on
    /// `k` for the exact-L2² rerank pool — typical 20.
    ///
    /// Errors:
    /// - `vectors.length` is not a multiple of `dim`
    /// - `dim == 0` or `vectors.length == 0`
    #[wasm_bindgen]
    pub fn build(
        vectors: &[f32],
        dim: u32,
        seed: u64,
        rerank_factor: u32,
    ) -> Result<RabitqIndex, JsValue> {
        let dim = dim as usize;
        if dim == 0 {
            return Err(JsValue::from_str("dim must be > 0"));
        }
        if vectors.is_empty() {
            return Err(JsValue::from_str("vectors must not be empty"));
        }
        if !vectors.len().is_multiple_of(dim) {
            return Err(JsValue::from_str(&format!(
                "vectors length {} is not a multiple of dim {}",
                vectors.len(),
                dim
            )));
        }

        let n = vectors.len() / dim;
        let items: Vec<(usize, Vec<f32>)> = (0..n)
            .map(|i| (i, vectors[i * dim..(i + 1) * dim].to_vec()))
            .collect();

        let inner =
            RabitqPlusIndex::from_vectors_parallel(dim, seed, rerank_factor as usize, items)
                .map_err(|e| JsValue::from_str(&format!("RabitqIndex.build: {e}")))?;

        Ok(Self { inner })
    }

    /// Find the `k` nearest neighbors of `query`. Returns hits in
    /// ascending distance.
    ///
    /// Errors:
    /// - `query.length != dim` of the index
    /// - `k == 0`
    #[wasm_bindgen]
    pub fn search(&self, query: &[f32], k: u32) -> Result<Vec<SearchResult>, JsValue> {
        if k == 0 {
            return Err(JsValue::from_str("k must be > 0"));
        }
        let hits = self
            .inner
            .search(query, k as usize)
            .map_err(|e| JsValue::from_str(&format!("RabitqIndex.search: {e}")))?;

        Ok(hits
            .into_iter()
            .map(|h| SearchResult {
                id: h.id as u32,
                distance: h.score,
            })
            .collect())
    }

    /// Number of vectors indexed.
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> u32 {
        self.inner.len() as u32
    }

    /// True iff the index has zero vectors. Mirrors Rust's `is_empty`
    /// convention; exposed because `wasm-bindgen` getter for `len`
    /// returns u32, so callers can't `idx.len === 0` reliably.
    #[wasm_bindgen(getter, js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }
}

/// Crate version string baked at build time.
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Tests for the WASM bindings live as `wasm_bindgen_test` and only run
// in a wasm32 environment via `wasm-pack test`. Native tests can't
// exercise the bindings because `wasm-bindgen 0.2.117` panics on
// `JsValue::from_str` outside a wasm runtime.
//
// The inner numerical correctness is covered by `ruvector-rabitq`'s
// own test suite; here we only verify the JS-facing surface.
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn build_and_search() {
        let dim = 32usize;
        let n = 100usize;
        let mut vectors = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                vectors[i * dim + j] = (i * 31 + j) as f32 / 100.0;
            }
        }
        let idx = RabitqIndex::build(&vectors, dim as u32, 42, 20).expect("build");
        assert_eq!(idx.len(), n as u32);
        assert!(!idx.is_empty());

        let query: Vec<f32> = vectors[..dim].to_vec();
        let hits = idx.search(&query, 5).expect("search");
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].id, 0);
        assert!(hits[0].distance < 1e-3);
    }

    #[wasm_bindgen_test]
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }
}
