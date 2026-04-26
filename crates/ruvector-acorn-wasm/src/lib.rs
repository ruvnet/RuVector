//! WASM bindings for ruvector-acorn.
//!
//! Exposes [`AcornIndex`] — predicate-agnostic filtered HNSW (ACORN,
//! Patel et al., SIGMOD 2024) — as a JavaScript-friendly class for use
//! in browsers, Cloudflare Workers, Deno, and Bun.
//!
//! ```ignore
//! import init, { AcornIndex } from "@ruvector/acorn-wasm";
//! await init();
//!
//! const dim = 128;
//! const n = 5_000;
//! const vectors = new Float32Array(n * dim);  // populate
//! // gamma=2 → ACORN-γ (best recall at low selectivity); gamma=1 → ACORN-1
//! const idx = AcornIndex.build(vectors, dim, 2);
//!
//! const query = new Float32Array(dim);  // populate
//! const evenIds = (id) => id % 2 === 0;
//! const results = idx.search(query, 10, evenIds);
//! //  → [{id, distance}, ...]
//! ```

#![allow(clippy::new_without_default)]

use ruvector_acorn::{AcornIndex1, AcornIndexGamma, FilteredIndex};
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
/// Mirrors the structure used by `@ruvector/rabitq-wasm` so callers
/// porting code between backends get identical shapes.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    /// Caller-supplied vector id (the position passed to `build`).
    #[wasm_bindgen(readonly)]
    pub id: u32,
    /// Approximate L2² distance.
    #[wasm_bindgen(readonly)]
    pub distance: f32,
}

/// Inner enum so we can ship one JS class with two backing index
/// variants. Hidden from the JS API surface.
enum Inner {
    G1(AcornIndex1),
    Gamma(AcornIndexGamma),
}

/// ACORN filtered HNSW index. Build once, run many filtered searches.
///
/// # Variants
/// - `gamma = 1` — standard HNSW edge budget (M=16). Smaller index,
///   good speed, recall drops at very low selectivity.
/// - `gamma = 2` — γ-augmented graph (M·γ = 32 edges per node).
///   ~2× memory, but holds 96% recall@10 at 1% predicate selectivity
///   where post-filter HNSW collapses to near-zero.
///
/// Default if you don't know which to pick: `gamma = 2`.
#[wasm_bindgen]
pub struct AcornIndex {
    inner: Inner,
    dim: usize,
}

#[wasm_bindgen]
impl AcornIndex {
    /// Build an index from a flat `Float32Array` of length `n * dim`.
    ///
    /// # Errors
    /// - `vectors.length` is not a multiple of `dim`
    /// - `dim == 0` or `vectors.length == 0`
    /// - `gamma == 0`
    #[wasm_bindgen]
    pub fn build(vectors: &[f32], dim: u32, gamma: u32) -> Result<AcornIndex, JsValue> {
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
        if gamma == 0 {
            return Err(JsValue::from_str("gamma must be >= 1"));
        }

        let n = vectors.len() / dim;
        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| vectors[i * dim..(i + 1) * dim].to_vec())
            .collect();

        let inner = if gamma == 1 {
            Inner::G1(AcornIndex1::build(data).map_err(acorn_err)?)
        } else {
            Inner::Gamma(AcornIndexGamma::new_with_gamma(data, gamma as usize).map_err(acorn_err)?)
        };

        Ok(Self { inner, dim })
    }

    /// Find the `k` nearest neighbors of `query` whose id passes
    /// `predicate`. Returns hits in ascending distance.
    ///
    /// `predicate` is called with each candidate `id: number` and must
    /// return a truthy value to admit the candidate. Calls cross the
    /// JS↔WASM boundary once per node visited (≤ ef per query, ~150
    /// default), not once per vector — overhead is bounded.
    ///
    /// # Errors
    /// - `query.length != dim` of the index
    /// - `k == 0`
    /// - `predicate` is not callable
    #[wasm_bindgen]
    pub fn search(
        &self,
        query: &[f32],
        k: u32,
        predicate: &js_sys::Function,
    ) -> Result<Vec<SearchResult>, JsValue> {
        if k == 0 {
            return Err(JsValue::from_str("k must be > 0"));
        }
        if query.len() != self.dim {
            return Err(JsValue::from_str(&format!(
                "query length {} != index dim {}",
                query.len(),
                self.dim
            )));
        }

        // Cell-error to surface the first JS-side throw without
        // unwinding through WASM.
        let pred_err: std::cell::Cell<Option<JsValue>> = std::cell::Cell::new(None);
        let pred_fn = |id: u32| -> bool {
            if pred_err.take().is_some() {
                // Already errored on a previous call — treat as fail
                // and the outer Err will be returned post-search.
                return false;
            }
            let arg = JsValue::from(id);
            match predicate.call1(&JsValue::NULL, &arg) {
                Ok(v) => v.is_truthy(),
                Err(e) => {
                    pred_err.set(Some(e));
                    false
                }
            }
        };

        let hits = match &self.inner {
            Inner::G1(idx) => idx.search(query, k as usize, &pred_fn),
            Inner::Gamma(idx) => idx.search(query, k as usize, &pred_fn),
        }
        .map_err(acorn_err)?;

        if let Some(e) = pred_err.take() {
            return Err(e);
        }

        Ok(hits
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    /// Vector dimensionality of the index.
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> u32 {
        self.dim as u32
    }

    /// Approximate heap size in bytes (graph edges + raw vectors).
    #[wasm_bindgen(getter, js_name = memoryBytes)]
    pub fn memory_bytes(&self) -> u32 {
        let bytes = match &self.inner {
            Inner::G1(idx) => idx.memory_bytes(),
            Inner::Gamma(idx) => idx.memory_bytes(),
        };
        bytes as u32
    }

    /// Variant label for diagnostics: `"ACORN-1 (γ=1, M=16)"` or
    /// `"ACORN-γ (γ=2, M=32)"`.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        match &self.inner {
            Inner::G1(idx) => idx.name().to_string(),
            Inner::Gamma(idx) => idx.name().to_string(),
        }
    }
}

fn acorn_err(e: ruvector_acorn::AcornError) -> JsValue {
    JsValue::from_str(&format!("AcornIndex: {e}"))
}

/// Crate version string baked at build time.
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Tests for the WASM bindings live as `wasm_bindgen_test` and only run
// in a wasm32 environment via `wasm-pack test`. Native tests can't
// exercise the bindings because `wasm-bindgen 0.2.117` panics on
// `JsValue::from_str` outside a wasm runtime — same gate as
// `ruvector-rabitq-wasm`.
//
// The inner numerical correctness is covered by `ruvector-acorn`'s own
// test suite; here we only verify the JS-facing surface.
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn build_and_search() {
        let dim = 16usize;
        let n = 200usize;
        let mut vectors = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                vectors[i * dim + j] = (i * 31 + j) as f32 / 100.0;
            }
        }
        let idx = AcornIndex::build(&vectors, dim as u32, 2).expect("build");
        assert_eq!(idx.dim(), dim as u32);

        // Predicate accepting all ids.
        let always_true = js_sys::Function::new_no_args("return true");
        let query: Vec<f32> = vectors[..dim].to_vec();
        let hits = idx.search(&query, 5, &always_true).expect("search");
        assert_eq!(hits.len(), 5);
        // Closest hit should be the seed point itself.
        assert_eq!(hits[0].id, 0);
        assert!(hits[0].distance < 1e-3);
    }

    #[wasm_bindgen_test]
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }
}
