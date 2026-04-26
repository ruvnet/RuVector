//! `RabitqIndex` Python class — wraps `ruvector_rabitq::RabitqPlusIndex`.
//!
//! M1 of the SDK plan picks the **plus** variant as the user-facing default
//! because it is the only `ruvector-rabitq` index that:
//!   1. has a published persistence format (`.rbpx`, see
//!      `crates/ruvector-rabitq/src/persist.rs`);
//!   2. supports per-call `rerank_factor` overrides via `search_with_rerank`;
//!   3. retains the original f32 vectors so we can serialise without forcing
//!      the user to keep a separate `items` Vec on the Python side.
//!
//! Future milestones (M2+) can add `FlatF32Index`, `RabitqIndex` (no rerank),
//! and `RabitqAsymIndex` as separate Python classes per the surface sketch in
//! `docs/sdk/03-api-surface.md`. M1 ships exactly one class.

use std::fs::File;
use std::io::{BufReader, BufWriter};

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use ruvector_rabitq::index::AnnIndex;
use ruvector_rabitq::{persist, RabitqPlusIndex};

use crate::error::to_pyerr;

/// Python-visible RaBitQ index. Backed by `RabitqPlusIndex` (symmetric 1-bit
/// scan + exact f32 rerank) — the variant the SDK plan picks for M1 because
/// it owns its originals and has a stable on-disk format.
///
/// `unsendable` because the underlying index is `Send + Sync` but pyo3 cannot
/// statically prove our wrapper is — and we never need cross-thread Python
/// access since all heavy work is done inside `py.allow_threads`. Marking it
/// unsendable is the conservative, free choice.
#[pyclass(name = "RabitqIndex", module = "ruvector._native", unsendable)]
pub struct RabitqIndex {
    inner: RabitqPlusIndex,
    // RabitqPlusIndex needs the original (id, vector) items handed back to
    // `persist::save_index` because the on-disk format is seed-based:
    // `(dim, seed, rerank_factor, items)` deterministically rebuilds. We
    // capture them at build time via `export_items()` on demand — no need
    // to keep a redundant copy here. The seed lives separately because
    // `RabitqPlusIndex` doesn't expose its own seed field.
    seed: u64,
}

#[pymethods]
impl RabitqIndex {
    /// Build an index from a 2D NumPy array of shape `(n, dim)`.
    ///
    /// `vectors` must be `dtype=float32` and contiguous in C order. Non-contig
    /// or wrong-dtype arrays raise `TypeError` at the boundary instead of
    /// silently copying — silent copies would be an O(n·dim) surprise on a
    /// "fast" build call.
    ///
    /// `rerank_factor` defaults to 20 per `docs/sdk/03-api-surface.md`'s
    /// "100% recall@10 at D=128" recommendation citing ADR-154.
    ///
    /// `seed` defaults to 42 to keep the build deterministic out-of-the-box —
    /// the doc-comment guarantee on `ruvector_rabitq` is that
    /// `(dim, seed, data)` round-trips bit-identically.
    #[staticmethod]
    #[pyo3(signature = (vectors, *, rerank_factor = 20, seed = 42))]
    fn build(
        py: Python<'_>,
        vectors: PyReadonlyArray2<'_, f32>,
        rerank_factor: u32,
        seed: u64,
    ) -> PyResult<Self> {
        // Validate the dtype/contig invariants up front. `PyReadonlyArray2`
        // already guarantees the dtype is f32 (otherwise the conversion at
        // the call site fails with `TypeError`). What it does NOT guarantee
        // is C-contiguity; we enforce it because the inner Rust API takes
        // owned `Vec<f32>` per row and we want a single slice per row, not
        // a strided view that would force a copy through ndarray.
        if !vectors.is_c_contiguous() {
            return Err(PyTypeError::new_err(
                "vectors must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        let shape = vectors.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "vectors must be 2D, got {}D",
                shape.len()
            )));
        }
        let n = shape[0];
        let dim = shape[1];
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }
        if n == 0 {
            return Err(PyValueError::new_err("vectors must contain at least 1 row"));
        }

        // Materialise (id, Vec<f32>) pairs. We allocate `n` rows of
        // `dim` f32s. The inner `from_vectors_parallel` is the only ctor
        // that uses rayon for the rotate+pack phase, so this is the
        // path with the right scaling characteristics for a 100k+ build.
        //
        // Copy is unavoidable — the inner API takes owned Vecs to amortise
        // across rayon workers. PyO3 cannot give us mutable owned access
        // to NumPy storage without breaking the readonly contract.
        let slice = vectors.as_slice()?; // contiguous view, len = n*dim
        let mut items: Vec<(usize, Vec<f32>)> = Vec::with_capacity(n);
        for i in 0..n {
            let row = &slice[i * dim..(i + 1) * dim];
            items.push((i, row.to_vec()));
        }

        // Heavy work: drop the GIL. `from_vectors_parallel` runs rotate+pack
        // on a rayon thread pool; releasing the GIL lets a cooperating
        // Python thread (e.g. the asyncio loop in M2) keep moving while
        // the build completes.
        let inner = py
            .allow_threads(|| {
                RabitqPlusIndex::from_vectors_parallel(dim, seed, rerank_factor as usize, items)
            })
            .map_err(to_pyerr)?;

        Ok(Self { inner, seed })
    }

    /// Search for the `k` nearest neighbours of a single `query` vector.
    ///
    /// Returns a list of `(id, score)` tuples in ascending score order
    /// (squared L2 — lower is closer). The id is the row index used at
    /// `build` time; M1 doesn't expose user-supplied ids yet (M2 adds
    /// them via `RuLake.upsert(ids=...)`).
    ///
    /// `rerank_factor` defaults to `None` which means "use the value the
    /// index was built with" (see `RabitqPlusIndex::rerank_factor`); pass
    /// an int to override per-call without rebuilding.
    #[pyo3(signature = (query, k, *, rerank_factor = None))]
    fn search(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<'_, f32>,
        k: usize,
        rerank_factor: Option<u32>,
    ) -> PyResult<Vec<(u32, f32)>> {
        if !query.is_c_contiguous() {
            return Err(PyTypeError::new_err(
                "query must be C-contiguous; pass np.ascontiguousarray(...) first",
            ));
        }
        if k == 0 {
            return Err(PyValueError::new_err("k must be > 0"));
        }
        let q = query.as_slice()?;

        // `search_with_rerank` is the right entry point even when no
        // override is requested, because it is the common path the
        // benchmark uses; routing both into the same code keeps behaviour
        // identical between defaulted and explicit calls.
        let rf = rerank_factor
            .map(|x| x as usize)
            .unwrap_or_else(|| self.inner.rerank_factor());

        // GIL-release window — pure Rust, no PyObject touched inside.
        let results = py
            .allow_threads(|| self.inner.search_with_rerank(q, k, rf))
            .map_err(to_pyerr)?;

        // Cast usize id → u32 to match the underlying SoA storage and
        // the persisted on-disk id width (see `persist.rs`'s u32 id
        // narrowing). `as u32` is safe because the inner storage already
        // checked the bound at build time.
        Ok(results
            .into_iter()
            .map(|r| (r.id as u32, r.score))
            .collect())
    }

    /// Save the index to `path` using the `.rbpx` v1 format.
    fn save(&self, path: &str) -> PyResult<()> {
        let items = self.inner.export_items();
        let f =
            File::create(path).map_err(|e| PyIOError::new_err(format!("create {path}: {e}")))?;
        let mut w = BufWriter::new(f);
        persist::save_index(&self.inner, self.seed, &items, &mut w).map_err(to_pyerr)?;
        // BufWriter::flush is implicit on drop, but make IO errors explicit
        // here rather than swallowed in Drop.
        use std::io::Write as _;
        w.flush()
            .map_err(|e| PyIOError::new_err(format!("flush {path}: {e}")))?;
        Ok(())
    }

    /// Load an index previously written by `save`. Returns a fresh
    /// `RabitqIndex`. The seed embedded in the file is recovered, so
    /// subsequent `save()` calls round-trip without the caller juggling it.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let f = File::open(path).map_err(|e| PyIOError::new_err(format!("open {path}: {e}")))?;
        let mut r = BufReader::new(f);
        // We re-read the header to recover the seed since `RabitqPlusIndex`
        // doesn't expose it. The cheapest correct path is to read+rewind:
        // 8B magic + 4B version + 4B dim + 8B seed = 24 bytes. But since
        // `persist::load_index` consumes the whole reader, we instead
        // re-open and peek separately.
        use std::io::Read as _;
        let mut header = [0u8; 24];
        let mut peek =
            File::open(path).map_err(|e| PyIOError::new_err(format!("open {path}: {e}")))?;
        peek.read_exact(&mut header)
            .map_err(|e| PyIOError::new_err(format!("read header from {path}: {e}")))?;
        if &header[0..8] != persist::MAGIC {
            return Err(PyIOError::new_err(format!(
                "{path}: bad magic — not an rbpx file"
            )));
        }
        let seed = u64::from_le_bytes(header[16..24].try_into().unwrap());

        let inner = persist::load_index(&mut r).map_err(to_pyerr)?;
        Ok(Self { inner, seed })
    }

    /// Number of indexed vectors (matches `AnnIndex::len`).
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Vector dimensionality.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Honest memory footprint in bytes — see
    /// `ruvector_rabitq::AnnIndex::memory_bytes` for what's counted.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// The rerank factor the index was built with (the default used by
    /// `search` when no override is passed).
    #[getter]
    fn rerank_factor(&self) -> u32 {
        self.inner.rerank_factor() as u32
    }

    /// Diagnostic-friendly repr: variant, n, dim, memory_bytes, rerank_factor.
    fn __repr__(&self) -> String {
        format!(
            "RabitqIndex(n={}, dim={}, memory_bytes={}, rerank_factor={})",
            self.inner.len(),
            self.inner.dim(),
            self.inner.memory_bytes(),
            self.inner.rerank_factor(),
        )
    }
}

/// Convenience exporter — the module init in `lib.rs` calls this to add
/// the class. Keeping it here means `lib.rs` doesn't need to know the
/// class type by name, which mirrors the NAPI module conventions used in
/// `crates/ruvector-diskann-node`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RabitqIndex>()?;
    Ok(())
}
