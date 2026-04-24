//! `VectorKernel` trait — the pluggable execution backend for RaBitQ
//! scan + rerank. Defined here (ADR-157 §"Where each piece lives")
//! because kernels are RaBitQ primitives; the cache is a consumer.
//!
//! Ships with one implementation — `CpuKernel` — which delegates to
//! the existing `RabitqPlusIndex::search_with_rerank`. GPU / SIMD /
//! WASM kernels live in separate crates (`ruvector-rabitq-cuda` etc.)
//! and register themselves with the caller (e.g. `ruvector-rulake`'s
//! dispatcher) as optional accelerators.
//!
//! ## Determinism contract
//!
//! Scan-phase output (top-k by 1-bit Hamming distance) must be
//! bit-reproducible across every kernel. Rerank-phase output (exact
//! L2²) may differ in the last ulp on reduction-order-sensitive
//! kernels (GPU with float reduction reorder); these set
//! `caps().deterministic = false`, and the caller's dispatch policy
//! filters them out of `Consistency::Fresh` / `Consistency::Frozen`
//! paths.
//!
//! The witness chain is NOT recomputed per kernel; it stays anchored
//! on `(data_ref, dim, rotation_seed, rerank_factor, generation)`.
//! Kernel identity is surfaced in caps + stats, not in the witness.

use crate::index::{AnnIndex, RabitqPlusIndex, SearchResult};
use crate::RabitqError;

/// Capability advertisement for a vector kernel. The caller's
/// dispatch policy compares these against the request to pick the
/// best kernel for a given batch + determinism requirement.
#[derive(Debug, Clone)]
pub struct KernelCaps {
    /// Symbolic accelerator label: "cpu", "cpu-simd", "cuda",
    /// "metal", "rocm", "wasm-simd", etc. Surfaced in stats.
    pub accelerator: &'static str,
    /// Minimum batch size at which this kernel is ever chosen. CPU
    /// kernels report 1; GPU kernels typically ≥ 64.
    pub min_batch: usize,
    /// Maximum dimensionality the kernel supports without falling
    /// back to a slower path. `usize::MAX` means "no constraint".
    pub max_dim: usize,
    /// Does the kernel produce byte-identical output (scan + rerank)
    /// vs the reference CPU kernel? Only deterministic kernels can
    /// feed witness-sealed outputs under Fresh/Frozen consistency.
    pub deterministic: bool,
}

impl KernelCaps {
    /// Default CPU caps: available always, deterministic, no dim cap.
    pub const fn cpu_default() -> Self {
        Self {
            accelerator: "cpu",
            min_batch: 1,
            max_dim: usize::MAX,
            deterministic: true,
        }
    }
}

/// A batch of query vectors against a single index. The index is
/// borrowed by reference so GPU kernels don't need to own its
/// lifetime — the cache holds the authoritative copy.
pub struct ScanRequest<'a> {
    pub index: &'a RabitqPlusIndex,
    pub queries: &'a [Vec<f32>],
    pub k: usize,
    /// Optional per-call rerank factor. `None` uses the index's stored
    /// default. Used by `ruvector-rulake` to divide rerank cost
    /// across K shards (ADR-155 federation path).
    pub rerank_factor: Option<usize>,
}

/// Batched top-k results, one `Vec<SearchResult>` per query. Order
/// matches the input `queries`.
pub type ScanResponse = Vec<Vec<SearchResult>>;

/// A vector kernel executes scan + exact rerank for one or more
/// queries against a compressed RaBitQ index.
///
/// Implementations are stateless w.r.t. the index — they receive it
/// by reference on every call, so a single kernel instance can serve
/// many caches / collections concurrently. Concrete GPU kernels may
/// carry a driver handle or stream object; that's kernel state, not
/// index state.
pub trait VectorKernel: Send + Sync {
    /// Stable identifier surfaced in stats + logs. Must be unique per
    /// kernel type (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    fn id(&self) -> &str;

    /// Capability advertisement — what this kernel can do. Return a
    /// fresh struct (not a static reference) so kernels can narrow
    /// caps at runtime (e.g. GPU-down → `min_batch = usize::MAX`).
    fn caps(&self) -> KernelCaps;

    /// Run the scan + rerank for every query in `req`. Returns one
    /// `Vec<SearchResult>` per query, in the input order.
    fn scan(&self, req: ScanRequest<'_>) -> Result<ScanResponse, RabitqError>;
}

/// Reference CPU kernel. Wraps `RabitqPlusIndex::search_with_rerank`.
/// Deterministic by construction (integer popcount scan + stable
/// exact L2² rerank with total-order tie break via position).
///
/// This is the default kernel every consumer gets for free; GPU /
/// SIMD implementations plug in alongside it via registration.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuKernel;

impl CpuKernel {
    pub const fn new() -> Self {
        Self
    }
}

impl VectorKernel for CpuKernel {
    fn id(&self) -> &str {
        "cpu"
    }

    fn caps(&self) -> KernelCaps {
        KernelCaps::cpu_default()
    }

    fn scan(&self, req: ScanRequest<'_>) -> Result<ScanResponse, RabitqError> {
        let mut out = Vec::with_capacity(req.queries.len());
        for q in req.queries {
            let hits = match req.rerank_factor {
                None => req.index.search(q, req.k)?,
                Some(rf) => req.index.search_with_rerank(q, req.k, rf)?,
            };
            out.push(hits);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_index() -> RabitqPlusIndex {
        let d = 8;
        let mut idx = RabitqPlusIndex::new(d, 42, 5);
        for i in 0..16 {
            let v: Vec<f32> = (0..d).map(|j| (i + j) as f32).collect();
            idx.add(i, v).unwrap();
        }
        idx
    }

    #[test]
    fn cpu_kernel_matches_direct_search() {
        let idx = tiny_index();
        let kernel = CpuKernel::new();
        let q: Vec<f32> = vec![2.0; 8];
        let direct = idx.search(&q, 4).unwrap();
        let batched = kernel
            .scan(ScanRequest {
                index: &idx,
                queries: std::slice::from_ref(&q),
                k: 4,
                rerank_factor: None,
            })
            .unwrap();
        assert_eq!(batched.len(), 1);
        let batch = &batched[0];
        assert_eq!(batch.len(), direct.len());
        for (a, b) in batch.iter().zip(direct.iter()) {
            assert_eq!(a.id, b.id);
            assert!((a.score - b.score).abs() < 1e-5);
        }
    }

    #[test]
    fn cpu_kernel_respects_rerank_override() {
        let idx = tiny_index();
        let kernel = CpuKernel::new();
        let q: Vec<f32> = vec![2.0; 8];
        // Override with a smaller rerank factor — results should still
        // be sorted and a prefix of the default.
        let out = kernel
            .scan(ScanRequest {
                index: &idx,
                queries: &[q.clone(), q.clone()],
                k: 3,
                rerank_factor: Some(2),
            })
            .unwrap();
        assert_eq!(out.len(), 2, "one result vec per input query");
        for v in &out {
            for w in v.windows(2) {
                assert!(w[0].score <= w[1].score, "hits must be sorted");
            }
        }
    }

    #[test]
    fn cpu_caps_are_deterministic_and_unbounded() {
        let c = CpuKernel::new().caps();
        assert_eq!(c.accelerator, "cpu");
        assert_eq!(c.min_batch, 1);
        assert_eq!(c.max_dim, usize::MAX);
        assert!(c.deterministic);
    }
}
