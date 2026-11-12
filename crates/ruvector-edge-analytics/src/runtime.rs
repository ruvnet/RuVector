//! Threading on wasm32 (ADR-351 M4 "rayon wasm test").
//!
//! `ruvector-mincut` depends on rayon unconditionally, but the paths this
//! crate calls (`MinCutBuilder` -> `DynamicMinCut` -> sparse Stoer-Wagner,
//! and `ApproxMinCut`) contain no rayon calls; its `par_iter` sites live in
//! `snn` and in `optimization::parallel` behind a `rayon` cfg that is never
//! set. If rayon is reached anyway, rayon-core >= 1.12 builds its global
//! pool lazily and, when `std::thread::spawn` reports `Unsupported`
//! (wasm32-unknown-unknown, wasm32-wasip1), rebuilds it as
//! `num_threads(1).use_current_thread()` (rayon-core `registry.rs`) — work
//! runs inline on the calling thread instead of panicking.
//! `tests/wasm.rs` asserts this on wasm32 at runtime.

/// Threads rayon will use. Initializes rayon's global pool on first call
/// (on native this spawns the worker threads). `1` on wasm32.
pub fn worker_threads() -> usize {
    rayon::current_num_threads()
}

/// Smoke-test rayon on this target: a parallel sum that must complete on
/// the calling thread when threads are unavailable. Returns the thread
/// count observed.
pub fn rayon_probe() -> (usize, u64) {
    use rayon::prelude::*;
    let sum = (0u64..10_000).into_par_iter().map(|x| x * 2).sum::<u64>();
    (rayon::current_num_threads(), sum)
}
