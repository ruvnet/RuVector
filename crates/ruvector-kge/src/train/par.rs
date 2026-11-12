//! Order-preserving parallel map for the trainer. With the `parallel`
//! feature (native builds) the work runs on rayon's pool; without it
//! (the default, and always on wasm32) it runs sequentially. Either way the
//! output is in input order and each item is computed by the same code, so
//! training results do not depend on the feature or the thread count.

/// `(0..n).map(f).collect()`, in parallel when enabled.
pub(crate) fn map_range<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
    {
        (0..n).map(f).collect()
    }
}

/// Whether [`map_range`] runs in parallel in this build.
#[must_use]
pub fn enabled() -> bool {
    cfg!(all(feature = "parallel", not(target_arch = "wasm32")))
}
