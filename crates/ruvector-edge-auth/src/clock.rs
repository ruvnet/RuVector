//! Time port. The auth crates never call `std::time` (it panics on
//! `wasm32-unknown-unknown`); the Worker implements this with `worker::Date`.

/// Wall-clock source in whole seconds since the Unix epoch.
#[cfg_attr(test, mockall::automock)]
pub trait Clock {
    /// Current time, seconds since 1970-01-01T00:00:00Z.
    fn now_unix(&self) -> u64;
}

impl<T: Clock + ?Sized> Clock for &T {
    fn now_unix(&self) -> u64 {
        (**self).now_unix()
    }
}
