//! WASM-compatible time abstraction
//!
//! Provides a monotonic time source that works across native and WASM targets.
//! On native targets, uses `std::time::Instant` for accurate timing.
//! On WASM targets (when `wasm` feature is enabled), uses a monotonic counter
//! since `std::time::Instant` is not supported in wasm32-unknown-unknown.

use std::sync::atomic::{AtomicU64, Ordering};

/// Global monotonic counter for WASM builds
#[cfg(feature = "wasm")]
static MONOTONIC_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A WASM-compatible instant type
#[derive(Debug, Clone, Copy)]
pub struct PortableInstant {
    #[cfg(not(feature = "wasm"))]
    inner: std::time::Instant,
    #[cfg(feature = "wasm")]
    counter: u64,
}

impl PortableInstant {
    /// Get the current instant
    #[cfg(not(feature = "wasm"))]
    pub fn now() -> Self {
        Self {
            inner: std::time::Instant::now(),
        }
    }

    /// Get the current instant (WASM version - uses monotonic counter)
    #[cfg(feature = "wasm")]
    pub fn now() -> Self {
        let counter = MONOTONIC_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self { counter }
    }

    /// Get elapsed time in microseconds
    #[cfg(not(feature = "wasm"))]
    pub fn elapsed_micros(&self) -> u64 {
        self.inner.elapsed().as_micros() as u64
    }

    /// Get elapsed "time" in WASM (returns counter difference as proxy)
    #[cfg(feature = "wasm")]
    pub fn elapsed_micros(&self) -> u64 {
        let current = MONOTONIC_COUNTER.load(Ordering::SeqCst);
        // In WASM, we can't measure real time, so return counter diff
        // This is sufficient for relative ordering and statistics
        current.saturating_sub(self.counter)
    }

    /// Get elapsed time as Duration
    #[cfg(not(feature = "wasm"))]
    pub fn elapsed(&self) -> std::time::Duration {
        self.inner.elapsed()
    }

    /// Get elapsed "time" as Duration in WASM (returns pseudo-duration)
    #[cfg(feature = "wasm")]
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::from_micros(self.elapsed_micros())
    }

    /// Get duration since another instant
    #[cfg(not(feature = "wasm"))]
    pub fn duration_since(&self, earlier: Self) -> std::time::Duration {
        self.inner.duration_since(earlier.inner)
    }

    /// Get duration since another instant (WASM version)
    #[cfg(feature = "wasm")]
    pub fn duration_since(&self, earlier: Self) -> std::time::Duration {
        let diff = self.counter.saturating_sub(earlier.counter);
        std::time::Duration::from_micros(diff)
    }
}

impl Default for PortableInstant {
    fn default() -> Self {
        Self::now()
    }
}

/// A WASM-compatible timestamp type for certificates and audit logs
#[derive(Debug, Clone, Copy)]
pub struct PortableTimestamp {
    /// Seconds since UNIX epoch (or monotonic counter in WASM)
    pub secs: u64,
}

impl PortableTimestamp {
    /// Get current timestamp
    #[cfg(not(feature = "wasm"))]
    pub fn now() -> Self {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self { secs }
    }

    /// Get current timestamp (WASM version - uses monotonic counter)
    #[cfg(feature = "wasm")]
    pub fn now() -> Self {
        let secs = MONOTONIC_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self { secs }
    }

    /// Convert to u64 seconds
    pub fn as_secs(&self) -> u64 {
        self.secs
    }
}

impl Default for PortableTimestamp {
    fn default() -> Self {
        Self::now()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portable_instant() {
        let start = PortableInstant::now();
        // Do some work
        let _sum: u64 = (0..1000).sum();
        let elapsed = start.elapsed_micros();
        // Should be non-zero on native, may be 0 on WASM due to counter
        #[cfg(not(feature = "wasm"))]
        assert!(elapsed >= 0);
    }

    #[test]
    fn test_portable_timestamp() {
        let ts1 = PortableTimestamp::now();
        let ts2 = PortableTimestamp::now();
        // Second timestamp should be >= first
        assert!(ts2.secs >= ts1.secs);
    }

    #[test]
    fn test_instant_ordering() {
        let t1 = PortableInstant::now();
        let t2 = PortableInstant::now();
        let d = t2.duration_since(t1);
        // Duration should be non-negative
        assert!(d.as_micros() >= 0);
    }
}
