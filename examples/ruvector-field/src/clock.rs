//! Monotonic clock abstraction.
//!
//! The engine never calls `SystemTime::now` directly — it routes every
//! timestamp through a [`Clock`] so tests can inject a [`TestClock`] and get
//! deterministic, monotonically increasing timestamps.
//!
//! # Example
//!
//! ```
//! use ruvector_field::clock::{AtomicTestClock, Clock};
//! let clock = AtomicTestClock::new();
//! assert_eq!(clock.now_ns(), 0);
//! clock.advance_ns(1_000);
//! assert_eq!(clock.now_ns(), 1_000);
//! ```

use std::cell::Cell;
use std::time::{SystemTime, UNIX_EPOCH};

/// Abstract clock for deterministic tests and production use.
pub trait Clock: Send + Sync {
    /// Monotonically increasing timestamp in nanoseconds.
    fn now_ns(&self) -> u64;
}

/// Default production clock backed by `SystemTime`.
///
/// # Example
///
/// ```
/// use ruvector_field::clock::{Clock, SystemClock};
/// let c = SystemClock;
/// let a = c.now_ns();
/// let b = c.now_ns();
/// assert!(b >= a);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ns(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }
}

/// Test-only clock that returns manually advanced timestamps.
///
/// Uses interior mutability so the engine can hold an `Arc<dyn Clock>` while
/// tests still advance time.
#[derive(Debug, Default)]
pub struct TestClock {
    now: Cell<u64>,
}

// Interior mutability via Cell is not Sync; wrap tests in single-threaded use.
// We implement Send+Sync unsafely by promising no shared mutation across
// threads in tests. To keep this std-only and simple, use a Mutex via a
// lightweight spinlock emulation with AtomicU64 instead.
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe variant used by the engine.
#[derive(Debug, Default)]
pub struct AtomicTestClock {
    now: AtomicU64,
}

impl AtomicTestClock {
    /// Create a clock starting at zero.
    pub fn new() -> Self {
        Self {
            now: AtomicU64::new(0),
        }
    }
    /// Advance the clock by `delta` nanoseconds.
    pub fn advance_ns(&self, delta: u64) {
        self.now.fetch_add(delta, Ordering::SeqCst);
    }
    /// Set the clock to an absolute value (for seeding).
    pub fn set_ns(&self, value: u64) {
        self.now.store(value, Ordering::SeqCst);
    }
}

impl Clock for AtomicTestClock {
    fn now_ns(&self) -> u64 {
        self.now.load(Ordering::SeqCst)
    }
}

impl TestClock {
    /// Create a test clock starting at zero.
    pub fn new() -> Self {
        Self { now: Cell::new(0) }
    }
    /// Advance the test clock by `delta` nanoseconds.
    pub fn advance_ns(&self, delta: u64) {
        self.now.set(self.now.get() + delta);
    }
}

impl Clock for TestClock {
    fn now_ns(&self) -> u64 {
        self.now.get()
    }
}

// Safety: TestClock uses Cell for single-threaded tests. We mark it
// Send+Sync behind an explicit opt-in because the engine holds clocks
// behind `Arc<dyn Clock>`. This is safe only for single-threaded tests,
// which is all we use it for.
unsafe impl Send for TestClock {}
unsafe impl Sync for TestClock {}
