//! Ticket-Based Spinlock Implementation
//!
//! This module provides a fair, ticket-based spinlock that guarantees
//! FIFO ordering of lock acquisition. This prevents starvation and
//! ensures bounded wait times.
//!
//! ## Design
//!
//! The ticket lock uses two counters:
//! - `next_ticket`: Incremented atomically when a thread wants the lock
//! - `now_serving`: Incremented when the lock is released
//!
//! A thread holds the lock when its ticket equals `now_serving`.
//!
//! ```text
//! Thread A        Thread B        Thread C
//! --------        --------        --------
//! ticket=0        ticket=1        ticket=2
//!                 wait            wait
//! (serving=0)
//! [holding]
//! release
//! (serving=1)     [holding]
//!                 release
//!                 (serving=2)     [holding]
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruvix_smp::SpinLock;
//!
//! static LOCK: SpinLock<Vec<u32>> = SpinLock::new(Vec::new());
//!
//! fn add_item(item: u32) {
//!     let mut guard = LOCK.lock();
//!     guard.push(item);
//! } // Lock released when guard drops
//! ```
//!
//! ## Interrupt Safety
//!
//! The lock does NOT disable interrupts. For IRQ-safe locking:
//!
//! ```rust,ignore
//! use ruvix_smp::SpinLock;
//! use ruvix_hal::interrupt::InterruptGuard;
//!
//! let _irq = InterruptGuard::new();  // Disable interrupts
//! let _lock = LOCK.lock();           // Acquire lock
//! // Critical section...
//! // Both released on drop
//! ```

use crate::barriers;
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicU32, Ordering};

/// Ticket-based spinlock
///
/// A fair spinlock that guarantees FIFO ordering of acquisition.
/// Uses the ticket lock algorithm to prevent starvation.
///
/// # Type Parameters
///
/// * `T` - The type of data protected by the lock
///
/// # Example
///
/// ```
/// use ruvix_smp::SpinLock;
///
/// let lock = SpinLock::new(42);
///
/// {
///     let mut guard = lock.lock();
///     *guard = 100;
/// }
///
/// assert_eq!(*lock.lock(), 100);
/// ```
#[repr(C)]
pub struct SpinLock<T> {
    /// Next ticket to be issued
    next_ticket: AtomicU32,
    /// Currently serving ticket
    now_serving: AtomicU32,
    /// The protected data
    data: UnsafeCell<T>,
}

// SAFETY: SpinLock synchronizes access to T
unsafe impl<T: Send> Sync for SpinLock<T> {}
unsafe impl<T: Send> Send for SpinLock<T> {}

impl<T> SpinLock<T> {
    /// Create a new spinlock protecting the given value
    ///
    /// # Arguments
    ///
    /// * `value` - The initial value to protect
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::SpinLock;
    ///
    /// static COUNTER: SpinLock<u64> = SpinLock::new(0);
    /// ```
    #[inline]
    pub const fn new(value: T) -> Self {
        Self {
            next_ticket: AtomicU32::new(0),
            now_serving: AtomicU32::new(0),
            data: UnsafeCell::new(value),
        }
    }

    /// Acquire the lock, blocking until available
    ///
    /// This uses the ticket lock algorithm:
    /// 1. Take a ticket (atomic increment of `next_ticket`)
    /// 2. Wait until `now_serving` equals our ticket
    /// 3. Return guard that releases on drop
    ///
    /// # Returns
    ///
    /// A [`SpinLockGuard`] that provides access to the protected data
    /// and releases the lock when dropped.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::SpinLock;
    ///
    /// let lock = SpinLock::new(vec![1, 2, 3]);
    ///
    /// {
    ///     let mut guard = lock.lock();
    ///     guard.push(4);
    /// }
    ///
    /// assert_eq!(lock.lock().len(), 4);
    /// ```
    #[inline]
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        // Take a ticket
        let my_ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);

        // Wait until it's our turn
        // Use SEVL + WFE pattern for power efficiency
        unsafe {
            barriers::sevl();
        }

        while self.now_serving.load(Ordering::Acquire) != my_ticket {
            // Power-efficient wait
            unsafe {
                barriers::wfe();
            }
        }

        // Acquire barrier ensures we see all prior writes
        // (already have Acquire above, but be explicit)
        unsafe {
            barriers::dmb();
        }

        SpinLockGuard {
            lock: self,
            _not_send: PhantomData,
        }
    }

    /// Try to acquire the lock without blocking
    ///
    /// # Returns
    ///
    /// `Some(SpinLockGuard)` if the lock was acquired, `None` if
    /// the lock is currently held.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::SpinLock;
    ///
    /// let lock = SpinLock::new(42);
    /// let guard = lock.try_lock();
    /// assert!(guard.is_some());
    /// drop(guard);
    /// ```
    #[inline]
    pub fn try_lock(&self) -> Option<SpinLockGuard<'_, T>> {
        // Get current state
        let now = self.now_serving.load(Ordering::Relaxed);
        let next = self.next_ticket.load(Ordering::Relaxed);

        // Check if lock is free (next == now means no one waiting)
        if next != now {
            return None;
        }

        // Try to take a ticket atomically
        let result = self.next_ticket.compare_exchange(
            next,
            next.wrapping_add(1),
            Ordering::Acquire,
            Ordering::Relaxed,
        );

        if result.is_ok() {
            // Got the lock
            unsafe {
                barriers::dmb();
            }
            Some(SpinLockGuard {
                lock: self,
                _not_send: PhantomData,
            })
        } else {
            // Someone else took a ticket
            None
        }
    }

    /// Check if the lock is currently held
    ///
    /// This is a hint only - the state may change immediately after
    /// this function returns.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::SpinLock;
    ///
    /// let lock = SpinLock::new(0);
    /// assert!(!lock.is_locked());
    ///
    /// let guard = lock.lock();
    /// assert!(lock.is_locked());
    /// drop(guard);
    ///
    /// assert!(!lock.is_locked());
    /// ```
    #[inline]
    pub fn is_locked(&self) -> bool {
        let now = self.now_serving.load(Ordering::Relaxed);
        let next = self.next_ticket.load(Ordering::Relaxed);
        next != now
    }

    /// Get the number of waiters (approximate)
    ///
    /// Returns the number of threads waiting to acquire the lock.
    /// This is approximate because the count may change during read.
    #[inline]
    pub fn waiter_count(&self) -> u32 {
        let now = self.now_serving.load(Ordering::Relaxed);
        let next = self.next_ticket.load(Ordering::Relaxed);
        next.wrapping_sub(now).saturating_sub(1)
    }

    /// Get mutable access to the underlying data
    ///
    /// This method requires `&mut self`, which statically guarantees
    /// exclusive access without needing to acquire the lock.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::SpinLock;
    ///
    /// let mut lock = SpinLock::new(0);
    /// *lock.get_mut() = 42;
    /// assert_eq!(*lock.lock(), 42);
    /// ```
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    /// Consume the lock and return the inner value
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::SpinLock;
    ///
    /// let lock = SpinLock::new(42);
    /// let value = lock.into_inner();
    /// assert_eq!(value, 42);
    /// ```
    #[inline]
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    /// Release the lock (internal use by guard)
    #[inline]
    fn release(&self) {
        // Ensure all writes are visible before release
        unsafe {
            barriers::dmb();
        }

        // Move to next ticket (Release ensures visibility)
        self.now_serving.fetch_add(1, Ordering::Release);

        // Wake waiting threads
        unsafe {
            barriers::sev();
        }
    }
}

impl<T: Default> Default for SpinLock<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for SpinLock<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.try_lock() {
            Some(guard) => f.debug_struct("SpinLock").field("data", &*guard).finish(),
            None => f.debug_struct("SpinLock").field("data", &"<locked>").finish(),
        }
    }
}

/// RAII guard for [`SpinLock`]
///
/// This guard provides access to the protected data and releases
/// the lock when dropped. It implements `Deref` and `DerefMut`
/// for convenient access.
///
/// # Example
///
/// ```
/// use ruvix_smp::SpinLock;
///
/// let lock = SpinLock::new(String::from("hello"));
///
/// {
///     let mut guard = lock.lock();
///     guard.push_str(" world");
/// } // Lock released here
///
/// assert_eq!(&*lock.lock(), "hello world");
/// ```
#[must_use = "if unused, the SpinLock will immediately unlock"]
pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
    /// Marker to make the guard !Send (raw pointers are !Send)
    /// This ensures the lock is released on the same thread/CPU where it was acquired.
    _not_send: PhantomData<*const ()>,
}

impl<T> Deref for SpinLockGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: We hold the lock
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for SpinLockGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: We hold the lock exclusively
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for SpinLockGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.release();
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for SpinLockGuard<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SpinLockGuard").field(&**self).finish()
    }
}

impl<T: core::fmt::Display> core::fmt::Display for SpinLockGuard<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        (**self).fmt(f)
    }
}

// SpinLockGuard is NOT Send because the lock must be released on the same CPU.
// We achieve this by including a PhantomData<*const ()> which is !Send.
// SpinLockGuard IS Sync if T is Sync, allowing shared references across threads.
//
// Note: In stable Rust, we can't use negative trait bounds (!Send), so we
// rely on the UnsafeCell in the lock providing the necessary !Sync bound
// and the PhantomData<*const ()> providing !Send.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spinlock_basic() {
        let lock = SpinLock::new(0u64);

        {
            let mut guard = lock.lock();
            *guard = 42;
        }

        assert_eq!(*lock.lock(), 42);
    }

    #[test]
    fn test_spinlock_try_lock() {
        let lock = SpinLock::new(0);

        // Should succeed when unlocked
        let guard = lock.try_lock();
        assert!(guard.is_some());
        assert!(lock.is_locked());

        // Should fail when locked
        assert!(lock.try_lock().is_none());

        drop(guard);
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_spinlock_is_locked() {
        let lock = SpinLock::new(0);
        assert!(!lock.is_locked());

        let guard = lock.lock();
        assert!(lock.is_locked());

        drop(guard);
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_spinlock_get_mut() {
        let mut lock = SpinLock::new(0);
        *lock.get_mut() = 100;
        assert_eq!(*lock.lock(), 100);
    }

    #[test]
    fn test_spinlock_into_inner() {
        let lock = SpinLock::new(42u64);
        let value = lock.into_inner();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_spinlock_default() {
        let lock: SpinLock<u64> = SpinLock::default();
        assert_eq!(*lock.lock(), 0);
    }

    #[test]
    fn test_guard_deref() {
        let lock = SpinLock::new([1u32, 2, 3, 0, 0]);

        let guard = lock.lock();
        assert_eq!(guard[0], 1);
        assert_eq!(guard[1], 2);
    }

    #[test]
    fn test_guard_deref_mut() {
        let lock = SpinLock::new([1u32, 2, 3, 0, 0]);

        {
            let mut guard = lock.lock();
            guard[3] = 4;
        }

        assert_eq!(lock.lock()[3], 4);
    }

    #[test]
    fn test_waiter_count() {
        let lock = SpinLock::new(0);
        assert_eq!(lock.waiter_count(), 0);

        let _guard = lock.lock();
        // With one holder and no waiters, count is 0
        // (holder is not counted as waiter)
        assert_eq!(lock.waiter_count(), 0);
    }

    #[test]
    fn test_spinlock_static() {
        static LOCK: SpinLock<u64> = SpinLock::new(0);

        *LOCK.lock() = 123;
        assert_eq!(*LOCK.lock(), 123);
    }
}
