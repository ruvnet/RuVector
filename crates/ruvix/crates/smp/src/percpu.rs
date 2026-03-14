//! Per-CPU Data Storage
//!
//! This module provides [`PerCpu<T>`], a container for storing per-CPU data.
//! Each CPU gets its own independent copy of the data, eliminating contention
//! and cache coherency overhead for frequently accessed data.
//!
//! ## Design
//!
//! The implementation uses a simple array indexed by CPU ID. This provides:
//! - O(1) access to the current CPU's data
//! - No locking required for read/write of own data
//! - Cache-line alignment to prevent false sharing
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruvix_smp::{PerCpu, current_cpu, CpuId};
//!
//! static COUNTERS: PerCpu<u64> = PerCpu::new([0; MAX_CPUS]);
//!
//! fn increment_local_counter() {
//!     let cpu = current_cpu();
//!     *COUNTERS.get_mut(cpu) += 1;
//! }
//!
//! fn total_count() -> u64 {
//!     COUNTERS.iter().sum()
//! }
//! ```
//!
//! ## Cache Considerations
//!
//! For best performance, the element type `T` should be cache-line sized
//! (64 bytes on most ARM64 systems) or use padding to prevent false sharing.

use crate::cpu::{CpuId, MAX_CPUS};
use core::cell::UnsafeCell;
use core::ops::{Index, IndexMut};

/// Per-CPU data container
///
/// Stores one value of type `T` per CPU, indexed by [`CpuId`].
/// Access is not synchronized - each CPU should only access its own slot.
///
/// # Type Parameters
///
/// * `T` - The type of data to store per-CPU. Must be `Send` to allow
///         initialization from any CPU.
///
/// # Example
///
/// ```
/// use ruvix_smp::{PerCpu, CpuId, MAX_CPUS};
///
/// // Static initialization
/// static DATA: PerCpu<u64> = PerCpu::new([0; MAX_CPUS]);
///
/// // Access for a specific CPU
/// let cpu0 = CpuId::BOOT_CPU;
/// let value = DATA.get(cpu0);
/// ```
#[repr(C)]
pub struct PerCpu<T> {
    /// The per-CPU data array
    ///
    /// Using UnsafeCell because we need interior mutability for
    /// the static mut pattern. Safety is ensured by the protocol:
    /// each CPU only accesses its own slot.
    data: UnsafeCell<[T; MAX_CPUS]>,
}

// SAFETY: PerCpu can be shared across threads because each CPU
// only accesses its own slot. The protocol ensures no data races.
unsafe impl<T: Send> Sync for PerCpu<T> {}
unsafe impl<T: Send> Send for PerCpu<T> {}

impl<T: Copy> PerCpu<T> {
    /// Create a new per-CPU container with uniform initialization
    ///
    /// All CPUs start with the same value.
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial array with values for each CPU
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::{PerCpu, MAX_CPUS};
    ///
    /// static COUNTERS: PerCpu<u64> = PerCpu::new([0; MAX_CPUS]);
    /// ```
    #[inline]
    pub const fn new(initial: [T; MAX_CPUS]) -> Self {
        Self {
            data: UnsafeCell::new(initial),
        }
    }

    /// Get a reference to this CPU's data
    ///
    /// # Arguments
    ///
    /// * `cpu` - The CPU ID to access
    ///
    /// # Safety
    ///
    /// The caller should ensure they are running on the specified CPU,
    /// or that external synchronization prevents concurrent access.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::{PerCpu, CpuId, MAX_CPUS};
    ///
    /// static DATA: PerCpu<u64> = PerCpu::new([42; MAX_CPUS]);
    ///
    /// let cpu = CpuId::BOOT_CPU;
    /// assert_eq!(*DATA.get(cpu), 42);
    /// ```
    #[inline]
    pub fn get(&self, cpu: CpuId) -> &T {
        // SAFETY: Array access is bounds-checked by CpuId
        unsafe { &(*self.data.get())[cpu.as_usize()] }
    }

    /// Get a mutable reference to this CPU's data
    ///
    /// # Arguments
    ///
    /// * `cpu` - The CPU ID to access
    ///
    /// # Safety
    ///
    /// The caller MUST be running on the specified CPU, or have
    /// exclusive access through external synchronization.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::{PerCpu, CpuId, MAX_CPUS};
    ///
    /// static DATA: PerCpu<u64> = PerCpu::new([0; MAX_CPUS]);
    ///
    /// let cpu = CpuId::BOOT_CPU;
    /// *DATA.get_mut(cpu) = 100;
    /// assert_eq!(*DATA.get(cpu), 100);
    /// ```
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn get_mut(&self, cpu: CpuId) -> &mut T {
        // SAFETY: Each CPU only accesses its own slot
        unsafe { &mut (*self.data.get())[cpu.as_usize()] }
    }

    /// Get the current CPU's data
    ///
    /// Convenience method that reads the current CPU ID and
    /// returns a reference to this CPU's data.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_smp::PerCpu;
    ///
    /// static DATA: PerCpu<u64> = PerCpu::new([0; MAX_CPUS]);
    ///
    /// let my_data = DATA.this_cpu();
    /// ```
    #[inline]
    pub fn this_cpu(&self) -> &T {
        self.get(crate::current_cpu())
    }

    /// Get the current CPU's data mutably
    ///
    /// Convenience method that reads the current CPU ID and
    /// returns a mutable reference to this CPU's data.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn this_cpu_mut(&self) -> &mut T {
        self.get_mut(crate::current_cpu())
    }

    /// Iterate over all CPU's data
    ///
    /// Returns an iterator over references to all CPU data slots.
    /// Useful for aggregating per-CPU data (e.g., summing counters).
    ///
    /// # Safety Note
    ///
    /// While iterating, other CPUs may be modifying their slots.
    /// For consistent snapshots, external synchronization is needed.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::{PerCpu, MAX_CPUS};
    ///
    /// static COUNTERS: PerCpu<u64> = PerCpu::new([0; MAX_CPUS]);
    ///
    /// let total: u64 = COUNTERS.iter().sum();
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        // SAFETY: We return immutable references
        unsafe { (*self.data.get()).iter() }
    }

    /// Set all CPUs to the same value
    ///
    /// # Arguments
    ///
    /// * `value` - Value to set for all CPUs
    ///
    /// # Safety
    ///
    /// Should only be called during single-threaded initialization
    /// or with external synchronization.
    #[inline]
    pub fn fill(&self, value: T) {
        // SAFETY: Caller ensures single-threaded access
        unsafe {
            (*self.data.get()).fill(value);
        }
    }
}

impl<T: Copy + Default> PerCpu<T> {
    /// Create with default values
    ///
    /// All CPUs start with `T::default()`.
    #[inline]
    pub fn zeroed() -> Self
    where
        T: Copy,
    {
        Self {
            data: UnsafeCell::new([T::default(); MAX_CPUS]),
        }
    }
}

impl<T: Copy> Index<CpuId> for PerCpu<T> {
    type Output = T;

    #[inline]
    fn index(&self, cpu: CpuId) -> &Self::Output {
        self.get(cpu)
    }
}

impl<T: Copy> IndexMut<CpuId> for PerCpu<T> {
    #[inline]
    fn index_mut(&mut self, cpu: CpuId) -> &mut Self::Output {
        // SAFETY: &mut self guarantees exclusive access
        unsafe { &mut (*self.data.get())[cpu.as_usize()] }
    }
}

/// Per-CPU data with cache line padding
///
/// Wraps a value with padding to prevent false sharing between
/// CPU cache lines. On most ARM64 systems, cache lines are 64 bytes.
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_smp::{PerCpu, MAX_CPUS};
/// use ruvix_smp::percpu::CacheAligned;
///
/// // Each counter is on its own cache line
/// static COUNTERS: PerCpu<CacheAligned<u64>> = PerCpu::new(
///     [CacheAligned::new(0); MAX_CPUS]
/// );
/// ```
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct CacheAligned<T> {
    value: T,
    // Padding is implicit due to align(64)
}

impl<T: Copy> CacheAligned<T> {
    /// Create a new cache-aligned value
    #[inline]
    pub const fn new(value: T) -> Self {
        Self { value }
    }

    /// Get the inner value
    #[inline]
    pub const fn get(&self) -> &T {
        &self.value
    }

    /// Get the inner value mutably
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Unwrap the value
    #[inline]
    pub const fn into_inner(self) -> T {
        self.value
    }
}

impl<T: Default + Copy> Default for CacheAligned<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for CacheAligned<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("CacheAligned").field(&self.value).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percpu_basic() {
        static DATA: PerCpu<u64> = PerCpu::new([0; MAX_CPUS]);

        let cpu0 = CpuId::BOOT_CPU;
        assert_eq!(*DATA.get(cpu0), 0);

        *DATA.get_mut(cpu0) = 42;
        assert_eq!(*DATA.get(cpu0), 42);
    }

    #[test]
    fn test_percpu_multiple_cpus() {
        static DATA: PerCpu<u64> = PerCpu::new([0; MAX_CPUS]);

        let cpu0 = CpuId::new(0).unwrap();
        let cpu1 = CpuId::new(1).unwrap();
        let cpu2 = CpuId::new(2).unwrap();

        *DATA.get_mut(cpu0) = 10;
        *DATA.get_mut(cpu1) = 20;
        *DATA.get_mut(cpu2) = 30;

        assert_eq!(*DATA.get(cpu0), 10);
        assert_eq!(*DATA.get(cpu1), 20);
        assert_eq!(*DATA.get(cpu2), 30);
    }

    #[test]
    fn test_percpu_iteration() {
        static DATA: PerCpu<u64> = PerCpu::new([1; MAX_CPUS]);

        let sum: u64 = DATA.iter().take(4).sum();
        assert_eq!(sum, 4);
    }

    #[test]
    fn test_percpu_fill() {
        static DATA: PerCpu<u64> = PerCpu::new([0; MAX_CPUS]);

        DATA.fill(99);

        let cpu0 = CpuId::BOOT_CPU;
        let cpu255 = CpuId::new(255).unwrap();

        assert_eq!(*DATA.get(cpu0), 99);
        assert_eq!(*DATA.get(cpu255), 99);
    }

    #[test]
    fn test_cache_aligned() {
        let aligned: CacheAligned<u64> = CacheAligned::new(42);

        assert_eq!(*aligned.get(), 42);
        assert_eq!(aligned.into_inner(), 42);

        // Verify alignment
        assert_eq!(core::mem::align_of::<CacheAligned<u64>>(), 64);
    }

    #[test]
    fn test_cache_aligned_percpu() {
        static DATA: PerCpu<CacheAligned<u64>> = PerCpu::new([CacheAligned::new(0); MAX_CPUS]);

        let cpu0 = CpuId::BOOT_CPU;
        *DATA.get_mut(cpu0).get_mut() = 123;

        assert_eq!(*DATA.get(cpu0).get(), 123);
    }
}
