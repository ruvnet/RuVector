//! # Memory-Mapped I/O Utilities
//!
//! This module provides type-safe wrappers for volatile MMIO operations with
//! proper memory barriers for ARM64 architecture.
//!
//! ## Memory Barriers
//!
//! - **DMB** (Data Memory Barrier) - Ensures memory accesses complete before next instruction
//! - **DSB** (Data Synchronization Barrier) - Stronger than DMB, waits for all instructions
//! - **ISB** (Instruction Synchronization Barrier) - Flushes pipeline, ensures subsequent instructions see barrier effects
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_drivers::mmio::{read_volatile, write_volatile, dsb, dmb};
//!
//! let uart_base = 0x0900_0000 as *mut u32;
//!
//! unsafe {
//!     // Write with barrier
//!     write_volatile(uart_base, 0x42);
//!     dsb();
//!
//!     // Read with barrier
//!     dmb();
//!     let value = read_volatile(uart_base);
//! }
//! ```

use core::ptr::{read_volatile as ptr_read_volatile, write_volatile as ptr_write_volatile};

/// Read a value from MMIO register with volatile semantics.
///
/// # Safety
///
/// - `addr` must be a valid MMIO register address
/// - `addr` must be properly aligned for type `T`
/// - Register must be readable without side effects
#[inline(always)]
pub unsafe fn read_volatile<T>(addr: *const T) -> T {
    ptr_read_volatile(addr)
}

/// Write a value to MMIO register with volatile semantics.
///
/// # Safety
///
/// - `addr` must be a valid MMIO register address
/// - `addr` must be properly aligned for type `T`
/// - Writing `value` must not violate hardware invariants
#[inline(always)]
pub unsafe fn write_volatile<T>(addr: *mut T, value: T) {
    ptr_write_volatile(addr, value);
}

/// Data Memory Barrier (DMB).
///
/// Ensures that all explicit memory accesses before this instruction complete
/// before any explicit memory accesses after this instruction start.
///
/// # Safety
///
/// This is safe to call at any time, but may affect performance if overused.
#[inline(always)]
pub fn dmb() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("dmb sy", options(nostack, preserves_flags));
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // No-op on non-ARM architectures for testing
    }
}

/// Data Synchronization Barrier (DSB).
///
/// Ensures that all instructions before this point have completed before
/// any instructions after this point execute. This is stronger than DMB.
///
/// # Safety
///
/// This is safe to call at any time, but may affect performance if overused.
#[inline(always)]
pub fn dsb() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // No-op on non-ARM architectures for testing
    }
}

/// Instruction Synchronization Barrier (ISB).
///
/// Flushes the pipeline and ensures that all instructions after this point
/// are fetched from cache or memory after the effects of all instructions
/// before this point are visible.
///
/// # Safety
///
/// This is safe to call at any time, but may affect performance if overused.
#[inline(always)]
pub fn isb() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("isb", options(nostack, preserves_flags));
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // No-op on non-ARM architectures for testing
    }
}

/// Type-safe MMIO register wrapper.
///
/// Provides safe read/write operations with automatic barriers.
#[repr(transparent)]
pub struct MmioReg<T> {
    addr: *mut T,
}

impl<T> MmioReg<T> {
    /// Create a new MMIO register wrapper.
    ///
    /// # Safety
    ///
    /// - `addr` must be a valid MMIO register address
    /// - `addr` must be properly aligned for type `T`
    /// - The register must remain valid for the lifetime of this wrapper
    #[inline]
    pub const unsafe fn new(addr: usize) -> Self {
        Self {
            addr: addr as *mut T,
        }
    }

    /// Read from the register with memory barrier.
    ///
    /// Performs: DMB -> READ -> DMB
    #[inline]
    pub fn read(&self) -> T
    where
        T: Copy,
    {
        dmb();
        // SAFETY: Caller guarantees addr is valid MMIO register
        let value = unsafe { read_volatile(self.addr) };
        dmb();
        value
    }

    /// Write to the register with memory barrier.
    ///
    /// Performs: DSB -> WRITE -> DSB
    #[inline]
    pub fn write(&mut self, value: T) {
        dsb();
        // SAFETY: Caller guarantees addr is valid MMIO register
        unsafe { write_volatile(self.addr, value) };
        dsb();
    }

    /// Read-modify-write operation with memory barriers.
    ///
    /// Applies the given function to the current value and writes back the result.
    #[inline]
    pub fn modify<F>(&mut self, f: F)
    where
        T: Copy,
        F: FnOnce(T) -> T,
    {
        dsb();
        // SAFETY: Caller guarantees addr is valid MMIO register
        let value = unsafe { read_volatile(self.addr) };
        let new_value = f(value);
        unsafe { write_volatile(self.addr, new_value) };
        dsb();
    }
}

// MmioReg is Send if the underlying type is Send (single-threaded access)
unsafe impl<T: Send> Send for MmioReg<T> {}
// MmioReg is not Sync because multiple threads must not access simultaneously
// (use external synchronization if needed)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barriers_compile() {
        // Just ensure barriers compile and don't panic
        dmb();
        dsb();
        isb();
    }
}
