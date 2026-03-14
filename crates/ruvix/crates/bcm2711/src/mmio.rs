//! # Memory-Mapped I/O Utilities for BCM2711/BCM2712
//!
//! This module provides type-safe wrappers for volatile MMIO operations with
//! proper memory barriers for ARM64 architecture on Raspberry Pi 4/5.
//!
//! ## Memory Barriers
//!
//! The BCM2711/BCM2712 requires memory barriers for device register access:
//!
//! - **DMB** (Data Memory Barrier) - Ensures memory accesses complete before next instruction
//! - **DSB** (Data Synchronization Barrier) - Stronger than DMB, waits for all instructions
//! - **ISB** (Instruction Synchronization Barrier) - Flushes pipeline
//!
//! ## VideoCore Considerations
//!
//! The VideoCore GPU shares memory with the ARM cores. When communicating via
//! the mailbox, use DSB to ensure all writes are visible to the GPU.
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_bcm2711::mmio::{read_volatile, write_volatile, dsb, dmb};
//!
//! let gpio_base = 0xFE200000 as *mut u32;
//!
//! unsafe {
//!     // Write with barrier
//!     write_volatile(gpio_base, 0x42);
//!     dsb();
//!
//!     // Read with barrier
//!     dmb();
//!     let value = read_volatile(gpio_base);
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
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Data Synchronization Barrier (DSB).
///
/// Ensures that all instructions before this point have completed before
/// any instructions after this point execute. This is stronger than DMB.
///
/// Required for VideoCore mailbox communication.
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
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Instruction Synchronization Barrier (ISB).
///
/// Flushes the pipeline and ensures that all instructions after this point
/// are fetched from cache or memory after the effects of all instructions
/// before this point are visible.
///
/// Required after modifying system registers or enabling MMU.
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
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
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

    /// Read from the register without memory barriers.
    ///
    /// Use when you need to manage barriers manually.
    ///
    /// # Safety
    ///
    /// Caller must ensure proper memory ordering.
    #[inline]
    pub unsafe fn read_raw(&self) -> T
    where
        T: Copy,
    {
        read_volatile(self.addr)
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

    /// Write to the register without memory barriers.
    ///
    /// Use when you need to manage barriers manually.
    ///
    /// # Safety
    ///
    /// Caller must ensure proper memory ordering.
    #[inline]
    pub unsafe fn write_raw(&mut self, value: T) {
        write_volatile(self.addr, value);
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

    /// Set specific bits in the register.
    #[inline]
    pub fn set_bits(&mut self, mask: T)
    where
        T: Copy + core::ops::BitOr<Output = T>,
    {
        self.modify(|v| v | mask);
    }

    /// Clear specific bits in the register.
    #[inline]
    pub fn clear_bits(&mut self, mask: T)
    where
        T: Copy + core::ops::Not<Output = T> + core::ops::BitAnd<Output = T>,
    {
        self.modify(|v| v & !mask);
    }
}

// MmioReg is Send if the underlying type is Send (single-threaded access)
unsafe impl<T: Send> Send for MmioReg<T> {}
// MmioReg is not Sync because multiple threads must not access simultaneously
// (use external synchronization if needed)

/// Delay loop using NOP instructions.
///
/// This provides a simple delay mechanism for hardware timing requirements.
/// The actual delay depends on CPU frequency.
///
/// # Arguments
///
/// * `cycles` - Number of NOP cycles to execute
#[inline]
pub fn delay_cycles(cycles: usize) {
    for _ in 0..cycles {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("nop", options(nomem, nostack, preserves_flags));
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            core::hint::spin_loop();
        }
    }
}

/// Delay for approximately the given number of microseconds.
///
/// This is a rough approximation based on estimated CPU frequency.
/// For precise timing, use the system timer instead.
///
/// # Arguments
///
/// * `us` - Microseconds to delay
#[inline]
pub fn delay_us(us: u32) {
    // Assuming ~1GHz clock, 1000 cycles per microsecond
    // Adjust based on actual CPU frequency
    delay_cycles(us as usize * 150);
}

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

    #[test]
    fn test_delay_cycles() {
        // Ensure delay doesn't panic
        delay_cycles(100);
    }
}
