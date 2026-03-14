//! AArch64 system register accessors
//!
//! This module provides safe wrappers around inline assembly for accessing
//! AArch64 system registers.
//!
//! # Safety
//!
//! All register access is `unsafe` because:
//! - Reading can have side effects (e.g., clearing interrupt status)
//! - Writing can affect system state (e.g., enabling MMU)
//! - Incorrect values can cause exceptions or undefined behavior

use core::arch::asm;

/// Read System Control Register (SCTLR_EL1)
///
/// # Safety
///
/// Reading SCTLR_EL1 is generally safe but may have side effects.
#[inline]
pub unsafe fn sctlr_el1_read() -> u64 {
    let val: u64;
    // SAFETY: Caller ensures this is safe to execute
    unsafe {
        asm!("mrs {}, sctlr_el1", out(reg) val, options(nostack, nomem, preserves_flags));
    }
    val
}

/// Write System Control Register (SCTLR_EL1)
///
/// # Safety
///
/// Writing SCTLR_EL1 can enable/disable MMU, caches, alignment checking, etc.
/// Caller must ensure the value is valid and won't cause system instability.
#[inline]
pub unsafe fn sctlr_el1_write(val: u64) {
    // SAFETY: Caller ensures value is safe to write
    unsafe {
        asm!("msr sctlr_el1, {}", in(reg) val, options(nostack, nomem, preserves_flags));
    }
}

/// Read Translation Control Register (TCR_EL1)
///
/// # Safety
///
/// Reading TCR_EL1 is generally safe but may have side effects.
#[inline]
pub unsafe fn tcr_el1_read() -> u64 {
    let val: u64;
    // SAFETY: Caller ensures this is safe to execute
    unsafe {
        asm!("mrs {}, tcr_el1", out(reg) val, options(nostack, nomem, preserves_flags));
    }
    val
}

/// Write Translation Control Register (TCR_EL1)
///
/// # Safety
///
/// Writing TCR_EL1 configures page table parameters.
/// Caller must ensure the value is valid for the current page tables.
#[inline]
pub unsafe fn set_tcr_el1(val: u64) {
    // SAFETY: Caller ensures value is safe to write
    unsafe {
        asm!("msr tcr_el1, {}", in(reg) val, options(nostack, nomem, preserves_flags));
    }
}

/// Write Memory Attribute Indirection Register (MAIR_EL1)
///
/// # Safety
///
/// Writing MAIR_EL1 configures memory attributes.
/// Caller must ensure the value is valid and consistent with page table entries.
#[inline]
pub unsafe fn set_mair_el1(val: u64) {
    // SAFETY: Caller ensures value is safe to write
    unsafe {
        asm!("msr mair_el1, {}", in(reg) val, options(nostack, nomem, preserves_flags));
    }
}

/// Write Translation Table Base Register 0 (TTBR0_EL1)
///
/// # Safety
///
/// Writing TTBR0_EL1 sets the page table for lower VA range.
/// Caller must ensure the value points to a valid, aligned page table.
#[inline]
pub unsafe fn set_ttbr0_el1(val: u64) {
    // SAFETY: Caller ensures value is safe to write
    unsafe {
        asm!("msr ttbr0_el1, {}", in(reg) val, options(nostack, nomem, preserves_flags));
    }
}

/// Write Translation Table Base Register 1 (TTBR1_EL1)
///
/// # Safety
///
/// Writing TTBR1_EL1 sets the page table for upper VA range.
/// Caller must ensure the value points to a valid, aligned page table.
#[inline]
pub unsafe fn set_ttbr1_el1(val: u64) {
    // SAFETY: Caller ensures value is safe to write
    unsafe {
        asm!("msr ttbr1_el1, {}", in(reg) val, options(nostack, nomem, preserves_flags));
    }
}

/// Write Vector Base Address Register (VBAR_EL1)
///
/// # Safety
///
/// Writing VBAR_EL1 sets the exception vector table base address.
/// Caller must ensure the value points to a valid, aligned vector table.
#[inline]
pub unsafe fn set_vbar_el1(val: usize) {
    // SAFETY: Caller ensures value is safe to write
    unsafe {
        asm!(
            "msr vbar_el1, {}",
            in(reg) val as u64,
            options(nostack, nomem, preserves_flags)
        );
    }
}

/// Read Exception Syndrome Register (ESR_EL1)
///
/// # Safety
///
/// Reading ESR_EL1 is safe and provides exception information.
/// Should only be called from exception handlers.
#[inline]
pub unsafe fn esr_el1_read() -> u64 {
    let val: u64;
    // SAFETY: Caller ensures this is safe to execute
    unsafe {
        asm!("mrs {}, esr_el1", out(reg) val, options(nostack, nomem, preserves_flags));
    }
    val
}

/// Read Fault Address Register (FAR_EL1)
///
/// # Safety
///
/// Reading FAR_EL1 is safe and provides fault address.
/// Should only be called from exception handlers.
#[inline]
pub unsafe fn far_el1_read() -> u64 {
    let val: u64;
    // SAFETY: Caller ensures this is safe to execute
    unsafe {
        asm!("mrs {}, far_el1", out(reg) val, options(nostack, nomem, preserves_flags));
    }
    val
}

/// Read Current Exception Level (CurrentEL)
///
/// # Safety
///
/// Reading CurrentEL is always safe.
#[inline]
pub unsafe fn current_el() -> u8 {
    let val: u64;
    // SAFETY: Reading CurrentEL is always safe
    unsafe {
        asm!("mrs {}, CurrentEL", out(reg) val, options(nostack, nomem, preserves_flags));
    }
    ((val >> 2) & 0x3) as u8
}

/// Data Synchronization Barrier
///
/// # Safety
///
/// Executing DSB is safe but may impact performance.
#[inline]
pub unsafe fn dsb() {
    // SAFETY: DSB is always safe to execute
    unsafe {
        asm!("dsb sy", options(nostack, nomem, preserves_flags));
    }
}

/// Instruction Synchronization Barrier
///
/// # Safety
///
/// Executing ISB is safe but may impact performance.
#[inline]
pub unsafe fn isb() {
    // SAFETY: ISB is always safe to execute
    unsafe {
        asm!("isb", options(nostack, nomem, preserves_flags));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_accessors_compile() {
        // These tests just verify that the functions compile
        // They cannot actually execute in unit tests (would require EL1)
    }
}
