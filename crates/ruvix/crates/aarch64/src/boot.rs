//! Boot sequence and early initialization for AArch64
//!
//! This module handles the transition from assembly to Rust, including:
//! - BSS initialization
//! - MMU setup
//! - Exception vector configuration
//! - Handoff to kernel main

use crate::mmu::Mmu;
use crate::registers::{sctlr_el1_read, sctlr_el1_write, set_vbar_el1};
use crate::VECTOR_ALIGNMENT;

/// BSS start symbol (defined in linker script)
extern "C" {
    static mut __bss_start: u8;
    static mut __bss_end: u8;
}

/// Exception vector table (defined in asm/vectors.S)
extern "C" {
    static exception_vectors: u8;
}

/// Early initialization called from assembly
///
/// # Safety
///
/// This function must only be called once during boot, before any other Rust code runs.
/// It assumes:
/// - Stack is valid
/// - Interrupts are disabled
/// - Running at EL1 or higher
#[no_mangle]
pub unsafe extern "C" fn early_init() -> ! {
    // SAFETY: Called once during boot, in order, before any other Rust code
    unsafe {
        // Clear BSS section
        init_bss();

        // Initialize MMU with identity mapping
        init_mmu();

        // Configure exception vectors
        init_exception_vectors();

        // Enable MMU and caches
        enable_mmu();
    }

    // Jump to kernel main (safe after initialization)
    kernel_main()
}

/// Clear the BSS section
///
/// # Safety
///
/// Must be called before any code that uses global variables in BSS.
unsafe fn init_bss() {
    // SAFETY: Symbols defined by linker script, valid during boot
    let start = unsafe { core::ptr::addr_of_mut!(__bss_start) };
    let end = unsafe { core::ptr::addr_of_mut!(__bss_end) };
    let len = (end as usize).wrapping_sub(start as usize);

    // SAFETY: BSS region is valid and uninitialized
    unsafe {
        core::ptr::write_bytes(start, 0, len);
    }
}

/// Initialize MMU with identity mapping
///
/// # Safety
///
/// Must be called during early boot before MMU is enabled.
unsafe fn init_mmu() {
    let mut mmu = Mmu::new();

    // SAFETY: Called during boot, no concurrent access
    unsafe {
        mmu.init();
    }
}

/// Configure exception vector table
///
/// # Safety
///
/// Must be called before enabling interrupts.
unsafe fn init_exception_vectors() {
    // SAFETY: exception_vectors symbol is properly aligned (see vectors.S)
    let vbar = unsafe { core::ptr::addr_of!(exception_vectors) } as usize;

    // Verify alignment
    assert_eq!(
        vbar & (VECTOR_ALIGNMENT - 1),
        0,
        "Exception vectors must be {VECTOR_ALIGNMENT}-byte aligned"
    );

    // SAFETY: VBAR_EL1 is being set to a valid, aligned vector table
    unsafe {
        set_vbar_el1(vbar);
    }
}

/// Enable MMU and caches
///
/// # Safety
///
/// Must be called after MMU is initialized and page tables are set up.
unsafe fn enable_mmu() {
    // SAFETY: Reading SCTLR_EL1 is always safe
    let mut sctlr = unsafe { sctlr_el1_read() };

    // Set control bits:
    // M (bit 0) = 1: Enable MMU
    // C (bit 2) = 1: Enable data cache
    // I (bit 12) = 1: Enable instruction cache
    // A (bit 1) = 1: Enable alignment checking
    sctlr |= (1 << 0) | (1 << 2) | (1 << 12) | (1 << 1);

    // SAFETY: Setting SCTLR_EL1 after MMU is configured
    unsafe {
        sctlr_el1_write(sctlr);
    }

    // ISB to ensure MMU is enabled before continuing
    // SAFETY: ISB is always safe
    unsafe {
        core::arch::asm!("isb", options(nostack, preserves_flags));
    }
}

/// Kernel main entry point
///
/// This is the first Rust function called after MMU is enabled.
///
/// # Safety
///
/// Called from early_init after all initialization is complete.
#[no_mangle]
pub extern "C" fn kernel_main() -> ! {
    // TODO: Call into ruvix-kernel main initialization

    // For now, just halt
    loop {
        // SAFETY: WFI (wait for interrupt) is safe to execute
        unsafe {
            core::arch::asm!("wfi", options(nostack, nomem, preserves_flags));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_alignment() {
        // Verify alignment constant is power of 2
        assert_eq!(VECTOR_ALIGNMENT.count_ones(), 1);
        assert_eq!(VECTOR_ALIGNMENT, 0x800);
    }
}
