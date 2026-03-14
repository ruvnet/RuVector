//! AArch64 architecture support for RuVix Cognition Kernel
//!
//! This crate provides low-level AArch64 support including:
//! - Boot sequence and early initialization
//! - Memory Management Unit (MMU) configuration
//! - Exception handling (sync, IRQ, FIQ, SError)
//! - System register access
//!
//! # Memory Layout
//!
//! ```text
//! 0x0000_0000_0000_0000 - 0x0000_FFFF_FFFF_FFFF: User space (TTBR0_EL1)
//! 0xFFFF_0000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF: Kernel space (TTBR1_EL1)
//! ```
//!
//! # Boot Sequence
//!
//! 1. Assembly entry point (`_start`) disables interrupts
//! 2. Stack pointer initialized
//! 3. BSS section cleared
//! 4. `early_init()` called to set up MMU
//! 5. Exception vectors configured
//! 6. Jump to `kernel_main()`

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod boot;
pub mod exception;
pub mod mmu;
pub mod registers;

// Re-export key types
pub use boot::{early_init, kernel_main};
pub use mmu::Mmu;
pub use registers::*;

/// AArch64 page size (4KB)
pub const PAGE_SIZE: usize = 0x1000;

/// AArch64 page shift (log2 of page size)
pub const PAGE_SHIFT: usize = 12;

/// Kernel virtual base address (upper half)
pub const KERNEL_VIRT_BASE: usize = 0xFFFF_0000_0000_0000;

/// Physical RAM base (platform-specific, QEMU virt = 0x4000_0000)
#[cfg(feature = "qemu-virt")]
pub const PHYS_RAM_BASE: usize = 0x4000_0000;

#[cfg(not(feature = "qemu-virt"))]
pub const PHYS_RAM_BASE: usize = 0x0000_0000;

/// Exception vector alignment requirement (2KB = 0x800)
pub const VECTOR_ALIGNMENT: usize = 0x800;

/// Convert virtual address to physical (simple offset for identity mapping)
#[inline]
pub const fn virt_to_phys(vaddr: usize) -> usize {
    vaddr.wrapping_sub(KERNEL_VIRT_BASE)
}

/// Convert physical address to virtual (simple offset for identity mapping)
#[inline]
pub const fn phys_to_virt(paddr: usize) -> usize {
    paddr.wrapping_add(KERNEL_VIRT_BASE)
}
