//! # Raspberry Pi Boot Support for RuVix
//!
//! This crate provides boot support for Raspberry Pi 4/5 as part of the
//! RuVix Cognition Kernel (ADR-087 Phase D).
//!
//! ## Boot Process Overview
//!
//! The Raspberry Pi boot process:
//!
//! 1. **GPU ROM** - Loads bootcode.bin from SD card
//! 2. **bootcode.bin** - Initializes SDRAM, loads start4.elf
//! 3. **start4.elf** - GPU firmware, parses config.txt, loads kernel8.img
//! 4. **kernel8.img** - Our kernel, entry at _start
//!
//! ## DTB (Device Tree Blob)
//!
//! The firmware passes the DTB address in register x0. The DTB contains:
//!
//! - Memory layout
//! - Boot arguments (cmdline)
//! - Device configuration
//! - Reserved memory regions
//!
//! ## Entry Point Requirements
//!
//! When our kernel starts:
//!
//! - x0 = DTB physical address
//! - x1 = 0 (reserved)
//! - x2 = 0 (reserved)
//! - x3 = 0 (reserved)
//! - MMU is OFF
//! - D-cache is OFF
//! - I-cache may be ON or OFF
//! - Interrupts are masked (DAIF)
//! - Running in EL2 (hypervisor) or EL1 (kernel)
//!
//! ## Memory Layout (4GB RPi 4)
//!
//! ```text
//! 0x0000_0000 - 0x0000_0FFF : Reserved (interrupt vectors)
//! 0x0000_1000 - 0x0007_FFFF : Kernel code/data
//! 0x0008_0000 - ...         : Traditional kernel load address
//! ...
//! 0x3B40_0000 - 0x3FFF_FFFF : GPU memory (configurable via config.txt)
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_rpi_boot::{BootInfo, parse_dtb_header};
//!
//! #[no_mangle]
//! pub extern "C" fn _start(dtb_addr: usize) -> ! {
//!     // Validate DTB
//!     let dtb = unsafe { parse_dtb_header(dtb_addr) };
//!     if let Some(info) = dtb {
//!         // DTB is valid, info contains size and version
//!     }
//!
//!     // Initialize early UART for debugging
//!     ruvix_rpi_boot::early_uart_init();
//!     ruvix_rpi_boot::early_print("RuVix booting...\n");
//!
//!     // Wake secondary CPUs
//!     ruvix_rpi_boot::wake_secondary_cpus();
//!
//!     loop {}
//! }
//! ```

#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod dtb;
pub mod early_uart;
pub mod spin_table;

pub use config::{parse_cmdline, BootConfig, ConfigError};
pub use dtb::{parse_dtb_header, DtbHeader, DtbInfo};
pub use early_uart::{early_print, early_uart_init};
pub use spin_table::{get_cpu_id, wake_secondary_cpus, SpinTable, CpuState};

use ruvix_bcm2711::{GPIO_BASE, MINI_UART_BASE, MAILBOX_BASE};

// =============================================================================
// Boot Constants
// =============================================================================

/// Default kernel load address (0x80000).
pub const KERNEL_LOAD_ADDRESS: usize = 0x0008_0000;

/// Stack size per CPU core (64 KB).
pub const STACK_SIZE_PER_CPU: usize = 64 * 1024;

/// Number of CPU cores on RPi 4.
pub const NUM_CPUS: usize = 4;

/// Total stack space needed (256 KB for 4 cores).
pub const TOTAL_STACK_SIZE: usize = STACK_SIZE_PER_CPU * NUM_CPUS;

/// Magic value for spin table wake signal.
pub const SPIN_TABLE_MAGIC: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// ARM local peripheral base for RPi 4.
pub const LOCAL_PERIPHERAL_BASE: usize = 0xFF80_0000;

// =============================================================================
// Boot Information
// =============================================================================

/// Boot information collected during early boot.
#[derive(Debug, Clone, Copy)]
pub struct BootInfo {
    /// Physical address of the DTB.
    pub dtb_addr: usize,
    /// Size of the DTB in bytes.
    pub dtb_size: usize,
    /// Total RAM size in bytes.
    pub ram_size: usize,
    /// Number of CPU cores detected.
    pub num_cpus: usize,
    /// Current exception level (1, 2, or 3).
    pub exception_level: u8,
    /// Board revision code.
    pub board_revision: u32,
}

impl BootInfo {
    /// Create a new boot info with default values.
    pub const fn new() -> Self {
        Self {
            dtb_addr: 0,
            dtb_size: 0,
            ram_size: 0,
            num_cpus: NUM_CPUS,
            exception_level: 0,
            board_revision: 0,
        }
    }
}

impl Default for BootInfo {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Exception Level Detection
// =============================================================================

/// Get the current exception level.
///
/// # Returns
///
/// - 1 for EL1 (kernel mode)
/// - 2 for EL2 (hypervisor mode)
/// - 3 for EL3 (secure monitor mode)
#[inline]
pub fn current_el() -> u8 {
    #[cfg(target_arch = "aarch64")]
    {
        let el: u64;
        unsafe {
            core::arch::asm!("mrs {}, CurrentEL", out(reg) el);
        }
        ((el >> 2) & 0x3) as u8
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        1 // Default to EL1 for non-ARM builds
    }
}

/// Check if we're running in EL2 (hypervisor mode).
#[inline]
pub fn is_el2() -> bool {
    current_el() == 2
}

/// Check if we're running in EL1 (kernel mode).
#[inline]
pub fn is_el1() -> bool {
    current_el() == 1
}

// =============================================================================
// Early Boot Utilities
// =============================================================================

/// Halt the CPU in a low-power loop.
///
/// This is used when the boot process fails and we need to stop execution.
#[inline(never)]
pub fn halt() -> ! {
    loop {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("wfe", options(nomem, nostack));
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            core::hint::spin_loop();
        }
    }
}

/// Delay for approximately the given number of cycles.
///
/// This is used for hardware timing during early boot.
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

/// Memory barrier (DSB).
#[inline]
pub fn dsb() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Instruction barrier (ISB).
#[inline]
pub fn isb() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("isb", options(nostack, preserves_flags));
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

// =============================================================================
// MMU Control (Pre-MMU State)
// =============================================================================

/// Disable the MMU (should already be disabled at boot).
///
/// This ensures we're in a known state before enabling our own page tables.
pub fn disable_mmu() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut sctlr: u64;
        core::arch::asm!("mrs {}, SCTLR_EL1", out(reg) sctlr);
        sctlr &= !(1 << 0); // Clear M bit (MMU enable)
        core::arch::asm!("msr SCTLR_EL1, {}", in(reg) sctlr);
        isb();
    }
}

/// Invalidate all TLB entries.
pub fn invalidate_tlb() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("tlbi vmalle1is", options(nostack));
        dsb();
        isb();
    }
}

/// Invalidate all instruction caches.
pub fn invalidate_icache() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("ic iallu", options(nostack));
        isb();
    }
}

// =============================================================================
// EL2 to EL1 Transition
// =============================================================================

/// Drop from EL2 to EL1.
///
/// The firmware typically starts us in EL2. This function transitions to EL1.
///
/// # Safety
///
/// Must only be called once, and only if currently in EL2.
pub unsafe fn drop_to_el1() {
    #[cfg(target_arch = "aarch64")]
    {
        // Configure EL2 to allow EL1 to function normally

        // HCR_EL2: Hypervisor Configuration Register
        // RW (bit 31) = 1: EL1 is AArch64
        let hcr_el2: u64 = 1 << 31;
        core::arch::asm!("msr HCR_EL2, {}", in(reg) hcr_el2);

        // SPSR_EL2: Saved Program Status Register
        // M[3:0] = 0b0101: EL1h (EL1 with SP_EL1)
        // D, A, I, F = 1: Mask all exceptions
        let spsr_el2: u64 = 0x3C5; // EL1h with DAIF masked
        core::arch::asm!("msr SPSR_EL2, {}", in(reg) spsr_el2);

        // ELR_EL2: Exception Link Register
        // Set to address of next instruction
        let el1_entry: u64;
        core::arch::asm!("adr {}, 1f", out(reg) el1_entry);
        core::arch::asm!("msr ELR_EL2, {}", in(reg) el1_entry);

        // SP_EL1: Stack pointer for EL1
        // Keep current stack pointer
        let sp: u64;
        core::arch::asm!("mov {}, sp", out(reg) sp);
        core::arch::asm!("msr SP_EL1, {}", in(reg) sp);

        // Return to EL1
        core::arch::asm!("eret");

        // Label for EL1 entry
        core::arch::asm!("1:");
    }
}

// =============================================================================
// Kernel Entry Stub
// =============================================================================

/// Kernel entry stub information.
///
/// This structure is placed at the beginning of the kernel image to allow
/// the firmware to identify it as a valid kernel.
#[repr(C)]
pub struct KernelHeader {
    /// Branch instruction to _start.
    pub branch: u32,
    /// Reserved (must be 0).
    pub reserved0: u32,
    /// Text offset from load address.
    pub text_offset: u64,
    /// Image size (may be 0 if unknown).
    pub image_size: u64,
    /// Flags (bit 0 = where to place kernel).
    pub flags: u64,
    /// Reserved fields.
    pub reserved1: [u64; 3],
    /// Magic number: "ARM\x64".
    pub magic: u32,
    /// PE offset (0 for Linux-style kernels).
    pub pe_offset: u32,
}

impl KernelHeader {
    /// ARM64 magic number ("ARM\x64").
    pub const MAGIC: u32 = 0x644D5241; // "ARM\x64" in little-endian
}

// =============================================================================
// Module Version
// =============================================================================

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boot_info_default() {
        let info = BootInfo::new();
        assert_eq!(info.dtb_addr, 0);
        assert_eq!(info.num_cpus, NUM_CPUS);
    }

    #[test]
    fn test_constants() {
        assert_eq!(KERNEL_LOAD_ADDRESS, 0x80000);
        assert_eq!(NUM_CPUS, 4);
        assert_eq!(STACK_SIZE_PER_CPU, 64 * 1024);
    }

    #[test]
    fn test_kernel_header_magic() {
        assert_eq!(KernelHeader::MAGIC, 0x644D5241);
    }

    #[test]
    fn test_current_el() {
        // On non-ARM platforms, should return 1
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert_eq!(current_el(), 1);
        }
    }
}
