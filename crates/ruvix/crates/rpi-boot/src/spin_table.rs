//! # Secondary CPU Spin Table Wake
//!
//! This module implements the spin table method for waking secondary CPU cores
//! on Raspberry Pi 4/5.
//!
//! ## Overview
//!
//! On multi-core ARM systems, secondary CPUs typically start in a holding pen,
//! waiting for the primary CPU to signal them. The spin table method uses a
//! memory location that secondary CPUs poll until they see a valid address.
//!
//! ## RPi 4 CPU Addresses
//!
//! The firmware places secondary CPUs in a spin loop, checking addresses:
//!
//! | CPU | Spin Table Address |
//! |-----|--------------------|
//! | 0 | Primary (boot CPU) |
//! | 1 | 0xE0 |
//! | 2 | 0xE8 |
//! | 3 | 0xF0 |
//!
//! ## Wake Process
//!
//! 1. Write the entry point address to the spin table
//! 2. Execute SEV (Send Event) instruction to wake WFE
//! 3. Secondary CPU reads address and jumps to it
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_rpi_boot::spin_table::{SpinTable, get_cpu_id, wake_secondary_cpus};
//!
//! // Define secondary CPU entry point
//! extern "C" fn secondary_entry() -> ! {
//!     let cpu_id = get_cpu_id();
//!     // Initialize per-CPU data
//!     // Enter idle loop or scheduler
//!     loop {}
//! }
//!
//! // Wake all secondary CPUs
//! let spin_table = SpinTable::new();
//! spin_table.wake_cpu(1, secondary_entry as usize);
//! spin_table.wake_cpu(2, secondary_entry as usize);
//! spin_table.wake_cpu(3, secondary_entry as usize);
//!
//! // Or wake all at once
//! wake_secondary_cpus(secondary_entry as usize);
//! ```
//!
//! ## Stack Setup
//!
//! Each secondary CPU needs its own stack. A common pattern:
//!
//! ```rust,no_run
//! const STACK_SIZE: usize = 64 * 1024; // 64 KB per CPU
//! static mut STACKS: [[u8; STACK_SIZE]; 4] = [[0; STACK_SIZE]; 4];
//!
//! fn get_stack_top(cpu_id: u8) -> usize {
//!     unsafe {
//!         let stack_base = STACKS[cpu_id as usize].as_ptr() as usize;
//!         stack_base + STACK_SIZE
//!     }
//! }
//! ```

use core::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Spin Table Addresses
// =============================================================================

/// Spin table address for CPU 1.
pub const SPIN_CPU1: usize = 0xE0;

/// Spin table address for CPU 2.
pub const SPIN_CPU2: usize = 0xE8;

/// Spin table address for CPU 3.
pub const SPIN_CPU3: usize = 0xF0;

/// Array of spin table addresses for all secondary CPUs.
pub const SPIN_TABLE_ADDRESSES: [usize; 3] = [SPIN_CPU1, SPIN_CPU2, SPIN_CPU3];

/// Number of secondary CPUs.
pub const NUM_SECONDARY_CPUS: usize = 3;

/// Total number of CPUs (including primary).
pub const TOTAL_CPUS: usize = 4;

// =============================================================================
// CPU State
// =============================================================================

/// CPU state enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuState {
    /// CPU is offline (not yet woken).
    Offline,
    /// CPU is in spin loop, waiting for work.
    Spinning,
    /// CPU is running kernel code.
    Online,
    /// CPU is in idle state (low power).
    Idle,
    /// CPU has halted due to error.
    Halted,
}

/// Per-CPU state tracking.
static CPU_STATES: [AtomicU64; TOTAL_CPUS] = [
    AtomicU64::new(CpuState::Online as u64),   // CPU 0 (primary) starts online
    AtomicU64::new(CpuState::Offline as u64),
    AtomicU64::new(CpuState::Offline as u64),
    AtomicU64::new(CpuState::Offline as u64),
];

/// Get the state of a CPU.
pub fn get_cpu_state(cpu_id: u8) -> CpuState {
    if cpu_id as usize >= TOTAL_CPUS {
        return CpuState::Offline;
    }

    match CPU_STATES[cpu_id as usize].load(Ordering::Acquire) {
        0 => CpuState::Offline,
        1 => CpuState::Spinning,
        2 => CpuState::Online,
        3 => CpuState::Idle,
        4 => CpuState::Halted,
        _ => CpuState::Offline,
    }
}

/// Set the state of a CPU.
pub fn set_cpu_state(cpu_id: u8, state: CpuState) {
    if (cpu_id as usize) < TOTAL_CPUS {
        CPU_STATES[cpu_id as usize].store(state as u64, Ordering::Release);
    }
}

// =============================================================================
// CPU ID Detection
// =============================================================================

/// Get the current CPU ID (0-3).
///
/// Reads the MPIDR_EL1 register to determine which CPU core we're running on.
///
/// # Returns
///
/// CPU ID (0, 1, 2, or 3 on RPi 4).
#[inline]
pub fn get_cpu_id() -> u8 {
    #[cfg(target_arch = "aarch64")]
    {
        let mpidr: u64;
        unsafe {
            core::arch::asm!("mrs {}, MPIDR_EL1", out(reg) mpidr);
        }
        // Aff0 (bits 0-7) contains the CPU ID on RPi 4
        (mpidr & 0xFF) as u8
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        0 // Default to CPU 0 on non-ARM
    }
}

/// Check if this is the primary (boot) CPU.
#[inline]
pub fn is_primary_cpu() -> bool {
    get_cpu_id() == 0
}

// =============================================================================
// Spin Table
// =============================================================================

/// Spin table driver for waking secondary CPUs.
pub struct SpinTable {
    /// Spin table base address (usually 0).
    base: usize,
}

impl SpinTable {
    /// Create a new spin table driver.
    #[inline]
    pub const fn new() -> Self {
        Self { base: 0 }
    }

    /// Create a spin table driver with a custom base address.
    ///
    /// # Safety
    ///
    /// The base address must be valid for spin table locations.
    #[inline]
    pub const unsafe fn with_base(base: usize) -> Self {
        Self { base }
    }

    /// Wake a specific secondary CPU.
    ///
    /// # Arguments
    ///
    /// * `cpu_id` - CPU ID to wake (1, 2, or 3)
    /// * `entry_point` - Address of the entry function
    ///
    /// # Returns
    ///
    /// `true` if the CPU was woken, `false` if the CPU ID is invalid.
    pub fn wake_cpu(&self, cpu_id: u8, entry_point: usize) -> bool {
        if cpu_id == 0 || cpu_id > 3 {
            return false;
        }

        let spin_addr = match cpu_id {
            1 => self.base + SPIN_CPU1,
            2 => self.base + SPIN_CPU2,
            3 => self.base + SPIN_CPU3,
            _ => return false,
        };

        // Write the entry point to the spin table location
        // SAFETY: We're writing to a known spin table address
        unsafe {
            let spin_ptr = spin_addr as *mut u64;
            core::ptr::write_volatile(spin_ptr, entry_point as u64);

            // Memory barrier to ensure write is visible
            #[cfg(target_arch = "aarch64")]
            core::arch::asm!("dsb sy", options(nostack, preserves_flags));

            // Send event to wake any CPUs in WFE
            #[cfg(target_arch = "aarch64")]
            core::arch::asm!("sev", options(nomem, nostack, preserves_flags));
        }

        set_cpu_state(cpu_id, CpuState::Spinning);
        true
    }

    /// Check if a CPU has been woken (has a non-zero entry point).
    ///
    /// # Arguments
    ///
    /// * `cpu_id` - CPU ID to check (1, 2, or 3)
    ///
    /// # Returns
    ///
    /// The entry point address, or 0 if not yet written.
    pub fn get_entry_point(&self, cpu_id: u8) -> u64 {
        if cpu_id == 0 || cpu_id > 3 {
            return 0;
        }

        let spin_addr = match cpu_id {
            1 => self.base + SPIN_CPU1,
            2 => self.base + SPIN_CPU2,
            3 => self.base + SPIN_CPU3,
            _ => return 0,
        };

        // SAFETY: We're reading from a known spin table address
        unsafe {
            let spin_ptr = spin_addr as *const u64;
            core::ptr::read_volatile(spin_ptr)
        }
    }

    /// Clear a spin table entry (set to 0).
    ///
    /// This can be used to reset a CPU's wake state.
    ///
    /// # Arguments
    ///
    /// * `cpu_id` - CPU ID to clear (1, 2, or 3)
    pub fn clear(&self, cpu_id: u8) {
        if cpu_id == 0 || cpu_id > 3 {
            return;
        }

        let spin_addr = match cpu_id {
            1 => self.base + SPIN_CPU1,
            2 => self.base + SPIN_CPU2,
            3 => self.base + SPIN_CPU3,
            _ => return,
        };

        // SAFETY: We're writing to a known spin table address
        unsafe {
            let spin_ptr = spin_addr as *mut u64;
            core::ptr::write_volatile(spin_ptr, 0);

            #[cfg(target_arch = "aarch64")]
            core::arch::asm!("dsb sy", options(nostack, preserves_flags));
        }

        set_cpu_state(cpu_id, CpuState::Offline);
    }
}

impl Default for SpinTable {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Wake all secondary CPUs with the same entry point.
///
/// # Arguments
///
/// * `entry_point` - Address of the secondary CPU entry function
pub fn wake_secondary_cpus(entry_point: usize) {
    let spin_table = SpinTable::new();

    for cpu_id in 1..=3 {
        spin_table.wake_cpu(cpu_id, entry_point);
    }
}

/// Wake all secondary CPUs with different entry points.
///
/// # Arguments
///
/// * `entry_points` - Array of entry points for CPUs 1, 2, and 3
pub fn wake_secondary_cpus_individual(entry_points: [usize; 3]) {
    let spin_table = SpinTable::new();

    for (i, &entry_point) in entry_points.iter().enumerate() {
        if entry_point != 0 {
            spin_table.wake_cpu((i + 1) as u8, entry_point);
        }
    }
}

/// Send an event to wake CPUs from WFE.
#[inline]
pub fn send_event() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("sev", options(nomem, nostack, preserves_flags));
    }
}

/// Wait for an event (low-power wait).
#[inline]
pub fn wait_for_event() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("wfe", options(nomem, nostack));
    }
}

/// Wait for interrupt (deeper low-power wait).
#[inline]
pub fn wait_for_interrupt() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("wfi", options(nomem, nostack));
    }
}

/// Count online CPUs.
pub fn count_online_cpus() -> usize {
    let mut count = 0;
    for i in 0..TOTAL_CPUS {
        if get_cpu_state(i as u8) == CpuState::Online {
            count += 1;
        }
    }
    count
}

/// Wait for a specific number of CPUs to come online.
///
/// # Arguments
///
/// * `expected` - Number of CPUs to wait for
/// * `timeout` - Maximum iterations to wait
///
/// # Returns
///
/// `true` if all CPUs came online, `false` if timeout.
pub fn wait_for_cpus_online(expected: usize, timeout: usize) -> bool {
    for _ in 0..timeout {
        if count_online_cpus() >= expected {
            return true;
        }
        for _ in 0..1000 {
            core::hint::spin_loop();
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spin_table_addresses() {
        assert_eq!(SPIN_CPU1, 0xE0);
        assert_eq!(SPIN_CPU2, 0xE8);
        assert_eq!(SPIN_CPU3, 0xF0);
    }

    #[test]
    fn test_cpu_state() {
        assert_eq!(CpuState::Offline as u64, 0);
        assert_eq!(CpuState::Online as u64, 2);
    }

    #[test]
    fn test_get_cpu_id() {
        // On non-ARM, should return 0
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert_eq!(get_cpu_id(), 0);
            assert!(is_primary_cpu());
        }
    }

    #[test]
    fn test_invalid_cpu_wake() {
        let spin_table = SpinTable::new();
        assert!(!spin_table.wake_cpu(0, 0x1000)); // Can't wake CPU 0
        assert!(!spin_table.wake_cpu(4, 0x1000)); // Invalid CPU
    }
}
