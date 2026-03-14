//! CPU identification and state management
//!
//! This module provides core CPU abstractions including:
//! - [`CpuId`]: A strongly-typed CPU identifier
//! - [`CpuState`]: CPU lifecycle states
//! - Helper functions for CPU queries

use core::fmt;

/// Maximum number of supported CPUs (256)
pub const MAX_CPUS: usize = 256;

/// CPU identifier (0-255)
///
/// A newtype wrapper providing type safety for CPU IDs.
/// Valid CPU IDs range from 0 to [`MAX_CPUS`]-1.
///
/// # Example
///
/// ```
/// use ruvix_smp::CpuId;
///
/// let cpu0 = CpuId::new(0).unwrap();
/// assert_eq!(cpu0.as_usize(), 0);
///
/// // Invalid CPU ID returns None
/// assert!(CpuId::new(255).is_some());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct CpuId(u8);

impl CpuId {
    /// The boot CPU (CPU 0)
    pub const BOOT_CPU: CpuId = CpuId(0);

    /// Create a new CPU ID
    ///
    /// # Arguments
    ///
    /// * `id` - The CPU ID value (0-255)
    ///
    /// # Returns
    ///
    /// `Some(CpuId)` if valid, `None` if `id >= MAX_CPUS`
    #[inline]
    pub const fn new(id: u8) -> Option<Self> {
        if (id as usize) < MAX_CPUS {
            Some(CpuId(id))
        } else {
            None
        }
    }

    /// Create a CPU ID without bounds checking
    ///
    /// # Safety
    ///
    /// Caller must ensure `id < MAX_CPUS`
    #[inline]
    pub const unsafe fn new_unchecked(id: u8) -> Self {
        CpuId(id)
    }

    /// Get the raw CPU ID value
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// Get the CPU ID as usize (for array indexing)
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Check if this is the boot CPU (CPU 0)
    #[inline]
    pub const fn is_boot_cpu(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Debug for CpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CpuId({})", self.0)
    }
}

impl fmt::Display for CpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CPU{}", self.0)
    }
}

impl From<CpuId> for u8 {
    #[inline]
    fn from(id: CpuId) -> Self {
        id.0
    }
}

impl From<CpuId> for usize {
    #[inline]
    fn from(id: CpuId) -> Self {
        id.0 as usize
    }
}

impl TryFrom<u8> for CpuId {
    type Error = CpuIdError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        CpuId::new(value).ok_or(CpuIdError::OutOfRange(value))
    }
}

impl TryFrom<usize> for CpuId {
    type Error = CpuIdError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value < MAX_CPUS {
            Ok(CpuId(value as u8))
        } else {
            Err(CpuIdError::OutOfRange(value as u8))
        }
    }
}

/// Error type for CPU ID conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuIdError {
    /// CPU ID is out of valid range
    OutOfRange(u8),
}

impl fmt::Display for CpuIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CpuIdError::OutOfRange(id) => {
                write!(f, "CPU ID {} is out of range (max {})", id, MAX_CPUS - 1)
            }
        }
    }
}

/// CPU lifecycle state
///
/// Represents the current operational state of a CPU in the system.
///
/// State transitions:
/// ```text
/// Offline -> Booting -> Online -> Halted
///                    |            |
///                    +<-----------+ (can resume)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CpuState {
    /// CPU is powered off or not present
    Offline = 0,
    /// CPU is in the boot process
    Booting = 1,
    /// CPU is running and ready for work
    Online = 2,
    /// CPU is halted (low-power state, can be woken)
    Halted = 3,
}

impl CpuState {
    /// Check if the CPU is available for scheduling
    #[inline]
    pub const fn is_schedulable(self) -> bool {
        matches!(self, CpuState::Online)
    }

    /// Check if the CPU is powered on
    #[inline]
    pub const fn is_powered(self) -> bool {
        !matches!(self, CpuState::Offline)
    }

    /// Check if the CPU can accept IPIs
    #[inline]
    pub const fn can_receive_ipi(self) -> bool {
        matches!(self, CpuState::Online | CpuState::Halted)
    }
}

impl Default for CpuState {
    fn default() -> Self {
        CpuState::Offline
    }
}

impl fmt::Display for CpuState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CpuState::Offline => write!(f, "offline"),
            CpuState::Booting => write!(f, "booting"),
            CpuState::Online => write!(f, "online"),
            CpuState::Halted => write!(f, "halted"),
        }
    }
}

/// Global CPU count (set during initialization)
///
/// This is initialized by the boot CPU and remains constant thereafter.
static mut CPU_COUNT: usize = 1;

/// Get the current CPU's ID
///
/// On ARM64, this reads the MPIDR_EL1 register and extracts the Aff0 field.
/// In test mode, returns CPU 0.
///
/// # Safety
///
/// This function uses inline assembly on ARM64 to read MPIDR_EL1.
/// The read itself is safe, but the function assumes the system is
/// properly initialized.
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_smp::current_cpu;
///
/// let cpu = current_cpu();
/// println!("Running on {}", cpu);
/// ```
#[inline]
pub fn current_cpu() -> CpuId {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        let mpidr: u64;
        // SAFETY: Reading MPIDR_EL1 is always safe
        unsafe {
            core::arch::asm!(
                "mrs {}, mpidr_el1",
                out(reg) mpidr,
                options(nostack, nomem, preserves_flags)
            );
        }
        // Extract Aff0 (bits 0-7) for the CPU ID
        let cpu_id = (mpidr & 0xFF) as u8;
        // SAFETY: CPU IDs from hardware are assumed valid
        unsafe { CpuId::new_unchecked(cpu_id) }
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        // In test mode or non-ARM64, return CPU 0
        CpuId::BOOT_CPU
    }
}

/// Check if a CPU is online
///
/// # Arguments
///
/// * `id` - CPU ID to check
///
/// # Returns
///
/// `true` if the CPU is in the `Online` state
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_smp::{cpu_online, CpuId};
///
/// if cpu_online(CpuId::BOOT_CPU) {
///     println!("Boot CPU is online");
/// }
/// ```
#[inline]
pub fn cpu_online(_id: CpuId) -> bool {
    // In a real implementation, this would check the topology
    // For now, only CPU 0 is considered online
    _id.is_boot_cpu()
}

/// Get the total number of CPUs in the system
///
/// This returns the number of CPUs detected during boot, not
/// the number currently online.
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_smp::cpu_count;
///
/// for i in 0..cpu_count() {
///     println!("CPU {} present", i);
/// }
/// ```
#[inline]
pub fn cpu_count() -> usize {
    // SAFETY: CPU_COUNT is only written once during boot
    unsafe { CPU_COUNT }
}

/// Set the CPU count (called during boot)
///
/// # Safety
///
/// Must only be called once during boot before any secondary
/// CPUs are brought online.
#[inline]
pub unsafe fn set_cpu_count(count: usize) {
    debug_assert!(count > 0 && count <= MAX_CPUS);
    // SAFETY: Caller guarantees single-threaded boot context
    unsafe {
        CPU_COUNT = count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_id_creation() {
        // Valid CPU IDs
        assert!(CpuId::new(0).is_some());
        assert!(CpuId::new(127).is_some());
        assert!(CpuId::new(255).is_some());

        // CPU 0 is boot CPU
        assert!(CpuId::BOOT_CPU.is_boot_cpu());
        assert!(CpuId::new(0).unwrap().is_boot_cpu());
        assert!(!CpuId::new(1).unwrap().is_boot_cpu());
    }

    #[test]
    fn test_cpu_id_conversion() {
        let cpu = CpuId::new(42).unwrap();
        assert_eq!(cpu.as_u8(), 42);
        assert_eq!(cpu.as_usize(), 42);
        assert_eq!(u8::from(cpu), 42);
        assert_eq!(usize::from(cpu), 42);
    }

    #[test]
    fn test_cpu_id_try_from() {
        assert!(CpuId::try_from(0u8).is_ok());
        assert!(CpuId::try_from(255u8).is_ok());
        assert!(CpuId::try_from(0usize).is_ok());
        assert!(CpuId::try_from(255usize).is_ok());

        // Out of range
        assert!(CpuId::try_from(256usize).is_err());
    }

    #[test]
    fn test_cpu_id_ordering() {
        let cpu0 = CpuId::new(0).unwrap();
        let cpu1 = CpuId::new(1).unwrap();
        let cpu2 = CpuId::new(2).unwrap();

        assert!(cpu0 < cpu1);
        assert!(cpu1 < cpu2);
        assert_eq!(cpu0, CpuId::BOOT_CPU);
    }

    #[test]
    fn test_cpu_state() {
        assert!(!CpuState::Offline.is_schedulable());
        assert!(!CpuState::Booting.is_schedulable());
        assert!(CpuState::Online.is_schedulable());
        assert!(!CpuState::Halted.is_schedulable());

        assert!(!CpuState::Offline.is_powered());
        assert!(CpuState::Booting.is_powered());
        assert!(CpuState::Online.is_powered());
        assert!(CpuState::Halted.is_powered());

        assert!(!CpuState::Offline.can_receive_ipi());
        assert!(!CpuState::Booting.can_receive_ipi());
        assert!(CpuState::Online.can_receive_ipi());
        assert!(CpuState::Halted.can_receive_ipi());
    }

    #[test]
    fn test_cpu_state_default() {
        assert_eq!(CpuState::default(), CpuState::Offline);
    }

    #[test]
    fn test_current_cpu_test_mode() {
        // In test mode, always returns CPU 0
        let cpu = current_cpu();
        assert!(cpu.is_boot_cpu());
    }

    #[test]
    fn test_cpu_count() {
        // Default count is 1
        assert!(cpu_count() >= 1);
    }
}
