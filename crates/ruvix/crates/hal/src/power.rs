//! # Power Management Abstraction
//!
//! Provides traits for CPU power state management, reset, and shutdown.
//!
//! ## Design
//!
//! - **WFI/WFE support** - Wait for interrupt/event low-power modes
//! - **PSCI integration** - ARM Power State Coordination Interface
//! - **Multi-core coordination** - Safe CPU hotplug
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_hal::PowerManagement;
//!
//! fn idle_loop<P: PowerManagement>(pm: &P) -> ! {
//!     loop {
//!         pm.wait_for_interrupt();
//!     }
//! }
//! ```

/// Power management error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerError {
    /// Operation not supported by hardware
    Unsupported,
    /// Invalid CPU ID
    InvalidCpu,
    /// CPU is already in the requested state
    AlreadyInState,
    /// Power state transition denied (e.g., last CPU shutdown)
    Denied,
    /// Hardware fault
    HardwareFault,
}

impl core::fmt::Display for PowerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported => write!(f, "operation not supported"),
            Self::InvalidCpu => write!(f, "invalid CPU ID"),
            Self::AlreadyInState => write!(f, "CPU already in requested state"),
            Self::Denied => write!(f, "power state transition denied"),
            Self::HardwareFault => write!(f, "power hardware fault"),
        }
    }
}

/// CPU power states (ARM64 PSCI)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    /// CPU is running
    On,
    /// CPU is in WFI/WFE (wake on interrupt)
    Idle,
    /// CPU is powered off (requires boot to restart)
    Off,
}

/// Reset types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetType {
    /// Warm reset (preserve DRAM)
    Warm,
    /// Cold reset (full power cycle)
    Cold,
    /// Shutdown (power off)
    Shutdown,
}

/// Power Management abstraction
///
/// This trait provides CPU power state control for ARM64 PSCI and similar.
///
/// ## Thread Safety
///
/// Power management operations MUST be safe to call from interrupt context.
///
/// ## Example Implementation (ARM64 PSCI)
///
/// ```rust,ignore
/// use ruvix_hal::{PowerManagement, power::*};
///
/// struct ArmPsci;
///
/// impl ArmPsci {
///     fn smc_call(function_id: u32, arg1: u64, arg2: u64, arg3: u64) -> i32 {
///         // Make SMC (Secure Monitor Call) to EL3 firmware
///         let result: i32;
///         unsafe {
///             core::arch::asm!(
///                 "smc #0",
///                 inout("x0") function_id as u64 => result as u64,
///                 in("x1") arg1,
///                 in("x2") arg2,
///                 in("x3") arg3,
///             );
///         }
///         result
///     }
/// }
///
/// impl PowerManagement for ArmPsci {
///     fn wait_for_interrupt(&self) {
///         unsafe {
///             core::arch::asm!("wfi");
///         }
///     }
///
///     fn wait_for_event(&self) {
///         unsafe {
///             core::arch::asm!("wfe");
///         }
///     }
///
///     fn send_event(&self) {
///         unsafe {
///             core::arch::asm!("sev");
///         }
///     }
///
///     fn reset(&self, reset_type: ResetType) -> Result<!, PowerError> {
///         let function_id = match reset_type {
///             ResetType::Warm => 0x84000009,    // PSCI_SYSTEM_RESET
///             ResetType::Cold => 0x84000009,    // Same as warm for most systems
///             ResetType::Shutdown => 0x84000008, // PSCI_SYSTEM_OFF
///         };
///
///         Self::smc_call(function_id, 0, 0, 0);
///
///         // If we reach here, reset failed
///         loop {
///             self.wait_for_interrupt();
///         }
///     }
///
///     fn cpu_on(&self, cpu_id: u32, entry_point: u64) -> Result<(), PowerError> {
///         const PSCI_CPU_ON: u32 = 0xC4000003;
///
///         let result = Self::smc_call(PSCI_CPU_ON, cpu_id as u64, entry_point, 0);
///
///         match result {
///             0 => Ok(()),              // SUCCESS
///             -1 => Err(PowerError::Unsupported), // NOT_SUPPORTED
///             -2 => Err(PowerError::InvalidCpu),  // INVALID_PARAMETERS
///             -4 => Err(PowerError::AlreadyInState), // ALREADY_ON
///             _ => Err(PowerError::HardwareFault),
///         }
///     }
///
///     fn cpu_off(&self) -> Result<!, PowerError> {
///         const PSCI_CPU_OFF: u32 = 0x84000002;
///
///         Self::smc_call(PSCI_CPU_OFF, 0, 0, 0);
///
///         // If we reach here, CPU_OFF failed
///         loop {
///             self.wait_for_interrupt();
///         }
///     }
///
///     fn get_cpu_state(&self, cpu_id: u32) -> Result<PowerState, PowerError> {
///         const PSCI_AFFINITY_INFO: u32 = 0xC4000004;
///
///         let result = Self::smc_call(PSCI_AFFINITY_INFO, cpu_id as u64, 0, 0);
///
///         match result {
///             0 => Ok(PowerState::On),
///             1 => Ok(PowerState::Off),
///             2 => Ok(PowerState::Idle), // ON_PENDING
///             _ => Err(PowerError::InvalidCpu),
///         }
///     }
///
///     fn current_cpu_id(&self) -> u32 {
///         let mpidr: u64;
///         unsafe {
///             core::arch::asm!("mrs {}, mpidr_el1", out(reg) mpidr);
///         }
///         // Extract Aff0 (CPU ID within cluster)
///         (mpidr & 0xFF) as u32
///     }
/// }
/// ```
pub trait PowerManagement {
    /// Wait for interrupt (WFI)
    ///
    /// Enters low-power state until an interrupt arrives.
    /// This is the primary idle mechanism.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_hal::PowerManagement;
    ///
    /// fn scheduler_idle<P: PowerManagement>(pm: &P) {
    ///     loop {
    ///         if no_tasks_ready() {
    ///             pm.wait_for_interrupt();
    ///         } else {
    ///             break;
    ///         }
    ///     }
    /// }
    /// ```
    fn wait_for_interrupt(&self);

    /// Wait for event (WFE)
    ///
    /// Enters low-power state until an event arrives.
    /// Used for spinlock contention and synchronization.
    fn wait_for_event(&self);

    /// Send event (SEV)
    ///
    /// Wakes all CPUs in WFE state.
    /// Used to wake CPUs after releasing a lock.
    fn send_event(&self);

    /// Reset or shutdown the system
    ///
    /// This function never returns on success. On error, it returns
    /// and the caller must handle the failure (typically by looping).
    ///
    /// # Arguments
    ///
    /// * `reset_type` - Type of reset to perform
    ///
    /// # Errors
    ///
    /// Returns `PowerError::Unsupported` if reset type is not available.
    /// Returns `PowerError::HardwareFault` if reset fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_hal::{PowerManagement, power::ResetType};
    ///
    /// fn panic_reset<P: PowerManagement>(pm: &P) -> ! {
    ///     pm.reset(ResetType::Warm);
    ///     // If reset fails, loop forever
    ///     loop { pm.wait_for_interrupt() }
    /// }
    /// ```
    fn reset(&self, reset_type: ResetType);

    /// Power on a secondary CPU
    ///
    /// # Arguments
    ///
    /// * `cpu_id` - Target CPU ID (MPIDR affinity)
    /// * `entry_point` - Physical address of entry function
    ///
    /// # Errors
    ///
    /// - `InvalidCpu` if CPU ID is invalid
    /// - `AlreadyInState` if CPU is already on
    /// - `Unsupported` if multi-core is not available
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_hal::PowerManagement;
    ///
    /// fn boot_secondary_cpus<P: PowerManagement>(pm: &P) -> Result<(), ruvix_hal::power::PowerError> {
    ///     for cpu_id in 1..4 {
    ///         pm.cpu_on(cpu_id, secondary_entry as u64)?;
    ///     }
    ///     Ok(())
    /// }
    ///
    /// extern "C" fn secondary_entry() -> ! {
    ///     // Secondary CPU initialization
    ///     loop {}
    /// }
    /// ```
    #[allow(unused_variables)]
    fn cpu_on(&self, cpu_id: u32, entry_point: u64) -> Result<(), PowerError> {
        Err(PowerError::Unsupported)
    }

    /// Power off the current CPU
    ///
    /// This function never returns on success. On error, it returns
    /// and the caller must handle the failure.
    ///
    /// # Errors
    ///
    /// Returns `PowerError::Denied` if this is the last CPU.
    /// Returns `PowerError::Unsupported` if CPU hotplug is not available.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_hal::{PowerManagement, power::PowerError};
    ///
    /// fn shutdown_cpu<P: PowerManagement>(pm: &P) -> ! {
    ///     match pm.cpu_off() {
    ///         Ok(()) => unreachable!(), // CPU was powered off
    ///         Err(_) => loop { pm.wait_for_interrupt() },
    ///     }
    /// }
    /// ```
    fn cpu_off(&self) -> Result<(), PowerError> {
        Err(PowerError::Unsupported)
    }

    /// Get power state of a CPU
    ///
    /// # Arguments
    ///
    /// * `cpu_id` - Target CPU ID
    ///
    /// # Errors
    ///
    /// - `InvalidCpu` if CPU ID is invalid
    /// - `Unsupported` if multi-core is not available
    #[allow(unused_variables)]
    fn get_cpu_state(&self, cpu_id: u32) -> Result<PowerState, PowerError> {
        Err(PowerError::Unsupported)
    }

    /// Get current CPU ID
    ///
    /// Returns the ID of the CPU executing this code.
    fn current_cpu_id(&self) -> u32 {
        0 // Default: single-core
    }

    /// Get number of CPUs available
    fn cpu_count(&self) -> u32 {
        1 // Default: single-core
    }

    /// Enter deep sleep (platform-specific)
    ///
    /// This is a hint to enter the deepest available low-power state.
    /// The CPU will wake on interrupt.
    fn deep_sleep(&self) {
        self.wait_for_interrupt();
    }
}

/// Helper for busy-wait spinning with WFE
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_hal::PowerManagement;
/// use ruvix_hal::power::SpinWait;
///
/// fn spin_on_lock<P: PowerManagement>(pm: &P, lock: &AtomicBool) {
///     let mut spin = SpinWait::new();
///     while lock.load(Ordering::Acquire) {
///         spin.wait(pm);
///     }
/// }
/// ```
#[derive(Debug)]
pub struct SpinWait {
    count: u32,
}

impl SpinWait {
    /// Create a new spin-wait helper
    pub const fn new() -> Self {
        Self { count: 0 }
    }

    /// Wait for a short duration
    ///
    /// Uses exponential backoff: spin → WFE
    pub fn wait<P: PowerManagement>(&mut self, pm: &P) {
        self.count += 1;

        if self.count < 10 {
            // Spin for a bit
            for _ in 0..100 {
                core::hint::spin_loop();
            }
        } else {
            // Use WFE for longer waits
            pm.wait_for_event();
        }
    }

    /// Reset spin counter
    pub fn reset(&mut self) {
        self.count = 0;
    }
}

impl Default for SpinWait {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockPower {
        cpu_id: u32,
    }

    impl MockPower {
        fn new() -> Self {
            Self { cpu_id: 0 }
        }
    }

    impl PowerManagement for MockPower {
        fn wait_for_interrupt(&self) {
            // Mock: do nothing
        }

        fn wait_for_event(&self) {
            // Mock: do nothing
        }

        fn send_event(&self) {
            // Mock: do nothing
        }

        fn reset(&self, _reset_type: ResetType) {
            // Mock: do nothing (would normally not return)
        }

        fn current_cpu_id(&self) -> u32 {
            self.cpu_id
        }
    }

    #[test]
    fn test_current_cpu_id() {
        let pm = MockPower::new();
        assert_eq!(pm.current_cpu_id(), 0);
    }

    #[test]
    fn test_cpu_count() {
        let pm = MockPower::new();
        assert_eq!(pm.cpu_count(), 1);
    }

    #[test]
    fn test_spin_wait() {
        let pm = MockPower::new();
        let mut spin = SpinWait::new();

        for _ in 0..5 {
            spin.wait(&pm);
        }

        spin.reset();
        assert_eq!(spin.count, 0);
    }
}
