//! CPU Topology Management
//!
//! This module provides [`CpuTopology`], which tracks the state of all CPUs
//! in the system. It manages CPU lifecycle (boot, halt, resume) and provides
//! queries for scheduling decisions.
//!
//! ## Design
//!
//! The topology uses a fixed-size array of atomic CPU states, allowing
//! lock-free state transitions. This is critical for multi-core boot where
//! locks may not be available.
//!
//! ## Boot Sequence
//!
//! 1. Primary CPU initializes topology with `init(num_cpus)`
//! 2. Primary boots itself with `boot_cpu(BOOT_CPU)`
//! 3. Secondary CPUs boot via PSCI or spin-table
//! 4. Each secondary calls `boot_cpu(my_id)` when ready
//! 5. Primary waits for all secondaries with `wait_for_all_cpus()`
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_smp::{CpuTopology, CpuId, CpuState};
//!
//! static TOPOLOGY: CpuTopology = CpuTopology::new();
//!
//! fn kernel_main() {
//!     // Initialize with 4 CPUs
//!     TOPOLOGY.init(4);
//!
//!     // Boot primary CPU
//!     TOPOLOGY.boot_cpu(CpuId::BOOT_CPU);
//!
//!     // Start secondaries via PSCI
//!     for i in 1..4 {
//!         psci_cpu_on(i, secondary_entry);
//!     }
//!
//!     // Wait for all to come online
//!     TOPOLOGY.wait_for_all_cpus();
//!
//!     println!("{} CPUs online", TOPOLOGY.online_count());
//! }
//!
//! fn secondary_entry() {
//!     let my_id = current_cpu();
//!     TOPOLOGY.boot_cpu(my_id);
//!     // Now online, enter scheduler
//! }
//! ```

use crate::barriers;
use crate::cpu::{CpuId, CpuState, MAX_CPUS};
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// CPU topology tracker
///
/// Tracks the state of all CPUs in the system. Uses atomic operations
/// for lock-free state transitions.
///
/// # Example
///
/// ```
/// use ruvix_smp::{CpuTopology, CpuId, CpuState};
///
/// static TOPOLOGY: CpuTopology = CpuTopology::new();
///
/// // Check CPU state
/// assert_eq!(TOPOLOGY.state(CpuId::BOOT_CPU), CpuState::Offline);
/// ```
pub struct CpuTopology {
    /// State of each CPU (atomic for lock-free access)
    states: [AtomicU8; MAX_CPUS],
    /// Number of CPUs in the system
    num_cpus: AtomicUsize,
    /// Number of CPUs currently online
    online_count: AtomicUsize,
    /// Boot order tracking (CPU IDs in boot order)
    boot_order: [AtomicU8; MAX_CPUS],
    /// Index into boot_order for next boot
    boot_index: AtomicUsize,
}

// SAFETY: All fields are atomic
unsafe impl Sync for CpuTopology {}
unsafe impl Send for CpuTopology {}

impl CpuTopology {
    /// Create a new CPU topology tracker
    ///
    /// All CPUs start in `Offline` state.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::CpuTopology;
    ///
    /// static TOPOLOGY: CpuTopology = CpuTopology::new();
    /// ```
    #[inline]
    pub const fn new() -> Self {
        // Workaround for const array initialization
        const OFFLINE: AtomicU8 = AtomicU8::new(CpuState::Offline as u8);
        const ZERO: AtomicU8 = AtomicU8::new(0);

        Self {
            states: [OFFLINE; MAX_CPUS],
            num_cpus: AtomicUsize::new(0),
            online_count: AtomicUsize::new(0),
            boot_order: [ZERO; MAX_CPUS],
            boot_index: AtomicUsize::new(0),
        }
    }

    /// Initialize topology with the number of CPUs
    ///
    /// Should be called once by the boot CPU before any secondary
    /// CPUs are started.
    ///
    /// # Arguments
    ///
    /// * `num_cpus` - Total number of CPUs in the system
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::CpuTopology;
    ///
    /// static TOPOLOGY: CpuTopology = CpuTopology::new();
    /// TOPOLOGY.init(4);
    /// assert_eq!(TOPOLOGY.cpu_count(), 4);
    /// ```
    #[inline]
    pub fn init(&self, num_cpus: usize) {
        debug_assert!(num_cpus > 0 && num_cpus <= MAX_CPUS);
        self.num_cpus.store(num_cpus, Ordering::Release);
    }

    /// Get the total number of CPUs
    #[inline]
    pub fn cpu_count(&self) -> usize {
        self.num_cpus.load(Ordering::Acquire)
    }

    /// Get the number of online CPUs
    #[inline]
    pub fn online_count(&self) -> usize {
        self.online_count.load(Ordering::Acquire)
    }

    /// Get a CPU's current state
    ///
    /// # Arguments
    ///
    /// * `cpu` - CPU ID to query
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::{CpuTopology, CpuId, CpuState};
    ///
    /// static TOPOLOGY: CpuTopology = CpuTopology::new();
    /// assert_eq!(TOPOLOGY.state(CpuId::BOOT_CPU), CpuState::Offline);
    /// ```
    #[inline]
    pub fn state(&self, cpu: CpuId) -> CpuState {
        let raw = self.states[cpu.as_usize()].load(Ordering::Acquire);
        Self::raw_to_state(raw)
    }

    /// Set a CPU's state
    ///
    /// # Arguments
    ///
    /// * `cpu` - CPU ID to modify
    /// * `state` - New state
    #[inline]
    fn set_state(&self, cpu: CpuId, state: CpuState) {
        self.states[cpu.as_usize()].store(state as u8, Ordering::Release);
    }

    /// Boot a CPU (transition Offline -> Booting -> Online)
    ///
    /// This should be called by each CPU during its boot sequence.
    /// It atomically transitions the state and updates the online count.
    ///
    /// # Arguments
    ///
    /// * `cpu` - CPU ID that is booting
    ///
    /// # Returns
    ///
    /// `true` if the CPU was successfully brought online, `false` if
    /// the CPU was already online or in an invalid state.
    ///
    /// # Example
    ///
    /// ```
    /// use ruvix_smp::{CpuTopology, CpuId, CpuState};
    ///
    /// static TOPOLOGY: CpuTopology = CpuTopology::new();
    /// TOPOLOGY.init(4);
    ///
    /// assert!(TOPOLOGY.boot_cpu(CpuId::BOOT_CPU));
    /// assert_eq!(TOPOLOGY.state(CpuId::BOOT_CPU), CpuState::Online);
    /// assert_eq!(TOPOLOGY.online_count(), 1);
    /// ```
    pub fn boot_cpu(&self, cpu: CpuId) -> bool {
        let idx = cpu.as_usize();

        // Try to transition Offline -> Booting
        let result = self.states[idx].compare_exchange(
            CpuState::Offline as u8,
            CpuState::Booting as u8,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );

        if result.is_err() {
            // Already booting or online
            return false;
        }

        // Record in boot order
        let order_idx = self.boot_index.fetch_add(1, Ordering::Relaxed);
        if order_idx < MAX_CPUS {
            self.boot_order[order_idx].store(cpu.as_u8(), Ordering::Release);
        }

        // Transition Booting -> Online
        self.set_state(cpu, CpuState::Online);
        self.online_count.fetch_add(1, Ordering::AcqRel);

        // Memory barrier to ensure state is visible
        unsafe {
            barriers::dsb();
            barriers::sev(); // Wake any waiting CPUs
        }

        true
    }

    /// Halt a CPU (transition Online -> Halted)
    ///
    /// The CPU will remain halted until resumed.
    ///
    /// # Arguments
    ///
    /// * `cpu` - CPU ID to halt
    ///
    /// # Returns
    ///
    /// `true` if the CPU was halted, `false` if not online.
    pub fn halt_cpu(&self, cpu: CpuId) -> bool {
        let idx = cpu.as_usize();

        let result = self.states[idx].compare_exchange(
            CpuState::Online as u8,
            CpuState::Halted as u8,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );

        if result.is_ok() {
            self.online_count.fetch_sub(1, Ordering::AcqRel);
            true
        } else {
            false
        }
    }

    /// Resume a halted CPU (transition Halted -> Online)
    ///
    /// # Arguments
    ///
    /// * `cpu` - CPU ID to resume
    ///
    /// # Returns
    ///
    /// `true` if the CPU was resumed, `false` if not halted.
    pub fn resume_cpu(&self, cpu: CpuId) -> bool {
        let idx = cpu.as_usize();

        let result = self.states[idx].compare_exchange(
            CpuState::Halted as u8,
            CpuState::Online as u8,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );

        if result.is_ok() {
            self.online_count.fetch_add(1, Ordering::AcqRel);
            unsafe {
                barriers::sev();
            }
            true
        } else {
            false
        }
    }

    /// Check if a CPU is online
    #[inline]
    pub fn is_online(&self, cpu: CpuId) -> bool {
        self.state(cpu) == CpuState::Online
    }

    /// Check if a CPU can receive IPIs
    #[inline]
    pub fn can_receive_ipi(&self, cpu: CpuId) -> bool {
        self.state(cpu).can_receive_ipi()
    }

    /// Wait for all CPUs to come online
    ///
    /// Spins until all CPUs (as specified by `init()`) are online.
    /// Uses WFE for power efficiency.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_smp::CpuTopology;
    ///
    /// static TOPOLOGY: CpuTopology = CpuTopology::new();
    ///
    /// fn kernel_main() {
    ///     TOPOLOGY.init(4);
    ///     TOPOLOGY.boot_cpu(CpuId::BOOT_CPU);
    ///     // Start secondaries...
    ///     TOPOLOGY.wait_for_all_cpus();
    ///     println!("All CPUs online!");
    /// }
    /// ```
    pub fn wait_for_all_cpus(&self) {
        let expected = self.cpu_count();
        if expected == 0 {
            return;
        }

        unsafe {
            barriers::sevl();
        }

        while self.online_count() < expected {
            unsafe {
                barriers::wfe();
            }
        }
    }

    /// Wait for a specific CPU to come online
    ///
    /// # Arguments
    ///
    /// * `cpu` - CPU ID to wait for
    pub fn wait_for_cpu(&self, cpu: CpuId) {
        unsafe {
            barriers::sevl();
        }

        while !self.is_online(cpu) {
            unsafe {
                barriers::wfe();
            }
        }
    }

    /// Get the boot order of CPUs
    ///
    /// Returns an iterator over CPU IDs in the order they booted.
    pub fn boot_order(&self) -> impl Iterator<Item = CpuId> + '_ {
        let count = self.boot_index.load(Ordering::Acquire).min(MAX_CPUS);
        (0..count).filter_map(move |i| {
            let id = self.boot_order[i].load(Ordering::Acquire);
            CpuId::new(id)
        })
    }

    /// Get all online CPUs
    ///
    /// Returns an iterator over all currently online CPU IDs.
    pub fn online_cpus(&self) -> impl Iterator<Item = CpuId> + '_ {
        let count = self.cpu_count();
        (0..count).filter_map(move |i| {
            let cpu = CpuId::new(i as u8)?;
            if self.is_online(cpu) {
                Some(cpu)
            } else {
                None
            }
        })
    }

    /// Convert raw u8 to CpuState
    #[inline]
    const fn raw_to_state(raw: u8) -> CpuState {
        match raw {
            0 => CpuState::Offline,
            1 => CpuState::Booting,
            2 => CpuState::Online,
            3 => CpuState::Halted,
            _ => CpuState::Offline, // Invalid -> treat as offline
        }
    }
}

impl Default for CpuTopology {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_new() {
        let topology = CpuTopology::new();
        assert_eq!(topology.cpu_count(), 0);
        assert_eq!(topology.online_count(), 0);
    }

    #[test]
    fn test_topology_init() {
        let topology = CpuTopology::new();
        topology.init(4);
        assert_eq!(topology.cpu_count(), 4);
    }

    #[test]
    fn test_topology_boot_cpu() {
        let topology = CpuTopology::new();
        topology.init(4);

        let cpu0 = CpuId::BOOT_CPU;
        assert_eq!(topology.state(cpu0), CpuState::Offline);

        assert!(topology.boot_cpu(cpu0));
        assert_eq!(topology.state(cpu0), CpuState::Online);
        assert_eq!(topology.online_count(), 1);

        // Double boot should fail
        assert!(!topology.boot_cpu(cpu0));
        assert_eq!(topology.online_count(), 1);
    }

    #[test]
    fn test_topology_halt_resume() {
        let topology = CpuTopology::new();
        topology.init(4);

        let cpu0 = CpuId::BOOT_CPU;
        topology.boot_cpu(cpu0);

        assert!(topology.halt_cpu(cpu0));
        assert_eq!(topology.state(cpu0), CpuState::Halted);
        assert_eq!(topology.online_count(), 0);

        assert!(topology.resume_cpu(cpu0));
        assert_eq!(topology.state(cpu0), CpuState::Online);
        assert_eq!(topology.online_count(), 1);
    }

    #[test]
    fn test_topology_multiple_cpus() {
        let topology = CpuTopology::new();
        topology.init(4);

        for i in 0..4 {
            let cpu = CpuId::new(i).unwrap();
            assert!(topology.boot_cpu(cpu));
        }

        assert_eq!(topology.online_count(), 4);

        for i in 0..4 {
            let cpu = CpuId::new(i).unwrap();
            assert!(topology.is_online(cpu));
        }
    }

    #[test]
    fn test_topology_boot_order() {
        let topology = CpuTopology::new();
        topology.init(4);

        // Boot in reverse order
        for i in (0..4).rev() {
            let cpu = CpuId::new(i).unwrap();
            topology.boot_cpu(cpu);
        }

        let mut boot_order = [CpuId::BOOT_CPU; 4];
        let mut count = 0;
        for cpu in topology.boot_order() {
            if count < 4 {
                boot_order[count] = cpu;
                count += 1;
            }
        }
        assert_eq!(count, 4);

        // First booted was CPU 3
        assert_eq!(boot_order[0], CpuId::new(3).unwrap());
        // Last booted was CPU 0
        assert_eq!(boot_order[3], CpuId::BOOT_CPU);
    }

    #[test]
    fn test_topology_online_cpus() {
        let topology = CpuTopology::new();
        topology.init(4);

        // Boot only even CPUs
        topology.boot_cpu(CpuId::new(0).unwrap());
        topology.boot_cpu(CpuId::new(2).unwrap());

        let count = topology.online_cpus().count();
        assert_eq!(count, 2);

        let mut has_cpu0 = false;
        let mut has_cpu2 = false;
        for cpu in topology.online_cpus() {
            if cpu == CpuId::new(0).unwrap() {
                has_cpu0 = true;
            }
            if cpu == CpuId::new(2).unwrap() {
                has_cpu2 = true;
            }
        }
        assert!(has_cpu0);
        assert!(has_cpu2);
    }

    #[test]
    fn test_topology_can_receive_ipi() {
        let topology = CpuTopology::new();
        topology.init(4);

        let cpu0 = CpuId::BOOT_CPU;

        // Offline CPU cannot receive IPI
        assert!(!topology.can_receive_ipi(cpu0));

        topology.boot_cpu(cpu0);
        // Online CPU can receive IPI
        assert!(topology.can_receive_ipi(cpu0));

        topology.halt_cpu(cpu0);
        // Halted CPU can receive IPI (to wake it)
        assert!(topology.can_receive_ipi(cpu0));
    }
}
