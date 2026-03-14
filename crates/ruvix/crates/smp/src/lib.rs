//! # RuVix Symmetric Multi-Processing (SMP) Support
//!
//! This crate provides symmetric multi-processing primitives for the RuVix
//! Cognition Kernel as specified in ADR-087 Phase C. It supports up to 256 CPUs
//! with efficient per-CPU data structures and synchronization primitives.
//!
//! ## Core Components
//!
//! | Component | Purpose |
//! |-----------|---------|
//! | [`CpuId`] | Newtype wrapper for CPU identifiers (0-255) |
//! | [`CpuState`] | State machine for CPU lifecycle |
//! | [`PerCpu<T>`] | Per-CPU data storage indexed by CPU ID |
//! | [`CpuTopology`] | System-wide CPU state tracking |
//! | [`SpinLock<T>`] | Ticket-based spinlock with fairness guarantees |
//! | [`IpiMessage`] | Inter-processor interrupt message types |
//!
//! ## Memory Barriers
//!
//! The [`barriers`] module provides ARM64 memory barrier operations:
//!
//! - `dmb()` - Data Memory Barrier
//! - `dsb()` - Data Synchronization Barrier
//! - `isb()` - Instruction Synchronization Barrier
//! - `sev()` - Send Event (wake waiting CPUs)
//! - `wfe()` - Wait For Event (low-power wait)
//!
//! ## Multi-Core Boot Sequence
//!
//! 1. Primary CPU (CPU 0) starts in `kernel_main()`
//! 2. Primary initializes [`CpuTopology`] and marks itself as `Online`
//! 3. Secondary CPUs are released from spin-table or PSCI
//! 4. Each secondary CPU:
//!    a. Reads its CPU ID from MPIDR_EL1
//!    b. Initializes per-CPU data
//!    c. Transitions: `Offline` -> `Booting` -> `Online`
//! 5. Primary waits for all secondaries using WFE/SEV
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_smp::{CpuTopology, SpinLock, current_cpu, cpu_count};
//!
//! static TOPOLOGY: CpuTopology = CpuTopology::new();
//! static SHARED_DATA: SpinLock<u64> = SpinLock::new(0);
//!
//! fn kernel_main() {
//!     // Initialize primary CPU
//!     let cpu = current_cpu();
//!     TOPOLOGY.boot_cpu(cpu);
//!
//!     // Access shared data
//!     {
//!         let mut guard = SHARED_DATA.lock();
//!         *guard += 1;
//!     } // Lock released here
//!
//!     // Boot secondary CPUs
//!     for i in 1..cpu_count() {
//!         boot_secondary(CpuId::new(i as u8).unwrap());
//!     }
//! }
//! ```
//!
//! ## Safety
//!
//! Most unsafe code is isolated to:
//! - Memory barrier inline assembly ([`barriers`] module)
//! - SpinLock internal atomics ([`SpinLock`])
//! - MPIDR_EL1 register access ([`current_cpu`])
//!
//! ## Features
//!
//! - `std`: Enable standard library support
//! - `alloc`: Enable alloc crate support
//! - `aarch64`: Enable ARM64-specific inline assembly
//! - `test-mode`: Use atomics instead of inline assembly for testing

#![no_std]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod barriers;
mod cpu;
mod ipi;
mod percpu;
mod spinlock;
mod topology;

pub use cpu::{cpu_count, cpu_online, current_cpu, CpuId, CpuState, MAX_CPUS};
pub use ipi::{send_ipi, IpiMessage, IpiTarget};
pub use percpu::PerCpu;
pub use spinlock::{SpinLock, SpinLockGuard};
pub use topology::CpuTopology;

/// SMP version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{
        cpu_count, cpu_online, current_cpu, CpuId, CpuState, CpuTopology, IpiMessage, PerCpu,
        SpinLock, SpinLockGuard,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
