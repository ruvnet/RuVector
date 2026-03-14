//! ARM64 Memory Barriers and CPU Synchronization
//!
//! This module provides memory barrier operations essential for correct
//! multi-core synchronization on ARM64 processors.
//!
//! ## Barrier Types
//!
//! | Barrier | Purpose | Use Case |
//! |---------|---------|----------|
//! | `dmb()` | Data Memory Barrier | Ordering memory accesses |
//! | `dsb()` | Data Synchronization Barrier | Completing memory operations |
//! | `isb()` | Instruction Synchronization Barrier | Pipeline flush |
//! | `sev()` | Send Event | Wake waiting CPUs |
//! | `wfe()` | Wait For Event | Low-power wait |
//! | `wfi()` | Wait For Interrupt | Idle loop |
//!
//! ## Memory Ordering
//!
//! ARM64 is weakly ordered, meaning loads and stores can be reordered
//! unless barriers are used. The barriers ensure ordering:
//!
//! ```text
//! CPU 0                    CPU 1
//! -----                    -----
//! store(data, 42)
//! dmb()                    dmb()
//! store(flag, 1)           load(flag) == 1
//!                          dmb()
//!                          load(data) == 42  // guaranteed!
//! ```
//!
//! ## WFE/SEV Protocol
//!
//! Wait-for-event (WFE) and send-event (SEV) provide efficient
//! spinlock implementations:
//!
//! ```text
//! Lock holder:             Lock waiter:
//! -----------              -----------
//!                          loop {
//!                              if try_acquire() { break; }
//!                              wfe()  // sleep until event
//!                          }
//! release_lock()
//! sev()  // wake waiters
//! ```
//!
//! ## Safety
//!
//! All barrier functions use inline assembly and are marked `unsafe`
//! because they can affect system behavior. However, executing barriers
//! themselves does not cause memory unsafety - they only affect ordering
//! and timing.

/// Data Memory Barrier (DMB)
///
/// Ensures that all explicit memory accesses before the DMB are observed
/// before any explicit memory accesses after the DMB.
///
/// # Domain
///
/// Uses `SY` (system) domain - affects all observers including other cores
/// and external memory.
///
/// # Usage
///
/// Use DMB when you need to order memory accesses but don't need to wait
/// for them to complete (use DSB for that).
///
/// # Safety
///
/// Executing DMB is always safe from a memory safety perspective, but
/// incorrect use can lead to subtle synchronization bugs.
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_smp::barriers::dmb;
///
/// // Ensure data is visible before flag
/// data.store(value, Ordering::Relaxed);
/// unsafe { dmb(); }
/// flag.store(true, Ordering::Relaxed);
/// ```
#[inline]
pub unsafe fn dmb() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: DMB is always safe to execute
        core::arch::asm!("dmb sy", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        // Use compiler fence in test mode
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Data Memory Barrier - Inner Shareable (DMB ISH)
///
/// Like `dmb()` but only affects inner shareable domain (other cores).
/// More efficient than full system barrier when external devices are
/// not involved.
///
/// # Safety
///
/// Same safety considerations as [`dmb()`].
#[inline]
pub unsafe fn dmb_ish() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: DMB ISH is always safe to execute
        core::arch::asm!("dmb ish", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Data Memory Barrier - Stores Only (DMB ST)
///
/// Only orders stores with respect to other stores. Loads can still
/// pass stores. Useful for write-only synchronization.
///
/// # Safety
///
/// Same safety considerations as [`dmb()`].
#[inline]
pub unsafe fn dmb_st() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: DMB ST is always safe to execute
        core::arch::asm!("dmb st", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Release);
    }
}

/// Data Synchronization Barrier (DSB)
///
/// Ensures that all memory accesses before the DSB have completed before
/// any instruction after the DSB executes. Stronger than DMB.
///
/// # Domain
///
/// Uses `SY` (system) domain.
///
/// # Usage
///
/// Use DSB when you need to ensure memory operations are complete, such as:
/// - Before modifying page tables
/// - Before enabling/disabling caches
/// - Before executing SEV to ensure stores are visible
///
/// # Safety
///
/// Executing DSB is always safe but has performance implications.
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_smp::barriers::{dsb, isb};
///
/// // Modify page table
/// page_table[index] = new_entry;
/// unsafe {
///     dsb();  // Ensure PTE write completes
///     isb();  // Flush pipeline
/// }
/// // TLB invalidation here
/// ```
#[inline]
pub unsafe fn dsb() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: DSB is always safe to execute
        core::arch::asm!("dsb sy", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Data Synchronization Barrier - Inner Shareable (DSB ISH)
///
/// Like `dsb()` but only affects inner shareable domain.
///
/// # Safety
///
/// Same safety considerations as [`dsb()`].
#[inline]
pub unsafe fn dsb_ish() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: DSB ISH is always safe to execute
        core::arch::asm!("dsb ish", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Instruction Synchronization Barrier (ISB)
///
/// Flushes the processor pipeline, ensuring all instructions after the ISB
/// are fetched from cache or memory after the ISB completes.
///
/// # Usage
///
/// Required after:
/// - Modifying instruction memory (self-modifying code)
/// - Modifying page tables
/// - Changing system registers that affect instruction execution
/// - Any operation that affects the instruction stream
///
/// # Safety
///
/// Executing ISB is always safe.
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_smp::barriers::isb;
///
/// // After writing VBAR_EL1
/// unsafe {
///     core::arch::asm!("msr vbar_el1, {}", in(reg) vector_table);
///     isb();  // Ensure new vector table is used
/// }
/// ```
#[inline]
pub unsafe fn isb() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: ISB is always safe to execute
        core::arch::asm!("isb", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        // No direct equivalent - use compiler fence
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// Send Event (SEV)
///
/// Signals an event to all CPUs in the system. This wakes any CPUs
/// that are waiting in a WFE instruction.
///
/// # Usage
///
/// Use after releasing a lock or signaling that shared data is ready:
///
/// ```rust,ignore
/// use ruvix_smp::barriers::{dsb, sev};
///
/// // Release lock
/// lock.store(0, Ordering::Release);
/// unsafe {
///     dsb();  // Ensure release is visible
///     sev();  // Wake waiting CPUs
/// }
/// ```
///
/// # Safety
///
/// Executing SEV is always safe. Spurious wakeups are handled by
/// the WFE protocol.
#[inline]
pub unsafe fn sev() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: SEV is always safe to execute
        core::arch::asm!("sev", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        // No equivalent needed in test mode
    }
}

/// Send Event - Local (SEVL)
///
/// Like SEV but only signals the local CPU. Useful for setting up
/// the event flag before entering a WFE loop.
///
/// # Safety
///
/// Executing SEVL is always safe.
#[inline]
pub unsafe fn sevl() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: SEVL is always safe to execute
        core::arch::asm!("sevl", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        // No equivalent needed in test mode
    }
}

/// Wait For Event (WFE)
///
/// Puts the CPU into a low-power state until an event occurs.
/// Events include:
/// - SEV from another CPU
/// - Interrupts (if not masked)
/// - External debug events
///
/// # Usage
///
/// Use in spinlock acquisition loops to reduce power consumption
/// and cache traffic:
///
/// ```rust,ignore
/// use ruvix_smp::barriers::{wfe, sevl};
///
/// unsafe {
///     sevl();  // Set local event flag
///     while !try_acquire_lock() {
///         wfe();  // Sleep until event
///     }
/// }
/// ```
///
/// # Safety
///
/// WFE may block indefinitely if no event occurs. Caller must ensure
/// the WFE will eventually be woken by a corresponding SEV.
#[inline]
pub unsafe fn wfe() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: WFE is always safe to execute (may block)
        core::arch::asm!("wfe", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        // Yield in test mode to prevent busy-wait
        core::hint::spin_loop();
    }
}

/// Wait For Interrupt (WFI)
///
/// Puts the CPU into a low-power state until an interrupt occurs.
/// Unlike WFE, only interrupts wake the CPU (not SEV).
///
/// # Usage
///
/// Use in the idle loop when no work is available:
///
/// ```rust,ignore
/// use ruvix_smp::barriers::wfi;
///
/// fn idle_loop() -> ! {
///     loop {
///         // Check for work
///         if let Some(task) = get_task() {
///             run_task(task);
///         } else {
///             unsafe { wfi(); }  // Sleep until interrupt
///         }
///     }
/// }
/// ```
///
/// # Safety
///
/// WFI may block indefinitely if interrupts are disabled. Caller
/// must ensure interrupts will eventually wake the CPU.
#[inline]
pub unsafe fn wfi() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: WFI is always safe to execute (may block)
        core::arch::asm!("wfi", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        // Yield in test mode
        core::hint::spin_loop();
    }
}

/// Yield hint
///
/// Provides a hint that the current thread is spinning. On ARM64,
/// this is implemented as YIELD which may improve performance on
/// SMT (Simultaneous Multi-Threading) cores.
///
/// # Safety
///
/// Always safe to execute.
#[inline]
pub unsafe fn cpu_yield() {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        // SAFETY: YIELD is always safe to execute
        core::arch::asm!("yield", options(nostack, nomem, preserves_flags));
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        core::hint::spin_loop();
    }
}

/// Combined barrier sequence for lock release
///
/// Executes DSB + SEV in sequence, which is the standard pattern
/// for releasing a lock and waking waiters.
///
/// # Safety
///
/// Same safety considerations as [`dsb()`] and [`sev()`].
#[inline]
pub unsafe fn release_barrier() {
    dsb();
    sev();
}

/// Combined barrier sequence for memory/cache operations
///
/// Executes DSB + ISB in sequence, which is required after
/// modifying page tables or cache control registers.
///
/// # Safety
///
/// Same safety considerations as [`dsb()`] and [`isb()`].
#[inline]
pub unsafe fn sync_barrier() {
    dsb();
    isb();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barriers_compile() {
        // Just verify the functions compile and can be called
        unsafe {
            dmb();
            dmb_ish();
            dmb_st();
            dsb();
            dsb_ish();
            isb();
            sev();
            sevl();
            cpu_yield();
            release_barrier();
            sync_barrier();
        }
    }

    #[test]
    fn test_wfe_wfi_yield() {
        // These should complete immediately in test mode
        unsafe {
            wfe();
            wfi();
            cpu_yield();
        }
    }
}
