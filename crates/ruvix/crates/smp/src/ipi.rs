//! Inter-Processor Interrupt (IPI) Support
//!
//! This module provides types and functions for sending inter-processor
//! interrupts between CPUs. IPIs are used for:
//!
//! - CPU synchronization (barriers)
//! - TLB shootdowns (page table changes)
//! - Scheduler events (task migration)
//! - Function calls on remote CPUs
//! - System-wide halt/reset
//!
//! ## Architecture
//!
//! On ARM64, IPIs use Software Generated Interrupts (SGIs) via the GIC:
//! - SGI 0-15 are available for software use
//! - We reserve SGI ranges for different message types
//!
//! ## Message Types
//!
//! | SGI | Message | Purpose |
//! |-----|---------|---------|
//! | 0 | Reschedule | Wake idle CPU for scheduling |
//! | 1 | TlbFlush | Invalidate TLB entries |
//! | 2 | FunctionCall | Execute function on target |
//! | 3 | Halt | Stop CPU (for debugging) |
//! | 4 | CallFunction | Generic function call |
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_smp::{send_ipi, IpiMessage, IpiTarget, CpuId};
//!
//! // Send reschedule to CPU 1
//! send_ipi(IpiTarget::Cpu(CpuId::new(1).unwrap()), IpiMessage::Reschedule);
//!
//! // Broadcast TLB flush to all other CPUs
//! send_ipi(IpiTarget::AllOther, IpiMessage::TlbFlush { asid: None });
//!
//! // Halt all CPUs (except self)
//! send_ipi(IpiTarget::AllOther, IpiMessage::Halt);
//! ```

use crate::cpu::CpuId;
use core::fmt;

/// IPI message types
///
/// Each message type corresponds to a specific SGI and handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpiMessage {
    /// Reschedule request
    ///
    /// Sent to wake an idle CPU to check its run queue.
    /// Used when a task is enqueued that could run on the target.
    Reschedule,

    /// TLB flush request
    ///
    /// Sent when page tables are modified and remote TLBs need invalidation.
    ///
    /// * `asid` - Optional ASID to flush (None = flush all)
    TlbFlush {
        /// ASID to flush, or None for global flush
        asid: Option<u16>,
    },

    /// Function call request
    ///
    /// Execute a function on the target CPU. The function must be
    /// stored in a shared location that the target can access.
    ///
    /// * `func_id` - Index into the IPI function table
    FunctionCall {
        /// Function identifier
        func_id: u32,
    },

    /// Halt request
    ///
    /// Stop the target CPU. Used for debugging or shutdown.
    Halt,

    /// Wake from halt
    ///
    /// Resume a halted CPU.
    WakeUp,

    /// Timer synchronization
    ///
    /// Synchronize timer state across CPUs.
    TimerSync,

    /// Generic data message
    ///
    /// Send arbitrary data to the target CPU. Limited to 32 bits.
    Data {
        /// Data payload
        payload: u32,
    },

    /// Performance monitoring request
    ///
    /// Request performance counter snapshot.
    PerfSnapshot,

    /// Debug break
    ///
    /// Enter debugger on target CPU.
    DebugBreak,

    /// Heartbeat
    ///
    /// Health check - target should respond.
    Heartbeat,
}

impl IpiMessage {
    /// Get the SGI number for this message type
    #[inline]
    pub const fn sgi(&self) -> u8 {
        match self {
            IpiMessage::Reschedule => 0,
            IpiMessage::TlbFlush { .. } => 1,
            IpiMessage::FunctionCall { .. } => 2,
            IpiMessage::Halt => 3,
            IpiMessage::WakeUp => 4,
            IpiMessage::TimerSync => 5,
            IpiMessage::Data { .. } => 6,
            IpiMessage::PerfSnapshot => 7,
            IpiMessage::DebugBreak => 8,
            IpiMessage::Heartbeat => 9,
        }
    }

    /// Check if this message requires a response
    #[inline]
    pub const fn requires_ack(&self) -> bool {
        matches!(
            self,
            IpiMessage::TlbFlush { .. }
                | IpiMessage::FunctionCall { .. }
                | IpiMessage::Heartbeat
        )
    }

    /// Get the priority of this message (lower = higher priority)
    #[inline]
    pub const fn priority(&self) -> u8 {
        match self {
            IpiMessage::Halt => 0,        // Highest
            IpiMessage::DebugBreak => 1,
            IpiMessage::TlbFlush { .. } => 2,
            IpiMessage::FunctionCall { .. } => 3,
            IpiMessage::Reschedule => 4,
            IpiMessage::WakeUp => 5,
            IpiMessage::TimerSync => 6,
            IpiMessage::PerfSnapshot => 7,
            IpiMessage::Heartbeat => 8,
            IpiMessage::Data { .. } => 9, // Lowest
        }
    }
}

impl fmt::Display for IpiMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpiMessage::Reschedule => write!(f, "RESCHED"),
            IpiMessage::TlbFlush { asid } => {
                if let Some(a) = asid {
                    write!(f, "TLB_FLUSH(ASID={})", a)
                } else {
                    write!(f, "TLB_FLUSH(ALL)")
                }
            }
            IpiMessage::FunctionCall { func_id } => write!(f, "FUNC_CALL({})", func_id),
            IpiMessage::Halt => write!(f, "HALT"),
            IpiMessage::WakeUp => write!(f, "WAKE"),
            IpiMessage::TimerSync => write!(f, "TIMER_SYNC"),
            IpiMessage::Data { payload } => write!(f, "DATA({:#x})", payload),
            IpiMessage::PerfSnapshot => write!(f, "PERF"),
            IpiMessage::DebugBreak => write!(f, "DEBUG"),
            IpiMessage::Heartbeat => write!(f, "HEARTBEAT"),
        }
    }
}

/// IPI target specification
///
/// Specifies which CPU(s) should receive the IPI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpiTarget {
    /// Send to a specific CPU
    Cpu(CpuId),

    /// Send to all CPUs (including self)
    All,

    /// Send to all other CPUs (excluding self)
    AllOther,

    /// Send to CPUs matching a bitmask
    ///
    /// Bit N set means send to CPU N.
    Mask(u64),
}

impl IpiTarget {
    /// Check if this target includes the specified CPU
    #[inline]
    pub fn includes(&self, cpu: CpuId, self_cpu: CpuId) -> bool {
        match self {
            IpiTarget::Cpu(target) => cpu == *target,
            IpiTarget::All => true,
            IpiTarget::AllOther => cpu != self_cpu,
            IpiTarget::Mask(mask) => (mask & (1u64 << cpu.as_u8())) != 0,
        }
    }
}

impl fmt::Display for IpiTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpiTarget::Cpu(cpu) => write!(f, "{}", cpu),
            IpiTarget::All => write!(f, "ALL"),
            IpiTarget::AllOther => write!(f, "ALL_OTHER"),
            IpiTarget::Mask(mask) => write!(f, "MASK({:#x})", mask),
        }
    }
}

/// IPI send result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpiResult {
    /// IPI sent successfully
    Ok,
    /// Target CPU is offline
    TargetOffline,
    /// Target CPU cannot receive IPIs
    TargetBusy,
    /// Invalid target specification
    InvalidTarget,
    /// IPI queue is full
    QueueFull,
    /// Hardware error
    HardwareError,
}

impl fmt::Display for IpiResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpiResult::Ok => write!(f, "OK"),
            IpiResult::TargetOffline => write!(f, "target offline"),
            IpiResult::TargetBusy => write!(f, "target busy"),
            IpiResult::InvalidTarget => write!(f, "invalid target"),
            IpiResult::QueueFull => write!(f, "queue full"),
            IpiResult::HardwareError => write!(f, "hardware error"),
        }
    }
}

/// Send an IPI to the specified target
///
/// This function sends an inter-processor interrupt to the target CPU(s).
/// On ARM64, this uses Software Generated Interrupts (SGIs) via the GIC.
///
/// # Arguments
///
/// * `target` - Which CPU(s) to send to
/// * `msg` - The IPI message to send
///
/// # Returns
///
/// Result indicating success or failure reason.
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_smp::{send_ipi, IpiMessage, IpiTarget, CpuId};
///
/// // Send reschedule to CPU 1
/// let result = send_ipi(
///     IpiTarget::Cpu(CpuId::new(1).unwrap()),
///     IpiMessage::Reschedule
/// );
///
/// match result {
///     IpiResult::Ok => println!("IPI sent"),
///     IpiResult::TargetOffline => println!("CPU 1 is offline"),
///     _ => println!("IPI failed: {}", result),
/// }
/// ```
///
/// # Platform Notes
///
/// On ARM64 with GICv3:
/// - Uses ICC_SGI1R_EL1 for SGI generation
/// - Target list encoded in affinity fields
/// - SGI number from message type
///
/// In test mode, this is a no-op that always succeeds.
#[inline]
pub fn send_ipi(target: IpiTarget, msg: IpiMessage) -> IpiResult {
    #[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
    {
        send_ipi_gicv3(target, msg)
    }

    #[cfg(any(not(target_arch = "aarch64"), not(feature = "aarch64"), feature = "test-mode"))]
    {
        // In test mode, just log and return success
        let _ = (target, msg);
        IpiResult::Ok
    }
}

/// Send IPI via GICv3 (ARM64 implementation)
#[cfg(all(target_arch = "aarch64", feature = "aarch64", not(feature = "test-mode")))]
fn send_ipi_gicv3(target: IpiTarget, msg: IpiMessage) -> IpiResult {
    use core::arch::asm;

    let sgi = msg.sgi() as u64;

    // Build ICC_SGI1R_EL1 value
    let sgi1r: u64 = match target {
        IpiTarget::Cpu(cpu) => {
            // Single CPU target
            let aff0 = cpu.as_u8() as u64;
            (sgi << 24) | (1u64 << aff0)
        }
        IpiTarget::All => {
            // All CPUs (IRM = 1, then we need self too)
            (sgi << 24) | (1u64 << 40) // IRM bit
        }
        IpiTarget::AllOther => {
            // All except self (IRM = 1)
            (sgi << 24) | (1u64 << 40)
        }
        IpiTarget::Mask(mask) => {
            // Target list (lower 16 bits)
            let target_list = mask & 0xFFFF;
            (sgi << 24) | target_list
        }
    };

    // SAFETY: Writing ICC_SGI1R_EL1 sends an SGI
    unsafe {
        asm!(
            "msr icc_sgi1r_el1, {}",
            in(reg) sgi1r,
            options(nostack, nomem, preserves_flags)
        );
        crate::barriers::isb();
    }

    IpiResult::Ok
}

/// IPI handler registration (for kernel use)
///
/// Each message type should have a registered handler.
#[derive(Debug, Clone, Copy)]
pub struct IpiHandler {
    /// Handler function
    pub handler: fn(IpiMessage, CpuId),
    /// Message types this handler processes
    pub msg_type: IpiMessage,
}

/// IPI statistics for debugging
#[derive(Debug, Default, Clone, Copy)]
pub struct IpiStats {
    /// Total IPIs sent
    pub sent: u64,
    /// Total IPIs received
    pub received: u64,
    /// IPIs that failed to send
    pub send_failures: u64,
    /// IPIs that required acknowledgment
    pub acked: u64,
}

impl IpiStats {
    /// Create new zero-initialized stats
    pub const fn new() -> Self {
        Self {
            sent: 0,
            received: 0,
            send_failures: 0,
            acked: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipi_message_sgi() {
        assert_eq!(IpiMessage::Reschedule.sgi(), 0);
        assert_eq!(IpiMessage::TlbFlush { asid: None }.sgi(), 1);
        assert_eq!(IpiMessage::FunctionCall { func_id: 0 }.sgi(), 2);
        assert_eq!(IpiMessage::Halt.sgi(), 3);
    }

    #[test]
    fn test_ipi_message_priority() {
        // Halt should have highest priority
        assert!(IpiMessage::Halt.priority() < IpiMessage::Reschedule.priority());
        assert!(IpiMessage::DebugBreak.priority() < IpiMessage::TlbFlush { asid: None }.priority());
    }

    #[test]
    fn test_ipi_message_requires_ack() {
        assert!(!IpiMessage::Reschedule.requires_ack());
        assert!(IpiMessage::TlbFlush { asid: None }.requires_ack());
        assert!(IpiMessage::FunctionCall { func_id: 0 }.requires_ack());
        assert!(!IpiMessage::Halt.requires_ack());
        assert!(IpiMessage::Heartbeat.requires_ack());
    }

    #[test]
    fn test_ipi_target_includes() {
        let cpu0 = CpuId::BOOT_CPU;
        let cpu1 = CpuId::new(1).unwrap();
        let cpu2 = CpuId::new(2).unwrap();

        // Specific CPU
        assert!(IpiTarget::Cpu(cpu1).includes(cpu1, cpu0));
        assert!(!IpiTarget::Cpu(cpu1).includes(cpu2, cpu0));

        // All
        assert!(IpiTarget::All.includes(cpu0, cpu0));
        assert!(IpiTarget::All.includes(cpu1, cpu0));

        // All other
        assert!(!IpiTarget::AllOther.includes(cpu0, cpu0));
        assert!(IpiTarget::AllOther.includes(cpu1, cpu0));

        // Mask
        let mask = 0b110; // CPUs 1 and 2
        assert!(!IpiTarget::Mask(mask).includes(cpu0, cpu0));
        assert!(IpiTarget::Mask(mask).includes(cpu1, cpu0));
        assert!(IpiTarget::Mask(mask).includes(cpu2, cpu0));
    }

    #[test]
    fn test_send_ipi_test_mode() {
        // In test mode, should always succeed
        let result = send_ipi(IpiTarget::All, IpiMessage::Reschedule);
        assert_eq!(result, IpiResult::Ok);

        let result = send_ipi(
            IpiTarget::Cpu(CpuId::BOOT_CPU),
            IpiMessage::TlbFlush { asid: Some(42) },
        );
        assert_eq!(result, IpiResult::Ok);
    }

    #[test]
    fn test_ipi_stats() {
        let stats = IpiStats::new();
        assert_eq!(stats.sent, 0);
        assert_eq!(stats.received, 0);
    }
}
