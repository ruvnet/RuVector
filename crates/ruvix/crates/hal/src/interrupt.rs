//! # Interrupt Controller Abstraction
//!
//! Provides traits for managing hardware interrupts (IRQ/FIQ) on ARM64
//! and other platforms.
//!
//! ## Design
//!
//! - **Priority-based routing** - Higher priority interrupts preempt lower
//! - **Edge and level triggering** - Configurable per-IRQ
//! - **IRQ-safe operations** - All methods callable from interrupt context
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_hal::{InterruptController, interrupt::{TriggerMode, InterruptError}};
//!
//! fn setup_timer<I: InterruptController>(irq: &mut I) -> Result<(), InterruptError> {
//!     const TIMER_IRQ: u32 = 30;
//!
//!     irq.set_priority(TIMER_IRQ, 1)?;
//!     irq.set_trigger_mode(TIMER_IRQ, TriggerMode::Edge)?;
//!     irq.enable(TIMER_IRQ)?;
//!
//!     Ok(())
//! }
//! ```

/// Interrupt error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptError {
    /// Invalid interrupt number
    InvalidIrq,
    /// Invalid priority level
    InvalidPriority,
    /// Hardware is not initialized
    NotInitialized,
    /// Operation not supported by hardware
    Unsupported,
}

impl core::fmt::Display for InterruptError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidIrq => write!(f, "invalid IRQ number"),
            Self::InvalidPriority => write!(f, "invalid priority level"),
            Self::NotInitialized => write!(f, "interrupt controller not initialized"),
            Self::Unsupported => write!(f, "operation not supported"),
        }
    }
}

/// Interrupt trigger mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerMode {
    /// Edge-triggered (pulse)
    Edge,
    /// Level-triggered (held high/low)
    Level,
}

/// Interrupt type (ARM64 specific)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptType {
    /// Fast interrupt (FIQ) - higher priority
    Fiq,
    /// Normal interrupt (IRQ)
    Irq,
}

/// Interrupt Controller abstraction
///
/// This trait provides IRQ/FIQ management for ARM64 GICv3 and similar
/// interrupt controllers.
///
/// ## Thread Safety
///
/// All methods MUST be IRQ-safe (callable from interrupt context).
///
/// ## Example Implementation (GICv3)
///
/// ```rust,ignore
/// use ruvix_hal::{InterruptController, interrupt::*};
///
/// struct Gicv3 {
///     gicd_base: usize, // Distributor
///     gicr_base: usize, // Redistributor
///     gicc_base: usize, // CPU interface
/// }
///
/// impl InterruptController for Gicv3 {
///     fn enable(&mut self, irq: u32) -> Result<(), InterruptError> {
///         if irq >= 1024 {
///             return Err(InterruptError::InvalidIrq);
///         }
///
///         // Set enable bit in GICD_ISENABLERn
///         let reg_offset = (irq / 32) * 4;
///         let bit_offset = irq % 32;
///         unsafe {
///             let reg_addr = (self.gicd_base + 0x100 + reg_offset) as *mut u32;
///             let value = core::ptr::read_volatile(reg_addr);
///             core::ptr::write_volatile(reg_addr, value | (1 << bit_offset));
///         }
///
///         Ok(())
///     }
///
///     fn disable(&mut self, irq: u32) -> Result<(), InterruptError> {
///         if irq >= 1024 {
///             return Err(InterruptError::InvalidIrq);
///         }
///
///         // Clear enable bit in GICD_ICENABLERn
///         let reg_offset = (irq / 32) * 4;
///         let bit_offset = irq % 32;
///         unsafe {
///             let reg_addr = (self.gicd_base + 0x180 + reg_offset) as *mut u32;
///             core::ptr::write_volatile(reg_addr, 1 << bit_offset);
///         }
///
///         Ok(())
///     }
///
///     fn acknowledge(&mut self) -> Option<u32> {
///         // Read GICC_IAR to get pending IRQ
///         unsafe {
///             let iar = core::ptr::read_volatile((self.gicc_base + 0x0C) as *const u32);
///             let irq = iar & 0x3FF;
///             if irq < 1020 {
///                 Some(irq)
///             } else {
///                 None // Spurious interrupt
///             }
///         }
///     }
///
///     fn end_of_interrupt(&mut self, irq: u32) -> Result<(), InterruptError> {
///         if irq >= 1024 {
///             return Err(InterruptError::InvalidIrq);
///         }
///
///         // Write GICC_EOIR to signal completion
///         unsafe {
///             core::ptr::write_volatile((self.gicc_base + 0x10) as *mut u32, irq);
///         }
///
///         Ok(())
///     }
///
///     fn set_priority(&mut self, irq: u32, priority: u8) -> Result<(), InterruptError> {
///         if irq >= 1024 {
///             return Err(InterruptError::InvalidIrq);
///         }
///
///         // Write GICD_IPRIORITYRn (8 bits per IRQ)
///         let reg_offset = irq;
///         unsafe {
///             let reg_addr = (self.gicd_base + 0x400 + reg_offset) as *mut u8;
///             core::ptr::write_volatile(reg_addr, priority);
///         }
///
///         Ok(())
///     }
///
///     fn set_trigger_mode(&mut self, irq: u32, mode: TriggerMode) -> Result<(), InterruptError> {
///         if irq >= 1024 {
///             return Err(InterruptError::InvalidIrq);
///         }
///
///         // Configure GICD_ICFGRn (2 bits per IRQ)
///         let reg_offset = (irq / 16) * 4;
///         let bit_offset = (irq % 16) * 2 + 1;
///         unsafe {
///             let reg_addr = (self.gicd_base + 0xC00 + reg_offset) as *mut u32;
///             let mut value = core::ptr::read_volatile(reg_addr);
///             match mode {
///                 TriggerMode::Edge => value |= 1 << bit_offset,
///                 TriggerMode::Level => value &= !(1 << bit_offset),
///             }
///             core::ptr::write_volatile(reg_addr, value);
///         }
///
///         Ok(())
///     }
///
///     fn is_pending(&self, irq: u32) -> Result<bool, InterruptError> {
///         if irq >= 1024 {
///             return Err(InterruptError::InvalidIrq);
///         }
///
///         // Read GICD_ISPENDRn
///         let reg_offset = (irq / 32) * 4;
///         let bit_offset = irq % 32;
///         unsafe {
///             let reg_addr = (self.gicd_base + 0x200 + reg_offset) as *const u32;
///             let value = core::ptr::read_volatile(reg_addr);
///             Ok((value & (1 << bit_offset)) != 0)
///         }
///     }
/// }
/// ```
pub trait InterruptController {
    /// Enable an interrupt
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number (0-1023 for GICv3)
    ///
    /// # Errors
    ///
    /// Returns `InvalidIrq` if IRQ number is out of range.
    fn enable(&mut self, irq: u32) -> Result<(), InterruptError>;

    /// Disable an interrupt
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number
    ///
    /// # Errors
    ///
    /// Returns `InvalidIrq` if IRQ number is out of range.
    fn disable(&mut self, irq: u32) -> Result<(), InterruptError>;

    /// Acknowledge and return pending interrupt
    ///
    /// This should be called at the start of the IRQ handler.
    /// Returns `None` if no interrupt is pending.
    fn acknowledge(&mut self) -> Option<u32>;

    /// Signal end of interrupt handling
    ///
    /// This MUST be called at the end of the IRQ handler to allow
    /// lower-priority interrupts to fire.
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number being completed
    ///
    /// # Errors
    ///
    /// Returns `InvalidIrq` if IRQ number is out of range.
    fn end_of_interrupt(&mut self, irq: u32) -> Result<(), InterruptError>;

    /// Set interrupt priority
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number
    /// * `priority` - Priority level (0 = highest, 255 = lowest)
    ///
    /// # Errors
    ///
    /// Returns `InvalidIrq` or `InvalidPriority` on invalid inputs.
    fn set_priority(&mut self, irq: u32, priority: u8) -> Result<(), InterruptError>;

    /// Set interrupt trigger mode
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number
    /// * `mode` - Edge or Level triggered
    ///
    /// # Errors
    ///
    /// Returns `InvalidIrq` if IRQ number is out of range.
    /// Returns `Unsupported` if hardware doesn't support configuration.
    fn set_trigger_mode(&mut self, irq: u32, mode: TriggerMode) -> Result<(), InterruptError>;

    /// Check if interrupt is pending
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number to check
    ///
    /// # Errors
    ///
    /// Returns `InvalidIrq` if IRQ number is out of range.
    fn is_pending(&self, irq: u32) -> Result<bool, InterruptError>;

    /// Get current CPU interrupt mask state
    ///
    /// Returns `true` if interrupts are globally masked.
    fn are_interrupts_masked(&self) -> bool {
        false // Default: assume enabled
    }

    /// Set interrupt affinity (route to specific CPU)
    ///
    /// Optional method for multi-core systems.
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number
    /// * `cpu_mask` - Bitmask of target CPUs (bit N = CPU N)
    ///
    /// # Errors
    ///
    /// Returns `Unsupported` if single-core or not implemented.
    #[allow(unused_variables)]
    fn set_affinity(&mut self, irq: u32, cpu_mask: u32) -> Result<(), InterruptError> {
        Err(InterruptError::Unsupported)
    }
}

/// Helper for RAII interrupt masking
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_hal::interrupt::InterruptGuard;
///
/// {
///     let _guard = InterruptGuard::new();
///     // Interrupts masked here
///     critical_section();
/// } // Interrupts restored on drop
/// ```
pub struct InterruptGuard {
    previous_state: bool,
}

impl InterruptGuard {
    /// Create guard and disable interrupts
    ///
    /// This is safe because we restore on drop.
    #[inline]
    pub fn new() -> Self {
        Self {
            previous_state: Self::disable_interrupts(),
        }
    }

    /// Disable interrupts and return previous state
    #[inline]
    fn disable_interrupts() -> bool {
        // Platform-specific implementation would go here
        // For now, return false (interrupts were enabled)
        false
    }

    /// Restore interrupts to previous state
    #[inline]
    fn restore_interrupts(enabled: bool) {
        // Platform-specific implementation would go here
        let _ = enabled;
    }
}

impl Drop for InterruptGuard {
    fn drop(&mut self) {
        Self::restore_interrupts(self.previous_state);
    }
}

impl Default for InterruptGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockIrqController {
        enabled: [bool; 32],
        priorities: [u8; 32],
        pending: Option<u32>,
    }

    impl MockIrqController {
        fn new() -> Self {
            Self {
                enabled: [false; 32],
                priorities: [128; 32],
                pending: None,
            }
        }
    }

    impl InterruptController for MockIrqController {
        fn enable(&mut self, irq: u32) -> Result<(), InterruptError> {
            if irq >= 32 {
                return Err(InterruptError::InvalidIrq);
            }
            self.enabled[irq as usize] = true;
            Ok(())
        }

        fn disable(&mut self, irq: u32) -> Result<(), InterruptError> {
            if irq >= 32 {
                return Err(InterruptError::InvalidIrq);
            }
            self.enabled[irq as usize] = false;
            Ok(())
        }

        fn acknowledge(&mut self) -> Option<u32> {
            self.pending.take()
        }

        fn end_of_interrupt(&mut self, irq: u32) -> Result<(), InterruptError> {
            if irq >= 32 {
                return Err(InterruptError::InvalidIrq);
            }
            Ok(())
        }

        fn set_priority(&mut self, irq: u32, priority: u8) -> Result<(), InterruptError> {
            if irq >= 32 {
                return Err(InterruptError::InvalidIrq);
            }
            self.priorities[irq as usize] = priority;
            Ok(())
        }

        fn set_trigger_mode(
            &mut self,
            _irq: u32,
            _mode: TriggerMode,
        ) -> Result<(), InterruptError> {
            Ok(())
        }

        fn is_pending(&self, irq: u32) -> Result<bool, InterruptError> {
            if irq >= 32 {
                return Err(InterruptError::InvalidIrq);
            }
            Ok(self.pending == Some(irq))
        }
    }

    #[test]
    fn test_enable_disable() {
        let mut irq = MockIrqController::new();

        irq.enable(5).unwrap();
        assert!(irq.enabled[5]);

        irq.disable(5).unwrap();
        assert!(!irq.enabled[5]);
    }

    #[test]
    fn test_invalid_irq() {
        let mut irq = MockIrqController::new();
        assert_eq!(irq.enable(100), Err(InterruptError::InvalidIrq));
    }

    #[test]
    fn test_priority() {
        let mut irq = MockIrqController::new();
        irq.set_priority(10, 42).unwrap();
        assert_eq!(irq.priorities[10], 42);
    }
}
