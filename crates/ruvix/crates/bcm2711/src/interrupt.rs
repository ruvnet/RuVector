//! # BCM2711 Interrupt Controller Driver
//!
//! This module implements a driver for the BCM2711 legacy interrupt controller.
//! Unlike QEMU virt which uses a GIC, the Raspberry Pi 4 uses the BCM2711's
//! built-in interrupt controller for backwards compatibility.
//!
//! ## Architecture
//!
//! The BCM2711 has a complex interrupt routing system:
//!
//! 1. **Legacy Interrupt Controller** - Compatible with BCM2835/BCM2836
//! 2. **GIC-400** - ARM Generic Interrupt Controller (optional, for newer code)
//! 3. **ARM Local Interrupt Controller** - Per-CPU interrupts
//!
//! This driver implements the legacy controller for maximum compatibility.
//!
//! ## Interrupt Sources
//!
//! Interrupts are grouped into three banks:
//!
//! - **GPU IRQs 0-31** (Bank 0) - GPU and system peripherals
//! - **GPU IRQs 32-63** (Bank 1) - More peripherals
//! - **ARM IRQs 0-7** (Basic) - ARM-specific interrupts
//!
//! ### Common Interrupt Numbers
//!
//! | IRQ | Source |
//! |-----|--------|
//! | 1 | System Timer Compare 1 |
//! | 3 | System Timer Compare 3 |
//! | 9 | USB Controller |
//! | 29 | Aux (Mini UART, SPI1, SPI2) |
//! | 48 | GPIO Bank 0 |
//! | 49 | GPIO Bank 1 |
//! | 50 | GPIO Bank 2 |
//! | 51 | GPIO Bank 3 |
//! | 52 | I2C |
//! | 53 | SPI |
//! | 54 | PCM Audio |
//! | 55 | SDHOST |
//! | 56 | UART (PL011) |
//! | 57 | EMMC |
//!
//! ## Register Map
//!
//! | Offset | Register | Description |
//! |--------|----------|-------------|
//! | 0x200 | IRQ_BASIC_PENDING | Basic pending register |
//! | 0x204 | IRQ_PENDING_1 | GPU IRQ pending 1 (0-31) |
//! | 0x208 | IRQ_PENDING_2 | GPU IRQ pending 2 (32-63) |
//! | 0x20C | FIQ_CONTROL | FIQ control register |
//! | 0x210 | ENABLE_IRQS_1 | Enable GPU IRQs 0-31 |
//! | 0x214 | ENABLE_IRQS_2 | Enable GPU IRQs 32-63 |
//! | 0x218 | ENABLE_BASIC_IRQS | Enable ARM basic IRQs |
//! | 0x21C | DISABLE_IRQS_1 | Disable GPU IRQs 0-31 |
//! | 0x220 | DISABLE_IRQS_2 | Disable GPU IRQs 32-63 |
//! | 0x224 | DISABLE_BASIC_IRQS | Disable ARM basic IRQs |
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_bcm2711::interrupt::BcmInterruptController;
//!
//! let mut irq = BcmInterruptController::new();
//! irq.init().unwrap();
//!
//! // Enable Mini UART interrupt (IRQ 29)
//! irq.enable(29).unwrap();
//!
//! // Check if interrupt is pending
//! if irq.is_pending(29) {
//!     // Handle interrupt
//!     // ...
//! }
//! ```

use crate::mmio::{dsb, MmioReg};
use crate::INTERRUPT_BASE;

// =============================================================================
// Register Offsets
// =============================================================================

/// IRQ basic pending register.
const IRQ_BASIC_PENDING: usize = 0x200;

/// GPU IRQ pending register 1 (IRQs 0-31).
const IRQ_PENDING_1: usize = 0x204;

/// GPU IRQ pending register 2 (IRQs 32-63).
const IRQ_PENDING_2: usize = 0x208;

/// FIQ control register.
const FIQ_CONTROL: usize = 0x20C;

/// Enable GPU IRQs 0-31.
const ENABLE_IRQS_1: usize = 0x210;

/// Enable GPU IRQs 32-63.
const ENABLE_IRQS_2: usize = 0x214;

/// Enable ARM basic IRQs.
const ENABLE_BASIC_IRQS: usize = 0x218;

/// Disable GPU IRQs 0-31.
const DISABLE_IRQS_1: usize = 0x21C;

/// Disable GPU IRQs 32-63.
const DISABLE_IRQS_2: usize = 0x220;

/// Disable ARM basic IRQs.
const DISABLE_BASIC_IRQS: usize = 0x224;

// =============================================================================
// Basic Pending Register Bits
// =============================================================================

/// ARM Timer interrupt pending.
const BASIC_ARM_TIMER: u32 = 1 << 0;

/// ARM Mailbox interrupt pending.
const BASIC_ARM_MAILBOX: u32 = 1 << 1;

/// ARM Doorbell 0 interrupt pending.
const BASIC_ARM_DOORBELL0: u32 = 1 << 2;

/// ARM Doorbell 1 interrupt pending.
const BASIC_ARM_DOORBELL1: u32 = 1 << 3;

/// GPU 0 halted interrupt.
const BASIC_GPU0_HALTED: u32 = 1 << 4;

/// GPU 1 halted interrupt.
const BASIC_GPU1_HALTED: u32 = 1 << 5;

/// Illegal access type 1 interrupt.
const BASIC_ILLEGAL_ACCESS_1: u32 = 1 << 6;

/// Illegal access type 0 interrupt.
const BASIC_ILLEGAL_ACCESS_0: u32 = 1 << 7;

/// One or more bits set in pending register 1.
const BASIC_PENDING_1: u32 = 1 << 8;

/// One or more bits set in pending register 2.
const BASIC_PENDING_2: u32 = 1 << 9;

// =============================================================================
// IRQ Numbers
// =============================================================================

/// System Timer Match 1.
pub const IRQ_TIMER1: u32 = 1;

/// System Timer Match 3.
pub const IRQ_TIMER3: u32 = 3;

/// USB Controller.
pub const IRQ_USB: u32 = 9;

/// Auxiliary peripherals (Mini UART, SPI1, SPI2).
pub const IRQ_AUX: u32 = 29;

/// GPIO Bank 0.
pub const IRQ_GPIO0: u32 = 48;

/// GPIO Bank 1.
pub const IRQ_GPIO1: u32 = 49;

/// GPIO Bank 2.
pub const IRQ_GPIO2: u32 = 50;

/// GPIO Bank 3.
pub const IRQ_GPIO3: u32 = 51;

/// I2C.
pub const IRQ_I2C: u32 = 52;

/// SPI.
pub const IRQ_SPI: u32 = 53;

/// PCM Audio.
pub const IRQ_PCM: u32 = 54;

/// SD Host.
pub const IRQ_SDHOST: u32 = 55;

/// UART (PL011).
pub const IRQ_UART: u32 = 56;

/// EMMC (SD Card Controller).
pub const IRQ_EMMC: u32 = 57;

// ARM Basic IRQs (64+)

/// ARM Timer (Basic IRQ 0).
pub const IRQ_ARM_TIMER: u32 = 64;

/// ARM Mailbox (Basic IRQ 1).
pub const IRQ_ARM_MAILBOX: u32 = 65;

/// ARM Doorbell 0 (Basic IRQ 2).
pub const IRQ_ARM_DOORBELL0: u32 = 66;

/// ARM Doorbell 1 (Basic IRQ 3).
pub const IRQ_ARM_DOORBELL1: u32 = 67;

// =============================================================================
// Types
// =============================================================================

/// Maximum IRQ number supported.
const MAX_IRQ: u32 = 71;

/// Number of GPU IRQs per bank.
const IRQS_PER_BANK: u32 = 32;

/// Start of ARM basic IRQs.
const BASIC_IRQ_START: u32 = 64;

/// Interrupt controller error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptError {
    /// IRQ number out of range.
    InvalidIrq,
}

/// Result type for interrupt operations.
pub type Result<T> = core::result::Result<T, InterruptError>;

// =============================================================================
// BCM Interrupt Controller
// =============================================================================

/// BCM2711 Legacy Interrupt Controller driver.
///
/// Provides IRQ enable/disable and pending status operations.
pub struct BcmInterruptController {
    base: usize,
}

impl BcmInterruptController {
    /// Create a new interrupt controller instance.
    #[inline]
    pub const fn new() -> Self {
        Self { base: INTERRUPT_BASE }
    }

    /// Create a new interrupt controller with a custom base address.
    ///
    /// # Safety
    ///
    /// The provided base address must be a valid interrupt controller base.
    #[inline]
    pub const unsafe fn with_base(base: usize) -> Self {
        Self { base }
    }

    /// Initialize the interrupt controller.
    ///
    /// Disables all interrupts.
    ///
    /// # Errors
    ///
    /// Returns `Err(InterruptError::InvalidIrq)` if initialization fails.
    pub fn init(&mut self) -> Result<()> {
        // SAFETY: base is guaranteed to be valid interrupt controller address
        unsafe {
            // Disable all GPU IRQs bank 1
            let mut disable1 = MmioReg::<u32>::new(self.base + DISABLE_IRQS_1);
            disable1.write(0xFFFF_FFFF);
            dsb();

            // Disable all GPU IRQs bank 2
            let mut disable2 = MmioReg::<u32>::new(self.base + DISABLE_IRQS_2);
            disable2.write(0xFFFF_FFFF);
            dsb();

            // Disable all ARM basic IRQs
            let mut disable_basic = MmioReg::<u32>::new(self.base + DISABLE_BASIC_IRQS);
            disable_basic.write(0xFF);
            dsb();
        }

        Ok(())
    }

    /// Enable an interrupt.
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number (0-63 for GPU IRQs, 64-71 for ARM basic)
    ///
    /// # Errors
    ///
    /// Returns `Err(InterruptError::InvalidIrq)` if IRQ is out of range.
    pub fn enable(&mut self, irq: u32) -> Result<()> {
        if irq > MAX_IRQ {
            return Err(InterruptError::InvalidIrq);
        }

        // SAFETY: base is guaranteed to be valid interrupt controller address
        unsafe {
            if irq < IRQS_PER_BANK {
                // GPU IRQs 0-31
                let mut enable = MmioReg::<u32>::new(self.base + ENABLE_IRQS_1);
                enable.write(1 << irq);
            } else if irq < BASIC_IRQ_START {
                // GPU IRQs 32-63
                let mut enable = MmioReg::<u32>::new(self.base + ENABLE_IRQS_2);
                enable.write(1 << (irq - IRQS_PER_BANK));
            } else {
                // ARM basic IRQs 64-71
                let mut enable = MmioReg::<u32>::new(self.base + ENABLE_BASIC_IRQS);
                enable.write(1 << (irq - BASIC_IRQ_START));
            }
            dsb();
        }

        Ok(())
    }

    /// Disable an interrupt.
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number (0-63 for GPU IRQs, 64-71 for ARM basic)
    ///
    /// # Errors
    ///
    /// Returns `Err(InterruptError::InvalidIrq)` if IRQ is out of range.
    pub fn disable(&mut self, irq: u32) -> Result<()> {
        if irq > MAX_IRQ {
            return Err(InterruptError::InvalidIrq);
        }

        // SAFETY: base is guaranteed to be valid interrupt controller address
        unsafe {
            if irq < IRQS_PER_BANK {
                // GPU IRQs 0-31
                let mut disable = MmioReg::<u32>::new(self.base + DISABLE_IRQS_1);
                disable.write(1 << irq);
            } else if irq < BASIC_IRQ_START {
                // GPU IRQs 32-63
                let mut disable = MmioReg::<u32>::new(self.base + DISABLE_IRQS_2);
                disable.write(1 << (irq - IRQS_PER_BANK));
            } else {
                // ARM basic IRQs 64-71
                let mut disable = MmioReg::<u32>::new(self.base + DISABLE_BASIC_IRQS);
                disable.write(1 << (irq - BASIC_IRQ_START));
            }
            dsb();
        }

        Ok(())
    }

    /// Check if an interrupt is pending.
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number (0-63 for GPU IRQs, 64-71 for ARM basic)
    ///
    /// # Returns
    ///
    /// `true` if the interrupt is pending, `false` otherwise or if invalid IRQ.
    pub fn is_pending(&self, irq: u32) -> bool {
        if irq > MAX_IRQ {
            return false;
        }

        // SAFETY: base is guaranteed to be valid interrupt controller address
        unsafe {
            if irq < IRQS_PER_BANK {
                // GPU IRQs 0-31
                let pending = MmioReg::<u32>::new(self.base + IRQ_PENDING_1);
                (pending.read() & (1 << irq)) != 0
            } else if irq < BASIC_IRQ_START {
                // GPU IRQs 32-63
                let pending = MmioReg::<u32>::new(self.base + IRQ_PENDING_2);
                (pending.read() & (1 << (irq - IRQS_PER_BANK))) != 0
            } else {
                // ARM basic IRQs 64-71
                let pending = MmioReg::<u32>::new(self.base + IRQ_BASIC_PENDING);
                (pending.read() & (1 << (irq - BASIC_IRQ_START))) != 0
            }
        }
    }

    /// Get all pending GPU IRQs in bank 1 (0-31).
    ///
    /// # Returns
    ///
    /// A bitmask of pending interrupts.
    pub fn pending_bank1(&self) -> u32 {
        // SAFETY: base is guaranteed to be valid interrupt controller address
        unsafe {
            let pending = MmioReg::<u32>::new(self.base + IRQ_PENDING_1);
            pending.read()
        }
    }

    /// Get all pending GPU IRQs in bank 2 (32-63).
    ///
    /// # Returns
    ///
    /// A bitmask of pending interrupts.
    pub fn pending_bank2(&self) -> u32 {
        // SAFETY: base is guaranteed to be valid interrupt controller address
        unsafe {
            let pending = MmioReg::<u32>::new(self.base + IRQ_PENDING_2);
            pending.read()
        }
    }

    /// Get all pending ARM basic IRQs.
    ///
    /// # Returns
    ///
    /// A bitmask of pending basic interrupts plus summary bits.
    pub fn pending_basic(&self) -> u32 {
        // SAFETY: base is guaranteed to be valid interrupt controller address
        unsafe {
            let pending = MmioReg::<u32>::new(self.base + IRQ_BASIC_PENDING);
            pending.read()
        }
    }

    /// Check if any GPU IRQs are pending.
    ///
    /// This is faster than checking individual IRQs.
    #[inline]
    pub fn any_gpu_pending(&self) -> bool {
        let basic = self.pending_basic();
        (basic & (BASIC_PENDING_1 | BASIC_PENDING_2)) != 0
    }

    /// Find the first pending GPU IRQ.
    ///
    /// # Returns
    ///
    /// The IRQ number of the first pending interrupt, or `None` if none pending.
    pub fn first_pending(&self) -> Option<u32> {
        let basic = self.pending_basic();

        // Check basic IRQs first (ARM timer, mailbox, etc.)
        for i in 0..8 {
            if (basic & (1 << i)) != 0 {
                return Some(BASIC_IRQ_START + i);
            }
        }

        // Check GPU bank 1
        if (basic & BASIC_PENDING_1) != 0 {
            let pending1 = self.pending_bank1();
            for i in 0..32 {
                if (pending1 & (1 << i)) != 0 {
                    return Some(i);
                }
            }
        }

        // Check GPU bank 2
        if (basic & BASIC_PENDING_2) != 0 {
            let pending2 = self.pending_bank2();
            for i in 0..32 {
                if (pending2 & (1 << i)) != 0 {
                    return Some(IRQS_PER_BANK + i);
                }
            }
        }

        None
    }

    /// Enable FIQ for a specific IRQ.
    ///
    /// Only one IRQ can be the FIQ source at a time.
    ///
    /// # Arguments
    ///
    /// * `irq` - Interrupt number to use as FIQ source
    ///
    /// # Errors
    ///
    /// Returns `Err(InterruptError::InvalidIrq)` if IRQ is out of range.
    pub fn enable_fiq(&mut self, irq: u32) -> Result<()> {
        if irq > 127 {
            return Err(InterruptError::InvalidIrq);
        }

        // SAFETY: base is guaranteed to be valid interrupt controller address
        unsafe {
            let mut fiq = MmioReg::<u32>::new(self.base + FIQ_CONTROL);
            // Bit 7 = FIQ enable, bits 0-6 = IRQ source
            fiq.write(0x80 | (irq & 0x7F));
            dsb();
        }

        Ok(())
    }

    /// Disable FIQ.
    pub fn disable_fiq(&mut self) {
        // SAFETY: base is guaranteed to be valid interrupt controller address
        unsafe {
            let mut fiq = MmioReg::<u32>::new(self.base + FIQ_CONTROL);
            fiq.write(0);
            dsb();
        }
    }
}

impl Default for BcmInterruptController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_irq_constants() {
        assert_eq!(IRQ_TIMER1, 1);
        assert_eq!(IRQ_AUX, 29);
        assert_eq!(IRQ_UART, 56);
        assert_eq!(IRQ_ARM_TIMER, 64);
    }

    #[test]
    fn test_irq_banks() {
        // IRQ 0-31 in bank 1
        assert!(IRQ_TIMER1 < IRQS_PER_BANK);
        assert!(IRQ_AUX < IRQS_PER_BANK);

        // IRQ 32-63 in bank 2
        assert!(IRQ_GPIO0 >= IRQS_PER_BANK && IRQ_GPIO0 < BASIC_IRQ_START);
        assert!(IRQ_UART >= IRQS_PER_BANK && IRQ_UART < BASIC_IRQ_START);

        // IRQ 64+ are basic ARM IRQs
        assert!(IRQ_ARM_TIMER >= BASIC_IRQ_START);
    }

    #[test]
    fn test_invalid_irq() {
        let mut irq = BcmInterruptController::new();
        assert!(irq.enable(72).is_err());
        assert!(irq.enable(100).is_err());
        assert!(irq.disable(255).is_err());
    }
}
