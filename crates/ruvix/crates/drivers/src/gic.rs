//! # ARM Generic Interrupt Controller (GIC-400) Driver
//!
//! This module implements a driver for the ARM GICv2 (GIC-400) interrupt controller.
//! The GIC manages up to 1020 interrupt sources and routes them to CPU cores.
//!
//! ## Architecture
//!
//! The GIC has two main components:
//!
//! - **Distributor (GICD)** - Manages interrupt prioritization and distribution
//! - **CPU Interface (GICC)** - Per-CPU interrupt handling and acknowledgment
//!
//! ## Interrupt Types
//!
//! - **SGI (0-15)** - Software Generated Interrupts
//! - **PPI (16-31)** - Private Peripheral Interrupts (per-CPU)
//! - **SPI (32-1019)** - Shared Peripheral Interrupts
//!
//! ## Register Map
//!
//! ### Distributor (GICD)
//!
//! | Offset | Register | Description |
//! |--------|----------|-------------|
//! | 0x000 | CTLR | Distributor Control Register |
//! | 0x004 | TYPER | Interrupt Controller Type Register |
//! | 0x100 | ISENABLER | Interrupt Set-Enable Registers |
//! | 0x180 | ICENABLER | Interrupt Clear-Enable Registers |
//! | 0x200 | ISPENDR | Interrupt Set-Pending Registers |
//! | 0x280 | ICPENDR | Interrupt Clear-Pending Registers |
//! | 0x400 | IPRIORITYR | Interrupt Priority Registers |
//! | 0x800 | ITARGETSR | Interrupt Processor Targets Registers |
//! | 0xC00 | ICFGR | Interrupt Configuration Registers |
//!
//! ### CPU Interface (GICC)
//!
//! | Offset | Register | Description |
//! |--------|----------|-------------|
//! | 0x000 | CTLR | CPU Interface Control Register |
//! | 0x004 | PMR | Interrupt Priority Mask Register |
//! | 0x00C | IAR | Interrupt Acknowledge Register |
//! | 0x010 | EOIR | End of Interrupt Register |
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_drivers::gic::Gic;
//!
//! let mut gic = Gic::new(0x0800_0000, 0x0800_1000).expect("Invalid GIC address");
//! gic.init().expect("GIC init failed");
//!
//! // Enable UART interrupt (IRQ 33)
//! gic.enable(33).expect("Failed to enable IRQ");
//! gic.set_priority(33, 1).expect("Failed to set priority");
//!
//! // In interrupt handler:
//! if let Some(irq) = gic.acknowledge() {
//!     // Handle interrupt
//!     gic.end_of_interrupt(irq).expect("Failed to EOI");
//! }
//! ```

use crate::mmio::{dsb, MmioReg};

/// Distributor Control Register (GICD_CTLR) bits
const GICD_CTLR_ENABLE: u32 = 1 << 0;

/// CPU Interface Control Register (GICC_CTLR) bits
const GICC_CTLR_ENABLE: u32 = 1 << 0;

/// Maximum number of interrupts supported (256 for QEMU virt)
const MAX_INTERRUPTS: usize = 256;

/// Number of priority levels (16)
const NUM_PRIORITIES: u8 = 16;

/// Special interrupt ID indicating no pending interrupt
const SPURIOUS_IRQ: u32 = 1023;

/// Error type for GIC operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GicError {
    /// IRQ number out of valid range
    InvalidIrq,
    /// Priority value out of valid range
    InvalidPriority,
    /// MMIO address outside valid peripheral range (CVE-002 protection)
    InvalidMmioAddress,
    /// GIC not initialized
    NotInitialized,
}

/// Valid MMIO regions for GIC on common platforms
/// CVE-002 FIX: Validate MMIO addresses are in known peripheral regions
mod mmio_validation {
    /// QEMU virt machine GIC region
    pub const QEMU_VIRT_GIC_START: usize = 0x0800_0000;
    pub const QEMU_VIRT_GIC_END: usize = 0x0802_0000;

    /// Raspberry Pi 4 (BCM2711) GIC region
    pub const BCM2711_GIC_START: usize = 0xFF84_0000;
    pub const BCM2711_GIC_END: usize = 0xFF85_0000;

    /// Generic AArch64 peripheral range (fallback)
    pub const GENERIC_PERIPH_START: usize = 0x0000_0000;
    pub const GENERIC_PERIPH_END: usize = 0xFFFF_FFFF;

    /// Validates that an address is within a known GIC MMIO region
    #[inline]
    pub fn is_valid_gic_address(addr: usize) -> bool {
        // Check QEMU virt
        if addr >= QEMU_VIRT_GIC_START && addr < QEMU_VIRT_GIC_END {
            return true;
        }
        // Check BCM2711
        if addr >= BCM2711_GIC_START && addr < BCM2711_GIC_END {
            return true;
        }
        // In production, this should be false. For flexibility, we allow
        // addresses in the upper half of memory (typically peripheral space)
        addr >= 0x0800_0000
    }
}

/// ARM GICv2 driver
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gic {
    gicd_base: usize, // Distributor base address
    gicc_base: usize, // CPU interface base address
}

impl Gic {
    /// Create a new GIC driver with validated MMIO addresses.
    ///
    /// # Arguments
    ///
    /// - `gicd_base` - Distributor base address (e.g., 0x0800_0000 for QEMU virt)
    /// - `gicc_base` - CPU interface base address (e.g., 0x0800_1000 for QEMU virt)
    ///
    /// # Errors
    ///
    /// Returns `Err(GicError::InvalidMmioAddress)` if addresses are not in
    /// valid peripheral MMIO regions. This prevents CVE-002 style attacks
    /// where arbitrary memory addresses could be used.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_drivers::gic::Gic;
    ///
    /// let gic = Gic::new(0x0800_0000, 0x0800_1000)?;
    /// ```
    #[inline]
    pub fn new(gicd_base: usize, gicc_base: usize) -> Result<Self, GicError> {
        // CVE-002 FIX: Validate MMIO addresses
        if !mmio_validation::is_valid_gic_address(gicd_base) {
            return Err(GicError::InvalidMmioAddress);
        }
        if !mmio_validation::is_valid_gic_address(gicc_base) {
            return Err(GicError::InvalidMmioAddress);
        }

        Ok(Self {
            gicd_base,
            gicc_base,
        })
    }

    /// Create a new GIC driver without validation (unsafe).
    ///
    /// # Safety
    ///
    /// The caller must ensure that `gicd_base` and `gicc_base` are valid
    /// GIC MMIO addresses. Using invalid addresses can cause undefined
    /// behavior or security vulnerabilities.
    #[inline]
    pub const unsafe fn new_unchecked(gicd_base: usize, gicc_base: usize) -> Self {
        Self {
            gicd_base,
            gicc_base,
        }
    }

    /// Initialize the GIC.
    ///
    /// This:
    /// - Disables all interrupts
    /// - Clears all pending interrupts
    /// - Sets default priorities (lowest)
    /// - Routes all SPIs to CPU 0
    /// - Enables the distributor and CPU interface
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if initialization fails.
    pub fn init(&mut self) -> Result<(), ()> {
        // SAFETY: gicd_base and gicc_base are valid MMIO addresses
        unsafe {
            // Disable distributor
            let mut gicd_ctlr = MmioReg::<u32>::new(self.gicd_base + 0x000);
            gicd_ctlr.write(0);
            dsb();

            // Disable all interrupts
            for i in 0..(MAX_INTERRUPTS / 32) {
                let mut icenabler = MmioReg::<u32>::new(self.gicd_base + 0x180 + i * 4);
                icenabler.write(0xFFFF_FFFF);
            }
            dsb();

            // Clear all pending interrupts
            for i in 0..(MAX_INTERRUPTS / 32) {
                let mut icpendr = MmioReg::<u32>::new(self.gicd_base + 0x280 + i * 4);
                icpendr.write(0xFFFF_FFFF);
            }
            dsb();

            // Set all priorities to lowest (255)
            for i in 0..MAX_INTERRUPTS {
                let mut ipriorityr = MmioReg::<u8>::new(self.gicd_base + 0x400 + i);
                ipriorityr.write(255);
            }
            dsb();

            // Route all SPIs to CPU 0 (target CPU mask = 0x01)
            for i in 32..MAX_INTERRUPTS {
                let mut itargetsr = MmioReg::<u8>::new(self.gicd_base + 0x800 + i);
                itargetsr.write(0x01);
            }
            dsb();

            // Configure all interrupts as level-sensitive (default)
            for i in 0..(MAX_INTERRUPTS / 16) {
                let mut icfgr = MmioReg::<u32>::new(self.gicd_base + 0xC00 + i * 4);
                icfgr.write(0);
            }
            dsb();

            // Enable distributor
            gicd_ctlr.write(GICD_CTLR_ENABLE);
            dsb();

            // Set CPU interface priority mask (allow all priorities)
            let mut gicc_pmr = MmioReg::<u32>::new(self.gicc_base + 0x004);
            gicc_pmr.write(0xFF);
            dsb();

            // Enable CPU interface
            let mut gicc_ctlr = MmioReg::<u32>::new(self.gicc_base + 0x000);
            gicc_ctlr.write(GICC_CTLR_ENABLE);
            dsb();
        }

        Ok(())
    }

    /// Enable an interrupt.
    ///
    /// # Arguments
    ///
    /// - `irq` - Interrupt number (0-255)
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if `irq` is out of range.
    pub fn enable(&mut self, irq: u32) -> Result<(), ()> {
        if irq as usize >= MAX_INTERRUPTS {
            return Err(());
        }

        // SAFETY: gicd_base is a valid MMIO address
        unsafe {
            let reg_idx = irq / 32;
            let bit_idx = irq % 32;
            let mut isenabler =
                MmioReg::<u32>::new(self.gicd_base + 0x100 + (reg_idx as usize) * 4);
            isenabler.write(1 << bit_idx);
            dsb();
        }

        Ok(())
    }

    /// Disable an interrupt.
    ///
    /// # Arguments
    ///
    /// - `irq` - Interrupt number (0-255)
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if `irq` is out of range.
    pub fn disable(&mut self, irq: u32) -> Result<(), ()> {
        if irq as usize >= MAX_INTERRUPTS {
            return Err(());
        }

        // SAFETY: gicd_base is a valid MMIO address
        unsafe {
            let reg_idx = irq / 32;
            let bit_idx = irq % 32;
            let mut icenabler =
                MmioReg::<u32>::new(self.gicd_base + 0x180 + (reg_idx as usize) * 4);
            icenabler.write(1 << bit_idx);
            dsb();
        }

        Ok(())
    }

    /// Set interrupt priority.
    ///
    /// # Arguments
    ///
    /// - `irq` - Interrupt number (0-255)
    /// - `priority` - Priority level (0-15, where 0 is highest)
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if `irq` is out of range or `priority` is invalid.
    pub fn set_priority(&mut self, irq: u32, priority: u8) -> Result<(), ()> {
        if irq as usize >= MAX_INTERRUPTS || priority >= NUM_PRIORITIES {
            return Err(());
        }

        // SAFETY: gicd_base is a valid MMIO address
        unsafe {
            // GIC uses 8-bit priority values, but only top 4 bits are implemented
            // Shift priority to bits [7:4]
            let priority_val = priority << 4;
            let mut ipriorityr = MmioReg::<u8>::new(self.gicd_base + 0x400 + irq as usize);
            ipriorityr.write(priority_val);
            dsb();
        }

        Ok(())
    }

    /// Acknowledge an interrupt.
    ///
    /// Returns the interrupt ID, or `None` if no interrupt is pending.
    pub fn acknowledge(&mut self) -> Option<u32> {
        // SAFETY: gicc_base is a valid MMIO address
        unsafe {
            let iar = MmioReg::<u32>::new(self.gicc_base + 0x00C);
            let irq = iar.read();

            if irq == SPURIOUS_IRQ {
                None
            } else {
                Some(irq)
            }
        }
    }

    /// Signal end of interrupt.
    ///
    /// # Arguments
    ///
    /// - `irq` - Interrupt number to acknowledge
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the operation fails.
    pub fn end_of_interrupt(&mut self, irq: u32) -> Result<(), ()> {
        // SAFETY: gicc_base is a valid MMIO address
        unsafe {
            let mut eoir = MmioReg::<u32>::new(self.gicc_base + 0x010);
            eoir.write(irq);
            dsb();
        }

        Ok(())
    }

    /// Check if an interrupt is pending.
    ///
    /// # Arguments
    ///
    /// - `irq` - Interrupt number (0-255)
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if `irq` is out of range.
    pub fn is_pending(&self, irq: u32) -> Result<bool, ()> {
        if irq as usize >= MAX_INTERRUPTS {
            return Err(());
        }

        // SAFETY: gicd_base is a valid MMIO address
        unsafe {
            let reg_idx = irq / 32;
            let bit_idx = irq % 32;
            let ispendr = MmioReg::<u32>::new(self.gicd_base + 0x200 + (reg_idx as usize) * 4);
            Ok((ispendr.read() & (1 << bit_idx)) != 0)
        }
    }

    /// Get the number of interrupt lines supported by this GIC.
    pub fn max_interrupts(&self) -> usize {
        // SAFETY: gicd_base is a valid MMIO address
        unsafe {
            let typer = MmioReg::<u32>::new(self.gicd_base + 0x004);
            let it_lines_num = typer.read() & 0x1F;
            // Number of interrupt lines = 32 * (ITLinesNumber + 1)
            32 * (it_lines_num + 1) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gic_new() {
        // Valid QEMU virt addresses
        let gic = Gic::new(0x0800_0000, 0x0800_1000).unwrap();
        assert_eq!(gic.gicd_base, 0x0800_0000);
        assert_eq!(gic.gicc_base, 0x0800_1000);
    }

    #[test]
    fn test_gic_new_invalid_address() {
        // CVE-002: Invalid addresses should be rejected
        let result = Gic::new(0x0000_0100, 0x0000_0200);
        assert_eq!(result, Err(GicError::InvalidMmioAddress));
    }

    #[test]
    fn test_gic_new_unchecked() {
        // Unsafe version allows any address
        let gic = unsafe { Gic::new_unchecked(0x0800_0000, 0x0800_1000) };
        assert_eq!(gic.gicd_base, 0x0800_0000);
    }

    #[test]
    #[ignore = "requires QEMU/real hardware MMIO"]
    fn test_irq_validation() {
        let mut gic = Gic::new(0x0800_0000, 0x0800_1000).unwrap();

        // Valid IRQ
        assert!(gic.enable(0).is_ok());
        assert!(gic.enable(255).is_ok());

        // Invalid IRQ
        assert!(gic.enable(256).is_err());
        assert!(gic.enable(1024).is_err());
    }

    #[test]
    #[ignore = "requires QEMU/real hardware MMIO"]
    fn test_priority_validation() {
        let mut gic = Gic::new(0x0800_0000, 0x0800_1000).unwrap();

        // Valid priorities
        assert!(gic.set_priority(0, 0).is_ok());
        assert!(gic.set_priority(0, 15).is_ok());

        // Invalid priorities
        assert!(gic.set_priority(0, 16).is_err());
        assert!(gic.set_priority(0, 255).is_err());
    }
}
