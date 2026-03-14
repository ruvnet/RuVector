//! # BCM2711/BCM2712 SoC Drivers for Raspberry Pi 4/5
//!
//! This crate provides low-level drivers for the Broadcom BCM2711 (Raspberry Pi 4)
//! and BCM2712 (Raspberry Pi 5) System-on-Chip devices.
//!
//! ## Supported Hardware
//!
//! - **Raspberry Pi 4 Model B** (BCM2711, Cortex-A72)
//! - **Raspberry Pi 5** (BCM2712, Cortex-A76)
//! - **Raspberry Pi 400** (BCM2711, Cortex-A72)
//! - **Raspberry Pi Compute Module 4** (BCM2711, Cortex-A72)
//!
//! ## Memory Map
//!
//! The BCM2711/BCM2712 uses a VideoCore-based memory map where the ARM sees
//! peripherals at a different address than the VideoCore GPU.
//!
//! ### RPi 4 (BCM2711) Peripheral Base Addresses
//!
//! | Bus Address | ARM Physical | Description |
//! |-------------|--------------|-------------|
//! | 0x7E00_0000 | 0xFE00_0000 | Main peripherals |
//! | 0x7C00_0000 | 0xFC00_0000 | PCIe / xHCI |
//! | 0xFF80_0000 | 0x6000_0000 | ARM local peripherals |
//!
//! ### Peripheral Offsets (from PERIPHERAL_BASE)
//!
//! | Offset | Size | Device |
//! |--------|------|--------|
//! | 0x00003000 | 4KB | System Timer |
//! | 0x0000B000 | 4KB | Interrupt Controller |
//! | 0x0000B880 | 4KB | Mailbox |
//! | 0x00200000 | 4KB | GPIO |
//! | 0x00201000 | 4KB | UART0 (PL011) |
//! | 0x00215000 | 4KB | UART1 (Mini UART / AUX) |
//! | 0x00300000 | 4KB | EMMC |
//! | 0x00340000 | 4KB | EMMC2 |
//! | 0x01800000 | 64KB | PCIe Controller |
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_bcm2711::{gpio::Gpio, mini_uart::MiniUart, mailbox::Mailbox};
//!
//! // Initialize GPIO for mini UART
//! let mut gpio = Gpio::new();
//! gpio.set_function(14, ruvix_bcm2711::gpio::Function::Alt5).unwrap();
//! gpio.set_function(15, ruvix_bcm2711::gpio::Function::Alt5).unwrap();
//! gpio.set_pull(14, ruvix_bcm2711::gpio::Pull::None).unwrap();
//! gpio.set_pull(15, ruvix_bcm2711::gpio::Pull::None).unwrap();
//!
//! // Initialize mini UART at 115200 baud
//! let mut uart = MiniUart::new();
//! uart.init().unwrap();
//! uart.write_str("Hello from RuVix!\n").unwrap();
//!
//! // Query firmware version via mailbox
//! let mailbox = Mailbox::new();
//! let version = mailbox.get_firmware_revision().unwrap();
//! ```
//!
//! ## Safety
//!
//! All drivers use MMIO with volatile operations and proper memory barriers.
//! The `unsafe` keyword is used only for hardware register access.

#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod gpio;
pub mod interrupt;
pub mod mailbox;
pub mod mini_uart;
mod mmio;

// Re-export MMIO utilities for other crates
pub use mmio::{dmb, dsb, isb, read_volatile, write_volatile, MmioReg};

// =============================================================================
// Memory Map Constants
// =============================================================================

/// Peripheral base address for Raspberry Pi 4 (BCM2711).
///
/// The ARM cores see peripherals at 0xFE00_0000, while the VideoCore
/// sees them at 0x7E00_0000. We use the ARM physical address.
#[cfg(not(feature = "rpi5"))]
pub const PERIPHERAL_BASE: usize = 0xFE00_0000;

/// Peripheral base address for Raspberry Pi 5 (BCM2712).
///
/// The RPi 5 has a different memory map with peripherals at a higher address.
#[cfg(feature = "rpi5")]
pub const PERIPHERAL_BASE: usize = 0x1F_0000_0000;

/// ARM local peripherals base address (RPi 4).
///
/// Contains local timer, mailboxes, and interrupt routing.
pub const LOCAL_PERIPHERAL_BASE: usize = 0xFF80_0000;

// =============================================================================
// Peripheral Offsets
// =============================================================================

/// System Timer offset from PERIPHERAL_BASE.
pub const SYSTEM_TIMER_OFFSET: usize = 0x0000_3000;

/// Interrupt Controller offset from PERIPHERAL_BASE.
pub const INTERRUPT_CONTROLLER_OFFSET: usize = 0x0000_B000;

/// VideoCore Mailbox offset from PERIPHERAL_BASE.
pub const MAILBOX_OFFSET: usize = 0x0000_B880;

/// GPIO Controller offset from PERIPHERAL_BASE.
pub const GPIO_OFFSET: usize = 0x0020_0000;

/// UART0 (PL011) offset from PERIPHERAL_BASE.
pub const UART0_OFFSET: usize = 0x0020_1000;

/// AUX peripherals offset (including Mini UART) from PERIPHERAL_BASE.
pub const AUX_OFFSET: usize = 0x0021_5000;

/// EMMC (SD Card) offset from PERIPHERAL_BASE.
pub const EMMC_OFFSET: usize = 0x0030_0000;

/// EMMC2 (for RPi 4) offset from PERIPHERAL_BASE.
pub const EMMC2_OFFSET: usize = 0x0034_0000;

/// PCIe Controller offset from PERIPHERAL_BASE.
pub const PCIE_OFFSET: usize = 0x0180_0000;

/// USB xHCI Controller offset from PERIPHERAL_BASE.
pub const XHCI_OFFSET: usize = 0x0200_0000;

// =============================================================================
// Computed Base Addresses
// =============================================================================

/// System Timer base address.
pub const SYSTEM_TIMER_BASE: usize = PERIPHERAL_BASE + SYSTEM_TIMER_OFFSET;

/// Interrupt Controller base address.
pub const INTERRUPT_BASE: usize = PERIPHERAL_BASE + INTERRUPT_CONTROLLER_OFFSET;

/// Mailbox base address.
pub const MAILBOX_BASE: usize = PERIPHERAL_BASE + MAILBOX_OFFSET;

/// GPIO base address.
pub const GPIO_BASE: usize = PERIPHERAL_BASE + GPIO_OFFSET;

/// UART0 (PL011) base address.
pub const UART0_BASE: usize = PERIPHERAL_BASE + UART0_OFFSET;

/// AUX peripherals base address.
pub const AUX_BASE: usize = PERIPHERAL_BASE + AUX_OFFSET;

/// Mini UART base address (within AUX).
pub const MINI_UART_BASE: usize = AUX_BASE + 0x40;

/// EMMC base address.
pub const EMMC_BASE: usize = PERIPHERAL_BASE + EMMC_OFFSET;

/// EMMC2 base address.
pub const EMMC2_BASE: usize = PERIPHERAL_BASE + EMMC2_OFFSET;

/// PCIe Controller base address.
pub const PCIE_BASE: usize = PERIPHERAL_BASE + PCIE_OFFSET;

/// USB xHCI Controller base address.
pub const XHCI_BASE: usize = PERIPHERAL_BASE + XHCI_OFFSET;

// =============================================================================
// Board Detection
// =============================================================================

/// Raspberry Pi board revision codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BoardRevision {
    /// Raspberry Pi 4 Model B (1GB)
    Rpi4B1Gb = 0xa03111,
    /// Raspberry Pi 4 Model B (2GB)
    Rpi4B2Gb = 0xb03111,
    /// Raspberry Pi 4 Model B (4GB)
    Rpi4B4Gb = 0xc03111,
    /// Raspberry Pi 4 Model B (8GB)
    Rpi4B8Gb = 0xd03114,
    /// Raspberry Pi 400
    Rpi400 = 0xc03130,
    /// Raspberry Pi Compute Module 4
    Cm4 = 0xa03140,
    /// Unknown board
    Unknown = 0,
}

impl From<u32> for BoardRevision {
    fn from(value: u32) -> Self {
        match value {
            0xa03111 => BoardRevision::Rpi4B1Gb,
            0xb03111 => BoardRevision::Rpi4B2Gb,
            0xc03111 => BoardRevision::Rpi4B4Gb,
            0xd03114 => BoardRevision::Rpi4B8Gb,
            0xc03130 => BoardRevision::Rpi400,
            0xa03140 => BoardRevision::Cm4,
            _ => BoardRevision::Unknown,
        }
    }
}

// =============================================================================
// Clock Frequencies
// =============================================================================

/// Core clock frequency (default 500 MHz for RPi 4).
pub const CORE_CLOCK_HZ: u32 = 500_000_000;

/// System clock frequency (250 MHz).
pub const SYSTEM_CLOCK_HZ: u32 = 250_000_000;

/// UART reference clock frequency (48 MHz for mini UART).
pub const UART_CLOCK_HZ: u32 = 500_000_000;

/// ARM timer frequency (from system counter).
pub const ARM_TIMER_FREQ_HZ: u32 = 54_000_000;

// =============================================================================
// Driver Version
// =============================================================================

/// BCM2711 driver version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::gpio::{Function, Gpio, Pull};
    pub use crate::interrupt::BcmInterruptController;
    pub use crate::mailbox::Mailbox;
    pub use crate::mini_uart::MiniUart;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_map_constants() {
        // Verify peripheral addresses are correct for RPi 4
        #[cfg(not(feature = "rpi5"))]
        {
            assert_eq!(PERIPHERAL_BASE, 0xFE00_0000);
            assert_eq!(GPIO_BASE, 0xFE20_0000);
            assert_eq!(MAILBOX_BASE, 0xFE00_B880);
            assert_eq!(AUX_BASE, 0xFE21_5000);
            assert_eq!(MINI_UART_BASE, 0xFE21_5040);
        }
    }

    #[test]
    fn test_board_revision() {
        assert_eq!(BoardRevision::from(0xa03111), BoardRevision::Rpi4B1Gb);
        assert_eq!(BoardRevision::from(0xd03114), BoardRevision::Rpi4B8Gb);
        assert_eq!(BoardRevision::from(0x12345678), BoardRevision::Unknown);
    }
}
