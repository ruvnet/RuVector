//! # RuVix Device Drivers
//!
//! This crate provides device drivers for the RuVix Cognition Kernel (ADR-087).
//! All drivers are designed for the QEMU virt machine on AArch64.
//!
//! ## Supported Devices
//!
//! - **PL011 UART** - ARM PrimeCell UART for serial console I/O
//! - **GICv2** - ARM Generic Interrupt Controller (GIC-400)
//! - **ARM Generic Timer** - System timer with deadline scheduling
//!
//! ## Memory Map (QEMU virt)
//!
//! | Device | Base Address | Size | Description |
//! |--------|--------------|------|-------------|
//! | PL011 UART | 0x0900_0000 | 4KB | Serial console |
//! | GIC Distributor | 0x0800_0000 | 4KB | Interrupt distribution |
//! | GIC CPU Interface | 0x0800_1000 | 4KB | CPU-local interrupt interface |
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_drivers::{pl011::Pl011, gic::Gic, timer::ArmGenericTimer};
//! use ruvix_hal::{Console, InterruptController, Timer};
//!
//! // Initialize UART
//! let mut uart = Pl011::new(0x0900_0000);
//! uart.init().expect("UART init failed");
//! uart.write_str("Hello, RuVix!\n").expect("UART write failed");
//!
//! // Initialize GIC
//! let mut gic = Gic::new(0x0800_0000, 0x0800_1000).expect("Invalid GIC address");
//! gic.init().expect("GIC init failed");
//! gic.enable(33).expect("Failed to enable UART IRQ");
//!
//! // Initialize timer
//! let timer = ArmGenericTimer::new();
//! let now = timer.now_ns();
//! ```
//!
//! ## Safety
//!
//! All drivers use MMIO (Memory-Mapped I/O) with volatile operations and proper
//! memory barriers. The `unsafe` keyword is used only for:
//!
//! - Reading/writing to hardware registers
//! - Executing barrier instructions (DSB, DMB, ISB)
//!
//! All unsafe operations are documented with SAFETY comments.

#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod gic;
pub mod mmio;
pub mod pl011;
pub mod timer;

pub use gic::Gic;
pub use pl011::Pl011;
pub use timer::ArmGenericTimer;

/// Driver version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{ArmGenericTimer, Gic, Pl011};
}
