//! # RuVix Hardware Abstraction Layer
//!
//! This crate defines the hardware abstraction traits that platform-specific
//! implementations must satisfy. It provides a clean separation between the
//! cognition kernel and hardware-specific code.
//!
//! ## Design Principles
//!
//! - **No unsafe code in trait definitions** - `#![forbid(unsafe_code)]`
//! - **No std dependency** - `#![no_std]` only
//! - **Zero-copy operations** - Use borrowed references where possible
//! - **Result-based error handling** - All fallible operations return `Result`
//! - **Platform-agnostic** - Traits work across ARM64, RISC-V, x86_64
//!
//! ## Architecture
//!
//! The HAL is divided into five main subsystems:
//!
//! - **Console** - Serial I/O for debugging and logging
//! - **Interrupt Controller** - IRQ/FIQ management and routing
//! - **Timer** - Monotonic time and deadline scheduling
//! - **MMU** - Virtual memory and page table management
//! - **Power Management** - CPU power states and reset control
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_hal::{Console, Timer, InterruptController};
//!
//! fn kernel_main<C: Console, T: Timer, I: InterruptController>(
//!     console: &mut C,
//!     timer: &T,
//!     irq: &mut I,
//! ) -> Result<(), ()> {
//!     console.write_str("RuVix booting...\n")?;
//!     let _now = timer.now_ns();
//!     console.write_str("Time: ")?;
//!     console.flush()?;
//!
//!     irq.enable(32)?; // Enable UART interrupt
//!     irq.set_priority(32, 1)?;
//!
//!     Ok(())
//! }
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod console;
pub mod interrupt;
pub mod mmu;
pub mod power;
pub mod timer;

pub use console::Console;
pub use interrupt::InterruptController;
pub use mmu::Mmu;
pub use power::PowerManagement;
pub use timer::Timer;

/// HAL version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{Console, InterruptController, Mmu, PowerManagement, Timer};
}
