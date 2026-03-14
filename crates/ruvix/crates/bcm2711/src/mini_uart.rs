//! # BCM2711 Mini UART (Auxiliary UART) Driver
//!
//! This module implements a driver for the BCM2711 mini UART, also known as
//! the auxiliary UART or UART1. This is the default UART used for console
//! output on Raspberry Pi 4/5.
//!
//! ## Overview
//!
//! The mini UART is simpler than the PL011 UART0:
//!
//! - Fixed 8N1 configuration (8 data bits, no parity, 1 stop bit)
//! - Smaller FIFOs (8 bytes vs 16 bytes)
//! - Baud rate tied to core clock (can vary with CPU frequency)
//! - No hardware flow control (RTS/CTS)
//!
//! ## Register Map
//!
//! The mini UART is part of the AUX (auxiliary) peripheral block.
//!
//! | Offset | Register | Description |
//! |--------|----------|-------------|
//! | 0x00 | AUX_IRQ | Auxiliary Interrupt status |
//! | 0x04 | AUX_ENABLES | Auxiliary enables |
//! | 0x40 | AUX_MU_IO | Mini UART I/O Data |
//! | 0x44 | AUX_MU_IER | Mini UART Interrupt Enable |
//! | 0x48 | AUX_MU_IIR | Mini UART Interrupt Identify |
//! | 0x4C | AUX_MU_LCR | Mini UART Line Control |
//! | 0x50 | AUX_MU_MCR | Mini UART Modem Control |
//! | 0x54 | AUX_MU_LSR | Mini UART Line Status |
//! | 0x58 | AUX_MU_MSR | Mini UART Modem Status |
//! | 0x5C | AUX_MU_SCRATCH | Mini UART Scratch |
//! | 0x60 | AUX_MU_CNTL | Mini UART Extra Control |
//! | 0x64 | AUX_MU_STAT | Mini UART Extra Status |
//! | 0x68 | AUX_MU_BAUD | Mini UART Baudrate |
//!
//! ## Baud Rate Calculation
//!
//! The mini UART baud rate is calculated as:
//!
//! ```text
//! baud = system_clock / (8 * (baud_reg + 1))
//! ```
//!
//! For a 500 MHz system clock and 115200 baud:
//!
//! ```text
//! baud_reg = (500000000 / (8 * 115200)) - 1 = 541
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_bcm2711::mini_uart::MiniUart;
//! use core::fmt::Write;
//!
//! let mut uart = MiniUart::new();
//! uart.init().unwrap();
//!
//! // Write using fmt::Write
//! writeln!(uart, "Hello from RuVix!").unwrap();
//!
//! // Read a character
//! if let Some(c) = uart.read_byte() {
//!     uart.write_byte(c).unwrap(); // Echo
//! }
//! ```
//!
//! ## GPIO Configuration
//!
//! The mini UART uses GPIO pins 14 (TXD) and 15 (RXD) in Alt5 function.
//! You must configure these pins before using the UART:
//!
//! ```rust,no_run
//! use ruvix_bcm2711::gpio::{Gpio, Function, Pull};
//!
//! let mut gpio = Gpio::new();
//! gpio.set_function(14, Function::Alt5).unwrap(); // TXD1
//! gpio.set_function(15, Function::Alt5).unwrap(); // RXD1
//! gpio.set_pull(14, Pull::None).unwrap();
//! gpio.set_pull(15, Pull::None).unwrap();
//! ```

use crate::mmio::{dsb, MmioReg};
use crate::{AUX_BASE, UART_CLOCK_HZ};
use core::fmt;

// =============================================================================
// Register Offsets (from AUX_BASE)
// =============================================================================

/// Auxiliary Interrupt Status.
const AUX_IRQ: usize = 0x00;

/// Auxiliary Enables.
const AUX_ENABLES: usize = 0x04;

/// Mini UART I/O Data.
const AUX_MU_IO: usize = 0x40;

/// Mini UART Interrupt Enable.
const AUX_MU_IER: usize = 0x44;

/// Mini UART Interrupt Identify.
const AUX_MU_IIR: usize = 0x48;

/// Mini UART Line Control.
const AUX_MU_LCR: usize = 0x4C;

/// Mini UART Modem Control.
const AUX_MU_MCR: usize = 0x50;

/// Mini UART Line Status.
const AUX_MU_LSR: usize = 0x54;

/// Mini UART Modem Status.
const AUX_MU_MSR: usize = 0x58;

/// Mini UART Scratch.
const AUX_MU_SCRATCH: usize = 0x5C;

/// Mini UART Extra Control.
const AUX_MU_CNTL: usize = 0x60;

/// Mini UART Extra Status.
const AUX_MU_STAT: usize = 0x64;

/// Mini UART Baud Rate.
const AUX_MU_BAUD: usize = 0x68;

// =============================================================================
// Register Bit Definitions
// =============================================================================

/// AUX_ENABLES: Enable mini UART.
const AUX_ENABLES_MU: u32 = 1 << 0;

/// AUX_ENABLES: Enable SPI1.
const AUX_ENABLES_SPI1: u32 = 1 << 1;

/// AUX_ENABLES: Enable SPI2.
const AUX_ENABLES_SPI2: u32 = 1 << 2;

/// AUX_MU_IER: Enable receive interrupt.
const IER_RX_ENABLE: u32 = 1 << 0;

/// AUX_MU_IER: Enable transmit interrupt.
const IER_TX_ENABLE: u32 = 1 << 1;

/// AUX_MU_IIR: Clear receive FIFO.
const IIR_RX_FIFO_CLEAR: u32 = 1 << 1;

/// AUX_MU_IIR: Clear transmit FIFO.
const IIR_TX_FIFO_CLEAR: u32 = 1 << 2;

/// AUX_MU_LCR: 8-bit mode.
const LCR_8BIT: u32 = 0b11;

/// AUX_MU_LSR: Data ready (RX FIFO has data).
const LSR_DATA_READY: u32 = 1 << 0;

/// AUX_MU_LSR: Transmitter empty (TX FIFO empty).
const LSR_TX_EMPTY: u32 = 1 << 5;

/// AUX_MU_LSR: Transmitter idle (completely finished).
const LSR_TX_IDLE: u32 = 1 << 6;

/// AUX_MU_CNTL: Enable receiver.
const CNTL_RX_ENABLE: u32 = 1 << 0;

/// AUX_MU_CNTL: Enable transmitter.
const CNTL_TX_ENABLE: u32 = 1 << 1;

/// AUX_MU_STAT: Symbol available (RX FIFO has data).
const STAT_RX_AVAILABLE: u32 = 1 << 0;

/// AUX_MU_STAT: Space available (TX FIFO has space).
const STAT_TX_AVAILABLE: u32 = 1 << 1;

/// AUX_MU_STAT: Receiver is idle.
const STAT_RX_IDLE: u32 = 1 << 2;

/// AUX_MU_STAT: Transmitter is idle.
const STAT_TX_IDLE: u32 = 1 << 3;

/// AUX_MU_STAT: Receiver overrun.
const STAT_RX_OVERRUN: u32 = 1 << 4;

/// AUX_MU_STAT: TX FIFO is full.
const STAT_TX_FULL: u32 = 1 << 5;

/// AUX_MU_STAT: TX FIFO is empty.
const STAT_TX_EMPTY: u32 = 1 << 8;

/// AUX_MU_STAT: TX done (FIFO empty and transmitter idle).
const STAT_TX_DONE: u32 = 1 << 9;

// =============================================================================
// Constants
// =============================================================================

/// Default baud rate.
pub const DEFAULT_BAUD_RATE: u32 = 115200;

/// Maximum FIFO depth.
const FIFO_DEPTH: usize = 8;

// =============================================================================
// Types
// =============================================================================

/// Mini UART error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UartError {
    /// Initialization failed.
    InitFailed,
    /// Write timeout.
    WriteTimeout,
    /// Buffer full.
    BufferFull,
}

/// Result type for UART operations.
pub type Result<T> = core::result::Result<T, UartError>;

// =============================================================================
// Mini UART Driver
// =============================================================================

/// BCM2711 Mini UART driver.
///
/// Provides serial console I/O via the auxiliary UART.
pub struct MiniUart {
    base: usize,
}

impl MiniUart {
    /// Create a new Mini UART driver instance.
    #[inline]
    pub const fn new() -> Self {
        Self { base: AUX_BASE }
    }

    /// Create a new Mini UART driver with a custom base address.
    ///
    /// # Safety
    ///
    /// The provided base address must be a valid AUX peripheral base.
    #[inline]
    pub const unsafe fn with_base(base: usize) -> Self {
        Self { base }
    }

    /// Initialize the Mini UART at the default baud rate (115200).
    ///
    /// This function:
    /// 1. Enables the mini UART in AUX_ENABLES
    /// 2. Disables TX/RX during configuration
    /// 3. Clears FIFOs
    /// 4. Configures 8-bit mode
    /// 5. Sets baud rate
    /// 6. Enables TX/RX
    ///
    /// # Note
    ///
    /// You must configure GPIO pins 14 and 15 for Alt5 function before calling this.
    ///
    /// # Errors
    ///
    /// Returns `Err(UartError::InitFailed)` if initialization fails.
    pub fn init(&mut self) -> Result<()> {
        self.init_with_baud(DEFAULT_BAUD_RATE)
    }

    /// Initialize the Mini UART with a specific baud rate.
    ///
    /// # Arguments
    ///
    /// * `baud` - Desired baud rate (e.g., 115200, 9600)
    ///
    /// # Errors
    ///
    /// Returns `Err(UartError::InitFailed)` if initialization fails.
    pub fn init_with_baud(&mut self, baud: u32) -> Result<()> {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            // Enable mini UART (and keep other AUX peripherals in current state)
            let mut enables = MmioReg::<u32>::new(self.base + AUX_ENABLES);
            enables.modify(|v| v | AUX_ENABLES_MU);
            dsb();

            // Disable TX/RX while configuring
            let mut cntl = MmioReg::<u32>::new(self.base + AUX_MU_CNTL);
            cntl.write(0);
            dsb();

            // Disable interrupts
            let mut ier = MmioReg::<u32>::new(self.base + AUX_MU_IER);
            ier.write(0);
            dsb();

            // Set 8-bit mode
            let mut lcr = MmioReg::<u32>::new(self.base + AUX_MU_LCR);
            lcr.write(LCR_8BIT);
            dsb();

            // Clear RTS (we don't use hardware flow control)
            let mut mcr = MmioReg::<u32>::new(self.base + AUX_MU_MCR);
            mcr.write(0);
            dsb();

            // Clear FIFOs
            let mut iir = MmioReg::<u32>::new(self.base + AUX_MU_IIR);
            iir.write(IIR_RX_FIFO_CLEAR | IIR_TX_FIFO_CLEAR);
            dsb();

            // Set baud rate
            // baud = system_clock / (8 * (baud_reg + 1))
            // baud_reg = (system_clock / (8 * baud)) - 1
            let baud_reg = (UART_CLOCK_HZ / (8 * baud)) - 1;
            let mut baud_rate = MmioReg::<u32>::new(self.base + AUX_MU_BAUD);
            baud_rate.write(baud_reg);
            dsb();

            // Enable TX and RX
            cntl.write(CNTL_TX_ENABLE | CNTL_RX_ENABLE);
            dsb();
        }

        Ok(())
    }

    /// Check if the transmit FIFO has space.
    #[inline]
    pub fn tx_ready(&self) -> bool {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let stat = MmioReg::<u32>::new(self.base + AUX_MU_STAT);
            (stat.read() & STAT_TX_AVAILABLE) != 0
        }
    }

    /// Check if the transmit FIFO is full.
    #[inline]
    pub fn tx_full(&self) -> bool {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let stat = MmioReg::<u32>::new(self.base + AUX_MU_STAT);
            (stat.read() & STAT_TX_FULL) != 0
        }
    }

    /// Check if the receive FIFO has data.
    #[inline]
    pub fn rx_ready(&self) -> bool {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let stat = MmioReg::<u32>::new(self.base + AUX_MU_STAT);
            (stat.read() & STAT_RX_AVAILABLE) != 0
        }
    }

    /// Write a single byte to the UART (blocking).
    ///
    /// # Arguments
    ///
    /// * `byte` - The byte to write
    ///
    /// # Errors
    ///
    /// Returns `Err(UartError::WriteTimeout)` if the FIFO remains full.
    pub fn write_byte(&mut self, byte: u8) -> Result<()> {
        // Wait for TX FIFO to have space
        let mut timeout = 1_000_000_u32;
        while !self.tx_ready() {
            timeout = timeout.saturating_sub(1);
            if timeout == 0 {
                return Err(UartError::WriteTimeout);
            }
            core::hint::spin_loop();
        }

        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let mut io = MmioReg::<u32>::new(self.base + AUX_MU_IO);
            io.write(u32::from(byte));
        }

        Ok(())
    }

    /// Write a single byte without blocking.
    ///
    /// Returns `Err(UartError::BufferFull)` if the TX FIFO is full.
    pub fn try_write_byte(&mut self, byte: u8) -> Result<()> {
        if !self.tx_ready() {
            return Err(UartError::BufferFull);
        }

        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let mut io = MmioReg::<u32>::new(self.base + AUX_MU_IO);
            io.write(u32::from(byte));
        }

        Ok(())
    }

    /// Read a single byte from the UART (non-blocking).
    ///
    /// Returns `None` if no data is available.
    pub fn read_byte(&mut self) -> Option<u8> {
        if !self.rx_ready() {
            return None;
        }

        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let io = MmioReg::<u32>::new(self.base + AUX_MU_IO);
            Some((io.read() & 0xFF) as u8)
        }
    }

    /// Read a single byte from the UART (blocking).
    ///
    /// Blocks until a byte is available.
    pub fn read_byte_blocking(&mut self) -> u8 {
        while !self.rx_ready() {
            core::hint::spin_loop();
        }

        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let io = MmioReg::<u32>::new(self.base + AUX_MU_IO);
            (io.read() & 0xFF) as u8
        }
    }

    /// Flush the transmit FIFO (blocking).
    ///
    /// Waits until all data has been transmitted.
    pub fn flush(&mut self) -> Result<()> {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let stat = MmioReg::<u32>::new(self.base + AUX_MU_STAT);

            // Wait for TX done (FIFO empty and transmitter idle)
            let mut timeout = 1_000_000_u32;
            while (stat.read() & STAT_TX_DONE) == 0 {
                timeout = timeout.saturating_sub(1);
                if timeout == 0 {
                    return Err(UartError::WriteTimeout);
                }
                core::hint::spin_loop();
            }
        }

        Ok(())
    }

    /// Write a byte slice to the UART.
    ///
    /// # Arguments
    ///
    /// * `data` - The bytes to write
    ///
    /// # Errors
    ///
    /// Returns `Err(UartError::WriteTimeout)` if a write times out.
    pub fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        for &byte in data {
            self.write_byte(byte)?;
        }
        Ok(())
    }

    /// Write a string to the UART.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to write
    ///
    /// # Errors
    ///
    /// Returns `Err(UartError::WriteTimeout)` if a write times out.
    pub fn write_str(&mut self, s: &str) -> Result<()> {
        self.write_bytes(s.as_bytes())
    }

    /// Enable receive interrupt.
    pub fn enable_rx_interrupt(&mut self) {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let mut ier = MmioReg::<u32>::new(self.base + AUX_MU_IER);
            ier.modify(|v| v | IER_RX_ENABLE);
        }
    }

    /// Disable receive interrupt.
    pub fn disable_rx_interrupt(&mut self) {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let mut ier = MmioReg::<u32>::new(self.base + AUX_MU_IER);
            ier.modify(|v| v & !IER_RX_ENABLE);
        }
    }

    /// Enable transmit interrupt.
    pub fn enable_tx_interrupt(&mut self) {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let mut ier = MmioReg::<u32>::new(self.base + AUX_MU_IER);
            ier.modify(|v| v | IER_TX_ENABLE);
        }
    }

    /// Disable transmit interrupt.
    pub fn disable_tx_interrupt(&mut self) {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let mut ier = MmioReg::<u32>::new(self.base + AUX_MU_IER);
            ier.modify(|v| v & !IER_TX_ENABLE);
        }
    }

    /// Get the number of bytes in the receive FIFO.
    pub fn rx_fifo_level(&self) -> usize {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let stat = MmioReg::<u32>::new(self.base + AUX_MU_STAT);
            ((stat.read() >> 16) & 0xF) as usize
        }
    }

    /// Get the number of bytes in the transmit FIFO.
    pub fn tx_fifo_level(&self) -> usize {
        // SAFETY: base is guaranteed to be valid AUX peripheral address
        unsafe {
            let stat = MmioReg::<u32>::new(self.base + AUX_MU_STAT);
            ((stat.read() >> 24) & 0xF) as usize
        }
    }
}

impl Default for MiniUart {
    fn default() -> Self {
        Self::new()
    }
}

// Implement core::fmt::Write for convenient text output
impl fmt::Write for MiniUart {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            self.write_byte(byte).map_err(|_| fmt::Error)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baud_rate_calculation() {
        // Test baud rate register calculation
        // baud_reg = (system_clock / (8 * baud)) - 1
        let baud_reg = (UART_CLOCK_HZ / (8 * 115200)) - 1;

        // For 500 MHz clock: 500000000 / (8 * 115200) - 1 = 541
        // This gives: 500000000 / (8 * 542) = 115313 baud (0.1% error)
        assert_eq!(baud_reg, 541);
    }

    #[test]
    fn test_register_bits() {
        assert_eq!(CNTL_TX_ENABLE, 0b10);
        assert_eq!(CNTL_RX_ENABLE, 0b01);
        assert_eq!(LCR_8BIT, 0b11);
    }
}
