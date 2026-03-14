//! # ARM PrimeCell PL011 UART Driver
//!
//! This module implements a driver for the ARM PL011 UART controller,
//! commonly used in QEMU virt machines and real ARM hardware.
//!
//! ## Configuration
//!
//! - **Baud rate**: 115200
//! - **Data bits**: 8
//! - **Parity**: None
//! - **Stop bits**: 1
//! - **FIFO**: Enabled (16-byte depth)
//!
//! ## Register Map
//!
//! | Offset | Register | Description |
//! |--------|----------|-------------|
//! | 0x000 | DR | Data Register |
//! | 0x018 | FR | Flag Register |
//! | 0x024 | IBRD | Integer Baud Rate Divisor |
//! | 0x028 | FBRD | Fractional Baud Rate Divisor |
//! | 0x02C | LCR_H | Line Control Register |
//! | 0x030 | CR | Control Register |
//! | 0x038 | IMSC | Interrupt Mask Set/Clear |
//! | 0x040 | MIS | Masked Interrupt Status |
//! | 0x044 | ICR | Interrupt Clear Register |
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_drivers::pl011::Pl011;
//! use ruvix_hal::Console;
//!
//! let mut uart = Pl011::new(0x0900_0000);
//! uart.init().expect("UART init failed");
//! uart.write_str("Hello, world!\n").expect("Write failed");
//! uart.flush().expect("Flush failed");
//! ```

use crate::mmio::{dsb, MmioReg};
use core::fmt;

/// PL011 UART register offsets
const DR: usize = 0x000; // Data Register
const FR: usize = 0x018; // Flag Register
const IBRD: usize = 0x024; // Integer Baud Rate Divisor
const FBRD: usize = 0x028; // Fractional Baud Rate Divisor
const LCR_H: usize = 0x02C; // Line Control Register
const CR: usize = 0x030; // Control Register
const IMSC: usize = 0x038; // Interrupt Mask Set/Clear
const MIS: usize = 0x040; // Masked Interrupt Status
const ICR: usize = 0x044; // Interrupt Clear Register

/// Flag Register (FR) bits
const FR_TXFF: u32 = 1 << 5; // Transmit FIFO full
const FR_RXFE: u32 = 1 << 4; // Receive FIFO empty
const FR_BUSY: u32 = 1 << 3; // UART busy

/// Line Control Register (LCR_H) bits
const LCR_H_FEN: u32 = 1 << 4; // Enable FIFOs
const LCR_H_WLEN_8: u32 = 0b11 << 5; // 8-bit word length

/// Control Register (CR) bits
const CR_UARTEN: u32 = 1 << 0; // UART enable
const CR_TXE: u32 = 1 << 8; // Transmit enable
const CR_RXE: u32 = 1 << 9; // Receive enable

/// Interrupt Mask/Status bits
const INT_RX: u32 = 1 << 4; // Receive interrupt
const INT_TX: u32 = 1 << 5; // Transmit interrupt
const INT_RT: u32 = 1 << 6; // Receive timeout
const INT_OE: u32 = 1 << 10; // Overrun error

/// UART clock frequency (24 MHz for QEMU virt)
const UART_CLK: u32 = 24_000_000;

/// Target baud rate (115200)
const BAUD_RATE: u32 = 115_200;

/// PL011 UART driver
pub struct Pl011 {
    base: usize,
}

impl Pl011 {
    /// Create a new PL011 UART driver.
    ///
    /// # Arguments
    ///
    /// - `base` - Base address of the PL011 UART registers (e.g., 0x0900_0000 for QEMU virt)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ruvix_drivers::pl011::Pl011;
    ///
    /// let uart = Pl011::new(0x0900_0000);
    /// ```
    #[inline]
    pub const fn new(base: usize) -> Self {
        Self { base }
    }

    /// Initialize the UART with 115200 8N1 configuration.
    ///
    /// This configures:
    /// - Baud rate: 115200
    /// - Data bits: 8
    /// - Parity: None
    /// - Stop bits: 1
    /// - FIFO: Enabled
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if initialization fails.
    pub fn init(&mut self) -> Result<(), ()> {
        // SAFETY: base is guaranteed to be a valid MMIO address
        unsafe {
            // Disable UART
            let mut cr = MmioReg::<u32>::new(self.base + CR);
            cr.write(0);
            dsb();

            // Wait for UART to finish transmitting
            let fr = MmioReg::<u32>::new(self.base + FR);
            while (fr.read() & FR_BUSY) != 0 {
                core::hint::spin_loop();
            }

            // Flush FIFOs by disabling them
            let mut lcr_h = MmioReg::<u32>::new(self.base + LCR_H);
            lcr_h.write(0);
            dsb();

            // Configure baud rate
            // Baud divisor = UART_CLK / (16 * BAUD_RATE)
            // For 24MHz clock and 115200 baud: divisor = 13.02
            // IBRD = 13, FBRD = 0.02 * 64 = 1
            let divisor = (UART_CLK * 4) / BAUD_RATE; // Fixed-point with 6 fractional bits
            let ibrd = divisor >> 6;
            let fbrd = divisor & 0x3F;

            let mut ibrd_reg = MmioReg::<u32>::new(self.base + IBRD);
            let mut fbrd_reg = MmioReg::<u32>::new(self.base + FBRD);
            ibrd_reg.write(ibrd);
            fbrd_reg.write(fbrd);
            dsb();

            // Configure line control: 8N1, enable FIFOs
            lcr_h.write(LCR_H_WLEN_8 | LCR_H_FEN);
            dsb();

            // Disable all interrupts (polling mode by default)
            let mut imsc = MmioReg::<u32>::new(self.base + IMSC);
            imsc.write(0);
            dsb();

            // Clear all interrupts
            let mut icr = MmioReg::<u32>::new(self.base + ICR);
            icr.write(0x7FF);
            dsb();

            // Enable UART, TX, and RX
            cr.write(CR_UARTEN | CR_TXE | CR_RXE);
            dsb();
        }

        Ok(())
    }

    /// Enable receive interrupt.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the operation fails.
    pub fn enable_rx_interrupt(&mut self) -> Result<(), ()> {
        // SAFETY: base is guaranteed to be a valid MMIO address
        unsafe {
            let mut imsc = MmioReg::<u32>::new(self.base + IMSC);
            imsc.modify(|val| val | INT_RX | INT_RT | INT_OE);
        }
        Ok(())
    }

    /// Disable receive interrupt.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the operation fails.
    pub fn disable_rx_interrupt(&mut self) -> Result<(), ()> {
        // SAFETY: base is guaranteed to be a valid MMIO address
        unsafe {
            let mut imsc = MmioReg::<u32>::new(self.base + IMSC);
            imsc.modify(|val| val & !(INT_RX | INT_RT | INT_OE));
        }
        Ok(())
    }

    /// Check if transmit FIFO is full.
    #[inline]
    fn tx_full(&self) -> bool {
        // SAFETY: base is guaranteed to be a valid MMIO address
        unsafe {
            let fr = MmioReg::<u32>::new(self.base + FR);
            (fr.read() & FR_TXFF) != 0
        }
    }

    /// Check if receive FIFO is empty.
    #[inline]
    fn rx_empty(&self) -> bool {
        // SAFETY: base is guaranteed to be a valid MMIO address
        unsafe {
            let fr = MmioReg::<u32>::new(self.base + FR);
            (fr.read() & FR_RXFE) != 0
        }
    }

    /// Write a single byte to the UART (blocking).
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the write operation fails.
    pub fn write_byte(&mut self, byte: u8) -> Result<(), ()> {
        // Wait for TX FIFO to have space
        while self.tx_full() {
            core::hint::spin_loop();
        }

        // SAFETY: base is guaranteed to be a valid MMIO address
        unsafe {
            let mut dr = MmioReg::<u32>::new(self.base + DR);
            dr.write(u32::from(byte));
        }

        Ok(())
    }

    /// Read a single byte from the UART (non-blocking).
    ///
    /// Returns `None` if no data is available.
    pub fn read_byte(&mut self) -> Option<u8> {
        if self.rx_empty() {
            return None;
        }

        // SAFETY: base is guaranteed to be a valid MMIO address
        unsafe {
            let dr = MmioReg::<u32>::new(self.base + DR);
            Some((dr.read() & 0xFF) as u8)
        }
    }

    /// Flush the transmit FIFO (blocking).
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the flush operation fails.
    pub fn flush(&mut self) -> Result<(), ()> {
        // SAFETY: base is guaranteed to be a valid MMIO address
        unsafe {
            let fr = MmioReg::<u32>::new(self.base + FR);
            while (fr.read() & FR_BUSY) != 0 {
                core::hint::spin_loop();
            }
        }
        Ok(())
    }
}

// Implement core::fmt::Write for convenient text output
impl fmt::Write for Pl011 {
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
    fn test_pl011_new() {
        let uart = Pl011::new(0x0900_0000);
        assert_eq!(uart.base, 0x0900_0000);
    }

    #[test]
    fn test_baud_rate_calculation() {
        // Verify baud rate divisor calculation
        let divisor = (UART_CLK * 4) / BAUD_RATE;
        let ibrd = divisor >> 6;
        let fbrd = divisor & 0x3F;

        // For 24MHz and 115200 baud:
        // divisor = 24000000 * 4 / 115200 = 833.33...
        // ibrd = 833 / 64 = 13
        // fbrd = 833 % 64 = 1
        assert_eq!(ibrd, 13);
        assert_eq!(fbrd, 1);
    }
}
