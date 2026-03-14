//! # Early UART Initialization
//!
//! This module provides minimal UART initialization for early boot messages
//! before the full kernel is running.
//!
//! ## Purpose
//!
//! During early boot, before the MMU is enabled and before the full kernel
//! drivers are initialized, we need a simple way to output debug messages.
//! This module provides that capability using the mini UART.
//!
//! ## Requirements
//!
//! For this to work, the firmware must have already set up:
//!
//! - GPIO pins 14/15 configured for mini UART (Alt5)
//! - `enable_uart=1` in config.txt
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_rpi_boot::early_uart::{early_uart_init, early_print, early_println};
//!
//! // Initialize UART (must be done once)
//! early_uart_init();
//!
//! // Print messages
//! early_print("RuVix booting...\n");
//! early_println("CPU ID: ");
//! early_print_hex(ruvix_rpi_boot::spin_table::get_cpu_id() as u64);
//! early_print("\n");
//! ```

use ruvix_bcm2711::{AUX_BASE, MINI_UART_BASE, GPIO_BASE};

// =============================================================================
// Register Offsets (duplicated for early boot independence)
// =============================================================================

/// AUX enables register offset from AUX_BASE.
const AUX_ENABLES: usize = 0x04;

/// Mini UART I/O register offset from AUX_BASE.
const AUX_MU_IO: usize = 0x40;

/// Mini UART interrupt enable offset from AUX_BASE.
const AUX_MU_IER: usize = 0x44;

/// Mini UART interrupt identify offset from AUX_BASE.
const AUX_MU_IIR: usize = 0x48;

/// Mini UART line control offset from AUX_BASE.
const AUX_MU_LCR: usize = 0x4C;

/// Mini UART modem control offset from AUX_BASE.
const AUX_MU_MCR: usize = 0x50;

/// Mini UART line status offset from AUX_BASE.
const AUX_MU_LSR: usize = 0x54;

/// Mini UART extra control offset from AUX_BASE.
const AUX_MU_CNTL: usize = 0x60;

/// Mini UART baud rate offset from AUX_BASE.
const AUX_MU_BAUD: usize = 0x68;

// =============================================================================
// Register Bits
// =============================================================================

/// Enable mini UART bit in AUX_ENABLES.
const AUX_MU_ENABLE: u32 = 1;

/// 8-bit mode for line control.
const LCR_8BIT: u32 = 0b11;

/// TX FIFO can accept data (bit 5 of LSR).
const LSR_TX_READY: u32 = 1 << 5;

/// Enable TX in control register.
const CNTL_TX_ENABLE: u32 = 1 << 1;

/// Enable RX in control register.
const CNTL_RX_ENABLE: u32 = 1 << 0;

/// Clear FIFOs bits.
const IIR_FIFO_CLEAR: u32 = 0b110;

// =============================================================================
// Clock and Baud Rate
// =============================================================================

/// System clock frequency (500 MHz for RPi 4).
const SYSTEM_CLOCK: u32 = 500_000_000;

/// Default baud rate.
const BAUD_RATE: u32 = 115200;

// =============================================================================
// MMIO Helpers
// =============================================================================

/// Write to an MMIO register.
#[inline(always)]
unsafe fn mmio_write(addr: usize, value: u32) {
    core::ptr::write_volatile(addr as *mut u32, value);
}

/// Read from an MMIO register.
#[inline(always)]
unsafe fn mmio_read(addr: usize) -> u32 {
    core::ptr::read_volatile(addr as *const u32)
}

/// Memory barrier.
#[inline(always)]
fn barrier() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

// =============================================================================
// Early UART Functions
// =============================================================================

/// Initialize the mini UART for early boot output.
///
/// This assumes GPIO pins 14/15 are already configured by the firmware
/// (which is the case when `enable_uart=1` is in config.txt).
///
/// This function initializes the UART at 115200 baud, 8N1.
pub fn early_uart_init() {
    unsafe {
        // Enable mini UART
        let enables = mmio_read(AUX_BASE + AUX_ENABLES);
        mmio_write(AUX_BASE + AUX_ENABLES, enables | AUX_MU_ENABLE);
        barrier();

        // Disable TX/RX while configuring
        mmio_write(AUX_BASE + AUX_MU_CNTL, 0);
        barrier();

        // Disable interrupts
        mmio_write(AUX_BASE + AUX_MU_IER, 0);

        // Set 8-bit mode
        mmio_write(AUX_BASE + AUX_MU_LCR, LCR_8BIT);

        // Clear modem control (no RTS)
        mmio_write(AUX_BASE + AUX_MU_MCR, 0);

        // Clear FIFOs
        mmio_write(AUX_BASE + AUX_MU_IIR, IIR_FIFO_CLEAR);
        barrier();

        // Set baud rate
        // baud_reg = (system_clock / (8 * baud)) - 1
        let baud_reg = (SYSTEM_CLOCK / (8 * BAUD_RATE)) - 1;
        mmio_write(AUX_BASE + AUX_MU_BAUD, baud_reg);
        barrier();

        // Enable TX and RX
        mmio_write(AUX_BASE + AUX_MU_CNTL, CNTL_TX_ENABLE | CNTL_RX_ENABLE);
        barrier();
    }
}

/// Check if the TX FIFO can accept data.
#[inline]
fn tx_ready() -> bool {
    unsafe { (mmio_read(AUX_BASE + AUX_MU_LSR) & LSR_TX_READY) != 0 }
}

/// Write a single byte to the UART (blocking).
pub fn early_putc(c: u8) {
    // Wait for TX to be ready
    while !tx_ready() {
        core::hint::spin_loop();
    }

    unsafe {
        mmio_write(AUX_BASE + AUX_MU_IO, c as u32);
    }
}

/// Print a string to the UART.
pub fn early_print(s: &str) {
    for c in s.bytes() {
        // Convert \n to \r\n for terminal compatibility
        if c == b'\n' {
            early_putc(b'\r');
        }
        early_putc(c);
    }
}

/// Print a string followed by a newline.
pub fn early_println(s: &str) {
    early_print(s);
    early_print("\n");
}

/// Print a hexadecimal number (64-bit).
pub fn early_print_hex(value: u64) {
    const HEX_CHARS: &[u8] = b"0123456789ABCDEF";

    early_print("0x");

    // Find the first non-zero nibble
    let mut started = false;
    for i in (0..16).rev() {
        let nibble = ((value >> (i * 4)) & 0xF) as usize;
        if nibble != 0 || started || i == 0 {
            early_putc(HEX_CHARS[nibble]);
            started = true;
        }
    }
}

/// Print a hexadecimal number (32-bit).
pub fn early_print_hex32(value: u32) {
    const HEX_CHARS: &[u8] = b"0123456789ABCDEF";

    early_print("0x");

    let mut started = false;
    for i in (0..8).rev() {
        let nibble = ((value >> (i * 4)) & 0xF) as usize;
        if nibble != 0 || started || i == 0 {
            early_putc(HEX_CHARS[nibble]);
            started = true;
        }
    }
}

/// Print a decimal number (unsigned).
pub fn early_print_dec(mut value: u64) {
    if value == 0 {
        early_putc(b'0');
        return;
    }

    // Buffer for digits (max 20 digits for u64)
    let mut buf = [0u8; 20];
    let mut i = 0;

    while value > 0 {
        buf[i] = b'0' + (value % 10) as u8;
        value /= 10;
        i += 1;
    }

    // Print in reverse
    while i > 0 {
        i -= 1;
        early_putc(buf[i]);
    }
}

/// Print boot banner.
pub fn early_print_banner() {
    early_print("\n");
    early_print("================================================================================\n");
    early_print("  RuVix Cognition Kernel - Phase D (Raspberry Pi 4/5)\n");
    early_print("  Version: ");
    early_print(env!("CARGO_PKG_VERSION"));
    early_print("\n");
    early_print("================================================================================\n");
    early_print("\n");
}

/// Print early diagnostic information.
pub fn early_diagnostics() {
    use crate::{current_el, get_cpu_id};

    early_print("[BOOT] CPU ID: ");
    early_print_dec(get_cpu_id() as u64);
    early_print("\n");

    early_print("[BOOT] Exception Level: EL");
    early_print_dec(current_el() as u64);
    early_print("\n");

    early_print("[BOOT] AUX_BASE: ");
    early_print_hex(AUX_BASE as u64);
    early_print("\n");

    early_print("[BOOT] GPIO_BASE: ");
    early_print_hex(GPIO_BASE as u64);
    early_print("\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baud_calculation() {
        let baud_reg = (SYSTEM_CLOCK / (8 * BAUD_RATE)) - 1;
        // 500000000 / (8 * 115200) - 1 = 541
        assert_eq!(baud_reg, 541);
    }
}
