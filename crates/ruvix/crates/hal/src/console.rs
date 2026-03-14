//! # Console Abstraction
//!
//! Provides traits for serial console output, essential for kernel debugging
//! and early boot logging.
//!
//! ## Design
//!
//! The console is write-only during early boot (before interrupt handlers).
//! It supports both blocking and buffered writes.
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_hal::{Console, console::ConsoleError};
//!
//! fn log<C: Console>(console: &mut C, msg: &str) -> Result<(), ConsoleError> {
//!     console.write_str("[ ")?;
//!     console.write_str(msg)?;
//!     console.write_str(" ]\n")?;
//!     console.flush()?;
//!     Ok(())
//! }
//! ```

/// Console error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleError {
    /// Hardware is not ready
    NotReady,
    /// Write buffer is full
    BufferFull,
    /// Hardware fault (e.g., UART error)
    HardwareFault,
    /// Invalid character (e.g., non-ASCII when ASCII-only)
    InvalidCharacter,
}

impl core::fmt::Display for ConsoleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotReady => write!(f, "console not ready"),
            Self::BufferFull => write!(f, "console buffer full"),
            Self::HardwareFault => write!(f, "console hardware fault"),
            Self::InvalidCharacter => write!(f, "invalid character"),
        }
    }
}

/// Console abstraction for serial output
///
/// This trait provides basic serial I/O for kernel debugging and logging.
/// Implementations typically target UART hardware on embedded platforms.
///
/// ## Thread Safety
///
/// Implementations MUST be safe to call from interrupt context (IRQ-safe).
/// Use spinlocks or disable interrupts if necessary.
///
/// ## Example Implementation
///
/// ```rust,ignore
/// use ruvix_hal::{Console, console::ConsoleError};
///
/// struct Pl011Uart {
///     base_addr: usize,
/// }
///
/// impl Console for Pl011Uart {
///     fn write_byte(&mut self, byte: u8) -> Result<(), ConsoleError> {
///         // Wait for TXFF (transmit FIFO full) to clear
///         while self.is_tx_full() {
///             core::hint::spin_loop();
///         }
///         // Write to data register
///         unsafe {
///             core::ptr::write_volatile(self.base_addr as *mut u8, byte);
///         }
///         Ok(())
///     }
///
///     fn write_str(&mut self, s: &str) -> Result<(), ConsoleError> {
///         for byte in s.bytes() {
///             self.write_byte(byte)?;
///         }
///         Ok(())
///     }
///
///     fn flush(&mut self) -> Result<(), ConsoleError> {
///         // Wait for TX FIFO to drain
///         while !self.is_tx_empty() {
///             core::hint::spin_loop();
///         }
///         Ok(())
///     }
///
///     fn is_ready(&self) -> bool {
///         !self.is_tx_full()
///     }
/// }
/// ```
pub trait Console {
    /// Write a single byte to the console
    ///
    /// This is a blocking operation that waits for the hardware to be ready.
    ///
    /// # Errors
    ///
    /// Returns `ConsoleError::NotReady` if hardware initialization failed.
    /// Returns `ConsoleError::HardwareFault` on UART errors.
    fn write_byte(&mut self, byte: u8) -> Result<(), ConsoleError>;

    /// Write a string to the console
    ///
    /// Default implementation calls `write_byte` for each byte.
    /// Implementations may override for better performance.
    ///
    /// # Errors
    ///
    /// Propagates errors from `write_byte`.
    fn write_str(&mut self, s: &str) -> Result<(), ConsoleError> {
        for byte in s.bytes() {
            self.write_byte(byte)?;
        }
        Ok(())
    }

    /// Flush any buffered output
    ///
    /// Blocks until all pending data has been transmitted.
    ///
    /// # Errors
    ///
    /// Returns `ConsoleError::HardwareFault` on transmission errors.
    fn flush(&mut self) -> Result<(), ConsoleError>;

    /// Check if console is ready for writing
    ///
    /// Returns `true` if `write_byte` would not block.
    fn is_ready(&self) -> bool;

    /// Write formatted output (optional, requires alloc)
    ///
    /// Default implementation is a no-op. Override if formatted output is needed.
    #[allow(unused_variables)]
    fn write_fmt(&mut self, args: core::fmt::Arguments) -> Result<(), ConsoleError> {
        Ok(())
    }
}

/// Helper macro for writing to console
///
/// # Example
///
/// ```rust,ignore
/// console_write!(console, "Boot time: {} ns\n", timer.now_ns());
/// ```
#[macro_export]
macro_rules! console_write {
    ($console:expr, $($arg:tt)*) => {{
        use core::fmt::Write;
        struct ConsoleWriter<'a, C: Console>(&'a mut C);
        impl<'a, C: Console> core::fmt::Write for ConsoleWriter<'a, C> {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                self.0.write_str(s).map_err(|_| core::fmt::Error)
            }
        }
        write!(ConsoleWriter($console), $($arg)*)
    }};
}

/// Console with buffered writes (optional trait)
///
/// Useful for reducing UART contention in multi-threaded scenarios.
pub trait BufferedConsole: Console {
    /// Write byte to buffer (non-blocking)
    fn try_write_byte(&mut self, byte: u8) -> Result<(), ConsoleError>;

    /// Get number of bytes in write buffer
    fn pending_bytes(&self) -> usize;

    /// Drain buffer to hardware
    fn drain_buffer(&mut self) -> Result<usize, ConsoleError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockConsole {
        buffer: [u8; 256],
        len: usize,
    }

    impl MockConsole {
        fn new() -> Self {
            Self {
                buffer: [0; 256],
                len: 0,
            }
        }

        fn as_str(&self) -> &str {
            core::str::from_utf8(&self.buffer[..self.len]).unwrap()
        }
    }

    impl Console for MockConsole {
        fn write_byte(&mut self, byte: u8) -> Result<(), ConsoleError> {
            if self.len >= self.buffer.len() {
                return Err(ConsoleError::BufferFull);
            }
            self.buffer[self.len] = byte;
            self.len += 1;
            Ok(())
        }

        fn flush(&mut self) -> Result<(), ConsoleError> {
            Ok(())
        }

        fn is_ready(&self) -> bool {
            self.len < self.buffer.len()
        }
    }

    #[test]
    fn test_write_str() {
        let mut console = MockConsole::new();
        console.write_str("Hello, RuVix!").unwrap();
        assert_eq!(console.as_str(), "Hello, RuVix!");
    }

    #[test]
    fn test_buffer_full() {
        let mut console = MockConsole::new();
        let result = console.write_str(&"x".repeat(300));
        assert_eq!(result, Err(ConsoleError::BufferFull));
    }
}
