//! # Boot Configuration (config.txt) Support
//!
//! This module provides parsing for Raspberry Pi boot configuration.
//! While we don't parse config.txt directly (that's done by the firmware),
//! we can parse the kernel command line passed via the DTB.
//!
//! ## Command Line Format
//!
//! The kernel command line is passed in the DTB `/chosen` node as the
//! `bootargs` property. Format:
//!
//! ```text
//! key1=value1 key2=value2 key3 key4="value with spaces"
//! ```
//!
//! ## Common Boot Arguments
//!
//! | Argument | Description |
//! |----------|-------------|
//! | `console=ttyS0,115200` | Serial console device |
//! | `root=/dev/mmcblk0p2` | Root filesystem |
//! | `rootfstype=ext4` | Root filesystem type |
//! | `elevator=deadline` | I/O scheduler |
//! | `quiet` | Suppress kernel messages |
//! | `debug` | Enable debug messages |
//! | `ruvix.heap_size=64M` | RuVix heap size |
//! | `ruvix.log_level=debug` | RuVix log level |
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_rpi_boot::config::{parse_cmdline, BootConfig};
//!
//! let cmdline = b"console=ttyS0,115200 ruvix.heap_size=64M quiet\0";
//! let config = parse_cmdline(cmdline);
//!
//! if config.quiet {
//!     // Suppress boot messages
//! }
//!
//! if let Some(baud) = config.console_baud {
//!     // Configure serial at specified baud rate
//! }
//! ```

/// Maximum command line length.
pub const MAX_CMDLINE_LENGTH: usize = 4096;

/// Maximum number of command line arguments.
pub const MAX_ARGS: usize = 64;

/// Configuration error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    /// Command line too long.
    CmdlineTooLong,
    /// Invalid argument format.
    InvalidFormat,
    /// Unknown argument.
    UnknownArg,
    /// Value parse error.
    ParseError,
}

/// Result type for configuration operations.
pub type Result<T> = core::result::Result<T, ConfigError>;

/// Boot configuration parsed from command line.
#[derive(Debug, Clone, Copy)]
pub struct BootConfig {
    /// Console device (e.g., "ttyS0").
    pub console_device: Option<ConsoleDevice>,
    /// Console baud rate.
    pub console_baud: Option<u32>,
    /// Quiet mode (suppress messages).
    pub quiet: bool,
    /// Debug mode (verbose messages).
    pub debug: bool,
    /// RuVix heap size in bytes.
    pub ruvix_heap_size: usize,
    /// RuVix log level.
    pub ruvix_log_level: LogLevel,
    /// Kernel page size (4KB or 64KB).
    pub page_size: PageSize,
    /// Enable SMP (multi-core).
    pub smp: bool,
    /// Maximum number of CPUs to use.
    pub maxcpus: u8,
}

impl BootConfig {
    /// Create a new boot configuration with default values.
    pub const fn new() -> Self {
        Self {
            console_device: None,
            console_baud: Some(115200),
            quiet: false,
            debug: false,
            ruvix_heap_size: 64 * 1024 * 1024, // 64 MB default
            ruvix_log_level: LogLevel::Info,
            page_size: PageSize::Small,
            smp: true,
            maxcpus: 4,
        }
    }
}

impl Default for BootConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Console device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleDevice {
    /// Serial port (ttyS0, ttyS1, etc.).
    Serial(u8),
    /// Mini UART (ttyAMA0).
    Ama(u8),
    /// USB serial.
    Usb(u8),
    /// Null console (discard output).
    Null,
}

/// Log level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// No logging.
    Off,
    /// Error messages only.
    Error,
    /// Warnings and errors.
    Warn,
    /// Informational messages.
    Info,
    /// Debug messages.
    Debug,
    /// Trace-level (very verbose).
    Trace,
}

/// Page size configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageSize {
    /// 4KB pages (standard).
    Small,
    /// 64KB pages (hugetlb).
    Large,
}

/// Parse a kernel command line into a boot configuration.
///
/// # Arguments
///
/// * `cmdline` - Null-terminated command line string
///
/// # Returns
///
/// A `BootConfig` with parsed values.
pub fn parse_cmdline(cmdline: &[u8]) -> BootConfig {
    let mut config = BootConfig::new();

    // Find the null terminator
    let len = cmdline.iter().position(|&b| b == 0).unwrap_or(cmdline.len());
    let cmdline = &cmdline[..len];

    // Parse arguments
    let mut i = 0;
    while i < cmdline.len() {
        // Skip whitespace
        while i < cmdline.len() && cmdline[i].is_ascii_whitespace() {
            i += 1;
        }

        if i >= cmdline.len() {
            break;
        }

        // Find end of argument
        let start = i;
        while i < cmdline.len() && !cmdline[i].is_ascii_whitespace() {
            i += 1;
        }

        let arg = &cmdline[start..i];
        parse_arg(arg, &mut config);
    }

    config
}

/// Parse a single command line argument.
fn parse_arg(arg: &[u8], config: &mut BootConfig) {
    // Handle boolean flags
    if arg == b"quiet" {
        config.quiet = true;
        return;
    }
    if arg == b"debug" {
        config.debug = true;
        config.ruvix_log_level = LogLevel::Debug;
        return;
    }
    if arg == b"nosmp" {
        config.smp = false;
        config.maxcpus = 1;
        return;
    }

    // Handle key=value pairs
    if let Some(eq_pos) = arg.iter().position(|&b| b == b'=') {
        let key = &arg[..eq_pos];
        let value = &arg[eq_pos + 1..];

        match key {
            b"console" => parse_console(value, config),
            b"maxcpus" => {
                if let Some(n) = parse_number(value) {
                    config.maxcpus = n.min(255) as u8;
                    if config.maxcpus == 0 {
                        config.smp = false;
                    }
                }
            }
            b"ruvix.heap_size" => {
                if let Some(size) = parse_size(value) {
                    config.ruvix_heap_size = size;
                }
            }
            b"ruvix.log_level" => {
                config.ruvix_log_level = parse_log_level(value);
            }
            b"ruvix.page_size" => {
                if value == b"64K" || value == b"64k" {
                    config.page_size = PageSize::Large;
                }
            }
            _ => {
                // Unknown argument, ignore
            }
        }
    }
}

/// Parse console argument (e.g., "ttyS0,115200").
fn parse_console(value: &[u8], config: &mut BootConfig) {
    // Find comma separator
    let (device, baud_str) = if let Some(comma) = value.iter().position(|&b| b == b',') {
        (&value[..comma], Some(&value[comma + 1..]))
    } else {
        (value, None)
    };

    // Parse device
    if device.starts_with(b"ttyS") {
        if let Some(n) = parse_digit(&device[4..]) {
            config.console_device = Some(ConsoleDevice::Serial(n));
        }
    } else if device.starts_with(b"ttyAMA") {
        if let Some(n) = parse_digit(&device[6..]) {
            config.console_device = Some(ConsoleDevice::Ama(n));
        }
    } else if device == b"null" {
        config.console_device = Some(ConsoleDevice::Null);
    }

    // Parse baud rate
    if let Some(baud_bytes) = baud_str {
        if let Some(baud) = parse_number(baud_bytes) {
            config.console_baud = Some(baud as u32);
        }
    }
}

/// Parse log level string.
fn parse_log_level(value: &[u8]) -> LogLevel {
    match value {
        b"off" | b"OFF" => LogLevel::Off,
        b"error" | b"ERROR" => LogLevel::Error,
        b"warn" | b"WARN" => LogLevel::Warn,
        b"info" | b"INFO" => LogLevel::Info,
        b"debug" | b"DEBUG" => LogLevel::Debug,
        b"trace" | b"TRACE" => LogLevel::Trace,
        _ => LogLevel::Info,
    }
}

/// Parse a size string (e.g., "64M", "1G", "512K").
fn parse_size(value: &[u8]) -> Option<usize> {
    if value.is_empty() {
        return None;
    }

    let last = value[value.len() - 1];
    let (num_bytes, multiplier) = match last {
        b'K' | b'k' => (&value[..value.len() - 1], 1024),
        b'M' | b'm' => (&value[..value.len() - 1], 1024 * 1024),
        b'G' | b'g' => (&value[..value.len() - 1], 1024 * 1024 * 1024),
        _ => (value, 1),
    };

    parse_number(num_bytes).map(|n| n * multiplier)
}

/// Parse a decimal number.
fn parse_number(bytes: &[u8]) -> Option<usize> {
    if bytes.is_empty() {
        return None;
    }

    let mut result: usize = 0;
    for &b in bytes {
        if !b.is_ascii_digit() {
            return None;
        }
        result = result.checked_mul(10)?;
        result = result.checked_add((b - b'0') as usize)?;
    }

    Some(result)
}

/// Parse a single digit.
fn parse_digit(bytes: &[u8]) -> Option<u8> {
    if bytes.is_empty() {
        return Some(0);
    }

    let mut result: u8 = 0;
    for &b in bytes {
        if !b.is_ascii_digit() {
            return None;
        }
        result = result.checked_mul(10)?;
        result = result.checked_add(b - b'0')?;
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_cmdline() {
        let config = parse_cmdline(b"\0");
        assert!(!config.quiet);
        assert!(!config.debug);
        assert_eq!(config.console_baud, Some(115200));
    }

    #[test]
    fn test_parse_quiet() {
        let config = parse_cmdline(b"quiet\0");
        assert!(config.quiet);
    }

    #[test]
    fn test_parse_debug() {
        let config = parse_cmdline(b"debug\0");
        assert!(config.debug);
        assert_eq!(config.ruvix_log_level, LogLevel::Debug);
    }

    #[test]
    fn test_parse_console() {
        let config = parse_cmdline(b"console=ttyS0,115200\0");
        assert_eq!(config.console_device, Some(ConsoleDevice::Serial(0)));
        assert_eq!(config.console_baud, Some(115200));
    }

    #[test]
    fn test_parse_heap_size() {
        let config = parse_cmdline(b"ruvix.heap_size=128M\0");
        assert_eq!(config.ruvix_heap_size, 128 * 1024 * 1024);
    }

    #[test]
    fn test_parse_maxcpus() {
        let config = parse_cmdline(b"maxcpus=2\0");
        assert_eq!(config.maxcpus, 2);
    }

    #[test]
    fn test_parse_nosmp() {
        let config = parse_cmdline(b"nosmp\0");
        assert!(!config.smp);
        assert_eq!(config.maxcpus, 1);
    }

    #[test]
    fn test_parse_multiple() {
        let config = parse_cmdline(b"console=ttyS0,115200 quiet ruvix.log_level=warn\0");
        assert!(config.quiet);
        assert_eq!(config.console_device, Some(ConsoleDevice::Serial(0)));
        assert_eq!(config.ruvix_log_level, LogLevel::Warn);
    }

    #[test]
    fn test_parse_number() {
        assert_eq!(parse_number(b"0"), Some(0));
        assert_eq!(parse_number(b"123"), Some(123));
        assert_eq!(parse_number(b"115200"), Some(115200));
        assert_eq!(parse_number(b""), None);
        assert_eq!(parse_number(b"12a3"), None);
    }

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size(b"512K"), Some(512 * 1024));
        assert_eq!(parse_size(b"64M"), Some(64 * 1024 * 1024));
        assert_eq!(parse_size(b"1G"), Some(1024 * 1024 * 1024));
        assert_eq!(parse_size(b"1024"), Some(1024));
    }
}
