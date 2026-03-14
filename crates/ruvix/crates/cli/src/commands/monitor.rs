//! Monitor command - UART/serial console for kernel debugging
//!
//! This module provides real serial port communication for RuVix kernel
//! debugging using the serialport crate.

use anyhow::{bail, Context, Result};
use clap::Args;
use colored::Colorize;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use serialport::{DataBits, FlowControl as SerialFlowControl, Parity as SerialParity, StopBits, SerialPort};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, BufWriter};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Arguments for the monitor command
#[derive(Args, Debug)]
pub struct MonitorArgs {
    /// Serial port path
    #[arg(
        short,
        long,
        help = "Serial port (e.g., /dev/ttyUSB0, COM3)",
        required_unless_present = "auto",
        required_unless_present = "list"
    )]
    pub port: Option<String>,

    /// Baud rate
    #[arg(short, long, default_value = "115200", help = "Baud rate")]
    pub baud: u32,

    /// Auto-detect serial port
    #[arg(long, help = "Auto-detect serial port")]
    pub auto: bool,

    /// List available ports
    #[arg(long, help = "List available serial ports")]
    pub list: bool,

    /// Log output to file
    #[arg(short, long, help = "Log to file")]
    pub log: Option<PathBuf>,

    /// Timestamp each line
    #[arg(long, help = "Add timestamps to output")]
    pub timestamp: bool,

    /// Data bits (5, 6, 7, 8)
    #[arg(long, default_value = "8", help = "Data bits")]
    pub data_bits: u8,

    /// Stop bits (1, 2)
    #[arg(long, default_value = "1", help = "Stop bits")]
    pub stop_bits: u8,

    /// Parity (none, odd, even)
    #[arg(long, default_value = "none", help = "Parity")]
    pub parity: Parity,

    /// Flow control (none, hardware, software)
    #[arg(long, default_value = "none", help = "Flow control")]
    pub flow_control: FlowControl,

    /// Reset DTR on connect
    #[arg(long, help = "Toggle DTR on connect (triggers reset on some boards)")]
    pub reset: bool,

    /// Exit on pattern match
    #[arg(long, help = "Exit when pattern is seen")]
    pub exit_on: Option<String>,

    /// Timeout in seconds
    #[arg(long, help = "Exit after timeout (seconds)")]
    pub timeout: Option<u64>,

    /// Hex dump mode
    #[arg(long, help = "Show hex dump of received data")]
    pub hex: bool,

    /// Suppress local echo
    #[arg(long, help = "Suppress local echo")]
    pub no_echo: bool,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum Parity {
    #[default]
    None,
    Odd,
    Even,
}

impl From<Parity> for SerialParity {
    fn from(p: Parity) -> Self {
        match p {
            Parity::None => SerialParity::None,
            Parity::Odd => SerialParity::Odd,
            Parity::Even => SerialParity::Even,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum FlowControl {
    #[default]
    None,
    Hardware,
    Software,
}

impl From<FlowControl> for SerialFlowControl {
    fn from(fc: FlowControl) -> Self {
        match fc {
            FlowControl::None => SerialFlowControl::None,
            FlowControl::Hardware => SerialFlowControl::Hardware,
            FlowControl::Software => SerialFlowControl::Software,
        }
    }
}

/// Convert data bits to serialport enum
fn data_bits_from_u8(bits: u8) -> Result<DataBits> {
    match bits {
        5 => Ok(DataBits::Five),
        6 => Ok(DataBits::Six),
        7 => Ok(DataBits::Seven),
        8 => Ok(DataBits::Eight),
        _ => bail!("Invalid data bits: {}. Must be 5, 6, 7, or 8.", bits),
    }
}

/// Convert stop bits to serialport enum
fn stop_bits_from_u8(bits: u8) -> Result<StopBits> {
    match bits {
        1 => Ok(StopBits::One),
        2 => Ok(StopBits::Two),
        _ => bail!("Invalid stop bits: {}. Must be 1 or 2.", bits),
    }
}

/// Execute the monitor command
pub fn execute(args: MonitorArgs, verbose: bool) -> Result<()> {
    // Handle list ports
    if args.list {
        return list_ports(verbose);
    }

    // Handle auto-detect
    let port = if args.auto {
        auto_detect_port(verbose)?
    } else {
        args.port.clone().context("Serial port is required")?
    };

    if verbose {
        println!("{}", "Monitor Configuration:".cyan().bold());
        println!("  Port:         {}", port.yellow());
        println!("  Baud:         {}", args.baud);
        println!("  Data bits:    {}", args.data_bits);
        println!("  Stop bits:    {}", args.stop_bits);
        println!("  Parity:       {:?}", args.parity);
        println!("  Flow control: {:?}", args.flow_control);
        if let Some(ref log) = args.log {
            println!("  Log file:     {}", log.display());
        }
        println!();
    }

    // Connect to serial port
    println!(
        "{} Connecting to {} at {} baud...",
        "[monitor]".cyan(),
        port.yellow(),
        args.baud
    );

    let mut serial = connect_serial(&port, &args)?;

    // Show help
    println!();
    println!("{}", "Serial Monitor Commands:".dimmed());
    println!("  {} - Quit monitor", "Ctrl+Q".yellow());
    println!("  {} - Send break", "Ctrl+B".yellow());
    println!("  {} - Toggle timestamps", "Ctrl+T".yellow());
    println!("  {} - Toggle hex mode", "Ctrl+H".yellow());
    println!();
    println!("{}", "--- Serial Output ---".dimmed());
    println!();

    // Run the monitor loop
    run_monitor_loop(&mut serial, &args)
}

fn list_ports(_verbose: bool) -> Result<()> {
    println!("{}", "Available Serial Ports:".cyan().bold());
    println!();

    let ports = serialport::available_ports().context("Failed to enumerate serial ports")?;

    if ports.is_empty() {
        println!("  No serial ports found.");
        println!();
        println!("  Common issues:");
        println!("    - Device not connected");
        println!("    - Missing driver");
        println!("    - Insufficient permissions (try: sudo usermod -aG dialout $USER)");
        return Ok(());
    }

    println!(
        "  {:<20} {:<30} {}",
        "PORT".bold(),
        "DESCRIPTION".bold(),
        "TYPE".bold()
    );
    println!("  {}", "-".repeat(70));

    for port in &ports {
        let (description, port_type) = match &port.port_type {
            serialport::SerialPortType::UsbPort(info) => {
                let desc = format!(
                    "{} (VID:{:04x} PID:{:04x})",
                    info.product.as_deref().unwrap_or("Unknown"),
                    info.vid,
                    info.pid
                );
                let manufacturer = info.manufacturer.as_deref().unwrap_or("Unknown");
                (desc, format!("USB - {}", manufacturer))
            }
            serialport::SerialPortType::BluetoothPort => {
                ("Bluetooth Serial".to_string(), "Bluetooth".to_string())
            }
            serialport::SerialPortType::PciPort => {
                ("PCI Serial".to_string(), "PCI".to_string())
            }
            serialport::SerialPortType::Unknown => {
                ("Unknown".to_string(), "System".to_string())
            }
        };

        println!(
            "  {:<20} {:<30} {}",
            port.port_name.yellow(),
            description,
            port_type.dimmed()
        );
    }

    println!();
    println!("  Found {} port(s)", ports.len());

    Ok(())
}

fn auto_detect_port(verbose: bool) -> Result<String> {
    if verbose {
        println!("{} Auto-detecting serial port...", "[monitor]".cyan());
    }

    let ports = serialport::available_ports().context("Failed to enumerate serial ports")?;

    // Priority: USB ports with common debug adapter VIDs
    let usb_debug_vids = [
        0x0403, // FTDI
        0x10C4, // Silicon Labs CP210x
        0x067B, // Prolific
        0x1A86, // QinHeng CH340/CH341
        0x2341, // Arduino
    ];

    // First, try to find a known debug adapter
    for port in &ports {
        if let serialport::SerialPortType::UsbPort(info) = &port.port_type {
            if usb_debug_vids.contains(&info.vid) {
                if verbose {
                    println!(
                        "  Found debug adapter: {} ({})",
                        port.port_name.green(),
                        info.product.as_deref().unwrap_or("Unknown")
                    );
                }
                return Ok(port.port_name.clone());
            }
        }
    }

    // Fall back to first USB port
    for port in &ports {
        if matches!(port.port_type, serialport::SerialPortType::UsbPort(_)) {
            if verbose {
                println!("  Using first USB port: {}", port.port_name.green());
            }
            return Ok(port.port_name.clone());
        }
    }

    // Fall back to first available port
    if let Some(port) = ports.first() {
        if verbose {
            println!("  Using first available port: {}", port.port_name.yellow());
        }
        return Ok(port.port_name.clone());
    }

    bail!("No serial ports found. Connect a device and try again.");
}

fn connect_serial(port: &str, args: &MonitorArgs) -> Result<Box<dyn SerialPort>> {
    let data_bits = data_bits_from_u8(args.data_bits)?;
    let stop_bits = stop_bits_from_u8(args.stop_bits)?;

    let mut serial = serialport::new(port, args.baud)
        .data_bits(data_bits)
        .stop_bits(stop_bits)
        .parity(args.parity.into())
        .flow_control(args.flow_control.into())
        .timeout(Duration::from_millis(100))
        .open()
        .with_context(|| format!("Failed to open serial port: {}", port))?;

    // Toggle DTR to reset device if requested
    if args.reset {
        println!("  Toggling DTR to reset device...");
        serial.write_data_terminal_ready(false)?;
        std::thread::sleep(Duration::from_millis(100));
        serial.write_data_terminal_ready(true)?;
        std::thread::sleep(Duration::from_millis(100));
    }

    println!(
        "{} Connected to {} @ {} baud ({}N{})",
        "OK".green(),
        port.yellow(),
        args.baud,
        args.data_bits,
        args.stop_bits
    );

    Ok(serial)
}

fn run_monitor_loop(serial: &mut Box<dyn SerialPort>, args: &MonitorArgs) -> Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    // Setup Ctrl+C handler
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).ok();

    // Open log file if requested
    let mut log_file: Option<BufWriter<File>> = if let Some(ref log_path) = args.log {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("Failed to open log file: {}", log_path.display()))?;
        Some(BufWriter::new(file))
    } else {
        None
    };

    // State for toggle options
    let mut show_timestamps = args.timestamp;
    let mut show_hex = args.hex;
    let mut line_buffer = String::new();

    // Enable raw mode for keyboard input
    enable_raw_mode().context("Failed to enable raw terminal mode")?;

    let start_time = std::time::Instant::now();
    let mut serial_buf = [0u8; 256];

    let result = (|| -> Result<()> {
        while running.load(Ordering::SeqCst) {
            // Check timeout
            if let Some(timeout_secs) = args.timeout {
                if start_time.elapsed().as_secs() >= timeout_secs {
                    println!();
                    println!("{} Timeout reached ({} seconds)", "[monitor]".cyan(), timeout_secs);
                    break;
                }
            }

            // Check for keyboard input
            if event::poll(Duration::from_millis(10))? {
                if let Event::Key(key_event) = event::read()? {
                    if let Some(action) = handle_key_event(key_event, &mut show_timestamps, &mut show_hex) {
                        match action {
                            KeyAction::Quit => {
                                println!();
                                println!("{} Exiting monitor", "[monitor]".cyan());
                                break;
                            }
                            KeyAction::SendBreak => {
                                serial.set_break()?;
                                std::thread::sleep(Duration::from_millis(250));
                                serial.clear_break()?;
                                eprintln!("\r{} Break sent", "[monitor]".cyan());
                            }
                            KeyAction::Send(c) => {
                                let buf = [c as u8];
                                serial.write_all(&buf)?;
                                if !args.no_echo {
                                    print!("{}", c);
                                    std::io::stdout().flush()?;
                                }
                            }
                            KeyAction::TimestampToggled => {
                                eprintln!("\r{} Timestamps: {}", "[monitor]".cyan(),
                                    if show_timestamps { "ON".green() } else { "OFF".red() });
                            }
                            KeyAction::HexToggled => {
                                eprintln!("\r{} Hex mode: {}", "[monitor]".cyan(),
                                    if show_hex { "ON".green() } else { "OFF".red() });
                            }
                        }
                    }
                }
            }

            // Read from serial port
            match serial.read(&mut serial_buf) {
                Ok(n) if n > 0 => {
                    let data = &serial_buf[..n];

                    // Write to log file
                    if let Some(ref mut log) = log_file {
                        log.write_all(data)?;
                    }

                    // Display data
                    if show_hex {
                        display_hex(data, show_timestamps);
                    } else {
                        for &byte in data {
                            if byte == b'\n' {
                                // End of line
                                if show_timestamps {
                                    let now = chrono::Local::now();
                                    print!("\r{} ", now.format("%H:%M:%S%.3f").to_string().dimmed());
                                }
                                println!("{}", line_buffer);

                                // Check for exit pattern
                                if let Some(ref pattern) = args.exit_on {
                                    if line_buffer.contains(pattern) {
                                        println!();
                                        println!(
                                            "{} Exit pattern '{}' matched",
                                            "[monitor]".cyan(),
                                            pattern.green()
                                        );
                                        return Ok(());
                                    }
                                }

                                line_buffer.clear();
                            } else if byte == b'\r' {
                                // Ignore carriage return
                            } else if byte.is_ascii() {
                                line_buffer.push(byte as char);
                            } else {
                                // Non-ASCII: show hex
                                line_buffer.push_str(&format!("\\x{:02x}", byte));
                            }
                        }
                    }

                    std::io::stdout().flush()?;
                }
                Ok(_) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => {
                    eprintln!("\r{} Serial error: {}", "[error]".red(), e);
                    break;
                }
            }
        }

        Ok(())
    })();

    // Restore terminal
    disable_raw_mode().ok();

    // Flush log file
    if let Some(ref mut log) = log_file {
        log.flush()?;
        if let Some(ref log_path) = args.log {
            println!("Log saved to: {}", log_path.display().to_string().yellow());
        }
    }

    println!();
    println!("{}", "--- End of session ---".dimmed());

    result
}

enum KeyAction {
    Quit,
    SendBreak,
    Send(char),
    TimestampToggled,
    HexToggled,
}

fn handle_key_event(event: KeyEvent, timestamps: &mut bool, hex: &mut bool) -> Option<KeyAction> {
    match (event.code, event.modifiers) {
        // Ctrl+Q: Quit
        (KeyCode::Char('q'), KeyModifiers::CONTROL) => Some(KeyAction::Quit),
        // Ctrl+C: Also quit
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => Some(KeyAction::Quit),
        // Ctrl+B: Send break
        (KeyCode::Char('b'), KeyModifiers::CONTROL) => Some(KeyAction::SendBreak),
        // Ctrl+T: Toggle timestamps
        (KeyCode::Char('t'), KeyModifiers::CONTROL) => {
            *timestamps = !*timestamps;
            Some(KeyAction::TimestampToggled)
        }
        // Ctrl+H: Toggle hex mode
        (KeyCode::Char('h'), KeyModifiers::CONTROL) => {
            *hex = !*hex;
            Some(KeyAction::HexToggled)
        }
        // Regular character
        (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => Some(KeyAction::Send(c)),
        // Enter
        (KeyCode::Enter, _) => Some(KeyAction::Send('\r')),
        // Backspace
        (KeyCode::Backspace, _) => Some(KeyAction::Send('\x08')),
        // Tab
        (KeyCode::Tab, _) => Some(KeyAction::Send('\t')),
        // Escape
        (KeyCode::Esc, _) => Some(KeyAction::Send('\x1b')),
        _ => None,
    }
}

fn display_hex(data: &[u8], with_timestamp: bool) {
    use std::io::Write;

    let now = chrono::Local::now();

    for chunk in data.chunks(16) {
        if with_timestamp {
            print!("{} ", now.format("%H:%M:%S%.3f").to_string().dimmed());
        }

        // Hex dump
        for (i, byte) in chunk.iter().enumerate() {
            if i == 8 {
                print!(" ");
            }
            print!("{:02x} ", byte);
        }

        // Padding for incomplete lines
        if chunk.len() < 16 {
            for i in chunk.len()..16 {
                if i == 8 {
                    print!(" ");
                }
                print!("   ");
            }
        }

        print!(" |");

        // ASCII representation
        for byte in chunk {
            if byte.is_ascii_graphic() || *byte == b' ' {
                print!("{}", *byte as char);
            } else {
                print!(".");
            }
        }

        println!("|");
    }

    std::io::stdout().flush().ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_bits_conversion() {
        assert!(data_bits_from_u8(8).is_ok());
        assert!(data_bits_from_u8(7).is_ok());
        assert!(data_bits_from_u8(6).is_ok());
        assert!(data_bits_from_u8(5).is_ok());
        assert!(data_bits_from_u8(9).is_err());
    }

    #[test]
    fn test_stop_bits_conversion() {
        assert!(stop_bits_from_u8(1).is_ok());
        assert!(stop_bits_from_u8(2).is_ok());
        assert!(stop_bits_from_u8(3).is_err());
    }

    #[test]
    fn test_parity_conversion() {
        assert_eq!(SerialParity::None, Parity::None.into());
        assert_eq!(SerialParity::Odd, Parity::Odd.into());
        assert_eq!(SerialParity::Even, Parity::Even.into());
    }

    #[test]
    fn test_flow_control_conversion() {
        assert_eq!(SerialFlowControl::None, FlowControl::None.into());
        assert_eq!(SerialFlowControl::Hardware, FlowControl::Hardware.into());
        assert_eq!(SerialFlowControl::Software, FlowControl::Software.into());
    }
}
