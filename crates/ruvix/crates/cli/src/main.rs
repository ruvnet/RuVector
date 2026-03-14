//! RuVix CLI - Host-side tooling for the RuVix Cognition Kernel
//!
//! This CLI provides commands for building, flashing, configuring, and monitoring
//! RuVix kernel images on AArch64 bare-metal targets.

use clap::{Parser, Subcommand};
use colored::Colorize;

mod commands;

use commands::{build, config, dtb, flash, keys, monitor, security};

/// RuVix - Cognition Kernel CLI
///
/// Host-side tooling for building, flashing, and managing RuVix bare-metal
/// kernel images. Supports secure boot, key management, DTB validation,
/// and serial monitoring.
#[derive(Parser)]
#[command(name = "ruvix")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
#[command(after_help = "\
EXAMPLES:
    Build a release kernel with secure boot:
        ruvix build --release --secure-boot --target aarch64-unknown-none

    Flash to a Raspberry Pi 4:
        ruvix flash --device /dev/sdb --image target/kernel8.img

    Generate signing keys:
        ruvix keys generate --output keys/

    Monitor serial output:
        ruvix monitor --port /dev/ttyUSB0 --baud 115200

    Run security audit:
        ruvix security audit --depth full

For more information, see: https://github.com/ruvnet/ruvector
")]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Suppress all output except errors
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Output format (text, json)
    #[arg(long, global = true, default_value = "text")]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// Build kernel image for bare-metal targets
    ///
    /// Compiles the RuVix kernel for the specified AArch64 target with
    /// optional secure boot signing and feature flags.
    #[command(after_help = "\
EXAMPLES:
    Build debug kernel:
        ruvix build

    Build release with secure boot:
        ruvix build --release --secure-boot

    Build for specific target with features:
        ruvix build --target aarch64-unknown-none --features \"smp,net\"

    Build with custom linker script:
        ruvix build --linker-script custom.ld
")]
    Build(build::BuildArgs),

    /// Flash kernel image to device
    ///
    /// Writes the compiled kernel image and optional DTB to the target
    /// device's boot partition. Supports SD cards and USB boot devices.
    #[command(after_help = "\
EXAMPLES:
    Flash to SD card:
        ruvix flash --device /dev/sdb --image kernel8.img

    Flash with DTB:
        ruvix flash --device /dev/sdb --image kernel8.img --dtb bcm2711-rpi-4-b.dtb

    Dry run (verify without writing):
        ruvix flash --device /dev/sdb --image kernel8.img --dry-run
")]
    Flash(flash::FlashArgs),

    /// Manage kernel configuration
    ///
    /// Get, set, or list RuVix kernel configuration options. Configuration
    /// can be stored in TOML or JSON format.
    Config {
        #[command(subcommand)]
        action: config::ConfigAction,
    },

    /// Key management for secure boot
    ///
    /// Generate, sign, verify, and manage cryptographic keys used for
    /// secure boot verification and kernel signing.
    Keys {
        #[command(subcommand)]
        action: keys::KeysAction,
    },

    /// Device Tree Blob (DTB) operations
    ///
    /// Validate, inspect, and dump Device Tree Blob files used for
    /// hardware description on ARM platforms.
    Dtb {
        #[command(subcommand)]
        action: dtb::DtbAction,
    },

    /// Serial/UART monitor
    ///
    /// Connect to the target device's serial port for real-time console
    /// output and kernel debugging.
    #[command(after_help = "\
EXAMPLES:
    Basic serial monitor:
        ruvix monitor --port /dev/ttyUSB0

    Custom baud rate:
        ruvix monitor --port /dev/ttyUSB0 --baud 921600

    Log to file:
        ruvix monitor --port /dev/ttyUSB0 --log serial.log

    Auto-detect port:
        ruvix monitor --auto
")]
    Monitor(monitor::MonitorArgs),

    /// Security audit and scanning
    ///
    /// Run security audits on kernel configuration, analyze potential
    /// vulnerabilities, and generate security reports.
    Security {
        #[command(subcommand)]
        action: security::SecurityAction,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Set up colored output
    if cli.quiet {
        colored::control::set_override(false);
    }

    let result = match cli.command {
        Commands::Build(args) => build::execute(args, cli.verbose),
        Commands::Flash(args) => flash::execute(args, cli.verbose),
        Commands::Config { action } => config::execute(action, cli.verbose),
        Commands::Keys { action } => keys::execute(action, cli.verbose),
        Commands::Dtb { action } => dtb::execute(action, cli.verbose),
        Commands::Monitor(args) => monitor::execute(args, cli.verbose),
        Commands::Security { action } => security::execute(action, cli.verbose),
    };

    if let Err(ref e) = result {
        if !cli.quiet {
            eprintln!("{}: {}", "error".red().bold(), e);
            if cli.verbose {
                // Print error chain
                let mut source = e.source();
                while let Some(cause) = source {
                    eprintln!("  {}: {}", "caused by".yellow(), cause);
                    source = cause.source();
                }
            }
        }
        std::process::exit(1);
    }

    Ok(())
}
