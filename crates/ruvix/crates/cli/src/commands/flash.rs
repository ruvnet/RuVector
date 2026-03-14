//! Flash command - write kernel images to target devices

use anyhow::{bail, Context, Result};
use clap::Args;
use colored::Colorize;
use std::path::PathBuf;

/// Arguments for the flash command
#[derive(Args, Debug)]
pub struct FlashArgs {
    /// Target device path (e.g., /dev/sdb)
    #[arg(
        short,
        long,
        help = "Device path (e.g., /dev/sdb, /dev/mmcblk0)",
        required_unless_present = "list_devices"
    )]
    pub device: Option<PathBuf>,

    /// Kernel image to flash
    #[arg(
        short,
        long,
        help = "Path to kernel image",
        required_unless_present = "list_devices"
    )]
    pub image: Option<PathBuf>,

    /// Device Tree Blob to include
    #[arg(long, help = "Path to DTB file")]
    pub dtb: Option<PathBuf>,

    /// Additional boot files to copy
    #[arg(long, value_delimiter = ',', help = "Additional boot files")]
    pub boot_files: Vec<PathBuf>,

    /// Perform dry run without actual write
    #[arg(long, help = "Simulate flash without writing")]
    pub dry_run: bool,

    /// Skip confirmation prompt
    #[arg(short = 'y', long, help = "Skip confirmation")]
    pub yes: bool,

    /// Verify after writing
    #[arg(long, default_value = "true", help = "Verify write integrity")]
    pub verify: bool,

    /// List available devices
    #[arg(long, help = "List available storage devices")]
    pub list_devices: bool,

    /// Boot partition number (for multi-partition devices)
    #[arg(long, default_value = "1", help = "Boot partition number")]
    pub partition: u32,

    /// Filesystem type for boot partition
    #[arg(long, default_value = "vfat", help = "Boot partition filesystem")]
    pub fs_type: String,
}

/// Execute the flash command
pub fn execute(args: FlashArgs, verbose: bool) -> Result<()> {
    // Handle list devices
    if args.list_devices {
        return list_available_devices(verbose);
    }

    // Validate required arguments
    let device = args.device.as_ref().context("Device path is required")?;
    let image = args.image.as_ref().context("Image path is required")?;

    if verbose {
        println!("{}", "Flash Configuration:".cyan().bold());
        println!("  Device:     {}", device.display().to_string().yellow());
        println!("  Image:      {}", image.display());
        if let Some(ref dtb) = args.dtb {
            println!("  DTB:        {}", dtb.display());
        }
        println!("  Partition:  {}", args.partition);
        println!("  Filesystem: {}", args.fs_type);
        println!("  Verify:     {}", args.verify);
        println!("  Dry Run:    {}", args.dry_run);
        println!();
    }

    // Step 1: Validate inputs
    println!("{} Validating inputs...", "[1/5]".cyan());
    validate_inputs(&args)?;

    // Step 2: Check device
    println!("{} Checking device...", "[2/5]".cyan());
    check_device(device)?;

    // Step 3: Confirm (unless --yes)
    if !args.yes && !args.dry_run {
        println!("{} Confirming operation...", "[3/5]".cyan());
        if !confirm_flash(device)? {
            println!("{}", "Aborted by user".yellow());
            return Ok(());
        }
    } else {
        println!("{} Skipping confirmation", "[3/5]".cyan());
    }

    // Step 4: Flash
    println!("{} Writing image to device...", "[4/5]".cyan());
    flash_image(&args)?;

    // Step 5: Verify
    if args.verify && !args.dry_run {
        println!("{} Verifying write...", "[5/5]".cyan());
        verify_flash(&args)?;
    } else {
        println!("{} Skipping verification", "[5/5]".cyan());
    }

    // Report success
    println!(
        "\n{} Successfully flashed {} to {}",
        "SUCCESS".green().bold(),
        image.display().to_string().yellow(),
        device.display().to_string().yellow()
    );

    Ok(())
}

fn list_available_devices(_verbose: bool) -> Result<()> {
    println!("{}", "Available Storage Devices:".cyan().bold());
    println!();

    // Stub: List common device paths
    let devices = [
        ("/dev/sda", "256GB", "Internal SSD", false),
        ("/dev/sdb", "32GB", "USB Flash Drive", true),
        ("/dev/mmcblk0", "16GB", "SD Card", true),
    ];

    println!(
        "  {:<15} {:<10} {:<20} {}",
        "DEVICE".bold(),
        "SIZE".bold(),
        "DESCRIPTION".bold(),
        "REMOVABLE".bold()
    );
    println!("  {}", "-".repeat(60));

    for (path, size, desc, removable) in devices {
        let removable_str = if removable {
            "Yes".green()
        } else {
            "No".dimmed()
        };
        println!("  {:<15} {:<10} {:<20} {}", path, size, desc, removable_str);
    }

    println!();
    println!(
        "  {} This is stub output. Real implementation would query /sys/block/",
        "[stub]".yellow()
    );

    Ok(())
}

fn validate_inputs(args: &FlashArgs) -> Result<()> {
    if let Some(ref image) = args.image {
        println!("  {} Checking image exists: {}", "[stub]".yellow(), image.display());
        // In real implementation: check file exists and is readable
    }

    if let Some(ref dtb) = args.dtb {
        println!("  {} Checking DTB exists: {}", "[stub]".yellow(), dtb.display());
    }

    for boot_file in &args.boot_files {
        println!("  {} Checking boot file: {}", "[stub]".yellow(), boot_file.display());
    }

    Ok(())
}

fn check_device(device: &PathBuf) -> Result<()> {
    println!("  {} Checking device: {}", "[stub]".yellow(), device.display());
    println!("  {} Verifying device is not mounted", "[stub]".yellow());
    println!("  {} Checking device permissions", "[stub]".yellow());

    // Stub warnings
    if device.to_string_lossy().contains("sda") {
        println!(
            "  {} Device appears to be a system disk!",
            "WARNING".red().bold()
        );
    }

    Ok(())
}

fn confirm_flash(device: &PathBuf) -> Result<bool> {
    println!(
        "\n  {} All data on {} will be ERASED!",
        "WARNING".red().bold(),
        device.display().to_string().yellow()
    );
    println!("  {} Would prompt: Continue? [y/N]", "[stub]".yellow());
    println!("  {} Assuming 'yes' for stub implementation", "[stub]".yellow());
    Ok(true)
}

fn flash_image(args: &FlashArgs) -> Result<()> {
    let device = args.device.as_ref().unwrap();
    let image = args.image.as_ref().unwrap();

    if args.dry_run {
        println!(
            "  {} DRY RUN: Would write {} to {}",
            "[stub]".yellow(),
            image.display(),
            device.display()
        );
    } else {
        println!(
            "  {} Would write {} to {}",
            "[stub]".yellow(),
            image.display(),
            device.display()
        );
    }

    // Simulate progress
    println!("  {} Writing kernel image...", "[stub]".yellow());

    if let Some(ref dtb) = args.dtb {
        println!("  {} Writing DTB: {}", "[stub]".yellow(), dtb.display());
    }

    for boot_file in &args.boot_files {
        println!("  {} Writing boot file: {}", "[stub]".yellow(), boot_file.display());
    }

    println!("  {} Syncing filesystem...", "[stub]".yellow());

    Ok(())
}

fn verify_flash(args: &FlashArgs) -> Result<()> {
    let device = args.device.as_ref().unwrap();
    let image = args.image.as_ref().unwrap();

    println!(
        "  {} Would verify {} matches {}",
        "[stub]".yellow(),
        device.display(),
        image.display()
    );
    println!("  {} Checksum verification: OK", "[stub]".yellow());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_devices() {
        let result = list_available_devices(false);
        assert!(result.is_ok());
    }
}
