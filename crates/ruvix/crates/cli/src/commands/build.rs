//! Build command - compile RuVix kernel images

use anyhow::{Context, Result};
use clap::Args;
use colored::Colorize;
use std::path::PathBuf;

/// Arguments for the build command
#[derive(Args, Debug)]
pub struct BuildArgs {
    /// Target triple for compilation
    #[arg(
        short,
        long,
        default_value = "aarch64-unknown-none",
        help = "Target architecture triple"
    )]
    pub target: String,

    /// Build in release mode with optimizations
    #[arg(short, long, help = "Enable release mode optimizations")]
    pub release: bool,

    /// Enable secure boot signing
    #[arg(
        long,
        help = "Sign kernel with secure boot keys",
        requires = "signing_key"
    )]
    pub secure_boot: bool,

    /// Path to signing key for secure boot
    #[arg(long, help = "Path to Ed25519 signing key")]
    pub signing_key: Option<PathBuf>,

    /// Comma-separated list of features to enable
    #[arg(
        short = 'F',
        long,
        value_delimiter = ',',
        help = "Features to enable (comma-separated)"
    )]
    pub features: Vec<String>,

    /// Output directory for built artifacts
    #[arg(short, long, default_value = "target", help = "Output directory")]
    pub output: PathBuf,

    /// Custom linker script
    #[arg(long, help = "Path to custom linker script")]
    pub linker_script: Option<PathBuf>,

    /// Generate position-independent code
    #[arg(long, help = "Enable position-independent code")]
    pub pic: bool,

    /// Strip debug symbols from output
    #[arg(long, help = "Strip debug symbols")]
    pub strip: bool,

    /// Clean build (remove target directory first)
    #[arg(long, help = "Clean before building")]
    pub clean: bool,

    /// Number of parallel jobs
    #[arg(short, long, help = "Number of parallel jobs")]
    pub jobs: Option<u32>,
}

/// Execute the build command
pub fn execute(args: BuildArgs, verbose: bool) -> Result<()> {
    if verbose {
        println!("{}", "Build Configuration:".cyan().bold());
        println!("  Target:       {}", args.target.yellow());
        println!("  Release:      {}", args.release);
        println!("  Secure Boot:  {}", args.secure_boot);
        if let Some(ref key) = args.signing_key {
            println!("  Signing Key:  {}", key.display());
        }
        if !args.features.is_empty() {
            println!("  Features:     {}", args.features.join(", ").green());
        }
        println!("  Output:       {}", args.output.display());
        if let Some(ref script) = args.linker_script {
            println!("  Linker:       {}", script.display());
        }
        println!();
    }

    // Clean if requested
    if args.clean {
        println!("{} Cleaning build directory...", "[1/4]".cyan());
        clean_build_dir(&args.output)?;
    }

    // Step 1: Validate environment
    println!("{} Validating build environment...", "[1/4]".cyan());
    validate_environment(&args.target)?;

    // Step 2: Configure build
    println!("{} Configuring build...", "[2/4]".cyan());
    let build_config = configure_build(&args)?;
    if verbose {
        println!("  {}", format!("{:?}", build_config).dimmed());
    }

    // Step 3: Compile
    println!("{} Compiling kernel...", "[3/4]".cyan());
    compile_kernel(&args, &build_config)?;

    // Step 4: Post-process (signing, stripping)
    println!("{} Post-processing...", "[4/4]".cyan());
    if args.secure_boot {
        sign_kernel(&args)?;
    }
    if args.strip {
        strip_symbols(&args)?;
    }

    // Report success
    let output_file = get_output_path(&args);
    println!(
        "\n{} Built kernel image: {}",
        "SUCCESS".green().bold(),
        output_file.display().to_string().yellow()
    );

    Ok(())
}

fn clean_build_dir(output: &PathBuf) -> Result<()> {
    println!(
        "  {} Would remove: {}",
        "[stub]".yellow(),
        output.display()
    );
    Ok(())
}

fn validate_environment(target: &str) -> Result<()> {
    println!(
        "  {} Checking for rustup target: {}",
        "[stub]".yellow(),
        target
    );
    println!("  {} Checking for cargo", "[stub]".yellow());
    println!("  {} Checking for llvm-objcopy", "[stub]".yellow());
    Ok(())
}

#[derive(Debug)]
struct BuildConfig {
    cargo_args: Vec<String>,
    rustflags: Vec<String>,
}

fn configure_build(args: &BuildArgs) -> Result<BuildConfig> {
    let mut cargo_args = vec![
        "build".to_string(),
        "--target".to_string(),
        args.target.clone(),
    ];

    if args.release {
        cargo_args.push("--release".to_string());
    }

    if let Some(jobs) = args.jobs {
        cargo_args.push("-j".to_string());
        cargo_args.push(jobs.to_string());
    }

    for feature in &args.features {
        cargo_args.push("--features".to_string());
        cargo_args.push(feature.clone());
    }

    let mut rustflags = Vec::new();

    if let Some(ref script) = args.linker_script {
        rustflags.push(format!("-C link-arg=-T{}", script.display()));
    }

    if args.pic {
        rustflags.push("-C relocation-model=pic".to_string());
    }

    println!(
        "  {} cargo {}",
        "[stub]".yellow(),
        cargo_args.join(" ")
    );

    Ok(BuildConfig {
        cargo_args,
        rustflags,
    })
}

fn compile_kernel(args: &BuildArgs, config: &BuildConfig) -> Result<()> {
    println!(
        "  {} Would run: cargo {}",
        "[stub]".yellow(),
        config.cargo_args.join(" ")
    );
    if !config.rustflags.is_empty() {
        println!(
            "  {} RUSTFLAGS: {}",
            "[stub]".yellow(),
            config.rustflags.join(" ")
        );
    }
    Ok(())
}

fn sign_kernel(args: &BuildArgs) -> Result<()> {
    if let Some(ref key) = args.signing_key {
        println!(
            "  {} Would sign with key: {}",
            "[stub]".yellow(),
            key.display()
        );
    }
    Ok(())
}

fn strip_symbols(args: &BuildArgs) -> Result<()> {
    println!(
        "  {} Would strip debug symbols from output",
        "[stub]".yellow()
    );
    Ok(())
}

fn get_output_path(args: &BuildArgs) -> PathBuf {
    let profile = if args.release { "release" } else { "debug" };
    args.output
        .join(&args.target)
        .join(profile)
        .join("kernel8.img")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_args() {
        let args = BuildArgs {
            target: "aarch64-unknown-none".to_string(),
            release: false,
            secure_boot: false,
            signing_key: None,
            features: vec![],
            output: PathBuf::from("target"),
            linker_script: None,
            pic: false,
            strip: false,
            clean: false,
            jobs: None,
        };
        assert_eq!(args.target, "aarch64-unknown-none");
        assert!(!args.release);
    }
}
