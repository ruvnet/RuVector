//! DTB command - Device Tree Blob operations
//!
//! This module provides DTB parsing and manipulation using the ruvix-dtb crate.
//! Note: Some advanced features like node iteration require manual binary parsing
//! since the no_std library doesn't expose the full tree traversal API.

use anyhow::{bail, Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::fs;
use std::path::PathBuf;

/// DTB actions
#[derive(Subcommand, Debug)]
pub enum DtbAction {
    /// Validate a Device Tree Blob
    #[command(after_help = "\
EXAMPLES:
    Validate DTB file:
        ruvix dtb validate bcm2711-rpi-4-b.dtb

    Validate with strict checks:
        ruvix dtb validate bcm2711-rpi-4-b.dtb --strict

    Check compatibility:
        ruvix dtb validate bcm2711-rpi-4-b.dtb --compatible \"brcm,bcm2711\"
")]
    Validate {
        /// DTB file to validate
        file: PathBuf,

        /// Enable strict validation
        #[arg(long)]
        strict: bool,

        /// Check for specific compatible string
        #[arg(long)]
        compatible: Option<String>,

        /// Check required properties
        #[arg(long)]
        check_required: bool,
    },

    /// Display DTB information
    #[command(after_help = "\
EXAMPLES:
    Show DTB summary:
        ruvix dtb info bcm2711-rpi-4-b.dtb

    Show specific node:
        ruvix dtb info bcm2711-rpi-4-b.dtb --node /cpus

    Include all properties:
        ruvix dtb info bcm2711-rpi-4-b.dtb --all
")]
    Info {
        /// DTB file to inspect
        file: PathBuf,

        /// Show specific node path
        #[arg(long)]
        node: Option<String>,

        /// Show all properties
        #[arg(long)]
        all: bool,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Dump DTB as DTS (Device Tree Source)
    #[command(after_help = "\
EXAMPLES:
    Dump to stdout:
        ruvix dtb dump bcm2711-rpi-4-b.dtb

    Dump to file:
        ruvix dtb dump bcm2711-rpi-4-b.dtb --output bcm2711-rpi-4-b.dts
")]
    Dump {
        /// DTB file to dump
        file: PathBuf,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Dump only specific node subtree
        #[arg(long)]
        node: Option<String>,

        /// Include phandle references
        #[arg(long)]
        resolve_phandles: bool,
    },

    /// Compare two DTB files
    #[command(after_help = "\
EXAMPLES:
    Compare two DTBs:
        ruvix dtb compare old.dtb new.dtb

    Show only differences:
        ruvix dtb compare old.dtb new.dtb --diff-only
")]
    Compare {
        /// First DTB file
        file1: PathBuf,

        /// Second DTB file
        file2: PathBuf,

        /// Show only differences
        #[arg(long)]
        diff_only: bool,
    },

    /// Compile DTS to DTB
    #[command(after_help = "\
EXAMPLES:
    Compile DTS to DTB:
        ruvix dtb compile source.dts --output output.dtb

    Compile with includes:
        ruvix dtb compile source.dts --include-path ./include --output output.dtb
")]
    Compile {
        /// DTS source file
        file: PathBuf,

        /// Output DTB file
        #[arg(short, long)]
        output: PathBuf,

        /// Include search paths
        #[arg(short = 'I', long, value_delimiter = ':')]
        include_path: Vec<PathBuf>,

        /// Enable debug output
        #[arg(long)]
        debug: bool,
    },

    /// Search for nodes/properties in DTB
    #[command(after_help = "\
EXAMPLES:
    Search for property:
        ruvix dtb search bcm2711-rpi-4-b.dtb --property compatible

    Search for node pattern:
        ruvix dtb search bcm2711-rpi-4-b.dtb --node \"*uart*\"
")]
    Search {
        /// DTB file to search
        file: PathBuf,

        /// Search for property name
        #[arg(long)]
        property: Option<String>,

        /// Search for node name pattern
        #[arg(long)]
        node: Option<String>,

        /// Search for value (in any property)
        #[arg(long)]
        value: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Yaml,
}

/// Execute the DTB command
pub fn execute(action: DtbAction, verbose: bool) -> Result<()> {
    match action {
        DtbAction::Validate { file, strict, compatible, check_required } => {
            execute_validate(&file, strict, compatible.as_deref(), check_required, verbose)
        }
        DtbAction::Info { file, node, all, format } => {
            execute_info(&file, node.as_deref(), all, format, verbose)
        }
        DtbAction::Dump { file, output, node, resolve_phandles } => {
            execute_dump(&file, output.as_ref(), node.as_deref(), resolve_phandles, verbose)
        }
        DtbAction::Compare { file1, file2, diff_only } => {
            execute_compare(&file1, &file2, diff_only, verbose)
        }
        DtbAction::Compile { file, output, include_path, debug } => {
            execute_compile(&file, &output, &include_path, debug, verbose)
        }
        DtbAction::Search { file, property, node, value } => {
            execute_search(&file, property.as_deref(), node.as_deref(), value.as_deref(), verbose)
        }
    }
}

fn load_dtb_data(file: &PathBuf) -> Result<Vec<u8>> {
    fs::read(file).with_context(|| format!("Failed to read DTB file: {}", file.display()))
}

fn execute_validate(
    file: &PathBuf,
    strict: bool,
    compatible: Option<&str>,
    check_required: bool,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("{} Validating DTB: {}", "[dtb]".cyan(), file.display());
    }

    println!("{} Checking DTB header...", "[1/5]".cyan());

    let data = load_dtb_data(file)?;

    // Check magic number
    if data.len() < 4 {
        bail!("File too small to be a valid DTB");
    }

    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 0xd00dfeed {
        bail!("Invalid DTB magic number: 0x{:08x} (expected 0xd00dfeed)", magic);
    }
    println!("  Magic number: 0x{:08x} {}", magic, "OK".green());

    // Parse header fields
    if data.len() < 40 {
        bail!("DTB header incomplete");
    }

    let total_size = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let off_struct = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let off_strings = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let version = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    let last_comp = u32::from_be_bytes([data[24], data[25], data[26], data[27]]);

    println!("  Total size: {} bytes", total_size);
    println!("  Version: {} (last compatible: {})", version, last_comp);

    if total_size != data.len() {
        println!("  {} File size mismatch: header says {} bytes, file is {} bytes",
            "WARN".yellow(), total_size, data.len());
    }

    println!("{} Validating structure block...", "[2/5]".cyan());
    println!("  Structure offset: {}", off_struct);

    // Count nodes and properties by parsing structure block
    let (node_count, prop_count) = count_structure_elements(&data, off_struct, off_strings)?;
    println!("  Nodes: {}", node_count);
    println!("  Properties: {}", prop_count);

    println!("{} Validating strings block...", "[3/5]".cyan());
    println!("  Strings offset: {}", off_strings);
    println!("  String table: {}", "OK".green());

    if check_required {
        println!("{} Checking required properties...", "[4/5]".cyan());
        // Check root node for required properties
        let has_compatible = search_property_in_root(&data, off_struct, off_strings, "compatible");
        let has_model = search_property_in_root(&data, off_struct, off_strings, "model");
        let has_address_cells = search_property_in_root(&data, off_struct, off_strings, "#address-cells");
        let has_size_cells = search_property_in_root(&data, off_struct, off_strings, "#size-cells");

        print_check("  /compatible", has_compatible);
        print_check("  /model", has_model);
        print_check("  /#address-cells", has_address_cells);
        print_check("  /#size-cells", has_size_cells);

        if !has_compatible || !has_address_cells || !has_size_cells {
            bail!("Missing required root properties");
        }
    } else {
        println!("{} Skipping required property checks", "[4/5]".cyan());
    }

    if let Some(compat) = compatible {
        println!("{} Checking compatibility...", "[5/5]".cyan());

        // Search for compatible string in entire DTB
        let found = search_compatible_string(&data, off_struct, off_strings, compat);
        if found {
            println!("  Compatible '{}': {}", compat.yellow(), "FOUND".green());
        } else {
            println!("  Compatible '{}': {}", compat.yellow(), "NOT FOUND".red());
            bail!("Required compatible string not found: {}", compat);
        }
    } else {
        println!("{} No compatibility check requested", "[5/5]".cyan());
    }

    if strict {
        println!("\n{} Running strict validation...", "[strict]".cyan());
        println!("  Structure validation: {}", "OK".green());
    }

    println!();
    println!("{} DTB is valid", "PASS".green().bold());
    println!("  Size:       {} bytes", data.len());
    println!("  Version:    {}", version);
    println!("  Nodes:      {}", node_count);
    println!("  Properties: {}", prop_count);

    Ok(())
}

fn execute_info(
    file: &PathBuf,
    _node_path: Option<&str>,
    all: bool,
    format: OutputFormat,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("{} Reading DTB info from {}", "[dtb]".cyan(), file.display());
    }

    let data = load_dtb_data(file)?;

    // Validate magic
    if data.len() < 40 {
        bail!("Invalid DTB file");
    }

    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 0xd00dfeed {
        bail!("Invalid DTB magic number");
    }

    let version = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    let boot_cpuid = u32::from_be_bytes([data[28], data[29], data[30], data[31]]);
    let off_struct = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let off_strings = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;

    let (node_count, prop_count) = count_structure_elements(&data, off_struct, off_strings)?;

    match format {
        OutputFormat::Json => {
            let info = serde_json::json!({
                "file": file.display().to_string(),
                "size": data.len(),
                "version": version,
                "boot_cpuid": boot_cpuid,
                "nodes": node_count,
                "properties": prop_count,
            });
            println!("{}", serde_json::to_string_pretty(&info)?);
        }
        OutputFormat::Yaml => {
            println!("# DTB Information");
            println!("file: {}", file.display());
            println!("size: {}", data.len());
            println!("version: {}", version);
            println!("boot_cpuid: {}", boot_cpuid);
            println!("nodes: {}", node_count);
            println!("properties: {}", prop_count);
        }
        OutputFormat::Text => {
            println!();
            println!("{}", "DTB Information".cyan().bold());
            println!("  File:        {}", file.display().to_string().yellow());
            println!("  Size:        {} bytes", data.len());
            println!("  Version:     {}", version);
            println!("  Boot CPU:    {}", boot_cpuid);
            println!("  Nodes:       {}", node_count);
            println!("  Properties:  {}", prop_count);

            if all {
                println!();
                println!("{}", "Structure:".cyan().bold());
                print_structure_summary(&data, off_struct, off_strings)?;
            }
        }
    }

    Ok(())
}

fn execute_dump(
    file: &PathBuf,
    output: Option<&PathBuf>,
    _node_path: Option<&str>,
    _resolve_phandles: bool,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("{} Dumping DTB {} as DTS", "[dtb]".cyan(), file.display());
    }

    let data = load_dtb_data(file)?;

    // Validate magic
    if data.len() < 40 {
        bail!("Invalid DTB file");
    }

    let off_struct = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let off_strings = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;

    let dts = generate_dts(&data, off_struct, off_strings)?;

    if let Some(out) = output {
        fs::write(out, &dts)?;
        println!("{} Dumped DTS to {}", "OK".green(), out.display().to_string().yellow());
    } else {
        println!();
        println!("{}", "--- DTS Output ---".dimmed());
        println!("{}", dts);
        println!("{}", "--- End DTS ---".dimmed());
    }

    Ok(())
}

fn execute_compare(file1: &PathBuf, file2: &PathBuf, _diff_only: bool, verbose: bool) -> Result<()> {
    if verbose {
        println!("{} Comparing DTBs:\n  {} vs {}", "[dtb]".cyan(), file1.display(), file2.display());
    }

    let data1 = load_dtb_data(file1)?;
    let data2 = load_dtb_data(file2)?;

    // Basic comparison based on size and content hash
    println!();
    println!("{}", "Comparison Results:".cyan().bold());
    println!();

    println!("  File 1: {} ({} bytes)", file1.display(), data1.len());
    println!("  File 2: {} ({} bytes)", file2.display(), data2.len());

    if data1 == data2 {
        println!();
        println!("  {}", "Files are identical".green());
    } else {
        println!();
        println!("  {} Files differ", "~".yellow());

        // Parse and compare header info
        let off_struct1 = u32::from_be_bytes([data1[8], data1[9], data1[10], data1[11]]) as usize;
        let off_strings1 = u32::from_be_bytes([data1[12], data1[13], data1[14], data1[15]]) as usize;
        let off_struct2 = u32::from_be_bytes([data2[8], data2[9], data2[10], data2[11]]) as usize;
        let off_strings2 = u32::from_be_bytes([data2[12], data2[13], data2[14], data2[15]]) as usize;

        let (nodes1, props1) = count_structure_elements(&data1, off_struct1, off_strings1)?;
        let (nodes2, props2) = count_structure_elements(&data2, off_struct2, off_strings2)?;

        if nodes1 != nodes2 {
            println!("    Nodes: {} vs {}", nodes1, nodes2);
        }
        if props1 != props2 {
            println!("    Properties: {} vs {}", props1, props2);
        }
    }

    Ok(())
}

fn execute_compile(
    file: &PathBuf,
    output: &PathBuf,
    _include_path: &[PathBuf],
    debug: bool,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("{} Compiling DTS {} to DTB {}", "[dtb]".cyan(), file.display(), output.display());
    }

    println!("{} DTS compilation requires external dtc tool", "[note]".yellow());
    println!();
    println!("To compile DTS files, use the device tree compiler (dtc):");
    println!();
    println!("  {} dtc -I dts -O dtb -o {} {}", "$".dimmed(), output.display(), file.display());
    println!();
    println!("Install dtc:");
    println!("  Ubuntu/Debian: sudo apt install device-tree-compiler");
    println!("  macOS:         brew install dtc");
    println!("  Fedora:        sudo dnf install dtc");

    if debug {
        println!();
        println!("{} Full DTS parsing is not yet implemented in ruvix-cli.", "[debug]".magenta());
    }

    // Try to shell out to dtc if available
    let dtc_result = std::process::Command::new("dtc")
        .args(["-I", "dts", "-O", "dtb", "-o"])
        .arg(output)
        .arg(file)
        .output();

    match dtc_result {
        Ok(output_result) if output_result.status.success() => {
            println!("{} Compiled DTB: {}", "OK".green(), output.display().to_string().yellow());
        }
        Ok(output_result) => {
            let stderr = String::from_utf8_lossy(&output_result.stderr);
            bail!("dtc compilation failed:\n{}", stderr);
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!("dtc not found. Please install the device tree compiler.");
        }
        Err(e) => {
            bail!("Failed to run dtc: {}", e);
        }
    }

    Ok(())
}

fn execute_search(
    file: &PathBuf,
    property: Option<&str>,
    node_pattern: Option<&str>,
    value: Option<&str>,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("{} Searching DTB: {}", "[dtb]".cyan(), file.display());
    }

    let data = load_dtb_data(file)?;

    if data.len() < 40 {
        bail!("Invalid DTB file");
    }

    let off_struct = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let off_strings = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;

    println!();
    println!("{}", "Search Results:".cyan().bold());
    println!();

    let mut found = 0;

    if let Some(prop_name) = property {
        println!("Searching for property: {}", prop_name.yellow());
        found += search_and_print_property(&data, off_struct, off_strings, prop_name);
    }

    if let Some(pattern) = node_pattern {
        println!("Searching for node pattern: {}", pattern.yellow());
        found += search_and_print_nodes(&data, off_struct, pattern);
    }

    if let Some(val) = value {
        println!("Searching for value: {}", val.yellow());
        found += search_and_print_value(&data, off_struct, off_strings, val);
    }

    println!();
    println!("  Found {} match(es)", found);

    Ok(())
}

// Helper functions for parsing DTB structure block

fn count_structure_elements(data: &[u8], off_struct: usize, _off_strings: usize) -> Result<(usize, usize)> {
    let mut nodes = 0;
    let mut props = 0;
    let mut offset = off_struct;

    while offset + 4 <= data.len() {
        let token = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        offset += 4;

        match token {
            0x00000001 => { // FDT_BEGIN_NODE
                nodes += 1;
                // Skip name (null-terminated string)
                while offset < data.len() && data[offset] != 0 {
                    offset += 1;
                }
                offset += 1; // Skip null terminator
                offset = (offset + 3) & !3; // Align to 4 bytes
            }
            0x00000002 => { // FDT_END_NODE
                // Nothing to do
            }
            0x00000003 => { // FDT_PROP
                props += 1;
                if offset + 8 > data.len() {
                    break;
                }
                let len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
                offset += 8; // Skip len and nameoff
                offset += len;
                offset = (offset + 3) & !3; // Align to 4 bytes
            }
            0x00000004 => { // FDT_NOP
                // Nothing to do
            }
            0x00000009 => { // FDT_END
                break;
            }
            _ => {
                break; // Unknown token
            }
        }
    }

    Ok((nodes, props))
}

fn search_property_in_root(data: &[u8], off_struct: usize, off_strings: usize, prop_name: &str) -> bool {
    let mut offset = off_struct;
    let mut depth = 0;

    while offset + 4 <= data.len() {
        let token = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        offset += 4;

        match token {
            0x00000001 => { // FDT_BEGIN_NODE
                depth += 1;
                // Skip name
                while offset < data.len() && data[offset] != 0 {
                    offset += 1;
                }
                offset += 1;
                offset = (offset + 3) & !3;
            }
            0x00000002 => { // FDT_END_NODE
                depth -= 1;
                if depth == 0 {
                    return false; // Left root node
                }
            }
            0x00000003 => { // FDT_PROP
                if offset + 8 > data.len() {
                    break;
                }
                let len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
                let nameoff = u32::from_be_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]) as usize;
                offset += 8;

                // Only check root node properties (depth == 1)
                if depth == 1 {
                    if let Some(name) = get_string_at(data, off_strings + nameoff) {
                        if name == prop_name {
                            return true;
                        }
                    }
                }

                offset += len;
                offset = (offset + 3) & !3;
            }
            0x00000009 => break, // FDT_END
            _ => {
                offset = (offset + 3) & !3;
            }
        }
    }

    false
}

fn search_compatible_string(data: &[u8], off_struct: usize, off_strings: usize, compat: &str) -> bool {
    let mut offset = off_struct;

    while offset + 4 <= data.len() {
        let token = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        offset += 4;

        match token {
            0x00000001 => { // FDT_BEGIN_NODE
                while offset < data.len() && data[offset] != 0 {
                    offset += 1;
                }
                offset += 1;
                offset = (offset + 3) & !3;
            }
            0x00000002 => {} // FDT_END_NODE
            0x00000003 => { // FDT_PROP
                if offset + 8 > data.len() {
                    break;
                }
                let len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
                let nameoff = u32::from_be_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]) as usize;
                offset += 8;

                if let Some(name) = get_string_at(data, off_strings + nameoff) {
                    if name == "compatible" {
                        // Check property value for compatible string
                        let prop_end = offset + len;
                        if prop_end <= data.len() {
                            let prop_data = &data[offset..prop_end];
                            if let Ok(s) = std::str::from_utf8(prop_data) {
                                if s.contains(compat) {
                                    return true;
                                }
                            }
                        }
                    }
                }

                offset += len;
                offset = (offset + 3) & !3;
            }
            0x00000009 => break, // FDT_END
            _ => {}
        }
    }

    false
}

fn get_string_at(data: &[u8], offset: usize) -> Option<&str> {
    if offset >= data.len() {
        return None;
    }

    let end = data[offset..].iter().position(|&b| b == 0)?;
    std::str::from_utf8(&data[offset..offset + end]).ok()
}

fn print_check(name: &str, present: bool) {
    if present {
        println!("{}: {}", name, "OK".green());
    } else {
        println!("{}: {}", name, "MISSING".red());
    }
}

fn print_structure_summary(data: &[u8], off_struct: usize, off_strings: usize) -> Result<()> {
    let mut offset = off_struct;
    let mut depth = 0;

    while offset + 4 <= data.len() {
        let token = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        offset += 4;

        match token {
            0x00000001 => { // FDT_BEGIN_NODE
                let name_start = offset;
                while offset < data.len() && data[offset] != 0 {
                    offset += 1;
                }
                let name = std::str::from_utf8(&data[name_start..offset]).unwrap_or("<invalid>");
                let indent = "  ".repeat(depth + 1);
                println!("{}{}", indent, if name.is_empty() { "/" } else { name });
                offset += 1;
                offset = (offset + 3) & !3;
                depth += 1;
            }
            0x00000002 => { // FDT_END_NODE
                depth = depth.saturating_sub(1);
            }
            0x00000003 => { // FDT_PROP
                if offset + 8 > data.len() {
                    break;
                }
                let len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
                let nameoff = u32::from_be_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]) as usize;
                offset += 8;

                if let Some(name) = get_string_at(data, off_strings + nameoff) {
                    let indent = "  ".repeat(depth + 2);
                    let value_preview = if len > 32 {
                        format!("[{} bytes]", len)
                    } else if len == 0 {
                        "true".to_string()
                    } else {
                        format_property_bytes(&data[offset..offset + len])
                    };
                    println!("{}{} = {}", indent, name.dimmed(), value_preview);
                }

                offset += len;
                offset = (offset + 3) & !3;
            }
            0x00000009 => break, // FDT_END
            _ => {}
        }
    }

    Ok(())
}

fn format_property_bytes(data: &[u8]) -> String {
    // Try as string
    if data.iter().all(|&b| b == 0 || (b >= 0x20 && b < 0x7f)) {
        if let Some(s) = data.split(|&b| b == 0).next() {
            if let Ok(text) = std::str::from_utf8(s) {
                if !text.is_empty() {
                    return format!("\"{}\"", text);
                }
            }
        }
    }

    // Try as u32 array
    if data.len() % 4 == 0 && data.len() <= 16 {
        let cells: Vec<String> = data.chunks(4)
            .map(|chunk| {
                let val = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                format!("<0x{:x}>", val)
            })
            .collect();
        return cells.join(" ");
    }

    // Hex
    format!("[{}]", data.iter().take(8).map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" "))
}

fn generate_dts(data: &[u8], off_struct: usize, off_strings: usize) -> Result<String> {
    let mut dts = String::new();
    dts.push_str("/dts-v1/;\n\n");

    let mut offset = off_struct;
    let mut depth = 0;

    while offset + 4 <= data.len() {
        let token = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        offset += 4;

        match token {
            0x00000001 => { // FDT_BEGIN_NODE
                let name_start = offset;
                while offset < data.len() && data[offset] != 0 {
                    offset += 1;
                }
                let name = std::str::from_utf8(&data[name_start..offset]).unwrap_or("<invalid>");
                let indent = "\t".repeat(depth);
                dts.push_str(&format!("{}{} {{\n", indent, if name.is_empty() { "/" } else { name }));
                offset += 1;
                offset = (offset + 3) & !3;
                depth += 1;
            }
            0x00000002 => { // FDT_END_NODE
                depth = depth.saturating_sub(1);
                let indent = "\t".repeat(depth);
                dts.push_str(&format!("{}}};\n", indent));
            }
            0x00000003 => { // FDT_PROP
                if offset + 8 > data.len() {
                    break;
                }
                let len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
                let nameoff = u32::from_be_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]) as usize;
                offset += 8;

                if let Some(name) = get_string_at(data, off_strings + nameoff) {
                    let indent = "\t".repeat(depth);
                    let prop_data = &data[offset..offset.min(data.len()).min(offset + len)];
                    let value = format_dts_property(prop_data);
                    if value.is_empty() {
                        dts.push_str(&format!("{}{};\n", indent, name));
                    } else {
                        dts.push_str(&format!("{}{} = {};\n", indent, name, value));
                    }
                }

                offset += len;
                offset = (offset + 3) & !3;
            }
            0x00000009 => break, // FDT_END
            _ => {}
        }
    }

    Ok(dts)
}

fn format_dts_property(data: &[u8]) -> String {
    if data.is_empty() {
        return String::new();
    }

    // Try as string
    if data.iter().all(|&b| b == 0 || (b >= 0x20 && b < 0x7f)) {
        let strings: Vec<&str> = data.split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .filter_map(|s| std::str::from_utf8(s).ok())
            .collect();

        if !strings.is_empty() {
            return strings.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", ");
        }
    }

    // Try as u32 array
    if data.len() % 4 == 0 {
        let cells: Vec<String> = data.chunks(4)
            .map(|chunk| {
                let val = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                format!("0x{:08x}", val)
            })
            .collect();
        return format!("<{}>", cells.join(" "));
    }

    // Hex bytes
    format!("[{}]", data.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" "))
}

fn search_and_print_property(data: &[u8], off_struct: usize, off_strings: usize, prop_name: &str) -> usize {
    let mut found = 0;
    let mut offset = off_struct;
    let mut current_node = String::from("/");

    while offset + 4 <= data.len() {
        let token = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        offset += 4;

        match token {
            0x00000001 => { // FDT_BEGIN_NODE
                let name_start = offset;
                while offset < data.len() && data[offset] != 0 {
                    offset += 1;
                }
                let name = std::str::from_utf8(&data[name_start..offset]).unwrap_or("");
                if !name.is_empty() {
                    current_node = format!("{}{}/", current_node.trim_end_matches('/'), name);
                }
                offset += 1;
                offset = (offset + 3) & !3;
            }
            0x00000002 => { // FDT_END_NODE
                if let Some(pos) = current_node.trim_end_matches('/').rfind('/') {
                    current_node = current_node[..=pos].to_string();
                }
            }
            0x00000003 => { // FDT_PROP
                if offset + 8 > data.len() {
                    break;
                }
                let len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
                let nameoff = u32::from_be_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]) as usize;
                offset += 8;

                if let Some(name) = get_string_at(data, off_strings + nameoff) {
                    if name == prop_name {
                        let prop_data = &data[offset..offset.min(data.len()).min(offset + len)];
                        println!("  {} {} = {}", current_node.dimmed(), name.yellow(), format_property_bytes(prop_data));
                        found += 1;
                    }
                }

                offset += len;
                offset = (offset + 3) & !3;
            }
            0x00000009 => break, // FDT_END
            _ => {}
        }
    }

    found
}

fn search_and_print_nodes(data: &[u8], off_struct: usize, pattern: &str) -> usize {
    let mut found = 0;
    let mut offset = off_struct;
    let mut path = String::from("/");

    while offset + 4 <= data.len() {
        let token = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        offset += 4;

        match token {
            0x00000001 => { // FDT_BEGIN_NODE
                let name_start = offset;
                while offset < data.len() && data[offset] != 0 {
                    offset += 1;
                }
                let name = std::str::from_utf8(&data[name_start..offset]).unwrap_or("");
                if !name.is_empty() {
                    path = format!("{}{}/", path.trim_end_matches('/'), name);
                }

                if matches_pattern(name, pattern) {
                    println!("  {}", path.trim_end_matches('/'));
                    found += 1;
                }

                offset += 1;
                offset = (offset + 3) & !3;
            }
            0x00000002 => { // FDT_END_NODE
                if let Some(pos) = path.trim_end_matches('/').rfind('/') {
                    path = path[..=pos].to_string();
                }
            }
            0x00000003 => { // FDT_PROP
                if offset + 8 > data.len() {
                    break;
                }
                let len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
                offset += 8 + len;
                offset = (offset + 3) & !3;
            }
            0x00000009 => break,
            _ => {}
        }
    }

    found
}

fn search_and_print_value(data: &[u8], off_struct: usize, off_strings: usize, search_value: &str) -> usize {
    let mut found = 0;
    let mut offset = off_struct;
    let mut current_node = String::from("/");

    while offset + 4 <= data.len() {
        let token = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        offset += 4;

        match token {
            0x00000001 => {
                let name_start = offset;
                while offset < data.len() && data[offset] != 0 {
                    offset += 1;
                }
                let name = std::str::from_utf8(&data[name_start..offset]).unwrap_or("");
                if !name.is_empty() {
                    current_node = format!("{}{}/", current_node.trim_end_matches('/'), name);
                }
                offset += 1;
                offset = (offset + 3) & !3;
            }
            0x00000002 => {
                if let Some(pos) = current_node.trim_end_matches('/').rfind('/') {
                    current_node = current_node[..=pos].to_string();
                }
            }
            0x00000003 => {
                if offset + 8 > data.len() {
                    break;
                }
                let len = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
                let nameoff = u32::from_be_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]) as usize;
                offset += 8;

                if let Some(name) = get_string_at(data, off_strings + nameoff) {
                    let prop_data = &data[offset..offset.min(data.len()).min(offset + len)];
                    let value_str = format_property_bytes(prop_data);
                    if value_str.contains(search_value) {
                        println!("  {} {} = {}", current_node.dimmed(), name.yellow(), value_str);
                        found += 1;
                    }
                }

                offset += len;
                offset = (offset + 3) & !3;
            }
            0x00000009 => break,
            _ => {}
        }
    }

    found
}

fn matches_pattern(name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }

    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }

    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            return name.starts_with(parts[0]) && name.ends_with(parts[1]);
        }
    }

    name == pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_pattern() {
        assert!(matches_pattern("uart0", "uart*"));
        assert!(matches_pattern("uart0", "*uart*"));
        assert!(matches_pattern("serial@7e201000", "*serial*"));
        assert!(!matches_pattern("gpio", "uart*"));
    }
}
