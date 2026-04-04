//! Climate Teleconnection Consciousness Explorer
//!
//! Applies IIT Phi to climate mode interactions to study how large-scale
//! climate oscillations form integrated information systems. Compares
//! neutral vs El Nino active conditions.

mod analysis;
mod data;
mod report;

fn main() {
    println!("+==========================================================+");
    println!("|  Climate Teleconnection Consciousness Explorer           |");
    println!("|  IIT 4.0 Phi Analysis of Climate Mode Interactions       |");
    println!("+==========================================================+");

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let null_samples = parse_arg(&args, "--null-samples", 50usize);
    let output = parse_str_arg(&args, "--output", "climate_report.svg");

    println!("\nConfiguration:");
    println!("  Climate indices: 7 (ENSO, NAO, PDO, AMO, IOD, SAM, QBO)");
    println!("  Null samples:    {}", null_samples);
    println!("  Output:          {}", output);

    // Step 1: Build climate correlation data
    println!("\n=== Step 1: Building Climate Mode Correlation Data ===");
    let neutral = data::build_neutral_correlations();
    let elnino = data::build_elnino_correlations();
    println!("  Neutral baseline: {} climate indices", neutral.n_indices);
    println!("  El Nino active:   {} climate indices", elnino.n_indices);

    // Step 2: Construct TPMs
    println!("\n=== Step 2: Constructing Transition Probability Matrices ===");
    let neutral_tpm = data::correlation_to_tpm(&neutral);
    let elnino_tpm = data::correlation_to_tpm(&elnino);
    println!("  Neutral TPM: {}x{}", neutral_tpm.size, neutral_tpm.size);
    println!("  El Nino TPM: {}x{}", elnino_tpm.size, elnino_tpm.size);

    // Step 3: Run analysis
    println!("\n=== Step 3: Consciousness Analysis ===");
    let results = analysis::run_analysis(
        &neutral, &neutral_tpm,
        &elnino, &elnino_tpm,
        null_samples,
    );

    // Step 4: Print report
    println!("\n=== Step 4: Results ===");
    report::print_summary(&results);

    // Step 5: Generate SVG
    let svg = report::generate_svg(&results, &neutral);
    std::fs::write(output, &svg).expect("Failed to write SVG report");
    println!(
        "\nSVG report saved to: {}",
        parse_str_arg(&args, "--output", "climate_report.svg")
    );

    // Final verdict
    println!("\n+==========================================================+");
    if results.elnino_increases_phi {
        println!("|  RESULT: El Nino INCREASES integrated information in     |");
        println!("|  the climate system -- teleconnections strengthen.       |");
    } else {
        println!("|  RESULT: El Nino does NOT increase integrated            |");
        println!("|  information -- system remains loosely coupled.          |");
    }
    if results.pacific_most_integrated {
        println!("|  The Pacific basin (ENSO, PDO, IOD) is the most          |");
        println!("|  integrated climate subsystem.                           |");
    }
    println!("+==========================================================+");
}

fn parse_arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == name)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn parse_str_arg<'a>(args: &'a [String], name: &str, default: &'a str) -> &'a str {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].as_str())
        .unwrap_or(default)
}
