//! Quantum Circuit Consciousness Explorer
//!
//! Applies IIT Phi to quantum circuit measurement statistics to explore
//! the relationship between entanglement and integrated information.

mod analysis;
mod data;
mod report;

fn main() {
    println!("+==========================================================+");
    println!("|  Quantum Circuit Consciousness Explorer -- IIT 4.0       |");
    println!("|  Bridging entanglement and integrated information        |");
    println!("+==========================================================+");

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let output = parse_str_arg(&args, "--output", "quantum_report.svg");
    let depth = parse_arg(&args, "--depth", 5usize);

    println!("\nConfiguration:");
    println!("  Output:        {}", output);
    println!("  Random depth:  {}", depth);

    // Step 1: Generate quantum circuit TPMs
    println!("\n=== Step 1: Generating Quantum Circuit Data ===");
    let circuits = data::generate_all_circuits(depth);
    for c in &circuits {
        println!(
            "  {}: {} qubits, {}x{} TPM",
            c.name,
            c.n_qubits,
            c.tpm_size(),
            c.tpm_size()
        );
    }

    // Step 2: Run analysis
    println!("\n=== Step 2: IIT Phi Analysis ===");
    let results = analysis::run_quantum_analysis(&circuits);

    // Step 3: Print text summary
    println!("\n=== Step 3: Results Summary ===");
    report::print_summary(&results);

    // Step 4: Generate SVG report
    let svg = report::generate_svg(&results);
    std::fs::write(output, &svg).expect("Failed to write SVG report");
    println!(
        "\nSVG report saved to: {}",
        parse_str_arg(&args, "--output", "quantum_report.svg")
    );

    // Entanglement hierarchy check
    println!("\n+==========================================================+");
    println!("|  Entanglement Hierarchy vs Phi:                          |");
    println!("|  Expected: Product < W < Bell <= GHZ                    |");
    println!("|  Actual:                                                 |");
    let mut sorted: Vec<_> = results
        .iter()
        .filter(|r| r.name != "Random Circuit")
        .collect();
    sorted.sort_by(|a, b| a.full_phi.partial_cmp(&b.full_phi).unwrap());
    for (i, r) in sorted.iter().enumerate() {
        println!(
            "|  {}. {:25} Phi = {:.6}                |",
            i + 1,
            r.name,
            r.full_phi
        );
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
