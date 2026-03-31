//! Gene Regulatory Network Consciousness Explorer
//!
//! Applies IIT Phi to gene regulatory networks to identify emergent
//! regulatory modules. Compares normal vs oncogenic (cancer) network
//! rewiring to study how disease alters integrated information.

mod analysis;
mod data;
mod report;

fn main() {
    println!("+==========================================================+");
    println!("|  Gene Regulatory Network Consciousness Explorer          |");
    println!("|  IIT 4.0 Phi Analysis of Regulatory Modules              |");
    println!("+==========================================================+");

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let null_samples = parse_arg(&args, "--null-samples", 50usize);
    let output = parse_str_arg(&args, "--output", "gene_report.svg");

    println!("\nConfiguration:");
    println!("  Genes:        16 (4 modules x 4 genes)");
    println!("  Null samples: {}", null_samples);
    println!("  Output:       {}", output);

    // Step 1: Build gene regulatory networks
    println!("\n=== Step 1: Building Gene Regulatory Networks ===");
    let normal = data::build_normal_network();
    let cancer = data::build_cancer_network();
    println!("  Normal network: {} genes, {} edges", normal.n_genes, normal.n_edges());
    println!("  Cancer network: {} genes, {} edges", cancer.n_genes, cancer.n_edges());

    // Step 2: Construct TPMs
    println!("\n=== Step 2: Constructing Transition Probability Matrices ===");
    let normal_tpm = data::network_to_tpm(&normal);
    let cancer_tpm = data::network_to_tpm(&cancer);
    println!("  Normal TPM: {}x{}", normal_tpm.size, normal_tpm.size);
    println!("  Cancer TPM: {}x{}", cancer_tpm.size, cancer_tpm.size);

    // Step 3: Run analysis
    println!("\n=== Step 3: Consciousness Analysis ===");
    let results = analysis::run_analysis(&normal, &normal_tpm, &cancer, &cancer_tpm, null_samples);

    // Step 4: Print report
    println!("\n=== Step 4: Results ===");
    report::print_summary(&results);

    // Step 5: Generate SVG
    let svg = report::generate_svg(&results, &normal);
    std::fs::write(output, &svg).expect("Failed to write SVG report");
    println!(
        "\nSVG report saved to: {}",
        parse_str_arg(&args, "--output", "gene_report.svg")
    );

    // Final verdict
    println!("\n+==========================================================+");
    if results.modules_more_integrated {
        println!("|  RESULT: Modules ARE the irreducible units of            |");
        println!("|  integrated information in the regulatory network.       |");
    } else {
        println!("|  RESULT: Full network is more integrated than modules.   |");
        println!("|  The regulatory network acts as a unified whole.         |");
    }
    if results.cancer_higher_cross_phi {
        println!("|  Cancer rewiring INCREASES cross-module integration,     |");
        println!("|  consistent with loss of modular boundaries.             |");
    } else {
        println!("|  Cancer rewiring does NOT increase cross-module Phi.     |");
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
