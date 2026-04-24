//! Ecosystem Consciousness Explorer
//!
//! Applies IIT Phi to food web networks to measure ecosystem integration
//! and resilience. Compares tropical rainforest, agricultural monoculture,
//! and coral reef ecosystems.

mod analysis;
mod data;
mod report;

fn main() {
    println!("+==========================================================+");
    println!("|   Ecosystem Consciousness Explorer -- IIT 4.0 Analysis   |");
    println!("|   Measuring food web integration via Phi                 |");
    println!("+==========================================================+");

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let output = parse_str_arg(&args, "--output", "ecosystem_report.svg");

    println!("\nConfiguration:");
    println!("  Output: {}", output);

    // Step 1: Generate food web data
    println!("\n=== Step 1: Generating Synthetic Food Webs ===");
    let ecosystems = data::generate_all_ecosystems();
    for eco in &ecosystems {
        println!(
            "  {}: {} species, {} connections",
            eco.name,
            eco.species.len(),
            eco.connection_count()
        );
    }

    // Step 2: Run consciousness analysis
    println!("\n=== Step 2: IIT Phi Analysis ===");
    let results = analysis::run_ecosystem_analysis(&ecosystems);

    // Step 3: Print text summary
    println!("\n=== Step 3: Results Summary ===");
    report::print_summary(&results);

    // Step 4: Generate SVG report
    let svg = report::generate_svg(&results);
    std::fs::write(output, &svg).expect("Failed to write SVG report");
    println!(
        "\nSVG report saved to: {}",
        parse_str_arg(&args, "--output", "ecosystem_report.svg")
    );

    // Final comparison
    println!("\n+==========================================================+");
    println!("|  Ecosystem Integration Ranking (by Phi):                 |");
    let mut sorted: Vec<_> = results.iter().collect();
    sorted.sort_by(|a, b| b.full_phi.partial_cmp(&a.full_phi).unwrap());
    for (i, r) in sorted.iter().enumerate() {
        println!(
            "|  {}. {:30} Phi = {:.6}       |",
            i + 1,
            r.name,
            r.full_phi
        );
    }
    println!("+==========================================================+");
}

fn parse_str_arg<'a>(args: &'a [String], name: &str, default: &'a str) -> &'a str {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].as_str())
        .unwrap_or(default)
}
