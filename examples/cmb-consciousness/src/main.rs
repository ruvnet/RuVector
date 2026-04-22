//! CMB Consciousness Explorer
//!
//! Explores the Cosmic Microwave Background radiation for signatures of
//! integrated information using IIT Phi, causal emergence, and MinCut analysis.

mod analysis;
mod cross_freq;
mod data;
mod emergence_sweep;
mod healpix;
mod report;

fn main() {
    println!("+==========================================================+");
    println!("|     CMB Consciousness Explorer -- IIT 4.0 Analysis       |");
    println!("|     Searching for integrated information in the CMB      |");
    println!("+==========================================================+");

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let n_bins = parse_arg(&args, "--bins", 16usize);
    let null_samples = parse_arg(&args, "--null-samples", 100usize);
    let alpha = parse_arg(&args, "--alpha", 1.0f64);
    let output = parse_str_arg(&args, "--output", "cmb_report.svg");

    println!("\nConfiguration:");
    println!("  Bins:         {}", n_bins);
    println!("  Null samples: {}", null_samples);
    println!("  Alpha:        {:.1}", alpha);
    println!("  Output:       {}", output);

    // Step 1: Acquire data
    println!("\n=== Step 1: Acquiring CMB Data ===");
    let ps = data::download_power_spectrum();
    println!(
        "  Multipole range: l = {}..{}",
        ps.ells[0] as u32,
        *ps.ells.last().unwrap() as u32
    );

    // Step 2: Construct TPM
    println!("\n=== Step 2: Constructing Transition Probability Matrix ===");
    let tpm = data::power_spectrum_to_tpm(&ps, n_bins, alpha);
    println!("  TPM size: {}x{}", tpm.size, tpm.size);
    println!(
        "  Bin edges (l): {:?}",
        tpm.bin_edges.iter().map(|x| *x as u32).collect::<Vec<_>>()
    );

    // Step 3: Run analysis
    println!("\n=== Step 3: Consciousness Analysis ===");
    let results = analysis::run_analysis(&tpm, &ps, n_bins, alpha, null_samples);

    // Step 4: Generate report
    println!("\n=== Step 4: Results ===");
    report::print_summary(&results, &tpm);

    // Step 5: Generate SVG
    let svg = report::generate_svg(&results, &tpm, &ps);
    std::fs::write(output, &svg).expect("Failed to write SVG report");
    println!(
        "\nSVG report saved to: {}",
        parse_str_arg(&args, "--output", "cmb_report.svg")
    );

    // Step 5: Cross-frequency foreground analysis
    println!("\n=== Step 5: Cross-Frequency Foreground Analysis ===");
    let _cross_freq_results = cross_freq::run_cross_frequency_analysis();

    // Step 6: Emergence sweep
    println!("\n=== Step 6: Emergence Sweep ===");
    let _sweep_results = emergence_sweep::run_emergence_sweep(&ps);

    // Step 7: Spatial Phi sky map
    println!("\n=== Step 7: Spatial Phi Sky Map ===");
    let sky_results = healpix::run_sky_mapping(&ps);
    let sky_svg = healpix::render_sky_map_svg(&sky_results);
    std::fs::write("cmb_sky_map.svg", &sky_svg).expect("Failed to write sky map");
    println!("  Sky map SVG saved to: cmb_sky_map.svg");

    // Final verdict
    println!("\n+==========================================================+");
    if results.p_value < 0.05 {
        println!("|  RESULT: Anomalous integrated information detected!     |");
        println!(
            "|  p = {:.4}, z = {:.2} -- warrants further investigation   |",
            results.p_value, results.z_score
        );
    } else {
        println!("|  RESULT: CMB consistent with Gaussian random field      |");
        println!(
            "|  p = {:.4}, z = {:.2} -- no evidence of structured        |",
            results.p_value, results.z_score
        );
        println!("|  intelligence at this resolution                        |");
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
