//! GW Consciousness Explorer
//!
//! Analyzes the gravitational wave stochastic background from pulsar timing
//! arrays for signatures of integrated information using IIT Phi. Tests whether
//! spectral correlations between frequency bins are consistent with independent
//! SMBH mergers (Phi~0) or a correlated cosmological source (Phi>0).

mod analysis;
mod data;
mod report;

fn main() {
    println!("+==========================================================+");
    println!("|   GW Background Consciousness Explorer -- IIT 4.0        |");
    println!("|   Searching for integrated information in the GWB        |");
    println!("+==========================================================+");

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let n_bins = parse_arg(&args, "--bins", 14usize);
    let null_samples = parse_arg(&args, "--null-samples", 100usize);
    let output = parse_str_arg(&args, "--output", "gw_report.svg");

    println!("\nConfiguration:");
    println!("  Frequency bins: {}", n_bins);
    println!("  Null samples:   {}", null_samples);
    println!("  Output:         {}", output);

    // Step 1: Generate GW spectra for each source model
    println!("\n=== Step 1: Generating GW Background Spectra ===");
    let models = ["smbh", "cosmic_strings", "primordial", "phase_transition"];
    let spectra: Vec<_> = models
        .iter()
        .map(|&m| {
            let spec = data::generate_nanograv_spectrum(m);
            println!(
                "  {:20} {} bins, f = {:.2e}..{:.2e} Hz",
                m,
                spec.n_bins,
                spec.frequencies[0],
                spec.frequencies.last().copied().unwrap_or(0.0),
            );
            (m, spec)
        })
        .collect();

    // Step 2: Build TPMs
    println!("\n=== Step 2: Constructing Transition Probability Matrices ===");
    let tpms: Vec<_> = spectra
        .iter()
        .map(|(name, spec)| {
            let tpm = data::gw_spectrum_to_tpm(spec, n_bins, 1.0);
            println!("  {:20} TPM size: {}x{}", name, tpm.size, tpm.size);
            (*name, tpm)
        })
        .collect();

    // Step 3: Run analysis
    println!("\n=== Step 3: Consciousness Analysis ===");
    let results = analysis::run_analysis(&tpms, &spectra, n_bins, null_samples);

    // Step 4: Print results
    println!("\n=== Step 4: Results ===");
    report::print_summary(&results);

    // Step 5: Generate SVG
    let svg = report::generate_svg(&results, &tpms, &spectra);
    std::fs::write(output, &svg).expect("Failed to write SVG report");
    println!(
        "\nSVG report saved to: {}",
        parse_str_arg(&args, "--output", "gw_report.svg")
    );

    // Final verdict
    println!("\n+==========================================================+");
    let smbh_phi = results
        .model_phis
        .iter()
        .find(|(n, _)| *n == "smbh")
        .map(|(_, p)| p.phi)
        .unwrap_or(0.0);
    let max_exotic_phi = results
        .model_phis
        .iter()
        .filter(|(n, _)| *n != "smbh")
        .map(|(_, p)| p.phi)
        .fold(0.0f64, f64::max);

    if max_exotic_phi > smbh_phi * 1.5 && results.p_value < 0.05 {
        println!("|  RESULT: Evidence for correlated cosmological source!   |");
        println!(
            "|  Phi(exotic) = {:.4} >> Phi(SMBH) = {:.4}              |",
            max_exotic_phi, smbh_phi
        );
    } else {
        println!("|  RESULT: GWB consistent with independent SMBH mergers  |");
        println!(
            "|  p = {:.4}, z = {:.2} -- no excess integration detected  |",
            results.p_value, results.z_score
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
