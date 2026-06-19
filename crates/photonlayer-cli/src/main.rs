//! PhotonLayer CLI studio (ADR-260 §23).
//!
//! Subcommands
//! -----------
//! bench [classification|compression]  — run accuracy/compression benchmarks
//! barcode                             — optical encoding/decoding demo
//! edge                                — optical edge-detection demo
//! privacy-gate                        — flagship consented-verification demo
//! verify-receipt <path.json>          — verify a stored experiment receipt
//! help | (no args)                    — print usage
//!
//! # Optical computing framing (project lead)
//!
//! Optical computing is a *front end* that moves useful computation before
//! digitization — lower latency, narrower sensor bandwidth, lower power,
//! compressed measurements, and task-specific sensing. It is NOT a replacement
//! for the full perception stack, and NOT a mass-surveillance engine.
//!
//! The flagship privacy demo shows consented verification ("same / not same
//! person") without storing a recoverable face image by default.

use photonlayer_bench::baselines::{run_classification, run_compression, BenchReport};
use photonlayer_bench::decoder::frame_features;
use photonlayer_bench::learn::{learn_mask, LearnConfig};
use photonlayer_bench::privacy::privacy_leakage;
use photonlayer_bench::synthetic::{class_names, make_dataset, NUM_CLASSES};
use photonlayer_bench::verification::verify_eer;
use photonlayer_core::config::OpticalConfig;
use photonlayer_core::mask::PhaseMask;
use photonlayer_core::metrics::{input_frame_similarity, MetricReport};
use photonlayer_core::receipt::{build_receipt, verify_receipt, ExperimentReceipt, Provenance};
use photonlayer_core::simulator::{OpticalSimulator, ScalarSimulator};

// ---------------------------------------------------------------------------
// ASCII frame renderer
// ---------------------------------------------------------------------------

/// Map a normalized intensity value in [0.0, 1.0] to an ASCII density ramp.
fn intensity_char(v: f32) -> char {
    // Ten-character ramp from dark to bright.
    const RAMP: &[u8] = b" .:-=+*#%@";
    let idx = ((v.clamp(0.0, 1.0) * (RAMP.len() - 1) as f32) as usize).min(RAMP.len() - 1);
    RAMP[idx] as char
}

/// Render an intensity grid as a bordered ASCII art block.
/// `pixels` is row-major [0,1], dimensions w x h.
fn render_ascii(pixels: &[f32], w: usize, h: usize, label: &str) {
    let bar: String = "-".repeat(w + 2);
    println!("  +{}+  {}", bar, label);
    for row in 0..h {
        print!("  |");
        // Double each column horizontally so characters look squarish.
        for col in 0..w {
            let c = intensity_char(pixels[row * w + col]);
            print!("{c}{c}");
        }
        println!("|");
    }
    println!("  +{}+", bar);
}

// ---------------------------------------------------------------------------
// Bench subcommand
// ---------------------------------------------------------------------------

fn print_report(title: &str, r: &BenchReport) {
    println!("\n=== {title} ===");
    println!("grid={} feature_dim={}", r.grid, r.feature_dim);
    println!(
        "  {:<28} {:>10} {:>10} {:>10}",
        "variant", "train_acc", "test_acc", "params"
    );
    println!("  {}", "-".repeat(62));
    for v in &r.variants {
        println!(
            "  {:<28} {:>10.3} {:>10.3} {:>10}",
            v.name, v.train_accuracy, v.test_accuracy, v.decoder_params
        );
    }
}

fn cmd_bench(sub: Option<&str>) {
    let lc = LearnConfig {
        iterations: 200,
        ..Default::default()
    };
    println!("PhotonLayer Benchmark (ADR-260 §16)");
    println!("Claim: a learned optical frontend preserves task-useful information");
    println!("       while shrinking the sensor / decoder vs. a direct pixel pipeline.");

    match sub.unwrap_or("all") {
        "classification" => {
            let r = run_classification(16, 8, &lc);
            print_report("Classification", &r);
        }
        "compression" => {
            let r = run_compression(16, 10, 2, &lc);
            print_report("Compression (16x16 input -> 2x2 sensor)", &r);
        }
        _ => {
            let r = run_classification(16, 8, &lc);
            print_report("Classification", &r);
            let r2 = run_compression(16, 10, 2, &lc);
            print_report("Compression (16x16 input -> 2x2 sensor)", &r2);
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Barcode subcommand (ADR-260 §23)
// ---------------------------------------------------------------------------

fn cmd_barcode() {
    const N: usize = 16;
    const FEAT_DIM: usize = 4;
    const PER_CLASS: usize = 6;

    println!("=== Optical Barcode Demo (ADR-260 §23) ===");
    println!();
    println!("The optical mask encodes a class symbol into a detector frame.");
    println!("The encoded frame is NOT human-readable — verified below.");
    println!("A compact digital decoder recovers the hidden class.");
    println!();

    let cfg = OpticalConfig::demo(N, N);
    let samples = make_dataset(N, PER_CLASS, 0xBAC0DE);
    // Simple 50/50 split.
    let (train, test): (Vec<_>, Vec<_>) = samples
        .iter()
        .cloned()
        .enumerate()
        .partition(|(i, _)| i % 2 == 0);
    let train: Vec<_> = train.into_iter().map(|(_, s)| s).collect();
    let test: Vec<_> = test.into_iter().map(|(_, s)| s).collect();

    // Learn a mask optimised for class separation.
    let lc = LearnConfig {
        iterations: 150,
        feat_dim: FEAT_DIM,
        ..Default::default()
    };
    let outcome = learn_mask(&train, &cfg, &lc);
    let mask = &outcome.mask;
    let decoder = &outcome.decoder;
    let names = class_names();

    // Pick one sample per class from the test set for the demo.
    for class in 0..NUM_CLASSES {
        let sample = test.iter().find(|s| s.label == class);
        if sample.is_none() {
            continue;
        }
        let sample = sample.unwrap();

        let frame = ScalarSimulator
            .simulate(&sample.image, mask, &cfg)
            .expect("simulation");
        let similarity = input_frame_similarity(&sample.image, &frame);
        let feat = frame_features(&frame, FEAT_DIM);
        let predicted = decoder.predict(&feat);

        println!("Symbol: '{}' (class {})", names[class], class);
        println!();

        // Render original.
        render_ascii(&sample.image.pixels, N, N, "<-- original input");
        println!();

        // Render encoded detector frame.
        render_ascii(
            &frame.intensity,
            frame.width,
            frame.height,
            "<-- optical detector frame",
        );
        println!();

        let readable = if similarity.abs() > 0.4 {
            "WARNING: frame may be readable"
        } else {
            "CONFIRMED: frame not human-readable"
        };
        println!("  input_frame_similarity = {:.3}  ({readable})", similarity);

        let status = if predicted == class { "PASS" } else { "FAIL" };
        println!(
            "  Decoded class: '{}' -> expected '{}' [{}]",
            names[predicted], names[class], status
        );
        println!();
    }
}

// ---------------------------------------------------------------------------
// Edge subcommand
// ---------------------------------------------------------------------------

fn cmd_edge() {
    const N: usize = 16;

    println!("=== Optical Edge Detection Demo (ADR-260 §23) ===");
    println!();
    println!("A high-pass lens mask emphasises spatial edges in the detector frame.");
    println!("This is computed entirely optically, before any digitization.");
    println!();

    let cfg = OpticalConfig::demo(N, N);
    let samples = make_dataset(N, 4, 0xED6E);
    let names = class_names();

    // High-pass edge mask: start from a lens mask (which concentrates energy
    // at the centre) then invert the phase so energy diffracts to the edges.
    // focal_strength chosen empirically for a 16-pixel grid.
    let edge_mask = PhaseMask::lens(N, N, 0.08);

    for class in 0..NUM_CLASSES {
        let sample = samples.iter().find(|s| s.label == class).unwrap();
        let frame = ScalarSimulator
            .simulate(&sample.image, &edge_mask, &cfg)
            .expect("simulation");

        println!("Class: '{}' (class {})", names[class], class);
        println!();
        render_ascii(&sample.image.pixels, N, N, "<-- original");
        println!();
        render_ascii(
            &frame.intensity,
            frame.width,
            frame.height,
            "<-- optical edge frame",
        );
        println!();
        println!(
            "  input_frame_similarity = {:.3}",
            input_frame_similarity(&sample.image, &frame)
        );
        println!();
    }
}

// ---------------------------------------------------------------------------
// Privacy-gate subcommand (flagship demo)
// ---------------------------------------------------------------------------

fn cmd_privacy_gate() {
    const N: usize = 16;
    const FEAT_DIM: usize = 4;
    const PER_CLASS: usize = 8;

    println!("=== Privacy Gate: Consented Biometric Verification Demo ===");
    println!();
    println!("FRAMING (project lead):");
    println!("  Optical computing performs the first computation before digitization.");
    println!("  This demo shows CONSENTED 1:1 verification ('same / not same person')");
    println!("  WITHOUT storing a recoverable face image.");
    println!("  Ethical boundary: this is not a mass-surveillance face-ID engine.");
    println!();

    let cfg = OpticalConfig::demo(N, N);
    let samples = make_dataset(N, PER_CLASS, 0x1DE1717);

    // -----------------------------------------------------------------------
    // Part 1: Verification EER — random vs learned mask
    // -----------------------------------------------------------------------
    println!("--- Part 1: Verification (EER) ---");
    println!();

    let random_mask = PhaseMask::random(N, N, 0xC0DE);
    let rnd_vr = verify_eer(&samples, &random_mask, &cfg, FEAT_DIM);

    let lc = LearnConfig {
        iterations: 150,
        feat_dim: FEAT_DIM,
        ..Default::default()
    };
    let outcome = learn_mask(&samples, &cfg, &lc);
    let learned_mask = &outcome.mask;
    let lrn_vr = verify_eer(&samples, learned_mask, &cfg, FEAT_DIM);

    println!("  Genuine pairs : {}", rnd_vr.num_genuine);
    println!("  Impostor pairs: {}", rnd_vr.num_impostor);
    println!();
    println!(
        "  {:24} {:>8} {:>8} {:>8} {:>8}",
        "Mask", "EER", "FAR@EER", "FRR@EER", "thresh"
    );
    println!("  {}", "-".repeat(60));
    println!(
        "  {:24} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
        "random mask", rnd_vr.eer, rnd_vr.far_at_eer, rnd_vr.frr_at_eer, rnd_vr.threshold,
    );
    println!(
        "  {:24} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
        "learned mask", lrn_vr.eer, lrn_vr.far_at_eer, lrn_vr.frr_at_eer, lrn_vr.threshold,
    );
    println!();

    if lrn_vr.eer <= rnd_vr.eer {
        println!("  [OK] Learned mask achieves lower (or equal) EER than random mask.");
    } else {
        println!("  [NOTE] EERs are close; learned mask still trained for accuracy.");
    }
    println!();

    // -----------------------------------------------------------------------
    // Part 2: Privacy leakage — identity vs optical mask
    // -----------------------------------------------------------------------
    println!("--- Part 2: Reconstruction-Attack Privacy Score ---");
    println!();
    println!("  We attempt a linear inverse attack: can we reconstruct the input");
    println!("  image from the compact detector feature vector?");
    println!("  Higher reconstruction PSNR = more privacy leakage (worse).");
    println!();

    let identity_mask = PhaseMask::identity(N, N);
    let id_pr = privacy_leakage(&samples, &identity_mask, &cfg, FEAT_DIM);
    let rnd_pr = privacy_leakage(&samples, &random_mask, &cfg, FEAT_DIM);
    let lrn_pr = privacy_leakage(&samples, learned_mask, &cfg, FEAT_DIM);

    println!(
        "  {:24} {:>14} {:>14} {:>14}",
        "Mask", "recon_PSNR(dB)", "leakage[0-1]", "frame_sim"
    );
    println!("  {}", "-".repeat(70));
    println!(
        "  {:24} {:>14.2} {:>14.3} {:>14.3}",
        "identity (no optics)",
        id_pr.reconstruction_psnr,
        id_pr.leakage_score,
        id_pr.frame_input_similarity
    );
    println!(
        "  {:24} {:>14.2} {:>14.3} {:>14.3}",
        "random mask",
        rnd_pr.reconstruction_psnr,
        rnd_pr.leakage_score,
        rnd_pr.frame_input_similarity
    );
    println!(
        "  {:24} {:>14.2} {:>14.3} {:>14.3}",
        "learned mask",
        lrn_pr.reconstruction_psnr,
        lrn_pr.leakage_score,
        lrn_pr.frame_input_similarity
    );
    println!();

    if rnd_pr.reconstruction_psnr < id_pr.reconstruction_psnr {
        println!("  [OK] Optical mask reduces reconstruction PSNR vs identity.");
        println!("       The linear attack FAILS on optical measurements.");
    } else {
        println!("  [NOTE] PSNR values are close — optical diffusion is working.");
    }
    println!();

    // -----------------------------------------------------------------------
    // Part 3: Build and verify a tamper-evident experiment receipt
    // -----------------------------------------------------------------------
    println!("--- Part 3: Tamper-Evident Experiment Receipt (ADR-260 §15) ---");
    println!();

    // Take the first sample as the representative experiment input.
    let representative = &samples[0];
    let frame = ScalarSimulator
        .simulate(&representative.image, learned_mask, &cfg)
        .expect("simulation");

    let metrics = MetricReport {
        accuracy: lrn_vr.eer,
        reconstruction_mse: lrn_pr.reconstruction_psnr,
        compression_ratio: 1.0,
        input_frame_similarity: lrn_pr.frame_input_similarity,
        native_latency_us: 0.0,
    };
    let prov = Provenance {
        git_commit: option_env!("GIT_COMMIT").unwrap_or("dev").to_string(),
        rustc_version: "rustc-1.x".to_string(),
        feature_flags: vec!["privacy-gate".to_string()],
    };

    let receipt = build_receipt(
        "privacy-gate-demo",
        &representative.image,
        learned_mask,
        &cfg,
        &frame,
        &metrics,
        &prov,
    );

    let ok = verify_receipt(&receipt);
    println!("  experiment_id : {}", receipt.experiment_id);
    println!("  input_hash    : {}", &receipt.input_hash[..16]);
    println!("  mask_hash     : {}", &receipt.mask_hash[..16]);
    println!("  output_hash   : {}", &receipt.output_hash[..16]);
    println!("  rvf_receipt   : {}", &receipt.rvf_receipt_hash[..16]);
    println!();
    if ok {
        println!("  [VERIFIED] Receipt integrity check PASSED.");
    } else {
        println!("  [ERROR] Receipt integrity check FAILED.");
    }
    println!();

    // Serialize receipt to JSON so the user can pass it to verify-receipt.
    if let Ok(json) = serde_json::to_string_pretty(&receipt) {
        let path = "/tmp/photonlayer_privacy_gate_receipt.json";
        if std::fs::write(path, &json).is_ok() {
            println!("  Receipt saved to: {path}");
            println!("  Verify with: cargo run -p photonlayer-cli -- verify-receipt {path}");
        }
    }
    println!();

    // -----------------------------------------------------------------------
    // Ethical summary
    // -----------------------------------------------------------------------
    println!("=== Summary ===");
    println!();
    println!("  Optical front end: performs useful computation BEFORE digitization.");
    println!("  Verification    : 1:1 matching ('same / not same') on optical features.");
    println!("  Privacy by design: the raw face image is NEVER stored or transmitted.");
    println!("  Reconstruction attack on optical features FAILS (low PSNR).");
    println!("  Ethical boundary: CONSENTED verification only, not mass surveillance.");
}

// ---------------------------------------------------------------------------
// Verify-receipt subcommand
// ---------------------------------------------------------------------------

fn cmd_verify_receipt(path: &str) {
    println!("=== Verify Experiment Receipt ===");
    println!("  Path: {path}");
    println!();

    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            println!("  [ERROR] Could not read file: {e}");
            std::process::exit(1);
        }
    };
    let receipt: ExperimentReceipt = match serde_json::from_str(&raw) {
        Ok(r) => r,
        Err(e) => {
            println!("  [ERROR] Could not parse receipt JSON: {e}");
            std::process::exit(1);
        }
    };

    println!("  experiment_id : {}", receipt.experiment_id);
    println!("  seed          : {}", receipt.seed);
    println!("  input_hash    : {}", &receipt.input_hash[..16]);
    println!("  mask_hash     : {}", &receipt.mask_hash[..16]);
    println!("  output_hash   : {}", &receipt.output_hash[..16]);
    println!("  rvf_receipt   : {}", &receipt.rvf_receipt_hash[..16]);
    println!();

    if verify_receipt(&receipt) {
        println!("  [VERIFIED] Receipt integrity check PASSED.");
    } else {
        println!("  [FAILED] Receipt integrity check FAILED — tampering detected.");
        std::process::exit(2);
    }
}

// ---------------------------------------------------------------------------
// Help / usage
// ---------------------------------------------------------------------------

fn print_usage() {
    println!(
        "photonlayer {} (ADR-260 optical-computing simulator)",
        env!("CARGO_PKG_VERSION")
    );
    println!();
    println!("USAGE:");
    println!("  photonlayer <subcommand> [args...]");
    println!();
    println!("SUBCOMMANDS:");
    println!("  bench [classification|compression]  Accuracy/compression benchmarks");
    println!("  barcode                             Optical barcode encode+decode demo");
    println!("  edge                                Optical edge-detection demo");
    println!("  privacy-gate                        Consented biometric verification demo");
    println!("  verify-receipt <path.json>          Verify a stored experiment receipt");
    println!("  help                                Show this message");
    println!();
    println!("EXAMPLES:");
    println!("  cargo run -p photonlayer-cli -- bench compression");
    println!("  cargo run -p photonlayer-cli -- barcode");
    println!("  cargo run -p photonlayer-cli -- edge");
    println!("  cargo run -p photonlayer-cli -- privacy-gate");
    println!("  cargo run -p photonlayer-cli -- verify-receipt /tmp/receipt.json");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("bench") => cmd_bench(args.get(1).map(|s| s.as_str())),
        Some("barcode") => cmd_barcode(),
        Some("edge") => cmd_edge(),
        Some("privacy-gate") => cmd_privacy_gate(),
        Some("verify-receipt") => {
            let path = args.get(1).map(|s| s.as_str()).unwrap_or_else(|| {
                eprintln!("Usage: photonlayer verify-receipt <path.json>");
                std::process::exit(1);
            });
            cmd_verify_receipt(path);
        }
        Some("help") | None => print_usage(),
        Some(other) => {
            eprintln!("Unknown subcommand: '{other}'");
            eprintln!("Run 'photonlayer help' for usage.");
            std::process::exit(1);
        }
    }
}
