#![allow(clippy::all)]
//! Real, end-to-end speculative decoding benchmark.
//!
//! Loads two real GGUF models (a small draft model and a larger main model
//! that **must share the same tokenizer/vocabulary** — see
//! `ruvllm::speculative` module docs) and measures actual tokens/sec for:
//!
//! - Baseline: the main model decoding autoregressively on its own
//!   (`CandleBackend::generate`, one real forward pass per token).
//! - Speculative: `SpeculativeDecoder` drafting with the small model and
//!   verifying batches of draft tokens against the main model in single
//!   forward passes (`crates/ruvllm/src/speculative.rs`).
//!
//! Both paths run real candle forward passes on real weights; nothing here
//! is simulated. Requires the `candle` feature (on by default).
//!
//! ## Usage
//!
//! ```bash
//! cargo run -p ruvllm --release --example speculative_bench -- \
//!     --main ./models/main-llama2-7b/model.gguf \
//!     --draft ./models/draft-tinyllama/model.gguf \
//!     --max-tokens 64 --lookahead 5
//! ```
//!
//! Tokenizers are expected at `tokenizer.json` next to each GGUF file
//! (same convention as `benchmark_model.rs`).

use std::env;
use std::path::PathBuf;
use std::time::Instant;

struct Config {
    main_path: PathBuf,
    draft_path: PathBuf,
    max_tokens: usize,
    lookahead: usize,
    iterations: usize,
    json_output: bool,
    prompts: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            main_path: PathBuf::new(),
            draft_path: PathBuf::new(),
            max_tokens: 64,
            lookahead: 5,
            iterations: 3,
            json_output: false,
            prompts: vec![
                "The quick brown fox jumps over".to_string(),
                "Once upon a time in a distant land,".to_string(),
                "The capital of France is".to_string(),
            ],
        }
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let mut cfg = Config::default();

    if args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_help();
        std::process::exit(0);
    }

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--main" => {
                i += 1;
                if i < args.len() {
                    cfg.main_path = PathBuf::from(&args[i]);
                }
            }
            "--draft" => {
                i += 1;
                if i < args.len() {
                    cfg.draft_path = PathBuf::from(&args[i]);
                }
            }
            "--max-tokens" | "-t" => {
                i += 1;
                if i < args.len() {
                    cfg.max_tokens = args[i].parse().unwrap_or(64);
                }
            }
            "--lookahead" | "-k" => {
                i += 1;
                if i < args.len() {
                    cfg.lookahead = args[i].parse().unwrap_or(5);
                }
            }
            "--iterations" | "-i" => {
                i += 1;
                if i < args.len() {
                    cfg.iterations = args[i].parse().unwrap_or(3);
                }
            }
            "--json" | "-j" => cfg.json_output = true,
            _ => {}
        }
        i += 1;
    }

    if cfg.main_path.as_os_str().is_empty() || cfg.draft_path.as_os_str().is_empty() {
        eprintln!("Error: --main and --draft model paths are required.\n");
        print_help();
        std::process::exit(1);
    }

    cfg
}

fn print_help() {
    println!("RuvLLM Speculative Decoding Benchmark");
    println!();
    println!("USAGE:");
    println!("    cargo run -p ruvllm --release --example speculative_bench -- \\");
    println!("        --main <MAIN_GGUF> --draft <DRAFT_GGUF> [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("        --main <PATH>        Path to main (target) GGUF model");
    println!("        --draft <PATH>       Path to draft GGUF model (must share the");
    println!("                             main model's tokenizer/vocabulary)");
    println!("    -t, --max-tokens <N>     Tokens to generate per prompt (default: 64)");
    println!("    -k, --lookahead <N>      Draft lookahead (default: 5)");
    println!("    -i, --iterations <N>     Iterations per prompt (default: 3)");
    println!("    -j, --json               Output results as JSON");
    println!("    -h, --help               Print help information");
}

#[cfg(feature = "candle")]
fn main() {
    use ruvllm::speculative::{SpeculativeConfig, SpeculativeDecoder};
    use ruvllm::{CandleBackend, GenerateParams, LlmBackend, ModelConfig};
    use std::sync::Arc;

    let cfg = parse_args();

    let load = |path: &PathBuf, label: &str| -> CandleBackend {
        println!("Loading {label} model: {}", path.display());
        let mut backend = CandleBackend::new().expect("failed to create CandleBackend");
        backend
            .load_gguf(path, &ModelConfig::default())
            .unwrap_or_else(|e| panic!("failed to load {label} GGUF at {path:?}: {e}"));
        let tokenizer_path = path
            .parent()
            .expect("model path has no parent directory")
            .join("tokenizer.json");
        backend.load_tokenizer(&tokenizer_path).unwrap_or_else(|e| {
            panic!("failed to load {label} tokenizer at {tokenizer_path:?}: {e}")
        });
        backend
    };

    // Three independent backend instances so baseline and speculative runs
    // never share (and can't corrupt) each other's KV cache state.
    let main_baseline = load(&cfg.main_path, "main (baseline)");
    let main_spec = Arc::new(load(&cfg.main_path, "main (speculative)"));
    let draft_spec = Arc::new(load(&cfg.draft_path, "draft (speculative)"));

    let params = GenerateParams {
        max_tokens: cfg.max_tokens,
        temperature: 0.0, // greedy — required for the exact-match verification in speculative.rs
        top_p: 1.0,
        top_k: 1,
        ..Default::default()
    };

    let spec_config = SpeculativeConfig {
        lookahead: cfg.lookahead,
        draft_temperature: 0.0,
        ..Default::default()
    };
    let decoder = SpeculativeDecoder::new(main_spec, draft_spec, spec_config);

    let mut baseline_tps = Vec::new();
    let mut spec_tps = Vec::new();

    for prompt in &cfg.prompts {
        for iter in 0..cfg.iterations {
            // --- Baseline: main model decoding on its own ---
            let start = Instant::now();
            let output = main_baseline
                .generate(prompt, params.clone())
                .expect("baseline generation failed");
            let elapsed = start.elapsed();
            let token_count = main_baseline
                .tokenizer()
                .expect("main tokenizer loaded")
                .encode(&output)
                .map(|t| t.len())
                .unwrap_or(0);
            let tps = token_count as f64 / elapsed.as_secs_f64();
            baseline_tps.push(tps);
            if !cfg.json_output {
                println!(
                    "[baseline]    prompt={:?} iter={} tokens={} time={:.3}s tok/s={:.2}",
                    &prompt[..prompt.len().min(30)],
                    iter,
                    token_count,
                    elapsed.as_secs_f64(),
                    tps
                );
            }

            // --- Speculative: draft + batched verify against main ---
            let start = Instant::now();
            let tokens = decoder
                .generate_tokens(
                    &decoder
                        .tokenizer()
                        .expect("tokenizer")
                        .encode(prompt)
                        .expect("encode prompt"),
                    &params,
                )
                .expect("speculative generation failed");
            let elapsed = start.elapsed();
            let tps = tokens.len() as f64 / elapsed.as_secs_f64();
            spec_tps.push(tps);
            let stats = decoder.stats();
            if !cfg.json_output {
                println!(
                    "[speculative] prompt={:?} iter={} tokens={} time={:.3}s tok/s={:.2} \
                     acceptance_rate={:.1}% avg_tokens/main_pass={:.2}",
                    &prompt[..prompt.len().min(30)],
                    iter,
                    tokens.len(),
                    elapsed.as_secs_f64(),
                    tps,
                    stats.acceptance_rate * 100.0,
                    stats.avg_tokens_per_main_pass
                );
            }
        }
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let baseline_mean = mean(&baseline_tps);
    let spec_mean = mean(&spec_tps);
    let final_stats = decoder.stats();

    if cfg.json_output {
        println!(
            "{{\"baseline_tokens_per_sec\":{:.4},\"speculative_tokens_per_sec\":{:.4},\
             \"speedup\":{:.4},\"acceptance_rate\":{:.4},\"avg_tokens_per_main_pass\":{:.4}}}",
            baseline_mean,
            spec_mean,
            spec_mean / baseline_mean,
            final_stats.acceptance_rate,
            final_stats.avg_tokens_per_main_pass
        );
    } else {
        println!();
        println!("=== Results (mean over {} runs) ===", baseline_tps.len());
        println!("Baseline (main model alone):    {baseline_mean:.2} tok/s");
        println!("Speculative (draft+verify):     {spec_mean:.2} tok/s");
        println!(
            "Speedup:                        {:.2}x",
            spec_mean / baseline_mean
        );
        println!(
            "Draft acceptance rate:          {:.1}%",
            final_stats.acceptance_rate * 100.0
        );
        println!(
            "Avg tokens per main forward:    {:.2}",
            final_stats.avg_tokens_per_main_pass
        );
    }
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("This example requires the `candle` feature: cargo run --features candle ...");
    std::process::exit(1);
}
