//! Honest, apples-to-apples throughput harness for the `lattice` and
//! `candle` ruvllm backends.
//!
//! Measures load time, decode throughput, and (for `lattice`) TTFT, and
//! prints a markdown table + one JSON line per run (to stderr) so numbers
//! are both human- and machine-readable.
//!
//! ## Candle timing rule
//!
//! `CandleBackend::generate_stream_v2` runs the full decode loop to
//! completion and then replays the tokens into the stream, so its first
//! token arrives only after the whole generation: it cannot measure TTFT.
//! This harness therefore times the candle backend via the blocking
//! `generate()` call (equivalent total work). For per-token timing use
//! `generate_detailed`, whose callback fires as each token is decoded.
//!
//! ## GPU serialization (fleet convention)
//!
//! Every process that drives the Metal GPU for timing measurements on this
//! machine must hold an exclusive advisory flock on
//! `/tmp/lion-metal-gpu-test.lock` for its lifetime; concurrent GPU work
//! corrupts both timing and numerics (lattice #628/#629). This example
//! acquires that lock (via a raw `flock(2)` FFI call — see `gpu_lock`
//! below — no new crate dependency) before touching either backend.
//!
//! ## Getting a model to bench (lattice backend)
//!
//! The lattice backend loads a local directory containing `tokenizer.json` +
//! `config.json`, plus either a Qwen3.5 safetensors checkpoint (f16) or a
//! lattice Q4 weight set (`*.q4` files).
//!
//! Fastest path, no quantization needed (f16 safetensors straight from HF):
//!
//! ```bash
//! pip install -U "huggingface_hub[cli]"
//! huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ~/.lattice/models/qwen3.5-0.8b
//! ```
//!
//! For the Q4 numbers you must quantize the checkpoint yourself with
//! lattice's streaming quantizer, then copy the tokenizer/config next to the
//! `.q4` output (the quantizer writes only weights):
//!
//! ```bash
//! cargo install lattice-inference --features metal-gpu,f16   # installs quantize_q4
//! quantize_q4 \
//!     --model-dir ~/.lattice/models/qwen3.5-0.8b \
//!     --output-dir ~/.lattice/models/qwen3.5-0.8b-q4
//! cp ~/.lattice/models/qwen3.5-0.8b/{tokenizer.json,config.json} \
//!     ~/.lattice/models/qwen3.5-0.8b-q4/
//! ```
//!
//! ## Usage
//!
//! ```bash
//! cargo run -p ruvllm --release --features lattice --example lattice_bench -- \
//!     --backend lattice --model ~/.lattice/models/qwen3.5-0.8b-q4
//!
//! cargo run -p ruvllm --release --features candle --example lattice_bench -- \
//!     --backend candle --model /path/to/llama-3.2-1b
//! ```
//!
//! Flags: `--max-tokens N` (default 128), `--runs N` (default 3, median
//! reported), `--warmup N` (default 1), `--prompt "..."`. Set `BENCH_GREEDY=1`
//! to decode greedily (top_k=1, temperature=0) — use this when comparing
//! against standalone-engine numbers so sampling cost does not skew the
//! comparison; default sampling adds meaningful per-token CPU cost on
//! lattice's 248k vocabulary.
#![allow(clippy::too_many_arguments)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

#[cfg(feature = "candle")]
use ruvllm::CandleBackend;
#[cfg(all(feature = "lattice", target_os = "macos"))]
use ruvllm::LatticeBackend;
use ruvllm::{GenerateParams, LlmBackend, ModelInfo, StreamEvent};

/// A fixed ~50-token prompt so every run (across backends and machines) is
/// measuring the same input. Token count is reported per-row from the
/// backend's own tokenizer rather than assumed, since BPE vocabularies
/// differ (see `PromptStats`).
const DEFAULT_PROMPT: &str = "The history of artificial intelligence began in antiquity, \
with myths and legends of artificial beings endowed with intelligence by master \
craftsmen. The seeds of modern AI were planted by philosophers who attempted to \
describe human thinking as a mechanical manipulation of symbols. This work culminated \
in the invention of the programmable digital computer, a machine based on the \
abstract essence of mathematical reasoning.";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    backend: String,
    model: PathBuf,
    prompt: String,
    max_tokens: usize,
    runs: usize,
    warmup: usize,
}

impl Args {
    fn parse() -> Self {
        let mut backend = String::new();
        let mut model: Option<PathBuf> = None;
        let mut prompt = DEFAULT_PROMPT.to_string();
        let mut max_tokens = 128usize;
        let mut runs = 3usize;
        let mut warmup = 1usize;

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--backend" => backend = it.next().expect("--backend requires a value"),
                "--model" => {
                    model = Some(PathBuf::from(it.next().expect("--model requires a value")))
                }
                "--prompt" => prompt = it.next().expect("--prompt requires a value"),
                "--max-tokens" => {
                    max_tokens = it
                        .next()
                        .expect("--max-tokens requires a value")
                        .parse()
                        .expect("--max-tokens must be an integer")
                }
                "--runs" => {
                    runs = it
                        .next()
                        .expect("--runs requires a value")
                        .parse()
                        .expect("--runs must be an integer")
                }
                "--warmup" => {
                    warmup = it
                        .next()
                        .expect("--warmup requires a value")
                        .parse()
                        .expect("--warmup must be an integer")
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    eprintln!("error: unrecognized argument '{other}'");
                    print_usage();
                    std::process::exit(2);
                }
            }
        }

        if backend.is_empty() {
            eprintln!("error: --backend is required (lattice|candle)");
            print_usage();
            std::process::exit(2);
        }
        let model = model.unwrap_or_else(|| {
            eprintln!("error: --model is required (path to a local model directory)");
            print_usage();
            std::process::exit(2);
        });

        Self {
            backend,
            model,
            prompt,
            max_tokens,
            runs,
            warmup,
        }
    }
}

fn print_usage() {
    eprintln!(
        "\nUsage: lattice_bench --backend <lattice|candle> --model <dir> [options]\n\n\
         Options:\n\
         \x20 --prompt <text>       Prompt text (default: embedded ~50-token prompt)\n\
         \x20 --max-tokens <n>      Max tokens to generate (default: 128)\n\
         \x20 --runs <n>            Timed runs to report (default: 3)\n\
         \x20 --warmup <n>          Warmup runs, excluded from stats (default: 1)\n"
    );
}

// ---------------------------------------------------------------------------
// GPU advisory flock (fleet convention: /tmp/lion-metal-gpu-test.lock)
// ---------------------------------------------------------------------------

/// Raw `flock(2)` bindings. `ruvllm` has no direct dependency that exposes
/// file locking (the transitive `libc` crate pulled in by other deps is not
/// usable from an example — only the package's own `[dependencies]` /
/// `[dev-dependencies]` are visible here), so this declares the two libc
/// symbols it needs directly via `extern "C"`. On any Unix target the C
/// runtime (and therefore `libc.so`/`libSystem.dylib`, which exports
/// `flock`) is always linked in, so no new Cargo dependency is required and
/// `Cargo.toml` is left untouched.
#[cfg(unix)]
mod gpu_lock {
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }

    const LOCK_EX: i32 = 2;

    /// Holds the lock file open for the process lifetime; the flock is
    /// released when this (and the underlying fd) is dropped, or when the
    /// process exits.
    pub struct GpuLock {
        _file: std::fs::File,
    }

    pub fn acquire() -> GpuLock {
        let path = "/tmp/lion-metal-gpu-test.lock";
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .unwrap_or_else(|e| panic!("failed to open GPU lock file {path}: {e}"));
        let fd = file.as_raw_fd();

        eprintln!("[gpu_lock] waiting for exclusive flock on {path} ...");
        let start = std::time::Instant::now();
        // SAFETY: `fd` is a valid, open file descriptor owned by `file` for
        // the duration of this call; `flock` only reads/writes kernel lock
        // state and does not touch the buffer.
        let rc = unsafe { flock(fd, LOCK_EX) };
        if rc != 0 {
            panic!(
                "flock(LOCK_EX) failed on {path}: {}",
                std::io::Error::last_os_error()
            );
        }
        eprintln!(
            "[gpu_lock] acquired after {:.1}s",
            start.elapsed().as_secs_f64()
        );
        GpuLock { _file: file }
    }
}

#[cfg(not(unix))]
mod gpu_lock {
    pub struct GpuLock;
    pub fn acquire() -> GpuLock {
        eprintln!("[gpu_lock] non-unix target: no GPU flock taken");
        GpuLock
    }
}

// ---------------------------------------------------------------------------
// Measurement types
// ---------------------------------------------------------------------------

/// One backend's honest identity for the report table: what was actually
/// loaded, not what was requested. Every row carries this so two rows can
/// never be misread as "the same model."
#[derive(Clone)]
struct ModelStats {
    model_path: String,
    quantization: String,
    num_parameters: usize,
}

impl ModelStats {
    fn from_info(model_path: &str, info: &ModelInfo) -> Self {
        Self {
            model_path: model_path.to_string(),
            quantization: info
                .quantization
                .map(|q| format!("{q:?}"))
                .unwrap_or_else(|| "unknown".to_string()),
            num_parameters: info.num_parameters,
        }
    }
}

/// One timed run's numbers. `ttft_ms` is only populated for the streamed
/// leg (lattice); candle's blocking-only measurement leaves it `None`.
#[derive(Debug, Clone, serde::Serialize)]
struct RunResult {
    run_index: usize,
    is_warmup: bool,
    leg: &'static str, // "stream" | "blocking"
    ttft_ms: Option<f64>,
    duration_ms: f64,
    total_tokens: usize,
    tokens_per_second: f64,
    /// True when `total_tokens` came from re-encoding the returned text
    /// (blocking legs) rather than being reported natively by the decode
    /// loop (streamed `Done` event) — an approximation, always labeled.
    tokens_approximate: bool,
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Print the markdown summary table for one backend/model run.
fn print_report(
    backend_name: &str,
    stats: &ModelStats,
    prompt_tokens: usize,
    max_tokens: usize,
    runs: &[RunResult],
) {
    println!("\n## {backend_name} backend — {}\n", stats.model_path);
    println!(
        "model={} | quantization={} | params={} | max_tokens={} | prompt_tokens={}\n",
        stats.model_path, stats.quantization, stats.num_parameters, max_tokens, prompt_tokens
    );

    for leg in ["stream", "blocking"] {
        let leg_runs: Vec<&RunResult> = runs
            .iter()
            .filter(|r| !r.is_warmup && r.leg == leg)
            .collect();
        if leg_runs.is_empty() {
            continue;
        }
        let approx = leg_runs[0].tokens_approximate;
        let ttfts: Vec<f64> = leg_runs.iter().filter_map(|r| r.ttft_ms).collect();
        let durations: Vec<f64> = leg_runs.iter().map(|r| r.duration_ms).collect();
        let toks_per_s: Vec<f64> = leg_runs.iter().map(|r| r.tokens_per_second).collect();

        println!(
            "### leg: {leg}{}\n",
            if approx {
                " (tokens approximate — re-encoded from output text)"
            } else {
                ""
            }
        );
        println!("| metric | median ({} runs) |", leg_runs.len());
        println!("|---|---|");
        if !ttfts.is_empty() {
            println!("| TTFT (ms) | {:.1} |", median(&ttfts));
        }
        println!("| duration (ms) | {:.1} |", median(&durations));
        println!("| decode tok/s | {:.2} |", median(&toks_per_s));
        println!();
    }
}

fn emit_jsonl(backend_name: &str, stats: &ModelStats, run: &RunResult) {
    let line = serde_json::json!({
        "backend": backend_name,
        "model_path": stats.model_path,
        "quantization": stats.quantization,
        "num_parameters": stats.num_parameters,
        "run_index": run.run_index,
        "is_warmup": run.is_warmup,
        "leg": run.leg,
        "ttft_ms": run.ttft_ms,
        "duration_ms": run.duration_ms,
        "total_tokens": run.total_tokens,
        "tokens_per_second": run.tokens_per_second,
        "tokens_approximate": run.tokens_approximate,
    });
    eprintln!("{line}");
}

// ---------------------------------------------------------------------------
// lattice backend leg
// ---------------------------------------------------------------------------

#[cfg(all(feature = "lattice", target_os = "macos"))]
fn run_lattice(args: &Args) {
    use ruvllm::ModelConfig;

    let model_path = args.model.display().to_string();

    let mut backend = LatticeBackend::new().expect("LatticeBackend::new is infallible today");

    let load_start = Instant::now();
    backend
        .load_model(&model_path, ModelConfig::default())
        .unwrap_or_else(|e| panic!("lattice load_model({model_path}) failed: {e}"));
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!("\nload_model wall time: {load_ms:.1} ms");

    let info = backend
        .model_info()
        .expect("model_info() must be Some after a successful load_model");
    let stats = ModelStats::from_info(&model_path, &info);

    let tokenizer = backend
        .tokenizer()
        .expect("tokenizer() must be Some after load");
    let prompt_tokens = tokenizer
        .encode(&args.prompt)
        .expect("prompt must tokenize")
        .len();

    let total = args.warmup + args.runs;
    let mut results = Vec::with_capacity(total * 2);

    for i in 0..total {
        let is_warmup = i < args.warmup;
        let params = if std::env::var("BENCH_GREEDY").is_ok() {
            GenerateParams::default()
                .with_max_tokens(args.max_tokens)
                .with_temperature(0.0)
                .with_top_k(1)
                .with_top_p(1.0)
                .with_repetition_penalty(1.0)
        } else {
            GenerateParams::default().with_max_tokens(args.max_tokens)
        };

        // Streamed leg: TTFT + native decode tok/s from the Done event.
        let stream_start = Instant::now();
        let stream = backend
            .generate_stream_v2(&args.prompt, params.clone())
            .expect("generate_stream_v2 failed");
        let mut ttft_ms = None;
        let mut total_tokens = 0usize;
        let mut tokens_per_second = 0.0f64;
        for event in stream {
            match event.expect("stream event error") {
                StreamEvent::Token(_) => {
                    if ttft_ms.is_none() {
                        ttft_ms = Some(stream_start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                StreamEvent::Done {
                    total_tokens: n,
                    tokens_per_second: tps,
                    ..
                } => {
                    total_tokens = n;
                    tokens_per_second = tps;
                }
                StreamEvent::Error(e) => panic!("lattice stream error: {e}"),
            }
        }
        let stream_duration_ms = stream_start.elapsed().as_secs_f64() * 1000.0;
        let stream_result = RunResult {
            run_index: i,
            is_warmup,
            leg: "stream",
            ttft_ms,
            duration_ms: stream_duration_ms,
            total_tokens,
            tokens_per_second,
            tokens_approximate: false,
        };
        emit_jsonl("lattice", &stats, &stream_result);
        results.push(stream_result);

        // Blocking leg: apples-to-apples wall-clock vs candle. Token count
        // is approximate — re-encoded from the returned text, since
        // `generate()` does not report a native token count.
        let block_start = Instant::now();
        let text = backend
            .generate(&args.prompt, params)
            .expect("generate failed");
        let block_duration_ms = block_start.elapsed().as_secs_f64() * 1000.0;
        let block_tokens = tokenizer
            .encode(&text)
            .expect("generated text must tokenize")
            .len();
        let block_tps = if block_duration_ms > 0.0 {
            block_tokens as f64 / (block_duration_ms / 1000.0)
        } else {
            0.0
        };
        let block_result = RunResult {
            run_index: i,
            is_warmup,
            leg: "blocking",
            ttft_ms: None,
            duration_ms: block_duration_ms,
            total_tokens: block_tokens,
            tokens_per_second: block_tps,
            tokens_approximate: true,
        };
        emit_jsonl("lattice", &stats, &block_result);
        results.push(block_result);
    }

    print_report("lattice", &stats, prompt_tokens, args.max_tokens, &results);
}

#[cfg(not(all(feature = "lattice", target_os = "macos")))]
fn run_lattice(_args: &Args) {
    eprintln!(
        "error: this binary was built without the 'lattice' feature (or not on macOS); \
         rebuild with --features lattice on macOS to use --backend lattice"
    );
    std::process::exit(1);
}

// ---------------------------------------------------------------------------
// candle backend leg
// ---------------------------------------------------------------------------

/// Candle is measured via the blocking `generate()` call.
/// `CandleBackend::generate_stream_v2` replays tokens only after the whole
/// generation has finished (see the module doc above), so it reports no
/// meaningful TTFT; do not use it here for timing.
#[cfg(feature = "candle")]
fn run_candle(args: &Args) {
    let model_path = args.model.display().to_string();

    let mut backend =
        CandleBackend::new().unwrap_or_else(|e| panic!("CandleBackend::new failed: {e}"));

    // `CandleBackend::load_model` dispatches on the path: a directory
    // containing `tokenizer.json` + `config.json` + `.safetensors` files is
    // loaded via `load_safetensors` (candle_backend.rs:1248-1297) — the same
    // "local directory" form the lattice backend expects, so `--model` takes
    // a directory for both backends. `ModelConfig::default()`'s
    // `architecture` field defaults to `ModelArchitecture::Llama`, which
    // matches the llama-3.2-1b checkpoint this harness targets; a different
    // model directory would need an explicit architecture override (not
    // exposed on this CLI — this harness is scoped to the two models named
    // in its usage docs above).
    //
    // Honesty guard (O3: precision label MUST match the actual artifact):
    // candle's `load_safetensors` echoes `config.quantization` verbatim into
    // `ModelInfo` (candle_backend.rs:949), and `ModelConfig::default()` says
    // `Some(Quantization::Q4K)` — a false label for a safetensors checkpoint.
    // Derive the label from the checkpoint's own config.json `torch_dtype`
    // instead; abort rather than print a wrong precision column.
    let torch_dtype = std::fs::read_to_string(args.model.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get("torch_dtype")
                .and_then(|d| d.as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| panic!("cannot read torch_dtype from {}/config.json", model_path));
    let quant = match torch_dtype.as_str() {
        "bfloat16" => ruvllm::backends::Quantization::Bf16,
        "float16" => ruvllm::backends::Quantization::F16,
        "float32" => ruvllm::backends::Quantization::None,
        other => panic!("unmapped torch_dtype '{other}' — refusing to guess a precision label"),
    };
    let model_config = ruvllm::ModelConfig {
        quantization: Some(quant),
        // candle's flash-attn feature is CUDA-only; the default (true) panics
        // with unimplemented! inside candle_transformers::models::llama on
        // Metal. Standard attention is candle's real macOS path.
        use_flash_attention: false,
        ..ruvllm::ModelConfig::default()
    };
    let load_start = Instant::now();
    backend
        .load_model(&model_path, model_config)
        .unwrap_or_else(|e| panic!("candle load_model({model_path}) failed: {e}"));
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!("\nload_model wall time: {load_ms:.1} ms");

    let info = backend
        .model_info()
        .expect("model_info() must be Some after a successful load_model");
    let stats = ModelStats::from_info(&model_path, &info);

    let tokenizer = backend
        .tokenizer()
        .expect("tokenizer() must be Some after load");
    let prompt_tokens = tokenizer
        .encode(&args.prompt)
        .expect("prompt must tokenize")
        .len();

    let total = args.warmup + args.runs;
    let mut results = Vec::with_capacity(total);

    for i in 0..total {
        let is_warmup = i < args.warmup;
        let params = if std::env::var("BENCH_GREEDY").is_ok() {
            GenerateParams::default()
                .with_max_tokens(args.max_tokens)
                .with_temperature(0.0)
                .with_top_k(1)
                .with_top_p(1.0)
                .with_repetition_penalty(1.0)
        } else {
            GenerateParams::default().with_max_tokens(args.max_tokens)
        };

        let block_start = Instant::now();
        let text = backend
            .generate(&args.prompt, params)
            .expect("generate failed");
        let block_duration_ms = block_start.elapsed().as_secs_f64() * 1000.0;
        let block_tokens = tokenizer
            .encode(&text)
            .expect("generated text must tokenize")
            .len();
        let block_tps = if block_duration_ms > 0.0 {
            block_tokens as f64 / (block_duration_ms / 1000.0)
        } else {
            0.0
        };
        let result = RunResult {
            run_index: i,
            is_warmup,
            leg: "blocking",
            ttft_ms: None,
            duration_ms: block_duration_ms,
            total_tokens: block_tokens,
            tokens_per_second: block_tps,
            tokens_approximate: true,
        };
        emit_jsonl("candle", &stats, &result);
        results.push(result);
    }

    print_report("candle", &stats, prompt_tokens, args.max_tokens, &results);
}

#[cfg(not(feature = "candle"))]
fn run_candle(_args: &Args) {
    eprintln!(
        "error: this binary was built without the 'candle' feature; \
         rebuild with --features candle to use --backend candle"
    );
    std::process::exit(1);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    // Hold the GPU flock for the process lifetime (fleet convention);
    // dropped implicitly at process exit.
    let _gpu_lock = gpu_lock::acquire();

    match args.backend.as_str() {
        "lattice" => run_lattice(&args),
        "candle" => run_candle(&args),
        other => {
            eprintln!("error: unknown --backend '{other}', expected 'lattice' or 'candle'");
            print_usage();
            std::process::exit(2);
        }
    }

    // Keep `Duration` import used even on build configs where no timing
    // path above happens to construct one directly (both backend arms do,
    // but this keeps the import warning-free under `-D warnings` on any
    // future refactor).
    let _ = Duration::from_secs(0);
}
