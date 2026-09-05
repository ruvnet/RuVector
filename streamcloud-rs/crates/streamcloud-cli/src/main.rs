//! `streamcloud` — native demo CLI for StreamCloud.
//!
//! Runs the streaming reconstruction pipeline over a synthetic (or, later, real)
//! video source and writes a streaming **MP4** plus optional **PNG** snapshots.
//!
//! ```text
//! streamcloud render --frames 60 --width 640 --height 480 --out out.mp4 \
//!                --png-dir frames --fps 20 --top-k 16
//! streamcloud inspect --weights model.safetensors
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use streamcloud_io::{mp4enc::Mp4Sink, save_png, FrameSink, MultiSink, PngSequenceSink};
use streamcloud_model::{ModelConfig, SyntheticReconstructor};
use streamcloud_pipeline::{scene, SoftwareRenderer, StreamingReconstructor};
use streamcloud_tensor::WeightIndex;

#[derive(Parser)]
#[command(
    name = "streamcloud",
    version,
    about = "StreamCloud — streaming 3D reconstruction demo"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run streaming reconstruction and export PNG + MP4.
    Render(RenderArgs),
    /// Inspect a safetensors checkpoint (header only — no multi-GB load).
    Inspect(InspectArgs),
}

#[derive(Parser)]
struct RenderArgs {
    /// Number of frames to process.
    #[arg(long, default_value_t = 60)]
    frames: u64,
    /// Output width (even).
    #[arg(long, default_value_t = 640)]
    width: usize,
    /// Output height (even).
    #[arg(long, default_value_t = 480)]
    height: usize,
    /// Output MP4 path.
    #[arg(long, default_value = "streamcloud_out.mp4")]
    out: PathBuf,
    /// Optional directory for per-frame PNG snapshots.
    #[arg(long)]
    png_dir: Option<PathBuf>,
    /// Frames per second for the MP4.
    #[arg(long, default_value_t = 20)]
    fps: u32,
    /// Top-K anchors retrieved per frame from the HNSW memory.
    #[arg(long, default_value_t = 16)]
    top_k: usize,
    /// Camera orbit per frame, radians (shows 3D parallax).
    #[arg(long, default_value_t = 0.03)]
    orbit: f32,
    /// Optional checkpoint to validate before running (header only).
    #[arg(long)]
    weights: Option<PathBuf>,
}

#[derive(Parser)]
struct InspectArgs {
    /// Path to a safetensors checkpoint.
    #[arg(long)]
    weights: PathBuf,
    /// Optional config.json to load alongside.
    #[arg(long)]
    config: Option<PathBuf>,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Render(a) => render(a),
        Command::Inspect(a) => inspect(a),
    }
}

fn render(a: RenderArgs) -> Result<()> {
    if a.width % 2 != 0 || a.height % 2 != 0 {
        anyhow::bail!("width and height must be even (H.264 4:2:0)");
    }

    if let Some(w) = &a.weights {
        match WeightIndex::from_file(w) {
            Ok(idx) => println!(
                "checkpoint ok: {} tensors, {} params ({:.2} GB f32-equivalent)",
                idx.tensors.len(),
                idx.total_params(),
                idx.total_params() as f64 * 4.0 / 1e9
            ),
            Err(e) => {
                eprintln!("warning: could not read checkpoint ({e}); using synthetic backend")
            }
        }
        println!("note: reconstruction uses the synthetic backend (see ADR-0006).");
    }

    let cfg = ModelConfig {
        keyframe_feature_dim: 512,
        anchor_top_k: a.top_k,
        ..Default::default()
    };
    let model = SyntheticReconstructor::new(cfg).with_sample_step(4);
    let mut pipeline =
        StreamingReconstructor::new(model, a.top_k).context("failed to init pipeline")?;
    let renderer = SoftwareRenderer::new(a.width, a.height);

    // Build the sink fan-out: MP4 always, PNG sequence if requested.
    let mut sinks: Vec<Box<dyn FrameSink>> = Vec::new();
    sinks.push(Box::new(
        Mp4Sink::new(&a.out, a.fps).context("failed to create MP4 sink")?,
    ));
    if let Some(dir) = &a.png_dir {
        sinks.push(Box::new(PngSequenceSink::new(dir, "frame_")?));
    }
    let mut sink = MultiSink::new(sinks);

    println!(
        "rendering {} frames @ {}x{}, fps={}, top_k={} → {}",
        a.frames,
        a.width,
        a.height,
        a.fps,
        a.top_k,
        a.out.display()
    );

    // Capture the final frame as a still PNG too.
    let mut last_rgba: Option<Vec<u8>> = None;
    let renderer_ref = &renderer;
    let mut frame_id = 0u64;
    let frames = scene::synthetic_stream(a.frames, a.width, a.height);
    // We re-implement the loop here (vs run_stream) to grab the last frame.
    for frame in frames {
        let (pc, _res) = pipeline.process(&frame)?;
        let rgba = renderer_ref.render(&pc, frame_id as f32 * a.orbit);
        sink.push(&rgba, a.width, a.height)?;
        last_rgba = Some(rgba);
        frame_id += 1;
    }
    sink.finish()?;

    if let (Some(rgba), Some(dir)) = (&last_rgba, a.png_dir.as_ref()) {
        let still = dir.join("final.png");
        save_png(&still, rgba, a.width, a.height)?;
        println!("final still: {}", still.display());
    }

    println!(
        "done: {} frames, {} keyframes in memory, MP4 {} bytes",
        frame_id,
        pipeline.memory().len(),
        std::fs::metadata(&a.out).map(|m| m.len()).unwrap_or(0)
    );
    Ok(())
}

fn inspect(a: InspectArgs) -> Result<()> {
    if let Some(c) = &a.config {
        let cfg = ModelConfig::from_file(c).context("failed to read config.json")?;
        println!("config: {cfg:#?}");
    }
    let idx = WeightIndex::from_file(&a.weights).context("failed to read checkpoint header")?;
    println!("checkpoint: {}", a.weights.display());
    println!("  tensors: {}", idx.tensors.len());
    println!("  params:  {}", idx.total_params());
    println!(
        "  approx:  {:.2} GB (f32-equivalent)",
        idx.total_params() as f64 * 4.0 / 1e9
    );
    // Show a few tensors.
    for (name, info) in idx.tensors.iter().take(8) {
        println!("    {name}: {:?} {}", info.shape, info.dtype);
    }
    Ok(())
}
