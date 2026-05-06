//! `ruview-vitals-replay` — synthetic + recorded ADR-018 broadcaster.
//!
//! Used to validate the vitals worker without touching the attached
//! ESP32 hardware. Two modes:
//!
//! * `--mode synth` — build frames in-memory whose subcarrier
//!   amplitudes are modulated by a configurable breathing + heart-rate
//!   sinusoid. Stable, deterministic, exercises the full DSP path
//!   (preprocessor → window → both extractors).
//! * `--mode jsonl --file PATH` — replay a RuView `.csi.jsonl`
//!   recording. The amplitudes feed straight into the ADR-018 I-channel
//!   (Q=0). Pacing follows the recorded timestamps where present;
//!   falls back to `--rate` otherwise.
//!
//! ## Usage
//!
//! ```text
//!   ruview-vitals-replay --target 127.0.0.1:5005 --mode synth \
//!       --breathing-bpm 15 --heart-rate-bpm 72 --duration-secs 60
//!
//!   ruview-vitals-replay --target 127.0.0.1:5005 --mode jsonl \
//!       --file /path/to/recording.csi.jsonl
//! ```
//!
//! The replay tool is *not* shipped to the cluster Pis — it lives in
//! the same crate as the worker for ease of CI but the systemd
//! deploy-bundle only installs `ruview-vitals-worker`.

use std::f64::consts::TAU;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::net::UdpSocket;
use tokio::time::sleep_until;
use tracing_subscriber::EnvFilter;

use ruview_vitals_worker::frame::{ADR018_HEADER_SIZE, CSI_MAGIC_V1};

/// One frame in a RuView `.csi.jsonl` recording.
#[derive(Debug, Clone, Deserialize)]
struct JsonlFrame {
    timestamp: f64,
    subcarriers: Vec<f64>,
    #[serde(default)]
    rssi: Option<f64>,
    #[serde(default)]
    noise_floor: Option<f64>,
    #[serde(default)]
    node_id: Option<u8>,
}

#[derive(Debug, Clone)]
struct Args {
    target: String,
    mode: Mode,
    node_id: u8,
    n_subcarriers: u16,
    n_antennas: u8,
    rate_fps: f64,
    duration_secs: f64,
    breathing_bpm: f64,
    heart_rate_bpm: f64,
    file: Option<PathBuf>,
    rssi_dbm: i8,
    noise_dbm: i8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    Synth,
    Jsonl,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            target: "127.0.0.1:5005".to_string(),
            mode: Mode::Synth,
            node_id: 7,
            n_subcarriers: 56,
            n_antennas: 1,
            rate_fps: 30.0,
            duration_secs: 60.0,
            breathing_bpm: 15.0,
            heart_rate_bpm: 72.0,
            file: None,
            rssi_dbm: -50,
            noise_dbm: -100,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--target" => {
                args.target = iter.next().ok_or_else(|| "--target needs a value".to_string())?;
            }
            "--mode" => {
                let m = iter.next().ok_or_else(|| "--mode needs a value".to_string())?;
                args.mode = match m.as_str() {
                    "synth" => Mode::Synth,
                    "jsonl" => Mode::Jsonl,
                    other => return Err(format!("unknown mode {other:?}")),
                };
            }
            "--file" => {
                let v = iter.next().ok_or_else(|| "--file needs a value".to_string())?;
                args.file = Some(PathBuf::from(v));
            }
            "--node-id" => {
                let v = iter.next().ok_or_else(|| "--node-id needs a value".to_string())?;
                args.node_id = v.parse().map_err(|e| format!("--node-id: {e}"))?;
            }
            "--n-subcarriers" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--n-subcarriers needs a value".to_string())?;
                args.n_subcarriers = v.parse().map_err(|e| format!("--n-subcarriers: {e}"))?;
            }
            "--n-antennas" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--n-antennas needs a value".to_string())?;
                args.n_antennas = v.parse().map_err(|e| format!("--n-antennas: {e}"))?;
            }
            "--rate" => {
                let v = iter.next().ok_or_else(|| "--rate needs a value".to_string())?;
                args.rate_fps = v.parse().map_err(|e| format!("--rate: {e}"))?;
            }
            "--duration-secs" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--duration-secs needs a value".to_string())?;
                args.duration_secs = v.parse().map_err(|e| format!("--duration-secs: {e}"))?;
            }
            "--breathing-bpm" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--breathing-bpm needs a value".to_string())?;
                args.breathing_bpm = v.parse().map_err(|e| format!("--breathing-bpm: {e}"))?;
            }
            "--heart-rate-bpm" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--heart-rate-bpm needs a value".to_string())?;
                args.heart_rate_bpm = v.parse().map_err(|e| format!("--heart-rate-bpm: {e}"))?;
            }
            "--rssi" => {
                let v = iter.next().ok_or_else(|| "--rssi needs a value".to_string())?;
                args.rssi_dbm = v.parse().map_err(|e| format!("--rssi: {e}"))?;
            }
            "--noise" => {
                let v = iter.next().ok_or_else(|| "--noise needs a value".to_string())?;
                args.noise_dbm = v.parse().map_err(|e| format!("--noise: {e}"))?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other:?}")),
        }
    }
    if args.mode == Mode::Jsonl && args.file.is_none() {
        return Err("--mode jsonl requires --file".into());
    }
    Ok(args)
}

fn print_usage() {
    eprintln!(
        "ruview-vitals-replay\n\
         \n\
         Usage:\n  \
           ruview-vitals-replay --target IP:PORT [--mode synth|jsonl] [--file PATH] \\\n  \
                                [--node-id N] [--n-subcarriers N] [--n-antennas N] \\\n  \
                                [--rate FPS] [--duration-secs S] \\\n  \
                                [--breathing-bpm BPM] [--heart-rate-bpm BPM] \\\n  \
                                [--rssi DBM] [--noise DBM]"
    );
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let filter = EnvFilter::try_from_env("RUVIEW_VITALS_LOG")
        .or_else(|_| EnvFilter::try_new("info"))
        .expect("filter");
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            print_usage();
            std::process::exit(2);
        }
    };
    if let Err(e) = run(args).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    socket.connect(&args.target).await?;
    tracing::info!(target=%args.target, mode=?args.mode, "ruview-vitals-replay");

    match args.mode {
        Mode::Synth => run_synth(&socket, &args).await,
        Mode::Jsonl => run_jsonl(&socket, &args).await,
    }
}

async fn run_synth(socket: &UdpSocket, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let total_frames = (args.rate_fps * args.duration_secs).round() as u64;
    let breath_hz = args.breathing_bpm / 60.0;
    let hr_hz = args.heart_rate_bpm / 60.0;
    let frame_period = Duration::from_secs_f64(1.0 / args.rate_fps);
    let start = tokio::time::Instant::now();

    let mut sent = 0u64;
    for i in 0..total_frames {
        let t = i as f64 / args.rate_fps;
        let breath_phase = TAU * breath_hz * t;
        let hr_phase = TAU * hr_hz * t;
        let buf = build_synth_frame(args, i, breath_phase, hr_phase);
        socket.send(&buf).await?;
        sent += 1;
        if sent % 30 == 0 {
            tracing::debug!(sent, "frames");
        }
        let next = start + frame_period * (i as u32 + 1);
        sleep_until(next).await;
    }
    tracing::info!(sent, "replay (synth) done");
    Ok(())
}

async fn run_jsonl(socket: &UdpSocket, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::BufRead;
    let path = args.file.as_ref().expect("file");
    let f = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(f);

    let mut sent = 0u64;
    let mut prev_ts: Option<f64> = None;
    let start = Instant::now();
    let mut elapsed_recording = 0.0_f64;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let frame: JsonlFrame = match serde_json::from_str(&line) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(lineno, error=%e, "skip malformed line");
                continue;
            }
        };
        let buf = jsonl_to_adr018(args, &frame);
        socket.send(&buf).await?;
        sent += 1;

        // Pace either by recorded timestamp deltas or by --rate fallback.
        if let Some(prev) = prev_ts {
            let delta = (frame.timestamp - prev).max(0.0);
            elapsed_recording += delta;
        }
        prev_ts = Some(frame.timestamp);
        let target_elapsed = if elapsed_recording > 0.0 {
            elapsed_recording
        } else {
            sent as f64 / args.rate_fps
        };
        let target = start + Duration::from_secs_f64(target_elapsed);
        let now = Instant::now();
        if target > now {
            tokio::time::sleep(target - now).await;
        }

        if sent % 30 == 0 {
            tracing::debug!(sent, "frames");
        }
    }
    tracing::info!(sent, "replay (jsonl) done");
    Ok(())
}

fn build_synth_frame(args: &Args, sample_index: u64, breath_phase: f64, hr_phase: f64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(
        ADR018_HEADER_SIZE + args.n_subcarriers as usize * 2 * args.n_antennas as usize,
    );
    buf.extend(CSI_MAGIC_V1.to_le_bytes());
    buf.push(args.node_id);
    buf.push(args.n_antennas);
    buf.extend(args.n_subcarriers.to_le_bytes());
    buf.push(11); // channel
    buf.push(args.rssi_dbm as u8);
    buf.push(args.noise_dbm as u8);
    buf.extend([0u8; 5]); // reserved
    let ts_us = (sample_index as f64 * 1.0e6 / args.rate_fps) as u32;
    buf.extend(ts_us.to_le_bytes());

    // Breathing modulation amplitude ≈ ±20 % of base; heart-rate ≈ ±5 %.
    // Per-subcarrier base shape introduces variance so the worker's
    // `subcarrier_variance` weighting picks them out.
    let breath_factor = 1.0 + 0.20 * breath_phase.sin();
    let hr_factor = 1.0 + 0.05 * hr_phase.sin();
    for ant in 0..args.n_antennas {
        for sc in 0..args.n_subcarriers {
            let base = 30.0 + 12.0 * ((sc as f64 * 0.18) + (ant as f64 * 0.5)).sin();
            let amp = (base * breath_factor * hr_factor).round().clamp(-127.0, 127.0) as i8;
            buf.push(amp as u8);
            buf.push(0u8); // Q=0 → phase=0
        }
    }
    buf
}

fn jsonl_to_adr018(args: &Args, frame: &JsonlFrame) -> Vec<u8> {
    let n_sub = frame.subcarriers.len().min(u16::MAX as usize) as u16;
    let mut buf = Vec::with_capacity(ADR018_HEADER_SIZE + n_sub as usize * 2);
    buf.extend(CSI_MAGIC_V1.to_le_bytes());
    buf.push(frame.node_id.unwrap_or(args.node_id));
    buf.push(1); // n_antennas — JSONL is per-frame folded already
    buf.extend(n_sub.to_le_bytes());
    buf.push(11);
    let rssi = frame.rssi.map(|v| v as i32).unwrap_or(args.rssi_dbm as i32);
    let noise = frame
        .noise_floor
        .map(|v| v as i32)
        .unwrap_or(args.noise_dbm as i32);
    buf.push(rssi.clamp(-128, 127) as u8);
    buf.push(noise.clamp(-128, 127) as u8);
    buf.extend([0u8; 5]);
    let ts_us = (frame.timestamp.fract() * 1.0e6) as u32;
    buf.extend(ts_us.to_le_bytes());

    for amp in &frame.subcarriers {
        let i = amp.round().clamp(-127.0, 127.0) as i8;
        buf.push(i as u8);
        buf.push(0u8);
    }
    buf
}
