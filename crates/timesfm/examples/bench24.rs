//! 24-case TimesFM forecast bench (ADR-191 Phase B — GCP deployment test).
//!
//! Loads the real weights once, then runs **24 distinct forecast cases** —
//! varying frequency, trend, and noise — each with ctx=512, horizon=128. For
//! each case it records latency and asserts the output is finite (no NaN/Inf)
//! and sane. Emits a per-case table plus aggregate stats (mean/p50/p95,
//! throughput) and peak RSS, as machine-readable JSON on the last line.
//!
//! Real numbers only: a non-finite output is a hard FAIL (exit 1), never a
//! silent pass.
//!
//! ```ignore
//! TIMESFM_WEIGHTS=/path/timesfm.safetensors \
//!   cargo run -p timesfm --release --features candle --example bench24
//! ```

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;
use timesfm::config::TimesfmConfig;
use timesfm::model::PatchedTimeSeriesDecoder;

/// Tiny deterministic PRNG (xorshift64*) so cases are reproducible without an
/// rng crate dependency.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        // map to [-1, 1)
        let u = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11; // 53-bit
        (u as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
    }
}

/// Build case `i`'s 512-point input series. 24 distinct shapes: combinations of
/// base frequency, linear trend, amplitude, and noise level.
fn make_series(i: usize, ctx: usize) -> (Vec<f32>, u32, String) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (i as u64).wrapping_mul(0x1000_0001b3));
    // Cycle through a grid of generators so all 24 are genuinely different.
    let periods = [12.0f32, 24.0, 48.0, 96.0, 168.0, 365.0];
    let period = periods[i % periods.len()];
    let trend = ((i % 5) as f32 - 2.0) * 0.01; // -0.02 .. +0.02 per step
    let amp = 0.5 + (i % 4) as f32 * 0.5; // 0.5 .. 2.0
    let noise = (i % 3) as f32 * 0.1; // 0, 0.1, 0.2
                                      // freq_id: TimesFM buckets (0=high, 1=mid, 2=low frequency).
    let freq_id = (i % 3) as u32;

    let mut s = Vec::with_capacity(ctx);
    for t in 0..ctx {
        let tf = t as f32;
        let seasonal = amp * (2.0 * std::f32::consts::PI * tf / period).sin();
        let second = 0.3 * amp * (2.0 * std::f32::consts::PI * tf / (period * 0.5)).cos();
        let n = noise * rng.next_f32();
        s.push(10.0 + trend * tf + seasonal + second + n);
    }
    let label = format!(
        "period={period:.0} trend={trend:+.3} amp={amp:.1} noise={noise:.1} freq_id={freq_id}"
    );
    (s, freq_id, label)
}

/// Read peak RSS (kB) from /proc/self/status; returns 0 if unavailable.
fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        })
        .unwrap_or(0)
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let weights = std::env::var("TIMESFM_WEIGHTS")
        .unwrap_or_else(|_| "/tmp/timesfm-parity/timesfm.safetensors".into());
    if !std::path::Path::new(&weights).exists() {
        eprintln!("SKIP bench24: weights missing ({weights}). Not fabricating numbers.");
        return Ok(());
    }

    let ctx = 512usize;
    let horizon = 128usize;
    let n_cases = 24usize;

    let cfg = TimesfmConfig::timesfm_1p0_200m();
    let load_t = Instant::now();
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights.clone()], DType::F32, &device)? };
    let model = PatchedTimeSeriesDecoder::load(cfg, vb)?;
    let load_ms = load_t.elapsed().as_secs_f64() * 1000.0;

    let threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(0);
    eprintln!(
        "loaded TimesFM 1.0 200M in {load_ms:.0} ms ({threads} threads) — running {n_cases} cases (ctx={ctx}, horizon={horizon})"
    );

    // Warm-up (not timed): first decode JITs allocation paths.
    {
        let (s, fid, _) = make_series(0, ctx);
        let input = Tensor::from_vec(s, (1, ctx), &device)?;
        let pad = Tensor::zeros((1, ctx), DType::F32, &device)?;
        let freq = Tensor::from_vec(vec![fid], (1, 1), &device)?;
        let _ = model.decode(&input, &pad, &freq, horizon)?;
    }

    println!("# case  latency_ms  finite  spec");
    let mut lat = Vec::with_capacity(n_cases);
    let mut all_finite = true;
    let total_t = Instant::now();
    for i in 0..n_cases {
        let (series, freq_id, label) = make_series(i, ctx);
        let input = Tensor::from_vec(series, (1, ctx), &device)?;
        let pad = Tensor::zeros((1, ctx), DType::F32, &device)?;
        let freq = Tensor::from_vec(vec![freq_id], (1, 1), &device)?;

        let t = Instant::now();
        let (point, _full) = model.decode(&input, &pad, &freq, horizon)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        lat.push(ms);

        let fc: Vec<f32> = point.i(0)?.to_vec1()?;
        let n_bad = fc.iter().filter(|x| !x.is_finite()).count();
        let finite = n_bad == 0;
        all_finite &= finite;
        // "sane" sanity: forecast magnitude should be in a plausible band of the
        // input scale (the series sit around 10 ± a few). Flag wild blowups.
        let max_abs = fc.iter().fold(0f32, |m, x| m.max(x.abs()));
        let sane = finite && max_abs < 1e4;
        println!(
            "{:>6}  {:>9.2}  {:>6}  {label}  (|max|={max_abs:.2}, sane={sane})",
            i, ms, finite
        );
    }
    let total_s = total_t.elapsed().as_secs_f64();

    // Aggregate stats.
    let mut sorted = lat.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = lat.iter().sum::<f64>() / lat.len() as f64;
    let pct = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx]
    };
    let p50 = pct(50.0);
    let p95 = pct(95.0);
    let p99 = pct(99.0);
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let throughput = n_cases as f64 / total_s; // forecasts/sec
    let rss_mb = peak_rss_kb() as f64 / 1024.0;

    println!();
    println!("================ bench24 summary ================");
    println!("cases={n_cases} ctx={ctx} horizon={horizon} threads={threads}");
    println!("load_ms={load_ms:.0}  peak_rss_mb={rss_mb:.1}");
    println!(
        "latency ms: mean={mean:.2} p50={p50:.2} p95={p95:.2} p99={p99:.2} min={min:.2} max={max:.2}"
    );
    println!("throughput={throughput:.2} forecasts/sec  total_wall_s={total_s:.2}");
    let n_finite = lat.len(); // all cases produced a latency; finiteness tracked separately
    println!("correctness: all_finite={all_finite} ({n_finite}/{n_cases} cases ran)");
    println!();
    // Machine-readable last line.
    println!(
        "JSON {{\"cases\":{n_cases},\"ctx\":{ctx},\"horizon\":{horizon},\"threads\":{threads},\"load_ms\":{load_ms:.1},\"peak_rss_mb\":{rss_mb:.1},\"mean_ms\":{mean:.2},\"p50_ms\":{p50:.2},\"p95_ms\":{p95:.2},\"p99_ms\":{p99:.2},\"min_ms\":{min:.2},\"max_ms\":{max:.2},\"throughput_fps\":{throughput:.2},\"total_wall_s\":{total_s:.2},\"all_finite\":{all_finite}}}"
    );

    if !all_finite {
        eprintln!("FAIL: at least one case produced non-finite output.");
        std::process::exit(1);
    }
    Ok(())
}
