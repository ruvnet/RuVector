//! ruvector-gpu-runner: budget-capped, auto-destroying vast.ai job runner.
//!
//! Subcommands:
//!   launch  — select an offer, gate on budget/credit/git/GCS, (optionally)
//!             create the instance and supervise it until DONE/FAILED/cap.
//!   reap    — find and destroy instances created by this tool (SIGKILL recovery).

mod audit;
mod budget;
mod cpu_perf;
mod gcs;
mod job;
mod offer;
#[cfg(all(test, unix))]
mod onstart_exec_tests;
mod resume;
mod runner;
mod vast;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::path::PathBuf;

use crate::audit::Audit;
use crate::offer::RELIABILITY_FLOOR;

// ubuntu24.04 (glibc 2.39): ort 2.0.0-rc.13 prebuilt onnxruntime needs glibc >= 2.38.
const DEFAULT_IMAGE: &str = "nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a";
/// `--cpu-mode` default: plain ubuntu:24.04 (multi-arch index digest,
/// 2026-09-27). Not a CUDA image, so no host driver/CUDA floor applies.
const DEFAULT_CPU_IMAGE: &str =
    "ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3";
const DEFAULT_GPUS: [&str; 2] = ["RTX 4090", "RTX A6000"];
const DEFAULT_MIN_CUDA: f64 = 12.6;

#[derive(Parser)]
#[command(
    name = "ruvector-gpu-runner",
    version,
    about = "Budget-capped vast.ai GPU/CPU job runner"
)]
struct Cli {
    /// JSONL audit log (default: $XDG_STATE_HOME/ruvector-gpu-runner/audit.jsonl).
    #[arg(long, global = true)]
    audit_log: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Launch(Box<LaunchArgs>),
    Reap {
        /// Destroy only this instance id (still idempotent + verified).
        #[arg(long)]
        id: Option<u64>,
        /// Actually destroy (default: list only).
        #[arg(long)]
        yes: bool,
    },
}

fn reliability(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if !(RELIABILITY_FLOOR..=1.0).contains(&v) {
        return Err(format!("must be within [{RELIABILITY_FLOOR}, 1.0]"));
    }
    Ok(v)
}

#[derive(clap::Args)]
struct LaunchArgs {
    /// Hard spend cap in USD (required).
    #[arg(long)]
    max_usd: f64,
    /// Hard wall-clock cap in hours (required, <= 11.5 because of signed-URL expiry).
    #[arg(long)]
    max_hours: f64,
    /// Do everything except the paid create call.
    #[arg(long)]
    dry_run: bool,
    /// CPU mode: rank by $/effective-core-hour, allow any GPU (or none), and
    /// default to a non-CUDA image. All other safety checks are unchanged.
    #[arg(long)]
    cpu_mode: bool,
    /// Minimum effective (allocated) CPU threads (`cpu_cores_effective`).
    #[arg(long, default_value_t = 0.0)]
    min_cpu_cores: f64,
    /// Minimum RAM allocated to the offer, GB (`cpu_ram`).
    #[arg(long, default_value_t = 0.0)]
    min_ram_gb: f64,
    /// CPU mode: minimum per-thread speed factor of the CPU family (EPYC
    /// Zen 2 = 1.0; see README). Xeon Phi / Atom / Celeron / Pentium /
    /// Opteron are refused regardless.
    #[arg(long, default_value_t = cpu_perf::DEFAULT_MIN_CPU_SPEED)]
    min_cpu_speed: f64,
    /// Allowed GPU model(s); repeatable. Default: RTX 4090, RTX A6000
    /// (GPU mode) or any (CPU mode).
    #[arg(long = "gpu")]
    gpus: Vec<String>,
    /// Exact GPU count. Default 1 (GPU mode) or any (CPU mode; 0 = any).
    #[arg(long)]
    num_gpus: Option<u32>,
    /// Default 24 (GPU mode) or 0 (CPU mode).
    #[arg(long)]
    min_gpu_ram_gb: Option<f64>,
    /// Host CUDA floor. Default 12.6, except 0 (unchecked) in --cpu-mode with a
    /// non-nvidia/cuda image.
    #[arg(long)]
    min_cuda: Option<f64>,
    /// Max planning rate ($/h, including disk storage).
    #[arg(long, default_value_t = 1.0)]
    max_dph: f64,
    #[arg(long, default_value_t = RELIABILITY_FLOOR, value_parser = reliability)]
    min_reliability: f64,
    /// Only datacenter hosts (otherwise they are merely preferred).
    #[arg(long)]
    require_datacenter: bool,
    #[arg(long, default_value_t = 60.0)]
    disk_gb: f64,
    #[arg(long, default_value_t = 2.0)]
    expected_upload_gb: f64,
    #[arg(long, default_value_t = 20.0)]
    expected_download_gb: f64,
    /// Container image pinned by digest (default depends on --cpu-mode).
    #[arg(long)]
    image: Option<String>,
    #[arg(long, default_value = "https://github.com/ruvnet/ruvector.git")]
    repo: String,
    #[arg(long, default_value = "feat/openjev")]
    git_ref: String,
    /// Full 40-char commit SHA to run (must be reachable from --git-ref on --repo).
    #[arg(long)]
    git_sha: String,
    /// Shell command run inside the checkout (cwd = repo root).
    #[arg(long)]
    cmd: String,
    /// Directory on the instance tarred and uploaded as artifacts.tar.gz.
    #[arg(long, default_value = "/workspace/out")]
    artifact_dir: String,
    /// Directory checkpointed every --checkpoint-secs and restored by
    /// --resume-from (default: --artifact-dir). Write checkpoints atomically
    /// (tmp file + rename): the uploader tars it while the job runs.
    #[arg(long)]
    checkpoint_dir: Option<String>,
    /// Checkpoint upload interval in seconds (0 disables periodic uploads).
    #[arg(long, default_value_t = 600)]
    checkpoint_secs: u64,
    /// Rolling checkpoint slots ckpt-0..N-1.tar.gz (2..=8).
    #[arg(long, default_value_t = 3)]
    checkpoint_ring: usize,
    /// Expected size of one checkpoint tarball, GB (for the transfer estimate).
    #[arg(long, default_value_t = 0.5)]
    expected_checkpoint_gb: f64,
    /// Restore a checkpoint before the command: a previous run id (uses its
    /// ckpt-latest.json pointer, sha256-verified) or gs://<bucket>/…tar.gz.
    #[arg(long)]
    resume_from: Option<String>,
    #[arg(long, default_value = "ruvector-openjev-artifacts")]
    bucket: String,
    #[arg(long, default_value = "ruv-dev")]
    gcp_project: String,
    /// Bucket location. Passed to `sign-url` so the keyless signer SA (which
    /// only holds objectCreator) never needs storage.buckets.get.
    #[arg(long, default_value = "us-central1")]
    bucket_region: String,
    #[arg(long, default_value = "rvgr-uploader@ruv-dev.iam.gserviceaccount.com")]
    signer_sa: String,
    /// Keyless read-only SA that signs the resume GET URL (objectViewer).
    #[arg(long, default_value = "rvgr-reader@ruv-dev.iam.gserviceaccount.com")]
    reader_sa: String,
    /// Local clone used to check SHA reachability.
    #[arg(long, default_value = ".")]
    local_repo: String,
    #[arg(long, default_value_t = 30)]
    poll_secs: u64,
    /// Destroy at this fraction of the $ / hour caps.
    #[arg(long, default_value_t = 0.95)]
    watchdog_fraction: f64,
    #[arg(long, default_value_t = 20.0)]
    startup_timeout_mins: f64,
    #[arg(long)]
    allow_concurrent: bool,
}

/// Host CUDA floor unless `--min-cuda` is given. A GPU launch always keeps the
/// 12.6 driver floor whatever the image; only a CPU-mode launch of a non
/// `nvidia/cuda` image leaves it unchecked.
fn default_min_cuda(cpu_mode: bool, image: &str) -> f64 {
    if !cpu_mode || image.contains("nvidia/cuda") {
        DEFAULT_MIN_CUDA
    } else {
        0.0
    }
}

/// Conservative effective instance -> GCS rate for the post-job tar + upload
/// (about 50 Mbit/s including gzip), in seconds per GB.
const XFER_SECS_PER_GB: f64 = 160.0;
/// Fixed reserve after the transfer window: loop stop, log, marker, slack.
const MARKER_RESERVE_SECS: u64 = 600;

/// Post-job transfer window: artifacts plus, when separate, one final
/// checkpoint. An in-flight periodic upload is cut, not awaited, so it does
/// not count. At least 2 minutes.
fn final_xfer_secs(upload_gb: f64, ckpt_gb: f64, final_ckpt: bool) -> u64 {
    let gb = upload_gb.max(0.0) + if final_ckpt { ckpt_gb.max(0.0) } else { 0.0 };
    ((gb * XFER_SECS_PER_GB).ceil() as u64).max(120)
}

/// In-instance timeout: the watchdog cap minus the transfer window minus the
/// marker reserve, so DONE/FAILED lands before the local watchdog destroys.
fn job_timeout_secs(max_hours: f64, watchdog_fraction: f64, xfer_secs: u64) -> Result<u64> {
    let cap = (max_hours * watchdog_fraction * 3600.0) as u64;
    let reserve = MARKER_RESERVE_SECS + xfer_secs;
    let t = cap.saturating_sub(reserve);
    if t < 60 {
        bail!(
            "--max-hours leaves {cap}s before the watchdog but {reserve}s are reserved for \
             the final transfers + marker; raise --max-hours or lower --expected-upload-gb / \
             --expected-checkpoint-gb"
        );
    }
    Ok(t)
}

/// Periodic checkpoint uploads over the whole cap, plus the final one.
fn checkpoint_uploads(max_hours: f64, secs: u64) -> f64 {
    if secs == 0 {
        0.0
    } else {
        (max_hours * 3600.0 / secs as f64).floor() + 1.0
    }
}

fn launch(a: LaunchArgs, audit: &Audit) -> Result<i32> {
    job::validate_sha(&a.git_sha)?;
    let image = a.image.clone().unwrap_or_else(|| {
        if a.cpu_mode {
            DEFAULT_CPU_IMAGE
        } else {
            DEFAULT_IMAGE
        }
        .to_string()
    });
    job::validate_image(&image)?;
    job::validate_artifact_dir(&a.artifact_dir)?;
    let checkpoint_dir = a
        .checkpoint_dir
        .clone()
        .unwrap_or_else(|| a.artifact_dir.clone());
    job::validate_checkpoint_dir(&checkpoint_dir)?;
    if a.checkpoint_secs > 0 && !(2..=gcs::MAX_CKPT_RING).contains(&a.checkpoint_ring) {
        bail!(
            "--checkpoint-ring must be within 2..={}",
            gcs::MAX_CKPT_RING
        );
    }
    if a.checkpoint_secs > 0 && a.checkpoint_secs < 60 {
        bail!("--checkpoint-secs must be 0 (off) or >= 60");
    }
    if !a.max_usd.is_finite() || !a.max_hours.is_finite() || a.max_usd <= 0.0 || a.max_hours <= 0.0
    {
        bail!("--max-usd and --max-hours must be positive");
    }
    let resume = a
        .resume_from
        .as_deref()
        .map(|s| resume::parse(s, &a.bucket))
        .transpose()?;
    let gpu_names = if a.gpus.is_empty() && !a.cpu_mode {
        DEFAULT_GPUS.iter().map(|s| s.to_string()).collect()
    } else {
        a.gpus
    };
    let min_cuda = a
        .min_cuda
        .unwrap_or_else(|| default_min_cuda(a.cpu_mode, &image));
    let ckpt_gb =
        a.expected_checkpoint_gb.max(0.0) * checkpoint_uploads(a.max_hours, a.checkpoint_secs);
    let run_id = format!(
        "{}{}-{}",
        vast::LABEL_PREFIX,
        chrono::Utc::now().format("%Y%m%dt%H%M%S"),
        &a.git_sha[..8]
    );
    let final_ckpt = a.checkpoint_secs > 0 && !job::same_dir(&checkpoint_dir, &a.artifact_dir);
    let final_xfer_secs =
        final_xfer_secs(a.expected_upload_gb, a.expected_checkpoint_gb, final_ckpt);
    let job_timeout_secs = job_timeout_secs(a.max_hours, a.watchdog_fraction, final_xfer_secs)?;
    let cfg = runner::LaunchCfg {
        filter: offer::OfferFilter {
            gpu_names,
            num_gpus: a.num_gpus.unwrap_or(if a.cpu_mode { 0 } else { 1 }),
            min_gpu_ram_gb: a
                .min_gpu_ram_gb
                .unwrap_or(if a.cpu_mode { 0.0 } else { 24.0 }),
            min_cuda,
            max_dph: a.max_dph,
            min_reliability: a.min_reliability,
            require_datacenter: a.require_datacenter,
            cpu_mode: a.cpu_mode,
            min_cpu_cores: a.min_cpu_cores,
            min_ram_gb: a.min_ram_gb,
            min_cpu_speed: a.min_cpu_speed,
        },
        cost: offer::CostModel {
            disk_gb: a.disk_gb,
            upload_gb: a.expected_upload_gb + ckpt_gb,
            download_gb: a.expected_download_gb
                + if resume.is_some() {
                    a.expected_checkpoint_gb.max(0.0)
                } else {
                    0.0
                },
        },
        budget: budget::Budget {
            max_usd: a.max_usd,
            max_hours: a.max_hours,
            watchdog_fraction: a.watchdog_fraction,
        },
        job: job::JobSpec {
            run_id: run_id.clone(),
            image,
            repo: a.repo,
            git_ref: a.git_ref,
            git_sha: a.git_sha.to_lowercase(),
            command: a.cmd,
            artifact_dir: a.artifact_dir,
            job_timeout_secs,
            checkpoint_dir,
            checkpoint_secs: a.checkpoint_secs,
            checkpoint_ring: if a.checkpoint_secs > 0 {
                a.checkpoint_ring
            } else {
                0
            },
            resume: None, // filled in by the runner after resolving `resume`
            final_xfer_secs,
        },
        gcs: gcs::GcsPlan {
            bucket: a.bucket,
            signer_sa: a.signer_sa,
            reader_sa: a.reader_sa,
            project: a.gcp_project,
            region: a.bucket_region,
            run_id: run_id.clone(),
        },
        resume,
        local_repo: a.local_repo,
        poll_secs: a.poll_secs,
        startup_timeout_mins: a.startup_timeout_mins,
        allow_concurrent: a.allow_concurrent,
        dry_run: a.dry_run,
    };
    println!(
        "run_id:           {run_id}{}{}",
        if cfg.dry_run { "  [DRY-RUN]" } else { "" },
        if cfg.filter.cpu_mode {
            "  [CPU-MODE]"
        } else {
            ""
        }
    );
    let client = vast::Client::from_env()?;
    runner::launch(&cfg, &client, audit)
}

fn reap(id: Option<u64>, yes: bool, audit: &Audit) -> Result<i32> {
    let client = vast::Client::from_env()?;
    let mine = client.list_mine()?;
    let mut targets: Vec<(u64, String)> = mine
        .iter()
        .filter_map(|i| {
            let iid = i.get("id").and_then(Value::as_u64)?;
            let label = i
                .get("label")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let ours = label.starts_with(vast::LABEL_PREFIX) || audit::live_ids().contains(&iid);
            (id.map_or(ours, |want| want == iid)).then_some((iid, label))
        })
        .collect();
    if let Some(want) = id {
        if targets.is_empty() {
            targets.push((want, "<not listed>".into()));
        }
    }
    if targets.is_empty() {
        println!(
            "no ruvector-gpu-runner instances found ({} total instances owned)",
            mine.len()
        );
        for stale in audit::live_ids() {
            audit::clear_live(stale);
        }
        return Ok(0);
    }
    for (iid, label) in &targets {
        println!("instance {iid} label={label}");
        if yes {
            let r = client.destroy_verified(*iid);
            audit.append(
                json!({"event": "destroy", "instance_id": iid, "reason": "reap",
                "label": label, "verified_gone": r.is_ok()}),
            )?;
            r?;
            audit::clear_live(*iid);
            println!("  destroyed + verified gone");
        }
    }
    if !yes {
        println!("(list only; pass --yes to destroy)");
    }
    Ok(0)
}

fn main() {
    let cli = Cli::parse();
    let audit = Audit::new(
        cli.audit_log
            .unwrap_or_else(|| audit::state_dir().join("audit.jsonl")),
    );
    let res = match cli.cmd {
        Cmd::Launch(a) => launch(*a, &audit),
        Cmd::Reap { id, yes } => reap(id, yes, &audit),
    };
    match res {
        Ok(code) => {
            eprintln!("audit log: {}", audit.path().display());
            std::process::exit(code)
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_cuda_floor_kept_in_gpu_mode_for_any_image() {
        let pytorch = format!("pytorch/pytorch@sha256:{}", "a".repeat(64));
        let cuda = format!("nvidia/cuda:12.4.1-devel@sha256:{}", "a".repeat(64));
        let ubuntu = format!("ubuntu:24.04@sha256:{}", "a".repeat(64));
        assert_eq!(default_min_cuda(false, &pytorch), DEFAULT_MIN_CUDA);
        assert_eq!(default_min_cuda(false, &ubuntu), DEFAULT_MIN_CUDA);
        assert_eq!(default_min_cuda(false, &cuda), DEFAULT_MIN_CUDA);
        assert_eq!(default_min_cuda(true, &cuda), DEFAULT_MIN_CUDA);
        assert_eq!(default_min_cuda(true, &ubuntu), 0.0);
        assert_eq!(default_min_cuda(true, &pytorch), 0.0);
    }

    #[test]
    fn timeout_reserves_transfer_window() {
        // 2 GB artifacts + 0.5 GB separate final checkpoint = 400 s window.
        let x = final_xfer_secs(2.0, 0.5, true);
        assert_eq!(x, 400);
        assert_eq!(final_xfer_secs(2.0, 0.5, false), 320);
        assert_eq!(final_xfer_secs(0.0, 0.0, false), 120);
        // A multi-GB output dir grows the reserve instead of a fixed 600 s.
        let big = final_xfer_secs(20.0, 20.0, true);
        assert_eq!(big, 6400);
        let t = job_timeout_secs(3.0, 0.95, x).unwrap();
        assert_eq!(t, (3.0 * 0.95 * 3600.0) as u64 - 600 - 400);
        let tb = job_timeout_secs(3.0, 0.95, big).unwrap();
        let cap = (3.0 * 0.95 * 3600.0) as u64;
        assert_eq!(tb, cap - 600 - 6400);
        // Too short a cap for the declared transfers is refused, not clamped.
        assert!(job_timeout_secs(0.25, 0.95, x).is_err());
        assert!(job_timeout_secs(1.0, 0.95, big).is_err());
    }
}
