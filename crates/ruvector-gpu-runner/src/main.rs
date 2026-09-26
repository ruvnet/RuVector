//! ruvector-gpu-runner: budget-capped, auto-destroying vast.ai job runner.
//!
//! Subcommands:
//!   launch  — select an offer, gate on budget/credit/git/GCS, (optionally)
//!             create the instance and supervise it until DONE/FAILED/cap.
//!   reap    — find and destroy instances created by this tool (SIGKILL recovery).

mod audit;
mod budget;
mod gcs;
mod job;
mod offer;
mod runner;
mod vast;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::path::PathBuf;

use crate::audit::Audit;
use crate::offer::RELIABILITY_FLOOR;

const DEFAULT_IMAGE: &str = "nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5";

#[derive(Parser)]
#[command(
    name = "ruvector-gpu-runner",
    version,
    about = "Budget-capped vast.ai GPU job runner"
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
    /// Allowed GPU model(s); repeatable.
    #[arg(long = "gpu", default_values_t = ["RTX 4090".to_string(), "RTX A6000".to_string()])]
    gpus: Vec<String>,
    #[arg(long, default_value_t = 1)]
    num_gpus: u32,
    #[arg(long, default_value_t = 24.0)]
    min_gpu_ram_gb: f64,
    #[arg(long, default_value_t = 12.4)]
    min_cuda: f64,
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
    /// Container image pinned by digest.
    #[arg(long, default_value = DEFAULT_IMAGE)]
    image: String,
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
    #[arg(long, default_value = "ruvector-openjev-artifacts")]
    bucket: String,
    #[arg(long, default_value = "ruv-dev")]
    gcp_project: String,
    #[arg(long, default_value = "rvgr-uploader@ruv-dev.iam.gserviceaccount.com")]
    signer_sa: String,
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

fn launch(a: LaunchArgs, audit: &Audit) -> Result<i32> {
    job::validate_sha(&a.git_sha)?;
    job::validate_image(&a.image)?;
    job::validate_artifact_dir(&a.artifact_dir)?;
    if !a.max_usd.is_finite() || !a.max_hours.is_finite() || a.max_usd <= 0.0 || a.max_hours <= 0.0
    {
        bail!("--max-usd and --max-hours must be positive");
    }
    let run_id = format!(
        "{}{}-{}",
        vast::LABEL_PREFIX,
        chrono::Utc::now().format("%Y%m%dt%H%M%S"),
        &a.git_sha[..8]
    );
    // In-instance timeout leaves 10 min for upload + marker before the local cap.
    let job_timeout_secs = ((a.max_hours * a.watchdog_fraction * 3600.0) as u64)
        .saturating_sub(600)
        .max(60);
    let cfg = runner::LaunchCfg {
        filter: offer::OfferFilter {
            gpu_names: a.gpus,
            num_gpus: a.num_gpus,
            min_gpu_ram_gb: a.min_gpu_ram_gb,
            min_cuda: a.min_cuda,
            max_dph: a.max_dph,
            min_reliability: a.min_reliability,
            require_datacenter: a.require_datacenter,
        },
        cost: offer::CostModel {
            disk_gb: a.disk_gb,
            upload_gb: a.expected_upload_gb,
            download_gb: a.expected_download_gb,
        },
        budget: budget::Budget {
            max_usd: a.max_usd,
            max_hours: a.max_hours,
            watchdog_fraction: a.watchdog_fraction,
        },
        job: job::JobSpec {
            run_id: run_id.clone(),
            image: a.image,
            repo: a.repo,
            git_ref: a.git_ref,
            git_sha: a.git_sha.to_lowercase(),
            command: a.cmd,
            artifact_dir: a.artifact_dir,
            job_timeout_secs,
        },
        gcs: gcs::GcsPlan {
            bucket: a.bucket,
            signer_sa: a.signer_sa,
            project: a.gcp_project,
            run_id: run_id.clone(),
        },
        local_repo: a.local_repo,
        poll_secs: a.poll_secs,
        startup_timeout_mins: a.startup_timeout_mins,
        allow_concurrent: a.allow_concurrent,
        dry_run: a.dry_run,
    };
    println!(
        "run_id:           {run_id}{}",
        if cfg.dry_run { "  [DRY-RUN]" } else { "" }
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
