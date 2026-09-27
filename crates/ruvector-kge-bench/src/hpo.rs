//! `--hpo`: run one dataset's pre-registered configs (ADR-007 §3, plan M4) as
//! concurrent valid-only jobs and write per-config receipts plus
//! `selection.json` (see [`crate::selection`]).
//!
//! Each job is this same binary re-invoked in the default `--config` mode
//! (`--threads T`, its own `--out`), so every job gets the runner's valid-only
//! training, early stop, atomic per-epoch checkpoints and sealed receipt
//! unchanged, plus process isolation and a clean kill at the wall cap. At
//! most `parallel_jobs` run at once; total threads = `parallel_jobs ×
//! threads_per_job`.
//!
//! Re-running the same plan is incremental: a config whose run dir already
//! holds a finished receipt is reused, and one with a checkpoint but no
//! receipt (e.g. killed at the wall cap) is resumed with `--resume`.
//!
//! The plan (`HpoPlan`) is written to the results dir, with a UTC timestamp,
//! **before** any job starts: it is the pre-commitment of which configs run
//! and with which stopping rule.

use crate::canon::{atomic_write, utc_now};
use crate::config::RunConfig;
use crate::receipt::{binary_sha256, git_state, host};
use crate::selection::{self, ConfigResult, HPO_SEED};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The pre-registered HPO plan for one dataset (the `--configs` file).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HpoPlan {
    pub dataset: String,
    /// Subset of C1..C8, in launch order.
    pub configs: Vec<String>,
    /// ADR-007 §3 caps HPO at 100 epochs.
    pub max_epochs: usize,
    /// Evaluations (not epochs) without a strictly better valid MRR before
    /// stopping. ADR-007 does not fix it: ASSUMED, recorded in every receipt.
    pub early_stop_patience: usize,
    #[serde(default = "one")]
    pub eval_every: usize,
    /// Free-text reason for the config subset / order (pre-committed).
    #[serde(default)]
    pub rationale: Option<String>,
}

fn one() -> usize {
    1
}

impl HpoPlan {
    pub fn validate(&self) -> Result<()> {
        if self.configs.is_empty() {
            bail!("plan lists no configs");
        }
        let mut seen = std::collections::BTreeSet::new();
        for c in &self.configs {
            if !seen.insert(c) {
                bail!("config {c} listed twice");
            }
            self.run_config(c, 1)?.validate()?;
        }
        if self.max_epochs == 0 || self.max_epochs > 100 {
            bail!("HPO max_epochs must be in 1..=100 (ADR-007 §3)");
        }
        if self.early_stop_patience == 0 || self.eval_every == 0 {
            bail!("early_stop_patience and eval_every must be >= 1");
        }
        Ok(())
    }

    /// The runner config for `id` (seed 100, grid recipe).
    pub fn run_config(&self, id: &str, threads: usize) -> Result<RunConfig> {
        if !(id.starts_with('C') && crate::config::grid(&self.dataset, id).is_ok()) {
            bail!(
                "{id} is not a pre-registered grid config of {}",
                self.dataset
            );
        }
        Ok(RunConfig {
            dataset: self.dataset.clone(),
            config_id: id.to_string(),
            recipe: None,
            seed: HPO_SEED,
            max_epochs: self.max_epochs,
            early_stop_patience: Some(self.early_stop_patience),
            eval_every: self.eval_every,
            threads: Some(threads),
        })
    }
}

/// Where and how the HPO runs.
#[derive(Debug, Clone)]
pub struct HpoOptions {
    pub parallel_jobs: usize,
    pub threads_per_job: usize,
    /// Heavy per-config run dirs (checkpoints, tables). Keep out of git.
    pub runs_dir: PathBuf,
    /// Small outputs: plan, per-config receipts, logs index, selection.json.
    pub results_dir: PathBuf,
    /// Kill still-running jobs after this (they stay resumable).
    pub wall_cap: Option<Duration>,
    pub cache_dir: PathBuf,
    pub repo: PathBuf,
    /// The binary to re-invoke (default: the current executable).
    pub exe: PathBuf,
    pub poll: Duration,
}

struct Job {
    id: String,
    child: Child,
    started: Instant,
}

fn read_json(p: &Path) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(p).with_context(|| format!("read {}", p.display()))?)
        .with_context(|| format!("parse {}", p.display()))
}

/// Refuse a receipt whose run config is not this plan's for `id` (e.g. a
/// stale run dir from a plan with another `max_epochs` / patience). The
/// thread count is not part of the recipe and is ignored.
pub fn check_matches_plan(receipt: &Value, plan: &HpoPlan, id: &str) -> Result<()> {
    let got: RunConfig = serde_json::from_value(receipt["config"]["run"].clone())
        .with_context(|| format!("{id}: receipt config.run"))?;
    let want = plan.run_config(id, got.threads.unwrap_or(1))?;
    if got != want {
        bail!("{id}: existing receipt was produced by a different run config than this plan (use a fresh --runs-dir)");
    }
    Ok(())
}

/// A finished (not interrupted) receipt in `dir`, if any.
fn finished_receipt(dir: &Path) -> Option<Value> {
    let r = read_json(&dir.join("receipt.json")).ok()?;
    (r["stopped"] != json!("interrupted")).then_some(r)
}

fn spawn(plan: &HpoPlan, id: &str, o: &HpoOptions) -> Result<Child> {
    let dir = o.runs_dir.join(id);
    std::fs::create_dir_all(&dir)?;
    let cfg = plan.run_config(id, o.threads_per_job)?;
    let cfg_path = dir.join("run-config.json");
    atomic_write(&cfg_path, serde_json::to_string_pretty(&cfg)?.as_bytes())?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("train.log"))?;
    let mut cmd = Command::new(&o.exe);
    cmd.arg("--config")
        .arg(&cfg_path)
        .arg("--out")
        .arg(&dir)
        .arg("--threads")
        .arg(o.threads_per_job.to_string())
        .arg("--cache-dir")
        .arg(&o.cache_dir)
        .arg("--repo")
        .arg(&o.repo);
    if dir.join(crate::checkpoint::CHECKPOINT_FILE).exists() {
        cmd.arg("--resume");
    }
    cmd.stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    cmd.spawn().with_context(|| format!("spawn job {id}"))
}

/// Run the plan and write `results_dir/{plan.json, receipts/Ck.json,
/// selection.json}`. Returns the selection.
pub fn run(plan: &HpoPlan, o: &HpoOptions) -> Result<Value> {
    plan.validate()?;
    if o.parallel_jobs == 0 || o.threads_per_job == 0 {
        bail!("--parallel-jobs and --threads-per-job must be >= 1");
    }
    std::fs::create_dir_all(&o.results_dir)?;
    std::fs::create_dir_all(o.results_dir.join("receipts"))?;
    let plan_v = json!({
        "plan": plan,
        "committed_at": utc_now(),
        "hpo_seed": HPO_SEED,
        "parallel_jobs": o.parallel_jobs,
        "threads_per_job": o.threads_per_job,
        "wall_cap_secs": o.wall_cap.map(|d| d.as_secs()),
        "patience_status": "ASSUMED (ADR-007 fixes early stop on valid MRR but not the patience)",
    });
    // Pre-commitment: never overwrite an existing plan with a different one.
    let plan_path = o.results_dir.join("plan.json");
    if plan_path.exists() {
        let old = read_json(&plan_path)?;
        if old["plan"] != json!(plan) {
            bail!(
                "{} holds a different plan; use a new results dir",
                plan_path.display()
            );
        }
    } else {
        atomic_write(
            &plan_path,
            serde_json::to_string_pretty(&plan_v)?.as_bytes(),
        )?;
    }
    let plan_v = read_json(&plan_path)?;

    let wall = Instant::now();
    let mut queue: Vec<String> = plan.configs.iter().rev().cloned().collect();
    let mut running: Vec<Job> = Vec::new();
    let mut status = serde_json::Map::new();
    loop {
        while running.len() < o.parallel_jobs {
            let Some(id) = queue.pop() else { break };
            if let Some(r) = finished_receipt(&o.runs_dir.join(&id)) {
                check_matches_plan(&r, plan, &id)?;
                eprintln!("[hpo] {id}: finished receipt present, reused");
                status.insert(id, json!({"status": "reused"}));
                continue;
            }
            if o.wall_cap.is_some_and(|c| wall.elapsed() >= c) {
                status.insert(id, json!({"status": "not_started_wall_cap"}));
                continue;
            }
            eprintln!("[hpo] {id}: start ({} threads)", o.threads_per_job);
            running.push(Job {
                child: spawn(plan, &id, o)?,
                id,
                started: Instant::now(),
            });
        }
        if running.is_empty() && queue.is_empty() {
            break;
        }
        std::thread::sleep(o.poll);
        let capped = o.wall_cap.is_some_and(|c| wall.elapsed() >= c);
        let mut still = Vec::new();
        for mut j in running.drain(..) {
            let secs = j.started.elapsed().as_secs_f64();
            match j.child.try_wait()? {
                Some(st) => {
                    let s = if st.success() { "finished" } else { "failed" };
                    eprintln!("[hpo] {}: {s} ({st}) after {secs:.0}s", j.id);
                    status.insert(
                        j.id,
                        json!({"status": s, "exit": st.code(), "wall_secs": secs}),
                    );
                }
                None if capped => {
                    let _ = j.child.kill();
                    let _ = j.child.wait();
                    eprintln!(
                        "[hpo] {}: killed at wall cap after {secs:.0}s (resumable)",
                        j.id
                    );
                    status.insert(
                        j.id,
                        json!({"status": "killed_wall_cap", "wall_secs": secs}),
                    );
                }
                None => still.push(j),
            }
        }
        running = still;
        if capped {
            for id in queue.drain(..) {
                status.insert(id, json!({"status": "not_started_wall_cap"}));
            }
        }
    }

    let mut results = Vec::new();
    for id in &plan.configs {
        let Some(r) = finished_receipt(&o.runs_dir.join(id)) else {
            continue;
        };
        check_matches_plan(&r, plan, id)?;
        let cr = ConfigResult::from_receipt(&plan.dataset, r)?;
        atomic_write(
            &o.results_dir.join("receipts").join(format!("{id}.json")),
            serde_json::to_string_pretty(&cr.receipt)?.as_bytes(),
        )?;
        results.push(cr);
    }
    if results.is_empty() {
        bail!(
            "no config finished; nothing to select (jobs: {})",
            Value::Object(status)
        );
    }
    let provenance = json!({
        "git": git_state(&o.repo),
        "binary_sha256": binary_sha256(),
        "crate": { "name": env!("CARGO_PKG_NAME"), "version": env!("CARGO_PKG_VERSION") },
        "host": host(),
        "hpo_wall_secs": wall.elapsed().as_secs_f64(),
        "parallel_jobs": o.parallel_jobs,
        "threads_per_job": o.threads_per_job,
    });
    let sel = selection::build(
        &plan.dataset,
        results,
        &plan_v,
        &Value::Object(status),
        provenance,
    )?;
    atomic_write(
        &o.results_dir.join("selection.json"),
        serde_json::to_string_pretty(&sel)?.as_bytes(),
    )?;
    Ok(sel)
}
