//! Launch flow: preflight -> (dry-run stop) -> create -> supervise -> destroy.

use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::audit::{self, Audit};
use crate::budget::{self, Budget, Watch};
use crate::gcs::{GcsPlan, MAX_SIGNED_HOURS};
use crate::job::{self, JobSpec, ResumeFetch};
use crate::offer::{self, CostModel, Estimate, Offer, OfferFilter};
use crate::resume::{self, ResumeSource};
use crate::vast::{Client, LABEL_PREFIX};

pub struct LaunchCfg {
    pub filter: OfferFilter,
    pub cost: CostModel,
    pub budget: Budget,
    pub job: JobSpec,
    pub gcs: GcsPlan,
    /// Checkpoint to restore before the command (resolved at launch).
    pub resume: Option<ResumeSource>,
    pub local_repo: String,
    pub poll_secs: u64,
    pub startup_timeout_mins: f64,
    pub allow_concurrent: bool,
    pub dry_run: bool,
}

/// Owns a live (billing) instance. Dropping it without an explicit destroy
/// (panic, early return, `?`) still destroys the instance.
struct LiveGuard<'a> {
    client: &'a Client,
    audit: &'a Audit,
    id: u64,
    run_id: String,
    created: Instant,
    hourly: f64,
    bandwidth: f64,
    done: bool,
}

impl LiveGuard<'_> {
    fn elapsed_h(&self) -> f64 {
        self.created.elapsed().as_secs_f64() / 3600.0
    }

    fn destroy(&mut self, reason: &str) -> Result<()> {
        if self.done {
            return Ok(());
        }
        eprintln!("[rvgr] destroying instance {} (reason: {reason})", self.id);
        let res = self.client.destroy_verified(self.id);
        let elapsed_h = self.elapsed_h();
        let _ = self.audit.append(json!({
            "event": "destroy", "run_id": self.run_id, "instance_id": self.id,
            "reason": reason, "verified_gone": res.is_ok(),
            "elapsed_h": elapsed_h, "hourly_usd": self.hourly,
            "est_cost_usd": elapsed_h * self.hourly + self.bandwidth,
            "stopped_at": chrono::Utc::now().to_rfc3339(),
            "error": res.as_ref().err().map(|e| format!("{e:#}")),
        }));
        if res.is_ok() {
            audit::clear_live(self.id);
            self.done = true;
        }
        res
    }
}

impl Drop for LiveGuard<'_> {
    fn drop(&mut self) {
        if !self.done {
            if let Err(e) = self.destroy("drop-guard") {
                eprintln!("[rvgr] !!! {e:#}");
            }
        }
    }
}

fn print_offer(o: &Offer, e: &Estimate) {
    println!("chosen offer:");
    println!("  offer_id        {}", o.id);
    println!(
        "  gpu             {} x{} ({:.0} GB)",
        o.gpu_name,
        o.num_gpus,
        o.gpu_ram / 1000.0
    );
    println!(
        "  host            machine {:?}, {}, datacenter={}",
        o.machine_id,
        o.geolocation.as_deref().unwrap_or("?"),
        o.is_datacenter()
    );
    println!(
        "  verification    {:?}, reliability {:.4}",
        o.verification.as_deref().unwrap_or("?"),
        o.reliability2.unwrap_or(0.0)
    );
    println!("  cuda_max_good   {:?}", o.cuda_max_good);
    println!(
        "  cpu             {} | {:.1} effective threads (host {:?}) | {:.0} GB RAM",
        o.cpu_name.as_deref().unwrap_or("?").trim(),
        o.effective_cores().unwrap_or(0.0),
        o.cpu_cores,
        o.cpu_ram.unwrap_or(0.0) / 1000.0
    );
    println!("cost estimate:");
    println!("  listed dph      ${:.4}/h", o.dph_total);
    println!("  compute         ${:.4}/h", e.compute_hourly_usd);
    println!("  disk storage    ${:.4}/h", e.storage_hourly_usd);
    println!("  planning rate   ${:.4}/h", e.hourly_usd);
    if let Some(c) = o.effective_cores() {
        println!(
            "  per core-hour   ${:.5} (planning rate / {c:.1} effective threads)",
            e.hourly_usd / c
        );
        let p = o.cpu_perf();
        println!(
            "  cpu family      {} (speed x{:.2} per thread; throughput {:.1})",
            p.family,
            p.factor,
            c * p.factor
        );
        if p.factor > 0.0 {
            println!(
                "  per tput-hour   ${:.5} (planning rate / throughput; CPU-mode rank key)",
                e.hourly_usd / (c * p.factor)
            );
        }
    }
    println!("  transfer        ${:.4} (one-off)", e.bandwidth_usd);
    println!(
        "  worst case      ${:.4} ({}h cap)",
        e.worst_case_usd, e.max_hours
    );
}

pub fn launch(cfg: &LaunchCfg, client: &Client, audit: &Audit) -> Result<i32> {
    let j = &cfg.job;
    let mut blockers: Vec<String> = Vec::new();
    if cfg.budget.max_hours + 0.5 > MAX_SIGNED_HOURS {
        blockers.push(format!("--max-hours must be <= {}", MAX_SIGNED_HOURS - 0.5));
    }

    blockers.extend(cfg.filter.problems());

    // 1. Offers (live, read-only). The server cannot order by $/core-hour,
    // so CPU mode pulls a wider page and ranks client-side.
    let limit = if cfg.filter.cpu_mode { 512 } else { 64 };
    let q = cfg.filter.query_json(cfg.cost.disk_gb, limit);
    let offers = client.search_offers(&q)?;
    let (ranked, rejected) = offer::rank(&offers, &cfg.filter, &cfg.cost);
    println!(
        "offers: {} returned, {} acceptable, {} rejected client-side",
        offers.len(),
        ranked.len(),
        rejected.len()
    );
    for (id, why) in rejected.iter().take(5) {
        println!("  rejected {id}: {why}");
    }
    let Some(chosen) = ranked.first().cloned() else {
        audit.append(
            json!({"event": "refuse", "run_id": j.run_id, "reason": "no acceptable offer",
            "dry_run": cfg.dry_run}),
        )?;
        bail!("no offer satisfies the filters");
    };
    for o in ranked.iter().take(5) {
        let e = offer::estimate(o, &cfg.cost, 1.0);
        let cpu = match (cfg.filter.cpu_mode, o.effective_cores()) {
            (true, Some(c)) => format!(
                " {c:.0}thr x{:.2} {:.0}GB ${:.5}/core-h ${:.5}/tput-h [{}]",
                o.cpu_perf().factor,
                o.cpu_ram.unwrap_or(0.0) / 1000.0,
                e.hourly_usd / c,
                o.usd_per_throughput_hour(&cfg.cost).unwrap_or(f64::NAN),
                o.cpu_name.as_deref().unwrap_or("?").trim()
            ),
            _ => String::new(),
        };
        println!(
            "  candidate {} {} x{} ${:.4}/h{cpu} rel={:.4} dc={} {}",
            o.id,
            if o.gpu_name.is_empty() {
                "no-gpu"
            } else {
                &o.gpu_name
            },
            o.num_gpus,
            e.hourly_usd,
            o.reliability2.unwrap_or(0.0),
            o.is_datacenter(),
            o.geolocation.as_deref().unwrap_or("")
        );
    }
    let est = offer::estimate(&chosen, &cfg.cost, cfg.budget.max_hours);
    print_offer(&chosen, &est);

    // 2. Budget + credit.
    let credit = client.credit_usd().ok();
    println!(
        "account credit:   {}",
        credit.map_or("unknown".into(), |c| format!("${c:.2}"))
    );
    blockers.extend(budget::plan_blockers(&cfg.budget, &est, credit));

    // 3. No stray instances from this tool.
    let stray: Vec<u64> = client
        .list_mine()?
        .iter()
        .filter(|i| {
            i.get("label")
                .and_then(Value::as_str)
                .is_some_and(|l| l.starts_with(LABEL_PREFIX))
        })
        .filter_map(|i| i.get("id").and_then(Value::as_u64))
        .collect();
    if !stray.is_empty() && !cfg.allow_concurrent {
        blockers.push(format!(
            "existing {LABEL_PREFIX}* instances {stray:?} (run `reap` or pass --allow-concurrent)"
        ));
    }

    // 4. Git ref/SHA reachable on the remote the instance clones from.
    match job::verify_remote_sha(&cfg.local_repo, &j.repo, &j.git_ref, &j.git_sha) {
        Ok(tip) => println!(
            "git:              {}@{} verified (remote tip {})",
            j.git_ref,
            &j.git_sha[..12],
            &tip[..12]
        ),
        Err(e) => blockers.push(format!("git: {e:#}")),
    }

    // 5. GCS sink.
    let need_reader = cfg.resume.is_some();
    let gcs_problems = cfg.gcs.readiness(need_reader);
    println!(
        "artifacts:        {}/{{artifacts.tar.gz,job.log,DONE,FAILED}}",
        cfg.gcs.prefix()
    );
    if j.checkpointing() {
        println!(
            "checkpoints:      {} every {}s -> {}/ckpt-{{0..{}}}.tar.gz + {} ({:.1} GB transfer budgeted)",
            j.checkpoint_dir,
            j.checkpoint_secs,
            cfg.gcs.prefix(),
            j.checkpoint_ring - 1,
            crate::gcs::CKPT_POINTER,
            cfg.cost.upload_gb
        );
    } else {
        println!("checkpoints:      off (--checkpoint-secs 0)");
    }
    if !gcs_problems.is_empty() {
        println!(
            "GCS setup needed (not performed automatically):\n{}",
            cfg.gcs.setup_commands(need_reader)
        );
        blockers.extend(gcs_problems.into_iter().map(|p| format!("gcs: {p}")));
    }

    // 6. Resume source (read-only, operator credentials).
    let mut job = j.clone();
    let mut resolved = None;
    if let Some(src) = &cfg.resume {
        match resume::resolve(src, &cfg.gcs) {
            Ok(r) => {
                println!(
                    "resume from:      {} (sha256 {})",
                    r.gs_url,
                    r.sha256.as_deref().unwrap_or("not recorded")
                );
                job.resume = Some(ResumeFetch {
                    sha256: r.sha256.clone(),
                });
                resolved = Some(r);
            }
            Err(e) => blockers.push(format!("resume: {e:#}")),
        }
    }
    let j = &job;

    let onstart = job::render_onstart(j);
    let placeholder: serde_json::Map<String, Value> = j
        .url_env_names()
        .into_iter()
        .map(|n| {
            let kind = if n == "RVGR_URL_RESUME" {
                "<signed GET>"
            } else {
                "<signed PUT>"
            };
            (n, json!(kind))
        })
        .collect();
    let mut body = json!({
        "client_id": "me", "image": j.image, "disk": cfg.cost.disk_gb, "label": j.run_id,
        "runtype": "ssh", "onstart": onstart, "env": placeholder,
    });
    println!("image:            {}", j.image);
    println!(
        "onstart:          {} bytes, job timeout {}s",
        onstart.len(),
        j.job_timeout_secs
    );
    let plan_rec = json!({
        "run_id": j.run_id, "offer_id": chosen.id, "gpu": chosen.gpu_name,
        "hourly_usd": est.hourly_usd, "worst_case_usd": est.worst_case_usd,
        "max_usd": cfg.budget.max_usd, "max_hours": cfg.budget.max_hours,
        "image": j.image, "git_ref": j.git_ref, "git_sha": j.git_sha,
        "blockers": blockers, "dry_run": cfg.dry_run,
        "cpu_mode": cfg.filter.cpu_mode, "cpu_name": chosen.cpu_name,
        "cpu_cores_effective": chosen.cpu_cores_effective, "cpu_ram_mb": chosen.cpu_ram,
        "usd_per_core_hour": chosen.usd_per_core_hour(&cfg.cost),
        "cpu_family": chosen.cpu_perf().family, "cpu_speed_factor": chosen.cpu_perf().factor,
        "usd_per_throughput_hour": chosen.usd_per_throughput_hour(&cfg.cost),
        "checkpoint_secs": j.checkpoint_secs, "checkpoint_ring": j.checkpoint_ring,
        "checkpoint_dir": j.checkpoint_dir,
        "resume_from": resolved.as_ref().map(|r| r.gs_url.clone()),
        "resume_sha256": resolved.as_ref().and_then(|r| r.sha256.clone()),
    });

    if cfg.dry_run {
        let mut rec = plan_rec;
        rec["event"] = json!("dry_run");
        audit.append(rec)?;
        println!("\nDRY-RUN: create call skipped (PUT /asks/{}/).", chosen.id);
        if blockers.is_empty() {
            println!("verdict: WOULD LAUNCH");
            return Ok(0);
        }
        println!("verdict: WOULD REFUSE");
        for b in &blockers {
            println!("  - {b}");
        }
        return Ok(2);
    }
    if !blockers.is_empty() {
        let mut rec = plan_rec;
        rec["event"] = json!("refuse");
        audit.append(rec)?;
        for b in &blockers {
            eprintln!("refusing: {b}");
        }
        return Ok(2);
    }

    // ---- Paid path below. ----
    let stop = Arc::new(AtomicBool::new(false));
    {
        let s = stop.clone();
        ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))?;
    }
    let ring = if j.checkpointing() {
        j.checkpoint_ring
    } else {
        0
    };
    let urls = cfg
        .gcs
        .sign_all((cfg.budget.max_hours + 0.5).min(MAX_SIGNED_HOURS), ring)?;
    let mut env = json!({"RVGR_URL_ARTIFACTS": urls.artifacts, "RVGR_URL_LOG": urls.log,
        "RVGR_URL_DONE": urls.done, "RVGR_URL_FAILED": urls.failed});
    for (i, u) in urls.ckpt_slots.into_iter().enumerate() {
        env[format!("RVGR_URL_CKPT_{i}")] = json!(u);
    }
    if let Some(u) = urls.ckpt_pointer {
        env["RVGR_URL_CKPT_LATEST"] = json!(u);
    }
    if let Some(r) = &resolved {
        // Short-lived: only needs to outlive boot + clone.
        let hours = ((cfg.startup_timeout_mins + 30.0) / 60.0).min(MAX_SIGNED_HOURS);
        env["RVGR_URL_RESUME"] = json!(cfg.gcs.sign_get(&r.gs_url, hours)?);
    }
    let missing: Vec<String> = j
        .url_env_names()
        .into_iter()
        .filter(|n| env.get(n).is_none())
        .collect();
    if !missing.is_empty() {
        bail!("internal: signed URLs missing for {missing:?}; nothing launched");
    }
    body["env"] = env;
    if stop.load(Ordering::SeqCst) {
        bail!("interrupted before create; nothing launched");
    }
    let id = match client.create(chosen.id, &body) {
        Ok(id) => id,
        Err(e) => {
            // A transport error may hide a successful create: sweep by label.
            eprintln!(
                "[rvgr] create error: {e:#}; checking for an orphan labelled {}",
                j.run_id
            );
            for inst in client.list_mine().unwrap_or_default() {
                if inst.get("label").and_then(Value::as_str) == Some(j.run_id.as_str()) {
                    if let Some(oid) = inst.get("id").and_then(Value::as_u64) {
                        let r = client.destroy_verified(oid);
                        audit.append(json!({"event": "destroy", "run_id": j.run_id,
                            "instance_id": oid, "reason": "orphan-after-create-error",
                            "verified_gone": r.is_ok()}))?;
                    }
                }
            }
            return Err(e);
        }
    };
    drop(body);
    let started_at = chrono::Utc::now().to_rfc3339();
    let mut guard = LiveGuard {
        client,
        audit,
        id,
        run_id: j.run_id.clone(),
        created: Instant::now(),
        hourly: est.hourly_usd,
        bandwidth: est.bandwidth_usd,
        done: false,
    };
    let mut rec = plan_rec;
    rec["event"] = json!("launch");
    rec["instance_id"] = json!(id);
    rec["started_at"] = json!(started_at);
    audit::mark_live(id, &rec)?;
    audit.append(rec)?;
    println!("launched instance {id}; supervising (Ctrl-C destroys it)");

    let reason = supervise(cfg, client, &mut guard, &stop);
    guard.destroy(&reason)?;
    println!("instance {id} destroyed and verified gone ({reason})");
    Ok(if reason == "job-done" { 0 } else { 1 })
}

fn supervise(cfg: &LaunchCfg, client: &Client, g: &mut LiveGuard, stop: &AtomicBool) -> String {
    let startup = Duration::from_secs_f64(cfg.startup_timeout_mins * 60.0);
    let mut running = false;
    let mut price_checked = false;
    loop {
        for _ in 0..cfg.poll_secs.max(5) {
            if stop.load(Ordering::SeqCst) {
                return "interrupted".into();
            }
            sleep(Duration::from_secs(1));
        }
        match budget::watchdog(&cfg.budget, g.hourly, g.bandwidth, g.elapsed_h()) {
            Watch::BudgetCap { spent_usd } => return format!("budget-cap (${spent_usd:.2})"),
            Watch::Timeout { .. } => return "timeout".into(),
            Watch::Ok { spent_usd } => eprint!(
                "\r[rvgr] {:.2}h elapsed, ~${spent_usd:.2} spent   ",
                g.elapsed_h()
            ),
        }
        if let Some(m) = cfg.gcs.marker() {
            return if m == "DONE" {
                "job-done".into()
            } else {
                "job-failed".into()
            };
        }
        let inst = match client.show(g.id) {
            Ok(Some(i)) => i,
            Ok(None) => return "instance-vanished".into(),
            Err(e) => {
                eprintln!("\n[rvgr] show failed: {e:#}");
                continue;
            }
        };
        if !price_checked {
            if let Some(actual) = inst.get("dph_total").and_then(Value::as_f64) {
                price_checked = true;
                if actual > g.hourly * 1.05 {
                    return format!(
                        "price-mismatch (actual ${actual:.4}/h > planned ${:.4}/h)",
                        g.hourly
                    );
                }
            }
        }
        let status = inst
            .get("actual_status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if status == "running" {
            running = true;
        } else if running && matches!(status, "exited" | "offline") {
            return format!("instance-{status}");
        } else if !running && g.created.elapsed() > startup {
            return format!("startup-timeout (status {status})");
        }
    }
}
