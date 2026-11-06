//! Subcommand handlers.

use super::io::{gate_path, load_gates, load_ledger, print_queue, read_json, write_json};
use super::{Ctx, LedgerOp};
use anyhow::{bail, Context, Result};
use ruvector_weaver::gate::{self, GateOptions};
use ruvector_weaver::ledger::{parse_measurement, Ledger};
use ruvector_weaver::model::{
    Ecosystem, GateReport, GateResult, Graph, LedgerEntry, Outcome, WorkKind,
};
use ruvector_weaver::prs;
use ruvector_weaver::registry::NpmRegistry;
use ruvector_weaver::util::{now, slug_key};
use ruvector_weaver::{graph, propose, queue, readme, release, scan};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

pub(super) fn entry(
    repo: &str,
    candidate: Option<String>,
    subject: Option<String>,
    kind: WorkKind,
    outcome: Outcome,
) -> LedgerEntry {
    LedgerEntry {
        ts: now(),
        repo: repo.to_ascii_lowercase(),
        candidate,
        subject,
        kind,
        signal_kind: None,
        outcome,
        pr: None,
        version: None,
        reason: None,
        measurements: vec![],
        gates: vec![],
        location: None,
        severity: None,
        blocked_on: vec![],
    }
}

pub(super) fn ledger_path(ctx: &Ctx) -> Result<PathBuf> {
    ctx.cfg
        .live_ledger()?
        .context("no live ledger path (set WEAVER_LEDGER)")
}

pub fn graph(ctx: &Ctx, fresh_downloads: bool, no_crates: bool, workers: usize) -> Result<()> {
    let opts = graph::GraphOptions {
        fresh_downloads,
        include_crates: !no_crates,
        workers,
    };
    let g = graph::build(&ctx.http, &ctx.cfg, &opts)?;
    write_json(&ctx.graph_path(), &g)?;
    let npm = g
        .packages
        .iter()
        .filter(|p| p.ecosystem == Ecosystem::Npm)
        .count();
    let plat = g.packages.iter().filter(|p| p.platform_binary).count();
    println!(
        "graph: {} packages ({npm} npm, {} crates, {plat} platform binaries kept as edges, skipped in rankings), {} consumer repos -> {}",
        g.packages.len(),
        g.packages.len() - npm,
        g.consumer_repos.len(),
        ctx.graph_path().display()
    );
    for r in g.consumer_repos.iter().take(15) {
        println!(
            "  {:>10}/mo  {:<40} {} pkgs",
            r.monthly_downloads,
            r.slug,
            r.packages.len()
        );
    }
    Ok(())
}

pub fn scan(ctx: &Ctx, mut repos: Vec<String>, all: bool, no_security: bool) -> Result<()> {
    let g: Graph = read_json(&ctx.graph_path())?;
    if all {
        repos.extend(g.consumer_repos.iter().map(|r| r.slug.clone()));
    }
    if repos.is_empty() {
        bail!("give repos (owner/name, URL or path) or --all");
    }
    let reg = NpmRegistry::new(ctx.http.clone());
    for r in repos {
        let (root, slug, commit) = match scan::fetch_repo(&ctx.work, &r) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("scan {r}: {e:#}");
                continue;
            }
        };
        let mut rep = scan::scan_dir(&root, &slug, &ctx.cfg, &reg, &g, !no_security)?;
        rep.commit = commit;
        let out = ctx
            .work
            .join("scan")
            .join(format!("{}.json", slug_key(&slug)));
        write_json(&out, &rep)?;
        let mut by: BTreeMap<&str, usize> = BTreeMap::new();
        for s in &rep.signals {
            *by.entry(s.kind.as_str()).or_default() += 1;
        }
        println!(
            "scan {slug}: {} files, {} signals {by:?} -> {}",
            rep.files_scanned,
            rep.signals.len(),
            out.display()
        );
        for n in &rep.notes {
            println!("  note: {n}");
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gate(
    ctx: &Ctx,
    package: &str,
    (consumer, consumer_package): (Option<String>, Option<String>),
    no_smoke: bool,
    new: bool,
    dir: Option<PathBuf>,
    run_tests: bool,
    record: bool,
) -> Result<()> {
    let g: Graph = read_json(&ctx.graph_path())?;
    let reg = NpmRegistry::new(ctx.http.clone());
    let ledger = load_ledger(ctx)?;
    let rep = if let Some(dir) = dir {
        let mut results: Vec<GateResult> = Vec::new();
        if new {
            results.extend(gate::new_package_source_gates(&dir, &ctx.cfg)?);
            if run_tests {
                results.push(gate::run_tests(&dir, Duration::from_secs(900))?);
            }
            if let Some(cap) = ctx.cfg.policy.lp.max_new_packages_per_day {
                let used =
                    ledger.new_packages_since(chrono::Utc::now() - chrono::Duration::hours(24));
                results.push(GateResult {
                    name: "rate_limit".into(),
                    passed: used < cap,
                    detail: format!("{used}/{cap} new names in 24h"),
                });
            }
        }
        results.extend(gate::pack_gates(&dir, &ctx.cfg, &ctx.http)?);
        let (name, checks) = release::dir_checks(&ctx.http, &dir, &ctx.cfg.policy.npm)?;
        if name != package {
            bail!("--dir holds {name}, not {package}");
        }
        results.extend(release::with_overrides(&ledger, package, checks));
        gate::finish(package, None, results, vec![])
    } else {
        if new {
            bail!("--new needs --dir <package dir>");
        }
        let opts = GateOptions {
            consumer: consumer.as_deref(),
            consumer_package: consumer_package.as_deref(),
            smoke: !no_smoke,
            ledger: Some(&ledger),
        };
        gate::gate(&ctx.cfg, &ctx.http, &reg, &g, package, &opts)?
    };
    write_json(&gate_path(&ctx.work, package), &rep)?;
    for r in &rep.results {
        println!(
            "  [{}] {:<20} {}",
            if r.passed { "pass" } else { "FAIL" },
            r.name,
            r.detail
        );
    }
    println!(
        "gate {package}: {}",
        if rep.passed {
            "PASSED"
        } else {
            "REJECTED (publish refused)"
        }
    );
    if record {
        let mut e = entry(
            consumer.as_deref().unwrap_or("-"),
            Some(package.to_string()),
            None,
            if new {
                WorkKind::Create
            } else {
                WorkKind::Integrate
            },
            Outcome::Gated,
        );
        e.version = rep.version.clone();
        e.reason = Some(if rep.passed {
            "all gates passed".into()
        } else {
            "gate rejected".into()
        });
        e.gates = rep.results.clone();
        Ledger::append(&ledger_path(ctx)?, &e)?;
    }
    if !rep.passed {
        std::process::exit(2);
    }
    Ok(())
}

fn build_queue(
    ctx: &Ctx,
    gates: &BTreeMap<String, GateReport>,
    props: &propose::Proposals,
) -> Result<ruvector_weaver::model::Queue> {
    let g: Graph = read_json(&ctx.graph_path())?;
    let reports = scan::load_reports(&ctx.work)?;
    let ledger = load_ledger(ctx)?;
    let live = prs::load_all(&ctx.work)?;
    let runs = ruvector_weaver::saturation::load_all(&ctx.work)?;
    Ok(queue::build(
        &ctx.cfg,
        queue::QueueInputs {
            graph: &g,
            reports: &reports,
            ledger: &ledger,
            gates,
            creates: props.items.clone(),
            create_suppressed: props.suppressed.clone(),
            prs: &live,
            runs: &runs,
        },
    ))
}

fn proposals(ctx: &Ctx) -> Result<propose::Proposals> {
    let g: Graph = read_json(&ctx.graph_path())?;
    let reports = scan::load_reports(&ctx.work)?;
    let ledger = load_ledger(ctx)?;
    Ok(propose::propose(
        &ctx.cfg,
        &g,
        &reports,
        &ledger,
        &NpmRegistry::new(ctx.http.clone()),
    ))
}

pub fn queue(ctx: &Ctx, gate_top: usize, no_create: bool, show: usize) -> Result<()> {
    let mut gates = load_gates(&ctx.work)?;
    let props = if no_create {
        propose::Proposals {
            items: vec![],
            suppressed: vec![],
        }
    } else {
        proposals(ctx)?
    };
    let mut q = build_queue(ctx, &gates, &props)?;
    if gate_top > 0 {
        let g: Graph = read_json(&ctx.graph_path())?;
        let ledger = load_ledger(ctx)?;
        let reg = NpmRegistry::new(ctx.http.clone());
        let mut todo: Vec<String> = Vec::new();
        for c in q.items.iter().filter_map(|i| i.candidate.clone()) {
            if !gates.contains_key(&c) && g.get(&c).is_some() && !todo.contains(&c) {
                todo.push(c);
            }
        }
        for c in todo.into_iter().take(gate_top) {
            let opts = GateOptions {
                consumer: None,
                consumer_package: None,
                smoke: true,
                ledger: Some(&ledger),
            };
            let rep = gate::gate(&ctx.cfg, &ctx.http, &reg, &g, &c, &opts)?;
            println!("gate {c}: {}", if rep.passed { "pass" } else { "REJECTED" });
            write_json(&gate_path(&ctx.work, &c), &rep)?;
            gates.insert(c, rep);
        }
        q = build_queue(ctx, &gates, &props)?;
    }
    write_json(&ctx.work.join("queue.json"), &q)?;
    print_queue(&q, show);
    Ok(())
}

pub fn propose_new(ctx: &Ctx) -> Result<()> {
    let q = build_queue(ctx, &load_gates(&ctx.work)?, &proposals(ctx)?)?;
    write_json(&ctx.work.join("queue.json"), &q)?;
    let creates: Vec<_> = q
        .items
        .iter()
        .filter(|i| i.kind == WorkKind::Create)
        .collect();
    println!(
        "propose-new: {} create item(s) in queue.json",
        creates.len()
    );
    for c in creates {
        println!(
            "  {:?} {} <- {}",
            c.status,
            c.candidate.as_deref().unwrap_or("-"),
            c.repo
        );
    }
    for s in q.suppressed.iter().filter(|s| s.id.starts_with("create::")) {
        println!("  suppressed {}: {}", s.id, s.reason);
    }
    Ok(())
}

pub fn ledger(ctx: &Ctx, op: LedgerOp) -> Result<()> {
    match op {
        LedgerOp::Record {
            repo,
            candidate,
            subject,
            kind,
            signal,
            outcome,
            pr,
            version,
            reason,
            measures,
            gates_from,
            blocked_on,
        } => {
            let mut e = entry(&repo, candidate, subject, kind, outcome);
            e.blocked_on = blocked_on;
            e.signal_kind = signal;
            e.pr = pr;
            e.version = version;
            e.reason = reason;
            e.measurements = measures
                .iter()
                .map(|m| parse_measurement(m))
                .collect::<Result<Vec<_>>>()?;
            if let Some(pkg) = gates_from {
                e.gates = read_json::<GateReport>(&gate_path(&ctx.work, &pkg))?.results;
            }
            let path = ledger_path(ctx)?;
            Ledger::append(&path, &e)?;
            println!(
                "ledger: recorded {outcome:?} for {} -> {}",
                e.repo,
                path.display()
            );
        }
        LedgerOp::Import { file } => {
            let path = ledger_path(ctx)?;
            let live = Ledger::load(&path)?;
            let incoming = Ledger::load(&file)?;
            let mut added = 0;
            for e in incoming
                .entries
                .iter()
                .filter(|e| !live.entries.contains(e))
            {
                Ledger::append(&path, e)?;
                added += 1;
            }
            println!(
                "ledger: imported {added} of {} entries from {} -> {}",
                incoming.entries.len(),
                file.display(),
                path.display()
            );
        }
        LedgerOp::Snapshot => {
            let live = ledger_path(ctx)?;
            let seed = ctx
                .cfg
                .ledger_seed
                .clone()
                .context("no in-repo seed (run with --config-dir config/weaver)")?;
            std::fs::copy(&live, &seed)?;
            println!("ledger: snapshot {} -> {}", live.display(), seed.display());
        }
        LedgerOp::Show { repo, json } => {
            let l = load_ledger(ctx)?;
            let rows: Vec<_> = l
                .entries
                .iter()
                .filter(|e| {
                    repo.as_ref()
                        .map(|r| e.repo.eq_ignore_ascii_case(r))
                        .unwrap_or(true)
                })
                .collect();
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
                return Ok(());
            }
            for e in &rows {
                println!(
                    "{} {:<11} {:<22} {:<36} {}",
                    &e.ts[..10.min(e.ts.len())],
                    format!("{:?}", e.outcome),
                    e.repo,
                    e.subject
                        .as_deref()
                        .or(e.candidate.as_deref())
                        .unwrap_or("-"),
                    e.reason.as_deref().unwrap_or("")
                );
            }
            println!("\nhit rates (Laplace-smoothed) -> queue weights:");
            let w = queue::signal_weights(&l);
            for (k, r) in l.hit_rates() {
                println!("  {:<18} hit {:.2}  weight {:.2}", k.as_str(), r, w[&k]);
            }
            println!(
                "busy repos: {:?}",
                l.busy_repos().keys().collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

pub fn readme(
    ctx: &Ctx,
    package: Option<String>,
    out_dir: Option<PathBuf>,
    lint: Option<PathBuf>,
) -> Result<()> {
    if let Some(f) = lint {
        let hits = readme::lint(&std::fs::read_to_string(&f)?);
        for (l, t) in &hits {
            println!("{}:{l}: unmeasured numeric claim: {t}", f.display());
        }
        if !hits.is_empty() {
            std::process::exit(2);
        }
        println!("readme lint: clean");
        return Ok(());
    }
    let pkg = package.context("give a package name or --lint <README.md>")?;
    let g: Graph = read_json(&ctx.graph_path())?;
    let gates = load_gates(&ctx.work)?;
    let r = readme::render(&ctx.cfg, &g, &load_ledger(ctx)?, gates.get(&pkg), &pkg);
    let dir = out_dir.unwrap_or_else(|| ctx.work.join("readme").join(slug_key(&pkg)));
    let ds = dir.to_string_lossy();
    if ds.ends_with("npm/packages/ruvector") || ds.contains("npm/packages/ruvector/") {
        bail!("refusing to write into npm/packages/ruvector (owned by other lanes)");
    }
    std::fs::create_dir_all(dir.join("assets"))?;
    std::fs::write(dir.join("README.md"), &r.readme)?;
    std::fs::write(dir.join("assets").join(&r.svg_file), &r.svg)?;
    let bad = readme::lint(&r.readme);
    println!(
        "readme {pkg} -> {} (lint: {})",
        dir.display(),
        if bad.is_empty() {
            "clean".to_string()
        } else {
            format!("{} hits", bad.len())
        }
    );
    Ok(())
}
