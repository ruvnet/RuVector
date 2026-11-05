//! Argument parsing and subcommand dispatch for the `weaver` binary.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use ruvector_weaver::config::{Config, EMBEDDED_LEDGER};
use ruvector_weaver::gate::{self, GateOptions};
use ruvector_weaver::http::Http;
use ruvector_weaver::ledger::{parse_measurement, Ledger};
use ruvector_weaver::model::{GateReport, Graph, LedgerEntry, Outcome, Queue, SignalKind, WorkKind};
use ruvector_weaver::registry::NpmRegistry;
use ruvector_weaver::util::{now, slug_key};
use ruvector_weaver::{graph, propose, queue, readme, scan};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Parser)]
#[command(name = "weaver", version, about = "ruvnet ecosystem fabric harness (ADR-355)")]
struct Cli {
    /// Work/cache root: graph.json, scan/, gates/, repos/, queue.json.
    #[arg(long, env = "WEAVER_WORK_DIR", default_value = "/data/scratch/weaver", global = true)]
    work_dir: PathBuf,
    /// Directory holding weaver.toml, capabilities.toml, ledger.jsonl.
    #[arg(long, env = "WEAVER_CONFIG_DIR", global = true)]
    config_dir: Option<PathBuf>,
    /// HTTP cache TTL in hours (0 disables reuse).
    #[arg(long, default_value_t = 12, global = true)]
    cache_ttl_hours: u64,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Crawl npm (maintainer) + crates.io (owner) into graph.json.
    Graph {
        #[arg(long)]
        fresh_downloads: bool,
        #[arg(long)]
        no_crates: bool,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Fetch repos and emit signals into scan/<repo>.json.
    Scan {
        repos: Vec<String>,
        /// Scan every consumer repo in graph.json.
        #[arg(long)]
        all: bool,
        #[arg(long)]
        no_security: bool,
    },
    /// Candidate health gate; writes gates/<pkg>.json.
    Gate {
        package: String,
        #[arg(long)]
        consumer: Option<String>,
        #[arg(long)]
        no_smoke: bool,
        /// New-package mode: source gates on --dir before publish.
        #[arg(long)]
        new: bool,
        #[arg(long)]
        dir: Option<PathBuf>,
        #[arg(long)]
        run_tests: bool,
        /// Append the result to the ledger as a `gated` entry.
        #[arg(long)]
        record: bool,
    },
    /// Rank work into queue.json (one lane per repo, batched).
    Queue {
        /// Gate the top N ungated candidates before ranking.
        #[arg(long, default_value_t = 0)]
        gate_top: usize,
        #[arg(long)]
        no_create: bool,
        /// Print the top N items.
        #[arg(long, default_value_t = 25)]
        show: usize,
    },
    /// Append to / inspect the outcome ledger.
    Ledger {
        #[command(subcommand)]
        op: LedgerOp,
    },
    /// Render an SEO README + animated SVG banner, or lint a README.
    Readme {
        package: Option<String>,
        #[arg(long)]
        out_dir: Option<PathBuf>,
        #[arg(long)]
        lint: Option<PathBuf>,
    },
    /// Capability gaps -> `create` items merged into queue.json.
    ProposeNew,
}

#[derive(Subcommand)]
enum LedgerOp {
    Record {
        #[arg(long)]
        repo: String,
        #[arg(long)]
        candidate: Option<String>,
        #[arg(long)]
        subject: Option<String>,
        #[arg(long, value_parser = parse_kind)]
        kind: WorkKind,
        #[arg(long, value_parser = parse_signal)]
        signal: Option<SignalKind>,
        #[arg(long, value_parser = parse_outcome)]
        outcome: Outcome,
        #[arg(long)]
        pr: Option<String>,
        #[arg(long)]
        version: Option<String>,
        #[arg(long)]
        reason: Option<String>,
        /// metric=value:unit[@baseline], repeatable.
        #[arg(long = "measure")]
        measures: Vec<String>,
        /// Attach gate results from gates/<pkg>.json.
        #[arg(long)]
        gates_from: Option<String>,
    },
    Show {
        #[arg(long)]
        repo: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

fn de<T: serde::de::DeserializeOwned>(s: &str) -> Result<T, String> {
    serde_json::from_value(serde_json::Value::String(s.to_string())).map_err(|e| e.to_string())
}
fn parse_kind(s: &str) -> Result<WorkKind, String> {
    de(s)
}
fn parse_signal(s: &str) -> Result<SignalKind, String> {
    de(s)
}
fn parse_outcome(s: &str) -> Result<Outcome, String> {
    de(s)
}

fn read_json<T: serde::de::DeserializeOwned + Default>(p: &Path) -> Result<T> {
    if !p.exists() {
        return Ok(T::default());
    }
    serde_json::from_str(&std::fs::read_to_string(p)?).with_context(|| format!("parsing {}", p.display()))
}

fn write_json<T: serde::Serialize>(p: &Path, v: &T) -> Result<()> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(p, serde_json::to_string_pretty(v)?)?;
    Ok(())
}

fn load_ledger(cfg: &Config) -> Result<Ledger> {
    match &cfg.ledger_path {
        Some(p) => Ledger::load(p),
        None => Ledger::parse(EMBEDDED_LEDGER),
    }
}

fn load_gates(work: &Path) -> Result<BTreeMap<String, GateReport>> {
    let mut out = BTreeMap::new();
    let dir = work.join("gates");
    if dir.exists() {
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            if p.extension().and_then(|x| x.to_str()) == Some("json") {
                let g: GateReport = read_json(&p)?;
                out.insert(g.package.clone(), g);
            }
        }
    }
    Ok(out)
}

fn gate_path(work: &Path, pkg: &str) -> PathBuf {
    work.join("gates").join(format!("{}.json", slug_key(pkg)))
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load(cli.config_dir.as_deref(), cli.work_dir.clone())?;
    let work = cli.work_dir.clone();
    let http = Http::new(Some(work.join("http-cache")), Duration::from_secs(cli.cache_ttl_hours * 3600));
    let graph_path = work.join("graph.json");
    match cli.cmd {
        Cmd::Graph { fresh_downloads, no_crates, workers } => {
            let opts = graph::GraphOptions { fresh_downloads, include_crates: !no_crates, workers };
            let g = graph::build(&http, &cfg, &opts)?;
            write_json(&graph_path, &g)?;
            let npm = g.packages.iter().filter(|p| p.ecosystem == ruvector_weaver::model::Ecosystem::Npm).count();
            let plat = g.packages.iter().filter(|p| p.platform_binary).count();
            println!("graph: {} packages ({npm} npm, {} crates, {plat} platform binaries skipped in rankings), {} consumer repos -> {}", g.packages.len(), g.packages.len() - npm, g.consumer_repos.len(), graph_path.display());
            for r in g.consumer_repos.iter().take(15) {
                println!("  {:>10}/mo  {:<40} {} pkgs", r.monthly_downloads, r.slug, r.packages.len());
            }
        }
        Cmd::Scan { mut repos, all, no_security } => {
            let g: Graph = read_json(&graph_path)?;
            if all {
                repos.extend(g.consumer_repos.iter().map(|r| r.slug.clone()));
            }
            if repos.is_empty() {
                bail!("give repos (owner/name, URL or path) or --all");
            }
            let reg = NpmRegistry::new(http.clone());
            for r in repos {
                let (root, slug, commit) = match scan::fetch_repo(&work, &r) {
                    Ok(x) => x,
                    Err(e) => {
                        eprintln!("scan {r}: {e:#}");
                        continue;
                    }
                };
                let mut rep = scan::scan_dir(&root, &slug, &cfg, &reg, &g, !no_security)?;
                rep.commit = commit;
                let out = work.join("scan").join(format!("{}.json", slug_key(&slug)));
                write_json(&out, &rep)?;
                let mut by: BTreeMap<&str, usize> = BTreeMap::new();
                for s in &rep.signals {
                    *by.entry(s.kind.as_str()).or_default() += 1;
                }
                println!("scan {slug}: {} files, {} signals {by:?} -> {}", rep.files_scanned, rep.signals.len(), out.display());
                for n in &rep.notes {
                    println!("  note: {n}");
                }
            }
        }
        Cmd::Gate { package, consumer, no_smoke, new, dir, run_tests, record } => {
            let g: Graph = read_json(&graph_path)?;
            let reg = NpmRegistry::new(http.clone());
            let ledger = load_ledger(&cfg)?;
            let rep = if new {
                let dir = dir.context("--new needs --dir <package dir>")?;
                let mut results = gate::new_package_source_gates(&dir, &cfg)?;
                if run_tests {
                    results.push(gate::run_tests(&dir, Duration::from_secs(900))?);
                }
                results.extend(gate::pack_gates(&dir, &cfg)?);
                if let Some(cap) = cfg.policy.lp.max_new_packages_per_day {
                    let used = ledger.new_packages_since(chrono::Utc::now() - chrono::Duration::hours(24));
                    results.push(ruvector_weaver::model::GateResult { name: "rate_limit".into(), passed: used < cap, detail: format!("{used}/{cap} new names in 24h") });
                }
                gate::finish(&package, None, results, vec![])
            } else {
                gate::gate(&cfg, &http, &reg, &g, &package, &GateOptions { consumer: consumer.as_deref(), smoke: !no_smoke })?
            };
            write_json(&gate_path(&work, &package), &rep)?;
            for r in &rep.results {
                println!("  [{}] {:<20} {}", if r.passed { "pass" } else { "FAIL" }, r.name, r.detail);
            }
            println!("gate {package}: {}", if rep.passed { "PASSED" } else { "REJECTED" });
            if record {
                let path = cfg.ledger_path.clone().context("--record needs a writable --config-dir")?;
                Ledger::append(&path, &LedgerEntry {
                    ts: now(),
                    repo: consumer.unwrap_or_else(|| "-".into()),
                    candidate: Some(package.clone()),
                    subject: None,
                    kind: if new { WorkKind::Create } else { WorkKind::Integrate },
                    signal_kind: None,
                    outcome: Outcome::Gated,
                    pr: None,
                    version: rep.version.clone(),
                    reason: Some(if rep.passed { "all gates passed".into() } else { "gate rejected".into() }),
                    measurements: vec![],
                    gates: rep.results.clone(),
                    location: None,
                    severity: None,
                })?;
            }
            if !rep.passed {
                std::process::exit(2);
            }
        }
        Cmd::Queue { gate_top, no_create, show } => {
            let g: Graph = read_json(&graph_path)?;
            let reports = scan::load_reports(&work)?;
            let ledger = load_ledger(&cfg)?;
            let reg = NpmRegistry::new(http.clone());
            let mut gates = load_gates(&work)?;
            let props = if no_create { propose::Proposals { items: vec![], suppressed: vec![] } } else { propose::propose(&cfg, &g, &reports, &ledger, &reg) };
            let build = |gates: &BTreeMap<String, GateReport>, p: &propose::Proposals| {
                queue::build(&cfg, queue::QueueInputs { graph: &g, reports: &reports, ledger: &ledger, gates, creates: p.items.clone(), create_suppressed: p.suppressed.clone() })
            };
            let mut q = build(&gates, &props);
            if gate_top > 0 {
                let todo: Vec<String> = q.items.iter().filter_map(|i| i.candidate.clone()).filter(|c| !gates.contains_key(c) && g.get(c).is_some()).fold(vec![], |mut v, c| { if !v.contains(&c) { v.push(c) } v }).into_iter().take(gate_top).collect();
                for c in todo {
                    let rep = gate::gate(&cfg, &http, &reg, &g, &c, &GateOptions { consumer: None, smoke: true })?;
                    println!("gate {c}: {}", if rep.passed { "pass" } else { "REJECTED" });
                    write_json(&gate_path(&work, &c), &rep)?;
                    gates.insert(c, rep);
                }
                q = build(&gates, &props);
            }
            write_json(&work.join("queue.json"), &q)?;
            print_queue(&q, show);
        }
        Cmd::ProposeNew => {
            let g: Graph = read_json(&graph_path)?;
            let reports = scan::load_reports(&work)?;
            let ledger = load_ledger(&cfg)?;
            let reg = NpmRegistry::new(http.clone());
            let p = propose::propose(&cfg, &g, &reports, &ledger, &reg);
            let q = queue::build(&cfg, queue::QueueInputs { graph: &g, reports: &reports, ledger: &ledger, gates: &load_gates(&work)?, creates: p.items, create_suppressed: p.suppressed });
            write_json(&work.join("queue.json"), &q)?;
            let creates: Vec<_> = q.items.iter().filter(|i| i.kind == WorkKind::Create).collect();
            println!("propose-new: {} create item(s) in queue.json", creates.len());
            for c in creates {
                println!("  {:?} {} <- {}", c.status, c.candidate.as_deref().unwrap_or("-"), c.repo);
            }
            for s in q.suppressed.iter().filter(|s| s.id.starts_with("create::")) {
                println!("  suppressed {}: {}", s.id, s.reason);
            }
        }
        Cmd::Ledger { op } => ledger_cmd(&cfg, op)?,
        Cmd::Readme { package, out_dir, lint } => {
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
            let g: Graph = read_json(&graph_path)?;
            let gates = load_gates(&work)?;
            let r = readme::render(&cfg, &g, &load_ledger(&cfg)?, gates.get(&pkg), &pkg);
            let dir = out_dir.unwrap_or_else(|| work.join("readme").join(slug_key(&pkg)));
            let ds = dir.to_string_lossy();
            if ds.ends_with("npm/packages/ruvector") || ds.contains("npm/packages/ruvector/") {
                bail!("refusing to write into npm/packages/ruvector (owned by other lanes)");
            }
            std::fs::create_dir_all(dir.join("assets"))?;
            std::fs::write(dir.join("README.md"), &r.readme)?;
            std::fs::write(dir.join("assets").join(&r.svg_file), &r.svg)?;
            let bad = readme::lint(&r.readme);
            println!("readme {pkg} -> {} (lint: {})", dir.display(), if bad.is_empty() { "clean".to_string() } else { format!("{} hits", bad.len()) });
        }
    }
    Ok(())
}

fn ledger_cmd(cfg: &Config, op: LedgerOp) -> Result<()> {
    match op {
        LedgerOp::Record { repo, candidate, subject, kind, signal, outcome, pr, version, reason, measures, gates_from } => {
            let path = cfg.ledger_path.clone().context("ledger record needs --config-dir (or run from the repo root)")?;
            let measurements = measures.iter().map(|m| parse_measurement(m)).collect::<Result<Vec<_>>>()?;
            let gates = match gates_from {
                Some(pkg) => read_json::<GateReport>(&gate_path(&cfg.work_dir, &pkg))?.results,
                None => vec![],
            };
            let e = LedgerEntry { ts: now(), repo: repo.to_ascii_lowercase(), candidate, subject, kind, signal_kind: signal, outcome, pr, version, reason, measurements, gates, location: None, severity: None };
            Ledger::append(&path, &e)?;
            println!("ledger: recorded {outcome:?} for {} -> {}", e.repo, path.display());
        }
        LedgerOp::Show { repo, json } => {
            let l = load_ledger(cfg)?;
            let rows: Vec<_> = l.entries.iter().filter(|e| repo.as_ref().map(|r| e.repo.eq_ignore_ascii_case(r)).unwrap_or(true)).collect();
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
                return Ok(());
            }
            for e in &rows {
                println!("{} {:<12} {:<22} {:<34} {}", &e.ts[..10.min(e.ts.len())], format!("{:?}", e.outcome), e.repo, e.candidate.as_deref().or(e.subject.as_deref()).unwrap_or("-"), e.reason.as_deref().unwrap_or(""));
            }
            println!("\nhit rates (Laplace-smoothed) -> queue weights:");
            let w = queue::signal_weights(&l);
            for (k, r) in l.hit_rates() {
                println!("  {:<18} hit {:.2}  weight {:.2}", k.as_str(), r, w[&k]);
            }
            println!("busy repos: {:?}", l.busy_repos().keys().collect::<Vec<_>>());
        }
    }
    Ok(())
}

fn print_queue(q: &Queue, show: usize) {
    println!("queue: {} items, {} lanes (max {}), {} suppressed, {} busy repos", q.items.len(), q.lanes.len(), q.max_lanes, q.suppressed.len(), q.busy_repos.len());
    for (i, it) in q.items.iter().take(show).enumerate() {
        let ev = it.evidence.first().map(|s| format!("{}:{} `{}`", s.file, s.line, ruvector_weaver::util::clip(&s.evidence, 70))).unwrap_or_default();
        println!("{:>3}. [{:<8}] {:>6.2} {:<9} {:<26} {:<32} {}", i + 1, format!("{:?}", it.status), it.score, format!("{:?}", it.kind), it.repo, it.candidate.as_deref().or(it.subject.as_deref()).unwrap_or("-"), ev);
    }
}
