//! Argument parsing and subcommand dispatch for the `weaver` binary.

mod cmds;
mod io;
mod sources;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ruvector_weaver::config::Config;
use ruvector_weaver::http::Http;
use ruvector_weaver::model::{Outcome, SignalKind, WorkKind};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "weaver",
    version,
    about = "ruvnet ecosystem fabric harness (ADR-355)"
)]
struct Cli {
    /// Work/cache root: graph.json, scan/, prs/, gates/, repos/, queue.json.
    #[arg(
        long,
        env = "WEAVER_WORK_DIR",
        default_value = "/data/scratch/weaver",
        global = true
    )]
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
pub enum Cmd {
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
    /// Open PRs per repo into prs/<repo>.json (land / close-with-ledger / dedupe).
    Prs {
        repos: Vec<String>,
        #[arg(long)]
        all: bool,
        /// Also fetch each PR's check rollup (one `gh pr view` per PR).
        #[arg(long)]
        checks: bool,
        /// Append land / close-with-ledger PRs to the ledger as `queued`.
        #[arg(long)]
        seed_ledger: bool,
    },
    /// Fresh-install top packages, `npm ls --all`, emit invalid_resolution
    /// items naming the consumer whose range won hoisting.
    Resolve {
        packages: Vec<String>,
        /// Without explicit packages: the top N npm packages by downloads.
        #[arg(long, default_value_t = 15)]
        top: usize,
    },
    /// Release-regression sweep over every npm package in graph.json.
    Release {
        packages: Vec<String>,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Candidate health gate; writes gates/<pkg>.json. Exit 2 on rejection.
    Gate {
        package: String,
        #[arg(long)]
        consumer: Option<String>,
        /// The exact consumer package taking the dependency (cycle check).
        #[arg(long)]
        consumer_package: Option<String>,
        #[arg(long)]
        no_smoke: bool,
        /// New-package mode: source gates on --dir before the first publish.
        #[arg(long)]
        new: bool,
        /// Local package dir: pre-publish release checks against the registry.
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
    /// Resolve what a consumer actually installs (`npm ls --all`) before
    /// filing a vulnerability report; exit 2 when no affected copy resolves.
    VulnEvidence {
        package: String,
        /// Consumer project dir (with node_modules / lockfile).
        #[arg(long)]
        dir: PathBuf,
        /// Affected semver range, e.g. "<0.1.9".
        #[arg(long)]
        affected: String,
    },
    /// Capability gaps -> `create` items merged into queue.json.
    ProposeNew,
}

#[derive(Subcommand)]
pub enum LedgerOp {
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
        /// Subject that must be shipped before this rejection re-opens (repeatable).
        #[arg(long = "blocked-on")]
        blocked_on: Vec<String>,
    },
    Show {
        #[arg(long)]
        repo: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Append entries from another ledger file (e.g. a pending file),
    /// skipping ones already present.
    Import { file: PathBuf },
    /// Write the live ledger over the in-repo seed snapshot (for an
    /// occasional snapshot PR; the live ledger itself never needs one).
    Snapshot,
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

/// Shared per-invocation context for the handlers.
pub struct Ctx {
    pub cfg: Config,
    pub work: PathBuf,
    pub http: Http,
}

impl Ctx {
    pub fn graph_path(&self) -> PathBuf {
        self.work.join("graph.json")
    }
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load(cli.config_dir.as_deref(), cli.work_dir.clone())?;
    let http = Http::new(
        Some(cli.work_dir.join("http-cache")),
        Duration::from_secs(cli.cache_ttl_hours * 3600),
    );
    let ctx = Ctx {
        cfg,
        work: cli.work_dir,
        http,
    };
    match cli.cmd {
        Cmd::Graph {
            fresh_downloads,
            no_crates,
            workers,
        } => cmds::graph(&ctx, fresh_downloads, no_crates, workers),
        Cmd::Scan {
            repos,
            all,
            no_security,
        } => cmds::scan(&ctx, repos, all, no_security),
        Cmd::Prs {
            repos,
            all,
            checks,
            seed_ledger,
        } => sources::prs(&ctx, repos, all, checks, seed_ledger),
        Cmd::Release { packages, workers } => sources::release(&ctx, packages, workers),
        Cmd::Resolve { packages, top } => sources::resolve(&ctx, packages, top),
        Cmd::Gate {
            package,
            consumer,
            consumer_package,
            no_smoke,
            new,
            dir,
            run_tests,
            record,
        } => cmds::gate(
            &ctx,
            &package,
            (consumer, consumer_package),
            no_smoke,
            new,
            dir,
            run_tests,
            record,
        ),
        Cmd::Queue {
            gate_top,
            no_create,
            show,
        } => cmds::queue(&ctx, gate_top, no_create, show),
        Cmd::ProposeNew => cmds::propose_new(&ctx),
        Cmd::VulnEvidence {
            package,
            dir,
            affected,
        } => sources::vuln_evidence(&package, &dir, &affected),
        Cmd::Ledger { op } => cmds::ledger(&ctx, op),
        Cmd::Readme {
            package,
            out_dir,
            lint,
        } => cmds::readme(&ctx, package, out_dir, lint),
    }
}
