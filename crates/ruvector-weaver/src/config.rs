//! Loop policy (`weaver.toml`) and the capability table (`capabilities.toml`).
//! Both are read from `--config-dir` when present and otherwise fall back to
//! the copies compiled into the binary, so tests and CI never depend on cwd.

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const EMBEDDED_POLICY: &str = include_str!("../../../config/weaver/weaver.toml");
pub const EMBEDDED_CAPABILITIES: &str = include_str!("../../../config/weaver/capabilities.toml");
pub const EMBEDDED_LEDGER: &str = include_str!("../../../config/weaver/ledger.jsonl");

#[derive(Debug, Clone, Deserialize)]
pub struct Policy {
    #[serde(rename = "loop")]
    pub lp: LoopPolicy,
    pub npm: NpmPolicy,
    pub crates: CratesPolicy,
    pub gate: GatePolicy,
    pub scan: ScanPolicy,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoopPolicy {
    pub max_lanes: usize,
    #[serde(default = "default_batch")]
    pub max_batch_per_lane: usize,
    /// `None` = unlimited (new names are bounded by the quality gates only).
    #[serde(default)]
    pub max_new_packages_per_tick: Option<usize>,
    #[serde(default)]
    pub max_new_packages_per_day: Option<usize>,
    pub max_failures: usize,
    /// Defer a repo's lane while it has more queued + in-progress Actions
    /// runs than this (account-wide runner caps, macOS especially).
    #[serde(default = "default_max_active_runs")]
    pub max_active_runs_per_repo: usize,
    /// Branch prefixes this loop owns; only their runs are ever cancelled.
    #[serde(default = "default_owned_prefixes")]
    pub owned_branch_prefixes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NpmPolicy {
    pub maintainer: String,
    pub ecosystem_scopes: Vec<String>,
    pub ecosystem_prefixes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CratesPolicy {
    pub user_id: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GatePolicy {
    pub max_age_days: i64,
    pub leak_patterns: Vec<String>,
    pub smoke_timeout_secs: u64,
    /// Per-package regexes for paths a package ships ON PURPOSE (e.g.
    /// agentic-flow's `.claude/`); never reported as leaks.
    #[serde(default)]
    pub leak_allow: std::collections::BTreeMap<String, Vec<String>>,
    /// "Lean release" bar that re-opens a candidate rejected for a heavy
    /// dependency tree once a newer version clears it.
    #[serde(default = "default_lean_deps")]
    pub lean_max_runtime_deps: usize,
    #[serde(default = "default_lean_bytes")]
    pub lean_max_unpacked_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScanPolicy {
    pub max_file_bytes: u64,
    pub skip_dirs: Vec<String>,
    pub security_timeout_secs: u64,
}

impl NpmPolicy {
    /// Does an import specifier's package name belong to the fabric?
    pub fn is_ecosystem(&self, name: &str) -> bool {
        if let Some((scope, _)) = name.split_once('/') {
            if name.starts_with('@') {
                return self.ecosystem_scopes.iter().any(|s| s == scope);
            }
        }
        self.ecosystem_prefixes
            .iter()
            .any(|p| name.starts_with(p.as_str()))
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Capability {
    pub package: String,
    pub category: String,
    pub summary: String,
    pub keywords: Vec<String>,
    #[serde(default = "one")]
    pub weight: f64,
}

fn one() -> f64 {
    1.0
}

fn default_batch() -> usize {
    6
}

#[derive(Debug, Clone, Deserialize)]
struct CapabilityFile {
    capability: Vec<Capability>,
}

/// Compiled capability table.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub entries: Vec<Capability>,
    matchers: Vec<(usize, String, Regex)>,
}

/// A keyword hit: which capability, which keyword.
#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityHit<'a> {
    pub capability: &'a Capability,
    pub keyword: String,
}

impl Capabilities {
    pub fn parse(text: &str) -> Result<Self> {
        let file: CapabilityFile = toml::from_str(text).context("parsing capabilities.toml")?;
        let mut matchers = Vec::new();
        for (i, cap) in file.capability.iter().enumerate() {
            for kw in &cap.keywords {
                let re = Regex::new(&format!(r"(?i)\b{kw}"))
                    .with_context(|| format!("bad keyword regex {kw:?} for {}", cap.package))?;
                matchers.push((i, kw.clone(), re));
            }
        }
        Ok(Self {
            entries: file.capability,
            matchers,
        })
    }

    /// First (table-order) capability whose keyword occurs in `text`,
    /// preferring the highest weight when several match.
    pub fn best_match(&self, text: &str) -> Option<CapabilityHit<'_>> {
        let mut best: Option<CapabilityHit<'_>> = None;
        for (i, kw, re) in &self.matchers {
            if re.is_match(text) {
                let cap = &self.entries[*i];
                let better = best
                    .as_ref()
                    .map(|b| cap.weight > b.capability.weight)
                    .unwrap_or(true);
                if better {
                    best = Some(CapabilityHit {
                        capability: cap,
                        keyword: kw.clone(),
                    });
                }
            }
        }
        best
    }

    /// Capability for a package name's tokens (`@ruvector/mincut` -> mincut).
    pub fn match_package_name(&self, name: &str) -> Option<CapabilityHit<'_>> {
        let bare = name.rsplit('/').next().unwrap_or(name);
        let tokens = bare
            .split(['-', '_', '.'])
            .filter(|t| !matches!(*t, "wasm" | "node" | "napi" | "ruvector" | "js" | "core"))
            .collect::<Vec<_>>()
            .join(" ");
        if tokens.is_empty() {
            return None;
        }
        self.best_match(&tokens)
    }

    pub fn get(&self, package: &str) -> Option<&Capability> {
        self.entries.iter().find(|c| c.package == package)
    }

    /// Every keyword regex, used by the stub-marker scanner.
    pub fn keyword_regexes(&self) -> impl Iterator<Item = (&Capability, &str, &Regex)> {
        self.matchers
            .iter()
            .map(|(i, kw, re)| (&self.entries[*i], kw.as_str(), re))
    }
}

/// Resolved configuration for one weaver invocation.
#[derive(Debug, Clone)]
pub struct Config {
    pub policy: Policy,
    pub capabilities: Capabilities,
    /// The LIVE ledger, outside the repo (`$WEAVER_LEDGER`, default
    /// `<work_dir>/ledger.jsonl`), so recording outcomes never needs a PR.
    /// `None` = embedded snapshot, read-only (tests).
    pub ledger_path: Option<PathBuf>,
    /// In-repo snapshot (`config/weaver/ledger.jsonl`) that seeds the live
    /// ledger the first time it is used on a machine.
    pub ledger_seed: Option<PathBuf>,
    /// Cache / work root (`graph.json`, `scan/`, `gates/`, `repos/`, `queue.json`).
    pub work_dir: PathBuf,
}

impl Config {
    pub fn load(config_dir: Option<&Path>, work_dir: PathBuf) -> Result<Self> {
        let dir = config_dir.map(Path::to_path_buf).or_else(|| {
            let p = PathBuf::from("config/weaver");
            p.join("weaver.toml").exists().then_some(p)
        });
        let read = |name: &str, embedded: &str| -> Result<String> {
            match &dir {
                Some(d) if d.join(name).exists() => std::fs::read_to_string(d.join(name))
                    .with_context(|| format!("reading {}", d.join(name).display())),
                _ => Ok(embedded.to_string()),
            }
        };
        let policy: Policy = toml::from_str(&read("weaver.toml", EMBEDDED_POLICY)?)
            .context("parsing weaver.toml")?;
        let capabilities = Capabilities::parse(&read("capabilities.toml", EMBEDDED_CAPABILITIES)?)?;
        let ledger_seed = dir.map(|d| d.join("ledger.jsonl")).filter(|p| p.exists());
        let ledger_path = Some(
            std::env::var_os("WEAVER_LEDGER")
                .map(PathBuf::from)
                .unwrap_or_else(|| work_dir.join("ledger.jsonl")),
        );
        Ok(Self {
            policy,
            capabilities,
            ledger_path,
            ledger_seed,
            work_dir,
        })
    }

    /// Embedded defaults only (tests).
    pub fn embedded(work_dir: PathBuf) -> Result<Self> {
        Ok(Self {
            policy: toml::from_str(EMBEDDED_POLICY)?,
            capabilities: Capabilities::parse(EMBEDDED_CAPABILITIES)?,
            ledger_path: None,
            ledger_seed: None,
            work_dir,
        })
    }

    /// Live ledger path, seeding it from the in-repo snapshot (or the
    /// embedded copy) when it does not exist yet.
    pub fn live_ledger(&self) -> Result<Option<PathBuf>> {
        let Some(path) = &self.ledger_path else {
            return Ok(None);
        };
        if !path.exists() {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let seed = match &self.ledger_seed {
                Some(s) => std::fs::read_to_string(s)
                    .with_context(|| format!("reading {}", s.display()))?,
                None => EMBEDDED_LEDGER.to_string(),
            };
            std::fs::write(path, seed).with_context(|| format!("seeding {}", path.display()))?;
        }
        Ok(Some(path.clone()))
    }
}

fn default_lean_deps() -> usize {
    5
}

fn default_lean_bytes() -> u64 {
    5_000_000
}

fn default_max_active_runs() -> usize {
    20
}

fn default_owned_prefixes() -> Vec<String> {
    vec!["weaver/".into(), "feat/weaver-".into()]
}
