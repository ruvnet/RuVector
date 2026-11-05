//! Serde types shared by every weaver subcommand. Every artifact weaver
//! writes (graph.json, scan/*.json, gates/*.json, queue.json, ledger.jsonl)
//! is one of these types, so the LLM side of the loop reads a stable schema.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ecosystem {
    Npm,
    Crates,
}

/// One published package in the ruvnet fabric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Package {
    pub name: String,
    pub ecosystem: Ecosystem,
    pub version: String,
    /// RFC 3339 timestamp of the latest publish, when known.
    pub last_publish: Option<String>,
    pub monthly_downloads: u64,
    pub description: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub optional_dependencies: Vec<String>,
    #[serde(default)]
    pub peer_dependencies: Vec<String>,
    pub repository: Option<String>,
    /// `owner/name`, lower-cased, derived from `repository`.
    pub repo_slug: Option<String>,
    /// Platform-suffixed NAPI binary (`-linux-x64-gnu` ...): kept in the
    /// graph for edges, excluded from every ranking.
    pub platform_binary: bool,
}

impl Package {
    /// Every runtime edge (deps + optional + peer).
    pub fn all_deps(&self) -> impl Iterator<Item = &String> {
        self.dependencies
            .iter()
            .chain(&self.optional_dependencies)
            .chain(&self.peer_dependencies)
    }
}

/// A source repository that publishes one or more fabric packages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerRepo {
    pub slug: String,
    pub packages: Vec<String>,
    /// Sum of monthly downloads of its non-platform packages.
    pub monthly_downloads: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Graph {
    pub generated_at: String,
    pub packages: Vec<Package>,
    pub consumer_repos: Vec<ConsumerRepo>,
}

impl Graph {
    pub fn get(&self, name: &str) -> Option<&Package> {
        self.packages.iter().find(|p| p.name == name)
    }

    pub fn repo(&self, slug: &str) -> Option<&ConsumerRepo> {
        let slug = slug.to_ascii_lowercase();
        self.consumer_repos.iter().find(|r| r.slug == slug)
    }

    /// Packages that depend on `name` (reverse edges).
    pub fn dependents(&self, name: &str) -> Vec<&Package> {
        self.packages
            .iter()
            .filter(|p| p.all_deps().any(|d| d == name))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalKind {
    /// Import of an ecosystem-looking package name that 404s on npm.
    MissingPackage,
    /// Import of a real ecosystem package not declared in the nearest package.json.
    UndeclaredImport,
    /// Ecosystem dependency declared but never imported.
    UnusedDeclared,
    /// Stub / TODO / simulated / fallback marker near a domain keyword.
    StubMarker,
    /// npm audit / cargo deny finding.
    Security,
}

impl SignalKind {
    pub const ALL: [SignalKind; 5] = [
        SignalKind::MissingPackage,
        SignalKind::UndeclaredImport,
        SignalKind::UnusedDeclared,
        SignalKind::StubMarker,
        SignalKind::Security,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            SignalKind::MissingPackage => "missing_package",
            SignalKind::UndeclaredImport => "undeclared_import",
            SignalKind::UnusedDeclared => "unused_declared",
            SignalKind::StubMarker => "stub_marker",
            SignalKind::Security => "security",
        }
    }
}

/// One observation in a consumer repo, always with file:line evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Signal {
    pub kind: SignalKind,
    pub repo: String,
    /// Path relative to the repo root.
    pub file: String,
    pub line: usize,
    /// The offending source line (trimmed, truncated).
    pub evidence: String,
    /// The package name observed in the consumer (imported / declared / vulnerable).
    pub subject: Option<String>,
    /// Domain keyword that matched (stub markers, capability lookups).
    pub keyword: Option<String>,
    /// Target package the capability table maps this signal to.
    pub candidate: Option<String>,
    pub confidence: f64,
    /// Security severity (`critical`, `high`, ...).
    pub severity: Option<String>,
    /// Extra occurrences folded into this signal.
    #[serde(default)]
    pub occurrences: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScanReport {
    pub repo: String,
    pub scanned_at: String,
    pub commit: Option<String>,
    pub files_scanned: usize,
    pub signals: Vec<Signal>,
    /// Tool failures / timeouts recorded instead of aborting.
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GateResult {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateReport {
    pub package: String,
    pub version: Option<String>,
    pub checked_at: String,
    pub passed: bool,
    pub results: Vec<GateResult>,
    /// Export names observed by the smoke import (the "verified API").
    #[serde(default)]
    pub exports: Vec<String>,
}

impl GateReport {
    pub fn failures(&self) -> Vec<&GateResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkKind {
    /// Wire an existing package into a consumer.
    Integrate,
    /// Replace a stub/fallback with the real package.
    Replace,
    /// Declare or drop a dependency.
    Hygiene,
    /// Security remediation.
    Security,
    /// Create + publish a new package that fills a gap (autopilot, gated).
    Create,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneStatus {
    /// Highest-ranked item for its repo; dispatch it this tick.
    Active,
    /// Independent item shipped in the same lane / PR as its repo's active item.
    Batched,
    /// Repo lane already taken this tick (or by an in-flight ledger entry).
    Deferred,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkItem {
    pub id: String,
    pub kind: WorkKind,
    pub status: LaneStatus,
    pub repo: String,
    pub subject: Option<String>,
    pub candidate: Option<String>,
    pub signal_kind: SignalKind,
    pub score: f64,
    pub evidence: Vec<Signal>,
    pub gates: Vec<GateResult>,
    /// `public_pr` or `private_advisory` (critical/high security findings).
    pub disclosure: String,
    pub required_test: String,
    pub brief: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Suppressed {
    pub id: String,
    pub repo: String,
    pub candidate: Option<String>,
    pub subject: Option<String>,
    pub reason: String,
}

/// One dispatchable lane: a repo plus the items its agent ships together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lane {
    pub repo: String,
    pub items: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Queue {
    pub generated_at: String,
    pub max_lanes: usize,
    pub lanes: Vec<Lane>,
    /// Repos whose lane is held by an in-flight ledger entry.
    pub busy_repos: BTreeMap<String, Vec<String>>,
    pub signal_weights: BTreeMap<String, f64>,
    pub items: Vec<WorkItem>,
    pub suppressed: Vec<Suppressed>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Shipped,
    Rejected,
    Failed,
    InProgress,
    /// Known backlog item, not yet dispatched: surfaces in the queue.
    Queued,
    /// Informational gate record; never changes lane state.
    Gated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Measurement {
    pub metric: String,
    pub value: f64,
    pub unit: String,
    #[serde(default)]
    pub baseline: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LedgerEntry {
    pub ts: String,
    pub repo: String,
    pub candidate: Option<String>,
    #[serde(default)]
    pub subject: Option<String>,
    pub kind: WorkKind,
    #[serde(default)]
    pub signal_kind: Option<SignalKind>,
    pub outcome: Outcome,
    #[serde(default)]
    pub pr: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub measurements: Vec<Measurement>,
    #[serde(default)]
    pub gates: Vec<GateResult>,
    /// `path:line` (repo-relative) for audit-sourced entries.
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default)]
    pub severity: Option<String>,
}
