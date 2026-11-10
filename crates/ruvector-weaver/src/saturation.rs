//! `infra:gha-queue-saturation`. GitHub Actions caps concurrent jobs (and
//! macOS runners especially) per account. Before the queue hands out lanes it
//! reads each repo's queued + in-progress run counts. Repos over
//! `max_active_runs_per_repo` are deferred, and runs superseded by a newer
//! run of the same workflow on the same branch become `cancel-superseded`
//! items. Only branches this loop owns (`owned_branch_prefixes`) are ever
//! listed, so the loop never cancels another lane's or a human's CI.

use crate::model::{LaneStatus, Signal, SignalKind, WorkItem, WorkKind};
use crate::util::{clip, now, run_timeout};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRef {
    pub id: u64,
    pub workflow: String,
    pub branch: String,
    pub run_number: u64,
    pub status: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepoRuns {
    pub repo: String,
    pub fetched_at: String,
    pub queued: u64,
    pub in_progress: u64,
    /// Superseded runs on branches this loop owns (safe to cancel).
    pub superseded: Vec<RunRef>,
}

impl RepoRuns {
    pub fn active(&self) -> u64 {
        self.queued + self.in_progress
    }
}

/// Parse `actions/runs` items into run refs.
pub fn parse_runs(v: &Value) -> Vec<RunRef> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|r| RunRef {
                    id: r["id"].as_u64().unwrap_or(0),
                    workflow: r["name"].as_str().unwrap_or("?").to_string(),
                    branch: r["head_branch"].as_str().unwrap_or("").to_string(),
                    run_number: r["run_number"].as_u64().unwrap_or(0),
                    status: r["status"].as_str().unwrap_or("").to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Runs superseded by a newer run of the same workflow on the same branch,
/// restricted to owned branches. The newest run of each pair is kept.
pub fn superseded(active: &[RunRef], owned_prefixes: &[String]) -> Vec<RunRef> {
    let mut groups: BTreeMap<(String, String), Vec<&RunRef>> = BTreeMap::new();
    for r in active.iter().filter(|r| {
        owned_prefixes
            .iter()
            .any(|p| r.branch.starts_with(p.as_str()))
    }) {
        groups
            .entry((r.branch.clone(), r.workflow.clone()))
            .or_default()
            .push(r);
    }
    let mut out = Vec::new();
    for (_, mut rs) in groups {
        rs.sort_by_key(|r| std::cmp::Reverse(r.run_number));
        out.extend(rs.into_iter().skip(1).cloned());
    }
    out.sort_by_key(|r| r.id);
    out
}

fn runs_with_status(repo: &str, status: &str, cwd: &Path) -> Result<(u64, Vec<RunRef>)> {
    let path = format!("repos/{repo}/actions/runs?status={status}&per_page=100");
    let o = run_timeout("gh", &["api", &path], cwd, Duration::from_secs(60))?;
    if o.status != Some(0) {
        bail!("gh api {path}: {}", clip(o.stderr.trim(), 200));
    }
    let v: Value = serde_json::from_str(&o.stdout)?;
    Ok((
        v["total_count"].as_u64().unwrap_or(0),
        parse_runs(&v["workflow_runs"]),
    ))
}

pub fn fetch(repo: &str, owned_prefixes: &[String], cwd: &Path) -> Result<RepoRuns> {
    let (queued, mut runs) = runs_with_status(repo, "queued", cwd)?;
    let (in_progress, more) = runs_with_status(repo, "in_progress", cwd)?;
    runs.extend(more);
    Ok(RepoRuns {
        repo: repo.to_ascii_lowercase(),
        fetched_at: now(),
        queued,
        in_progress,
        superseded: superseded(&runs, owned_prefixes),
    })
}

/// Saturated repos, as `busy_repos`-style reasons.
pub fn saturated(runs: &[RepoRuns], max_active: usize) -> BTreeMap<String, Vec<String>> {
    runs.iter()
        .filter(|r| r.active() as usize > max_active)
        .map(|r| {
            (
                r.repo.clone(),
                vec![format!(
                    "gha-saturated: {} queued + {} in progress (max {max_active})",
                    r.queued, r.in_progress
                )],
            )
        })
        .collect()
}

/// A direct (agent-free) hygiene item: cancel the listed superseded runs.
pub fn cancel_item(r: &RepoRuns) -> Option<WorkItem> {
    if r.superseded.is_empty() {
        return None;
    }
    let ids: Vec<String> = r.superseded.iter().map(|x| x.id.to_string()).collect();
    let detail: Vec<String> = r
        .superseded
        .iter()
        .map(|x| format!("{} {}#{} ({})", x.id, x.workflow, x.run_number, x.branch))
        .collect();
    let cmd = format!(
        "for id in {}; do gh run cancel \"$id\" --repo {}; done",
        ids.join(" "),
        r.repo
    );
    let sig = Signal {
        kind: SignalKind::MainRed,
        repo: r.repo.clone(),
        file: format!("https://github.com/{}/actions", r.repo),
        line: 0,
        evidence: clip(
            &format!(
                "{} superseded run(s) on loop-owned branches: {}",
                r.superseded.len(),
                detail.join("; ")
            ),
            400,
        ),
        subject: Some("infra:cancel-superseded".into()),
        keyword: Some("cancel-superseded".into()),
        candidate: None,
        confidence: 1.0,
        severity: None,
        occurrences: r.superseded.len(),
    };
    Some(WorkItem {
        id: crate::queue::item_id(WorkKind::Hygiene, &r.repo, "infra:cancel-superseded"),
        kind: WorkKind::Hygiene,
        status: LaneStatus::Active,
        repo: r.repo.clone(),
        subject: Some("infra:cancel-superseded".into()),
        candidate: None,
        signal_kind: SignalKind::MainRed,
        score: 100.0,
        disclosure: "public_pr".into(),
        required_test: "Each listed run reports `cancelled`; no run outside the list was touched.".into(),
        evidence: vec![sig],
        gates: vec![],
        brief: format!(
            "[weaver direct] cancel superseded CI runs in {} (no agent needed; run from the tick).\nOnly runs on loop-owned branches are listed; never cancel anything else.\n\n{cmd}\n",
            r.repo
        ),
    })
}

pub fn load_all(work_dir: &Path) -> Result<Vec<RepoRuns>> {
    let dir = work_dir.join("runs");
    let mut out = Vec::new();
    if dir.exists() {
        for e in std::fs::read_dir(&dir)? {
            let p = e?.path();
            if p.extension().and_then(|x| x.to_str()) == Some("json") {
                out.push(serde_json::from_str(&std::fs::read_to_string(&p)?)?);
            }
        }
    }
    Ok(out)
}
