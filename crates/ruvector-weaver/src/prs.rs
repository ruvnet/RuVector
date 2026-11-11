//! `weaver prs <repo>`: open pull requests as a signal source. Automated
//! routines (daily security hardening, research nightlies, fix batches)
//! leave many PRs open; the loop should land the fix/security/deps ones,
//! record negative research results in the ledger and close those PRs, and
//! never dispatch a lane that re-implements an open PR.

use crate::model::{LaneStatus, Signal, SignalKind, WorkItem, WorkKind};
use crate::propose::name_tokens;
use crate::util::{clip, now, par_map, run_timeout};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrClass {
    /// fix / security / perf / deps / test / build PR: land it.
    Land,
    /// research PR with a REJECT / negative verdict: ledger + close.
    CloseWithLedger,
    /// feature / docs / positive research: not auto-landed, used for dedupe.
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenPr {
    pub number: u64,
    pub title: String,
    pub url: String,
    pub head: String,
    pub updated_at: String,
    pub draft: bool,
    pub labels: Vec<String>,
    /// `passing`, `failing`, `pending`, `none` or `unknown` (not fetched).
    pub checks: String,
    pub stale_days: i64,
    pub class: PrClass,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepoPrs {
    pub repo: String,
    pub fetched_at: String,
    pub prs: Vec<OpenPr>,
    /// Workflows failing on the default branch's head commit.
    #[serde(default)]
    pub main_red: Vec<String>,
    #[serde(default)]
    pub main_sha: Option<String>,
}

fn release_title_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"^[@\w./-]+ v?\d+\.\d+\.\d+\S*:").expect("release title re")
    })
}

pub fn classify(title: &str, head: &str, labels: &[String]) -> PrClass {
    let t = title.to_ascii_lowercase();
    let research = t.starts_with("research") || t.contains("(nightly") || t.starts_with("bench(");
    let negative = [
        "reject",
        "negative result",
        "falls short",
        "failed fresh",
        "no gain",
        "does not beat",
    ]
    .iter()
    .any(|w| t.contains(w));
    if research && negative {
        return PrClass::CloseWithLedger;
    }
    let landable = [
        "fix",
        "security",
        "perf",
        "build(deps)",
        "chore(deps)",
        "test",
        "build",
        "deps",
    ]
    .iter()
    .any(|p| t.starts_with(p))
        // Release PRs: "ruvector 0.3.12: fix ..." / "pkg v1.2.0: ...".
        || release_title_re().is_match(&t)
        || labels
            .iter()
            .any(|l| l == "security-scan" || l == "dependencies")
        || head.starts_with("dependabot/");
    if landable && !research {
        PrClass::Land
    } else {
        PrClass::Other
    }
}

/// Reduce a `statusCheckRollup` array to one word.
pub fn rollup(v: &Value) -> String {
    let Some(a) = v.as_array() else {
        return "unknown".into();
    };
    if a.is_empty() {
        return "none".into();
    }
    let bad = [
        "FAILURE",
        "ERROR",
        "TIMED_OUT",
        "CANCELLED",
        "ACTION_REQUIRED",
    ];
    let concl = |x: &Value| {
        x["conclusion"]
            .as_str()
            .or(x["state"].as_str())
            .unwrap_or("")
            .to_string()
    };
    if a.iter().any(|x| bad.contains(&concl(x).as_str())) {
        "failing".into()
    } else if a.iter().any(|x| {
        x["status"]
            .as_str()
            .map(|s| s != "COMPLETED")
            .unwrap_or(false)
            || concl(x) == "PENDING"
    }) {
        "pending".into()
    } else {
        "passing".into()
    }
}

pub fn parse_list(repo: &str, v: &Value) -> Vec<OpenPr> {
    let today = chrono::Utc::now();
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|p| {
                    let title = p["title"].as_str().unwrap_or_default().to_string();
                    let head = p["headRefName"].as_str().unwrap_or_default().to_string();
                    let labels: Vec<String> = p["labels"]
                        .as_array()
                        .map(|l| {
                            l.iter()
                                .filter_map(|x| x["name"].as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    let updated_at = p["updatedAt"].as_str().unwrap_or_default().to_string();
                    let stale_days = chrono::DateTime::parse_from_rfc3339(&updated_at)
                        .map(|t| (today - t.with_timezone(&chrono::Utc)).num_days())
                        .unwrap_or(0);
                    let number = p["number"].as_u64().unwrap_or(0);
                    OpenPr {
                        class: classify(&title, &head, &labels),
                        url: p["url"]
                            .as_str()
                            .map(String::from)
                            .unwrap_or_else(|| format!("https://github.com/{repo}/pull/{number}")),
                        checks: p
                            .get("statusCheckRollup")
                            .map(rollup)
                            .unwrap_or_else(|| "unknown".into()),
                        draft: p["isDraft"].as_bool().unwrap_or(false),
                        number,
                        title,
                        head,
                        updated_at,
                        labels,
                        stale_days,
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `gh pr list` for one repo; with `checks`, one `gh pr view` per PR (the
/// rollup in the list query times out on repos with ~100 open PRs).
pub fn fetch(repo: &str, checks: bool, cwd: &Path) -> Result<RepoPrs> {
    let t = Duration::from_secs(120);
    let o = run_timeout(
        "gh",
        &[
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "500",
            "--json",
            "number,title,updatedAt,headRefName,isDraft,labels,url",
        ],
        cwd,
        t,
    )?;
    if o.status != Some(0) {
        bail!("gh pr list {repo}: {}", clip(o.stderr.trim(), 300));
    }
    let v: Value = serde_json::from_str(&o.stdout).context("gh pr list JSON")?;
    let mut prs = parse_list(repo, &v);
    if checks {
        let got = par_map(&prs, 4, |p| {
            let n = p.number.to_string();
            run_timeout(
                "gh",
                &[
                    "pr",
                    "view",
                    &n,
                    "--repo",
                    repo,
                    "--json",
                    "statusCheckRollup",
                ],
                cwd,
                t,
            )
            .ok()
            .filter(|o| o.status == Some(0))
            .and_then(|o| serde_json::from_str::<Value>(&o.stdout).ok())
            .map(|v| rollup(&v["statusCheckRollup"]))
            .unwrap_or_else(|| "unknown".into())
        });
        for (p, c) in prs.iter_mut().zip(got) {
            p.checks = c;
        }
    }
    let (main_sha, main_red) = main_status(repo, cwd).unwrap_or((None, vec![]));
    Ok(RepoPrs {
        repo: repo.to_ascii_lowercase(),
        fetched_at: now(),
        prs,
        main_red,
        main_sha,
    })
}

pub fn subject(n: u64) -> String {
    format!("pr:#{n}")
}

/// Work item for a landable / closable PR.
pub fn item(repo: &str, p: &OpenPr, downloads: f64, weight: f64) -> Option<WorkItem> {
    let kind = match p.class {
        PrClass::Land => WorkKind::Land,
        PrClass::CloseWithLedger => WorkKind::CloseWithLedger,
        PrClass::Other => return None,
    };
    let t = p.title.to_ascii_lowercase();
    let conf = match kind {
        WorkKind::CloseWithLedger => 0.6,
        _ if t.starts_with("security") || t.contains("(security)") => 0.9,
        _ if t.starts_with("fix") => 0.8,
        _ => 0.65,
    };
    let stale = 1.0 + (p.stale_days.clamp(0, 30) as f64) / 30.0;
    let sig = Signal {
        kind: SignalKind::OpenPr,
        repo: repo.to_string(),
        file: p.url.clone(),
        line: p.number as usize,
        evidence: clip(
            &format!(
                "#{} {} [checks {}, {} days idle{}]",
                p.number,
                p.title,
                p.checks,
                p.stale_days,
                if p.draft { ", draft" } else { "" }
            ),
            220,
        ),
        subject: Some(subject(p.number)),
        keyword: Some(p.head.clone()),
        candidate: None,
        confidence: conf,
        severity: None,
        occurrences: 1,
    };
    Some(WorkItem {
        id: crate::queue::item_id(kind, repo, &subject(p.number)),
        kind,
        status: LaneStatus::Deferred,
        repo: repo.to_string(),
        subject: Some(subject(p.number)),
        candidate: None,
        signal_kind: SignalKind::OpenPr,
        score: (10.0 + downloads).ln() * conf * stale * weight,
        disclosure: "public_pr".into(),
        required_test: match kind {
            WorkKind::CloseWithLedger => format!("Ledger entry (outcome rejected, reason = the PR's verdict and evidence) recorded BEFORE `gh pr close {}`.", p.number),
            _ => format!("PR #{} rebased on main with every required check green, then merged (squash).", p.number),
        },
        evidence: vec![sig],
        gates: vec![],
        brief: String::new(),
    })
}

/// An open PR whose title/branch already covers an item's candidate or
/// subject (every significant token present, one of them >= 4 chars).
pub fn covering<'a>(it: &WorkItem, prs: &'a [OpenPr]) -> Option<&'a OpenPr> {
    if matches!(it.kind, WorkKind::Land | WorkKind::CloseWithLedger) {
        return None;
    }
    let names: Vec<String> = [it.candidate.as_deref(), it.subject.as_deref()]
        .into_iter()
        .flatten()
        // Package names and audit slugs only: `release:`, `pr:#`, `deps:`
        // and `<repo>:` subjects describe the work, not what a PR touches.
        .filter(|n| n.starts_with("audit:") || !n.contains(':'))
        .map(|n| n.trim_start_matches("audit:").to_string())
        .collect();
    let word = |hay: &str, t: &str| {
        hay.match_indices(t).any(|(i, _)| {
            let before = hay[..i]
                .chars()
                .last()
                .is_none_or(|c| !c.is_ascii_alphanumeric());
            let after = hay[i + t.len()..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_ascii_alphanumeric());
            before && after
        })
    };
    prs.iter().find(|p| {
        let hay = format!("{} {}", p.title, p.head).to_ascii_lowercase();
        names.iter().any(|n| {
            let bare = n.rsplit('/').next().unwrap_or(n).to_ascii_lowercase();
            let toks: Vec<String> = name_tokens(n)
                .into_iter()
                .filter(|t| t.len() >= 3)
                .collect();
            if toks.len() >= 2 {
                // Multi-token names: every token as a whole word.
                toks.iter().any(|t| t.len() >= 4) && toks.iter().all(|t| word(&hay, t))
            } else {
                // One significant token ("learning-wasm"): the full name.
                bare.len() >= 4 && word(&hay, &bare)
            }
        })
    })
}

pub fn load_all(work_dir: &Path) -> Result<Vec<RepoPrs>> {
    let dir = work_dir.join("prs");
    let mut out = Vec::new();
    if dir.exists() {
        for e in std::fs::read_dir(&dir)? {
            let p = e?.path();
            if p.extension().and_then(|x| x.to_str()) == Some("json") {
                out.push(
                    serde_json::from_str(&std::fs::read_to_string(&p)?)
                        .with_context(|| format!("parsing {}", p.display()))?,
                );
            }
        }
    }
    Ok(out)
}

/// Failing workflows on the default branch's latest head commit
/// (latest run per workflow name; cancelled/skipped runs are ignored).
pub fn red_workflows(runs: &Value) -> (Option<String>, Vec<String>) {
    let Some(a) = runs.as_array() else {
        return (None, vec![]);
    };
    let Some(head) = a.first().and_then(|r| r["headSha"].as_str()) else {
        return (None, vec![]);
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut red = Vec::new();
    for r in a.iter().filter(|r| r["headSha"].as_str() == Some(head)) {
        let name = r["workflowName"]
            .as_str()
            .or(r["name"].as_str())
            .unwrap_or("?")
            .to_string();
        if !seen.insert(name.clone()) {
            continue;
        }
        if matches!(
            r["conclusion"].as_str(),
            Some("failure" | "timed_out" | "startup_failure")
        ) {
            red.push(name);
        }
    }
    red.sort();
    (Some(head.to_string()), red)
}

fn main_status(repo: &str, cwd: &Path) -> Option<(Option<String>, Vec<String>)> {
    let t = Duration::from_secs(60);
    let b = run_timeout(
        "gh",
        &[
            "repo",
            "view",
            repo,
            "--json",
            "defaultBranchRef",
            "--jq",
            ".defaultBranchRef.name",
        ],
        cwd,
        t,
    )
    .ok()?;
    let branch = b.stdout.trim().to_string();
    if b.status != Some(0) || branch.is_empty() {
        return None;
    }
    let o = run_timeout(
        "gh",
        &[
            "run",
            "list",
            "--repo",
            repo,
            "--branch",
            &branch,
            "--limit",
            "40",
            "--json",
            "workflowName,name,conclusion,headSha,status",
        ],
        cwd,
        t,
    )
    .ok()?;
    let v: Value = serde_json::from_str(&o.stdout).ok()?;
    Some(red_workflows(&v))
}

/// Hygiene item: the default branch is already red.
pub fn main_red_item(r: &RepoPrs, downloads: f64, weight: f64) -> WorkItem {
    let sha = r.main_sha.clone().unwrap_or_default();
    let sig = Signal {
        kind: SignalKind::MainRed,
        repo: r.repo.clone(),
        file: format!("https://github.com/{}/commit/{}", r.repo, sha),
        line: 0,
        evidence: clip(
            &format!(
                "main is red at {}: {}",
                &sha[..sha.len().min(9)],
                r.main_red.join(", ")
            ),
            220,
        ),
        subject: Some("ci:main-red".into()),
        keyword: None,
        candidate: None,
        confidence: 0.9,
        severity: None,
        occurrences: r.main_red.len(),
    };
    WorkItem {
        id: crate::queue::item_id(WorkKind::Hygiene, &r.repo, "ci:main-red"),
        kind: WorkKind::Hygiene,
        status: LaneStatus::Deferred,
        repo: r.repo.clone(),
        subject: Some("ci:main-red".into()),
        candidate: None,
        signal_kind: SignalKind::MainRed,
        score: (10.0 + downloads).ln() * 0.9 * weight,
        disclosure: "public_pr".into(),
        required_test:
            "Every workflow listed in the evidence is green on the default branch after the fix."
                .into(),
        evidence: vec![sig],
        gates: vec![],
        brief: String::new(),
    }
}
