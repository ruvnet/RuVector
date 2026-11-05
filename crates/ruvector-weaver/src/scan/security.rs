//! Security lane: `npm audit --json --package-lock-only` for every shallow
//! package-lock.json, `cargo deny check advisories` when a Cargo.lock is at
//! the root. Time-boxed; a failing tool becomes a note, never aborts a scan.
//! Critical/high findings are routed to private advisories by the queue.

use crate::model::{Signal, SignalKind};
use crate::util::{clip, run_timeout};
use serde_json::Value;
use std::path::Path;
use std::time::Duration;

/// Parse `npm audit --json` (v7+ `vulnerabilities` map) into signals.
pub fn parse_npm_audit(repo: &str, lock_rel: &str, lock_text: &str, audit: &Value) -> Vec<Signal> {
    let Some(vulns) = audit["vulnerabilities"].as_object() else {
        return vec![];
    };
    let mut out = Vec::new();
    for (name, v) in vulns {
        let severity = v["severity"].as_str().unwrap_or("unknown").to_string();
        let titles: Vec<String> = v["via"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x["title"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let needle = format!("\"node_modules/{name}\"");
        let line = lock_text
            .lines()
            .position(|l| l.contains(&needle))
            .map(|i| i + 1)
            .unwrap_or(1);
        let fix = match &v["fixAvailable"] {
            Value::Bool(true) => "fix available".to_string(),
            Value::Object(o) => format!(
                "fix: {}@{}",
                o.get("name").and_then(|x| x.as_str()).unwrap_or(name),
                o.get("version").and_then(|x| x.as_str()).unwrap_or("?")
            ),
            _ => "no fix".to_string(),
        };
        let confidence = match severity.as_str() {
            "critical" => 1.0,
            "high" => 0.9,
            "moderate" => 0.6,
            _ => 0.3,
        };
        out.push(Signal {
            kind: SignalKind::Security,
            repo: repo.to_string(),
            file: lock_rel.to_string(),
            line,
            evidence: clip(
                &format!(
                    "npm audit: {name} [{severity}] {} ({fix})",
                    titles.join("; ")
                ),
                240,
            ),
            subject: Some(name.clone()),
            keyword: Some("npm-audit".into()),
            candidate: None,
            confidence,
            severity: Some(severity),
            occurrences: 1,
        });
    }
    out
}

/// Parse `cargo deny --format json check advisories` stderr lines.
pub fn parse_cargo_deny(repo: &str, stderr: &str) -> Vec<Signal> {
    let mut out = Vec::new();
    for line in stderr.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let f = &v["fields"];
        let Some(adv) = f.get("advisory").filter(|a| a.is_object()) else {
            continue;
        };
        let id = adv["id"].as_str().unwrap_or("advisory");
        let pkg = adv["package"].as_str().unwrap_or("?");
        let sev = f["severity"].as_str().unwrap_or("error");
        let cvss_sev = adv["cvss"].as_str().map(|_| "high");
        out.push(Signal {
            kind: SignalKind::Security,
            repo: repo.to_string(),
            file: "Cargo.lock".into(),
            line: 1,
            evidence: clip(
                &format!(
                    "cargo deny: {id} in {pkg}: {}",
                    adv["title"].as_str().unwrap_or("")
                ),
                240,
            ),
            subject: Some(pkg.to_string()),
            keyword: Some(format!("cargo-deny:{id}")),
            candidate: None,
            confidence: if sev == "error" { 0.8 } else { 0.4 },
            severity: Some(cvss_sev.unwrap_or("moderate").to_string()),
            occurrences: 1,
        });
    }
    out
}

/// Run the security tools under `root`; `notes` collects tool failures.
pub fn run(repo: &str, root: &Path, timeout: Duration, notes: &mut Vec<String>) -> Vec<Signal> {
    let mut out = Vec::new();
    let locks: Vec<_> = walkdir::WalkDir::new(root)
        .max_depth(3)
        .into_iter()
        .filter_entry(|e| e.file_name() != "node_modules" && e.file_name() != ".git")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name() == "package-lock.json")
        .take(6)
        .collect();
    for lock in locks {
        let dir = lock.path().parent().unwrap_or(root);
        let rel = lock
            .path()
            .strip_prefix(root)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        match run_timeout(
            "npm",
            &["audit", "--json", "--package-lock-only"],
            dir,
            timeout,
        ) {
            Ok(o) if o.timed_out => notes.push(format!("npm audit timed out in {rel}")),
            Ok(o) => match serde_json::from_str::<Value>(&o.stdout) {
                Ok(v) => {
                    let text = std::fs::read_to_string(lock.path()).unwrap_or_default();
                    out.extend(parse_npm_audit(repo, &rel, &text, &v));
                }
                Err(_) => notes.push(format!("npm audit in {rel}: unparseable output")),
            },
            Err(e) => notes.push(format!("npm audit in {rel}: {e}")),
        }
    }
    if root.join("Cargo.lock").exists() {
        match run_timeout(
            "cargo",
            &["deny", "--format", "json", "check", "advisories"],
            root,
            timeout,
        ) {
            Ok(o) if o.timed_out => notes.push("cargo deny timed out".into()),
            Ok(o) => out.extend(parse_cargo_deny(repo, &o.stderr)),
            Err(e) => notes.push(format!("cargo deny: {e}")),
        }
    }
    out
}

/// Finding from a published upstream advisory (npm audit / RustSec):
/// already public, so a dependency-bump PR is the right channel.
pub fn is_upstream_advisory(s: &Signal) -> bool {
    s.kind == SignalKind::Security
        && s.keyword
            .as_deref()
            .map(|k| k == "npm-audit" || k.starts_with("cargo-deny:"))
            .unwrap_or(false)
}
