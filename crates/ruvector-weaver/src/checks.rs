//! Gate checks that need context beyond one tarball, written to avoid the
//! two false positives seen on agentic-flow 3.4.0:
//!
//! - **leak_scan**: a package may ship a directory on purpose (agentic-flow
//!   ships `.claude/`). `[gate.leak_allow]` lists those per package, and a
//!   leak path that the previous published version already carried is
//!   reported as carried over, not new. Only NEW leak paths fail the gate.
//! - **no_cycle**: the cycle is checked against the published dependency
//!   graph from the exact consuming package (`--consumer-package`), or
//!   against the consumer repo's packages *excluding the candidate's own repo
//!   siblings*. agent-booster is published from ruvnet/agentic-flow, so
//!   "agentic-flow -> agent-booster" is a sibling edge, not a cycle.

use crate::config::Config;
use crate::gate::{leak_scan, reaches};
use crate::model::{GateResult, Graph};
use anyhow::{Context, Result};
use regex::Regex;
use std::collections::BTreeSet;

fn res(name: &str, passed: bool, detail: String) -> GateResult {
    GateResult {
        name: name.into(),
        passed,
        detail,
    }
}

/// Split leak paths into (new, carried over from the previous version,
/// allowlisted for this package).
pub fn classify_leaks(
    leaks: &[String],
    allow: &[String],
    prev_files: Option<&[String]>,
) -> Result<(Vec<String>, Vec<String>, Vec<String>)> {
    let allow: Vec<Regex> = allow
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("bad leak_allow pattern {p}")))
        .collect::<Result<_>>()?;
    let prev: BTreeSet<&str> = prev_files
        .unwrap_or(&[])
        .iter()
        .map(String::as_str)
        .collect();
    let (mut new, mut carried, mut allowed) = (vec![], vec![], vec![]);
    for l in leaks {
        if allow.iter().any(|r| r.is_match(l)) {
            allowed.push(l.clone());
        } else if prev.contains(l.as_str()) {
            carried.push(l.clone());
        } else {
            new.push(l.clone());
        }
    }
    Ok((new, carried, allowed))
}

/// The `leak_scan` gate result. `prev` is the previous published version
/// (label, file list) when known.
pub fn leak_result(
    cfg: &Config,
    pkg: &str,
    files: &[String],
    prev: Option<(&str, &[String])>,
) -> Result<GateResult> {
    let leaks = leak_scan(files, &cfg.policy.gate.leak_patterns)?;
    if leaks.is_empty() {
        return Ok(res("leak_scan", true, "clean".into()));
    }
    let allow = cfg
        .policy
        .gate
        .leak_allow
        .get(pkg)
        .cloned()
        .unwrap_or_default();
    let (new, carried, allowed) = classify_leaks(&leaks, &allow, prev.map(|p| p.1))?;
    let mut parts = Vec::new();
    if !new.is_empty() {
        parts.push(format!(
            "NEW leak paths: {}",
            crate::util::clip(&new.join(", "), 300)
        ));
    }
    if !carried.is_empty() {
        parts.push(format!(
            "{} carried over from {}",
            carried.len(),
            prev.map(|p| p.0).unwrap_or("previous")
        ));
    }
    if !allowed.is_empty() {
        parts.push(format!("{} allowlisted for {pkg}", allowed.len()));
    }
    Ok(res("leak_scan", new.is_empty(), parts.join("; ")))
}

/// File list of a published version's tarball.
pub fn version_files(http: &crate::http::Http, pkg: &str, version: &str) -> Option<Vec<String>> {
    let url = format!(
        "https://registry.npmjs.org/{}/{version}",
        crate::http::encode_pkg(pkg)
    );
    let m = http.get_json(&url).ok()??;
    let tarball = m["dist"]["tarball"].as_str()?.to_string();
    let bytes = http.get_bytes(&tarball, 200 * 1024 * 1024).ok()?;
    crate::gate::tarball_files(&bytes).ok()
}

/// Previous published version (before `latest`) and its file list.
pub fn previous_files(http: &crate::http::Http, pkg: &str) -> Option<(String, Vec<String>)> {
    let url = format!(
        "https://registry.npmjs.org/{}",
        crate::http::encode_pkg(pkg)
    );
    let doc = http.get_json(&url).ok()??;
    let (prev, _) = crate::release::latest_pair(&doc)?;
    let files = version_files(http, pkg, &prev.version)?;
    Some((prev.version, files))
}

/// The `no_cycle` gate result over the published dependency graph.
pub fn cycle_result(
    graph: &Graph,
    pkg: &str,
    consumer: &str,
    consumer_package: Option<&str>,
) -> GateResult {
    let own_repo = graph.get(pkg).and_then(|p| p.repo_slug.clone());
    let consumer = graph.canon(consumer);
    let targets: BTreeSet<String> = match consumer_package {
        Some(cp) => BTreeSet::from([cp.to_string()]),
        None => graph
            .repo(&consumer)
            .map(|r| r.packages.clone())
            .unwrap_or_default()
            .into_iter()
            .filter(|t| t != pkg)
            // Siblings published from the candidate's own repo are not a
            // cross-package cycle; name --consumer-package to check one.
            .filter(|t| {
                own_repo.is_none() || graph.get(t).and_then(|p| p.repo_slug.clone()) != own_repo
            })
            .collect(),
    };
    if consumer_package.is_none() && own_repo.as_deref() == Some(consumer.as_str()) {
        return res(
            "no_cycle",
            true,
            format!("{pkg} is published from {consumer} itself (intra-repo); pass --consumer-package to check a specific package"),
        );
    }
    match reaches(graph, pkg, &targets) {
        Some(path) => {
            let edges: Vec<String> = path
                .windows(2)
                .map(|w| {
                    let range = graph
                        .get(&w[0])
                        .and_then(|p| p.dep_ranges.get(&w[1]).cloned())
                        .unwrap_or_default();
                    format!("{} -> {}@{}", w[0], w[1], range)
                })
                .collect();
            res(
                "no_cycle",
                false,
                format!("published-graph cycle: {}", edges.join(", ")),
            )
        }
        None => res(
            "no_cycle",
            true,
            format!(
                "{pkg}'s published dependency closure does not reach {}",
                if consumer_package.is_some() {
                    consumer_package.unwrap_or_default().to_string()
                } else {
                    format!("{consumer}'s packages")
                }
            ),
        ),
    }
}
