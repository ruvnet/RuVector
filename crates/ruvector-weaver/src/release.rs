//! Release-regression detection (incident 2026-11: `@ruvector/ruvllm`
//! 4.0.0/4.0.1 were published from ruvnet/ruvllm over the 3.x line from
//! ruvnet/RuVector, replacing a 137-file runtime with a 16-file unrelated
//! harness, and `ruvector`'s open `>=2.0.0` optional range picked it up).
//!
//! Checks, per package: repository changed between consecutive versions;
//! file count / unpacked size / export surface dropped sharply; the same npm
//! name declared by a second repo; open-ended ranges on ecosystem deps.
//! `weaver gate` refuses a publish when any check trips, unless the ledger
//! holds an explicit `override` entry for that exact check.

use crate::config::NpmPolicy;
use crate::graph::repo_slug;
use crate::ledger::Ledger;
use crate::model::{GateResult, Outcome, Signal, SignalKind};
use crate::util::clip;
use serde_json::Value;

/// What one published (or about-to-be-published) version looks like.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Snapshot {
    pub version: String,
    pub repo: Option<String>,
    pub file_count: u64,
    pub unpacked_size: u64,
    pub export_keys: usize,
}

fn export_keys(v: &Value) -> usize {
    let e = match &v["exports"] {
        Value::Object(o) => o.len(),
        Value::String(_) => 1,
        _ => 0,
    };
    let b = match &v["bin"] {
        Value::Object(o) => o.len(),
        Value::String(_) => 1,
        _ => 0,
    };
    e + b + usize::from(v["main"].is_string())
}

pub fn snapshot(version_doc: &Value) -> Snapshot {
    let repo = match &version_doc["repository"] {
        Value::String(s) => repo_slug(s),
        Value::Object(o) => o.get("url").and_then(|u| u.as_str()).and_then(repo_slug),
        _ => None,
    };
    Snapshot {
        version: version_doc["version"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        repo,
        file_count: version_doc["dist"]["fileCount"].as_u64().unwrap_or(0),
        unpacked_size: version_doc["dist"]["unpackedSize"].as_u64().unwrap_or(0),
        export_keys: export_keys(version_doc),
    }
}

/// (previous, latest) snapshots from a full packument, ordered by publish
/// time; a stable latest is compared with the previous stable version.
pub fn latest_pair(packument: &Value) -> Option<(Snapshot, Snapshot)> {
    let latest = packument["dist-tags"]["latest"].as_str()?;
    let times = packument["time"].as_object()?;
    let mut vs: Vec<(&str, &str)> = times
        .iter()
        .filter(|(k, _)| *k != "created" && *k != "modified")
        .filter_map(|(k, t)| t.as_str().map(|t| (k.as_str(), t)))
        .filter(|(k, _)| packument["versions"].get(*k).is_some())
        .collect();
    vs.sort_by(|a, b| a.1.cmp(b.1));
    let pos = vs.iter().position(|(k, _)| *k == latest)?;
    let stable = !latest.contains('-');
    let prev = vs[..pos]
        .iter()
        .rev()
        .find(|(k, _)| !stable || !k.contains('-'))?
        .0;
    Some((
        snapshot(&packument["versions"][prev]),
        snapshot(&packument["versions"][latest]),
    ))
}

fn res(name: &str, passed: bool, detail: String) -> GateResult {
    GateResult {
        name: name.into(),
        passed,
        detail,
    }
}

/// Compare consecutive versions. Thresholds: files or exports below half,
/// unpacked size below 40 %, with floors so tiny packages do not trip.
pub fn compare(prev: &Snapshot, next: &Snapshot) -> Vec<GateResult> {
    let mut out = Vec::new();
    match (&prev.repo, &next.repo) {
        (Some(a), Some(b)) if a != b => out.push(res(
            "repo_changed",
            false,
            format!("{} came from {a}, {} from {b}", prev.version, next.version),
        )),
        _ => out.push(res(
            "repo_changed",
            true,
            format!("{} -> {}: same repository", prev.version, next.version),
        )),
    }
    let files_drop = prev.file_count >= 8 && next.file_count * 2 < prev.file_count;
    out.push(res(
        "file_count_drop",
        !files_drop,
        format!("files {} -> {}", prev.file_count, next.file_count),
    ));
    let size_drop =
        prev.unpacked_size >= 50_000 && next.unpacked_size * 10 < prev.unpacked_size * 4;
    out.push(res(
        "size_drop",
        !size_drop,
        format!(
            "unpacked bytes {} -> {}",
            prev.unpacked_size, next.unpacked_size
        ),
    ));
    let api_drop = prev.export_keys >= 4 && next.export_keys * 2 < prev.export_keys;
    out.push(res(
        "export_surface_drop",
        !api_drop,
        format!(
            "entry/export keys {} -> {}",
            prev.export_keys, next.export_keys
        ),
    ));
    out
}

/// Ranges that silently accept any future major (`>=x` with no upper
/// bound, `*`, `latest`, `x`, empty).
pub fn is_open_range(range: &str) -> bool {
    let r = range.trim();
    matches!(r, "" | "*" | "x" | "latest" | "next")
        || (r.starts_with(">") && !r.contains('<') && !r.contains("||"))
}

/// Open ranges on ecosystem deps in a manifest JSON: (dep, range).
pub fn open_ranges(manifest: &Value, npm: &NpmPolicy) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for sec in ["dependencies", "optionalDependencies", "peerDependencies"] {
        if let Some(o) = manifest[sec].as_object() {
            for (dep, r) in o {
                let r = r.as_str().unwrap_or_default();
                if npm.is_ecosystem(dep) && is_open_range(r) && !r.starts_with("workspace:") {
                    out.push((dep.clone(), r.to_string()));
                }
            }
        }
    }
    out
}

pub fn subject(check: &str, name: &str) -> String {
    format!("release:{check}:{name}")
}

/// Has the ledger explicitly accepted this tripped check?
pub fn overridden(ledger: &Ledger, check: &str, pkg: &str) -> bool {
    let s = subject(check, pkg);
    ledger
        .entries
        .iter()
        .rev()
        .find(|e| e.subject.as_deref() == Some(s.as_str()) && e.outcome != Outcome::Gated)
        .map(|e| e.outcome == Outcome::Override)
        .unwrap_or(false)
}

/// Apply ledger overrides: a tripped check with an override passes, and
/// says so in its detail.
pub fn with_overrides(ledger: &Ledger, pkg: &str, results: Vec<GateResult>) -> Vec<GateResult> {
    results
        .into_iter()
        .map(|mut r| {
            if !r.passed && overridden(ledger, &r.name, pkg) {
                r.passed = true;
                r.detail = format!("{} (ledger override)", r.detail);
            }
            r
        })
        .collect()
}

/// Queue signals for a package's tripped checks. The lane belongs to the
/// repo that owned the previous version (the line that was replaced).
pub fn signals(
    pkg: &str,
    prev: &Snapshot,
    results: &[GateResult],
    fallback_repo: &str,
) -> Vec<Signal> {
    let repo = prev
        .repo
        .clone()
        .unwrap_or_else(|| fallback_repo.to_string());
    results
        .iter()
        .filter(|r| !r.passed)
        .map(|r| Signal {
            kind: SignalKind::ReleaseRegression,
            repo: repo.clone(),
            file: format!("https://www.npmjs.com/package/{pkg}"),
            line: 0,
            evidence: clip(&format!("{pkg}: {} ({})", r.name, r.detail), 220),
            subject: Some(subject(&r.name, pkg)),
            keyword: Some(r.name.clone()),
            candidate: Some(pkg.to_string()),
            confidence: 1.0,
            severity: Some("high".into()),
            occurrences: 1,
        })
        .collect()
}

/// Registry-side release checks for `pkg` (latest vs previous version).
pub fn registry_checks(
    http: &crate::http::Http,
    pkg: &str,
) -> anyhow::Result<Option<(Snapshot, Snapshot, Vec<GateResult>)>> {
    let url = format!(
        "https://registry.npmjs.org/{}",
        crate::http::encode_pkg(pkg)
    );
    let Some(doc) = http.get_json(&url)? else {
        return Ok(None);
    };
    let Some((prev, latest)) = latest_pair(&doc) else {
        return Ok(None);
    };
    // Compare with the immediate predecessor AND the last release of the
    // previous major: two bad versions in a row (4.0.0, 4.0.1) must not
    // hide the 3.x -> 4.x break.
    let mut bases = vec![prev.clone()];
    if let Some(pm) = previous_major(&doc).filter(|pm| pm.version != prev.version) {
        bases.push(pm);
    }
    let latest = canonical(latest);
    let mut merged: Vec<GateResult> = Vec::new();
    let mut worst = prev;
    for b in bases {
        let b = canonical(b);
        for r in compare(&b, &latest) {
            match merged.iter_mut().find(|m| m.name == r.name) {
                Some(m) if m.passed && !r.passed => {
                    *m = r;
                    worst = b.clone();
                }
                Some(_) => {}
                None => {
                    if !r.passed {
                        worst = b.clone();
                    }
                    merged.push(r);
                }
            }
        }
    }
    Ok(Some((worst, latest, merged)))
}

/// Last stable release of the major line before `latest`'s major.
pub fn previous_major(packument: &Value) -> Option<Snapshot> {
    let latest = packument["dist-tags"]["latest"].as_str()?;
    let major = |v: &str| v.split('.').next().and_then(|m| m.parse::<u64>().ok());
    let lm = major(latest)?;
    let times = packument["time"].as_object()?;
    let mut best: Option<(&str, &str)> = None;
    for (k, t) in times {
        let Some(t) = t.as_str() else { continue };
        if k.contains('-') || packument["versions"].get(k.as_str()).is_none() {
            continue;
        }
        if major(k).map(|m| m < lm).unwrap_or(false) && best.map(|(_, bt)| t > bt).unwrap_or(true) {
            best = Some((k.as_str(), t));
        }
    }
    best.map(|(k, _)| snapshot(&packument["versions"][k]))
}

/// Resolve GitHub renames (`ruvnet/claude-flow` -> `ruvnet/ruflo`) so a
/// renamed repo is not reported as a repository change. Cached; falls back
/// to the given slug when `gh` cannot answer.
pub fn canonical_slug(slug: &str) -> String {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<BTreeMap<String, String>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(v) = cache.lock().expect("slug cache").get(slug) {
        return v.clone();
    }
    let path = format!("repos/{slug}");
    let got = crate::util::run_timeout(
        "gh",
        &["api", &path, "--jq", ".full_name"],
        std::path::Path::new("."),
        std::time::Duration::from_secs(30),
    )
    .ok()
    .filter(|o| o.status == Some(0))
    .map(|o| o.stdout.trim().to_ascii_lowercase())
    .filter(|s| s.contains('/'))
    .unwrap_or_else(|| slug.to_string());
    cache
        .lock()
        .expect("slug cache")
        .insert(slug.to_string(), got.clone());
    got
}

fn canonical(mut s: Snapshot) -> Snapshot {
    s.repo = s.repo.map(|r| canonical_slug(&r));
    s
}

/// Pre-publish release checks for a local package dir against the
/// registry's current latest: repository, file count, size, exports, and
/// open ranges in the manifest about to ship.
pub fn dir_checks(
    http: &crate::http::Http,
    dir: &std::path::Path,
    npm: &NpmPolicy,
) -> anyhow::Result<(String, Vec<GateResult>)> {
    use anyhow::Context;
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("package.json")).context("reading package.json")?,
    )?;
    let name = manifest["name"]
        .as_str()
        .context("package.json has no name")?
        .to_string();
    let o = crate::util::run_timeout(
        "npm",
        &["pack", "--dry-run", "--json", "--ignore-scripts"],
        dir,
        std::time::Duration::from_secs(180),
    )?;
    let pack: Value = serde_json::from_str(&o.stdout).unwrap_or(Value::Null);
    let mut next = snapshot(&manifest);
    next.file_count = pack[0]["entryCount"].as_u64().unwrap_or(0);
    next.unpacked_size = pack[0]["unpackedSize"].as_u64().unwrap_or(0);
    let latest_url = format!(
        "https://registry.npmjs.org/{}/latest",
        crate::http::encode_pkg(&name)
    );
    let mut out = match http.get_json(&latest_url)? {
        Some(latest) => compare(&canonical(snapshot(&latest)), &canonical(next)),
        None => vec![res(
            "repo_changed",
            true,
            "first publish of this name".into(),
        )],
    };
    let open = open_ranges(&manifest, npm);
    out.push(res(
        "open_range",
        open.is_empty(),
        if open.is_empty() {
            "no open-ended ecosystem ranges".into()
        } else {
            open.iter()
                .map(|(d, r)| format!("{d}@{r}"))
                .collect::<Vec<_>>()
                .join(", ")
        },
    ));
    Ok((name, out))
}

/// Patched versions from the ledger: shipped `security` entries whose
/// `version` names `<candidate>@<ver>` (comma-separated lists allowed).
pub fn patched_versions(ledger: &Ledger) -> std::collections::BTreeMap<String, String> {
    let mut out: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for e in ledger
        .entries
        .iter()
        .filter(|e| e.outcome == Outcome::Shipped && e.kind == crate::model::WorkKind::Security)
    {
        let (Some(c), Some(vs)) = (&e.candidate, &e.version) else {
            continue;
        };
        for item in vs.split(',').map(str::trim) {
            if let Some(v) = item.strip_prefix(&format!("{c}@")) {
                let newer = out
                    .get(c)
                    .and_then(|old| crate::semver::parse(old))
                    .zip(crate::semver::parse(v))
                    .map(|(o, n)| n > o)
                    .unwrap_or(true);
                if newer {
                    out.insert(c.clone(), v.to_string());
                }
            }
        }
    }
    out
}

/// Range signals over the published fabric:
/// - `fix_unreachable`: every version a dependent's declared range admits
///   predates a shipped security fix (high priority). A range that starts
///   past the fix (`^2.3.0` vs 2.2.3) reaches it;
/// - `zero_caret`: a `^0.y`/`~0.y` range that already excludes the dep's
///   current release (0.x carets behave like pins).
pub fn range_signals(
    graph: &crate::model::Graph,
    patched: &std::collections::BTreeMap<String, String>,
    npm: &NpmPolicy,
) -> Vec<Signal> {
    let mut out = Vec::new();
    for p in graph.packages.iter().filter(|p| !p.platform_binary) {
        let repo = p
            .repo_slug
            .clone()
            .unwrap_or_else(|| "ruvnet/ruvector".into());
        for (dep, range) in &p.dep_ranges {
            let mk = |check: &str, conf: f64, sev: &str, ev: String| Signal {
                kind: SignalKind::ReleaseRegression,
                repo: repo.clone(),
                file: format!("https://www.npmjs.com/package/{}", p.name),
                line: 0,
                evidence: clip(&ev, 220),
                subject: Some(subject(check, &format!("{}->{dep}", p.name))),
                keyword: Some(check.to_string()),
                candidate: Some(dep.clone()),
                confidence: conf,
                severity: Some(sev.into()),
                occurrences: 1,
            };
            if let Some(fixed) = patched.get(dep) {
                // Unreachable only if no admitted version is at or past the
                // fix. Excluding `fixed` itself is fine when the range starts
                // after it (`^2.3.0` vs a 2.2.3 fix).
                if crate::semver::reaches_fix(range, fixed) == Some(false) {
                    out.push(mk("fix_unreachable", 1.0, "high", format!(
                        "{}@{} declares {dep} \"{range}\", which excludes the security fix in {dep}@{fixed}",
                        p.name, p.version
                    )));
                    continue;
                }
            }
            let current = graph.get(dep).map(|d| d.version.clone());
            if let Some(cur) =
                current.filter(|_| npm.is_ecosystem(dep) && crate::semver::is_zero_caret(range))
            {
                if crate::semver::satisfies(range, &cur) == Some(false) {
                    out.push(mk("zero_caret", 0.4, "low", format!(
                        "{}@{} pins {dep} \"{range}\" (0.x caret) and misses the current {dep}@{cur}",
                        p.name, p.version
                    )));
                }
            }
        }
    }
    out
}
