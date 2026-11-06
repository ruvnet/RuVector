//! `weaver gate <pkg>`: candidate health. Recent publish, non-empty tarball
//! with real entrypoints, tarball leak scan, dependency-cycle check against
//! the consumer, and a smoke import in a throwaway npm project. For NEW
//! packages (autopilot creation) it adds: real implementation (no stub /
//! re-export-only source), tests present (and optionally passing), and the
//! new-name rate limit from config.

use crate::config::Config;
use crate::http::Http;
use crate::model::{GateReport, GateResult, Graph};
use crate::registry::Registry;
use crate::util::{clip, now, run_timeout};
use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;
use std::collections::{BTreeSet, VecDeque};
use std::io::Read;
use std::path::Path;
use std::time::Duration;

fn res(name: &str, passed: bool, detail: impl Into<String>) -> GateResult {
    GateResult {
        name: name.to_string(),
        passed,
        detail: detail.into(),
    }
}

/// Paths in a tarball file list that match a leak pattern.
pub fn leak_scan(files: &[String], patterns: &[String]) -> Result<Vec<String>> {
    let res: Vec<Regex> = patterns
        .iter()
        .map(|p| Regex::new(p).with_context(|| format!("bad leak pattern {p}")))
        .collect::<Result<_>>()?;
    Ok(files
        .iter()
        .filter(|f| res.iter().any(|r| r.is_match(f)))
        .cloned()
        .collect())
}

/// Entry points a manifest declares (`main`, `module`, `exports`, `bin`),
/// normalised to tarball-relative paths without `./`.
pub fn entrypoints(manifest: &Value) -> Vec<String> {
    let mut out = BTreeSet::new();
    let mut push = |s: &str| {
        if !s.contains('*') {
            out.insert(s.trim_start_matches("./").to_string());
        }
    };
    for k in ["main", "module", "browser"] {
        if let Some(s) = manifest[k].as_str() {
            push(s);
        }
    }
    fn walk_exports(v: &Value, push: &mut dyn FnMut(&str)) {
        match v {
            Value::String(s) => push(s),
            Value::Object(o) => o.values().for_each(|x| walk_exports(x, push)),
            Value::Array(a) => a.iter().for_each(|x| walk_exports(x, push)),
            _ => {}
        }
    }
    walk_exports(&manifest["exports"], &mut push);
    match &manifest["bin"] {
        Value::String(s) => push(s),
        Value::Object(o) => o.values().filter_map(|v| v.as_str()).for_each(&mut push),
        _ => {}
    }
    out.into_iter().filter(|p| !p.ends_with(".d.ts")).collect()
}

/// Entry points declared but absent from the tarball (`index.js` implied
/// when nothing is declared).
pub fn missing_entrypoints(manifest: &Value, files: &[String]) -> (Vec<String>, Vec<String>) {
    let mut eps = entrypoints(manifest);
    if eps.is_empty() {
        eps.push("index.js".into());
    }
    let set: BTreeSet<&str> = files.iter().map(String::as_str).collect();
    let missing = eps
        .iter()
        .filter(|e| {
            let e = e.as_str();
            !(set.contains(e)
                || set.contains(format!("{e}.js").as_str())
                || set.contains(format!("{e}/index.js").as_str()))
        })
        .cloned()
        .collect();
    (eps, missing)
}

/// Does `from`'s dependency closure (within the graph) reach any of `targets`?
pub fn reaches(graph: &Graph, from: &str, targets: &BTreeSet<String>) -> Option<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut q = VecDeque::from([vec![from.to_string()]]);
    while let Some(path) = q.pop_front() {
        let cur = path.last().expect("non-empty path").clone();
        if !seen.insert(cur.clone()) {
            continue;
        }
        let Some(p) = graph.get(&cur) else { continue };
        for d in p.all_deps() {
            let mut next = path.clone();
            next.push(d.clone());
            if targets.contains(d) {
                return Some(next);
            }
            q.push_back(next);
        }
    }
    None
}

/// Tarball file list (paths without the leading `package/`).
pub fn tarball_files(bytes: &[u8]) -> Result<Vec<String>> {
    let gz = flate2::read::GzDecoder::new(bytes);
    let mut ar = tar::Archive::new(gz);
    let mut out = Vec::new();
    for e in ar.entries()? {
        let mut e = e?;
        let p = e.path()?.to_string_lossy().to_string();
        let p = p.split_once('/').map(|(_, r)| r.to_string()).unwrap_or(p);
        if e.header().entry_type().is_file() {
            out.push(p);
        }
        let mut sink = [0u8; 8192];
        while e.read(&mut sink)? > 0 {}
    }
    out.sort();
    Ok(out)
}

/// Smoke import in a temp npm project: `require`, falling back to `import()`.
/// On success returns the module's export names.
pub fn smoke(
    cfg: &Config,
    pkg: &str,
    version: &str,
) -> Result<std::result::Result<Vec<String>, String>> {
    let root = cfg.work_dir.join("smoke");
    std::fs::create_dir_all(&root)?;
    let dir = tempfile::Builder::new()
        .prefix("smoke-")
        .tempdir_in(&root)?;
    std::fs::write(
        dir.path().join("package.json"),
        r#"{"name":"weaver-smoke","private":true}"#,
    )?;
    let t = Duration::from_secs(cfg.policy.gate.smoke_timeout_secs);
    let spec = format!("{pkg}@{version}");
    let i = run_timeout(
        "npm",
        &[
            "install",
            &spec,
            "--ignore-scripts",
            "--no-audit",
            "--no-fund",
            "--loglevel=error",
        ],
        dir.path(),
        t,
    )?;
    if i.timed_out || i.status != Some(0) {
        return Ok(Err(format!(
            "npm install failed: {}",
            clip(i.stderr.trim(), 300)
        )));
    }
    let js = r#"const p=process.argv[1];const show=m=>{console.log(JSON.stringify(Object.keys(m||{}).slice(0,60)))};
try{show(require(p))}catch(e){import(p).then(show).catch(err=>{console.error(String(e&&e.message)+' | '+String(err&&err.message));process.exit(1)})}"#;
    let n = run_timeout(
        "node",
        &["-e", js, pkg],
        dir.path(),
        Duration::from_secs(60),
    )?;
    if n.status != Some(0) {
        return Ok(Err(format!(
            "import failed: {}",
            clip(n.stderr.trim(), 300)
        )));
    }
    let keys: Vec<String> = n
        .stdout
        .lines()
        .last()
        .and_then(|l| serde_json::from_str(l).ok())
        .unwrap_or_default();
    Ok(Ok(keys))
}

/// True when a queue item carries no package to gate: no candidate, or the
/// `-` placeholder briefs and listings print for one (e.g. `ci:main-red`).
/// Querying the registry for `-` returns HTTP 405.
pub fn is_packageless(pkg: &str) -> bool {
    let p = pkg.trim();
    p.is_empty() || p == "-"
}

pub struct GateOptions<'a> {
    pub consumer: Option<&'a str>,
    pub smoke: bool,
    /// The exact package in the consumer repo that will take the dependency.
    pub consumer_package: Option<&'a str>,
    /// Release-regression checks (repo change, surface drop) with ledger
    /// overrides; `None` skips them.
    pub ledger: Option<&'a crate::ledger::Ledger>,
}

/// Registry-side gate for an existing (or just-published) package.
pub fn gate(
    cfg: &Config,
    http: &Http,
    reg: &dyn Registry,
    graph: &Graph,
    pkg: &str,
    opts: &GateOptions,
) -> Result<GateReport> {
    let mut results = Vec::new();
    let mut exports = Vec::new();
    let Some(m) = reg.latest(pkg)? else {
        results.push(res("exists", false, "404 on registry.npmjs.org"));
        return Ok(finish(pkg, None, results, exports));
    };
    let version = m["version"].as_str().unwrap_or_default().to_string();
    results.push(res("exists", true, format!("{pkg}@{version}")));

    let published = graph.get(pkg).and_then(|p| p.last_publish.clone());
    match published
        .as_deref()
        .and_then(|d| chrono::DateTime::parse_from_rfc3339(d).ok())
    {
        Some(t) => {
            let age = (chrono::Utc::now() - t.with_timezone(&chrono::Utc)).num_days();
            let ok = age <= cfg.policy.gate.max_age_days;
            results.push(res(
                "recent_publish",
                ok,
                format!("last publish {age} days ago"),
            ));
        }
        None => results.push(res(
            "recent_publish",
            true,
            "publish date unknown (not in graph); not blocking",
        )),
    }

    let file_count = m["dist"]["fileCount"].as_u64().unwrap_or(0);
    match m["dist"]["tarball"].as_str() {
        Some(url) => match http
            .get_bytes(url, 200 * 1024 * 1024)
            .and_then(|b| tarball_files(&b))
        {
            Ok(files) => {
                let code = files
                    .iter()
                    .filter(|f| {
                        !f.ends_with(".md") && *f != "package.json" && !f.starts_with("LICENSE")
                    })
                    .count();
                results.push(res(
                    "tarball_nonempty",
                    code > 0,
                    format!(
                        "{} files ({code} non-doc; registry fileCount {file_count})",
                        files.len()
                    ),
                ));
                let (eps, missing) = missing_entrypoints(&m, &files);
                results.push(res(
                    "entrypoints",
                    missing.is_empty(),
                    if missing.is_empty() {
                        format!("present: {}", eps.join(", "))
                    } else {
                        format!("missing: {}", missing.join(", "))
                    },
                ));
                let prev = if leak_scan(&files, &cfg.policy.gate.leak_patterns)?.is_empty() {
                    None
                } else {
                    crate::checks::previous_files(http, pkg)
                };
                results.push(crate::checks::leak_result(
                    cfg,
                    pkg,
                    &files,
                    prev.as_ref().map(|(v, f)| (v.as_str(), f.as_slice())),
                )?);
            }
            Err(e) => results.push(res(
                "tarball_nonempty",
                false,
                format!("tarball unreadable: {e}"),
            )),
        },
        None => results.push(res("tarball_nonempty", false, "no dist.tarball")),
    }

    if let Some(consumer) = opts.consumer {
        results.push(crate::checks::cycle_result(
            graph,
            pkg,
            consumer,
            opts.consumer_package,
        ));
    }

    if let Some(ledger) = opts.ledger {
        match crate::release::registry_checks(http, pkg)? {
            Some((_, _, checks)) => {
                results.extend(crate::release::with_overrides(ledger, pkg, checks))
            }
            None => results.push(res(
                "release_history",
                true,
                "single version; nothing to compare",
            )),
        }
    }

    if opts.smoke {
        match smoke(cfg, pkg, &version)? {
            Ok(keys) => {
                results.push(res("smoke_import", true, format!("{} exports", keys.len())));
                exports = keys;
            }
            Err(why) => results.push(res("smoke_import", false, why)),
        }
    }
    Ok(finish(pkg, Some(version), results, exports))
}

pub fn finish(
    pkg: &str,
    version: Option<String>,
    results: Vec<GateResult>,
    exports: Vec<String>,
) -> GateReport {
    GateReport {
        package: pkg.to_string(),
        version,
        checked_at: now(),
        passed: results.iter().all(|r| r.passed),
        results,
        exports,
    }
}

/// Source-level gates for a NEW package directory before publish.
pub fn new_package_source_gates(dir: &Path, cfg: &Config) -> Result<Vec<GateResult>> {
    let mut out = Vec::new();
    let mut code_lines = 0usize;
    let mut reexport_only = true;
    let mut stub_hits = Vec::new();
    let mut tests = 0usize;
    for e in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_entry(|e| {
            !cfg.policy
                .scan
                .skip_dirs
                .iter()
                .any(|s| e.depth() > 0 && e.file_name() == s.as_str())
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let rel = e
            .path()
            .strip_prefix(dir)
            .unwrap_or(e.path())
            .to_string_lossy()
            .to_string();
        let ext = e.path().extension().and_then(|x| x.to_str()).unwrap_or("");
        if !["js", "mjs", "cjs", "ts", "rs"].contains(&ext) || rel.ends_with(".d.ts") {
            continue;
        }
        let text = std::fs::read_to_string(e.path()).unwrap_or_default();
        if crate::scan::markers::is_test(&rel) || text.contains("#[test]") {
            tests += 1;
            if !text.contains("#[test]") {
                continue;
            }
        }
        for (i, l) in text.lines().enumerate() {
            let t = l.trim();
            if t.is_empty() || t.starts_with("//") || t.starts_with('*') || t.starts_with("/*") {
                continue;
            }
            code_lines += 1;
            let reexport = t.starts_with("export * from")
                || t.starts_with("module.exports = require(")
                || t.starts_with("export {") && t.contains(" from ")
                || t.starts_with("pub use ");
            if !reexport && !t.starts_with("import ") && !t.starts_with("use ") {
                reexport_only = false;
            }
            if crate::scan::markers::marker_re().is_match(l) && !t.contains("fallback") {
                stub_hits.push(format!("{rel}:{}", i + 1));
            }
        }
    }
    out.push(res(
        "real_implementation",
        !reexport_only && code_lines >= 30,
        format!("{code_lines} code lines; re-export only: {reexport_only}"),
    ));
    out.push(res(
        "no_stub_markers",
        stub_hits.is_empty(),
        if stub_hits.is_empty() {
            "none".into()
        } else {
            clip(&stub_hits.join(", "), 300)
        },
    ));
    out.push(res(
        "tests_present",
        tests > 0,
        format!("{tests} test files"),
    ));
    Ok(out)
}

/// `npm test` in the package dir, time-boxed.
pub fn run_tests(dir: &Path, timeout: Duration) -> Result<GateResult> {
    let o = run_timeout("npm", &["test", "--silent"], dir, timeout)?;
    let ok = o.status == Some(0) && !o.timed_out;
    Ok(res(
        "tests_pass",
        ok,
        if ok {
            "npm test passed".to_string()
        } else {
            clip(&format!("{}{}", o.stdout, o.stderr), 300)
        },
    ))
}

/// Tarball list for a local dir via `npm pack --dry-run --json`.
pub fn pack_gates(dir: &Path, cfg: &Config, http: &Http) -> Result<Vec<GateResult>> {
    let o = run_timeout(
        "npm",
        &["pack", "--dry-run", "--json", "--ignore-scripts"],
        dir,
        Duration::from_secs(180),
    )?;
    let v: Value = serde_json::from_str(&o.stdout).unwrap_or(Value::Null);
    let files: Vec<String> = v[0]["files"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|f| f["path"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("package.json")).unwrap_or_default(),
    )
    .unwrap_or(Value::Null);
    let (eps, missing) = missing_entrypoints(&manifest, &files);
    // Pre-publish: the version being replaced is the registry's current latest.
    let name = manifest["name"].as_str().unwrap_or_default().to_string();
    let prev = if name.is_empty() || leak_scan(&files, &cfg.policy.gate.leak_patterns)?.is_empty() {
        None
    } else {
        http.get_json(&format!(
            "https://registry.npmjs.org/{}/latest",
            crate::http::encode_pkg(&name)
        ))
        .ok()
        .flatten()
        .and_then(|m| m["version"].as_str().map(String::from))
        .and_then(|v| crate::checks::version_files(http, &name, &v).map(|f| (v, f)))
    };
    let leak = crate::checks::leak_result(
        cfg,
        &name,
        &files,
        prev.as_ref().map(|(v, f)| (v.as_str(), f.as_slice())),
    )?;
    Ok(vec![
        res(
            "tarball_nonempty",
            files.len() > 1,
            format!("{} files", files.len()),
        ),
        res(
            "entrypoints",
            missing.is_empty(),
            if missing.is_empty() {
                format!("present: {}", eps.join(", "))
            } else {
                format!("missing: {}", missing.join(", "))
            },
        ),
        leak,
    ])
}
