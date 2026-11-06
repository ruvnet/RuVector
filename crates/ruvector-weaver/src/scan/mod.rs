//! `weaver scan <repo>`: fetch a consumer repo into the cache and emit
//! signals with file:line evidence (404 imports, undeclared / unused
//! ecosystem deps, stub markers near domain keywords, security findings).

pub mod imports;
pub mod markers;
pub mod security;

use crate::config::Config;
use crate::model::{Graph, ScanReport, Signal, SignalKind};
use crate::registry::Registry;
use crate::util::{clip, now, run_timeout, slug_key};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

const SOURCE_EXT: &[&str] = &["js", "mjs", "cjs", "ts", "tsx", "mts", "cts", "jsx"];

/// A parsed package.json.
#[derive(Debug, Clone)]
pub struct Manifest {
    /// Directory relative to the repo root ("" for the root).
    pub dir: String,
    pub rel: String,
    pub name: Option<String>,
    /// dependencies + optional + peer (runtime edges).
    pub runtime: BTreeSet<String>,
    /// runtime + devDependencies.
    pub declared: BTreeSet<String>,
    pub text: String,
}

fn keys(v: &Value) -> impl Iterator<Item = String> + '_ {
    v.as_object().into_iter().flat_map(|o| o.keys().cloned())
}

pub fn parse_manifest(dir: &str, rel: &str, text: &str) -> Option<Manifest> {
    let v: Value = serde_json::from_str(text).ok()?;
    let runtime: BTreeSet<String> = keys(&v["dependencies"])
        .chain(keys(&v["optionalDependencies"]))
        .chain(keys(&v["peerDependencies"]))
        .collect();
    let mut declared = runtime.clone();
    declared.extend(keys(&v["devDependencies"]));
    Some(Manifest {
        dir: dir.to_string(),
        rel: rel.to_string(),
        name: v["name"].as_str().map(String::from),
        runtime,
        declared,
        text: text.to_string(),
    })
}

/// Clone (depth 1) or refresh `slug` under `<work>/repos`. Local paths are
/// used in place. Returns (root, commit).
pub fn fetch_repo(work_dir: &Path, repo: &str) -> Result<(PathBuf, String, Option<String>)> {
    let local = Path::new(repo);
    if local.is_dir() {
        let slug = local
            .file_name()
            .map(|s| format!("local/{}", s.to_string_lossy()))
            .unwrap_or_else(|| "local/repo".into());
        let commit = git_head(local);
        return Ok((local.to_path_buf(), slug, commit));
    }
    let slug = crate::graph::repo_slug(repo)
        .or_else(|| (repo.matches('/').count() == 1).then(|| repo.to_ascii_lowercase()))
        .with_context(|| format!("not a repo slug, URL or directory: {repo}"))?;
    if !slug
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_./".contains(c))
        || slug.contains("..")
    {
        bail!("refusing suspicious repo slug {slug:?}");
    }
    let dir = work_dir.join("repos").join(slug_key(&slug));
    let t = Duration::from_secs(600);
    if dir.join(".git").exists() {
        let o = run_timeout("git", &["fetch", "--depth", "1", "origin", "HEAD"], &dir, t)?;
        if o.status == Some(0) {
            run_timeout("git", &["reset", "--hard", "-q", "FETCH_HEAD"], &dir, t)?;
        }
    } else {
        std::fs::create_dir_all(dir.parent().expect("repos dir"))?;
        let url = format!("https://github.com/{slug}.git");
        let target = dir.to_string_lossy().to_string();
        let o = run_timeout(
            "git",
            &[
                "clone",
                "-q",
                "--depth",
                "1",
                "--single-branch",
                &url,
                &target,
            ],
            work_dir,
            t,
        )?;
        if o.status != Some(0) {
            bail!("git clone {url} failed: {}", clip(o.stderr.trim(), 300));
        }
    }
    let commit = git_head(&dir);
    Ok((dir, slug, commit))
}

fn git_head(dir: &Path) -> Option<String> {
    run_timeout("git", &["rev-parse", "HEAD"], dir, Duration::from_secs(20))
        .ok()
        .filter(|o| o.status == Some(0))
        .map(|o| o.stdout.trim().to_string())
}

struct Tree {
    sources: Vec<(String, String)>,
    manifests: Vec<Manifest>,
}

fn walk(root: &Path, cfg: &Config) -> Tree {
    let skip: BTreeSet<&str> = cfg
        .policy
        .scan
        .skip_dirs
        .iter()
        .map(String::as_str)
        .collect();
    let mut sources = Vec::new();
    let mut manifests = Vec::new();
    let it = walkdir::WalkDir::new(root).into_iter().filter_entry(|e| {
        !(e.file_type().is_dir()
            && e.depth() > 0
            && skip.contains(e.file_name().to_string_lossy().as_ref()))
    });
    for e in it
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let Ok(rel) = e.path().strip_prefix(root) else {
            continue;
        };
        let rel = rel.to_string_lossy().replace('\\', "/");
        if e.metadata()
            .map(|m| m.len() > cfg.policy.scan.max_file_bytes)
            .unwrap_or(true)
        {
            continue;
        }
        let name = e.file_name().to_string_lossy();
        if name == "package.json" {
            if let Ok(text) = std::fs::read_to_string(e.path()) {
                let dir = rel
                    .trim_end_matches("package.json")
                    .trim_end_matches('/')
                    .to_string();
                if let Some(m) = parse_manifest(&dir, &rel, &text) {
                    manifests.push(m);
                }
            }
            continue;
        }
        let ext = e.path().extension().and_then(|x| x.to_str()).unwrap_or("");
        if !SOURCE_EXT.contains(&ext) || name.ends_with(".d.ts") || name.ends_with(".min.js") {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(e.path()) {
            sources.push((rel, text));
        }
    }
    sources.sort();
    manifests.sort_by_key(|a| std::cmp::Reverse(a.dir.len()));
    Tree { sources, manifests }
}

/// Deepest manifest whose dir contains `rel` (manifests sorted deepest-first).
fn nearest<'a>(manifests: &'a [Manifest], rel: &str) -> Option<&'a Manifest> {
    manifests
        .iter()
        .find(|m| m.dir.is_empty() || rel.starts_with(&format!("{}/", m.dir)))
}

fn manifest_line(m: &Manifest, dep: &str) -> usize {
    let needle = format!("\"{dep}\"");
    m.text
        .lines()
        .position(|l| l.contains(&needle))
        .map(|i| i + 1)
        .unwrap_or(1)
}

/// Pure scan of a checked-out tree (unit-tested on fixtures).
pub fn scan_dir(
    root: &Path,
    repo: &str,
    cfg: &Config,
    registry: &dyn Registry,
    graph: &Graph,
    with_security: bool,
) -> Result<ScanReport> {
    let tree = walk(root, cfg);
    let npm = &cfg.policy.npm;
    let local_names: BTreeSet<String> = tree
        .manifests
        .iter()
        .filter_map(|m| m.name.clone())
        .collect();
    let mut folded: BTreeMap<(SignalKind, String, String), Signal> = BTreeMap::new();
    let mut imported_in: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut notes = Vec::new();
    let mut add = |s: Signal, scope: String| {
        let key = (
            s.kind,
            s.subject
                .clone()
                .or(s.candidate.clone())
                .unwrap_or_default(),
            scope,
        );
        folded
            .entry(key)
            .and_modify(|e| e.occurrences += 1)
            .or_insert(s);
    };

    for (rel, text) in &tree.sources {
        let near = nearest(&tree.manifests, rel);
        let scope = near.map(|m| m.dir.clone()).unwrap_or_default();
        for imp in imports::extract(text) {
            imported_in
                .entry(scope.clone())
                .or_default()
                .insert(imp.package.clone());
            let name = &imp.package;
            let ecosystem = npm.is_ecosystem(name) || graph.get(name).is_some();
            // NAPI loaders probe every platform binary by design; not a signal.
            if !ecosystem || local_names.contains(name) || crate::graph::is_platform_binary(name) {
                continue;
            }
            let base = |kind, confidence, candidate: Option<String>| Signal {
                kind,
                repo: repo.to_string(),
                file: rel.clone(),
                line: imp.line,
                evidence: clip(&imp.text, 200),
                subject: Some(name.clone()),
                keyword: None,
                candidate,
                confidence,
                severity: None,
                occurrences: 1,
            };
            let exists = graph.get(name).is_some() || registry.exists(name).unwrap_or(true);
            if !exists {
                let hit = cfg.capabilities.match_package_name(name);
                let mut s = base(
                    SignalKind::MissingPackage,
                    0.95,
                    hit.as_ref().map(|h| h.capability.package.clone()),
                );
                s.keyword = hit.map(|h| h.keyword);
                add(s, scope.clone());
            } else if near.map(|m| !m.declared.contains(name)).unwrap_or(true) {
                let test = markers::is_test_or_doc(rel) || rel.contains("example");
                let c = if test {
                    0.45
                } else if imp.dynamic {
                    0.7
                } else {
                    0.85
                };
                add(
                    base(SignalKind::UndeclaredImport, c, Some(name.clone())),
                    scope.clone(),
                );
            }
        }
        for s in markers::scan_file(repo, rel, text, &cfg.capabilities) {
            let scope = format!("{}#{}", s.file, s.candidate.clone().unwrap_or_default());
            add(s, scope);
        }
    }

    // Declared ecosystem runtime deps that nothing in scope imports or names.
    for m in &tree.manifests {
        let used = imported_in.get(&m.dir).cloned().unwrap_or_default();
        for dep in m
            .runtime
            .iter()
            .filter(|d| npm.is_ecosystem(d) || graph.get(d).is_some())
        {
            if used.contains(dep) || local_names.contains(dep) {
                continue;
            }
            let quoted = [format!("'{dep}"), format!("\"{dep}"), format!("`{dep}")];
            let mentioned = tree.sources.iter().any(|(rel, t)| {
                (m.dir.is_empty() || rel.starts_with(&format!("{}/", m.dir)))
                    && quoted.iter().any(|q| t.contains(q.as_str()))
            });
            if mentioned {
                continue;
            }
            let line = manifest_line(m, dep);
            add(
                Signal {
                    kind: SignalKind::UnusedDeclared,
                    repo: repo.to_string(),
                    file: m.rel.clone(),
                    line,
                    evidence: clip(m.text.lines().nth(line - 1).unwrap_or("").trim(), 200),
                    subject: Some(dep.clone()),
                    keyword: None,
                    candidate: Some(dep.clone()),
                    confidence: 0.5,
                    severity: None,
                    occurrences: 1,
                },
                m.dir.clone(),
            );
        }
    }

    let mut signals: Vec<Signal> = folded.into_values().collect();
    signals.extend(release_signals(repo, &tree.manifests, graph, npm));
    if with_security {
        let t = Duration::from_secs(cfg.policy.scan.security_timeout_secs);
        signals.extend(security::run(repo, root, t, &mut notes));
    }
    signals.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.file.cmp(&b.file))
            .then(a.line.cmp(&b.line))
    });
    Ok(ScanReport {
        repo: repo.to_string(),
        scanned_at: now(),
        commit: None,
        files_scanned: tree.sources.len(),
        signals,
        notes,
    })
}

/// Release-regression signals from a repo's manifests: an npm name this
/// repo declares that the registry attributes to a different repo, and
/// open-ended ranges on ecosystem deps.
fn release_signals(
    repo: &str,
    manifests: &[Manifest],
    graph: &Graph,
    npm: &crate::config::NpmPolicy,
) -> Vec<Signal> {
    let mut out = Vec::new();
    let sig = |m: &Manifest, line: usize, check: &str, name: &str, ev: String| Signal {
        kind: SignalKind::ReleaseRegression,
        repo: repo.to_string(),
        file: m.rel.clone(),
        line,
        evidence: clip(&ev, 220),
        subject: Some(crate::release::subject(check, name)),
        keyword: Some(check.to_string()),
        candidate: Some(name.to_string()),
        confidence: if check == "name_collision" { 1.0 } else { 0.8 },
        severity: Some("high".into()),
        occurrences: 1,
    };
    for m in manifests {
        let Ok(v) = serde_json::from_str::<Value>(&m.text) else {
            continue;
        };
        if let (Some(name), false) = (&m.name, v["private"].as_bool() == Some(true)) {
            let owner = graph.get(name).and_then(|p| p.repo_slug.clone());
            if let Some(owner) = owner.filter(|o| {
                !o.eq_ignore_ascii_case(&graph.canon(repo)) && !repo.starts_with("local/")
            }) {
                let line = m
                    .text
                    .lines()
                    .position(|l| l.contains("\"name\""))
                    .map_or(1, |i| i + 1);
                out.push(sig(
                    m,
                    line,
                    "name_collision",
                    name,
                    format!(
                        "{name} is declared here but its latest npm release comes from {owner}"
                    ),
                ));
            }
        }
        for (dep, range) in crate::release::open_ranges(&v, npm) {
            let line = manifest_line(m, &dep);
            out.push(sig(
                m,
                line,
                "open_range",
                &dep,
                format!("\"{dep}\": \"{range}\" accepts any future major from any repo"),
            ));
        }
    }
    out
}

/// Load every `scan/*.json` report under the work dir.
pub fn load_reports(work_dir: &Path) -> Result<Vec<ScanReport>> {
    let dir = work_dir.join("scan");
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    for e in std::fs::read_dir(&dir)? {
        let p = e?.path();
        if p.extension().and_then(|x| x.to_str()) == Some("json") {
            let r: ScanReport = serde_json::from_str(&std::fs::read_to_string(&p)?)
                .with_context(|| format!("parsing {}", p.display()))?;
            out.push(r);
        }
    }
    out.sort_by(|a, b| a.repo.cmp(&b.repo));
    Ok(out)
}
