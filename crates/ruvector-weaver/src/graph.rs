//! `weaver graph`: crawl npm (`maintainer:<m>`) and crates.io (owner id),
//! collect version / last publish / monthly downloads / dependency edges /
//! repository, and derive the consumer-repo set.

use crate::config::Config;
use crate::http::{encode_pkg, Http};
use crate::model::{ConsumerRepo, Ecosystem, Graph, Package};
use crate::registry::str_array;
use crate::util::{now, par_map};
use anyhow::Result;
use regex::Regex;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub struct GraphOptions {
    pub fresh_downloads: bool,
    pub include_crates: bool,
    pub workers: usize,
}

/// `-linux-x64-gnu`, `-darwin-arm64`, `-win32-x64-msvc`, ... NAPI binaries.
pub fn is_platform_binary(name: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"-(linux|darwin|win32|android|freebsd|openbsd|sunos|wasm32)-(x64|arm64|ia32|arm|riscv64|s390x|ppc64|universal|wasi)(-(gnu|musl|msvc|gnueabihf|musleabihf|eabi|androideabi))?$")
            .expect("platform regex")
    })
    .is_match(name)
}

/// `git+https://github.com/RuvNet/RuVector.git#main` -> `ruvnet/ruvector`.
pub fn repo_slug(url: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"(?i)(?:github\.com[/:]|^github:)([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
            .expect("slug regex")
    });
    let c = re.captures(url.trim())?;
    let name = c[2].trim_end_matches(".git");
    Some(format!("{}/{}", &c[1], name).to_ascii_lowercase())
}

fn repo_url(v: &Value) -> Option<String> {
    match &v["repository"] {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("url").and_then(|u| u.as_str()).map(String::from),
        _ => None,
    }
}

fn dep_names(v: &Value) -> Vec<String> {
    v.as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default()
}

/// Every npm package maintained by `maintainer`, paginated (250 / page).
fn npm_search(http: &Http, maintainer: &str) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut from = 0usize;
    loop {
        let url = format!(
            "https://registry.npmjs.org/-/v1/search?text=maintainer:{maintainer}&size=250&from={from}"
        );
        let Some(v) = http.get_json(&url)? else { break };
        let objs = v["objects"].as_array().cloned().unwrap_or_default();
        let total = v["total"].as_u64().unwrap_or(0) as usize;
        if objs.is_empty() {
            break;
        }
        from += objs.len();
        out.extend(objs);
        if from >= total {
            break;
        }
    }
    Ok(out)
}

fn npm_package(http: &Http, npm: &crate::config::NpmPolicy, obj: &Value) -> Package {
    let p = &obj["package"];
    let name = p["name"].as_str().unwrap_or_default().to_string();
    let latest = http
        .get_json(&format!(
            "https://registry.npmjs.org/{}/latest",
            encode_pkg(&name)
        ))
        .ok()
        .flatten()
        .unwrap_or(Value::Null);
    let repository =
        repo_url(&latest).or_else(|| p["links"]["repository"].as_str().map(String::from));
    Package {
        platform_binary: is_platform_binary(&name),
        open_ranges: crate::release::open_ranges(&latest, npm)
            .into_iter()
            .map(|(d, r)| format!("{d}@{r}"))
            .collect(),
        version: latest["version"]
            .as_str()
            .or(p["version"].as_str())
            .unwrap_or_default()
            .to_string(),
        last_publish: p["date"].as_str().map(String::from),
        monthly_downloads: obj["downloads"]["monthly"].as_u64().unwrap_or(0),
        unpacked_size: latest["dist"]["unpackedSize"].as_u64().unwrap_or(0),
        runtime_deps: latest["dependencies"].as_object().map_or(0, |o| o.len()),
        dep_ranges: ["dependencies", "optionalDependencies", "peerDependencies"]
            .iter()
            .filter_map(|k| latest[*k].as_object())
            .flat_map(|o| {
                o.iter()
                    .map(|(d, r)| (d.clone(), r.as_str().unwrap_or_default().to_string()))
            })
            .collect(),
        description: p["description"].as_str().unwrap_or_default().to_string(),
        keywords: str_array(&p["keywords"]),
        dependencies: dep_names(&latest["dependencies"]),
        optional_dependencies: dep_names(&latest["optionalDependencies"]),
        peer_dependencies: dep_names(&latest["peerDependencies"]),
        repo_slug: repository.as_deref().and_then(repo_slug),
        repository,
        ecosystem: Ecosystem::Npm,
        name,
    }
}

/// api.npmjs.org last-month downloads: bulk (128 / call) for unscoped
/// names, one call per scoped name (bulk does not support scopes).
fn fresh_downloads(http: &Http, names: &[String], workers: usize) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    let (scoped, unscoped): (Vec<&String>, Vec<&String>) =
        names.iter().partition(|n| n.starts_with('@'));
    for chunk in unscoped.chunks(128) {
        let list = chunk
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let url = format!("https://api.npmjs.org/downloads/point/last-month/{list}");
        if let Ok(Some(v)) = http.get_json(&url) {
            if chunk.len() == 1 {
                out.insert(chunk[0].clone(), v["downloads"].as_u64().unwrap_or(0));
            } else if let Some(o) = v.as_object() {
                for (k, e) in o {
                    out.insert(k.clone(), e["downloads"].as_u64().unwrap_or(0));
                }
            }
        }
    }
    let got = par_map(&scoped, workers.min(4), |n| {
        let url = format!("https://api.npmjs.org/downloads/point/last-month/{n}");
        http.get_json(&url)
            .ok()
            .flatten()
            .and_then(|v| v["downloads"].as_u64())
            .map(|d| ((*n).clone(), d))
    });
    out.extend(got.into_iter().flatten());
    out
}

fn crates_for_user(http: &Http, user_id: u64) -> Result<Vec<Package>> {
    let mut out = Vec::new();
    for page in 1..=50 {
        let url =
            format!("https://crates.io/api/v1/crates?user_id={user_id}&per_page=100&page={page}");
        let Some(v) = http.get_json(&url)? else { break };
        let crates = v["crates"].as_array().cloned().unwrap_or_default();
        if crates.is_empty() {
            break;
        }
        for c in &crates {
            let repository = c["repository"].as_str().map(String::from);
            out.push(Package {
                name: c["name"].as_str().unwrap_or_default().to_string(),
                ecosystem: Ecosystem::Crates,
                version: c["max_stable_version"]
                    .as_str()
                    .or(c["max_version"].as_str())
                    .unwrap_or_default()
                    .to_string(),
                last_publish: c["updated_at"].as_str().map(String::from),
                // crates.io reports a 90-day window; approximate a month.
                monthly_downloads: c["recent_downloads"].as_u64().unwrap_or(0) / 3,
                description: c["description"]
                    .as_str()
                    .unwrap_or_default()
                    .trim()
                    .to_string(),
                keywords: str_array(&c["keywords"]),
                dependencies: vec![],
                optional_dependencies: vec![],
                peer_dependencies: vec![],
                repo_slug: repository.as_deref().and_then(repo_slug),
                repository,
                platform_binary: false,
                open_ranges: vec![],
                unpacked_size: 0,
                runtime_deps: 0,
                dep_ranges: BTreeMap::new(),
            });
        }
        if crates.len() < 100 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1100)); // crates.io: 1 req/s
    }
    Ok(out)
}

/// Group packages by repository into the consumer-repo set.
pub fn consumer_repos(packages: &[Package]) -> Vec<ConsumerRepo> {
    let mut by_repo: BTreeMap<String, ConsumerRepo> = BTreeMap::new();
    for p in packages.iter().filter(|p| !p.platform_binary) {
        let Some(slug) = &p.repo_slug else { continue };
        let r = by_repo.entry(slug.clone()).or_insert_with(|| ConsumerRepo {
            slug: slug.clone(),
            packages: vec![],
            monthly_downloads: 0,
        });
        r.packages.push(p.name.clone());
        r.monthly_downloads += p.monthly_downloads;
    }
    let mut v: Vec<_> = by_repo.into_values().collect();
    v.sort_by_key(|a| std::cmp::Reverse(a.monthly_downloads));
    v
}

pub fn build(http: &Http, cfg: &Config, opts: &GraphOptions) -> Result<Graph> {
    let objs = npm_search(http, &cfg.policy.npm.maintainer)?;
    let mut packages = par_map(&objs, opts.workers, |o| {
        npm_package(http, &cfg.policy.npm, o)
    });
    if opts.fresh_downloads {
        let names: Vec<String> = packages
            .iter()
            .filter(|p| !p.platform_binary)
            .map(|p| p.name.clone())
            .collect();
        let dl = fresh_downloads(http, &names, opts.workers);
        for p in &mut packages {
            if let Some(d) = dl.get(&p.name) {
                p.monthly_downloads = *d;
            }
        }
    }
    if opts.include_crates {
        packages.extend(crates_for_user(http, cfg.policy.crates.user_id)?);
    }
    // Resolve GitHub renames once per distinct slug.
    let mut slugs: Vec<String> = packages
        .iter()
        .filter_map(|p| p.repo_slug.clone())
        .collect();
    slugs.sort();
    slugs.dedup();
    let canon = par_map(&slugs, opts.workers.min(4), |s| {
        crate::release::canonical_slug(s)
    });
    let repo_aliases: BTreeMap<String, String> = slugs
        .into_iter()
        .zip(canon)
        .filter(|(a, b)| a != b)
        .collect();
    for p in &mut packages {
        if let Some(c) = p.repo_slug.as_ref().and_then(|s| repo_aliases.get(s)) {
            p.repo_slug = Some(c.clone());
        }
    }
    packages.sort_by(|a, b| {
        b.monthly_downloads
            .cmp(&a.monthly_downloads)
            .then(a.name.cmp(&b.name))
    });
    Ok(Graph {
        generated_at: now(),
        consumer_repos: consumer_repos(&packages),
        repo_aliases,
        packages,
    })
}
