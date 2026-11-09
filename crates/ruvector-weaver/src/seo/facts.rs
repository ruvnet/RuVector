//! Fact gathering: pure parsers over a registry packument and `gh api
//! repos/<slug>` JSON (unit-tested on fixtures), plus thin network
//! fetchers the CLI uses. Demand measurement for the taxonomy lives here
//! too: npm keyword totals and GitHub topic repo counts.

use super::audit::{PkgFacts, RepoFacts};
use crate::http::{encode_pkg, Http};
use crate::registry::{str_array, url_escape};
use crate::util::run_timeout;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::time::Duration;

fn url_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Object(o) => o.get("url").and_then(|u| u.as_str()).map(String::from),
        _ => None,
    }
}

/// Facts from a full packument (`registry.npmjs.org/<name>`).
pub fn from_packument(name: &str, doc: &Value) -> PkgFacts {
    let latest = doc["dist-tags"]["latest"].as_str().unwrap_or_default();
    let m = &doc["versions"][latest];
    PkgFacts {
        name: name.to_string(),
        version: latest.to_string(),
        homepage: m["homepage"].as_str().map(String::from),
        description: m["description"].as_str().map(String::from),
        keywords: str_array(&m["keywords"]),
        repository_url: url_of(&m["repository"]),
        repository_directory: m["repository"]["directory"].as_str().map(String::from),
        bugs: url_of(&m["bugs"]),
        readme: doc["readme"]
            .as_str()
            .or(m["readme"].as_str())
            .map(String::from),
        deprecated: m["deprecated"].as_str().map(String::from),
        unpacked_size: m["dist"]["unpackedSize"].as_u64(),
        has_bin: !m["bin"].is_null(),
        os_cpu: m["os"].is_array() || m["cpu"].is_array(),
        maintainers: doc["maintainers"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x["name"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        import_ok: None,
    }
}

/// Facts from `gh api repos/<slug>`.
pub fn from_gh_repo(slug: &str, v: &Value) -> RepoFacts {
    RepoFacts {
        slug: v["full_name"]
            .as_str()
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_else(|| slug.to_string()),
        owner: v["owner"]["login"].as_str().unwrap_or_default().to_string(),
        homepage: v["homepage"]
            .as_str()
            .filter(|h| !h.is_empty())
            .map(String::from),
        topics: str_array(&v["topics"]),
        archived: v["archived"].as_bool().unwrap_or(false),
        fork: v["fork"].as_bool().unwrap_or(false),
        private: v["private"].as_bool().unwrap_or(false),
        default_branch: v["default_branch"].as_str().unwrap_or("main").to_string(),
        missing: false,
    }
}

pub fn fetch_packument(http: &Http, name: &str) -> Result<Option<Value>> {
    http.get_json(&format!("https://registry.npmjs.org/{}", encode_pkg(name)))
}

pub fn gh_json(args: &[&str]) -> Option<Value> {
    let o = run_timeout("gh", args, Path::new("."), Duration::from_secs(60)).ok()?;
    if o.status != Some(0) {
        return None;
    }
    serde_json::from_str(&o.stdout).ok()
}

pub fn fetch_repo(slug: &str) -> RepoFacts {
    match gh_json(&["api", &format!("repos/{slug}")]) {
        Some(v) => from_gh_repo(slug, &v),
        None => RepoFacts {
            slug: slug.to_string(),
            missing: true,
            ..RepoFacts::default()
        },
    }
}

/// One topic's measured demand.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Demand {
    pub topic: String,
    /// `registry.npmjs.org/-/v1/search?text=keywords:<t>` total.
    pub npm_kw_total: Option<u64>,
    /// Sum of the top-5 results' monthly downloads.
    pub npm_top5_monthly: Option<u64>,
    /// `gh api search/repositories q=topic:<t>` total_count.
    pub gh_topic_repos: Option<u64>,
}

pub fn parse_npm_demand(v: &Value) -> (Option<u64>, Option<u64>) {
    let total = v["total"].as_u64();
    let top5 = v["objects"].as_array().map(|a| {
        a.iter()
            .take(5)
            .map(|o| o["downloads"]["monthly"].as_u64().unwrap_or(0))
            .sum()
    });
    (total, top5)
}

pub fn measure(http: &Http, topic: &str) -> Demand {
    let url = format!(
        "https://registry.npmjs.org/-/v1/search?text=keywords:{}&size=5",
        url_escape(topic)
    );
    let (npm_kw_total, npm_top5_monthly) = http
        .get_json(&url)
        .ok()
        .flatten()
        .map(|v| parse_npm_demand(&v))
        .unwrap_or((None, None));
    let mut gh = None;
    for _ in 0..3 {
        gh = gh_json(&[
            "api",
            "-X",
            "GET",
            "search/repositories",
            "-f",
            &format!("q=topic:{topic}"),
            "-f",
            "per_page=1",
        ])
        .and_then(|v| v["total_count"].as_u64());
        if gh.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_secs(5));
    }
    // GitHub search API: 30 requests/minute authenticated.
    std::thread::sleep(Duration::from_millis(2200));
    Demand {
        topic: topic.to_string(),
        npm_kw_total,
        npm_top5_monthly,
        gh_topic_repos: gh,
    }
}
