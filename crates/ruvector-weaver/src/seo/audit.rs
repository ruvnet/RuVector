//! `weaver seo audit`: per-package and per-repo metadata findings, ranked
//! by monthly downloads x issue count. Pure over gathered facts so it is
//! unit-tested offline; the CLI gathers facts from the registry and `gh`.

use super::taxonomy::{norm, Taxonomy};
use crate::model::{Graph, Package};
use serde::{Deserialize, Serialize};

/// Registry-side facts for one package (latest version unless noted).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PkgFacts {
    pub name: String,
    pub version: String,
    pub homepage: Option<String>,
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub repository_url: Option<String>,
    pub repository_directory: Option<String>,
    pub bugs: Option<String>,
    /// Packument `readme` (what npmjs.com renders); `None` = unknown.
    pub readme: Option<String>,
    pub deprecated: Option<String>,
    pub unpacked_size: Option<u64>,
    pub has_bin: bool,
    /// Manifest declares `os` / `cpu` (a platform build).
    pub os_cpu: bool,
    pub maintainers: Vec<String>,
    /// Smoke import from an existing gate report; `None` = not verified.
    pub import_ok: Option<bool>,
}

/// GitHub-side facts for one repo.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepoFacts {
    pub slug: String,
    pub owner: String,
    pub homepage: Option<String>,
    pub topics: Vec<String>,
    pub archived: bool,
    pub fork: bool,
    pub private: bool,
    pub default_branch: String,
    /// `gh` could not read the repo (renamed, private, missing).
    pub missing: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Issue {
    pub code: String,
    pub detail: String,
}

fn issue(code: &str, detail: impl Into<String>) -> Issue {
    Issue {
        code: code.into(),
        detail: detail.into(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkgAudit {
    pub name: String,
    pub repo: Option<String>,
    pub version: String,
    pub monthly_downloads: u64,
    pub last_publish: Option<String>,
    pub stale: bool,
    pub primary_topics: Vec<String>,
    pub import: String,
    /// Ships a `bin`: a failed `require` does not mean the CLI is broken.
    #[serde(default)]
    pub has_bin: bool,
    pub dependents: Vec<String>,
    pub issues: Vec<Issue>,
    pub score: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoAudit {
    pub slug: String,
    pub owned: bool,
    pub homepage: Option<String>,
    pub topics: Vec<String>,
    pub proposed_topics: Vec<String>,
    pub monthly_downloads: u64,
    pub issues: Vec<Issue>,
    pub score: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaleRec {
    pub name: String,
    pub monthly_downloads: u64,
    pub last_publish: Option<String>,
    pub import: String,
    pub dependents: Vec<String>,
    pub recommendation: String,
    pub why: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditReport {
    pub generated_at: String,
    pub homepage: String,
    pub packages: Vec<PkgAudit>,
    pub repos: Vec<RepoAudit>,
    pub stale_low_download: Vec<StaleRec>,
}

pub fn same_url(a: &str, b: &str) -> bool {
    let n = |s: &str| {
        s.trim()
            .trim_end_matches('/')
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .trim_start_matches("www.")
            .to_ascii_lowercase()
    };
    n(a) == n(b)
}

pub fn has_install(md: &str) -> bool {
    let l = md.to_ascii_lowercase();
    [
        "npm install",
        "npm i ",
        "pnpm add",
        "yarn add",
        "npx ",
        "bun add",
    ]
    .iter()
    .any(|p| l.contains(p))
}

pub fn has_links_block(md: &str) -> bool {
    let l = md.to_ascii_lowercase();
    l.contains(super::LINKS_START) || (l.contains("works with") && l.contains("used by"))
}

pub fn age_days(ts: Option<&str>, now: chrono::DateTime<chrono::Utc>) -> Option<i64> {
    let t = chrono::DateTime::parse_from_rfc3339(ts?).ok()?;
    Some((now - t.with_timezone(&chrono::Utc)).num_days())
}

/// Published non-platform npm packages that depend on `name`.
pub fn dependents(graph: &Graph, name: &str) -> Vec<String> {
    let mut v: Vec<String> = graph
        .dependents(name)
        .into_iter()
        .filter(|p| !p.platform_binary)
        .map(|p| p.name.clone())
        .collect();
    v.sort();
    v.dedup();
    v
}

pub fn audit_package(
    tax: &Taxonomy,
    graph: &Graph,
    p: &Package,
    f: &PkgFacts,
    now: chrono::DateTime<chrono::Utc>,
) -> PkgAudit {
    let m = &tax.meta;
    let mut out = Vec::new();
    if !f
        .homepage
        .as_deref()
        .is_some_and(|h| same_url(h, &m.homepage))
    {
        out.push(issue(
            "homepage",
            format!(
                "{} != {}",
                f.homepage.as_deref().unwrap_or("(none)"),
                m.homepage
            ),
        ));
    }
    let n = f.keywords.len();
    if n < m.keywords_min || n > m.keywords_max {
        out.push(issue(
            "keyword_count",
            format!("{n} keywords (want {}-{})", m.keywords_min, m.keywords_max),
        ));
    }
    let off: Vec<&String> = f
        .keywords
        .iter()
        .filter(|k| !tax.keyword_ok(&p.name, k))
        .collect();
    if !off.is_empty() {
        let shown: Vec<&str> = off.iter().take(8).map(|s| s.as_str()).collect();
        out.push(issue(
            "keywords_off_taxonomy",
            format!("{} not in taxonomy: {}", off.len(), shown.join(", ")),
        ));
    }
    let primary = tax.primary_for(graph, p);
    if primary.len() < 3 {
        out.push(issue(
            "primary_topics",
            format!("only {} relevant taxonomy topic(s)", primary.len()),
        ));
    }
    match f
        .description
        .as_deref()
        .map(str::trim)
        .filter(|d| !d.is_empty())
    {
        None => out.push(issue("description_missing", "no description")),
        Some(d) => {
            let len = d.chars().count();
            if len > m.description_max {
                out.push(issue(
                    "description_long",
                    format!("{len} chars > {}", m.description_max),
                ));
            }
            if !primary.is_empty() && !tax.has_phrase(d, &primary[..1]) {
                let phrase = tax
                    .topic(&primary[0])
                    .map(|t| t.phrase.clone())
                    .unwrap_or_default();
                out.push(issue(
                    "description_phrase",
                    format!("lacks primary phrase {phrase:?}"),
                ));
            }
        }
    }
    if f.repository_url.is_none() {
        out.push(issue("repository_missing", "no repository"));
    }
    if f.bugs.is_none() {
        out.push(issue("bugs_missing", "no bugs url"));
    }
    let monorepo = p
        .repo_slug
        .as_deref()
        .and_then(|s| graph.repo(s))
        .is_some_and(|r| r.packages.len() > 1);
    if monorepo && f.repository_directory.is_none() {
        out.push(issue(
            "directory_missing",
            "monorepo package without repository.directory",
        ));
    }
    match f.readme.as_deref().map(str::trim) {
        None | Some("") | Some("ERROR: No README data found!") => {
            out.push(issue("readme_missing", "no README on npm"))
        }
        Some(md) => {
            if !has_install(md) {
                out.push(issue("readme_install", "no install / quick-start command"));
            }
            if !has_links_block(md) {
                out.push(issue("readme_links", "no \"Works with / Used by\" section"));
            }
            if !md.contains("ruv.io") {
                out.push(issue("readme_ruvio", "no ruv.io link"));
            }
        }
    }
    let age = age_days(p.last_publish.as_deref(), now);
    let stale = age.is_some_and(|a| a > m.stale_days);
    if stale {
        out.push(issue(
            "stale",
            format!("last publish {} days ago", age.unwrap_or_default()),
        ));
    }
    let import = match f.import_ok {
        Some(true) => "ok",
        Some(false) => "fail",
        None => "unknown",
    };
    if f.import_ok == Some(false) {
        out.push(issue("import_fail", "registry install/import gate failed"));
    }
    if let Some(d) = &f.deprecated {
        out.push(issue("deprecated", d.clone()));
    }
    PkgAudit {
        name: p.name.clone(),
        repo: p.repo_slug.clone(),
        version: f.version.clone(),
        monthly_downloads: p.monthly_downloads,
        last_publish: p.last_publish.clone(),
        stale,
        primary_topics: primary,
        import: import.into(),
        has_bin: f.has_bin,
        dependents: dependents(graph, &p.name),
        score: p.monthly_downloads.max(1) * out.len() as u64,
        issues: out,
    }
}

pub fn audit_repo(tax: &Taxonomy, graph: &Graph, f: &RepoFacts, owner: &str) -> RepoAudit {
    let mut out = Vec::new();
    let proposed = tax.repo_topics(graph, &f.slug);
    if f.missing {
        out.push(issue("repo_unreadable", "gh could not read the repo"));
    } else {
        if !f
            .homepage
            .as_deref()
            .is_some_and(|h| same_url(h, &tax.meta.homepage))
        {
            out.push(issue(
                "homepage",
                format!(
                    "{} != {}",
                    f.homepage
                        .as_deref()
                        .filter(|h| !h.is_empty())
                        .unwrap_or("(none)"),
                    tax.meta.homepage
                ),
            ));
        }
        if f.topics.is_empty() {
            out.push(issue("topics_missing", "no repo topics"));
        } else if f.topics.len() > tax.meta.repo_topics_max {
            out.push(issue("topics_over", format!("{} topics", f.topics.len())));
        }
        let off: Vec<&String> = f.topics.iter().filter(|t| tax.topic(t).is_none()).collect();
        if !off.is_empty() {
            out.push(issue(
                "topics_off_taxonomy",
                format!(
                    "{} off-taxonomy: {}",
                    off.len(),
                    off.iter()
                        .take(8)
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
    }
    let dl = graph.repo(&f.slug).map_or(0, |r| r.monthly_downloads);
    RepoAudit {
        slug: f.slug.clone(),
        owned: f.owner.eq_ignore_ascii_case(owner) && !f.missing,
        homepage: f.homepage.clone(),
        topics: f.topics.clone(),
        proposed_topics: proposed,
        monthly_downloads: dl,
        score: dl.max(1) * out.len() as u64,
        issues: out,
    }
}

/// Recommendation for a stale (> stale_days) package under 1k downloads/mo.
/// Never "deprecate" on downloads or name alone (ADR-356): a deprecation
/// candidate must also fail the import gate AND have no published consumer.
pub fn stale_rec(a: &PkgAudit) -> StaleRec {
    let (rec, why) = match (a.import.as_str(), a.dependents.is_empty()) {
        ("ok", false) => (
            "refresh+republish",
            format!("imports cleanly; used by {}", a.dependents.join(", ")),
        ),
        ("fail", true) if a.has_bin => (
            "leave as-is (CLI: verify the bin, import n/a)",
            "require/import fails but it ships a bin; a CLI-only package needs a bin check, not an import".to_string(),
        ),
        ("fail", true) => (
            "deprecate-candidate (owner, 2FA, after migration audit)",
            "registry import fails and no published package depends on it".to_string(),
        ),
        ("fail", false) => (
            "fix+republish",
            format!("import fails but used by {}", a.dependents.join(", ")),
        ),
        ("ok", true) => (
            "leave as-is",
            "imports cleanly, no published consumer: no reason to churn".into(),
        ),
        _ => (
            "leave as-is (unverified)",
            "import not verified; run `weaver seo audit --smoke-stale`".into(),
        ),
    };
    StaleRec {
        name: a.name.clone(),
        monthly_downloads: a.monthly_downloads,
        last_publish: a.last_publish.clone(),
        import: a.import.clone(),
        dependents: a.dependents.clone(),
        recommendation: rec.into(),
        why,
    }
}

/// Sort by score desc, then name; and derive stale recommendations.
pub fn finish(report: &mut AuditReport) {
    report
        .packages
        .sort_by(|a, b| b.score.cmp(&a.score).then(a.name.cmp(&b.name)));
    report
        .repos
        .sort_by(|a, b| b.score.cmp(&a.score).then(a.slug.cmp(&b.slug)));
    report.stale_low_download = report
        .packages
        .iter()
        .filter(|a| a.stale && a.monthly_downloads < 1000)
        .map(stale_rec)
        .collect();
    report.stale_low_download.sort_by(|a, b| {
        a.recommendation
            .cmp(&b.recommendation)
            .then(a.name.cmp(&b.name))
    });
}

/// Plain-text ranked table (top `n`).
pub fn table(report: &AuditReport, n: usize) -> String {
    let mut s = String::from(
        "| # | package | downloads/mo | issues | score | codes |\n|---:|---|---:|---:|---:|---|\n",
    );
    for (i, a) in report.packages.iter().take(n).enumerate() {
        let codes: Vec<&str> = a.issues.iter().map(|x| x.code.as_str()).collect();
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            i + 1,
            a.name,
            a.monthly_downloads,
            a.issues.len(),
            a.score,
            codes.join(" ")
        ));
    }
    s
}

pub fn norm_topics(v: &[String]) -> Vec<String> {
    v.iter().map(|t| norm(t)).collect()
}
