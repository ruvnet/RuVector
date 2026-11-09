//! `weaver seo propose`: the exact package.json / README / repo-topic change
//! for a package or repo. Cross-links come ONLY from real dependency edges
//! in the graph; nothing here adds a dependency.

use super::audit::{dependents, has_install, PkgFacts, RepoFacts};
use super::taxonomy::{norm, Taxonomy};
use super::{diff, flow, jsontext, FOOTER_END, FOOTER_START, LINKS_END, LINKS_START};
use super::{FLOW_END, FLOW_START};
use crate::model::{Graph, Package};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A package's source files in a local checkout of its repo.
#[derive(Debug, Clone)]
pub struct SourcePkg {
    /// Directory relative to the repo root ("" for the root).
    pub rel_dir: String,
    pub package_json: String,
    /// (file name, contents) of the README next to package.json.
    pub readme: Option<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEdit {
    pub path: String,
    pub before: String,
    pub after: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PkgProposal {
    pub name: String,
    pub repo: Option<String>,
    pub homepage: String,
    pub description: String,
    pub keywords: Vec<String>,
    pub repository: Value,
    pub bugs: Value,
    pub works_with: Vec<String>,
    pub used_by: Vec<String>,
    pub edits: Vec<FileEdit>,
    pub diff: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepoProposal {
    pub slug: String,
    pub homepage_old: Option<String>,
    pub homepage_new: String,
    pub topics_old: Vec<String>,
    pub topics_new: Vec<String>,
    pub add: Vec<String>,
    pub remove: Vec<String>,
}

fn npm_link(n: &str) -> String {
    format!("[`{n}`](https://www.npmjs.com/package/{n})")
}

fn clip_sentence(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    if let Some(i) = cut.rfind(". ").filter(|i| *i > max / 2) {
        return cut[..=i].trim().to_string();
    }
    let i = cut.rfind(' ').unwrap_or(cut.len());
    format!("{}…", cut[..i].trim_end_matches([',', ';', ':', ' ']))
}

/// Description: taxonomy override, else the current text clipped to the
/// limit and prefixed with the primary phrase when it lacks one.
pub fn description(tax: &Taxonomy, name: &str, current: &str, primary: &[String]) -> String {
    if let Some(d) = tax.package(name).and_then(|p| p.description.clone()) {
        return d;
    }
    let max = tax.meta.description_max;
    let mut d = current.trim().to_string();
    if !primary.is_empty() && !tax.has_phrase(&d, &primary[..1]) {
        if let Some(t) = tax.topic(&primary[0]) {
            let mut p = t.phrase.clone();
            if let Some(c) = p.get(..1) {
                p = c.to_ascii_uppercase() + &p[1..];
            }
            d = if d.is_empty() { p } else { format!("{p}: {d}") };
        }
    }
    clip_sentence(&d, max)
}

/// Keywords: primaries, then the package's own taxonomy keywords, then
/// other relevant topics, brands, repo topics; deduped, <= target.
pub fn keywords(tax: &Taxonomy, graph: &Graph, p: &Package, current: &[String]) -> Vec<String> {
    let primary = tax.primary_for(graph, p);
    let mut out: Vec<String> = Vec::new();
    // One keyword per canonical topic: aliases (`rl`, `vector-store`) never
    // crowd out a distinct topic.
    let push = |k: String, out: &mut Vec<String>| {
        let canon = |x: &str| tax.topic(x).map_or(x.to_string(), |t| t.slug.clone());
        let c = canon(&k);
        if out.len() < tax.meta.keywords_target && !out.iter().any(|o| canon(o) == c) {
            out.push(c);
        }
    };
    for t in &primary {
        push(t.clone(), &mut out);
    }
    for b in tax.brands_for(p) {
        push(b, &mut out);
    }
    for k in current {
        let k = norm(k);
        if tax.topic(&k).is_some() || tax.keyword_ok(&p.name, &k) {
            push(k, &mut out);
        }
    }
    for t in tax.secondary_for(p, &primary) {
        push(t, &mut out);
    }
    if out.len() < tax.meta.keywords_min {
        for t in tax.repo_topics(graph, p.repo_slug.as_deref().unwrap_or("")) {
            push(t, &mut out);
        }
    }
    out
}

fn replace_block(md: &str, start: &str, end: &str, block: &str) -> String {
    if let (Some(a), Some(b)) = (md.find(start), md.find(end)) {
        if a < b {
            return format!("{}{}{}", &md[..a], block, &md[b + end.len()..]);
        }
    }
    let mut s = md.trim_end().to_string();
    s.push_str("\n\n");
    s.push_str(block);
    s.push('\n');
    s
}

/// Insert the flow image right after the first H1 (or at the top).
fn place_flow(md: &str, block: &str) -> String {
    if md.contains(FLOW_START) {
        return replace_block(md, FLOW_START, FLOW_END, block);
    }
    let mut lines: Vec<&str> = md.lines().collect();
    let at = lines
        .iter()
        .position(|l| l.starts_with("# "))
        .map(|i| i + 1)
        .unwrap_or(0);
    lines.insert(at, "");
    lines.insert(at + 1, block);
    let mut s = lines.join("\n");
    s.push('\n');
    s
}

pub fn links_block(
    tax: &Taxonomy,
    primary: &[String],
    works: &[(String, &str)],
    used: &[(String, &str)],
) -> String {
    let mut b = format!("{LINKS_START}\n## Works with / Used by\n\n");
    let phrases: Vec<String> = primary
        .iter()
        .filter_map(|t| tax.topic(t).map(|x| x.phrase.clone()))
        .take(3)
        .collect();
    if !phrases.is_empty() {
        b.push_str(&format!(
            "Part of the [ruv.io]({}) open-source stack for {}.\n\n",
            tax.meta.homepage,
            phrases.join(", ")
        ));
    }
    let list = |v: &[(String, &str)]| {
        v.iter()
            .map(|(n, k)| {
                if k.is_empty() {
                    npm_link(n)
                } else {
                    format!("{} ({k})", npm_link(n))
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    b.push_str(&format!(
        "- **Works with** (its runtime dependencies): {}\n",
        if works.is_empty() {
            "no other ruvnet package (standalone)".to_string()
        } else {
            list(works)
        }
    ));
    b.push_str(&format!(
        "- **Used by** (published packages that depend on it): {}\n",
        if used.is_empty() {
            "none yet".to_string()
        } else {
            list(&used[..used.len().min(12)])
        }
    ));
    b.push_str(LINKS_END);
    b
}

pub fn footer_block(tax: &Taxonomy, slug: Option<&str>) -> String {
    let src = slug
        .map(|s| format!(" · [Source](https://github.com/{s})"))
        .unwrap_or_default();
    format!(
        "{FOOTER_START}\n---\n\nBuilt by [rUv](https://ruv.io) · [ruv.io]({}){src}\n{FOOTER_END}",
        tax.meta.homepage
    )
}

/// Raw GitHub URL for a committed asset (renders on npmjs.com too).
pub fn raw_url(slug: &str, branch: &str, rel_dir: &str, file: &str) -> String {
    let dir = rel_dir.trim_matches('/');
    let path = if dir.is_empty() {
        format!("assets/{file}")
    } else {
        format!("{dir}/assets/{file}")
    };
    format!("https://raw.githubusercontent.com/{slug}/{branch}/{path}")
}

#[allow(clippy::too_many_arguments)]
pub fn propose_package(
    tax: &Taxonomy,
    graph: &Graph,
    p: &Package,
    facts: &PkgFacts,
    src: Option<&SourcePkg>,
    repo: Option<&RepoFacts>,
) -> Result<PkgProposal> {
    let primary = tax.primary_for(graph, p);
    let slug = p.repo_slug.clone();
    let mut notes = Vec::new();
    let desc = description(
        tax,
        &p.name,
        facts.description.as_deref().unwrap_or(""),
        &primary,
    );
    let kws = keywords(tax, graph, p, &facts.keywords);
    if kws.len() < tax.meta.keywords_min {
        notes.push(format!(
            "only {} taxonomy keywords fit; add a topic with measured demand rather than padding",
            kws.len()
        ));
    }
    let works: Vec<String> = p
        .all_deps()
        .filter(|d| graph.get(d).is_some_and(|x| !x.platform_binary))
        .cloned()
        .collect();
    let used = dependents(graph, &p.name);
    // optional / peer edges are real edges too (npm installs optional deps
    // by default): label them rather than drop them.
    let works_l: Vec<(String, &str)> = works.iter().map(|d| (d.clone(), edge_kind(p, d))).collect();
    let used_l: Vec<(String, &str)> = used
        .iter()
        .map(|u| {
            (
                u.clone(),
                graph.get(u).map_or("", |x| edge_kind(x, &p.name)),
            )
        })
        .collect();
    let dir = src
        .map(|s| s.rel_dir.clone())
        .or(facts.repository_directory.clone())
        .filter(|d| !d.is_empty());
    let repository = match (&slug, &dir) {
        (Some(s), Some(d)) => {
            json!({"type": "git", "url": format!("git+https://github.com/{s}.git"), "directory": d})
        }
        (Some(s), None) => json!({"type": "git", "url": format!("git+https://github.com/{s}.git")}),
        _ => {
            notes.push("no repository slug known: repository left unchanged".into());
            Value::Null
        }
    };
    let bugs = slug
        .as_ref()
        .map(|s| json!({"url": format!("https://github.com/{s}/issues")}))
        .unwrap_or(Value::Null);
    let mut edits = Vec::new();
    if let Some(s) = src {
        let pj_path = join(&s.rel_dir, "package.json");
        let mut t = s.package_json.clone();
        t = jsontext::set(&t, "description", &json!(desc), &["version", "name"])?;
        t = jsontext::set(
            &t,
            "homepage",
            &json!(tax.meta.homepage),
            &["description", "version"],
        )?;
        t = jsontext::set(&t, "keywords", &json!(kws), &["homepage", "description"])?;
        if !repository.is_null() {
            t = jsontext::set(&t, "repository", &repository, &["keywords", "homepage"])?;
            t = jsontext::set(&t, "bugs", &bugs, &["repository"])?;
        }
        edits.push(FileEdit {
            path: pj_path,
            before: s.package_json.clone(),
            after: t,
        });
        let (rname, before, mut md) = match s.readme.clone() {
            Some((n, t)) => (n, t.clone(), t.replace("\r\n", "\n")),
            None => (
                "README.md".to_string(),
                String::new(),
                format!("# {}\n\n{}\n", p.name, desc),
            ),
        };
        if !has_install(&md) {
            md = before_first_h2(
                &md,
                &format!(
                    "<!-- weaver:seo:install:start -->\n## Install\n\n```bash\nnpm install {}\n```\n<!-- weaver:seo:install:end -->",
                    p.name
                ),
            );
        }
        let flow_steps = tax
            .package(&p.name)
            .map(|e| e.flow.clone())
            .unwrap_or_default();
        if !flow_steps.is_empty() {
            let file = flow::file_name(&p.name);
            let svg = flow::render(&p.name, &flow_steps)?;
            let branch = repo
                .map(|r| r.default_branch.clone())
                .filter(|b| !b.is_empty())
                .unwrap_or_else(|| "main".into());
            let url = raw_url(
                slug.as_deref().unwrap_or_default(),
                &branch,
                &s.rel_dir,
                &file,
            );
            let block = format!(
                "{FLOW_START}\n![{}: {}]({url})\n{FLOW_END}",
                p.name,
                flow_steps.join(" → ")
            );
            md = place_flow(&md, &block);
            let svg_path = join(&join(&s.rel_dir, "assets"), &file);
            edits.push(FileEdit {
                path: svg_path,
                before: String::new(),
                after: svg,
            });
        }
        // Never duplicate a hand-written section: generated blocks are added
        // only where the README lacks one (or replace their own markers).
        let had_ruvio =
            md.replace(FOOTER_START, "").contains("ruv.io") && !md.contains(FOOTER_START);
        if md.contains(LINKS_START) || !super::audit::has_links_block(&md) {
            md = replace_block(
                &md,
                LINKS_START,
                LINKS_END,
                &links_block(tax, &primary, &works_l, &used_l),
            );
        }
        if !had_ruvio {
            md = replace_block(
                &md,
                FOOTER_START,
                FOOTER_END,
                &footer_block(tax, slug.as_deref()),
            );
        }
        if before.contains("\r\n") {
            md = md.replace('\n', "\r\n");
        }
        edits.insert(
            1,
            FileEdit {
                path: join(&s.rel_dir, &rname),
                before,
                after: md,
            },
        );
        let bad = crate::readme::lint(&edits[1].after);
        let had = crate::readme::lint(&edits[1].before).len();
        if bad.len() > had {
            anyhow::bail!("proposal for {} adds unmeasured numeric claims", p.name);
        }
    } else {
        notes.push("no local source: package.json / README diff needs a checkout".into());
    }
    let diff = edits
        .iter()
        .map(|e| diff::unified(&e.path, &e.before, &e.after))
        .collect::<Vec<_>>()
        .join("");
    Ok(PkgProposal {
        name: p.name.clone(),
        repo: slug,
        homepage: tax.meta.homepage.clone(),
        description: desc,
        keywords: kws,
        repository,
        bugs,
        works_with: works,
        used_by: used,
        edits,
        diff,
        notes,
    })
}

pub fn propose_repo(tax: &Taxonomy, graph: &Graph, f: &RepoFacts) -> RepoProposal {
    let new = tax.repo_topics(graph, &f.slug);
    let old: Vec<String> = f.topics.clone();
    RepoProposal {
        slug: f.slug.clone(),
        homepage_old: f.homepage.clone().filter(|h| !h.is_empty()),
        homepage_new: tax.meta.homepage.clone(),
        add: new.iter().filter(|t| !old.contains(t)).cloned().collect(),
        remove: old.iter().filter(|t| !new.contains(t)).cloned().collect(),
        topics_old: old,
        topics_new: new,
    }
}

fn join(dir: &str, f: &str) -> String {
    let d = dir.trim_matches('/');
    if d.is_empty() {
        f.to_string()
    } else {
        format!("{d}/{f}")
    }
}

/// `x.y.z` -> `x.y.(z+1)` from the higher of source and registry versions.
pub fn bump_patch(source: &str, registry: &str) -> Option<String> {
    let parse = |v: &str| -> Option<(u64, u64, u64)> {
        let core = v.split(['-', '+']).next()?;
        let mut it = core.split('.').map(|x| x.parse::<u64>().ok());
        Some((it.next()??, it.next()??, it.next()??))
    };
    let a = parse(source);
    let b = parse(registry);
    let (x, y, z) = match (a, b) {
        (Some(a), Some(b)) => a.max(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        _ => return None,
    };
    Some(format!("{x}.{y}.{}", z + 1))
}

/// Insert `block` before the first `## ` heading (after the intro), or at the end.
fn before_first_h2(md: &str, block: &str) -> String {
    let mut lines: Vec<&str> = md.lines().collect();
    match lines.iter().position(|l| l.starts_with("## ")) {
        Some(i) => {
            lines.insert(i, "");
            lines.insert(i, block);
            let mut s = lines.join("\n");
            s.push('\n');
            s
        }
        None => replace_block(md, "\u{0}", "\u{0}", block),
    }
}

/// How `p` depends on `dep`: "" (dependencies), "optional" or "peer".
pub fn edge_kind(p: &Package, dep: &str) -> &'static str {
    if p.dependencies.iter().any(|d| d == dep) {
        ""
    } else if p.optional_dependencies.iter().any(|d| d == dep) {
        "optional"
    } else if p.peer_dependencies.iter().any(|d| d == dep) {
        "peer"
    } else {
        ""
    }
}
