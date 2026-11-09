//! Portfolio baseline (ADR-356 rollout step 1): one row per package of the
//! owner's pinned inventory, every fact resolved from the published
//! manifest, unknowns carried as the literal `"unknown"`, the owner's class
//! kept as `class_hint` and checked against the manifest, plus a disposition.

use super::audit::PkgFacts;
use super::taxonomy::Taxonomy;
use crate::graph::{is_platform_binary, repo_slug};
use serde::{Serialize, Serializer};
use std::collections::BTreeMap;

/// A fact that is either known or explicitly `"unknown"`.
#[derive(Debug, Clone, PartialEq)]
pub enum Fact<T> {
    Known(T),
    Unknown,
}

impl<T> From<Option<T>> for Fact<T> {
    fn from(o: Option<T>) -> Self {
        o.map_or(Fact::Unknown, Fact::Known)
    }
}

impl<T: Serialize> Serialize for Fact<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Fact::Known(v) => v.serialize(s),
            Fact::Unknown => s.serialize_str("unknown"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub name: String,
    pub in_inventory: bool,
    pub in_graph: bool,
    pub version: Fact<String>,
    pub homepage: Fact<String>,
    pub homepage_is_ruvio: bool,
    pub keywords: Vec<String>,
    pub readme_present: Fact<bool>,
    /// Runtime edges into the ruvnet fabric (deps + optional + peer).
    pub dependency_edges: Vec<String>,
    /// Registry `dist.unpackedSize` of the latest version (not a measured
    /// cold install; that is recorded in the ledger for pilots only).
    pub unpacked_size_bytes: Fact<u64>,
    pub deprecated: Fact<bool>,
    pub monthly_downloads: Fact<u64>,
    pub maintainer_ruvnet: Fact<bool>,
    pub repo: Fact<String>,
    pub directory: Fact<String>,
    pub class_hint: Fact<String>,
    pub class: String,
    pub class_check: String,
    pub disposition: String,
}

/// Owner plan section heading -> class slug.
pub fn class_of_heading(h: &str) -> Option<&'static str> {
    let h = h.to_ascii_lowercase();
    Some(if h.starts_with("pilot") {
        "pilot"
    } else if h.starts_with("entry point") {
        "entry"
    } else if h.starts_with("adapter") {
        "adapter-example"
    } else if h.starts_with("feature") {
        "feature"
    } else if h.starts_with("platform") {
        "platform"
    } else if h.starts_with("migration") {
        "alias"
    } else if h.starts_with("discovery") {
        "discovery"
    } else {
        return None;
    })
}

/// `name -> class` from the owner's distribution-plan markdown tables.
pub fn parse_class_hints(md: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut class: Option<&str> = None;
    for l in md.lines() {
        if let Some(h) = l.strip_prefix("## ") {
            class = class_of_heading(h);
            continue;
        }
        let Some(c) = class else { continue };
        if let Some(rest) = l.strip_prefix("| [`") {
            if let Some(end) = rest.find("`]") {
                out.insert(rest[..end].to_string(), c.to_string());
            }
        }
    }
    out
}

/// Inventory JSON (`{"packages": [names...]}` or `[names...]`).
pub fn parse_inventory(v: &serde_json::Value) -> Vec<String> {
    let arr = v.get("packages").unwrap_or(v);
    let mut out: Vec<String> = arr
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    x.as_str()
                        .map(String::from)
                        .or_else(|| x["name"].as_str().map(String::from))
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out.dedup();
    out
}

fn alias_text(desc: &str) -> bool {
    let d = desc.to_ascii_lowercase();
    [
        "alias",
        "renamed",
        "moved to",
        "use ",
        "deprecated",
        "superseded",
        "compat",
    ]
    .iter()
    .any(|k| d.contains(k))
}

/// (class, check) from the hint and the manifest.
pub fn classify(name: &str, hint: Option<&str>, f: Option<&PkgFacts>) -> (String, String) {
    let platform = is_platform_binary(name) || f.is_some_and(|f| f.os_cpu);
    let Some(f) = f else {
        return (
            hint.unwrap_or("unknown").to_string(),
            "unverifiable: no published manifest".into(),
        );
    };
    let derived = if platform {
        Some("platform")
    } else if f.deprecated.is_some() {
        Some("alias")
    } else {
        None
    };
    match (hint, derived) {
        (Some("platform"), Some("platform")) => (
            "platform".into(),
            "consistent: os/cpu or platform suffix".into(),
        ),
        (Some("platform"), _) => (
            "platform".into(),
            "conflict: hint platform but no os/cpu or platform suffix".into(),
        ),
        (Some(h), Some("platform")) => (
            "platform".into(),
            format!("conflict: hint {h} but manifest is a platform build"),
        ),
        (Some("alias"), Some("alias")) => ("alias".into(), "consistent: deprecated".into()),
        (Some("alias"), _) => (
            "alias".into(),
            if alias_text(f.description.as_deref().unwrap_or("")) {
                "consistent: description reads as an alias/rename".into()
            } else {
                "unverified: not deprecated, no rename text (migration audit needed)".into()
            },
        ),
        (Some("entry"), _) | (Some("pilot"), _) if !f.has_bin && f.version.is_empty() => (
            hint.unwrap_or_default().into(),
            "unverifiable: no version".into(),
        ),
        (Some(h), Some("alias")) => (h.into(), "note: deprecated on the registry".into()),
        (Some(h), _) => (
            h.into(),
            "hint kept (not contradicted by the manifest)".into(),
        ),
        (None, Some(d)) => (d.into(), "derived from manifest (no hint)".into()),
        (None, None) => (
            "discovery".into(),
            "no hint: ownership/discovery review".into(),
        ),
    }
}

pub fn disposition(
    tax: &Taxonomy,
    name: &str,
    class: &str,
    f: Option<&PkgFacts>,
    repo: Option<&str>,
) -> String {
    if f.is_none() {
        return "blocker: no published manifest (404 or unreadable)".into();
    }
    if repo.is_some_and(|r| tax.hands_off_repo(r)) || tax.hands_off_package(name) {
        return format!("handed off: owned by the {} lane", repo.unwrap_or("owning"));
    }
    if f.is_some_and(|f| f.deprecated.is_some()) {
        return "deprecated: leave (no republish)".into();
    }
    match class {
        "platform" => "shared metadata via the source template; no hand publish".into(),
        "alias" => "migration audit; never deprecate on downloads or name alone".into(),
        "pilot" => "pilot: tested quickstart, measured clean install, animated flow".into(),
        "entry" => "wave 1: metadata + README contract on the next normal release".into(),
        "adapter-example" => "runnable path from the flagship + metadata on next release".into(),
        "feature" => "feature-fit: one concrete integration or a documented no-fit".into(),
        _ => "ownership review: confirm maintainer + canonical repo before editing".into(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn row(
    tax: &Taxonomy,
    name: &str,
    in_inventory: bool,
    pkg: Option<&crate::model::Package>,
    f: Option<&PkgFacts>,
    hint: Option<&str>,
    downloads: Option<u64>,
    owner: &str,
) -> Row {
    let repo = f
        .and_then(|f| f.repository_url.as_deref())
        .and_then(repo_slug)
        .or(pkg.and_then(|p| p.repo_slug.clone()));
    let (class, class_check) = classify(name, hint, f);
    let disposition = disposition(tax, name, &class, f, repo.as_deref());
    let homepage = f.and_then(|f| f.homepage.clone());
    Row {
        name: name.to_string(),
        in_inventory,
        in_graph: pkg.is_some(),
        version: f
            .map(|f| f.version.clone())
            .filter(|v| !v.is_empty())
            .into(),
        homepage_is_ruvio: homepage
            .as_deref()
            .is_some_and(|h| super::audit::same_url(h, &tax.meta.homepage)),
        homepage: homepage.into(),
        keywords: f.map(|f| f.keywords.clone()).unwrap_or_default(),
        readme_present: f
            .and_then(|f| f.readme.as_deref())
            .map(|r| !r.trim().is_empty() && !r.contains("No README data found"))
            .into(),
        dependency_edges: pkg
            .map(|p| p.all_deps().cloned().collect())
            .unwrap_or_default(),
        unpacked_size_bytes: f.and_then(|f| f.unpacked_size).into(),
        deprecated: f.map(|f| f.deprecated.is_some()).into(),
        monthly_downloads: downloads.into(),
        maintainer_ruvnet: f
            .filter(|f| !f.maintainers.is_empty())
            .map(|f| f.maintainers.iter().any(|m| m.eq_ignore_ascii_case(owner)))
            .into(),
        directory: f.and_then(|f| f.repository_directory.clone()).into(),
        repo: repo.into(),
        class_hint: hint.map(String::from).into(),
        class,
        class_check,
        disposition,
    }
}
