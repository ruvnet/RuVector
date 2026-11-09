//! `weaver seo …` handlers: gather facts (registry, gh, local checkout),
//! then call the pure `ruvector_weaver::seo` functions.

use super::io::{load_gates, read_json, write_json};
use super::Ctx;
use anyhow::{Context, Result};
use clap::Subcommand;
use ruvector_weaver::config::EMBEDDED_TOPICS;
use ruvector_weaver::model::{Ecosystem, Graph, Package};
use ruvector_weaver::seo::audit::{PkgFacts, RepoFacts};
use ruvector_weaver::seo::facts;
use ruvector_weaver::seo::propose::{self, SourcePkg};
use ruvector_weaver::seo::taxonomy::Taxonomy;
use ruvector_weaver::util::{now, par_map, slug_key};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Subcommand)]
pub enum SeoOp {
    /// Measure topic demand (npm keyword total, top-5 downloads, GitHub topic repos).
    Demand {
        topics: Vec<String>,
        /// Re-measure every topic in topics.toml.
        #[arg(long)]
        taxonomy: bool,
        /// Record each measurement in the ledger (subject seo:topic:<t>).
        #[arg(long)]
        record: bool,
    },
    /// Metadata audit of npm packages and their repos (JSON + ranked table).
    Audit {
        packages: Vec<String>,
        #[arg(long)]
        all: bool,
        /// Smoke-import stale (>stale_days) packages under 1k downloads/mo.
        #[arg(long)]
        smoke_stale: bool,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Rows in the printed table.
        #[arg(long, default_value_t = 30)]
        top: usize,
    },
    /// Per-package portfolio baseline over the owner's pinned inventory.
    Baseline {
        #[arg(long)]
        inventory: PathBuf,
        /// Owner distribution-plan markdown (class hints).
        #[arg(long)]
        plan: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Exact diff for package.json / README / repo topics.
    Propose {
        packages: Vec<String>,
        #[arg(long)]
        repo: Option<String>,
        #[arg(long)]
        all: bool,
        /// Local checkout of the repo (default: weaver's scan clone).
        #[arg(long)]
        checkout: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Apply: repo metadata now (gh), package changes as a branch + PR.
    Apply {
        #[arg(long)]
        repo: String,
        /// Apply repo-level homepage + topics.
        #[arg(long)]
        repo_level: bool,
        /// Packages for the metadata PR (comma list).
        #[arg(long, value_delimiter = ',')]
        packages: Vec<String>,
        /// Or: the top N packages of the repo by downloads.
        #[arg(long, default_value_t = 0)]
        top: usize,
        #[arg(long)]
        checkout: Option<PathBuf>,
        #[arg(long)]
        dry_run: bool,
    },
}

pub fn taxonomy(config_dir: Option<&Path>) -> Result<Taxonomy> {
    let dir = config_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("config/weaver"));
    let p = dir.join("topics.toml");
    let text = if p.exists() {
        std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?
    } else {
        EMBEDDED_TOPICS.to_string()
    };
    Taxonomy::parse(&text)
}

pub fn run(ctx: &Ctx, config_dir: Option<&Path>, op: SeoOp) -> Result<()> {
    let tax = taxonomy(config_dir)?;
    match op {
        SeoOp::Demand {
            topics,
            taxonomy,
            record,
        } => demand(ctx, &tax, topics, taxonomy, record),
        SeoOp::Audit {
            packages,
            all,
            smoke_stale,
            out,
            top,
        } => super::seo_report::audit_cmd(ctx, &tax, packages, all, smoke_stale, out, top),
        SeoOp::Baseline {
            inventory,
            plan,
            out,
        } => super::seo_report::baseline_cmd(ctx, &tax, &inventory, plan.as_deref(), out),
        SeoOp::Propose {
            packages,
            repo,
            all,
            checkout,
            json,
        } => propose_cmd(ctx, &tax, packages, repo, all, checkout, json),
        SeoOp::Apply {
            repo,
            repo_level,
            packages,
            top,
            checkout,
            dry_run,
        } => super::seo_apply::apply(
            ctx, &tax, &repo, repo_level, packages, top, checkout, dry_run,
        ),
    }
}

fn demand(
    ctx: &Ctx,
    tax: &Taxonomy,
    mut topics: Vec<String>,
    all: bool,
    record: bool,
) -> Result<()> {
    if all {
        topics.extend(
            tax.topics
                .iter()
                .filter(|t| t.kind == ruvector_weaver::seo::taxonomy::TopicKind::Topic)
                .map(|t| t.slug.clone()),
        );
    }
    let mut out = Vec::new();
    for t in topics {
        let d = facts::measure(&ctx.http, &t);
        println!(
            "{:<32} npm_kw_total={:<8} npm_top5_monthly={:<12} gh_topic_repos={}",
            d.topic,
            fmt(d.npm_kw_total),
            fmt(d.npm_top5_monthly),
            fmt(d.gh_topic_repos)
        );
        if record {
            let mut e = super::cmds::entry(
                "ruvnet/ruvector",
                None,
                Some(format!("seo:topic:{}", d.topic)),
                ruvector_weaver::model::WorkKind::Hygiene,
                ruvector_weaver::model::Outcome::Gated,
            );
            e.reason = Some(format!("topic demand measured {}", now()));
            for (k, v) in [
                ("npm_kw_total", d.npm_kw_total),
                ("npm_top5_monthly", d.npm_top5_monthly),
                ("gh_topic_repos", d.gh_topic_repos),
            ] {
                if let Some(v) = v {
                    e.measurements.push(ruvector_weaver::model::Measurement {
                        metric: k.into(),
                        value: v as f64,
                        unit: "count".into(),
                        baseline: None,
                    });
                }
            }
            ruvector_weaver::ledger::Ledger::append(&super::cmds::ledger_path(ctx)?, &e)?;
        }
        out.push(d);
    }
    write_json(&ctx.work.join("seo").join("demand.json"), &out)
}

fn fmt(v: Option<u64>) -> String {
    v.map_or("unknown".into(), |x| x.to_string())
}

pub fn npm_packages(g: &Graph) -> Vec<&Package> {
    g.packages
        .iter()
        .filter(|p| p.ecosystem == Ecosystem::Npm && !p.platform_binary)
        .collect()
}

/// Facts for `names` (cached packuments), with import status from gate
/// reports and the seo smoke cache.
pub fn gather(ctx: &Ctx, names: &[String]) -> BTreeMap<String, PkgFacts> {
    let gates = load_gates(&ctx.work).unwrap_or_default();
    let smoke = load_smoke(&ctx.work);
    let got = par_map(names, 8, |n| {
        facts::fetch_packument(&ctx.http, n)
            .ok()
            .flatten()
            .map(|doc| facts::from_packument(n, &doc))
    });
    let mut out = BTreeMap::new();
    for (n, f) in names.iter().zip(got) {
        if let Some(mut f) = f {
            f.import_ok = smoke.get(n).copied().or_else(|| {
                gates.get(n).and_then(|g| {
                    g.results
                        .iter()
                        .find(|r| r.name == "smoke_import")
                        .map(|r| r.passed)
                })
            });
            out.insert(n.clone(), f);
        }
    }
    out
}

pub fn smoke_path(work: &Path) -> PathBuf {
    work.join("seo").join("smoke.json")
}

pub fn load_smoke(work: &Path) -> BTreeMap<String, bool> {
    read_json(&smoke_path(work)).unwrap_or_default()
}

pub fn repo_facts(slugs: &[String]) -> BTreeMap<String, RepoFacts> {
    let got = par_map(slugs, 4, |s| facts::fetch_repo(s));
    slugs.iter().cloned().zip(got).collect()
}

/// package name -> repo-relative dir, for every package.json in a checkout.
pub fn index_checkout(root: &Path) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let skip = [
        "node_modules",
        ".git",
        "target",
        "dist",
        "build",
        ".next",
        "vendor",
    ];
    for e in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !skip.contains(&e.file_name().to_str().unwrap_or("")))
        .flatten()
    {
        if e.file_name() != "package.json" {
            continue;
        }
        let Ok(t) = std::fs::read_to_string(e.path()) else {
            continue;
        };
        let Some(name) = serde_json::from_str::<serde_json::Value>(&t)
            .ok()
            .and_then(|v| v["name"].as_str().map(String::from))
        else {
            continue;
        };
        let rel = e
            .path()
            .parent()
            .and_then(|d| d.strip_prefix(root).ok())
            .map(|d| d.to_string_lossy().to_string())
            .unwrap_or_default();
        out.entry(name).or_default().push(rel);
    }
    out
}

pub fn source_for(
    root: &Path,
    idx: &BTreeMap<String, Vec<String>>,
    name: &str,
    dir_hint: Option<&str>,
) -> Option<SourcePkg> {
    let dirs = idx.get(name)?;
    let rel = dirs
        .iter()
        .find(|d| Some(d.as_str()) == dir_hint)
        .or_else(|| dirs.iter().min_by_key(|d| d.len()))?
        .clone();
    let dir = root.join(&rel);
    let package_json = std::fs::read_to_string(dir.join("package.json")).ok()?;
    let readme = ["README.md", "readme.md", "Readme.md"]
        .iter()
        .find_map(|f| {
            std::fs::read_to_string(dir.join(f))
                .ok()
                .map(|t| (f.to_string(), t))
        });
    Some(SourcePkg {
        rel_dir: rel,
        package_json,
        readme,
    })
}

fn propose_cmd(
    ctx: &Ctx,
    tax: &Taxonomy,
    mut packages: Vec<String>,
    repo: Option<String>,
    all: bool,
    checkout: Option<PathBuf>,
    json: bool,
) -> Result<()> {
    let g: Graph = read_json(&ctx.graph_path())?;
    if let Some(r) = &repo {
        let rf = facts::fetch_repo(r);
        let rp = propose::propose_repo(tax, &g, &rf);
        println!(
            "repo {}: homepage {:?} -> {}",
            rp.slug, rp.homepage_old, rp.homepage_new
        );
        println!(
            "  topics [{}] -> [{}]",
            rp.topics_old.join(", "),
            rp.topics_new.join(", ")
        );
        if packages.is_empty() && !all {
            if let Some(cr) = g.repo(r) {
                packages = cr.packages.clone();
            }
        }
    }
    if all {
        packages = npm_packages(&g).iter().map(|p| p.name.clone()).collect();
    }
    let facts = gather(ctx, &packages);
    let mut props = Vec::new();
    let mut idx_cache: BTreeMap<String, (PathBuf, BTreeMap<String, Vec<String>>)> = BTreeMap::new();
    for name in &packages {
        let Some(p) = g.get(name) else { continue };
        let f = facts.get(name).cloned().unwrap_or_default();
        let src = p.repo_slug.as_ref().and_then(|s| {
            let (root, idx) = idx_cache.entry(s.clone()).or_insert_with(|| {
                let root = checkout
                    .clone()
                    .unwrap_or_else(|| ctx.work.join("repos").join(slug_key(s)));
                let idx = index_checkout(&root);
                (root, idx)
            });
            source_for(root, idx, name, f.repository_directory.as_deref())
        });
        let rf = p.repo_slug.as_ref().map(|s| RepoFacts {
            slug: s.clone(),
            default_branch: "main".into(),
            ..RepoFacts::default()
        });
        match propose::propose_package(tax, &g, p, &f, src.as_ref(), rf.as_ref()) {
            Ok(pp) => {
                if !json {
                    println!("== {} ({})", pp.name, pp.repo.as_deref().unwrap_or("-"));
                    println!("  description: {}", pp.description);
                    println!(
                        "  keywords ({}): {}",
                        pp.keywords.len(),
                        pp.keywords.join(", ")
                    );
                    for n in &pp.notes {
                        println!("  note: {n}");
                    }
                    print!("{}", pp.diff);
                }
                props.push(pp);
            }
            Err(e) => eprintln!("propose {name}: {e:#}"),
        }
    }
    let path = ctx.work.join("seo").join("proposals.json");
    write_json(&path, &props)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&props)?);
    }
    eprintln!(
        "seo propose: {} proposals -> {}",
        props.len(),
        path.display()
    );
    Ok(())
}
