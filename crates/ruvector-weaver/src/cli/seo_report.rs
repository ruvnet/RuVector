//! `weaver seo audit` / `weaver seo baseline` handlers.

use super::io::{read_json, write_json};
use super::seo::{gather, load_smoke, npm_packages, repo_facts, smoke_path};
use super::Ctx;
use anyhow::{bail, Result};
use ruvector_weaver::model::{Ecosystem, Graph, Package};
use ruvector_weaver::seo::audit::{self, AuditReport, PkgFacts};
use ruvector_weaver::seo::baseline;
use ruvector_weaver::seo::taxonomy::Taxonomy;
use ruvector_weaver::util::{now, par_map};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub fn audit_cmd(
    ctx: &Ctx,
    tax: &Taxonomy,
    packages: Vec<String>,
    all: bool,
    smoke_stale: bool,
    out: Option<PathBuf>,
    top: usize,
) -> Result<()> {
    let g: Graph = read_json(&ctx.graph_path())?;
    let pkgs: Vec<&Package> = npm_packages(&g)
        .into_iter()
        .filter(|p| all || packages.contains(&p.name))
        .collect();
    if pkgs.is_empty() {
        bail!("give package names or --all");
    }
    let names: Vec<String> = pkgs.iter().map(|p| p.name.clone()).collect();
    let mut facts = gather(ctx, &names);
    let nowt = chrono::Utc::now();
    if smoke_stale {
        let mut cache = load_smoke(&ctx.work);
        let todo: Vec<(String, String)> = pkgs
            .iter()
            .filter(|p| p.monthly_downloads < 1000)
            .filter(|p| {
                audit::age_days(p.last_publish.as_deref(), nowt).unwrap_or(0) > tax.meta.stale_days
            })
            .filter_map(|p| {
                facts
                    .get(&p.name)
                    .map(|f| (p.name.clone(), f.version.clone()))
            })
            .filter(|(n, _)| facts.get(n).is_some_and(|f| f.import_ok.is_none()))
            .collect();
        eprintln!("seo audit: smoke-importing {} stale packages", todo.len());
        let res = par_map(&todo, 6, |(n, v)| {
            ruvector_weaver::gate::smoke(&ctx.cfg, n, v)
                .map(|r| r.is_ok())
                .unwrap_or(false)
        });
        for ((n, _), ok) in todo.iter().zip(res) {
            cache.insert(n.clone(), ok);
            if let Some(f) = facts.get_mut(n) {
                f.import_ok = Some(ok);
            }
        }
        write_json(&smoke_path(&ctx.work), &cache)?;
    }
    let mut slugs: Vec<String> = pkgs.iter().filter_map(|p| p.repo_slug.clone()).collect();
    slugs.sort();
    slugs.dedup();
    let repos = repo_facts(&slugs);
    let mut report = AuditReport {
        generated_at: now(),
        homepage: tax.meta.homepage.clone(),
        ..AuditReport::default()
    };
    for p in &pkgs {
        let f = facts.get(&p.name).cloned().unwrap_or_else(|| PkgFacts {
            name: p.name.clone(),
            ..PkgFacts::default()
        });
        report
            .packages
            .push(audit::audit_package(tax, &g, p, &f, nowt));
    }
    for f in repos.values() {
        report.repos.push(audit::audit_repo(
            tax,
            &g,
            f,
            &ctx.cfg.policy.npm.maintainer,
        ));
    }
    audit::finish(&mut report);
    let path = out.unwrap_or_else(|| ctx.work.join("seo-audit.json"));
    write_json(&path, &report)?;
    print!("{}", audit::table(&report, top));
    let issues: usize = report.packages.iter().map(|p| p.issues.len()).sum();
    println!(
        "seo audit: {} packages ({} issues), {} repos, {} stale<1k -> {}",
        report.packages.len(),
        issues,
        report.repos.len(),
        report.stale_low_download.len(),
        path.display()
    );
    Ok(())
}

pub fn baseline_cmd(
    ctx: &Ctx,
    tax: &Taxonomy,
    inventory: &Path,
    plan: Option<&Path>,
    out: Option<PathBuf>,
) -> Result<()> {
    let inv =
        baseline::parse_inventory(&serde_json::from_str(&std::fs::read_to_string(inventory)?)?);
    let hints = match plan {
        Some(p) => baseline::parse_class_hints(&std::fs::read_to_string(p)?),
        None => BTreeMap::new(),
    };
    let g: Graph = read_json(&ctx.graph_path())?;
    let mut names: Vec<String> = inv.clone();
    let extra: Vec<String> = g
        .packages
        .iter()
        .filter(|p| p.ecosystem == Ecosystem::Npm && !inv.contains(&p.name))
        .map(|p| p.name.clone())
        .collect();
    names.extend(extra.iter().cloned());
    let facts = gather(ctx, &names);
    let dl = ruvector_weaver::graph::fresh_downloads(&ctx.http, &names, 8);
    let owner = ctx.cfg.policy.npm.maintainer.clone();
    let rows: Vec<baseline::Row> = names
        .iter()
        .map(|n| {
            baseline::row(
                tax,
                n,
                inv.contains(n),
                g.get(n),
                facts.get(n),
                hints.get(n).map(String::as_str),
                dl.get(n).copied(),
                &owner,
            )
        })
        .collect();
    let missing_hint: Vec<&String> = inv.iter().filter(|n| !hints.contains_key(*n)).collect();
    let not_in_graph: Vec<&String> = inv.iter().filter(|n| g.get(n).is_none()).collect();
    let mut by_class: BTreeMap<&str, usize> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.in_inventory) {
        *by_class.entry(r.class.as_str()).or_default() += 1;
    }
    let doc = serde_json::json!({
        "generated_at": now(),
        "inventory": inventory.display().to_string(),
        "inventory_count": inv.len(),
        "graph_only_reconcile": extra,
        "inventory_not_in_graph": not_in_graph,
        "inventory_without_class_hint": missing_hint,
        "class_counts": by_class,
        "rows": rows,
    });
    let path = out.unwrap_or_else(|| ctx.work.join("seo").join("baseline.json"));
    write_json(&path, &doc)?;
    println!(
        "seo baseline: {} inventory + {} graph-only rows, classes {:?} -> {}",
        inv.len(),
        extra.len(),
        by_class,
        path.display()
    );
    Ok(())
}
