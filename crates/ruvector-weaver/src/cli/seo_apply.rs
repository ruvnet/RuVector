//! `weaver seo apply --repo r`: repo metadata via `gh repo edit` (old values
//! recorded in the ledger BEFORE the edit), package metadata as a branch +
//! PR with patch bumps. Never publishes.

use super::cmds::{entry, ledger_path};
use super::io::read_json;
use super::seo::{gather, index_checkout, source_for};
use super::Ctx;
use anyhow::{bail, Context, Result};
use ruvector_weaver::ledger::Ledger;
use ruvector_weaver::model::{Graph, Measurement, Outcome, WorkKind};
use ruvector_weaver::seo::taxonomy::Taxonomy;
use ruvector_weaver::seo::{apply as ap, facts, propose};
use ruvector_weaver::util::run_timeout;
use std::path::{Path, PathBuf};
use std::time::Duration;

const ATTR_COMMIT: &str = "Co-Authored-By: claude-flow <ruv@ruv.net>";
const ATTR_PR: &str = "🤖 Generated with [claude-flow](https://github.com/ruvnet/claude-flow)";

fn sh(cmd: &str, args: &[&str], cwd: &Path) -> Result<String> {
    let o = run_timeout(cmd, args, cwd, Duration::from_secs(600))?;
    if o.timed_out || o.status != Some(0) {
        bail!("{cmd} {}: {}", args.join(" "), o.stderr.trim());
    }
    Ok(o.stdout)
}

fn m(metric: &str, v: usize) -> Measurement {
    Measurement {
        metric: metric.into(),
        value: v as f64,
        unit: "count".into(),
        baseline: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn apply(
    ctx: &Ctx,
    tax: &Taxonomy,
    repo: &str,
    repo_level: bool,
    packages: Vec<String>,
    top: usize,
    checkout: Option<PathBuf>,
    dry_run: bool,
) -> Result<()> {
    let g: Graph = read_json(&ctx.graph_path())?;
    let slug = g.canon(repo);
    if tax.hands_off_repo(&slug) {
        bail!("{slug} is owned by another lane (topics.toml hands_off_repos): propose only");
    }
    let rf = facts::fetch_repo(&slug);
    if rf.missing {
        bail!("gh cannot read {slug}");
    }
    if repo_level {
        if !rf
            .owner
            .eq_ignore_ascii_case(&ctx.cfg.policy.npm.maintainer)
        {
            bail!(
                "{slug} is owned by {}, not {}",
                rf.owner,
                ctx.cfg.policy.npm.maintainer
            );
        }
        if rf.archived {
            bail!("{slug} is archived");
        }
        let rp = propose::propose_repo(tax, &g, &rf);
        if rp.topics_new.is_empty() {
            bail!("{slug}: no taxonomy topics derivable; add a [[repo]] entry");
        }
        println!("{}", ap::repo_reason(&rp));
        let meets = ruvector_weaver::seo::audit::audit_repo(tax, &g, &rf, &rf.owner)
            .issues
            .is_empty();
        if ap::repo_noop(&rp) || meets {
            println!("{slug}: already meets ADR-356 (homepage + taxonomy topics); skipped");
        } else if !dry_run {
            let mut e = entry(
                &slug,
                None,
                Some("seo:repo-settings".to_string()),
                WorkKind::Hygiene,
                Outcome::Gated,
            );
            e.reason = Some(ap::repo_reason(&rp));
            e.measurements = vec![
                m("topics_before", rp.topics_old.len()),
                m("topics_after", rp.topics_new.len()),
            ];
            let lp = ledger_path(ctx)?;
            Ledger::append(&lp, &e)?;
            let args = ap::repo_edit_args(&rp);
            let argv: Vec<&str> = args.iter().map(String::as_str).collect();
            let res = sh("gh", &argv, Path::new("."));
            let after = facts::fetch_repo(&slug);
            let ok = res.is_ok()
                && after
                    .homepage
                    .as_deref()
                    .is_some_and(|h| ruvector_weaver::seo::audit::same_url(h, &tax.meta.homepage))
                && {
                    let mut a = after.topics.clone();
                    let mut b = rp.topics_new.clone();
                    a.sort();
                    b.sort();
                    a == b
                };
            e.outcome = if ok {
                Outcome::Shipped
            } else {
                Outcome::Failed
            };
            e.ts = ruvector_weaver::util::now();
            if let Err(err) = &res {
                e.reason = Some(format!("{} | gh error: {err:#}", ap::repo_reason(&rp)));
            }
            Ledger::append(&lp, &e)?;
            println!(
                "{slug}: repo metadata {}",
                if ok {
                    "applied + read back"
                } else {
                    "FAILED readback"
                }
            );
        }
    }
    let mut names = packages;
    if names.is_empty() && top > 0 {
        let mut ps: Vec<_> = g
            .repo(&slug)
            .map(|r| r.packages.iter().filter_map(|n| g.get(n)).collect())
            .unwrap_or_default();
        ps.sort_by_key(|p| std::cmp::Reverse(p.monthly_downloads));
        names = ps
            .into_iter()
            .filter(|p| !tax.hands_off_package(&p.name))
            .take(top)
            .map(|p| p.name.clone())
            .collect();
    }
    if names.is_empty() {
        return Ok(());
    }
    for n in &names {
        if tax.hands_off_package(n) {
            bail!("{n} is owned by another lane (hands_off_packages)");
        }
    }
    package_pr(ctx, tax, &g, &slug, &rf, &names, checkout, dry_run)
}

#[allow(clippy::too_many_arguments)]
fn package_pr(
    ctx: &Ctx,
    tax: &Taxonomy,
    g: &Graph,
    slug: &str,
    rf: &ruvector_weaver::seo::audit::RepoFacts,
    names: &[String],
    checkout: Option<PathBuf>,
    dry_run: bool,
) -> Result<()> {
    let tmp;
    let root = match checkout {
        Some(c) => c,
        None => {
            let base = ctx.work.join("seo-apply");
            std::fs::create_dir_all(&base)?;
            tmp = tempfile::Builder::new()
                .prefix("apply-")
                .tempdir_in(&base)?;
            let dest = tmp.path().join("repo");
            sh(
                "gh",
                &[
                    "repo",
                    "clone",
                    slug,
                    &dest.to_string_lossy(),
                    "--",
                    "--depth",
                    "1",
                ],
                Path::new("."),
            )?;
            dest
        }
    };
    let facts = gather(ctx, names);
    let idx = index_checkout(&root);
    let mut written: Vec<String> = Vec::new();
    let mut lines = Vec::new();
    for n in names {
        let p = g.get(n).with_context(|| format!("{n} not in graph"))?;
        let f = facts.get(n).cloned().unwrap_or_default();
        let src = source_for(&root, &idx, n, f.repository_directory.as_deref())
            .with_context(|| format!("{n}: package.json not found in {}", root.display()))?;
        let pp = propose::propose_package(tax, g, p, &f, Some(&src), Some(rf))?;
        let cur = ruvector_weaver::seo::jsontext::get(&src.package_json, "version")
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default();
        let ver = propose::bump_patch(&cur, &f.version).context("unparseable version")?;
        print!("{}", pp.diff);
        for (path, text) in ap::with_version(&pp, &ver)? {
            let full = root.join(&path);
            if !dry_run {
                if let Some(d) = full.parent() {
                    std::fs::create_dir_all(d)?;
                }
                std::fs::write(&full, text)?;
            }
            written.push(path);
        }
        lines.push(format!("- `{n}` {} -> {ver}: homepage, {} keywords, description, repository/bugs, Works with / Used by, ruv.io footer", f.version, pp.keywords.len()));
    }
    if dry_run {
        println!("dry run: {} files would change", written.len());
        return Ok(());
    }
    let date = chrono::Utc::now().format("%Y%m%d").to_string();
    let branch = ap::branch_for(slug, &date);
    sh("git", &["checkout", "-b", &branch], &root)?;
    let mut add = vec!["add", "--"];
    add.extend(written.iter().map(String::as_str));
    sh("git", &add, &root)?;
    let title = format!(
        "SEO metadata: homepage https://ruv.io, keywords, cross-links ({} packages, ADR-356)",
        names.len()
    );
    let body = format!(
        "Portfolio distribution layer (ADR-356, ruvnet/RuVector). Metadata only; no dependency added.\n\n{}\n\nCross-links come from real dependency edges in the weaver graph. Publish through this repo's normal release path after merge.\n\n{ATTR_PR}\n",
        lines.join("\n")
    );
    let msg = format!("{title}\n\n{}\n\n{ATTR_COMMIT}\n", lines.join("\n"));
    sh("git", &["commit", "-m", &msg], &root)?;
    sh("git", &["push", "-u", "origin", &branch], &root)?;
    let url = sh(
        "gh",
        &[
            "pr", "create", "--repo", slug, "--head", &branch, "--title", &title, "--body", &body,
        ],
        &root,
    )?;
    let url = url.trim().to_string();
    println!("{slug}: PR {url}");
    for n in names {
        let mut e = entry(
            slug,
            Some(n.clone()),
            Some(format!("seo:{n}")),
            WorkKind::Hygiene,
            Outcome::Gated,
        );
        e.pr = Some(url.clone());
        e.reason =
            Some("seo metadata PR opened; publish via the repo release path after merge".into());
        Ledger::append(&ledger_path(ctx)?, &e)?;
    }
    Ok(())
}
