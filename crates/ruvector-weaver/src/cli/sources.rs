//! Signal-source handlers that talk to GitHub / the registry: open PRs and
//! the release-regression sweep.

use super::cmds::{entry, ledger_path};
use super::io::{load_ledger, read_json, write_json};
use super::Ctx;
use anyhow::{bail, Result};
use ruvector_weaver::ledger::Ledger;
use ruvector_weaver::model::{Ecosystem, Graph, Outcome, ScanReport, SignalKind, WorkKind};
use ruvector_weaver::prs::{self, PrClass, RepoPrs};
use ruvector_weaver::release;
use ruvector_weaver::util::{now, par_map, slug_key};

pub fn prs(ctx: &Ctx, mut repos: Vec<String>, all: bool, checks: bool, seed: bool) -> Result<()> {
    if all {
        let g: Graph = read_json(&ctx.graph_path())?;
        repos.extend(g.consumer_repos.iter().map(|r| r.slug.clone()));
    }
    if repos.is_empty() {
        bail!("give repos (owner/name) or --all");
    }
    let ledger = load_ledger(ctx)?;
    for r in repos {
        let rp: RepoPrs = match prs::fetch(&r, checks, &ctx.work) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("prs {r}: {e:#}");
                continue;
            }
        };
        write_json(
            &ctx.work
                .join("prs")
                .join(format!("{}.json", slug_key(&rp.repo))),
            &rp,
        )?;
        let count = |c: PrClass| rp.prs.iter().filter(|p| p.class == c).count();
        println!(
            "prs {}: {} open ({} land, {} close-with-ledger, {} other)",
            rp.repo,
            rp.prs.len(),
            count(PrClass::Land),
            count(PrClass::CloseWithLedger),
            count(PrClass::Other)
        );
        if seed {
            let path = ledger_path(ctx)?;
            let mut added = 0;
            for p in rp.prs.iter().filter(|p| p.class != PrClass::Other) {
                let subj = prs::subject(p.number);
                let known = ledger.entries.iter().any(|e| {
                    e.repo.eq_ignore_ascii_case(&rp.repo)
                        && e.subject.as_deref() == Some(subj.as_str())
                });
                if known {
                    continue;
                }
                let kind = if p.class == PrClass::Land {
                    WorkKind::Land
                } else {
                    WorkKind::CloseWithLedger
                };
                let mut e = entry(&rp.repo, None, Some(subj), kind, Outcome::Queued);
                e.signal_kind = Some(SignalKind::OpenPr);
                e.pr = Some(p.url.clone());
                e.reason = Some(p.title.clone());
                Ledger::append(&path, &e)?;
                added += 1;
            }
            println!("  ledger: seeded {added} queued PR entries");
        }
    }
    Ok(())
}

pub fn release(ctx: &Ctx, packages: Vec<String>, workers: usize) -> Result<()> {
    let g: Graph = read_json(&ctx.graph_path())?;
    let names: Vec<String> = if packages.is_empty() {
        g.packages
            .iter()
            .filter(|p| p.ecosystem == Ecosystem::Npm && !p.platform_binary)
            .map(|p| p.name.clone())
            .collect()
    } else {
        packages
    };
    let found = par_map(&names, workers, |n| {
        let fallback = g
            .get(n)
            .and_then(|p| p.repo_slug.clone())
            .unwrap_or_else(|| "ruvnet/ruvector".into());
        match release::registry_checks(&ctx.http, n) {
            Ok(Some((prev, _, res))) => release::signals(n, &prev, &res, &fallback),
            _ => vec![],
        }
    });
    let mut signals: Vec<_> = found.into_iter().flatten().collect();
    let patched = release::patched_versions(&load_ledger(ctx)?);
    signals.extend(release::range_signals(&g, &patched, &ctx.cfg.policy.npm));
    for p in g
        .packages
        .iter()
        .filter(|p| !p.platform_binary && !p.open_ranges.is_empty())
    {
        let repo = p
            .repo_slug
            .clone()
            .unwrap_or_else(|| "ruvnet/ruvector".into());
        for r in &p.open_ranges {
            let dep = r
                .rsplit_once('@')
                .map(|(d, _)| d)
                .filter(|d| !d.is_empty())
                .unwrap_or(r);
            signals.push(ruvector_weaver::model::Signal {
                kind: SignalKind::ReleaseRegression,
                repo: repo.clone(),
                file: format!("https://www.npmjs.com/package/{}", p.name),
                line: 0,
                evidence: format!("{}@{} publishes open range {r}", p.name, p.version),
                subject: Some(release::subject(
                    "open_range",
                    &format!("{}->{dep}", p.name),
                )),
                keyword: Some("open_range".into()),
                candidate: Some(p.name.clone()),
                confidence: 0.8,
                severity: Some("high".into()),
                occurrences: 1,
            });
        }
    }
    for s in &signals {
        println!(
            "  [{}] {:<22} {}",
            s.keyword.as_deref().unwrap_or("-"),
            s.repo,
            s.evidence
        );
    }
    let rep = ScanReport {
        repo: "_release".into(),
        scanned_at: now(),
        commit: None,
        files_scanned: names.len(),
        signals,
        notes: vec![],
    };
    write_json(&ctx.work.join("scan").join("_release.json"), &rep)?;
    println!(
        "release: {} packages checked, {} findings -> scan/_release.json",
        names.len(),
        rep.signals.len()
    );
    Ok(())
}

pub fn vuln_evidence(package: &str, dir: &std::path::Path, affected: &str) -> Result<()> {
    let o = ruvector_weaver::util::run_timeout(
        "npm",
        &["ls", package, "--all", "--json"],
        dir,
        std::time::Duration::from_secs(180),
    )?;
    // `npm ls` exits non-zero on extraneous/invalid trees but still prints JSON.
    let tree: serde_json::Value = serde_json::from_str(&o.stdout).map_err(|e| {
        anyhow::anyhow!(
            "npm ls produced no JSON ({e}); is {} an installed project?",
            dir.display()
        )
    })?;
    let copies = ruvector_weaver::vuln::resolved_copies(&tree, package, affected);
    for c in &copies {
        println!(
            "  [{}] {} {}",
            if c.affected { "AFFECTED" } else { "ok" },
            c.path,
            c.version
        );
    }
    let (bad, why) = ruvector_weaver::vuln::verdict(package, &copies);
    println!("vuln-evidence {package} ({affected}): {why}");
    if !bad {
        std::process::exit(2);
    }
    Ok(())
}
