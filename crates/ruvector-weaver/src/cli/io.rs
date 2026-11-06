//! JSON artifact I/O and loaders shared by the handlers.

use super::Ctx;
use anyhow::{Context, Result};
use ruvector_weaver::config::EMBEDDED_LEDGER;
use ruvector_weaver::ledger::Ledger;
use ruvector_weaver::model::{GateReport, Queue};
use ruvector_weaver::util::{clip, slug_key};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub fn read_json<T: serde::de::DeserializeOwned + Default>(p: &Path) -> Result<T> {
    if !p.exists() {
        return Ok(T::default());
    }
    serde_json::from_str(&std::fs::read_to_string(p)?)
        .with_context(|| format!("parsing {}", p.display()))
}

pub fn write_json<T: serde::Serialize>(p: &Path, v: &T) -> Result<()> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(p, serde_json::to_string_pretty(v)?)?;
    Ok(())
}

pub fn load_ledger(ctx: &Ctx) -> Result<Ledger> {
    match ctx.cfg.live_ledger()? {
        Some(p) => Ledger::load(&p),
        None => Ledger::parse(EMBEDDED_LEDGER),
    }
}

pub fn gate_path(work: &Path, pkg: &str) -> PathBuf {
    work.join("gates").join(format!("{}.json", slug_key(pkg)))
}

pub fn load_gates(work: &Path) -> Result<BTreeMap<String, GateReport>> {
    let mut out = BTreeMap::new();
    let dir = work.join("gates");
    if dir.exists() {
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            if p.extension().and_then(|x| x.to_str()) == Some("json") {
                let g: GateReport = read_json(&p)?;
                out.insert(g.package.clone(), g);
            }
        }
    }
    Ok(out)
}

pub fn print_queue(q: &Queue, show: usize) {
    println!(
        "queue: {} items, {} lanes (max {}), {} suppressed, {} busy repos",
        q.items.len(),
        q.lanes.len(),
        q.max_lanes,
        q.suppressed.len(),
        q.busy_repos.len()
    );
    for (i, it) in q.items.iter().take(show).enumerate() {
        let ev = it
            .evidence
            .first()
            .map(|s| {
                let at = if s.line == 0 {
                    s.file.clone()
                } else {
                    format!("{}:{}", s.file, s.line)
                };
                format!("{at} `{}`", clip(&s.evidence, 70))
            })
            .unwrap_or_default();
        println!(
            "{:>3}. [{:<8}] {:>6.2} {:<15} {:<22} {:<30} {}",
            i + 1,
            format!("{:?}", it.status),
            it.score,
            format!("{:?}", it.kind),
            it.repo,
            clip(
                it.candidate
                    .as_deref()
                    .or(it.subject.as_deref())
                    .unwrap_or("-"),
                30
            ),
            ev
        );
    }
}
