//! `weaver propose-new`: capability gaps -> `create` work items in
//! queue.json (autopilot, user decision 2026-11-05). A gap is a 404 import
//! of an ecosystem-looking name that no capability-table entry covers. The
//! pre-dispatch gates enforced here, each recorded on the item:
//!   1. gap_evidence  - file:line evidence in at least one consumer repo,
//!      and the same lane wires the package into that consumer;
//!   2. no_duplicate  - no capability entry and no ruvnet registry hit
//!      already covers it (no near-duplicates / renames);
//!   5. rate_limit    - optional per-tick / per-day caps from weaver.toml.
//!
//! Gates 3 (real implementation, tests, tarball, leaks, post-publish smoke)
//! and 4 (README via `weaver readme`) run in `weaver gate --new` / `weaver
//! readme --lint` inside the lane, and are recorded to the ledger there.

use crate::config::Config;
use crate::ledger::{Ledger, PairState};
use crate::model::{
    GateResult, Graph, LaneStatus, ScanReport, Signal, Suppressed, WorkItem, WorkKind,
};
use crate::queue::{item_id, signal_weights};
use crate::registry::{Registry, SearchHit};
use std::collections::BTreeMap;

fn res(name: &str, passed: bool, detail: impl Into<String>) -> GateResult {
    GateResult {
        name: name.into(),
        passed,
        detail: detail.into(),
    }
}

/// Significant tokens of a package name (`@ruvector/ast-diff` -> [ast, diff]).
pub fn name_tokens(name: &str) -> Vec<String> {
    let bare = name.rsplit('/').next().unwrap_or(name);
    bare.split(['-', '_', '.'])
        .filter(|t| {
            t.len() >= 2
                && !matches!(
                    *t,
                    "wasm" | "node" | "napi" | "ruvector" | "js" | "core" | "sdk" | "lib"
                )
        })
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

/// A registry hit counts as an existing implementation when its name
/// carries a gap token as a whole name segment.
pub fn duplicate_of<'a>(gap: &str, hits: &'a [SearchHit]) -> Option<&'a SearchHit> {
    let toks = name_tokens(gap);
    if toks.is_empty() {
        return None;
    }
    hits.iter().find(|h| {
        h.name != gap && {
            let ht = name_tokens(&h.name);
            toks.iter().all(|t| ht.contains(t))
        }
    })
}

pub struct Proposals {
    pub items: Vec<WorkItem>,
    pub suppressed: Vec<Suppressed>,
}

pub fn propose(
    cfg: &Config,
    graph: &Graph,
    reports: &[ScanReport],
    ledger: &Ledger,
    reg: &dyn Registry,
) -> Proposals {
    let lp = &cfg.policy.lp;
    let mut gaps: BTreeMap<(String, String), Vec<Signal>> = BTreeMap::new();
    for s in reports.iter().flat_map(|r| &r.signals) {
        if s.kind == crate::model::SignalKind::MissingPackage && s.candidate.is_none() {
            if let Some(name) = &s.subject {
                gaps.entry((graph.canon(&s.repo), name.clone()))
                    .or_default()
                    .push(s.clone());
            }
        }
    }
    let mut canon = ledger.clone();
    for e in &mut canon.entries {
        e.repo = graph.canon(&e.repo);
    }
    let ledger = &canon;
    let day_used = ledger.new_packages_since(chrono::Utc::now() - chrono::Duration::hours(24));
    let weight = signal_weights(ledger)[&crate::model::SignalKind::MissingPackage];
    let mut out = Proposals {
        items: vec![],
        suppressed: vec![],
    };
    let mut emitted = 0usize;
    for ((repo, name), sigs) in gaps {
        let id = item_id(WorkKind::Create, &repo, &name);
        let sup = |reason: String| Suppressed {
            id: id.clone(),
            repo: repo.clone(),
            candidate: Some(name.clone()),
            subject: Some(name.clone()),
            reason,
        };
        if let PairState::Blocked { outcome, reason } = ledger.pair_state(
            &repo,
            Some(&name),
            Some(&name),
            WorkKind::Create,
            Some(crate::model::SignalKind::MissingPackage),
            lp.max_failures,
        ) {
            out.suppressed
                .push(sup(format!("ledger {outcome:?}: {reason}")));
            continue;
        }
        let mut gates = vec![res(
            "gap_evidence",
            !sigs.is_empty(),
            format!(
                "{} consumer reference(s), first {}:{}",
                sigs.len(),
                sigs[0].file,
                sigs[0].line
            ),
        )];
        let query = format!(
            "{} maintainer:{}",
            name_tokens(&name).join(" "),
            cfg.policy.npm.maintainer
        );
        let dup = match reg.search(&query) {
            Ok(hits) => duplicate_of(&name, &hits).map(|h| h.name.clone()),
            Err(e) => {
                out.suppressed
                    .push(sup(format!("no_duplicate check failed (fail closed): {e}")));
                continue;
            }
        };
        if let Some(d) = dup {
            out.suppressed.push(sup(format!(
                "no_duplicate: {d} already covers it; integrate that instead"
            )));
            continue;
        }
        gates.push(res(
            "no_duplicate",
            true,
            "no capability entry and no ruvnet registry match",
        ));
        let tick_ok = lp
            .max_new_packages_per_tick
            .map(|m| emitted < m)
            .unwrap_or(true);
        let day_ok = lp
            .max_new_packages_per_day
            .map(|m| day_used + emitted < m)
            .unwrap_or(true);
        if !(tick_ok && day_ok) {
            out.suppressed.push(sup(
                "rate_limit: new-package cap reached for this tick/day".into()
            ));
            continue;
        }
        gates.push(res(
            "rate_limit",
            true,
            match lp.max_new_packages_per_tick {
                Some(m) => format!("{}/{m} this tick", emitted + 1),
                None => "unlimited (gates only)".into(),
            },
        ));
        emitted += 1;
        let downloads = graph.repo(&repo).map(|r| r.monthly_downloads).unwrap_or(0) as f64;
        let score =
            (10.0 + downloads).ln() * 0.95 * weight * (1.0 + 0.1 * (sigs.len() as f64).ln());
        out.items.push(WorkItem {
            id,
            kind: WorkKind::Create,
            status: LaneStatus::Deferred,
            repo: repo.clone(),
            subject: Some(name.clone()),
            candidate: Some(name.clone()),
            signal_kind: crate::model::SignalKind::MissingPackage,
            score,
            disclosure: "public_pr".into(),
            required_test: format!("Package `{name}` has its own passing tests; the consumer imports it from the registry after publish; `weaver gate {name} --new --dir <pkg>` and `weaver gate {name} --consumer {repo}` pass."),
            evidence: sigs.into_iter().take(5).collect(),
            gates,
            brief: String::new(),
        });
    }
    out
}
