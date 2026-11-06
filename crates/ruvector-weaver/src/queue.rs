//! `weaver queue`: rank work items across every scanned repo, dedupe them
//! against the ledger, assign ONE lane per repo (with batching of
//! independent items), and attach a self-contained agent brief to each.

use crate::brief::{brief, kind_str, required_test};
use crate::config::Config;
use crate::ledger::{Ledger, PairState};
use crate::model::{
    GateReport, GateResult, Graph, Lane, LaneStatus, LedgerEntry, Queue, ScanReport, Signal,
    SignalKind, Suppressed, WorkItem, WorkKind,
};
use crate::prs::{self, RepoPrs};
use crate::scan::security::is_upstream_advisory;
use crate::util::now;
use std::collections::{BTreeMap, BTreeSet};

pub fn base_weight(k: SignalKind) -> f64 {
    match k {
        SignalKind::Security => 1.2,
        SignalKind::MissingPackage => 1.0,
        SignalKind::UndeclaredImport => 0.9,
        SignalKind::StubMarker => 0.6,
        SignalKind::UnusedDeclared => 0.5,
        SignalKind::OpenPr => 0.8,
        SignalKind::ReleaseRegression => 1.6,
    }
}

/// Base weight scaled by the ledger hit rate (prior 0.5 -> x1.0).
pub fn signal_weights(ledger: &Ledger) -> BTreeMap<SignalKind, f64> {
    ledger
        .hit_rates()
        .into_iter()
        .map(|(k, r)| (k, base_weight(k) * 2.0 * r))
        .collect()
}

pub fn work_kind(s: &Signal) -> WorkKind {
    match s.kind {
        SignalKind::MissingPackage if s.candidate.is_none() => WorkKind::Create,
        SignalKind::MissingPackage | SignalKind::StubMarker => WorkKind::Replace,
        SignalKind::UndeclaredImport | SignalKind::UnusedDeclared => WorkKind::Hygiene,
        SignalKind::Security => WorkKind::Security,
        SignalKind::OpenPr => WorkKind::Land,
        SignalKind::ReleaseRegression => WorkKind::ReleaseFix,
    }
}

pub fn disclosure(kind: WorkKind, severity: Option<&str>) -> &'static str {
    match (kind, severity) {
        (WorkKind::Security, Some("critical" | "high")) => "private_advisory",
        (WorkKind::Security, None | Some("unknown")) => "triage_then_route",
        _ => "public_pr",
    }
}

pub fn item_id(kind: WorkKind, repo: &str, name: &str) -> String {
    format!("{}::{repo}::{name}", kind_str(kind))
}

/// Turn a queued ledger backlog entry into a work item.
fn backlog_item(e: &LedgerEntry) -> WorkItem {
    let (file, line) = e
        .location
        .as_deref()
        .and_then(|l| {
            l.rsplit_once(':')
                .and_then(|(f, n)| n.parse().ok().map(|n| (f.to_string(), n)))
        })
        .unwrap_or_else(|| (e.location.clone().unwrap_or_else(|| "(audit)".into()), 0));
    let name = e
        .subject
        .clone()
        .or(e.candidate.clone())
        .unwrap_or_default();
    let sig = Signal {
        kind: e.signal_kind.unwrap_or(SignalKind::Security),
        repo: e.repo.clone(),
        file,
        line,
        evidence: e.reason.clone().unwrap_or_default(),
        subject: e.subject.clone(),
        keyword: None,
        candidate: e.candidate.clone(),
        confidence: 0.9,
        severity: e.severity.clone(),
        occurrences: 1,
    };
    WorkItem {
        id: item_id(e.kind, &e.repo, &name),
        kind: e.kind,
        status: LaneStatus::Deferred,
        repo: e.repo.clone(),
        subject: e.subject.clone(),
        candidate: e.candidate.clone(),
        signal_kind: sig.kind,
        score: 0.0,
        disclosure: disclosure(e.kind, e.severity.as_deref()).to_string(),
        required_test: required_test(e.kind, &name),
        evidence: vec![sig],
        gates: vec![],
        brief: String::new(),
    }
}

pub struct QueueInputs<'a> {
    pub graph: &'a Graph,
    pub reports: &'a [ScanReport],
    pub ledger: &'a Ledger,
    pub gates: &'a BTreeMap<String, GateReport>,
    /// Pre-built `create` items (from propose-new), already gate-filtered.
    pub creates: Vec<WorkItem>,
    pub create_suppressed: Vec<Suppressed>,
    /// Live open-PR snapshots (`weaver prs`).
    pub prs: &'a [RepoPrs],
}

/// Does a rejection's reason say the candidate was too heavy to depend on?
pub fn is_heavy_rejection(e: &LedgerEntry) -> bool {
    e.outcome == crate::model::Outcome::Rejected
        && e.reason
            .as_deref()
            .map(|r| {
                let r = r.to_ascii_lowercase();
                r.contains("heavy dep") || r.contains("dependency tree") || r.contains("heavy tree")
            })
            .unwrap_or(false)
}

/// The candidate's current release clears the size/dep gate and is newer
/// than the version that was rejected.
pub fn lean_release(graph: &Graph, cfg: &Config, e: &LedgerEntry) -> Option<String> {
    let p = graph.get(e.candidate.as_deref()?)?;
    let rejected_at = e
        .version
        .as_deref()
        .and_then(|v| v.rsplit('@').next())
        .unwrap_or("");
    let g = &cfg.policy.gate;
    let lean = p.runtime_deps <= g.lean_max_runtime_deps
        && p.unpacked_size > 0
        && p.unpacked_size <= g.lean_max_unpacked_bytes;
    (p.version != rejected_at && lean).then(|| {
        format!(
            "{}@{} clears the size/dep gate ({} runtime deps, {} bytes unpacked; rejected at {})",
            p.name,
            p.version,
            p.runtime_deps,
            p.unpacked_size,
            if rejected_at.is_empty() {
                "?"
            } else {
                rejected_at
            }
        )
    })
}

/// Ledger view where heavy-dependency rejections whose candidate shipped a
/// lean release are re-queued (the rejection itself stays in history).
pub fn requeue_lean(ledger: &Ledger, graph: &Graph, cfg: &Config) -> Ledger {
    let mut out = ledger.clone();
    let latest: Vec<&LedgerEntry> = ledger.latest_by_key().into_values().collect();
    for (i, e) in ledger.entries.iter().enumerate() {
        if !is_heavy_rejection(e) || !latest.iter().any(|l| std::ptr::eq(*l, e)) {
            continue;
        }
        if let Some(why) = lean_release(graph, cfg, e) {
            let o = &mut out.entries[i];
            o.outcome = crate::model::Outcome::Queued;
            o.reason = Some(format!(
                "re-queued: {why}. Was: {}",
                e.reason.as_deref().unwrap_or("")
            ));
        }
    }
    out
}

/// Map every repo name through the graph's rename aliases, keep only the
/// newest scan / PR snapshot per canonical repo, and re-queue lean releases.
fn canonicalize(inp: &QueueInputs, cfg: &Config) -> (Ledger, Vec<ScanReport>, Vec<RepoPrs>) {
    let g = inp.graph;
    let mut ledger = requeue_lean(inp.ledger, g, cfg);
    for e in &mut ledger.entries {
        e.repo = g.canon(&e.repo);
    }
    let mut reports: BTreeMap<String, ScanReport> = BTreeMap::new();
    for r in inp.reports {
        let mut r = r.clone();
        for s in &mut r.signals {
            s.repo = g.canon(&s.repo);
        }
        let key = g.canon(&r.repo);
        match reports.get(&key) {
            Some(old) if old.scanned_at >= r.scanned_at => {}
            _ => {
                reports.insert(key, r);
            }
        }
    }
    let mut prs: BTreeMap<String, RepoPrs> = BTreeMap::new();
    for p in inp.prs {
        let mut p = p.clone();
        p.repo = g.canon(&p.repo);
        match prs.get(&p.repo) {
            Some(old) if old.fetched_at >= p.fetched_at => {}
            _ => {
                prs.insert(p.repo.clone(), p);
            }
        }
    }
    (
        ledger,
        reports.into_values().collect(),
        prs.into_values().collect(),
    )
}

pub fn build(cfg: &Config, inp: QueueInputs) -> Queue {
    let lp = &cfg.policy.lp;
    let (requeued, reports, live_prs) = canonicalize(&inp, cfg);
    let ledger = &requeued;
    let weights = signal_weights(ledger);
    let mut groups: BTreeMap<String, (WorkKind, Vec<Signal>)> = BTreeMap::new();
    for s in reports.iter().flat_map(|r| &r.signals) {
        let kind = work_kind(s);
        if kind == WorkKind::Create {
            continue; // handled by propose-new
        }
        let name = s
            .candidate
            .clone()
            .or(s.subject.clone())
            .unwrap_or_default();
        let key = match kind {
            // Upstream advisories: one dependency-bump item per lockfile.
            WorkKind::Security if is_upstream_advisory(s) => {
                item_id(kind, &s.repo, &format!("deps:{}", s.file))
            }
            WorkKind::Security => item_id(kind, &s.repo, s.subject.as_deref().unwrap_or(&name)),
            _ => item_id(kind, &s.repo, &name),
        };
        groups
            .entry(key)
            .or_insert_with(|| (kind, vec![]))
            .1
            .push(s.clone());
    }

    let mut items = Vec::new();
    let mut suppressed = inp.create_suppressed;
    for (id, (kind, mut sigs)) in groups {
        sigs.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top = sigs[0].clone();
        let upstream = is_upstream_advisory(&top);
        let subject = if upstream {
            Some(format!("deps:{}", top.file))
        } else if sigs.iter().all(|s| s.subject == top.subject) {
            top.subject.clone()
        } else {
            None
        };
        let candidate = top.candidate.clone();
        if let PairState::Blocked { outcome, reason } = ledger.pair_state(
            &top.repo,
            candidate.as_deref(),
            subject.as_deref(),
            kind,
            Some(top.kind),
            lp.max_failures,
        ) {
            suppressed.push(Suppressed {
                id,
                repo: top.repo.clone(),
                candidate,
                subject,
                reason: format!("ledger {outcome:?}: {reason}"),
            });
            continue;
        }
        let gate = candidate.as_deref().and_then(|c| inp.gates.get(c));
        if let Some(g) = gate.filter(|g| !g.passed) {
            let why: Vec<String> = g
                .failures()
                .iter()
                .map(|r| format!("{}: {}", r.name, r.detail))
                .collect();
            suppressed.push(Suppressed {
                id,
                repo: top.repo.clone(),
                candidate,
                subject,
                reason: format!("gate rejected: {}", why.join("; ")),
            });
            continue;
        }
        let downloads = inp
            .graph
            .repo(&top.repo)
            .map(|r| r.monthly_downloads)
            .unwrap_or(0) as f64;
        let occ: usize = sigs.iter().map(|s| s.occurrences.max(1)).sum();
        let health = if gate.is_some() { 1.0 } else { 0.6 };
        let score = (10.0 + downloads).ln()
            * top.confidence
            * (1.0 + 0.1 * (occ as f64).ln())
            * health
            * weights[&top.kind];
        sigs.truncate(5);
        items.push(WorkItem {
            id,
            kind,
            status: LaneStatus::Deferred,
            repo: top.repo.clone(),
            disclosure: if upstream {
                "public_pr".to_string()
            } else {
                disclosure(kind, top.severity.as_deref()).to_string()
            },
            required_test: required_test(
                kind,
                candidate.as_deref().or(subject.as_deref()).unwrap_or("-"),
            ),
            subject,
            candidate,
            signal_kind: top.kind,
            score,
            evidence: sigs,
            gates: gate.map(|g| g.results.clone()).unwrap_or_default(),
            brief: String::new(),
        });
    }
    let live: BTreeMap<String, &RepoPrs> = live_prs.iter().map(|r| (r.repo.clone(), r)).collect();
    let is_pr = |s: &Option<String>| s.as_deref().map(|x| x.starts_with("pr:#")).unwrap_or(false);
    for e in ledger.queued() {
        if is_pr(&e.subject) {
            if let Some(r) = live.get(&e.repo.to_ascii_lowercase()) {
                let open = r
                    .prs
                    .iter()
                    .any(|p| Some(prs::subject(p.number)) == e.subject);
                if !open {
                    suppressed.push(Suppressed { id: item_id(e.kind, &e.repo, e.subject.as_deref().unwrap_or("")), repo: e.repo.clone(), candidate: None, subject: e.subject.clone(), reason: "PR no longer open since it was queued: record its outcome in the ledger".into() });
                }
                continue; // live PR items below replace the seeded snapshot
            }
        }
        let mut it = backlog_item(e);
        let downloads = inp
            .graph
            .repo(&it.repo)
            .map(|r| r.monthly_downloads)
            .unwrap_or(0) as f64;
        it.score = (10.0 + downloads).ln() * 0.9 * weights[&it.signal_kind];
        items.push(it);
    }
    items.extend(inp.creates);
    for r in &live_prs {
        let downloads = inp
            .graph
            .repo(&r.repo)
            .map(|x| x.monthly_downloads)
            .unwrap_or(0) as f64;
        for p in &r.prs {
            let Some(it) = prs::item(&r.repo, p, downloads, weights[&SignalKind::OpenPr]) else {
                continue;
            };
            match ledger.pair_state(
                &it.repo,
                None,
                it.subject.as_deref(),
                it.kind,
                Some(SignalKind::OpenPr),
                lp.max_failures,
            ) {
                PairState::Blocked { outcome, reason }
                    if outcome != crate::model::Outcome::Queued =>
                {
                    suppressed.push(Suppressed {
                        id: it.id.clone(),
                        repo: it.repo.clone(),
                        candidate: None,
                        subject: it.subject.clone(),
                        reason: format!("ledger {outcome:?}: {reason}"),
                    });
                }
                _ => items.push(it),
            }
        }
    }
    // Never dispatch a lane that re-implements an open PR.
    let mut kept = Vec::with_capacity(items.len());
    for it in items {
        let open = live
            .get(&it.repo.to_ascii_lowercase())
            .map(|r| r.prs.as_slice())
            .unwrap_or(&[]);
        match prs::covering(&it, open) {
            Some(p) => suppressed.push(Suppressed {
                id: it.id.clone(),
                repo: it.repo.clone(),
                candidate: it.candidate.clone(),
                subject: it.subject.clone(),
                reason: format!("open PR #{} already covers it: {}", p.number, p.title),
            }),
            None => kept.push(it),
        }
    }
    let mut items = kept;
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });

    let busy = ledger.busy_repos();
    let lanes = assign_lanes(&mut items, &busy, lp.max_lanes, lp.max_batch_per_lane);
    for it in &mut items {
        let gate = it.candidate.as_deref().and_then(|c| inp.gates.get(c));
        it.brief = brief(it, inp.graph, gate);
    }
    Queue {
        generated_at: now(),
        max_lanes: lp.max_lanes,
        lanes,
        busy_repos: busy,
        signal_weights: weights
            .into_iter()
            .map(|(k, v)| (k.as_str().to_string(), v))
            .collect(),
        items,
        suppressed,
    }
}

/// One lane per repo; the top item is `active`, independent items (distinct
/// candidate, no shared evidence file, public disclosure) ride along as
/// `batched`. Repos with an in-flight ledger lane stay deferred.
pub fn assign_lanes(
    items: &mut [WorkItem],
    busy: &BTreeMap<String, Vec<String>>,
    max_lanes: usize,
    max_batch: usize,
) -> Vec<Lane> {
    let mut lanes: Vec<Lane> = Vec::new();
    let mut lane_of: BTreeMap<String, usize> = BTreeMap::new();
    let mut files: Vec<BTreeSet<String>> = Vec::new();
    let mut cands: Vec<BTreeSet<String>> = Vec::new();
    for it in items.iter_mut() {
        let repo = it.repo.to_ascii_lowercase();
        it.status = LaneStatus::Deferred;
        if busy.contains_key(&repo) {
            continue;
        }
        let my_files: BTreeSet<String> = it.evidence.iter().map(|s| s.file.clone()).collect();
        let my_cand = it
            .candidate
            .clone()
            .or(it.subject.clone())
            .unwrap_or_default();
        match lane_of.get(&repo).copied() {
            None if lanes.len() < max_lanes => {
                lane_of.insert(repo.clone(), lanes.len());
                lanes.push(Lane {
                    repo,
                    items: vec![it.id.clone()],
                });
                files.push(my_files);
                cands.push(BTreeSet::from([my_cand]));
                it.status = LaneStatus::Active;
            }
            Some(l) => {
                let lead_private = items_private(&lanes[l]);
                let independent = it.disclosure == "public_pr"
                    && !lead_private
                    && !cands[l].contains(&my_cand)
                    && files[l].is_disjoint(&my_files);
                if independent && lanes[l].items.len() < max_batch {
                    lanes[l].items.push(it.id.clone());
                    files[l].extend(my_files);
                    cands[l].insert(my_cand);
                    it.status = LaneStatus::Batched;
                }
            }
            None => {}
        }
    }
    lanes
}

fn items_private(l: &Lane) -> bool {
    l.items
        .first()
        .map(|id| id.starts_with("security::"))
        .unwrap_or(false)
}

/// Gate results attached to a queue item (for ledger records).
pub fn gate_results(q: &Queue, id: &str) -> Vec<GateResult> {
    q.items
        .iter()
        .find(|i| i.id == id)
        .map(|i| i.gates.clone())
        .unwrap_or_default()
}
