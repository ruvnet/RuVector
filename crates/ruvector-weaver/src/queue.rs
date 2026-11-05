//! `weaver queue`: rank work items across every scanned repo, dedupe them
//! against the ledger, assign ONE lane per repo (with batching of
//! independent items), and attach a self-contained agent brief to each.

use crate::config::Config;
use crate::ledger::{Ledger, PairState};
use crate::model::{
    GateReport, GateResult, Graph, Lane, LaneStatus, LedgerEntry, Queue, ScanReport, Signal, SignalKind, Suppressed,
    WorkItem, WorkKind,
};
use crate::util::now;
use std::collections::{BTreeMap, BTreeSet};

pub fn base_weight(k: SignalKind) -> f64 {
    match k {
        SignalKind::Security => 1.2,
        SignalKind::MissingPackage => 1.0,
        SignalKind::UndeclaredImport => 0.9,
        SignalKind::StubMarker => 0.6,
        SignalKind::UnusedDeclared => 0.5,
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
    let k = serde_json::to_value(kind).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
    format!("{k}::{repo}::{name}")
}

fn required_test(kind: WorkKind, cand: &str) -> String {
    match kind {
        WorkKind::Replace => format!("A test that exercises the code path through the real `{cand}` (no fallback) and asserts its result; the existing suite stays green."),
        WorkKind::Integrate => format!("A test that imports `{cand}` from the declared dependency and exercises the integrated path."),
        WorkKind::Hygiene => format!("Build + test pass after declaring (or removing) `{cand}`; a clean install resolves it from the registry."),
        WorkKind::Security => "A regression test that fails before the fix and passes after; `npm audit`/`cargo deny` clean for this finding.".into(),
        WorkKind::Create => format!("Package `{cand}` has its own passing tests; the consumer has a test importing it from the registry after publish; `weaver gate {cand} --new` passes."),
    }
}

fn ledger_cmd(it: &WorkItem) -> String {
    let opt = |flag: &str, v: &Option<String>| v.as_ref().map(|x| format!(" --{flag} '{x}'")).unwrap_or_default();
    let k = serde_json::to_value(it.kind).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
    format!(
        "weaver ledger record --repo {}{}{} --kind {k} --signal {} --outcome <shipped|rejected|failed> --pr <url> --version <pkg@ver> --reason '<one line>' [--measure metric=value:unit@baseline]",
        it.repo,
        opt("candidate", &it.candidate),
        opt("subject", &it.subject),
        it.signal_kind.as_str()
    )
}

fn task(kind: WorkKind, cand: &str, subject: &str) -> String {
    match kind {
        WorkKind::Replace => format!("Replace the stub/fallback/404 probe (`{subject}`) with the real `{cand}`: declare it (optionalDependency if the consumer must stay installable without it), call its verified API, keep a clear error path, remove dead probes."),
        WorkKind::Integrate => format!("Wire `{cand}` into the consumer where the evidence points, declared in the nearest package.json."),
        WorkKind::Hygiene => format!("`{cand}`: declare it in the nearest package.json if it is imported (optional if probed dynamically), or drop the unused declaration."),
        WorkKind::Security => format!("Fix `{subject}` (bump/patch/replace). Assess severity FIRST: critical/high goes to a private GitHub security advisory, never a public PR or issue."),
        WorkKind::Create => format!("Create and publish NEW package `{cand}` that fills this gap, then wire it into the consumer in the same lane. Autopilot gates (all enforced by `weaver gate --new`): gap evidence, no near-duplicate in the fabric, real implementation + passing tests, non-empty tarball with working entrypoint, clean leak scan, registry smoke import after publish, README via `weaver readme` (ledger-only numbers)."),
    }
}

pub fn brief(it: &WorkItem, graph: &Graph, gate: Option<&GateReport>) -> String {
    let cand = it.candidate.clone().unwrap_or_else(|| "-".into());
    let subject = it.subject.clone().unwrap_or_else(|| cand.clone());
    let mut b = format!("[weaver lane] {:?} in {} -> {}\n\nEvidence ({} signal(s), {}):\n", it.kind, it.repo, cand, it.evidence.len(), it.signal_kind.as_str());
    for s in &it.evidence {
        b.push_str(&format!("  - {}:{}  `{}`{}\n", s.file, s.line, s.evidence, if s.occurrences > 1 { format!(" (+{} more)", s.occurrences - 1) } else { String::new() }));
    }
    if let Some(p) = it.candidate.as_deref().and_then(|c| graph.get(c)) {
        b.push_str(&format!("\nCandidate: {}@{} (last publish {}, {} downloads/mo)\n", p.name, p.version, p.last_publish.as_deref().unwrap_or("?"), p.monthly_downloads));
    }
    match gate {
        Some(g) => {
            let names: Vec<String> = g.results.iter().map(|r| format!("{}={}", r.name, if r.passed { "pass" } else { "FAIL" })).collect();
            b.push_str(&format!("Gates ({}): {}\n", g.checked_at, names.join(", ")));
            if !g.exports.is_empty() {
                b.push_str(&format!("Verified API (smoke-import exports): {}\n", g.exports.join(", ")));
            }
        }
        None if it.kind != WorkKind::Security => b.push_str(&format!("Gates: not run yet. FIRST run `weaver gate {cand} --consumer {}` and stop if it fails.\n", it.repo)),
        None => {}
    }
    b.push_str(&format!("\nTask: {}\nRequired test: {}\nDisclosure: {}\n", task(it.kind, &cand, &subject), it.required_test, it.disclosure));
    b.push_str("Rules: one PR in this repo; no numeric performance claims unless measured and recorded with --measure; never touch files owned by another lane; publish tokens only via `gcloud secrets versions access` at publish time, never printed or written.\n");
    b.push_str(&format!("Report back with: {}\n", ledger_cmd(it)));
    b
}

/// Turn a queued ledger backlog entry into a work item.
fn backlog_item(e: &LedgerEntry) -> WorkItem {
    let (file, line) = e
        .location
        .as_deref()
        .and_then(|l| l.rsplit_once(':').and_then(|(f, n)| n.parse().ok().map(|n| (f.to_string(), n))))
        .unwrap_or_else(|| (e.location.clone().unwrap_or_else(|| "(audit)".into()), 0));
    let name = e.subject.clone().or(e.candidate.clone()).unwrap_or_default();
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
}

pub fn build(cfg: &Config, inp: QueueInputs) -> Queue {
    let lp = &cfg.policy.lp;
    let weights = signal_weights(inp.ledger);
    let mut groups: BTreeMap<String, (WorkKind, Vec<Signal>)> = BTreeMap::new();
    for s in inp.reports.iter().flat_map(|r| &r.signals) {
        let kind = work_kind(s);
        if kind == WorkKind::Create {
            continue; // handled by propose-new
        }
        let name = s.candidate.clone().or(s.subject.clone()).unwrap_or_default();
        let key = match kind {
            // Security findings stay one item per vulnerable package.
            WorkKind::Security => item_id(kind, &s.repo, s.subject.as_deref().unwrap_or(&name)),
            _ => item_id(kind, &s.repo, &name),
        };
        groups.entry(key).or_insert_with(|| (kind, vec![])).1.push(s.clone());
    }

    let mut items = Vec::new();
    let mut suppressed = inp.create_suppressed;
    for (id, (kind, mut sigs)) in groups {
        sigs.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        let top = sigs[0].clone();
        let subject = if sigs.iter().all(|s| s.subject == top.subject) { top.subject.clone() } else { None };
        let candidate = top.candidate.clone();
        if let PairState::Blocked { outcome, reason } = inp.ledger.pair_state(&top.repo, candidate.as_deref(), subject.as_deref(), kind, lp.max_failures) {
            suppressed.push(Suppressed { id, repo: top.repo.clone(), candidate, subject, reason: format!("ledger {outcome:?}: {reason}") });
            continue;
        }
        let gate = candidate.as_deref().and_then(|c| inp.gates.get(c));
        if let Some(g) = gate.filter(|g| !g.passed) {
            let why: Vec<String> = g.failures().iter().map(|r| format!("{}: {}", r.name, r.detail)).collect();
            suppressed.push(Suppressed { id, repo: top.repo.clone(), candidate, subject, reason: format!("gate rejected: {}", why.join("; ")) });
            continue;
        }
        let downloads = inp.graph.repo(&top.repo).map(|r| r.monthly_downloads).unwrap_or(0) as f64;
        let occ: usize = sigs.iter().map(|s| s.occurrences.max(1)).sum();
        let health = if gate.is_some() { 1.0 } else { 0.6 };
        let score = (10.0 + downloads).ln() * top.confidence * (1.0 + 0.1 * (occ as f64).ln()) * health * weights[&top.kind];
        sigs.truncate(5);
        items.push(WorkItem {
            id,
            kind,
            status: LaneStatus::Deferred,
            repo: top.repo.clone(),
            disclosure: disclosure(kind, top.severity.as_deref()).to_string(),
            required_test: required_test(kind, candidate.as_deref().or(subject.as_deref()).unwrap_or("-")),
            subject,
            candidate,
            signal_kind: top.kind,
            score,
            evidence: sigs,
            gates: gate.map(|g| g.results.clone()).unwrap_or_default(),
            brief: String::new(),
        });
    }
    for e in inp.ledger.queued() {
        let mut it = backlog_item(e);
        let downloads = inp.graph.repo(&it.repo).map(|r| r.monthly_downloads).unwrap_or(0) as f64;
        it.score = (10.0 + downloads).ln() * 0.9 * weights[&it.signal_kind];
        items.push(it);
    }
    items.extend(inp.creates);
    items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then(a.id.cmp(&b.id)));

    let busy = inp.ledger.busy_repos();
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
        signal_weights: weights.into_iter().map(|(k, v)| (k.as_str().to_string(), v)).collect(),
        items,
        suppressed,
    }
}

/// One lane per repo; the top item is `active`, independent items (distinct
/// candidate, no shared evidence file, public disclosure) ride along as
/// `batched`. Repos with an in-flight ledger lane stay deferred.
pub fn assign_lanes(items: &mut [WorkItem], busy: &BTreeMap<String, Vec<String>>, max_lanes: usize, max_batch: usize) -> Vec<Lane> {
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
        let my_cand = it.candidate.clone().or(it.subject.clone()).unwrap_or_default();
        match lane_of.get(&repo).copied() {
            None if lanes.len() < max_lanes => {
                lane_of.insert(repo.clone(), lanes.len());
                lanes.push(Lane { repo, items: vec![it.id.clone()] });
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
    l.items.first().map(|id| id.starts_with("security::")).unwrap_or(false)
}

/// Gate results attached to a queue item (for ledger records).
pub fn gate_results(q: &Queue, id: &str) -> Vec<GateResult> {
    q.items.iter().find(|i| i.id == id).map(|i| i.gates.clone()).unwrap_or_default()
}
