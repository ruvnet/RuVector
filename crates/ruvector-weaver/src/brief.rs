//! Self-contained agent briefs: each queue item carries everything a lane
//! needs (evidence, verified API, gates, task, required test, disclosure
//! route and the exact ledger command to report back with).

use crate::model::{GateReport, Graph, WorkItem, WorkKind};

pub fn kind_str(kind: WorkKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default()
}

pub fn required_test(kind: WorkKind, cand: &str) -> String {
    match kind {
        WorkKind::Replace => format!("A test that exercises the code path through the real `{cand}` (no fallback) and asserts its result; the existing suite stays green."),
        WorkKind::Integrate => format!("A test that imports `{cand}` from the declared dependency and exercises the integrated path."),
        WorkKind::Hygiene => format!("Build + test pass after declaring (or removing) `{cand}`; a clean install resolves it from the registry."),
        WorkKind::Security => "A regression test that fails before the fix and passes after; `npm audit`/`cargo deny` clean for this finding.".into(),
        WorkKind::Create => format!("Package `{cand}` has its own passing tests; the consumer has a test importing it from the registry after publish; `weaver gate {cand} --new` passes."),
        WorkKind::Land => format!("`{cand}` rebased on main with every required check green, then merged (squash)."),
        WorkKind::CloseWithLedger => format!("Ledger entry (outcome rejected, reason = the verdict and its evidence) recorded BEFORE closing `{cand}`."),
        WorkKind::ReleaseFix => format!("`weaver gate {cand}` passes its release checks without an override after the fix (repository, file count, size, exports, ranges)."),
    }
}

fn ledger_cmd(it: &WorkItem) -> String {
    let opt = |flag: &str, v: &Option<String>| {
        v.as_ref()
            .map(|x| format!(" --{flag} '{x}'"))
            .unwrap_or_default()
    };
    format!(
        "weaver ledger record --repo {}{}{} --kind {} --signal {} --outcome <shipped|rejected|failed> --pr <url> --version <pkg@ver> --reason '<one line>' [--measure metric=value:unit@baseline]",
        it.repo,
        opt("candidate", &it.candidate),
        opt("subject", &it.subject),
        kind_str(it.kind),
        it.signal_kind.as_str()
    )
}

fn task(kind: WorkKind, cand: &str, subject: &str) -> String {
    match kind {
        WorkKind::Replace => format!("Replace the stub/fallback/404 probe (`{subject}`) with the real `{cand}`: declare it (optionalDependency if the consumer must stay installable without it), call its verified API, keep a clear error path, remove dead probes."),
        WorkKind::Integrate => format!("Wire `{cand}` into the consumer where the evidence points, declared in the nearest package.json."),
        WorkKind::Hygiene => format!("`{cand}`: declare it in the nearest package.json if it is imported (optional if probed dynamically), or drop the unused declaration."),
        WorkKind::Security => format!("Fix `{subject}` (bump/patch/replace). Assess severity FIRST: first-party critical/high goes to a private GitHub security advisory, never a public PR or issue; upstream advisories (npm audit / RustSec) are already public and ship as a dependency bump PR."),
        WorkKind::Create => format!("Create and publish NEW package `{cand}` that fills this gap, then wire it into the consumer in the same lane. Autopilot gates (enforced by `weaver gate --new` / `weaver gate`): gap evidence, no near-duplicate in the fabric, real implementation + passing tests, non-empty tarball with working entrypoint, clean leak scan, registry smoke import after publish, README via `weaver readme` (ledger-only numbers)."),
        WorkKind::Land => format!("Land open PR `{subject}`: rebase on main, fix CI until every required check is green, merge (squash). If it no longer applies, close it with a one-line reason and record `failed`."),
        WorkKind::CloseWithLedger => format!("Research PR `{subject}` carries a REJECT / negative verdict. Record it with `weaver ledger record --outcome rejected --reason '<verdict + evidence>'` first, then `gh pr close` with a comment linking the ledger entry."),
        WorkKind::ReleaseFix => format!("Release regression on `{cand}` (`{subject}`): restore the intended line (republish from the owning repo), rename a colliding package, or pin the open range with an upper bound. If the change is intended, record an explicit `--outcome override --subject '{subject}'` ledger entry instead; `weaver gate` refuses the publish otherwise."),
    }
}

pub fn brief(it: &WorkItem, graph: &Graph, gate: Option<&GateReport>) -> String {
    let cand = it.candidate.clone().unwrap_or_else(|| "-".into());
    let subject = it.subject.clone().unwrap_or_else(|| cand.clone());
    let mut b = format!(
        "[weaver lane] {:?} in {} -> {}\n\nEvidence ({} signal(s), {}):\n",
        it.kind,
        it.repo,
        cand,
        it.evidence.len(),
        it.signal_kind.as_str()
    );
    for s in &it.evidence {
        b.push_str(&format!(
            "  - {}:{}  `{}`{}\n",
            s.file,
            s.line,
            s.evidence,
            if s.occurrences > 1 {
                format!(" (+{} more)", s.occurrences - 1)
            } else {
                String::new()
            }
        ));
    }
    if let Some(p) = it.candidate.as_deref().and_then(|c| graph.get(c)) {
        b.push_str(&format!(
            "\nCandidate: {}@{} (last publish {}, {} downloads/mo)\n",
            p.name,
            p.version,
            p.last_publish.as_deref().unwrap_or("?"),
            p.monthly_downloads
        ));
    }
    match gate {
        Some(g) => {
            let names: Vec<String> = g.results.iter().map(|r| format!("{}={}", r.name, if r.passed { "pass" } else { "FAIL" })).collect();
            b.push_str(&format!("Gates ({}): {}\n", g.checked_at, names.join(", ")));
            if !g.exports.is_empty() {
                b.push_str(&format!("Verified API (smoke-import exports): {}\n", g.exports.join(", ")));
            }
        }
        // Package-less items (no candidate) have nothing to gate.
        None if it.candidate.as_deref().is_some_and(|c| !crate::gate::is_packageless(c))
            && matches!(it.kind, WorkKind::Replace | WorkKind::Integrate | WorkKind::Hygiene) => b.push_str(&format!("Gates: not run yet. FIRST run `weaver gate {cand} --consumer {}` and stop if it fails.\n", it.repo)),
        None => {}
    }
    b.push_str(&format!(
        "\nTask: {}\nRequired test: {}\nDisclosure: {}\n",
        task(it.kind, &cand, &subject),
        it.required_test,
        it.disclosure
    ));
    b.push_str("Rules: one PR in this repo; no numeric performance claims unless measured and recorded with --measure; never touch files owned by another lane; publish tokens only via `gcloud secrets versions access` at publish time, never printed or written.\n");
    b.push_str(&format!("Report back with: {}\n", ledger_cmd(it)));
    b
}
