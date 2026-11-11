//! Offline tests: signal detection on fixture repos, ranking + lanes,
//! ledger dedupe and reweighting, gate leak scan, README claim lint and
//! new-package proposals. No network: registry access uses MemRegistry.

use ruvector_weaver::config::{Capabilities, Config, EMBEDDED_CAPABILITIES, EMBEDDED_LEDGER};
use ruvector_weaver::gate::{leak_scan, missing_entrypoints, reaches};
use ruvector_weaver::graph::{consumer_repos, is_platform_binary, repo_slug};
use ruvector_weaver::ledger::{parse_measurement, Ledger, PairState};
use ruvector_weaver::model::*;
use ruvector_weaver::registry::{MemRegistry, SearchHit};
use ruvector_weaver::{propose, queue, readme, scan, svg};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn cfg() -> Config {
    Config::embedded(std::env::temp_dir().join("weaver-test")).unwrap()
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn registry() -> MemRegistry {
    MemRegistry::with_names([
        "@ruvector/rvf",
        "@ruvector/gnn",
        "@ruvector/learning-wasm",
        "@ruvector/mincut-wasm",
        "@ruvector/attention",
    ])
}

fn scan_fixture() -> ScanReport {
    scan::scan_dir(
        &fixture("consumer-a"),
        "ruvnet/consumer-a",
        &cfg(),
        &registry(),
        &Graph::default(),
        false,
    )
    .unwrap()
}

fn find<'a>(r: &'a ScanReport, kind: SignalKind, subject: &str) -> Option<&'a Signal> {
    r.signals
        .iter()
        .find(|s| s.kind == kind && s.subject.as_deref() == Some(subject))
}

#[test]
fn detects_404_probes_with_file_line_and_capability() {
    let r = scan_fixture();
    let napi =
        find(&r, SignalKind::MissingPackage, "@ruvector/mincut").expect("404 @ruvector/mincut");
    assert_eq!((napi.file.as_str(), napi.line), ("src/MincutService.ts", 6));
    assert_eq!(napi.candidate.as_deref(), Some("@ruvector/mincut-wasm"));
    let wasm = find(&r, SignalKind::MissingPackage, "ruvector-mincut-wasm")
        .expect("404 ruvector-mincut-wasm");
    assert_eq!(wasm.line, 10);
    assert_eq!(wasm.candidate.as_deref(), Some("@ruvector/mincut-wasm"));
    // A 404 with no capability entry is a gap (candidate None).
    assert!(find(
        &r,
        SignalKind::MissingPackage,
        "@ruvector/mincut-linux-arm-musleabihf"
    )
    .is_none());
    let ast = find(&r, SignalKind::MissingPackage, "@ruvector/ast").expect("404 @ruvector/ast");
    assert_eq!(ast.candidate, None);
    assert_eq!((ast.file.as_str(), ast.line), ("src/ast-analyzer.ts", 3));
}

#[test]
fn detects_undeclared_and_unused_against_nearest_manifest() {
    let r = scan_fixture();
    let lw = find(&r, SignalKind::UndeclaredImport, "@ruvector/learning-wasm")
        .expect("undeclared learning-wasm");
    assert_eq!((lw.file.as_str(), lw.line), ("src/learning.ts", 3));
    // rvf is declared at the root but NOT in packages/inner's manifest.
    let inner =
        find(&r, SignalKind::UndeclaredImport, "@ruvector/rvf").expect("undeclared in inner");
    assert_eq!(inner.file, "packages/inner/src/main.js");
    let gnn = find(&r, SignalKind::UnusedDeclared, "@ruvector/gnn").expect("unused gnn");
    assert_eq!((gnn.file.as_str(), gnn.line), ("package.json", 6));
    // Declared + imported, comment-only mentions, and non-ecosystem deps are silent.
    assert!(find(&r, SignalKind::UnusedDeclared, "@ruvector/rvf").is_none());
    assert!(r
        .signals
        .iter()
        .all(|s| s.subject.as_deref() != Some("@ruvector/attention")));
    assert!(r
        .signals
        .iter()
        .all(|s| s.subject.as_deref() != Some("left-pad")));
}

#[test]
fn stub_markers_map_to_capabilities_and_skip_tests() {
    let r = scan_fixture();
    let stub = r
        .signals
        .iter()
        .find(|s| s.kind == SignalKind::StubMarker)
        .expect("stub marker");
    assert_eq!(stub.file, "src/MincutService.ts");
    assert_eq!(stub.candidate.as_deref(), Some("@ruvector/mincut-wasm"));
    assert!(stub.evidence.contains("fallback"));
    assert!(r.signals.iter().all(|s| !s.file.starts_with("tests/")));
}

#[test]
fn capability_name_matching() {
    let caps = Capabilities::parse(EMBEDDED_CAPABILITIES).unwrap();
    assert_eq!(
        caps.match_package_name("@ruvector/mincut")
            .unwrap()
            .capability
            .package,
        "@ruvector/mincut-wasm"
    );
    assert_eq!(
        caps.match_package_name("@ruvector/rabitq")
            .unwrap()
            .capability
            .package,
        "@ruvector/rabitq-wasm"
    );
    assert!(caps.match_package_name("@ruvector/ast").is_none());
    assert!(caps.match_package_name("@ruvector/coverage").is_none());
}

fn entry(
    repo: &str,
    cand: Option<&str>,
    subj: Option<&str>,
    kind: WorkKind,
    sk: SignalKind,
    outcome: Outcome,
) -> LedgerEntry {
    LedgerEntry {
        ts: "2026-11-05T00:00:00Z".into(),
        repo: repo.into(),
        candidate: cand.map(String::from),
        subject: subj.map(String::from),
        kind,
        signal_kind: Some(sk),
        outcome,
        pr: None,
        version: None,
        reason: Some("r".into()),
        measurements: vec![],
        gates: vec![],
        location: None,
        severity: None,
        blocked_on: vec![],
    }
}

#[test]
fn ledger_dedupe_and_failure_budget() {
    let mut l = Ledger::default();
    l.entries.push(entry(
        "ruvnet/a",
        Some("@ruvector/mincut-wasm"),
        Some("@ruvector/mincut"),
        WorkKind::Replace,
        SignalKind::MissingPackage,
        Outcome::Shipped,
    ));
    // Same candidate, other subject, same kind -> blocked (never re-proposed).
    assert!(matches!(
        l.pair_state(
            "ruvnet/a",
            Some("@ruvector/mincut-wasm"),
            Some("ruvector-mincut-wasm"),
            WorkKind::Replace,
            None,
            2
        ),
        PairState::Blocked {
            outcome: Outcome::Shipped,
            ..
        }
    ));
    // Same candidate, different work kind and subject -> open.
    assert_eq!(
        l.pair_state(
            "ruvnet/a",
            Some("@ruvector/mincut-wasm"),
            Some("x"),
            WorkKind::Hygiene,
            None,
            2
        ),
        PairState::Open
    );
    // Other repo -> open.
    assert_eq!(
        l.pair_state(
            "ruvnet/b",
            Some("@ruvector/mincut-wasm"),
            None,
            WorkKind::Replace,
            None,
            2
        ),
        PairState::Open
    );
    l.entries.push(entry(
        "ruvnet/c",
        Some("p"),
        None,
        WorkKind::Integrate,
        SignalKind::StubMarker,
        Outcome::Failed,
    ));
    assert_eq!(
        l.pair_state("ruvnet/c", Some("p"), None, WorkKind::Integrate, None, 2),
        PairState::Open
    );
    l.entries.push(entry(
        "ruvnet/c",
        Some("p"),
        None,
        WorkKind::Integrate,
        SignalKind::StubMarker,
        Outcome::Failed,
    ));
    assert!(matches!(
        l.pair_state("ruvnet/c", Some("p"), None, WorkKind::Integrate, None, 2),
        PairState::Blocked {
            outcome: Outcome::Failed,
            ..
        }
    ));
    // A later shipped entry replaces in_progress as the latest state.
    l.entries.push(entry(
        "ruvnet/d",
        Some("q"),
        None,
        WorkKind::Integrate,
        SignalKind::StubMarker,
        Outcome::InProgress,
    ));
    assert!(l.busy_repos().contains_key("ruvnet/d"));
    l.entries.push(entry(
        "ruvnet/d",
        Some("q"),
        None,
        WorkKind::Integrate,
        SignalKind::StubMarker,
        Outcome::Shipped,
    ));
    assert!(!l.busy_repos().contains_key("ruvnet/d"));
}

#[test]
fn ledger_hit_rates_reweight_signal_types() {
    let mut l = Ledger::default();
    for _ in 0..3 {
        l.entries.push(entry(
            "r/x",
            Some("a"),
            None,
            WorkKind::Replace,
            SignalKind::MissingPackage,
            Outcome::Shipped,
        ));
    }
    for _ in 0..3 {
        l.entries.push(entry(
            "r/x",
            Some("b"),
            None,
            WorkKind::Replace,
            SignalKind::StubMarker,
            Outcome::Rejected,
        ));
    }
    let h = l.hit_rates();
    assert!((h[&SignalKind::MissingPackage] - 0.8).abs() < 1e-9); // (3+1)/(3+2)
    assert!((h[&SignalKind::StubMarker] - 0.2).abs() < 1e-9); // (0+1)/(3+2)
    assert!((h[&SignalKind::Security] - 0.5).abs() < 1e-9); // unseen prior
    let w = queue::signal_weights(&l);
    assert!(w[&SignalKind::MissingPackage] > queue::base_weight(SignalKind::MissingPackage));
    assert!(w[&SignalKind::StubMarker] < queue::base_weight(SignalKind::StubMarker));
}

#[test]
fn seeded_ledger_parses_and_blocks_known_lanes() {
    let l = Ledger::parse(EMBEDDED_LEDGER).unwrap();
    assert!(l.entries.len() >= 30);
    let busy = l.busy_repos();
    for repo in ["ruvnet/claude-flow", "ruvnet/ruvector"] {
        assert!(
            busy.contains_key(repo),
            "{repo} should hold an in-flight lane"
        );
    }
    assert!(matches!(
        l.pair_state(
            "ruvnet/claude-flow",
            None,
            Some("@ruvector/ast"),
            WorkKind::Create,
            None,
            2
        ),
        PairState::Blocked {
            outcome: Outcome::InProgress,
            ..
        }
    ));
    assert!(matches!(
        l.pair_state(
            "ruvnet/agentic-flow",
            Some("midstreamer"),
            None,
            WorkKind::Integrate,
            None,
            2
        ),
        PairState::Blocked {
            outcome: Outcome::Rejected,
            ..
        }
    ));
    assert!(l.queued().len() >= 7);
}

fn graph_with(repos: &[(&str, u64)]) -> Graph {
    let packages: Vec<Package> = repos
        .iter()
        .map(|(slug, dl)| Package {
            name: slug.rsplit('/').next().unwrap().to_string(),
            ecosystem: Ecosystem::Npm,
            version: "1.0.0".into(),
            last_publish: None,
            monthly_downloads: *dl,
            description: String::new(),
            keywords: vec![],
            dependencies: vec![],
            optional_dependencies: vec![],
            peer_dependencies: vec![],
            repository: None,
            repo_slug: Some(slug.to_string()),
            platform_binary: false,
            open_ranges: vec![],
            unpacked_size: 0,
            runtime_deps: 0,
            dep_ranges: BTreeMap::new(),
        })
        .collect();
    Graph {
        generated_at: String::new(),
        consumer_repos: consumer_repos(&packages),
        packages,
        repo_aliases: BTreeMap::new(),
    }
}

fn report_for(repo: &str) -> ScanReport {
    let mut r = scan_fixture();
    r.repo = repo.into();
    for s in &mut r.signals {
        s.repo = repo.into();
    }
    r
}

#[test]
fn ranking_one_lane_per_repo_with_batching() {
    let c = cfg();
    let g = graph_with(&[("ruvnet/big", 500_000), ("ruvnet/small", 10)]);
    let reports = vec![report_for("ruvnet/big"), report_for("ruvnet/small")];
    let ledger = Ledger::default();
    let gates = BTreeMap::new();
    let q = queue::build(
        &c,
        queue::QueueInputs {
            graph: &g,
            reports: &reports,
            ledger: &ledger,
            gates: &gates,
            creates: vec![],
            create_suppressed: vec![],
            prs: &[],
            runs: &[],
        },
    );
    assert_eq!(q.lanes.len(), 2);
    for lane in &q.lanes {
        let active = q
            .items
            .iter()
            .filter(|i| i.repo == lane.repo && i.status == LaneStatus::Active)
            .count();
        assert_eq!(active, 1, "exactly one active item per repo");
    }
    // Downloads dominate: the big repo's best item outranks the small one's.
    let top = |repo: &str| {
        q.items
            .iter()
            .filter(|i| i.repo == repo)
            .map(|i| i.score)
            .fold(0.0, f64::max)
    };
    assert!(top("ruvnet/big") > top("ruvnet/small"));
    // Batched items never share a candidate or evidence file with the lane.
    for lane in &q.lanes {
        let items: Vec<&WorkItem> = lane
            .items
            .iter()
            .map(|id| q.items.iter().find(|i| &i.id == id).unwrap())
            .collect();
        let files: Vec<&String> = items
            .iter()
            .flat_map(|i| i.evidence.iter().map(|s| &s.file).collect::<BTreeSet<_>>())
            .collect();
        assert_eq!(
            files.iter().collect::<BTreeSet<_>>().len(),
            files.len(),
            "batched items share an evidence file"
        );
        let cands: BTreeSet<_> = items
            .iter()
            .map(|i| i.candidate.clone().or(i.subject.clone()))
            .collect();
        assert_eq!(cands.len(), items.len());
    }
    // Every item carries a self-contained brief with evidence and a ledger command.
    assert!(q
        .items
        .iter()
        .all(|i| i.brief.contains("Evidence") && i.brief.contains("weaver ledger record")));
}

#[test]
fn busy_repo_lane_is_deferred_and_ledger_pairs_suppressed() {
    let c = cfg();
    let g = graph_with(&[("ruvnet/big", 500_000)]);
    let reports = vec![report_for("ruvnet/big")];
    let mut ledger = Ledger::default();
    ledger.entries.push(entry(
        "ruvnet/big",
        Some("@ruvector/mincut-wasm"),
        Some("@ruvector/mincut"),
        WorkKind::Replace,
        SignalKind::MissingPackage,
        Outcome::InProgress,
    ));
    let gates = BTreeMap::new();
    let q = queue::build(
        &c,
        queue::QueueInputs {
            graph: &g,
            reports: &reports,
            ledger: &ledger,
            gates: &gates,
            creates: vec![],
            create_suppressed: vec![],
            prs: &[],
            runs: &[],
        },
    );
    assert!(q.lanes.is_empty());
    assert!(q.items.iter().all(|i| i.status == LaneStatus::Deferred));
    assert!(q
        .suppressed
        .iter()
        .any(|s| s.candidate.as_deref() == Some("@ruvector/mincut-wasm")));
}

#[test]
fn failed_gate_rejects_candidate() {
    let c = cfg();
    let g = graph_with(&[("ruvnet/big", 1000)]);
    let reports = vec![report_for("ruvnet/big")];
    let ledger = Ledger::default();
    let mut gates = BTreeMap::new();
    gates.insert(
        "@ruvector/learning-wasm".to_string(),
        GateReport {
            package: "@ruvector/learning-wasm".into(),
            passed: false,
            results: vec![GateResult {
                name: "leak_scan".into(),
                passed: false,
                detail: "leaked: .env".into(),
            }],
            ..Default::default()
        },
    );
    let q = queue::build(
        &c,
        queue::QueueInputs {
            graph: &g,
            reports: &reports,
            ledger: &ledger,
            gates: &gates,
            creates: vec![],
            create_suppressed: vec![],
            prs: &[],
            runs: &[],
        },
    );
    assert!(q
        .items
        .iter()
        .all(|i| i.candidate.as_deref() != Some("@ruvector/learning-wasm")));
    assert!(q.suppressed.iter().any(|s| s.reason.contains("leak_scan")));
}

#[test]
fn security_findings_route_to_private_advisory() {
    assert_eq!(
        queue::disclosure(WorkKind::Security, Some("critical")),
        "private_advisory"
    );
    assert_eq!(
        queue::disclosure(WorkKind::Security, Some("high")),
        "private_advisory"
    );
    assert_eq!(
        queue::disclosure(WorkKind::Security, None),
        "triage_then_route"
    );
    assert_eq!(
        queue::disclosure(WorkKind::Security, Some("low")),
        "public_pr"
    );
    let audit = serde_json::json!({"vulnerabilities": {"tar": {"severity": "high", "via": [{"title": "Path traversal"}], "fixAvailable": true}}});
    let lock = "{\n  \"packages\": {\n    \"node_modules/tar\": {}\n  }\n}";
    let s =
        ruvector_weaver::scan::security::parse_npm_audit("r/x", "package-lock.json", lock, &audit);
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].line, 3);
    assert_eq!(s[0].severity.as_deref(), Some("high"));
}

#[test]
fn gate_leak_scan_flags_secrets_and_agent_state() {
    let c = cfg();
    let files: Vec<String> = [
        "dist/index.js",
        ".claude/settings.json",
        ".claude-flow/metrics.json",
        "data/memory.db",
        ".env",
        ".env.local",
        ".npmrc",
        "lib/claude.js",
        "src/env.ts",
        "README.md",
        "keys/server.pem",
        ".swarm/state.json",
        "CLAUDE.md",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let leaked = leak_scan(&files, &c.policy.gate.leak_patterns).unwrap();
    let want = [
        ".claude/settings.json",
        ".claude-flow/metrics.json",
        "data/memory.db",
        ".env",
        ".env.local",
        ".npmrc",
        "keys/server.pem",
        ".swarm/state.json",
    ];
    assert_eq!(
        leaked.iter().map(String::as_str).collect::<BTreeSet<_>>(),
        want.into_iter().collect()
    );
}

#[test]
fn gate_entrypoints_and_cycles() {
    let m = serde_json::json!({"main": "dist/index.js", "exports": {".": {"import": "./dist/index.mjs", "types": "./dist/index.d.ts"}}, "bin": {"x": "bin/x.js"}});
    let (eps, missing) = missing_entrypoints(&m, &["dist/index.js".into(), "bin/x.js".into()]);
    assert_eq!(eps.len(), 3);
    assert_eq!(missing, vec!["dist/index.mjs".to_string()]);
    let mut g = graph_with(&[("ruvnet/agentic-flow", 1), ("ruvnet/midstream", 1)]);
    g.packages
        .iter_mut()
        .find(|p| p.name == "midstream")
        .unwrap()
        .dependencies = vec!["agentic-flow".into()];
    let targets = BTreeSet::from(["agentic-flow".to_string()]);
    assert_eq!(
        reaches(&g, "midstream", &targets),
        Some(vec!["midstream".into(), "agentic-flow".into()])
    );
    assert_eq!(
        reaches(
            &g,
            "agentic-flow",
            &BTreeSet::from(["midstream".to_string()])
        ),
        None
    );
}

#[test]
fn readme_lint_only_allows_ledger_numbers() {
    let md = format!("# x\nIt is 52x faster than Morph.\n{}\n| latency | 1.2 ms |\n{}\nRecall improves to 0.97 here.\n", readme::MEASURED_START, readme::MEASURED_END);
    let hits = readme::lint(&md);
    assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![2, 6]);
    assert_eq!(
        readme::strip_claims(
            "Ultra-fast code editing engine - 52x faster than Morph LLM at $0 cost"
        ),
        "Ultra-fast code editing engine"
    );
}

#[test]
fn rendered_readme_is_lint_clean_and_banner_follows_convention() {
    let c = cfg();
    let mut l = Ledger::default();
    let mut e = entry(
        "ruvnet/agentdb",
        Some("@ruvector/mincut-wasm"),
        None,
        WorkKind::Replace,
        SignalKind::MissingPackage,
        Outcome::Shipped,
    );
    e.measurements
        .push(parse_measurement("p50_latency=0.8:ms@3.1").unwrap());
    l.entries.push(e);
    let mut g = graph_with(&[("ruvnet/ruvector", 100)]);
    g.packages[0].name = "@ruvector/mincut-wasm".into();
    g.packages[0].description = "Weighted dynamic minimum cut - 12x faster than JS".into();
    let r = readme::render(&c, &g, &l, None, "@ruvector/mincut-wasm");
    assert!(
        readme::lint(&r.readme).is_empty(),
        "{:?}",
        readme::lint(&r.readme)
    );
    assert!(r.readme.contains("| p50_latency | 0.8 ms | 3.1 ms |"));
    assert!(r.readme.contains("## Related packages") && r.readme.contains("@ruvector/kge"));
    assert!(!r.readme.contains("12x"));
    assert_eq!(r.svg_file, "ruvector-mincut-wasm.svg");
    for needle in [
        "viewBox=\"0 0 1000 240\"",
        "aria-labelledby=\"title desc\"",
        "prefers-reduced-motion",
        "@keyframes flow",
        "#0c1324",
    ] {
        assert!(r.svg.contains(needle), "banner missing {needle}");
    }
    assert_eq!(svg::esc("<a&b>"), "&lt;a&amp;b&gt;");
}

#[test]
fn propose_new_creates_gated_items_or_suppresses() {
    let mut c = cfg();
    let reports = vec![scan_fixture()];
    let g = Graph::default();
    let ledger = Ledger::default();
    let p = propose::propose(&c, &g, &reports, &ledger, &MemRegistry::default());
    let ast = p
        .items
        .iter()
        .find(|i| i.candidate.as_deref() == Some("@ruvector/ast"))
        .expect("create item for the gap");
    assert_eq!(ast.kind, WorkKind::Create);
    assert!(ast.gates.iter().all(|g| g.passed));
    assert!(["gap_evidence", "no_duplicate", "rate_limit"]
        .iter()
        .all(|n| ast.gates.iter().any(|g| g.name == *n)));
    // Near-duplicate in the fabric -> suppressed, not created.
    let mut reg = MemRegistry::default();
    reg.hits.push(SearchHit {
        name: "@ruvector/ast-tools".into(),
        description: "AST analysis".into(),
        keywords: vec![],
    });
    let p = propose::propose(&c, &g, &reports, &ledger, &reg);
    assert!(p.items.is_empty());
    assert!(p.suppressed[0].reason.contains("@ruvector/ast-tools"));
    // Optional per-tick cap.
    c.policy.lp.max_new_packages_per_tick = Some(0);
    let p = propose::propose(&c, &g, &reports, &ledger, &MemRegistry::default());
    assert!(p.items.is_empty() && p.suppressed[0].reason.starts_with("rate_limit"));
    // Default policy: unlimited.
    assert_eq!(cfg().policy.lp.max_new_packages_per_tick, None);
    assert_eq!(cfg().policy.lp.max_lanes, 8);
}

#[test]
fn graph_helpers() {
    assert!(is_platform_binary("@ruvector/core-linux-x64-gnu"));
    assert!(is_platform_binary("ruvector-darwin-arm64"));
    assert!(is_platform_binary("@ruvector/rvf-node-win32-x64-msvc"));
    assert!(is_platform_binary("@ruvector/sona-linux-arm-musleabihf"));
    assert!(is_platform_binary("@ruvector/sona-android-arm-eabi"));
    assert!(!is_platform_binary("@ruvector/mincut-wasm"));
    assert_eq!(
        repo_slug("git+https://github.com/ruvnet/RuVector.git").as_deref(),
        Some("ruvnet/ruvector")
    );
    assert_eq!(
        repo_slug("github:ruvnet/agentdb").as_deref(),
        Some("ruvnet/agentdb")
    );
    assert_eq!(
        repo_slug("https://github.com/ruvnet/ruvector/tree/main/npm/packages/rvf").as_deref(),
        Some("ruvnet/ruvector")
    );
}

// ---- open PRs ---------------------------------------------------------------

use ruvector_weaver::prs::{self, OpenPr, PrClass, RepoPrs};
use ruvector_weaver::release;

fn pr(n: u64, title: &str, head: &str) -> OpenPr {
    OpenPr {
        number: n,
        title: title.into(),
        url: format!("https://github.com/ruvnet/big/pull/{n}"),
        head: head.into(),
        updated_at: "2026-11-01T00:00:00Z".into(),
        draft: false,
        labels: vec![],
        checks: "unknown".into(),
        stale_days: 4,
        class: prs::classify(title, head, &[]),
    }
}

#[test]
fn pr_classification() {
    use PrClass::*;
    let c = |t: &str| prs::classify(t, "x", &[]);
    assert_eq!(
        c("research(sona): coherence-gated ReasoningBank retrieval (REJECT, root cause found)"),
        CloseWithLedger
    );
    assert_eq!(c("research: incremental-boundary mincut search — real 2-4x speedup, falls short of 10x threshold"), CloseWithLedger);
    assert_eq!(c("research(ann): frontier-cut conductance as adaptive ef_search signal (negative result)"), CloseWithLedger);
    assert_eq!(
        c("security: daily hardening 2026-11-05 (IPv6-embedded address check)"),
        Land
    );
    assert_eq!(
        c("fix(core): refill selective filtered top-k from exact eligible candidates"),
        Land
    );
    assert_eq!(
        c("build(deps): bump follow-redirects from 1.16.1 to 1.16.2 in /npm"),
        Land
    );
    assert_eq!(
        c("feat: RuGraph graph AI language with native runtime"),
        Other
    );
    assert_eq!(
        c("research(agent-memory): sampled betweenness centrality bridge protection"),
        Other
    );
    assert_eq!(
        prs::classify("chore: x", "dependabot/npm_and_yarn/x", &[]),
        Land
    );
}

#[test]
fn open_prs_become_items_and_block_duplicate_work() {
    let c = cfg();
    let g = graph_with(&[("ruvnet/big", 1000)]);
    let reports = vec![report_for("ruvnet/big")];
    let mut ledger = Ledger::default();
    // A seeded PR that has since been merged.
    let mut gone = entry(
        "ruvnet/big",
        None,
        Some("pr:#7"),
        WorkKind::Land,
        SignalKind::OpenPr,
        Outcome::Queued,
    );
    gone.reason = Some("fix: old".into());
    ledger.entries.push(gone);
    let live = vec![RepoPrs {
        repo: "ruvnet/big".into(),
        fetched_at: String::new(),
        main_red: vec!["Workspace CI".into(), "Security Audit".into()],
        main_sha: Some("e985713aa".into()),
        prs: vec![
            pr(
                1364,
                "ruvector 0.3.12: fix hooks graph-mincut, wire @ruvector/mincut-wasm",
                "fix/ruvector-graph-mincut",
            ),
            pr(
                1295,
                "research(sona): coherence-gated retrieval (REJECT, root cause found)",
                "claude/x",
            ),
            pr(
                1225,
                "feat: RuGraph graph AI language",
                "feat/graph-language-v1",
            ),
        ],
    }];
    let gates = BTreeMap::new();
    let q = queue::build(
        &c,
        queue::QueueInputs {
            graph: &g,
            reports: &reports,
            ledger: &ledger,
            gates: &gates,
            creates: vec![],
            create_suppressed: vec![],
            prs: &live,
            runs: &[],
        },
    );
    let kinds: BTreeSet<_> = q
        .items
        .iter()
        .filter(|i| i.signal_kind == SignalKind::OpenPr)
        .map(|i| (i.kind, i.subject.clone().unwrap()))
        .collect();
    assert!(kinds.contains(&(WorkKind::Land, "pr:#1364".into())));
    assert!(kinds.contains(&(WorkKind::CloseWithLedger, "pr:#1295".into())));
    assert!(
        !kinds.iter().any(|(_, s)| s == "pr:#1225"),
        "feature PRs are not auto-landed"
    );
    // The mincut-wasm replacement is already an open PR: not re-dispatched.
    assert!(q
        .items
        .iter()
        .all(|i| i.candidate.as_deref() != Some("@ruvector/mincut-wasm")));
    assert!(q
        .suppressed
        .iter()
        .any(|s| s.reason.starts_with("open PR #1364")));
    assert!(q
        .suppressed
        .iter()
        .any(|s| s.subject.as_deref() == Some("pr:#7") && s.reason.contains("no longer open")));
    // A red main is its own hygiene item, and every brief in that repo says so.
    let red = q
        .items
        .iter()
        .find(|i| i.signal_kind == SignalKind::MainRed)
        .expect("main-red item");
    assert!(red.evidence[0]
        .evidence
        .contains("Workspace CI, Security Audit"));
    assert!(q
        .items
        .iter()
        .filter(|i| i.repo == "ruvnet/big")
        .all(|i| i.brief.contains("Main baseline: already red on main")));
}

#[test]
fn red_workflows_reads_latest_head_only() {
    let runs = serde_json::json!([
        {"workflowName": "CI", "conclusion": "failure", "headSha": "new"},
        {"workflowName": "Lint", "conclusion": "success", "headSha": "new"},
        {"workflowName": "CI", "conclusion": "success", "headSha": "new"},
        {"workflowName": "Audit", "conclusion": "failure", "headSha": "old"}
    ]);
    let (sha, red) = prs::red_workflows(&runs);
    assert_eq!(sha.as_deref(), Some("new"));
    assert_eq!(red, vec!["CI".to_string()]);
}

#[test]
fn covering_pr_matches_audit_subjects() {
    let open = vec![pr(
        1317,
        "perf(core): ultra optimize filtered search with shared inverted postings",
        "perf/metadata-filter-index-1313",
    )];
    let item = WorkItem {
        id: "x".into(),
        kind: WorkKind::Security,
        status: LaneStatus::Deferred,
        repo: "ruvnet/big".into(),
        subject: Some("audit:core-filtered-search".into()),
        candidate: Some("ruvector-core".into()),
        signal_kind: SignalKind::Security,
        score: 0.0,
        evidence: vec![],
        gates: vec![],
        disclosure: "public_pr".into(),
        required_test: String::new(),
        brief: String::new(),
    };
    assert_eq!(prs::covering(&item, &open).map(|p| p.number), Some(1317));
    let mut other = item.clone();
    other.subject = Some("audit:rvf-qr-seed-url-length".into());
    other.candidate = Some("@ruvector/rvf".into());
    assert!(prs::covering(&other, &open).is_none());
}

// ---- release regressions ------------------------------------------------------

fn snap(v: &str, repo: &str, files: u64, size: u64, keys: usize) -> release::Snapshot {
    release::Snapshot {
        version: v.into(),
        repo: Some(repo.into()),
        file_count: files,
        unpacked_size: size,
        export_keys: keys,
    }
}

#[test]
fn release_compare_catches_the_ruvllm_incident_shape() {
    let prev = snap("3.9.0", "ruvnet/ruvector", 137, 2_400_000, 6);
    let next = snap("4.0.0", "ruvnet/ruvllm", 16, 90_000, 2);
    let r = release::compare(&prev, &next);
    let failed: BTreeSet<&str> = r
        .iter()
        .filter(|x| !x.passed)
        .map(|x| x.name.as_str())
        .collect();
    assert_eq!(
        failed,
        BTreeSet::from([
            "repo_changed",
            "file_count_drop",
            "size_drop",
            "export_surface_drop"
        ])
    );
    let ok = release::compare(&prev, &snap("3.9.1", "ruvnet/ruvector", 140, 2_500_000, 6));
    assert!(ok.iter().all(|x| x.passed));
    let sigs = release::signals("@ruvector/ruvllm", &prev, &r, "ruvnet/ruvllm");
    assert_eq!(sigs.len(), 4);
    assert!(sigs
        .iter()
        .all(|s| s.repo == "ruvnet/ruvector" && s.kind == SignalKind::ReleaseRegression));
}

#[test]
fn release_latest_pair_orders_by_publish_time() {
    let doc = serde_json::json!({
        "dist-tags": {"latest": "4.0.1"},
        "time": {"created": "x", "modified": "y", "3.9.0": "2026-10-01T00:00:00Z", "4.0.0-rc.1": "2026-10-20T00:00:00Z", "4.0.1": "2026-11-01T00:00:00Z"},
        "versions": {
            "3.9.0": {"version": "3.9.0", "repository": {"url": "git+https://github.com/ruvnet/RuVector.git"}, "dist": {"fileCount": 137, "unpackedSize": 2400000}},
            "4.0.0-rc.1": {"version": "4.0.0-rc.1", "dist": {"fileCount": 1}},
            "4.0.1": {"version": "4.0.1", "repository": "github:ruvnet/ruvllm", "dist": {"fileCount": 16, "unpackedSize": 90000}}
        }
    });
    let (p, n) = release::latest_pair(&doc).unwrap();
    assert_eq!((p.version.as_str(), n.version.as_str()), ("3.9.0", "4.0.1"));
    assert_eq!(p.repo.as_deref(), Some("ruvnet/ruvector"));
    assert_eq!(n.repo.as_deref(), Some("ruvnet/ruvllm"));
    assert_eq!(release::previous_major(&doc).unwrap().version, "3.9.0");
}

#[test]
fn release_gate_refuses_without_ledger_override() {
    let prev = snap("3.9.0", "ruvnet/ruvector", 137, 2_400_000, 6);
    let next = snap("4.1.0", "ruvnet/ruvector", 137, 2_400_000, 6);
    // Restoring from the original repo after a hijacked line still trips
    // repo_changed vs the bad latest, until the ledger says it is intended.
    let bad = snap("4.0.1", "ruvnet/ruvllm", 16, 90_000, 2);
    let r = release::compare(&bad, &next);
    let mut l = Ledger::default();
    let gated = release::with_overrides(&l, "@ruvector/ruvllm", r.clone());
    assert!(gated.iter().any(|x| x.name == "repo_changed" && !x.passed));
    l.entries.push(entry(
        "ruvnet/ruvector",
        Some("@ruvector/ruvllm"),
        Some("release:repo_changed:@ruvector/ruvllm"),
        WorkKind::ReleaseFix,
        SignalKind::ReleaseRegression,
        Outcome::Override,
    ));
    let gated = release::with_overrides(&l, "@ruvector/ruvllm", r);
    assert!(gated.iter().all(|x| x.passed));
    assert!(gated.iter().any(|x| x.detail.contains("ledger override")));
    assert!(release::compare(&prev, &next).iter().all(|x| x.passed));
}

#[test]
fn open_ranges_and_name_collisions_in_scans() {
    assert!(release::is_open_range(">=2.0.0"));
    assert!(release::is_open_range("latest"));
    assert!(release::is_open_range("*"));
    assert!(!release::is_open_range(">=2.0.0 <3.0.0"));
    assert!(!release::is_open_range("^2.0.0"));
    let mut g = graph_with(&[("ruvnet/elsewhere", 10)]);
    g.packages[0].name = "consumer-a".into();
    let r = scan::scan_dir(
        &fixture("consumer-a"),
        "ruvnet/consumer-a",
        &cfg(),
        &registry(),
        &g,
        false,
    )
    .unwrap();
    let subj = |s: &str| r.signals.iter().find(|x| x.subject.as_deref() == Some(s));
    let col = subj("release:name_collision:consumer-a").expect("name collision");
    assert_eq!((col.file.as_str(), col.line), ("package.json", 2));
    assert!(col.evidence.contains("ruvnet/elsewhere"));
    let sona = subj("release:open_range:@ruvector/sona").expect("open range latest");
    assert_eq!(sona.line, 8);
    assert!(subj("release:open_range:@ruvector/kge").is_some());
    assert!(subj("release:open_range:@ruvector/rvf").is_none());
}

#[test]
fn seeded_ledger_holds_the_ruvllm_incident() {
    let l = Ledger::parse(EMBEDDED_LEDGER).unwrap();
    let st = l.pair_state(
        "ruvnet/ruvector",
        Some("@ruvector/ruvllm"),
        Some("release:size_drop:@ruvector/ruvllm"),
        WorkKind::ReleaseFix,
        Some(SignalKind::ReleaseRegression),
        2,
    );
    assert!(matches!(
        st,
        PairState::Blocked {
            outcome: Outcome::InProgress,
            ..
        }
    ));
}

#[test]
fn heavy_rejection_requeues_on_lean_release() {
    let c = cfg();
    let mut l = Ledger::default();
    let mut e = entry(
        "ruvnet/agentdb",
        Some("sublinear-time-solver"),
        Some("agentdb:ppr-fallback"),
        WorkKind::Replace,
        SignalKind::StubMarker,
        Outcome::Rejected,
    );
    e.reason = Some("heavy dependency tree (18 runtime deps)".into());
    e.version = Some("sublinear-time-solver@1.7.2".into());
    l.entries.push(e);
    let mut g = graph_with(&[
        ("ruvnet/sublinear-time-solver", 1000),
        ("ruvnet/agentdb", 10),
    ]);
    let p = g
        .packages
        .iter_mut()
        .find(|p| p.name == "sublinear-time-solver")
        .unwrap();
    p.version = "1.7.2".into();
    p.runtime_deps = 18;
    p.unpacked_size = 9_845_144;
    // Same heavy version: still rejected, no item.
    let q = queue::build(
        &c,
        queue::QueueInputs {
            graph: &g,
            reports: &[],
            ledger: &l,
            gates: &BTreeMap::new(),
            creates: vec![],
            create_suppressed: vec![],
            prs: &[],
            runs: &[],
        },
    );
    assert!(q.items.is_empty());
    // Lean release ships: the rejection is re-queued as a work item.
    let p = g
        .packages
        .iter_mut()
        .find(|p| p.name == "sublinear-time-solver")
        .unwrap();
    p.version = "1.8.0".into();
    p.runtime_deps = 1;
    p.unpacked_size = 400_000;
    let q = queue::build(
        &c,
        queue::QueueInputs {
            graph: &g,
            reports: &[],
            ledger: &l,
            gates: &BTreeMap::new(),
            creates: vec![],
            create_suppressed: vec![],
            prs: &[],
            runs: &[],
        },
    );
    let it = q
        .items
        .iter()
        .find(|i| i.subject.as_deref() == Some("agentdb:ppr-fallback"))
        .expect("re-queued");
    assert!(it.evidence[0]
        .evidence
        .starts_with("re-queued: sublinear-time-solver@1.8.0"));
    // Non-heavy rejections are never re-opened.
    l.entries[0].reason = Some("dependency cycle".into());
    let q = queue::build(
        &c,
        queue::QueueInputs {
            graph: &g,
            reports: &[],
            ledger: &l,
            gates: &BTreeMap::new(),
            creates: vec![],
            create_suppressed: vec![],
            prs: &[],
            runs: &[],
        },
    );
    assert!(q.items.is_empty());
}

#[test]
fn renamed_repos_share_one_lane_and_one_ledger_history() {
    let c = cfg();
    let mut g = graph_with(&[("ruvnet/ruflo", 1000)]);
    g.repo_aliases
        .insert("ruvnet/claude-flow".into(), "ruvnet/ruflo".into());
    let mut ledger = Ledger::default();
    ledger.entries.push(entry(
        "ruvnet/claude-flow",
        Some("@ruvector/mincut-wasm"),
        Some("@ruvector/mincut"),
        WorkKind::Replace,
        SignalKind::MissingPackage,
        Outcome::InProgress,
    ));
    // The same repo scanned under both names.
    let reports = vec![report_for("ruvnet/claude-flow"), report_for("ruvnet/ruflo")];
    let q = queue::build(
        &c,
        queue::QueueInputs {
            graph: &g,
            reports: &reports,
            ledger: &ledger,
            gates: &BTreeMap::new(),
            creates: vec![],
            create_suppressed: vec![],
            prs: &[],
            runs: &[],
        },
    );
    assert!(q.busy_repos.contains_key("ruvnet/ruflo"));
    assert!(q.lanes.is_empty());
    assert!(q.items.iter().all(|i| i.repo == "ruvnet/ruflo"));
    let ids: BTreeSet<_> = q.items.iter().map(|i| i.id.clone()).collect();
    assert_eq!(
        ids.len(),
        q.items.len(),
        "no duplicate items from the two names"
    );
    assert!(q.items.iter().all(|i| i.evidence.len()
        == i.evidence
            .iter()
            .map(|s| (&s.file, s.line))
            .collect::<BTreeSet<_>>()
            .len()));
}

#[test]
fn covering_pr_needs_the_full_name_not_a_common_word() {
    let open = vec![
        pr(
            4477,
            "feat: independently verify learning, model transfer and provider effects",
            "feat/verify",
        ),
        pr(
            4839,
            "fix(memory): exit non-zero when a refused sql.js save loses writes",
            "fix/memory",
        ),
    ];
    let mk = |cand: &str, subj: Option<&str>| WorkItem {
        id: "x".into(),
        kind: WorkKind::Replace,
        status: LaneStatus::Deferred,
        repo: "ruvnet/big".into(),
        subject: subj.map(String::from),
        candidate: Some(cand.into()),
        signal_kind: SignalKind::StubMarker,
        score: 0.0,
        evidence: vec![],
        gates: vec![],
        disclosure: "public_pr".into(),
        required_test: String::new(),
        brief: String::new(),
    };
    assert!(prs::covering(&mk("@ruvector/learning-wasm", None), &open).is_none());
    assert!(prs::covering(
        &mk(
            "@claude-flow/plugin-gastown-bridge",
            Some("release:open_range:@claude-flow/memory")
        ),
        &open
    )
    .is_none());
    let wired = vec![pr(
        1364,
        "ruvector 0.3.12: wire @ruvector/mincut-wasm",
        "fix/x",
    )];
    assert!(prs::covering(&mk("@ruvector/mincut-wasm", None), &wired).is_some());
}

#[test]
fn security_fix_unreachable_and_zero_caret_pins() {
    let mut l = Ledger::default();
    let mut e = entry(
        "ruvnet/metaharness",
        Some("@metaharness/darwin"),
        Some("advisory:GHSA-x"),
        WorkKind::Security,
        SignalKind::Security,
        Outcome::Shipped,
    );
    e.version = Some("@metaharness/darwin@0.13.0, metaharness@0.4.20".into());
    l.entries.push(e);
    let patched = release::patched_versions(&l);
    assert_eq!(
        patched.get("@metaharness/darwin").map(String::as_str),
        Some("0.13.0")
    );
    let mut g = graph_with(&[
        ("ruvnet/ruvector", 10),
        ("ruvnet/ruflo", 10),
        ("ruvnet/metaharness", 10),
        ("ruvnet/x", 10),
    ]);
    let set = |g: &mut Graph, name: &str, f: &dyn Fn(&mut Package)| {
        f(g.packages.iter_mut().find(|p| p.name == name).unwrap())
    };
    set(&mut g, "ruvector", &|p| {
        p.dep_ranges
            .insert("@metaharness/darwin".into(), "^0.9.1".into());
    });
    set(&mut g, "ruflo", &|p| {
        p.dep_ranges
            .insert("@metaharness/darwin".into(), "^0.13.0".into());
    });
    set(&mut g, "metaharness", &|p| {
        p.name = "@ruvector/rvf".into();
        p.version = "0.4.2".into();
    });
    set(&mut g, "x", &|p| {
        p.name = "@ruvector/old-user".into();
        p.dep_ranges.insert("@ruvector/rvf".into(), "^0.3.0".into());
    });
    let s = release::range_signals(&g, &patched, &cfg().policy.npm);
    let subj: BTreeSet<_> = s.iter().filter_map(|x| x.subject.clone()).collect();
    assert!(subj.contains("release:fix_unreachable:ruvector->@metaharness/darwin"));
    assert!(
        !subj.iter().any(|x| x.contains("ruflo->")),
        "^0.13.0 admits the fix"
    );
    assert!(subj.contains("release:zero_caret:@ruvector/old-user->@ruvector/rvf"));
    let fix = s
        .iter()
        .find(|x| x.keyword.as_deref() == Some("fix_unreachable"))
        .unwrap();
    assert_eq!(
        (fix.repo.as_str(), fix.severity.as_deref()),
        ("ruvnet/ruvector", Some("high"))
    );
}

#[test]
fn fix_unreachable_ignores_ranges_that_start_past_the_fix() {
    // Regression: a security fix shipped in @ruvector/attention@2.2.3 and a
    // dependent declares "^2.3.0". The range excludes 2.2.3 itself, but every
    // version it admits descends from the fix, so it must not be flagged.
    let mut l = Ledger::default();
    let mut e = entry(
        "ruvnet/ruvector",
        Some("@ruvector/attention"),
        Some("advisory:GHSA-y"),
        WorkKind::Security,
        SignalKind::Security,
        Outcome::Shipped,
    );
    e.version = Some("@ruvector/attention@2.2.3".into());
    l.entries.push(e);
    let patched = release::patched_versions(&l);
    let mut g = graph_with(&[("ruvnet/ruvector", 10), ("ruvnet/ruflo", 10)]);
    let set = |g: &mut Graph, name: &str, f: &dyn Fn(&mut Package)| {
        f(g.packages.iter_mut().find(|p| p.name == name).unwrap())
    };
    set(&mut g, "ruvector", &|p| {
        p.dep_ranges
            .insert("@ruvector/attention".into(), "^2.3.0".into());
    });
    set(&mut g, "ruflo", &|p| {
        p.dep_ranges
            .insert("@ruvector/attention".into(), "^2.2.0 <2.2.3".into());
    });
    let s = release::range_signals(&g, &patched, &cfg().policy.npm);
    let subj: BTreeSet<_> = s.iter().filter_map(|x| x.subject.clone()).collect();
    assert!(
        !subj
            .iter()
            .any(|x| x.contains("ruvector->@ruvector/attention")),
        "^2.3.0 descends from the 2.2.3 fix: {subj:?}"
    );
    assert!(
        subj.contains("release:fix_unreachable:ruflo->@ruvector/attention"),
        "a range capped below the fix is still flagged: {subj:?}"
    );
}

#[test]
fn live_ledger_lives_outside_the_repo_and_is_seeded_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = Config::embedded(dir.path().to_path_buf()).unwrap();
    c.ledger_path = Some(dir.path().join("ledger.jsonl"));
    let p = c.live_ledger().unwrap().unwrap();
    let seeded = Ledger::load(&p).unwrap();
    assert_eq!(
        seeded.entries.len(),
        Ledger::parse(EMBEDDED_LEDGER).unwrap().entries.len()
    );
    Ledger::append(
        &p,
        &entry(
            "ruvnet/x",
            Some("y"),
            None,
            WorkKind::Integrate,
            SignalKind::StubMarker,
            Outcome::Shipped,
        ),
    )
    .unwrap();
    // A second use never re-seeds over live history.
    let p2 = c.live_ledger().unwrap().unwrap();
    assert_eq!(
        Ledger::load(&p2).unwrap().entries.len(),
        seeded.entries.len() + 1
    );
}

#[test]
fn vuln_evidence_distinguishes_nested_stale_copies() {
    use ruvector_weaver::vuln::{resolved_copies, verdict};
    let tree = serde_json::json!({"dependencies": {
        "@ruvector/sona": {"version": "0.1.10"},
        "agentdb": {"version": "3.4.0", "dependencies": {"@ruvector/sona": {"version": "0.1.7"}}}
    }});
    let c = resolved_copies(&tree, "@ruvector/sona", "<0.1.9");
    assert_eq!(c.len(), 2);
    let (bad, why) = verdict("@ruvector/sona", &c);
    assert!(bad && why.contains("agentdb > @ruvector/sona (0.1.7, nested stale copy)"));
    assert!(!why.contains("0.1.10"));
    let clean = serde_json::json!({"dependencies": {"@ruvector/sona": {"version": "0.1.10"}}});
    let (bad, why) = verdict(
        "@ruvector/sona",
        &resolved_copies(&clean, "@ruvector/sona", "<0.1.9"),
    );
    assert!(!bad && why.contains("outside the affected range"));
}

#[test]
fn invalid_resolution_names_the_winning_consumer() {
    use ruvector_weaver::resolve;
    let tree = serde_json::json!({"dependencies": {
        "ruvector": {"version": "0.3.13", "dependencies": {
            "@ruvector/ruvllm": {"version": "2.7.0", "invalid": "\"^4.1.0\" from node_modules/ruvector"}
        }},
        "metaharness": {"version": "0.4.20", "dependencies": {
            "@ruvector/ruvllm": {"version": "2.7.0"}
        }}
    }});
    let mut g = graph_with(&[("ruvnet/ruvector", 10), ("ruvnet/metaharness", 10)]);
    for p in &mut g.packages {
        let r = if p.name == "ruvector" {
            "^4.1.0"
        } else {
            "^2.5.6"
        };
        p.dep_ranges.insert("@ruvector/ruvllm".into(), r.into());
    }
    let s = resolve::signals(&g, "ruvector", &tree);
    assert_eq!(s.len(), 1);
    assert_eq!(
        s[0].repo, "ruvnet/metaharness",
        "the lane belongs to the range that must move"
    );
    assert_eq!(
        s[0].subject.as_deref(),
        Some("release:invalid_resolution:ruvector->@ruvector/ruvllm")
    );
    assert!(s[0].evidence.ends_with("won by metaharness"));
    let clean = serde_json::json!({"dependencies": {"ruvector": {"version": "0.3.13"}}});
    assert!(resolve::signals(&g, "ruvector", &clean).is_empty());
}

#[test]
fn blocked_rejection_requeues_only_when_every_blocker_ships() {
    let c = cfg();
    let g = graph_with(&[("ruvnet/agentdb", 10), ("ruvnet/ruvector", 10)]);
    let mut l = Ledger::default();
    let mut e = entry(
        "ruvnet/agentdb",
        Some("@ruvector/rvf"),
        Some("agentdb:rvf-0.4-migration"),
        WorkKind::Integrate,
        SignalKind::StubMarker,
        Outcome::Rejected,
    );
    e.blocked_on = vec![
        "rvf:per-vector-metadata".into(),
        "rvf:vector-readback".into(),
    ];
    l.entries.push(e);
    let build = |l: &Ledger| {
        queue::build(
            &c,
            queue::QueueInputs {
                graph: &g,
                reports: &[],
                ledger: l,
                gates: &BTreeMap::new(),
                creates: vec![],
                create_suppressed: vec![],
                prs: &[],
                runs: &[],
            },
        )
    };
    let has = |q: &Queue| {
        q.items
            .iter()
            .any(|i| i.subject.as_deref() == Some("agentdb:rvf-0.4-migration"))
    };
    assert!(!has(&build(&l)), "no blocker shipped");
    l.entries.push(entry(
        "ruvnet/ruvector",
        Some("@ruvector/rvf"),
        Some("rvf:per-vector-metadata"),
        WorkKind::Replace,
        SignalKind::StubMarker,
        Outcome::Shipped,
    ));
    assert!(!has(&build(&l)), "one of two blockers shipped");
    assert_eq!(
        ruvector_weaver::requeue::open_blockers(&l, &l.entries[0]),
        vec!["rvf:vector-readback"]
    );
    l.entries.push(entry(
        "ruvnet/ruvector",
        Some("@ruvector/rvf"),
        Some("rvf:vector-readback"),
        WorkKind::Replace,
        SignalKind::StubMarker,
        Outcome::Shipped,
    ));
    let q = build(&l);
    let it = q
        .items
        .iter()
        .find(|i| i.subject.as_deref() == Some("agentdb:rvf-0.4-migration"))
        .expect("re-queued");
    assert!(it.evidence[0]
        .evidence
        .starts_with("re-queued: every blocker shipped"));
    // blocked_on round-trips through JSONL and stays off entries without it.
    let line = serde_json::to_string(&l.entries[0]).unwrap();
    assert!(line.contains("\"blocked_on\""));
    assert!(!serde_json::to_string(&l.entries[1])
        .unwrap()
        .contains("blocked_on"));
}

#[test]
fn saturated_repos_are_deferred_and_only_owned_superseded_runs_are_cancelled() {
    use ruvector_weaver::saturation::{self, RepoRuns, RunRef};
    let run = |id: u64, wf: &str, br: &str, n: u64| RunRef {
        id,
        workflow: wf.into(),
        branch: br.into(),
        run_number: n,
        status: "queued".into(),
    };
    let active = vec![
        run(1, "Build Native", "weaver/lane-a", 10),
        run(2, "Build Native", "weaver/lane-a", 11),
        run(3, "Build Native", "main", 7),
        run(4, "Build Native", "main", 8),
        run(5, "CI", "feat/someone-else", 1),
        run(6, "CI", "feat/someone-else", 2),
        run(7, "CI", "weaver/lane-a", 3),
    ];
    let owned = cfg().policy.lp.owned_branch_prefixes.clone();
    let sup = saturation::superseded(&active, &owned);
    assert_eq!(
        sup.iter().map(|r| r.id).collect::<Vec<_>>(),
        vec![1],
        "only the older owned run of the same workflow+branch"
    );
    let c = cfg();
    let reports = vec![report_for("ruvnet/big"), report_for("ruvnet/small")];
    let g = graph_with(&[("ruvnet/big", 1000), ("ruvnet/small", 10)]);
    let runs = vec![
        RepoRuns {
            repo: "ruvnet/big".into(),
            fetched_at: String::new(),
            queued: 30,
            in_progress: 4,
            superseded: sup,
        },
        RepoRuns {
            repo: "ruvnet/small".into(),
            fetched_at: String::new(),
            queued: 1,
            in_progress: 0,
            superseded: vec![],
        },
    ];
    let q = queue::build(
        &c,
        queue::QueueInputs {
            graph: &g,
            reports: &reports,
            ledger: &Ledger::default(),
            gates: &BTreeMap::new(),
            creates: vec![],
            create_suppressed: vec![],
            prs: &[],
            runs: &runs,
        },
    );
    assert!(q.busy_repos["ruvnet/big"][0].starts_with("gha-saturated: 30 queued + 4 in progress"));
    assert!(q.lanes.iter().all(|l| l.repo != "ruvnet/big"));
    assert!(q.lanes.iter().any(|l| l.repo == "ruvnet/small"));
    assert_eq!(q.direct.len(), 1);
    let cancel = q.items.iter().find(|i| i.id == q.direct[0]).unwrap();
    assert!(cancel.brief.contains("gh run cancel") && cancel.brief.contains("for id in 1;"));
    assert!(!cancel.brief.contains(" 3 ") && !cancel.brief.contains(" 5 "));
}

#[test]
fn leak_scan_allows_intentional_dirs_and_only_fails_new_paths() {
    use ruvector_weaver::checks::{classify_leaks, leak_result};
    let c = cfg();
    let files: Vec<String> = [
        ".claude/skills/a.md",
        ".claude/agents/b.md",
        "dist/index.js",
        ".env",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // agentic-flow ships .claude/ on purpose (config allowlist).
    let r = leak_result(&c, "agentic-flow", &files[..3], None).unwrap();
    assert!(r.passed, "{}", r.detail);
    assert!(r.detail.contains("2 allowlisted"));
    // Another package: .claude/ carried over from the previous version passes...
    let prev: Vec<String> = files[..3].to_vec();
    let r = leak_result(&c, "other-pkg", &files[..3], Some(("1.2.3", &prev))).unwrap();
    assert!(
        r.passed && r.detail.contains("2 carried over from 1.2.3"),
        "{}",
        r.detail
    );
    // ...but a NEW leak path (.env) fails, even alongside allowlisted ones.
    let r = leak_result(&c, "agentic-flow", &files, Some(("3.3.1", &prev))).unwrap();
    assert!(
        !r.passed && r.detail.contains("NEW leak paths: .env"),
        "{}",
        r.detail
    );
    let (new, carried, allowed) = classify_leaks(&files, &[], None).unwrap();
    assert_eq!((new.len(), carried.len(), allowed.len()), (4, 0, 0));
}

#[test]
fn no_cycle_uses_the_published_graph_and_ignores_repo_siblings() {
    use ruvector_weaver::checks::cycle_result;
    // agent-booster is published from ruvnet/agentic-flow and depends only on express.
    let mut g = graph_with(&[("ruvnet/agentic-flow", 10), ("ruvnet/agentic-flow", 10)]);
    g.packages[0].name = "agentic-flow".into();
    g.packages[0].dependencies = vec!["agent-booster".into()];
    g.packages[0]
        .dep_ranges
        .insert("agent-booster".into(), "^0.2.2".into());
    g.packages[1].name = "agent-booster".into();
    g.packages[1].dependencies = vec!["express".into()];
    g.consumer_repos = consumer_repos(&g.packages);
    let r = cycle_result(&g, "agentic-flow", "ruvnet/agentic-flow", None);
    assert!(r.passed, "intra-repo is not a cycle: {}", r.detail);
    let r = cycle_result(
        &g,
        "agent-booster",
        "ruvnet/agentic-flow",
        Some("agentic-flow"),
    );
    assert!(
        r.passed,
        "agent-booster does not depend on agentic-flow: {}",
        r.detail
    );
    // A real published cycle is still caught, with the declared ranges as evidence.
    g.packages[1].dependencies.push("agentic-flow".into());
    g.packages[1]
        .dep_ranges
        .insert("agentic-flow".into(), "^3.0.0".into());
    let r = cycle_result(
        &g,
        "agent-booster",
        "ruvnet/agentic-flow",
        Some("agentic-flow"),
    );
    assert!(
        !r.passed && r.detail.contains("agent-booster -> agentic-flow@^3.0.0"),
        "{}",
        r.detail
    );
}

// ---- package-less items -----------------------------------------------------

#[test]
fn packageless_items_are_never_gated() {
    use ruvector_weaver::brief::brief;
    use ruvector_weaver::gate::is_packageless;
    assert!(is_packageless("-"));
    assert!(is_packageless(" - "));
    assert!(is_packageless(""));
    assert!(!is_packageless("ruvector"));
    assert!(!is_packageless("@ruvector/core"));

    let mut item = WorkItem {
        id: "ci".into(),
        kind: WorkKind::Hygiene,
        status: LaneStatus::Deferred,
        repo: "ruvnet/big".into(),
        subject: Some("ci:main-red".into()),
        candidate: None,
        signal_kind: SignalKind::MainRed,
        score: 0.0,
        evidence: vec![],
        gates: vec![],
        disclosure: "public_pr".into(),
        required_test: String::new(),
        brief: String::new(),
    };
    let g = Graph::default();
    assert!(!brief(&item, &g, None).contains("weaver gate"));
    item.candidate = Some("-".into());
    assert!(!brief(&item, &g, None).contains("weaver gate"));
    item.candidate = Some("ruvector".into());
    assert!(brief(&item, &g, None).contains("weaver gate ruvector --consumer ruvnet/big"));
}

#[test]
fn gate_cli_skips_packageless_item() {
    let work = std::env::temp_dir().join(format!("weaver-gate-skip-{}", std::process::id()));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_weaver"))
        .args(["--work-dir"])
        .arg(&work)
        .args(["--cache-ttl-hours", "0", "gate", "-"])
        .output()
        .expect("run weaver");
    let _ = std::fs::remove_dir_all(&work);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("gate skipped"));
}
