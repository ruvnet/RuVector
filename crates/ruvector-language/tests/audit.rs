use ruvector_language::{audit::*, *};
#[path = "../examples/support/memory.rs"]
mod memory;
fn options(interventions: Vec<Intervention>) -> AuditOptions {
    AuditOptions {
        interventions,
        max_execution_work: 1_000_000_000,
    }
}
fn incident() -> RunRequest {
    RunRequest {
        program: parser::parse(include_str!("../examples/incident.rvg")).unwrap(),
        snapshot: serde_json::from_str(include_str!("../examples/snapshot.json")).unwrap(),
        context: serde_json::from_str(include_str!("../examples/context.json")).unwrap(),
    }
}
#[test]
fn necessary_evidence_retracts_but_absence_does_not_refute() {
    let r = incident();
    let report = run(
        &r,
        &options(vec![Intervention::Observation {
            id: "db-error-1".into(),
        }]),
    )
    .unwrap();
    assert_eq!(report.trials[0].changes.len(), 2);
    for c in &report.trials[0].changes {
        assert_eq!(c.before, Some(Status::Supported));
        assert_eq!(c.after, Some(Status::Unknown));
    }
    assert!(report.execution_work <= report.reserved_execution_work);
    assert_eq!(r.snapshot.observations.len(), 2);
}
#[test]
fn redundant_observation_and_source_removal_have_different_effects() {
    let mut r = incident();
    let mut duplicate = r.snapshot.observations[0].clone();
    duplicate.id = "redundant".into();
    r.snapshot.observations.push(duplicate);
    let report = run(
        &r,
        &options(vec![
            Intervention::Observation {
                id: "db-error-1".into(),
            },
            Intervention::Source {
                id: "telemetry-a".into(),
            },
        ]),
    )
    .unwrap();
    assert!(report.trials[0].changes.is_empty());
    assert_eq!(report.trials[0].changed_decisions, 2);
    assert_eq!(report.trials[1].removed_records, 2);
    assert_eq!(report.trials[1].changes.len(), 2);
}
#[test]
fn graph_ablation_reports_missing_domain_separately_from_unknown() {
    let report = run(
        &incident(),
        &options(vec![Intervention::Edge {
            id: "api-db".into(),
        }]),
    )
    .unwrap();
    assert_eq!(report.trials[0].changes.len(), 2);
    assert!(report.trials[0].changes.iter().all(|c| c.after.is_none()));
}
#[test]
fn conflict_removal_preserves_remaining_negative_evidence() {
    let mut r = incident();
    let mut o = r.snapshot.observations[0].clone();
    o.id = "negative".into();
    o.value = false;
    r.snapshot.observations.push(o);
    let report = run(
        &r,
        &options(vec![Intervention::Observation {
            id: "db-error-1".into(),
        }]),
    )
    .unwrap();
    let c = report.trials[0]
        .changes
        .iter()
        .find(|c| c.predicate == "Failing")
        .unwrap();
    assert_eq!(
        (c.before.clone(), c.after.clone()),
        (Some(Status::Conflicting), Some(Status::Refuted))
    );
}
#[test]
fn replay_is_exact_and_report_identifies_changed_inputs() {
    let mut r = incident();
    let o = options(vec![Intervention::Source {
        id: "telemetry-a".into(),
    }]);
    let a = run(&r, &o).unwrap();
    assert_eq!(a, run(&r, &o).unwrap());
    r.context.now = 1100;
    let expired = run(&r, &o).unwrap();
    assert_ne!(a.context_hash, expired.context_hash);
    assert!(expired.trials[0].changes.is_empty());
}
#[test]
fn rejects_invalid_targets_duplicate_trials_and_excessive_budget() {
    let r = incident();
    let t = Intervention::Source {
        id: "telemetry-a".into(),
    };
    for o in [
        options(vec![]),
        options(vec![t.clone(); 65]),
        options(vec![t.clone(), t.clone()]),
        options(vec![Intervention::Edge {
            id: "missing".into(),
        }]),
    ] {
        assert_eq!(run(&r, &o).unwrap_err().code, "INVALID_AUDIT");
    }
    let mut o = options(vec![t]);
    o.max_execution_work = 1;
    assert_eq!(run(&r, &o).unwrap_err().code, "AUDIT_BUDGET");
    let mut denied = r;
    denied.context.capabilities.clear();
    assert_eq!(
        run(
            &denied,
            &options(vec![Intervention::Source {
                id: "telemetry-a".into()
            }])
        )
        .unwrap_err()
        .code,
        "UNAUTHORIZED"
    );
}
#[test]
fn refuses_prediction_even_when_not_emitted() {
    let mut r = incident();
    r.program.steps.insert(
        0,
        Step {
            bind: "unsafe".into(),
            op: Op::Predict {
                input: "x".into(),
                relation: "r".into(),
                model: "m".into(),
                k: 1,
            },
        },
    );
    assert_eq!(
        run(
            &r,
            &options(vec![Intervention::Edge {
                id: "api-db".into()
            }])
        )
        .unwrap_err()
        .code,
        "INVALID_AUDIT"
    );
}
#[test]
fn generated_memory_cases_match_independent_oracle_under_expiry_and_permutation() {
    for seed in 0..128 {
        let mut r = memory::request(32, seed);
        for now in [1000, 1100] {
            r.context.now = now;
            let a = runtime::execute(&r.program, &r.snapshot, &r.context, None).unwrap();
            assert_eq!(
                memory::actual(&a),
                memory::reference(&r),
                "seed {seed}, time {now}"
            );
            r.snapshot.nodes.reverse();
            r.snapshot.edges.reverse();
            r.snapshot.observations.reverse();
            r.snapshot.rules.reverse();
            let b = runtime::execute(&r.program, &r.snapshot, &r.context, None).unwrap();
            assert_eq!(a, b);
        }
    }
}
#[test]
fn generated_audit_deltas_match_full_independent_recomputation() {
    for seed in 0..32 {
        let r = memory::request(32, seed);
        let old = memory::reference(&r);
        let report = run(
            &r,
            &options(vec![
                Intervention::Source {
                    id: "primary".into(),
                },
                Intervention::Edge { id: "e0".into() },
            ]),
        )
        .unwrap();
        for trial in report.trials {
            let mut altered = r.clone();
            match trial.intervention {
                Intervention::Source { id } => {
                    altered.snapshot.observations.retain(|o| o.source != id)
                }
                Intervention::Edge { id } => altered.snapshot.edges.retain(|e| e.id != id),
                _ => unreachable!(),
            }
            let new = memory::reference(&altered);
            let keys: std::collections::BTreeSet<_> = old.keys().chain(new.keys()).collect();
            let expected: Vec<_> = keys
                .into_iter()
                .filter(|k| old.get(*k) != new.get(*k))
                .map(|k| Change {
                    subject: k.0.clone(),
                    predicate: k.1.clone(),
                    before: old.get(k).cloned(),
                    after: new.get(k).cloned(),
                })
                .collect();
            assert_eq!(trial.changes, expected);
        }
    }
}
#[test]
fn audit_cli_runs_real_process_and_rejects_ambiguous_flags() {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let r = incident();
    let mut child = Command::new(env!("CARGO_BIN_EXE_rvg-core"))
        .args([
            "audit",
            "--request",
            "-",
            "--audit-options",
            "examples/audit-options.json",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&r).unwrap())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let report: AuditReport = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report.trials.len(), 3);
    let out = Command::new(env!("CARGO_BIN_EXE_rvg-core"))
        .args(["audit", "--request", "missing", "--models", "missing"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8(out.stderr).unwrap().contains("USAGE"));
}
