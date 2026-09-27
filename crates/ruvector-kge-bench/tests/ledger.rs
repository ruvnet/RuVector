//! Ledger CI rules and the `kge-ledger-check` binary.

use ruvector_kge_bench::ledger::{check, Line, Selection};
use std::process::Command;

fn intent(ds: &str, seed: u64, hash: &str) -> String {
    Line::Intent {
        dataset: ds.into(),
        seed,
        config_hash: hash.into(),
        tag_sha: "T".into(),
        checkpoint_sha256: "C".into(),
        utc: "u".into(),
    }
    .to_json_line()
}
fn result(ds: &str, seed: u64, hash: &str) -> String {
    Line::Result {
        dataset: ds.into(),
        seed,
        config_hash: hash.into(),
        tag_sha: "T".into(),
        checkpoint_sha256: "C".into(),
        receipt_sha256: "R".into(),
        bottom_ranks_sha256: "B".into(),
        random_ranks_sha256: "D".into(),
        utc: "u".into(),
    }
    .to_json_line()
}
fn sel() -> Selection {
    Selection::parse(
        r#"{"datasets":{"wn18rr":{"config_hash":"H"},"fb15k237":{"config_hash":"F"}}}"#,
    )
    .unwrap()
}
fn join(lines: &[String]) -> String {
    lines.iter().map(|l| format!("{l}\n")).collect()
}

#[test]
fn rules() {
    let ok = join(&[
        intent("wn18rr", 0, "H"),
        result("wn18rr", 0, "H"),
        intent("fb15k237", 4, "F"),
    ]);
    check(&ok, Some(&sel()), None, Some("T")).unwrap();
    check("", None, None, None).unwrap();
    let bad: Vec<(&str, String)> = vec![
        (
            "duplicate intent key",
            join(&[intent("wn18rr", 0, "H"), intent("wn18rr", 0, "H")]),
        ),
        (
            "duplicate result key",
            join(&[
                intent("wn18rr", 0, "H"),
                result("wn18rr", 0, "H"),
                result("wn18rr", 0, "H"),
            ]),
        ),
        ("result without intent", join(&[result("wn18rr", 1, "H")])),
        ("seed outside 0..4", join(&[intent("wn18rr", 5, "H")])),
        ("hpo seed", join(&[intent("wn18rr", 100, "H")])),
        ("non-selected hash", join(&[intent("wn18rr", 0, "X")])),
        ("unknown dataset", join(&[intent("codexm", 0, "H")])),
        ("malformed", "{not json}\n".into()),
        (
            "unknown kind",
            "{\"kind\":\"final\",\"dataset\":\"wn18rr\",\"seed\":0}\n".into(),
        ),
    ];
    for (why, text) in bad {
        assert!(
            check(&text, Some(&sel()), None, Some("T")).is_err(),
            "{why}"
        );
    }
    // Tag SHA mismatch and missing selection.
    assert!(check(&ok, Some(&sel()), None, Some("OTHER")).is_err());
    assert!(check(&ok, None, None, None).is_err());
    // Append-only.
    let prev = join(&[intent("wn18rr", 0, "H")]);
    check(&ok, Some(&sel()), Some(&prev), None).unwrap();
    let edited = ok.replacen("\"seed\":0", "\"seed\":1", 1);
    assert!(
        check(&edited, Some(&sel()), Some(&prev), None).is_err(),
        "modified line"
    );
    assert!(
        check("", Some(&sel()), Some(&prev), None).is_err(),
        "deleted line"
    );
}

#[test]
fn ledger_check_binary() {
    let bin = env!("CARGO_BIN_EXE_kge-ledger-check");
    let d = tempfile::tempdir().unwrap();
    let (l, s, p) = (
        d.path().join("ledger.jsonl"),
        d.path().join("selection.json"),
        d.path().join("prev.jsonl"),
    );
    std::fs::write(&s, r#"{"datasets":{"wn18rr":{"config_hash":"H"}}}"#).unwrap();
    std::fs::write(
        &l,
        join(&[intent("wn18rr", 0, "H"), result("wn18rr", 0, "H")]),
    )
    .unwrap();
    let run = |extra: &[&std::path::Path]| {
        let mut c = Command::new(bin);
        c.arg(&l).arg("--selection").arg(&s);
        if let Some(prev) = extra.first() {
            c.arg("--previous").arg(prev);
        }
        c.status().unwrap().success()
    };
    assert!(run(&[]));
    std::fs::write(
        &l,
        join(&[intent("wn18rr", 0, "H"), intent("wn18rr", 0, "H")]),
    )
    .unwrap();
    assert!(!run(&[]), "duplicate key must fail");
    std::fs::write(&p, join(&[intent("wn18rr", 1, "H")])).unwrap();
    std::fs::write(&l, join(&[intent("wn18rr", 0, "H")])).unwrap();
    assert!(!run(&[&p]), "rewritten history must fail");
    // The committed (empty) ledger passes.
    let repo_ledger = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../npm/packages/kge/bench/results/final/ledger.jsonl");
    assert!(Command::new(bin)
        .arg(repo_ledger)
        .status()
        .unwrap()
        .success());
}
