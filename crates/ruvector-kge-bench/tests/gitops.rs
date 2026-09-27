//! `GitCli` against a local bare "origin" in a temp dir (never the real
//! remote): the tag gate refuses a missing and an unsigned tag, the ledger is
//! read from origin's branch (not the working tree), and a stale push is
//! rejected as a non-fast-forward.

use ruvector_kge_bench::final_mode::{authorize, FinalGit, FinalRequest};
use ruvector_kge_bench::gitops::{parse_signers, parse_verify_output, GitCli};
use ruvector_kge_bench::ledger::{LEDGER_BRANCH, LEDGER_PATH, SIGNERS_PATH};
use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> String {
    let o = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stdout).trim().to_string()
}

fn setup() -> (tempfile::TempDir, std::path::PathBuf) {
    let d = tempfile::tempdir().unwrap();
    let (origin, work) = (d.path().join("origin.git"), d.path().join("work"));
    git(
        d.path(),
        &[
            "init",
            "-q",
            "--bare",
            "-b",
            "main",
            &origin.to_string_lossy(),
        ],
    );
    git(
        d.path(),
        &["init", "-q", "-b", "main", &work.to_string_lossy()],
    );
    for (k, v) in [
        ("user.name", "t"),
        ("user.email", "t@example.invalid"),
        ("commit.gpgsign", "false"),
        ("tag.gpgsign", "false"),
    ] {
        git(&work, &["config", k, v]);
    }
    git(
        &work,
        &["remote", "add", "origin", &origin.to_string_lossy()],
    );
    std::fs::create_dir_all(work.join("npm/packages/kge/bench/results/final")).unwrap();
    std::fs::write(work.join(SIGNERS_PATH), "# test\nABCDEF0123\n").unwrap();
    std::fs::write(work.join(LEDGER_PATH), "").unwrap();
    git(&work, &["add", "-A"]);
    git(&work, &["commit", "-q", "-m", "init"]);
    git(
        &work,
        &[
            "push",
            "-q",
            "origin",
            "main",
            &format!("main:{LEDGER_BRANCH}"),
        ],
    );
    (d, work)
}

fn req() -> FinalRequest {
    FinalRequest {
        dataset: "wn18rr".into(),
        seed: 0,
        config_hash: "H".into(),
        config_is_grid: true,
        checkpoint_sha256: "C".into(),
    }
}

#[test]
fn tag_gate_on_a_local_origin() {
    let (_d, work) = setup();
    let g = GitCli::new(&work);
    // No tag on origin.
    assert!(g.origin_tag("kge-prereg-v1").unwrap().is_none());
    let e = authorize(&g, &req()).unwrap_err();
    assert!(
        format!("{e:#}").contains("does not exist on origin"),
        "{e:#}"
    );
    // An annotated but unsigned tag, pushed: verify-tag fails -> refused.
    git(&work, &["tag", "-a", "kge-prereg-v1", "-m", "prereg"]);
    // A local-only tag would not count; push it.
    git(&work, &["push", "-q", "origin", "kge-prereg-v1"]);
    let t = g.origin_tag("kge-prereg-v1").unwrap().unwrap();
    assert_eq!(t.commit, git(&work, &["rev-parse", "HEAD"]));
    assert_ne!(t.tag_object, t.commit);
    assert!(
        g.verify_tag_signer(&t).is_err(),
        "unsigned tag must fail verification"
    );
    let e = authorize(&g, &req()).unwrap_err();
    assert!(format!("{e:#}").contains("signature/signer"), "{e:#}");
    assert!(g.is_ancestor_of_head(&t.commit).unwrap());
}

#[test]
fn ledger_reads_origin_and_rejects_stale_push() {
    let (_d, work) = setup();
    let g = GitCli::new(&work);
    let base = g.fetch_origin_ledger().unwrap();
    assert_eq!(base.ledger, "");
    assert!(base.selection.is_none());
    // A working-tree edit is invisible to the gate.
    std::fs::write(work.join(LEDGER_PATH), "{\"garbage\":1}\n").unwrap();
    assert_eq!(g.fetch_origin_ledger().unwrap().ledger, "");
    g.push_ledger_line(&base, "{\"line\":1}").unwrap();
    let now = g.fetch_origin_ledger().unwrap();
    assert_eq!(now.ledger, "{\"line\":1}\n");
    // Pushing on top of the old base (a concurrent writer got there first) is rejected.
    assert!(g.push_ledger_line(&base, "{\"line\":2}").is_err());
    assert_eq!(g.fetch_origin_ledger().unwrap().ledger, "{\"line\":1}\n");
    // The caller's branch and working-tree ledger are untouched.
    assert_eq!(
        std::fs::read_to_string(work.join(LEDGER_PATH)).unwrap(),
        "{\"garbage\":1}\n"
    );
    assert_eq!(git(&work, &["rev-parse", "--abbrev-ref", "HEAD"]), "main");
}

#[test]
fn signer_parsing() {
    assert_eq!(
        parse_signers("# c\nab cd 01\n\nEF # tail\n"),
        vec!["ABCD01", "EF"]
    );
    let gpg = "[GNUPG:] NEWSIG\n[GNUPG:] VALIDSIG AAAA1111 2026-09-27 1790000000 0 4 0 1 10 00 BBBB2222\n";
    assert_eq!(parse_verify_output(gpg), vec!["AAAA1111", "BBBB2222"]);
    let ssh = "Good \"git\" signature for x@y with ED25519 key SHA256:abcDEF\n";
    assert_eq!(parse_verify_output(ssh), vec!["SHA256:ABCDEF"]);
    assert!(
        parse_verify_output("Could not verify signature. with ED25519 key SHA256:x").is_empty()
    );
}
