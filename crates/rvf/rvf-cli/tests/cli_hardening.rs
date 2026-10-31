//! CLI behaviour checks run against the real `rvf` binary.
use std::process::Command;

fn rvf(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rvf")).args(args).output().unwrap();
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

fn store(dir: &std::path::Path) -> String {
    let p = dir.join("s.rvf").to_string_lossy().to_string();
    let v = dir.join("v.json");
    std::fs::write(&v, r#"[{"id":1,"vector":[1,0,0,0]},{"id":2,"vector":[0,1,0,0]}]"#).unwrap();
    assert!(rvf(&["create", "-d", "4", &p]).0);
    assert!(rvf(&["ingest", "-i", v.to_str().unwrap(), &p]).0);
    p
}

#[test]
fn non_finite_query_vectors_are_refused() {
    let d = tempfile::tempdir().unwrap();
    let p = store(d.path());
    for bad in ["nan,0,0,0", "inf,0,0,0", "0,-inf,0,0"] {
        let (ok, out) = rvf(&["query", "-v", bad, "-k", "1", &p]);
        assert!(!ok, "query {bad} succeeded: {out}");
        assert!(out.contains("finite"), "{out}");
    }
    assert!(rvf(&["query", "-v", "1,0,0,0", "-k", "1", &p]).0);
}

#[test]
fn errors_name_the_code() {
    let d = tempfile::tempdir().unwrap();
    let p = store(d.path());
    let (ok, out) = rvf(&["query", "-v", "1,0", "-k", "1", &p]);
    assert!(!ok);
    assert!(out.contains("DimensionMismatch"), "error should name the code: {out}");
}

#[test]
fn verify_witness_exit_status_matches_its_verdict() {
    let d = tempfile::tempdir().unwrap();
    let p = store(d.path());
    let (ok, out) = rvf(&["verify-witness", &p]);
    let failed = out.contains("failed verification");
    assert_eq!(ok, !failed, "exit status success={ok} but output says failed={failed}:\n{out}");
    let (ok, out) = rvf(&["verify-witness", "--json", &p]);
    let invalid = out.contains("\"invalid\"");
    assert_eq!(ok, !invalid, "--json: exit status success={ok} but status invalid={invalid}:\n{out}");
}
