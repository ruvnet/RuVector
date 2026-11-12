//! CLI regression tests for #1276 items 1, 3, 4, 5 and #1275 (exit codes).
use std::path::Path;
use std::process::{Command, Output};

fn rvf(args: &[&str], cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rvf"))
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap()
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn vectors_json(dir: &Path, n: usize, dim: usize) {
    let recs: Vec<String> = (0..n)
        .map(|i| {
            let v: Vec<String> = (0..dim).map(|d| format!("{}", i + d)).collect();
            format!("{{\"id\":{},\"vector\":[{}]}}", i + 1, v.join(","))
        })
        .collect();
    std::fs::write(dir.join("v.json"), format!("[{}]", recs.join(","))).unwrap();
}

fn setup(n: usize) -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    assert!(rvf(&["create", "-d", "4", "s.rvf"], d.path())
        .status
        .success());
    vectors_json(d.path(), n, 4);
    assert!(
        rvf(&["ingest", "-i", "v.json", "-b", "100", "s.rvf"], d.path())
            .status
            .success()
    );
    d
}

#[test]
fn truncated_store_status_and_query_exit_nonzero() {
    let d = setup(300);
    let full = std::fs::read(d.path().join("s.rvf")).unwrap();
    // Find the last manifest header (magic RVFS + type 5) and cut inside it.
    let mut last = 0;
    for p in (0..full.len().saturating_sub(64)).step_by(64) {
        if full[p..p + 4] == 0x5256_4653u32.to_le_bytes() && full[p + 5] == 5 {
            last = p;
        }
    }
    std::fs::write(d.path().join("cut.rvf"), &full[..last + 70]).unwrap();
    let o = rvf(&["status", "cut.rvf"], d.path());
    assert!(!o.status.success(), "status must fail: {}", text(&o));
    assert!(text(&o).contains("TruncatedSegment"), "{}", text(&o));
    let o = rvf(&["query", "cut.rvf", "-v", "1,2,3,4"], d.path());
    assert!(!o.status.success(), "query must fail: {}", text(&o));
    let o = rvf(&["inspect", "cut.rvf"], d.path());
    assert!(!o.status.success(), "inspect must fail: {}", text(&o));
    // Explicit opt-in recovers the last committed snapshot (and warns).
    let o = rvf(&["status", "--recover", "cut.rvf"], d.path());
    assert!(o.status.success(), "{}", text(&o));
    assert!(text(&o).contains("ends inside a segment"), "{}", text(&o));
    // The intact file is unaffected.
    assert!(rvf(&["status", "s.rvf"], d.path()).status.success());
}

#[test]
fn nan_and_inf_query_vectors_are_refused() {
    let d = setup(20);
    for bad in ["nan,0,0,0", "0,inf,0,0", "0,0,-inf,0"] {
        let o = rvf(&["query", "s.rvf", "-v", bad], d.path());
        assert!(!o.status.success(), "{bad}: {}", text(&o));
        assert!(text(&o).contains("not finite"), "{bad}: {}", text(&o));
    }
    assert!(rvf(&["query", "s.rvf", "-v", "1,0,0,0"], d.path())
        .status
        .success());
}

#[test]
fn fully_rejected_ingest_exits_nonzero_and_keeps_epoch() {
    let d = setup(20);
    let before = text(&rvf(&["status", "s.rvf"], d.path()));
    vectors_json(d.path(), 3, 2); // wrong dimension
    let o = rvf(&["ingest", "-i", "v.json", "s.rvf"], d.path());
    assert!(!o.status.success(), "{}", text(&o));
    assert!(text(&o).contains("Rejected:"), "{}", text(&o));
    let after = text(&rvf(&["status", "s.rvf"], d.path()));
    assert_eq!(before, after, "status (incl. epoch) unchanged");
}

#[test]
fn errors_name_the_error_code() {
    let d = setup(20);
    let o = rvf(&["query", "s.rvf", "-v", "1,2"], d.path());
    assert!(!o.status.success());
    assert!(text(&o).contains("DimensionMismatch"), "{}", text(&o));
}
