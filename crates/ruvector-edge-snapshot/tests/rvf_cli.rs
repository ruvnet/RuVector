//! Cross-check an export against the real `rvf` CLI (rvf-runtime reader).
//!
//! Build the CLI once from this tree and point `RVF_CLI` at it:
//!
//! ```text
//! cd crates/rvf && CARGO_TARGET_DIR=/data/scratch/ruvector-edge-target/m3-rvfcli cargo build -p rvf-cli
//! RVF_CLI=/data/scratch/ruvector-edge-target/m3-rvfcli/debug/rvf \
//!   cargo test -p ruvector-edge-snapshot --test rvf_cli -- --nocapture
//! ```
//!
//! The export is a witnessed one (the gateway's `:export` shape: sidecar +
//! vec pairs, the `RVEW` export-witness `PROFILE_SEG`, then the runtime
//! manifest), verified offline with `verify_export` first. The test runs
//! `rvf inspect <file> --json` and
//! `rvf query <file> --vector <row 17> --k 3 --json` and asserts the runtime
//! opened the file (no checksum or manifest error), saw the dimension, the
//! row count and the `Profile`/`Vec` segments, and returns runtime id 17 (the
//! export ordinal of `doc-000000017`) at distance 0. Without `RVF_CLI` the
//! test only writes the file and says so.

mod common;
use common::*;
use ruvector_edge_snapshot::*;
use std::process::Command;

#[test]
fn rvf_cli_reads_the_export() {
    let dim = 8;
    let data = rows(250, dim, 77);
    let mut file = Vec::new();
    let mut sink = |b: &[u8]| file.extend_from_slice(b);
    let mut x = RvfExporter::new(dim, Metric::L2, 64).unwrap();
    for r in &data {
        x.push(r, &mut sink).unwrap();
    }
    let id = ExportIdentity::new(TENANT_A, UID, [1; 32]).unwrap();
    let signer = TestSigner::fixed(9);
    x.finish_witnessed(&mut sink, &id, Some(&signer)).unwrap();
    let mut v = Ed25519Verifier::new();
    v.add_tenant_key(TENANT_A, "snap-k1", signer.public())
        .unwrap();
    verify_export(&file, TENANT_A, SignaturePolicy::Required(&v)).unwrap();
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("edge-export.rvf");
    std::fs::write(&path, &file).unwrap();

    let Ok(cli) = std::env::var("RVF_CLI") else {
        eprintln!(
            "RVF_CLI not set; wrote {} ({} bytes) without cross-check",
            path.display(),
            file.len()
        );
        return;
    };
    let run = |args: &[&str]| {
        let o = Command::new(&cli).args(args).output().expect("run rvf");
        let text =
            String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr);
        assert!(o.status.success(), "rvf {args:?} failed: {text}");
        eprintln!("$ rvf {}\n{text}", args.join(" "));
        text.split_whitespace().collect::<String>()
    };
    let p = path.to_str().unwrap();
    let inspect = run(&["inspect", p, "--json"]);
    assert!(inspect.contains("\"dimension\":8"), "{inspect}");
    assert!(inspect.contains("\"total_vectors\":250"), "{inspect}");
    assert!(
        inspect.contains("\"seg_type_name\":\"Profile\""),
        "{inspect}"
    );
    assert!(inspect.contains("\"seg_type_name\":\"Vec\""), "{inspect}");

    let q: Vec<String> = data[17].values.iter().map(|v| format!("{v:?}")).collect();
    let query = run(&[
        "query",
        p,
        &format!("--vector={}", q.join(",")),
        "--k",
        "3",
        "--json",
    ]);
    // Observed: {"count":3,"results":[{"distance":0.0,"id":17},…]}
    assert!(
        query.contains("\"results\":[{\"distance\":0.0,\"id\":17}"),
        "{query}"
    );
}
