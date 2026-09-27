//! The shared JS/Rust norm contract (ADR-008 §2): every case in
//! `npm/packages/typesafe/bench/test/norm-golden.json` must match exactly,
//! both the normalized text and sha256(norm).

use std::path::Path;

use ruvector_typesafe_train::norm::{norm, sha256_hex, sha256_norm};

#[test]
fn all_52_bench_golden_vectors_match() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../npm/packages/typesafe/bench/test/norm-golden.json");
    let s = std::fs::read_to_string(&p).expect("norm-golden.json is part of the repo contract");
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    let cases = v["cases"].as_array().unwrap();
    assert_eq!(
        cases.len(),
        52,
        "golden file changed size — re-check the contract"
    );
    let mut fails = Vec::new();
    for c in cases {
        let (name, input, want, sha) = (
            c["name"].as_str().unwrap(),
            c["input"].as_str().unwrap(),
            c["norm"].as_str().unwrap(),
            c["sha256"].as_str().unwrap(),
        );
        let got = norm(input);
        if got != want || sha256_norm(input) != sha || sha256_hex(want.as_bytes()) != sha {
            fails.push(format!("{name}: norm({input:?}) = {got:?}, want {want:?}"));
        }
    }
    assert!(
        fails.is_empty(),
        "{} golden failures:\n{}",
        fails.len(),
        fails.join("\n")
    );
}
