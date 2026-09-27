//! Checkpoint + resume is bit-identical to an uninterrupted run; checkpoints
//! are verified and bound to their run; exports carry no triples.

mod common;

use common::{synth_dataset, synth_run};
use ruvector_kge_bench::export::load_tables;
use ruvector_kge_bench::receipt::validate;
use ruvector_kge_bench::runner::{run, RunOptions};
use serde_json::Value;
use std::path::Path;

fn opts(out: &Path, resume: bool, stop: Option<usize>) -> RunOptions {
    RunOptions {
        out: out.into(),
        resume,
        stop_after_epoch: stop,
        repo: ".".into(),
        verbose: false,
    }
}

/// Per-epoch records without the wall-clock fields.
fn epochs_sans_time(r: &Value) -> Vec<Value> {
    r["epochs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let mut e = e.clone();
            e.as_object_mut().unwrap().remove("train_secs");
            e.as_object_mut().unwrap().remove("eval_secs");
            e
        })
        .collect()
}

#[test]
fn resume_is_bitwise_identical() {
    let ds = synth_dataset();
    let cfg = synth_run(6);
    let d = tempfile::tempdir().unwrap();
    let (a, b) = (d.path().join("a"), d.path().join("b"));

    let ra = run(&cfg, &ds, &opts(&a, false, None)).unwrap();
    let rb1 = run(&cfg, &ds, &opts(&b, false, Some(3))).unwrap();
    assert_eq!(rb1["stopped"], "interrupted");
    assert_eq!(rb1["epochs"].as_array().unwrap().len(), 3);
    let rb = run(&cfg, &ds, &opts(&b, true, None)).unwrap();

    for r in [&ra, &rb1, &rb] {
        assert_eq!(validate(r).unwrap(), "valid");
    }
    assert_eq!(rb["stopped"], "max_epochs");
    assert_eq!(rb["resumed_at"], serde_json::json!([3]));
    assert_eq!(
        epochs_sans_time(&ra),
        epochs_sans_time(&rb),
        "per-epoch losses and valid metrics"
    );
    assert_eq!(
        ra["best"], rb["best"],
        "best-on-valid epoch, metrics, per-query ranks and weights hash"
    );
    for sub in ["last", "best"] {
        assert_eq!(
            std::fs::read(a.join(sub).join("weights.bin")).unwrap(),
            std::fs::read(b.join(sub).join("weights.bin")).unwrap(),
            "{sub} tables must be bitwise identical"
        );
    }
    // Starting fresh over an existing checkpoint is refused.
    assert!(run(&cfg, &ds, &opts(&a, false, None)).is_err());
}

#[test]
fn checkpoint_is_bound_and_verified() {
    let ds = synth_dataset();
    let cfg = synth_run(4);
    let d = tempfile::tempdir().unwrap();
    let out = d.path().join("run");
    run(&cfg, &ds, &opts(&out, false, Some(2))).unwrap();

    // Different recipe -> different config hash -> refused.
    let mut other = cfg.clone();
    other.recipe.as_mut().unwrap().lr = 0.2;
    let e = run(&other, &ds, &opts(&out, true, None)).unwrap_err();
    assert!(e.to_string().contains("different run config"), "{e}");
    // Different thread count -> refused (determinism is per thread count).
    let mut t = cfg.clone();
    t.threads = Some(3);
    assert!(run(&t, &ds, &opts(&out, true, None)).is_err());
    // A flipped byte fails the trailer checksum.
    let p = out.join("checkpoint.bin");
    let mut bytes = std::fs::read(&p).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x40;
    std::fs::write(&p, &bytes).unwrap();
    let e = run(&cfg, &ds, &opts(&out, true, None)).unwrap_err();
    assert!(e.to_string().contains("checksum mismatch"), "{e}");
}

#[test]
fn early_stop_and_tables_only_export() {
    let ds = synth_dataset();
    let mut cfg = synth_run(40);
    cfg.early_stop_patience = Some(2);
    cfg.recipe.as_mut().unwrap().lr = 2.0; // overshoots quickly, so valid stops improving
    let d = tempfile::tempdir().unwrap();
    let out = d.path().join("run");
    let r = run(&cfg, &ds, &opts(&out, false, None)).unwrap();
    let n = r["epochs"].as_array().unwrap().len();
    assert_eq!(r["stopped"], "early_stop", "ran {n} epochs");
    assert!(n < 40);
    let best = r["best"]["epoch"].as_u64().unwrap() as usize;
    assert_eq!(n, best + 3, "stops after `patience` non-improving evals");

    // Export: exactly (E + 2R) x D f32s, a manifest with no triple or label
    // fields, and it reloads only against its own dataset.
    let (t, m) = load_tables(&out.join("best"), Some(&ds)).unwrap();
    assert_eq!(m.epoch, best);
    let w = std::fs::read(out.join("best/weights.bin")).unwrap();
    assert_eq!(
        w.len(),
        (ds.num_entities + 2 * ds.num_relations) * t.dims() * 4
    );
    let man = std::fs::read_to_string(out.join("best/manifest.json")).unwrap();
    for banned in ["ent0", "rel0", "\"s\"", "triples"] {
        assert!(!man.contains(banned), "manifest leaks {banned}");
    }
    let mut other = ds.clone();
    other.entity_vocab_hash = "0".repeat(64);
    assert!(load_tables(&out.join("best"), Some(&other)).is_err());
}
