//! Integration tests on the real pinned inputs. They need the cache populated
//! (`openjev fetch --download --cache $OPENJEV_CACHE`), so they are #[ignore]d:
//!   OPENJEV_CACHE=~/.cache/openjev cargo test -p ruvector-typesafe-train --release -- --ignored

use std::path::{Path, PathBuf};

use ruvector_typesafe_train::config::Config;
use ruvector_typesafe_train::data::{read_jsonl, write_jsonl, Row, TICKETS};
use ruvector_typesafe_train::leakage::LeakageError;
use ruvector_typesafe_train::{export, pins, prep, train};

fn cache() -> PathBuf {
    PathBuf::from(
        std::env::var("OPENJEV_CACHE").expect("set OPENJEV_CACHE to the pinned-input cache"),
    )
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sources() -> prep::Sources {
    prep::Sources {
        tickets_fixture: repo().join("npm/packages/typesafe/bench/fixtures/tickets-decisions.json"),
        cache: cache(),
        allow_download: false,
        datasets: vec![],
    }
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("openjev-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
#[ignore]
fn prep_counts_and_planted_collision_aborts_train() {
    let dir = tmp("leak");
    let out = prep::run(&sources(), &dir).unwrap();
    let tickets_train = out.train.iter().filter(|r| r.dataset == TICKETS).count();
    let tickets_val = out.val.iter().filter(|r| r.dataset == TICKETS).count();
    assert_eq!((tickets_train, tickets_val), (136, 150));
    assert_eq!(out.report.total_intersection(), 0);
    // 12 humanised intent names equal public test utterances (e.g. CLINC150 "goodbye").
    let dropped: usize = out.report.dropped_descriptions.values().map(Vec::len).sum();
    assert_eq!(dropped, 12, "{:?}", out.report.dropped_descriptions);

    // Plant one frozen *test* ticket (case/punctuation changed) into train.jsonl.
    let fx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sources().tickets_fixture).unwrap()).unwrap();
    let test_text = fx["split"]["test"][0]["text"]
        .as_str()
        .unwrap()
        .to_uppercase()
        + "!!";
    let mut rows = read_jsonl(&dir.join("train.jsonl")).unwrap();
    let mut planted = Row::new(TICKETS, "planted".into(), test_text, "billing".into());
    planted.urgent = Some(false);
    planted.frustration = Some(0);
    rows.push(planted);
    write_jsonl(&dir.join("train.jsonl"), &rows).unwrap();

    let args = train::TrainArgs {
        config: Config::v0(),
        config_sha256: String::new(),
        data: dir.clone(),
        sources: sources(),
        base: cache().join(pins::BASE_SAFETENSORS.name),
        tokenizer: cache().join(pins::TOKENIZER.name),
        seed: 1,
        device: candle_core::Device::Cpu,
        device_name: "cpu".into(),
        out: dir.join("run"),
    };
    let err = train::run(&args).expect_err("train must refuse leaked data");
    let leak = err.downcast_ref::<LeakageError>().expect("a LeakageError");
    assert_eq!(leak.0.total_intersection(), 1);
    assert!(
        !dir.join("run/encoder.safetensors").exists(),
        "no step may have run"
    );
}

#[test]
#[ignore]
fn real_identity_transplant_is_byte_identical() {
    let out = tmp("ident").join("model.onnx");
    let rep = export::run(
        &cache().join(pins::TEMPLATE_ONNX.name),
        &cache().join(pins::BASE_SAFETENSORS.name),
        None,
        &out,
    )
    .unwrap();
    assert_eq!(rep.float_initializers, 197);
    assert_eq!(rep.named, 125);
    assert_eq!(rep.anonymous_transposed, 72);
    assert!(rep.byte_identical_to_template);
    assert_eq!(rep.output_sha256, pins::TEMPLATE_ONNX.sha256);
}
