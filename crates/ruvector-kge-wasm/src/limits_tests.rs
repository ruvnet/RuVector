//! F1 size-limit tests for the binding glue. Duplicated VERBATIM in
//! `ruvector-kge-ffi/src/limits_tests.rs` and `ruvector-kge-wasm/src/limits_tests.rs`.

use crate::model::KgeModel;

fn kind_of(envelope: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(envelope).ok()?;
    v["error"]["kind"].as_str().map(str::to_string)
}

#[test]
fn f1_constructor_rejects_50k_dims_with_limit_envelope() {
    let err = KgeModel::new(r#"{"dims":50000}"#)
        .err()
        .expect("must reject");
    assert_eq!(kind_of(&err).as_deref(), Some("limit"), "{err}");
    assert!(KgeModel::new(r#"{"dims":4096}"#).is_ok());
    // A cap too small for one row is also a limit error; a cap above the
    // global ceiling is clamped, not honoured.
    let err = KgeModel::new(r#"{"dims":8,"maxTableBytes":16}"#)
        .err()
        .unwrap();
    assert_eq!(kind_of(&err).as_deref(), Some("limit"));
    assert!(KgeModel::new(r#"{"dims":8,"maxTableBytes":18446744073709551615}"#).is_ok());
}

#[test]
fn f1_add_triples_growth_rejected_atomically() {
    // 64 dims x 4 B = 256 B/row; a 4 KiB cap fits 16 rows (e.g. 15 + 1).
    let mut m = KgeModel::new(r#"{"dims":64,"maxTableBytes":4096}"#).unwrap();
    let ok = m.add_triples_json(r#"[{"s":"a","r":"r","o":"b"}]"#);
    assert!(kind_of(&ok).is_none(), "{ok}");
    let before = m.to_json();
    let batch: Vec<String> = (0..20)
        .map(|i| format!(r#"{{"s":"e{i}","r":"r","o":"f{i}"}}"#))
        .collect();
    let out = m.add_triples_json(&format!("[{}]", batch.join(",")));
    assert_eq!(kind_of(&out).as_deref(), Some("limit"), "{out}");
    assert_eq!(
        m.to_json(),
        before,
        "a rejected batch must not mutate the model"
    );
}

#[test]
fn f1_default_cap_rejects_huge_entity_growth() {
    // 4096 dims x 4 B = 16 KiB/row: 2 GiB fits ~131k rows, far below 1M
    // entities — the byte cap, not the row cap, is what fires here.
    let mut m = KgeModel::new(r#"{"dims":4096}"#).unwrap();
    let batch: Vec<String> = (0..70_000)
        .map(|i| format!(r#"{{"s":"a{i}","r":"r","o":"b{i}"}}"#))
        .collect();
    let out = m.add_triples_json(&format!("[{}]", batch.join(",")));
    assert_eq!(kind_of(&out).as_deref(), Some("limit"), "{out}");
    assert!(m.tables.is_none());
}

#[test]
fn f1_load_rejects_oversized_model_before_allocation() {
    // Declared dims far above MAX_DIMS; no tables, so a naive load would only
    // blow up later in `ensure_built`. The hash is bogus on purpose: the size
    // gate must fire first, before the payload is materialised or hashed.
    let json = r#"{"sha256":"00","model":{"version":1,
        "config":{"scorer":"hole","dims":50000,"seed":1},
        "entities":{"labels":["a","b"]},"relations":{"labels":["r"]},
        "triples":[],"tables":null}}"#;
    let err = KgeModel::from_json(json).err().expect("must reject");
    assert_eq!(kind_of(&err).as_deref(), Some("limit"), "{err}");

    // In-range dims but a vocab whose tables would exceed the byte cap.
    let labels: Vec<String> = (0..140_000).map(|i| format!("\"e{i}\"")).collect();
    let json = format!(
        r#"{{"sha256":"00","model":{{"version":1,
        "config":{{"scorer":"hole","dims":4096,"seed":1}},
        "entities":{{"labels":[{}]}},"relations":{{"labels":["r"]}},
        "triples":[],"tables":null}}}}"#,
        labels.join(",")
    );
    let err = KgeModel::from_json(&json).err().expect("must reject");
    assert_eq!(kind_of(&err).as_deref(), Some("limit"), "{err}");
}

#[test]
fn f1_round_trip_unchanged_for_in_range_models() {
    let mut m = KgeModel::new(r#"{"dims":8}"#).unwrap();
    m.add_triples_json(r#"[{"s":"a","r":"r","o":"b"}]"#);
    m.ensure_built();
    let saved = m.to_json();
    assert!(
        !saved.contains("maxTableBytes"),
        "unset cap must not be serialized"
    );
    let back = KgeModel::from_json(&saved).expect("round trip");
    assert_eq!(back.to_json(), saved);
}

/// Wrap a raw `model` payload in an envelope whose sha256 is computed exactly
/// as `from_json` recomputes it (Value path, so duplicate keys keep the last
/// copy): the hash is not a MAC, so an attacker can always do this.
fn envelope_with_valid_hash(model_part: &str) -> String {
    use sha2::{Digest, Sha256};
    let v: serde_json::Value = serde_json::from_str(model_part).unwrap();
    let parsed: KgeModel = serde_json::from_value(v).unwrap();
    let body = serde_json::to_string(&parsed).unwrap();
    let hash: String = Sha256::digest(body.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!(r#"{{"sha256":"{hash}","model":{model_part}}}"#)
}

#[test]
fn f1_load_rejects_duplicate_config_key_bypass() {
    // Harmless `config` first, oversized last: `Value` keeps the last copy.
    let model = r#"{"version":1,
        "config":{"scorer":"hole","dims":8,"seed":1},
        "config":{"scorer":"hole","dims":50000,"seed":1},
        "entities":{"labels":["a","b"]},"relations":{"labels":["r"]},
        "triples":[],"tables":null}"#;
    let err = KgeModel::from_json(&envelope_with_valid_hash(model))
        .err()
        .expect("duplicate-key payload must not load");
    let kind = kind_of(&err);
    assert!(
        matches!(kind.as_deref(), Some("limit") | Some("invalid")),
        "{err}"
    );
}

#[test]
fn f1_load_rejects_duplicate_entities_key_bypass() {
    let labels: Vec<String> = (0..140_000).map(|i| format!("\"e{i}\"")).collect();
    let model = format!(
        r#"{{"version":1,
        "config":{{"scorer":"hole","dims":4096,"seed":1}},
        "entities":{{"labels":[]}},
        "entities":{{"labels":[{}]}},"relations":{{"labels":["r"]}},
        "triples":[],"tables":null}}"#,
        labels.join(",")
    );
    let err = KgeModel::from_json(&envelope_with_valid_hash(&model))
        .err()
        .expect("duplicate-key payload must not load");
    let kind = kind_of(&err);
    assert!(
        matches!(kind.as_deref(), Some("limit") | Some("invalid")),
        "{err}"
    );
}

#[test]
fn f1_post_parse_gate_rejects_oversized_model_independently() {
    // The header precheck is only an early rejection; the post-parse gate must
    // stand on its own for anything the header pass could miss.
    let big: KgeModel = serde_json::from_str(
        r#"{"version":1,"config":{"scorer":"hole","dims":50000,"seed":1},
        "entities":{"labels":["a"]},"relations":{"labels":["r"]},
        "triples":[],"tables":null}"#,
    )
    .unwrap();
    let err = big.validate_loaded().expect_err("must reject");
    assert_eq!(kind_of(&err).as_deref(), Some("limit"), "{err}");

    let odd: KgeModel = serde_json::from_str(
        r#"{"version":1,"config":{"scorer":"hole","dims":7,"seed":1},
        "entities":{"labels":["a"]},"relations":{"labels":["r"]},
        "triples":[],"tables":null}"#,
    )
    .unwrap();
    let err = odd.validate_loaded().expect_err("odd dims must reject");
    assert_eq!(kind_of(&err).as_deref(), Some("invalid"), "{err}");
}

#[test]
fn f1_load_fails_closed_on_malformed_json() {
    let err = KgeModel::from_json("{not json").err().expect("must reject");
    assert_eq!(kind_of(&err).as_deref(), Some("invalid"), "{err}");
}
