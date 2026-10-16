use ruvector_language::{parser, runtime, Context, Effect, Node, Output, Snapshot};
use std::collections::{BTreeMap, BTreeSet};

fn execute(properties: serde_json::Value) -> ruvector_language::Receipt {
    let program = parser::parse(
        r#"program canonical {
        effect graph.read; effect decision.emit; budget 100000;
        let n = select ["api"]; let e = observe n "Failing" within 300;
        let d = assess e; emit d;
    }"#,
    )
    .unwrap();
    let snapshot = Snapshot {
        version: 1,
        revision: "same".into(),
        nodes: vec![Node {
            id: "api".into(),
            label: "Service".into(),
            properties: BTreeMap::from([("nested".into(), properties)]),
            vector: None,
        }],
        edges: vec![],
        observations: vec![],
        rules: vec![],
    };
    let context = Context {
        now: 100,
        capabilities: BTreeSet::from([Effect::GraphRead, Effect::DecisionEmit]),
        max_steps: 100000,
        max_results: 100,
    };
    runtime::execute(&program, &snapshot, &context, None).unwrap()
}

#[test]
fn reordered_nested_objects_produce_identical_receipts_with_preserve_order() {
    let left: serde_json::Value =
        serde_json::from_str(r#"{"z":{"b":2,"a":1},"a":[{"d":4,"c":3}]}"#).unwrap();
    let right: serde_json::Value =
        serde_json::from_str(r#"{"a":[{"c":3,"d":4}],"z":{"a":1,"b":2}}"#).unwrap();
    // Ensure this regression really exercises the insertion-ordered backend.
    assert_ne!(left.to_string(), right.to_string());
    let first = execute(left);
    let second = execute(right);
    assert_eq!(first, second);
    runtime::verify_receipt(&first).unwrap();
}

#[test]
fn nested_content_changes_and_array_reordering_change_snapshot_hash() {
    let original = execute(serde_json::json!({"x": [{"a":1}, {"a":2}]}));
    let reversed = execute(serde_json::json!({"x": [{"a":2}, {"a":1}]}));
    let changed = execute(serde_json::json!({"x": [{"a":1}, {"a":3}]}));
    assert_ne!(original.snapshot_hash, reversed.snapshot_hash);
    assert_ne!(original.snapshot_hash, changed.snapshot_hash);
    assert_eq!(original.output, reversed.output);
    assert!(matches!(original.output, Output::Decisions(_)));
}
