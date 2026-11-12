use agl_types::{
    AdmissionError, Authority, FitnessVector, Genome, HardGates, Mutation, MutationScope,
};
use serde_json::{json, Value};
use std::{env, fs};

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: adapter MUTATION_JSON NOW OUTPUT_JSON"
    );
    let record: Value = serde_json::from_str(&fs::read_to_string(&args[1]).unwrap()).unwrap();
    let now: u64 = args[2].parse().unwrap();
    let parent: Genome = serde_json::from_value(record["genome"].clone()).unwrap();
    let candidate: Mutation = serde_json::from_value(record["mutation"].clone()).unwrap();
    assert!(candidate.admissible(&parent, now).is_ok());
    assert!(!MutationScope::ApplicationCode.auto_promotable());

    let mut negatives = serde_json::Map::new();
    let check = |name: &str, value: Value, expected: fn(&AdmissionError) -> bool| {
        let mutation: Mutation = serde_json::from_value(value).unwrap();
        let error = mutation.admissible(&parent, now).unwrap_err();
        assert!(expected(&error), "unexpected {name} result: {error:?}");
        (name.to_string(), json!(format!("{error:?}")))
    };
    let base = record["mutation"].clone();
    let mut value = base.clone();
    value["parent_genome_hash"] = json!("wrong");
    negatives.extend([check("parent_mismatch", value, |error| {
        matches!(error, AdmissionError::ParentMismatch)
    })]);
    let mut value = base.clone();
    value["requested_authority"] = json!("constitutional");
    negatives.extend([check("authority_expansion", value, |error| {
        matches!(error, AdmissionError::AuthorityExpansion { .. })
    })]);
    let mut value = base.clone();
    value["requested_authority"] = json!("auto_reversible");
    negatives.extend([check("authority_insufficient", value, |error| {
        matches!(error, AdmissionError::AuthorityInsufficient { .. })
    })]);
    let mut value = base.clone();
    value["rollback_target"] = Value::Null;
    negatives.extend([check("no_rollback", value, |error| {
        matches!(error, AdmissionError::NoRollback)
    })]);
    let mut value = base.clone();
    value["preserved_invariants"][0]["holds"] = json!(false);
    negatives.extend([check("invariant_regressed", value, |error| {
        matches!(error, AdmissionError::InvariantRegressed(_))
    })]);
    let mut value = base;
    value["expires_at"] = json!(now);
    negatives.extend([check("expired", value, |error| {
        matches!(error, AdmissionError::Expired)
    })]);

    let gates = HardGates::default();
    let passing = FitnessVector {
        task_quality: 1.0,
        safety: 1.0,
        governance: 1.0,
        reliability: 1.0,
        p99_overhead_ms: 0.0,
        false_positive_rate: 0.0,
        regression_count: 0,
        rollback_verified: true,
    };
    assert!(passing.passes_hard_gates(&gates));
    let mut failed = Vec::new();
    let mut cases = vec![
        passing.clone(),
        passing.clone(),
        passing.clone(),
        passing.clone(),
        passing.clone(),
        passing,
    ];
    cases[0].safety = 0.0;
    cases[1].governance = 0.0;
    cases[2].false_positive_rate = 1.0;
    cases[3].p99_overhead_ms = 100.0;
    cases[4].regression_count = 1;
    cases[5].rollback_verified = false;
    for (index, case) in cases.iter().enumerate() {
        assert!(!case.passes_hard_gates(&gates));
        failed.push(index);
    }
    let receipt = json!({
        "schema": 1,
        "autogenous_commit": "905aa6cbe213392f8b3cab5d4f17bc3a48e0a509",
        "mutation_admissible": true,
        "application_code_auto_promotable": false,
        "negative_admission_gates": negatives,
        "synthetic_hard_gate_negatives": failed,
        "candidate_fitness_evaluated": false,
        "deployment_gate_pass": false,
        "authority": Authority::Governed,
        "limitations": [
            "Synthetic fitness vectors only establish hard AND semantics.",
            "No physical safety or latency fitness was invented."
        ]
    });
    fs::write(
        &args[3],
        serde_json::to_string_pretty(&receipt).unwrap() + "\n",
    )
    .unwrap();
    println!(
        "{}",
        json!({
            "mutation_admissible": true,
            "negative_gates": 6,
            "application_code_auto_promotable": false
        })
    );
}
