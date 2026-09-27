use agl_types::{Authority, Genome, Mutation};
use serde_json::{json, Value};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "Usage: esp32-retention-authority RECORD_JSON NOW_UNIX");
    let record: Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let parent: Genome = serde_json::from_value(record["genome"].clone()).unwrap();
    let mutation: Mutation = serde_json::from_value(record["mutation"].clone()).unwrap();
    let now: u64 = args[2].parse().unwrap();
    let admission = mutation.admissible(&parent, now);
    assert!(admission.is_ok(), "typed structural admission rejected: {admission:?}");
    assert!(!mutation.scope.auto_promotable(), "application code unexpectedly auto promotable");
    let mut overreach = mutation.clone();
    overreach.requested_authority = Authority::Constitutional;
    assert!(overreach.admissible(&parent, now).is_err());
    let mut insufficient = mutation.clone();
    insufficient.requested_authority = Authority::AutoReversible;
    assert!(insufficient.admissible(&parent, now).is_err());
    let mut irreversible = mutation.clone();
    irreversible.rollback_target = None;
    assert!(irreversible.admissible(&parent, now).is_err());
    let mut regressed = mutation.clone();
    regressed.preserved_invariants.clear();
    assert!(regressed.admissible(&parent, now).is_err());
    assert!(mutation.admissible(&parent, mutation.expires_at.unwrap()).is_err());
    println!("{}", serde_json::to_string_pretty(&json!({
        "structural_admission":true,"auto_promotable":false,"authorized_delivery":"PR review only",
        "deployment_promoted":false,"fitness_vector_evaluated":false,
        "negative_checks":{"authority_expansion":true,"insufficient_scope_authority":true,"missing_rollback":true,"invariant_regression":true,"expiry_boundary":true},
        "evaluated_at_unix":now,"autogenous_commit":"905aa6cbe213392f8b3cab5d4f17bc3a48e0a509",
        "limitations":["Structural admission does not verify declared invariant evidence.","Governed ceiling is limited here to the authorized PR workflow, not deployment.","No fitness safety, reliability or latency values invented; production hard gates not evaluated."]
    })).unwrap());
}
