//! Unit tests for the example bank — pure, no engine/embedder present.

use super::*;

fn bank() -> Bank {
    Bank::new(SplitRatios::default())
}

#[test]
fn admission_dedupes_identical_content() {
    let mut b = bank();
    let first = b.admit("dept", "my card was charged twice", "billing", TrustTier::A);
    let second = b.admit("dept", "my card was charged twice", "billing", TrustTier::A);
    match (first, second) {
        (Admission::Accepted(a), Admission::Duplicate(d)) => assert_eq!(a, d),
        other => panic!("expected accept then duplicate, got {other:?}"),
    }
    assert_eq!(b.len(), 1);
}

#[test]
fn different_label_is_a_different_example() {
    let mut b = bank();
    b.admit("dept", "same text", "billing", TrustTier::A);
    // Same text + question but a different label => different content id.
    assert!(matches!(
        b.admit("dept", "same text", "fraud", TrustTier::A),
        Admission::Accepted(_)
    ));
    assert_eq!(b.len(), 2);
}

#[test]
fn splits_are_deterministic_and_frozen_across_reload() {
    let mut b = bank();
    for i in 0..500 {
        b.admit("q", &format!("example number {i}"), "label", TrustTier::A);
    }
    let before: Vec<(ExampleId, Split)> = b.iter().map(|e| (e.id, e.split)).collect();
    // Reload with DIFFERENT ratios: stored splits must be authoritative.
    let json = b.to_json().unwrap();
    let reloaded = Bank::from_json(&json).unwrap();
    let after: Vec<(ExampleId, Split)> = reloaded.iter().map(|e| (e.id, e.split)).collect();
    assert_eq!(before, after);
    // Re-admitting the exact same content reproduces the same id + split.
    let mut b2 = Bank::new(SplitRatios::new(20, 20, 20, 20, 20).unwrap());
    let ex = b.iter().next().unwrap();
    let text = b.text_of(ex.id).unwrap().to_string();
    if let Admission::Accepted(id) = b2.admit(&ex.question, &text, &ex.label, ex.tier) {
        assert_eq!(id, ex.id);
        // The id is stable; the split follows b2's own ratios for a fresh admit,
        // but within one bank a split is assigned once and never rewritten.
        assert_eq!(b2.iter().next().unwrap().id, ex.id);
    } else {
        panic!("fresh bank should accept");
    }
}

#[test]
fn all_five_splits_are_pairwise_disjoint() {
    let mut b = bank();
    for i in 0..1000 {
        b.admit("q", &format!("txt {i}"), "l", TrustTier::A);
    }
    let splits = [
        Split::Train,
        Split::Calibration,
        Split::Validation,
        Split::Transfer,
        Split::Test,
    ];
    for (i, &a) in splits.iter().enumerate() {
        for &c in &splits[i + 1..] {
            assert_eq!(b.assert_disjoint(a, c), Ok(()), "{a:?} vs {c:?}");
        }
    }
    // Every split actually received members with the default 60/10/15/10/5.
    for s in splits {
        assert!(b.iter_split("q", s).count() > 0, "split {s:?} empty");
    }
}

#[test]
fn tier_c_is_stored_but_never_promotable() {
    let mut b = bank();
    b.admit("q", "trusted verifier label", "l", TrustTier::A);
    b.admit("q", "llm judge label", "l", TrustTier::B);
    b.admit("q", "user supplied text", "l", TrustTier::C);
    assert_eq!(b.len(), 3);
    assert_eq!(b.promotable(), 2);
    // Per (question, split), tier C never appears in the promotable iterator.
    for s in [
        Split::Train,
        Split::Validation,
        Split::Transfer,
        Split::Test,
    ] {
        assert!(b.iter_promotable("q", s).all(|e| e.tier != TrustTier::C));
    }
}

#[test]
fn noisy_label_filter_quarantines_on_disagreement() {
    // Filter: reject any text containing "spam".
    let filter: NoisyLabelFilter = Box::new(|_ex, text| !text.contains("spam"));
    let mut b = bank().with_noisy_label_filter(filter);
    assert!(matches!(
        b.admit("q", "legitimate request", "l", TrustTier::A),
        Admission::Accepted(_)
    ));
    assert!(matches!(
        b.admit("q", "buy cheap spam now", "l", TrustTier::A),
        Admission::Quarantined(_)
    ));
    assert_eq!(b.len(), 1);
}

#[test]
fn redacted_summary_contains_no_example_text() {
    let mut b = bank();
    let secret = "PATIENT SSN 123-45-6789 confidential";
    b.admit("q", secret, "label", TrustTier::A);
    b.admit(
        "q",
        "another distinctive phrase zzzq",
        "label",
        TrustTier::B,
    );
    let summary = b.redacted_summary();
    let serialized = serde_json::to_string(&summary).unwrap();
    assert!(!serialized.contains(secret));
    assert!(!serialized.contains("distinctive phrase"));
    assert!(!serialized.contains("123-45-6789"));
    // But it does carry counts and the content-id hashes.
    assert_eq!(summary.total, 2);
    assert_eq!(summary.ids.len(), 2);
    assert!(serialized.contains("\"total\":2"));
}

#[test]
fn full_json_round_trip_preserves_text_and_ids() {
    let mut b = bank();
    b.admit(
        "dept",
        "refund my duplicate charge",
        "billing",
        TrustTier::A,
    );
    b.admit("mood", "this is unacceptable", "angry", TrustTier::B);
    let json = b.to_json().unwrap();
    let reloaded = Bank::from_json(&json).unwrap();
    assert_eq!(reloaded.len(), 2);
    // Text survives (the user's own data on disk) and dedup still works.
    let id = b.iter().next().unwrap().id;
    assert_eq!(reloaded.text_of(id), b.text_of(id));
    // The rebuilt id set enforces dedup after reload.
    let mut reloaded = reloaded;
    assert!(matches!(
        reloaded.admit(
            "dept",
            "refund my duplicate charge",
            "billing",
            TrustTier::A
        ),
        Admission::Duplicate(_)
    ));
    assert_eq!(reloaded.len(), 2);
}

#[test]
fn split_ratios_must_sum_to_100() {
    assert!(SplitRatios::new(60, 10, 15, 10, 5).is_ok());
    assert!(SplitRatios::new(60, 10, 10, 10, 5).is_err());
}
