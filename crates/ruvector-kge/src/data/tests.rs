//! Tests for the parent [`crate::data`] module. Kept in a sibling file so
//! `data.rs` stays under the 500-line ceiling.

use super::*;

fn t(s: u32, r: u32, o: u32) -> Triple {
    Triple::new(s, r, o)
}

#[test]
fn data_dedup_and_counts() {
    let store = TripleStore::new(vec![t(0, 0, 1), t(0, 0, 1), t(2, 1, 3)]).unwrap();
    assert_eq!(store.len(), 2, "duplicate removed");
    assert_eq!(store.num_entities(), 4);
    assert_eq!(store.num_relations(), 2);
    assert_eq!(store.true_tails(0, 0).unwrap().len(), 1);
    assert!(store.true_heads(1, 3).unwrap().contains(&2));
}

#[test]
fn data_split4_transfer_holdout() {
    // 5 relations; transfer must hold out whole relations, disjoint+frozen.
    let triples: Vec<Triple> = (0..500).map(|i| t(i % 100, i % 5, (i + 1) % 100)).collect();
    let store = TripleStore::new(triples).unwrap();
    let a = store.split4(9, [0.7, 0.1, 0.1, 0.1]).unwrap();
    let b = store.split4(9, [0.7, 0.1, 0.1, 0.1]).unwrap();
    assert_eq!(a, b, "same seed => frozen 4-way split");
    a.assert_disjoint().unwrap();
    assert_eq!(a.total(), store.len(), "partition covers every triple once");

    let xfer_rels = a.transfer_relations();
    assert!(!xfer_rels.is_empty(), "at least one relation held out");
    // The held-out relations appear in NO other split — the mix differs.
    for part in [&a.train, &a.valid, &a.test] {
        assert!(
            part.iter().all(|tr| !xfer_rels.contains(&tr.r)),
            "transfer relations must be absent from train/valid/test"
        );
    }
    // With < 2 relations, no relation can be held out.
    let tiny = TripleStore::new(vec![t(0, 0, 1), t(1, 0, 2)]).unwrap();
    let s = tiny.split4(1, [0.8, 0.0, 0.1, 0.1]).unwrap();
    assert!(
        s.transfer.is_empty(),
        "no holdout possible with one relation"
    );
}

#[test]
fn data_split_disjoint_and_frozen() {
    let triples: Vec<Triple> = (0..100).map(|i| t(i, i % 5, (i + 1) % 100)).collect();
    let store = TripleStore::new(triples).unwrap();
    let a = store.split(7, [0.8, 0.1, 0.1]).unwrap();
    let b = store.split(7, [0.8, 0.1, 0.1]).unwrap();
    assert_eq!(a, b, "same seed => identical (frozen) split");
    a.assert_disjoint().unwrap();
    assert_eq!(a.total(), 100, "partition covers every triple once");
    assert!(a.train.len() > a.valid.len());
    let c = store.split(8, [0.8, 0.1, 0.1]).unwrap();
    assert_ne!(a, c, "different seed => different split");
}

#[test]
fn data_frozen_across_reload() {
    let triples: Vec<Triple> = (0..60).map(|i| t(i, i % 3, (i * 7) % 60)).collect();
    let store = TripleStore::new(triples).unwrap();
    let before = store.split(3, [0.7, 0.15, 0.15]).unwrap();
    let json = serde_json::to_string(&store).unwrap();
    let reloaded: TripleStore = serde_json::from_str(&json).unwrap();
    let after = reloaded.split(3, [0.7, 0.15, 0.15]).unwrap();
    assert_eq!(before, after, "split is frozen across serialize/reload");
}

#[test]
fn data_vocab_redacts_and_maps() {
    let mut v = Vocab::new();
    let a = v.intern_entity("alice").unwrap();
    let a2 = v.intern_entity("alice").unwrap();
    let b = v.intern_entity("bob").unwrap();
    assert_eq!(a, a2);
    assert_ne!(a, b);
    assert_eq!(v.entity_id("bob"), Some(b));
    let dbg = format!("{v:?}");
    assert!(!dbg.contains("alice"), "Debug must not leak labels");
    assert!(dbg.contains('2'), "Debug shows counts");
    let json = serde_json::to_string(&v).unwrap();
    let rt: Vocab = serde_json::from_str(&json).unwrap();
    assert_eq!(rt.entity_id("alice"), Some(a), "index rebuilt on reload");
}

#[test]
fn data_input_limit_typed() {
    // A single triple that names a relation id past the cap.
    let bad = TripleStore::with_counts(vec![t(0, 0, 1)], None, Some(MAX_RELATIONS + 1));
    assert!(matches!(bad, Err(KgeError::Limit("relations"))));
    let bad_e = TripleStore::with_counts(vec![t(0, 0, 1)], Some(MAX_ENTITIES + 1), None);
    assert!(matches!(bad_e, Err(KgeError::Limit("entities"))));
}
