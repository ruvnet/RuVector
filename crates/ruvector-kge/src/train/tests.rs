use super::grad::testing::DistMult;
use super::*;
use crate::data::TripleStore;
use crate::{eval, Tables, Triple};

/// Synthetic KG: 200 entities in 10 clusters of 20, plus 4 dedicated
/// "tag" entities per cluster reused across relations. Relations are all
/// symmetric (same-cluster style) so a symmetric DistMult can fit them —
/// asymmetric relations (successor) are unlearnable by DistMult and would
/// only add noise to the margin check.
fn synthetic_kg() -> (TripleStore, usize, usize) {
    let clusters = 5usize;
    let per = 20usize;
    let num_entities = clusters * per; // 100 — kept small so the 1-vs-all
                                       // (O(|E|) per positive per side) test
                                       // stays fast in debug.
    let num_relations = 3;
    let mut triples: Vec<Triple> = Vec::new();
    for c in 0..clusters {
        let base = (c * per) as u32;
        for a in 0..per as u32 {
            // r0: same-cluster neighbour (ring within the cluster).
            let b = base + (a + 1) % per as u32;
            triples.push(Triple::new(base + a, 0, b));
            triples.push(Triple::new(b, 0, base + a)); // symmetric
                                                       // r1: two-step neighbour.
            let d = base + (a + 2) % per as u32;
            triples.push(Triple::new(base + a, 1, d));
            triples.push(Triple::new(d, 1, base + a));
            // r2: cluster tag (the cluster's first member as anchor).
            triples.push(Triple::new(base + a, 2, base));
            triples.push(Triple::new(base, 2, base + a));
        }
    }
    let store = TripleStore::with_counts(triples, Some(num_entities), Some(num_relations)).unwrap();
    (store, num_entities, num_relations)
}

fn train_store(split_train: &[Triple], ne: usize, nr: usize) -> TripleStore {
    TripleStore::with_counts(split_train.to_vec(), Some(ne), Some(nr)).unwrap()
}

#[test]
fn train_lifts_filtered_mrr_over_random() {
    let (store, ne, nr) = synthetic_kg();
    let split = store.split(1, [0.85, 0.0, 0.15]).unwrap();
    split.assert_disjoint().unwrap();
    let train = train_store(&split.train, ne, nr);

    let dims = 32;
    let scorer = DistMult::new(dims);
    let mut tables = Tables::new(ne, nr, dims, 42);

    let cfg = TrainConfig {
        dims,
        epochs: 35,
        batch_size: 100,
        lr: 0.5,
        optimizer: OptimKind::Adagrad { epsilon: 1e-8 },
        loss: LossKind::OneVsAll,
        n3_lambda: 5e-4,
        seed: 7,
        ..Default::default()
    };

    // Baseline MRR on random init (before training).
    let ecfg = eval::EvalConfig::random(123);
    let baseline = eval::evaluate(&tables, &scorer, &store, &split.test, &ecfg).unwrap();

    let mut first_loss = None;
    let mut last_loss = 0.0f32;
    Trainer::fit(&mut tables, &scorer, &train, &cfg, |p| {
        if first_loss.is_none() {
            first_loss = Some(p.loss);
        }
        last_loss = p.loss;
    })
    .unwrap();

    let trained = eval::evaluate(&tables, &scorer, &store, &split.test, &ecfg).unwrap();

    let first = first_loss.unwrap();
    println!(
        "[train] loss {:.4} -> {:.4}; filtered MRR baseline {:.4} -> trained {:.4} (Hits@10 {:.3} -> {:.3})",
        first, last_loss, baseline.combined.mrr, trained.combined.mrr,
        baseline.combined.hits10, trained.combined.hits10
    );

    assert!(
        last_loss < first,
        "loss should decrease: {first} -> {last_loss}"
    );
    assert!(
        trained.combined.mrr > baseline.combined.mrr + 0.2,
        "trained MRR {:.4} should beat baseline {:.4} by > 0.2",
        trained.combined.mrr,
        baseline.combined.mrr
    );
    assert!(
        trained.combined.mrr > 0.3,
        "trained filtered MRR {:.4} should be well above the ~1/N random floor",
        trained.combined.mrr
    );
}

#[test]
fn train_deterministic_same_seed() {
    let (store, ne, nr) = synthetic_kg();
    let split = store.split(2, [0.9, 0.0, 0.1]).unwrap();
    let train = train_store(&split.train, ne, nr);
    let dims = 16;
    let scorer = DistMult::new(dims);
    let cfg = TrainConfig {
        dims,
        epochs: 10,
        batch_size: 128,
        lr: 0.3,
        optimizer: OptimKind::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        loss: LossKind::SelfAdversarial {
            neg_count: 8,
            temperature: 1.0,
            margin: 6.0,
        },
        n3_lambda: 0.0,
        seed: 99,
        ..Default::default()
    };
    let mut a = Tables::new(ne, nr, dims, 5);
    let mut b = Tables::new(ne, nr, dims, 5);
    Trainer::fit(&mut a, &scorer, &train, &cfg, |_| {}).unwrap();
    Trainer::fit(&mut b, &scorer, &train, &cfg, |_| {}).unwrap();
    assert_eq!(a, b, "same seed must yield identical tables after training");
}

#[test]
fn train_self_adversarial_reduces_loss() {
    let (store, ne, nr) = synthetic_kg();
    let split = store.split(3, [0.9, 0.0, 0.1]).unwrap();
    let train = train_store(&split.train, ne, nr);
    let dims = 32;
    let scorer = DistMult::new(dims);
    let cfg = TrainConfig {
        dims,
        epochs: 40,
        batch_size: 200,
        lr: 0.05,
        optimizer: OptimKind::Adagrad { epsilon: 1e-8 },
        loss: LossKind::SelfAdversarial {
            neg_count: 16,
            temperature: 0.5,
            margin: 3.0,
        },
        n3_lambda: 0.0,
        seed: 11,
        ..Default::default()
    };
    let mut tables = Tables::new(ne, nr, dims, 8);
    let mut first = None;
    let mut last = 0.0;
    Trainer::fit(&mut tables, &scorer, &train, &cfg, |p| {
        if first.is_none() {
            first = Some(p.loss);
        }
        last = p.loss;
    })
    .unwrap();
    let first = first.unwrap();
    println!("[train/self-adv] loss {first:.4} -> {last:.4}");
    assert!(
        last < first,
        "self-adversarial loss should drop: {first} -> {last}"
    );
}

#[test]
fn train_validates_boundary() {
    let (store, ne, nr) = synthetic_kg();
    let scorer = DistMult::new(8);
    let mut tables = Tables::new(ne, nr, 8, 1);
    // dims mismatch
    let bad = TrainConfig {
        dims: 16,
        ..Default::default()
    };
    assert!(matches!(
        Trainer::fit(&mut tables, &scorer, &store, &bad, |_| {}),
        Err(KgeError::Dims { .. })
    ));
    // batch_size 0
    let bad2 = TrainConfig {
        dims: 8,
        batch_size: 0,
        ..Default::default()
    };
    assert!(matches!(
        Trainer::fit(&mut tables, &scorer, &store, &bad2, |_| {}),
        Err(KgeError::Invalid(_))
    ));
}

#[test]
fn train_config_partial_json_defaults() {
    // An empty object deserializes to the full default config.
    let c: TrainConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(c, TrainConfig::default());
    // Internally-tagged enums fill their per-field defaults from the tag.
    let c: TrainConfig =
        serde_json::from_str(r#"{"loss":{"kind":"self_adversarial"},"optimizer":{"kind":"adam"}}"#)
            .unwrap();
    assert!(matches!(
        c.loss,
        LossKind::SelfAdversarial {
            neg_count: 16,
            margin,
            ..
        } if margin == 9.0
    ));
    assert!(matches!(c.optimizer, OptimKind::Adam { beta1, .. } if beta1 == 0.9));
}
