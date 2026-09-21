//! CPU mini-batch training over the embedding [`Tables`] (ADR-003 §4).
//!
//! Deterministic in `TrainConfig::seed`: own PRNG for shuffling and negative
//! sampling (no `rand`), sparse per-row optimizer state, order-independent
//! gradient application. The scorer is any [`Differentiable`]; training never
//! special-cases a scorer.

pub mod grad;
pub mod loss;
mod negatives;
pub mod optim;

pub use grad::Differentiable;
pub use optim::OptimKind;

use crate::data::{Rng, TripleStore};
use crate::{KgeError, Result, Tables};
use optim::{Grads, Optimizer};
use serde::{Deserialize, Serialize};

/// Which training regime to run (ADR-003 §1). A bandit arm, not hardcoded.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LossKind {
    /// Self-adversarial negative sampling (RotatE).
    SelfAdversarial {
        #[serde(default = "neg_count_default")]
        neg_count: usize,
        #[serde(default = "temperature_default")]
        temperature: f32,
        #[serde(default = "margin_default")]
        margin: f32,
    },
    /// 1-vs-all cross-entropy over every entity (ComplEx-N3), both sides.
    #[default]
    OneVsAll,
}

fn neg_count_default() -> usize {
    16
}
fn temperature_default() -> f32 {
    1.0
}
fn margin_default() -> f32 {
    9.0
}

/// Training hyperparameters. Every field has a serde default so a partial
/// config deserializes (the HPO loop varies a subset per arm, ADR-004).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    #[serde(default = "dims_default")]
    pub dims: usize,
    #[serde(default = "epochs_default")]
    pub epochs: usize,
    #[serde(default = "batch_size_default")]
    pub batch_size: usize,
    #[serde(default = "lr_default")]
    pub lr: f32,
    #[serde(default)]
    pub optimizer: OptimKind,
    #[serde(default)]
    pub loss: LossKind,
    #[serde(default = "n3_lambda_default")]
    pub n3_lambda: f32,
    #[serde(default)]
    pub seed: u64,
}

fn dims_default() -> usize {
    128
}
fn epochs_default() -> usize {
    100
}
fn batch_size_default() -> usize {
    256
}
fn lr_default() -> f32 {
    0.1
}
fn n3_lambda_default() -> f32 {
    1e-3
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            dims: dims_default(),
            epochs: epochs_default(),
            batch_size: batch_size_default(),
            lr: lr_default(),
            optimizer: OptimKind::default(),
            loss: LossKind::default(),
            n3_lambda: n3_lambda_default(),
            seed: 0,
        }
    }
}

/// Per-epoch progress handed to the callback. Carries no clock or timestamp
/// (ADR-005: no time in the core).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    pub epoch: usize,
    pub epochs: usize,
    pub num_batches: usize,
    /// Mean data-loss per positive over the epoch.
    pub loss: f32,
    /// Mean N3 penalty per positive over the epoch.
    pub n3_penalty: f32,
}

/// The trainer. `fit` is an associated function taking the tables by mutable
/// reference (ADR-003 signature).
pub struct Trainer;

impl Trainer {
    /// Train `tables` on `store`'s triples with `scorer`, calling `callback`
    /// once per epoch. Returns after `config.epochs` epochs.
    pub fn fit(
        tables: &mut Tables,
        scorer: &dyn Differentiable,
        store: &TripleStore,
        config: &TrainConfig,
        mut callback: impl FnMut(&Progress),
    ) -> Result<()> {
        validate(tables, scorer, store, config)?;

        let positives = store.triples().to_vec();
        let n = positives.len();
        if n == 0 {
            return Ok(());
        }
        let num_batches = n.div_ceil(config.batch_size);

        let mut opt = Optimizer::new(config.optimizer, config.lr);
        let mut sample_rng = Rng::seeded(config.seed ^ 0xA5A5_0F0F_1234_5678);
        let mut order: Vec<usize> = (0..n).collect();
        let mut n3_buf: Vec<f32> = Vec::new();

        for epoch in 0..config.epochs {
            shuffle(&mut order, config.seed, epoch);
            let mut epoch_loss = 0.0f64;
            let mut epoch_n3 = 0.0f64;

            for batch in order.chunks(config.batch_size) {
                let mut grads = Grads::new(config.dims);
                for &i in batch {
                    let t = positives[i];
                    let data_loss = match config.loss {
                        LossKind::SelfAdversarial {
                            neg_count,
                            temperature,
                            margin,
                        } => loss::self_adversarial_step(
                            tables,
                            scorer,
                            t,
                            neg_count,
                            temperature,
                            margin,
                            &mut sample_rng,
                            &mut grads,
                        )?,
                        LossKind::OneVsAll => loss::one_vs_all_step(tables, scorer, t, &mut grads)?,
                    };
                    epoch_loss += data_loss as f64;
                    if config.n3_lambda > 0.0 {
                        epoch_n3 +=
                            apply_n3(tables, t, config.n3_lambda, &mut grads, &mut n3_buf)? as f64;
                    }
                }
                if !grads.is_empty() {
                    opt.apply(tables, &grads)?;
                }
            }

            let progress = Progress {
                epoch,
                epochs: config.epochs,
                num_batches,
                loss: (epoch_loss / n as f64) as f32,
                n3_penalty: (epoch_n3 / n as f64) as f32,
            };
            callback(&progress);
        }
        Ok(())
    }
}

fn validate(
    tables: &Tables,
    scorer: &dyn Differentiable,
    store: &TripleStore,
    config: &TrainConfig,
) -> Result<()> {
    if config.batch_size == 0 {
        return Err(KgeError::Invalid("batch_size must be > 0".into()));
    }
    if config.epochs == 0 {
        return Err(KgeError::Invalid("epochs must be >= 1".into()));
    }
    if tables.dims() != config.dims {
        return Err(KgeError::Dims {
            expected: config.dims,
            got: tables.dims(),
        });
    }
    if scorer.dims() != config.dims {
        return Err(KgeError::Dims {
            expected: config.dims,
            got: scorer.dims(),
        });
    }
    if tables.num_entities() < store.num_entities() {
        return Err(KgeError::Invalid(
            "tables have fewer entities than the store".into(),
        ));
    }
    if tables.num_relations() < store.num_relations() {
        return Err(KgeError::Invalid(
            "tables have fewer relations than the store".into(),
        ));
    }
    Ok(())
}

/// Apply N3 to the three rows of a positive triple, accumulating gradients and
/// returning the total penalty.
fn apply_n3(
    tables: &Tables,
    t: crate::Triple,
    lambda: f32,
    grads: &mut Grads,
    buf: &mut Vec<f32>,
) -> Result<f32> {
    let mut penalty = 0.0;
    penalty += loss::n3_grad(tables.entity(t.s)?, lambda, buf);
    grads.add_entity(t.s, buf);
    penalty += loss::n3_grad(tables.relation(t.r)?, lambda, buf);
    grads.add_relation(t.r, buf);
    penalty += loss::n3_grad(tables.entity(t.o)?, lambda, buf);
    grads.add_entity(t.o, buf);
    Ok(penalty)
}

fn shuffle(order: &mut [usize], seed: u64, epoch: usize) {
    let mut rng = Rng::seeded(seed ^ 0x3C3C_5A5A_DEAD_BEEF ^ epoch as u64);
    for i in (1..order.len()).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        order.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
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
        let store =
            TripleStore::with_counts(triples, Some(num_entities), Some(num_relations)).unwrap();
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
        let c: TrainConfig = serde_json::from_str(
            r#"{"loss":{"kind":"self_adversarial"},"optimizer":{"kind":"adam"}}"#,
        )
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
}
