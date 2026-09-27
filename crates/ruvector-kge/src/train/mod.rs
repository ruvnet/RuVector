//! CPU mini-batch training over the embedding [`Tables`] (ADR-003 §4,
//! ADR-007 §3).
//!
//! Deterministic in `TrainConfig::seed`: own PRNG for shuffling, negative
//! sampling and initialisation (no `rand`), per-row optimizer state (sparse or
//! dense, numerically identical), order-independent gradient application. The
//! scorer is any [`Differentiable`]; training never special-cases a scorer —
//! a scorer that is [`Bilinear`](crate::scorer::Bilinear) (reached through
//! [`Differentiable::as_bilinear`]) takes the batched 1-N path
//! ([`one_to_n`]) for 1-vs-all losses.

mod config;
pub mod grad;
pub mod init;
pub mod loss;
mod negatives;
pub mod one_to_n;
pub mod optim;
pub mod reciprocal;
mod rp;

pub use config::{LossKind, N3Form, Reduction, TrainConfig};
pub use grad::Differentiable;
pub use init::Init;
pub use one_to_n::{NaiveOneToN, OneToN};
pub use optim::{OptimKind, StateLayout};

use crate::data::{Rng, TripleStore};
use crate::{KgeError, Result, Tables, Triple};
use optim::{Grads, Optimizer};

/// Per-epoch progress handed to the callback. Carries no clock or timestamp
/// (ADR-005: no time in the core).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    pub epoch: usize,
    pub epochs: usize,
    pub num_batches: usize,
    /// Mean data-loss per training example over the epoch (per positive; per
    /// direction under `reciprocal`). Unaffected by `loss_reduction`.
    pub loss: f32,
    /// Mean N3 penalty per training example over the epoch.
    pub n3_penalty: f32,
    /// Mean weighted relation-prediction loss per example (0 when off).
    pub rp_loss: f32,
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
        callback: impl FnMut(&Progress),
    ) -> Result<()> {
        Self::fit_with_kernel(tables, scorer, store, config, &NaiveOneToN, callback)
    }

    /// [`Trainer::fit`] with an explicit batched 1-N kernel (used only when
    /// the scorer is bilinear and the loss is 1-vs-all). The kernel lane plugs
    /// its GEMM implementation in here.
    pub fn fit_with_kernel(
        tables: &mut Tables,
        scorer: &dyn Differentiable,
        store: &TripleStore,
        config: &TrainConfig,
        kernel: &dyn OneToN,
        mut callback: impl FnMut(&Progress),
    ) -> Result<()> {
        validate(tables, scorer, store, config)?;
        init::apply_init(tables, config.init, config.seed);

        let examples: Vec<Triple> = if config.reciprocal {
            reciprocal::augment(tables, store.triples())?
        } else {
            store.triples().to_vec()
        };
        let n = examples.len();
        if n == 0 {
            return Ok(());
        }
        let num_batches = n.div_ceil(config.batch_size);

        let bilinear = match config.loss {
            LossKind::OneVsAll => scorer.as_bilinear().filter(|b| b.index_is_identity()),
            LossKind::SelfAdversarial { .. } => None,
        };
        let (ne, nr, d) = (tables.num_entities(), tables.num_relations(), config.dims);
        // The batched path needs dense entity grads; dense vs sparse
        // accumulation is bitwise identical, so this changes storage only.
        let grad_layout = if bilinear.is_some() {
            StateLayout::Dense
        } else {
            config.optim_state
        };
        let mut grads = Grads::with_layout(grad_layout, d, ne, nr);
        let mut opt = Optimizer::new(config.optimizer, config.lr, config.optim_state, d, ne, nr);
        let mut sample_rng = Rng::seeded(config.seed ^ 0xA5A5_0F0F_1234_5678);
        let mut order: Vec<usize> = (0..n).collect();
        let mut n3_buf: Vec<f32> = Vec::new();
        let mut batch_ex: Vec<Triple> = Vec::with_capacity(config.batch_size);

        for epoch in 0..config.epochs {
            shuffle(&mut order, config.seed, epoch);
            let (mut epoch_loss, mut epoch_n3, mut epoch_rp) = (0.0f64, 0.0f64, 0.0f64);

            for batch in order.chunks(config.batch_size) {
                grads.clear();
                if config.loss_reduction == Reduction::Mean {
                    grads.set_scale(1.0 / batch.len() as f32);
                }
                batch_ex.clear();
                batch_ex.extend(batch.iter().map(|&i| examples[i]));

                if let Some(b) = bilinear {
                    epoch_loss += one_to_n::batched_step(
                        tables,
                        b,
                        kernel,
                        &batch_ex,
                        !config.reciprocal,
                        &mut grads,
                    )? as f64;
                }
                for &t in &batch_ex {
                    let data_loss = match config.loss {
                        _ if bilinear.is_some() => 0.0,
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
                        LossKind::OneVsAll if config.reciprocal => {
                            loss::one_vs_all_tail_step(tables, scorer, t, &mut grads)?
                        }
                        LossKind::OneVsAll => loss::one_vs_all_step(tables, scorer, t, &mut grads)?,
                    };
                    epoch_loss += data_loss as f64;
                    if config.rp_weight > 0.0 {
                        epoch_rp += rp::relation_prediction_step(
                            tables,
                            scorer,
                            t,
                            config.rp_weight,
                            &mut grads,
                        )? as f64;
                    }
                    if config.n3_lambda > 0.0 {
                        epoch_n3 += apply_n3(
                            tables,
                            t,
                            config.n3_lambda,
                            config.n3_form,
                            &mut grads,
                            &mut n3_buf,
                        )? as f64;
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
                rp_loss: (epoch_rp / n as f64) as f32,
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
    if config.reciprocal {
        reciprocal::validate(tables, store)?;
    }
    if !(config.rp_weight.is_finite() && config.rp_weight >= 0.0) {
        return Err(KgeError::Invalid(
            "rp_weight must be finite and >= 0".into(),
        ));
    }
    if config.n3_form == N3Form::Moduli && !config.dims.is_multiple_of(2) {
        return Err(KgeError::Invalid(
            "n3_form \"moduli\" needs even dims ([re; im] layout)".into(),
        ));
    }
    config.init.validate()
}

/// Apply N3 to the three rows of a training example, accumulating gradients
/// and returning the total penalty.
fn apply_n3(
    tables: &Tables,
    t: Triple,
    lambda: f32,
    form: N3Form,
    grads: &mut Grads,
    buf: &mut Vec<f32>,
) -> Result<f32> {
    let reg = match form {
        N3Form::Elementwise => loss::n3_grad,
        N3Form::Moduli => loss::n3_moduli_grad,
    };
    let mut penalty = 0.0;
    penalty += reg(tables.entity(t.s)?, lambda, buf);
    grads.add_entity(t.s, buf);
    penalty += reg(tables.relation(t.r)?, lambda, buf);
    grads.add_relation(t.r, buf);
    penalty += reg(tables.entity(t.o)?, lambda, buf);
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
mod tests;
#[cfg(test)]
mod tests_recipe;
#[cfg(test)]
mod tests_wiring;
