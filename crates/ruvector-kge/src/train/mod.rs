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
mod session;

pub use config::{LossKind, N3Form, OneNKernel, Reduction, TrainConfig};
pub use grad::Differentiable;
pub use init::Init;
pub use one_to_n::{NaiveOneToN, OneToN};
pub use optim::{OptimKind, OptimState, RowsState, StateLayout};
pub use session::{TrainSession, TrainState};

use crate::data::{Rng, TripleStore};
use crate::{KgeError, Result, Tables, Triple};
use optim::Grads;

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
    ///
    /// The batched 1-N kernel is chosen by [`TrainConfig::one_n_kernel`].
    pub fn fit(
        tables: &mut Tables,
        scorer: &dyn Differentiable,
        store: &TripleStore,
        config: &TrainConfig,
        callback: impl FnMut(&Progress),
    ) -> Result<()> {
        match config.one_n_kernel {
            OneNKernel::Naive => {
                Self::fit_with_kernel(tables, scorer, store, config, &NaiveOneToN, callback)
            }
            OneNKernel::Gemm => {
                let kernel = crate::kernel::GemmOneToN::new();
                Self::fit_with_kernel(tables, scorer, store, config, &kernel, callback)
            }
        }
    }

    /// [`Trainer::fit`] with an explicit batched 1-N kernel (used only when
    /// the scorer is bilinear and the loss is 1-vs-all). The kernel lane plugs
    /// its GEMM implementation in here.
    ///
    /// A loop over [`TrainSession`]: the session carries every piece of state
    /// between epochs, so a checkpointed-and-resumed session reproduces this
    /// function bit for bit (plan M3).
    pub fn fit_with_kernel(
        tables: &mut Tables,
        scorer: &dyn Differentiable,
        store: &TripleStore,
        config: &TrainConfig,
        kernel: &dyn OneToN,
        mut callback: impl FnMut(&Progress),
    ) -> Result<()> {
        let mut session = TrainSession::new(tables, scorer, store, config, kernel)?;
        if session.num_examples() == 0 {
            return Ok(());
        }
        for _ in 0..config.epochs {
            let progress = session.run_epoch(tables)?;
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
