//! An epoch-at-a-time training session with exportable state (plan M3:
//! checkpoint + resume). [`Trainer::fit_with_kernel`](super::Trainer) is a
//! thin loop over this type, so a session that is stopped after epoch `k`,
//! exported with [`TrainSession::export_state`], and rebuilt with
//! [`TrainSession::resume`] over the same tables continues **bit-identically**
//! to an uninterrupted `fit`: the state is exactly what the loop carries
//! between epochs — the persistent shuffle permutation, the negative-sampling
//! RNG, the optimizer rows, and the next epoch index. Gradients are
//! per-batch scratch and are not state.
//!
//! The core stays fs-free (ADR-005): [`TrainState`] is plain data; callers
//! (the bench binary) serialise it.

use super::optim::{Grads, OptimState, Optimizer, StateLayout};
use super::{apply_n3, init, loss, one_to_n, reciprocal, rp, shuffle, validate};
use super::{Differentiable, LossKind, OneToN, Progress, Reduction, TrainConfig};
use crate::data::{Rng, TripleStore};
use crate::scorer::Bilinear;
use crate::{KgeError, Result, Tables, Triple};

/// Everything a session carries from one epoch to the next.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainState {
    /// Index of the next epoch to run (= epochs completed).
    pub next_epoch: usize,
    /// The persistent example permutation (shuffled in place every epoch).
    pub order: Vec<u32>,
    /// Negative-sampling RNG state (consumed only by `SelfAdversarial`).
    pub sample_rng: u64,
    /// Optimizer rows and step counter.
    pub optimizer: OptimState,
}

/// One training run, advanced one epoch at a time.
pub struct TrainSession<'a> {
    scorer: &'a dyn Differentiable,
    kernel: &'a dyn OneToN,
    bilinear: Option<&'a dyn Bilinear>,
    config: TrainConfig,
    examples: Vec<Triple>,
    grads: Grads,
    opt: Optimizer,
    sample_rng: Rng,
    order: Vec<usize>,
    n3_buf: Vec<f32>,
    batch_ex: Vec<Triple>,
    next_epoch: usize,
}

impl<'a> TrainSession<'a> {
    /// Start a fresh run: validate, apply `config.init` to `tables`, and build
    /// the example list and optimizer (exactly what `fit` does before epoch 0).
    pub fn new(
        tables: &mut Tables,
        scorer: &'a dyn Differentiable,
        store: &TripleStore,
        config: &TrainConfig,
        kernel: &'a dyn OneToN,
    ) -> Result<Self> {
        validate(tables, scorer, store, config)?;
        init::apply_init(tables, config.init, config.seed);
        Self::build(tables, scorer, store, config, kernel)
    }

    /// Continue a run from `state` over `tables` as they were when the state
    /// was exported. `config.init` is **not** applied. Errors if the state
    /// does not fit this store/config (wrong example count, bad permutation,
    /// optimizer shape or kind mismatch).
    pub fn resume(
        tables: &Tables,
        scorer: &'a dyn Differentiable,
        store: &TripleStore,
        config: &TrainConfig,
        kernel: &'a dyn OneToN,
        state: &TrainState,
    ) -> Result<Self> {
        validate(tables, scorer, store, config)?;
        let mut s = Self::build(tables, scorer, store, config, kernel)?;
        let n = s.examples.len();
        if state.order.len() != n {
            return Err(KgeError::Invalid(format!(
                "resume: state has {} examples, this run has {n}",
                state.order.len()
            )));
        }
        let mut seen = vec![false; n];
        for &i in &state.order {
            let i = i as usize;
            if i >= n || std::mem::replace(&mut seen[i], true) {
                return Err(KgeError::Invalid(
                    "resume: order is not a permutation".into(),
                ));
            }
        }
        s.order = state.order.iter().map(|&i| i as usize).collect();
        s.sample_rng = Rng::from_state(state.sample_rng)?;
        s.opt.import_state(&state.optimizer)?;
        s.next_epoch = state.next_epoch;
        Ok(s)
    }

    fn build(
        tables: &Tables,
        scorer: &'a dyn Differentiable,
        store: &TripleStore,
        config: &TrainConfig,
        kernel: &'a dyn OneToN,
    ) -> Result<Self> {
        let examples: Vec<Triple> = if config.reciprocal {
            reciprocal::augment(tables, store.triples())?
        } else {
            store.triples().to_vec()
        };
        if examples.len() > u32::MAX as usize {
            return Err(KgeError::Limit("triples"));
        }
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
        let n = examples.len();
        Ok(Self {
            scorer,
            kernel,
            bilinear,
            config: *config,
            examples,
            grads: Grads::with_layout(grad_layout, d, ne, nr),
            opt: Optimizer::new(config.optimizer, config.lr, config.optim_state, d, ne, nr),
            sample_rng: Rng::seeded(config.seed ^ 0xA5A5_0F0F_1234_5678),
            order: (0..n).collect(),
            n3_buf: Vec::new(),
            batch_ex: Vec::with_capacity(config.batch_size),
            next_epoch: 0,
        })
    }

    /// Number of training examples per epoch (reciprocal-augmented).
    pub fn num_examples(&self) -> usize {
        self.examples.len()
    }

    /// Index of the next epoch [`run_epoch`](Self::run_epoch) will run.
    pub fn next_epoch(&self) -> usize {
        self.next_epoch
    }

    /// Snapshot of the between-epoch state (pair it with the tables).
    pub fn export_state(&self) -> TrainState {
        TrainState {
            next_epoch: self.next_epoch,
            order: self.order.iter().map(|&i| i as u32).collect(),
            sample_rng: self.sample_rng.state(),
            optimizer: self.opt.export_state(),
        }
    }

    /// Run one epoch over `tables` and return its progress record. Does not
    /// consult `config.epochs` (the caller decides when to stop).
    pub fn run_epoch(&mut self, tables: &mut Tables) -> Result<Progress> {
        let config = self.config;
        let epoch = self.next_epoch;
        let n = self.examples.len();
        let num_batches = n.div_ceil(config.batch_size);
        shuffle(&mut self.order, config.seed, epoch);
        let (mut epoch_loss, mut epoch_n3, mut epoch_rp) = (0.0f64, 0.0f64, 0.0f64);

        for batch in self.order.chunks(config.batch_size) {
            let grads = &mut self.grads;
            grads.clear();
            if config.loss_reduction == Reduction::Mean {
                grads.set_scale(1.0 / batch.len() as f32);
            }
            self.batch_ex.clear();
            self.batch_ex
                .extend(batch.iter().map(|&i| self.examples[i]));

            if let Some(b) = self.bilinear {
                epoch_loss += one_to_n::batched_step(
                    tables,
                    b,
                    self.kernel,
                    &self.batch_ex,
                    !config.reciprocal,
                    grads,
                )? as f64;
            }
            for &t in &self.batch_ex {
                let data_loss = match config.loss {
                    _ if self.bilinear.is_some() => 0.0,
                    LossKind::SelfAdversarial {
                        neg_count,
                        temperature,
                        margin,
                    } => loss::self_adversarial_step(
                        tables,
                        self.scorer,
                        t,
                        neg_count,
                        temperature,
                        margin,
                        &mut self.sample_rng,
                        grads,
                    )?,
                    LossKind::OneVsAll if config.reciprocal => {
                        loss::one_vs_all_tail_step(tables, self.scorer, t, grads)?
                    }
                    LossKind::OneVsAll => loss::one_vs_all_step(tables, self.scorer, t, grads)?,
                };
                epoch_loss += data_loss as f64;
                if config.rp_weight > 0.0 {
                    epoch_rp += rp::relation_prediction_step(
                        tables,
                        self.scorer,
                        t,
                        config.rp_weight,
                        grads,
                    )? as f64;
                }
                if config.n3_lambda > 0.0 {
                    epoch_n3 += apply_n3(
                        tables,
                        t,
                        config.n3_lambda,
                        config.n3_form,
                        grads,
                        &mut self.n3_buf,
                    )? as f64;
                }
            }
            if !grads.is_empty() {
                self.opt.apply(tables, grads)?;
            }
        }
        self.next_epoch += 1;
        let denom = n.max(1) as f64;
        Ok(Progress {
            epoch,
            epochs: config.epochs,
            num_batches,
            loss: (epoch_loss / denom) as f32,
            n3_penalty: (epoch_n3 / denom) as f32,
            rp_loss: (epoch_rp / denom) as f32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Init, N3Form, OneNKernel, OptimKind, Trainer};
    use super::*;
    use crate::scorer::complex::ComplEx;
    use crate::{HolE, Tables, Triple};

    fn store(ne: usize, nr: usize) -> TripleStore {
        let t: Vec<Triple> = (0..40u32)
            .map(|i| Triple::new(i % ne as u32, i % nr as u32, (i * 7 + 3) % ne as u32))
            .collect();
        TripleStore::with_counts(t, Some(ne), Some(nr)).unwrap()
    }

    fn cfg(
        optimizer: OptimKind,
        loss: LossKind,
        layout: StateLayout,
        reciprocal: bool,
    ) -> TrainConfig {
        TrainConfig {
            dims: 8,
            epochs: 5,
            batch_size: 7,
            lr: 0.05,
            optimizer,
            loss,
            n3_lambda: 0.01,
            seed: 23,
            reciprocal,
            init: Init::Normal { scale: 0.2 },
            n3_form: N3Form::Moduli,
            loss_reduction: Reduction::Mean,
            rp_weight: 0.1,
            optim_state: layout,
            one_n_kernel: OneNKernel::Gemm,
        }
    }

    /// fit(5) == session(2) → export → resume → session(3), bitwise, across
    /// optimizer kinds, layouts, the batched 1-N path (ComplEx) and the
    /// RNG-consuming SelfAdversarial path (HolE).
    #[test]
    fn resume_is_bitwise_identical_to_uninterrupted_fit() {
        let (ne, nr) = (15usize, 3usize);
        let st = store(ne, nr);
        let adam = OptimKind::Adam {
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 1e-6,
        };
        let adagrad = OptimKind::Adagrad { epsilon: 1e-3 };
        let sa = LossKind::SelfAdversarial {
            neg_count: 4,
            temperature: 1.0,
            margin: 3.0,
        };
        let complex = ComplEx::new(8).unwrap();
        let hole = HolE::new(8).unwrap();
        let cases: Vec<(&dyn Differentiable, TrainConfig)> = vec![
            (
                &complex,
                cfg(adagrad, LossKind::OneVsAll, StateLayout::Dense, true),
            ),
            (
                &complex,
                cfg(adam, LossKind::OneVsAll, StateLayout::Sparse, true),
            ),
            (
                &complex,
                cfg(adam, LossKind::OneVsAll, StateLayout::Dense, false),
            ),
            (&hole, cfg(adam, sa, StateLayout::Sparse, false)),
            (&hole, cfg(adagrad, sa, StateLayout::Dense, false)),
        ];
        let kernel = crate::kernel::GemmOneToN::new();
        for (i, (scorer, c)) in cases.into_iter().enumerate() {
            let rel_rows = if c.reciprocal { 2 * nr } else { nr };
            let mut full = Tables::new(ne, rel_rows, 8, 5);
            let mut full_losses = Vec::new();
            Trainer::fit_with_kernel(&mut full, scorer, &st, &c, &kernel, |p| {
                full_losses.push(p.loss.to_bits())
            })
            .unwrap();

            let mut part = Tables::new(ne, rel_rows, 8, 5);
            let mut losses = Vec::new();
            let state = {
                let mut s = TrainSession::new(&mut part, scorer, &st, &c, &kernel).unwrap();
                for _ in 0..2 {
                    losses.push(s.run_epoch(&mut part).unwrap().loss.to_bits());
                }
                s.export_state()
            };
            // A "new process": fresh tables cloned from the checkpointed rows.
            let mut resumed = part.clone();
            let mut s = TrainSession::resume(&resumed, scorer, &st, &c, &kernel, &state).unwrap();
            assert_eq!(s.next_epoch(), 2);
            for _ in 2..5 {
                losses.push(s.run_epoch(&mut resumed).unwrap().loss.to_bits());
            }
            assert_eq!(losses, full_losses, "case {i}: per-epoch losses diverged");
            assert_eq!(
                resumed.entities_raw(),
                full.entities_raw(),
                "case {i}: entities"
            );
            assert_eq!(
                resumed.relations_raw(),
                full.relations_raw(),
                "case {i}: relations"
            );
            // State round-trips exactly.
            assert_eq!(s.export_state().next_epoch, 5);
        }
    }

    #[test]
    fn resume_rejects_foreign_state() {
        let (ne, nr) = (15usize, 3usize);
        let st = store(ne, nr);
        let c = cfg(
            OptimKind::Adagrad { epsilon: 1e-3 },
            LossKind::OneVsAll,
            StateLayout::Dense,
            true,
        );
        let complex = ComplEx::new(8).unwrap();
        let kernel = crate::kernel::GemmOneToN::new();
        let mut t = Tables::new(ne, 2 * nr, 8, 5);
        let good = TrainSession::new(&mut t, &complex, &st, &c, &kernel)
            .unwrap()
            .export_state();

        let mut bad = good.clone();
        bad.order.pop();
        assert!(TrainSession::resume(&t, &complex, &st, &c, &kernel, &bad).is_err());
        let mut bad = good.clone();
        bad.order[0] = bad.order[1];
        assert!(TrainSession::resume(&t, &complex, &st, &c, &kernel, &bad).is_err());
        let mut bad = good.clone();
        bad.sample_rng = 0;
        assert!(TrainSession::resume(&t, &complex, &st, &c, &kernel, &bad).is_err());
        let mut bad = good.clone();
        bad.optimizer.rows.push(RowsStateAlias::default());
        assert!(TrainSession::resume(&t, &complex, &st, &c, &kernel, &bad).is_err());
        let mut bad = good;
        bad.optimizer.rows[0] = RowsStateAlias {
            ids: vec![ne as u32],
            data: vec![0.0; 8],
        };
        assert!(TrainSession::resume(&t, &complex, &st, &c, &kernel, &bad).is_err());
    }

    use super::super::optim::RowsState as RowsStateAlias;
}
