//! The real [`Evaluator`] (ADR-004): fits each proposal's [`Knobs`] on the
//! frozen `train` split and scores it against the current incumbent on `valid`,
//! with a transfer-holdout check and a token-gated `test` closure. It is the
//! seam the coordinator wires the trainer to — `campaign.rs` never mentions a
//! scorer or a table, only [`Evaluator::fit_and_eval`] / [`set_incumbent`].
//!
//! ## The paired-outcome mapping (documented, ADR-004 gate 2)
//!
//! The gate reads `(incumbent_correct, candidate_correct)` per validation query
//! and only *discordant* pairs move wealth (McNemar). We derive that pair from
//! a per-query **rank improvement** rather than an absolute Hits@k hit, because
//! HPO wins are small, consistent rank shifts the tuning loop must detect:
//!
//! | per-query ranks (lower is better) | encoded pair    | gate reads it as |
//! |-----------------------------------|-----------------|------------------|
//! | `candidate < incumbent` (better)  | `(false, true)` | champion win     |
//! | `candidate > incumbent` (worse)   | `(true, false)` | baseline win     |
//! | `candidate == incumbent`          | `(false, false)`| concordant, skip |
//!
//! Integer ranks (not `f32` reciprocals) make the tie exact. **Consequence:**
//! the receipt's `val.baseline_accuracy` / `val.champion_accuracy` (derived from
//! these booleans) are McNemar *win-fractions*, not Hits@k — the true quality
//! signal is `val_mrr`. See the [`ArmOutcome::val_paired`](super::ArmOutcome)
//! note.
//!
//! Deterministic: table init, training and tie-break RNGs are all seeded, so the
//! same [`TripleStore`] + [`Split4`] + seed reproduce every number.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use super::proposals::{Knobs, Loss, Optimizer, Proposal};
use super::{ArmOutcome, Evaluator, TestScoreFn};
use crate::data::{Split4, TripleStore};
use crate::eval::{evaluate, evaluate_ranks, EvalConfig};
use crate::scorer::{HolE, RotatE};
use crate::train::{Differentiable, LossKind, OptimKind, TrainConfig, Trainer};
use crate::{Result, ScorerKind, Tables, Triple};

/// Mini-batch size for every arm's training. Not an HPO axis (ADR-004 varies
/// dims/lr/loss/n3), so it is fixed rather than threaded through [`Knobs`].
const BATCH_SIZE: usize = 256;

/// `SelfAdversarial` margin — the trainer's serde default (ADR-003); not a
/// [`Knobs`] field, so the neg-sampling arms borrow it.
const SELF_ADVERSARIAL_MARGIN: f32 = 9.0;

/// Divisor turning the raw `epochs × triples` work into the normalised `cost`
/// in the bandit reward `val_mrr − λ·cost`. Chosen so a small-graph campaign's
/// cost is comparable in magnitude to its MRR deltas (a modest tie-break toward
/// cheaper arms, never a term that dominates the reward — by design).
const COST_SCALE: f32 = 100_000.0;

/// Everything one fit produced, kept only long enough for the *next*
/// [`Evaluator::set_incumbent`] to adopt it without retraining.
struct LastFit {
    id: u64,
    tables: Arc<Tables>,
    valid_ranks: Vec<usize>,
    transfer_mrr: f32,
}

/// Fits and evaluates KGE proposals over a frozen [`Split4`] for the campaign.
///
/// The `store` is the full graph — the filter set for filtered ranking — while
/// `split` is the frozen train/valid/transfer/test partition. Pairing is always
/// against the *current* incumbent's stored per-query ranks, which
/// [`set_incumbent`](Evaluator::set_incumbent) refreshes on every promotion.
pub struct TrainerEvaluator {
    store: Arc<TripleStore>,
    split: Arc<Split4>,
    seed: u64,
    eval_seed: u64,
    n_train: usize,
    incumbent_valid_ranks: Vec<usize>,
    incumbent_transfer_mrr: Option<f32>,
    last_fit: Option<LastFit>,
    /// Trained tables of every *promoted* proposal, keyed by its id, so the
    /// coordinator can install the champion's exact tables — the champion is the
    /// best-reward promoted arm, which need not be the last incumbent, so a
    /// single-slot cache would lose it.
    retained: HashMap<u64, Arc<Tables>>,
    test_scorings: Rc<Cell<u32>>,
}

impl TrainerEvaluator {
    /// Build an evaluator over `store` (the full graph) and its frozen `split`.
    /// `seed` drives table init and training; a fixed derived seed drives the
    /// eval tie-break so ranking is reproducible yet independent of training.
    #[must_use]
    pub fn new(store: TripleStore, split: Split4, seed: u64) -> Self {
        let n_train = split.train.len();
        Self {
            store: Arc::new(store),
            split: Arc::new(split),
            seed,
            eval_seed: seed ^ 0x0ED1_7A5C_9B3E_2F41,
            n_train,
            incumbent_valid_ranks: Vec::new(),
            incumbent_transfer_mrr: None,
            last_fit: None,
            retained: HashMap::new(),
            test_scorings: Rc::new(Cell::new(0)),
        }
    }

    /// The trained tables of a promoted proposal (the champion), for install.
    #[must_use]
    pub fn tables(&self, id: u64) -> Option<Arc<Tables>> {
        self.retained.get(&id).cloned()
    }

    /// How many times a `score_test` closure was invoked (must be exactly 2 per
    /// campaign: baseline + final champion — ADR-004 gate 1).
    #[must_use]
    pub fn test_scorings(&self) -> u32 {
        self.test_scorings.get()
    }

    /// Fit `knobs` on `train`, then measure per-query valid ranks and the
    /// transfer MRR, dispatching on the scorer kind (both are `Differentiable`).
    fn train_and_measure(
        &self,
        knobs: &Knobs,
        tables: &mut Tables,
        train_store: &TripleStore,
        cfg: &TrainConfig,
        ecfg: &EvalConfig,
    ) -> Result<(Vec<usize>, f32, u32)> {
        match knobs.scorer {
            ScorerKind::Hole => {
                let sc = HolE::new(knobs.dims)?;
                self.fit_measure(&sc, tables, train_store, cfg, ecfg)
            }
            ScorerKind::Rotate => {
                let sc = RotatE::new(knobs.dims)?;
                self.fit_measure(&sc, tables, train_store, cfg, ecfg)
            }
        }
    }

    fn fit_measure<S: Differentiable>(
        &self,
        scorer: &S,
        tables: &mut Tables,
        train_store: &TripleStore,
        cfg: &TrainConfig,
        ecfg: &EvalConfig,
    ) -> Result<(Vec<usize>, f32, u32)> {
        Trainer::fit(tables, scorer, train_store, cfg, |_| {})?;
        let valid_ranks = evaluate_ranks(tables, scorer, &self.store, &self.split.valid, ecfg)?;
        let (transfer_mrr, transfer_n) = if self.split.transfer.is_empty() {
            (0.0, 0)
        } else {
            let rep = evaluate(tables, scorer, &self.store, &self.split.transfer, ecfg)?;
            (rep.combined.mrr, rep.combined.count as u32)
        };
        Ok((valid_ranks, transfer_mrr, transfer_n))
    }

    /// An outcome that can never promote (empty paired evidence, zero MRR),
    /// used when a scorer/table build fails so a bad arm is refused, not panicked.
    fn degenerate_outcome(&self) -> ArmOutcome {
        let counter = self.test_scorings.clone();
        let n_test = self.split.test.len() as u32;
        ArmOutcome {
            val_paired: Vec::new(),
            val_mrr: 0.0,
            transfer_baseline: self.incumbent_transfer_mrr.unwrap_or(0.0),
            transfer_candidate: 0.0,
            transfer_n: self.split.transfer.len() as u32,
            cost: 0.0,
            score_test: Box::new(move || {
                counter.set(counter.get() + 1);
                (0.0, n_test)
            }),
        }
    }
}

impl Evaluator for TrainerEvaluator {
    fn fit_and_eval(&mut self, proposal: &Proposal) -> ArmOutcome {
        let knobs = &proposal.knobs;
        let dims = knobs.dims;
        let ne = self.store.num_entities().max(1);
        let nr = self.store.num_relations().max(1);
        let cfg = train_config(knobs, self.seed);
        let ecfg = EvalConfig::random(self.eval_seed);

        let train_store =
            match TripleStore::with_counts(self.split.train.clone(), Some(ne), Some(nr)) {
                Ok(s) => s,
                Err(_) => return self.degenerate_outcome(),
            };
        let mut tables = Tables::new(ne, nr, dims, self.seed);

        let (valid_ranks, transfer_mrr, transfer_n) =
            match self.train_and_measure(knobs, &mut tables, &train_store, &cfg, &ecfg) {
                Ok(v) => v,
                Err(_) => return self.degenerate_outcome(),
            };

        let val_mrr = mrr_of(&valid_ranks);
        let val_paired = pair_ranks(&self.incumbent_valid_ranks, &valid_ranks);
        let transfer_baseline = self.incumbent_transfer_mrr.unwrap_or(transfer_mrr);
        let cost = (knobs.epochs as f32 * self.n_train as f32) / COST_SCALE;

        // Retain the trained tables so a following `set_incumbent` adopts them
        // (and the champion install reads them) without a second training run.
        let arc_tables = Arc::new(tables);
        self.last_fit = Some(LastFit {
            id: proposal.id,
            tables: arc_tables.clone(),
            valid_ranks,
            transfer_mrr,
        });

        // Lazy, token-gated test scoring (ADR-004 gate 1): the campaign mints
        // exactly two `TestToken`s, so this closure runs only for the baseline
        // and the final champion.
        let counter = self.test_scorings.clone();
        let store = self.store.clone();
        let split = self.split.clone();
        let scorer_kind = knobs.scorer;
        let eval_seed = self.eval_seed;
        let score_test: TestScoreFn = Box::new(move || {
            counter.set(counter.get() + 1);
            test_mrr(
                scorer_kind,
                dims,
                &arc_tables,
                &store,
                &split.test,
                eval_seed,
            )
        });

        ArmOutcome {
            val_paired,
            val_mrr,
            transfer_baseline,
            transfer_candidate: transfer_mrr,
            transfer_n,
            cost,
            score_test,
        }
    }

    fn set_incumbent(&mut self, proposal: &Proposal) {
        let ready = self.last_fit.as_ref().is_some_and(|f| f.id == proposal.id);
        if !ready {
            // Defensive: the campaign always fits immediately before promoting,
            // so this fits an un-fitted incumbent rather than reusing stale data.
            let _ = self.fit_and_eval(proposal);
        }
        if let Some(fit) = self.last_fit.take() {
            if fit.id == proposal.id {
                self.incumbent_valid_ranks = fit.valid_ranks;
                self.incumbent_transfer_mrr = Some(fit.transfer_mrr);
                self.retained.insert(fit.id, fit.tables);
            }
        }
    }
}

/// Map [`Knobs`] onto a trainer [`TrainConfig`]. `CrossEntropy` is the trainer's
/// 1-vs-all loss; `Bce`/`Margin` are the negative-sampling family, so both route
/// to `SelfAdversarial` (`Knobs` has no per-loss margin, so the trainer default
/// is used). `batch_size` is fixed ([`BATCH_SIZE`]); `dims` is set by the arm.
fn train_config(knobs: &Knobs, seed: u64) -> TrainConfig {
    TrainConfig {
        dims: knobs.dims,
        epochs: knobs.epochs,
        batch_size: BATCH_SIZE,
        lr: knobs.lr,
        optimizer: match knobs.optimizer {
            Optimizer::Adam => OptimKind::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            Optimizer::Adagrad => OptimKind::Adagrad { epsilon: 1e-8 },
        },
        loss: match knobs.loss {
            Loss::CrossEntropy => LossKind::OneVsAll,
            Loss::Bce | Loss::Margin => LossKind::SelfAdversarial {
                neg_count: knobs.neg_count,
                temperature: knobs.temperature,
                margin: SELF_ADVERSARIAL_MARGIN,
            },
        },
        n3_lambda: knobs.n3_lambda,
        seed,
    }
}

/// Encode per-query candidate-vs-incumbent ranks into the gate's paired form
/// (see the module docs for the full mapping table). Lower rank is better, so
/// `candidate < incumbent` is a champion win.
fn pair_ranks(incumbent: &[usize], candidate: &[usize]) -> Vec<(bool, bool)> {
    incumbent
        .iter()
        .zip(candidate)
        .map(|(&i, &c)| {
            if c < i {
                (false, true)
            } else if c > i {
                (true, false)
            } else {
                (false, false)
            }
        })
        .collect()
}

/// Mean reciprocal rank of a per-query integer-rank vector.
fn mrr_of(ranks: &[usize]) -> f32 {
    if ranks.is_empty() {
        return 0.0;
    }
    let sum: f64 = ranks.iter().map(|&r| 1.0 / r as f64).sum();
    (sum / ranks.len() as f64) as f32
}

/// Filtered combined test MRR of one arm's trained `tables`, dispatched on the
/// scorer kind. A build/eval failure scores 0 (a champion that cannot be scored
/// is reported as no gain, never a panic).
fn test_mrr(
    kind: ScorerKind,
    dims: usize,
    tables: &Tables,
    filter: &TripleStore,
    test: &[Triple],
    seed: u64,
) -> (f32, u32) {
    let ecfg = EvalConfig::random(seed);
    let report = match kind {
        ScorerKind::Hole => match HolE::new(dims) {
            Ok(sc) => evaluate(tables, &sc, filter, test, &ecfg),
            Err(e) => Err(e),
        },
        ScorerKind::Rotate => match RotatE::new(dims) {
            Ok(sc) => evaluate(tables, &sc, filter, test, &ecfg),
            Err(e) => Err(e),
        },
    };
    match report {
        Ok(r) => (r.combined.mrr, r.combined.count as u32),
        Err(_) => (0.0, test.len() as u32),
    }
}

#[cfg(test)]
mod tests;
