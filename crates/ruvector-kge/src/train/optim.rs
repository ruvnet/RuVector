//! Sparse per-row optimizers (Adagrad, Adam) over the embedding [`Tables`].
//!
//! Only rows touched in a batch are updated. State is keyed by row id in a
//! `BTreeMap` so accumulation and iteration are order-deterministic (no
//! `RandomState`), which keeps training reproducible for a given seed. Each
//! row is updated independently, so update order never affects the result.

use crate::{Result, Tables};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Which optimizer a run uses. Betas/epsilon are carried here with defaults so
/// the schedule is a single HPO arm (ADR-003 §3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind")]
pub enum OptimKind {
    Adagrad {
        #[serde(default = "eps_default")]
        epsilon: f32,
    },
    Adam {
        #[serde(default = "beta1_default")]
        beta1: f32,
        #[serde(default = "beta2_default")]
        beta2: f32,
        #[serde(default = "eps_default")]
        epsilon: f32,
    },
}

fn eps_default() -> f32 {
    1e-8
}
fn beta1_default() -> f32 {
    0.9
}
fn beta2_default() -> f32 {
    0.999
}

impl Default for OptimKind {
    fn default() -> Self {
        OptimKind::Adagrad {
            epsilon: eps_default(),
        }
    }
}

/// Accumulates loss gradients per touched embedding row within one batch.
#[derive(Debug)]
pub(crate) struct Grads {
    dims: usize,
    ent: BTreeMap<u32, Vec<f32>>,
    rel: BTreeMap<u32, Vec<f32>>,
}

impl Grads {
    pub(crate) fn new(dims: usize) -> Self {
        Self {
            dims,
            ent: BTreeMap::new(),
            rel: BTreeMap::new(),
        }
    }

    pub(crate) fn add_entity(&mut self, id: u32, g: &[f32]) {
        accumulate(
            self.ent.entry(id).or_insert_with(|| vec![0.0; self.dims]),
            g,
        );
    }
    pub(crate) fn add_relation(&mut self, id: u32, g: &[f32]) {
        accumulate(
            self.rel.entry(id).or_insert_with(|| vec![0.0; self.dims]),
            g,
        );
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ent.is_empty() && self.rel.is_empty()
    }
}

fn accumulate(dst: &mut [f32], g: &[f32]) {
    for (d, &v) in dst.iter_mut().zip(g) {
        *d += v;
    }
}

/// Per-row optimizer state.
enum State {
    Adagrad {
        ent: BTreeMap<u32, Vec<f32>>,
        rel: BTreeMap<u32, Vec<f32>>,
    },
    Adam {
        ent_m: BTreeMap<u32, Vec<f32>>,
        ent_v: BTreeMap<u32, Vec<f32>>,
        rel_m: BTreeMap<u32, Vec<f32>>,
        rel_v: BTreeMap<u32, Vec<f32>>,
        t: u64,
    },
}

/// A stateful optimizer applied to touched rows once per batch.
pub(crate) struct Optimizer {
    kind: OptimKind,
    lr: f32,
    state: State,
}

impl Optimizer {
    pub(crate) fn new(kind: OptimKind, lr: f32) -> Self {
        let state = match kind {
            OptimKind::Adagrad { .. } => State::Adagrad {
                ent: BTreeMap::new(),
                rel: BTreeMap::new(),
            },
            OptimKind::Adam { .. } => State::Adam {
                ent_m: BTreeMap::new(),
                ent_v: BTreeMap::new(),
                rel_m: BTreeMap::new(),
                rel_v: BTreeMap::new(),
                t: 0,
            },
        };
        Self { kind, lr, state }
    }

    /// Apply one batch of accumulated gradients to the tables. Rows are
    /// updated independently; iteration order does not affect the result.
    pub(crate) fn apply(&mut self, tables: &mut Tables, grads: &Grads) -> Result<()> {
        let lr = self.lr;
        match (&self.kind, &mut self.state) {
            (OptimKind::Adagrad { epsilon }, State::Adagrad { ent, rel }) => {
                let eps = *epsilon;
                for (&id, g) in &grads.ent {
                    let acc = ent.entry(id).or_insert_with(|| vec![0.0; g.len()]);
                    adagrad_step(tables.entity_mut(id)?, g, acc, lr, eps);
                }
                for (&id, g) in &grads.rel {
                    let acc = rel.entry(id).or_insert_with(|| vec![0.0; g.len()]);
                    adagrad_step(tables.relation_mut(id)?, g, acc, lr, eps);
                }
            }
            (
                OptimKind::Adam {
                    beta1,
                    beta2,
                    epsilon,
                },
                State::Adam {
                    ent_m,
                    ent_v,
                    rel_m,
                    rel_v,
                    t,
                },
            ) => {
                *t += 1;
                let (b1, b2, eps) = (*beta1, *beta2, *epsilon);
                let bc1 = 1.0 - b1.powi(*t as i32);
                let bc2 = 1.0 - b2.powi(*t as i32);
                for (&id, g) in &grads.ent {
                    let m = ent_m.entry(id).or_insert_with(|| vec![0.0; g.len()]);
                    let v = ent_v.entry(id).or_insert_with(|| vec![0.0; g.len()]);
                    adam_step(tables.entity_mut(id)?, g, m, v, lr, b1, b2, eps, bc1, bc2);
                }
                for (&id, g) in &grads.rel {
                    let m = rel_m.entry(id).or_insert_with(|| vec![0.0; g.len()]);
                    let v = rel_v.entry(id).or_insert_with(|| vec![0.0; g.len()]);
                    adam_step(tables.relation_mut(id)?, g, m, v, lr, b1, b2, eps, bc1, bc2);
                }
            }
            _ => unreachable!("optimizer kind and state constructed together"),
        }
        Ok(())
    }
}

fn adagrad_step(row: &mut [f32], g: &[f32], acc: &mut [f32], lr: f32, eps: f32) {
    for ((p, &gi), a) in row.iter_mut().zip(g).zip(acc.iter_mut()) {
        *a += gi * gi;
        *p -= lr * gi / (a.sqrt() + eps);
    }
}

#[allow(clippy::too_many_arguments)]
fn adam_step(
    row: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
) {
    for (((p, &gi), mi), vi) in row.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
        *mi = b1 * *mi + (1.0 - b1) * gi;
        *vi = b2 * *vi + (1.0 - b2) * gi * gi;
        let mhat = *mi / bc1;
        let vhat = *vi / bc2;
        *p -= lr * mhat / (vhat.sqrt() + eps);
    }
}
