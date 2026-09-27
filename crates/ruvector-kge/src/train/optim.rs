//! Per-row optimizers (Adagrad, Adam) over the embedding [`Tables`], with a
//! sparse or dense storage layout for both gradients and optimizer state.
//!
//! Only rows touched in a batch are updated ("lazy" semantics, in both
//! layouts). Sparse state is keyed by row id in a `BTreeMap`; dense state is a
//! flat `rows × dims` buffer (Lacroix 2018 runs dense Adagrad — with 1-N every
//! entity row is touched every batch, so the map is pure overhead). Touched
//! rows are always visited in ascending id order and each row is updated
//! independently, so the two layouts produce **bitwise-identical** tables:
//! the layout is a storage choice, never a numerical one. For Adagrad this is
//! also exactly torch's dense `Adagrad` (a zero-gradient row is a no-op there
//! too); for Adam, untouched rows keep their moments frozen (lazy Adam). The
//! batched 1-N path marks every entity row touched (softmax gives each a
//! gradient), so there entity rows update every batch, as in dense Adam.

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

/// Storage layout for per-row gradients and optimizer state
/// (`TrainConfig::optim_state`). Numerically identical; `Dense` is the fast
/// choice for 1-N training, `Sparse` for negative sampling on large tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StateLayout {
    #[default]
    Sparse,
    Dense,
}

/// One table's worth (entities or relations) of per-row vectors.
#[derive(Debug)]
enum Rows {
    Sparse(BTreeMap<u32, Vec<f32>>),
    Dense { buf: Vec<f32>, touched: Vec<bool> },
}

impl Rows {
    fn new(layout: StateLayout, rows: usize, dims: usize) -> Self {
        match layout {
            StateLayout::Sparse => Rows::Sparse(BTreeMap::new()),
            StateLayout::Dense => Rows::Dense {
                buf: vec![0.0; rows * dims],
                touched: vec![false; rows],
            },
        }
    }

    /// The row for `id`, zero-initialised on first use. Marks it touched.
    fn row_mut(&mut self, id: u32, dims: usize) -> &mut [f32] {
        match self {
            Rows::Sparse(m) => m.entry(id).or_insert_with(|| vec![0.0; dims]),
            Rows::Dense { buf, touched } => {
                let i = id as usize;
                touched[i] = true;
                &mut buf[i * dims..(i + 1) * dims]
            }
        }
    }

    /// Touched rows in ascending id order.
    fn iter(&self, dims: usize) -> Box<dyn Iterator<Item = (u32, &[f32])> + '_> {
        match self {
            Rows::Sparse(m) => Box::new(m.iter().map(|(&id, v)| (id, v.as_slice()))),
            Rows::Dense { buf, touched } => Box::new(
                touched
                    .iter()
                    .enumerate()
                    .filter(|(_, &t)| t)
                    .map(move |(i, _)| (i as u32, &buf[i * dims..(i + 1) * dims])),
            ),
        }
    }

    #[cfg(test)]
    fn get(&self, id: u32, dims: usize) -> Option<&[f32]> {
        match self {
            Rows::Sparse(m) => m.get(&id).map(|v| v.as_slice()),
            Rows::Dense { buf, touched } => {
                let i = id as usize;
                (i < touched.len() && touched[i]).then(|| &buf[i * dims..(i + 1) * dims])
            }
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Rows::Sparse(m) => m.is_empty(),
            Rows::Dense { touched, .. } => !touched.iter().any(|&t| t),
        }
    }

    /// Reset to "nothing touched" (dense keeps its allocation).
    fn clear(&mut self, dims: usize) {
        match self {
            Rows::Sparse(m) => m.clear(),
            Rows::Dense { buf, touched } => {
                for (i, t) in touched.iter_mut().enumerate() {
                    if *t {
                        buf[i * dims..(i + 1) * dims].fill(0.0);
                        *t = false;
                    }
                }
            }
        }
    }
}

/// Accumulates loss gradients per touched embedding row within one batch.
///
/// Every `add_*` multiplies by the batch `scale` (1 for summed losses, `1/B`
/// for batch-averaged ones); `x * 1.0 == x` exactly, so the summed default is
/// bitwise unchanged. `*_prescaled` adds values that already carry the scale
/// (the batched 1-N kernel scales its own outputs).
#[derive(Debug)]
pub(crate) struct Grads {
    dims: usize,
    scale: f32,
    ent: Rows,
    rel: Rows,
}

impl Grads {
    /// Sparse accumulator (any table size; tests / oracles).
    #[cfg(test)]
    pub(crate) fn new(dims: usize) -> Self {
        Self {
            dims,
            scale: 1.0,
            ent: Rows::new(StateLayout::Sparse, 0, dims),
            rel: Rows::new(StateLayout::Sparse, 0, dims),
        }
    }

    /// Accumulator in `layout` for tables of `entities` × `relations` rows.
    pub(crate) fn with_layout(
        layout: StateLayout,
        dims: usize,
        entities: usize,
        relations: usize,
    ) -> Self {
        Self {
            dims,
            scale: 1.0,
            ent: Rows::new(layout, entities, dims),
            rel: Rows::new(layout, relations, dims),
        }
    }

    pub(crate) fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    pub(crate) fn add_entity(&mut self, id: u32, g: &[f32]) {
        let s = self.scale;
        accumulate(self.ent.row_mut(id, self.dims), g, s);
    }
    pub(crate) fn add_relation(&mut self, id: u32, g: &[f32]) {
        let s = self.scale;
        accumulate(self.rel.row_mut(id, self.dims), g, s);
    }
    pub(crate) fn add_entity_prescaled(&mut self, id: u32, g: &[f32]) {
        accumulate(self.ent.row_mut(id, self.dims), g, 1.0);
    }
    pub(crate) fn add_relation_prescaled(&mut self, id: u32, g: &[f32]) {
        accumulate(self.rel.row_mut(id, self.dims), g, 1.0);
    }

    /// The whole dense `E × dims` entity gradient buffer, every row marked
    /// touched — the batched 1-N kernel accumulates into it directly. `None`
    /// for a sparse accumulator.
    pub(crate) fn dense_entities_mut(&mut self) -> Option<&mut [f32]> {
        match &mut self.ent {
            Rows::Sparse(_) => None,
            Rows::Dense { buf, touched } => {
                touched.fill(true);
                Some(buf.as_mut_slice())
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ent.is_empty() && self.rel.is_empty()
    }

    /// Forget all accumulated rows (keeps a dense allocation) and reset the
    /// scale to 1.
    pub(crate) fn clear(&mut self) {
        self.ent.clear(self.dims);
        self.rel.clear(self.dims);
        self.scale = 1.0;
    }

    /// Accumulated gradient of entity row `id`, if touched (tests / oracles).
    #[cfg(test)]
    pub(crate) fn entity(&self, id: u32) -> Option<&[f32]> {
        self.ent.get(id, self.dims)
    }
    /// Accumulated gradient of relation row `id`, if touched.
    #[cfg(test)]
    pub(crate) fn relation(&self, id: u32) -> Option<&[f32]> {
        self.rel.get(id, self.dims)
    }
}

fn accumulate(dst: &mut [f32], g: &[f32], scale: f32) {
    for (d, &v) in dst.iter_mut().zip(g) {
        *d += v * scale;
    }
}

/// Per-row optimizer state.
enum State {
    Adagrad {
        ent: Rows,
        rel: Rows,
    },
    Adam {
        ent_m: Rows,
        ent_v: Rows,
        rel_m: Rows,
        rel_v: Rows,
        t: u64,
    },
}

/// A stateful optimizer applied to touched rows once per batch.
pub(crate) struct Optimizer {
    kind: OptimKind,
    lr: f32,
    dims: usize,
    state: State,
}

impl Optimizer {
    /// `entities`/`relations` size the dense layout (ignored when sparse).
    pub(crate) fn new(
        kind: OptimKind,
        lr: f32,
        layout: StateLayout,
        dims: usize,
        entities: usize,
        relations: usize,
    ) -> Self {
        let rows = |n| Rows::new(layout, n, dims);
        let state = match kind {
            OptimKind::Adagrad { .. } => State::Adagrad {
                ent: rows(entities),
                rel: rows(relations),
            },
            OptimKind::Adam { .. } => State::Adam {
                ent_m: rows(entities),
                ent_v: rows(entities),
                rel_m: rows(relations),
                rel_v: rows(relations),
                t: 0,
            },
        };
        Self {
            kind,
            lr,
            dims,
            state,
        }
    }

    /// Apply one batch of accumulated gradients to the tables. Rows are
    /// updated independently, in ascending id order.
    pub(crate) fn apply(&mut self, tables: &mut Tables, grads: &Grads) -> Result<()> {
        let (lr, d) = (self.lr, self.dims);
        match (&self.kind, &mut self.state) {
            (OptimKind::Adagrad { epsilon }, State::Adagrad { ent, rel }) => {
                let eps = *epsilon;
                for (id, g) in grads.ent.iter(grads.dims) {
                    adagrad_step(tables.entity_mut(id)?, g, ent.row_mut(id, d), lr, eps);
                }
                for (id, g) in grads.rel.iter(grads.dims) {
                    adagrad_step(tables.relation_mut(id)?, g, rel.row_mut(id, d), lr, eps);
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
                let h = AdamHyper {
                    lr,
                    b1,
                    b2,
                    eps,
                    bc1,
                    bc2,
                };
                for (id, g) in grads.ent.iter(grads.dims) {
                    let row = tables.entity_mut(id)?;
                    adam_step(row, g, ent_m.row_mut(id, d), ent_v.row_mut(id, d), &h);
                }
                for (id, g) in grads.rel.iter(grads.dims) {
                    let row = tables.relation_mut(id)?;
                    adam_step(row, g, rel_m.row_mut(id, d), rel_v.row_mut(id, d), &h);
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

struct AdamHyper {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
}

fn adam_step(row: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], h: &AdamHyper) {
    for (((p, &gi), mi), vi) in row.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
        *mi = h.b1 * *mi + (1.0 - h.b1) * gi;
        *vi = h.b2 * *vi + (1.0 - h.b2) * gi * gi;
        let mhat = *mi / h.bc1;
        let vhat = *vi / h.bc2;
        *p -= h.lr * mhat / (vhat.sqrt() + h.eps);
    }
}
