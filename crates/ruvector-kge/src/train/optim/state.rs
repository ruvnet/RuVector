//! Checkpoint export/import of optimizer state (plan M3). A child module so it
//! can reach the private row storage without widening its visibility.

use super::{Optimizer, Rows, State};
use crate::Result;

/// One table's worth of optimizer rows, exported for a checkpoint: the
/// touched row ids in ascending order and their values concatenated
/// (`ids.len() × dims`). The same form for both layouts.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RowsState {
    pub ids: Vec<u32>,
    pub data: Vec<f32>,
}

/// Exported optimizer state (plan M3 checkpoints). `rows` is `[ent, rel]`
/// for Adagrad and `[ent_m, ent_v, rel_m, rel_v]` for Adam; `t` is Adam's
/// step counter (0 for Adagrad).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OptimState {
    pub t: u64,
    pub rows: Vec<RowsState>,
}

impl Rows {
    fn export(&self, dims: usize) -> RowsState {
        let mut out = RowsState::default();
        for (id, v) in self.iter(dims) {
            out.ids.push(id);
            out.data.extend_from_slice(v);
        }
        out
    }

    /// Restore exported rows into this (freshly built, empty) table.
    fn import(&mut self, st: &RowsState, dims: usize, rows: usize) -> Result<()> {
        if st.data.len() != st.ids.len() * dims {
            return Err(crate::KgeError::Invalid(
                "optimizer state: data is not ids x dims".into(),
            ));
        }
        let mut prev: Option<u32> = None;
        for (k, &id) in st.ids.iter().enumerate() {
            if (id as usize) >= rows || prev.is_some_and(|p| p >= id) {
                return Err(crate::KgeError::Invalid(
                    "optimizer state: row ids out of range or not ascending".into(),
                ));
            }
            prev = Some(id);
            self.row_mut(id, dims)
                .copy_from_slice(&st.data[k * dims..(k + 1) * dims]);
        }
        Ok(())
    }
}

impl Optimizer {
    fn tables_mut(&mut self) -> Vec<&mut Rows> {
        match &mut self.state {
            State::Adagrad { ent, rel } => vec![ent, rel],
            State::Adam {
                ent_m,
                ent_v,
                rel_m,
                rel_v,
                ..
            } => vec![ent_m, ent_v, rel_m, rel_v],
        }
    }

    /// Snapshot of the per-row state (see [`OptimState`]).
    pub(crate) fn export_state(&self) -> OptimState {
        let d = self.dims;
        match &self.state {
            State::Adagrad { ent, rel } => OptimState {
                t: 0,
                rows: vec![ent.export(d), rel.export(d)],
            },
            State::Adam {
                ent_m,
                ent_v,
                rel_m,
                rel_v,
                t,
            } => OptimState {
                t: *t,
                rows: [ent_m, ent_v, rel_m, rel_v]
                    .iter()
                    .map(|r| r.export(d))
                    .collect(),
            },
        }
    }

    /// Load `st` into a freshly constructed optimizer of the same kind and
    /// shape. Row bounds are checked against the tables it was sized for.
    pub(crate) fn import_state(&mut self, st: &OptimState) -> Result<()> {
        let (d, ne, nr) = (self.dims, self.entities, self.relations);
        if let State::Adam { t, .. } = &mut self.state {
            *t = st.t;
        } else if st.t != 0 {
            return Err(crate::KgeError::Invalid(
                "optimizer state: step counter on a non-Adam optimizer".into(),
            ));
        }
        let mut tables = self.tables_mut();
        if tables.len() != st.rows.len() {
            return Err(crate::KgeError::Invalid(
                "optimizer state: kind mismatch (row-table count)".into(),
            ));
        }
        let half = tables.len() / 2;
        for (i, (rows, s)) in tables.iter_mut().zip(&st.rows).enumerate() {
            rows.import(s, d, if i < half { ne } else { nr })?;
        }
        Ok(())
    }
}
