//! Loop 3 — continual update on graph growth with Elastic Weight Consolidation
//! (ADR-004). New triples are a delta: grow the tables for new entities
//! (initialised from the mean of their neighbours' embeddings, ADR-004 loop 3),
//! then fine-tune with an EWC penalty
//! `L_EWC = λ/2 · Σ Fᵢ(θᵢ − θ*ᵢ)²` over the Fisher diagonal estimated from the
//! previous training gradients. A permanent control arm (typesafe
//! `ControlArm`) compares the pre- and post-update models on a frozen slice.
//!
//! **Stated v1 limitation (ADR-004, arXiv:2604.19401, 2026):** standard EWC only
//! penalises drift in *existing* entity embeddings. New entities from graph
//! growth can still interfere with old ones by competing for score mass — a
//! forgetting mode the Fisher penalty does not cover. v1 documents it; v2
//! revisits with incremental distillation (arXiv:2405.04453) or incremental
//! LoRA (arXiv:2407.05705) if EWC-only proves insufficient in our benchmarks.

use crate::{EntityId, Tables, Triple};
use ruvector_typesafe_core::loop_gate::ControlArm;

/// EWC state: the Fisher diagonal and the reference optimum `θ*`, kept for the
/// entity and relation tables separately. Both are grown (zero-padded) when new
/// entities arrive, so a new row has `F = 0` and is fit freely by the data
/// gradient.
pub struct ContinualUpdate {
    pub lambda: f32,
    dims: usize,
    fisher_ent: Vec<f32>,
    star_ent: Vec<f32>,
    fisher_rel: Vec<f32>,
    star_rel: Vec<f32>,
}

impl ContinualUpdate {
    /// Estimate the Fisher diagonal as the mean of squared gradients over the
    /// previous training steps, and snapshot the current parameters as `θ*`.
    /// Each gradient sample must match the table layout (`entities_raw` /
    /// `relations_raw` lengths).
    #[must_use]
    pub fn from_gradients(
        tables: &Tables,
        ent_grads: &[Vec<f32>],
        rel_grads: &[Vec<f32>],
        lambda: f32,
    ) -> Self {
        let fisher_ent = mean_of_squares(ent_grads, tables.entities_raw().len());
        let fisher_rel = mean_of_squares(rel_grads, tables.relations_raw().len());
        Self {
            lambda,
            dims: tables.dims(),
            fisher_ent,
            star_ent: tables.entities_raw().to_vec(),
            fisher_rel,
            star_rel: tables.relations_raw().to_vec(),
        }
    }

    /// Grow `old` by `neighbours.len()` new entities. Each new row is the mean
    /// of its neighbours' embeddings (ADR-004 loop 3); an empty neighbour list
    /// leaves the deterministic random init rather than a NaN mean. Fisher and
    /// `θ*` are zero-padded for the new rows.
    #[must_use]
    pub fn grow(&mut self, old: &Tables, neighbours: &[Vec<EntityId>], seed: u64) -> Tables {
        let d = old.dims();
        let n_old = old.num_entities();
        let n_new = neighbours.len();
        let mut grown = Tables::new(n_old + n_new, old.num_relations(), d, seed);
        grown.entities_raw_mut()[..n_old * d].copy_from_slice(old.entities_raw());
        grown
            .relations_raw_mut()
            .copy_from_slice(old.relations_raw());

        for (k, nb) in neighbours.iter().enumerate() {
            if nb.is_empty() {
                continue; // keep the random init
            }
            let mut mean = vec![0.0f32; d];
            let mut count = 0.0f32;
            for &e in nb {
                if let Ok(row) = old.entity(e) {
                    for (m, v) in mean.iter_mut().zip(row) {
                        *m += *v;
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for m in &mut mean {
                    *m /= count;
                }
                let new_id = (n_old + k) as EntityId;
                grown
                    .entity_mut(new_id)
                    .expect("new row in bounds")
                    .copy_from_slice(&mean);
            }
        }

        self.fisher_ent.resize((n_old + n_new) * d, 0.0);
        self.star_ent.resize((n_old + n_new) * d, 0.0);
        grown
    }

    /// One-step-per-iteration gradient descent on the delta with the EWC
    /// penalty folded in: `θ ← θ − lr·(g_data + λ·F·(θ − θ*))`. `grad` returns
    /// the data-loss gradient over the delta, in `(entities_raw, relations_raw)`
    /// layout, for the current parameters.
    ///
    /// Stability: `lr·λ·max(F) < 2`, or the penalty term oscillates.
    pub fn fit_delta<F>(&self, tables: &mut Tables, lr: f32, steps: usize, grad: F)
    where
        F: Fn(&Tables) -> (Vec<f32>, Vec<f32>),
    {
        for _ in 0..steps {
            let (dg_ent, dg_rel) = grad(tables);
            step_table(
                tables.entities_raw_mut(),
                &dg_ent,
                &self.fisher_ent,
                &self.star_ent,
                self.lambda,
                lr,
            );
            step_table(
                tables.relations_raw_mut(),
                &dg_rel,
                &self.fisher_rel,
                &self.star_rel,
                self.lambda,
                lr,
            );
        }
    }

    /// Analytic drift bound for an old-entity parameter under a constant
    /// data gradient of magnitude `grad_mag`: the EWC fixed point is
    /// `|θ − θ*| = |g| / (λ·F)`. Returns `f32::INFINITY` where `F = 0`.
    #[must_use]
    pub fn entity_drift_bound(&self, param_index: usize, grad_mag: f32) -> f32 {
        let f = self.fisher_ent.get(param_index).copied().unwrap_or(0.0);
        if f <= 0.0 || self.lambda <= 0.0 {
            f32::INFINITY
        } else {
            grad_mag.abs() / (self.lambda * f)
        }
    }

    /// Feed a frozen slice through the permanent control arm: the old model is
    /// the control, the updated model the champion. `correct` returns
    /// `(old_correct, updated_correct)` per triple. The alarm fires when the
    /// updated model trails the old by more than `threshold` (ADR-004 gate 4).
    #[must_use]
    pub fn control_comparison<F>(
        &self,
        frozen: &[Triple],
        threshold: f32,
        min_samples: u32,
        mut correct: F,
    ) -> ControlArm
    where
        F: FnMut(&Triple) -> (bool, bool),
    {
        let mut arm = ControlArm::new(threshold, min_samples);
        for t in frozen {
            let (old_ok, new_ok) = correct(t);
            arm.observe(old_ok, new_ok);
        }
        arm
    }

    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }
}

fn mean_of_squares(grads: &[Vec<f32>], len: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; len];
    if grads.is_empty() {
        return acc;
    }
    for g in grads {
        for (a, v) in acc.iter_mut().zip(g) {
            *a += v * v;
        }
    }
    let n = grads.len() as f32;
    for a in &mut acc {
        *a /= n;
    }
    acc
}

fn step_table(theta: &mut [f32], grad: &[f32], fisher: &[f32], star: &[f32], lambda: f32, lr: f32) {
    for (i, t) in theta.iter_mut().enumerate() {
        let g = grad.get(i).copied().unwrap_or(0.0);
        let penalty = match (fisher.get(i), star.get(i)) {
            (Some(&f), Some(&s)) => lambda * f * (*t - s),
            _ => 0.0,
        };
        *t -= lr * (g + penalty);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewc_bounds_old_entities_while_new_ones_fit() {
        // 4 old entities, 1 relation, dims 2.
        let old = Tables::new(4, 1, 2, 7);
        // Strong Fisher on every param: gradient magnitude 10 → F = 100.
        let ent_grad = vec![10.0f32; old.entities_raw().len()];
        let rel_grad = vec![10.0f32; old.relations_raw().len()];
        let mut cu = ContinualUpdate::from_gradients(&old, &[ent_grad], &[rel_grad], 1.0);
        // F = 100 for old rows.
        assert!((cu.entity_drift_bound(0, 0.5) - 0.005).abs() < 1e-4);

        // Grow one new entity from the mean of entities 0 and 1.
        let mut grown = cu.grow(&old, &[vec![0, 1]], 9);
        let star0 = grown.entity(0).unwrap()[0];
        let new_id = 4u32;
        // New row initialised to the neighbour mean (F = 0 for it).
        assert!((cu.entity_drift_bound((new_id as usize) * 2, 0.5)).is_infinite());

        // Constant adversarial gradient +0.5 on old-entity-0 elem0 AND on the
        // whole new row (to fit it). lr·λ·F = 0.01·1·100 = 1 < 2 (stable).
        let lr = 0.01;
        cu.fit_delta(&mut grown, lr, 400, |t| {
            let mut g = vec![0.0f32; t.entities_raw().len()];
            g[0] = 0.5; // old entity 0, element 0
            g[new_id as usize * 2] = 0.5; // new entity, element 0
            g[new_id as usize * 2 + 1] = 0.5;
            (g, vec![0.0f32; t.relations_raw().len()])
        });

        let old_drift = (grown.entity(0).unwrap()[0] - star0).abs();
        let new_shift = grown.entity(new_id).unwrap()[0].abs();
        // Old param pinned near θ* within the analytic bound (+ epsilon).
        assert!(old_drift <= 0.005 + 1e-3, "old drifted {old_drift}");
        // New param moved far more than the old bound.
        assert!(new_shift > 0.3, "new shift {new_shift}");
    }

    #[test]
    fn control_arm_alarms_on_injected_regression() {
        let frozen: Vec<Triple> = (0..10).map(|i| Triple::new(i, 0, i + 1)).collect();
        let old = Tables::new(1, 1, 2, 1);
        let cu = ContinualUpdate::from_gradients(&old, &[], &[], 1.0);

        // Updated model wrong on every frozen triple the old model got right.
        let regressed = cu.control_comparison(&frozen, 0.1, 5, |_t| (true, false));
        assert!(regressed.alarm());
        assert!(regressed.drift() < 0.0);

        // Healthy update: no alarm.
        let healthy = cu.control_comparison(&frozen, 0.1, 5, |_t| (true, true));
        assert!(!healthy.alarm());
    }

    #[test]
    fn grow_with_empty_neighbours_keeps_random_init() {
        let old = Tables::new(2, 1, 3, 4);
        let mut cu = ContinualUpdate::from_gradients(&old, &[], &[], 1.0);
        let grown = cu.grow(&old, &[vec![]], 4);
        // Row exists and is finite (not NaN from a 0/0 mean).
        assert_eq!(grown.num_entities(), 3);
        assert!(grown.entity(2).unwrap().iter().all(|x| x.is_finite()));
    }
}
