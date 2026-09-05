//! Dynamic maintenance engine (issue #669, Phase 4).
//!
//! Supports every edit class (vertex/edge insertion and deletion,
//! restriction/observation/weight updates, orthogonal frame updates) with
//! lazy cell-level dirty tracking, cheap [`SyndromeEstimate`]s between
//! synchronizations, and an exact [`DynamicCohomology::flush`] whose result
//! is equivalent to fresh batch assembly within solver tolerance.

mod edits;

pub use edits::{CohomologyEdit, EditReceipt, SyndromeEstimate};

use crate::affine::{AffineSheafSystem, EdgeField, EdgeWeights};
use crate::block_sparse::CoboundaryBuilder;
use crate::operator::{LinearRestriction, RestrictionMap};
use crate::partition::CellPartition;
use crate::repair::{propose_repair, EdgeCost, RepairPlan, RepairPolicy};
use crate::syndrome::{compute_syndrome_warm, SyndromeConfig, SyndromeResult};
use crate::transport::{frame_change, transport_restriction};
use crate::witness::quantize;
use crate::{CohomologyError, EdgeKey, Endpoint, GaugePolicy, ValidationLimits, VertexId};
use std::collections::{BTreeMap, BTreeSet};

/// The dynamic cohomology interface (issue #669 core interface).
pub trait DynamicCohomology {
    /// Apply one edit, marking incident cells dirty.
    fn apply_edit(&mut self, edit: CohomologyEdit) -> EditReceipt;
    /// Cheap estimate from cached cell contributions.
    fn estimate(&self) -> SyndromeEstimate;
    /// Exact synchronization: equivalent to fresh batch assembly within
    /// solver tolerance.
    fn flush(&mut self) -> Result<SyndromeResult, CohomologyError>;
    /// Propose a group-sparse repair for the current contradiction.
    fn propose_repair(&mut self, policy: RepairPolicy) -> Result<RepairPlan, CohomologyError>;
}

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Boundary validation limits.
    pub limits: ValidationLimits,
    /// Syndrome/solver configuration.
    pub syndrome: SyndromeConfig,
    /// Bounded cell capacity (vertices per cell).
    pub cell_capacity: usize,
    /// Rebuild the partition after this many structural edits (bounds
    /// fragmentation drift under long edit streams).
    pub rebuild_interval: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            limits: ValidationLimits::default(),
            syndrome: SyndromeConfig::default(),
            cell_capacity: 64,
            rebuild_interval: 4096,
        }
    }
}

#[derive(Clone)]
struct EdgeState {
    rho_u: RestrictionMap,
    rho_v: RestrictionMap,
    weight: f64,
    observation: Vec<f64>,
    cost: EdgeCost,
    protected: bool,
}

struct FlushCache {
    topology_epoch: u64,
    observation_epoch: u64,
    x_star: Vec<f64>,
    energy: f64,
    /// Per-cell energy attribution (edge energy attributed to the tail cell).
    cell_energy: Vec<f64>,
}

/// Dynamic sheaf engine with bounded-cell dirty tracking.
pub struct DynamicSheafEngine {
    config: EngineConfig,
    vertices: BTreeMap<VertexId, usize>,
    edges: BTreeMap<EdgeKey, EdgeState>,
    partition: CellPartition,
    dirty: BTreeSet<usize>,
    topology_epoch: u64,
    observation_epoch: u64,
    structural_edits: u64,
    cache: Option<FlushCache>,
}

impl DynamicSheafEngine {
    /// New empty engine.
    pub fn new(config: EngineConfig) -> Self {
        let capacity = config.cell_capacity;
        Self {
            config,
            vertices: BTreeMap::new(),
            edges: BTreeMap::new(),
            partition: CellPartition::build(std::iter::empty(), capacity),
            dirty: BTreeSet::new(),
            topology_epoch: 0,
            observation_epoch: 0,
            structural_edits: 0,
            cache: None,
        }
    }

    /// Current topology epoch.
    pub fn topology_epoch(&self) -> u64 {
        self.topology_epoch
    }

    /// Current observation epoch.
    pub fn observation_epoch(&self) -> u64 {
        self.observation_epoch
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Set the repair cost of an edge (outside the edit stream: costs are
    /// policy metadata, not sheaf data).
    pub fn set_edge_cost(&mut self, key: &EdgeKey, cost: EdgeCost) -> Result<(), CohomologyError> {
        self.edges
            .get_mut(key)
            .map(|e| e.cost = cost)
            .ok_or(CohomologyError::UnknownEdge(key.u, key.v))
    }

    /// Mark an edge protected (immutable to repair without authorization).
    pub fn set_protected(&mut self, key: &EdgeKey, protected: bool) -> Result<(), CohomologyError> {
        self.edges
            .get_mut(key)
            .map(|e| e.protected = protected)
            .ok_or(CohomologyError::UnknownEdge(key.u, key.v))
    }

    /// Per-edge costs in canonical order (for the repair engine).
    pub fn edge_costs(&self) -> Vec<EdgeCost> {
        self.edges.values().map(|e| e.cost).collect()
    }

    /// Per-edge protected flags in canonical order.
    pub fn protected_mask(&self) -> Vec<bool> {
        self.edges.values().map(|e| e.protected).collect()
    }

    /// Assemble the current state into a batch [`AffineSheafSystem`]
    /// (deterministic: BTreeMap order is canonical order).
    pub fn build_system(&self) -> Result<AffineSheafSystem, CohomologyError> {
        let mut builder = CoboundaryBuilder::with_limits(self.config.limits.clone());
        for (&id, &dim) in &self.vertices {
            builder.add_vertex(id, dim)?;
        }
        for (key, state) in &self.edges {
            builder.add_edge(key.u, key.v, state.rho_u.clone(), state.rho_v.clone())?;
        }
        let op = builder.build();
        let mut observations = vec![0.0; op.dim_c1()];
        let mut weights = Vec::with_capacity(self.edges.len());
        for (key, state) in &self.edges {
            let (off, dim) = op.edge_slice(key).expect("edge indexed");
            observations[off..off + dim].copy_from_slice(&state.observation);
            weights.push(state.weight);
        }
        let mut system = AffineSheafSystem::new(
            op,
            EdgeField { data: observations },
            EdgeWeights { values: weights },
            GaugePolicy::MinimumNorm,
        )?;
        system.topology_epoch = self.topology_epoch;
        system.observation_epoch = self.observation_epoch;
        Ok(system)
    }

    fn mark_dirty_vertex(&mut self, v: VertexId, out: &mut Vec<usize>) {
        if let Some(c) = self.partition.cell_of(v) {
            if self.dirty.insert(c) || !out.contains(&c) {
                out.push(c);
            }
        }
    }

    fn try_apply(&mut self, edit: &CohomologyEdit, dirty: &mut Vec<usize>) -> Result<(), CohomologyError> {
        match edit {
            CohomologyEdit::InsertVertex { id, dim } => {
                if self.vertices.contains_key(id) {
                    return Err(CohomologyError::DuplicateVertex(*id));
                }
                if *dim == 0 || *dim > self.config.limits.max_dim {
                    return Err(CohomologyError::ValidationFailed(format!(
                        "vertex stalk dim {dim} outside (0, {}]",
                        self.config.limits.max_dim
                    )));
                }
                self.vertices.insert(*id, *dim);
                let cell = self.partition.insert_vertex(*id);
                self.dirty.insert(cell);
                dirty.push(cell);
                self.topology_epoch += 1;
                self.structural_edits += 1;
            }
            CohomologyEdit::RemoveVertex { id } => {
                if !self.vertices.contains_key(id) {
                    return Err(CohomologyError::UnknownVertex(*id));
                }
                let incident: Vec<EdgeKey> = self
                    .edges
                    .keys()
                    .filter(|k| k.u == *id || k.v == *id)
                    .copied()
                    .collect();
                for key in incident {
                    self.mark_dirty_vertex(key.u, dirty);
                    self.mark_dirty_vertex(key.v, dirty);
                    self.edges.remove(&key);
                }
                self.mark_dirty_vertex(*id, dirty);
                self.vertices.remove(id);
                self.partition.remove_vertex(*id);
                self.topology_epoch += 1;
                self.structural_edits += 1;
            }
            CohomologyEdit::InsertEdge { a, b, rho_a, rho_b, weight, observation } => {
                let key = EdgeKey::new(*a, *b)?;
                if self.edges.contains_key(&key) {
                    return Err(CohomologyError::DuplicateEdge(key.u, key.v));
                }
                let da = *self.vertices.get(a).ok_or(CohomologyError::UnknownVertex(*a))?;
                let db = *self.vertices.get(b).ok_or(CohomologyError::UnknownVertex(*b))?;
                rho_a.validate(&self.config.limits)?;
                rho_b.validate(&self.config.limits)?;
                if rho_a.input_dim() != da || rho_b.input_dim() != db
                    || rho_a.output_dim() != rho_b.output_dim()
                {
                    return Err(CohomologyError::DimensionMismatch {
                        expected: da,
                        actual: rho_a.input_dim(),
                        context: format!("edge ({a}, {b}) restriction dims"),
                    });
                }
                if observation.len() != rho_a.output_dim() {
                    return Err(CohomologyError::DimensionMismatch {
                        expected: rho_a.output_dim(),
                        actual: observation.len(),
                        context: format!("edge ({a}, {b}) observation"),
                    });
                }
                if !weight.is_finite() || *weight <= 0.0 || observation.iter().any(|v| !v.is_finite())
                {
                    return Err(CohomologyError::ValidationFailed(
                        "edge weight must be positive/finite and observation finite".into(),
                    ));
                }
                let (rho_u, rho_v) = if a < b {
                    (rho_a.clone(), rho_b.clone())
                } else {
                    (rho_b.clone(), rho_a.clone())
                };
                self.edges.insert(
                    key,
                    EdgeState {
                        rho_u,
                        rho_v,
                        weight: *weight,
                        observation: observation.clone(),
                        cost: EdgeCost::default(),
                        protected: false,
                    },
                );
                self.mark_dirty_vertex(key.u, dirty);
                self.mark_dirty_vertex(key.v, dirty);
                self.topology_epoch += 1;
                self.structural_edits += 1;
            }
            CohomologyEdit::RemoveEdge { a, b } => {
                let key = EdgeKey::new(*a, *b)?;
                if self.edges.remove(&key).is_none() {
                    return Err(CohomologyError::UnknownEdge(key.u, key.v));
                }
                self.mark_dirty_vertex(key.u, dirty);
                self.mark_dirty_vertex(key.v, dirty);
                self.topology_epoch += 1;
                self.structural_edits += 1;
            }
            CohomologyEdit::UpdateRestriction { a, b, endpoint, map } => {
                let key = EdgeKey::new(*a, *b)?;
                map.validate(&self.config.limits)?;
                let vertex_dim = |v: VertexId, s: &Self| {
                    s.vertices.get(&v).copied().ok_or(CohomologyError::UnknownVertex(v))
                };
                let (du, dv) = (vertex_dim(key.u, self)?, vertex_dim(key.v, self)?);
                let state = self
                    .edges
                    .get_mut(&key)
                    .ok_or(CohomologyError::UnknownEdge(key.u, key.v))?;
                let (expected_in, slot_dim) = match endpoint {
                    Endpoint::Tail => (du, state.rho_u.output_dim()),
                    Endpoint::Head => (dv, state.rho_v.output_dim()),
                };
                if map.input_dim() != expected_in || map.output_dim() != slot_dim {
                    return Err(CohomologyError::DimensionMismatch {
                        expected: expected_in,
                        actual: map.input_dim(),
                        context: format!("restriction update on ({}, {})", key.u, key.v),
                    });
                }
                match endpoint {
                    Endpoint::Tail => state.rho_u = map.clone(),
                    Endpoint::Head => state.rho_v = map.clone(),
                }
                self.mark_dirty_vertex(key.u, dirty);
                self.mark_dirty_vertex(key.v, dirty);
                self.topology_epoch += 1;
            }
            CohomologyEdit::UpdateObservation { a, b, value } => {
                let key = EdgeKey::new(*a, *b)?;
                let state = self
                    .edges
                    .get_mut(&key)
                    .ok_or(CohomologyError::UnknownEdge(key.u, key.v))?;
                if value.len() != state.observation.len() {
                    return Err(CohomologyError::DimensionMismatch {
                        expected: state.observation.len(),
                        actual: value.len(),
                        context: format!("observation update on ({}, {})", key.u, key.v),
                    });
                }
                if value.iter().any(|v| !v.is_finite()) {
                    return Err(CohomologyError::ValidationFailed(
                        "observation contains non-finite values".into(),
                    ));
                }
                state.observation = value.clone();
                self.mark_dirty_vertex(key.u, dirty);
                self.mark_dirty_vertex(key.v, dirty);
                self.observation_epoch += 1;
            }
            CohomologyEdit::UpdateWeight { a, b, weight } => {
                let key = EdgeKey::new(*a, *b)?;
                if !weight.is_finite() || *weight <= 0.0 {
                    return Err(CohomologyError::ValidationFailed(format!(
                        "weight {weight} must be finite and positive"
                    )));
                }
                let state = self
                    .edges
                    .get_mut(&key)
                    .ok_or(CohomologyError::UnknownEdge(key.u, key.v))?;
                state.weight = *weight;
                self.mark_dirty_vertex(key.u, dirty);
                self.mark_dirty_vertex(key.v, dirty);
                self.observation_epoch += 1;
            }
            CohomologyEdit::UpdateFrame { vertex, q_old, q_new } => {
                let dim = *self
                    .vertices
                    .get(vertex)
                    .ok_or(CohomologyError::UnknownVertex(*vertex))?;
                if q_old.rows != dim {
                    return Err(CohomologyError::DimensionMismatch {
                        expected: dim,
                        actual: q_old.rows,
                        context: format!("frame update on vertex {vertex}"),
                    });
                }
                let r = frame_change(q_old, q_new, &self.config.limits)?;
                let incident: Vec<EdgeKey> = self
                    .edges
                    .keys()
                    .filter(|k| k.u == *vertex || k.v == *vertex)
                    .copied()
                    .collect();
                // Validate all transports before mutating any (atomicity).
                let mut updates = Vec::with_capacity(incident.len());
                for key in &incident {
                    let state = self.edges.get(key).expect("incident edge");
                    let new_map = if key.u == *vertex {
                        transport_restriction(&state.rho_u, &r)?
                    } else {
                        transport_restriction(&state.rho_v, &r)?
                    };
                    updates.push((*key, new_map));
                }
                for (key, new_map) in updates {
                    let state = self.edges.get_mut(&key).expect("incident edge");
                    if key.u == *vertex {
                        state.rho_u = new_map;
                    } else {
                        state.rho_v = new_map;
                    }
                    self.mark_dirty_vertex(key.u, dirty);
                    self.mark_dirty_vertex(key.v, dirty);
                }
                self.mark_dirty_vertex(*vertex, dirty);
                self.topology_epoch += 1;
            }
        }
        Ok(())
    }
}

impl DynamicCohomology for DynamicSheafEngine {
    fn apply_edit(&mut self, edit: CohomologyEdit) -> EditReceipt {
        let edit_hash = edits::edit_hash(&edit);
        let mut dirty = Vec::new();
        let result = self.try_apply(&edit, &mut dirty);
        dirty.sort_unstable();
        dirty.dedup();
        EditReceipt {
            accepted: result.is_ok(),
            topology_epoch: self.topology_epoch,
            observation_epoch: self.observation_epoch,
            dirty_cells: dirty,
            edit_hash,
            error: result.err().map(|e| e.to_string()),
        }
    }

    fn estimate(&self) -> SyndromeEstimate {
        let (clean_energy, is_exact) = match &self.cache {
            Some(cache) => {
                let fresh = cache.topology_epoch == self.topology_epoch
                    && cache.observation_epoch == self.observation_epoch
                    && self.dirty.is_empty();
                if fresh {
                    (cache.energy, true)
                } else {
                    let clean: f64 = cache
                        .cell_energy
                        .iter()
                        .enumerate()
                        .filter(|(c, _)| !self.dirty.contains(c))
                        .map(|(_, e)| e)
                        .sum();
                    (clean, false)
                }
            }
            None => (0.0, self.edges.is_empty() && self.dirty.is_empty()),
        };
        SyndromeEstimate {
            clean_energy,
            dirty_cell_count: self.dirty.len(),
            is_exact,
            topology_epoch: self.topology_epoch,
            observation_epoch: self.observation_epoch,
        }
    }

    fn flush(&mut self) -> Result<SyndromeResult, CohomologyError> {
        // Bound partition fragmentation drift under long edit streams.
        if self.structural_edits >= self.config.rebuild_interval {
            self.partition = CellPartition::build(
                self.vertices.keys().copied(),
                self.config.cell_capacity,
            );
            self.structural_edits = 0;
            self.cache = None; // cell attribution layout changed
        }
        let system = self.build_system()?;
        let warm = self
            .cache
            .as_ref()
            .filter(|c| c.topology_epoch == self.topology_epoch)
            .map(|c| c.x_star.as_slice());
        let result = compute_syndrome_warm(&system, &self.config.syndrome, warm)?;

        // Attribute per-edge syndrome energy to the tail vertex's cell.
        let mut cell_energy = vec![0.0; self.partition.cell_count()];
        for contribution in &result.ranked_support {
            if let Some(cell) = self.partition.cell_of(contribution.edge.u) {
                cell_energy[cell] += contribution.energy;
            }
        }
        self.cache = Some(FlushCache {
            topology_epoch: self.topology_epoch,
            observation_epoch: self.observation_epoch,
            x_star: result.x_star.clone(),
            energy: result.energy,
            cell_energy,
        });
        self.dirty.clear();
        // Numerical drift guard: quantized cached energy must round-trip.
        let scale = self.config.syndrome.quantization_scale;
        debug_assert_eq!(
            quantize(result.energy, scale),
            quantize(self.cache.as_ref().expect("cache just set").energy, scale)
        );
        Ok(result)
    }

    fn propose_repair(&mut self, policy: RepairPolicy) -> Result<RepairPlan, CohomologyError> {
        let syndrome = self.flush()?;
        let system = self.build_system()?;
        let costs = self.edge_costs();
        let protected = self.protected_mask();
        propose_repair(
            &system,
            &syndrome,
            Some(&costs),
            Some(&protected),
            &policy,
            &self.config.syndrome,
        )
    }
}
