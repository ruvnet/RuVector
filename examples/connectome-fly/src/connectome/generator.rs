//! Deterministic stochastic-block-model generator for a synthetic
//! fly-like connectome. See
//! `docs/research/connectome-ruvector/02-connectome-layer.md` for the
//! target statistics this implementation calibrates against.

use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::LogNormal;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::persist::ConnectomeError;
use super::schema::{
    ConnectomeConfig, ConnectomeSerCfg, FlyWireNeuronId, NeuronClass, NeuronId, NeuronMeta, Sign,
    Synapse,
};

/// A synthetic fly-like connectome. Stores neuron metadata and a
/// flattened CSR outgoing adjacency (`row_ptr`, `synapses`) for
/// cache-friendly LIF dispatch, plus per-class indices used by the
/// stimulus and motif encoders.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Connectome {
    pub(super) cfg: ConnectomeSerCfg,
    pub(super) meta: Vec<NeuronMeta>,
    /// Flattened outgoing synapses.
    pub(super) synapses: Vec<Synapse>,
    /// CSR row pointer: `synapses[row_ptr[i]..row_ptr[i+1]]` are the
    /// outgoing synapses of neuron `i`.
    pub(super) row_ptr: Vec<u32>,
    /// Pre-computed index of sensory neuron ids.
    pub(super) sensory: Vec<NeuronId>,
    /// Pre-computed index of motor neuron ids.
    pub(super) motor: Vec<NeuronId>,
    /// Pre-computed index grouped by class.
    pub(super) by_class: Vec<Vec<NeuronId>>,
    /// Stable FlyWire root ids, parallel to `meta` / dense ids.
    /// `None` for SBM-generated connectomes; `Some` when loaded via the
    /// `flywire` module. Serialized at the tail of the bincode blob so
    /// existing synthetic blobs remain round-trippable.
    #[serde(default)]
    pub(super) flywire_ids: Option<Vec<FlyWireNeuronId>>,
}

impl Connectome {
    /// Generate a deterministic synthetic connectome from `cfg`.
    pub fn generate(cfg: &ConnectomeConfig) -> Self {
        let n = cfg.num_neurons as usize;
        assert!(n > 0, "num_neurons must be > 0");
        assert!(cfg.num_modules > 0, "num_modules must be > 0");
        let mut rng = Xoshiro256StarStar::seed_from_u64(cfg.seed);
        let (class_table, module_of) = build_class_assignment(cfg, &mut rng);
        let mut meta: Vec<NeuronMeta> = (0..n)
            .map(|i| NeuronMeta {
                class: class_table[i],
                module: module_of[i],
                bias_pa: rng.gen_range(-1.2..1.2),
            })
            .collect();
        let hub_set = compute_hubs(cfg);
        let target_edges = (cfg.avg_out_degree * n as f32) as usize;
        let mut buckets: Vec<SmallVec<[Synapse; 32]>> = vec![SmallVec::new(); n];
        let weight_dist = LogNormal::new(cfg.weight_log_mu as f64, cfg.weight_log_sigma as f64)
            .expect("valid lognormal");

        let mut drawn: usize = 0;
        let mut attempts: usize = 0;
        // Generous cap: at low between-module probabilities we spend
        // many rejected draws before admitting one. 64× target plus a
        // scale-invariant floor keeps 10k-neuron builds feasible.
        let max_attempts = target_edges.saturating_mul(64).saturating_add(1 << 15);
        while drawn < target_edges && attempts < max_attempts {
            attempts += 1;
            let pre = rng.gen_range(0..n);
            let post = rng.gen_range(0..n);
            if pre == post {
                continue;
            }
            let p = connection_probability(cfg, module_of[pre], module_of[post], &hub_set);
            if rng.gen::<f32>() >= p {
                continue;
            }
            if buckets[pre].iter().any(|s| s.post.0 as usize == post) {
                continue;
            }
            let w_raw: f64 = weight_dist.sample(&mut rng);
            let mut weight = (w_raw as f32).clamp(1e-4, 20.0);
            let sign = if rng.gen::<f32>() < meta[pre].class.inhibitory_rate() {
                Sign::Inhibitory
            } else {
                Sign::Excitatory
            };
            if matches!(sign, Sign::Inhibitory) {
                weight *= 1.3;
            }
            let dmod = module_distance(module_of[pre], module_of[post], cfg.num_modules);
            let delay_ms = (cfg.delay_min_ms + 0.15 * dmod as f32 + rng.gen_range(0.0..1.5))
                .clamp(cfg.delay_min_ms, cfg.delay_max_ms);
            buckets[pre].push(Synapse {
                post: NeuronId(post as u32),
                weight,
                delay_ms,
                sign,
            });
            drawn += 1;
        }

        let mut row_ptr: Vec<u32> = Vec::with_capacity(n + 1);
        let mut synapses: Vec<Synapse> = Vec::with_capacity(drawn);
        row_ptr.push(0);
        for b in &buckets {
            synapses.extend(b.iter().copied());
            row_ptr.push(synapses.len() as u32);
        }
        for m in meta.iter_mut() {
            if m.class.is_sensory() {
                m.bias_pa = -0.5;
            } else if m.class.is_motor() {
                m.bias_pa = 0.5;
            }
        }
        let mut by_class: Vec<Vec<NeuronId>> = vec![Vec::new(); 15];
        let mut sensory: Vec<NeuronId> = Vec::new();
        let mut motor: Vec<NeuronId> = Vec::new();
        for (i, m) in meta.iter().enumerate() {
            by_class[m.class as usize].push(NeuronId(i as u32));
            if m.class.is_sensory() {
                sensory.push(NeuronId(i as u32));
            }
            if m.class.is_motor() {
                motor.push(NeuronId(i as u32));
            }
        }
        Self {
            cfg: ConnectomeSerCfg::from(cfg),
            meta,
            synapses,
            row_ptr,
            sensory,
            motor,
            by_class,
            flywire_ids: None,
        }
    }

    /// Construct a `Connectome` directly from already-assembled parts.
    ///
    /// Used by the `flywire` loader to install parsed FlyWire v783
    /// records without going through the synthetic SBM path. Callers
    /// are responsible for supplying a CSR-consistent `(row_ptr,
    /// synapses)` pair: `row_ptr.len() == meta.len() + 1` and
    /// `row_ptr[i] <= row_ptr[i+1] <= synapses.len()`.
    ///
    /// Sensory / motor / by-class indices are derived from `meta`.
    /// `flywire_ids`, if provided, must be parallel to `meta`.
    pub(super) fn from_parts(
        cfg: ConnectomeSerCfg,
        meta: Vec<NeuronMeta>,
        synapses: Vec<Synapse>,
        row_ptr: Vec<u32>,
        flywire_ids: Option<Vec<FlyWireNeuronId>>,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), meta.len() + 1);
        debug_assert_eq!(*row_ptr.last().unwrap_or(&0) as usize, synapses.len());
        if let Some(ids) = &flywire_ids {
            debug_assert_eq!(ids.len(), meta.len());
        }
        let mut by_class: Vec<Vec<NeuronId>> = vec![Vec::new(); 15];
        let mut sensory: Vec<NeuronId> = Vec::new();
        let mut motor: Vec<NeuronId> = Vec::new();
        for (i, m) in meta.iter().enumerate() {
            by_class[m.class as usize].push(NeuronId(i as u32));
            if m.class.is_sensory() {
                sensory.push(NeuronId(i as u32));
            }
            if m.class.is_motor() {
                motor.push(NeuronId(i as u32));
            }
        }
        Self {
            cfg,
            meta,
            synapses,
            row_ptr,
            sensory,
            motor,
            by_class,
            flywire_ids,
        }
    }

    /// Parallel array of stable FlyWire root ids, if this connectome
    /// was loaded from a FlyWire v783 release. `None` for SBM-generated
    /// connectomes.
    #[inline]
    pub fn flywire_ids(&self) -> Option<&[FlyWireNeuronId]> {
        self.flywire_ids.as_deref()
    }

    /// Total number of neurons.
    #[inline]
    pub fn num_neurons(&self) -> usize {
        self.meta.len()
    }

    /// Total number of outgoing synapses (each directed edge counted once).
    #[inline]
    pub fn num_synapses(&self) -> usize {
        self.synapses.len()
    }

    /// Meta for neuron `id`.
    #[inline]
    pub fn meta(&self, id: NeuronId) -> &NeuronMeta {
        &self.meta[id.idx()]
    }

    /// All neuron metadata as a slice.
    #[inline]
    pub fn all_meta(&self) -> &[NeuronMeta] {
        &self.meta
    }

    /// Outgoing synapses of neuron `id`.
    #[inline]
    pub fn outgoing(&self, id: NeuronId) -> &[Synapse] {
        let s = self.row_ptr[id.idx()] as usize;
        let e = self.row_ptr[id.idx() + 1] as usize;
        &self.synapses[s..e]
    }

    /// Flat synapse array (used by LIF and benches).
    #[inline]
    pub fn synapses(&self) -> &[Synapse] {
        &self.synapses
    }

    /// CSR row pointer.
    #[inline]
    pub fn row_ptr(&self) -> &[u32] {
        &self.row_ptr
    }

    /// Pre-computed sensory-neuron index.
    #[inline]
    pub fn sensory_neurons(&self) -> &[NeuronId] {
        &self.sensory
    }

    /// Pre-computed motor-neuron index.
    #[inline]
    pub fn motor_neurons(&self) -> &[NeuronId] {
        &self.motor
    }

    /// Neurons grouped by class.
    #[inline]
    pub fn by_class(&self) -> &[Vec<NeuronId>] {
        &self.by_class
    }

    /// Number of modules the connectome was generated with.
    #[inline]
    pub fn num_modules(&self) -> u16 {
        self.cfg.num_modules
    }

    /// Seed this connectome was generated with.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.cfg.seed
    }

    /// Serialize to a compact binary blob (bincode).
    pub fn to_bytes(&self) -> Result<Vec<u8>, ConnectomeError> {
        bincode::serialize(self).map_err(ConnectomeError::from)
    }

    /// Deserialize from a bincode blob.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ConnectomeError> {
        bincode::deserialize(bytes).map_err(ConnectomeError::from)
    }

    /// Return a copy of the connectome with the specified flat synapse
    /// indices zeroed-out. Used by the counterfactual perturbation
    /// harness (ADR-154 §3.4 AC-5).
    pub fn with_synapse_weights_zeroed(&self, flat_ids: &[usize]) -> Self {
        let mut out = self.clone();
        for &i in flat_ids {
            if i < out.synapses.len() {
                out.synapses[i].weight = 0.0;
            }
        }
        out
    }
}

// -------------------------------------------------------------------
// Internal helpers
// -------------------------------------------------------------------

fn build_class_assignment<R: Rng>(
    cfg: &ConnectomeConfig,
    rng: &mut R,
) -> (Vec<NeuronClass>, Vec<u16>) {
    let n = cfg.num_neurons as usize;
    let m = cfg.num_modules as usize;
    let mut class_table: Vec<NeuronClass> = Vec::with_capacity(n);
    let mut module_of: Vec<u16> = Vec::with_capacity(n);

    // Biased class distribution roughly matching the research §02
    // table weightings.
    let base_weights: [(NeuronClass, f32); 15] = [
        (NeuronClass::PhotoReceptor, 0.03),
        (NeuronClass::Chemosensory, 0.02),
        (NeuronClass::Mechanosensory, 0.02),
        (NeuronClass::OpticLocal, 0.18),
        (NeuronClass::KenyonCell, 0.14),
        (NeuronClass::MbOutput, 0.015),
        (NeuronClass::CentralComplex, 0.06),
        (NeuronClass::LateralAccessory, 0.05),
        (NeuronClass::Descending, 0.015),
        (NeuronClass::Ascending, 0.02),
        (NeuronClass::Motor, 0.03),
        (NeuronClass::LocalInter, 0.09),
        (NeuronClass::Projection, 0.10),
        (NeuronClass::Modulatory, 0.02),
        (NeuronClass::Other, 0.24),
    ];
    let total: f32 = base_weights.iter().map(|(_, w)| *w).sum();
    for i in 0..n {
        let r: f32 = rng.gen::<f32>() * total;
        let mut acc = 0.0;
        let mut chosen = NeuronClass::Other;
        for (c, w) in &base_weights {
            acc += *w;
            if r <= acc {
                chosen = *c;
                break;
            }
        }
        class_table.push(chosen);
        let bias = match chosen {
            NeuronClass::PhotoReceptor => 0,
            NeuronClass::Chemosensory => 1,
            NeuronClass::Mechanosensory => 2,
            NeuronClass::KenyonCell | NeuronClass::MbOutput => 3,
            NeuronClass::Motor | NeuronClass::Descending => 4,
            NeuronClass::CentralComplex => 5,
            _ => 6 + (i % (m.saturating_sub(6).max(1))),
        };
        module_of.push((bias % m) as u16);
    }
    (class_table, module_of)
}

fn compute_hubs(cfg: &ConnectomeConfig) -> Vec<u16> {
    (0..cfg.num_hub_modules).collect()
}

fn connection_probability(cfg: &ConnectomeConfig, m_pre: u16, m_post: u16, hubs: &[u16]) -> f32 {
    if m_pre == m_post {
        cfg.p_within
    } else if hubs.contains(&m_pre) && hubs.contains(&m_post) {
        cfg.p_between + cfg.p_hub_boost
    } else {
        cfg.p_between
    }
}

#[inline]
fn module_distance(a: u16, b: u16, n: u16) -> u16 {
    let d = a.abs_diff(b);
    d.min(n - d)
}
