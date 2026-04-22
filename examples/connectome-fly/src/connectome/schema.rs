//! Public types for the connectome layer.
//!
//! Neuron / synapse / class / sign enums plus the `ConnectomeConfig`
//! struct that parameterizes the SBM generator in
//! `super::generator`. Derivations favour `Serialize` / `Deserialize`
//! so the full connectome round-trips through a bincode blob without
//! the generator being involved.

use serde::{Deserialize, Serialize};

/// Global id of a neuron in the connectome. Dense `0 .. num_neurons`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NeuronId(pub u32);

impl NeuronId {
    /// Raw index.
    #[inline]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
}

/// Stable FlyWire v783 root id (64-bit). Carried alongside the dense
/// `NeuronId` when a `Connectome` is loaded from FlyWire so analyses can
/// round-trip back to the published identifier space. Opaque newtype;
/// see `docs/research/connectome-ruvector/02-connectome-layer.md` §3.1.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FlyWireNeuronId(pub u64);

impl FlyWireNeuronId {
    /// Raw id.
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Synapse sign. `+1` excitatory, `-1` inhibitory. Neuromodulatory
/// edges are *not* represented in the fast path
/// (`docs/research/connectome-ruvector/03-neural-dynamics.md` §2.2).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i8)]
pub enum Sign {
    /// Excitatory.
    Excitatory = 1,
    /// Inhibitory.
    Inhibitory = -1,
}

/// Coarse neuron class. Matches the ~15 broad functional categories
/// used in the research (§02 and §05 of the research docs).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum NeuronClass {
    /// Photoreceptor.
    PhotoReceptor = 0,
    /// Chemosensory / olfactory.
    Chemosensory = 1,
    /// Mechanosensory.
    Mechanosensory = 2,
    /// Optic-lobe local.
    OpticLocal = 3,
    /// Mushroom-body Kenyon.
    KenyonCell = 4,
    /// Mushroom-body output.
    MbOutput = 5,
    /// Central complex.
    CentralComplex = 6,
    /// Lateral accessory lobe.
    LateralAccessory = 7,
    /// Descending / command.
    Descending = 8,
    /// Ascending.
    Ascending = 9,
    /// Motor.
    Motor = 10,
    /// Local interneuron (GABA-dominated).
    LocalInter = 11,
    /// Projection neuron.
    Projection = 12,
    /// Neuromodulatory cell (present but not on fast path; rendered
    /// as slow excitatory here because the ADR defers slow pools).
    Modulatory = 13,
    /// Catch-all.
    Other = 14,
}

impl NeuronClass {
    /// Base inhibitory probability for this class.
    #[inline]
    pub fn inhibitory_rate(self) -> f32 {
        match self {
            NeuronClass::LocalInter => 0.95,
            NeuronClass::OpticLocal => 0.30,
            NeuronClass::MbOutput => 0.10,
            NeuronClass::Descending => 0.05,
            _ => 0.05,
        }
    }

    /// Whether this class participates in sensory input in the demo.
    #[inline]
    pub fn is_sensory(self) -> bool {
        matches!(
            self,
            NeuronClass::PhotoReceptor | NeuronClass::Chemosensory | NeuronClass::Mechanosensory
        )
    }

    /// Whether this class drives motor output.
    #[inline]
    pub fn is_motor(self) -> bool {
        matches!(self, NeuronClass::Motor | NeuronClass::Descending)
    }
}

/// One synapse.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Synapse {
    /// Post-synaptic neuron.
    pub post: NeuronId,
    /// Weight in pA-equivalent (positive; sign is separate).
    pub weight: f32,
    /// Axonal + synaptic delay, ms.
    pub delay_ms: f32,
    /// Excitatory or inhibitory.
    pub sign: Sign,
}

/// Per-neuron metadata attached alongside the CSR outgoing table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronMeta {
    /// Functional class.
    pub class: NeuronClass,
    /// Module (stochastic-block index).
    pub module: u16,
    /// Resting bias current (nA), small.
    pub bias_pa: f32,
}

/// Configuration for the synthetic connectome.
#[derive(Clone, Debug)]
pub struct ConnectomeConfig {
    /// Number of neurons. Default 1024. Scalable to 10k.
    pub num_neurons: u32,
    /// Number of modules. Default 70.
    pub num_modules: u16,
    /// Designated hub modules (denser inter-module edges). Default 6.
    pub num_hub_modules: u16,
    /// Target average out-degree (synapses / neuron). Default ~48,
    /// giving ~49k edges at N=1024.
    pub avg_out_degree: f32,
    /// Log-normal mean (log μ) for synapse weight.
    pub weight_log_mu: f32,
    /// Log-normal sigma (log σ) for synapse weight.
    pub weight_log_sigma: f32,
    /// Min delay clamp (ms).
    pub delay_min_ms: f32,
    /// Max delay clamp (ms).
    pub delay_max_ms: f32,
    /// Baseline intra-module connection probability contribution.
    pub p_within: f32,
    /// Inter-module connection probability contribution (non-hub).
    pub p_between: f32,
    /// Hub-module inter-connection boost.
    pub p_hub_boost: f32,
    /// Deterministic seed.
    pub seed: u64,
}

impl Default for ConnectomeConfig {
    fn default() -> Self {
        Self {
            num_neurons: 1024,
            num_modules: 70,
            num_hub_modules: 6,
            avg_out_degree: 48.0,
            weight_log_mu: -0.5,
            weight_log_sigma: 0.9,
            delay_min_ms: 0.5,
            delay_max_ms: 10.0,
            p_within: 0.12,
            p_between: 0.004,
            p_hub_boost: 0.018,
            seed: 0x51FE_D0FF_CAFE_BABE,
        }
    }
}

/// Compact serializable copy of the config fields that influence
/// generation (used in the persisted blob).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectomeSerCfg {
    pub(crate) num_neurons: u32,
    pub(crate) num_modules: u16,
    pub(crate) num_hub_modules: u16,
    pub(crate) seed: u64,
}

impl From<&ConnectomeConfig> for ConnectomeSerCfg {
    fn from(c: &ConnectomeConfig) -> Self {
        Self {
            num_neurons: c.num_neurons,
            num_modules: c.num_modules,
            num_hub_modules: c.num_hub_modules,
            seed: c.seed,
        }
    }
}
