//! FlyWire v783 on-disk record schema.
//!
//! Three serde structs, one per published TSV file in the release:
//!
//! - [`NeuronRecord`]   — one row per neuron; union of fields across
//!   `neurons.tsv` plus the parts of `classification.tsv` / NT tables
//!   that the loader consumes in a single pass.
//! - [`SynapseRecord`]  — one row per directed pre→post edge in
//!   `connections.tsv`.
//! - [`CellTypeRecord`] — one row per neuron in
//!   `classification.tsv`; used as an override table when the primary
//!   `neurons.tsv` lacks a cell-type assignment.
//!
//! The column names match the published v783 schema (see
//! `docs/research/connectome-ruvector/02-connectome-layer.md` §2).
//! Unknown columns are ignored by the CSV reader so adding downstream
//! fields (e.g. `hemilineage`) does not require a schema version bump.

use serde::{Deserialize, Serialize};

/// One row of the neurons TSV.
///
/// Columns mirror the FlyWire v783 release. `neuron_id` is the stable
/// 64-bit root id; `supervoxel_id` is the coarse segmentation handle
/// (kept for provenance, not used by the loader in v1); `cell_type`,
/// `nt_type`, `side`, `nerve`, and `flow` are all string-enum encoded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuronRecord {
    /// Stable FlyWire root id.
    pub neuron_id: u64,
    /// Supervoxel id (provenance only).
    #[serde(default)]
    pub supervoxel_id: u64,
    /// Cell type, e.g. "KC_g", "MBON01", "DNp01". Empty string
    /// (deserialized to `None`) is allowed when the classification is
    /// unresolved.
    #[serde(default)]
    pub cell_type: Option<String>,
    /// Dominant predicted neurotransmitter: "ACH", "GLUT", "GABA",
    /// "SER", "OCT", "DOP", "HIST".
    pub nt_type: String,
    /// Anatomical side: "left", "right", "center".
    #[serde(default)]
    pub side: Option<String>,
    /// Peripheral nerve id (Wikipedia naming), if afferent / efferent.
    #[serde(default)]
    pub nerve: Option<String>,
    /// Flow class: "afferent", "efferent", "intrinsic".
    #[serde(default)]
    pub flow: Option<String>,
    /// Optional super-class label (e.g. "optic", "central", "motor").
    #[serde(default)]
    pub super_class: Option<String>,
}

/// One row of the connections TSV.
///
/// `pre_id` and `post_id` are stable FlyWire root ids; both must resolve
/// to a row in the neurons TSV or the loader errors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SynapseRecord {
    /// Pre-synaptic neuron id.
    pub pre_id: u64,
    /// Post-synaptic neuron id.
    pub post_id: u64,
    /// Neuropil region label (e.g. "MB_CA_L").
    #[serde(default)]
    pub neuropil: Option<String>,
    /// Aggregated synapse count for this directed pair.
    pub syn_count: u32,
    /// Effective weight reported by the release; loader uses
    /// `syn_count` when this field is absent or zero.
    #[serde(default)]
    pub syn_weight: f32,
    /// Per-edge NT prediction (optional; falls back to the pre
    /// neuron's dominant NT when unset).
    #[serde(default)]
    pub nt_type: Option<String>,
}

/// One row of the classification TSV.
///
/// Provides authoritative cell-type / super-class labels that can
/// override or fill in the fields on [`NeuronRecord`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellTypeRecord {
    /// Stable FlyWire root id.
    pub neuron_id: u64,
    /// Primary cell-type label.
    pub cell_type: String,
    /// Optional coarse super-class.
    #[serde(default)]
    pub super_class: Option<String>,
}

impl NeuronRecord {
    /// Effective cell-type string after folding in the classification
    /// override. `class_override` wins over `self.cell_type` when both
    /// are present.
    pub fn effective_cell_type(&self, class_override: Option<&str>) -> Option<String> {
        class_override
            .map(str::to_owned)
            .or_else(|| self.cell_type.clone())
    }
}

/// Parsed, normalized neurotransmitter tag. Distinct from the
/// `Sign` enum in the outer schema because several NTs (DA / 5-HT /
/// OA) are neuromodulatory and do not carry a fast-path sign; the
/// loader materializes them as Excitatory in the fast path per the
/// research doc §4 table and records the NT identity on the side.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NeuroTransmitter {
    /// Acetylcholine — fast excitation.
    Acetylcholine,
    /// Glutamate — excitation in central circuits (v1 default).
    Glutamate,
    /// GABA — fast inhibition.
    Gaba,
    /// Histamine — photoreceptor output, inhibitory.
    Histamine,
    /// Serotonin — neuromodulator, rendered excitatory in the fast path.
    Serotonin,
    /// Dopamine — neuromodulator, rendered excitatory in the fast path.
    Dopamine,
    /// Octopamine — neuromodulator, rendered excitatory in the fast path.
    Octopamine,
}

impl NeuroTransmitter {
    /// Whether this NT is routed through the slow neuromodulatory
    /// pool in the research schema. The fast path still assigns a
    /// sign so the LIF engine has something to integrate; this flag
    /// surfaces the category so analysis code can exclude slow edges.
    pub fn is_modulatory(self) -> bool {
        matches!(
            self,
            NeuroTransmitter::Serotonin | NeuroTransmitter::Dopamine | NeuroTransmitter::Octopamine
        )
    }
}
