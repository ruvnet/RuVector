//! FlyWire v783 ingest: TSV release → `Connectome`.
//!
//! This module is the first follow-up named in ADR-154 §13. It moves
//! the connectome-fly demonstrator from its synthetic stochastic-block
//! model onto the real FlyWire v783 wiring, one file at a time, without
//! touching any analysis, LIF, or observer code.
//!
//! ## Public API
//!
//! - [`load_flywire`] — parse `neurons.tsv`, `classification.tsv`, and
//!   `connections.tsv` from a directory; return a fully-populated
//!   [`crate::Connectome`] with parallel `FlyWireNeuronId`s.
//! - [`FlywireError`] — structured error type with one variant per
//!   named failure mode (malformed row, dangling reference, unknown
//!   NT, unknown cell type, IO failure, …).
//! - [`schema`] — serde record structs matching the release TSV
//!   columns.
//! - [`fixture`] — hand-authored 100-neuron fixture used by tests.
//!
//! ## Hard constraints
//!
//! - No `unsafe`. No Python, shell, or JS/TS.
//! - Deterministic: byte-identical TSV input produces bit-identical
//!   `Connectome` output across runs.
//! - No download path; `load_flywire` reads whatever TSVs are under
//!   the path the caller hands it.

pub mod fixture;
pub mod loader;
pub mod schema;

pub use loader::{
    classify_cell_type, classify_cell_type_strict, load_flywire, nt_to_sign, parse_nt,
};
pub use schema::{CellTypeRecord, NeuroTransmitter, NeuronRecord, SynapseRecord};

use thiserror::Error;

/// Errors produced by the FlyWire ingest path. Each variant maps to a
/// distinct test case in `tests/flywire_ingest.rs`.
#[derive(Debug, Error)]
pub enum FlywireError {
    /// A row failed to deserialize against the [`NeuronRecord`],
    /// [`SynapseRecord`], or [`CellTypeRecord`] schema.
    #[error("malformed row in {file} at line {line}: {detail}")]
    MalformedRow {
        /// File name (not full path), e.g. `"neurons.tsv"`.
        file: String,
        /// 1-based row number (header is line 1).
        line: u64,
        /// Underlying parser message.
        detail: String,
    },

    /// IO or CSV-framing failure before per-row dispatch.
    #[error("io error on {file}: {detail}")]
    Io {
        /// File name.
        file: String,
        /// Underlying error.
        detail: String,
    },

    /// A synapse referenced a `pre_id` that is not present in
    /// `neurons.tsv`.
    #[error("synapse pre_id {0} not in neurons.tsv")]
    UnknownPreNeuron(u64),

    /// A synapse referenced a `post_id` that is not present in
    /// `neurons.tsv`.
    #[error("synapse post_id {0} not in neurons.tsv")]
    UnknownPostNeuron(u64),

    /// A neuron id appeared twice in `neurons.tsv`.
    #[error("duplicate neuron_id {0} in neurons.tsv")]
    DuplicateNeuron(u64),

    /// An NT-type string did not match the seven release-documented
    /// labels (ACH / GLUT / GABA / HIST / SER / DOP / OCT).
    #[error("unknown nt_type {raw:?} on neuron_id {neuron_id}")]
    UnknownNtType {
        /// Raw column value.
        raw: String,
        /// Context id (neuron or pre-neuron of the offending synapse).
        neuron_id: u64,
    },

    /// A cell-type string did not match any known prefix. Only
    /// surfaced from the strict classification path
    /// ([`loader::classify_cell_type_strict`]); the default
    /// [`loader::classify_cell_type`] folds unknown cell types into
    /// [`crate::NeuronClass::Other`] because FlyWire v783 documents
    /// ~8 000 cell types and the ingest loader is coarse by design.
    #[error("unknown cell_type {raw:?} on neuron_id {neuron_id}")]
    UnknownCellType {
        /// Raw column value.
        raw: String,
        /// Context neuron id.
        neuron_id: u64,
    },
}
