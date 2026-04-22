//! Connectome schema, stochastic-block-model generator, and compact
//! binary serialization. Split across four submodules:
//!
//! - `schema`    — public types (`NeuronId`, `FlyWireNeuronId`, `Sign`,
//!                `NeuronClass`, `Synapse`, `NeuronMeta`,
//!                `ConnectomeConfig`).
//! - `generator` — deterministic SBM generator + helpers.
//! - `persist`   — bincode-backed binary round-trip.
//! - `flywire`   — FlyWire v783 TSV ingest (real wiring path).
//!
//! See `docs/research/connectome-ruvector/02-connectome-layer.md` for
//! the schema design and the log-normal / hub-module statistics this
//! generator targets, and ADR-154 §13 for the FlyWire ingest hand-off.

pub mod flywire;
pub mod generator;
pub mod persist;
pub mod schema;

pub use flywire::{load_flywire, FlywireError};
pub use generator::Connectome;
pub use persist::ConnectomeError;
pub use schema::{
    ConnectomeConfig, FlyWireNeuronId, NeuronClass, NeuronId, NeuronMeta, Sign, Synapse,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism_same_seed_same_bytes() {
        let cfg = ConnectomeConfig {
            num_neurons: 256,
            ..ConnectomeConfig::default()
        };
        let a = Connectome::generate(&cfg);
        let b = Connectome::generate(&cfg);
        assert_eq!(a.num_neurons(), b.num_neurons());
        assert_eq!(a.num_synapses(), b.num_synapses());
        assert_eq!(a.row_ptr(), b.row_ptr());
        let ab = a.to_bytes().expect("ser");
        let bb = b.to_bytes().expect("ser");
        assert_eq!(ab, bb);
    }

    #[test]
    fn scales_to_10k() {
        let cfg = ConnectomeConfig {
            num_neurons: 10_000,
            avg_out_degree: 24.0,
            ..ConnectomeConfig::default()
        };
        let c = Connectome::generate(&cfg);
        assert_eq!(c.num_neurons(), 10_000);
        assert!(c.num_synapses() > 100_000);
    }
}
