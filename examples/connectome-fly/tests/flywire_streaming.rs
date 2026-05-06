//! Equivalence + determinism tests for `load_flywire_streaming`.
//!
//! Invariant: the streaming loader produces a `Connectome` byte-
//! identical to `load_flywire` on the same input. Streaming is purely
//! a memory optimisation — ADR-154 §13.

use std::fs;
use std::path::PathBuf;

use connectome_fly::connectome::flywire::{
    fixture, load_flywire_streaming,
    loader::{self},
};
use connectome_fly::connectome::Connectome;
use tempfile::TempDir;

fn write_fixture_dir() -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    let root: PathBuf = dir.path().to_path_buf();
    fs::write(root.join("neurons.tsv"), fixture::neurons_tsv()).expect("write neurons");
    fs::write(root.join("connections.tsv"), fixture::connections_tsv())
        .expect("write connections");
    fs::write(
        root.join("classification.tsv"),
        fixture::classification_tsv(),
    )
    .expect("write classification");
    dir
}

fn serialize_bytes(c: &Connectome) -> Vec<u8> {
    c.to_bytes().expect("serialize connectome to bytes")
}

#[test]
fn streaming_matches_non_streaming_byte_identical() {
    let dir = write_fixture_dir();
    let a = loader::load_flywire(dir.path()).expect("non-streaming loader");
    let b = load_flywire_streaming(dir.path()).expect("streaming loader");

    let ba = serialize_bytes(&a);
    let bb = serialize_bytes(&b);
    assert_eq!(
        ba.len(),
        bb.len(),
        "streaming and non-streaming produce different-sized Connectome bytes (non-stream={} stream={})",
        ba.len(),
        bb.len()
    );
    assert_eq!(
        ba, bb,
        "streaming and non-streaming produce different Connectome bytes on the same fixture"
    );
    assert_eq!(a.num_neurons(), b.num_neurons());
    assert_eq!(a.synapses().len(), b.synapses().len());
}

#[test]
fn streaming_is_deterministic_across_repeat_loads() {
    let dir = write_fixture_dir();
    let a = load_flywire_streaming(dir.path()).expect("load 1");
    let b = load_flywire_streaming(dir.path()).expect("load 2");
    let ba = serialize_bytes(&a);
    let bb = serialize_bytes(&b);
    assert_eq!(
        ba, bb,
        "streaming loader is non-deterministic: two loads of the same TSV fixture produced different bytes"
    );
}

#[test]
fn streaming_errors_on_missing_neurons_file() {
    let dir = TempDir::new().expect("tempdir");
    fs::write(dir.path().join("connections.tsv"), fixture::connections_tsv())
        .expect("write connections");
    let res = load_flywire_streaming(dir.path());
    assert!(
        res.is_err(),
        "streaming loader should error when neurons.tsv is missing; got Ok"
    );
}

#[test]
fn streaming_errors_on_dangling_pre_reference() {
    let dir = TempDir::new().expect("tempdir");
    // Minimal valid neurons file with one entry. Column names must
    // match the serde field names on NeuronRecord / SynapseRecord
    // (see src/connectome/flywire/schema.rs).
    let neurons = "\
neuron_id\tnt_type\tflow\n\
1000\tACH\tintrinsic\n";
    // Connection references pre_id 9999 which is not in the neurons file.
    let connections = "\
pre_id\tpost_id\tsyn_count\n\
9999\t1000\t3\n";
    fs::write(dir.path().join("neurons.tsv"), neurons).expect("write neurons");
    fs::write(dir.path().join("connections.tsv"), connections).expect("write connections");
    let res = load_flywire_streaming(dir.path());
    match res {
        Err(connectome_fly::connectome::flywire::FlywireError::UnknownPreNeuron(id)) => {
            assert_eq!(id, 9999);
        }
        other => panic!(
            "expected FlywireError::UnknownPreNeuron(9999); got {:?}",
            other.map(|_| "Ok(Connectome)")
        ),
    }
}
