//! FlyWire v783 ingest — acceptance tests.
//!
//! These tests exercise every named failure mode of the loader plus a
//! round-trip on the 100-neuron fixture. The fixture lives as Rust
//! string constants (see `src/connectome/flywire/fixture.rs`) so CI
//! does not need the ~2 GB FlyWire release on disk.

use std::fs;
use std::path::PathBuf;

use connectome_fly::connectome::flywire::{
    classify_cell_type, classify_cell_type_strict, fixture, load_flywire, nt_to_sign, parse_nt,
};
use connectome_fly::{FlyWireNeuronId, FlywireError, NeuronClass, Sign};
use tempfile::TempDir;

fn setup_fixture() -> (TempDir, fixture::FixturePaths) {
    let dir = TempDir::new().expect("temp dir");
    let paths = fixture::write_fixture(dir.path()).expect("write fixture");
    (dir, paths)
}

#[test]
fn schema_round_trip_neuron_and_synapse_counts_match_fixture() {
    let (dir, _paths) = setup_fixture();
    let c = load_flywire(dir.path()).expect("load fixture");
    assert_eq!(
        c.num_neurons(),
        fixture::EXPECTED_NEURONS,
        "neuron count mismatch vs fixture declaration",
    );
    // Connection count in the fixture is 159 directed edges; some may
    // be dropped as self-loops or by NT filtering. We expect no
    // drops in the fixture (no self-loops authored), so equality holds.
    assert_eq!(
        c.num_synapses(),
        fixture::EXPECTED_SYNAPSES,
        "synapse count mismatch vs fixture declaration",
    );
}

#[test]
fn flywire_ids_are_parallel_to_dense_ids() {
    let (dir, _paths) = setup_fixture();
    let c = load_flywire(dir.path()).expect("load fixture");
    let ids = c.flywire_ids().expect("flywire_ids set after load");
    assert_eq!(ids.len(), c.num_neurons());
    assert_eq!(ids[0], FlyWireNeuronId(10_000_001));
    assert_eq!(ids[99], FlyWireNeuronId(10_000_100));
    // Monotonic in the fixture (authored sequentially).
    for win in ids.windows(2) {
        assert!(win[0].raw() < win[1].raw());
    }
}

#[test]
fn determinism_two_loads_bit_identical_bincode() {
    let (dir, _paths) = setup_fixture();
    let a = load_flywire(dir.path()).expect("load 1");
    let b = load_flywire(dir.path()).expect("load 2");
    assert_eq!(a.num_neurons(), b.num_neurons());
    assert_eq!(a.num_synapses(), b.num_synapses());
    let ab = a.to_bytes().expect("ser a");
    let bb = b.to_bytes().expect("ser b");
    assert_eq!(ab, bb, "FlyWire ingest is not deterministic");
}

#[test]
fn nt_to_sign_covers_release_documented_labels() {
    // Excitatory.
    for raw in ["ACH", "GLUT", "ACETYLCHOLINE", "Glutamate"] {
        let nt = parse_nt(raw, 0).expect(raw);
        assert_eq!(nt_to_sign(nt), Sign::Excitatory);
    }
    // Inhibitory.
    for raw in ["GABA", "HIST", "histamine"] {
        let nt = parse_nt(raw, 0).expect(raw);
        assert_eq!(nt_to_sign(nt), Sign::Inhibitory);
    }
    // Neuromodulatory — mapped to excitatory in the fast path per
    // research doc §4 (slow pool lives outside the fast path).
    for raw in ["DOP", "SER", "OCT", "5-HT", "DA", "OA"] {
        let nt = parse_nt(raw, 0).expect(raw);
        assert_eq!(nt_to_sign(nt), Sign::Excitatory);
    }
}

#[test]
fn unknown_nt_type_is_a_named_error_not_silent_default() {
    let err = parse_nt("PANIC", 42).expect_err("must reject unknown NT");
    match err {
        FlywireError::UnknownNtType { raw, neuron_id } => {
            assert_eq!(raw, "PANIC");
            assert_eq!(neuron_id, 42);
        }
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn cell_type_coverage_hits_key_classes() {
    let (dir, _paths) = setup_fixture();
    let c = load_flywire(dir.path()).expect("load fixture");
    // Every coarse class that exists in the fixture must be populated.
    // The fixture is authored to cover these explicitly.
    for cls in [
        NeuronClass::PhotoReceptor,
        NeuronClass::Chemosensory,
        NeuronClass::Mechanosensory,
        NeuronClass::OpticLocal,
        NeuronClass::KenyonCell,
        NeuronClass::MbOutput,
        NeuronClass::CentralComplex,
        NeuronClass::LateralAccessory,
        NeuronClass::Descending,
        NeuronClass::Ascending,
        NeuronClass::Motor,
        NeuronClass::LocalInter,
        NeuronClass::Projection,
        NeuronClass::Modulatory,
    ] {
        assert!(
            !c.by_class()[cls as usize].is_empty(),
            "class {cls:?} unexpectedly empty after fixture load",
        );
    }
    // Sensory + motor indices must also be populated (ADR §3.4 AC
    // stimulus / readout needs them).
    assert!(!c.sensory_neurons().is_empty());
    assert!(!c.motor_neurons().is_empty());
}

#[test]
fn classify_cell_type_known_prefixes() {
    assert_eq!(
        classify_cell_type(Some("KC_g"), None).unwrap(),
        NeuronClass::KenyonCell,
    );
    assert_eq!(
        classify_cell_type(Some("MBON05"), None).unwrap(),
        NeuronClass::MbOutput,
    );
    assert_eq!(
        classify_cell_type(Some("DNp01"), None).unwrap(),
        NeuronClass::Descending,
    );
    assert_eq!(
        classify_cell_type(Some("Motor_leg_1"), None).unwrap(),
        NeuronClass::Motor,
    );
    assert_eq!(
        classify_cell_type(Some("LN_GABA_A"), None).unwrap(),
        NeuronClass::LocalInter,
    );
    // Flow fallback when cell type is missing.
    assert_eq!(
        classify_cell_type(None, Some("efferent")).unwrap(),
        NeuronClass::Motor,
    );
    // Both missing falls through to Other.
    assert_eq!(classify_cell_type(None, None).unwrap(), NeuronClass::Other);
}

#[test]
fn malformed_tsv_surfaces_row_level_error() {
    let dir = TempDir::new().expect("temp");
    // Valid neurons + classification files.
    fs::write(dir.path().join("neurons.tsv"), fixture::neurons_tsv()).unwrap();
    fs::write(
        dir.path().join("classification.tsv"),
        fixture::classification_tsv(),
    )
    .unwrap();
    // Broken connections file: header is valid, but the second data
    // row has a non-integer pre_id.
    let broken = "pre_id\tpost_id\tneuropil\tsyn_count\tsyn_weight\tnt_type\n\
                  10000005\t10000013\tMB_CA_L\t12\t12.0\tACH\n\
                  BROKEN\t10000013\tMB_CA_L\t12\t12.0\tACH\n";
    fs::write(dir.path().join("connections.tsv"), broken).unwrap();

    let err = load_flywire(dir.path()).expect_err("must fail on BROKEN row");
    match err {
        FlywireError::MalformedRow { file, line, .. } => {
            assert_eq!(file, "connections.tsv");
            assert_eq!(line, 3, "expected line 3 (header=1, first data=2)");
        }
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn unknown_cell_type_folds_to_other_in_default_mode() {
    // Default classify_cell_type: unmapped -> Other. FlyWire has ~8k
    // cell types and the coarse bucket is the v1 contract.
    let class = classify_cell_type(Some("ZZZ_novel_type"), None).unwrap();
    assert_eq!(class, NeuronClass::Other);
}

#[test]
fn unknown_cell_type_is_a_named_error_in_strict_mode() {
    // Strict path surfaces `FlywireError::UnknownCellType` so callers
    // that want to audit prefix coverage can opt in.
    let err = classify_cell_type_strict(Some("ZZZ_novel_type"), None, 99)
        .expect_err("strict must reject unknown cell type");
    match err {
        FlywireError::UnknownCellType { raw, neuron_id } => {
            assert_eq!(raw, "ZZZ_novel_type");
            assert_eq!(neuron_id, 99);
        }
        other => panic!("wrong variant: {other:?}"),
    }
    // Known types still pass under strict mode.
    assert_eq!(
        classify_cell_type_strict(Some("KC_g"), None, 1).unwrap(),
        NeuronClass::KenyonCell,
    );
}

#[test]
fn unknown_nt_type_in_neurons_file_fails_load() {
    let dir = TempDir::new().expect("temp");
    // Replace the very first NT label with a bogus one.
    let bad_neurons = fixture::neurons_tsv().replacen(
        "10000001\t9000001\tPR_R1\tHIST\t",
        "10000001\t9000001\tPR_R1\tBOGUS\t",
        1,
    );
    fs::write(dir.path().join("neurons.tsv"), bad_neurons).unwrap();
    fs::write(
        dir.path().join("classification.tsv"),
        fixture::classification_tsv(),
    )
    .unwrap();
    fs::write(
        dir.path().join("connections.tsv"),
        fixture::connections_tsv(),
    )
    .unwrap();

    let err = load_flywire(dir.path()).expect_err("must fail on BOGUS nt_type");
    match err {
        FlywireError::UnknownNtType { raw, neuron_id } => {
            assert_eq!(raw, "BOGUS");
            assert_eq!(neuron_id, 10_000_001);
        }
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn dangling_synapse_reference_is_a_named_error() {
    let dir = TempDir::new().expect("temp");
    fs::write(dir.path().join("neurons.tsv"), fixture::neurons_tsv()).unwrap();
    fs::write(
        dir.path().join("classification.tsv"),
        fixture::classification_tsv(),
    )
    .unwrap();
    // Append a synapse pointing at a nonexistent post_id.
    let mut connections = fixture::connections_tsv();
    connections.push_str("10000005\t99999999\tSMP_L\t3\t3.0\tACH\n");
    fs::write(dir.path().join("connections.tsv"), connections).unwrap();

    let err = load_flywire(dir.path()).expect_err("must fail on dangling post_id");
    match err {
        FlywireError::UnknownPostNeuron(id) => assert_eq!(id, 99_999_999),
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn duplicate_neuron_id_is_a_named_error() {
    let dir = TempDir::new().expect("temp");
    // Duplicate the first neuron row at the tail.
    let mut neurons = fixture::neurons_tsv();
    neurons.push_str("10000001\t9000001\tPR_R1\tHIST\tleft\tOCN\tafferent\tsensory\n");
    fs::write(dir.path().join("neurons.tsv"), neurons).unwrap();
    fs::write(
        dir.path().join("classification.tsv"),
        fixture::classification_tsv(),
    )
    .unwrap();
    fs::write(
        dir.path().join("connections.tsv"),
        fixture::connections_tsv(),
    )
    .unwrap();

    let err = load_flywire(dir.path()).expect_err("must fail on duplicate neuron_id");
    match err {
        FlywireError::DuplicateNeuron(id) => assert_eq!(id, 10_000_001),
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn classification_file_is_optional() {
    // No classification.tsv — cell-type is taken from neurons.tsv
    // directly. The loader must still succeed.
    let dir = TempDir::new().expect("temp");
    fs::write(dir.path().join("neurons.tsv"), fixture::neurons_tsv()).unwrap();
    fs::write(
        dir.path().join("connections.tsv"),
        fixture::connections_tsv(),
    )
    .unwrap();
    // Intentionally do NOT write classification.tsv.
    let c = load_flywire(dir.path()).expect("load without classification");
    assert_eq!(c.num_neurons(), fixture::EXPECTED_NEURONS);
}

#[test]
fn missing_neurons_file_surfaces_io_error() {
    let dir = TempDir::new().expect("temp");
    // No neurons.tsv at all.
    let err = load_flywire(dir.path()).expect_err("must fail without neurons.tsv");
    match err {
        FlywireError::Io { file, .. } => {
            assert_eq!(file, "neurons.tsv");
        }
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn synapse_signs_follow_nt_mapping_in_fixture() {
    let (dir, _paths) = setup_fixture();
    let c = load_flywire(dir.path()).expect("load fixture");
    // Fixture includes several GABA and HIST edges — expect inhibitory
    // synapses to be a non-zero fraction but bounded above by the
    // balance of excitatory ACH / GLUT edges.
    let mut inh = 0_usize;
    let mut exc = 0_usize;
    for s in c.synapses() {
        match s.sign {
            Sign::Inhibitory => inh += 1,
            Sign::Excitatory => exc += 1,
        }
    }
    assert!(inh > 0, "fixture has no inhibitory edges: unexpected");
    assert!(exc > 0, "fixture has no excitatory edges: unexpected");
    let frac = inh as f32 / c.num_synapses() as f32;
    assert!(
        (0.05..0.5).contains(&frac),
        "inhibitory fraction {frac:.3} out of expected band [0.05, 0.5]",
    );
}

#[test]
fn dir_label_on_io_error_uses_filename_only() {
    // Defensive: the Io variant reports a short filename, not a full
    // path. This keeps the error deterministic across tempdir roots.
    let bogus = PathBuf::from("/nonexistent/__connectome_fly_test__");
    let err = load_flywire(&bogus).expect_err("must fail on missing dir");
    match err {
        FlywireError::Io { file, .. } => assert_eq!(file, "neurons.tsv"),
        other => panic!("wrong variant: {other:?}"),
    }
}
