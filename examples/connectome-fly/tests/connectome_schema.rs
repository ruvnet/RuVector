//! Connectome schema + serialization round-trip invariants.

use connectome_fly::{Connectome, ConnectomeConfig, NeuronClass};

#[test]
fn generate_hits_target_scale_defaults() {
    let cfg = ConnectomeConfig::default();
    let c = Connectome::generate(&cfg);
    assert_eq!(c.num_neurons(), 1024);
    // Target avg out-degree 48 → on the order of 20k–60k synapses
    // depending on random rejection dynamics. Bound wide enough to be
    // stable across seeds, tight enough to catch regressions that
    // zero out edge generation.
    assert!(
        c.num_synapses() > 10_000,
        "synapse count too low: {}",
        c.num_synapses()
    );
    assert!(
        c.num_synapses() < 70_000,
        "synapse count too high: {}",
        c.num_synapses()
    );
    assert!(!c.sensory_neurons().is_empty());
    assert!(!c.motor_neurons().is_empty());
}

#[test]
fn serialization_is_byte_identical_same_seed() {
    let cfg = ConnectomeConfig {
        num_neurons: 512,
        ..ConnectomeConfig::default()
    };
    let a = Connectome::generate(&cfg);
    let ab = a.to_bytes().expect("serialize");
    let b = Connectome::from_bytes(&ab).expect("deserialize");
    assert_eq!(a.num_neurons(), b.num_neurons());
    assert_eq!(a.num_synapses(), b.num_synapses());
    // Round-trip bytes exactly.
    let bb = b.to_bytes().expect("serialize twice");
    assert_eq!(ab, bb, "round-trip serialization is not stable");
}

#[test]
fn inhibitory_fraction_is_in_target_band() {
    let cfg = ConnectomeConfig::default();
    let c = Connectome::generate(&cfg);
    // Fraction of *synapses* marked inhibitory. Target in §02 is ~10%
    // on the population, but local interneurons push this toward 15–25%
    // at the synapse level because they are densely fan-out.
    let mut inh = 0_u64;
    for s in c.synapses() {
        if matches!(s.sign, connectome_fly::Sign::Inhibitory) {
            inh += 1;
        }
    }
    let frac = inh as f32 / c.num_synapses() as f32;
    assert!(
        (0.05..0.35).contains(&frac),
        "inhibitory fraction {frac:.3} out of expected [0.05, 0.35]"
    );
}

#[test]
fn class_coverage_is_nonempty_for_key_classes() {
    let cfg = ConnectomeConfig::default();
    let c = Connectome::generate(&cfg);
    let by_class = c.by_class();
    // KenyonCell, Motor, LocalInter should all be present in N=1024.
    for cls in [
        NeuronClass::KenyonCell,
        NeuronClass::Motor,
        NeuronClass::LocalInter,
    ] {
        assert!(
            !by_class[cls as usize].is_empty(),
            "class {:?} unexpectedly empty",
            cls
        );
    }
}

#[test]
fn weight_log_normal_stats_roughly_match_config() {
    let cfg = ConnectomeConfig::default();
    let c = Connectome::generate(&cfg);
    let mut logs = Vec::with_capacity(c.num_synapses());
    for s in c.synapses() {
        if s.weight > 0.0 {
            logs.push(s.weight.ln());
        }
    }
    let mean: f32 = logs.iter().sum::<f32>() / logs.len() as f32;
    // Generator applies an extra 1.3× for inhibitory weights, so the
    // measured log-mean shifts slightly upward from the config mean.
    let configured_mu = cfg.weight_log_mu;
    assert!(
        (mean - configured_mu).abs() < 0.25,
        "log-weight mean drifted: measured={mean:.3} configured={configured_mu:.3}"
    );
}
