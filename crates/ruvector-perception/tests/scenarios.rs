//! The brief's flagship scenarios:
//! 1. Move an inert object: RF/vibration/acoustic support, thermal contradicts,
//!    novelty high, action = observe (a structured delta witness, not a label).
//! 2. A bedtime routine whose return never happens -> absence safety signal.

use ruvector_perception::{
    novelty_level, Action, DeltaEngine, EngineConfig, Modality, ProofGate, Reading, SequenceMonitor,
};

fn warmup(eng: &mut DeltaEngine) {
    // Build responsiveness in table_left_zone across RF/vibration/acoustic/thermal
    // (all historically react here); other zones stay quiet.
    for i in 0..8u64 {
        let hi = (i % 2) as f32;
        let rs = vec![
            Reading::new("table_left_zone", Modality::Rf, hi),
            Reading::new("table_left_zone", Modality::Vibration, hi),
            Reading::new("table_left_zone", Modality::Acoustic, hi),
            Reading::new("table_left_zone", Modality::Thermal, 20.0 + hi),
            Reading::new("table_right_zone", Modality::Rf, 0.0),
            Reading::new("window_zone", Modality::Rf, 0.0),
        ];
        eng.observe(&rs, i);
    }
}

#[test]
fn inert_object_move_produces_structured_delta_witness() {
    let mut eng = DeltaEngine::new(EngineConfig::default());
    warmup(&mut eng);

    // Construct the event relative to learned baselines: RF/vibration/acoustic
    // jump (object moved), thermal exactly at baseline (no heat -> silent).
    let bl = |m| eng.state().baseline("table_left_zone", m);
    let (bl_rf, bl_vib, bl_ac, bl_th) = (
        bl(Modality::Rf),
        bl(Modality::Vibration),
        bl(Modality::Acoustic),
        bl(Modality::Thermal),
    );
    let event = vec![
        Reading::new("table_left_zone", Modality::Rf, bl_rf + 3.0),
        Reading::new("table_left_zone", Modality::Vibration, bl_vib + 3.0),
        Reading::new("table_left_zone", Modality::Acoustic, bl_ac + 3.0),
        Reading::new("table_left_zone", Modality::Thermal, bl_th), // silent
        Reading::new(
            "table_right_zone",
            Modality::Rf,
            eng.state().baseline("table_right_zone", Modality::Rf),
        ),
        Reading::new(
            "window_zone",
            Modality::Rf,
            eng.state().baseline("window_zone", Modality::Rf),
        ),
    ];

    let prev = eng.state().baseline("table_left_zone", Modality::Rf); // touch state
    let _ = prev;
    let w = eng.observe(&event, 100);

    // The exact witness shape from the brief.
    assert_eq!(w.changed_boundary, "table_left_zone");
    assert!(w.supporting_modalities.contains(&Modality::Rf));
    assert!(w.supporting_modalities.contains(&Modality::Vibration));
    assert!(w.supporting_modalities.contains(&Modality::Acoustic));
    assert!(!w.supporting_modalities.contains(&Modality::Thermal));
    assert!(
        w.contradicting_modalities.contains(&Modality::Thermal),
        "thermal should contradict (usually reacts here, stayed silent): {:?}",
        w.contradicting_modalities
    );
    assert_eq!(novelty_level(w.novelty, &ProofGate::default()), "high");
    assert!(w.coherence > 0.5, "boundary not clean: {}", w.coherence);
    // Contradicted evidence is capped at Observe — it does not escalate.
    assert_eq!(w.action, Action::Observe);
    // Auditable evidence chain.
    assert_eq!(w.evidence_hash.len(), 64);
    assert!(
        w.prev_hash.is_some(),
        "witness should chain to the warmup history"
    );
}

#[test]
fn missing_routine_return_is_a_safety_signal() {
    let mut routine = SequenceMonitor::new(
        vec![
            "bed_exit".into(),
            "bathroom_path".into(),
            "return_path".into(),
        ],
        100,
    );
    routine.observe_zone("bed_exit", 0);
    routine.observe_zone("bathroom_path", 10);
    // The return edge never appears: the sequence graph stays incomplete.
    assert!(routine.check(50).is_none());
    let absence = routine.check(300).expect("overdue return is a signal");
    assert_eq!(absence.missing_step, "return_path");
    assert_eq!(absence.after, "bathroom_path");
}
