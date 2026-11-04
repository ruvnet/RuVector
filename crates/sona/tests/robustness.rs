//! Robustness against malformed caller input (NaN, wrong lengths, rank 0).
//!
//! These inputs arrive straight from JavaScript through the NAPI and WASM
//! bindings, so they must never panic (a panic aborts the Node process) and
//! must never poison the learned state.

use ruvector_sona::{BaseLoRA, SonaConfig, SonaEngine};

const DIM: usize = 16;

fn embedding(seed: usize) -> Vec<f32> {
    (0..DIM)
        .map(|i| ((i + seed) % 7) as f32 / 7.0 + 0.1)
        .collect()
}

fn submit(engine: &SonaEngine, emb: Vec<f32>, reward: f32, quality: f32) {
    let mut builder = engine.begin_trajectory(emb.clone());
    builder.add_step(emb, vec![], reward);
    engine.end_trajectory(builder, quality);
}

fn micro_out(engine: &SonaEngine, input: &[f32]) -> Vec<f32> {
    let mut out = input.to_vec();
    engine.apply_micro_lora(input, &mut out);
    out
}

#[test]
fn nan_quality_does_not_stop_instant_learning() {
    let engine = SonaEngine::new(DIM);
    submit(&engine, embedding(0), f32::NAN, f32::NAN);
    engine.flush();

    let fisher_ok = engine.coordinator().instant().ewc().map_or(true, |ewc| {
        ewc.read().importance_scores().iter().all(|f| f.is_finite())
    });
    assert!(
        fisher_ok,
        "a NaN trajectory poisoned the instant-loop EWC++ Fisher"
    );

    let probe = embedding(3);
    let before = micro_out(&engine, &probe);
    for i in 0..20 {
        submit(&engine, embedding(i), 1.0, 0.9);
    }
    engine.flush();
    let after = micro_out(&engine, &probe);

    assert!(after.iter().all(|v| v.is_finite()));
    assert_ne!(
        before, after,
        "micro-LoRA stopped learning after one NaN trajectory"
    );
}

#[test]
fn nan_trajectory_does_not_poison_base_lora() {
    let engine = SonaEngine::new(DIM);
    submit(&engine, embedding(0), 1.0, f32::NAN);
    let mut nan_embedding = embedding(1);
    nan_embedding[0] = f32::NAN;
    submit(&engine, nan_embedding, 1.0, 0.9);
    submit(&engine, embedding(2), f32::INFINITY, 0.9);
    engine.force_learn();

    let input = embedding(4);
    let mut out = input.clone();
    engine.apply_base_lora(0, &input, &mut out);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "base-LoRA output is not finite: {:?}",
        out
    );
}

#[test]
fn base_lora_wrong_length_does_not_panic() {
    let lora = BaseLoRA::new(8, 2, 1);
    let input = vec![1.0f32; 20];
    let mut output = vec![1.0f32; 20];
    lora.forward_layer(0, &input, &mut output);
    assert_eq!(
        output,
        vec![1.0f32; 20],
        "mismatched lengths must be a no-op"
    );

    let mut short = vec![1.0f32; 3];
    lora.forward_layer(0, &[1.0; 3], &mut short);
    assert_eq!(short, vec![1.0f32; 3]);
}

#[test]
fn base_lora_rank_zero_is_finite() {
    let engine = SonaEngine::with_config(SonaConfig {
        hidden_dim: DIM,
        embedding_dim: DIM,
        base_lora_rank: 0,
        ..Default::default()
    });
    let input = embedding(0);
    let mut out = input.clone();
    engine.apply_base_lora(0, &input, &mut out);
    assert_eq!(out, input, "rank-0 base LoRA must be the identity, not NaN");
}
