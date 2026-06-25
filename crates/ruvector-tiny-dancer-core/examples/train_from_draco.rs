//! End-to-end FastGRNN training from a DRACO routing dataset (ADR-252).
//!
//! Demonstrates the full pipeline the way `@ruvector/tiny-dancer` would use it:
//!   DRACO rows ({ embedding, scores } + prices)  →  TrainingDataset::from_draco
//!   →  Trainer::train (real gradients + Adam)  →  model.save(.safetensors)
//!   →  FastGRNN::load  →  inference (route a new query).
//!
//! Run:  cargo run --example train_from_draco -p ruvector-tiny-dancer-core

use std::collections::HashMap;

use ruvector_tiny_dancer_core::model::{FastGRNN, FastGRNNConfig};
use ruvector_tiny_dancer_core::training::{DracoRow, Trainer, TrainingConfig, TrainingDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Synthesize a DRACO dataset. Quality of the cheap model (haiku) tracks a
    //    simple signal in the embedding; opus is reliably strong. Same shape as
    //    @metaharness/router's `fromExamples` / `trainRouter`.
    const DIM: usize = 8;
    let prices: HashMap<String, f32> = [("haiku".into(), 1.0_f32), ("opus".into(), 15.0_f32)]
        .into_iter()
        .collect();

    let mut rng_state = 0x1234_5678_u64;
    let mut next = || {
        // small deterministic LCG so the example is reproducible
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let mut rows = Vec::new();
    for _ in 0..300 {
        let emb: Vec<f32> = (0..DIM).map(|_| next()).collect();
        // "easy" query (cheap model good enough) when the first coords are positive.
        let easy = emb[0] + emb[1] > 0.0;
        let haiku_q = if easy { 0.9 } else { 0.5 };
        let opus_q = 0.93;
        let scores: HashMap<String, f32> = [("haiku".into(), haiku_q), ("opus".into(), opus_q)]
            .into_iter()
            .collect();
        rows.push(DracoRow {
            embedding: emb,
            scores,
        });
    }

    let dataset = TrainingDataset::from_draco(&rows, &prices, 0.05)?;
    println!("DRACO dataset: {} rows, dim {}", dataset.len(), DIM);

    // 2. Train a FastGRNN router.
    let model_config = FastGRNNConfig {
        input_dim: DIM,
        hidden_dim: 12,
        output_dim: 1,
        ..Default::default()
    };
    let train_config = TrainingConfig {
        learning_rate: 0.05,
        batch_size: 32,
        epochs: 40,
        early_stopping_patience: None,
        l2_reg: 0.0,
        ..Default::default()
    };
    let mut model = FastGRNN::new(model_config.clone())?;
    let metrics = Trainer::new(&model_config, train_config).train(&mut model, &dataset)?;
    let last = metrics.last().unwrap();
    println!(
        "trained: train_loss={:.4} train_acc={:.3} val_acc={:.3}",
        last.train_loss, last.train_accuracy, last.val_accuracy
    );

    // 3. Persist + reload (safetensors).
    let dir = std::env::temp_dir();
    let path = dir.join("tiny-dancer-router.safetensors");
    model.save(&path)?;
    println!(
        "saved -> {} ({} bytes)",
        path.display(),
        std::fs::metadata(&path)?.len()
    );
    let loaded = FastGRNN::load(&path)?;

    // 4. Route two new queries: "use light model?" = sigmoid score >= 0.5.
    let easy_q = {
        let mut v = vec![0.0f32; DIM];
        v[0] = 0.8;
        v[1] = 0.6;
        v
    };
    let hard_q = {
        let mut v = vec![0.0f32; DIM];
        v[0] = -0.8;
        v[1] = -0.6;
        v
    };
    let s_easy = loaded.forward(&easy_q, None)?;
    let s_hard = loaded.forward(&hard_q, None)?;
    println!(
        "route(easy) score={:.3} -> {}",
        s_easy,
        if s_easy >= 0.5 {
            "haiku (cheap)"
        } else {
            "opus"
        }
    );
    println!(
        "route(hard) score={:.3} -> {}",
        s_hard,
        if s_hard >= 0.5 {
            "haiku (cheap)"
        } else {
            "opus"
        }
    );

    Ok(())
}
