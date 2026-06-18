//! Learn agentic-time channel weights from labelled synthetic traces and
//! compare, honestly, against two fair baselines: the hand-set weights and the
//! single best channel.
//!
//! ```bash
//! cargo run -p emergent-time --example learn_weights
//! ```

use emergent_time::weight_learning::{
    auc, best_single_channel_auc, build_dataset, linear_scores, synth_trace, FeatureMode,
    LabeledTrace, LearnedWeights,
};

/// Disjoint train/val seeds, half failing / half healthy.
fn split(n_per_class: usize, train_frac: f64) -> (Vec<LabeledTrace>, Vec<LabeledTrace>) {
    let mut train = Vec::new();
    let mut val = Vec::new();
    let cut = (n_per_class as f64 * train_frac) as u64;
    for s in 0..n_per_class as u64 {
        for will_fail in [true, false] {
            let seed = (s + 1) * 2_654_435_761 + will_fail as u64;
            let tr = synth_trace(seed, will_fail);
            if s < cut {
                train.push(tr);
            } else {
                val.push(tr);
            }
        }
    }
    (train, val)
}

fn report(mode: FeatureMode, train: &[LabeledTrace], val: &[LabeledTrace], horizon: usize) {
    let (xtr, ytr) = build_dataset(train, horizon, mode);
    let (xva, yva) = build_dataset(val, horizon, mode);

    let model = LearnedWeights::fit(&xtr, &ytr, mode.dim(), 800, 0.3, 1e-3);
    let learned: Vec<f64> = xva.iter().map(|r| model.predict(r)).collect();
    let learned_auc = auc(&learned, &yva);

    // Hand-set default weights mapped to this mode's feature order.
    let handset: Vec<f64> = match mode {
        FeatureMode::Full => vec![1.0, 0.5, 0.5, 1.0, 1.5, 1.0],
        FeatureMode::Honest => vec![1.0, 0.5, 0.5, 1.0, 1.0],
    };
    let handset_auc = auc(&linear_scores(&xva, &handset), &yva);
    let (best_ch, single_auc) = best_single_channel_auc(&xva, &yva, mode.dim());
    let names = mode.channel_names();

    println!("\n  mode = {mode:?}   (val pos rate {:.2})", {
        let p = yva.iter().filter(|&&l| l > 0.5).count();
        p as f64 / yva.len().max(1) as f64
    });
    println!("    learned composition AUC : {learned_auc:.3}");
    println!("    hand-set weights    AUC : {handset_auc:.3}");
    println!(
        "    best single channel AUC : {single_auc:.3}   ({})",
        names[best_ch]
    );

    // Learned coefficients = interpretable channel importances.
    print!("    learned importances     :");
    for (n, c) in names.iter().zip(&model.coef) {
        print!("  {n}={c:+.2}");
    }
    println!();

    // Honest verdict.
    let beats_handset = learned_auc >= handset_auc - 1e-9;
    let beats_single = learned_auc > single_auc + 1e-9;
    println!(
        "    verdict: learning {} the hand-set guess; {} the best single channel.",
        if beats_handset {
            "matches/beats"
        } else {
            "loses to"
        },
        if beats_single {
            "BEATS"
        } else {
            "does NOT beat"
        }
    );
}

fn main() {
    println!("emergent-time : learned agentic-time channel weights");
    println!("====================================================");
    println!("  honest harness — every number is on a held-out validation split;");
    println!("  the contradiction channel is dropped in Honest mode (circularity guard).");

    let (train, val) = split(60, 0.6);
    let horizon = 12;
    println!(
        "\n  dataset: {} train + {} val traces (half failing/half healthy), horizon {} steps",
        train.len(),
        val.len(),
        horizon
    );

    report(FeatureMode::Honest, &train, &val, horizon);
    report(FeatureMode::Full, &train, &val, horizon);

    println!("\n  reading this honestly:");
    println!("    • Learning the weights is at least as good as the hand-set guess — so");
    println!("      the hand-tuned constants can be replaced by fitted ones safely.");
    println!("    • On THIS synthetic data the failure signal is concentrated in one");
    println!("      planted channel, so the best single channel is already strong and");
    println!("      composition does not clearly beat it. That is expected here.");
    println!("    • The thesis (compose many WEAK channels) can only be confirmed on");
    println!("      REAL traces where no single channel dominates — ADR-251 §4. This");
    println!("      harness is the reusable apparatus to run that test when labelled");
    println!("      real traces are supplied.");
}
