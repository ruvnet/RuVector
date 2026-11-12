//! CPU smoke train on a tiny random BERT: the real step loss (SupCon + heads +
//! urgent/frustration + OE), the real sampler and the real AdamW must drive
//! the loss down within a few dozen steps.

use std::collections::BTreeMap;

use candle_core::Device;
use ruvector_typesafe_train::config::Config;
use ruvector_typesafe_train::data::{Row, BANKING77, CLINC150, HWU64, TICKETS};
use ruvector_typesafe_train::model::{random_encoder, BertConfig, Heads};
use ruvector_typesafe_train::optim::{lr_scale, AdamW};
use ruvector_typesafe_train::sampler::Sampler;
use ruvector_typesafe_train::train::step::{head_name, step_loss, StepCtx, H_FRUST, H_URGENT};

const VOCAB: usize = 64;

/// Token ids encode the label so the task is learnable: [CLS]=1, label token, noise.
fn toks(label: usize, j: usize) -> Vec<u32> {
    vec![1, 10 + label as u32, 40 + (j % 7) as u32, 2]
}

fn build() -> (Vec<Row>, StepCtx, Config) {
    let mut cfg = Config::v0();
    cfg.batch.public_p = 4;
    cfg.batch.public_k = 2;
    cfg.batch.clinc_oos = 3;
    cfg.batch.tickets_p = 4;
    let mut rows = Vec::new();
    let mut tokens = Vec::new();
    let mut label_index = BTreeMap::new();
    let mut desc_tokens = BTreeMap::new();
    for ds in [TICKETS, BANKING77, CLINC150, HWU64] {
        let mut li = BTreeMap::new();
        for l in 0..4 {
            let name = format!("L{l}");
            li.insert(name.clone(), l);
            desc_tokens.insert((ds.to_string(), name.clone()), vec![1, 10 + l as u32, 2]);
            for j in 0..6 {
                let mut r = Row::new(
                    ds,
                    format!("{ds}-{l}-{j}"),
                    format!("{l} {j}"),
                    name.clone(),
                );
                if ds == TICKETS {
                    r.urgent = Some(l % 2 == 0);
                    r.frustration = Some((l % 3) as u8);
                    if j == 0 {
                        r.soft = Some(
                            [(name.clone(), 0.75), (format!("L{}", (l + 1) % 4), 0.25)].into(),
                        );
                    }
                }
                rows.push(r);
                tokens.push(toks(l, j));
            }
        }
        label_index.insert(ds.to_string(), li);
    }
    for j in 0..5 {
        let mut r = Row::new(
            CLINC150,
            format!("oos-{j}"),
            format!("oos {j}"),
            "oos".into(),
        );
        r.oos = true;
        rows.push(r);
        tokens.push(vec![1, 60, 2]);
    }
    // One label without a description (as after the leakage exclusion).
    desc_tokens.remove(&(TICKETS.to_string(), "L0".to_string()));
    let (urgent_w, frust_w) = StepCtx::class_weights(&rows);
    let ctx = StepCtx {
        rows: rows.clone(),
        tokens,
        desc_tokens,
        label_index,
        urgent_w,
        frust_w,
    };
    (rows, ctx, cfg)
}

#[test]
fn loss_decreases_on_cpu() {
    let dev = Device::Cpu;
    let (rows, ctx, cfg) = build();
    let (vm, bert) = random_encoder(BertConfig::tiny(VOCAB), 11, &dev).unwrap();
    let mut spec: Vec<(String, usize)> = [TICKETS, BANKING77, CLINC150, HWU64]
        .iter()
        .map(|d| (head_name(d).to_string(), 4))
        .collect();
    spec.push((H_URGENT.into(), 2));
    spec.push((H_FRUST.into(), 3));
    let heads = Heads::new(&spec, 32, 20.0, 11, &dev).unwrap();
    let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.01);
    opt.add_group(&vm, 3e-3).unwrap();
    opt.add_group(&heads.vm, 1e-2).unwrap();
    let mut sampler = Sampler::new(&rows, &cfg, 3);
    let steps = 60;
    let mut first = Vec::new();
    let mut last = Vec::new();
    let mut saw = std::collections::BTreeSet::new();
    for s in 0..steps {
        let b = sampler.next_batch();
        let out = step_loss(&bert, &heads, &ctx, &cfg, &b).unwrap();
        for k in out.parts.keys() {
            saw.insert(*k);
        }
        let g = out.loss.backward().unwrap();
        opt.step(&g, lr_scale(s, 5, steps), 1.0).unwrap();
        let t = out.parts["total"];
        assert!(t.is_finite());
        if s < 10 {
            first.push(t);
        } else if s >= steps - 10 {
            last.push(t);
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    assert!(
        mean(&last) < 0.8 * mean(&first),
        "first {} last {}",
        mean(&first),
        mean(&last)
    );
    for k in ["supcon", "head", "urgent", "frustration", "oe", "total"] {
        assert!(saw.contains(k), "loss part {k} never produced");
    }
}
