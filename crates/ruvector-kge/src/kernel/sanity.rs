//! WN18RR sanity run of the ComplEx-N3-R recipe on the GEMM kernel (plan M2
//! integration). `#[ignore]`d and env-gated: it reads the canonical
//! commit-pinned files the npm bench caches, **train and valid only** (test is
//! never opened), and reports filtered valid MRR under BOTTOM and RANDOM ties.
//!
//! ```text
//! KGE_WN18RR_DIR=npm/packages/kge/bench/.cache KGE_EPOCHS=8 KGE_RANK=500 \
//! CARGO_TARGET_DIR=... cargo test --release -p ruvector-kge --features parallel \
//!     --lib kernel::sanity -- --ignored --nocapture
//! ```
//!
//! Reciprocal evaluation (the recipe trains tail queries only) goes through
//! `EvalConfig::reciprocal`: each valid triple `(s, r, o)` is ranked as the
//! tail query `(s, r, ?)` and its head query as `(o, r⁻¹, ?)`. Filtering uses train +
//! valid (no test), so it is slightly pessimistic against the usual
//! train + valid + test filter. The entity count is padded to the canonical
//! 40,943 (entities seen only in test are untrained candidate rows).

use crate::data::{TripleStore, Vocab};
use crate::eval::{evaluate_rank_pair_with, EvalConfig};
use crate::scorer::ComplEx;
use crate::train::{TrainConfig, Trainer};
use crate::{Tables, Triple};
use std::time::Instant;

const WN18RR_ENTITIES: usize = 40_943;
const WN18RR_RELATIONS: usize = 11;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn read(vocab: &mut Vocab, path: &std::path::Path) -> Vec<Triple> {
    let text = std::fs::read_to_string(path).expect("read split");
    text.lines()
        .map(|l| l.trim_end_matches('\r'))
        .filter(|l| !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 3, "bad line {l:?}");
            let s = vocab.intern_entity(f[0]).unwrap();
            let r = vocab.intern_relation(f[1]).unwrap();
            let o = vocab.intern_entity(f[2]).unwrap();
            Triple::new(s, r, o)
        })
        .collect()
}

#[test]
#[ignore]
fn sanity_wn18rr_valid_complex_n3_r() {
    let Ok(dir) = std::env::var("KGE_WN18RR_DIR") else {
        println!("KGE_WN18RR_DIR unset; skipping");
        return;
    };
    let dir = std::path::Path::new(&dir);
    let (epochs, rank) = (env_usize("KGE_EPOCHS", 8), env_usize("KGE_RANK", 500));
    let (threads, batch) = (env_usize("KGE_THREADS", 16), env_usize("KGE_BATCH", 1000));
    let n3 = std::env::var("KGE_N3")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1f32);

    let mut vocab = Vocab::new();
    let train = read(&mut vocab, &dir.join("wn18rr@2e440e0f-train.txt"));
    let valid = read(&mut vocab, &dir.join("wn18rr@2e440e0f-valid.txt"));
    assert_eq!((train.len(), valid.len()), (86_835, 3_034));
    assert_eq!(vocab.num_relations(), WN18RR_RELATIONS);
    assert!(vocab.num_entities() <= WN18RR_ENTITIES);
    let (ne, nr, d) = (WN18RR_ENTITIES, WN18RR_RELATIONS, 2 * rank);
    let store = TripleStore::with_counts(train.clone(), Some(ne), Some(nr)).unwrap();

    let cfg: TrainConfig = serde_json::from_str(&format!(
        r#"{{"loss":{{"kind":"one_vs_all"}},"reciprocal":true,"n3_form":"moduli",
            "loss_reduction":"mean","init":{{"kind":"normal","scale":0.001}},
            "optimizer":{{"kind":"adagrad"}},"optim_state":"dense","n3_lambda":{n3},
            "rp_weight":0.05,"dims":{d},"epochs":{epochs},"batch_size":{batch},
            "lr":0.1,"seed":1,"one_n_kernel":"gemm"}}"#
    ))
    .unwrap();
    println!(
        "SANITY wn18rr: seen entities {} (padded to {ne}), train {}, valid {}, cfg {}",
        vocab.num_entities(),
        train.len(),
        valid.len(),
        serde_json::to_string(&cfg).unwrap()
    );
    let sc = ComplEx::new(d).unwrap();
    let mut tables = Tables::new(ne, 2 * nr, d, 7);
    let start = Instant::now();
    pool(threads, || {
        Trainer::fit(&mut tables, &sc, &store, &cfg, |p| {
            println!(
                "SANITY epoch {}/{} loss {:.4} n3 {:.5} rp {:.4} wall {:.1}s",
                p.epoch + 1,
                p.epochs,
                p.loss,
                p.n3_penalty,
                p.rp_loss,
                start.elapsed().as_secs_f64()
            );
        })
        .unwrap()
    });
    let train_s = start.elapsed().as_secs_f64();

    let mut all = train.clone();
    all.extend_from_slice(&valid);
    // Base-id filter store: reciprocal eval maps the head query onto
    // (o, r⁻¹, ?) itself (eval/counts.rs), batched on the GEMM kernel.
    let filter = TripleStore::with_counts(all, Some(ne), Some(nr)).unwrap();
    let ecfg = EvalConfig::random(42).with_reciprocal(true);
    let eval_start = Instant::now();
    let pair = pool(threads, || {
        evaluate_rank_pair_with(&tables, &sc, &filter, &valid, &ecfg).unwrap()
    });
    let eval_s = eval_start.elapsed().as_secs_f64();
    // Pair layout is [t0, h0, t1, h1, ..]: tails first, then heads.
    fn sides(v: &[usize]) -> Vec<usize> {
        let side = |k: usize| v.iter().skip(k).step_by(2).copied();
        side(0).chain(side(1)).collect()
    }
    let (bottom, random) = (sides(&pair.bottom), sides(&pair.random));
    assert_eq!(bottom.len(), 2 * valid.len());
    let mrr = |r: &[usize]| r.iter().map(|&k| 1.0 / k as f64).sum::<f64>() / r.len() as f64;
    let hits =
        |r: &[usize], k: usize| r.iter().filter(|&&x| x <= k).count() as f64 / r.len() as f64;
    let half = valid.len();
    println!(
        "SANITY wn18rr VALID filtered(train+valid) MRR bottom {:.4} random {:.4} | \
         tail {:.4}/{:.4} head {:.4}/{:.4} | H@1 {:.4} H@10 {:.4} (random) | \
         train {:.1}s eval {:.1}s threads {threads}",
        mrr(&bottom),
        mrr(&random),
        mrr(&bottom[..half]),
        mrr(&random[..half]),
        mrr(&bottom[half..]),
        mrr(&random[half..]),
        hits(&random, 1),
        hits(&random, 10),
        train_s,
        eval_s
    );
}

fn pool<R: Send>(t: usize, f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "parallel")]
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
            .install(f)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = t;
        f()
    }
}
