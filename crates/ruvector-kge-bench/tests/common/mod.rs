//! Synthetic dataset + configs shared by the integration tests.
#![allow(dead_code)]

use ruvector_kge_bench::config::{Recipe, RunConfig};
use ruvector_kge_bench::datasets::{from_bytes, Dataset, DatasetSpec, Expected};
use std::collections::BTreeSet;

/// A deterministic synthetic KG: `ne` entities, 3 relations with structure
/// (r0: e→e+1, r1: e→e+7, r2: e→2e), split disjointly into train/valid/test.
pub fn synth_dataset() -> Dataset {
    let ne = 48u32;
    let mut all = BTreeSet::new();
    for e in 0..ne {
        all.insert((e, 0, (e + 1) % ne));
        all.insert((e, 1, (e + 7) % ne));
        all.insert((e, 2, (2 * e) % ne));
    }
    let (mut tr, mut va, mut te) = (String::new(), String::new(), String::new());
    let (mut ntr, mut nva, mut nte) = (0, 0, 0);
    for (i, (s, r, o)) in all.iter().enumerate() {
        let line = format!("ent{s:03}\trel{r}\tent{o:03}\n");
        match i % 10 {
            0 => {
                va.push_str(&line);
                nva += 1;
            }
            1 => {
                te.push_str(&line);
                nte += 1;
            }
            _ => {
                tr.push_str(&line);
                ntr += 1;
            }
        }
    }
    let spec: &'static DatasetSpec = Box::leak(Box::new(DatasetSpec {
        name: "synth",
        commit: "0000000000000000000000000000000000000000",
        cache_ext: ".txt",
        urls: ["u/train.txt", "u/valid.txt", "u/test.txt"],
        pins: ["", "", ""],
        expected: Expected {
            train: ntr,
            valid: nva,
            test: nte,
            entities: ne as usize,
            relations: 3,
        },
        licence: "synthetic",
    }));
    from_bytes(spec, [tr.as_bytes(), va.as_bytes(), te.as_bytes()]).unwrap()
}

pub fn recipe() -> Recipe {
    Recipe {
        complex_rank: 6,
        batch_size: 16,
        lr: 0.1,
        n3_lambda: 0.01,
        rp_weight: 0.05,
    }
}

/// A custom-recipe run on the synthetic dataset.
pub fn synth_run(max_epochs: usize) -> RunConfig {
    RunConfig {
        dataset: "synth".into(),
        config_id: "custom".into(),
        recipe: Some(recipe()),
        seed: 3,
        max_epochs,
        early_stop_patience: None,
        eval_every: 1,
        threads: Some(2),
    }
}
