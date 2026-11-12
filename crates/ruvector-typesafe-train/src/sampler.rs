//! Seeded class-balanced P×K sampler (plan Step 3 "batch shapes").
//!
//! One dataset per step, drawn by mix weight; then P labels × K examples, the
//! P label-description texts (extra SupCon positives), and on CLINC150 steps
//! `clinc_oos` rows from `oos_train`. All randomness is a ChaCha8 stream from
//! the run seed, so a (seed, data, config) triple fixes the batch sequence.

use std::collections::BTreeMap;

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::config::Config;
use crate::data::{Row, CLINC150};

pub struct DatasetIndex {
    pub name: &'static str,
    /// label → train row indices (in-scope rows only).
    pub by_label: BTreeMap<String, Vec<usize>>,
    pub oos: Vec<usize>,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepBatch {
    pub dataset: &'static str,
    /// (row index, label) for the P×K examples.
    pub rows: Vec<(usize, String)>,
    /// Labels whose description text is appended (one per sampled label).
    pub desc_labels: Vec<String>,
    pub oos_rows: Vec<usize>,
}

pub struct Sampler {
    pub datasets: Vec<DatasetIndex>,
    rng: ChaCha8Rng,
    p: Vec<(usize, usize)>,
    clinc_oos: usize,
}

impl Sampler {
    pub fn new(train: &[Row], cfg: &Config, seed: u64) -> Self {
        let mut datasets = Vec::new();
        let mut p = Vec::new();
        for name in cfg.active() {
            let mut by_label: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            let mut oos = Vec::new();
            for (i, r) in train.iter().enumerate().filter(|(_, r)| r.dataset == name) {
                if r.oos {
                    oos.push(i);
                } else {
                    by_label.entry(r.label.clone()).or_default().push(i);
                }
            }
            if by_label.is_empty() {
                continue;
            }
            p.push(cfg.pk(name));
            datasets.push(DatasetIndex {
                name,
                by_label,
                oos,
                weight: cfg.mix[name],
            });
        }
        Self {
            datasets,
            rng: ChaCha8Rng::seed_from_u64(seed),
            p,
            clinc_oos: cfg.batch.clinc_oos,
        }
    }

    pub fn next_batch(&mut self) -> StepBatch {
        let total: f64 = self.datasets.iter().map(|d| d.weight).sum();
        let mut x = self.rng.gen::<f64>() * total;
        let mut pick = self.datasets.len() - 1;
        for (i, d) in self.datasets.iter().enumerate() {
            if x < d.weight {
                pick = i;
                break;
            }
            x -= d.weight;
        }
        let (p, k) = self.p[pick];
        let d = &self.datasets[pick];
        let labels: Vec<&String> = d.by_label.keys().collect();
        let chosen: Vec<String> = labels
            .choose_multiple(&mut self.rng, p.min(labels.len()))
            .map(|s| (*s).clone())
            .collect();
        let mut rows = Vec::with_capacity(chosen.len() * k);
        for l in &chosen {
            let pool = &d.by_label[l];
            if pool.len() >= k {
                for &i in pool.choose_multiple(&mut self.rng, k) {
                    rows.push((i, l.clone()));
                }
            } else {
                for _ in 0..k {
                    rows.push((pool[self.rng.gen_range(0..pool.len())], l.clone()));
                }
            }
        }
        let oos_rows = if d.name == CLINC150 && !d.oos.is_empty() {
            d.oos
                .choose_multiple(&mut self.rng, self.clinc_oos.min(d.oos.len()))
                .copied()
                .collect()
        } else {
            Vec::new()
        };
        StepBatch {
            dataset: d.name,
            rows,
            desc_labels: chosen,
            oos_rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BANKING77, HWU64, TICKETS};

    fn rows() -> Vec<Row> {
        let mut v = Vec::new();
        for (ds, n_labels) in [(TICKETS, 8), (BANKING77, 40), (CLINC150, 40), (HWU64, 40)] {
            for l in 0..n_labels {
                for j in 0..5 {
                    v.push(Row::new(
                        ds,
                        format!("{ds}-{l}-{j}"),
                        format!("t {l} {j}"),
                        format!("L{l}"),
                    ));
                }
            }
        }
        for j in 0..30 {
            let mut r = Row::new(CLINC150, format!("oos-{j}"), format!("o {j}"), "oos".into());
            r.oos = true;
            v.push(r);
        }
        v
    }

    #[test]
    fn same_seed_same_sequence_different_seed_differs() {
        let (r, c) = (rows(), Config::v0());
        let mut a = Sampler::new(&r, &c, 7);
        let mut b = Sampler::new(&r, &c, 7);
        let mut z = Sampler::new(&r, &c, 8);
        let sa: Vec<_> = (0..20).map(|_| a.next_batch()).collect();
        let sb: Vec<_> = (0..20).map(|_| b.next_batch()).collect();
        let sz: Vec<_> = (0..20).map(|_| z.next_batch()).collect();
        assert_eq!(sa, sb);
        assert_ne!(sa, sz);
    }

    #[test]
    fn batch_shapes_follow_the_plan() {
        let (r, c) = (rows(), Config::v0());
        let mut s = Sampler::new(&r, &c, 1);
        let mut seen = BTreeMap::new();
        for _ in 0..400 {
            let b = s.next_batch();
            let (p, k) = c.pk(b.dataset);
            assert_eq!(b.rows.len(), p * k, "{}", b.dataset);
            assert_eq!(b.desc_labels.len(), p);
            assert_eq!(b.oos_rows.len(), if b.dataset == CLINC150 { 16 } else { 0 });
            // every sampled row carries its own label and dataset
            for (i, l) in &b.rows {
                assert_eq!(&r[*i].label, l);
                assert_eq!(r[*i].dataset, b.dataset);
                assert!(!r[*i].oos);
            }
            *seen.entry(b.dataset).or_insert(0) += 1;
        }
        // mix 0.20/0.25/0.30/0.25 over 400 draws
        assert!(seen[CLINC150] > seen[TICKETS], "{seen:?}");
        assert_eq!(seen.len(), 4);
    }
}
