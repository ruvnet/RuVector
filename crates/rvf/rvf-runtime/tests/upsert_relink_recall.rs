//! In-place upsert vs delete-then-insert recall on the same churn
//! (rvf:upsert-relink-recall).
//!
//! Mirrors agentdb's paired test (tests/backends/rvf-049-upsert-lock.test.ts):
//! 3,000 x 384 cosine, 300 updates of ids `(k * 37) % N`, 200 queries,
//! recall@10 against the exact top-10 of the final vectors. The reinsert arm
//! does what the TypeScript SDK does on `delete` + `ingest`: tombstone the old
//! label and ingest under a fresh one, so the old node stays in the graph.
//!
//! rvf 0.4.9 measured upsert 0.966 vs reinsert 0.9985 (seed 12345) through the
//! SDK. The relink must now keep upsert within 0.005 of reinsert.

use std::collections::HashSet;

use rvf_runtime::options::{DistanceMetric, QueryOptions, RvfOptions};
use rvf_runtime::RvfStore;
use tempfile::TempDir;

const D: usize = 384;

/// agentdb's JS generator, emulated exactly: `(seed * 1103515245 + 12345) >>> 0`
/// runs in f64, which loses precision past 2^53, so the sequence cycles after
/// a few thousand draws and the dataset holds many exact duplicate vectors
/// (identical texts embed identically, so this is a realistic workload).
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> f32 {
        let x = (self.0 as f64) * 1103515245.0 + 12345.0;
        self.0 = x.rem_euclid(4294967296.0) as u32;
        (self.0 as f64 / 4294967296.0 - 0.5) as f32
    }
    fn unit(&mut self) -> Vec<f32> {
        let mut v: Vec<f32> = (0..D).map(|_| self.next()).collect();
        let s = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= s);
        v
    }
}

fn store(dir: &TempDir, name: &str) -> RvfStore {
    RvfStore::create(
        &dir.path().join(name),
        RvfOptions {
            dimension: D as u16,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        },
    )
    .unwrap()
}

fn fill(s: &mut RvfStore, base: &[Vec<f32>]) {
    for start in (0..base.len()).step_by(500) {
        let end = (start + 500).min(base.len());
        let refs: Vec<&[f32]> = base[start..end].iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (start as u64 + 1..=end as u64).collect();
        s.ingest_batch(&refs, &ids, None).unwrap();
    }
}

/// Returns (recall upsert, recall reinsert).
fn paired(seed: u32, n: usize, churn: usize, q: usize) -> (f64, f64) {
    let mut rng = Lcg(seed);
    let base: Vec<Vec<f32>> = (0..n).map(|_| rng.unit()).collect();
    let updates: Vec<(usize, Vec<f32>)> = (0..churn).map(|k| ((k * 37) % n, rng.unit())).collect();
    let queries: Vec<Vec<f32>> = (0..q).map(|_| rng.unit()).collect();

    let dir = TempDir::new().unwrap();
    let mut up = store(&dir, "upsert.rvf");
    let mut re = store(&dir, "reinsert.rvf");
    fill(&mut up, &base);
    fill(&mut re, &base);
    let opts = QueryOptions::default();
    up.query(&queries[0], 10, &opts).unwrap();
    re.query(&queries[0], 10, &opts).unwrap();

    // label of each logical id in the reinsert store (SDK allocates a new one).
    let mut re_label: Vec<u64> = (1..=n as u64).collect();
    let mut latest = base.clone();
    for (next_label, (idx, v)) in (n as u64 + 1..).zip(updates.iter()) {
        up.ingest_batch(&[v.as_slice()], &[*idx as u64 + 1], None)
            .unwrap();
        re.delete(&[re_label[*idx]]).unwrap();
        re_label[*idx] = next_label;
        re.ingest_batch(&[v.as_slice()], &[next_label], None)
            .unwrap();
        latest[*idx] = v.clone();
    }
    let mut fresh = store(&dir, "fresh.rvf");
    fill(&mut fresh, &latest);
    fresh.query(&queries[0], 10, &opts).unwrap();
    let logical_of_re: std::collections::HashMap<u64, usize> =
        re_label.iter().enumerate().map(|(i, &l)| (l, i)).collect();

    let (mut hit_up, mut hit_re, mut hit_fr) = (0usize, 0usize, 0usize);
    for qv in &queries {
        let mut scored: Vec<(usize, f32)> = latest
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.iter().zip(qv).map(|(a, b)| a * b).sum::<f32>()))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let truth: HashSet<usize> = scored[..10].iter().map(|&(i, _)| i).collect();

        let got_up: Vec<usize> = up
            .query(qv, 10, &opts)
            .unwrap()
            .iter()
            .map(|r| r.id as usize - 1)
            .collect();
        let got_re: Vec<usize> = re
            .query(qv, 10, &opts)
            .unwrap()
            .iter()
            .map(|r| logical_of_re[&r.id])
            .collect();
        assert_eq!(got_up.iter().collect::<HashSet<_>>().len(), got_up.len());
        hit_up += got_up.iter().filter(|i| truth.contains(i)).count();
        hit_re += got_re.iter().filter(|i| truth.contains(i)).count();
        hit_fr += fresh
            .query(qv, 10, &opts)
            .unwrap()
            .iter()
            .filter(|r| truth.contains(&(r.id as usize - 1)))
            .count();
    }
    let total = (q * 10) as f64;
    eprintln!("  fresh-build recall={:.4}", hit_fr as f64 / total);
    (hit_up as f64 / total, hit_re as f64 / total)
}

/// The full agentdb-sized measurement (slow in debug; run with --release).
#[test]
#[ignore]
fn upsert_recall_matches_reinsert_3000x384() {
    // Seed 4242 cycles after ~400 draws: ~400 distinct vectors, so every arm
    // (fresh build included) is tie-limited near 0.4. Reported, not asserted.
    for seed in [12345u32, 777, 4242] {
        let (up, re) = paired(seed, 3000, 300, 200);
        eprintln!("seed={seed} recall@10 upsert={up:.4} reinsert={re:.4}");
        if seed == 4242 {
            continue;
        }
        assert!(
            up >= re - 0.005,
            "seed {seed}: upsert {up} vs reinsert {re}"
        );
    }
}

/// A smaller paired run that fits the default test budget. N must exceed the
/// generator period (~2,021 distinct vectors for seed 12345) so the store
/// holds duplicates; below it both arms read 1.0 and the test cannot fail.
#[test]
fn upsert_recall_matches_reinsert_small() {
    let (up, re) = paired(12345, 2600, 300, 80);
    eprintln!("small recall@10 upsert={up:.4} reinsert={re:.4}");
    assert!(up >= re - 0.01, "upsert {up} vs reinsert {re}");
}
