//! CI recall gate (not ignored): the *default* configuration
//! (`QuantConfig::new` → Hadamard, `QueryOptions::default()` → factor 50)
//! must reach recall@10 ≥ 0.95 against exact search for every metric, at
//! one fixed factor. `tests/scale.rs` repeats the gate at the 50k × 384
//! design point in release.

mod common;

use common::*;
use ruvector_edge_quant::*;

const N: usize = 10_000;
const DIM: usize = 384;
const Q: usize = 50;
const K: usize = 10;

#[test]
fn default_config_recall_at_10_is_at_least_095_for_every_metric() {
    let (data, qs) = clustered(N, Q, DIM, 100, 0.5, 7);
    for metric in [Metric::Cosine, Metric::L2, Metric::Dot] {
        let cfg = QuantConfig::new(DIM, metric, 0x5EED);
        assert_eq!(cfg.rotation, RandomRotationKind::HadamardSigned);
        let mut s = QuantShard::new(cfg).unwrap();
        s.upsert(&rows(&data)).unwrap();
        let mut src = MemRows::from_data(&data);
        let opts = QueryOptions::default();
        let mut truth = Vec::with_capacity(Q);
        let mut got = Vec::with_capacity(Q);
        for q in &qs {
            truth.push(exact_topk(&data, q, K, metric));
            let out = s.query(q, &opts, Some(&mut src)).unwrap();
            assert!(!out.stats.clamped);
            got.push(out.hits.iter().map(|h| h.key).collect::<Vec<_>>());
        }
        let r = recall(&truth, &got);
        assert!(r >= 0.95, "{metric:?}: recall@{K} {r:.4} < 0.95");
    }
}
