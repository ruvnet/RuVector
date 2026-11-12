//! Deterministic, transcendental-free fixture data shared by the native
//! fixture check (`tests/fixtures_native.rs`) and the wasm32 runtime test
//! (`tests/wasm.rs`): the same rows are generated bit-identically on every
//! target, so a snapshot written natively can be compared byte for byte
//! with one written on wasm32.
#![allow(dead_code)]

use ruvector_edge_quant::persist::{save_to, MAX_FRAME_BYTES};
use ruvector_edge_quant::*;
use std::collections::HashMap;
use std::fmt::Write as _;

/// Fixture dimension (one full code word).
pub const DIM: usize = 64;
/// Fixture rows.
pub const N: u64 = 200;
/// Queries answered per fixture.
pub const QUERIES: u64 = 5;
/// Rotation seed.
pub const SEED: u64 = 0x5EED;
/// `(file stem, rotation)` of each checked-in fixture.
pub const KINDS: [(&str, RandomRotationKind); 2] = [
    ("hadamard_200x64", RandomRotationKind::HadamardSigned),
    ("haar_200x64", RandomRotationKind::HaarDense),
];

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Components are `k / 2^24 · 2 − 1` for integer `k < 2^24`: exact in f32.
pub fn vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dim)
        .map(|_| (xorshift(&mut s) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)
        .collect()
}

/// Row `i` has key `1000 + 3i` (keys are not positions).
pub fn rows(n: u64, dim: usize) -> Vec<(u64, Vec<f32>)> {
    (0..n).map(|i| (1_000 + 3 * i, vector(i, dim))).collect()
}

/// Build the fixture shard natively or on wasm32.
pub fn build(kind: RandomRotationKind) -> QuantShard {
    let mut cfg = QuantConfig::new(DIM, Metric::L2, SEED);
    cfg.rotation = kind;
    let mut s = QuantShard::new(cfg).unwrap();
    let rows = rows(N, DIM);
    let refs: Vec<(u64, &[f32])> = rows.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    s.upsert(&refs).unwrap();
    s
}

/// The shard as one v2 byte stream.
pub fn snapshot(s: &QuantShard) -> Vec<u8> {
    let mut out = Vec::new();
    save_to(s, &mut out, MAX_FRAME_BYTES).unwrap();
    out
}

/// f32 originals keyed like the store.
pub struct Rows(pub HashMap<u64, Vec<f32>>);

impl Rows {
    pub fn new(n: u64, dim: usize) -> Self {
        Rows(rows(n, dim).into_iter().collect())
    }
}

impl RerankSource for Rows {
    fn fetch(&mut self, keys: &[u64]) -> std::result::Result<Vec<Option<Vec<f32>>>, String> {
        Ok(keys.iter().map(|k| self.0.get(k).cloned()).collect())
    }
}

/// Top-5 per query, estimate-only and reranked, as `key:distance-bits`
/// lines (the pinned native answers).
pub fn answers(s: &QuantShard) -> String {
    let mut src = Rows::new(N, DIM);
    let mut out = String::new();
    for j in 0..QUERIES {
        let q = vector(10_000 + j, DIM);
        for rerank in [false, true] {
            let opts = QueryOptions {
                top_k: 5,
                rerank_factor: 4,
            };
            let hits = if rerank {
                s.query(&q, &opts, Some(&mut src)).unwrap().hits
            } else {
                s.query(&q, &opts, None).unwrap().hits
            };
            for h in hits {
                write!(out, "{}:{:08x} ", h.key, h.distance.to_bits()).unwrap();
            }
            out.push('\n');
        }
    }
    out
}
