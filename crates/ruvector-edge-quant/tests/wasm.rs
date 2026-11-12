//! wasm32 runtime checks (ADR-351 M4). Run with
//! `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner
//! cargo test -p ruvector-edge-quant --release --target wasm32-unknown-unknown
//! --test wasm -- --nocapture` (Node). On native targets this file compiles
//! to nothing; `tests/fixtures_native.rs` pins the same fixtures natively.
#![cfg(target_arch = "wasm32")]

mod fixture;

use fixture::*;
use ruvector_edge_quant::persist::{load_frames, load_from, save_frames, MAX_FRAME_BYTES};
use ruvector_edge_quant::*;
use wasm_bindgen_test::{console_log, wasm_bindgen_test};

const FIXTURES: [(&[u8], &str); 2] = [
    (
        include_bytes!("fixtures/hadamard_200x64.rbqx"),
        include_str!("fixtures/hadamard_200x64.hits"),
    ),
    (
        include_bytes!("fixtures/haar_200x64.rbqx"),
        include_str!("fixtures/haar_200x64.hits"),
    ),
];

#[wasm_bindgen_test]
fn native_snapshots_load_and_answer_like_native() {
    for ((stem, kind), (bytes, hits)) in KINDS.into_iter().zip(FIXTURES) {
        // Cold load regenerates the rotation on wasm32 and checks it against
        // the fingerprint written natively.
        let s = load_from(&mut &bytes[..], Budget::default())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(s.config().rotation, kind);
        assert_eq!(answers(&s), hits, "{stem}: answers differ from native");
        // Encoding on wasm32 writes the native bytes exactly.
        assert_eq!(
            snapshot(&build(kind)),
            bytes,
            "{stem}: wasm32 encode differs"
        );
    }
}

/// M4 design point on the edge target: 50k × 384 (default Hadamard) cold
/// load and query under the default budgets; prints wall time per work
/// unit so `max_load_units` / `max_query_units` can be checked for wasm32.
#[wasm_bindgen_test]
fn cold_load_50k_x_384_fits_the_default_budget() {
    const N: u64 = 50_000;
    const D: usize = 384;
    let mut s = QuantShard::new(QuantConfig::new(D, Metric::Cosine, SEED)).unwrap();
    let data = fixture::rows(N, D);
    for chunk in data.chunks(500) {
        let refs: Vec<(u64, &[f32])> = chunk.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        s.upsert(&refs).unwrap();
    }
    let frames = save_frames(&s, MAX_FRAME_BYTES).unwrap();
    let bytes: usize = frames.iter().map(Vec::len).sum();
    drop(s);

    let t = js_sys::Date::now();
    let s = load_frames(frames.iter().map(Vec::as_slice), Budget::default()).unwrap();
    let load_ms = js_sys::Date::now() - t;
    let units = budget::load_units(bytes as u64, N, D, RandomRotationKind::HadamardSigned);
    assert!(units <= Budget::default().max_load_units);
    assert!(s.resident_bytes() <= Budget::default().max_resident_bytes);

    let mut src = Rows(data.into_iter().collect());
    let t = js_sys::Date::now();
    let mut q_units = 0;
    for j in 0..20 {
        let q = vector(90_000 + j, D);
        let out = s
            .query(&q, &QueryOptions::default(), Some(&mut src))
            .unwrap();
        assert_eq!(out.hits.len(), 10);
        q_units = out.stats.units;
    }
    let query_ms = (js_sys::Date::now() - t) / 20.0;
    console_log!(
        "wasm32 50k x 384: cold load {load_ms:.1} ms ({units} units, {:.2} ns/unit), \
         query {query_ms:.2} ms ({q_units} units, {:.2} ns/unit), resident {} B",
        load_ms * 1e6 / units as f64,
        query_ms * 1e6 / q_units as f64,
        s.resident_bytes()
    );
}

#[wasm_bindgen_test]
fn budget_exhaustion_is_413_on_wasm() {
    let b = Budget {
        max_load_units: 1,
        ..Budget::default()
    };
    let e = load_from(&mut &FIXTURES[0].0[..], b).err().unwrap();
    assert_eq!(e.status(), 413);
    let mut s = build(RandomRotationKind::HadamardSigned);
    s.set_budget(Budget {
        max_query_units: 1,
        ..Budget::default()
    });
    let e = s.query(&vector(1, DIM), &QueryOptions::default(), None);
    assert_eq!(e.unwrap_err().status(), 413);
}
