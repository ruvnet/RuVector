//! Regression tests for inputs a JS caller controls: router/adapter JSON,
//! KV-cache bounds and SONA configuration.
//!
//! Native: `cargo test -p ruvllm-wasm --test input_hardening`
//! WASM:   `wasm-pack test --node crates/ruvllm-wasm -- --test input_hardening`
//!
//! Cases that build a `JsValue` (error paths) or call `performance.now()` (SONA)
//! only run on wasm32.

use ruvllm_wasm::micro_lora::{MicroLoraConfigWasm, MicroLoraWasm};
use ruvllm_wasm::{HnswRouterWasm, KvCacheConfigWasm, KvCacheWasm};
use wasm_bindgen_test::wasm_bindgen_test;

/// Deterministic pseudo-random unit-ish vectors (no RNG dependency).
fn embedding(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn hnsw_finds_every_inserted_pattern() {
    let dim = 32;
    let n = 300;
    let mut router = HnswRouterWasm::new(dim, n);
    for i in 0..n {
        assert!(router.add_pattern(&embedding(i as u64, dim), &format!("p{i}"), "{}"));
    }
    let mut hits = 0;
    for i in 0..n {
        let res = router.route(&embedding(i as u64, dim), 1);
        if res.first().map(|r| r.name()) == Some(format!("p{i}")) {
            hits += 1;
        }
    }
    // An exact-match query must find itself; anything below ~95% means part of
    // the graph is unreachable from the entry point.
    assert!(hits * 100 >= n * 95, "self-recall {hits}/{n}");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn hnsw_zero_ef_and_zero_k_do_not_trap() {
    let dim = 8;
    let mut router = HnswRouterWasm::new(dim, 64);
    for i in 0..40 {
        router.add_pattern(&embedding(i, dim), &format!("p{i}"), "{}");
    }
    router.set_ef_search(0);
    assert!(router.route(&embedding(3, dim), 0).is_empty());
    assert_eq!(router.route(&embedding(3, dim), 1).len(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn hnsw_non_finite_embeddings_do_not_trap() {
    let dim = 4;
    let mut router = HnswRouterWasm::new(dim, 64);
    for i in 0..30 {
        let mut e = embedding(i, dim);
        if i % 3 == 0 {
            e[1] = f32::NAN;
        }
        if i % 5 == 0 {
            e[2] = f32::INFINITY;
        }
        router.add_pattern(&e, &format!("p{i}"), "{}");
    }
    let _ = router.route(&[f32::NAN; 4], 5);
    assert!(!router.route(&embedding(1, dim), 5).is_empty());
}

#[cfg(target_arch = "wasm32")] // error paths build a JsValue
#[wasm_bindgen_test]
fn hnsw_from_json_rejects_inconsistent_graph() {
    let mut router = HnswRouterWasm::new(4, 16);
    for i in 0..6 {
        router.add_pattern(&embedding(i, 4), &format!("p{i}"), "{}");
    }
    let good = router.to_json().unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&good).unwrap();

    // max_layer beyond the stored layers
    let mut bad = v.clone();
    bad["graph"]["max_layer"] = serde_json::json!(9);
    assert!(HnswRouterWasm::from_json(&bad.to_string()).is_err());

    // entry point past the pattern list
    let mut bad = v.clone();
    bad["graph"]["entry_point"] = serde_json::json!(1000);
    assert!(HnswRouterWasm::from_json(&bad.to_string()).is_err());

    // stored embedding shorter than the declared dimensions
    let mut bad = v.clone();
    bad["dimensions"] = serde_json::json!(5);
    assert!(HnswRouterWasm::from_json(&bad.to_string()).is_err());

    // neighbour index past the pattern list
    let layer0 = v["graph"]["layers"][0].as_object_mut().unwrap();
    let (_, node) = layer0.iter_mut().next().unwrap();
    node["neighbors"] = serde_json::json!([4096]);
    assert!(HnswRouterWasm::from_json(&v.to_string()).is_err());

    assert!(HnswRouterWasm::from_json(&good).is_ok());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn kv_cache_never_exceeds_max_tokens() {
    let mut cfg = KvCacheConfigWasm::new();
    cfg.set_num_kv_heads(1);
    cfg.set_head_dim(2);
    cfg.set_tail_length(16);
    cfg.set_max_tokens(4);
    let cache = KvCacheWasm::new(&cfg);
    for _ in 0..10 {
        cache.append(&[0.1, 0.2], &[0.3, 0.4]).unwrap();
        assert!(cache.token_count() <= 4, "tokens={}", cache.token_count());
    }
    assert_eq!(cache.token_count(), 4);
}

#[cfg(target_arch = "wasm32")] // error paths build a JsValue
#[wasm_bindgen_test]
fn micro_lora_from_json_rejects_mismatched_weights() {
    let mut cfg = MicroLoraConfigWasm::new();
    cfg.set_in_features(8);
    cfg.set_out_features(8);
    let lora = MicroLoraWasm::new(&cfg);
    let good = lora.to_json().unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&good).unwrap();
    v["adapter"]["in_features"] = serde_json::json!(4096);
    assert!(MicroLoraWasm::from_json(&v.to_string()).is_err());
    let mut v: serde_json::Value = serde_json::from_str(&good).unwrap();
    v["adapter"]["lora_b"] = serde_json::json!([0.0]);
    assert!(MicroLoraWasm::from_json(&v.to_string()).is_err());
    let restored = MicroLoraWasm::from_json(&good).unwrap();
    assert_eq!(restored.apply(&[0.5; 8]).unwrap().len(), 8);
}

#[cfg(target_arch = "wasm32")]
mod sona {
    use ruvllm_wasm::{SonaConfigWasm, SonaInstantWasm};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn zero_hidden_dim_does_not_trap() {
        let mut cfg = SonaConfigWasm::new();
        cfg.set_hidden_dim(0);
        let mut sona = SonaInstantWasm::new(cfg);
        assert!(sona.instant_adapt(0.9).applied());
    }

    #[wasm_bindgen_test]
    fn non_finite_quality_is_ignored() {
        let mut sona = SonaInstantWasm::new(SonaConfigWasm::new());
        assert!(!sona.instant_adapt(f32::NAN).applied());
        assert!(!sona.instant_adapt(f32::INFINITY).applied());
        assert!(sona.stats().avg_quality().is_finite());
    }

    #[wasm_bindgen_test]
    fn from_json_clamps_config() {
        let cfg_json = r#"{"hidden_dim":0,"micro_lora_rank":99,"learning_rate":5.0,
            "ema_decay":7.0,"pattern_capacity":4294967295,"ewc_lambda":0.1,
            "quality_threshold":0.5}"#;
        let cfg = SonaConfigWasm::from_json(cfg_json).unwrap();
        assert!(cfg.pattern_capacity() <= 1000 && cfg.hidden_dim() >= 1);
        assert!(cfg.micro_lora_rank() <= 4 && cfg.ema_decay() <= 1.0);

        let json = format!(
            r#"{{"config":{cfg_json},"quality_ema":0.5,"adaptations":0,
            "patterns_recorded":0,"successful_patterns":0,"current_rank":99}}"#
        );
        let mut sona = SonaInstantWasm::from_json(&json).unwrap();
        assert!(sona.stats().current_rank() <= 4);
        let r = sona.instant_adapt(0.9);
        assert!(r.quality_ema().is_finite() && r.quality_ema() <= 1.0);
    }
}
