//! Regression tests for defects found in the 2026-11-04 review of the
//! `@ruvector/tiny-dancer` NAPI surface. Each one used to panic (aborting the
//! Node process when reached from the sync `new Router()` constructor) or let
//! caller input degrade the router for every other caller.

use chrono::Utc;
use ruvector_tiny_dancer_core::circuit_breaker::{CircuitBreaker, CircuitState};
use ruvector_tiny_dancer_core::{
    Candidate, FastGRNN, FastGRNNConfig, Router, RouterConfig, RoutingRequest,
};
use safetensors::{tensor::TensorView, Dtype};
use std::collections::HashMap;
use std::time::Duration;

/// Write a safetensors file holding `tensors` (name, dtype, shape, bytes) and
/// the given model config JSON.
fn write_model(
    path: &std::path::Path,
    config_json: &str,
    tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)],
) {
    let mut views = HashMap::new();
    for (name, dtype, shape, bytes) in tensors {
        views.insert(
            (*name).to_string(),
            TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
        );
    }
    let mut meta = HashMap::new();
    meta.insert("config".to_string(), config_json.to_string());
    let bytes = safetensors::serialize(&views, &Some(meta)).unwrap();
    std::fs::write(path, bytes).unwrap();
}

fn f32_bytes(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| 0.01f32.to_le_bytes()).collect()
}

fn config_json(input: usize, hidden: usize, output: usize) -> String {
    format!(
        r#"{{"input_dim":{input},"hidden_dim":{hidden},"output_dim":{output},"nu":1.0,"zeta":1.0,"rank":null}}"#
    )
}

/// Full, well-formed tensor set for the given dims, with `w_reset` replaced by
/// `w_reset_override` when provided.
fn tensors(
    i: usize,
    h: usize,
    o: usize,
    w_reset_override: Option<(Dtype, Vec<usize>, Vec<u8>)>,
) -> Vec<(&'static str, Dtype, Vec<usize>, Vec<u8>)> {
    let (rd, rs, rb) = w_reset_override.unwrap_or((Dtype::F32, vec![h, i], f32_bytes(h * i)));
    vec![
        ("w_reset", rd, rs, rb),
        ("w_update", Dtype::F32, vec![h, i], f32_bytes(h * i)),
        ("w_candidate", Dtype::F32, vec![h, i], f32_bytes(h * i)),
        ("w_recurrent", Dtype::F32, vec![h, h], f32_bytes(h * h)),
        ("w_output", Dtype::F32, vec![o, h], f32_bytes(o * h)),
        ("b_reset", Dtype::F32, vec![h], f32_bytes(h)),
        ("b_update", Dtype::F32, vec![h], f32_bytes(h)),
        ("b_candidate", Dtype::F32, vec![h], f32_bytes(h)),
        ("b_output", Dtype::F32, vec![o], f32_bytes(o)),
    ]
}

#[test]
fn load_rejects_non_f32_tensor_instead_of_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad-dtype.safetensors");
    // 3 bytes of U8: not a multiple of 4 — bytemuck::cast_slice used to panic.
    write_model(
        &path,
        &config_json(5, 8, 1),
        &tensors(5, 8, 1, Some((Dtype::U8, vec![3], vec![1, 2, 3]))),
    );
    assert!(FastGRNN::load(&path).is_err());
}

#[test]
fn load_rejects_zero_output_dim_instead_of_panicking_in_forward() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zero-out.safetensors");
    write_model(&path, &config_json(5, 8, 0), &tensors(5, 8, 0, None));
    // Either load refuses it, or forward returns an error — never a panic.
    if let Ok(model) = FastGRNN::load(&path) {
        assert!(model.forward(&[0.0; 5], None).is_err());
    }
}

#[test]
fn load_round_trips_a_saved_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ok.safetensors");
    let model = FastGRNN::new(FastGRNNConfig::default()).unwrap();
    model.save(&path).unwrap();
    let loaded = FastGRNN::load(&path).unwrap();
    let x = [0.1, 0.2, 0.3, 0.4, 0.5];
    assert_eq!(
        model.forward(&x, None).unwrap().to_bits(),
        loaded.forward(&x, None).unwrap().to_bits()
    );
}

#[test]
fn forward_rejects_wrong_hidden_state_length() {
    let model = FastGRNN::new(FastGRNNConfig::default()).unwrap();
    assert!(model.forward(&[0.0; 5], Some(&[0.0; 3])).is_err());
}

#[test]
fn router_rejects_model_with_incompatible_input_dim() {
    // trainRouter writes models with input_dim = embedding length; Router's
    // feature pipeline always produces 5 features. Such a model used to be
    // accepted, then fail every route and trip the circuit breaker.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dim8.safetensors");
    FastGRNN::new(FastGRNNConfig {
        input_dim: 8,
        ..Default::default()
    })
    .unwrap()
    .save(&path)
    .unwrap();
    let config = RouterConfig {
        model_path: path.to_string_lossy().into_owned(),
        ..Default::default()
    };
    assert!(Router::new(config).is_err());
}

fn candidate(embedding: Vec<f32>, success_rate: f32) -> Candidate {
    Candidate {
        id: "c".to_string(),
        embedding,
        metadata: HashMap::new(),
        created_at: Utc::now().timestamp(),
        access_count: 1,
        success_rate,
    }
}

fn strict_router() -> Router {
    Router::new(RouterConfig {
        model_path: "/nonexistent/tiny-dancer-review.safetensors".to_string(),
        circuit_breaker_threshold: 1,
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn non_finite_caller_input_is_rejected_without_tripping_breaker() {
    let router = strict_router();
    let bad_requests = vec![
        RoutingRequest {
            query_embedding: vec![f32::NAN, 0.5, 0.5],
            candidates: vec![candidate(vec![0.5; 3], 0.9)],
            metadata: None,
        },
        RoutingRequest {
            query_embedding: vec![0.5; 3],
            candidates: vec![candidate(vec![f32::INFINITY, 0.5, 0.5], 0.9)],
            metadata: None,
        },
        RoutingRequest {
            query_embedding: vec![0.5; 3],
            candidates: vec![candidate(vec![0.5; 3], f32::NAN)],
            metadata: None,
        },
    ];
    for req in bad_requests {
        assert!(router.route(req).is_err());
        assert_eq!(
            router.circuit_breaker_status(),
            Some(true),
            "caller input error must not open the circuit breaker"
        );
    }
    // A valid request still routes afterwards.
    let ok = router.route(RoutingRequest {
        query_embedding: vec![0.5; 3],
        candidates: vec![candidate(vec![0.4; 3], 0.9)],
        metadata: None,
    });
    assert!(ok.is_ok());
}

#[test]
fn zero_norm_embeddings_do_not_trip_breaker() {
    let router = strict_router();
    let _ = router.route(RoutingRequest {
        query_embedding: vec![0.0; 3],
        candidates: vec![candidate(vec![0.0; 3], 0.9)],
        metadata: None,
    });
    assert_eq!(router.circuit_breaker_status(), Some(true));
}

#[test]
fn extreme_created_at_does_not_overflow() {
    let router = strict_router();
    let mut c = candidate(vec![0.4; 3], 0.9);
    c.created_at = i64::MIN;
    let res = router
        .route(RoutingRequest {
            query_embedding: vec![0.5; 3],
            candidates: vec![c],
            metadata: None,
        })
        .unwrap();
    assert_eq!(res.decisions.len(), 1);
}

#[test]
fn half_open_needs_fresh_successes_to_close() {
    let cb = CircuitBreaker::with_timeout(2, Duration::ZERO);
    // Successes earned while closed must not count toward half-open recovery.
    for _ in 0..3 {
        cb.record_success();
    }
    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
    assert!(cb.is_closed()); // timeout elapsed -> half-open probe allowed
    assert_eq!(cb.state(), CircuitState::HalfOpen);
    cb.record_success();
    assert_eq!(
        cb.state(),
        CircuitState::HalfOpen,
        "one probe success must not close the circuit"
    );
}
