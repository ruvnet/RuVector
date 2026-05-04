//! Pi-gated end-to-end integration test for the NPU + cpu-fallback
//! workers running on real hardware.
//!
//! Iter 172 (ADR-176 follow-up). Locks in the iter-163 / iter-149
//! throughput numbers as regression gates by running the same
//! cluster-bench-style workload from this test process. Skips
//! entirely when `RUVECTOR_TEST_PI_HOST` is unset so CI / dev-box
//! `cargo test` is unaffected.
//!
//! Usage:
//!   RUVECTOR_TEST_PI_HOST=cognitum-v0:50051 \
//!     cargo test -p ruvector-hailo-cluster --test pi_hardware_integration \
//!     -- --nocapture --test-threads=1

use ruvector_hailo_cluster::transport::{EmbeddingTransport, WorkerEndpoint};
use ruvector_hailo_cluster::{GrpcTransport, HailoClusterEmbedder};
use std::sync::Arc;
use std::time::Instant;

fn pi_host() -> Option<String> {
    std::env::var("RUVECTOR_TEST_PI_HOST").ok()
}

fn cluster(addr: &str) -> HailoClusterEmbedder {
    let workers = vec![WorkerEndpoint::new("pi", addr)];
    let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
        Arc::new(GrpcTransport::new().expect("GrpcTransport::new"));
    HailoClusterEmbedder::new(workers, transport, 384, "").expect("HailoClusterEmbedder::new")
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[test]
fn pi_worker_returns_real_semantic_vectors() {
    let Some(addr) = pi_host() else {
        eprintln!("skipping — set RUVECTOR_TEST_PI_HOST=<host>:<port>");
        return;
    };
    let c = cluster(&addr);

    // Three reference phrases — same set the iter-167 worker
    // self-test uses. If we get the same ranking from the cluster
    // side, we know:
    //   * worker is up
    //   * NPU/cpu-fallback path is loaded
    //   * tokenizer + embeddings + encoder + pool agree
    let v0 = c
        .embed_one_blocking("the quick brown fox jumps over the lazy dog")
        .unwrap();
    let v1 = c
        .embed_one_blocking("a puppy sprints across the meadow")
        .unwrap();
    let v2 = c
        .embed_one_blocking("kafka topic partition rebalancing strategy")
        .unwrap();

    assert_eq!(v0.len(), 384);
    assert_eq!(v1.len(), 384);
    assert_eq!(v2.len(), 384);

    let sim_close = cos(&v0, &v1);
    let sim_far = cos(&v0, &v2);
    eprintln!(
        "Pi {}: sim(dog,puppy)={:.4}  sim(dog,kafka)={:.4}  Δ={:+.4}",
        addr,
        sim_close,
        sim_far,
        sim_close - sim_far
    );
    assert!(
        sim_close > sim_far,
        "ranking violation: sim(dog,puppy)={:.4} <= sim(dog,kafka)={:.4}",
        sim_close,
        sim_far
    );
    assert!(
        sim_close - sim_far > 0.10,
        "ranking margin too thin: Δ={:+.4} (encoder may be degenerate)",
        sim_close - sim_far
    );
}

#[test]
fn pi_worker_throughput_above_floor() {
    let Some(addr) = pi_host() else {
        return;
    };
    let c = cluster(&addr);

    // iter-149 cpu-fallback baseline = 7 / sec
    // iter-163 NPU                   = 67 / sec
    // Floor is 5 / sec — catches a regression that would drop the
    // cpu-fallback path below useful, while still allowing the much
    // weaker Pi 4 (~3-4 / sec estimated) to fail loudly.
    const FLOOR_EMBEDS_PER_SEC: f64 = 5.0;
    const SAMPLES: usize = 30;

    // Warm up so the first-call model load doesn't skew the bench.
    let _ = c.embed_one_blocking("warm-up").unwrap();

    let t0 = Instant::now();
    for i in 0..SAMPLES {
        let s = format!("benchmark sentence number {} of {}", i, SAMPLES);
        let v = c.embed_one_blocking(&s).unwrap();
        assert_eq!(v.len(), 384);
    }
    let elapsed = t0.elapsed();
    let rate = SAMPLES as f64 / elapsed.as_secs_f64();
    eprintln!(
        "Pi {}: {} embeds in {:.2}s = {:.1} embeds/sec",
        addr,
        SAMPLES,
        elapsed.as_secs_f64(),
        rate
    );
    assert!(
        rate >= FLOOR_EMBEDS_PER_SEC,
        "throughput {:.1} / sec below floor {:.1} (regression?)",
        rate,
        FLOOR_EMBEDS_PER_SEC
    );
}

#[test]
fn pi_worker_handles_padding_and_truncation() {
    let Some(addr) = pi_host() else {
        return;
    };
    let c = cluster(&addr);

    // Empty string → tokenizer emits [CLS][SEP] → encoder runs on
    // 2 attended positions, 126 PAD. Output should still be a
    // finite unit vector.
    let v_empty = c.embed_one_blocking("").unwrap();
    assert_eq!(v_empty.len(), 384);
    assert!(v_empty.iter().all(|x| x.is_finite()));

    // Long input → tokenizer truncates to seq=128. Should still work.
    let long: String = "lorem ipsum dolor sit amet ".repeat(200);
    let v_long = c.embed_one_blocking(&long).unwrap();
    assert_eq!(v_long.len(), 384);
    assert!(v_long.iter().all(|x| x.is_finite()));
}
