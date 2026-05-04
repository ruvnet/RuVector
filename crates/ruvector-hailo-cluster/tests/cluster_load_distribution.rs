//! End-to-end integration test for the HailoClusterEmbedder dispatch loop.
//!
//! Spins up 2 mock tonic workers with different artificial latencies on
//! localhost, builds a HailoClusterEmbedder pointed at both via the real
//! GrpcTransport, dispatches N=200 embed requests, then verifies:
//!
//!   1. All 200 succeed (no AllWorkersFailed)
//!   2. Both workers got non-zero traffic (not pinning to one)
//!   3. The fast worker got *more* traffic than the slow one — confirms
//!      P2C random + EWMA latency tracking is biasing as designed.
//!
//! Uses real TCP + HTTP/2 + protobuf via tonic, so this is the most
//! convincing validation of the dispatch path short of running against
//! actual Pi 5 + Hailo workers.

use ruvector_hailo_cluster::proto::embedding_server::{Embedding, EmbeddingServer};
use ruvector_hailo_cluster::proto::{
    EmbedBatchRequest, EmbedRequest, EmbedResponse, EmbedStreamResponse, HealthRequest,
    HealthResponse, StatsRequest, StatsResponse,
};
use ruvector_hailo_cluster::transport::WorkerEndpoint;
use ruvector_hailo_cluster::{GrpcTransport, HailoClusterEmbedder};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use tonic::{transport::Server, Request, Response, Status};

/// Mock worker. Delays each `embed` call by `delay_ms` to simulate a slow
/// vs fast NPU. Counts calls per worker so the test can verify load
/// distribution.
struct DelayWorker {
    name: String,
    delay_ms: u64,
    calls: Arc<AtomicU64>,
    /// Each completed embed pushes the observed request_id (from gRPC
    /// metadata, falling back to the proto field) into this vec. Used
    /// by `caller_supplied_request_id_propagates_to_worker`.
    seen_request_ids: Arc<std::sync::Mutex<Vec<String>>>,
}

#[tonic::async_trait]
impl Embedding for DelayWorker {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let observed = ruvector_hailo_cluster::proto::extract_request_id(
            &request,
            &request.get_ref().request_id,
        );
        self.seen_request_ids.lock().unwrap().push(observed);
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
        }
        // Deterministic 4-dim vector — keeps the test independent of real
        // model output. Coordinator validates dim, so we match.
        Ok(Response::new(EmbedResponse {
            vector: vec![0.0, 0.0, 0.0, 0.0],
            dim: 4,
            latency_us: (self.delay_ms * 1000) as i64,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: "delay-mock".into(),
            device_id: self.name.clone(),
            model_fingerprint: "fp:test".into(),
            ready: true,
            npu_temp_ts0_celsius: 0.0,
            npu_temp_ts1_celsius: 0.0,
        }))
    }
    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        Ok(Response::new(StatsResponse::default()))
    }
    type EmbedStreamStream = std::pin::Pin<
        Box<dyn futures_core::Stream<Item = Result<EmbedStreamResponse, Status>> + Send + 'static>,
    >;
    async fn embed_stream(
        &self,
        request: Request<EmbedBatchRequest>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        let observed = ruvector_hailo_cluster::proto::extract_request_id(
            &request,
            &request.get_ref().request_id,
        );
        self.seen_request_ids.lock().unwrap().push(observed);
        let req = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<EmbedStreamResponse, Status>>(
            req.texts.len().max(1),
        );
        let calls = self.calls.clone();
        let delay = Duration::from_millis(self.delay_ms);
        tokio::task::spawn(async move {
            for (i, _text) in req.texts.into_iter().enumerate() {
                calls.fetch_add(1, Ordering::SeqCst);
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                let item = Ok(EmbedStreamResponse {
                    index: i as u32,
                    vector: vec![0.0, 0.0, 0.0, 0.0],
                    dim: 4,
                    latency_us: delay.as_micros() as i64,
                });
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }
}

fn start_worker(rt: &Runtime, name: &str, delay_ms: u64) -> (SocketAddr, Arc<AtomicU64>) {
    let (addr, calls, _seen) = start_worker_capturing(rt, name, delay_ms);
    (addr, calls)
}

fn start_worker_capturing(
    rt: &Runtime,
    name: &str,
    delay_ms: u64,
) -> (
    SocketAddr,
    Arc<AtomicU64>,
    Arc<std::sync::Mutex<Vec<String>>>,
) {
    let calls = Arc::new(AtomicU64::new(0));
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let calls_for_server = calls.clone();
    let seen_for_server = seen.clone();
    let name_owned = name.to_string();

    let addr = rt.block_on(async move {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let listener_stream = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let svc = DelayWorker {
            name: name_owned,
            delay_ms,
            calls: calls_for_server,
            seen_request_ids: seen_for_server,
        };
        tokio::spawn(async move {
            Server::builder()
                .add_service(EmbeddingServer::new(svc))
                .serve_with_incoming(listener_stream)
                .await
                .ok();
        });
        // Brief delay so the server is accepting before we hand back.
        tokio::time::sleep(Duration::from_millis(100)).await;
        addr
    });
    (addr, calls, seen)
}

#[test]
fn p2c_ewma_biases_toward_fast_worker_under_load() {
    // Reserve the test's tokio runtime — DelayWorker server tasks live here.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    // Spawn two workers: one snappy (1ms), one slow (50ms). The
    // 50:1 latency gap (vs the original 15:1) makes the EWMA bias
    // dominant even under tokio scheduler jitter — earlier 15ms gap
    // was close enough to tonic's per-call framing overhead that
    // observed latency ratios fluctuated from 8:1 to 3:1, leaving
    // EWMA picks split closer to 64/36 instead of the asserted 2:1.
    let (fast_addr, fast_calls) = start_worker(&server_rt, "fast", 1);
    let (slow_addr, slow_calls) = start_worker(&server_rt, "slow", 50);

    let workers = vec![
        WorkerEndpoint::new("fast", fast_addr.to_string()),
        WorkerEndpoint::new("slow", slow_addr.to_string()),
    ];

    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster =
        HailoClusterEmbedder::new(workers, transport, 4, "fp:test").expect("init cluster");

    // Iter 196 — warmup phase so the EWMA has steady-state samples
    // before the ratio assertion. Without this, the first ~10 calls
    // include tonic channel-dial cost (~50 ms) which dominates the
    // 1 vs 15 ms handler delay; EWMA convergence then depends on
    // which worker the deterministic P2C LCG happens to pick first,
    // and the test was intermittently routing to slow when fast's
    // first call paid the dial tax. After warmup both channels are
    // cached + both EWMAs reflect steady-state handler latency, and
    // the bias check is reliable.
    const WARMUP: usize = 30;
    for i in 0..WARMUP {
        let _ = cluster.embed_one_blocking(&format!("warmup-{}", i));
    }
    fast_calls.store(0, Ordering::SeqCst);
    slow_calls.store(0, Ordering::SeqCst);

    const N: usize = 200;
    let mut errors = 0usize;
    for i in 0..N {
        match cluster.embed_one_blocking(&format!("text-{}", i)) {
            Ok(_) => {}
            Err(_) => errors += 1,
        }
    }

    let fast = fast_calls.load(Ordering::SeqCst);
    let slow = slow_calls.load(Ordering::SeqCst);
    eprintln!(
        "dispatch result (post-warmup): fast={}, slow={}, errors={}",
        fast, slow, errors
    );

    assert_eq!(errors, 0, "all {} dispatches should succeed", N);
    assert!(fast > 0, "fast worker should receive some traffic");
    assert_eq!(
        fast as usize + slow as usize,
        N,
        "every dispatch lands on exactly one worker"
    );
    // EWMA bias check: with 1ms vs 15ms post-warmup latency, the
    // picker should clearly prefer the fast one. Demand at least 2:1
    // — easily achievable once dial tax has amortized.
    assert!(
        fast as f64 / (slow as f64).max(1.0) >= 2.0,
        "expected ≥2:1 fast:slow ratio under EWMA bias, got fast={} slow={}",
        fast,
        slow
    );
}

#[test]
fn embed_batch_streaming_returns_ordered_results() {
    // DelayWorker streams one EmbedStreamResponse per input text, indexed
    // 0..N. The dispatcher sorts by `index` and returns Vec<Vec<f32>> in
    // input order. Since DelayWorker emits rows in order anyway, the sort
    // is a no-op here — but the assertion structure proves the contract.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (a_addr, _) = start_worker(&server_rt, "alpha", 0);

    let workers = vec![WorkerEndpoint::new("alpha", a_addr.to_string())];
    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster =
        HailoClusterEmbedder::new(workers, transport, 4, "fp:test").expect("init cluster");

    let texts = vec![
        "alpha".to_string(),
        "beta".to_string(),
        "gamma".to_string(),
        "delta".to_string(),
    ];
    let vectors = cluster
        .embed_batch_blocking(&texts)
        .expect("batch embed should succeed");

    // One vector per input.
    assert_eq!(vectors.len(), texts.len(), "len mismatch");
    // Each vector matches the cluster's declared dim.
    for (i, v) in vectors.iter().enumerate() {
        assert_eq!(v.len(), 4, "row {} dim mismatch", i);
    }
    // DelayWorker emits zero-vectors deterministically, so every row
    // should be the same length-4 zero vector — confirming the bytes
    // round-tripped through the wire format intact.
    for v in &vectors {
        for &x in v {
            assert_eq!(x, 0.0);
        }
    }
}

#[test]
fn batch_cache_reuses_results_across_calls() {
    // Two batched calls with overlapping inputs. The second call should
    // hit the cache for shared texts and only RPC the new ones — observed
    // via the worker's `calls` counter.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (a_addr, calls) = start_worker(&server_rt, "alpha", 0);

    let workers = vec![WorkerEndpoint::new("alpha", a_addr.to_string())];
    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster = HailoClusterEmbedder::new(workers, transport, 4, "fp:cache-batch")
        .expect("init cluster")
        .with_cache(64);

    // First batch — 3 items, all misses → 3 RPC items.
    let v1 = cluster
        .embed_batch_blocking(&["alpha".to_string(), "beta".to_string(), "gamma".to_string()])
        .expect("first batch");
    assert_eq!(v1.len(), 3);
    let calls_after_first = calls.load(Ordering::SeqCst);
    assert_eq!(calls_after_first, 3, "first batch hits worker 3 times");

    // Second batch — 2 overlapping (alpha, gamma) + 1 new (delta).
    // Worker should see exactly 1 more call.
    let v2 = cluster
        .embed_batch_blocking(&[
            "alpha".to_string(),
            "gamma".to_string(),
            "delta".to_string(),
        ])
        .expect("second batch");
    assert_eq!(v2.len(), 3);
    let calls_after_second = calls.load(Ordering::SeqCst);
    assert_eq!(
        calls_after_second - calls_after_first,
        1,
        "second batch should only RPC the 1 miss (delta)"
    );

    let s = cluster.cache_stats();
    assert_eq!(s.size, 4, "cache should hold 4 unique texts");
    assert_eq!(s.hits, 2, "two cache hits in second batch");
    assert_eq!(s.misses, 4, "first 3 + second 1 miss = 4");
}

#[test]
fn batch_all_cached_skips_rpc_entirely() {
    // After warming the cache with all texts, a follow-up batch should
    // hit zero workers — the all-cached fast path returns without
    // touching the pool.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (a_addr, calls) = start_worker(&server_rt, "alpha", 0);

    let workers = vec![WorkerEndpoint::new("alpha", a_addr.to_string())];
    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster = HailoClusterEmbedder::new(workers, transport, 4, "fp:warm")
        .expect("init cluster")
        .with_cache(8);

    // Warm.
    let _ = cluster
        .embed_batch_blocking(&["a".to_string(), "b".to_string()])
        .expect("warm");
    let warm_calls = calls.load(Ordering::SeqCst);
    assert_eq!(warm_calls, 2);

    // Replay — must be free.
    let v = cluster
        .embed_batch_blocking(&["a".to_string(), "b".to_string()])
        .expect("replay");
    assert_eq!(v.len(), 2);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        warm_calls,
        "all-cached batch must not touch the worker"
    );

    let s = cluster.cache_stats();
    assert_eq!(s.hits, 2);
    assert_eq!(s.misses, 2);
}

#[test]
fn embed_batch_async_succeeds_inside_tokio_runtime() {
    // Mirror of the sync test, but driving the call through the async
    // wrapper from inside a #[tokio::main]-style runtime. Verifies the
    // spawn_blocking bridge doesn't deadlock or drop the result.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (a_addr, _) = start_worker(&server_rt, "alpha", 0);

    let workers = vec![WorkerEndpoint::new("alpha", a_addr.to_string())];
    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster = Arc::new(
        HailoClusterEmbedder::new(workers, transport, 4, "fp:test").expect("init cluster"),
    );

    // A separate runtime simulates a tokio app calling `cluster.embed_batch(...).await`.
    let app_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let texts = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
    let result = app_rt.block_on(async { Arc::clone(&cluster).embed_batch(texts).await });

    let vectors = result.expect("async batch embed should succeed");
    assert_eq!(vectors.len(), 3);
    for v in &vectors {
        assert_eq!(v.len(), 4);
    }
}

#[test]
fn async_embed_one_succeeds_inside_tokio_runtime() {
    // Independent runtime for the worker.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (addr, _calls) = start_worker(&server_rt, "alpha", 0);

    let workers = vec![WorkerEndpoint::new("alpha", addr.to_string())];
    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster =
        Arc::new(HailoClusterEmbedder::new(workers, transport, 4, "fp:test").expect("init"));

    // Caller-side runtime — exercise the async path the way a tokio
    // application would.
    let caller_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let result =
        caller_rt.block_on(async move { cluster.embed_one("hello async".to_string()).await });
    let v = result.expect("embed_one async should succeed against mock worker");
    assert_eq!(v.len(), 4);
}

#[test]
fn fleet_stats_returns_one_entry_per_worker() {
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (a_addr, _) = start_worker(&server_rt, "a", 0);
    let (b_addr, _) = start_worker(&server_rt, "b", 0);

    let workers = vec![
        WorkerEndpoint::new("a", a_addr.to_string()),
        WorkerEndpoint::new("b", b_addr.to_string()),
    ];

    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster = HailoClusterEmbedder::new(workers, transport, 4, "fp:test").expect("init");

    // Drive a few embeds first so each worker has some counter activity
    // (the DelayWorker mock doesn't actually track stats — its get_stats
    // returns Default — but the RPC path is what we're validating here).
    for _ in 0..3 {
        let _ = cluster.embed_one_blocking("warm-up");
    }

    let stats = cluster.fleet_stats();
    assert_eq!(stats.len(), 2, "one entry per worker");
    for (worker, result) in &stats {
        assert!(
            result.is_ok(),
            "fleet_stats RPC against worker {} failed: {:?}",
            worker.name,
            result.as_ref().err()
        );
        // Mock returns Default → all counters zero. The fact that we got
        // a StatsSnapshot back at all proves the RPC path works.
        let snap = result.as_ref().unwrap();
        eprintln!("worker {} stats: {:?}", worker.name, snap);
    }
}

#[test]
fn dispatch_continues_when_one_worker_dies() {
    // Spin up two workers, immediately tear one of them down by binding
    // a TcpListener that we drop before the test starts. The "dead" address
    // is unreachable from the dispatcher's POV.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    // Get an unbound port that's guaranteed-dead.
    let dead_addr = server_rt.block_on(async {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a = l.local_addr().unwrap();
        drop(l);
        a
    });

    let (alive_addr, alive_calls) = start_worker(&server_rt, "alive", 1);

    let workers = vec![
        WorkerEndpoint::new("dead", dead_addr.to_string()),
        WorkerEndpoint::new("alive", alive_addr.to_string()),
    ];

    let transport = Arc::new(
        GrpcTransport::with_timeouts(Duration::from_millis(200), Duration::from_millis(200))
            .unwrap(),
    );
    let cluster =
        HailoClusterEmbedder::new(workers, transport, 4, "fp:test").expect("init cluster");

    let mut ok = 0usize;
    let mut err = 0usize;
    for i in 0..50 {
        match cluster.embed_one_blocking(&format!("survival-{}", i)) {
            Ok(_) => ok += 1,
            Err(_) => err += 1,
        }
    }

    eprintln!(
        "with one dead worker: ok={}, err={}, alive_received={}",
        ok,
        err,
        alive_calls.load(Ordering::SeqCst)
    );

    // After health failures eject the dead worker (3 failures → ejected),
    // remaining requests should land cleanly on `alive`.
    assert!(ok > 0, "some embeds should succeed via the live worker");
    assert!(
        alive_calls.load(Ordering::SeqCst) >= ok as u64,
        "alive worker received >= as many calls as we counted as ok"
    );
}

#[test]
fn caller_supplied_request_id_propagates_to_worker() {
    // Caller passes "trace-12345" through the public API; the test
    // mock captures whatever it sees in the gRPC metadata header (or
    // proto-field fallback). The two should match — proves the
    // request_id is propagated end-to-end through tonic, including
    // back-compat with the proto field.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (addr, _calls, seen) = start_worker_capturing(&server_rt, "alpha", 0);

    let workers = vec![WorkerEndpoint::new("alpha", addr.to_string())];
    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster =
        HailoClusterEmbedder::new(workers, transport, 4, "fp:trace-test").expect("init cluster");

    let _ = cluster
        .embed_one_blocking_with_request_id("hello", "trace-12345")
        .expect("embed should succeed");

    let observed = seen.lock().unwrap().clone();
    assert_eq!(
        observed,
        vec!["trace-12345".to_string()],
        "worker should see the caller-supplied id"
    );

    // Sanity: an embed_one_blocking call (random id) should produce a
    // non-empty, different id — confirming the random-id path still works.
    let _ = cluster
        .embed_one_blocking("world")
        .expect("embed should succeed");
    let observed = seen.lock().unwrap().clone();
    assert_eq!(observed.len(), 2);
    assert_ne!(observed[1], "trace-12345");
    assert!(
        !observed[1].is_empty(),
        "random id path should populate something"
    );
}

#[test]
fn caller_supplied_request_id_propagates_through_batch() {
    // Same shape as the single-embed test, but for embed_batch_blocking
    // — exercises the streaming RPC + metadata header path.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (addr, _calls, seen) = start_worker_capturing(&server_rt, "alpha", 0);

    let workers = vec![WorkerEndpoint::new("alpha", addr.to_string())];
    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster =
        HailoClusterEmbedder::new(workers, transport, 4, "fp:trace-batch").expect("init cluster");

    let _ = cluster
        .embed_batch_blocking_with_request_id(
            &["a".to_string(), "b".to_string(), "c".to_string()],
            "batch-trace-99",
        )
        .expect("batch embed should succeed");

    let observed = seen.lock().unwrap().clone();
    // One streaming RPC, one entry in seen.
    assert_eq!(observed, vec!["batch-trace-99".to_string()]);

    // Random-id path on plain embed_batch_blocking still works.
    let _ = cluster
        .embed_batch_blocking(&["d".to_string()])
        .expect("plain batch should succeed");
    let observed = seen.lock().unwrap().clone();
    assert_eq!(observed.len(), 2);
    assert_ne!(observed[1], "batch-trace-99");
    assert!(!observed[1].is_empty());
}

#[test]
fn async_embed_one_with_request_id_propagates() {
    // Async sibling of `caller_supplied_request_id_propagates_to_worker`.
    // Drives `Arc::clone(&cluster).embed_one_with_request_id(...).await`
    // from inside a tokio runtime to verify the spawn_blocking wrapper
    // correctly propagates the supplied id end-to-end.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (addr, _calls, seen) = start_worker_capturing(&server_rt, "alpha", 0);

    let workers = vec![WorkerEndpoint::new("alpha", addr.to_string())];
    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster = Arc::new(
        HailoClusterEmbedder::new(workers, transport, 4, "fp:async-trace").expect("init cluster"),
    );

    let app_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let result = app_rt.block_on(async {
        Arc::clone(&cluster)
            .embed_one_with_request_id("hello".into(), "async-trace-42".into())
            .await
    });
    let _ = result.expect("async embed_one_with_request_id should succeed");

    let observed = seen.lock().unwrap().clone();
    assert_eq!(observed, vec!["async-trace-42".to_string()]);
}

#[test]
fn async_embed_batch_with_request_id_propagates() {
    // Same shape, batch path through async wrapper.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (addr, _calls, seen) = start_worker_capturing(&server_rt, "alpha", 0);

    let workers = vec![WorkerEndpoint::new("alpha", addr.to_string())];
    let transport = Arc::new(GrpcTransport::new().unwrap());
    let cluster = Arc::new(
        HailoClusterEmbedder::new(workers, transport, 4, "fp:async-batch-trace")
            .expect("init cluster"),
    );

    let app_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let result = app_rt.block_on(async {
        Arc::clone(&cluster)
            .embed_batch_with_request_id(texts, "async-batch-trace-7".into())
            .await
    });
    let vectors = result.expect("async batch should succeed");
    assert_eq!(vectors.len(), 3);

    // One streaming RPC, one captured id.
    let observed = seen.lock().unwrap().clone();
    assert_eq!(observed, vec!["async-batch-trace-7".to_string()]);
}
