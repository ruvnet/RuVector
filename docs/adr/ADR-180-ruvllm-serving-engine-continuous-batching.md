---
adr: 180
title: "Wire ruvllm's ServingEngine continuous batching into ruvllm-pi-worker"
status: proposed
date: 2026-05-05
authors: [ruvnet, claude-flow]
related: [ADR-179, ADR-176, ADR-167]
branch: feature/ruvllm-pi-cluster-batching
---

# ADR-180 — ServingEngine continuous batching on Pi 5

## Status

**Proposed.** Direct successor to ADR-179. ADR-179 hit a hard ceiling
at **20.5 tok/s aggregate (4-Pi Q4_K_M)** because each worker uses
`Mutex<CandleBackend>` — every request serializes behind one lock.
ADR-180 replaces that with `ruvllm::serving::ServingEngine`, which
already implements continuous batching with PagedAttention KV-cache.

## Context

ADR-179 iter 28 proved the bottleneck is **per-worker
serialization**, not network or memory bandwidth at the cluster
level. Sending 2 parallel requests to a single Pi takes 4552 ms
(2× the 1-request 1785 ms), with zero aggregate gain. The
single-mutex backend is the cap.

`crates/ruvllm/src/serving/` ships a complete continuous-batching
implementation:

- `ServingEngine` — the public driver
- `ContinuousBatchScheduler` — selects which prefill / decode tasks
  run each iteration
- `KvCacheManager` + `PagedAttention` — paged-KV slot allocation
- `submit_async(InferenceRequest) -> GenerationResult` — async
  oneshot that returns when the request finishes
- `run_async()` — main loop that the worker spawns once

We never wired this into `ruvllm-pi-worker` because iter 9 needed
the simplest possible end-to-end smoke. The `Mutex<CandleBackend>`
in `engine::PiEngine` is the iter-9 placeholder; ADR-180 replaces
it.

## Decision

Replace `engine::PiEngine` (in
`crates/ruvector-hailo-cluster/src/bin/ruvllm-pi-worker.rs`) with a
thin wrapper around `ruvllm::serving::ServingEngine`:

```rust
struct PiEngine {
    engine: Arc<ServingEngine>,
}

impl PiEngine {
    fn load(model_path: &str, max_inflight: usize) -> Result<Self> {
        let backend = Arc::new({
            let mut b = CandleBackend::new()?;
            let cfg = ModelConfig {
                use_flash_attention: false, // ADR-179 iter 10
                quantization: None,         // GGUF auto-detected on disk
                ..Default::default()
            };
            b.load_model(model_path, cfg)?;
            b
        });
        let engine_cfg = ServingEngineConfig {
            max_concurrent_requests: max_inflight,
            enable_speculative: false,      // iter 1 — defer spec decode
            ..Default::default()
        };
        let engine = Arc::new(ServingEngine::new(backend, engine_cfg));
        // spawn the scheduler loop
        let e = Arc::clone(&engine);
        tokio::spawn(async move { let _ = e.run_async().await; });
        Ok(Self { engine })
    }

    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let req = InferenceRequest::new(prompt, max_tokens);
        let result = self.engine.submit_async(req).await?;
        Ok(result.generated_text.unwrap_or_default())
    }
}
```

Per-connection request handlers in `handle_conn` call
`engine.generate(prompt, max_tokens).await` instead of locking the
backend. The scheduler loop interleaves prefill + decode across
all in-flight requests within a single forward pass.

### Configuration shape

New env vars in `/etc/ruvllm-pi-worker.env`:

```
RUVLLM_MAX_INFLIGHT      max concurrent requests (default 4 → 8 → 16 sweep)
RUVLLM_ENABLE_SPECULATIVE  enable draft-model speculative decode (default off until iter-2)
RUVLLM_KV_BLOCK_SIZE     PagedAttention block size (default 16 tokens)
RUVLLM_KV_CACHE_BLOCKS   total KV blocks per worker (sized to ~50% RAM)
```

### Acceptance

ADR-180 v1 ships when:

- [ ] All 4 Pis serve via ServingEngine, scheduler loop alive
- [ ] 4 parallel requests to ONE Pi land in roughly the same wall
      time as 1 (continuous batching working)
- [ ] 4-Pi cluster with 4 in-flight per Pi (16 total in-flight)
      hits **≥40 tok/s aggregate** (2× ADR-179 SOTA)
- [ ] Per-request quality preserved (perplexity within 1% of
      iter-26 reference on the same prompts)

ADR-180 v2 (deferred): enable speculative decoding with a 0.5B
draft model co-located on each Pi. Projected another 1.5-2× on
top.

## Implementation plan

| Iter | Step |
|---|---|
| 1 | Replace `engine::PiEngine::load` with the ServingEngine-based wrapper |
| 2 | Replace `engine::PiEngine::generate` with `submit_async` |
| 3 | Cross-build aarch64; deploy to cluster-1 only; smoke 1 request |
| 4 | Smoke 4 parallel requests to cluster-1 — measure batched vs solo wall |
| 5 | Replicate to all 4 Pis, run cluster bench |
| 6 | Sweep `max_concurrent_requests` ∈ {2, 4, 8, 16} per Pi |
| 7 | Measure perplexity (iter 26 reference vs ADR-180 batched) |
| 8 | Convergence per ADR-179 rule, email report |

## Risks

| Risk | Mitigation |
|---|---|
| `ServingEngine` requires `LlmBackend::generate_stream_v2` (TokenStream) which CandleBackend may have unimplemented | Audit `crates/ruvllm/src/backends/candle_backend.rs` impl; fall back to `generate_stream` v1 if needed |
| KV-cache memory pressure with 16 in-flight 1B-class requests | Cap via `RUVLLM_KV_CACHE_BLOCKS` sized to ~50% of 8 GB; reject when full |
| Tokio runtime in worker conflicts with ServingEngine's own runtime expectations | Use the worker's existing multi-thread runtime; ServingEngine is runtime-agnostic |
| Quality drop from speculative decoding (when iter-2 enables it) | Disable for ADR-180 v1; re-enable later with strict acceptance threshold |

## Acceptance criteria as test cases

```rust
#[tokio::test]
async fn batched_throughput_beats_serialized() {
    let engine = PiEngine::load("/var/lib/ruvllm/models/tinyllama-1.1b-q4", 4)?;
    let prompts: Vec<_> = (0..4).map(|i| format!("Prompt {i}: tell me about")).collect();
    let t0 = Instant::now();
    let results = futures::future::try_join_all(
        prompts.iter().map(|p| engine.generate(p, 16))
    ).await?;
    let elapsed = t0.elapsed();
    // ADR-180 acceptance: 4 parallel requests in ≤1.5× the wall of 1
    assert!(elapsed < Duration::from_millis(2700)); // vs ~1800 ms for 1 req
}
```

## Tracking

- Branch: `feature/ruvllm-pi-cluster-batching` (off `feature/ruvllm-pi-cluster`)
- Iteration log appended to existing `RUVLLM_CLUSTER_PLAN.md`
- Email convergence report via Resend `cluster@cognitum.one` → ruv@ruv.net
