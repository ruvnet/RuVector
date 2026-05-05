# ADR-180 + ADR-181 implementation plan

Branch: `feature/ruvllm-batching-quant`
Started: 2026-05-05
Cluster: 4× Pi 5 + AI HAT+ (cognitum-v0, cognitum-cluster-1, -2, -3)
Cron: `5694314c` (every 5 min)
Target: ≥40 tok/s aggregate (ADR-180), ≥80 tok/s aggregate (ADR-181)

## Baseline (post-ADR-179 release)

| Iter | Stack | Aggregate | per-Pi |
|---|---|---:|---:|
| 26 | candle Q4_K_M, Mutex backend | 20.5 tok/s | 5.1 (parallel) / 9.0 (solo) |

## Phase A — ADR-180: ServingEngine continuous batching (iters 1-10)

| Iter | Goal |
|---|---|
| 1 | Branch off main + plan doc + ServingEngine API audit |
| 2 | Replace `engine::PiEngine` with `ServingEngine` wrapper |
| 3 | Wire `submit_async` into `handle_conn` request flow |
| 4 | Spawn `run_async` scheduler loop on worker startup |
| 5 | Cross-build aarch64 + smoke single-Pi (cluster-1) |
| 6 | Send 4 parallel requests to ONE Pi — measure batched vs solo |
| 7 | Roll out to all 4 Pis + restart services |
| 8 | 4-Pi cluster bench, max_inflight ∈ {1, 4, 8, 16} sweep |
| 9 | Quality gate: perplexity vs ADR-179 baseline (5 prompts) |
| 10 | Phase A convergence check or iterate |

## Phase B — ADR-181: pi_quant + BitNet b1.58 (iters 11-20)

| Iter | Goal |
|---|---|
| 11 | Audit `crates/ruvllm/src/quantize/pi_quant.rs` API |
| 12 | Convert TinyLlama-1.1B fp16 → pi_quant 3-bit blob (host) |
| 13 | Add `Quantization::PiQuant3` variant + dispatch in `candle_backend` |
| 14 | Stage pi_quant blob on cluster-1, smoke |
| 15 | Cluster bench Phase B intermediate |
| 16 | Audit `crates/ruvllm/src/bitnet/quantizer.rs` |
| 17 | Convert TinyLlama → BitNet b1.58 ternary blob |
| 18 | Wire `BitNetBackend` into `LlmBackend` trait |
| 19 | Stage + cluster bench |
| 20 | Phase B convergence check or iterate |

## Convergence rule

Stop when:
- 4-Pi aggregate tok/s holds for 2 consecutive iterations (no improvement) AND
- perplexity stays within 1% of fp16 reference

On convergence:
1. CronDelete `5694314c`
2. git push branch
3. gh pr create
4. cargo publish if ruvllm crate touched
5. Email summary to ruv@ruv.net via Resend `cluster@cognitum.one`

## Iter 1 (this commit)

**Done:**
- Branched `feature/ruvllm-batching-quant` off main (post ADR-179 merge)
- This plan doc
- Audited `LlmBackend` trait + `InferenceRequest::new` + `GenerateParams`

**Key API findings:**
- `LlmBackend::encode(&str) -> Result<Vec<u32>>` exists — worker can
  tokenize before submitting
- `LlmBackend::decode(&[u32]) -> Result<String>` exists — for detokenizing
  the result
- `InferenceRequest::new(Vec<u32>, GenerateParams)` — needs prompt
  pre-tokenized
- `ServingEngine::submit_async(InferenceRequest) -> Result<GenerationResult>`
  is the async oneshot API
- `ServingEngine::run_async()` is the scheduler loop — spawn once

**Wiring shape (planned for iter 2):**
```rust
struct PiEngine {
    backend: Arc<dyn LlmBackend>,    // for encode/decode
    engine: Arc<ServingEngine>,
}

async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
    let tokens = self.backend.encode(prompt)?;
    let params = GenerateParams { max_tokens, ..Default::default() };
    let req = InferenceRequest::new(tokens, params);
    let result = self.engine.submit_async(req).await?;
    self.backend.decode(&result.generated_tokens)
}
```

The `Arc<dyn LlmBackend>` is shared between PiEngine (for tokenize/
detokenize) AND ServingEngine (for the actual forward pass). Mutex
goes away — ServingEngine has its own scheduler that calls into
backend-internal interior-mutability state.

**Iter 2 plan:**
- Implement the above struct + replace existing PiEngine in
  `ruvllm-pi-worker.rs`
- Verify host build still works
- Cross-build aarch64
- Iter 3 wires the request handler
