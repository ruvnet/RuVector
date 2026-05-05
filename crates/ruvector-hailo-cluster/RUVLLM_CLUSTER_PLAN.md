# ruvllm on the 4-node Pi 5 cluster — implementation plan

Branch: `feature/ruvllm-pi-cluster`
Started: 2026-05-04

## Goal

Deploy in-tree `crates/ruvllm` (the existing Rust LLM inference engine)
across the 4-Pi cluster (cognitum-v0, cognitum-cluster-1, cognitum-cluster-2,
cognitum-cluster-3 — all Pi 5 + AI HAT+) with quantization (pi_quant,
turbo_quant, RaBitQ, QuIP). Target models: phi-3-mini, Qwen2.5-1.5B,
Llama-3.2-1B. Iterate to SOTA per-node throughput + tail latency.

## What's already in tree

- `crates/ruvllm/` — engine (serving, kv_cache, paged_attention, kernels, lora, moe, intelligence)
- `crates/ruvllm/src/quantize/` — `pi_quant.rs`, `pi_quant_simd.rs`, `turbo_quant.rs`,
  `turboquant_profile.rs`, `quip.rs`, `hadamard.rs`, `incoherence.rs`, `ruvltra_quant.rs`
- `crates/ruvector-rabitq/` — separate crate (RaBitQ already implemented)
- `crates/ruvllm-cli/` — `ruvllm` binary with `serve | quantize | benchmark | download | chat | info | list`
- `crates/ruvllm/benches/` — `pi_quant_bench.rs`, `turbo_quant_bench.rs`, `serving_bench.rs`, etc.

## Iteration plan (5-min cycles, /loop 0dd7f865)

| Iter | Goal | Acceptance |
|---|---|---|
| 1 | Survey + plan + try aarch64 build | This doc + build status known |
| 2 | Cross-build `ruvllm-cli` for aarch64 (no-default-features → minimum viable) | binary at `target/aarch64.../ruvllm` |
| 3 | scp binary to all 4 Pis; smoke `ruvllm --version` over ssh | binary runs on Pi 5 |
| 4 | Download Llama-3.2-1B (smallest of the 3) into `/var/lib/ruvllm/models/` on cognitum-v0 | model file present, HF-format |
| 5 | Quantize once on cognitum-v0 with pi_quant | quantized blob produced |
| 6 | Serve on cognitum-v0 (port 50053), `ruvllm chat` smoke from ruvultra | first token response |
| 7 | Replicate model + service to all 4 nodes | 4× `ruvllm serve` listening |
| 8 | Add `ruvllm-cluster-bench` (mirror of hailo bench) for completion RPCs | per-node + 4-node throughput numbers |
| 9 | Apply turbo_quant on top of pi_quant (composable) | quality + throughput delta |
| 10 | RaBitQ on KV-cache (`crates/ruvector-rabitq` + sparse_inference's ruvllm.rs hook, ADR-154) | KV-cache memory reduction |
| 11 | **BitNet b1.58 ternary weights** via `crates/ruvllm/src/bitnet/` (ADR-024) — 1.58-bit weight conversion for Llama-3.2-1B (smallest first) | quantized weight blob + eval harness clean |
| 12 | Quality sweep across {fp16, pi_quant, turbo_quant, BitNet b1.58, +RaBitQ-KV} for all 3 models | ≤1% perplexity gap target on at least one quant per model |
| 13 | Cross-product matrix: model × quant — pick winners per model | best (tok/s × quality) per model |
| 14 | Optimize: NPU dispatch via Hailo-8 — investigate which transformer ops compile | feasibility note |
| 15+ | Push throughput / latency frontier per quantization scheme | iterate to SOTA |

Convergence rule per loop directive: stop when tok/s + p50 don't
improve for 2 consecutive iterations across both throughput AND quality
(perplexity within 1% of fp16 reference).

## Architecture

```
                        ┌──────────────────┐
                        │  ruvllm-cli      │  on each Pi 5
                        │  (serve mode)    │
                        │  port 50053      │
                        │  pi_quant Q4     │
                        │  pool=N requests │
                        └────────▲─────────┘
                                 │ gRPC completion
              ┌──────────────────┼──────────────────┐
              │                  │                  │
        cognitum-v0        cluster-1            cluster-2/3
        :50051 embed                            ...
        :50053 llm
                                 │
                  ┌──────────────┴──────────────┐
                  │ ruvllm-cluster-bench (new)  │  on ruvultra
                  │ P2C+EWMA across 4 :50053    │
                  └─────────────────────────────┘
```

## Open questions (for iter 1)

1. Does `ruvllm-cli` build for aarch64 with no-default-features? Likely needs feature gating audit (metal/cuda/ane should be off).
2. Where does ruvllm currently load models from? GGUF? HF safetensors? Both?
3. What's the gRPC interface on `serve`? Or is it OpenAI-compatible HTTP? The Python `ruos-llm-serve` on ruvultra answers `/v1/models` so probably OpenAI-compat.
4. KV-cache size at Pi-5 RAM limits — Llama-3.2-1B Q4 is ~600 MB weights + KV per request. 4-8 in-flight requests fit in 8 GB.

## Iter 1 result

(Pending build attempt below.)

## Iter 1 (2026-05-04, ~20:05)

**Done:**
- branch `feature/ruvllm-pi-cluster` created off main
- ADR-179 drafted at `docs/adr/ADR-179-ruvllm-pi-cluster-deployment.md`
- Surveyed ruvllm crate — engine + quantization + serving all in tree
- Identified `ruvllm-cli` as the binary entry point
- aarch64 cross target installed ✓

**Blocker for iter 2:**
- `cargo build --target aarch64-unknown-linux-gnu --release -p ruvllm-cli`
  fails on `openssl-sys 0.9.112` — needs aarch64 OpenSSL libs OR a
  rustls-only feature path. Options:
  1. `apt install libssl-dev:arm64` + a cross sysroot env (heavyweight)
  2. Vendor: `OPENSSL_VENDORED=1` + cross-build openssl (slow)
  3. Audit ruvllm-cli's transitive deps and pin a feature subset
     that doesn't pull `openssl-sys` (best — likely `hub` HF download
     pulls reqwest/tls/openssl)
- Iter 2 plan: option 3 — find which dep pulls openssl, build with
  feature subset that excludes it, fall back to rustls for any HTTP.

**Files staged:**
- `docs/adr/ADR-179-ruvllm-pi-cluster-deployment.md`
- `crates/ruvector-hailo-cluster/RUVLLM_CLUSTER_PLAN.md`

## Iter 2 (2026-05-04, ~20:10)

**Done:**
- Identified `hf-hub` → `native-tls` → `openssl-sys` as the cross-build blocker
- Patched `crates/ruvllm-cli/Cargo.toml` and `crates/ruvllm/Cargo.toml`:
  `hf-hub = { default-features = false, features = ["tokio", "rustls-tls"] }`
- Added workspace-level `.cargo/config.toml` aarch64 stanza:
  `linker = "aarch64-linux-gnu-gcc"` + Cortex-A76 rustflags (matches
  iter-84 hailo-cluster ultra profile for the `+lse +rcpc +fp16 +crc`
  feature set that gave the embed cluster its 65% perf bump)
- Identified that the user's shell `RUSTFLAGS=-C link-arg=-fuse-ld=mold`
  overrides config rustflags entirely; cross-build needs `RUSTFLAGS=`
  prefix.
- Build now passes openssl AND linker stages — cleanly hits the
  Cortex-A76 + rustls path.

**New blocker (iter 3 plan):**
- `hf-hub` 0.4.3 feature `rustls-tls` only switches reqwest's TLS;
  the sync `hf_hub::api::sync` API still requires `ureq` feature,
  and `ureq` brings back native-tls. `crates/ruvllm/src/backends/candle_backend.rs:462`
  uses sync API.
- **Decision:** don't try to make `ruvllm-cli` cross-build the whole
  HF download flow. Instead, create a new minimal binary
  `crates/ruvector-hailo-cluster/src/bin/ruvllm-pi-worker.rs` that:
  - Uses ruvllm as a library (engine + serving + quantize)
  - Loads model from a local `.safetensors` / `.gguf` path (no hf-hub)
  - Exposes gRPC on `:50053` (mirrors hailo worker pattern on `:50051`)
  - Models rsync'd from ruvultra → Pis ahead of time
- This avoids the hf-hub mess + reuses our embedding cluster's deploy
  conventions (systemd unit, env file, install script).

**Files staged:**
- `.cargo/config.toml` (workspace)
- `crates/ruvllm/Cargo.toml`
- `crates/ruvllm-cli/Cargo.toml`

## Iter 3 (2026-05-04, ~20:18)

**Done:**
- Created `crates/ruvector-hailo-cluster/src/bin/ruvllm-pi-worker.rs`
  scaffold (env contract, TCP listener, version banner). Mirrors the
  hailo worker's env-var documentation style.
- Added `[[bin]]` entry in `crates/ruvector-hailo-cluster/Cargo.toml`
- **Cross-build to aarch64 succeeds end-to-end.** Binary at
  `target/aarch64-unknown-linux-gnu/release/ruvllm-pi-worker`,
  size 1.18 MB. Compiles with the Cortex-A76 rustflags from the
  workspace `.cargo/config.toml`.
- Smoke probe works on host: `nc localhost 50053` returns version
  banner + bind addr.

**Iter 4 plan:**
- scp aarch64 binary to all 4 Pis (`/usr/local/bin/ruvllm-pi-worker`)
- write `ruvllm-pi-worker.service` systemd unit + `ruvllm-pi-worker.env.example`
- write `install-ruvllm-pi-worker.sh` (mirror of `install.sh`,
  reuse `ruvector-worker` user pattern but new state dir
  `/var/lib/ruvllm/`)
- Run scaffold-version on a Pi, confirm it accepts a TCP connection
  on `:50053`. No model loading yet — just prove the deploy pipeline.

## Iter 4 (2026-05-04, ~20:42)

**Done:**
- `deploy/ruvllm-pi-worker.service` (systemd unit, mirrors hailo
  worker hardening: NoNewPrivileges, ProtectSystem=strict, MemoryMax=4G,
  TasksMax=64, runs as `ruvllm-worker`)
- `deploy/ruvllm-pi-worker.env.example` (env contract for iters 5+)
- `deploy/install-ruvllm-pi-worker.sh` (idempotent installer, mirrors
  install.sh for the embed worker)
- aarch64 binary rsync'd to all 4 Pis
- Installed + service started on all 4 Pis
- TCP probe returns version banner from each `:50053` port

**Issues fixed:**
- systemd's `MemoryDenyWriteExecute=no` line had an inline `#` comment
  on the same line — systemd doesn't parse those, warns on parse.
  Moved the comment to its own line.

**Cluster state:**
- 4× Pi 5 + AI HAT+ each running TWO worker services:
  - `:50051` ruvector-hailo-worker (embeddings, NPU)
  - `:50053` ruvllm-pi-worker (scaffold; LLM completions, soon)

**Iter 5 plan:**
- Wire `ruvllm::serving::ServingEngine` into `ruvllm-pi-worker`. Need:
  - A `LlmBackend` impl (probably reuse `crates/ruvllm/src/backends/`
    candle one, but call it with already-on-disk weights — no hf-hub)
  - Tokenizer load from local path
  - First test: Llama-3.2-1B fp16 (no quantization) — get one token
    out, prove the engine wires. Quantization layered after.
- Stage Llama-3.2-1B from ruvultra's HuggingFace cache to Pi via rsync.

## Iter 5–7 (2026-05-04 ~22:50 → ~23:10)

**Substitution decided:**
- `Llama-3.2-1B` requires HF license accept (token not configured on
  ruvultra). Cached models available locally (`~/.cache/huggingface/hub/`):
  - `Qwen2.5-0.5B-Instruct` (954 MB, smallest — chosen as engine-wiring proof)
  - `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `TinyLlama-1.1B-Chat-v1.0`,
    `Phi-4-mini-instruct`
- **Qwen2.5-0.5B substitutes for Llama-3.2-1B** in iter 5–8. Llama-3.2-1B
  re-enters scope post-engine-wiring once we configure an HF token.
- cognitum-v0 has only **1.8 GB free root** (the original SD card,
  pre-clone) — too tight for 940 MB model + KV; skip it for now,
  stage on cluster-1/2/3 only (each 29 GB free).

**Rsync challenges:**
- Iter 5 first attempt — parallel rsync from 3 background tasks
  collided in `/tmp/qwen2.5-0.5b/` and over WiFi. Slow (~5 MB/s/Pi).
- Iter 6 cleanup — `pkill -f "rsync.*qwen2.5-0.5b"` matched its own
  command line, killing the parent bash + all backgrounded tasks
  (exit 144). Foot-gun documented.
- Iter 7 (this one) — sequential rsync via background `b13vuf2ct`,
  uses `--partial` so cluster-1's 320 MB partial resumes.

**Files staged (one-shot when rsync finishes):**
- `/var/lib/ruvllm/models/qwen2.5-0.5b/{config,tokenizer,model.safetensors,...}`
  on cluster-1, cluster-2, cluster-3.

**Iter 8 plan (waiting on rsync):**
- Update `/etc/ruvllm-pi-worker.env` on each cluster Pi to point
  `RUVLLM_MODEL_PATH=/var/lib/ruvllm/models/qwen2.5-0.5b/model.safetensors`.
- Wire `ruvllm::serving::ServingEngine` + a `LlmBackend` that loads
  from this local path. The candle backend's `get_safetensors_files`
  takes `&hf_hub::api::sync::ApiRepo` — need a thin local-path
  adapter or a different backend entry point.
- Bring up engine with `RUVLLM_QUANTIZE=none` (fp16 first to prove
  pipeline). Quantization layered after.
