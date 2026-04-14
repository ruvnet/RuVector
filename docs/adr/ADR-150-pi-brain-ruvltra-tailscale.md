# ADR-150: π Brain + RuvLtra via Tailscale — Semantic Embedding Upgrade

## Status

Proposed

## Date

2026-04-14

## Context

The pi.ruv.io brain currently uses `ruvllm::HashEmbedder` (FNV-1a + bigrams, 128-dim) as its default embedding engine. When the corpus exceeds 50 documents, it upgrades to `RlmEmbedder` (contextual neighbor-weighted re-embedding using the local corpus). Both are **hash-based** — they don't understand semantics.

A query for `"seizure detection"` and a stored memory titled `"epilepsy prediction via graph mincut"` are semantically related but share few tokens. The hash embedder returns low similarity, the memory doesn't surface, and the brain's most valuable knowledge stays buried.

We already have a production-ready transformer in the workspace: **RuvLtra** (`crates/ruvllm/src/models/ruvltra.rs`) — a Qwen-0.5B-based model with SONA continuous learning integration, ANE-optimized for Apple Silicon. It produces 896-dim semantic embeddings and can run locally on a Mac Mini or laptop with 4-8 GB RAM.

The brain runs on Cloud Run (ephemeral, no GPU, 2 GB RAM limit for the container). We cannot run a 1B parameter model there. But we have a local Mac Mini (`ruv-mac-mini.local`) with Apple Silicon that already has RuvLtra loaded and running.

**Tailscale** provides a zero-config mesh VPN that assigns each device a stable 100.64.x.x address. The brain on Cloud Run can reach the Mac Mini's Tailscale IP over a private, encrypted tunnel.

## Decision

Offload embedding generation from the Cloud Run brain to a local RuvLtra inference server running on the Mac Mini, connected via Tailscale. The brain falls back to `HashEmbedder` when the Tailscale endpoint is unreachable.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    pi.ruv.io (Cloud Run)                  │
│                                                           │
│  EmbeddingEngine::embed(text)                            │
│       │                                                   │
│       ├──[Tailscale available?]                          │
│       │                                                   │
│       ├──YES──► HTTP POST to ruvltra endpoint            │
│       │        at http://100.64.x.x:8090/embed          │
│       │                                                   │
│       └──NO──►  HashEmbedder fallback (existing)         │
│                                                           │
└─────────────────────────┬────────────────────────────────┘
                          │
                    Tailscale mesh VPN
                    (encrypted, P2P)
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│             Mac Mini (ruv-mac-mini.local)                 │
│                                                           │
│  ruvltra-embed-server (new binary in crates/ruvllm):     │
│    - Listens on :8090 (Tailscale interface only)         │
│    - Loads ruvltra-small-q4.gguf (~662 MB, ANE-accel)    │
│    - POST /embed {text: "..."} → {vec: [896 f32]}        │
│    - SONA continuous learning on query patterns          │
│                                                           │
│  System: macOS, M-series, 16+ GB RAM, always-on          │
│  Startup: `cargo run --release -p ruvllm --bin            │
│            ruvltra-embed-server`                          │
└──────────────────────────────────────────────────────────┘
```

### What We Already Have

| Component | Location | Status |
|---|---|---|
| RuvLtra model | `crates/ruvllm/src/models/ruvltra.rs` | Implemented, ANE-optimized |
| GGUF loader | `crates/ruvllm/src/gguf/` | Loads 4-bit quantized weights |
| SONA integration | `crates/ruvllm/src/sona/` | Continuous learning on queries |
| Hub registry | `crates/ruvllm/src/hub/registry.rs` | Downloads models from HF |
| Brain embedder | `crates/mcp-brain-server/src/embeddings.rs` | Hash + RLM fallback chain |

### What We Need to Build

1. **`ruvltra-embed-server` binary** (~300 lines) in `crates/ruvllm/src/bin/`
   - Axum HTTP server on port 8090
   - Loads RuvLtra with `Q4` quantization (~662 MB RAM)
   - Endpoint: `POST /embed` → mean-pooled last-hidden-state
   - Endpoint: `GET /health` → model loaded + token count
   - SONA learns which queries are most frequent

2. **`TailscaleEmbedder` in brain** (~150 lines) in `crates/mcp-brain-server/src/embeddings.rs`
   - New variant added to the embedder chain
   - Reads `RUVLTRA_EMBED_URL` env var (e.g., `http://100.64.1.5:8090`)
   - Falls back to `RlmEmbedder` on timeout or connection error
   - Caches embeddings for identical input text (LRU, 1000 entries)

3. **Tailscale deployment** (no code, just setup)
   - Install Tailscale on Mac Mini: `brew install tailscale && tailscale up`
   - Install Tailscale on Cloud Run: use the [Tailscale Cloud Run integration](https://tailscale.com/kb/1242/tailscale-serve) with a sidecar or Tailscale Kubernetes operator
   - Cloud Run env var: `RUVLTRA_EMBED_URL=http://ruv-mac-mini:8090`

### Embedding Upgrade Path

The brain currently has 10,764 memories with 128-dim HashEmbedder embeddings. Migration strategy:

| Phase | Action | Duration |
|---|---|---|
| **Phase 1** | Add TailscaleEmbedder as new variant, keep HashEmbedder as default | 1 day |
| **Phase 2** | Route new writes through RuvLtra (896-dim), store alongside old embeddings | Week 1 |
| **Phase 3** | Background job re-embeds old memories (10K memories × ~50ms = ~9 min) | Single run |
| **Phase 4** | Switch search path to 896-dim embeddings, deprecate 128-dim | Week 2 |
| **Phase 5** | Delete old 128-dim embeddings from Firestore | Week 3 |

### Performance Expectations

| Metric | Current (HashEmbedder) | New (RuvLtra via Tailscale) |
|---|---|---|
| Embedding dimension | 128 | 896 (7x more semantic info) |
| Embedding latency | 0.5 ms (local, CPU) | ~50 ms (Tailscale + ANE inference) |
| Semantic quality | ~0.3 NDCG@10 (hash) | ~0.85 NDCG@10 (transformer) |
| Memory per embedding | 512 bytes | 3,584 bytes |
| Cloud Run memory | N/A | Same (inference offloaded) |
| Mac Mini memory | 0 | ~1.5 GB (Q4 model + KV cache) |

### Why This Works

1. **No Cloud Run resource increase.** Model runs on the Mac Mini, which already exists and has ANE.
2. **Network is fast enough.** Tailscale is peer-to-peer (~1-5 ms within same city) and encrypted. 50 ms total latency per embedding is acceptable — embeddings are generated once on write, not on every query.
3. **Graceful degradation.** If the Mac Mini is offline or Tailscale is unreachable, the brain falls back to HashEmbedder. No outage, just degraded semantics.
4. **SONA learns what matters.** The embed server tracks query patterns and fine-tunes LoRA adapters to the brain's actual workload — semantics improve over time without manual training.
5. **Cost: effectively $0.** Tailscale is free for personal use (up to 100 devices). The Mac Mini runs 24/7 anyway. No API fees.

### Security Considerations

- **Tailscale ACLs:** Lock down the `:8090` port to ONLY the Cloud Run service's Tailscale node. No public internet exposure.
- **API key:** Embed server requires `X-Brain-Auth` header matching the `BRAIN_SYSTEM_KEY` from gcloud secrets. Defense in depth.
- **Rate limiting:** Max 100 embeddings/second per client to prevent runaway brain loops.
- **PII scrubbing:** RVF_PII_STRIP is already enabled on the brain. Embeddings don't store raw text, just vectors.
- **Audit log:** Every embed request logged with timestamp + hash of input (not raw text). Visible in Firestore.

### Alternatives Considered

1. **Run RuvLtra on Cloud Run directly.** Rejected: requires 4+ GB RAM, 30+ s cold start, and no ANE acceleration. Cost would jump 10x.

2. **Use Gemini API for embeddings.** Rejected: ~$0.00013 per 1K tokens = ~$2/day at current rate, and no SONA learning.

3. **Run local model on user's laptop.** Rejected: laptops aren't always online. Mac Mini is a stable always-on endpoint.

4. **Self-host on a cheap VPS.** Rejected: no GPU/ANE, 5-10x slower inference, and $20-40/month. The Mac Mini is free capacity.

## Consequences

### Positive
- **7x richer embeddings** (128 → 896 dims) — semantic search becomes actually semantic
- **No Cloud Run resource increase** — inference offloaded to existing hardware
- **SONA continuous learning** — embeddings improve with every query
- **Zero new infrastructure cost** — Tailscale is free, Mac Mini already exists
- **Graceful degradation** — HashEmbedder fallback means no outages

### Negative
- **Single point of failure** — if Mac Mini goes offline, embeddings degrade (but brain keeps running)
- **Added latency** — 50 ms per embedding instead of 0.5 ms (acceptable on write path)
- **Operational complexity** — now managing a local server + Tailscale mesh
- **Migration cost** — 10,764 memories must be re-embedded (one-time, ~10 min)

### Risks
- **Tailscale downtime** (rare but possible): mitigated by HashEmbedder fallback
- **Mac Mini hardware failure**: mitigated by SSH keys + documented setup; can rebuild in ~30 min
- **Embedding distribution shift**: mitigated by keeping old embeddings during transition

## Implementation Plan

### Week 1: Server side
- Day 1-2: Build `ruvltra-embed-server` binary with /embed and /health endpoints
- Day 3: Deploy to Mac Mini, verify inference latency (~50 ms)
- Day 4: Install Tailscale on Mac Mini, note assigned 100.64.x.x IP
- Day 5: Install Tailscale sidecar on Cloud Run (via init container pattern)

### Week 2: Brain side
- Day 1-2: Add `TailscaleEmbedder` variant in `embeddings.rs`
- Day 3: Update `EmbeddingEngine::embed` to try Tailscale first, fall back to HashEmbedder
- Day 4: Add env vars `RUVLTRA_EMBED_URL` and `RUVLTRA_EMBED_KEY` to Cloud Run
- Day 5: Deploy, verify new memories use 896-dim embeddings

### Week 3: Migration
- Day 1: Build `re-embed-corpus` admin endpoint that re-embeds all memories
- Day 2: Run migration (~10 min for 10K memories)
- Day 3: Switch search to use new embeddings
- Day 4-5: Monitor, tune, document

## References

- RuvLtra source: `crates/ruvllm/src/models/ruvltra.rs`
- Brain embedder: `crates/mcp-brain-server/src/embeddings.rs`
- Hub registry: `crates/ruvllm/src/hub/registry.rs`
- SONA integration: ADR-067 SONA Self-Optimizing Neural Architecture
- Tailscale docs: https://tailscale.com/kb/
- Tailscale on Cloud Run: https://tailscale.com/kb/1242/tailscale-serve
- ADR-149: Brain performance optimizations (pre-requisite — embeddings must be pre-normalized)
