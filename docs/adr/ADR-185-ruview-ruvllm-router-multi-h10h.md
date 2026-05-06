---
adr: 185
title: "ruview-ruvllm-router — multi-backend LLM routing across heterogeneous cognitum hardware"
status: accepted
date: 2026-05-05
authors: [ruvnet, claude-flow]
related: [ADR-183, ADR-184]
hardware: yes
---

# ADR-185 — ruview-ruvllm-router: multi-backend LLM routing

## Status

**Accepted.** Second Hailo-10H (AI HAT+ 2) installed on cognitum-v0 (2026-05-05).
This ADR documents the routing layer that optimises LLM serving across all cluster
hardware configurations.

---

## Context

After ADR-184 deployed `ruview-ruvllm-h10` on cluster-3, the cluster gained a second
Hailo-10H on cognitum-v0. A static per-node service works for a single node, but:

- Two H10H nodes means double the concurrent LLM capacity.
- v0 is the brain node — local LLM avoids the Tailscale RTT for every streaming token.
- Future nodes may have H8 (embedding only), H10H, or no NPU.
- The brain's `RUVIEW_LLM_BACKEND` env var points to a single endpoint — a router
  gives the brain a stable single address regardless of backend topology.

### Hardware inventory (2026-05-05)

| Node | IP | NPU | Role |
|---|---|---|---|
| cognitum-v0 | 100.77.59.83 | **Hailo-10H** (new) | brain + LLM backend |
| cognitum-cluster-1 | 100.80.54.16 | Hailo-8 | CSI embedding only |
| cognitum-cluster-2 | 100.77.220.24 | Hailo-8 | CSI embedding only |
| cognitum-cluster-3 | 100.73.75.53 | **Hailo-10H** (ADR-184) | LLM backend |

### Supported hardware configurations

The router handles all combinations automatically via health checks:

| Scenario | Router behaviour |
|---|---|
| Both H10H nodes up | least-busy routing across v0 + cluster-3 |
| cluster-3 down | all traffic to v0 (transparent failover) |
| v0 H10H down | all traffic to cluster-3 (transparent failover) |
| Both down | 503 with clear error |
| H8 nodes (cluster-1/2) | not LLM-capable; excluded from router |

---

## Decision

New crate `ruview-ruvllm-router` on cognitum-v0:
- Listens on gRPC `:50060` and HTTP `:8882`
- Pools configured backends (comma-separated env var)
- Routes each `Generate` request to the least-busy healthy backend
- Health-checks backends every 30 s; marks unavailable ones unhealthy
- HTTP `/backends` endpoint exposes pool status for monitoring
- Brain updated to `RUVIEW_LLM_BACKEND=grpc://100.77.59.83:50060` (local, no Tailscale hop)

### Architecture

```
cognitum-v0 (brain node)
├── ruview-mcp-brain-mini.service       (:9876 HTTP)
│   └── RUVIEW_LLM_BACKEND=grpc://127.0.0.1:50060
├── ruview-ruvllm-h10.service  [NEW]    (:50058 gRPC, :8880 HTTP)  ← local H10H backend
├── ruview-ruvllm-router.service [NEW]  (:50060 gRPC, :8882 HTTP)  ← router
│   ├── backend[0]: 127.0.0.1:50058    (v0 local H10H — 0ms latency)
│   └── backend[1]: 100.73.75.53:50058 (cluster-3 via Tailscale — ~1ms)
│
cognitum-cluster-3
└── ruview-ruvllm-h10.service           (:50058 gRPC, :8880 HTTP)  ← existing H10H backend
```

### Routing algorithm

**Least-busy**: select the healthy backend with the fewest concurrent `active` requests.

- Both idle → pick backend[0] (v0 local, zero RTT)
- v0 busy (active=1) + cluster-3 idle (active=0) → route to cluster-3
- v0 unhealthy → all to cluster-3 automatically
- All unhealthy → 503

This is a simple, correct algorithm. It does not require central coordination and
degrades gracefully under partial failure.

### Performance impact

| Metric | ADR-184 (single node) | ADR-185 (router + 2 nodes) |
|---|---|---|
| tok/s (single request) | ~8 tok/s | ~8 tok/s (same per backend) |
| tok/s (2 concurrent requests) | ~4 tok/s each | ~8 tok/s each (separate backends) |
| Brain LLM latency (first chunk) | ~1ms Tailscale RTT | ~0ms (local backend first) |
| Availability | single-node SPOF | 2-node HA (failover < 30s) |

---

## Implementation Plan

| Iter | Milestone |
|---|---|
| 1 | Install H10H driver + packages on cognitum-v0 (blacklist H8) |
| 2 | Copy hailo-ollama binary + library symlink; verify `/dev/hailo0` |
| 3 | Pull `llama3.2:1b` model to v0 |
| 4 | Deploy `ruview-ruvllm-h10` on v0 (port :50058, loopback HTTP) |
| 5 | Build + deploy `ruview-ruvllm-router` on v0 |
| 6 | Update brain `RUVIEW_LLM_BACKEND` → `grpc://127.0.0.1:50060` |
| 7 | Smoke test router: `/health` shows 2/2 backends; generate round-trips both |
| 8 | Update `cluster-smoke-test.sh` — add router assertions |
| 9 | Update ADR-183 smoke test count |
| 10 | Commit + update PR #425 |

---

## Implementation Log

| Iter | Status | Notes |
|---|---|---|
| 1 | ✅ done | H10H detected (`1e60:45c4`); `hailo_pci` blacklisted; `hailo1x_pci` loaded; `/dev/hailo0` present |
| 2 | ✅ done | hailo-ollama copied from cluster-3; libhailort.so.5.1.1 installed; 5.2.0 ABI symlink in `aarch64-linux-gnu/`; binary resolves |
| 3 | ✅ done | Blob (1.875 GB) rsync'd via Tailscale (hailo-ollama auto-download stalled; manual rsync --append succeeded) |
| 4 | ✅ done | `ruview-ruvllm-h10` built (aarch64), installed, service unit + env deployed; `hailo_ok=True` on v0 |
| 5 | ✅ done | `ruview-ruvllm-router` crate compiled, deployed on v0 `:50060`/`:8882` |
| 6 | ✅ done | brain `RUVIEW_LLM_BACKEND=grpc://127.0.0.1:50060`; brain-mini env updated |
| 7 | ✅ done | router `/health` shows 2/2 backends healthy (v0 local + cluster-3 via Tailscale) |
| 8 | ✅ done | cluster-smoke-test.sh: **38/38 PASS** (ADR-183/184/185 + ADR-018 CSI bridge + H8 worker) |
| 9 | ✅ done | smoke test updated to 38 assertions (iter 13) |
| 10 | ✅ done | committed to feat/realtime-dense-pointcloud (PR #425) |

---

## Alternatives Considered

| Alternative | Reason Rejected |
|---|---|
| Point brain at cluster-3 directly (no router) | v0's local H10H unused; no failover |
| DNS round-robin | No health checking; sends traffic to dead backends |
| Envoy/nginx proxy | External dependency; 20MB+ binary for a Pi cluster |
| Speculative decoding (draft on v0, verify on cluster-3) | Both nodes run same model (llama3.2:1b) — same vocabulary, no gain; needs larger verifier |

---

## Acceptance Criteria

- [x] `/dev/hailo0` present on v0 after reboot
- [x] `ruview-ruvllm-h10` running on v0 (gRPC :50058, HTTP :8880)
- [x] `llama3.2:1b` generates tokens on v0 H10H
- [x] `ruview-ruvllm-router` running on v0 (gRPC :50060, HTTP :8882)
- [x] Router `/health` shows 2/2 backends healthy
- [x] Brain `RUVIEW_LLM_BACKEND=grpc://127.0.0.1:50060`
- [x] `cluster-smoke-test.sh` passes with all assertions included (38/38)
