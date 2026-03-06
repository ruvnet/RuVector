# ADR-083: Brain Server Training Loops — Closing the Store→Learn Gap

**Status**: Accepted
**Date**: 2026-03-03
**Authors**: RuVector Team
**Deciders**: ruv
**Related**: ADR-081 (Brain Server v0.2.8–0.2.10), ADR-082 (Security Hardening)

## 1. Context

After injecting 258 memories, 856 votes, and running 30+ cross-domain transfers, a training audit revealed that the brain server's higher-order learning subsystems were architecturally present but not actively learning:

| Subsystem | Has Code | Was Training | Why Not |
|---|---|---|---|
| SONA (pattern learning) | Yes — gradient, EWC, ReasoningBank | No | `force_learn()` never called; `tick()` only fires on `/v1/status` hits |
| LoRA Federation | Yes — Byzantine-robust median+MAD aggregation | Client-driven | Works as designed; server aggregates client-submitted weights |
| Pareto Frontier | Yes — `evolve_population()` exists | No | `evolve_population()` was never called from any route or background task |
| GWT Workspace | Yes — attention filter | Per-request only | Transient re-ranking, no persistent learning |
| Midstream | Yes — scheduler, solver, strange loop | No | All flags default to `false`; scheduler has zero tasks submitted |
| Training Preferences | Yes — DPO pair export | Export-only | Working as designed; clients consume for offline training |

The gap: the server **stores knowledge** but does not **learn from knowledge**. The missing piece is a training loop that periodically processes accumulated data.

## 2. Decision

### 2.1 Background Training Loop

Added a `tokio::spawn` background task in `main.rs` that runs every 5 minutes:

- Waits 60 seconds after startup (let data load complete)
- Every 5 minutes, checks if new memories or votes have arrived
- If any new data exists, runs `run_training_cycle()`:
  1. SONA `force_learn()` — drains trajectory buffer, extracts patterns via k-means, applies EWC++ constraints
  2. Domain `evolve_population()` — records policy kernels into Pareto front, evolves population

### 2.2 Explicit Training Endpoint

Added `POST /v1/train` for on-demand training:

- Authenticated (requires valid API key)
- Runs the same `run_training_cycle()` as the background loop
- Returns `TrainingCycleResult` with SONA patterns, Pareto growth, memory/vote counts

### 2.3 CLI Command

Added `ruvector brain train`:
- Calls `POST /v1/train`
- Displays SONA message, pattern count, Pareto growth, memory/vote counts
- Supports `--json` flag

### 2.4 MCP Tool

Added `brain_train` MCP tool for agent-triggered training.

### 2.5 Vote Dedup Refinements (ADR-082 follow-up)

- **Author exemption**: Content authors now bypass IP vote dedup (self-votes are already blocked by store-level check)
- **24h TTL**: Vote dedup entries expire after 24 hours and are evicted during periodic cleanup

## 3. Results

After deploying, 3 training cycles produced:

| Metric | Before | After |
|--------|--------|-------|
| Pareto frontier size | 0 | 24 |
| SONA patterns | 0 | 0 (needs 100 trajectories minimum) |
| Domain population | Static | Evolving with fitness tracking |

SONA will begin extracting patterns once 100+ search/share operations accumulate trajectories (its minimum threshold for k-means clustering).

## 4. Files Modified

| File | Changes |
|------|---------|
| `crates/mcp-brain-server/src/main.rs` | Background training loop (tokio::spawn, 5 min interval) |
| `crates/mcp-brain-server/src/routes.rs` | `POST /v1/train` endpoint, `run_training_cycle()` function, `create_router()` returns `(Router, AppState)` |
| `crates/mcp-brain-server/src/types.rs` | `TrainingCycleResult` struct |
| `crates/mcp-brain-server/src/rate_limit.rs` | 24h TTL on vote dedup entries, cleanup in `maybe_cleanup()` |
| `npm/packages/ruvector/bin/cli.js` | `brain train` command |
| `npm/packages/ruvector/bin/mcp-server.js` | `brain_train` MCP tool |

## 5. What Remains (Future Work)

| Subsystem | Status | Next Step |
|---|---|---|
| SONA | Active, needs volume | Will start learning after ~100 searches (natural usage) |
| LoRA | Working | Clients need to submit computed LoRA updates |
| Pareto | Now growing | Accumulates each training cycle |
| Midstream | Scaffolding | Enable flags + submit scheduler tasks |
| GWT | Working per-request | Consider persistence for cross-session attention |
| Training Prefs | Export working | Build external DPO trainer that consumes this API |
