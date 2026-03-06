# ADR-078: npx ruvector Midstream & Brain AGI Integration

**Status**: Proposed
**Date**: 2026-03-03
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-070 (npx ruvector Unified Integration), ADR-076 (AGI Capability Wiring), ADR-077 (Midstream Brain Integration)

## 1. Context

The mcp-brain-server backend at π.ruv.io now has 8 AGI subsystems deployed and operational (ADR-076, ADR-077):

- **SONA** — 3-tier hierarchical learning engine
- **GWT** — Global Workspace Theory attention competition
- **Temporal Delta Tracking** — Knowledge evolution velocity
- **Meta-Learning Exploration** — Thompson Sampling with curiosity/regret
- **Nanosecond Scheduler** — Background task scheduling
- **Temporal Attractor Studio** — Lyapunov exponent analysis for embedding stability
- **Temporal Neural Solver** — Certified temporal predictions with solver gates
- **Strange Loop** — Recursive meta-cognitive reasoning

The backend exposes 5 diagnostic endpoints (`/v1/status`, `/v1/sona/stats`, `/v1/temporal`, `/v1/explore`, `/v1/midstream`) and returns AGI-enriched search results. However, **none of these capabilities are exposed through the `npx ruvector` CLI (167 commands) or the MCP server (118 tools)**.

The existing brain CLI commands (13) and MCP tools (11) were implemented against the Phase 1-7 API surface. They don't surface AGI diagnostics, midstream analytics, or the enriched scoring pipeline metadata.

### Current State

| Layer | Brain Commands | AGI/Midstream Commands |
|-------|---------------|----------------------|
| Backend (mcp-brain-server) | 33 REST endpoints | 5 AGI endpoints, 4 midstream subsystems |
| MCP Client (mcp-brain) | 20 MCP tools | 0 |
| npm CLI (npx ruvector) | 13 brain subcommands | 0 |
| npm MCP (mcp-server.js) | 11 brain MCP tools | 0 |

### Gap

Users cannot:
1. View SONA learning patterns, trajectories, or background tick state
2. Monitor temporal delta velocity or trend
3. Inspect meta-learning regret, curiosity scores, or plateau status
4. View midstream scheduler metrics, attractor analysis, or strange-loop version
5. See AGI scoring layer contributions in search results
6. Toggle midstream feature flags without redeploying

## 2. Decision

Extend `npx ruvector` with 2 new CLI command groups and 12 new MCP tools that surface all AGI and midstream capabilities from the backend.

### 2.1 New CLI Commands

#### `ruvector brain agi` Subcommand Group (6 commands)

```
ruvector brain agi status          # Combined AGI + midstream diagnostics
ruvector brain agi sona            # SONA patterns, trajectories, background ticks
ruvector brain agi temporal        # Temporal delta velocity, trend, total deltas
ruvector brain agi explore         # Meta-learning curiosity, regret, plateau, Pareto
ruvector brain agi midstream       # Scheduler ticks, attractor categories, solver, strange-loop
ruvector brain agi flags           # List current feature flag state from /v1/status
```

All commands support `--json` for machine-readable output and `--url`/`--key` for backend override.

**Example output:**

```
$ npx ruvector brain agi status
┌─────────────────────────────────────┐
│  π.ruv.io AGI Diagnostics          │
├─────────────────────────────────────┤
│  SONA                              │
│    Patterns: 12  Trajectories: 45  │
│    Background ticks: 3             │
│                                     │
│  GWT Attention                     │
│    Workspace load: 0.42            │
│    Avg salience: 0.31              │
│                                     │
│  Temporal                          │
│    Total deltas: 237               │
│    Velocity: 14.0/hr               │
│    Trend: growing                  │
│                                     │
│  Meta-Learning                     │
│    Avg regret: 0.023               │
│    Plateau: learning               │
│                                     │
│  Midstream                         │
│    Scheduler ticks: 1,204          │
│    Attractor categories: 5         │
│    Strange-loop: v0.3.0            │
└─────────────────────────────────────┘
```

#### `ruvector midstream` Command Group (4 commands)

```
ruvector midstream status           # Midstream platform overview
ruvector midstream attractor [cat]  # Lyapunov analysis per category
ruvector midstream scheduler        # Scheduler metrics (ticks, tasks/sec)
ruvector midstream benchmark        # Run latency benchmark against backend
```

### 2.2 New MCP Tools

| Tool Name | Endpoint | Description |
|-----------|----------|-------------|
| `brain_agi_status` | GET /v1/status | Combined AGI diagnostics (SONA + GWT + temporal + meta + midstream fields) |
| `brain_sona_stats` | GET /v1/sona/stats | SONA patterns, trajectories, background ticks |
| `brain_temporal` | GET /v1/temporal | Temporal delta velocity, trend, total deltas |
| `brain_explore` | GET /v1/explore | Meta-learning curiosity, regret, plateau, Pareto |
| `brain_midstream` | GET /v1/midstream | Midstream scheduler, attractor, solver, strange-loop |
| `brain_flags` | GET /v1/status | Extract and display feature flag state |
| `midstream_status` | GET /v1/midstream | Full midstream platform diagnostics |
| `midstream_attractor` | GET /v1/midstream | Attractor categories with Lyapunov exponents |
| `midstream_scheduler` | GET /v1/midstream | Nanosecond scheduler performance metrics |
| `midstream_benchmark` | Multi-endpoint | Run sequential + concurrent latency benchmark |
| `midstream_search` | GET /v1/memories/search | Search with midstream scoring metadata in response |
| `midstream_health` | GET /v1/health + /v1/midstream | Combined health + midstream subsystem check |

### 2.3 Enhanced Brain Commands

Existing commands get optional AGI metadata:

| Command | Enhancement |
|---------|------------|
| `brain search` | Add `--verbose` flag to show per-result AGI scoring breakdown (SONA pattern boost, GWT attention winner, meta curiosity, attractor stability, strange-loop bonus) |
| `brain status` | Include AGI and midstream fields in default output (already returned by backend, just not displayed) |
| `brain share` | Show attractor update status when midstream is enabled |

### 2.4 Response Schema Extension

The backend already returns AGI/midstream fields in `/v1/status`. The CLI and MCP server need to parse and display them:

```json
{
  "sona_patterns": 12,
  "sona_trajectories": 45,
  "gwt_workspace_load": 0.42,
  "gwt_avg_salience": 0.31,
  "knowledge_velocity": 14.0,
  "temporal_deltas": 237,
  "meta_avg_regret": 0.023,
  "meta_plateau_status": "learning",
  "midstream_scheduler_ticks": 1204,
  "midstream_attractor_categories": 5,
  "midstream_strange_loop_version": "0.3.0"
}
```

## 3. Implementation Plan

### Phase 1: CLI `brain agi` Subcommands (cli.js)

**File**: `npm/packages/ruvector/bin/cli.js`

1. Add `agi` subcommand group under existing `brainCmd`:

```javascript
const agiCmd = brainCmd.command('agi')
  .description('AGI subsystem diagnostics — SONA, GWT, temporal, meta-learning, midstream');

agiCmd.command('status')
  .description('Combined AGI + midstream diagnostics from π.ruv.io')
  .option('--json', 'Output as JSON')
  .action(async (opts) => {
    const piBrain = await requirePiBrain();
    const config = getBrainConfig(opts);
    const client = new piBrain.PiBrainClient(config);
    const status = await client.status();
    // Extract and format AGI fields
  });
```

2. Add individual `sona`, `temporal`, `explore`, `midstream`, `flags` subcommands following same pattern.

3. Each command calls the corresponding backend endpoint via `PiBrainClient` and formats output with chalk.

**Estimated changes**: ~200 lines added to cli.js

### Phase 2: CLI `midstream` Command Group (cli.js)

**File**: `npm/packages/ruvector/bin/cli.js`

1. Add new top-level command group:

```javascript
const midstreamCmd = program.command('midstream')
  .description('Midstream real-time streaming analysis platform');
```

2. Add `status`, `attractor`, `scheduler`, `benchmark` subcommands.

3. The `benchmark` command runs sequential + concurrent latency tests:
   - 3 sequential requests to each endpoint (health, status, search, write)
   - 20 concurrent search requests
   - Reports p50/p90/p99 latencies

**Estimated changes**: ~150 lines added to cli.js

### Phase 3: MCP Tools (mcp-server.js)

**File**: `npm/packages/ruvector/bin/mcp-server.js`

1. Add 12 new tool definitions to the tools array:

```javascript
{
  name: 'brain_agi_status',
  description: 'Combined AGI subsystem diagnostics from the shared brain',
  inputSchema: {
    type: 'object',
    properties: {},
  }
},
// ... 11 more tools
```

2. Add handler cases:

```javascript
case 'brain_agi_status':
case 'brain_sona_stats':
case 'brain_temporal':
case 'brain_explore':
case 'brain_midstream':
case 'brain_flags': {
  const { PiBrainClient } = require('@ruvector/pi-brain');
  const client = new PiBrainClient({ url, key });
  const endpointMap = {
    brain_agi_status: 'status',
    brain_sona_stats: 'sona/stats',
    brain_temporal: 'temporal',
    brain_explore: 'explore',
    brain_midstream: 'midstream',
    brain_flags: 'status',
  };
  const subCmd = endpointMap[name];
  const result = await client.raw(`/v1/${subCmd}`);
  // For brain_flags, extract only flag-related fields
  // For brain_agi_status, extract AGI fields
  return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
}
```

**Estimated changes**: ~250 lines added to mcp-server.js

### Phase 4: PiBrainClient Extension (@ruvector/pi-brain)

**File**: `npm/packages/pi-brain/` (or inline in cli.js if pi-brain not available)

Add methods:
- `sonaStats()` — GET /v1/sona/stats
- `temporal()` — GET /v1/temporal
- `explore()` — GET /v1/explore
- `midstream()` — GET /v1/midstream
- `raw(path)` — GET any /v1/* endpoint (generic)

If `@ruvector/pi-brain` is not yet published, implement these as inline fetch calls in cli.js using the existing `getBrainConfig()` pattern.

### Phase 5: Enhanced `brain search --verbose`

**File**: `npm/packages/ruvector/bin/cli.js`

Add `--verbose` flag to `brain search` that displays per-result metadata when available. The backend already includes `quality_score`, `witness_hash`, etc. in search results. The verbose mode would format these prominently:

```
$ npx ruvector brain search "neural embeddings" --verbose
1. Neural Embedding Patterns (pattern)
   Quality: 0.89 | Votes: 12↑ 1↓ | Witness: 3a7f...
   Contributor: anon-abc123 | Created: 2026-03-01
   Tags: neural, embedding, rust
```

### Phase 6: Tests

**File**: `npm/packages/ruvector/test/cli-commands.js`

Add tests for new commands:
- `brain agi status --help` — verify command exists
- `brain agi sona --help` — verify subcommand exists
- `midstream status --help` — verify new command group
- `midstream benchmark --help` — verify benchmark command

**File**: `npm/packages/ruvector/test/integration.js`

Add integration test: verify new MCP tools appear in tool list.

**Estimated**: 12-15 new test cases

## 4. File Summary

| File | Action | Phase | Est. Lines |
|------|--------|-------|-----------|
| `npm/packages/ruvector/bin/cli.js` | Add `brain agi` + `midstream` commands | 1,2,5 | +400 |
| `npm/packages/ruvector/bin/mcp-server.js` | Add 12 MCP tools | 3 | +250 |
| `npm/packages/pi-brain/src/client.ts` | Add AGI/midstream methods | 4 | +60 |
| `npm/packages/ruvector/test/cli-commands.js` | New command tests | 6 | +50 |
| `npm/packages/ruvector/test/integration.js` | MCP tool integration tests | 6 | +30 |
| `npm/packages/ruvector/package.json` | Version bump 0.2.3 → 0.2.4 | 3 | +1 |

## 5. API Surface After Integration

| Layer | Before | After | Delta |
|-------|--------|-------|-------|
| CLI Commands | 167 | 177 | +10 |
| CLI Groups | 12 | 13 | +1 (`midstream`) |
| Brain Subcommands | 13 | 19 | +6 (`agi` group) |
| MCP Tools | 118 | 130 | +12 |
| Brain MCP Tools | 11 | 17 | +6 |

## 6. Backend Endpoints Consumed

All endpoints already exist and are deployed (revision ruvbrain-00071-wp7):

| Endpoint | Auth | New Consumers |
|----------|------|--------------|
| GET /v1/status | Yes | `brain agi status`, `brain agi flags`, `brain_agi_status`, `brain_flags` |
| GET /v1/sona/stats | Yes | `brain agi sona`, `brain_sona_stats` |
| GET /v1/temporal | Yes | `brain agi temporal`, `brain_temporal` |
| GET /v1/explore | Yes | `brain agi explore`, `brain_explore` |
| GET /v1/midstream | Yes | `brain agi midstream`, `midstream status`, `brain_midstream`, `midstream_status` |
| GET /v1/health | No | `midstream benchmark`, `midstream_health` |
| GET /v1/memories/search | Yes | `midstream_search` (with scoring metadata) |

No backend changes required — all endpoints are live and returning the expected JSON schemas.

## 7. Feature Flags

No new environment variables. The CLI/MCP tools read from the existing backend responses. Feature flag state is reported by the backend in `/v1/status` response.

For local development/testing, existing env vars apply:
- `BRAIN_URL` — Override backend URL (default: π.ruv.io Cloud Run URL)
- `BRAIN_API_KEY` — API key for authentication

## 8. Backward Compatibility

- All new commands are additive — no existing commands modified
- `brain status` enhanced output is backward-compatible (new fields appended)
- `brain search --verbose` is opt-in (default output unchanged)
- MCP tools follow existing naming convention (`brain_*`, `midstream_*`)
- `@ruvector/pi-brain` methods are additive (existing API unchanged)
- Version bump is minor (0.2.3 → 0.2.4) — no breaking changes

## 9. Consequences

### Positive

- Users gain full visibility into all 8 AGI subsystems via CLI and MCP
- Claude Code agents can inspect brain learning state (SONA patterns, meta-learning regret) to make informed decisions
- Midstream platform metrics enable monitoring and alerting on knowledge evolution
- `midstream benchmark` provides a standardized latency test for the brain backend
- Feature flag visibility enables debugging deployment issues

### Negative

- cli.js grows by ~400 lines (8,500 → 8,900) — still within single-file budget
- mcp-server.js grows by ~250 lines (3,500 → 3,750) — acceptable
- 12 additional MCP tools increase tool list size (agents may need to filter)
- `@ruvector/pi-brain` gains a dependency on 5 new endpoint paths

### Risks

- Backend `/v1/status` response schema may evolve — CLI should handle missing fields gracefully
- Midstream feature flags defaulting to `false` means `brain agi midstream` shows zeros until enabled
- `midstream benchmark` generates real traffic — should include rate limiting awareness

## 10. Verification

1. `npm test` in `npm/packages/ruvector/` — 55 existing + 12-15 new tests pass
2. `npx ruvector brain agi status --json` returns valid JSON with AGI fields
3. `npx ruvector midstream status --json` returns midstream diagnostics
4. `npx ruvector midstream benchmark` completes within 30s and reports latencies
5. MCP server lists 130 tools (was 118)
6. Claude Code can call `brain_agi_status` tool and receive formatted response
7. `npx ruvector brain search "test" --verbose` shows enhanced output
8. All commands work with `--url` and `--key` overrides
