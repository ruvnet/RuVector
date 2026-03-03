# ADR-080: npx ruvector Deep Capability Audit

**Status:** Accepted
**Date:** 2026-03-03
**Author:** ruvnet

## Context

The `ruvector` npm package (v0.2.5) is the primary CLI and MCP entry point for the ruvector ecosystem, providing `npx ruvector` access to vector database operations, self-learning hooks, brain AGI subsystems, edge compute, and 91+ MCP tools. This ADR documents a comprehensive audit of all capabilities, coverage gaps, and security findings.

## Package Overview

| Field | Value |
|-------|-------|
| **Package** | `ruvector` on npm |
| **Version** | 0.2.5 |
| **CLI entry** | `bin/cli.js` (8,911 lines) |
| **MCP entry** | `bin/mcp-server.js` (~3,816 lines) |
| **Node.js** | >=18.0.0 |
| **Dependencies** | 8 required, 1 optional, 3 peer (optional) |
| **Published files** | `bin/`, `dist/`, `README.md`, `LICENSE` |

## CLI Inventory

### Summary

- **Total commands**: ~179 registered, ~145 unique
- **Command groups**: 14 main groups + standalone commands
- **Lazy-loaded modules**: GNN, Attention, ora, ruvector core, pi-brain, ruvllm
- **Startup time**: ~55ms (lazy loading optimization)

### Command Groups (14)

| Group | Subcommands | Description |
|-------|-------------|-------------|
| **hooks** | 55 | Self-learning intelligence hooks — routing, memory, trajectories, AST, diff, coverage, compression, learning algorithms |
| **brain** | 22 | Shared intelligence — search, share, vote, sync, AGI subsystems (SONA, GWT, temporal, meta-learning, midstream) |
| **workers** | 14 | Background analysis — dispatch, presets, phases, custom workers |
| **rvf** | 11 | RuVector Format — create, ingest, query, derive, segments, examples, download |
| **sona** | 6 | SONA adaptive learning — status, patterns, train, export |
| **embed** | 5 | Embeddings — text, adaptive LoRA, ONNX, neural, benchmark |
| **attention** | 5 | Attention mechanisms — compute, benchmark, hyperbolic, list |
| **edge** | 5 | Distributed P2P compute — status, join, balance, tasks, dashboard |
| **native** | 4 | Native ONNX/VectorDB workers — run, benchmark, list, compare |
| **mcp** | 4 | MCP server — start, info, tools, test |
| **gnn** | 4 | Graph Neural Networks — layer, compress, search, info |
| **identity** | 4 | Pi key management — generate, show, export, import |
| **llm** | 4 | LLM embeddings/inference via ruvllm |
| **midstream** | 4 | Real-time streaming — status, attractor, scheduler, benchmark |
| **route** | 3 | Semantic routing — classify, benchmark, info |

### Standalone Commands (15)

`create`, `insert`, `search`, `stats`, `benchmark`, `info`, `install`, `graph`, `router`, `server`, `cluster`, `export`, `import`, `doctor`, `setup`

### Stub/Coming-Soon Commands (4)

| Command | Status | Note |
|---------|--------|------|
| `router` | Coming Soon | npm package in development |
| `server` | Coming Soon | HTTP/gRPC server planned |
| `cluster` | Coming Soon | Distributed cluster planned |
| `graph` | Requires @ruvector/graph-node | Optional package not installed by default |

### External API Commands

| Commands | Service | URL |
|----------|---------|-----|
| `brain *` (16 commands) | pi.ruv.io | `https://pi.ruv.io` |
| `brain agi *` (6 commands) | pi.ruv.io AGI endpoints | `/v1/sona`, `/v1/temporal`, `/v1/explore`, `/v1/midstream` |
| `edge *` (5 commands) | Edge genesis node | Cloud Run endpoint |
| `midstream attractor` | pi.ruv.io | `/v1/midstream` |
| `rvf download` | GCS + GitHub | Storage + raw GitHub |

## MCP Server Inventory

### Summary

- **Total tools**: 91 (base) + 12 (AGI/midstream) = 103 registered inputSchemas
- **Transport modes**: stdio (default), SSE (HTTP)
- **Version**: 0.2.5 (hardcoded in 2 locations)

### Tool Groups (9)

| Group | Tools | Description |
|-------|-------|-------------|
| **hooks** | 49 | Intelligence, memory, routing, learning, compression, AST, diff, coverage, security, RAG |
| **workers** | 12 | Background analysis dispatch, presets, phases, custom workers |
| **rvf** | 10 | Vector store CRUD, compact, derive, segments, examples |
| **brain** | 11 | Shared knowledge search, share, vote, sync, partition, transfer |
| **brain_agi** | 6 | AGI diagnostics — SONA, temporal, explore, midstream, flags |
| **midstream** | 6 | Real-time analysis — status, attractor, scheduler, benchmark, search, health |
| **edge** | 4 | Distributed compute — status, join, balance, tasks |
| **rvlite** | 3 | SQL/Cypher/SPARQL query engines over vector data |
| **identity** | 2 | Pi key generation and display |

### Stub Tools (~6 of 91, ~7%)

`hooks_attention_info`, `hooks_gnn_info`, `workers_triggers`, `workers_presets`, `workers_phases` — return hardcoded fallback data when packages unavailable. Brain AGI tools require external service.

### Functional Tools (~85 of 91, ~93%)

All hooks intelligence, RVF CRUD, brain services, edge network, identity crypto, worker dispatch, and query engine tools have real implementations.

## Security Findings

### Strong Defenses

| Defense | Coverage |
|---------|----------|
| **Path validation** (`validateRvfPath()`) | All RVF tools — null byte check, realpath resolution, CWD confinement, blocked system paths |
| **Shell sanitization** (`sanitizeShellArg()`) | All hooks/workers using execSync — removes metacharacters, backticks, `$()`, pipes, semicolons |
| **Numeric validation** (`sanitizeNumericArg()`) | Hooks/workers with numeric args — parseInt with NaN fallback |
| **Null byte defense** | Both path and shell sanitizers strip `\0` |
| **Chalk ESM fix** | Consistent `_chalk.default \|\| _chalk` pattern at line 7-8 |

### Concerns (10 findings)

| # | Finding | Severity | Location |
|---|---------|----------|----------|
| 1 | execSync with shell invocation despite sanitization | Medium | hooks_init, hooks_pretrain, analysis tools |
| 2 | Intelligence data load/save paths not validated by `validateRvfPath()` | Medium | mcp-server.js lines 171-191 |
| 3 | No fetch timeout on brain/edge/midstream API calls | Medium | Could hang/DoS |
| 4 | No rate limiting on external API calls | Medium | Brain, edge, midstream tools |
| 5 | Environment variable values used unsanitized in fetch/crypto | Medium | BRAIN_URL, PI, EDGE_GENESIS_URL |
| 6 | Pi key prefix logged in responses | High | identity_show, mcp-server.js line 3555 |
| 7 | No limits on vector dimensions or query result sizes | Medium | rvf_create, rvf_query, rvlite_sql |
| 8 | 51% of MCP tools lack input validation | Medium | hooks_remember, hooks_recall, brain tools |
| 9 | workers_dispatch returns `success: true` on error | Low | mcp-server.js line 2730 |
| 10 | Inconsistent `isError` flag usage across tools | Low | Error response formatting |

## Test Coverage Analysis

### Test Suite

| File | Tests | Quality |
|------|-------|---------|
| `test/cli-commands.js` | 63 active + 6 dynamic | Mixed — many help-only |
| `test/integration.js` | 6 test groups | Good — module, types, structure |
| `test/benchmark-cli.js` | 7 benchmark commands | Good — latency + lazy loading |

### Coverage Matrix

| Capability | CLI Test | Integration Test | Benchmark |
|-----------|----------|-----------------|-----------|
| create/insert/search/stats | **None** | **None** | **None** |
| GNN operations | Help only | No | No |
| Attention operations | Help only | No | No |
| Hooks routing/memory | Basic | No | No |
| Brain AGI commands | Help only | No | No |
| Midstream commands | Help only | No | No |
| Module loading | No | Yes | No |
| Type definitions | No | Yes | No |
| MCP tool count | No | Yes (103) | No |
| CLI startup latency | No | No | Yes (<100ms budget) |
| Lazy loading overhead | No | No | Yes |

### Critical Gaps

1. **No functional database tests** — `create`, `insert`, `search`, `stats` are the primary documented use case but have zero test coverage
2. **Performance claims unvalidated** — "sub-millisecond queries", "52,000 inserts/sec", "150x HNSW speedup" have no benchmarks
3. **MCP tool functionality untested** — only tool count validated, not individual tool behavior
4. **Brain AGI connectivity untested** — commands only tested for `--help` output

## Code Quality

### Strengths

- Well-organized 14-group command hierarchy
- Consistent lazy-loading pattern (GNN, Attention, ora, ruvector core)
- Graceful degradation when optional packages missing
- Version sourced from package.json (not hardcoded in cli.js)
- Comprehensive hooks system (55 subcommands covering full dev lifecycle)
- RVF path validation is thorough

### Issues

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | Dead code in router command (unreachable block) | Low | cli.js line 1807 |
| 2 | brain page/node actions return "not yet available" | Low | cli.js lines 8120-8180 |
| 3 | Uninitialized variables in conditional blocks | Low | cli.js lines 4757, 4769 |
| 4 | Error suppression in brain/edge catch blocks | Low | cli.js lines 7907-7908 |

## Decision

Document findings and prioritize fixes:

### P0 — Security (address before next publish)
- Add fetch timeout (30s) to all external API calls (brain, edge, midstream)
- Stop logging Pi key prefix in identity_show responses
- Add `validateRvfPath()` to intelligence data load/save paths

### P1 — Test Coverage (next sprint)
- Add functional tests for `create`, `insert`, `search`, `stats` commands
- Add MCP tool functional tests (at least one per group)
- Add connectivity test for brain AGI endpoints (mock or live)

### P2 — Code Quality (backlog)
- Remove dead code in router command
- Add input validation to remaining 51% of MCP tools
- Add resource limits (max dimensions, max result count)
- Fix workers_dispatch error reporting

### P3 — Documentation (backlog)
- Add performance benchmarks to validate README claims
- Mark stub commands more clearly in README
- Document external service dependencies and fallback behavior

## Consequences

- Full visibility into the 145-command, 91-tool npm package surface area
- 10 security findings documented with severity and fix priority
- Test coverage gaps identified — core database operations completely untested
- Clear prioritized action plan for hardening before next publish

## Appendix: Dependency Tree

### Required
```
@modelcontextprotocol/sdk ^1.0.0
@ruvector/attention        ^0.1.3
@ruvector/core             ^0.1.25
@ruvector/gnn              ^0.1.22
@ruvector/sona             ^0.1.4
chalk                      ^4.1.2  (CJS compat via .default || fallback)
commander                  ^11.1.0
ora                        ^5.4.1  (lazy loaded)
```

### Optional
```
@ruvector/rvf              ^0.1.0
```

### Peer (all optional)
```
@ruvector/pi-brain         >=0.1.0  (brain commands)
@ruvector/ruvllm           >=2.0.0  (llm commands)
@ruvector/router           >=0.1.0  (router command, not yet published)
```

### External Services
```
https://pi.ruv.io               — Brain AGI, midstream (Cloud Run)
edge-net-genesis (Cloud Run)    — Edge compute network
storage.googleapis.com          — RVF examples
raw.githubusercontent.com       — RVF manifest fallback
```
