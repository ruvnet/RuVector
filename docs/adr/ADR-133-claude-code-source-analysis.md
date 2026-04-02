# ADR-133: Claude Code CLI Source Code Analysis

## Status

Deployed (2026-04-02)

## Date

2026-04-02

## Context

Understanding Claude Code's internal architecture is critical for building effective extensions (agents, hooks, skills, MCP servers) and for debugging integration issues. The CLI ships as a minified Bun 1.3.11 Single Executable Application (229 MB) with no public source code, making direct inspection necessary.

## Decision

Perform systematic reverse-engineering analysis of the Claude Code CLI binary using string extraction, pattern matching, and structural analysis. Store findings as structured research documents in `/docs/research/claude-code-rvsource/`.

### Approach: Agentic Jujutsu

Use Claude Code's own agent capabilities (background research agents, parallel tool calls) to analyze itself — the tool examines its own binary.

## Architecture Findings

### Binary Structure

- **Format**: Bun 1.3.11 SEA (Single Executable Application)
- **Size**: 229 MB binary, ~12.8 MB bundled/minified JavaScript
- **Entry**: `cli.js` — single bundle with all application code
- **Runtime**: Bun (not Node.js) with native module support

### Code Metrics (from decompilation)

| Metric | Count |
|--------|-------|
| Classes | 1,557 |
| Functions | 19,464 |
| Async functions | 884 |
| Arrow functions | 23,537 |
| Environment variables | 498 |
| Built-in tools | 25+ |
| Slash commands | 39 |
| Permission modes | 6 |
| Auth providers | 5 |
| Extension points | 13 |

### Core Modules Identified

| Module | Minified Symbol | Role |
|--------|-----------------|------|
| Agent Loop | `s$` | Async generator — recursive `yield*` after tool execution |
| Tool Registry | `XF0` | `validateInput`/`call` interface, 25+ built-in tools |
| Permission Checker | — | 5 modes, sandbox (bubblewrap/seatbelt), managed settings |
| Context Manager | — | Auto-compaction via `clear_tool_uses_20250919` API feature |
| MCP Client | — | stdio/SSE/WebSocket transports, OAuth/OIDC, MCPB bundles |
| Streaming Handler | — | SSE event processing, 3 stream modes |

### Key Architectural Patterns

1. **Recursive generator loop**: The agent loop is an async generator that `yield*`s itself after each tool execution, producing 13 distinct event types
2. **Loopback proxy for MCP**: Tool calls from MCP are proxied to REST endpoints via HTTP loopback on localhost
3. **Deferred tool loading**: MCP tool schemas loaded lazily via `ToolSearch` to keep initial context small
4. **Context compaction**: Automatic context window management with micro-compaction for stale tool results

## Research Output

### Documents (18 files, ~3,000 lines)

| File | Content |
|------|---------|
| `00-index.md` | Master index |
| `01` - `13` | Architecture analysis (overview, tools, agent loop, permissions, MCP, hooks, context, config, subagents, models, telemetry, dependencies, extensions) |
| `14` - `18` | Source code analysis (extraction, core modules, call graphs, class hierarchy, state machines) |

### Extracted Source (6 RVF-header files)

Module extractions in `extracted/` with metadata headers:
- `agent-loop.rvf`, `tool-dispatch.rvf`, `permission-system.rvf`
- `mcp-client.rvf`, `context-manager.rvf`, `streaming-handler.rvf`

**Note**: These use text format with RVF-style metadata headers, not binary RVF containers. Future work: convert to proper RVF cognitive containers with vector embeddings using `rvf-cli`.

### Decompiler Script

`/scripts/claude-code-decompile.sh` — automated extraction tool:
- Finds Claude Code source (npm package or Bun SEA binary)
- Extracts JavaScript via `strings` + pattern matching
- Basic beautification and module splitting
- Generates files with metadata headers

## Consequences

### Positive

- Comprehensive internal reference for building Claude Code extensions
- Call graphs and state machines clarify execution flow
- Decompiler script enables analysis of future versions
- Understanding of tool dispatch enables better MCP server design

### Negative

- Analysis based on minified code — some mappings are speculative
- Findings may become stale as Claude Code updates frequently
- Binary extraction captures string literals well but misses control flow nuances

### Risks

- Decompiled source is for internal research only — respect Anthropic's terms of service
- Minified symbol names (`s$`, `XF0`, `ye`) may change between versions

## References

- [Claude Code CLI documentation](https://docs.anthropic.com/en/docs/claude-code)
- [MCP specification](https://modelcontextprotocol.io)
- Research output: `/docs/research/claude-code-rvsource/`
