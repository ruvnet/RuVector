# Claude Code CLI Source Analysis: Research Index

## Project

Deep-dive reverse engineering of the Claude Code CLI (v2.1.90) internal
architecture, based on binary analysis, string extraction, pattern matching,
and configuration schema examination.

## Methodology

This analysis used "agentic jujutsu" -- the tool analyzed itself by:

1. Locating the binary and extension files on disk
2. Extracting the embedded JavaScript source from the Bun SEA binary
3. Pattern-matching against 12.8 MB of application code for class names,
   function signatures, string literals, and configuration patterns
4. Analyzing the 76-property settings schema
5. Cross-referencing 498 environment variables
6. Mapping tool definitions, hook events, and MCP protocol methods

## Research Documents

| # | Document | Description |
|---|----------|-------------|
| 01 | [Overview and Binary Structure](./01-overview-and-binary-structure.md) | Binary format, installation paths, Bun SEA architecture, version management |
| 02 | [Tool System](./02-tool-system.md) | 25+ built-in tools, MCP tool integration, tool schemas, validation, content block types |
| 03 | [Agent Loop and Execution Flow](./03-agent-loop-and-execution-flow.md) | Entry points, main loop, streaming, conversation management, slash commands, output formats |
| 04 | [Permission System](./04-permission-system.md) | 6 permission modes, permission flow, sandbox integration, managed settings |
| 05 | [MCP Integration](./05-mcp-integration.md) | 4 transports, 13 protocol methods, connection management, OAuth, tool discovery |
| 06 | [Hooks System](./06-hooks-system.md) | 6 hook events, command/HTTP hook types, lifecycle, security controls |
| 07 | [Context and Session Management](./07-context-and-session-management.md) | Token budgets, auto-compaction, session persistence, CLAUDE.md, file checkpointing, prompt caching |
| 08 | [Configuration and Environment](./08-configuration-and-environment.md) | Settings hierarchy, 76 settings, 498 env vars, home directory structure |
| 09 | [Agent and Subagent System](./09-agent-and-subagent-system.md) | Agent types, task/subagent lifecycle, skill system, plugin marketplace |
| 10 | [Models and API](./10-models-and-api.md) | 27+ model IDs, 5 provider backends, API endpoints, prompt caching, effort levels |
| 11 | [Telemetry and Observability](./11-telemetry-and-observability.md) | OpenTelemetry, Datadog, Perfetto, debugging, cost tracking |
| 12 | [Dependency Graph](./12-dependency-graph.md) | Module relationships, data flow, state management, initialization sequence |
| 13 | [Extension Points](./13-extension-points.md) | 13 extension mechanisms from CLAUDE.md to Agent SDK |
| 14 | [Source Extraction](./14-source-extraction.md) | Binary analysis, code metrics, extraction methods, dependency identification |
| 15 | [Core Module Analysis](./15-core-module-analysis.md) | Agent loop, tool dispatch, permissions, context management, MCP, streaming |
| 16 | [Call Graphs](./16-call-graphs.md) | Mermaid call graphs: boot, agent loop, tool dispatch, permissions, MCP, compaction |
| 17 | [Class Hierarchy](./17-class-hierarchy.md) | 1,557 classes, inheritance trees, AppState type, tool registry |
| 18 | [State Machines](./18-state-machines.md) | Agent loop, permission, session, streaming, MCP, sandbox state machines |

### Extracted Source (RVF files)

| File | Module | Confidence |
|------|--------|------------|
| [agent-loop.rvf](./extracted/agent-loop.rvf) | Core async generator (s$) | High |
| [tool-dispatch.rvf](./extracted/tool-dispatch.rvf) | Tool registry and dispatch | High |
| [permission-system.rvf](./extracted/permission-system.rvf) | Permission checker and sandbox | High |
| [mcp-client.rvf](./extracted/mcp-client.rvf) | MCP protocol client | High |
| [context-manager.rvf](./extracted/context-manager.rvf) | Token counting and compaction | High |
| [streaming-handler.rvf](./extracted/streaming-handler.rvf) | SSE event processing | High |

### Decompiler Script

`/scripts/claude-code-decompile.sh [output-dir]` - Automated extraction and analysis

## Key Findings

### Architecture Summary

- **Runtime**: Bun 1.3.11 Single Executable Application (229 MB binary)
- **Application code**: ~12.8 MB of bundled, minified JavaScript
- **UI**: React 18.3.1 WebView (VS Code) + Ink-style terminal (CLI)
- **API**: Anthropic Messages API with streaming SSE
- **Extension**: MCP client protocol with 4 transports

### By the Numbers

| Metric | Count |
|--------|-------|
| Built-in tools | 25+ |
| Slash commands | 39 |
| Environment variables | 498 |
| Settings properties | 76 |
| Supported models | 27+ |
| MCP protocol methods | 13 |
| Hook event types | 6 |
| Permission modes | 5 (acceptEdits, bypassPermissions, default, dontAsk, plan) |
| Extension mechanisms | 13 |
| Auth providers | 5 |
| MCP transports | 4 |
| Output formats | 3 |
| **Source code classes** | **1,557** |
| **Functions (estimated)** | **19,464** |
| **Async generators (core loops)** | **6** |
| **Bundle size (minified)** | **11 MB / 4,836 lines** |

### Architecture Pattern

Claude Code follows a **plugin-oriented monolith** pattern:
- Single binary deployment (Bun SEA)
- Modular internal architecture with clear subsystem boundaries
- Extensive extension surface (MCP, hooks, agents, skills, plugins)
- Multi-provider backend abstraction (Anthropic/AWS/GCP/Azure)
- Layered security (permissions -> sandbox -> hooks -> managed settings)

## Limitations

- Source is minified/mangled: variable names are meaningless (e.g., `Yq`, `f9`)
- Cannot trace exact function boundaries or module structure
- V8 snapshot region (~100MB) could not be decompiled
- Some patterns may be from bundled dependencies, not Claude Code itself
- This analysis reflects v2.1.90; architecture may change between versions
