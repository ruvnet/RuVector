# ruvector: vector search and persistent agent memory

![RuVector memory and feedback loop](https://raw.githubusercontent.com/ruvnet/RuVector/main/assets/ruvector/build-loop.svg)

[![npm version](https://img.shields.io/npm/v/ruvector.svg)](https://www.npmjs.com/package/ruvector)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ruvnet/RuVector/blob/main/LICENSE)

The `ruvector` npm package provides Node.js vector storage, semantic memory commands, and MCP integration. It is part of [Reuven Cohen's RuVector stack](https://github.com/ruvnet/RuVector). Specialized graph, learning, browser, and runtime libraries have their own packages and setup.

[Quick start](#quick-start) · [Claude Code](#claude-code) · [Node.js tutorial](https://github.com/ruvnet/RuVector/blob/main/examples/nodejs/README.md) · [Full library catalog](https://github.com/ruvnet/RuVector#library-and-capability-map)

## Quick start

Run in your project directory:

```bash
npm install ruvector
npx ruvector info
npx ruvector hooks remember --semantic --type decision "Keep customer inference in Canada."
npx ruvector hooks recall --semantic --top-k 3 "Where should inference run?"
```

The first semantic command downloads a local embedding model. Memory belongs to the current project. Run recall again from the same directory in a new terminal to check persistence. Commit your lockfile to preserve dependency resolution.

**Backend check:** native vector operations require a working native backend. If it cannot load, the limited fallback does not provide persistent vector storage. Browser applications should use the [WASM package and tutorial](https://github.com/ruvnet/RuVector/blob/main/examples/wasm-vanilla/README.md). See [supported deployment surfaces](https://github.com/ruvnet/RuVector#deployment-surfaces).

## Claude Code

```bash
claude mcp add --scope project --env RUVECTOR_MCP_PROFILE=readonly --transport stdio ruvector -- npx -y ruvector mcp start
claude mcp get ruvector
```

Open Claude Code in the project, approve the project server when prompted, and check `/mcp`. Ask it to retrieve relevant project memory and identify the supporting records.

[Complete Claude Code instructions and CLAUDE.md sample](https://github.com/ruvnet/RuVector#claude-code-setup)

### Other MCP clients

Use command `npx`, arguments `["-y", "ruvector", "mcp", "start"]`, and environment `RUVECTOR_MCP_PROFILE=readonly`. Launch from the project directory.

```bash
npx ruvector mcp tools
RUVECTOR_MCP_PROFILE=readonly npx ruvector mcp start
```

Available tools vary with package version and policy. `RUVECTOR_MCP_DENY` overrides allow rules; `RUVECTOR_MCP_ALLOW` and profiles restrict access. Without a policy, the broader compatibility surface remains available. [Policy and agent integration](https://github.com/ruvnet/RuVector#agent-integration).

## Build with the SDK

Follow the [Node.js write and reopen tutorial](https://github.com/ruvnet/RuVector/blob/main/examples/nodejs/README.md). It uses `VectorDB`, `OnnxEmbedder`, and search parameter `k`, with a separate read process to verify storage.

[Node.js API](https://github.com/ruvnet/RuVector/blob/main/docs/api/NODEJS_API.md) · [Current exports](https://github.com/ruvnet/RuVector/blob/main/npm/packages/ruvector/src/index.ts)

## Choose a capability

| System | Start with | Tutorial |
| :--- | :--- | :--- |
| System 0: Sense and respond | Local embeddings, npm `@ruvector/typesafe` | [Typed decisions](https://github.com/ruvnet/RuVector/blob/main/npm/packages/typesafe/README.md) |
| System 1: Learn and remember | `ruvector`, `@ruvector/kge`, `@ruvector/sona` | [Memory](https://github.com/ruvnet/RuVector/blob/main/examples/nodejs/README.md), [KGE](https://github.com/ruvnet/RuVector/blob/main/npm/packages/kge/README.md), [SONA](https://github.com/ruvnet/RuVector/blob/main/npm/packages/sona/README.md) |
| System 2: Reason and orchestrate | MCP, MetaHarness integrations, RVF artifacts | [Examples hub](https://github.com/ruvnet/RuVector/blob/main/examples/README.md), [RVF](https://github.com/ruvnet/RuVector/blob/main/examples/rvf/README.md) |

Reading memory does not train weights. Learning requires explicit outcomes and an evaluation policy. Contrastive learning and graph coherence provide different signals; neither establishes factual truth.

## Coding hooks and harnesses

[Hooks reference](https://github.com/ruvnet/RuVector/blob/main/npm/packages/ruvector/HOOKS.md) covers setup, generated configuration, and workflow integration.

```bash
npx ruvector hooks --help
npx ruvector hooks verify
npx ruvector harness doctor --json
```

Review generated hooks before using them. Harness execution and hosted Brain commands have different effects and data boundaries from local retrieval. Start with the [governance and deployment guide](https://github.com/ruvnet/RuVector#security-and-governance).

## Troubleshooting

| Symptom | Check |
| :--- | :--- |
| Native backend unavailable | Run `npx ruvector info`; check platform and native dependency installation |
| Empty recall after changing directory | Return to the directory that owns the memory store |
| Embedding dimension mismatch | Use the original model and dimension; review `npx ruvector hooks reembed --help` before migrating |
| Claude cannot find the server | Run `claude mcp get ruvector`, then inspect `/mcp` in that project |
| A write tool is unavailable | Check the configured MCP profile and explicit tool policy |

[Benchmarks with scope and evidence](https://github.com/ruvnet/RuVector#reproduce-the-evidence) · [Known boundaries](https://github.com/ruvnet/RuVector#known-boundaries) · [Issues](https://github.com/ruvnet/RuVector/issues) · [MIT license](https://github.com/ruvnet/RuVector/blob/main/LICENSE)
