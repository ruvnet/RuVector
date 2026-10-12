---
name: ruvector-constellation
description: Select and integrate RuVector npm packages for vector search, browser WASM, or MCP tools, with optional Ruflo orchestration and MetaHarness evaluation. Use when wiring these packages into an application or coding agent.
---

# RuVector constellation

Choose packages from the workload and runtime, then verify the installed capability. A package name or successful install does not prove its native backend, optional integrations, or MCP tools work.

## Select the surface

| Need | Starting package | Boundary |
| --- | --- | --- |
| Node vector search with CLI and general MCP | `ruvector` | Application facade; requires Node 20+ in the current source |
| Focused Node vector primitives | `@ruvector/core` | Native/platform loading must be tested on the target OS and CPU |
| Browser vector search | `@ruvector/wasm` | Verify browser exports and WASM assets with the application's bundler |
| RVF storage | `@ruvector/rvf` | Use its installed README and exports for the matching version |
| Focused RVF MCP | `@ruvector/rvf-mcp-server` | CLI supports `--transport stdio` |
| Agent coordination | `ruflo` | Separate application-level integration; opt in when requested |
| Evaluation, routing, or improvement gates | `metaharness`, `@metaharness/*` | Select the required capability, not the whole family |

For other packages, inspect their registry metadata and source manifest rather than guessing names from features. Distinguish the package's own API from tools provided by a facade. Keep orchestration and agent configuration out of native/WASM leaves; connect them at the application, adapter, or MCP layer. Avoid dependency cycles.

## Install and establish evidence

Inspect the project's lockfile and runtime first. For example:

```bash
npm view ruvector version engines repository dependencies optionalDependencies --json
npm install --save-exact ruvector
npx --no-install ruvector --help
npx --no-install ruvector info
npx --no-install ruvector harness status --json
```

Use the existing package manager when it differs from npm. Check the installed version's help before using a command from a newer checkout. `harness status` loads optional integrations; it can report missing packages without making core RuVector unusable. Current source declares several MetaHarness packages as optional dependencies, which npm normally installs; optional does not mean absent by default. For an intentionally lean installation, assess the consequences of `--omit=optional` for native backends and test the fallback on the target runtime.

Verify one application operation with a temporary database: insert a known vector, query it, and assert the expected result. Read the installed types/examples for that version's signatures; do not invent vector constructors or assume an optional native binary exists.

## Connect MCP

Use the installed `ruvector` executable with arguments `mcp start`. For a client with a JSON `mcpServers` configuration, merge an entry such as the following after resolving the absolute executable path in the project:

```json
{
  "mcpServers": {
    "ruvector": {
      "command": "/absolute/project/node_modules/.bin/ruvector",
      "args": ["mcp", "start"],
      "env": {
        "RUVECTOR_MCP_ALLOW": "hooks_capabilities,metaharness_status"
      }
    }
  }
}
```

Use the client's own configuration format and platform executable conventions. Preserve unrelated servers. The CLI also supports `npx ruvector mcp start` for exploratory use; prefer an installed, locked version for persistent configuration.

Complete the MCP initialization handshake, send `notifications/initialized`, then request `tools/list`. Call an advertised discovery tool and record its response. Expand the explicit allowlist to the tools the application actually needs. In current source, no policy means all registered tools are available; `RUVECTOR_MCP_DENY` wins over allow, and a configured profile is combined with the allowlist. The named `readonly` profile includes operations such as export and workspace probe; assess their side effects before selecting it. Do not equate a help listing with a working protocol handshake.

## Add optional connections

- For MetaHarness, use `ruvector harness status --json` or the advertised `metaharness_status` tool to identify available integrations. Install only the missing package needed by the requested feature, for example `npm install --save-exact @metaharness/router`. Read `ruvector harness route --help` before routing; it expects example scores, prices, and a query embedding. Rerun discovery and exercise the selected capability. Missing optional packages should produce an actionable diagnostic, not break core vector search.
- For Ruflo, inspect `npm view ruflo version engines --json`, install a compatible exact version, and check `npx --no-install ruflo mcp --help`. Its documented MCP entry is `ruflo mcp start`. Configure it as a separate server when coordination is needed. Do not run project initialization solely to expose MCP: initialization changes agent configuration. Keep the two servers' tool names and responsibilities explicit.
- Skill installation supplies agent instructions. It neither installs these npm runtimes nor registers MCP servers or plugins. Register a plugin only through the target host's documented manifest and lifecycle; an MCP connection alone is not a plugin installation.

Report installed versions, runtime/platform, exercised operation, advertised MCP tools, and unavailable capabilities. When improving a connection, retain the baseline workload and compare quality, latency, cost, and regressions before promoting a change. Do not infer permission to publish packages or deploy services from an integration request.
