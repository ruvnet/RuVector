# ADR-137: npm Decompiler CLI and MCP Tools

## Status

Deployed (2026-04-03) — CLI command + 6 MCP tools implemented. Decompiler library with reconstruction, validation, and `--runnable` mode.

## Date

2026-04-03

## Context

The `ruvector-decompiler` Rust crate (ADR-135) provides MinCut-based JavaScript decompilation with witness chains. Currently it's only usable as a Rust library or via `cargo run --example`. To reach developers where they work, it needs npm distribution as both a CLI tool and MCP server.

### Target UX

```bash
# CLI: decompile any npm package
npx ruvector decompile express
npx ruvector decompile @anthropic-ai/claude-code --version 2.1.90
npx ruvector decompile ./bundle.min.js --output ./decompiled/

# MCP: use from Claude Code / AI agents
claude mcp add ruvector -- npx ruvector mcp
# Then: "decompile the lodash package and explain the debounce implementation"
```

## Decision

Add `decompile` subcommand to the existing `npx ruvector` CLI (`npm/packages/ruvector/`) and expose decompilation via the existing MCP server. The decompiler runs as WASM (in-browser/Node.js) for portability.

### Architecture

```
npx ruvector decompile <package>
       │
       ▼
┌──────────────────────────────┐
│ @ruvector/cli                │
│                              │
│  ┌────────────────────────┐  │
│  │ decompile command      │  │
│  │                        │  │
│  │ 1. Fetch from npm      │  │  registry.npmjs.org
│  │ 2. Extract bundle      │  │  unpkg.com / tarball
│  │ 3. Run decompiler      │──┼──▶ ruvector-decompiler-wasm
│  │ 4. Output results      │  │    (pure Rust → WASM)
│  └────────────────────────┘  │
│                              │
│  ┌────────────────────────┐  │
│  │ MCP server             │  │
│  │                        │  │
│  │ tools:                 │  │
│  │  • decompile_package   │  │  Fetch + decompile npm package
│  │  • decompile_file      │  │  Decompile local .js file
│  │  • decompile_url       │  │  Decompile from URL
│  │  • decompile_search    │  │  Search decompiled modules
│  │  • decompile_diff      │  │  Compare two versions
│  │  • decompile_witness   │  │  Verify witness chain
│  └────────────────────────┘  │
└──────────────────────────────┘
```

### CLI Commands

#### `npx ruvector decompile <target>`

| Argument | Description | Example |
|----------|-------------|---------|
| `<package>` | npm package name | `npx ruvector decompile express` |
| `<package>@<version>` | Specific version | `npx ruvector decompile lodash@4.17.21` |
| `<file.js>` | Local file | `npx ruvector decompile ./dist/bundle.min.js` |
| `<url>` | Remote URL | `npx ruvector decompile https://unpkg.com/react` |

| Flag | Description | Default |
|------|-------------|---------|
| `--output, -o` | Output directory | `./decompiled/<name>/` |
| `--format` | Output format: `modules`, `single`, `json`, `rvf` | `modules` |
| `--confidence` | Minimum confidence threshold | `0.3` |
| `--witness` | Generate witness chain proof | `true` |
| `--source-map` | Generate V3 source maps | `true` |
| `--diff <version>` | Compare against another version | — |
| `--json` | JSON output (for piping) | `false` |
| `--quiet, -q` | Suppress progress output | `false` |
| `--runnable` | Guaranteed-runnable output (validated renames only) | `false` |
| `--validate` | Run operational validation after decompilation | `false` |
| `--reconstruct` | Apply AI name recovery + JSDoc + style fixes | `true` |

#### Output Formats

**`modules`** (default) — graph-derived folder hierarchy:
```
decompiled/express/
├── README.md
├── source/                # Decompiled code (graph-derived folders)
│   ├── core/
│   ├── tools/
│   ├── config/
│   └── uncategorized/
├── rvf/                   # RVF containers (separate from source)
│   ├── master.rvf
│   └── core.rvf
├── witness.json           # Merkle witness chain
└── metrics.json           # Declarations, confidence, timing
```

**`runnable`** — single validated file (guaranteed to execute):
```
decompiled/express/
├── express.runnable.js    # Renames validated one-by-one via vm sandbox
├── witness.json
└── rename-report.json     # Applied/rejected renames with reasons
```

**`single`** — beautified single file with module comments:
```
decompiled/express/decompiled.js
```

**`json`** — structured JSON (for programmatic use):
```json
{
  "modules": [...],
  "inferred_names": [...],
  "witness_chain": {...},
  "metrics": {...}
}
```

**`rvf`** — binary RVF container:
```
decompiled/express/express-v4.18.2.rvf
```

### MCP Tools

Six tools exposed via the existing `npx ruvector mcp` server:

#### `decompile_package`

```json
{
  "name": "decompile_package",
  "description": "Decompile an npm package. Fetches from registry, extracts bundle, runs MinCut partitioning, infers original names with AI.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "package": { "type": "string", "description": "npm package name (e.g., 'express', '@anthropic-ai/claude-code')" },
      "version": { "type": "string", "description": "Version (default: latest)" },
      "min_confidence": { "type": "number", "description": "Minimum confidence threshold (0-1, default: 0.3)" }
    },
    "required": ["package"]
  }
}
```

Returns: module list with inferred names, metrics, witness chain root.

#### `decompile_file`

```json
{
  "name": "decompile_file",
  "description": "Decompile a local JavaScript file using MinCut partitioning and AI name inference.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": { "type": "string", "description": "Path to .js file" },
      "min_confidence": { "type": "number" }
    },
    "required": ["path"]
  }
}
```

#### `decompile_url`

Fetch and decompile from any URL (unpkg, CDN, raw GitHub).

#### `decompile_search`

Search across previously decompiled modules by string/pattern:

```json
{
  "name": "decompile_search",
  "description": "Search decompiled code for patterns, function names, or string literals.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Search query (regex supported)" },
      "package": { "type": "string", "description": "Limit to specific package" },
      "module": { "type": "string", "description": "Limit to specific module" }
    },
    "required": ["query"]
  }
}
```

#### `decompile_diff`

Compare two versions of the same package:

```json
{
  "name": "decompile_diff",
  "description": "Compare decompiled output between two versions of a package. Shows added/removed/changed modules and declarations.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "package": { "type": "string" },
      "version_a": { "type": "string" },
      "version_b": { "type": "string" }
    },
    "required": ["package", "version_a", "version_b"]
  }
}
```

#### `decompile_witness`

Verify a witness chain:

```json
{
  "name": "decompile_witness",
  "description": "Verify the cryptographic witness chain of a decompilation. Proves output derives faithfully from input.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "witness_path": { "type": "string", "description": "Path to witness.json" },
      "source_path": { "type": "string", "description": "Path to original bundle" }
    },
    "required": ["witness_path"]
  }
}
```

### WASM Compilation

The `ruvector-decompiler` crate compiles to WASM for Node.js:

```
crates/ruvector-decompiler/        →  Rust source
crates/ruvector-decompiler-wasm/   →  wasm-bindgen wrapper (new)
npm/packages/ruvector/             →  npm CLI + MCP (existing, add decompile command)
```

The WASM module includes:
- Parser with memchr
- Louvain partitioner (rayon → single-thread in WASM)
- Name inferrer with 210 patterns
- Witness chain (SHA3-256)
- Source map generator
- Transformer inference (pure Rust, no ONNX needed)

Estimated WASM size: ~500KB (with model weights embedded: ~700KB).

### Integration Points

| Integration | How | Use Case |
|-------------|-----|----------|
| **Claude Code** | `claude mcp add ruvector` | "Decompile this dependency" |
| **CI/CD** | `npx ruvector decompile --json` | Audit npm deps in pipeline |
| **Dashboard** | Import WASM module | In-browser decompilation |
| **Brain server** | New MCP tools on `pi.ruv.io` | Remote decompilation API |
| **RVF corpus** | `--format rvf` | Searchable decompiled packages |

### Implementation Plan

| Phase | Work | Timeline |
|-------|------|----------|
| 1 | Compile `ruvector-decompiler` to WASM (`crates/ruvector-decompiler-wasm/`) | 1 week |
| 2 | Add `decompile` subcommand to `npm/packages/ruvector/` CLI | 3 days |
| 3 | Add 6 MCP tools to existing MCP server | 3 days |
| 4 | Publish to npm: `@ruvector/decompiler` | 1 day |
| 5 | Add to brain server MCP tools at `mcp.pi.ruv.io` | 2 days |

## Consequences

### Positive

- **One command**: `npx ruvector decompile express` — instant, no install
- **AI-assisted auditing**: Claude Code can decompile and explain any dependency
- **Supply chain security**: CI pipelines verify npm packages with witness chains
- **Cross-version tracking**: `decompile_diff` shows exactly what changed between releases
- **Portable**: WASM runs in Node.js, browser, and edge functions

### Negative

- WASM is single-threaded (no rayon) — Louvain is slower than native
- WASM module size ~700KB with embedded model weights
- npm registry rate limits may affect bulk decompilation

### Risks

| Risk | Mitigation |
|------|------------|
| WASM performance | Profile hot paths, optimize critical loops |
| npm rate limiting | Local caching of fetched packages |
| Model accuracy | Self-learning improves over time; fallback to patterns |
| Legal concerns | Only decompile packages you have rights to; witness proves no modification |

## References

- [ADR-135: MinCut Decompiler](./ADR-135-mincut-decompiler-with-witness-chains.md)
- [ADR-136: GPU Training Pipeline](./ADR-136-gpu-trained-deobfuscation-model.md)
- [ADR-134: Claude Code Integration](./ADR-134-ruvector-claude-code-deep-integration.md)
- [npm/packages/ruvector/](../../npm/packages/ruvector/) — existing CLI
- [crates/ruvector-decompiler/](../../crates/ruvector-decompiler/) — Rust decompiler
