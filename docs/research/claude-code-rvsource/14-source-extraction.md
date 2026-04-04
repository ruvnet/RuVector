# 14 - Source Extraction and Code Metrics

## Binary Analysis

### Distribution Formats

Claude Code ships in two forms:

| Format | Path | Size | Version |
|--------|------|------|---------|
| Bun SEA (ELF) | `~/.local/share/claude/versions/2.1.90` | 229,902,976 bytes (219 MB) | 2.1.90 |
| NPM bundle | `@anthropic-ai/claude-code/cli.js` | 11,044,554 bytes (10.5 MB) | 2.0.62 |

The Bun SEA binary is a dynamically linked ELF executable embedding the Bun runtime (v1.2+) with
the JavaScript bundle inlined. The NPM package contains the same logical code as a single minified
`cli.js` file plus tree-sitter WASM modules and a vendored ripgrep binary.

### Binary Structure (Bun SEA)

```
ELF 64-bit LSB executable, x86-64
├── Bun runtime (~219 MB)
│   ├── V8/JavaScriptCore bindings
│   ├── libc, libcrypto, libssl
│   └── Node.js compatibility layer
├── Embedded JS bundle (~11 MB equivalent)
│   ├── Minified application code
│   ├── Bundled npm dependencies (lodash, zod, ink, etc.)
│   └── Tree-sitter grammars (WASM)
└── Bun SEA metadata markers
    ├── @bun @bytecode @bun-cjs
    └── {"method":"Bun.canReload"}
```

### NPM Package Layout

```
@anthropic-ai/claude-code/
├── cli.js              11,044,554 bytes  (single minified bundle)
├── sdk-tools.d.ts          65,511 bytes  (TypeScript type defs for tools)
├── package.json             1,196 bytes
├── tree-sitter-bash.wasm 1,380,769 bytes
├── tree-sitter.wasm        205,498 bytes
├── vendor/
│   └── ripgrep/            (bundled rg binary)
├── LICENSE.md
└── README.md
```

### Version Management

Multiple versions coexist under `~/.local/share/claude/versions/`:
- `2.1.86` (228,280,960 bytes) - 2025-03-27
- `2.1.87` (228,280,960 bytes) - 2025-03-29
- `2.1.90` (229,902,976 bytes) - 2025-04-02 (current)

The active version is symlinked: `~/.local/bin/claude -> ~/.local/share/claude/versions/2.1.90`

## Code Metrics (from cli.js v2.0.62)

| Metric | Count |
|--------|-------|
| File size | 11,044,554 bytes |
| Lines (minified) | 4,836 |
| Estimated functions | 19,464 |
| Async functions | 884 |
| Arrow functions | 23,537 |
| Classes | 1,557 |
| Class inheritance (extends) | 956 |
| `for await` loops | 41 |
| `yield*` statements | 18 |
| Async generators | 6 (core loop functions) |
| Node.js built-in imports | 25+ modules |

### Estimated Original Source Size

The minified bundle is ~11 MB in 4,836 lines. Given typical minification ratios (3-5x for
well-structured TypeScript), the original source is estimated at 33-55 MB / 50,000-150,000 lines
across hundreds of modules.

### Minification Characteristics

The code uses aggressive mangling:
- All local variables shortened to 1-3 characters (`A`, `Q`, `B`, `G`, `Z`, `Y`)
- Module-scoped functions use short hashes (`s$`, `ye`, `nB`, `QA`, `wG`)
- Class names are 2-4 character hashes (`L6`, `V9`, `GI`, `RX`, `oJ`, `c90`)
- Original names survive only in string literals and public API surfaces

### Key Identifiable Functions (from code patterns)

| Minified Name | Likely Original | Evidence |
|--------------|-----------------|----------|
| `s$` | `agentLoop` / `queryLoop` | Core async generator; receives messages, systemPrompt, canUseTool, toolUseContext |
| `ye` | `resolveModel` | Takes permissionMode, mainLoopModel, exceeds200kTokens |
| `Ll` | `createInitialAppState` | Returns full AppState object with all fields |
| `QA` | `trackEvent` / `telemetry` | Called everywhere with event name + payload |
| `nB` | `updateSettings` | Writes to settings store |
| `wG` | `logDebug` | Debug logging with event names like "query_query_start" |
| `Y7` | `getMainLoopModel` | Returns current model |
| `Y0` | `getCwd` | Returns current working directory |
| `GB` | `getProjectDir` | Returns project directory |
| `Bd` | `getContextWindow` | Takes model, returns context window size |
| `xC` | `getToolPermissionContext` | Returns permission context object |
| `Sn` | `extractTextContent` | Extracts text from API response |
| `B0` | `getAgentId` | Returns current agent ID |

## Extraction Methods

### Method 1: NPM Package (recommended)

The `cli.js` from the NPM package is directly analyzable JavaScript:

```bash
CLI="$(npm root -g)/claude-flow/node_modules/@anthropic-ai/claude-code/cli.js"
# or install directly:
npm pack @anthropic-ai/claude-code && tar xzf anthropic-ai-claude-code-*.tgz
```

### Method 2: Binary Strings

```bash
strings ~/.local/share/claude/versions/2.1.90 | grep -c 'function\|class '
# Returns ~9,887 readable JS fragments
```

The binary contains the same JS but embedded within the Bun SEA container. The NPM
package provides cleaner access to the same code.

### Method 3: Bun SEA Extraction

The Bun SEA format embeds a bytecode blob. To extract:
```bash
# Find the JS entry section
strings -t d binary | grep '#!/usr/bin/env' | head -1
# Use offset to extract the embedded module
```

## Source Files Referenced

- Extracted module analysis: `extracted/agent-loop.rvf`
- Tool dispatch patterns: `extracted/tool-dispatch.rvf`
- Permission system: `extracted/permission-system.rvf`
- MCP client: `extracted/mcp-client.rvf`
- Context manager: `extracted/context-manager.rvf`
- Streaming handler: `extracted/streaming-handler.rvf`

## Dependencies (Node.js Built-in Imports)

```
assert, async_hooks, child_process, crypto, events, fs, fs/promises,
http, https, module, net, os, path, process, stream, tty, url, util, zlib
node:buffer, node:child_process, node:crypto, node:fs, node:fs/promises,
node:http, node:https, node:module, node:net, node:os, node:path,
node:process, node:stream, node:timers/promises, node:tty, node:url,
node:util, node:zlib
```

## Bundled Third-Party Libraries (identified from code patterns)

- Zod (schema validation - `S.string()`, `S.enum()`, `S.record()`)
- Ink / React (terminal UI - `createElement`, `useCallback`, `useEffect`, `useRef`)
- Sentry (error tracking - `globalEventProcessors`, `_dispatching`)
- GrowthBook (feature flags - `stickyBucketService`, `getExperiment`)
- Statsig (experimentation - `_getFeatureGateImpl`)
- Sharp (image processing - `@img/sharp-*` optional deps)
- node-forge (crypto - `aes.startEncrypting`, `aes.createDecryptionCipher`)
- tree-sitter (AST parsing - WASM modules)
- ripgrep (file search - vendored binary)
