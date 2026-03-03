# ADR-070: npx ruvector Unified Integration

**Status**: Proposed
**Date**: 2026-02-28
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-065 (npm Publishing Strategy), ADR-064 (Pi Brain Infrastructure), ADR-066 (SSE MCP Transport), ADR-069 (Edge-Net Integration)

## 1. Context

The RuVector npm ecosystem has grown to 55+ packages across multiple concerns:

- **`ruvector`** (v0.1.100): Core vector database with native/WASM/RVF backend auto-detection, CLI (`npx ruvector`), GNN wrappers, SONA embeddings, ONNX, parallel intelligence
- **`@ruvector/pi-brain`**: Pi brain CLI + SDK + MCP stdio proxy for the shared intelligence at `pi.ruv.io`
- **`@ruvector/edge-net`**: Distributed P2P browser compute network (WASM, Web Workers, rUv credits)
- **50+ other packages**: solver, rvf, gnn, attention, sona, ruvllm, tiny-dancer, ospipe, etc.

These systems are independently installable and operate in isolation. There is no unified entry point that lets a user:
1. Access the shared brain from the CLI
2. Join the edge compute network from Node.js
3. Start an MCP server that bridges all capabilities
4. Manage their π identity and rUv credits from one place

The `ruvector` CLI already has a `commander`-based command structure with subcommands for vector operations (insert, search, benchmark, etc.). Extending it to include brain, edge, and MCP integration creates a single `npx ruvector` entry point for the entire ecosystem.

## 2. Decision

Extend the existing `ruvector` CLI with three new command groups: `brain`, `edge`, and `mcp`. These commands call into `@ruvector/pi-brain` and `@ruvector/edge-net` as optional peer dependencies — they are lazy-loaded only when invoked, so the core `ruvector` package remains lightweight.

```
npx ruvector brain share "JWT refresh pattern" --category pattern
npx ruvector brain search "auth" --limit 10
npx ruvector edge status
npx ruvector edge join --contribution 0.3
npx ruvector mcp start --transport sse --url https://pi.ruv.io
npx ruvector identity generate
npx ruvector identity show
```

## 3. Architecture

### 3.1 Command Hierarchy

```
npx ruvector
  ├── insert       (existing) Vector insert
  ├── search       (existing) Vector search
  ├── benchmark    (existing) Performance benchmark
  ├── info         (existing) System info
  │
  ├── brain        (NEW) Shared intelligence
  │   ├── share    Share knowledge
  │   ├── search   Semantic search
  │   ├── get      Retrieve by ID
  │   ├── vote     Quality vote
  │   ├── list     List memories
  │   ├── delete   Delete own
  │   ├── transfer Domain transfer
  │   ├── drift    Check drift
  │   ├── partition Knowledge topology
  │   ├── status   System health
  │   ├── sync     LoRA weight sync
  │   ├── page     Brainpedia CRUD
  │   └── node     WASM node publish
  │
  ├── edge         (NEW) Distributed compute
  │   ├── status   Network status (genesis, relay, nodes)
  │   ├── join     Join as compute node
  │   ├── balance  Check rUv balance
  │   ├── tasks    List available compute tasks
  │   └── dashboard Open dashboard URL
  │
  ├── mcp          (NEW) MCP server management
  │   ├── start    Start MCP server (stdio or SSE)
  │   ├── tools    List available MCP tools
  │   └── test     Test MCP connection
  │
  └── identity     (NEW) π identity management
      ├── generate Generate new π key
      ├── show     Display current pseudonym
      ├── export   Export key for backup
      └── import   Import key from backup
```

### 3.2 Dependency Strategy

```
ruvector (core)
  ├── commander, chalk, ora          (bundled)
  ├── @ruvector/core                 (optional - native backend)
  ├── @ruvector/rvf                  (optional - RVF backend)
  │
  ├── @ruvector/pi-brain             (optional peer dep - brain commands)
  │   └── @modelcontextprotocol/sdk
  │
  └── @ruvector/edge-net             (optional peer dep - edge commands)
      └── wasm-bindgen (WASM runtime)
```

When a user runs `npx ruvector brain search "auth"` without `@ruvector/pi-brain` installed, the CLI prints:

```
Edge command requires @ruvector/pi-brain. Install with:
  npm install @ruvector/pi-brain
```

This keeps the core package at ~2MB while allowing the full ecosystem to be progressively adopted.

### 3.3 Environment Configuration

All commands read from a unified config hierarchy:

| Source | Priority | Example |
|--------|----------|---------|
| CLI flags | 1 (highest) | `--url https://pi.ruv.io` |
| Environment vars | 2 | `PI=key`, `BRAIN_URL=url` |
| `.env` file | 3 | `PI=abc123...` in project root |
| `~/.ruvector/config.json` | 4 | Global config |
| Defaults | 5 (lowest) | `https://pi.ruv.io` |

The π key (`PI` env var) is shared across brain, edge, and MCP commands. One identity, one key, three systems.

### 3.4 Identity Derivation Chain

```
User's π key (64 hex chars)
  │
  ├── SHAKE-256(key) ──► Brain pseudonym (contributor ID)
  │                       Used for: brain share, vote, delete
  │
  ├── Ed25519(key) ────► Edge Pi-Key (node identity)
  │                       Used for: edge join, rUv transactions
  │
  └── HMAC-SHA256(key, "mcp") ──► MCP session token
                                   Used for: SSE MCP auth
```

A single key derives three identities through different cryptographic paths. The SHAKE-256 path matches the brain server's `auth.rs` pseudonym derivation. The Ed25519 path matches edge-net's `pikey` module. The HMAC path provides MCP session auth.

## 4. New CLI Commands

### 4.1 `ruvector brain`

Wraps `@ruvector/pi-brain`'s `PiBrainClient`:

```typescript
// Lazy load
const { PiBrainClient } = await import('@ruvector/pi-brain');
const client = new PiBrainClient({ url: opts.url, key: opts.key });
```

| Command | Maps to | Description |
|---------|---------|-------------|
| `brain share <title> -c <category> -t <tags>` | `POST /v1/memories` | Share knowledge |
| `brain search <query> -l <limit>` | `GET /v1/memories/search` | Semantic search |
| `brain get <id>` | `GET /v1/memories/:id` | Retrieve with provenance |
| `brain vote <id> <up\|down>` | `POST /v1/memories/:id/vote` | Quality vote |
| `brain list [-c category] [-l limit]` | `GET /v1/memories/list` | List memories |
| `brain delete <id>` | `DELETE /v1/memories/:id` | Delete own contribution |
| `brain transfer <source> <target>` | `POST /v1/transfer` | Domain transfer |
| `brain drift [--domain <d>]` | `GET /v1/drift` | Drift detection |
| `brain partition [--domain <d>]` | `GET /v1/partition` | Knowledge topology |
| `brain status` | `GET /v1/status` | System health |
| `brain sync [pull\|push\|both]` | `POST /v1/lora/submit` | LoRA sync |

### 4.2 `ruvector edge`

Wraps `@ruvector/edge-net` for Node.js (non-browser) usage:

| Command | Description |
|---------|-------------|
| `edge status` | Query genesis node for network stats, rUv supply, sunset phase |
| `edge join --contribution 0.3` | Join as compute node (headless, Node.js Web Worker polyfill) |
| `edge balance` | Check rUv balance for current identity |
| `edge tasks` | List available distributed compute tasks |
| `edge dashboard` | Open edge-net dashboard in browser |

Edge commands hit the deployed services:
- Genesis: `https://edge-net-genesis-875130704813.us-central1.run.app`
- Relay: `https://edge-net-relay-875130704813.us-central1.run.app`
- Dashboard: `https://edge-net-dashboard-875130704813.us-central1.run.app`

### 4.3 `ruvector mcp`

Manages MCP server lifecycle:

| Command | Description |
|---------|-------------|
| `mcp start` | Start stdio MCP server (default, for `claude mcp add`) |
| `mcp start --transport sse --port 8080` | Start SSE MCP server locally |
| `mcp tools` | List all 22 available MCP tools |
| `mcp test` | Send test JSON-RPC to verify connection |

The `mcp start` command replaces `cargo run -p mcp-brain` for users who don't have Rust installed:

```bash
# Before (requires Rust toolchain)
claude mcp add pi-brain -- cargo run -p mcp-brain

# After (just Node.js)
claude mcp add pi-brain -- npx ruvector mcp start
```

### 4.4 `ruvector identity`

Manages the π key:

| Command | Description |
|---------|-------------|
| `identity generate` | Generate new π key, display, copy to clipboard |
| `identity show` | Show current key's pseudonym, edge Pi-Key, reputation |
| `identity export` | Export key to file (encrypted with passphrase) |
| `identity import <file>` | Import key from encrypted backup |

## 5. Implementation

### 5.1 File Changes

| File | Change |
|------|--------|
| `npm/packages/ruvector/bin/cli.js` | Add `brain`, `edge`, `mcp`, `identity` command groups |
| `npm/packages/ruvector/package.json` | Add `@ruvector/pi-brain` and `@ruvector/edge-net` as optional peer deps |
| `npm/packages/ruvector/src/commands/brain.ts` | Brain command handlers with lazy `pi-brain` import |
| `npm/packages/ruvector/src/commands/edge.ts` | Edge command handlers with lazy `edge-net` import |
| `npm/packages/ruvector/src/commands/mcp.ts` | MCP server start/test with transport selection |
| `npm/packages/ruvector/src/commands/identity.ts` | Key generation, derivation, export/import |

### 5.2 Lazy Loading Pattern

```typescript
async function requirePiBrain(): Promise<typeof import('@ruvector/pi-brain')> {
  try {
    return await import('@ruvector/pi-brain');
  } catch {
    console.error(chalk.red('Brain commands require @ruvector/pi-brain'));
    console.error(chalk.yellow('  npm install @ruvector/pi-brain'));
    process.exit(1);
  }
}
```

### 5.3 Output Formatting

All commands output JSON by default when piped (`!process.stdout.isTTY`) and human-readable tables/colors when interactive. The `--json` flag forces JSON output.

```bash
# Human-readable
npx ruvector brain status
# Memories: 42 | Contributors: 7 | Quality: 0.82 | Drift: stable

# Machine-readable
npx ruvector brain status --json
# {"total_memories":42,"total_contributors":7,...}

# Piped
npx ruvector brain search "auth" | jq '.[] | .title'
```

## 6. Security

### 6.1 Key Storage

The π key is never stored in plaintext on disk by the CLI. Options:
- Environment variable (`PI=...`)
- `.env` file (user's responsibility to gitignore)
- System keychain via `keytar` (optional dependency)
- Encrypted file via `identity export/import`

### 6.2 Network Security

All CLI commands communicate over HTTPS. The brain client validates TLS certificates. No HTTP fallback.

### 6.3 Dependency Isolation

Brain and edge dependencies are optional peers, not bundled. This prevents supply chain attacks through transitive dependencies from affecting users who only use the core vector database.

## 7. Versioning

The `ruvector` CLI version is independent of the brain and edge package versions. The CLI detects compatible version ranges at runtime:

```typescript
const pkg = await import('@ruvector/pi-brain/package.json');
if (!semver.satisfies(pkg.version, '>=0.1.0')) {
  console.warn(chalk.yellow(`pi-brain ${pkg.version} may not be compatible`));
}
```

## 8. Testing

| Test | Description |
|------|-------------|
| `brain` commands without `pi-brain` installed | Graceful error with install instructions |
| `brain status` with mock server | Returns formatted status |
| `brain search "query"` with live backend | Returns results |
| `edge status` against genesis | Returns network stats |
| `mcp start` stdio | Responds to JSON-RPC initialize |
| `mcp tools` | Lists 22 tools |
| `identity generate` | Produces valid 64-char hex key |
| `identity show` | Derives correct SHAKE-256 pseudonym |
| JSON output mode | All commands produce valid JSON with `--json` |
| Pipe detection | Auto-JSON when stdout is not TTY |

## 9. Migration Path

### Phase 1: CLI Extension (1 week)
Add command groups to `bin/cli.js`. Wire `brain` commands to `@ruvector/pi-brain`. Add `identity` commands with key generation and SHAKE-256 derivation.

### Phase 2: Edge Integration (1 week)
Add `edge` commands. Create Node.js adapter for `@ruvector/edge-net` (which targets browsers). Implement headless join for server-side compute contribution.

### Phase 3: MCP Proxy (1 week)
Add `mcp start` with stdio and SSE transport support. Replace `cargo run -p mcp-brain` as the recommended MCP setup for non-Rust users.

### Phase 4: Publish (1 week)
Bump `ruvector` to 0.2.0. Update README. Publish `@ruvector/pi-brain` and ensure version compatibility. Update landing page docs.

## 10. Consequences

### Positive
- Single entry point: `npx ruvector` provides access to vector DB, shared brain, edge compute, and MCP
- Progressive adoption: core package stays lightweight, features opt-in via peer deps
- No Rust required: `npx ruvector mcp start` replaces `cargo run -p mcp-brain`
- Unified identity: one π key for all three systems

### Negative
- CLI complexity increases — more surface area to maintain
- Optional peer deps can confuse users (unclear what's installed)
- Node.js adapter for edge-net (browser-targeted WASM) may have compatibility gaps

### Neutral
- Version coordination between `ruvector`, `pi-brain`, and `edge-net` requires semver discipline
- Existing `pi-brain` CLI (`npx pi-brain`) continues to work independently

## 11. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-065 | npm Publishing Strategy — package categories, publish order |
| ADR-064 | Pi Brain Infrastructure — Cloud Run deployment, domains |
| ADR-066 | SSE MCP Transport — SSE protocol that `mcp start --transport sse` exposes |
| ADR-069 | Edge-Net Integration — distributed compute that `edge` commands access |
| ADR-059 | Shared Brain Google Cloud — backend that `brain` commands call |
| ADR-060 | Shared Brain Capabilities — 7 capabilities exposed through `brain` subcommands |
