# obsidian-brain

Obsidian plugin that bridges your vault with the **RuVector Brain** (local `mcp-brain-server-local` on `127.0.0.1:9876`) and the embedder (`127.0.0.1:9877`). Implements ADR-SYS-0025.

## Features

| Command / UI | What it does |
| --- | --- |
| `Cmd+Shift+B` semantic search | DiskANN search over the brain's memory graph. Results open matching vault notes if known, otherwise show content inline. |
| Related panel (right sidebar) | Live "similar memories" for the active note; refreshes when you switch notes. |
| Auto-index on save | Debounced per-file embed + store into the brain. AIDefence-scanned first. |
| Bulk-sync vault → brain | Modal with progress bar, include/exclude folder filters, content-hash dedup. |
| DPO preference pairs | Mark chosen / rejected / export pairs as a markdown table in `Brain/Exports/`. |
| Graph overlay | Writes per-category color groups into `.obsidian/graph.json` and tags notes with `#brain/<category>`. Reversible. |
| Brain status bar | Live health + memory count; click for engine / index mode. |
| `pi.ruv.io: pull memories into local brain` | Mirrors a slice of the shared pi.ruv.io brain into the local brain + `Brain/Pi/<title>.md` stubs. AIDefence-scanned, content-hash deduped. |
| `pi.ruv.io: search shared brain directly` | Modal that queries pi.ruv.io's `/v1/memories/search` with your bearer token and shows scored results inline. |
| `pi.ruv.io: status` | Fetches `/v1/status` — total memories, graph edges, embedding dim, drift status. |

## Install

### Option A — from this repo

```bash
# In the plugin dir:
npm install
npm run build

# Copy into your vault (or run the helper):
./scripts/setup.sh /path/to/your-vault
```

Then in Obsidian: **Settings → Community plugins → Installed plugins → RuVector Brain → Enable**.

### Option B — BRAT

Install with [BRAT](https://github.com/TfTHacker/obsidian42-brat) from
the dedicated distribution repo:

```
BRAT → Add beta plugin → ruvnet/obsidian-brain
```

The [ruvnet/obsidian-brain](https://github.com/ruvnet/obsidian-brain)
repo publishes tagged releases of `main.js`, `manifest.json`, and
`styles.css` built from the source in this directory.

## Running the brain locally

The plugin expects two local HTTP services on loopback:

- **Brain** — `mcp-brain-server-local` (axum, DiskANN, AIDefence) on `:9876`
- **Embedder** — any service answering `POST /embed` on `:9877` and returning `{ "vectors": [[...]] }`

Build + run the brain:

```bash
cargo build --release --bin mcp-brain-server-local --features local -p mcp-brain-server
./target/release/mcp-brain-server-local --port 9876
```

For systemd user units, copy `systemd/*.service` into `~/.config/systemd/user/` and:

```bash
systemctl --user daemon-reload
systemctl --user enable --now ruvector-brain.service
systemctl --user enable --now ruvector-embedder.service
```

Both units bind loopback only (`IPAddressDeny=any` + `IPAddressAllow=127.0.0.0/8`).

## Why not an embedded vector DB?

The brain already ships DiskANN, AIDefence, embeddings and a stable HTTP surface. It is shared across the CLI, MCP tools and agent runtime. Duplicating a second index inside the plugin would fragment the knowledge base without adding capability.

## Settings

All settings live in **Settings → RuVector Brain**. Key ones:

- **Brain URL / Embedder URL** — default `http://127.0.0.1:9876` / `:9877`.
- **Auto-index on save** — off by default. Debounce configurable.
- **AIDefence scan before indexing** — on by default. When the brain is unreachable, indexing bails rather than leaking notes.
- **Bulk-sync include/exclude folders** — comma-separated paths, anchored at vault root.
- **DPO default direction** — label stored on every preference pair (e.g. `quality`, `tone`).
- **pi.ruv.io URL / bearer token** — required for the pi commands. Token is stored in `.obsidian/plugins/obsidian-brain/data.json`.
- **Pull limit / query** — how many pi memories to pull per sync, optionally filtered by a semantic query.

## Development

```bash
npm install
npm run dev       # esbuild watch → main.js (reload via "Reload plugin" command in Obsidian)
npm run build     # production bundle + tsc --noEmit
npm run typecheck
```

## Live dev session (`scripts/run-dev.sh`)

One command boots everything: brain subprocess, an embedder (prefers the
real `ruvultra-embedder` at `:9877`, falls back to a 16-dim stub), a demo
vault with twelve seed notes, optional pi.ruv.io pull, graph color groups,
and a fully populated `indexState` so click-to-open works.

```bash
# Minimum — runs entirely offline.
./scripts/run-dev.sh

# Pull 30 pi.ruv.io memories into the local brain and vault (requires
# bearer token in $BRAIN_API_KEY).
PI_LIMIT=30 ./scripts/run-dev.sh

# Filter the pi pull with a semantic query.
PI_QUERY="hnsw diskann" PI_LIMIT=20 ./scripts/run-dev.sh

# Use a custom vault location.
./scripts/run-dev.sh ~/notes/brain-vault
```

The script:

1. Builds `mcp-brain-server-local` + the plugin on first run (cached after).
2. Extracts the Obsidian AppImage once into `~/.cache/obsidian-brain-e2e/`.
3. Writes a minimal `.obsidian/` with the plugin enabled and Bases
   disabled (it otherwise drops `Untitled.base` into the vault root).
4. Probes `:9877` — if the real embedder is up, uses it directly;
   otherwise starts a tiny in-process stub.
5. Calls `scripts/seed-dev.py` after the brain health-checks to seed
   memories, populate `indexState`, surface AIDefence 422s, pull pi
   memories, and write `.obsidian/graph.json`.
6. Launches Obsidian under an isolated `HOME` (nothing touches the real
   Obsidian vault registry).

### Dim-mismatch guard

`seed-dev.py` refuses to seed when the brain already holds vectors at a
different dim than the current embedder (DiskANN silently misbehaves if
you mix dims). Wipe `.brain-data/` to recover:

```bash
rm -rf ~/obsidian-brain-vault/.brain-data
```

## Testing

Three real-service test suites, no mocks. `npm test` runs everything
except the Obsidian E2E.

### 1. Protocol tests (`tests/protocol/`) — always-on

Spins up a real `mcp-brain-server-local` subprocess with a scratch SQLite DB
and a small in-process mock embedder, then exercises every endpoint the
plugin depends on (`/health`, `/brain/info`, `/brain/index_stats`,
`/brain/search`, `/memories`, `/memories/:id`, `/security/scan`,
`/preference_pairs`, `/embed`). Validates the exact response shapes the
`BrainClient` parses.

Prereq (one-time):

```bash
cargo build --release -p mcp-brain-server --features local --bin mcp-brain-server-local
```

`npm test` picks up the binary at `../../target/release/mcp-brain-server-local`
or at `$RUVBRAIN_BIN`.

### 2. pi.ruv.io protocol tests (`tests/protocol/pi-server.test.ts`) — gated

Asserts `/v1/status`, bearer-gated `/v1/memories/list` and
`/v1/memories/search`, and that the 401 is correctly returned without the
bearer. Only runs when `BRAIN_API_KEY` is set:

```bash
BRAIN_API_KEY=<your-token> npm test
```

### 3. Obsidian E2E (`tests/e2e/`) — opt-in, real Obsidian

Downloads the real Obsidian AppImage (cached at `~/.cache/obsidian-brain-e2e/`),
unpacks it, provisions a disposable vault, enables the built plugin + a
companion *harness plugin* that runs inside Obsidian and exercises commands
end-to-end, launches the app under `xvfb-run`, and asserts the harness's
JSON report.

Prereq:

```bash
sudo apt install xvfb libfuse2t64   # or libfuse2 on older distros
```

Run:

```bash
OBSIDIAN_E2E=1 npm test -- tests/e2e
```

The harness validates, inside the real Obsidian runtime:

1. `obsidian-brain` plugin loads
2. All commands are registered
3. Status bar populates with live brain state
4. `Index current note` persists a note into the brain
5. `BrainClient.search` returns the note we just indexed
6. `bulkSync` completes with zero failures
7. Graph overlay writes `#brain/*` color groups to `.obsidian/graph.json`
8. `pi.ruv.io: pull/search/status` commands are registered
9. *(when `BRAIN_API_KEY` is set)* `PiClient.status` roundtrips against live pi.ruv.io

Source layout:

```
src/
  main.ts           # plugin entry, commands, status bar, persistence
  brain.ts          # HTTP client (brain + embedder + AIDefence scan)
  indexer.ts        # per-file + bulk indexing with hash dedup
  settings.ts       # settings tab + defaults
  search-modal.ts   # Cmd+Shift+B modal + fuzzy fallback
  related-view.ts   # right-panel ItemView
  bulk-sync.ts      # bulk import modal with progress bar
  dpo.ts            # chosen / rejected / export preference pairs
  graph-overlay.ts  # category colour groups in graph.json
  pi-client.ts      # pi.ruv.io HTTP client (status/list/search)
  pi-sync.ts        # pi pull modal + pi search modal, writes Brain/Pi/*.md
scripts/
  run-dev.sh        # live dev session: builds, seeds, launches Obsidian
  seed-dev.py       # idempotent seeder: local notes, pi pull, indexState
  setup.sh          # one-shot install into an existing vault
```

## Protocol

The plugin only uses these brain endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | status / backend / version |
| `GET` | `/brain/info` | memory count, db path |
| `GET` | `/brain/index_stats` | engine + mode |
| `POST` | `/brain/search` | `{query?, query_vector?, k}` → results |
| `POST` | `/memories` | `{category, content, embedding?}` — creates memory |
| `GET` | `/memories/:id` | fetch single memory (content hydrated from blob store) |
| `POST` | `/security/scan` | AIDefence scan arbitrary text |
| `GET` / `POST` | `/preference_pairs` | DPO list / create |
| `POST` | `/embed` *(embedder)* | `{texts: [...]}` → `{vectors: [[...]]}` |

## License

MIT — see repository root.
