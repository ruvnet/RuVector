# ADR-152: Obsidian Brain Plugin — RuVector Brain bridge for Obsidian

## Status

Accepted (phases 1–3 delivered, phase 4 roadmap in §10)

## Date

2026-04-20

## Aliases

- **ADR-SYS-0025** — original planning identifier used during the
  scoping session and referenced in the plugin's `manifest.json`.
  This ADR is the canonical record.

## Context

Obsidian is the largest-installed-base personal knowledge base UI with a
mature plugin API. The RuVector **local brain**
(`mcp-brain-server-local`) already ships:

- DiskANN Vamana graph index for vector search (ADR-146)
- AIDefence regex set for prompt-injection / PII screening (ADR-082)
- SQLite content store + blob directory
- A stable HTTP surface (`/memories`, `/brain/search`,
  `/security/scan`, `/preference_pairs`, …)
- A companion embedder process (`ruvultra-embedder`,
  bge-small-en-v1.5, 384-dim, candle-cuda)

And the shared **pi.ruv.io** brain (ADR-059, ADR-060, ADR-064, ADR-150)
aggregates knowledge across contributors: 12K+ memories, 1.2M+ graph
edges, semantic search via RuvLtra embeddings.

Obsidian users today have no principled way to push their notes through
AIDefence, index them against DiskANN, surface semantically related
memories, or participate in pi.ruv.io's collective knowledge graph.
Existing "semantic search" plugins duplicate vector infrastructure
inside the plugin process — fragmenting the knowledge base and
bypassing AIDefence entirely.

## Decision

Ship an Obsidian plugin that is a **thin client** over the existing
brain servers. The plugin owns no vector storage, no embedder, no
security regex — it orchestrates user-visible workflows and delegates
every semantic operation to:

1. **Local brain** (`mcp-brain-server-local` on `127.0.0.1:9876`)
2. **Local embedder** (`ruvultra-embedder` on `127.0.0.1:9877`;
   stub fallback at `:19877` when real one is absent)
3. **pi.ruv.io shared brain** (`https://pi.ruv.io`, bearer-gated)

The plugin's only persistent state is:

- `data.json` — settings + `indexState` mapping vault paths to memory
  ids (so click-to-open works without querying the brain).
- Frontmatter `brain-category` / `#brain/<category>` tags written by
  the graph overlay, reversible.

### Architectural principle

> The plugin stores mappings, not memories. Memories live in the
> brain. If the brain is wiped, the plugin resurrects its mapping by
> re-running Bulk-sync.

### Module layout (`examples/obsidian-brain/src/`)

| Module | Responsibility |
| --- | --- |
| `main.ts` | Plugin lifecycle, command registration, status bar, persistence |
| `brain.ts` | HTTP client for the local brain + embedder + AIDefence scan |
| `indexer.ts` | Per-file auto-index, bulk sync, rename / delete handling, content-hash dedup |
| `settings.ts` | Settings tab; schema for all persisted options |
| `search-modal.ts` | `Cmd+Shift+B` semantic search UI with fuzzy fallback when brain offline |
| `related-view.ts` | Right-panel `ItemView` — refreshes on active-leaf change |
| `bulk-sync.ts` | Modal with progress bar, include/exclude filters |
| `dpo.ts` | Preference-pair workflow + export to `Brain/Exports/` |
| `graph-overlay.ts` | Category color groups in `.obsidian/graph.json`; reversible |
| `pi-client.ts` | pi.ruv.io HTTP client (`/v1/status`, `/v1/memories/list`, `/v1/memories/search`) |
| `pi-sync.ts` | pi pull modal + direct-search modal; mirrors into local brain + `Brain/Pi/` |

## Consequences

### Positive

- **Zero duplication of vector infra** — one brain, many clients.
- **AIDefence is non-bypassable** — every note goes through the local
  brain's `/memories` endpoint, which runs the regex set inline.
- **Cross-client consistency** — the same memory id referenced by the
  CLI, MCP server, or agent runtime appears in Obsidian's Related panel.
- **Offline-aware** — search modal falls back to local fuzzy ranking
  when the brain is unreachable; auto-index fails *closed* rather than
  silently leaking unscanned notes.
- **Pi integration pulls the collective brain into the personal vault**
  without coupling vault storage to pi's retention policies.

### Negative / tradeoffs

- Requires a local brain subprocess — not a drop-in for users who
  cannot run a Rust binary alongside Obsidian.
- Network round-trip per search (sub-30 ms for local, ~200 ms for pi).
- Embedding dim is determined by the running embedder — mixing dims in
  the same scratch brain corrupts DiskANN. Mitigated by a dim-mismatch
  guard in `scripts/seed-dev.py` that refuses to seed when brain dim
  ≠ embedder dim.

### Neutral

- Plugin is Obsidian-desktop-only (`isDesktopOnly: false` in
  manifest but the brain subprocess requirement is desktop-bound).
- Bundle size ≈30 KB (esbuild, minified).

## Scope — delivered

### Phase 1 — MVP (~400 LOC target, delivered ~650 LOC)

- [x] `Cmd+Shift+B` semantic search modal with fuzzy fallback
- [x] Auto-index on save (debounced, AIDefence-scanned)
- [x] Settings tab (brain URL, embedder URL, category, debounce, filters)
- [x] Status bar showing live health + memory count

### Phase 2 — Related + sync (~300 LOC target, delivered ~420 LOC)

- [x] Right-panel Related memories view, refreshes on leaf change
- [x] Bulk-sync vault → brain modal with include/exclude + progress
- [x] Rename / delete event handlers that keep `indexState` consistent

### Phase 3 — Graph + DPO (~300 LOC target, delivered ~440 LOC)

- [x] Graph overlay: writes `tag:#brain/<category>` color groups to
  `.obsidian/graph.json`, stamps notes with matching tags, reversible
- [x] DPO preference-pair workflow: mark chosen → pair with rejected →
  export table under `Brain/Exports/`

### Beyond the original ADR

- [x] **pi.ruv.io integration** — client, pull modal, search modal,
  status command, settings section. Mirrors memories into
  `Brain/Pi/<title>.md` stubs with `pi-id` + `pi-source` frontmatter.
- [x] **Real embedder autodetect** — `scripts/run-dev.sh` probes
  `:9877` for the real `ruvultra-embedder` (bge-small-en-v1.5, 384-dim,
  candle-cuda) and uses it directly when available; falls back to an
  in-process 16-dim stub for offline development.
- [x] **Dim-mismatch guard** — seed helper refuses to seed when the
  brain already holds vectors at a different dim than the current
  embedder.

## Testing strategy

**No mocks.** All tests hit real services. Three suites live under
`examples/obsidian-brain/tests/`:

| Suite | Gate | Validates |
| --- | --- | --- |
| `protocol/brain-server.test.ts` | always-on | Spins up a real `mcp-brain-server-local` subprocess + in-process embedder; asserts every endpoint shape the plugin parses. 9 tests. |
| `protocol/pi-server.test.ts` | `BRAIN_API_KEY` | Asserts `/v1/status`, bearer-gated `/v1/memories/list` + `/v1/memories/search`, and 401 without the bearer. 4 tests. |
| `e2e/obsidian-e2e.test.ts` | `OBSIDIAN_E2E=1` | Downloads the real Obsidian AppImage, extracts it (no FUSE needed at runtime), provisions a disposable vault with an isolated HOME, launches Obsidian under `xvfb-run`, and runs a companion *harness plugin* that exercises commands + views from inside the real Obsidian runtime. 9 harness checks. |

Current status: **13/13 protocol tests pass**, **9/9 E2E harness
checks pass** on a real Obsidian 1.6.5 AppImage.

## Operational

### Dev-session bootstrap

`examples/obsidian-brain/scripts/run-dev.sh` boots everything in one
command:

1. Builds `mcp-brain-server-local` (release) if missing.
2. Builds the plugin bundle (`main.js`) if missing.
3. Extracts the Obsidian AppImage once (cached under
   `~/.cache/obsidian-brain-e2e/`).
4. Probes `:9877` for the real embedder, starts a stub only if absent.
5. Starts the brain subprocess pointed at the scratch vault.
6. Delegates seeding to `scripts/seed-dev.py` (idempotent: POSTs each
   note, captures memory ids, writes complete `indexState`, surfaces
   AIDefence 422s, pulls pi.ruv.io memories when `BRAIN_API_KEY` is
   set, writes `.obsidian/graph.json` color groups).
7. Launches real Obsidian under an isolated `HOME` — never touches the
   developer's real Obsidian vault registry.

Environment knobs: `BRAIN_API_KEY`, `PI_LIMIT`, `PI_QUERY`, `PI_URL`,
`BRAIN_PORT`, `EMBED_PORT`, `RUVBRAIN_BIN`.

### Distribution

Plugin is installable by:

- `scripts/setup.sh <vault>` — copies the built bundle into a vault's
  `.obsidian/plugins/obsidian-brain/`.
- BRAT — once the repo publishes a tagged GitHub release with
  `main.js`, `manifest.json`, `styles.css` as release assets.

## Roadmap — phase 4 capabilities

The following capabilities build on the delivered surface and are
tracked here for future PRs:

| Priority | Capability | Notes |
| --- | --- | --- |
| P0 | **Q&A / RAG modal** | Retrieves top-k memories for a question and renders a grounded answer inline. Can be LLM-free (just top-k context cards) or wired to a local LLM if available. |
| P0 | **pi.ruv.io write-through** | `POST /v1/memories` to publish selected notes back to the shared brain. Requires bearer + confirmation dialog. |
| P1 | **Find similar to selection** | Right-click on highlighted text → semantic search on just that fragment. Uses `editor.getSelection()`. |
| P1 | **Tag / category filter on search** | `Cmd+Shift+B` accepts `category:<x>` prefix or exposes a dropdown. Backend already supports it via `?category=`. |
| P1 | **Pi pull category filter** | New setting `piPullCategory` + `list?category=` on the pull. Avoids default pulls dominated by self-reflection training logs. |
| P2 | **Jump-to-passage on result open** | Find the matching span and scroll to it rather than opening the whole note at line 1. |
| P2 | **Inline wikilink suggestions** | Command analyzes active note, finds semantically related vault notes, proposes `[[links]]`. |
| P2 | **Offline queue** | On transient brain unavailability, queue pending writes in `data.json.pending[]` and replay on reconnect. |
| P3 | **Memory explorer view** | Dedicated panel listing brain memories with category/date filters — complements the graph view. |
| P3 | **MMR diversification** | Optional post-process of search results to diversify redundant near-duplicates. |
| P3 | **Keyboard nav in Related panel** | Arrow keys + Enter, no mouse required. |
| P3 | **Daily recall note** | Auto-surface "memories from this day last year". |
| P4 | **Multi-brain** | Federated search across local + pi + team brains; result provenance per row. |
| P4 | **Canvas integration** | Drag brain memories onto Obsidian Canvas as cards. |

## Security / privacy

- Bearer token for pi.ruv.io is persisted in the Obsidian vault's
  `data.json` — treated as sensitive by Obsidian Sync; users should
  not sync the plugin's data.json if they sync their vault.
- AIDefence scan runs server-side in Rust; the plugin never inspects
  content for security patterns.
- The plugin talks exclusively to loopback brain + bearer-authed pi.
  No third-party analytics, no telemetry.

## References

- ADR-146 — DiskANN Vamana implementation (local brain's search engine)
- ADR-059 / ADR-060 / ADR-064 — pi.ruv.io architecture + capabilities
- ADR-082 — AIDefence hardening
- ADR-150 — pi Brain + RuvLtra via Tailscale (embedding upgrade on pi)
- `examples/obsidian-brain/README.md` — user-facing install + ops
- PR #365 — initial delivery on the `feat/obsidian-brain-plugin` branch
