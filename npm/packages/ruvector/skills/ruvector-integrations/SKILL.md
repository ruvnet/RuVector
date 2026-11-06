---
name: ruvector-integrations
description: Use the optional native and WASM accelerators that the `ruvector` npm package wires in, including RaBitQ-accelerated hook memory recall, the DiskANN on-disk index (CLI and MCP), MinCut/Louvain/spectral code-graph analysis, and `npx ruvector add` extras. Use when a task involves large hook memory stores, building or searching a vector index on disk, finding module boundaries or communities in a codebase, or deciding which ruvector extra to install. Do not use for generic vector-database questions that the core `ruvector` API already answers.
---

# ruvector integrations

`ruvector` keeps a dependency-light core and loads its accelerators lazily.
Each accelerator below is either an **optional dependency**, installed by
default and skipped silently where it cannot install, or an **extra**, which
you install with `npx ruvector add <name>`. Every path still works when its
package is missing: it falls back to a pure-JS implementation or prints an
install hint.

Check what is installed:

```bash
npx ruvector add --list          # extras and their install state
npx ruvector add --list --json   # machine-readable
```

Over MCP, `hooks_capabilities` reports the same list.

## Hook memory recall: RaBitQ (`@ruvector/rabitq-wasm`, optional dependency)

The MCP tools `hooks_recall` and `hooks_suggest_context`, and the library's
`IntelligenceEngine.recall()`, scan every memory embedding while the store is
small. (The one-shot CLI command `hooks recall` always scans.) Once a store reaches `RUVECTOR_RABITQ_THRESHOLD` memories (default
**5000**; `0` turns it off), recall works in two steps:

1. A 1-bit RaBitQ index proposes a candidate pool of 10 x k.
2. The pool is re-scored with exact cosine similarity, so the scores you get
   back are exact.

How the index behaves:

- **Build timing.** The index is built on the second eligible recall in a
  process. A one-shot hook command never pays the build cost (about 0.4 s for
  5k x 384). The long-lived MCP server builds it once and then answers in about
  0.5 to 3 ms instead of a full scan (about 9 ms at 5k, 38 ms at 20k).
- **Inserts and deletes after the build** take effect immediately, without a
  rebuild. The index rebuilds once those changes reach about 10% of its size.
- **Which engine ran.** `IntelligenceEngine#getStats().recallEngine` reports
  `'rabitq'` or `'brute-force'` for the most recent recall.

If the package is missing, recall stays on the exact scan. Nothing else
changes.

## DiskANN on-disk index (`@ruvector/diskann`, extra)

```bash
npx ruvector add diskann
npx ruvector diskann build ./idx -i vectors.json [--max-degree 64] [--alpha 1.2] [--json]
npx ruvector diskann search ./idx -v 0.1,0.2,0.3 -k 10 [--json]
```

- `vectors.json` is a JSON array of `{ "id": "...", "vector": [ ... ] }`. All
  vectors must share one dimension, and ids must be unique.
- Search distances are squared L2.
- The MCP tools are `diskann_build` (`path`, `vectors`) and `diskann_search`
  (`path`, `vector`, `k`). Their paths are confined to the server's working
  directory (and `RUVECTOR_MCP_ROOTS`).
- From the library, use `buildDiskAnnIndex(dir, records)`,
  `searchDiskAnnIndex(dir, query, k)`, or the lower-level `DiskAnnIndex`.

## Code-graph analysis (`@ruvector/mincut-wasm`, optional dependency)

```bash
npx ruvector hooks graph-mincut src/*.ts
npx ruvector hooks graph-cluster src/*.ts --method louvain
npx ruvector hooks graph-cluster src/*.ts --method spectral --clusters 3
```

These commands build an undirected graph from relative imports between the
files you pass. They recognise both `from './x'` and `require('./x')`, with or
without an extension.

- **`graph-mincut`** returns `groups`, `cutWeight`, `modularity` and `engine`.
  `engine` is `mincut-wasm` when the WASM solver is installed and `typescript`
  otherwise. Both engines are exact and give the same cut.
- **`graph-cluster`** returns `assignments` (file to cluster), `groups` and
  `modularity`. The spectral method also returns `eigenvalues`. Both methods
  are deterministic.

The MCP equivalents are `hooks_graph_mincut` and `hooks_graph_cluster`.

## Other extras

| Extra | Package | What it backs |
|---|---|---|
| `rabitq` | `@ruvector/rabitq-wasm` | `RabitqIndex` library API (also used by recall, above) |
| `diskann` | `@ruvector/diskann` | `ruvector diskann`, `diskann_*` MCP tools |
| `rvlite` | `rvlite` | `rvlite_sql`, `rvlite_cypher`, `rvlite_sparql` MCP tools |
| `rvf` | `@ruvector/rvf` | `ruvector rvf ...`, `rvf_*` MCP tools |
| `graph` | `@ruvector/graph-node` | `CodeGraph` (incl. PageRank) library API |

`npx ruvector add <name>` uses the project's own package manager. It uses
`npm`, `pnpm`, `yarn` or `bun` according to the lockfile, and
`--dry-run` prints the exact command without running it.

## Installing this skill into a project

```bash
npx ruvector skills install            # .claude/skills (Claude Code) and .agents/skills (Codex)
npx ruvector skills install --target codex
npx ruvector skills list
```
