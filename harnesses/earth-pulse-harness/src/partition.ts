// SPDX-License-Identifier: MIT
// Signal-class partitioning of the pulse-event graph via ruVector dynamic
// MinCut (`@ruvector/mincut-wasm`). This is the "cluster the events without
// forcing one theory too early" step: build a similarity graph over event
// embeddings, then let a dynamic minimum-cut / expander decomposition tell us
// how many distinct signal classes the population splits into (e.g. ocean-forced
// vs. gliding-tremor vs. artifact) and how weakly they are joined.
//
// Dynamic = edges are inserted/deleted as new events stream in and the
// decomposition is re-evaluated, so a sudden jump in the class count is a
// regime change / anomaly (a second mechanism appearing). See ADR-006.
//
// The published wasm's *timed* paths (WasmMinCut.insertEdge, Wrapper.query)
// panic under the Node wasm runtime ("time not implemented"), so we drive the
// untimed `WasmThreeLevelHierarchy` (φ-expander decomposition, arXiv:2512.13105)
// and fall back to an exact connected-components partition if the wasm is
// unavailable — the ADR-150 graceful-degradation pattern the harness follows.

import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { cosineSimilarity } from './embed-events.js';
import type { EventEmbedding } from './types.js';
import type { EmbeddingField } from './memory.js';

export interface PartitionResult {
  /**
   * Number of distinct signal classes the event population splits into. From
   * the ruVector hierarchy this is the Level-2 cluster count (num_clusters),
   * which equals 1 for a coherent population and rises as genuinely separate
   * classes (disconnected / very weakly joined) appear. From the fallback it is
   * the connected-component count.
   */
  classes: number;
  /**
   * φ-expander subgraph count (Level-0). A finer structural decomposition than
   * `classes` — a single dense class can still yield several expanders — so it
   * is reported for diagnostics, not as the class count. Undefined for the
   * fallback engine.
   */
  expanders?: number;
  vertices: number;
  edges: number;
  /** Which engine produced the result. */
  backend: 'ruvector-mincut-wasm' | 'fallback-connected-components';
}

interface MincutModule {
  WasmThreeLevelHierarchy: new () => {
    insertEdge(u: bigint, v: bigint, weight: number): void;
    build(): void;
    globalMinCut(): number;
    stats(): { num_expanders: number; num_clusters: number; num_vertices: number; num_edges: number };
    free(): void;
  };
}

let cached: MincutModule | null | undefined;

/** Load + init `@ruvector/mincut-wasm` from local bytes (Node has no fetch URL). */
export async function loadMincut(): Promise<MincutModule | null> {
  if (cached !== undefined) return cached;
  try {
    const mod = (await import('@ruvector/mincut-wasm')) as unknown as {
      default: (opts: { module_or_path: Uint8Array }) => Promise<unknown>;
    } & MincutModule;
    const require = createRequire(import.meta.url);
    const wasmPath = require.resolve('@ruvector/mincut-wasm/ruvector_mincut_wasm_bg.wasm');
    await mod.default({ module_or_path: readFileSync(wasmPath) });
    cached = mod;
  } catch {
    cached = null;
  }
  return cached;
}

export interface PartitionOptions {
  /** Embedding facet to compare (default 'waveform'; see ADR-002 — keep facets separate). */
  field?: EmbeddingField;
  /** Cosine-similarity threshold above which two events are joined by an edge. */
  threshold?: number;
}

interface Vertexed {
  eventId: string;
  emb: EventEmbedding;
}

/** Build the weighted similarity edge list (only edges above the threshold). */
function buildEdges(
  events: Vertexed[],
  field: EmbeddingField,
  threshold: number,
): { edges: [number, number, number][]; n: number } {
  const edges: [number, number, number][] = [];
  for (let i = 0; i < events.length; i++) {
    for (let j = i + 1; j < events.length; j++) {
      const s = cosineSimilarity(events[i].emb[field], events[j].emb[field]);
      if (s >= threshold) edges.push([i, j, s]);
    }
  }
  return { edges, n: events.length };
}

/** Connected-components fallback (union-find) — exact, dependency-free. */
function componentsPartition(edges: [number, number, number][], n: number): PartitionResult {
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (const [u, v] of edges) {
    const ru = find(u);
    const rv = find(v);
    if (ru !== rv) parent[ru] = rv;
  }
  const roots = new Set<number>();
  for (let i = 0; i < n; i++) roots.add(find(i));
  return {
    classes: roots.size,
    vertices: n,
    edges: edges.length,
    backend: 'fallback-connected-components',
  };
}

/**
 * Partition the event graph into signal classes. Prefers ruVector dynamic
 * MinCut (φ-expander decomposition); falls back to connected components.
 */
export async function partitionEvents(
  events: Vertexed[],
  opts: PartitionOptions = {},
): Promise<PartitionResult> {
  const field = opts.field ?? 'waveform';
  const threshold = opts.threshold ?? 0.5;
  const { edges, n } = buildEdges(events, field, threshold);

  const mod = await loadMincut();
  if (!mod) return componentsPartition(edges, n);

  const h = new mod.WasmThreeLevelHierarchy();
  try {
    for (const [u, v, w] of edges) h.insertEdge(BigInt(u), BigInt(v), w);
    h.build();
    const s = h.stats();
    return {
      classes: s.num_clusters,
      expanders: s.num_expanders,
      vertices: n,
      edges: edges.length,
      backend: 'ruvector-mincut-wasm',
    };
  } finally {
    h.free();
  }
}

/**
 * Dynamic stream: re-decompose after each batch of events arrives. A jump in
 * the class count signals a new mechanism entering the record (regime change).
 * Returns the class count after each cumulative batch.
 */
export async function streamPartition(
  batches: Vertexed[][],
  opts: PartitionOptions = {},
): Promise<PartitionResult[]> {
  const seen: Vertexed[] = [];
  const out: PartitionResult[] = [];
  for (const batch of batches) {
    seen.push(...batch);
    out.push(await partitionEvents(seen, opts));
  }
  return out;
}
