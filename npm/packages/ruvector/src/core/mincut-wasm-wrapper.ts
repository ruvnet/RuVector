/**
 * MinCut WASM Wrapper - optional acceleration + dynamic/routing capabilities
 * from @ruvector/mincut-wasm (Rust, ruvector-mincut 2.3.0).
 *
 * The pure-TS `minCut` in ./graph-algorithms stays the default and the
 * fallback: it is exact (Stoer-Wagner, differentially fuzzed against
 * brute-force enumeration) and needs no native dependency. This wrapper adds
 * what TS cannot offer:
 *
 *   - `DynamicMinCut` — incremental insert/delete/update that maintains the
 *     cut across edits instead of recomputing the whole graph each time.
 *   - `RoadRouter` — exact directed shortest paths with turn restrictions,
 *     closures and landmark acceleration.
 *
 * @ruvector/mincut-wasm is an OPTIONAL dependency. Every entry point here
 * degrades predictably when it is absent: `isMinCutWasmAvailable()` reports
 * false, `minCutFast` transparently falls back to the TS implementation, and
 * the capability classes throw one actionable install error.
 *
 * Vertex IDs cross the WASM boundary as u64, so the bindings require BigInt.
 * That is an implementation detail of the native module; this wrapper accepts
 * plain numbers or strings and does the conversion.
 */

import { buildGraph, minCut as minCutTs } from './graph-algorithms';
import type { Graph, Partition } from './graph-algorithms';

let wasmModule: any = null;
let loadError: Error | null = null;

const INSTALL_HINT =
  'Install with: npm install @ruvector/mincut-wasm';

/** Lazy-load the optional native module (house convention: see gnn-wrapper). */
function getMinCutWasm(): any {
  if (wasmModule) return wasmModule;
  if (loadError) throw loadError;
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    wasmModule = require('@ruvector/mincut-wasm');
    return wasmModule;
  } catch (e: any) {
    loadError = new Error(
      `@ruvector/mincut-wasm is not installed or failed to load: ${e.message}\n` +
        INSTALL_HINT
    );
    throw loadError;
  }
}

/**
 * Whether the native accelerator is usable. Never throws — call this to pick
 * a code path rather than catching from the capability classes.
 */
export function isMinCutWasmAvailable(): boolean {
  try {
    getMinCutWasm();
    return true;
  } catch {
    return false;
  }
}

/** Map arbitrary node labels onto the contiguous u64 IDs the native side wants. */
function indexNodes(nodes: string[]): Map<string, bigint> {
  const ids = new Map<string, bigint>();
  nodes.forEach((n, i) => ids.set(n, BigInt(i)));
  return ids;
}

/**
 * Exact global minimum cut, using the native solver when present and the
 * pure-TS solver otherwise. Both are exact, so the RESULT is the same either
 * way — only the speed differs. Shape matches `minCut` for drop-in use.
 */
export function minCutFast(graph: Graph): Partition {
  if (!isMinCutWasmAvailable()) return minCutTs(graph);

  // The native solver only ever learns about vertices that appear in an edge.
  // A declared-but-edgeless vertex is its own component, so the true minimum
  // cut is 0 -- isolate it. Handing this to the native side would silently
  // drop the vertex and report the cut of the remaining subgraph instead,
  // disagreeing with `minCut` on the same input.
  const touched = new Set<string>();
  for (const { from, to } of graph.edges) {
    touched.add(from);
    touched.add(to);
  }
  if (graph.nodes.length >= 2 && graph.nodes.some((n) => !touched.has(n))) {
    return minCutTs(graph);
  }

  const { WasmMinCut } = getMinCutWasm();
  const ids = indexNodes(graph.nodes);
  const solver = new WasmMinCut();
  let edgeCount = 0;
  for (const { from, to, weight = 1 } of graph.edges) {
    const a = ids.get(from);
    const b = ids.get(to);
    if (a === undefined || b === undefined || a === b) continue;
    solver.insertEdge(a, b, weight);
    edgeCount++;
  }
  // The native solver needs at least one edge to describe a cut; below that
  // the TS path already returns the correct degenerate answer.
  if (edgeCount === 0) return minCutTs(graph);

  // The native side returns { s, t } as JSON with plain numbers, not a tuple.
  const { s, t } = solver.partition() as { s: number[]; t: number[] };
  const back = graph.nodes;
  const groups = [
    (s ?? []).map((i) => back[Number(i)]).filter(Boolean),
    (t ?? []).map((i) => back[Number(i)]).filter(Boolean),
  ].filter((g) => g.length > 0);

  return {
    groups,
    cutWeight: solver.minCutValue(),
    modularity: minCutTs(graph).modularity,
  };
}

/**
 * Incremental minimum cut over an evolving graph — the capability the TS
 * implementation does not have. Edits maintain the cut rather than
 * recomputing from scratch, so a long edit stream is far cheaper than
 * calling `minCut` after every change.
 *
 * Requires @ruvector/mincut-wasm; construction throws with an install hint
 * if it is missing. Check `isMinCutWasmAvailable()` first to branch cleanly.
 */
export class DynamicMinCut {
  private solver: any;
  private ids = new Map<string, bigint>();
  private labels: string[] = [];

  constructor() {
    const { WasmMinCut } = getMinCutWasm();
    this.solver = new WasmMinCut();
  }

  /** Stable u64 ID for a node label, assigned on first use. */
  private id(node: string): bigint {
    let v = this.ids.get(node);
    if (v === undefined) {
      v = BigInt(this.labels.length);
      this.ids.set(node, v);
      this.labels.push(node);
    }
    return v;
  }

  insertEdge(from: string, to: string, weight = 1): number {
    this.solver.insertEdge(this.id(from), this.id(to), weight);
    return this.minCutValue();
  }

  deleteEdge(from: string, to: string): number {
    this.solver.deleteEdge(this.id(from), this.id(to));
    return this.minCutValue();
  }

  updateEdge(from: string, to: string, weight: number): number {
    this.solver.updateEdge(this.id(from), this.id(to), weight);
    return this.minCutValue();
  }

  minCutValue(): number {
    return this.solver.minCutValue();
  }

  /** Current cut as node-label groups, in the same shape as `Partition.groups`. */
  partition(): string[][] {
    const { s, t } = this.solver.partition() as { s: number[]; t: number[] };
    return [
      (s ?? []).map((i) => this.labels[Number(i)]).filter(Boolean),
      (t ?? []).map((i) => this.labels[Number(i)]).filter(Boolean),
    ].filter((g) => g.length > 0);
  }

  isConnected(): boolean {
    return this.solver.isConnected();
  }

  numVertices(): number {
    return this.solver.numVertices();
  }

  numEdges(): number {
    return this.solver.numEdges();
  }
}

export interface RoadArc {
  from: number;
  to: number;
  cost: number;
}

export interface RoadRoute {
  cost: number;
  nodes: number[];
  arcs: number[];
  settled: number;
}

/**
 * Exact directed routing with turn restrictions, closures and landmark A*.
 * Distinct from min-cut: this is a separate directed engine in the same
 * native module, exposed here so the npm package can use it directly.
 *
 * Costs are caller-chosen integer units (e.g. milliseconds). Arc IDs are
 * input positions. Requires @ruvector/mincut-wasm.
 */
export class RoadRouter {
  private router: any;

  /**
   * @param nodeCount number of vertices; arc endpoints must be < nodeCount
   * @param arcs directed arcs, in the order that defines their IDs
   * @param forbiddenTurns pairs of consecutive arc IDs that may not be chained
   */
  constructor(nodeCount: number, arcs: RoadArc[], forbiddenTurns: Array<[number, number]> = []) {
    const { WasmRoadRouter } = getMinCutWasm();
    const endpoints = new Uint32Array(arcs.length * 2);
    const costs = new Uint32Array(arcs.length);
    arcs.forEach((a, i) => {
      endpoints[i * 2] = a.from;
      endpoints[i * 2 + 1] = a.to;
      costs[i] = a.cost;
    });
    const turns = new Uint32Array(forbiddenTurns.length * 2);
    forbiddenTurns.forEach(([a, b], i) => {
      turns[i * 2] = a;
      turns[i * 2 + 1] = b;
    });
    this.router = new WasmRoadRouter(nodeCount, endpoints, costs, turns);
  }

  /** Exact shortest path, or undefined when the target is unreachable. */
  route(source: number, target: number, options: { landmarks?: boolean; budget?: number } = {}): RoadRoute | undefined {
    const { landmarks = false, budget = 5_000_000 } = options;
    return this.router.route(source, target, landmarks, budget) ?? undefined;
  }

  /** Precompute landmark bounds to accelerate repeated queries. */
  prepare(landmarkNodes: number[], budget = 100_000_000): void {
    this.router.prepare(new Uint32Array(landmarkNodes), budget);
  }

  /** Atomic cost update; pass 0xffffffff as a cost to close an arc. */
  update(arcIds: number[], costs: number[]): void {
    this.router.update(new Uint32Array(arcIds), new Uint32Array(costs));
  }
}

export default {
  isMinCutWasmAvailable,
  minCutFast,
  DynamicMinCut,
  RoadRouter,
};
