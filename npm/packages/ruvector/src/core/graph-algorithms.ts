/**
 * Graph Algorithms - MinCut, Spectral Clustering, Community Detection
 *
 * Provides graph partitioning and clustering algorithms for:
 * - Code module detection
 * - Dependency clustering
 * - Architecture analysis
 * - Refactoring suggestions
 */

export interface Graph {
  nodes: string[];
  edges: Array<{ from: string; to: string; weight?: number }>;
  adjacency: Map<string, Map<string, number>>;
}

export interface Partition {
  groups: string[][];
  cutWeight: number;
  modularity: number;
}

export interface SpectralResult {
  clusters: Map<string, number>;
  eigenvalues: number[];
  coordinates: Map<string, number[]>;
}

/**
 * Build adjacency representation from edges
 */
export function buildGraph(
  nodes: string[],
  edges: Array<{ from: string; to: string; weight?: number }>
): Graph {
  const adjacency = new Map<string, Map<string, number>>();

  for (const node of nodes) {
    adjacency.set(node, new Map());
  }

  for (const { from, to, weight = 1 } of edges) {
    if (!adjacency.has(from)) adjacency.set(from, new Map());
    if (!adjacency.has(to)) adjacency.set(to, new Map());

    // Undirected graph - add both directions
    adjacency.get(from)!.set(to, weight);
    adjacency.get(to)!.set(from, weight);
  }

  return { nodes, edges, adjacency };
}

/**
 * Minimum Cut (Stoer-Wagner algorithm)
 *
 * Finds the minimum weight cut that partitions the graph into two parts.
 * Useful for finding loosely coupled module boundaries.
 */
export function minCut(graph: Graph): Partition {
  const n = graph.nodes.length;
  if (n < 2) {
    return { groups: [graph.nodes], cutWeight: 0, modularity: 0 };
  }

  // Copy adjacency for modification
  const adj = new Map<string, Map<string, number>>();
  for (const [node, neighbors] of graph.adjacency) {
    adj.set(node, new Map(neighbors));
  }

  let minCutWeight = Infinity;
  let bestPartition: string[][] = [];
  const merged = new Map<string, string[]>(); // Track merged nodes

  for (const node of graph.nodes) {
    merged.set(node, [node]);
  }

  let remaining = [...graph.nodes];

  // Stoer-Wagner phases
  while (remaining.length > 1) {
    // Maximum adjacency search
    const inA = new Set<string>([remaining[0]]);
    const weights = new Map<string, number>();

    for (const node of remaining) {
      if (!inA.has(node)) {
        weights.set(node, adj.get(remaining[0])?.get(node) || 0);
      }
    }

    let lastAdded = remaining[0];
    let beforeLast = remaining[0];

    while (inA.size < remaining.length) {
      // Find node with maximum weight to A
      let maxWeight = -Infinity;
      let maxNode = '';

      for (const [node, weight] of weights) {
        if (!inA.has(node) && weight > maxWeight) {
          maxWeight = weight;
          maxNode = node;
        }
      }

      if (!maxNode) break;

      beforeLast = lastAdded;
      lastAdded = maxNode;
      inA.add(maxNode);

      // Update weights
      for (const [neighbor, w] of adj.get(maxNode) || []) {
        if (!inA.has(neighbor)) {
          weights.set(neighbor, (weights.get(neighbor) || 0) + w);
        }
      }
    }

    // Cut of the phase
    const cutWeight = weights.get(lastAdded) || 0;

    if (cutWeight < minCutWeight) {
      minCutWeight = cutWeight;
      const lastGroup = merged.get(lastAdded) || [lastAdded];
      const otherNodes = remaining.filter(n => n !== lastAdded).flatMap(n => merged.get(n) || [n]);
      bestPartition = [lastGroup, otherNodes];
    }

    // Merge last two nodes
    if (remaining.length > 1) {
      // Merge lastAdded into beforeLast
      const mergedNodes = [...(merged.get(beforeLast) || []), ...(merged.get(lastAdded) || [])];
      merged.set(beforeLast, mergedNodes);

      // Update adjacency
      for (const [neighbor, w] of adj.get(lastAdded) || []) {
        if (neighbor !== beforeLast) {
          const current = adj.get(beforeLast)?.get(neighbor) || 0;
          adj.get(beforeLast)?.set(neighbor, current + w);
          adj.get(neighbor)?.set(beforeLast, current + w);
        }
      }

      // Remove lastAdded
      remaining = remaining.filter(n => n !== lastAdded);
      adj.delete(lastAdded);
      for (const [, neighbors] of adj) {
        neighbors.delete(lastAdded);
      }
    }
  }

  const modularity = calculateModularity(graph, bestPartition);

  return {
    groups: bestPartition.filter(g => g.length > 0),
    cutWeight: minCutWeight,
    modularity,
  };
}

/**
 * Spectral Clustering (using power iteration)
 *
 * Uses graph Laplacian eigenvectors for clustering.
 * Good for finding natural clusters in code dependencies.
 */
export function spectralClustering(graph: Graph, k: number = 2): SpectralResult {
  // Spectral clustering uses the eigenvectors of the normalized Laplacian
  // L = I - D^(-1/2) A D^(-1/2) with the SMALLEST eigenvalues (they vary slowly inside a well-connected group). Plain power
  // iteration on L converges to the LARGEST ones, so the clusters ignored the groups (two planted groups came out mixed).
  // Now: power iteration on M = 2I - L (same eigenvectors, order reversed), starting from L's known first eigenvector
  // D^(1/2)·1, rows normalized (Ng–Jordan–Weiss), and a deterministic k-means start (farthest point), so a run is repeatable.
  const n = graph.nodes.length;
  const nodeIndex = new Map(graph.nodes.map((node, i) => [node, i]));
  const clusters = new Map<string, number>();
  if (n === 0) {
    return { clusters, eigenvalues: [], coordinates: new Map() };
  }

  const degree = new Float64Array(n);
  const A: Array<Array<[number, number]>> = Array.from({ length: n }, () => []);
  for (const [node, neighbors] of graph.adjacency) {
    const i = nodeIndex.get(node);
    if (i === undefined) continue;
    for (const [neighbor, weight] of neighbors) {
      const j = nodeIndex.get(neighbor);
      if (j === undefined || j === i) continue;
      A[i].push([j, weight]);
      degree[i] += weight;
    }
  }
  const invSqrt = Float64Array.from(degree, (d) => (d > 0 ? 1 / Math.sqrt(d) : 0));
  // y = M x = 2x - L x = x + D^(-1/2) A D^(-1/2) x   (an isolated node has L_ii = 0, so M_ii = 2)
  const applyM = (x: Float64Array) => {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = degree[i] > 0 ? x[i] : 2 * x[i];
      for (const [j, w] of A[i]) s += invSqrt[i] * w * invSqrt[j] * x[j];
      y[i] = s;
    }
    return y;
  };
  const rayleighL = (x: Float64Array) => { const y = applyM(x); return 2 - dotProduct(x, y); };
  const deflate = (v: Float64Array, basis: Float64Array[]) => {
    for (const b of basis) { const d = dotProduct(v, b); for (let i = 0; i < n; i++) v[i] -= d * b[i]; }
  };

  const m = Math.min(k, n);
  const basis: Float64Array[] = [];
  const eigenvalues: number[] = [];
  // L's first eigenvector is D^(1/2)·1 (eigenvalue 0) when there are edges
  const first = Float64Array.from(degree, (d) => Math.sqrt(d));
  if (dotProduct(first, first) > 0) { normalize(first); basis.push(first); eigenvalues.push(rayleighL(first)); }
  let seed = 12345;
  const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
  while (basis.length < m) {
    let v = Float64Array.from({ length: n }, rand);
    deflate(v, basis); normalize(v);
    for (let iter = 0; iter < 500; iter++) {
      const next = applyM(v);
      deflate(next, basis); normalize(next);
      let diff = 0; for (let i = 0; i < n; i++) diff = Math.max(diff, Math.abs(next[i] - v[i]));
      v = next;
      if (diff < 1e-10) break;
    }
    basis.push(v);
    eigenvalues.push(rayleighL(v));
  }

  const coordinates = new Map<string, number[]>();
  const rows: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = basis.map((b) => b[i]);
    const len = Math.sqrt(row.reduce((a, x) => a + x * x, 0));
    const unit = len > 0 ? row.map((x) => x / len) : row;
    rows.push(unit);
    coordinates.set(graph.nodes[i], unit);
  }

  const clusterAssignment = kMeans(rows, k);
  for (let i = 0; i < n; i++) {
    clusters.set(graph.nodes[i], clusterAssignment[i]);
  }
  return { clusters, eigenvalues, coordinates };
}

/**
 * Louvain Community Detection
 *
 * Greedy modularity optimization for finding communities.
 * Good for detecting natural module boundaries.
 */
export function louvainCommunities(graph: Graph): Map<string, number> {
  // Louvain done the standard way, both phases. The version before compared each
  // neighbour community's gain with 0 without subtracting the loss of leaving its own (whose total still counted the
  // node), so nodes swapped back and forth and the loop never ended (a 50-node graph ran forever); m was also halved
  // twice, and there was no aggregation phase. Now: (1) local moving, a node REMOVED from its community before the gains
  // are compared, moving only when another community beats its own; (2) the communities become single nodes and (1)
  // runs again, until nothing merges. Pass and level caps guard the end.
  const strength = new Map<string, number>();
  let twoM = 0;
  for (const node of graph.nodes) {
    let w = 0;
    for (const [neighbor, weight] of graph.adjacency.get(node) || []) if (neighbor !== node) w += weight;
    strength.set(node, w);
    twoM += w; // both directions are stored, so the sum of strengths is 2m
  }
  const member = new Map<string, string>(); // original node -> its node at the current level
  for (const node of graph.nodes) member.set(node, node);
  if (twoM === 0) {
    const alone = new Map<string, number>();
    graph.nodes.forEach((n, i) => alone.set(n, i));
    return alone;
  }

  let nodes = [...graph.nodes];
  let adj = graph.adjacency as Map<string, Map<string, number>>;
  let k = strength;
  for (let level = 0; level < 32; level++) {
    const part = louvainLocalMoving(nodes, adj, k, twoM);
    const count = new Set(part.values()).size;
    if (count === nodes.length) break; // nothing merged: done
    // aggregate: each community becomes one node; edges between communities summed (inside ones dropped: they no
    // longer decide a move, and each node keeps its full strength in k)
    const nextAdj = new Map<string, Map<string, number>>();
    const nextK = new Map<string, number>();
    const id = (c: number) => `L${level}c${c}`;
    for (const u of nodes) {
      const cu = id(part.get(u)!);
      nextK.set(cu, (nextK.get(cu) || 0) + k.get(u)!);
      if (!nextAdj.has(cu)) nextAdj.set(cu, new Map());
      for (const [v, w] of adj.get(u) || []) {
        const cv = id(part.get(v)!);
        if (cv === cu) continue;
        nextAdj.get(cu)!.set(cv, (nextAdj.get(cu)!.get(cv) || 0) + w);
      }
    }
    for (const [orig, at] of member) member.set(orig, id(part.get(at)!));
    nodes = [...nextK.keys()];
    adj = nextAdj;
    k = nextK;
  }

  // Number the final communities contiguously, in node order
  const communities = new Map<string, number>();
  const renumber = new Map<string, number>();
  for (const node of graph.nodes) {
    const at = member.get(node)!;
    if (!renumber.has(at)) renumber.set(at, renumber.size);
    communities.set(node, renumber.get(at)!);
  }
  return communities;
}

/** Louvain phase 1 (local moving) on one level; returns each node's community. */
function louvainLocalMoving(
  nodes: string[],
  adj: Map<string, Map<string, number>>,
  k: Map<string, number>,
  twoM: number
): Map<string, number> {
  const community = new Map<string, number>();
  nodes.forEach((n, i) => community.set(n, i));
  const tot = new Map<number, number>(); // Σ_tot: the summed strength of each community
  for (const n of nodes) tot.set(community.get(n)!, k.get(n)!);
  for (let pass = 0, improved = true; improved && pass < 100; pass++) {
    improved = false;
    for (const node of nodes) {
      const own = community.get(node)!;
      const ki = k.get(node)!;
      const kIn = new Map<number, number>(); // edge weight from this node into each neighbouring community
      for (const [neighbor, weight] of adj.get(node) || []) {
        if (neighbor === node) continue;
        const c = community.get(neighbor)!;
        kIn.set(c, (kIn.get(c) || 0) + weight);
      }
      tot.set(own, tot.get(own)! - ki); // take the node out of its community
      // gain of inserting it into c (up to a constant factor): k_i,in(c) - k_i * Σ_tot(c) / 2m
      const gain = (c: number) => (kIn.get(c) || 0) - (ki * (tot.get(c) || 0)) / twoM;
      let best = own;
      let bestGain = gain(own);
      for (const c of kIn.keys()) {
        const g = gain(c);
        if (g > bestGain + 1e-12) { best = c; bestGain = g; }
      }
      tot.set(best, (tot.get(best) || 0) + ki);
      if (best !== own) { community.set(node, best); improved = true; }
    }
  }
  return community;
}

/**
 * Calculate modularity of a partition
 */
export function calculateModularity(graph: Graph, partition: string[][]): number {
  let m = 0;
  for (const { weight = 1 } of graph.edges) {
    m += weight;
  }
  // Adjacency is symmetric, but graph.edges already counts each edge once.

  if (m === 0) return 0;

  let modularity = 0;

  for (const group of partition) {
    const groupSet = new Set(group);

    // Edges within group
    let inGroup = 0;
    let degreeSum = 0;

    for (const node of group) {
      for (const [neighbor, weight] of graph.adjacency.get(node) || []) {
        if (groupSet.has(neighbor)) {
          inGroup += weight;
        }
        degreeSum += weight;
      }
    }

    inGroup /= 2; // Count each edge once
    modularity += inGroup / m - Math.pow(degreeSum / (2 * m), 2);
  }

  return modularity;
}

/**
 * Find bridges (edges whose removal disconnects components)
 */
export function findBridges(graph: Graph): Array<{ from: string; to: string }> {
  const bridges: Array<{ from: string; to: string }> = [];
  const visited = new Set<string>();
  const discovery = new Map<string, number>();
  const low = new Map<string, number>();
  const parent = new Map<string, string | null>();
  let time = 0;

  function dfs(node: string) {
    visited.add(node);
    discovery.set(node, time);
    low.set(node, time);
    time++;

    for (const [neighbor] of graph.adjacency.get(node) || []) {
      if (!visited.has(neighbor)) {
        parent.set(neighbor, node);
        dfs(neighbor);

        low.set(node, Math.min(low.get(node)!, low.get(neighbor)!));

        if (low.get(neighbor)! > discovery.get(node)!) {
          bridges.push({ from: node, to: neighbor });
        }
      } else if (neighbor !== parent.get(node)) {
        low.set(node, Math.min(low.get(node)!, discovery.get(neighbor)!));
      }
    }
  }

  for (const node of graph.nodes) {
    if (!visited.has(node)) {
      parent.set(node, null);
      dfs(node);
    }
  }

  return bridges;
}

/**
 * Find articulation points (nodes whose removal disconnects components)
 */
export function findArticulationPoints(graph: Graph): string[] {
  const points: string[] = [];
  const visited = new Set<string>();
  const discovery = new Map<string, number>();
  const low = new Map<string, number>();
  const parent = new Map<string, string | null>();
  let time = 0;

  function dfs(node: string) {
    visited.add(node);
    discovery.set(node, time);
    low.set(node, time);
    time++;

    let children = 0;

    for (const [neighbor] of graph.adjacency.get(node) || []) {
      if (!visited.has(neighbor)) {
        children++;
        parent.set(neighbor, node);
        dfs(neighbor);

        low.set(node, Math.min(low.get(node)!, low.get(neighbor)!));

        // Root with 2+ children or non-root with low[v] >= disc[u]
        if (
          (parent.get(node) === null && children > 1) ||
          (parent.get(node) !== null && low.get(neighbor)! >= discovery.get(node)!)
        ) {
          if (!points.includes(node)) {
            points.push(node);
          }
        }
      } else if (neighbor !== parent.get(node)) {
        low.set(node, Math.min(low.get(node)!, discovery.get(neighbor)!));
      }
    }
  }

  for (const node of graph.nodes) {
    if (!visited.has(node)) {
      parent.set(node, null);
      dfs(node);
    }
  }

  return points;
}

// Helper functions
function normalize(v: Float64Array) {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  const norm = Math.sqrt(sum);
  if (norm > 0) {
    for (let i = 0; i < v.length; i++) {
      v[i] /= norm;
    }
  }
}

function dotProduct(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function kMeans(points: number[][], k: number, maxIter: number = 100): number[] {
  const n = points.length;
  if (n === 0 || k === 0) return [];

  const dim = points[0].length;

  // deterministic farthest-point initialization (it was random: a run could differ from
  // the last, and two starting points in one group split that group)
  const dist2 = (a: number[], b: number[]) => a.reduce((s, x, d) => s + (x - b[d]) * (x - b[d]), 0);
  const centroids: number[][] = [[...points[0]]];
  while (centroids.length < Math.min(k, n)) {
    let best = 0, bestD = -1;
    for (let i = 0; i < n; i++) {
      const d = Math.min(...centroids.map((c) => dist2(points[i], c)));
      if (d > bestD) { bestD = d; best = i; }
    }
    centroids.push([...points[best]]);
  }

  const assignment = new Array(n).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign points to nearest centroid
    let changed = false;
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let minC = 0;
      for (let c = 0; c < centroids.length; c++) {
        let dist = 0;
        for (let d = 0; d < dim; d++) {
          dist += Math.pow(points[i][d] - centroids[c][d], 2);
        }
        if (dist < minDist) {
          minDist = dist;
          minC = c;
        }
      }
      if (assignment[i] !== minC) {
        assignment[i] = minC;
        changed = true;
      }
    }

    if (!changed) break;

    // Update centroids
    const counts = new Array(k).fill(0);
    for (let c = 0; c < centroids.length; c++) {
      for (let d = 0; d < dim; d++) {
        centroids[c][d] = 0;
      }
    }

    for (let i = 0; i < n; i++) {
      const c = assignment[i];
      counts[c]++;
      for (let d = 0; d < dim; d++) {
        centroids[c][d] += points[i][d];
      }
    }

    for (let c = 0; c < centroids.length; c++) {
      if (counts[c] > 0) {
        for (let d = 0; d < dim; d++) {
          centroids[c][d] /= counts[c];
        }
      }
    }
  }

  return assignment;
}

export default {
  buildGraph,
  minCut,
  spectralClustering,
  louvainCommunities,
  calculateModularity,
  findBridges,
  findArticulationPoints,
};
