// Vector search + graph expansion with two @ruvector packages.
//
//   @ruvector/core       native HNSW vector index: "which notes look like my query?"
//   @ruvector/graph-wasm property graph + Cypher:  "what do those notes link to?"
//
// Runs offline. The toy embedding below is a hashed bag of words, so matches are
// lexical, not semantic. Swap `embed()` for a real embedding model in practice.

import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import core from '@ruvector/core';
import { initSync, GraphDB } from '@ruvector/graph-wasm';

const DIM = 64;

function embed(text) {
  const v = new Float32Array(DIM);
  for (const word of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    let h = 2166136261;
    for (const ch of word) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
    v[(h >>> 0) % DIM] += 1;
  }
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}

const notes = [
  { id: 'hnsw', text: 'HNSW graph index for fast approximate nearest neighbor vector search' },
  { id: 'quant', text: 'Quantization compresses vectors to cut memory for nearest neighbor search' },
  { id: 'cypher', text: 'Cypher queries match nodes and relationships in a property graph' },
  { id: 'rag', text: 'Retrieval augmented generation fetches relevant documents before the LLM answers' },
  { id: 'router', text: 'Semantic routing sends a query to the agent whose embedding is closest' },
];
const links = [
  ['rag', 'hnsw', 'USES'],
  ['rag', 'cypher', 'USES'],
  ['router', 'hnsw', 'USES'],
  ['hnsw', 'quant', 'PAIRS_WITH'],
];

// 1. Vector index. @ruvector/core needs Float32Array vectors, PascalCase metric
//    names, and writes to ./ruvector.db unless storagePath is set.
const workDir = mkdtempSync(join(tmpdir(), 'ruvector-example-'));
const vectors = new core.VectorDb({
  dimensions: DIM,
  distanceMetric: 'Cosine',
  storagePath: join(workDir, 'vectors.db'),
});
await vectors.insertBatch(notes.map((n) => ({ id: n.id, vector: embed(n.text) })));

// 2. Graph. The web-target WASM build is initialised synchronously from the file on disk.
const require = createRequire(import.meta.url);
initSync({ module: readFileSync(require.resolve('@ruvector/graph-wasm/ruvector_graph_wasm_bg.wasm')) });
const graph = new GraphDB('cosine');
const nodeId = new Map(); // note id -> graph node id
const noteOf = new Map(); // graph node id -> note id
for (const n of notes) {
  // Storing the embedding as a node property lets the node join a hyperedge later.
  const gid = graph.createNode(['Note'], { noteId: n.id, text: n.text, embedding: Array.from(embed(n.text)) });
  nodeId.set(n.id, gid);
  noteOf.set(gid, n.id);
}
for (const [from, to, type] of links) graph.createEdge(nodeId.get(from), nodeId.get(to), type, {});
graph.createHyperedge([nodeId.get('hnsw'), nodeId.get('quant'), nodeId.get('router')], 'nearest-neighbor stack', null, 0.9);

// 3. Query: top-k by vector similarity, then expand each hit through the graph.
const { edges } = await graph.query('MATCH (a:Note)-[r]->(b:Note) RETURN a, r, b');
const query = 'retrieve relevant documents for an LLM answer';
const hits = await vectors.search({ vector: embed(query), k: 2 });

console.log(`query: "${query}"\n`);
for (const hit of hits) {
  // With the Cosine metric, score is a distance: 0 means identical direction.
  console.log(`${hit.id}  (cosine distance ${hit.score.toFixed(3)})`);
  for (const e of edges) {
    if (e.from === nodeId.get(hit.id)) console.log(`  -[${e.type}]-> ${noteOf.get(e.to)}`);
    if (e.to === nodeId.get(hit.id)) console.log(`  <-[${e.type}]- ${noteOf.get(e.from)}`);
  }
}

const stats = graph.stats();
console.log(`\nvectors: ${await vectors.len()}, graph: ${stats.nodeCount} nodes, ${stats.edgeCount} edges, ${stats.hyperedgeCount} hyperedge`);

rmSync(workDir, { recursive: true, force: true });
