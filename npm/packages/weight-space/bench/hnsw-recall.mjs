// HNSW (@ruvector/core) top-5 agreement with exact search, on real MiniLM embeddings (eval/adr-corpus.mjs)
// and on uniform random 384-d vectors (the adversarial case). Usage: node bench/hnsw-recall.mjs corpus.json
import { readFileSync } from 'node:fs';
import { FlatVectorStore, RuvectorVectorStore } from '../dist/esm/index.js';
const c = JSON.parse(readFileSync(process.argv[2] ?? 'corpus.json', 'utf8'));
const vecs = [...c.docs.map(d => d.vector), ...c.queries.map(q => q.vector)];
// real embeddings: index 1244 vectors, query with 200 of them perturbed
let s = 7; const r = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
const qs = Array.from({ length: 200 }, (_, i) => vecs[(i * 6) % vecs.length].map(v => v + (r() - 0.5) * 0.02));
for (const [label, data] of [['real-minilm-1244', vecs], ['random-10000', Array.from({ length: 10000 }, () => Array.from({ length: 384 }, () => r() * 2 - 1))]]) {
  const flat = new FlatVectorStore(384); for (let i = 0; i < data.length; i++) await flat.upsert(String(i), data[i]);
  const queries = label.startsWith('real') ? qs : Array.from({ length: 200 }, () => Array.from({ length: 384 }, () => r() * 2 - 1));
  for (const cfg of [{ m: 16, efConstruction: 100, efSearch: 64 }, { m: 32, efConstruction: 200, efSearch: 200 }]) {
    const h = await RuvectorVectorStore.open(384, { hnsw: { ...cfg, maxElements: data.length } });
    for (let i = 0; i < data.length; i++) await h.upsert(String(i), data[i]);
    let agree = 0; let t = 0;
    for (const q of queries) { const a = new Set((await flat.search(q, 5)).map(x => x.id)); const t0 = performance.now(); const hh = await h.search(q, 5); t += performance.now() - t0; agree += hh.filter(x => a.has(x.id)).length; }
    console.log(label, JSON.stringify(cfg), 'recall@5', (agree / (queries.length * 5)).toFixed(3), 'meanMs', (t / queries.length).toFixed(3));
    await h.close();
  }
}
