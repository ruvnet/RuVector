#!/usr/bin/env node
// Build the paraphrase-retrieval corpus from this repo's ADRs and embed it.
//
//   corpus  = ADR titles ("# ADR-123: <title>" -> "<title>"), one document per ADR file
//   queries = up to 3 sentences from the same ADR's body that describe it in other words
//
// A query never contains its ADR's full title (that would leak the answer), never cites another
// ADR number, and is 10-40 words long. Everything is deterministic: same repo -> same corpus.
//
// Usage: node eval/adr-corpus.mjs --adr-dir ../../../docs/adr --ruvector <path-to-ruvector-pkg> --out corpus.json
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createRequire } from 'node:module';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1]]);
  return acc;
}, []));
const adrDir = resolve(args['adr-dir'] ?? '../../../docs/adr');
const out = resolve(args.out ?? 'corpus.json');
const ruvectorPath = args.ruvector ? resolve(args.ruvector) : null;
const PER_DOC = Number(args['per-doc'] ?? 3);

function sentencesOf(body) {
  const kept = [];
  let inCode = false;
  for (const raw of body.split('\n')) {
    const line = raw.trim();
    if (line.startsWith('```')) { inCode = !inCode; continue; }
    if (inCode || !line || /^[#|>\-*\d]/.test(line) || line.startsWith('<')) continue;
    kept.push(line);
  }
  return kept.join(' ')
    .replace(/`[^`]*`/g, (m) => m.slice(1, -1))
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\*\*|__/g, '')
    .split(/(?<=[.!?])\s+(?=[A-Z])/)
    .map((s) => s.trim());
}

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();

const docs = [];
const queries = [];
for (const file of readdirSync(adrDir).filter((f) => /^ADR-\d+.*\.md$/.test(f)).sort()) {
  const text = readFileSync(join(adrDir, file), 'utf8');
  const m = text.match(/^#\s+(.+)$/m);
  if (!m) continue;
  const title = m[1].replace(/^ADR[-\s]?\d+[a-z]?\s*[:.\-—–]\s*/i, '').trim();
  if (title.split(/\s+/).length < 3) continue;
  const body = text.slice(m.index + m[0].length);
  const docId = docs.length;
  const picked = [];
  for (const s of sentencesOf(body)) {
    const words = s.split(/\s+/).length;
    if (words < 10 || words > 40) continue;
    if (/ADR-?\d/i.test(s)) continue;
    if (norm(s).includes(norm(title))) continue;
    if (!/[a-z]{3}/.test(s) || /[{}=;]/.test(s)) continue;
    picked.push(s);
    if (picked.length === PER_DOC) break;
  }
  if (picked.length === 0) continue;
  docs.push({ id: docId, file, title });
  for (const q of picked) queries.push({ text: q, doc: docId });
}

const req = createRequire(join(ruvectorPath ?? process.cwd(), 'package.json'));
const onnx = req(ruvectorPath ? join(ruvectorPath, 'dist/core/onnx-embedder.js') : 'ruvector/dist/core/onnx-embedder.js');
await onnx.initOnnxEmbedder();
const embedAll = async (texts) => {
  const vecs = [];
  for (let i = 0; i < texts.length; i += 64) {
    const batch = await onnx.embedBatch(texts.slice(i, i + 64));
    for (const r of batch) vecs.push(Array.from(r.embedding, (v) => Math.round(v * 1e6) / 1e6));
  }
  return vecs;
};
const docVecs = await embedAll(docs.map((d) => d.title));
const queryVecs = await embedAll(queries.map((q) => q.text));
docs.forEach((d, i) => { d.vector = docVecs[i]; });
queries.forEach((q, i) => { q.vector = queryVecs[i]; });
writeFileSync(out, JSON.stringify({
  source: 'docs/adr titles (corpus) and body sentences (queries)',
  embedder: onnx.getActiveModelId?.() ?? 'all-MiniLM-L6-v2',
  dimensions: docVecs[0].length,
  docs,
  queries,
}));
console.log(`docs=${docs.length} queries=${queries.length} dims=${docVecs[0].length} -> ${out}`);
await onnx.shutdown?.();
process.exit(0);
