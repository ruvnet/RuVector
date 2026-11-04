#!/usr/bin/env node
/**
 * gen-package-banners.mjs — deterministic generator for the animated SVG
 * headers shown at the top of each user-facing npm package README.
 *
 * Usage:
 *   node scripts/gen-package-banners.mjs           # write assets/ruvector/packages/<slug>.svg
 *   node scripts/gen-package-banners.mjs --check   # exit 1 if any file on disk differs
 *
 * Style follows the existing RuVector visuals (assets/ruvector/system-*-library-header.svg,
 * build-loop.svg): #0c1324 card, #5cf2c3 accent, Arial, dashed `.wire` flow, staged
 * fades, <title>/<desc> + role="img" for accessibility, and a prefers-reduced-motion
 * block that stops every animation. No external fonts, images or scripts.
 *
 * Output is a pure function of the PACKAGES table (no clock, no randomness). Every
 * SVG is checked for well-formed XML and for the 20 KB size budget before writing.
 * Zero dependencies — Node built-ins only.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { CLASS_CSS, KEYFRAMES, renderMotif, esc } from './lib/package-banner-motifs.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = path.join(ROOT, 'assets/ruvector/packages');
const MAX_BYTES = 20 * 1024;
const RAW_BASE = 'https://raw.githubusercontent.com/ruvnet/RuVector/main/assets/ruvector/packages';

/**
 * slug = directory under npm/packages; name = published npm name;
 * kicker = small caps category; tagline = one line (wrapped to 2 if long);
 * motif = key in scripts/lib/package-banner-motifs.mjs; desc = what the animation shows.
 */
export const PACKAGES = [
  { slug: 'ruvector', name: 'ruvector', kicker: 'VECTOR DATABASE', motif: 'knn', tagline: 'Vector search and persistent agent memory for Node.js — HNSW, ONNX embeddings, GNN and attention.', desc: 'A query point pulses while its nearest neighbours light up one by one.' },
  { slug: 'all', name: '@ruvector/all', kicker: 'UMBRELLA BUNDLE', motif: 'bundle', opts: { items: ['ruvector', 'rvf', 'sona', 'graph', 'rvlite', 'router'] }, tagline: 'Opt-in umbrella bundle that installs the whole RuVector toolkit in one go.', desc: 'Package chips light up in turn and flow into a single install command.' },
  { slug: 'sona', name: '@ruvector/sona', kicker: 'ADAPTIVE LEARNING', motif: 'loop', opts: { steps: ['trajectory', 'LoRA adapt', 'EWC++ guard', 'ReasoningBank'] }, tagline: 'Self-Optimizing Neural Architecture — runtime-adaptive learning with LoRA, EWC++ and ReasoningBank.', desc: 'An adaptation loop cycles through trajectory, LoRA adapt, EWC++ guard and ReasoningBank.' },
  { slug: 'rvf-wasm', name: '@ruvector/rvf-wasm', kicker: 'RVF / WEBASSEMBLY', motif: 'segments', opts: { labels: ['VEC', 'INDEX', 'MANIFEST', 'WITNESS'], cap: 'WASM microkernel · query RVF in browsers + edge', top: 'query' }, tagline: 'RuVector Format WASM microkernel for browser and edge vector operations.', desc: 'A query flows over a segmented RVF container whose segments light up in turn.' },
  { slug: 'rvf-solver', name: '@ruvector/rvf-solver', kicker: 'RVF / SELF-LEARNING SOLVER', motif: 'solver', tagline: 'Self-learning temporal solver — Thompson Sampling, PolicyKernel and ReasoningBank.', desc: 'Four policy arms rise and fall as Thompson sampling picks the best one.' },
  { slug: 'tiny-dancer', name: '@ruvector/tiny-dancer', kicker: 'NEURAL ROUTER', motif: 'fanout', opts: { src: 'request', targets: ['fast model', 'large model', 'tool agent', 'fallback'], broken: 3, cap: 'FastGRNN routing · uncertainty · circuit breaker' }, tagline: 'FastGRNN neural router for AI agent orchestration, with circuit breakers and uncertainty estimates.', desc: 'A request fans out and each route to a model lights in turn; the fallback route is marked by the circuit breaker.' },
  { slug: 'rabitq-wasm', name: '@ruvector/rabitq-wasm', kicker: 'QUANTIZATION / WEBASSEMBLY', motif: 'bits', tagline: 'RaBitQ 1-bit quantized vector index in WebAssembly — 32× compression with high-recall rerank.', desc: 'Float coordinate bars collapse while matching 1-bit codes appear.' },
  { slug: 'emergent-time', name: '@ruvector/emergent-time', kicker: 'AGENTIC TIME', motif: 'clock', tagline: "Measure an AI agent's internal time and 7-state health from per-channel state deltas.", desc: 'Three channel waveforms flow into a stack of seven health states that light in sequence.' },
  { slug: 'rvlite', name: 'rvlite', kicker: 'EMBEDDED VECTOR DATABASE', motif: 'lanes', tagline: 'Lightweight vector database with SQL, SPARQL and Cypher — runs in Node.js, the browser and at the edge.', desc: 'SQL, Cypher and SPARQL lanes light up and flow into one rvlite store.' },
  { slug: 'diskann', name: '@ruvector/diskann', kicker: 'BILLION-SCALE ANN', motif: 'diskhops', tagline: 'DiskANN / Vamana — SSD-friendly approximate nearest-neighbour search with product quantization.', desc: 'Graph hops light up across nodes held in memory while full vectors are fetched from an SSD.' },
  { slug: 'ruvllm-wasm', name: '@ruvector/ruvllm-wasm', kicker: 'RUVLLM / WEBASSEMBLY', motif: 'llm', tagline: 'RuvLLM runtime pieces in WebAssembly — KV cache, chat templates, HNSW routing and MicroLoRA.', desc: 'Tokens pass through a chat template and fill a two-tier KV cache.' },
  { slug: 'graph-wasm', name: '@ruvector/graph-wasm', kicker: 'GRAPH DATABASE / WEBASSEMBLY', motif: 'graph', opts: { cap: 'MATCH (a)-[r]->(b) RETURN b' }, tagline: 'Neo4j-compatible hypergraph database in WebAssembly — Cypher queries, SIMD, knowledge graphs.', desc: 'Graph nodes and edges light up as a Cypher pattern is matched.' },
  { slug: 'ruvector-extensions', name: 'ruvector-extensions', kicker: 'EXTENSIONS', motif: 'extensions', tagline: 'Embeddings, persistence, exports, temporal tracking and UI for the ruvector vector database.', desc: 'Extension modules light up and connect to the ruvector core.' },
  { slug: 'pi-brain', name: '@ruvector/pi-brain', kicker: 'SHARED BRAIN', motif: 'brain', tagline: 'CLI and SDK for π, the RuVector shared brain — share, search and transfer learning across AI sessions.', desc: 'Several sessions send knowledge into a pulsing central π brain.' },
  { slug: 'node', name: '@ruvector/node', kicker: 'NODE.JS', motif: 'gnn', tagline: 'Unified RuVector package for Node.js — high-performance vector database with GNN capabilities.', desc: 'Messages pass along graph edges as nodes light up in turn.' },
  { slug: 'ruvector-cnn', name: '@ruvector/cnn', kicker: 'IMAGE EMBEDDINGS', motif: 'layers', tagline: 'CNN feature extraction for image embeddings — SIMD-optimized, pure Rust compiled to WASM.', desc: 'An image grid passes through shrinking convolution layers into an embedding vector.' },
  { slug: 'ruvbot', name: 'ruvbot', kicker: 'AI ASSISTANT', motif: 'bot', tagline: 'Self-learning AI assistant with vector memory, security controls and multi-model LLM support.', desc: 'Chat messages appear in turn while recalled context flows from a memory store.' },
  { slug: 'postgres-cli', name: '@ruvector/postgres-cli', kicker: 'POSTGRESQL', motif: 'sql', tagline: 'Vector database CLI for PostgreSQL — a pgvector drop-in with SQL functions, GNN layers and attention.', desc: 'A terminal types a vector similarity query that flows into PostgreSQL.' },
  { slug: 'rvf-mcp-server', name: '@ruvector/rvf-mcp-server', kicker: 'MODEL CONTEXT PROTOCOL', motif: 'mcp', tagline: 'MCP server for RuVector Format vector stores — stdio and SSE transports for AI agents.', desc: 'MCP clients connect through the server to an RVF store.' },
  { slug: 'agentic-synth', name: '@ruvector/agentic-synth', kicker: 'SYNTHETIC DATA', motif: 'synth', tagline: 'Synthetic data generator for AI/ML training, RAG systems and agentic workflows.', desc: 'A schema flows into generated data rows that appear one by one.' },
  { slug: 'rvf', name: '@ruvector/rvf', kicker: 'RVF / TYPESCRIPT SDK', motif: 'segments', opts: { labels: ['VEC', 'INDEX', 'MODEL', 'KERNEL', 'WITNESS'], cap: 'one cognitive container · Node (N-API) or WASM backend', top: 'ingest · query · derive' }, tagline: 'Unified TypeScript SDK for the RuVector Format (RVF) cognitive container.', desc: 'Segments of an RVF cognitive container light up in turn as data flows over it.' },
  { slug: 'rvf-node', name: '@ruvector/rvf-node', kicker: 'RVF / NATIVE BINDINGS', motif: 'segments', opts: { labels: ['VEC', 'INDEX', 'MANIFEST', 'WITNESS'], cap: 'Rust RVF engine via N-API · native speed', top: 'Node.js → N-API' }, tagline: 'Native Node.js bindings for the RuVector Format (RVF) vector database, built with Rust via N-API.', desc: 'Calls from Node.js flow over a segmented RVF container whose segments light in turn.' },
  { slug: 'core', name: '@ruvector/core', kicker: 'VECTOR ENGINE', motif: 'hnsw', tagline: 'High-performance vector database core with HNSW indexing, built in Rust for similarity search.', desc: 'A search descends greedily through the layers of an HNSW graph.' },
  { slug: 'router', name: '@ruvector/router', kicker: 'SEMANTIC ROUTER', motif: 'fanout', opts: { src: 'intent', targets: ['search', 'code', 'billing', 'chat'], cap: 'vector intent matching · HNSW · SIMD' }, tagline: 'Semantic router for AI agents — vector-based intent matching with HNSW indexing and SIMD.', desc: 'An intent is matched against routes, each lighting up in turn.' },
  { slug: 'graph-data-generator', name: '@ruvector/graph-data-generator', kicker: 'SYNTHETIC GRAPHS', motif: 'graph', opts: { cap: 'generated entities + relations for Neo4j' }, tagline: 'AI-powered synthetic graph data for Neo4j knowledge graphs, social networks and temporal events.', desc: 'Generated graph nodes and edges appear one after another.' },
  { slug: 'spiking-neural', name: '@ruvector/spiking-neural', kicker: 'SPIKING NEURAL NETWORKS', motif: 'spikes', tagline: 'High-performance spiking neural network (SNN) library with SIMD optimization — CLI and SDK.', desc: 'Spike trains flow along four neuron rows of a raster plot.' },
  { slug: 'rudag', name: '@ruvector/rudag', kicker: 'DAG / WEBASSEMBLY', motif: 'dag', tagline: 'Fast DAG library in Rust/WASM — topological sort, critical path and task scheduling.', desc: 'DAG nodes light in topological order while the critical path flows.' },
  { slug: 'burst-scaling', name: '@ruvector/burst-scaling', kicker: 'AUTO-SCALING', motif: 'burst', tagline: 'Adaptive burst scaling for ruvector — handles 10–50× traffic spikes.', desc: 'A traffic spike grows while the capacity line steps up to follow it.' },
  { slug: 'raft', name: '@ruvector/raft', kicker: 'CONSENSUS', motif: 'heartbeat', tagline: 'Raft consensus for distributed systems — leader election, log replication and fault tolerance.', desc: 'A leader pulses heartbeats to four followers that acknowledge in turn.' },
  { slug: 'scipix', name: '@ruvector/scipix', kicker: 'SCIENTIFIC OCR', motif: 'ocr', tagline: 'OCR client for scientific documents — extract LaTeX and MathML from equations and papers.', desc: 'A scan line sweeps a page and LaTeX and MathML outputs light up.' },
  { slug: 'ospipe', name: '@ruvector/ospipe', kicker: 'PERSONAL AI MEMORY', motif: 'pipeline', opts: { cap: 'Screenpipe frames → embeddings → searchable memory' }, tagline: 'RuVector-enhanced personal AI memory SDK for Screenpipe pipes.', desc: 'Screen frames are embedded and stored as searchable memory points.' },
  { slug: 'ospipe-wasm', name: '@ruvector/ospipe-wasm', kicker: 'PERSONAL AI MEMORY / WASM', motif: 'pipeline', opts: { cap: 'in-browser embed · dedupe · search' }, tagline: 'OSpipe personal AI memory search in the browser — WebAssembly bindings.', desc: 'Frames are embedded in the browser and stored as searchable memory points.' },
  { slug: 'acorn-wasm', name: '@ruvector/acorn-wasm', kicker: 'FILTERED SEARCH / WEBASSEMBLY', motif: 'filter', tagline: 'ACORN predicate-agnostic filtered HNSW in WebAssembly — high-recall search with metadata filters.', desc: 'A search path hops only across graph nodes that match a metadata filter.' },
  { slug: 'agentic-integration', name: '@ruvector/agentic-integration', kicker: 'AGENT COORDINATION', motif: 'swarm', tagline: 'Distributed agent coordination for ruvector with claude-flow integration.', desc: 'A coordinator dispatches work to regional agent groups that light up in turn.' },
];

/** Greedy word wrap at ~64 chars; at most two lines (the table keeps taglines short). */
function wrap(s, max = 64) {
  const out = [''];
  for (const w of s.split(' ')) {
    const cur = out[out.length - 1];
    if (cur && (cur + ' ' + w).length > max) out.push(w); else out[out.length - 1] = cur ? cur + ' ' + w : w;
  }
  if (out.length > 2) throw new Error(`tagline too long for two lines: ${s}`);
  return out;
}

export function renderBanner(p) {
  const motif = renderMotif(p.motif, p.opts);
  const lines = wrap(p.tagline);
  const body = [
    '<rect width="1000" height="240" rx="18" fill="#0c1324"/>',
    '<path d="M0 42H1000M0 226H1000" stroke="#223047"/>',
    `<text x="30" y="29" fill="#5cf2c3" font-size="13" letter-spacing="3">RUVECTOR / NPM / ${esc(p.kicker)}</text>`,
    `<text x="30" y="92" fill="#f2f6ff" font-size="${p.name.length > 24 ? 25 : 29}" font-weight="bold">${esc(p.name)}</text>`,
    ...lines.map((l, i) => `<text x="30" y="${126 + i * 22}" fill="#abbad0" font-size="16">${esc(l)}</text>`),
    `<text x="30" y="200" fill="#8297b1" font-size="14">npm install ${esc(p.name)}</text>`,
    '<rect x="600" y="58" width="370" height="152" rx="14" fill="#111e32" stroke="#223047"/>',
    motif,
  ].join('');
  // Emit only the CSS a banner actually uses, in a fixed order (determinism).
  const used = Object.keys(CLASS_CSS).filter((c) => body.includes(`class="${c}"`));
  const kf = [...new Set(used.map((c) => KEYFRAMES[c]).filter(Boolean))];
  const css = 'text{font-family:Arial,sans-serif}' + used.map((c) => CLASS_CSS[c]).join('') + kf.join('') +
    `@media(prefers-reduced-motion:reduce){${used.map((c) => '.' + c).join(',') || 'text'}{animation:none}}`;
  const title = `${p.name} — ${p.tagline}`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="240" viewBox="0 0 1000 240" role="img" aria-labelledby="title desc">` +
    `<title id="title">${esc(title)}</title><desc id="desc">${esc('Animated RuVector package banner. ' + p.desc)}</desc>` +
    `<style>${css}</style>${body}</svg>\n`;
}

/** Minimal well-formedness check: balanced tags, quoted attributes, legal entities. */
export function checkXml(xml) {
  const stack = [];
  const re = /<(\/?)([A-Za-z][\w:.-]*)((?:\s+[\w:.-]+="[^"<]*")*)\s*(\/?)>|<!\[CDATA\[|<!--|<\?|<!|<|&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)/g;
  let m;
  while ((m = re.exec(xml))) {
    if (m[0] === '<' || m[0].startsWith('&') || m[0].startsWith('<!') || m[0].startsWith('<?')) {
      throw new Error(`bad markup at offset ${m.index}: ${xml.slice(m.index, m.index + 40)}`);
    }
    const [, close, tag, , self] = m;
    if (close) { if (stack.pop() !== tag) throw new Error(`mismatched </${tag}> at ${m.index}`); } else if (!self) stack.push(tag);
  }
  if (stack.length) throw new Error(`unclosed: ${stack.join(',')}`);
  if (!xml.startsWith('<svg ')) throw new Error('root is not <svg>');
}

export function readmeHeader(p) {
  return `<p align="center"><img src="${RAW_BASE}/${p.slug}.svg" alt="${esc(`${p.name}: ${p.tagline}`)}" width="100%"></p>`;
}

function main() {
  const check = process.argv.includes('--check');
  const seen = new Set();
  let stale = 0;
  fs.mkdirSync(OUT_DIR, { recursive: true });
  for (const p of [...PACKAGES].sort((a, b) => a.slug.localeCompare(b.slug))) {
    if (seen.has(p.slug) || !/^[a-z0-9-]+$/.test(p.slug)) throw new Error(`bad or duplicate slug: ${p.slug}`);
    seen.add(p.slug);
    const svg = renderBanner(p);
    checkXml(svg);
    const bytes = Buffer.byteLength(svg);
    if (bytes > MAX_BYTES) throw new Error(`${p.slug}.svg is ${bytes} bytes (> ${MAX_BYTES})`);
    const file = path.join(OUT_DIR, `${p.slug}.svg`);
    if (check) {
      const same = fs.existsSync(file) && fs.readFileSync(file, 'utf8') === svg;
      if (!same) stale++;
      console.log(`${same ? 'ok   ' : 'STALE'} ${p.slug.padEnd(22)} ${bytes} B`);
    } else {
      fs.writeFileSync(file, svg);
      console.log(`wrote ${p.slug.padEnd(22)} ${bytes} B  xml ok`);
    }
  }
  if (check && stale) { console.error(`${stale} banner(s) out of date — run without --check`); process.exit(1); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
