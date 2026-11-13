'use strict';
/**
 * `ruvector weights` CLI group and `weights_*` MCP tools, backed by the
 * @ruvector/weight-space dependency (versioned ruv-fann weight snapshots indexed by
 * task embedding).
 *
 * Nothing here loads @ruvector/weight-space (or the ONNX embedder) at require time:
 * both are required inside the operation that needs them, so `ruvector --help` and
 * the MCP server start without them. The package is a hard dependency, so it is
 * normally present; if an install is pruned or does not hoist it, every operation
 * fails with ERR_WEIGHT_SPACE_MISSING and the install hint below.
 *
 * Path handling: every operation takes a `resolvePath` function. The MCP server
 * passes its validateRvfPath (working directory + RUVECTOR_MCP_ROOTS); the CLI
 * passes confineToWorkingDir, which applies the same rule.
 */

const fs = require('fs');
const path = require('path');
const recordValidate = require('./record-validate');

const WEIGHT_SPACE_PKG = '@ruvector/weight-space';
const WEIGHT_SPACE_VERSION = '0.1.0';
const WEIGHT_SPACE_INSTALL_HINT = `npm i ${WEIGHT_SPACE_PKG}@${WEIGHT_SPACE_VERSION}`;
const DEFAULT_REGISTRY = path.join('.ruvector', 'weights.jsonl');

const MAX_MODEL_FILE_BYTES = 16 * 1024 * 1024;
const MAX_TASK_CHARS = 256;
const MAX_NOTE_CHARS = 4096;
const MAX_TAGS = 64;
const MAX_TAG_CHARS = 128;
const MAX_TEXT_CHARS = 8192;

function missingWeightSpaceError() {
  const err = new Error(
    `${WEIGHT_SPACE_PKG} is not installed, but it is a dependency of ruvector, so this install is incomplete. Install it with: ${WEIGHT_SPACE_INSTALL_HINT}`,
  );
  err.code = 'ERR_WEIGHT_SPACE_MISSING';
  err.install = WEIGHT_SPACE_INSTALL_HINT;
  return err;
}

/** require('@ruvector/weight-space'), or throw ERR_WEIGHT_SPACE_MISSING with the install hint. */
function loadWeightSpace() {
  try {
    require.resolve(WEIGHT_SPACE_PKG);
  } catch (e) {
    if (e && e.code === 'MODULE_NOT_FOUND') throw missingWeightSpaceError();
    throw e;
  }
  return require(WEIGHT_SPACE_PKG);
}

// ── path confinement (CLI) ─────────────────────────────────────────────────

function realOrResolved(p) {
  try { return fs.realpathSync(path.resolve(p)); } catch { return path.resolve(p); }
}

function isWithin(root, target) {
  const rel = path.relative(root, target);
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

/**
 * Resolve `p` (following symlinks of the path or its parent) and refuse anything
 * outside the working directory or RUVECTOR_MCP_ROOTS. Same rule as the MCP
 * server's validateRvfPath.
 */
function confineToWorkingDir(p) {
  if (typeof p !== 'string' || p.length === 0) throw new Error('Path must be a non-empty string');
  if (p.includes('\0')) throw new Error('Path contains null bytes');
  let resolved = path.resolve(p);
  try {
    resolved = fs.realpathSync(resolved);
  } catch {
    try { resolved = path.join(fs.realpathSync(path.dirname(resolved)), path.basename(resolved)); } catch { /* keep */ }
  }
  const roots = [realOrResolved(process.cwd())];
  for (const r of (process.env.RUVECTOR_MCP_ROOTS || '').split(path.delimiter)) {
    if (r.trim()) roots.push(realOrResolved(r.trim()));
  }
  if (!roots.some((root) => isWithin(root, resolved))) {
    throw new Error(`Access denied: path resolves to '${resolved}', which is outside the working directory. ` +
      'Set RUVECTOR_MCP_ROOTS to allow additional directories.');
  }
  return resolved;
}

// ── input parsing / validation ─────────────────────────────────────────────

/** Parse `--embedding` as a JSON array (`[0.1,0.2]`) or comma-separated numbers (`0.1,0.2`). */
function parseEmbeddingArg(raw) {
  if (Array.isArray(raw)) return Array.from(recordValidate.validateVector(raw, undefined, 'embedding'));
  const text = String(raw).trim();
  let values;
  if (text.startsWith('[')) {
    try { values = JSON.parse(text); } catch { throw new Error(`Invalid embedding "${raw}": not a JSON array`); }
  } else {
    values = text.split(',').map((s) => (s.trim() === '' ? NaN : Number(s)));
  }
  return Array.from(recordValidate.validateVector(values, undefined, 'embedding'));
}

function optionalString(value, max, name) {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== 'string') throw new Error(`${name} must be a string`);
  if (value.length > max) throw new Error(`${name} must be at most ${max} characters`);
  return value;
}

function parseTags(value) {
  if (value === undefined || value === null) return undefined;
  const list = Array.isArray(value) ? value : String(value).split(',').map((t) => t.trim()).filter(Boolean);
  if (list.length > MAX_TAGS) throw new Error(`at most ${MAX_TAGS} tags`);
  for (const t of list) {
    if (typeof t !== 'string' || t.length === 0 || t.length > MAX_TAG_CHARS) {
      throw new Error(`each tag must be a non-empty string of at most ${MAX_TAG_CHARS} characters`);
    }
  }
  return list;
}

function parseId(value, name = 'id') {
  if (typeof value !== 'string' || !/^[A-Za-z0-9_-]{1,128}$/.test(value)) {
    throw new Error(`${name} must be a snapshot id such as ws-0123456789abcdef`);
  }
  return value;
}

function readModelFile(filePath) {
  const st = fs.statSync(filePath);
  if (!st.isFile()) throw new Error(`model file '${filePath}' is not a regular file`);
  if (st.size > MAX_MODEL_FILE_BYTES) throw new Error(`model file is larger than ${MAX_MODEL_FILE_BYTES} bytes`);
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (e) {
    throw new Error(`model file '${filePath}' is not valid JSON: ${e.message}`);
  }
}

/** Embed text with ruvector's ONNX embedder (lazy; may download the model on first use). */
async function embedText(text, kind) {
  const value = optionalString(text, MAX_TEXT_CHARS, 'text');
  if (!value || !value.trim()) throw new Error('text must be a non-empty string');
  const onnx = require('../dist/core/onnx-embedder.js');
  const ready = await onnx.initOnnxEmbedder();
  if (ready === false) throw new Error('ONNX embedder is unavailable; pass an explicit embedding instead');
  const result = kind === 'query' ? await onnx.embedQuery(value) : await onnx.embed(value);
  return Array.from(result.embedding);
}

/** Exactly one of `embedding` / `text` is required. */
async function resolveEmbedding({ embedding, text }, kind) {
  const hasEmbedding = embedding !== undefined && embedding !== null && embedding !== '';
  const hasText = text !== undefined && text !== null && text !== '';
  if (hasEmbedding === hasText) throw new Error('exactly one of an embedding or a text description is required');
  return hasEmbedding ? parseEmbeddingArg(embedding) : embedText(text, kind);
}

// ── registry operations ────────────────────────────────────────────────────

function summarize(rec) {
  return {
    id: rec.id,
    version: rec.version,
    parent: rec.parent,
    task: rec.task,
    hash: rec.hash,
    topology: rec.topology,
    createdAt: rec.createdAt,
    op: rec.op,
    tags: rec.tags,
    note: rec.note,
    outcomes: rec.outcomes,
  };
}

/** Open the registry at a confined path, run `fn(reg, ws)`, and always close it. */
async function withRegistry(registry, resolvePath, fn) {
  const ws = loadWeightSpace();
  const registryPath = resolvePath(registry || DEFAULT_REGISTRY);
  // The @ruvector/core index lives next to the log; confine it too (symlinks).
  resolvePath(registryPath + '.index.db');
  const reg = await ws.WeightRegistry.open({ path: registryPath });
  try {
    return await fn(reg, ws, registryPath);
  } finally {
    await reg.close();
  }
}

async function saveSnapshot(opts, resolvePath) {
  const ws = loadWeightSpace();
  let model = opts.model;
  if (opts.modelPath !== undefined) {
    if (model !== undefined) throw new Error('pass either a model object or a model file, not both');
    model = readModelFile(resolvePath(opts.modelPath));
  }
  if (model === undefined || model === null) throw new Error('a model (ruv-fann dense-layers JSON) is required');
  const clean = ws.validateModel(model);
  const embedding = await resolveEmbedding(opts, 'passage');
  const meta = {};
  const task = optionalString(opts.task, MAX_TASK_CHARS, 'task');
  if (task !== undefined) meta.task = task;
  const tags = parseTags(opts.tags);
  if (tags !== undefined) meta.tags = tags;
  const note = optionalString(opts.note, MAX_NOTE_CHARS, 'note');
  if (note !== undefined) meta.note = note;
  return withRegistry(opts.registry, resolvePath, async (reg, _ws, registryPath) => {
    const rec = await reg.save(clean, embedding, meta);
    return { registry: registryPath, snapshot: summarize(rec) };
  });
}

async function nearestSnapshots(opts, resolvePath) {
  loadWeightSpace();
  const k = recordValidate.validateK(opts.k, 5);
  const embedding = await resolveEmbedding(opts, 'query');
  return withRegistry(opts.registry, resolvePath, async (reg, _ws, registryPath) => {
    const hits = await reg.nearest(embedding, k);
    return {
      registry: registryPath,
      results: hits.map((h) => ({ id: h.id, similarity: h.similarity, snapshot: summarize(h.record) })),
    };
  });
}

async function diffSnapshots(opts, resolvePath) {
  const a = parseId(opts.idA, 'idA');
  const b = parseId(opts.idB, 'idB');
  return withRegistry(opts.registry, resolvePath, async (reg) => ({ a, b, diff: reg.diff(a, b) }));
}

async function lineageOf(opts, resolvePath) {
  const id = parseId(opts.id);
  return withRegistry(opts.registry, resolvePath, async (reg) => ({ id, lineage: reg.lineage(id).map(summarize) }));
}

async function getSnapshot(opts, resolvePath) {
  const id = parseId(opts.id);
  return withRegistry(opts.registry, resolvePath, async (reg, ws) => {
    const rec = reg.get(id);
    if (!rec) throw new ws.WeightSpaceError(`unknown snapshot '${id}'`, 'ENOTFOUND');
    return { snapshot: { ...summarize(rec), embedding: rec.embedding, model: rec.model } };
  });
}

/** Error payload shared by the CLI (--json) and MCP. */
function errorPayload(e) {
  const out = { success: false, error: e && e.message ? e.message : String(e) };
  if (e && e.code) out.code = e.code;
  if (e && e.install) out.install = e.install;
  return out;
}

// ── MCP ────────────────────────────────────────────────────────────────────

const REGISTRY_PROP = {
  type: 'string',
  description: `Registry event log (JSON lines; confined to the working directory). Default ${DEFAULT_REGISTRY}`,
};
const EMBEDDING_PROP = { type: 'array', items: { type: 'number' }, description: 'Task embedding (all snapshots in a registry share one dimension)' };
const NEEDS = `Runs on ${WEIGHT_SPACE_PKG}; if it is missing, install it with: ${WEIGHT_SPACE_INSTALL_HINT}.`;

const BASE_MCP_TOOLS = [
  {
    name: 'weights_save',
    description: `Save a ruv-fann model snapshot to a weight registry, keyed by a task embedding. ${NEEDS}`,
    inputSchema: {
      type: 'object',
      properties: {
        registry: REGISTRY_PROP,
        model: { type: 'object', description: 'ruv-fann dense-layers model ({schema, layers})' },
        model_path: { type: 'string', description: 'Model JSON file (alternative to model; confined to the working directory)' },
        embedding: EMBEDDING_PROP,
        task: { type: 'string', description: 'Task name' },
        tags: { type: 'array', items: { type: 'string' } },
        note: { type: 'string' },
      },
      required: ['embedding'],
    },
  },
  {
    name: 'weights_nearest',
    description: `Find the k registry snapshots whose task embeddings are most similar (cosine) to an embedding. ${NEEDS}`,
    inputSchema: {
      type: 'object',
      properties: { registry: REGISTRY_PROP, embedding: EMBEDDING_PROP, k: { type: 'number', default: 5 } },
      required: ['embedding'],
    },
  },
  {
    name: 'weights_diff',
    description: `Parameter-level difference between two snapshots (L2, cosine, per-layer changes). ${NEEDS}`,
    inputSchema: {
      type: 'object',
      properties: { registry: REGISTRY_PROP, id_a: { type: 'string' }, id_b: { type: 'string' } },
      required: ['id_a', 'id_b'],
    },
  },
  {
    name: 'weights_lineage',
    description: `Ancestry of a snapshot from its root down to the snapshot itself. ${NEEDS}`,
    inputSchema: {
      type: 'object',
      properties: { registry: REGISTRY_PROP, id: { type: 'string' } },
      required: ['id'],
    },
  },
];

// Edit and fine-tune operations (set-layer, perturb, learn) live in weights-edit.js;
// the shared registry plumbing is injected so the two files do not import each other.
const edit = require('./weights-edit').create({
  withRegistry, parseId, summarize, optionalString, parseTags, REGISTRY_PROP, NEEDS, MAX_NOTE_CHARS,
});
const MCP_TOOLS = [...BASE_MCP_TOOLS, ...edit.MCP_TOOLS];
const MCP_TOOL_NAMES = new Set(MCP_TOOLS.map((t) => t.name));

/** Run a weights_* MCP tool. `resolvePath` is the server's path validator. */
async function runMcpTool(name, args, resolvePath) {
  const a = args && typeof args === 'object' ? args : {};
  try {
    let payload;
    if (name === 'weights_save') {
      if (!Array.isArray(a.embedding)) throw new Error('embedding must be an array of numbers');
      payload = await saveSnapshot({
        registry: a.registry, model: a.model, modelPath: a.model_path, embedding: a.embedding,
        task: a.task, tags: a.tags, note: a.note,
      }, resolvePath);
    } else if (name === 'weights_nearest') {
      if (!Array.isArray(a.embedding)) throw new Error('embedding must be an array of numbers');
      payload = await nearestSnapshots({ registry: a.registry, embedding: a.embedding, k: a.k }, resolvePath);
    } else if (name === 'weights_diff') {
      payload = await diffSnapshots({ registry: a.registry, idA: a.id_a, idB: a.id_b }, resolvePath);
    } else if (name === 'weights_lineage') {
      payload = await lineageOf({ registry: a.registry, id: a.id }, resolvePath);
    } else if (Object.prototype.hasOwnProperty.call(edit.RUNNERS, name)) {
      payload = await edit.RUNNERS[name](a, resolvePath);
    } else {
      throw new Error(`unknown weights tool '${name}'`);
    }
    return { content: [{ type: 'text', text: JSON.stringify({ success: true, ...payload }, null, 2) }] };
  } catch (e) {
    return { content: [{ type: 'text', text: JSON.stringify(errorPayload(e), null, 2) }], isError: true };
  }
}

// ── CLI ────────────────────────────────────────────────────────────────────

function topologyLabel(t) {
  return `${t.sizes.join('->')} [${t.activations.join(',')}] ${t.parameters} params`;
}

function printHuman(chalk, kind, result) {
  if (kind === 'save') {
    const s = result.snapshot;
    console.log(chalk.green(`Saved snapshot ${s.id}`));
    console.log(`  version:  ${s.version}`);
    console.log(`  hash:     ${s.hash}`);
    console.log(`  topology: ${topologyLabel(s.topology)}`);
    console.log(chalk.dim(`  registry: ${result.registry}`));
  } else if (kind === 'nearest') {
    if (result.results.length === 0) console.log(chalk.yellow('No snapshots in the registry.'));
    result.results.forEach((r, i) => {
      const s = r.snapshot;
      console.log(`  ${i + 1}. ${s.id}  similarity=${r.similarity.toFixed(6)}  v${s.version}  task=${s.task ?? '-'}  ${topologyLabel(s.topology)}`);
    });
  } else if (kind === 'diff') {
    const d = result.diff;
    if (!d.compatible) {
      console.log(chalk.yellow(`${result.a} and ${result.b} are not comparable: ${d.reason}`));
    } else {
      console.log(`  ${result.a} -> ${result.b}: l2=${d.l2.toFixed(6)} cosine=${d.cosine.toFixed(6)} changed=${d.changed}/${d.parameters}`);
      for (const l of d.layers) {
        console.log(chalk.dim(`    layer ${l.index}: l2=${l.l2.toFixed(6)} maxAbs=${l.maxAbs.toFixed(6)} changed=${l.changed}/${l.parameters}`));
      }
    }
  } else if (kind === 'lineage') {
    result.lineage.forEach((s) => console.log(`  v${s.version}  ${s.id}  op=${s.op}  task=${s.task ?? '-'}  hash=${s.hash.slice(0, 12)}`));
  } else if (kind === 'get') {
    const s = result.snapshot;
    console.log(`  ${s.id}  v${s.version}  parent=${s.parent ?? '-'}  task=${s.task ?? '-'}`);
    console.log(`  hash: ${s.hash}`);
    console.log(`  topology: ${topologyLabel(s.topology)}`);
    console.log(`  tags: ${s.tags.join(',') || '-'}  note: ${s.note ?? '-'}  outcomes: ${s.outcomes.length}`);
  } else if (Object.prototype.hasOwnProperty.call(edit.PRINTERS, kind)) {
    edit.PRINTERS[kind](chalk, result);
  }
}

/** Register `ruvector weights save|nearest|diff|lineage|get|set-layer|perturb|learn` on a commander program. */
function registerWeightsCommands(program, chalk) {
  const cmd = program
    .command('weights')
    .description(`Versioned ruv-fann weight registry indexed by task embedding (${WEIGHT_SPACE_PKG})`);

  // Exit explicitly (the ONNX embedder can keep the event loop alive), but only
  // once stdout/stderr have drained: pipe writes are async on some platforms.
  const exitAfterFlush = (code) => {
    let pending = 2;
    const done = () => { if (--pending === 0) process.exit(code); };
    process.stdout.write('', done);
    process.stderr.write('', done);
  };

  const run = (kind, build) => async (...argv) => {
    const opts = argv[argv.length - 2] && typeof argv[argv.length - 2] === 'object' ? argv[argv.length - 2] : {};
    try {
      const result = await build(...argv);
      if (opts.json) console.log(JSON.stringify({ success: true, ...result }, null, 2));
      else printHuman(chalk, kind, result);
      exitAfterFlush(0);
    } catch (e) {
      const payload = errorPayload(e);
      if (opts.json) {
        console.log(JSON.stringify(payload, null, 2));
      } else {
        console.error(chalk.red(payload.error));
        if (payload.install) console.error(chalk.yellow(`  Install it:  ${payload.install}`));
      }
      exitAfterFlush(1);
    }
  };

  const common = (c) => c
    .option('--registry <path>', 'Registry event log (JSON lines)', DEFAULT_REGISTRY)
    .option('--json', 'Output as JSON');

  common(cmd.command('save')
    .description('Save a model snapshot keyed by a task embedding')
    .requiredOption('--model <file>', 'ruv-fann dense-layers model JSON file')
    .option('--embedding <values>', 'Task embedding: JSON array or comma-separated numbers')
    .option('--text <description>', 'Task description, embedded with the ONNX embedder')
    .option('--task <name>', 'Task name')
    .option('--tags <a,b>', 'Comma-separated tags')
    .option('--note <text>', 'Free-form note'))
    .action(run('save', (opts) => saveSnapshot({ ...opts, modelPath: opts.model, model: undefined }, confineToWorkingDir)));

  common(cmd.command('nearest')
    .description('Snapshots whose task embeddings are most similar to a query')
    .option('--embedding <values>', 'Query embedding: JSON array or comma-separated numbers')
    .option('--text <description>', 'Query task description, embedded with the ONNX embedder')
    .option('-k, --k <n>', 'Number of results', '5'))
    .action(run('nearest', (opts) => nearestSnapshots(opts, confineToWorkingDir)));

  common(cmd.command('diff <idA> <idB>')
    .description('Parameter-level difference between two snapshots'))
    .action(run('diff', (idA, idB, opts) => diffSnapshots({ ...opts, idA, idB }, confineToWorkingDir)));

  common(cmd.command('lineage <id>')
    .description('Ancestry of a snapshot, root first'))
    .action(run('lineage', (id, opts) => lineageOf({ ...opts, id }, confineToWorkingDir)));

  common(cmd.command('get <id>')
    .description('One snapshot, including its model and embedding'))
    .action(run('get', (id, opts) => getSnapshot({ ...opts, id }, confineToWorkingDir)));

  edit.registerCommands(cmd, run, common, confineToWorkingDir);

  return cmd;
}

module.exports = {
  WEIGHT_SPACE_PKG,
  WEIGHT_SPACE_VERSION,
  WEIGHT_SPACE_INSTALL_HINT,
  DEFAULT_REGISTRY,
  MCP_TOOLS,
  MCP_TOOL_NAMES,
  loadWeightSpace,
  confineToWorkingDir,
  parseEmbeddingArg,
  saveSnapshot,
  nearestSnapshots,
  diffSnapshots,
  lineageOf,
  getSnapshot,
  errorPayload,
  runMcpTool,
  registerWeightsCommands,
};
