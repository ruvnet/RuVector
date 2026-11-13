'use strict';
/**
 * Edit and fine-tune operations for the weight registry: `weights set-layer`,
 * `weights perturb`, `weights learn`, and the matching `weights_set_layer`,
 * `weights_perturb`, `weights_learn` MCP tools. Each one writes a CHILD snapshot of an
 * existing snapshot (the registry is append-only and snapshots are immutable), so the
 * lineage records exactly what was changed.
 *
 * This file holds only argument validation and the call into @ruvector/weight-space;
 * the shared registry plumbing (path confinement, open/close, summaries) is injected
 * from weights.js so the two files do not import each other.
 */

const fs = require('fs');

const MAX_JSON_FILE_BYTES = 16 * 1024 * 1024;
const MAX_TRAJECTORIES = 10000;
const MAX_VECTOR_LEN = 4096;
const MAX_PATCH_VALUES = 1000000;
const MAX_LAYER_INDEX = 63;
const MAX_LAYER_LIST = 64;
const MAX_EPOCHS = 10000;
const MAX_SEED = 4294967295;
const PATCH_KEYS = new Set(['weights', 'bias', 'activation', 'steepness']);

/** A finite number within bounds, from a number or a CLI string. `dflt` when absent. */
function num(value, name, { min, max, int = false, dflt } = {}) {
  if (value === undefined || value === null || value === '') return dflt;
  const n = typeof value === 'number' ? value : Number(String(value).trim());
  if (!Number.isFinite(n)) throw new Error(`${name} must be a finite number`);
  if (int && !Number.isInteger(n)) throw new Error(`${name} must be an integer`);
  if (min !== undefined && n < min) throw new Error(`${name} must be at least ${min}`);
  if (max !== undefined && n > max) throw new Error(`${name} must be at most ${max}`);
  return n;
}

function flag(value, name) {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== 'boolean') throw new Error(`${name} must be true or false`);
  return value;
}

/** A layer index from a number or string. */
function layerIndex(value, name = 'layer') {
  const n = num(value, name, { min: 0, max: MAX_LAYER_INDEX, int: true });
  if (n === undefined) throw new Error(`${name} is required`);
  return n;
}

/** `--layers 0,1` or `[0,1]`: a short list of layer indexes. */
function layerList(value) {
  if (value === undefined || value === null || value === '') return undefined;
  const list = Array.isArray(value) ? value : String(value).split(',').map((s) => s.trim()).filter(Boolean);
  if (list.length === 0 || list.length > MAX_LAYER_LIST) throw new Error(`layers must list 1 to ${MAX_LAYER_LIST} layer indexes`);
  return list.map((v, i) => layerIndex(v, `layers[${i}]`));
}

function finiteVector(value, name) {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_VECTOR_LEN) {
    throw new Error(`${name} must be an array of 1 to ${MAX_VECTOR_LEN} numbers`);
  }
  for (const v of value) if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`${name} must contain only finite numbers`);
  return value;
}

function readJsonFile(filePath, label) {
  const st = fs.statSync(filePath);
  if (!st.isFile()) throw new Error(`${label} file '${filePath}' is not a regular file`);
  if (st.size > MAX_JSON_FILE_BYTES) throw new Error(`${label} file is larger than ${MAX_JSON_FILE_BYTES} bytes`);
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (e) {
    throw new Error(`${label} file '${filePath}' is not valid JSON: ${e.message}`);
  }
}

/** Exactly one of an inline value or a confined file path. */
function inlineOrFile(inline, filePath, label, resolvePath) {
  const hasInline = inline !== undefined && inline !== null;
  const hasFile = filePath !== undefined && filePath !== null && filePath !== '';
  if (hasInline === hasFile) throw new Error(`pass exactly one of ${label} or a ${label} file`);
  return hasInline ? inline : readJsonFile(resolvePath(filePath), label);
}

function validatePatch(patch) {
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw new Error('patch must be an object');
  const keys = Object.keys(patch);
  if (keys.length === 0) throw new Error('patch must change at least one of weights, bias, activation, steepness');
  for (const k of keys) if (!PATCH_KEYS.has(k)) throw new Error(`patch has an unsupported field '${k}' (allowed: ${[...PATCH_KEYS].join(', ')})`);
  const out = {};
  if (patch.weights !== undefined) {
    if (!Array.isArray(patch.weights) || patch.weights.length > MAX_PATCH_VALUES) throw new Error(`patch.weights must be an array of at most ${MAX_PATCH_VALUES} numbers`);
    for (const v of patch.weights) if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error('patch.weights must contain only finite numbers');
    out.weights = patch.weights;
  }
  if (patch.bias !== undefined) {
    if (patch.bias !== null) {
      if (!Array.isArray(patch.bias) || patch.bias.length > MAX_VECTOR_LEN) throw new Error(`patch.bias must be null or an array of at most ${MAX_VECTOR_LEN} numbers`);
      for (const v of patch.bias) if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error('patch.bias must contain only finite numbers');
    }
    out.bias = patch.bias;
  }
  if (patch.activation !== undefined) {
    if (typeof patch.activation !== 'string' || patch.activation.length > 64) throw new Error('patch.activation must be an activation name');
    out.activation = patch.activation;
  }
  if (patch.steepness !== undefined) out.steepness = num(patch.steepness, 'patch.steepness', { min: 0 });
  return out;
}

function validateTrajectories(rows) {
  if (!Array.isArray(rows) || rows.length === 0 || rows.length > MAX_TRAJECTORIES) {
    throw new Error(`trajectories must be an array of 1 to ${MAX_TRAJECTORIES} rows`);
  }
  return rows.map((r, i) => {
    if (!r || typeof r !== 'object' || Array.isArray(r)) throw new Error(`trajectories[${i}] must be an object`);
    const row = {
      input: finiteVector(r.input, `trajectories[${i}].input`),
      output: finiteVector(r.output, `trajectories[${i}].output`),
      score: num(r.score, `trajectories[${i}].score`),
    };
    if (row.score === undefined) throw new Error(`trajectories[${i}].score is required`);
    if (r.correction !== undefined && r.correction !== null) row.correction = finiteVector(r.correction, `trajectories[${i}].correction`);
    return row;
  });
}

function create(h) {
  const { withRegistry, parseId, summarize, optionalString, parseTags, REGISTRY_PROP, NEEDS, MAX_NOTE_CHARS } = h;

  /** note/tags carried onto the child snapshot. */
  function childMeta(opts) {
    const meta = {};
    const note = optionalString(opts.note, MAX_NOTE_CHARS, 'note');
    if (note !== undefined) meta.note = note;
    const tags = parseTags(opts.tags);
    if (tags !== undefined) meta.tags = tags;
    return meta;
  }

  async function setLayerOp(opts, resolvePath) {
    const id = parseId(opts.id);
    const index = layerIndex(opts.layer);
    const patch = validatePatch(inlineOrFile(opts.patch, opts.patchPath, 'patch', resolvePath));
    const meta = childMeta(opts);
    return withRegistry(opts.registry, resolvePath, async (reg, _ws, registryPath) => ({
      registry: registryPath, parent: id, snapshot: summarize(await reg.setLayer(id, index, patch, meta)),
    }));
  }

  async function perturbOp(opts, resolvePath) {
    const id = parseId(opts.id);
    const options = {};
    const scale = num(opts.scale, 'scale', { min: 0, max: 10 });
    if (scale !== undefined) options.scale = scale;
    const seed = num(opts.seed, 'seed', { min: 0, max: MAX_SEED, int: true });
    if (seed !== undefined) options.seed = seed;
    const layers = layerList(opts.layers);
    if (layers !== undefined) options.layers = layers;
    const bias = flag(opts.bias, 'bias');
    if (bias !== undefined) options.bias = bias;
    const relative = flag(opts.relative, 'relative');
    if (relative !== undefined) options.relative = relative;
    const meta = childMeta(opts);
    return withRegistry(opts.registry, resolvePath, async (reg, _ws, registryPath) => ({
      registry: registryPath, parent: id, snapshot: summarize(await reg.perturb(id, options, meta)),
    }));
  }

  async function learnOp(opts, resolvePath) {
    const id = parseId(opts.id);
    const trajectories = validateTrajectories(inlineOrFile(opts.trajectories, opts.trajectoriesPath, 'trajectories', resolvePath));
    const options = {};
    const set = (key, value) => { if (value !== undefined) options[key] = value; };
    set('epochs', num(opts.epochs, 'epochs', { min: 1, max: MAX_EPOCHS, int: true }));
    set('learningRate', num(opts.learningRate, 'learning rate', { min: 1e-9, max: 10 }));
    set('seed', num(opts.seed, 'seed', { min: 0, max: MAX_SEED, int: true }));
    set('successThreshold', num(opts.successThreshold, 'success threshold', { min: 0, max: 1 }));
    set('holdoutFraction', num(opts.holdoutFraction, 'holdout fraction', { min: 0.01, max: 0.9 }));
    set('minImprovement', num(opts.minImprovement, 'minimum improvement', { min: 0, max: 1 }));
    if (opts.optimizer !== undefined && opts.optimizer !== null) {
      if (opts.optimizer !== 'sgd' && opts.optimizer !== 'adamw') throw new Error("optimizer must be 'sgd' or 'adamw'");
      options.optimizer = opts.optimizer;
    }
    const note = optionalString(opts.note, MAX_NOTE_CHARS, 'note');
    if (note !== undefined) options.note = note;
    return withRegistry(opts.registry, resolvePath, async (reg, _ws, registryPath) => {
      const r = await reg.learnFromOutcomes(id, trajectories, options);
      // `accepted: false` is a normal result (the held-out check did not improve), not an error.
      return {
        registry: registryPath, parent: r.parent, accepted: r.accepted, reason: r.reason,
        child: r.child ? summarize(r.child) : null, rows: r.rows, holdoutMse: r.holdoutMse, trainLoss: r.trainLoss,
      };
    });
  }

  const idProp = { type: 'string', description: 'Parent snapshot id (the registry is append-only: the result is a new child snapshot)' };
  const notes = { note: { type: 'string' }, tags: { type: 'array', items: { type: 'string' } } };
  const MCP_TOOLS = [
    {
      name: 'weights_set_layer',
      description: `Write a child snapshot with one layer patched (weights, bias, activation or steepness; the layer shape cannot change). ${NEEDS}`,
      inputSchema: {
        type: 'object',
        properties: {
          registry: REGISTRY_PROP, id: idProp, layer: { type: 'integer', minimum: 0, maximum: MAX_LAYER_INDEX },
          patch: { type: 'object', description: 'Fields to change: weights, bias, activation, steepness' },
          patch_path: { type: 'string', description: 'JSON file holding the patch (alternative to patch; confined to the working directory)' },
          ...notes,
        },
        required: ['id', 'layer'],
      },
    },
    {
      name: 'weights_perturb',
      description: `Write a child snapshot with seeded Gaussian noise added to the weights (same seed, same noise). ${NEEDS}`,
      inputSchema: {
        type: 'object',
        properties: {
          registry: REGISTRY_PROP, id: idProp,
          scale: { type: 'number', minimum: 0, maximum: 10, description: 'Noise standard deviation (default 0.01)' },
          seed: { type: 'integer', minimum: 0, maximum: MAX_SEED, description: 'Default 1' },
          layers: { type: 'array', items: { type: 'integer', minimum: 0, maximum: MAX_LAYER_INDEX }, description: 'Layers to perturb (default all)' },
          bias: { type: 'boolean', description: 'Also perturb biases (default true)' },
          relative: { type: 'boolean', description: "Scale noise by each layer's weight RMS (default false)" },
          ...notes,
        },
        required: ['id'],
      },
    },
    {
      name: 'weights_learn',
      description: `Fine-tune a snapshot on outcome trajectories with ruv-fann. A child snapshot is written ONLY if a held-out check improves; otherwise nothing is written and the result says why. ${NEEDS}`,
      inputSchema: {
        type: 'object',
        properties: {
          registry: REGISTRY_PROP, id: idProp,
          trajectories: {
            type: 'array', maxItems: MAX_TRAJECTORIES,
            description: 'Rows of {input, output, score, correction?}. score >= successThreshold trains toward output; a failure trains toward its correction (failures without one are skipped)',
            items: { type: 'object' },
          },
          trajectories_path: { type: 'string', description: 'JSON file holding the rows (alternative to trajectories; confined to the working directory)' },
          epochs: { type: 'integer', minimum: 1, maximum: MAX_EPOCHS },
          learning_rate: { type: 'number', minimum: 1e-9, maximum: 10 },
          optimizer: { type: 'string', enum: ['sgd', 'adamw'] },
          seed: { type: 'integer', minimum: 0, maximum: MAX_SEED },
          success_threshold: { type: 'number', minimum: 0, maximum: 1 },
          holdout_fraction: { type: 'number', minimum: 0.01, maximum: 0.9 },
          min_improvement: { type: 'number', minimum: 0, maximum: 1 },
          note: { type: 'string' },
        },
        required: ['id'],
      },
    },
  ];

  /** MCP argument (snake_case) -> operation, keyed by tool name. */
  const RUNNERS = {
    weights_set_layer: (a, rp) => setLayerOp({ registry: a.registry, id: a.id, layer: a.layer, patch: a.patch, patchPath: a.patch_path, note: a.note, tags: a.tags }, rp),
    weights_perturb: (a, rp) => perturbOp({ registry: a.registry, id: a.id, scale: a.scale, seed: a.seed, layers: a.layers, bias: a.bias, relative: a.relative, note: a.note, tags: a.tags }, rp),
    weights_learn: (a, rp) => learnOp({
      registry: a.registry, id: a.id, trajectories: a.trajectories, trajectoriesPath: a.trajectories_path, epochs: a.epochs,
      learningRate: a.learning_rate, optimizer: a.optimizer, seed: a.seed, successThreshold: a.success_threshold,
      holdoutFraction: a.holdout_fraction, minImprovement: a.min_improvement, note: a.note,
    }, rp),
  };

  const PRINTERS = {
    'set-layer': (chalk, r) => printChild(chalk, r),
    perturb: (chalk, r) => printChild(chalk, r),
    learn: (chalk, r) => {
      if (!r.accepted) {
        console.log(chalk.yellow(`Not accepted: ${r.reason}`));
      } else {
        console.log(chalk.green(`Accepted: child snapshot ${r.child.id} (v${r.child.version}) of ${r.parent}`));
      }
      console.log(`  held-out MSE: ${r.holdoutMse.before.toFixed(6)} -> ${r.holdoutMse.after.toFixed(6)}`);
      console.log(`  rows: train=${r.rows.train} holdout=${r.rows.holdout} successes=${r.rows.successes} corrections=${r.rows.corrections} skipped=${r.rows.skipped}`);
      console.log(chalk.dim(`  registry: ${r.registry}`));
    },
  };

  function printChild(chalk, r) {
    const s = r.snapshot;
    console.log(chalk.green(`Wrote snapshot ${s.id} (v${s.version}, op=${s.op}) as a child of ${r.parent}`));
    console.log(`  hash: ${s.hash}`);
    console.log(chalk.dim(`  registry: ${r.registry}`));
  }

  /** Register the three subcommands. `run` and `common` come from weights.js's CLI wiring. */
  function registerCommands(cmd, run, common, confine) {
    common(cmd.command('set-layer <id> <layer>')
      .description('Write a child snapshot with one layer patched')
      .requiredOption('--patch <file>', 'JSON file with the fields to change (weights, bias, activation, steepness)')
      .option('--note <text>', 'Free-form note')
      .option('--tags <a,b>', 'Comma-separated tags'))
      .action(run('set-layer', (id, layer, opts) => setLayerOp({ ...opts, id, layer, patchPath: opts.patch, patch: undefined }, confine)));

    common(cmd.command('perturb <id>')
      .description('Write a child snapshot with seeded Gaussian noise added to the weights')
      .option('--scale <n>', 'Noise standard deviation (default 0.01)')
      .option('--seed <n>', 'Noise seed (default 1)')
      .option('--layers <a,b>', 'Comma-separated layer indexes (default all)')
      .option('--no-bias', 'Leave biases unchanged')
      .option('--relative', "Scale noise by each layer's weight RMS")
      .option('--note <text>', 'Free-form note')
      .option('--tags <a,b>', 'Comma-separated tags'))
      .action(run('perturb', (id, opts) => perturbOp({ ...opts, id, bias: opts.bias === false ? false : undefined, relative: opts.relative === true ? true : undefined }, confine)));

    common(cmd.command('learn <id>')
      .description('Fine-tune a snapshot on outcome trajectories; a child is written only if a held-out check improves')
      .requiredOption('--trajectories <file>', 'JSON file: rows of {input, output, score, correction?}')
      .option('--epochs <n>', 'Training epochs')
      .option('--learning-rate <n>', 'Learning rate')
      .option('--optimizer <name>', 'sgd or adamw')
      .option('--seed <n>', 'Seed for the holdout split')
      .option('--success-threshold <n>', 'Score at or above this counts as a success (default 0.5)')
      .option('--holdout-fraction <n>', 'Fraction of rows held out (default 0.25)')
      .option('--min-improvement <n>', 'Required relative MSE reduction to accept (default 0.01)')
      .option('--note <text>', 'Free-form note'))
      .action(run('learn', (id, opts) => learnOp({ ...opts, id, trajectoriesPath: opts.trajectories, trajectories: undefined }, confine)));
  }

  return { MCP_TOOLS, RUNNERS, PRINTERS, registerCommands, setLayerOp, perturbOp, learnOp };
}

module.exports = { create };
