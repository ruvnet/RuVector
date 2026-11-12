/**
 * ruv-fann dense-model types, validation against ruv-fann LIMITS, canonical JSON and hashing.
 *
 * The types mirror `ruv-fann` 0.1.0 (`src/index.d.ts`) structurally, so this package can be
 * compiled to CommonJS even though ruv-fann itself is ESM-only.
 */
import { createHash } from 'node:crypto';

export type DenseActivation = 'Linear' | 'Threshold' | 'ThresholdSymmetric' | 'Sigmoid' |
  'SigmoidSymmetric' | 'Tanh' | 'Gaussian' | 'GaussianSymmetric' | 'Elliot' |
  'ElliotSymmetric' | 'LinearPiece' | 'LinearPieceSymmetric' | 'ReLU' | 'ReLULeaky' |
  'Sin' | 'Cos' | 'SinSymmetric' | 'CosSymmetric' | 'Gelu' | 'Swish' | 'LeakyRelu';

export const ACTIVATIONS: readonly DenseActivation[] = Object.freeze([
  'Linear', 'Threshold', 'ThresholdSymmetric', 'Sigmoid', 'SigmoidSymmetric', 'Tanh', 'Gaussian',
  'GaussianSymmetric', 'Elliot', 'ElliotSymmetric', 'LinearPiece', 'LinearPieceSymmetric', 'ReLU',
  'ReLULeaky', 'Sin', 'Cos', 'SinSymmetric', 'CosSymmetric', 'Gelu', 'Swish', 'LeakyRelu',
] as DenseActivation[]);

export interface DenseLayer {
  readonly inputs: number;
  readonly outputs: number;
  /** Row-major [outputs, inputs]: weight from input `i` to output `o` is `weights[o * inputs + i]`. */
  readonly weights: readonly number[];
  readonly bias?: readonly number[] | null;
  readonly activation: DenseActivation;
  readonly steepness: number;
}

export interface DenseModel {
  readonly schema: 'ruv-fann.dense-layers/1';
  readonly layers: readonly DenseLayer[];
}

/**
 * ruv-fann 0.1.0 `LIMITS`, mirrored so validation can run synchronously before the WASM runtime
 * is loaded. `loadFann()` checks that the installed ruv-fann reports the same values.
 */
export const LIMITS = Object.freeze({
  maxRows: 4096,
  maxDimension: 65536,
  maxLayers: 64,
  maxParameters: 1_000_000,
  maxModelBytes: 4 * 1024 * 1024,
  maxBytes: 64 * 1024 * 1024,
  maxEpochs: 1000,
  maxTrainingWork: 100_000_000,
});
export type Limits = typeof LIMITS;

/** Raised for any model, shape or argument that ruv-fann would reject. */
export class WeightSpaceError extends Error {
  constructor(message: string, readonly code: string) {
    super(message);
    this.name = 'WeightSpaceError';
  }
}

function fail(message: string, code = 'EINVALID'): never {
  throw new WeightSpaceError(message, code);
}

function int(value: unknown, min: number, max: number, name: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < min || (value as number) > max) {
    fail(`${name} must be an integer from ${min} to ${max}`, 'ERANGE');
  }
  return value as number;
}

function finiteArray(value: unknown, count: number, name: string): number[] {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) fail(`${name} must be an array`);
  const arr = value as ArrayLike<number>;
  if (arr.length !== count) fail(`${name} must have length ${count}, got ${arr.length}`, 'ESHAPE');
  const out = new Array<number>(count);
  for (let i = 0; i < count; i++) {
    const v = arr[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) fail(`${name}[${i}] must be a finite number`);
    out[i] = v;
  }
  return out;
}

/** Validate one layer and return a normalised plain copy. */
export function validateLayer(layer: unknown, index = 0): DenseLayer {
  if (!layer || typeof layer !== 'object' || Array.isArray(layer)) fail(`layer ${index} must be an object`);
  const l = layer as Record<string, unknown>;
  const inputs = int(l.inputs, 1, LIMITS.maxDimension, `layer ${index} inputs`);
  const outputs = int(l.outputs, 1, LIMITS.maxDimension, `layer ${index} outputs`);
  if (inputs * outputs > LIMITS.maxParameters) fail(`layer ${index} exceeds ${LIMITS.maxParameters} parameters`, 'ERANGE');
  if (typeof l.activation !== 'string' || !ACTIVATIONS.includes(l.activation as DenseActivation)) {
    fail(`layer ${index} activation must be one of ${ACTIVATIONS.join(', ')}`);
  }
  const steepness = l.steepness === undefined ? 1 : l.steepness;
  if (typeof steepness !== 'number' || !Number.isFinite(steepness)) fail(`layer ${index} steepness must be finite`);
  return {
    inputs,
    outputs,
    weights: finiteArray(l.weights, inputs * outputs, `layer ${index} weights`),
    bias: l.bias == null ? null : finiteArray(l.bias, outputs, `layer ${index} bias`),
    activation: l.activation as DenseActivation,
    steepness,
  };
}

/** Validate a whole model against ruv-fann's schema and LIMITS; returns a normalised plain copy. */
export function validateModel(model: unknown): DenseModel {
  return validateModelWithJson(model).model;
}

/** `validateModel` plus the canonical JSON it already had to build for the byte-budget check. */
export function validateModelWithJson(model: unknown): { model: DenseModel; json: string } {
  if (typeof model === 'string') {
    if (Buffer.byteLength(model) > LIMITS.maxModelBytes) fail('model exceeds JSON byte budget', 'ERANGE');
    model = JSON.parse(model);
  }
  if (!model || typeof model !== 'object' || Array.isArray(model)) fail('model must be an object');
  const m = model as Record<string, unknown>;
  if (m.schema !== 'ruv-fann.dense-layers/1') fail("model schema must be 'ruv-fann.dense-layers/1'");
  if (!Array.isArray(m.layers)) fail('model layers must be an array');
  int(m.layers.length, 1, LIMITS.maxLayers, 'layer count');
  const layers: DenseLayer[] = [];
  let params = 0;
  for (let i = 0; i < m.layers.length; i++) {
    const layer = validateLayer(m.layers[i], i);
    if (i > 0 && layer.inputs !== layers[i - 1].outputs) {
      fail(`layer ${i} inputs (${layer.inputs}) must equal layer ${i - 1} outputs (${layers[i - 1].outputs})`, 'ESHAPE');
    }
    params += layer.inputs * layer.outputs + (layer.bias ? layer.outputs : 0);
    if (params > LIMITS.maxParameters) fail(`model exceeds ${LIMITS.maxParameters} parameters`, 'ERANGE');
    layers.push(layer);
  }
  const out: DenseModel = { schema: 'ruv-fann.dense-layers/1', layers };
  const json = canonicalJson(out);
  if (Buffer.byteLength(json) > LIMITS.maxModelBytes) fail('model exceeds JSON byte budget', 'ERANGE');
  return { model: out, json };
}

/** Canonical JSON (fixed key order) used for hashing and storage. */
export function canonicalJson(model: DenseModel): string {
  return JSON.stringify({
    schema: model.schema,
    layers: model.layers.map((l) => ({
      inputs: l.inputs,
      outputs: l.outputs,
      weights: l.weights,
      bias: l.bias ?? null,
      activation: l.activation,
      steepness: l.steepness,
    })),
  });
}

/** sha256 (hex) of the canonical JSON: identical weights always hash identically. */
export function modelHash(model: DenseModel, canonical?: string): string {
  return createHash('sha256').update(canonical ?? canonicalJson(model)).digest('hex');
}

export function parameterCount(model: DenseModel): number {
  let n = 0;
  for (const l of model.layers) n += l.inputs * l.outputs + (l.bias ? l.outputs : 0);
  return n;
}

/** Topology summary, e.g. `{ sizes: [4, 8, 1], activations: ['Tanh', 'Linear'], parameters: 49 }`. */
export interface Topology {
  sizes: number[];
  activations: DenseActivation[];
  parameters: number;
}

export function topologyOf(model: DenseModel): Topology {
  return {
    sizes: [model.layers[0].inputs, ...model.layers.map((l) => l.outputs)],
    activations: model.layers.map((l) => l.activation),
    parameters: parameterCount(model),
  };
}

export function sameTopology(a: DenseModel, b: DenseModel): boolean {
  if (a.layers.length !== b.layers.length) return false;
  return a.layers.every((l, i) => l.inputs === b.layers[i].inputs && l.outputs === b.layers[i].outputs
    && (l.bias == null) === (b.layers[i].bias == null));
}

/** Recursively freeze a model so a stored snapshot can never be mutated in place. */
export function freezeModel(model: DenseModel): DenseModel {
  for (const l of model.layers) {
    Object.freeze(l.weights);
    if (l.bias) Object.freeze(l.bias);
    Object.freeze(l);
  }
  Object.freeze(model.layers);
  return Object.freeze(model);
}

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample via Box-Muller over the given uniform source. */
export function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/** Build a dense MLP with Xavier-uniform weights. `sizes` = [inputs, hidden..., outputs]. */
export function createMlp(
  sizes: number[],
  options: { hidden?: DenseActivation; output?: DenseActivation; seed?: number; scale?: number } = {},
): DenseModel {
  if (!Array.isArray(sizes) || sizes.length < 2) fail('sizes must list at least [inputs, outputs]');
  const rand = rng(options.seed ?? 1);
  const layers: DenseLayer[] = [];
  for (let i = 0; i + 1 < sizes.length; i++) {
    const inputs = sizes[i];
    const outputs = sizes[i + 1];
    const limit = Math.sqrt(6 / (inputs + outputs)) * (options.scale ?? 1);
    const weights = new Array<number>(inputs * outputs);
    for (let w = 0; w < weights.length; w++) weights[w] = (rand() * 2 - 1) * limit;
    layers.push({
      inputs, outputs, weights, bias: new Array<number>(outputs).fill(0),
      activation: i + 2 === sizes.length ? (options.output ?? 'Linear') : (options.hidden ?? 'Tanh'),
      steepness: 1,
    });
  }
  return validateModel({ schema: 'ruv-fann.dense-layers/1', layers });
}
