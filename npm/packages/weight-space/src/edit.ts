/**
 * Weight editing primitives. Every function is pure: it returns a new, validated model and never
 * mutates its input. Shapes are checked against the row-major [outputs, inputs] layout and LIMITS.
 */
import {
  DenseActivation, DenseLayer, DenseModel, WeightSpaceError, gaussian, rng, sameTopology,
  topologyOf, validateLayer, validateModel,
} from './model.js';

export interface LayerView {
  index: number;
  inputs: number;
  outputs: number;
  activation: DenseActivation;
  steepness: number;
  /** Row-major [outputs, inputs] copy. */
  weights: number[];
  bias: number[] | null;
}

function layerIndex(model: DenseModel, index: number): number {
  if (!Number.isSafeInteger(index) || index < 0 || index >= model.layers.length) {
    throw new WeightSpaceError(`layer index must be an integer from 0 to ${model.layers.length - 1}`, 'ERANGE');
  }
  return index;
}

/** Copy of one layer. Editing the returned arrays does not affect the model. */
export function getLayer(model: DenseModel, index: number): LayerView {
  const l = model.layers[layerIndex(model, index)];
  return {
    index, inputs: l.inputs, outputs: l.outputs, activation: l.activation, steepness: l.steepness,
    weights: Array.from(l.weights), bias: l.bias ? Array.from(l.bias) : null,
  };
}

export interface LayerPatch {
  weights?: ArrayLike<number>;
  bias?: ArrayLike<number> | null;
  activation?: DenseActivation;
  steepness?: number;
}

/**
 * Return a new model with layer `index` patched. The layer's shape cannot change (weights must
 * keep length outputs*inputs, bias length outputs), so adjacent layers stay compatible.
 */
export function setLayer(model: DenseModel, index: number, patch: LayerPatch): DenseModel {
  const base = validateModel(model);
  const i = layerIndex(base, index);
  if (!patch || typeof patch !== 'object') throw new WeightSpaceError('patch must be an object', 'EINVALID');
  const allowed = ['weights', 'bias', 'activation', 'steepness'];
  for (const key of Object.keys(patch)) {
    if (!allowed.includes(key)) throw new WeightSpaceError(`unknown patch field '${key}'`, 'EINVALID');
  }
  const old = base.layers[i];
  const next = validateLayer({
    inputs: old.inputs,
    outputs: old.outputs,
    weights: patch.weights !== undefined ? Array.from(patch.weights) : old.weights,
    bias: patch.bias === undefined ? old.bias : patch.bias === null ? null : Array.from(patch.bias),
    activation: patch.activation ?? old.activation,
    steepness: patch.steepness ?? old.steepness,
  }, i);
  const layers = base.layers.slice();
  layers[i] = next;
  return validateModel({ schema: base.schema, layers });
}

export interface PerturbOptions {
  /** Standard deviation of the Gaussian noise (default 0.01). */
  scale?: number;
  /** Seed for reproducible noise (default 1). */
  seed?: number;
  /** Which layers to perturb (default all). */
  layers?: number[];
  /** Also perturb biases (default true). */
  bias?: boolean;
  /** Scale noise by each layer's weight RMS instead of using an absolute scale (default false). */
  relative?: boolean;
}

/** Return a new model with seeded Gaussian noise added to selected layers. */
export function perturb(model: DenseModel, options: PerturbOptions = {}): DenseModel {
  const base = validateModel(model);
  const scale = options.scale ?? 0.01;
  if (typeof scale !== 'number' || !Number.isFinite(scale) || scale < 0) throw new WeightSpaceError('scale must be a finite number >= 0', 'ERANGE');
  const targets = new Set((options.layers ?? base.layers.map((_, i) => i)).map((i) => layerIndex(base, i)));
  const rand = rng(options.seed ?? 1);
  const layers: DenseLayer[] = base.layers.map((l, i) => {
    if (!targets.has(i)) return l;
    let sigma = scale;
    if (options.relative) {
      let ss = 0;
      for (const w of l.weights) ss += w * w;
      sigma = scale * Math.sqrt(ss / l.weights.length);
    }
    const weights = l.weights.map((w) => w + sigma * gaussian(rand));
    const bias = l.bias ? (options.bias === false ? l.bias : l.bias.map((b) => b + sigma * gaussian(rand))) : null;
    return { ...l, weights, bias };
  });
  return validateModel({ schema: base.schema, layers });
}

export interface LayerDiff {
  index: number;
  /** L2 norm of (b - a) over weights and biases. */
  l2: number;
  maxAbs: number;
  meanAbs: number;
  /** Number of parameters whose value differs. */
  changed: number;
  parameters: number;
  activationChanged: boolean;
  steepnessChanged: boolean;
}

export type ModelDiff =
  | { compatible: true; l2: number; cosine: number; changed: number; parameters: number; layers: LayerDiff[] }
  | { compatible: false; reason: string; a: ReturnType<typeof topologyOf>; b: ReturnType<typeof topologyOf> };

/** Parameter-level difference between two models of the same topology. */
export function diff(a: DenseModel, b: DenseModel): ModelDiff {
  const ma = validateModel(a);
  const mb = validateModel(b);
  if (!sameTopology(ma, mb)) {
    return { compatible: false, reason: 'topology differs', a: topologyOf(ma), b: topologyOf(mb) };
  }
  let total = 0, dot = 0, na = 0, nb = 0, changed = 0, parameters = 0;
  const layers: LayerDiff[] = ma.layers.map((la, i) => {
    const lb = mb.layers[i];
    const va = la.bias ? [...la.weights, ...la.bias] : la.weights;
    const vb = lb.bias ? [...lb.weights, ...lb.bias] : lb.weights;
    let ss = 0, max = 0, sumAbs = 0, ch = 0;
    for (let p = 0; p < va.length; p++) {
      const d = vb[p] - va[p];
      const ad = Math.abs(d);
      ss += d * d;
      sumAbs += ad;
      if (ad > max) max = ad;
      if (d !== 0) ch++;
      dot += va[p] * vb[p];
      na += va[p] * va[p];
      nb += vb[p] * vb[p];
    }
    total += ss;
    changed += ch;
    parameters += va.length;
    return {
      index: i, l2: Math.sqrt(ss), maxAbs: max, meanAbs: sumAbs / va.length, changed: ch, parameters: va.length,
      activationChanged: la.activation !== lb.activation, steepnessChanged: la.steepness !== lb.steepness,
    };
  });
  const cosine = na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : (na === nb ? 1 : 0);
  return { compatible: true, l2: Math.sqrt(total), cosine, changed, parameters, layers };
}
