'use strict';

const { version } = require('../package.json');

let nativeModule = null;
try {
  nativeModule = require('../build/Release/snn_simd.node');
} catch {
  // Native acceleration is optional; the portable implementation is complete.
}

const native = nativeModule !== null;

function finite(value, fallback, name) {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved)) throw new TypeError(`${name} must be finite`);
  return resolved;
}

function assertLength(actual, expected, name) {
  if (actual.length !== expected) {
    throw new RangeError(`${name} must contain ${expected} values (received ${actual.length})`);
  }
}

class LIFLayer {
  constructor(nNeurons, params = {}) {
    if (!Number.isSafeInteger(nNeurons) || nNeurons <= 0) {
      throw new RangeError('n_neurons must be a positive safe integer');
    }
    this.n_neurons = nNeurons;
    this.dt = finite(params.dt, 1, 'dt');
    this.tau = finite(params.tau, 20, 'tau');
    if (this.dt <= 0 || this.tau <= 0) throw new RangeError('dt and tau must be positive');
    this.v_rest = finite(params.v_rest, -70, 'v_rest');
    this.v_reset = finite(params.v_reset, -75, 'v_reset');
    this.v_thresh = finite(params.v_thresh, -50, 'v_thresh');
    this.resistance = finite(params.resistance, 10, 'resistance');
    this.inhibition_strength = finite(params.inhibition_strength, 10, 'inhibition_strength');
    this.lateral_inhibition = Boolean(params.lateral_inhibition);
    this.voltages = new Float32Array(nNeurons);
    this.currents = new Float32Array(nNeurons);
    this.spikes = new Float32Array(nNeurons);
    this.voltages.fill(this.v_rest);
  }

  setCurrents(currents) {
    assertLength(currents, this.n_neurons, 'currents');
    this.currents.set(currents);
  }

  update(currents) {
    if (currents !== undefined) this.setCurrents(currents);
    let count = 0;
    if (nativeModule) {
      nativeModule.lifUpdate(
        this.voltages,
        this.currents,
        this.dt,
        this.tau,
        this.v_rest,
        this.resistance
      );
      count = nativeModule.detectSpikes(
        this.voltages,
        this.spikes,
        this.v_thresh,
        this.v_reset
      );
      if (this.lateral_inhibition && count > 0) {
        nativeModule.lateralInhibition(this.voltages, this.spikes, this.inhibition_strength);
      }
    } else {
      for (let i = 0; i < this.n_neurons; i++) {
        const dv =
          (-(this.voltages[i] - this.v_rest) + this.resistance * this.currents[i]) *
          this.dt /
          this.tau;
        this.voltages[i] += dv;
        if (this.voltages[i] >= this.v_thresh) {
          this.spikes[i] = 1;
          this.voltages[i] = this.v_reset;
          count++;
        } else {
          this.spikes[i] = 0;
        }
      }
      if (this.lateral_inhibition && count > 0) {
        for (let winner = 0; winner < this.n_neurons; winner++) {
          if (!this.spikes[winner]) continue;
          for (let i = 0; i < this.n_neurons; i++) {
            if (i !== winner) this.voltages[i] -= this.inhibition_strength;
          }
        }
      }
    }
    return count;
  }

  getSpikes() {
    return new Float32Array(this.spikes);
  }

  reset() {
    this.voltages.fill(this.v_rest);
    this.currents.fill(0);
    this.spikes.fill(0);
  }
}

class SynapticLayer {
  constructor(nPre, nPost, params = {}) {
    if (!Number.isSafeInteger(nPre) || nPre <= 0 || !Number.isSafeInteger(nPost) || nPost <= 0) {
      throw new RangeError('synapse dimensions must be positive safe integers');
    }
    this.n_pre = nPre;
    this.n_post = nPost;
    this.a_plus = finite(params.a_plus, 0.01, 'a_plus');
    this.a_minus = finite(params.a_minus, 0.01, 'a_minus');
    this.w_min = finite(params.w_min, 0, 'w_min');
    this.w_max = finite(params.w_max, 1, 'w_max');
    if (this.w_min > this.w_max) throw new RangeError('w_min cannot exceed w_max');
    const mean = finite(params.init_weight, 0.5, 'init_weight');
    const spread = Math.max(0, finite(params.init_std, 0.1, 'init_std'));
    this.weights = new Float32Array(nPre * nPost);
    this.pre_trace = new Float32Array(nPre);
    this.post_trace = new Float32Array(nPost);
    for (let i = 0; i < this.weights.length; i++) {
      const u1 = Math.max(Number.EPSILON, Math.random());
      const gaussian = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * Math.random());
      this.weights[i] = Math.min(this.w_max, Math.max(this.w_min, mean + gaussian * spread));
    }
  }

  forward(preSpikes, postCurrents = new Float32Array(this.n_post)) {
    assertLength(preSpikes, this.n_pre, 'pre_spikes');
    assertLength(postCurrents, this.n_post, 'post_currents');
    if (nativeModule) {
      nativeModule.computeCurrents(postCurrents, preSpikes, this.weights);
    } else {
      postCurrents.fill(0);
      for (let post = 0; post < this.n_post; post++) {
        const offset = post * this.n_pre;
        let sum = 0;
        for (let pre = 0; pre < this.n_pre; pre++) {
          sum += preSpikes[pre] * this.weights[offset + pre];
        }
        postCurrents[post] = sum;
      }
    }
    return postCurrents;
  }

  learn(preSpikes, postSpikes, decay = 0.95) {
    assertLength(preSpikes, this.n_pre, 'pre_spikes');
    assertLength(postSpikes, this.n_post, 'post_spikes');
    if (nativeModule) {
      nativeModule.stdpUpdate(
        this.weights,
        preSpikes,
        postSpikes,
        this.pre_trace,
        this.post_trace,
        this.a_plus,
        this.a_minus,
        this.w_min,
        this.w_max
      );
      nativeModule.updateTraces(this.pre_trace, preSpikes, decay);
      nativeModule.updateTraces(this.post_trace, postSpikes, decay);
      return;
    }
    for (let post = 0; post < this.n_post; post++) {
      for (let pre = 0; pre < this.n_pre; pre++) {
        const index = post * this.n_pre + pre;
        const delta =
          this.a_plus * postSpikes[post] * this.pre_trace[pre] -
          this.a_minus * preSpikes[pre] * this.post_trace[post];
        this.weights[index] = Math.min(
          this.w_max,
          Math.max(this.w_min, this.weights[index] + delta)
        );
      }
    }
    for (let i = 0; i < this.n_pre; i++) {
      this.pre_trace[i] = this.pre_trace[i] * decay + preSpikes[i];
    }
    for (let i = 0; i < this.n_post; i++) {
      this.post_trace[i] = this.post_trace[i] * decay + postSpikes[i];
    }
  }

  getWeightStats() {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    for (const weight of this.weights) {
      min = Math.min(min, weight);
      max = Math.max(max, weight);
      sum += weight;
    }
    return { min, max, mean: sum / this.weights.length };
  }
}

class SpikingNeuralNetwork {
  constructor(layerSizes, params = {}) {
    if (!Array.isArray(layerSizes) || layerSizes.length < 2) {
      throw new TypeError('layer_sizes must contain at least input and output sizes');
    }
    this.dt = finite(params.dt, 1, 'dt');
    this.time = 0;
    this.layers = layerSizes.map((size) => new LIFLayer(size, params));
    this.synapses = layerSizes.slice(0, -1).map(
      (size, i) => new SynapticLayer(size, layerSizes[i + 1], params)
    );
  }

  step(input) {
    assertLength(input, this.layers[0].n_neurons, 'input');
    this.layers[0].spikes.set(input);
    let previous = this.layers[0].spikes;
    let totalSpikes = 0;
    for (let i = 0; i < this.synapses.length; i++) {
      const layer = this.layers[i + 1];
      this.synapses[i].forward(previous, layer.currents);
      totalSpikes += layer.update();
      this.synapses[i].learn(previous, layer.spikes);
      previous = layer.spikes;
    }
    this.time += this.dt;
    return totalSpikes;
  }

  run(duration, inputGenerator) {
    if (!Number.isFinite(duration) || duration < 0) throw new RangeError('duration must be non-negative');
    const output = [];
    const steps = Math.ceil(duration / this.dt);
    for (let i = 0; i < steps; i++) {
      this.step(inputGenerator(this.time));
      output.push(this.getOutput());
    }
    return output;
  }

  getOutput() {
    return this.layers[this.layers.length - 1].getSpikes();
  }

  getStats() {
    return {
      time: this.time,
      native,
      layers: this.layers.map((layer, i) => ({
        neurons: layer.n_neurons,
        spikes: layer.spikes.reduce((sum, value) => sum + value, 0),
        synapses: i < this.synapses.length ? this.synapses[i].getWeightStats() : null
      }))
    };
  }

  reset() {
    this.time = 0;
    for (const layer of this.layers) layer.reset();
    for (const synapse of this.synapses) {
      synapse.pre_trace.fill(0);
      synapse.post_trace.fill(0);
    }
  }
}

function createFeedforwardSNN(layerSizes, params) {
  return new SpikingNeuralNetwork(layerSizes, params);
}

function rateEncoding(values, dt = 1, maxRate = 100) {
  if (!Number.isFinite(dt) || dt < 0 || !Number.isFinite(maxRate) || maxRate < 0) {
    throw new RangeError('dt and max_rate must be non-negative finite numbers');
  }
  const spikes = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const normalized = Math.min(1, Math.max(0, Number(values[i]) || 0));
    spikes[i] = Math.random() < normalized * maxRate * dt / 1000 ? 1 : 0;
  }
  return spikes;
}

function temporalEncoding(values, time, tStart = 0, tWindow = 50) {
  if (!Number.isFinite(tWindow) || tWindow <= 0) throw new RangeError('t_window must be positive');
  const spikes = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const normalized = Math.min(1, Math.max(0, Number(values[i]) || 0));
    const spikeTime = tStart + (1 - normalized) * tWindow;
    spikes[i] = time >= spikeTime && time < spikeTime + 1 ? 1 : 0;
  }
  return spikes;
}

const SIMDOps = {
  dotProduct(a, b) {
    assertLength(b, a.length, 'right operand');
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
  },
  distance(a, b) {
    assertLength(b, a.length, 'right operand');
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const delta = a[i] - b[i];
      sum += delta * delta;
    }
    return Math.sqrt(sum);
  },
  cosineSimilarity(a, b) {
    assertLength(b, a.length, 'right operand');
    let aa = 0;
    let bb = 0;
    let ab = 0;
    for (let i = 0; i < a.length; i++) {
      aa += a[i] * a[i];
      bb += b[i] * b[i];
      ab += a[i] * b[i];
    }
    return aa === 0 || bb === 0 ? 0 : ab / Math.sqrt(aa * bb);
  }
};

module.exports = {
  LIFLayer,
  SynapticLayer,
  SpikingNeuralNetwork,
  createFeedforwardSNN,
  rateEncoding,
  temporalEncoding,
  SIMDOps,
  native,
  version
};
