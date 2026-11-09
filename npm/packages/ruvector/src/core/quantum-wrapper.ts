/**
 * Quantum Wrapper — state-vector quantum circuit simulation via the OPTIONAL
 * @ruvector/ruqu-wasm package (Rust ruqu-core compiled to WebAssembly).
 *
 * Nothing here touches the engine until a simulation is requested, so
 * requiring this module costs nothing at CLI/MCP startup.
 *
 * @ruvector/ruqu-wasm is an ES module that self-locates its .wasm through
 * `import.meta.url`. ruvector compiles to CommonJS, where tsc would rewrite a
 * plain `import()` into `require()` — so the engine is loaded through a real
 * dynamic import and initialised synchronously from the .wasm bytes.
 *
 * Circuits are validated here, before the engine is loaded, so bad input fails
 * with a precise message (which gate, which field) without paying for the WASM
 * load; the engine (^2.2.0) also rejects non-finite angles itself. The seed is
 * passed to the engine's `simulate(circuit, seed)` as well as to the shot
 * sampler. Measurement and reset are not accepted as circuit ops; use `shots`
 * (seeded sampling of the final state) instead.
 *
 * Bit order: qubit 0 is the least-significant bit, i.e. the RIGHTMOST
 * character of a basis-state bitstring (`x 0` on 2 qubits gives "01").
 */

import * as fs from 'fs';
import * as path from 'path';
import { pathToFileURL } from 'url';

const PKG = '@ruvector/ruqu-wasm';
export const QUANTUM_INSTALL_HINT = `npm i ${PKG}`;

const dynamicImport = new Function('specifier', 'return import(specifier)') as (specifier: string) => Promise<any>;

/** Hard limits applied to every circuit. Callers may tighten, never loosen past ENGINE_MAX_QUBITS. */
export interface QuantumLimits {
  maxQubits: number;
  maxGates: number;
  maxShots: number;
  maxTop: number;
}

/** The engine's own WASM limit (state vector 2^25 * 16 bytes = 512 MiB). */
export const ENGINE_MAX_QUBITS = 25;

/** Defaults for untrusted callers (MCP): 2^20 amplitudes = 16 MiB of state. */
export const DEFAULT_LIMITS: Readonly<QuantumLimits> = Object.freeze({
  maxQubits: 20,
  maxGates: 10000,
  maxShots: 100000,
  maxTop: 256,
});

/** Limits for the local CLI user: up to the engine's own ceiling. */
export const CLI_LIMITS: Readonly<QuantumLimits> = Object.freeze({
  maxQubits: ENGINE_MAX_QUBITS,
  maxGates: 1000000,
  maxShots: 10000000,
  maxTop: 4096,
});

type GateKind = 'single' | 'rotation' | 'pair' | 'pairRotation' | 'barrier';

const GATES: Readonly<Record<string, { kind: GateKind; method: string }>> = Object.freeze(
  Object.assign(Object.create(null), {
    h: { kind: 'single', method: 'h' },
    x: { kind: 'single', method: 'x' },
    y: { kind: 'single', method: 'y' },
    z: { kind: 'single', method: 'z' },
    s: { kind: 'single', method: 's' },
    t: { kind: 'single', method: 't' },
    rx: { kind: 'rotation', method: 'rx' },
    ry: { kind: 'rotation', method: 'ry' },
    rz: { kind: 'rotation', method: 'rz' },
    cnot: { kind: 'pair', method: 'cnot' },
    cx: { kind: 'pair', method: 'cnot' },
    cz: { kind: 'pair', method: 'cz' },
    swap: { kind: 'pair', method: 'swap' },
    rzz: { kind: 'pairRotation', method: 'rzz' },
    barrier: { kind: 'barrier', method: 'barrier' },
  })
);

export const SUPPORTED_GATES: readonly string[] = Object.freeze(Object.keys(GATES));

/** One validated gate: `qubits` holds 0, 1 or 2 indices; `angle` only for rotations. */
export interface QuantumGate {
  gate: string;
  qubits: number[];
  angle?: number;
}

export interface QuantumCircuitSpec {
  qubits: number;
  gates: QuantumGate[];
}

export interface QuantumSimulateOptions {
  shots?: number;
  seed?: number;
  top?: number;
  /** Include the full 2^n probability vector (CLI only — can be large). */
  full?: boolean;
  limits?: Partial<QuantumLimits>;
}

export interface QuantumState {
  state: string;
  index: number;
  probability: number;
}

export interface QuantumSimulateResult {
  numQubits: number;
  gateCount: number;
  depth: number;
  executionTimeMs: number;
  wallTimeMs: number;
  memoryBytes: number;
  /** Most probable basis states, highest first (probability > 1e-12). */
  top: QuantumState[];
  /** Sampled measurement counts keyed by bitstring, when shots > 0. */
  counts?: Record<string, number>;
  shots: number;
  seed?: number;
  probabilities?: number[];
  bitOrder: 'qubit 0 = rightmost bit';
}

function resolveLimits(partial?: Partial<QuantumLimits>): QuantumLimits {
  const out = { ...DEFAULT_LIMITS, ...(partial || {}) };
  out.maxQubits = Math.min(out.maxQubits, ENGINE_MAX_QUBITS);
  return out;
}

function fail(msg: string): never {
  throw new Error(msg);
}

function stateBytes(n: number): number {
  return Math.pow(2, n) * 16;
}

function formatBytes(b: number): string {
  if (b >= 1 << 30) return `${(b / (1 << 30)).toFixed(1)} GiB`;
  if (b >= 1 << 20) return `${(b / (1 << 20)).toFixed(1)} MiB`;
  if (b >= 1 << 10) return `${(b / (1 << 10)).toFixed(1)} KiB`;
  return `${b} B`;
}

function checkQubitCount(n: unknown, limits: QuantumLimits): number {
  if (typeof n !== 'number' || !Number.isInteger(n) || n < 1) {
    fail(`qubits must be a positive integer, got ${JSON.stringify(n)}`);
  }
  if (n > limits.maxQubits) {
    fail(
      `qubits=${n} exceeds the cap of ${limits.maxQubits} ` +
        `(state vector would need ${formatBytes(stateBytes(n))}; 2^n * 16 bytes)`
    );
  }
  return n;
}

function checkIndex(q: unknown, n: number, where: string): number {
  if (typeof q !== 'number' || !Number.isInteger(q) || q < 0 || q >= n) {
    fail(`${where}: qubit index ${JSON.stringify(q)} out of range 0..${n - 1}`);
  }
  return q;
}

function checkAngle(a: unknown, where: string): number {
  if (typeof a !== 'number' || !Number.isFinite(a)) {
    fail(`${where}: angle must be a finite number, got ${JSON.stringify(a)}`);
  }
  return a;
}

/**
 * Validate and normalise an untrusted circuit description.
 *
 * Accepted gate shapes: `{gate:'h', qubits:[0]}`, `{gate:'h', qubit:0}`,
 * `{gate:'cnot', control:0, target:1}`, `{gate:'rx', qubits:[0], angle:1.57}`,
 * `{gate:'rzz', qubits:[0,1], angle:0.5}`, `{gate:'barrier'}`.
 * `op` is accepted as an alias for `gate`.
 */
export function validateCircuit(spec: unknown, limitsIn?: Partial<QuantumLimits>): QuantumCircuitSpec {
  const limits = resolveLimits(limitsIn);
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) fail('circuit must be an object {qubits, gates}');
  const s = spec as Record<string, unknown>;
  const n = checkQubitCount(s.qubits, limits);
  const rawGates = s.gates === undefined ? [] : s.gates;
  if (!Array.isArray(rawGates)) fail('gates must be an array');
  if (rawGates.length > limits.maxGates) fail(`gate count ${rawGates.length} exceeds the cap of ${limits.maxGates}`);

  const gates: QuantumGate[] = rawGates.map((g: unknown, i: number) => {
    const where = `gates[${i}]`;
    if (!g || typeof g !== 'object' || Array.isArray(g)) fail(`${where} must be an object`);
    const r = g as Record<string, unknown>;
    const name = r.gate ?? r.op;
    if (typeof name !== 'string') fail(`${where}: missing gate name`);
    const key = name.trim().toLowerCase();
    if (!Object.prototype.hasOwnProperty.call(GATES, key)) {
      if (key === 'measure' || key === 'measure_all' || key === 'reset') {
        fail(`${where}: '${key}' is not supported as a circuit op; use shots to sample measurements`);
      }
      fail(`${where}: unknown gate '${name}' (supported: ${SUPPORTED_GATES.join(', ')})`);
    }
    const def = GATES[key];
    let qs: unknown[];
    if (Array.isArray(r.qubits)) qs = r.qubits;
    else if (r.control !== undefined || r.target !== undefined) qs = [r.control, r.target];
    else if (r.qubit !== undefined) qs = [r.qubit];
    else qs = [];
    const want = def.kind === 'barrier' ? 0 : def.kind === 'single' || def.kind === 'rotation' ? 1 : 2;
    if (qs.length !== want) fail(`${where}: '${key}' takes ${want} qubit(s), got ${qs.length}`);
    const qubits = qs.map((q) => checkIndex(q, n, where));
    if (want === 2 && qubits[0] === qubits[1]) fail(`${where}: '${key}' needs two distinct qubits`);
    const out: QuantumGate = { gate: key, qubits };
    if (def.kind === 'rotation' || def.kind === 'pairRotation') out.angle = checkAngle(r.angle ?? r.theta, where);
    return out;
  });
  return { qubits: n, gates };
}

/**
 * Parse the compact text form used by the CLI, e.g. `"h 0; cx 0 1"` or
 * `"rx 0 1.5708"` (one gate per `;` or newline; angle last). Also accepts the
 * presets `bell` and `ghz` / `ghz:N`.
 */
export function parseCircuitText(
  text: string,
  qubits?: number,
  limitsIn?: Partial<QuantumLimits>
): { qubits: number; gates: unknown[] } {
  const limits = resolveLimits(limitsIn);
  const t = String(text).trim();
  const preset = /^(bell|ghz)(?::(\d+))?$/i.exec(t);
  if (preset) {
    const kind = preset[1].toLowerCase();
    const n = kind === 'bell' ? 2 : preset[2] ? parseInt(preset[2], 10) : qubits ?? 3;
    // Checked before building: ghz:N allocates N gates.
    return ghzCircuit(checkQubitCount(n, limits));
  }
  const gates: Record<string, unknown>[] = [];
  let maxIndex = -1;
  for (const raw of t.split(/[;\n]+/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    if (gates.length >= limits.maxGates) fail(`gate count exceeds the cap of ${limits.maxGates}`);
    const [name, ...args] = line.split(/[\s,]+/);
    const key = name.toLowerCase();
    const def = GATES[key];
    const nums = args.map((a) => Number(a));
    let qs: number[] = nums;
    let angle: number | undefined;
    if (def && (def.kind === 'rotation' || def.kind === 'pairRotation')) {
      angle = nums[nums.length - 1];
      qs = nums.slice(0, -1);
    }
    for (const q of qs) if (Number.isInteger(q)) maxIndex = Math.max(maxIndex, q);
    gates.push(angle === undefined ? { gate: name, qubits: qs } : { gate: name, qubits: qs, angle });
  }
  return { qubits: qubits ?? Math.max(1, maxIndex + 1), gates };
}

/**
 * Turn a string circuit into a spec: inline JSON (`'{"qubits":2,...}'`), a
 * preset (`bell`, `ghz`, `ghz:N`) or text ops (`"h 0; cx 0 1"`). Shared by the
 * CLI and the MCP tool so both accept the same shorthand under the caller's
 * limits. Never reads files.
 */
export function circuitFromString(text: string, qubits?: number, limits?: Partial<QuantumLimits>): unknown {
  const s = String(text).trim();
  if (s.startsWith('{')) {
    try {
      return JSON.parse(s);
    } catch (e) {
      fail(`circuit JSON does not parse: ${(e as Error).message}`);
    }
  }
  return parseCircuitText(s, qubits, limits);
}

/** n-qubit GHZ circuit: H on qubit 0, then a CNOT chain. n=2 is the Bell state. */
export function ghzCircuit(n: number): QuantumCircuitSpec {
  const gates: QuantumGate[] = [{ gate: 'h', qubits: [0] }];
  for (let i = 0; i < n - 1; i++) gates.push({ gate: 'cnot', qubits: [i, i + 1] });
  return { qubits: n, gates };
}

// ── engine loading ──────────────────────────────────────────────────────────

let enginePromise: Promise<any> | null = null;

function resolveEnginePath(): string | null {
  try {
    return require.resolve(PKG, { paths: [process.cwd(), __dirname] });
  } catch {
    return null;
  }
}

/** True when @ruvector/ruqu-wasm resolves. Never throws, never loads the engine. */
export function isQuantumAvailable(): boolean {
  return resolveEnginePath() !== null;
}

/** Load and initialise the engine once. Rejects with an install hint when absent. */
export function loadQuantum(): Promise<any> {
  if (enginePromise) return enginePromise;
  const main = resolveEnginePath();
  if (!main) {
    return Promise.reject(new Error(`${PKG} is not installed. Install it with: ${QUANTUM_INSTALL_HINT}`));
  }
  enginePromise = (async () => {
    const mod = await dynamicImport(pathToFileURL(main).href);
    const wasm = path.join(path.dirname(main), 'ruqu_wasm_bg.wasm');
    mod.initSync({ module: fs.readFileSync(wasm) });
    return mod;
  })();
  enginePromise.catch(() => {
    enginePromise = null;
  });
  return enginePromise;
}

function engineVersion(): string | null {
  const main = resolveEnginePath();
  if (!main) return null;
  try {
    return JSON.parse(fs.readFileSync(path.join(path.dirname(main), 'package.json'), 'utf8')).version || null;
  } catch {
    return null;
  }
}

// ── sampling ────────────────────────────────────────────────────────────────

/** mulberry32: small, fast, seedable PRNG for reproducible shot sampling. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function bitstring(index: number, n: number): string {
  return index.toString(2).padStart(n, '0');
}

function sampleCounts(probs: ArrayLike<number>, n: number, shots: number, seed: number): Record<string, number> {
  const support: number[] = [];
  const cdf: number[] = [];
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > 0) {
      acc += p;
      support.push(i);
      cdf.push(acc);
    }
  }
  const rand = mulberry32(seed);
  const counts: Record<string, number> = Object.create(null);
  for (let s = 0; s < shots; s++) {
    const u = rand() * acc;
    let lo = 0;
    let hi = cdf.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cdf[mid] < u) lo = mid + 1;
      else hi = mid;
    }
    const key = bitstring(support[lo], n);
    counts[key] = (counts[key] || 0) + 1;
  }
  return { ...counts };
}

function topStates(probs: ArrayLike<number>, n: number, k: number): QuantumState[] {
  const idx: number[] = [];
  for (let i = 0; i < probs.length; i++) if (probs[i] > 1e-12) idx.push(i);
  idx.sort((a, b) => probs[b] - probs[a] || a - b);
  return idx.slice(0, k).map((i) => ({ state: bitstring(i, n), index: i, probability: probs[i] }));
}

function checkIntOption(v: unknown, name: string, min: number, max: number, dflt: number): number {
  if (v === undefined || v === null) return dflt;
  if (typeof v !== 'number' || !Number.isInteger(v) || v < min || v > max) {
    fail(`${name} must be an integer in ${min}..${max}, got ${JSON.stringify(v)}`);
  }
  return v;
}

// ── operations ──────────────────────────────────────────────────────────────

/** Validate (before loading anything), then simulate a circuit. */
export async function simulate(spec: unknown, options: QuantumSimulateOptions = {}): Promise<QuantumSimulateResult> {
  const limits = resolveLimits(options.limits);
  const circuit = validateCircuit(spec, limits);
  const shots = checkIntOption(options.shots, 'shots', 0, limits.maxShots, 0);
  const top = checkIntOption(options.top, 'top', 1, limits.maxTop, 16);
  const seed = checkIntOption(options.seed, 'seed', 0, 0xffffffff, 0x5eed);

  const q = await loadQuantum();
  const c = new q.WasmQuantumCircuit(circuit.qubits);
  try {
    for (const g of circuit.gates) {
      const m = GATES[g.gate].method;
      if (g.angle !== undefined) c[m](...g.qubits, g.angle);
      else c[m](...g.qubits);
    }
    const t0 = process.hrtime.bigint();
    const r = q.simulate(c, seed);
    const wallTimeMs = Number(process.hrtime.bigint() - t0) / 1e6;
    const probs: number[] = r.probabilities;
    const out: QuantumSimulateResult = {
      numQubits: r.num_qubits,
      gateCount: r.gate_count,
      depth: c.depth,
      executionTimeMs: r.execution_time_ms,
      wallTimeMs,
      memoryBytes: q.estimate_memory(circuit.qubits),
      top: topStates(probs, circuit.qubits, top),
      shots,
      bitOrder: 'qubit 0 = rightmost bit',
    };
    if (shots > 0) {
      out.counts = sampleCounts(probs, circuit.qubits, shots, seed);
      out.seed = seed;
    }
    if (options.full) out.probabilities = Array.from(probs);
    return out;
  } finally {
    c.free();
  }
}

export interface QuantumInfo {
  available: boolean;
  package: string;
  version: string | null;
  engineMaxQubits: number | null;
  limits: QuantumLimits;
  gates: readonly string[];
  bitOrder: string;
  memoryBytes?: Record<string, number>;
  install?: string;
}

/** Engine availability, version, caps and per-size state-vector memory. */
export async function info(limitsIn?: Partial<QuantumLimits>): Promise<QuantumInfo> {
  const limits = resolveLimits(limitsIn);
  const base = {
    package: PKG,
    limits,
    gates: SUPPORTED_GATES,
    bitOrder: 'qubit 0 = rightmost bit',
  };
  if (!isQuantumAvailable()) {
    return { available: false, version: null, engineMaxQubits: null, install: QUANTUM_INSTALL_HINT, ...base };
  }
  const q = await loadQuantum();
  const memoryBytes: Record<string, number> = {};
  for (const n of [10, 16, 20, 25]) if (n <= q.max_qubits()) memoryBytes[String(n)] = q.estimate_memory(n);
  return { available: true, version: engineVersion(), engineMaxQubits: q.max_qubits(), memoryBytes, ...base };
}

export interface QuantumBenchRow {
  qubits: number;
  gates: number;
  repeats: number;
  medianMs: number;
  minMs: number;
  memoryBytes: number;
}

/** Time GHZ-state simulation (H + CNOT chain) at each size; reports median wall time. */
export async function bench(sizes: number[] = [10, 16, 20], repeats = 5): Promise<QuantumBenchRow[]> {
  const q = await loadQuantum();
  const rows: QuantumBenchRow[] = [];
  for (const n of sizes) {
    checkQubitCount(n, CLI_LIMITS);
    const times: number[] = [];
    let gates = 0;
    for (let r = 0; r < repeats; r++) {
      const c = new q.WasmQuantumCircuit(n);
      try {
        c.h(0);
        for (let i = 0; i < n - 1; i++) c.cnot(i, i + 1);
        gates = c.gate_count;
        const t0 = process.hrtime.bigint();
        q.simulate(c);
        times.push(Number(process.hrtime.bigint() - t0) / 1e6);
      } finally {
        c.free();
      }
    }
    times.sort((a, b) => a - b);
    rows.push({
      qubits: n,
      gates,
      repeats,
      medianMs: times[Math.floor(times.length / 2)],
      minMs: times[0],
      memoryBytes: q.estimate_memory(n),
    });
  }
  return rows;
}

/** Grover search over 2^qubits items for the given target indices. */
export async function grover(
  qubits: number,
  targets: number[],
  seed?: number,
  limitsIn?: Partial<QuantumLimits>
): Promise<{ measuredState: number; measured: string; targetFound: boolean; successProbability: number; iterations: number }> {
  const limits = resolveLimits(limitsIn);
  const n = checkQubitCount(qubits, limits);
  if (!Array.isArray(targets) || targets.length === 0) fail('grover needs at least one target index');
  for (const t of targets) {
    if (!Number.isInteger(t) || t < 0 || t >= Math.pow(2, n)) fail(`target ${t} out of range 0..${Math.pow(2, n) - 1}`);
  }
  const s = checkIntOption(seed, 'seed', 0, 0xffffffff, 0x5eed);
  const q = await loadQuantum();
  const r = q.grover_search(n, new Uint32Array(targets), s);
  return {
    measuredState: r.measured_state,
    measured: bitstring(r.measured_state, n),
    targetFound: r.target_found,
    successProbability: r.success_probability,
    iterations: r.num_iterations,
  };
}
